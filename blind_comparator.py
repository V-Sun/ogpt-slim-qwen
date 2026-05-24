#!/usr/bin/env python3
"""Blind pairwise comparator — issue text + patch A diff + patch B diff only.

Tracks 2 (nano), 4 (mini), 6 (5.4 full) of the scope-reset experiment plan.

Usage:
    python3 blind_comparator.py --model azure/gpt-5.4-nano --mode smoke
    python3 blind_comparator.py --model azure/gpt-5.4-nano --mode pilot
    python3 blind_comparator.py --model azure/gpt-5.4-nano --mode full

Outputs:
    outputs/blind_cmp_{model_slug}_{mode}/pairwise_votes.jsonl
    One row per (instance_id, patch_a_idx, patch_b_idx), votes list embedded.
    Position-swap (A/B flipped) is run for every pair; winner is de-swapped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
PROPOSALS    = ROOT / "outputs" / "canonical_greedy8" / "proposals"
K_LIST       = list(range(8))
PILOT_SEED   = 42
PILOT_N      = 50
SMOKE_N      = 5
BUDGET_CAP   = 400.0

PROMPT = """\
You are comparing two candidate patches for a GitHub issue.
Choose which patch is more likely to correctly fix the issue.

ISSUE:
{issue_text}

PATCH A:
{patch_a}

PATCH B:
{patch_b}

Rules:
- Pick A if Patch A more directly and correctly addresses the root cause in the issue.
- Pick B if Patch B does.
- TIE only if both patches make identical or equally correct changes to the same code path.
- Do NOT choose TIE simply because both look plausible — read carefully and commit.

Respond with EXACTLY this JSON, no markdown fences, no extra text:
{{"winner": "A"|"B"|"TIE", "confidence": <0.5-1.0>, "reasoning": "<one sentence>"}}
"""


# ── Azure creds ───────────────────────────────────────────────────────────────

def _load_creds() -> dict:
    # Normalise: AZURE_DEPLOYMENT_NAME → AZURE_DEPLOYMENT
    if not os.getenv("AZURE_DEPLOYMENT") and os.getenv("AZURE_DEPLOYMENT_NAME"):
        os.environ["AZURE_DEPLOYMENT"] = os.environ["AZURE_DEPLOYMENT_NAME"]
    if not all(os.getenv(k) for k in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_DEPLOYMENT")):
        try:
            sys.path.insert(0, str(ROOT.parent / "orchestra-gpt"))
            import config as _c  # noqa: F401
            sys.path.pop(0)
            # config.py may override AZURE_DEPLOYMENT_NAME; sync back
            if not os.getenv("AZURE_DEPLOYMENT") and os.getenv("AZURE_DEPLOYMENT_NAME"):
                os.environ["AZURE_DEPLOYMENT"] = os.environ["AZURE_DEPLOYMENT_NAME"]
        except Exception:
            pass
    missing = [k for k in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_DEPLOYMENT")
               if not os.getenv(k)]
    if missing:
        sys.exit(f"FATAL: missing env vars: {missing}")
    # Always use 2025-03-01-preview for the Responses API with reasoning
    return {
        "api_key":        os.environ["AZURE_API_KEY"],
        "azure_endpoint": os.environ["AZURE_API_BASE"],
        "api_version":    "2025-03-01-preview",
        "deployment":     os.environ["AZURE_DEPLOYMENT"],
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_patches() -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for k in K_LIST:
        p = PROPOSALS / f"proposer_{k}.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            patch = (r.get("model_patch") or "").strip()
            if patch:
                out.setdefault(r["instance_id"], {})[k] = patch
    return out


def load_issues(iids: list[str]) -> dict[str, str]:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    return {r["instance_id"]: r.get("problem_statement", "") or ""
            for r in ds if r["instance_id"] in set(iids)}


def select_instances(all_iids: list[str], mode: str) -> list[str]:
    iids = sorted(all_iids)
    if mode == "smoke":
        random.seed(PILOT_SEED)
        return random.sample(iids, min(SMOKE_N, len(iids)))
    if mode == "pilot":
        random.seed(PILOT_SEED)
        return random.sample(iids, min(PILOT_N, len(iids)))
    return iids


# ── API call ──────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text.strip()


async def _one_call(client, deployment: str, sem: asyncio.Semaphore,
                    prompt: str, reasoning_effort: str) -> str:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.responses.create(
                    model=deployment,
                    input=[{"role": "user", "content": prompt}],
                    reasoning={"effort": reasoning_effort},
                    max_output_tokens=16000,
                )
                return resp.output_text or ""
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
    return ""


def _parse(text: str, swapped: bool = False) -> dict | None:
    try:
        obj = json.loads(_strip_fences(text))
        if not isinstance(obj, dict):
            return None
        winner = str(obj.get("winner") or "").upper()
        if winner not in ("A", "B", "TIE"):
            return None
        # De-swap: if prompt was B/A, flip the winner back
        if swapped and winner == "A":
            winner = "B"
        elif swapped and winner == "B":
            winner = "A"
        return {
            "winner":     winner,
            "confidence": float(obj.get("confidence") or 0.5),
            "reasoning":  str(obj.get("reasoning") or "")[:400],
        }
    except Exception:
        return None


# ── Main async runner ─────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    from openai import AsyncAzureOpenAI

    creds = _load_creds()
    deployment = creds["deployment"]
    reasoning_effort = os.environ.get("AZURE_REASONING_EFFORT", "high")

    patches   = load_patches()
    all_iids  = sorted(patches)
    instances = select_instances(all_iids, args.mode)
    all_pairs = list(combinations(K_LIST, 2))  # 28 pairs

    print(f"[blind_comparator] mode={args.mode}  instances={len(instances)}  "
          f"pairs={len(all_pairs)}  R={args.votes}  swap={args.position_swap}  "
          f"model={deployment}  reasoning={reasoning_effort}")

    issues = load_issues(instances)

    out_dir  = ROOT / "outputs" / f"blind_cmp_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pairwise_votes.jsonl"

    # Resume: load done keys
    done: set[tuple[str, int, int]] = set()
    if out_path.exists():
        for line in out_path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["instance_id"], int(r["patch_a_idx"]), int(r["patch_b_idx"])))
    print(f"[blind_comparator] {len(done)} pairs already done — skipping")

    sem       = asyncio.Semaphore(args.concurrency)
    budget    = {"spent": 0.0, "calls": 0}
    fout      = out_path.open("a")
    lock      = asyncio.Lock()
    t0        = time.time()
    completed = [0]

    client = AsyncAzureOpenAI(
        api_key=creds["api_key"],
        azure_endpoint=creds["azure_endpoint"],
        api_version=creds["api_version"],
    )

    async def do_pair(iid: str, a: int, b: int) -> None:
        p_iid = patches.get(iid, {})
        patch_a = p_iid.get(a, "")
        patch_b = p_iid.get(b, "")
        if not patch_a or not patch_b:
            return

        issue = issues.get(iid, "")
        votes = []

        for vote_idx in range(args.votes):
            # Forward: A=a, B=b
            prompt_fwd = PROMPT.format(
                issue_text=issue[:3000],
                patch_a=patch_a[:3500],
                patch_b=patch_b[:3500],
            )
            text_fwd = await _one_call(client, deployment, sem, prompt_fwd, reasoning_effort)
            parsed_fwd = _parse(text_fwd, swapped=False)

            async with lock:
                budget["calls"] += 1

            vote = {
                "vote_idx": vote_idx,
                "primary": {"parsed": parsed_fwd or {"winner": None, "confidence": 0.5}},
            }

            if args.position_swap:
                # Swapped: A=b, B=a → de-swap winner
                prompt_bak = PROMPT.format(
                    issue_text=issue[:3000],
                    patch_a=patch_b[:3500],
                    patch_b=patch_a[:3500],
                )
                text_bak = await _one_call(client, deployment, sem, prompt_bak, reasoning_effort)
                parsed_bak = _parse(text_bak, swapped=True)
                async with lock:
                    budget["calls"] += 1
                vote["secondary"] = {"parsed": parsed_bak or {"winner": None, "confidence": 0.5}}

            votes.append(vote)

        row = {
            "instance_id":  iid,
            "patch_a_idx":  a,
            "patch_b_idx":  b,
            "votes":        votes,
        }
        async with lock:
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            completed[0] += 1
            if completed[0] % 20 == 0:
                elapsed = time.time() - t0
                rate = completed[0] / elapsed * 60 if elapsed > 0 else 0
                print(f"[heartbeat] completed={completed[0]} "
                      f"rate={rate:.0f}/min cost=${budget['spent']:.2f}")
            if budget["spent"] > BUDGET_CAP:
                print(f"[BUDGET] ${budget['spent']:.2f} exceeds cap — stopping")
                raise SystemExit(1)

    tasks = []
    for iid in instances:
        if iid not in patches:
            continue
        for a, b in all_pairs:
            if (iid, a, b) in done:
                continue
            if a not in patches[iid] or b not in patches[iid]:
                continue
            tasks.append(do_pair(iid, a, b))

    print(f"[blind_comparator] {len(tasks)} pairs to process")
    await asyncio.gather(*tasks)
    fout.close()
    print(f"[blind_comparator] done  completed={completed[0]}  "
          f"cost=${budget['spent']:.2f}  output={out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="azure/gpt-5.4-nano")
    p.add_argument("--mode",          choices=["smoke", "pilot", "full"], default="pilot")
    p.add_argument("--votes",         type=int, default=5)
    p.add_argument("--concurrency",   type=int, default=64)
    p.add_argument("--position-swap", action="store_true", default=True)
    p.add_argument("--no-position-swap", dest="position_swap", action="store_false")
    p.add_argument("--run-id",        default=None)
    args = p.parse_args()

    if args.run_id is None:
        slug = args.model.replace("/", "_").replace(".", "").replace("-", "_")
        args.run_id = f"{slug}_{args.mode}"

    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
