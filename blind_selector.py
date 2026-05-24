#!/usr/bin/env python3
"""Blind binary selector — issue text + patch diff only, no execution hints.

Tracks 1 (nano), 3 (mini), 5 (5.4 full) of the scope-reset experiment plan.

Usage:
    python3 blind_selector.py --model azure/gpt-5.4-nano --mode smoke
    python3 blind_selector.py --model azure/gpt-5.4-nano --mode pilot
    python3 blind_selector.py --model azure/gpt-5.4-nano --mode full

Outputs:
    outputs/blind_sel_{model_slug}_{mode}/binary_votes.jsonl
    One row per (instance_id, patch_idx), binary_votes list embedded.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
PROPOSALS    = ROOT / "outputs" / "canonical_greedy8" / "proposals"
K_LIST       = list(range(8))
PILOT_SEED   = 42
PILOT_N      = 50
SMOKE_N      = 5
BUDGET_CAP   = 400.0   # hard stop across entire session

PROMPT = """\
You are evaluating whether a code patch correctly fixes the GitHub issue described below.

ISSUE:
{issue_text}

PROPOSED PATCH:
{patch_diff}

Does this patch fix the issue?

Respond with EXACTLY this JSON, no markdown fences, no extra text:
{{"resolves": true|false, "confidence": <integer 1-5>, "reasoning": "<one sentence>"}}
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
    """Return {iid: {k: patch_diff}}."""
    out: dict[str, dict[int, str]] = {}
    for k in K_LIST:
        p = PROPOSALS / f"proposer_{k}.jsonl"
        if not p.exists():
            print(f"[warn] {p} not found, skipping K={k}")
            continue
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            iid   = r["instance_id"]
            patch = (r.get("model_patch") or "").strip()
            if patch:
                out.setdefault(iid, {})[k] = patch
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
    return iids  # full


# ── API call ──────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text.strip()


async def _one_call(client, deployment: str, sem: asyncio.Semaphore,
                    prompt: str, reasoning_effort: str) -> tuple[str, float]:
    """Return (raw_text, cost_usd_estimate)."""
    from openai import AsyncAzureOpenAI  # noqa: F811
    async with sem:
        for attempt in range(3):
            try:
                kwargs: dict = dict(
                    model=deployment,
                    input=[{"role": "user", "content": prompt}],
                    reasoning={"effort": reasoning_effort},
                    max_output_tokens=16000,
                )
                resp = await client.responses.create(**kwargs)
                text = resp.output_text or ""
                # Rough cost estimate: nano ~$0.0001/call, mini ~$0.001, full ~$0.01
                cost = 0.0
                return text, cost
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0.0


def _parse(text: str) -> dict | None:
    try:
        obj = json.loads(_strip_fences(text))
        if not isinstance(obj, dict):
            return None
        if obj.get("resolves") not in (True, False):
            return None
        return {
            "resolves":   bool(obj["resolves"]),
            "confidence": int(obj.get("confidence") or 0),
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

    patches  = load_patches()
    all_iids = sorted(patches)
    instances = select_instances(all_iids, args.mode)
    print(f"[blind_selector] mode={args.mode}  instances={len(instances)}  "
          f"model={deployment}  reasoning={reasoning_effort}  R={args.votes}")

    issues = load_issues(instances)

    out_dir = ROOT / "outputs" / f"blind_sel_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "binary_votes.jsonl"

    # Load already-done keys to support resume
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        for line in out_path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["instance_id"], int(r["patch_idx"])))
    print(f"[blind_selector] {len(done)} (iid, k) pairs already done — skipping")

    sem    = asyncio.Semaphore(args.concurrency)
    budget = {"spent": 0.0, "calls": 0}
    fout   = out_path.open("a")
    lock   = asyncio.Lock()
    t0     = time.time()
    completed = [0]

    client = AsyncAzureOpenAI(
        api_key=creds["api_key"],
        azure_endpoint=creds["azure_endpoint"],
        api_version=creds["api_version"],
    )

    async def do_pair(iid: str, k: int) -> None:
        if (iid, k) not in patches.get(iid, {}) and k not in patches.get(iid, {}):
            return
        patch = patches[iid].get(k, "")
        if not patch:
            return
        issue = issues.get(iid, "")

        votes = []
        for vote_idx in range(args.votes):
            prompt = PROMPT.format(
                issue_text=issue[:3000],
                patch_diff=patch[:4000],
            )
            text, cost = await _one_call(client, deployment, sem, prompt, reasoning_effort)
            parsed = _parse(text)
            async with lock:
                budget["spent"] += cost
                budget["calls"] += 1
            votes.append({
                "vote_idx":   vote_idx,
                "resolves":   parsed["resolves"] if parsed else None,
                "confidence": parsed["confidence"] if parsed else 0,
                "reasoning":  parsed["reasoning"] if parsed else "",
                "abstain_reason": None if parsed else "parse_fail",
            })

        row = {
            "instance_id":   iid,
            "patch_idx":     k,
            "binary_votes":  votes,
            "patch_resolves": None,  # filled by offline scoring
        }
        async with lock:
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            completed[0] += 1
            if completed[0] % 50 == 0:
                elapsed = time.time() - t0
                rate = completed[0] / elapsed * 60 if elapsed > 0 else 0
                print(f"[heartbeat] completed={completed[0]} "
                      f"rate={rate:.0f}/min cost=${budget['spent']:.2f}")
            if budget["spent"] > BUDGET_CAP:
                print(f"[BUDGET] ${budget['spent']:.2f} exceeds cap ${BUDGET_CAP} — stopping")
                raise SystemExit(1)

    tasks = []
    for iid in instances:
        if iid not in patches:
            continue
        for k in K_LIST:
            if (iid, k) in done:
                continue
            if k not in patches[iid]:
                continue
            tasks.append(do_pair(iid, k))

    print(f"[blind_selector] {len(tasks)} (iid, k) pairs to process")
    await asyncio.gather(*tasks)
    fout.close()
    print(f"[blind_selector] done  completed={completed[0]}  "
          f"cost=${budget['spent']:.2f}  output={out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="azure/gpt-5.4-nano",
                   help="Model string (used for labeling only; set AZURE_DEPLOYMENT env var)")
    p.add_argument("--mode",        choices=["smoke", "pilot", "full"], default="pilot")
    p.add_argument("--votes",       type=int, default=5)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--run-id",      default=None,
                   help="Output dir suffix (defaults to model slug + mode)")
    args = p.parse_args()

    if args.run_id is None:
        slug = args.model.replace("/", "_").replace(".", "").replace("-", "_")
        args.run_id = f"{slug}_{args.mode}"

    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
