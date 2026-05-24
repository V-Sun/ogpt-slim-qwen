#!/usr/bin/env python3
"""Trace-aware binary patch selector.

Binary yes/no classifier (like direct_binary_dynamic_hints.py) augmented with
rich proposer trajectory context: THOUGHT reasoning, bash commands run, and
errors encountered by the agent. Pre-eval blind — no post-patch harness results.

Trajectory context comes from .traj.json files in the proposer shard dirs.
It captures HOW the agent arrived at the patch, giving the model insight into
agent confidence, the code paths explored, and whether the agent hit errors.

Output: binary_votes.jsonl — same format as direct_binary_dynamic_hints.py,
compatible with all existing aggregation and plotting scripts.

Usage:
    python3 trace_selector.py --all --model azure/gpt-5.4-nano --votes-per-patch 3 \\
        --k-list 1,3,5,7,8,9,12,15 --concurrency 256 \\
        --output-dir outputs/trace_selector_nano_greedy8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ceiling as rc

REPO = Path(__file__).resolve().parent
DATA_REPO = Path("/home/vsun/orchestra-gpt")
DEFAULT_PATCHES_DIR = DATA_REPO / "outputs" / "oracle_preds_v3"
DEFAULT_OUTPUT_DIR = REPO / "outputs" / "trace_selector_greedy8"
TRAJ_SHARD_DIRS = sorted((DATA_REPO / "outputs").glob("propsonly_xhigh_500_shard_*"))

GREEDY8_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
CONCURRENCY = int(os.getenv("TRACE_SELECTOR_CONCURRENCY", "256"))
VOTES_PER_PATCH = 3
MAX_OUTPUT_TOKENS = 2048
HARD_BUDGET_USD = 250.0

# Max chars per trace section
MAX_THOUGHTS = 1200
MAX_COMMANDS = 600
MAX_ERRORS = 600

_DIFF_FILE_RE = re.compile(r'^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)', re.MULTILINE)


# ---------------------------------------------------------------------------
# Trajectory index + extraction (mirrors trace_comparator.py)
# ---------------------------------------------------------------------------

def build_traj_index() -> dict[tuple[str, int], Path]:
    index: dict[tuple[str, int], Path] = {}
    for shard_dir in TRAJ_SHARD_DIRS:
        for task_dir in shard_dir.iterdir():
            if not task_dir.is_dir():
                continue
            iid = task_dir.name
            for tf in task_dir.glob("proposer_*.traj.json"):
                try:
                    k = int(tf.stem.replace("proposer_", "").replace(".traj", ""))
                except ValueError:
                    continue
                index[(iid, k)] = tf
    return index


def extract_trace_context(traj_path: Path, max_steps: int = 40) -> dict[str, str]:
    try:
        d = json.load(traj_path.open())
    except Exception:
        return {"thoughts": "", "commands": "", "errors": ""}

    msgs = d.get("messages", [])
    thoughts: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    step = 0

    for m in msgs:
        if step >= max_steps:
            break
        role = m.get("role", "")

        if role == "assistant":
            step += 1
            output = m.get("output", [])
            if not isinstance(output, list):
                output = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            t = (c.get("text") or "").strip()
                            if t:
                                thoughts.append(t[:400])
                elif item.get("type") == "function_call":
                    raw_args = item.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}
                    cmd = (args.get("command") or "").strip()
                    if cmd:
                        commands.append(cmd[:200])

        elif role == "tool":
            raw = m.get("extra", {}).get("raw_output") or m.get("content") or ""
            if isinstance(raw, list):
                raw = " ".join(str(x) for x in raw)
            raw = str(raw)
            low = raw.lower()
            if any(kw in low for kw in ("error", "traceback", "exception", "failed", "errno")):
                errors.append(raw[:300])

    thoughts_str = "\n".join(thoughts)[:MAX_THOUGHTS]
    commands_str = "\n".join(f"$ {c}" for c in commands)[:MAX_COMMANDS]
    errors_str = "\n---\n".join(errors)[:MAX_ERRORS]
    return {"thoughts": thoughts_str, "commands": commands_str, "errors": errors_str}


def format_trace(ctx: dict[str, str]) -> str:
    parts = []
    if ctx.get("thoughts"):
        parts.append(f"Agent reasoning:\n{ctx['thoughts']}")
    if ctx.get("commands"):
        parts.append(f"Commands run:\n{ctx['commands']}")
    if ctx.get("errors"):
        parts.append(f"Errors encountered:\n{ctx['errors']}")
    return "\n\n".join(parts) if parts else "(no trajectory available)"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """You are evaluating a single software patch against the GitHub issue it claims to fix.

Answer YES if running the patched code would pass the failing test(s) and not break passing tests.
Answer NO otherwise.

You have access to:
  - The issue text
  - The exact failing test names (FAIL_TO_PASS)
  - The test source code (test_patch) showing what the correct fix must achieve
  - The agent's trajectory — how this patch was produced: reasoning, commands, errors
  - The files this patch modifies
  - The proposed patch diff

Use the trajectory to assess agent confidence: did the agent explore the relevant code path?
Did it encounter errors suggesting its fix may be incomplete? Did it reason correctly about
what the failing test requires?

ISSUE:
{problem_statement}

FAILING TESTS (must pass after patch):
{fail_to_pass}

TEST SOURCE (defines what a passing patch must achieve):
{test_patch}

AGENT TRAJECTORY (how this patch was produced):
{trace_context}

FILES MODIFIED BY PATCH:
{patch_files}

PROPOSED PATCH:
{patch}

Respond with EXACTLY this JSON, no markdown fences, no extra text:
{{
  "resolves": <true|false>,
  "reasoning": "<one sentence: does the patch address the root cause the test exercises>",
  "confidence": <integer 1-5>
}}"""


def parse_binary(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(rc._strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "resolves" not in obj or obj["resolves"] not in (True, False):
        return None
    return {
        "resolves": bool(obj["resolves"]),
        "reasoning": (obj.get("reasoning") or "")[:500],
        "confidence": int(obj.get("confidence", 0) or 0),
    }


def patch_files(patch_text: str, max_files: int = 5) -> list[str]:
    if not patch_text:
        return []
    seen, seen_set = [], set()
    for m in _DIFF_FILE_RE.finditer(patch_text):
        path = m.group(1)
        if path in ("/dev/null", "") or path in seen_set:
            continue
        seen.append(path)
        seen_set.add(path)
        if len(seen) >= max_files:
            break
    return seen


# ---------------------------------------------------------------------------
# Per-vote driver
# ---------------------------------------------------------------------------

async def _do_one_vote(
    client, deployment, sem,
    iid: str, k: int, vote_idx: int,
    prompt: str, out_dir: Path, budget,
) -> dict:
    budget.issued += 1
    result = await rc._api_call(
        client, deployment, prompt,
        temperature=0.7, max_output_tokens=MAX_OUTPUT_TOKENS, sem=sem,
    )
    budget.add("critic", result)
    rc.append_usage(
        out_dir, "critic",
        {"iid": iid, "patch_idx": k, "vote_idx": vote_idx},
        result, "trace_selector", 0.7,
    )
    if result.error:
        budget.abstains_critic += 1
        return {
            "vote_idx": vote_idx, "resolves": None, "confidence": 0,
            "reasoning": "", "abstain_reason": result.error,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "reasoning_tokens": result.reasoning_tokens,
        }
    parsed = parse_binary(result.text)
    if parsed is None:
        budget.abstains_critic += 1
        return {
            "vote_idx": vote_idx, "resolves": None, "confidence": 0,
            "reasoning": "", "abstain_reason": "parse_failed",
            "raw_response": result.text[:500],
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "reasoning_tokens": result.reasoning_tokens,
        }
    return {
        "vote_idx": vote_idx,
        "resolves": parsed["resolves"],
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"],
        "raw_response": result.text[:500],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_patches(patches_dir: Path, k_list: list[int]) -> tuple[dict[str, dict[int, str]], list[str]]:
    by_iid: dict[str, dict[int, str]] = {}
    for k in k_list:
        f = patches_dir / f"proposer_{k}.jsonl"
        if not f.exists():
            raise FileNotFoundError(f)
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                iid = row["instance_id"]
                by_iid.setdefault(iid, {})[k] = row.get("model_patch", "") or ""
    return by_iid, sorted(by_iid)


def load_dataset_fields(iids: list[str]) -> dict[str, dict]:
    from datasets import load_dataset
    wanted = set(iids)
    out: dict[str, dict] = {}
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    for inst in ds:
        iid = inst["instance_id"]
        if iid not in wanted:
            continue
        ftp_raw = inst.get("FAIL_TO_PASS", "[]")
        try:
            ftp = json.loads(ftp_raw) if isinstance(ftp_raw, str) else list(ftp_raw)
        except Exception:
            ftp = []
        out[iid] = {
            "problem_statement": inst.get("problem_statement", "") or "",
            "ftp": [n for n in ftp if isinstance(n, str)],
            "test_patch": inst.get("test_patch", "") or "",
        }
    return out


def load_truth(reports_dir: Path, k_list: list[int]) -> dict[str, dict[int, bool]]:
    truth: dict[str, dict[int, bool]] = {}
    for k in k_list:
        for rep in reports_dir.glob(f"oracle_*_proposer_{k}/*/*/report.json"):
            try:
                obj = json.load(rep.open())
            except Exception:
                continue
            iid = rep.parent.name
            rec = obj.get(iid, obj) if isinstance(obj, dict) else {}
            if isinstance(rec, dict):
                truth.setdefault(iid, {})[k] = bool(rec.get("resolved", False))
    return truth


def load_done_keys(jsonl: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not jsonl.exists():
        return done
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
                done.add((r["instance_id"], int(r["patch_idx"])))
            except Exception:
                continue
    return done


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def run(args, instances, patches_by_iid, fields_by_iid, traj_index, truth) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "binary_votes.jsonl"
    done = load_done_keys(out_jsonl)

    env = rc._require_env()
    client = rc._make_client(env)
    deployment = args.model.split("/", 1)[1] if args.model.startswith("azure/") else args.model
    if not deployment:
        deployment = env["deployment"]
    sem = asyncio.Semaphore(CONCURRENCY)
    budget = rc.Budget()
    hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))

    print(f"[trace-sel] pre-extracting trajectories for "
          f"{len(instances)} instances × {len(args.k_indices)} K's…")
    trace_cache: dict[tuple[str, int], dict[str, str]] = {}
    n_found = 0
    for iid in instances:
        for k in args.k_indices:
            tf = traj_index.get((iid, k))
            if tf and tf.exists():
                trace_cache[(iid, k)] = extract_trace_context(tf)
                n_found += 1
            else:
                trace_cache[(iid, k)] = {}
    print(f"[trace-sel] {n_found}/{len(instances)*len(args.k_indices)} trajectories found")

    # Build all (iid, k) tasks
    task_keys: list[tuple[str, int, int]] = []
    coros = []
    pk_truth: dict[tuple[str, int], bool | None] = {}

    for iid in instances:
        df = fields_by_iid.get(iid, {})
        patches = patches_by_iid.get(iid, {})
        ftp = df.get("ftp", [])
        ftp_str = "\n".join(f"  - {n}" for n in ftp[:10]) or "  (none)"
        test_patch_str = (df.get("test_patch") or "")[:2500] or "(not available)"
        problem = (df.get("problem_statement") or "")[:2000]

        for k in args.k_indices:
            if (iid, k) in done:
                continue
            patch = patches.get(k, "") or ""
            pk_truth[(iid, k)] = truth.get(iid, {}).get(k)

            if not patch.strip():
                row = {
                    "instance_id": iid, "patch_idx": k,
                    "patch_resolves": pk_truth[(iid, k)],
                    "binary_votes": [
                        {"vote_idx": v, "resolves": None, "confidence": 0,
                         "reasoning": "", "abstain_reason": "empty_patch"}
                        for v in range(args.votes_per_patch)
                    ],
                }
                with out_jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done.add((iid, k))
                continue

            ctx = trace_cache.get((iid, k), {})
            trace_str = format_trace(ctx)
            files = patch_files(patch)
            files_str = "\n".join(f"  - {p}" for p in files) or "  (none)"

            prompt = PROMPT.format(
                problem_statement=problem,
                fail_to_pass=ftp_str,
                test_patch=test_patch_str,
                trace_context=trace_str,
                patch_files=files_str,
                patch=patch[:8000],
            )

            for vi in range(args.votes_per_patch):
                task_keys.append((iid, k, vi))
                coros.append(_do_one_vote(
                    client, deployment, sem, iid, k, vi, prompt, out_dir, budget,
                ))

    print(f"[trace-sel] {len(coros)} API calls queued ({len(instances)} instances, "
          f"{len(args.k_indices)} K's, {args.votes_per_patch} votes each)")
    if not coros:
        print("[trace-sel] nothing to do (all done)")
        hb.cancel()
        return

    async def _keyed(key, coro):
        return key, await coro

    futures = [asyncio.ensure_future(_keyed(key, c)) for key, c in zip(task_keys, coros)]
    pending: dict[tuple[str, int], dict[int, dict]] = {}
    halted = False
    completed = 0

    for fut in asyncio.as_completed(futures):
        if budget.check_halt() and not halted:
            print(f"[trace-sel] HALT: budget ${budget.spent_usd:.2f} >= ${HARD_BUDGET_USD}")
            halted = True
            for f in futures:
                if not f.done():
                    f.cancel()
        try:
            (iid, k, vi), vote = await fut
        except asyncio.CancelledError:
            continue

        bucket = pending.setdefault((iid, k), {})
        bucket[vi] = vote
        completed += 1

        if len(bucket) == args.votes_per_patch:
            votes = [bucket[v] for v in range(args.votes_per_patch)]
            row = {
                "instance_id": iid, "patch_idx": k,
                "patch_resolves": pk_truth.get((iid, k)),
                "binary_votes": votes,
            }
            with out_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done.add((iid, k))
            del pending[(iid, k)]

        if completed % 500 == 0:
            print(f"[trace-sel] {completed}/{len(coros)} votes done, "
                  f"cost=${budget.spent_usd:.2f}")

    hb.cancel()
    try:
        await hb
    except asyncio.CancelledError:
        pass

    print(f"[trace-sel] done: {completed} votes, {budget.n_calls} API calls, "
          f"${budget.spent_usd:.2f}, abstains={budget.abstains_critic}")
    summarize(out_jsonl, args.k_indices, truth, out_dir)


def summarize(jsonl: Path, k_list: list[int], truth: dict, out_dir: Path) -> None:
    rows_by_iid: dict[str, list[dict]] = defaultdict(list)
    with jsonl.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows_by_iid[r["instance_id"]].append(r)

    resolved_most_yes = total_most_yes = 0
    resolved_conf = total_conf = 0

    for iid, rows in rows_by_iid.items():
        scores: dict[int, int] = {}
        conf_scores: dict[int, float] = {}
        for r in rows:
            k = int(r["patch_idx"])
            votes = r.get("binary_votes", [])
            yes = sum(1 for v in votes if v.get("resolves") is True)
            avg_conf = sum(v.get("confidence", 0) for v in votes if v.get("resolves") is True)
            scores[k] = yes
            conf_scores[k] = avg_conf

        if not scores:
            continue

        # most-yes selection
        best_k = max(scores, key=lambda k: (scores[k], k))
        if iid in truth and best_k in truth[iid]:
            total_most_yes += 1
            if truth[iid][best_k]:
                resolved_most_yes += 1

        # confidence-weighted selection
        best_k_conf = max(conf_scores, key=lambda k: (conf_scores[k], k))
        if iid in truth and best_k_conf in truth[iid]:
            total_conf += 1
            if truth[iid][best_k_conf]:
                resolved_conf += 1

    rate_my = resolved_most_yes / total_most_yes if total_most_yes else 0.0
    rate_cf = resolved_conf / total_conf if total_conf else 0.0
    print(f"\n[trace-sel] most-yes:   {resolved_most_yes}/{total_most_yes} = {rate_my:.4f}")
    print(f"[trace-sel] conf-weighted: {resolved_conf}/{total_conf} = {rate_cf:.4f}")

    summary = {
        "most_yes": {"resolved": resolved_most_yes, "total": total_most_yes, "resolve_rate": rate_my},
        "confidence_weighted": {"resolved": resolved_conf, "total": total_conf, "resolve_rate": rate_cf},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true",
                   help="Run on all instances in patches dir")
    p.add_argument("--instances-path", default="",
                   help="JSON file with list of instance IDs to run")
    p.add_argument("--patches-dir", default=str(DEFAULT_PATCHES_DIR))
    p.add_argument("--harness-reports-dir",
                   default=str(DATA_REPO / "logs" / "run_evaluation"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--model", default=os.getenv("AZURE_DEPLOYMENT", "gpt-5.4-nano"))
    p.add_argument("--votes-per-patch", type=int, default=VOTES_PER_PATCH)
    p.add_argument("--k-list", default=",".join(str(k) for k in GREEDY8_K_LIST))
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--max-instances", type=int, default=0)
    args = p.parse_args()

    args.k_indices = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if not args.k_indices:
        raise SystemExit("FATAL: --k-list is empty")
    if args.concurrency:
        global CONCURRENCY
        CONCURRENCY = args.concurrency

    patches_dir = Path(args.patches_dir)
    reports_dir = Path(args.harness_reports_dir)

    print(f"[load] patches from {patches_dir}")
    patches_by_iid, all_iids = load_patches(patches_dir, args.k_indices)

    if args.instances_path:
        req = json.load(Path(args.instances_path).open())
        instances = [str(x["instance_id"] if isinstance(x, dict) else x) for x in req
                     if (x["instance_id"] if isinstance(x, dict) else x) in set(all_iids)]
    elif args.all:
        instances = list(all_iids)
    else:
        raise SystemExit("FATAL: provide --all or --instances-path")

    if args.max_instances:
        instances = instances[:args.max_instances]

    print(f"[load] {len(instances)} instances, K={args.k_indices}")
    print("[load] dataset fields…")
    fields_by_iid = load_dataset_fields(instances)
    print("[load] truth…")
    truth = load_truth(reports_dir, args.k_indices)
    print("[load] trajectory index…")
    traj_index = build_traj_index()
    print(f"[load] {len(traj_index)} trajectory entries found")

    asyncio.run(run(args, instances, patches_by_iid, fields_by_iid, traj_index, truth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
