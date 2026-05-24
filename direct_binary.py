"""Direct-binary classifier — single-patch yes/no resolves judgment.

Frame test for the comparator failure: instead of the structural pairwise
comparison, ask each patch in isolation "does this resolve the issue?"
R=5 votes per patch, no critic gate, no Copeland tournament. Pick the
patch with the most "yes" votes per instance, tiebreak by lowest k_index.

This script is deliberately separate from run_ceiling.py — it imports the
async API plumbing but adds nothing to the multi-stage pipeline.

Settings (matched to the existing diversity-run pilot for apples-to-apples):
  K = 0..7   (8 patches per instance)
  R = 5      (votes per patch)
  pilot = 50 instances (from pilot_instance_subset)

Total calls: 50 × 8 × 5 = 2,000.

Caches:
  outputs/direct_binary_pilot/binary_votes.jsonl  (1 row per (iid, k))
  outputs/direct_binary_pilot/usage.jsonl         (1 row per API call)

Run:
  python3 direct_binary.py --smoke 1   # 1-instance × 8 patches × R=5 = 40 calls
  python3 direct_binary.py --full      # all 50 pilot instances
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Import the same async plumbing run_ceiling.py uses
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ceiling as rc

REPO = Path(__file__).resolve().parent

# Output paths
OUT_DIR = REPO / "outputs" / "direct_binary_pilot"
SMOKE_DIR = REPO / "outputs" / "direct_binary_smoke"

# Experiment scale
K_PATCHES = 8
R_VOTES = 5
GLOBAL_SEED = 42

# Reuse run_ceiling's constants
CONCURRENCY = rc.CONCURRENCY
PRICE_INPUT_PER_M = rc.PRICE_INPUT_PER_M
PRICE_OUTPUT_PER_M = rc.PRICE_OUTPUT_PER_M
PARSE_RETRY_MAX = rc.PARSE_RETRY_MAX
PARSE_RETRY_TEMP_BUMP = rc.PARSE_RETRY_TEMP_BUMP
BINARY_MAX_OUTPUT = 4096   # smaller than critic since output JSON is tiny
HARD_BUDGET_USD = 100.00   # this experiment is small, tight cap

# ----------------------------------------------------------------------------
# Prompt + parser
# ----------------------------------------------------------------------------


BINARY_PROMPT = """You are evaluating a single patch against the GitHub issue it claims to fix.
Output yes if and only if running the patched code passes the failing test
described in the issue and does not break any test that previously passed.

Trace the failure path from the issue to the originating code. Check whether
the patch modifies that code path. Then state whether the patch produces the
expected behavior on the smallest input that exhibits the failure.

ISSUE:
{problem_statement}

PROPOSED PATCH:
{patch}

Respond with EXACTLY this JSON, no markdown fences, no extra text:
{{
  "resolves": <true|false>,
  "reasoning": "<one sentence: smallest input under original code produces X; under patch produces Y>",
  "confidence": <integer 1-5>
}}
"""


def parse_binary(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(rc._strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "resolves" not in obj:
        return None
    if obj["resolves"] not in (True, False):
        return None
    return {
        "resolves": bool(obj["resolves"]),
        "reasoning": (obj.get("reasoning") or "")[:500],
        "confidence": int(obj.get("confidence", 0) or 0),
    }


# ----------------------------------------------------------------------------
# Per-vote driver
# ----------------------------------------------------------------------------


async def _do_one_binary_vote(
    client, deployment, sem, problem, patch,
    iid: str, k: int, vote_idx: int,
    out_dir: Path, budget: rc.Budget,
) -> dict:
    base_temp = 0.7
    prompt = BINARY_PROMPT.format(
        problem_statement=problem[:2000], patch=patch[:8000]
    )
    last_text = ""
    last_result = None
    for attempt in range(PARSE_RETRY_MAX):
        temp = min(1.5, base_temp + attempt * PARSE_RETRY_TEMP_BUMP)
        budget.issued += 1
        result = await rc._api_call(
            client, deployment, prompt,
            temperature=temp,
            max_output_tokens=BINARY_MAX_OUTPUT,
            sem=sem,
        )
        budget.add("critic", result)  # bucket as critic-side spending
        rc.append_usage(
            out_dir, "critic",  # kind="critic" for usage.jsonl bucketing
            {"iid": iid, "patch_idx": k, "vote_idx": vote_idx, "attempt": attempt},
            result, "direct_binary", temp,
        )
        last_result = result
        last_text = result.text
        if result.error:
            continue
        parsed = parse_binary(result.text)
        if parsed is not None:
            return {
                "vote_idx": vote_idx,
                "temperature": base_temp,
                "resolves": parsed["resolves"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "raw_response": result.text[:2000],
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_tokens": result.reasoning_tokens,
            }
    # All retries failed — abstain
    budget.abstains_critic += 1
    return {
        "vote_idx": vote_idx,
        "temperature": base_temp,
        "resolves": None,
        "confidence": 0,
        "reasoning": "",
        "raw_response": last_text[:2000],
        "abstain_reason": (last_result.error if last_result and last_result.error
                           else "parse_failed"),
        "input_tokens": last_result.input_tokens if last_result else 0,
        "output_tokens": last_result.output_tokens if last_result else 0,
        "reasoning_tokens": last_result.reasoning_tokens if last_result else 0,
    }


# ----------------------------------------------------------------------------
# Stage runner: streaming as_completed across (iid, k, vote_idx)
# ----------------------------------------------------------------------------


def load_done_keys(jsonl: Path) -> set[tuple[str, int]]:
    done = set()
    if not jsonl.exists():
        return done
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["instance_id"], int(r["patch_idx"])))
    return done


async def run_binary(
    client, deployment, instances, patches_by_iid,
    resolve_by_iid, problems_by_iid, out_dir: Path,
    budget: rc.Budget,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "binary_votes.jsonl"
    done_keys = load_done_keys(out_jsonl)
    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.time()

    task_keys = []
    coros = []
    pk_resolves = {}
    for iid in instances:
        problem = problems_by_iid.get(iid, "")
        patches = patches_by_iid.get(iid, [""] * K_PATCHES)
        for k in range(K_PATCHES):
            if (iid, k) in done_keys:
                continue
            patch = patches[k] or ""
            patch_resolves = resolve_by_iid.get(iid, {}).get(k, None)
            pk_resolves[(iid, k)] = patch_resolves
            if not patch.strip():
                # Empty patch: no API calls, write abstain row
                row = {
                    "instance_id": iid,
                    "patch_idx": k,
                    "patch_resolves": patch_resolves,
                    "binary_votes": [
                        {"vote_idx": v, "resolves": None, "confidence": 0,
                         "reasoning": "", "abstain_reason": "empty_patch"}
                        for v in range(R_VOTES)
                    ],
                }
                with out_jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done_keys.add((iid, k))
                continue
            for vi in range(R_VOTES):
                task_keys.append((iid, k, vi))
                coros.append(_do_one_binary_vote(
                    client, deployment, sem, problem, patch,
                    iid, k, vi, out_dir, budget,
                ))

    print(f"[binary] {len(coros)} API calls queued across {len(instances)} instances")
    if not coros:
        return

    async def _keyed(key, coro):
        return key, await coro
    futures = [asyncio.ensure_future(_keyed(k, c))
               for k, c in zip(task_keys, coros)]

    pending: dict[tuple[str, int], dict[int, dict]] = {}
    halted = False
    completed = 0
    for fut in asyncio.as_completed(futures):
        if budget.check_halt() and not halted:
            print(f"[binary] HALT: budget ${budget.spent_usd:.2f} ≥ ${HARD_BUDGET_USD}")
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
        if len(bucket) == R_VOTES:
            votes = [bucket[v] for v in range(R_VOTES)]
            row = {
                "instance_id": iid,
                "patch_idx": k,
                "patch_resolves": pk_resolves[(iid, k)],
                "binary_votes": votes,
            }
            with out_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_keys.add((iid, k))
            del pending[(iid, k)]
        completed += 1

    elapsed = time.time() - t_start
    print(f"[binary] done: {completed} vote-events, "
          f"{budget.n_calls} API calls, ${budget.spent_usd:.2f} cost, "
          f"{elapsed:.0f}s wall-clock, abstains={budget.abstains_critic}")


# ----------------------------------------------------------------------------
# Inline analysis
# ----------------------------------------------------------------------------


def analyze(out_dir: Path, label: str = "direct-binary"):
    jsonl = out_dir / "binary_votes.jsonl"
    if not jsonl.exists():
        print(f"[analyze] no cache at {jsonl}")
        return
    by_iid: dict[str, list[dict]] = {}
    with jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_iid.setdefault(r["instance_id"], []).append(r)

    iids = sorted(by_iid.keys())
    n_inst = len(iids)
    n_resolved_majority = 0      # most-yes-votes, lowest-k tiebreak
    n_resolved_confidence = 0    # confidence-weighted
    yes_counts = []
    abstain_count = 0

    for iid in iids:
        rows = sorted(by_iid[iid], key=lambda r: r["patch_idx"])
        # Per-patch yes-count, conf-sum-when-yes
        scores_majority = []
        scores_confidence = []
        for row in rows:
            yes = sum(1 for v in row["binary_votes"]
                      if v.get("resolves") is True)
            conf_yes = sum(int(v.get("confidence", 0))
                           for v in row["binary_votes"]
                           if v.get("resolves") is True)
            abstains = sum(1 for v in row["binary_votes"]
                           if v.get("resolves") is None)
            abstain_count += abstains
            yes_counts.append(yes)
            scores_majority.append((row["patch_idx"], yes,
                                    bool(row.get("patch_resolves"))))
            scores_confidence.append((row["patch_idx"], conf_yes,
                                      bool(row.get("patch_resolves"))))
        # Tiebreak: highest score, then lowest k_index
        scores_majority.sort(key=lambda x: (-x[1], x[0]))
        winner_resolves = scores_majority[0][2]
        if winner_resolves:
            n_resolved_majority += 1
        scores_confidence.sort(key=lambda x: (-x[1], x[0]))
        if scores_confidence[0][2]:
            n_resolved_confidence += 1

    # Yes-count histogram across all patches
    from collections import Counter
    hist = Counter(yes_counts)
    print(f"\n=== {label} analysis on {n_inst} instances ===")
    print(f"Per-patch yes-count distribution (out of R={R_VOTES}):")
    for c in range(R_VOTES + 1):
        print(f"  {c} yes votes: {hist.get(c, 0):>4d} patches")
    print(f"Total abstains: {abstain_count} (across {n_inst*K_PATCHES*R_VOTES} possible vote slots)")
    print()
    print(f"Resolve rate (most-yes, tiebreak lowest-k):  {n_resolved_majority}/{n_inst} = {n_resolved_majority/n_inst:.3f}")
    print(f"Resolve rate (confidence-weighted):          {n_resolved_confidence}/{n_inst} = {n_resolved_confidence/n_inst:.3f}")

    # Decision tree
    rate = n_resolved_majority / n_inst
    print()
    print("=" * 60)
    print(f"DECISION (most-yes resolve rate: {rate:.1%})")
    print("=" * 60)
    if rate >= 0.70:
        print(f"  ≥70.0% → framing was the bottleneck.")
        print(f"  RECOMMEND: K-best-8 with binary classifier.")
    elif rate >= 0.67:
        print(f"  67-69% → small framing effect, residual signal weak.")
        print(f"  RECOMMEND: K-best-8 with binary classifier as the main lever.")
    elif rate >= 0.64:
        print(f"  64-66% → framing isn't the issue, but doesn't rule out information-deficit.")
        print(f"  RECOMMEND: execution-hints experiment (~$50) before writing up.")
    else:
        print(f"  <64% → unexpectedly low; investigate parse-fail rate or model issues.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


async def amain(args):
    env = rc._require_env()
    client = rc._make_client(env)

    print("[load] patches…")
    # rc.K_PATCHES might have been overridden globally; force-set ours
    rc.K_PATCHES = K_PATCHES
    patches_by_iid, all_iids = rc.load_patches()
    print(f"[load] {len(all_iids)} instances, K={K_PATCHES}")

    print("[load] resolve booleans…")
    resolve_by_iid = rc.load_resolve_bools()

    print("[load] problem statements…")
    problems_by_iid = rc.load_problem_statements(all_iids)

    pilot = rc.pilot_instance_subset(all_iids)
    print(f"[load] pilot subset: {len(pilot)} instances")

    budget = rc.Budget()
    hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))

    if args.smoke:
        smoke_iids = pilot[:int(args.smoke)]
        print(f"[smoke] {len(smoke_iids)} instances: {smoke_iids}")
        await run_binary(client, env["deployment"], smoke_iids,
                         patches_by_iid, resolve_by_iid, problems_by_iid,
                         SMOKE_DIR, budget)
        analyze(SMOKE_DIR, label=f"smoke ({len(smoke_iids)} inst)")
    elif args.full:
        await run_binary(client, env["deployment"], pilot,
                         patches_by_iid, resolve_by_iid, problems_by_iid,
                         OUT_DIR, budget)
        analyze(OUT_DIR, label="direct-binary full pilot")
    else:
        print("specify --smoke N or --full")
        return 2

    hb.cancel()
    try:
        await hb
    except asyncio.CancelledError:
        pass
    print(f"[done] calls={budget.n_calls} cost=${budget.spent_usd:.2f} "
          f"abstains={budget.abstains_critic}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", type=int, default=0,
                   help="Smoke test on first N pilot instances")
    p.add_argument("--full", action="store_true",
                   help="Run on all 50 pilot instances")
    args = p.parse_args()
    if not (args.smoke or args.full):
        p.print_help()
        return 2
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
