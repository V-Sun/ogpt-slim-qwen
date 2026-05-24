"""Anchored-judgment classifier — 5-category framing of the same direct-binary
question with execution-grounded hints.

Categories:
  CLEAR FIX           → strong yes  (weight +2)
  PARTIAL PROGRESS    → weak yes    (weight +1)
  SAME FAILURE        → no          (weight 0)
  NEW FAILURE         → no          (weight 0)
  INSUFFICIENT EVIDENCE → abstain   (skip)

Aggregation per patch (R=5 votes):
  weighted_yes_score = 2 * (#CLEAR FIX) + 1 * (#PARTIAL PROGRESS)
  Highest score wins, tiebreak lowest k.

Same K=0..7, same hints (FAIL_TO_PASS + error trace + file paths).

Decision tree at pilot:
  ≥74%: scale to full 500
  72-73%: marginal, halt and write up direct-binary as final
  <72%: anchored framing didn't help

Caches:
  outputs/anchored_judgment_smoke/
  outputs/anchored_judgment_pilot/
  outputs/anchored_judgment_full500/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ceiling as rc
import direct_binary_dynamic_hints as dyn

REPO = Path(__file__).resolve().parent
SMOKE_DIR = REPO / "outputs" / "anchored_judgment_smoke"
PILOT_DIR = REPO / "outputs" / "anchored_judgment_pilot"
FULL500_DIR = REPO / "outputs" / "anchored_judgment_full500"

K_PATCHES = 8
R_VOTES = 5
CONCURRENCY = 128  # share with parallel runs
PARSE_RETRY_MAX = rc.PARSE_RETRY_MAX
PARSE_RETRY_TEMP_BUMP = rc.PARSE_RETRY_TEMP_BUMP
ANCHORED_MAX_OUTPUT = 4096
HARD_BUDGET_USD = 200.00

# Category → score
CATEGORY_SCORE = {
    "CLEAR FIX": 2,
    "PARTIAL PROGRESS": 1,
    "SAME FAILURE": 0,
    "NEW FAILURE": 0,
    "INSUFFICIENT EVIDENCE": None,  # abstain
}


# ----------------------------------------------------------------------------
# Anchored prompt
# ----------------------------------------------------------------------------


ANCHORED_PROMPT = """You are evaluating whether a patch materially addressed the failing behavior
described in a GitHub issue. The patch is one of several candidates; your task
is to judge THIS specific patch against the original failure.

Frame your judgment in ONE of five categories:

  - CLEAR FIX: The patch modifies the code path producing the failure, and
    the patched code would yield the issue's described correct behavior on
    the input that exhibits the failure. The failing test would now pass.
  - PARTIAL PROGRESS: The patch addresses some portion of the failure (e.g.
    handles some inputs that triggered the bug, fixes part of a multi-step
    failure, or moves logic in the right direction) but doesn't fully
    eliminate the failing behavior described in the issue.
  - SAME FAILURE: The patch does not change the code path producing the
    failure. Running the failing test would produce the same error. (Patch
    may modify unrelated code, or may reshape code without changing
    behavior on the failing input.)
  - NEW FAILURE: The patch changes behavior in a way that introduces a
    DIFFERENT failure — e.g. signature mismatch breaks callers, the patch's
    new logic raises a different exception, or a previously-passing test
    would now fail.
  - INSUFFICIENT EVIDENCE: You cannot tell from the available text which
    of the above applies. (Use this sparingly — only when the patch is
    truly ambiguous, the trace is missing, or the issue is unclear.)

Below you have:
  - The issue text
  - The exact failing test names (FAIL_TO_PASS)
  - The actual error trace from running the failing test on the unpatched code
  - The list of files the proposed patch modifies
  - The proposed patch

ISSUE:
{problem_statement}

FAILING TESTS:
{fail_to_pass}

ERROR TRACE FROM RUNNING THE FAILING TEST:
{error_trace}

FILES MODIFIED BY PROPOSED PATCH:
{patch_files}

PROPOSED PATCH:
{patch}

Respond with EXACTLY this JSON, no markdown fences:
{{
  "category": "CLEAR FIX" | "PARTIAL PROGRESS" | "SAME FAILURE" | "NEW FAILURE" | "INSUFFICIENT EVIDENCE",
  "reasoning": "<one sentence: link the trace's error origin to what the patch does>",
  "confidence": <integer 1-5>
}}
"""


def parse_anchored(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(rc._strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    cat = obj.get("category")
    if cat not in CATEGORY_SCORE:
        # be lenient: try canonicalizing case
        if isinstance(cat, str):
            cat_up = cat.strip().upper()
            for k in CATEGORY_SCORE:
                if cat_up == k:
                    cat = k
                    break
            else:
                return None
        else:
            return None
    return {
        "category": cat,
        "reasoning": (obj.get("reasoning") or "")[:500],
        "confidence": int(obj.get("confidence", 0) or 0),
    }


# ----------------------------------------------------------------------------
# File-list parser
# ----------------------------------------------------------------------------


_DIFF_FILE_RE = re.compile(r'^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)', re.MULTILINE)


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


# ----------------------------------------------------------------------------
# Per-vote driver
# ----------------------------------------------------------------------------


async def _do_one_vote(
    client, deployment, sem, problem, patch, ftp_str, error_trace, files_str,
    iid, k, vote_idx, out_dir, budget,
):
    base_temp = 0.7
    prompt = ANCHORED_PROMPT.format(
        problem_statement=problem[:2000],
        fail_to_pass=ftp_str,
        error_trace=error_trace[:3000],
        patch_files=files_str,
        patch=patch[:8000],
    )
    last_text = ""
    last_result = None
    for attempt in range(PARSE_RETRY_MAX):
        temp = min(1.5, base_temp + attempt * PARSE_RETRY_TEMP_BUMP)
        budget.issued += 1
        result = await rc._api_call(
            client, deployment, prompt,
            temperature=temp, max_output_tokens=ANCHORED_MAX_OUTPUT, sem=sem,
        )
        budget.add("critic", result)
        rc.append_usage(out_dir, "critic",
                        {"iid": iid, "patch_idx": k, "vote_idx": vote_idx,
                         "attempt": attempt},
                        result, "anchored_judgment", temp)
        last_result, last_text = result, result.text
        if result.error:
            continue
        parsed = parse_anchored(result.text)
        if parsed is not None:
            return {
                "vote_idx": vote_idx,
                "temperature": base_temp,
                "category": parsed["category"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "raw_response": result.text[:2000],
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_tokens": result.reasoning_tokens,
            }
    budget.abstains_critic += 1
    return {
        "vote_idx": vote_idx,
        "temperature": base_temp,
        "category": None,
        "confidence": 0,
        "reasoning": "",
        "raw_response": last_text[:2000],
        "abstain_reason": (last_result.error if last_result and last_result.error
                           else "parse_failed"),
        "input_tokens": last_result.input_tokens if last_result else 0,
        "output_tokens": last_result.output_tokens if last_result else 0,
        "reasoning_tokens": last_result.reasoning_tokens if last_result else 0,
    }


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


async def run_anchored(
    client, deployment, instances, patches_by_iid, resolve_by_iid,
    problems_by_iid, dataset_fields, eval_index, out_dir, budget,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "anchored_votes.jsonl"
    done_keys = load_done_keys(out_jsonl)
    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.time()

    # Pre-extract dynamic hints per instance
    print(f"[hints] extracting error traces for {len(instances)} instances…")
    hint_cache = {}
    n_with_trace = 0
    for iid in instances:
        df = dataset_fields.get(iid, {})
        ftp = df.get("ftp", [])
        ftp_str = "\n".join(f"  - {n}" for n in ftp[:10]) or "  (none)"
        trace, _ = dyn.get_dynamic_hint_for_iid(iid, ftp, eval_index)
        if trace:
            n_with_trace += 1
        else:
            trace = "(no failing-test trace cached — likely all proposers resolved this instance)"
        hint_cache[iid] = (ftp_str, trace)
    print(f"[hints] traces for {n_with_trace}/{len(instances)} instances")

    task_keys = []
    coros = []
    pk_resolves = {}
    for iid in instances:
        ftp_str, trace = hint_cache[iid]
        problem = problems_by_iid.get(iid, "")
        patches = patches_by_iid.get(iid, [""] * K_PATCHES)
        for k in range(K_PATCHES):
            if (iid, k) in done_keys:
                continue
            patch = patches[k] or ""
            patch_resolves = resolve_by_iid.get(iid, {}).get(k, None)
            pk_resolves[(iid, k)] = patch_resolves
            files = patch_files(patch, max_files=5)
            files_str = "\n".join(f"  - {p}" for p in files) or "  (no patch / parse failed)"
            if not patch.strip():
                row = {
                    "instance_id": iid, "patch_idx": k,
                    "patch_resolves": patch_resolves,
                    "anchored_votes": [
                        {"vote_idx": v, "category": None, "confidence": 0,
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
                coros.append(_do_one_vote(
                    client, deployment, sem, problem, patch,
                    ftp_str, trace, files_str,
                    iid, k, vi, out_dir, budget,
                ))

    print(f"[anchored] {len(coros)} API calls queued across {len(instances)} instances")
    if not coros:
        return

    async def _keyed(key, coro):
        return key, await coro
    futures = [asyncio.ensure_future(_keyed(k, c)) for k, c in zip(task_keys, coros)]
    pending = {}
    halted = False
    completed = 0
    for fut in asyncio.as_completed(futures):
        if budget.check_halt() and not halted:
            print(f"[anchored] HALT: budget ${budget.spent_usd:.2f} ≥ ${HARD_BUDGET_USD}")
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
                "instance_id": iid, "patch_idx": k,
                "patch_resolves": pk_resolves[(iid, k)],
                "anchored_votes": votes,
            }
            with out_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_keys.add((iid, k))
            del pending[(iid, k)]
        completed += 1

    elapsed = time.time() - t_start
    print(f"[anchored] done: {completed} vote-events, "
          f"{budget.n_calls} API calls, ${budget.spent_usd:.2f} cost, "
          f"{elapsed:.0f}s wall-clock, abstains={budget.abstains_critic}")


def analyze(out_dir: Path, label: str = "anchored-judgment"):
    jsonl = out_dir / "anchored_votes.jsonl"
    if not jsonl.exists():
        print(f"[analyze] no cache at {jsonl}")
        return 0.0
    by_iid = defaultdict(list)
    with jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_iid[r["instance_id"]].append(r)
    iids = sorted(by_iid.keys())
    n_inst = len(iids)
    n_resolved = 0
    cat_counts = Counter()
    abstain_count = 0

    for iid in iids:
        rows = sorted(by_iid[iid], key=lambda r: r["patch_idx"])
        scored = []
        for row in rows:
            score = 0
            for v in row["anchored_votes"]:
                cat = v.get("category")
                cat_counts[cat] += 1
                w = CATEGORY_SCORE.get(cat)
                if w is None:
                    abstain_count += 1
                else:
                    score += w
            scored.append((row["patch_idx"], score, bool(row.get("patch_resolves"))))
        scored.sort(key=lambda x: (-x[1], x[0]))
        if scored and scored[0][2]:
            n_resolved += 1

    rate = n_resolved / max(1, n_inst)
    print(f"\n=== {label} on {n_inst} instances ===")
    print(f"Category distribution (across {n_inst}×{K_PATCHES}×{R_VOTES} = "
          f"{n_inst * K_PATCHES * R_VOTES} vote slots):")
    for c in ["CLEAR FIX", "PARTIAL PROGRESS", "SAME FAILURE", "NEW FAILURE",
              "INSUFFICIENT EVIDENCE", None]:
        n = cat_counts.get(c, 0)
        label_c = "<abstain>" if c is None else c
        print(f"  {label_c:25s}: {n:>5d}")
    print(f"  Abstains (parse-fail or empty): {abstain_count}")
    print()
    print(f"Resolve rate (weighted-yes score, tiebreak lowest-k): "
          f"{n_resolved}/{n_inst} = {rate:.3f}")
    print()
    print("=" * 64)
    print(f"DECISION (resolve rate: {rate:.1%})")
    print("=" * 64)
    if rate >= 0.74:
        print(f"  ≥74% → scale to full 500 (~$100 projected). This is the headline.")
    elif rate >= 0.72:
        print(f"  72-73% → marginal. Halt + write up direct-binary (72.0%) as final.")
    else:
        print(f"  <72% → anchored framing didn't help over direct-binary (72.0%).")
    return rate


async def amain(args):
    env = rc._require_env()
    client = rc._make_client(env)

    print("[load] patches…")
    rc.K_PATCHES = K_PATCHES
    patches_by_iid, all_iids = rc.load_patches()
    print(f"[load] {len(all_iids)} instances, K={K_PATCHES}")
    print("[load] resolve booleans…")
    resolve_by_iid = rc.load_resolve_bools()
    print("[load] problem statements…")
    problems_by_iid = rc.load_problem_statements(all_iids)
    print("[load] dataset fields (FAIL_TO_PASS)…")
    dataset_fields = dyn.load_dataset_fields(all_iids)
    eval_index = {}
    if args.allow_cached_eval_traces:
        print("[load] LEGACY/NON-BLIND: indexing failing eval dirs…")
        eval_index = dyn._index_failing_eval_dirs()
        print(f"[load] index covers {len(eval_index)} iids")
    else:
        print("[load] pre-eval blind mode: not reading cached eval reports/test_output.txt")

    pilot = rc.pilot_instance_subset(all_iids)
    budget = rc.Budget()
    hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))

    if args.smoke:
        smoke_iids = pilot[:int(args.smoke)]
        print(f"[smoke] {len(smoke_iids)} instances: {smoke_iids}")
        await run_anchored(client, env["deployment"], smoke_iids,
                           patches_by_iid, resolve_by_iid, problems_by_iid,
                           dataset_fields, eval_index, SMOKE_DIR, budget)
        analyze(SMOKE_DIR, label=f"smoke ({len(smoke_iids)} inst)")
    elif args.pilot:
        await run_anchored(client, env["deployment"], pilot,
                           patches_by_iid, resolve_by_iid, problems_by_iid,
                           dataset_fields, eval_index, PILOT_DIR, budget)
        analyze(PILOT_DIR, label="pilot 50")
    elif args.all:
        await run_anchored(client, env["deployment"], all_iids,
                           patches_by_iid, resolve_by_iid, problems_by_iid,
                           dataset_fields, eval_index, FULL500_DIR, budget)
        analyze(FULL500_DIR, label="full 500")
    else:
        print("specify --smoke N, --pilot, or --all")
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
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--allow-cached-eval-traces", action="store_true",
                   help="LEGACY/NON-BLIND: mine cached eval report test_output.txt before selection.")
    args = p.parse_args()
    if not (args.smoke or args.pilot or args.all):
        p.print_help()
        return 2
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
