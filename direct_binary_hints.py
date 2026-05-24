"""Direct-binary classifier with execution hints (FAIL_TO_PASS test names +
example file paths from each patch).

Tests whether more grounding context — what the failing test is, what files
the patch touches — closes the gap between text-only direct-binary (66.0%)
and oracle (74%).

Mirrors direct_binary.py exactly except for the augmented prompt + caches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ceiling as rc

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "outputs" / "direct_binary_hints_pilot"
SMOKE_DIR = REPO / "outputs" / "direct_binary_hints_smoke"
FULL500_DIR = REPO / "outputs" / "direct_binary_hints_full500"
GREEDY8_EXTRA_DIR = REPO / "outputs" / "direct_binary_hints_greedy8_extra"

K_PATCHES = 8
R_VOTES = 5
GLOBAL_SEED = 42
CONCURRENCY = rc.CONCURRENCY
PRICE_INPUT_PER_M = rc.PRICE_INPUT_PER_M
PRICE_OUTPUT_PER_M = rc.PRICE_OUTPUT_PER_M
PARSE_RETRY_MAX = rc.PARSE_RETRY_MAX
PARSE_RETRY_TEMP_BUMP = rc.PARSE_RETRY_TEMP_BUMP
BINARY_MAX_OUTPUT = 4096
HARD_BUDGET_USD = 150.00  # full-500 mode budget cap

# ----------------------------------------------------------------------------
# Hints loading
# ----------------------------------------------------------------------------


def load_fail_to_pass(iids: list[str]) -> dict[str, list[str]]:
    """Load FAIL_TO_PASS test names per instance from SWE-Bench Verified."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
        wanted = set(iids)
        out: dict[str, list[str]] = {}
        for inst in ds:
            iid = inst["instance_id"]
            if iid not in wanted:
                continue
            ftp_raw = inst.get("FAIL_TO_PASS", "[]")
            try:
                names = json.loads(ftp_raw) if isinstance(ftp_raw, str) else list(ftp_raw)
            except Exception:
                names = []
            out[iid] = [n for n in names if isinstance(n, str)]
        return out
    except Exception as e:
        sys.stderr.write(f"WARNING: FAIL_TO_PASS load failed ({e}); proceeding empty.\n")
        return {iid: [] for iid in iids}


_DIFF_FILE_RE = re.compile(r'^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)', re.MULTILINE)


def patch_files(patch_text: str, max_files: int = 5) -> list[str]:
    """Extract example file paths from a unified diff patch (deduped, ordered)."""
    if not patch_text:
        return []
    seen = []
    seen_set = set()
    for m in _DIFF_FILE_RE.finditer(patch_text):
        path = m.group(1)
        if path in ("/dev/null", ""):
            continue
        if path in seen_set:
            continue
        seen.append(path)
        seen_set.add(path)
        if len(seen) >= max_files:
            break
    return seen


# ----------------------------------------------------------------------------
# Prompt + parser
# ----------------------------------------------------------------------------


HINTS_BINARY_PROMPT = """You are evaluating a single patch against the GitHub issue it claims to fix.
Output yes if and only if running the patched code passes the failing tests
listed below and does not break any test that previously passed.

Trace the failure path from the issue and the named failing tests to the
originating code. Check whether the patch modifies that code path. Then state
whether the patch produces the expected behavior on the smallest input
that exhibits the failure.

ISSUE:
{problem_statement}

FAILING TESTS (FAIL_TO_PASS — these must pass after applying the patch):
{fail_to_pass}

FILES MODIFIED BY PATCH:
{patch_files}

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
    if "resolves" not in obj or obj["resolves"] not in (True, False):
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
    client, deployment, sem, problem, patch, fail_to_pass, patch_files_str,
    iid: str, k: int, vote_idx: int,
    out_dir: Path, budget: rc.Budget,
) -> dict:
    base_temp = 0.7
    prompt = HINTS_BINARY_PROMPT.format(
        problem_statement=problem[:2000],
        fail_to_pass=fail_to_pass,
        patch_files=patch_files_str,
        patch=patch[:8000],
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
        budget.add("critic", result)
        rc.append_usage(
            out_dir, "critic",
            {"iid": iid, "patch_idx": k, "vote_idx": vote_idx, "attempt": attempt},
            result, "direct_binary_hints", temp,
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
# Stage runner: streaming
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


async def run_binary_hints(
    client, deployment, instances, patches_by_iid,
    resolve_by_iid, problems_by_iid, fail_to_pass_by_iid,
    out_dir: Path, budget: rc.Budget,
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
        ftp_list = fail_to_pass_by_iid.get(iid, [])
        ftp_str = "\n".join(f"  - {n}" for n in ftp_list[:10]) or "  (none provided)"
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
                    client, deployment, sem, problem, patch, ftp_str, files_str,
                    iid, k, vi, out_dir, budget,
                ))

    print(f"[binary-hints] {len(coros)} API calls queued across {len(instances)} instances")
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
            print(f"[binary-hints] HALT: budget ${budget.spent_usd:.2f} ≥ ${HARD_BUDGET_USD}")
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
    print(f"[binary-hints] done: {completed} vote-events, "
          f"{budget.n_calls} API calls, ${budget.spent_usd:.2f} cost, "
          f"{elapsed:.0f}s wall-clock, abstains={budget.abstains_critic}")


# ----------------------------------------------------------------------------
# Inline analysis (same as direct_binary.py)
# ----------------------------------------------------------------------------


def analyze(out_dir: Path, label: str = "direct-binary-hints"):
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
    n_resolved_majority = 0
    n_resolved_confidence = 0
    yes_counts = []
    abstain_count = 0

    for iid in iids:
        rows = sorted(by_iid[iid], key=lambda r: r["patch_idx"])
        scores_majority = []
        scores_confidence = []
        for row in rows:
            yes = sum(1 for v in row["binary_votes"] if v.get("resolves") is True)
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
        scores_majority.sort(key=lambda x: (-x[1], x[0]))
        if scores_majority[0][2]:
            n_resolved_majority += 1
        scores_confidence.sort(key=lambda x: (-x[1], x[0]))
        if scores_confidence[0][2]:
            n_resolved_confidence += 1

    from collections import Counter
    hist = Counter(yes_counts)
    print(f"\n=== {label} analysis on {n_inst} instances ===")
    print(f"Per-patch yes-count distribution (out of R={R_VOTES}):")
    for c in range(R_VOTES + 1):
        print(f"  {c} yes votes: {hist.get(c, 0):>4d} patches")
    print(f"Total abstains: {abstain_count}")
    print()
    print(f"Resolve rate (most-yes, tiebreak lowest-k):  {n_resolved_majority}/{n_inst} = {n_resolved_majority/n_inst:.3f}")
    print(f"Resolve rate (confidence-weighted):          {n_resolved_confidence}/{n_inst} = {n_resolved_confidence/n_inst:.3f}")

    rate = n_resolved_majority / n_inst
    print()
    print("=" * 64)
    print(f"DECISION (most-yes resolve rate: {rate:.1%})")
    print("=" * 64)
    if rate >= 0.70:
        print(f"  ≥70% → grounding context was the missing piece.")
        print(f"  RECOMMEND: K-best-8 + execution-hints. Paper has positive contribution.")
    elif rate >= 0.67:
        print(f"  67-69% → small but real grounding effect.")
        print(f"  RECOMMEND: K-best-8 + execution-hints.")
    elif rate >= 0.64:
        print(f"  64-66% → grounding doesn't help either.")
        print(f"  RECOMMEND: HALT all spending. Write up the negative result.")
        print(f"  Story: weak-model committees on SWE-bench fail with or without grounding")
        print(f"  due to systematic yes-bias; aggregation cannot recover what individual")
        print(f"  judgments lack.")
    else:
        print(f"  <64% → unexpectedly low; investigate parse-fail rate or model issues.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def load_patches_subset(k_list: list[int]):
    """Load patches for arbitrary K indices (not just 0..N-1).
    Returns (patches_by_iid_then_localidx, sorted_iids, k_list).
    The local indices are just k_list's order, but the cache row's
    patch_idx will store the *actual* K value.
    """
    import json as _json
    PATCH_DIR = Path("/home/vsun/orchestra-gpt/outputs/oracle_preds_v3")
    by_iid = {}
    for k in k_list:
        f = PATCH_DIR / f"proposer_{k}.jsonl"
        if not f.exists():
            sys.stderr.write(f"FATAL: missing patches file {f}\n")
            sys.exit(2)
        with f.open() as fh:
            for line in fh:
                row = _json.loads(line)
                iid = row["instance_id"]
                if iid not in by_iid:
                    by_iid[iid] = {}
                by_iid[iid][k] = row.get("model_patch", "") or ""
    return by_iid, sorted(by_iid.keys())


async def run_binary_hints_subset(
    client, deployment, instances, patches_by_iid_dict,
    resolve_by_iid, problems_by_iid, fail_to_pass_by_iid, k_list,
    out_dir: Path, budget,
):
    """Run binary-hints classifier on specific (iid, k) pairs where k ∈ k_list.
    Writes rows with the *actual* K value as patch_idx (not local index)."""
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
        ftp_list = fail_to_pass_by_iid.get(iid, [])
        ftp_str = "\n".join(f"  - {n}" for n in ftp_list[:10]) or "  (none provided)"
        patches_for_iid = patches_by_iid_dict.get(iid, {})
        for k in k_list:
            if (iid, k) in done_keys:
                continue
            patch = patches_for_iid.get(k, "") or ""
            patch_resolves = resolve_by_iid.get(iid, {}).get(k, None)
            pk_resolves[(iid, k)] = patch_resolves
            files = patch_files(patch, max_files=5)
            files_str = "\n".join(f"  - {p}" for p in files) or "  (no patch / parse failed)"
            if not patch.strip():
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
                    client, deployment, sem, problem, patch, ftp_str, files_str,
                    iid, k, vi, out_dir, budget,
                ))
    print(f"[binary-hints] {len(coros)} API calls queued across {len(instances)} instances "
          f"(K={k_list})")
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
            print(f"[binary-hints] HALT: budget ${budget.spent_usd:.2f} ≥ ${HARD_BUDGET_USD}")
            halted = True
            for f in futures:
                if not f.done(): f.cancel()
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
    print(f"[binary-hints] done: {completed} vote-events, "
          f"{budget.n_calls} API calls, ${budget.spent_usd:.2f} cost, "
          f"{elapsed:.0f}s wall-clock, abstains={budget.abstains_critic}")


async def amain(args):
    env = rc._require_env()
    client = rc._make_client(env)

    if args.k_list:
        # Custom K-subset path (e.g., greedy-8 extras)
        k_list = [int(x.strip()) for x in args.k_list.split(",")]
        print(f"[load] patches for K-list = {k_list}")
        patches_dict, all_iids = load_patches_subset(k_list)
        print(f"[load] {len(all_iids)} instances")

        print("[load] resolve booleans…")
        resolve_by_iid = rc.load_resolve_bools()
        print("[load] problem statements…")
        problems_by_iid = rc.load_problem_statements(all_iids)
        print("[load] FAIL_TO_PASS…")
        fail_to_pass = load_fail_to_pass(all_iids)

        budget = rc.Budget()
        hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))
        await run_binary_hints_subset(
            client, env["deployment"], all_iids,
            patches_dict, resolve_by_iid, problems_by_iid,
            fail_to_pass, k_list, GREEDY8_EXTRA_DIR, budget,
        )
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        print(f"[done] calls={budget.n_calls} cost=${budget.spent_usd:.2f} "
              f"abstains={budget.abstains_critic}")
        return 0

    print("[load] patches…")
    rc.K_PATCHES = K_PATCHES
    patches_by_iid, all_iids = rc.load_patches()
    print(f"[load] {len(all_iids)} instances, K={K_PATCHES}")

    print("[load] resolve booleans…")
    resolve_by_iid = rc.load_resolve_bools()

    print("[load] problem statements…")
    problems_by_iid = rc.load_problem_statements(all_iids)

    print("[load] FAIL_TO_PASS…")
    fail_to_pass = load_fail_to_pass(all_iids)
    n_with_ftp = sum(1 for iid in all_iids if fail_to_pass.get(iid))
    print(f"[load] FAIL_TO_PASS for {n_with_ftp}/{len(all_iids)} instances")

    pilot = rc.pilot_instance_subset(all_iids)
    print(f"[load] pilot subset: {len(pilot)} instances")

    budget = rc.Budget()
    hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))

    if args.smoke:
        smoke_iids = pilot[:int(args.smoke)]
        print(f"[smoke] {len(smoke_iids)} instances: {smoke_iids}")
        await run_binary_hints(client, env["deployment"], smoke_iids,
                               patches_by_iid, resolve_by_iid, problems_by_iid,
                               fail_to_pass, SMOKE_DIR, budget)
        analyze(SMOKE_DIR, label=f"hints smoke ({len(smoke_iids)} inst)")
    elif args.all:
        print(f"[all] {len(all_iids)} instances (full SWE-Bench Verified subset)")
        await run_binary_hints(client, env["deployment"], all_iids,
                               patches_by_iid, resolve_by_iid, problems_by_iid,
                               fail_to_pass, FULL500_DIR, budget)
        analyze(FULL500_DIR, label="direct-binary + hints (full 500)")
    elif args.full:
        await run_binary_hints(client, env["deployment"], pilot,
                               patches_by_iid, resolve_by_iid, problems_by_iid,
                               fail_to_pass, OUT_DIR, budget)
        analyze(OUT_DIR, label="direct-binary + hints (full pilot)")
    else:
        print("specify --smoke N or --full or --all")
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
    p.add_argument("--all", action="store_true",
                   help="Run on all 493 SWE-Bench Verified instances "
                        "with patches (writes to outputs/direct_binary_hints_full500/)")
    p.add_argument("--k-list", default="",
                   help="Comma-separated K indices (e.g. '8,9,12,15') for "
                        "greedy-8 bridge. Writes to outputs/direct_binary_hints_greedy8_extra/.")
    args = p.parse_args()
    if not (args.smoke or args.full or args.all or args.k_list):
        p.print_help()
        return 2
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
