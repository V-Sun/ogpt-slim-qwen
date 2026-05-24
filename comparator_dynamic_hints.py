"""Round-robin comparator with execution-grounded hints, 50 pilot.

Setup:
  K=8 patches (K=0..7)
  All C(8,2) = 28 matchups per instance
  R = 5 comparators per matchup, each comparator does 2 position-swap calls
  Single committed comparator prompt + dynamic hints
  (FAIL_TO_PASS names + error trace from test_output.txt + file paths)

Aggregation per user spec:
  - Per comparator vote: weight 1 if swap_1 == swap_2, else 0.5 counted as TIE
  - Sum weighted A/B/TIE per matchup → matchup winner
  - Per patch: round-robin score = matchups won
  - Tournament winner = highest score, tiebreak by lowest patch_idx

Decision tree:
  ≥75%: scale to full 500
  72-74%: marginal, halt and decide
  <72%: write up direct-binary as final

Output: outputs/comparator_dynamic_hints_{smoke,pilot}/
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ceiling as rc
import direct_binary_dynamic_hints as dyn  # for trace extraction

REPO = Path(__file__).resolve().parent
SMOKE_DIR = REPO / "outputs" / "comparator_dynamic_hints_smoke"
PILOT_DIR = REPO / "outputs" / "comparator_dynamic_hints_pilot"
FULL_DIR = REPO / "outputs" / "comparator_dynamic_hints_full500"

K_PATCHES = 8
R_VOTES = 5
CONCURRENCY = 128  # share deployment with dynamic-hints full run
PARSE_RETRY_MAX = rc.PARSE_RETRY_MAX
PARSE_RETRY_TEMP_BUMP = rc.PARSE_RETRY_TEMP_BUMP
COMPARATOR_MAX_OUTPUT = 8192
HARD_BUDGET_USD = 150.00


# ----------------------------------------------------------------------------
# Prompt: original structural-comparison + dynamic hints
# ----------------------------------------------------------------------------


COMPARATOR_HINTS_PROMPT = """You are deciding which of two patches more correctly resolves a GitHub issue.

Below you have:
  - The issue text
  - The exact failing test names (FAIL_TO_PASS)
  - The actual error trace from running the failing test on the unpatched code
  - The list of files each patch modifies
  - The two patches

Your job: determine which patch (A or B) actually resolves the failing test
shown in the trace. Trace the error frame; check which patch modifies the
code path producing that error; check the patch's logic would make the failing
assertion pass without breaking the passing assertions.

ISSUE:
{problem_statement}

FAILING TESTS (must pass after patch):
{fail_to_pass}

ACTUAL ERROR TRACE FROM RUNNING THE FAILING TEST:
{error_trace}

PATCH A (modifies: {a_files}):
{patch_a}

PATCH B (modifies: {b_files}):
{patch_b}

REQUIRED STRUCTURAL COMPARISON (fill these in BEFORE deciding):

  1. Failing test hypothesis: state the smallest hypothesis about what is
     wrong. One sentence. (Use the trace above to ground this.)

  2. A_changes: specific lines/functions Patch A modifies.
  3. B_changes: specific lines/functions Patch B modifies.

  4. A_consistent_with_hypothesis: do A's changes plausibly cause the failing
     test to start passing? (true/false)
  5. B_consistent_with_hypothesis: same for B.

  6. A_collateral: does A change behavior on inputs unrelated to the failure?
  7. B_collateral: same for B.

If exactly one patch is consistent with the hypothesis and resolves the
traced error, that one wins. If both are consistent, prefer less collateral.
If both fail the hypothesis (neither addresses the trace), output TIE.
If functionally equivalent, output TIE.

Respond in this EXACT JSON format — JSON only, no markdown fences:
{{
  "hypothesis": "<one sentence>",
  "a_changes": "<files/functions modified by A>",
  "b_changes": "<files/functions modified by B>",
  "a_consistent": <true|false>,
  "b_consistent": <true|false>,
  "a_collateral": <true|false>,
  "b_collateral": <true|false>,
  "winner": "A" | "B" | "TIE",
  "confidence": <integer 1-5>,
  "reasoning": "<one sentence>"
}}
"""


def parse_comparator(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(rc._strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    w = obj.get("winner")
    if w not in ("A", "B", "TIE"):
        return None
    return {
        "winner": w,
        "confidence": int(obj.get("confidence", 0) or 0),
        "reasoning": (obj.get("reasoning") or "")[:300],
    }


# ----------------------------------------------------------------------------
# One comparator vote = 2 position-swap API calls
# ----------------------------------------------------------------------------


async def _one_swap_call(
    client, deployment, sem, problem, patch_a, patch_b, ftp_str, trace,
    a_files_str, b_files_str,
    iid, a_idx, b_idx, vote_idx, swap, out_dir, budget,
):
    base_temp = 0.7
    # swap=1: shown as (A=patch_a, B=patch_b)
    # swap=2: shown as (A=patch_b, B=patch_a) — we map letter back at vote level
    if swap == 1:
        first, second = patch_a, patch_b
        first_files, second_files = a_files_str, b_files_str
    else:
        first, second = patch_b, patch_a
        first_files, second_files = b_files_str, a_files_str
    prompt = COMPARATOR_HINTS_PROMPT.format(
        problem_statement=problem[:2000],
        fail_to_pass=ftp_str,
        error_trace=trace[:2500],
        a_files=first_files,
        b_files=second_files,
        patch_a=first[:6000],
        patch_b=second[:6000],
    )
    last_text = ""
    last_result = None
    for attempt in range(PARSE_RETRY_MAX):
        temp = min(1.5, base_temp + attempt * PARSE_RETRY_TEMP_BUMP)
        budget.issued += 1
        result = await rc._api_call(
            client, deployment, prompt,
            temperature=temp, max_output_tokens=COMPARATOR_MAX_OUTPUT, sem=sem,
        )
        budget.add("comparator", result)
        rc.append_usage(out_dir, "comparator",
                        {"iid": iid, "a_idx": a_idx, "b_idx": b_idx,
                         "vote_idx": vote_idx, "swap": swap, "attempt": attempt},
                        result, "comparator_dynamic_hints", temp)
        last_result, last_text = result, result.text
        if result.error:
            continue
        parsed = parse_comparator(result.text)
        if parsed is not None:
            return parsed["winner"], parsed["confidence"]
    budget.abstains_comparator += 1
    return None, 0


async def _do_one_comparator_vote(
    client, deployment, sem,
    problem, patch_a, patch_b, ftp_str, trace, a_files_str, b_files_str,
    iid, a_idx, b_idx, vote_idx, out_dir, budget,
):
    w1, c1 = await _one_swap_call(
        client, deployment, sem, problem, patch_a, patch_b, ftp_str, trace,
        a_files_str, b_files_str, iid, a_idx, b_idx, vote_idx, 1,
        out_dir, budget,
    )
    w2_raw, c2 = await _one_swap_call(
        client, deployment, sem, problem, patch_a, patch_b, ftp_str, trace,
        a_files_str, b_files_str, iid, a_idx, b_idx, vote_idx, 2,
        out_dir, budget,
    )
    # Map swap-2's winner: in swap-2, the model saw A=patch_b, B=patch_a.
    # So returned 'A' means patch_b won → caller's letter "B"; vice versa.
    if w2_raw is None:
        w2 = None
    elif w2_raw == "A":
        w2 = "B"
    elif w2_raw == "B":
        w2 = "A"
    else:
        w2 = "TIE"
    return {
        "comparator_idx": vote_idx,
        "position_swap_1": w1,
        "position_swap_2": w2,
        "confidence_1": c1,
        "confidence_2": c2,
    }


# ----------------------------------------------------------------------------
# File-list helpers (reuse from direct_binary_hints — but inline minimal)
# ----------------------------------------------------------------------------


_DIFF_FILE_RE = re.compile(r'^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)', re.MULTILINE)


def patch_files_str(patch_text: str, max_files: int = 5) -> str:
    if not patch_text:
        return "(empty patch)"
    seen, seen_set = [], set()
    for m in _DIFF_FILE_RE.finditer(patch_text):
        path = m.group(1)
        if path in ("/dev/null", "") or path in seen_set:
            continue
        seen.append(path)
        seen_set.add(path)
        if len(seen) >= max_files:
            break
    return ", ".join(seen) or "(no file paths parsed)"


# ----------------------------------------------------------------------------
# Resume + main runner
# ----------------------------------------------------------------------------


def load_done_matchups(jsonl: Path) -> set[tuple[str, int, int]]:
    done = set()
    if not jsonl.exists():
        return done
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["instance_id"], int(r["patch_a_idx"]), int(r["patch_b_idx"])))
    return done


async def run_comparator_hints(
    client, deployment, instances, patches_by_iid, resolve_by_iid,
    problems_by_iid, dataset_fields, eval_index, out_dir, budget,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "comparator_votes.jsonl"
    done_keys = load_done_matchups(out_jsonl)
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
        trace, _src = dyn.get_dynamic_hint_for_iid(iid, ftp, eval_index)
        if trace:
            n_with_trace += 1
        else:
            trace = "(no failing-test trace cached — likely all proposers resolved this instance)"
        hint_cache[iid] = (ftp_str, trace)
    print(f"[hints] traces for {n_with_trace}/{len(instances)} instances")

    # Build all (iid, a, b, vi) coroutines
    task_keys = []
    coros = []
    for iid in instances:
        ftp_str, trace = hint_cache[iid]
        problem = problems_by_iid.get(iid, "")
        patches = patches_by_iid.get(iid, [""] * K_PATCHES)
        for a, b in itertools.combinations(range(K_PATCHES), 2):
            if (iid, a, b) in done_keys:
                continue
            pa, pb = patches[a] or "", patches[b] or ""
            if not pa.strip() or not pb.strip():
                # empty patch: cache TIE row
                row = {
                    "instance_id": iid, "patch_a_idx": a, "patch_b_idx": b,
                    "comparator_votes": [
                        {"comparator_idx": vi, "position_swap_1": None,
                         "position_swap_2": None, "confidence_1": 0,
                         "confidence_2": 0,
                         "abstain_reason": "empty_patch"}
                        for vi in range(R_VOTES)
                    ],
                }
                with out_jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done_keys.add((iid, a, b))
                continue
            a_files_str = patch_files_str(pa)
            b_files_str = patch_files_str(pb)
            for vi in range(R_VOTES):
                task_keys.append((iid, a, b, vi))
                coros.append(_do_one_comparator_vote(
                    client, deployment, sem, problem, pa, pb,
                    ftp_str, trace, a_files_str, b_files_str,
                    iid, a, b, vi, out_dir, budget,
                ))

    print(f"[comparator-hints] {len(coros)} comparator-votes queued "
          f"(2 swap calls each = {2 * len(coros)} API calls)")
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
            print(f"[comparator-hints] HALT: budget ${budget.spent_usd:.2f} ≥ ${HARD_BUDGET_USD}")
            halted = True
            for f in futures:
                if not f.done():
                    f.cancel()
        try:
            (iid, a, b, vi), vote = await fut
        except asyncio.CancelledError:
            continue
        bucket = pending.setdefault((iid, a, b), {})
        bucket[vi] = vote
        if len(bucket) == R_VOTES:
            votes = [bucket[v] for v in range(R_VOTES)]
            row = {
                "instance_id": iid, "patch_a_idx": a, "patch_b_idx": b,
                "comparator_votes": votes,
            }
            with out_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_keys.add((iid, a, b))
            del pending[(iid, a, b)]
        completed += 1

    elapsed = time.time() - t_start
    print(f"[comparator-hints] done: {completed} matchup-vote-events, "
          f"{budget.n_calls} API calls, ${budget.spent_usd:.2f} cost, "
          f"{elapsed:.0f}s wall-clock, abstains_cmp={budget.abstains_comparator}")


# ----------------------------------------------------------------------------
# Aggregation per user spec
# ----------------------------------------------------------------------------


def aggregate_matchup_winner(votes: list[dict]) -> str:
    """Per user: weight 1 if swap_1==swap_2, else 0.5 counted as TIE.
    Sum weighted A/B/TIE; max wins. Returns 'A', 'B', or 'TIE'."""
    a_score = b_score = tie_score = 0.0
    for v in votes:
        s1 = v.get("position_swap_1")
        s2 = v.get("position_swap_2")
        if s1 is None or s2 is None:
            continue  # both swaps abstained
        if s1 == s2:
            if s1 == "A":
                a_score += 1.0
            elif s1 == "B":
                b_score += 1.0
            else:
                tie_score += 1.0
        else:
            tie_score += 0.5  # disagreement → half-weight TIE
    if a_score > b_score and a_score > tie_score:
        return "A"
    if b_score > a_score and b_score > tie_score:
        return "B"
    return "TIE"


def round_robin_winner(matchup_winners: dict[tuple[int, int], str],
                       k_patches: int = K_PATCHES) -> int:
    """Per patch: count matchups won. Tournament winner = max wins,
    tiebreak by lowest patch_idx."""
    wins = {p: 0 for p in range(k_patches)}
    for (a, b), w in matchup_winners.items():
        if w == "A":
            wins[a] += 1
        elif w == "B":
            wins[b] += 1
    return min(range(k_patches), key=lambda p: (-wins[p], p))


def analyze(out_dir: Path, label: str = "comparator + dynamic hints"):
    jsonl = out_dir / "comparator_votes.jsonl"
    if not jsonl.exists():
        print(f"[analyze] no cache at {jsonl}")
        return 0.0
    by_iid = defaultdict(dict)  # iid -> {(a,b): votes}
    resolves = {}                # iid -> {k: bool}  (filled from existing critic cache)
    with jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_iid[r["instance_id"]][(r["patch_a_idx"], r["patch_b_idx"])] = \
                r.get("comparator_votes", [])
    # Load resolves from existing critic cache
    crit_cache = REPO / "outputs" / "stage1_critics_full" / "critic_votes.jsonl"
    with crit_cache.open() as fh:
        for line in fh:
            r = json.loads(line)
            resolves.setdefault(r["instance_id"], {})[r["patch_idx"]] = bool(r.get("patch_resolves"))

    n_resolved = 0
    n_inst = len(by_iid)
    matchup_winner_dist = Counter()
    swap_agreement_pct = []
    for iid in sorted(by_iid):
        matchup_winners = {}
        for (a, b), votes in by_iid[iid].items():
            w = aggregate_matchup_winner(votes)
            matchup_winners[(a, b)] = w
            matchup_winner_dist[w] += 1
            for v in votes:
                s1, s2 = v.get("position_swap_1"), v.get("position_swap_2")
                if s1 is not None and s2 is not None:
                    swap_agreement_pct.append(1.0 if s1 == s2 else 0.0)
        winner = round_robin_winner(matchup_winners)
        if resolves.get(iid, {}).get(winner, False):
            n_resolved += 1

    rate = n_resolved / max(1, n_inst)
    print(f"\n=== {label} analysis on {n_inst} instances ===")
    print(f"Matchup winner distribution: {dict(matchup_winner_dist)}")
    if swap_agreement_pct:
        agree = sum(swap_agreement_pct) / len(swap_agreement_pct)
        print(f"Position-swap agreement rate: {agree:.1%} "
              f"({len(swap_agreement_pct)} comparator votes)")
    print()
    print(f"Resolve rate (round-robin winner, lowest-k tiebreak): {n_resolved}/{n_inst} = {rate:.3f}")
    print()
    print("=" * 64)
    print(f"DECISION (resolve rate: {rate:.1%})")
    print("=" * 64)
    if rate >= 0.75:
        print(f"  ≥75% → scale to full 500.")
    elif rate >= 0.72:
        print(f"  72-74% → marginal, halt and decide.")
    else:
        print(f"  <72% → write up direct-binary as final result.")
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
    print("[load] FAIL_TO_PASS / test_patch / base_commit…")
    dataset_fields = dyn.load_dataset_fields(all_iids)
    print(f"[load] dataset fields for {len(dataset_fields)}/{len(all_iids)}")
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
        await run_comparator_hints(client, env["deployment"], smoke_iids,
                                   patches_by_iid, resolve_by_iid, problems_by_iid,
                                   dataset_fields, eval_index, SMOKE_DIR, budget)
        analyze(SMOKE_DIR, label=f"smoke ({len(smoke_iids)} inst)")
    elif args.pilot:
        await run_comparator_hints(client, env["deployment"], pilot,
                                   patches_by_iid, resolve_by_iid, problems_by_iid,
                                   dataset_fields, eval_index, PILOT_DIR, budget)
        analyze(PILOT_DIR, label="pilot 50")
    elif args.all:
        await run_comparator_hints(client, env["deployment"], all_iids,
                                   patches_by_iid, resolve_by_iid, problems_by_iid,
                                   dataset_fields, eval_index, FULL_DIR, budget)
        analyze(FULL_DIR, label="full 500")
    else:
        print("specify --smoke N, --pilot, or --all")
        return 2

    hb.cancel()
    try:
        await hb
    except asyncio.CancelledError:
        pass
    print(f"[done] calls={budget.n_calls} cost=${budget.spent_usd:.2f} "
          f"abstains_cmp={budget.abstains_comparator}")
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
