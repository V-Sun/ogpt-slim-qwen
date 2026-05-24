#!/usr/bin/env python3
"""King-of-the-hill patch selection aggregator.

Reads pairwise comparator votes (any votes jsonl from anchored_dynamic_comparator or
trace_comparator) and applies two aggregation strategies beyond Copeland:

1. BRACKET: Fixed single-elimination bracket over the greedy-8 pool.
   Quarter-finals: (1v3), (5v7), (8v9), (12v15)
   Semi-finals:    (QF1 winner vs QF2 winner), (QF3 winner vs QF4 winner)
   Final:          (SF1 winner vs SF2 winner)
   Tiebreak at each match: confidence margin, then lower k-index.

2. SEQUENTIAL KING: Start with K=1 as the reigning king. Each subsequent K
   challenges the king; winner retains the title. Order: [0, 1, 2, 3, 4, 5, 6, 7].
   Tiebreak: challenger does NOT dethrone the king on a tie.

3. STRICT DOMINANCE: A patch is "king" iff it beats every other patch head-to-head
   (majority of votes). If no unique dominator exists, falls back to Copeland.

4. COPELAND (baseline): Reproduced from the same vote data for direct comparison.

Usage:
    python3 king_of_the_hill.py --votes-path outputs/final_files/comparator_blind_greedy8_votes.jsonl
    python3 king_of_the_hill.py --votes-path outputs/trace_comparator_nano_greedy8/pairwise_trace_comparator_votes.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DATA_REPO = Path("/home/vsun/orchestra-gpt")
DEFAULT_REPORTS_DIR = DATA_REPO / "logs" / "run_evaluation"
GREEDY8_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

BRACKET_QF = [(0, 1), (2, 3), (4, 5), (6, 7)]


# ---------------------------------------------------------------------------
# Truth loading (from harness report.json files)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Vote parsing
# ---------------------------------------------------------------------------

def tally_matchup(votes: list[dict]) -> tuple[int, int, float]:
    """Return (a_wins, b_wins, net_confidence_margin).

    net_confidence_margin > 0 means A ahead, < 0 means B ahead.
    """
    a_wins = b_wins = 0
    margin = 0.0
    for v in votes:
        p = (v.get("primary") or {}).get("parsed") or {}
        winner = p.get("winner", "TIE")
        conf = float(p.get("confidence", 0.5) or 0.5)
        if winner == "A":
            a_wins += 1
            margin += conf
        elif winner == "B":
            b_wins += 1
            margin -= conf
    return a_wins, b_wins, margin


def build_pairwise(rows: list[dict]) -> dict[tuple[int, int], tuple[int, int, float]]:
    """Build {(a, b): (a_wins, b_wins, margin)} for a < b always."""
    result: dict[tuple[int, int], tuple[int, int, float]] = {}
    for row in rows:
        if "skip_reason" in row or not row.get("votes"):
            continue
        a = int(row["patch_a_idx"])
        b = int(row["patch_b_idx"])
        aw, bw, mg = tally_matchup(row["votes"])
        if (a, b) not in result:
            result[(a, b)] = (aw, bw, mg)
        else:
            ea, eb, em = result[(a, b)]
            result[(a, b)] = (ea + aw, eb + bw, em + mg)
    return result


def head_to_head(pairwise: dict, a: int, b: int) -> tuple[int, int, float]:
    """Return (a_wins, b_wins, margin) for any ordering of a, b."""
    if (a, b) in pairwise:
        return pairwise[(a, b)]
    if (b, a) in pairwise:
        bw, aw, mg = pairwise[(b, a)]
        return aw, bw, -mg
    return 0, 0, 0.0


def match_winner(pairwise: dict, a: int, b: int, challenger_needs_win: bool = False) -> int:
    """Return the winning k-index, using confidence margin then lower-k as tiebreak.

    challenger_needs_win: if True (sequential-king mode), a tie means the
    current king (a) retains the title.
    """
    aw, bw, mg = head_to_head(pairwise, a, b)
    if aw > bw:
        return a
    if bw > aw:
        return b
    # Tied on vote count — use confidence margin
    if mg > 0:
        return a
    if mg < 0:
        return b
    # Fully tied
    if challenger_needs_win:
        return a  # king retains on tie
    return min(a, b)  # lower k-index wins


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------

def copeland(pairwise: dict, k_list: list[int]) -> int:
    scores: dict[int, int] = {k: 0 for k in k_list}
    for i, ka in enumerate(k_list):
        for kb in k_list[i + 1:]:
            aw, bw, _ = head_to_head(pairwise, ka, kb)
            if aw > bw:
                scores[ka] += 1
                scores[kb] -= 1
            elif bw > aw:
                scores[kb] += 1
                scores[ka] -= 1
    return max(k_list, key=lambda k: (scores[k], -k))


def bracket(pairwise: dict, k_list: list[int]) -> int:
    """Single-elimination bracket. QF bracket is fixed for greedy-8.

    Falls back to sequential if k_list doesn't match BRACKET_QF exactly.
    """
    available = set(k_list)
    rounds = []

    if set(k_list) == set(GREEDY8_K_LIST):
        rounds = [list(BRACKET_QF)]
    else:
        # Generic bracket: pair up adjacent elements
        pairs = []
        lst = list(k_list)
        while len(lst) >= 2:
            pairs.append((lst.pop(0), lst.pop(0)))
        rounds = [pairs]
        if lst:
            # Bye — advance the remaining one directly
            byes = lst
        else:
            byes = []

    survivors = list(k_list)
    for rnd_pairs in rounds:
        next_round = []
        in_match = set()
        for a, b in rnd_pairs:
            if a not in available or b not in available:
                # One absent — other advances if present
                for x in (a, b):
                    if x in available:
                        next_round.append(x)
                        in_match.add(x)
                continue
            w = match_winner(pairwise, a, b)
            next_round.append(w)
            in_match.update([a, b])
        byes_this = [k for k in survivors if k not in in_match]
        survivors = next_round + byes_this

    while len(survivors) > 1:
        next_round = []
        for i in range(0, len(survivors) - 1, 2):
            a, b = survivors[i], survivors[i + 1]
            next_round.append(match_winner(pairwise, a, b))
        if len(survivors) % 2 == 1:
            next_round.append(survivors[-1])
        survivors = next_round

    return survivors[0] if survivors else k_list[0]


def sequential_king(pairwise: dict, k_list: list[int]) -> int:
    """Current king must be beaten outright — ties preserve the king."""
    king = k_list[0]
    for challenger in k_list[1:]:
        king = match_winner(pairwise, king, challenger, challenger_needs_win=True)
    return king


def strict_dominance(pairwise: dict, k_list: list[int]) -> int:
    """Return k that beats ALL others. Falls back to Copeland if none found."""
    for k in k_list:
        if all(
            head_to_head(pairwise, k, other)[0] > head_to_head(pairwise, k, other)[1]
            for other in k_list if other != k
        ):
            return k
    return copeland(pairwise, k_list)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    selections: dict[str, int],
    truth: dict[str, dict[int, bool]],
    label: str,
) -> None:
    resolved = total = 0
    for iid, chosen_k in selections.items():
        if iid in truth and chosen_k in truth[iid]:
            total += 1
            if truth[iid][chosen_k]:
                resolved += 1
    rate = resolved / total if total else 0.0
    print(f"  {label:30s}  {resolved:4d}/{total:<4d}  {rate:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--votes-path", required=True,
                   help="Pairwise comparator votes jsonl (blind or trace)")
    p.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR),
                   help="Harness run_evaluation/ dir for truth loading")
    p.add_argument("--k-list", default=",".join(str(k) for k in GREEDY8_K_LIST))
    p.add_argument("--out", default="",
                   help="Optional path to write JSON summary")
    args = p.parse_args()

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    votes_path = Path(args.votes_path)
    reports_dir = Path(args.reports_dir)

    print(f"[load] votes from {votes_path}")
    rows_by_iid: dict[str, list[dict]] = defaultdict(list)
    n_rows = 0
    with votes_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_iid[row["instance_id"]].append(row)
            n_rows += 1
    print(f"[load] {n_rows} rows across {len(rows_by_iid)} instances")

    print(f"[load] truth from {reports_dir}")
    truth = load_truth(reports_dir, k_list)
    print(f"[load] truth loaded for {len(truth)} instances")

    # Try to supplement truth from votes rows (trace_comparator attaches it)
    for iid, rows in rows_by_iid.items():
        for row in rows:
            for side, idx_key, res_key in [("a", "patch_a_idx", "patch_a_resolves"),
                                            ("b", "patch_b_idx", "patch_b_resolves")]:
                k = row.get(idx_key)
                resolved = row.get(res_key)
                if k is not None and resolved is not None:
                    truth.setdefault(iid, {})[int(k)] = bool(resolved)

    instances = sorted(rows_by_iid)
    print(f"\n[eval] {len(instances)} instances, K={k_list}\n")
    print(f"  {'Method':30s}  {'Resolved':>9}  Rate")
    print(f"  {'-'*30}  {'-'*9}  ------")

    results: dict[str, dict] = {}
    methods = [
        ("Copeland", copeland),
        ("Bracket elimination", bracket),
        ("Sequential king", sequential_king),
        ("Strict dominance (→Copeland)", strict_dominance),
    ]

    for label, fn in methods:
        selections: dict[str, int] = {}
        for iid in instances:
            pw = build_pairwise(rows_by_iid[iid])
            if not pw:
                selections[iid] = k_list[0]
                continue
            try:
                selections[iid] = fn(pw, k_list)
            except Exception as e:
                selections[iid] = k_list[0]
                print(f"  [warn] {iid} {label}: {e}", file=sys.stderr)

        resolved = total = 0
        for iid, chosen_k in selections.items():
            if iid in truth and chosen_k in truth[iid]:
                total += 1
                if truth[iid][chosen_k]:
                    resolved += 1
        rate = resolved / total if total else 0.0
        print(f"  {label:30s}  {resolved:4d}/{total:<4d}  {rate:.4f}")
        results[label] = {"resolved": resolved, "total": total, "resolve_rate": rate,
                          "selections": selections}

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "votes_path": str(votes_path),
            "n_instances": len(instances),
            "k_list": k_list,
            "methods": {
                label: {"resolved": v["resolved"], "total": v["total"],
                        "resolve_rate": v["resolve_rate"]}
                for label, v in results.items()
            },
        }
        out.write_text(json.dumps(summary, indent=2))
        print(f"\n[out] written to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
