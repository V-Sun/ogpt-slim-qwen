#!/usr/bin/env python3
"""Offline aggregation of the comparator-hints pilot cache.

Process killed before printing pilot decision; reconstruct it from the cache.

Aggregation:
  per vote: weight 1 if swap_1==swap_2 (and both 'A'|'B'), else 0.5 to TIE if
            swap_1!=swap_2 (and both non-null). Abstains are skipped.
  per matchup: sum weights for A, B, TIE; winner = max (lowest-k tiebreak among A/B,
               TIE if TIE has the most weight).
  per instance: Copeland-style — count wins (A/B winners), tiebreak lowest-k.
  resolve check: load_resolve_bools from orchestra-gpt evaluation cache.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/vsun/ogpt-slim-qwen")
CACHE = ROOT / "outputs/comparator_dynamic_hints_pilot/comparator_votes.jsonl"

# Reuse run_ceiling.load_resolve_bools and pilot subset
sys.path.insert(0, str(ROOT))
import importlib
rc = importlib.import_module("run_ceiling")


def aggregate_votes(votes: list[dict]) -> dict:
    """Returns {'A': float, 'B': float, 'TIE': float}."""
    out = {"A": 0.0, "B": 0.0, "TIE": 0.0}
    for v in votes:
        s1 = v.get("position_swap_1")
        s2 = v.get("position_swap_2")
        if s1 is None or s2 is None:
            continue  # abstain
        if s1 not in ("A", "B", "TIE") or s2 not in ("A", "B", "TIE"):
            continue
        if s1 == s2:
            out[s1] += 1.0
        elif {s1, s2} == {"A", "B"}:
            out["TIE"] += 1.0  # disagreement → tie
        else:
            # one is TIE, one is A or B — give half-weight to the non-TIE direction
            non_tie = s1 if s1 != "TIE" else s2
            out[non_tie] += 0.5
            out["TIE"] += 0.5
    return out


def matchup_winner(weights: dict) -> str:
    """Return 'A', 'B', or 'TIE'. Ties go to TIE."""
    if not any(weights.values()):
        return "TIE"
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    if items[0][1] == items[1][1]:
        return "TIE"
    return items[0][0]


def main():
    rows: list[dict] = []
    with CACHE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Build {iid: {(a, b): weights}}
    matchups: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for r in rows:
        iid = r["instance_id"]
        a, b = int(r["patch_a_idx"]), int(r["patch_b_idx"])
        weights = aggregate_votes(r["comparator_votes"] or [])
        matchups[iid][(a, b)] = weights

    print(f"loaded: {len(rows)} matchups across {len(matchups)} instances")
    expected_per_instance = 28  # C(8,2)
    incomplete = [iid for iid, m in matchups.items() if len(m) < expected_per_instance]
    print(f"complete instances: {len(matchups) - len(incomplete)}")
    print(f"incomplete instances: {len(incomplete)}  (skipping)")

    # Per instance Copeland: each patch's win count
    truth = rc.load_resolve_bools()  # {iid: {k: bool}}
    n_total = 0
    n_resolved = 0
    no_truth = 0
    winner_dist = defaultdict(int)
    matchup_winner_dist = defaultdict(int)

    for iid, m in matchups.items():
        if len(m) < expected_per_instance:
            continue
        wins = defaultdict(int)
        ties = defaultdict(int)
        for (a, b), w in m.items():
            mw = matchup_winner(w)
            matchup_winner_dist[mw] += 1
            if mw == "A":
                wins[a] += 1
            elif mw == "B":
                wins[b] += 1
            else:
                ties[a] += 1
                ties[b] += 1
        # Pick winner: max wins; tiebreak lowest-k
        if not wins:
            # all ties → lowest k
            winner_k = 0
        else:
            ranked = sorted(range(8), key=lambda k: (-wins[k], k))
            winner_k = ranked[0]
        winner_dist[winner_k] += 1

        # Resolve check
        t = truth.get(iid, {})
        winner_truth = t.get(winner_k)
        if winner_truth is None:
            no_truth += 1
            continue
        n_total += 1
        if winner_truth is True:
            n_resolved += 1

    print()
    print(f"=== Comparator + dynamic hints, 50 pilot ===")
    print(f"matchup winner distribution: {dict(matchup_winner_dist)}")
    print(f"winner-K distribution:       {dict(winner_dist)}")
    print(f"scored: {n_total}    no-truth winners: {no_truth}")
    if n_total:
        print(f"Resolve rate: {n_resolved}/{n_total}  =  {100*n_resolved/n_total:.1f}%")


if __name__ == "__main__":
    main()
