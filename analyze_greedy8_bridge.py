#!/usr/bin/env python3
"""Re-aggregate static-hints binary votes using the greedy-8 K-subset {0,1,2,3,4,5,6,7}.

Combines:
  outputs/direct_binary_hints_full500/binary_votes.jsonl       (K=0..7)
  outputs/direct_binary_hints_greedy8_extra/binary_votes.jsonl (K=8,9,12,15)

Produces:
  - resolve rate using K=0..7 baseline (sanity check vs prior 72.0%)
  - resolve rate using greedy-8 lineup {0,1,2,3,4,5,6,7}
  - lineup oracle ceilings (any-of-K resolves)
  - per-K coverage diagnostics
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/vsun/ogpt-slim-qwen")
FULL500 = ROOT / "outputs/direct_binary_hints_full500/binary_votes.jsonl"
GREEDY = ROOT / "outputs/direct_binary_hints_greedy8_extra/binary_votes.jsonl"

BASELINE_K = list(range(8))                    # K=0..7
GREEDY8 = [0, 1, 2, 3, 4, 5, 6, 7]


def load_votes(path: Path) -> dict:
    """{iid: {k: {'yes': int, 'total': int, 'resolves_truth': bool|None}}}"""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row["instance_id"]
            k = int(row["patch_idx"])
            votes = row.get("binary_votes") or []
            yes = sum(1 for v in votes if v.get("resolves") is True)
            no  = sum(1 for v in votes if v.get("resolves") is False)
            total = yes + no
            out[iid][k] = {
                "yes": yes,
                "total": total,
                "resolves_truth": row.get("patch_resolves"),  # may be None
            }
    return out


def merge(a: dict, b: dict) -> dict:
    """Right-overrides for any (iid,k) collision; both sources should be disjoint."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for src in (a, b):
        for iid, kmap in src.items():
            for k, v in kmap.items():
                out[iid][k] = v
    return out


def pick_winner(kmap: dict[int, dict], lineup: list[int]) -> tuple[int | None, dict]:
    """Most-yes within lineup, tiebreak lowest-k. Skip K's missing from cache."""
    candidates = []
    for k in lineup:
        if k not in kmap:
            continue
        v = kmap[k]
        if v["total"] == 0:
            # No usable votes (all abstained / unparseable) — treat as 0 yes
            candidates.append((k, 0, v))
        else:
            candidates.append((k, v["yes"], v))
    if not candidates:
        return None, {}
    # Sort: highest yes-count first, then lowest k
    candidates.sort(key=lambda t: (-t[1], t[0]))
    winner_k, _, vinfo = candidates[0]
    return winner_k, vinfo


def evaluate(votes: dict, lineup: list[int], label: str):
    """Print resolve rate, oracle ceiling, and coverage for a lineup."""
    n_total = 0
    n_resolved = 0
    oracle_resolved = 0
    no_winner = 0
    no_truth = 0
    incomplete = 0
    n_complete = 0
    per_k_count = defaultdict(int)

    for iid, kmap in votes.items():
        # Coverage check: how many K's of the lineup are present
        present = [k for k in lineup if k in kmap]
        for k in present:
            per_k_count[k] += 1

        if len(present) < len(lineup):
            incomplete += 1
        else:
            n_complete += 1

        # Oracle: any K in lineup resolves
        resolved_in_lineup = [k for k in present if kmap[k].get("resolves_truth") is True]
        if resolved_in_lineup:
            oracle_resolved += 1

        winner_k, vinfo = pick_winner(kmap, lineup)
        if winner_k is None:
            no_winner += 1
            continue
        truth = vinfo.get("resolves_truth")
        if truth is None:
            no_truth += 1
            continue
        n_total += 1
        if truth is True:
            n_resolved += 1

    print(f"\n=== {label} (lineup={lineup}) ===")
    print(f"  instances total:       {len(votes)}")
    print(f"  complete coverage:     {n_complete}")
    print(f"  incomplete coverage:   {incomplete}")
    print(f"  no winner pickable:    {no_winner}")
    print(f"  winner truth==None:    {no_truth}")
    print(f"  scored:                {n_total}")
    if n_total:
        print(f"  resolved:              {n_resolved}/{n_total}  =  {100*n_resolved/n_total:.1f}%")
    print(f"  oracle (any-K hits):   {oracle_resolved}/{len(votes)}  =  {100*oracle_resolved/len(votes):.1f}%")
    print(f"  per-K row count:")
    for k in lineup:
        print(f"    K={k:2d}: {per_k_count[k]}")


def main():
    full = load_votes(FULL500)
    greedy = load_votes(GREEDY)
    print(f"loaded full500: {sum(len(v) for v in full.values())} (iid,k) rows  /  {len(full)} iids")
    print(f"loaded greedy:  {sum(len(v) for v in greedy.values())} (iid,k) rows  /  {len(greedy)} iids")

    combined = merge(full, greedy)
    print(f"combined:       {sum(len(v) for v in combined.values())} (iid,k) rows  /  {len(combined)} iids")

    # Headline reproduction
    evaluate(combined, BASELINE_K, "K=0..7 baseline (sanity vs prior 72.0%)")

    # Greedy-8 lineup
    evaluate(combined, GREEDY8, "Greedy-8 {0,1,2,3,4,5,6,7} (oracle ceiling 80.3%)")


if __name__ == "__main__":
    main()
