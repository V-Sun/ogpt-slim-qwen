#!/usr/bin/env python3
"""Greedy-8 re-aggregation that loads truth labels for K=8,9,12,15 too.

The original analyzer relied on per-row `patch_resolves` in the cache, which
was None for K=8/9/12/15 because load_resolve_bools is K=0..7-bound. Here
we extend K_PATCHES to 16 before calling load_resolve_bools, so K=8..15
truth is included.

Inputs:
  outputs/direct_binary_hints_full500/binary_votes.jsonl       (K=0..7, static hints)
  outputs/direct_binary_hints_greedy8_extra/binary_votes.jsonl (K=8,9,12,15, static hints)
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/vsun/ogpt-slim-qwen")
sys.path.insert(0, str(ROOT))
import importlib
rc = importlib.import_module("run_ceiling")
rc.K_PATCHES = 16  # extend to cover K=0..15 in load_resolve_bools

FULL500 = ROOT / "outputs/direct_binary_hints_full500/binary_votes.jsonl"
GREEDY = ROOT / "outputs/direct_binary_hints_greedy8_extra/binary_votes.jsonl"
GREEDY8 = [0, 1, 2, 3, 4, 5, 6, 7]
BASELINE_K = list(range(8))


def load_yes_counts(path: Path) -> dict:
    """{iid: {k: yes_count}}"""
    out: dict[str, dict[int, int]] = defaultdict(dict)
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
            yes = sum(1 for v in (row.get("binary_votes") or []) if v.get("resolves") is True)
            out[iid][k] = yes
    return out


def evaluate(yc: dict, truth: dict, lineup: list[int], label: str):
    n_total = 0
    n_resolved = 0
    oracle_resolved = 0
    no_winner = 0
    no_truth = 0
    n_complete = 0

    for iid in yc:
        present = [k for k in lineup if k in yc[iid]]
        if len(present) == len(lineup):
            n_complete += 1

        # Oracle: any K in lineup resolves (using authoritative truth)
        if any(truth.get(iid, {}).get(k) is True for k in present):
            oracle_resolved += 1

        cands = [(k, yc[iid][k]) for k in lineup if k in yc[iid]]
        if not cands:
            no_winner += 1
            continue
        cands.sort(key=lambda t: (-t[1], t[0]))
        winner_k = cands[0][0]
        t = truth.get(iid, {}).get(winner_k)
        if t is None:
            no_truth += 1
            continue
        n_total += 1
        if t is True:
            n_resolved += 1

    print(f"\n=== {label} (lineup={lineup}) ===")
    print(f"  instances total:      {len(yc)}")
    print(f"  complete coverage:    {n_complete}")
    print(f"  no winner pickable:   {no_winner}")
    print(f"  winner truth==None:   {no_truth}")
    print(f"  scored:               {n_total}")
    if n_total:
        print(f"  resolved:             {n_resolved}/{n_total}  =  {100*n_resolved/n_total:.1f}%")
    print(f"  oracle (any-K hits):  {oracle_resolved}/{len(yc)}  =  {100*oracle_resolved/len(yc):.1f}%")


def main():
    print("loading truth for K=0..15 (extended K_PATCHES)...")
    truth = rc.load_resolve_bools()
    n_iids_with_truth = len(truth)
    n_total_truth = sum(len(t) for t in truth.values())
    k_distribution = defaultdict(int)
    for iid, t in truth.items():
        for k in t:
            k_distribution[k] += 1
    print(f"  {n_iids_with_truth} iids with truth, {n_total_truth} (iid,k) labels")
    print(f"  per-K coverage: {dict(sorted(k_distribution.items()))}")

    yc_full = load_yes_counts(FULL500)
    yc_greedy = load_yes_counts(GREEDY)
    print(f"\nfull500 yes-counts: {sum(len(v) for v in yc_full.values())} (iid,k) over {len(yc_full)} iids")
    print(f"greedy   yes-counts: {sum(len(v) for v in yc_greedy.values())} (iid,k) over {len(yc_greedy)} iids")

    # Merge
    combined: dict[str, dict[int, int]] = defaultdict(dict)
    for src in (yc_full, yc_greedy):
        for iid, kmap in src.items():
            combined[iid].update(kmap)

    evaluate(combined, truth, BASELINE_K, "K=0..7 baseline (sanity)")
    evaluate(combined, truth, GREEDY8, "Greedy-8 {0,1,2,3,4,5,6,7}")


if __name__ == "__main__":
    main()
