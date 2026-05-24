#!/usr/bin/env python3
"""Resolve-rate analyzer for the xhigh+greedy8+dynamic-hints full500 run.

Cache layout: one directory with K=0,1,2,3,4,5,6,7 all in the same
binary_votes.jsonl. Same row schema as direct_binary_hints_*.

Usage:
  python3 analyze_xhigh_greedy8.py [path]
  default path: outputs/direct_binary_dynamic_hints_xhigh_greedy8_full500/binary_votes.jsonl
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
rc.K_PATCHES = 16  # extend so load_resolve_bools loads K=0..15

DEFAULT = ROOT / "outputs/direct_binary_dynamic_hints_xhigh_greedy8_full500/binary_votes.jsonl"
GREEDY8 = [0, 1, 2, 3, 4, 5, 6, 7]
TRUTH = rc.load_resolve_bools()  # {iid: {k: bool}} loaded once


def load_votes(path: Path) -> dict:
    """{iid: {k: {'yes': int, 'total': int, 'resolves_truth': bool|None}}}"""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
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
            no = sum(1 for v in votes if v.get("resolves") is False)
            total = yes + no
            # If duplicate (iid,k), keep the most-complete row (most votes)
            existing = out[iid].get(k)
            if existing and (existing["yes"] + existing["total"] - existing["yes"]) >= total:
                continue
            out[iid][k] = {
                "yes": yes,
                "total": total,
                "n_votes": len(votes),
                # IMPORTANT: cache's patch_resolves is None for K=8+ due to old bug
                # — pull truth authoritatively from load_resolve_bools (K=0..15)
                "resolves_truth": TRUTH.get(iid, {}).get(k),
            }
    return out


def evaluate(votes: dict, lineup: list[int], label: str):
    n_total = 0
    n_resolved = 0
    oracle_resolved = 0
    no_winner = 0
    no_truth = 0
    incomplete = 0
    n_complete = 0
    per_k_count = defaultdict(int)
    yes_dist = defaultdict(int)

    for iid, kmap in votes.items():
        present = [k for k in lineup if k in kmap]
        for k in present:
            per_k_count[k] += 1

        if len(present) < len(lineup):
            incomplete += 1
        else:
            n_complete += 1

        resolved_in_lineup = [k for k in present if kmap[k].get("resolves_truth") is True]
        if resolved_in_lineup:
            oracle_resolved += 1

        # Pick winner: most-yes among lineup, tiebreak lowest-k
        cands = []
        for k in lineup:
            if k not in kmap:
                continue
            v = kmap[k]
            cands.append((k, v["yes"]))
        if not cands:
            no_winner += 1
            continue
        cands.sort(key=lambda t: (-t[1], t[0]))
        winner_k, winner_yes = cands[0]
        yes_dist[winner_yes] += 1
        truth = kmap[winner_k].get("resolves_truth")
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
    print(f"  winner yes-count dist: {dict(sorted(yes_dist.items()))}")
    print(f"  per-K row count:")
    for k in lineup:
        print(f"    K={k:2d}: {per_k_count[k]}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print(f"loading: {path}")
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        sys.exit(1)
    votes = load_votes(path)
    n_rows = sum(len(v) for v in votes.values())
    print(f"loaded: {n_rows} (iid,k) rows over {len(votes)} iids")
    evaluate(votes, GREEDY8, "xhigh+dynamic+greedy-8 {0,1,2,3,4,5,6,7}")


if __name__ == "__main__":
    main()
