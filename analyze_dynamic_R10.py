#!/usr/bin/env python3
"""Combine R=5 (existing) + R=5 extension caches into R=10 dynamic-hints aggregation.

Question: do unanimous yes=5/5 ties at R=5 differentiate at R=10?

Compares:
  - R=5 baseline (most-yes, tiebreak lowest-k)
  - R=10 same scheme
  - R=10 confidence-weighted variants
  - Failure-mode decomposition (tiebreak vs discrimination misses)
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
TRUTH = rc.load_resolve_bools()

R5 = ROOT / "outputs/direct_binary_dynamic_hints_full500/binary_votes.jsonl"
R5_EXT = ROOT / "outputs/direct_binary_dynamic_hints_full500_R10ext/binary_votes.jsonl"


def load_votes(path: Path) -> dict:
    """{iid: {k: list of vote dicts}}"""
    out: dict[str, dict[int, list]] = defaultdict(dict)
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
            out[iid][k] = votes
    return out


def aggregate(per_iid_k_votes: dict, lineup: list[int], label: str, *, tiebreak: str = "lowest_k"):
    n_resolved = 0
    n_total = 0
    n_complete = 0
    yes_count_dist = defaultdict(int)

    for iid, kmap in per_iid_k_votes.items():
        present = [k for k in lineup if k in kmap]
        if len(present) == len(lineup):
            n_complete += 1

        cands = []
        for k in lineup:
            if k not in kmap:
                continue
            votes = kmap[k]
            yes = sum(1 for v in votes if v.get("resolves") is True)
            yes_confs = [v.get("confidence") or 0 for v in votes if v.get("resolves") is True]
            mean_yc = sum(yes_confs) / len(yes_confs) if yes_confs else 0
            cands.append((k, yes, mean_yc, len(votes)))

        if not cands:
            continue

        if tiebreak == "lowest_k":
            cands.sort(key=lambda c: (-c[1], c[0]))
        elif tiebreak == "yes_conf_mean":
            cands.sort(key=lambda c: (-c[1], -c[2], c[0]))
        else:
            raise ValueError(tiebreak)

        winner_k = cands[0][0]
        yes_count_dist[cands[0][1]] += 1
        t = TRUTH.get(iid, {}).get(winner_k)
        if t is None:
            continue
        n_total += 1
        if t is True:
            n_resolved += 1

    print(f"{label:<60}  {n_resolved}/{n_total} = {100*n_resolved/n_total:.1f}%  complete={n_complete}")


def failure_modes(per_iid_k_votes: dict, lineup: list[int], R: int):
    hits = misses_tie = misses_disc = no_resolver = 0
    for iid, kmap in per_iid_k_votes.items():
        if len(kmap) < len(lineup):
            continue
        cands = []
        for k in lineup:
            yes = sum(1 for v in kmap[k] if v.get("resolves") is True)
            cands.append((k, yes))
        cands.sort(key=lambda c: (-c[1], c[0]))
        winner_k, winner_y = cands[0]
        t = TRUTH.get(iid, {})
        if not any(t.get(k) is True for k in lineup):
            no_resolver += 1
            continue
        if t.get(winner_k) is True:
            hits += 1
            continue
        resolver_yes = [yes for k, yes in cands if t.get(k) is True]
        if max(resolver_yes) >= winner_y:
            misses_tie += 1
        else:
            misses_disc += 1
    total = hits + misses_tie + misses_disc
    print(f"  R={R}: {hits} hits / {misses_tie} tiebreak miss / {misses_disc} disc miss / {no_resolver} oracle miss")
    if total:
        print(f"        hit rate on winnable: {100*hits/total:.1f}%   ceiling if tiebreak solved: {100*(hits + misses_tie)/total:.1f}%")


def main():
    print("Loading caches...")
    r5 = load_votes(R5)
    ext = load_votes(R5_EXT)
    print(f"  R=5    cache: {sum(len(v) for v in r5.values())} (iid,k) rows over {len(r5)} iids")
    print(f"  R=ext  cache: {sum(len(v) for v in ext.values())} (iid,k) rows over {len(ext)} iids")

    # Build R=10 by merging votes per (iid, k)
    r10: dict[str, dict[int, list]] = defaultdict(dict)
    for src in (r5, ext):
        for iid, kmap in src.items():
            for k, votes in kmap.items():
                r10[iid].setdefault(k, []).extend(votes)
    n_combined = sum(len(v) for v in r10.values())
    print(f"  R=10 combined: {n_combined} (iid,k) rows over {len(r10)} iids")

    # Vote count distribution per (iid, k) — should be 10 if both caches have it
    vote_counts = defaultdict(int)
    for iid, kmap in r10.items():
        for k, votes in kmap.items():
            vote_counts[len(votes)] += 1
    print(f"  vote-count dist over (iid,k): {dict(sorted(vote_counts.items()))}")

    LINEUP = list(range(8))
    print()
    print("=== Resolve rates ===")
    aggregate(r5, LINEUP, "R=5  most-yes, lowest-k")
    aggregate(r5, LINEUP, "R=5  most-yes, yes-conf-mean tiebreak", tiebreak="yes_conf_mean")
    aggregate(r10, LINEUP, "R=10 most-yes, lowest-k")
    aggregate(r10, LINEUP, "R=10 most-yes, yes-conf-mean tiebreak", tiebreak="yes_conf_mean")

    print()
    print("=== Failure-mode decomposition ===")
    failure_modes(r5, LINEUP, 5)
    failure_modes(r10, LINEUP, 10)


if __name__ == "__main__":
    main()
