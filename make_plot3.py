#!/usr/bin/env python3
"""Generate Plot 3 — Comparator count vs resolve rate (Layer 3).

Reads:
  - pairwise comparator votes from anchored_dynamic_comparator output dir
  - binary R=10 cache for selector-side reference rate
  - per-instance candidate set (currently full K-list)

For m_votes in 1..M:
  - take all C(M, m) subsets of vote_idx (subsample for speed)
  - aggregate Copeland (confidence-weighted) on each subset's votes per pair
  - report resolve rate over instances with patch_resolves truth

Style: single solid line, denominator C, oracle reference lines.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["font.family"] = "DejaVu Sans"


def load_binary(paths: list[str]) -> dict[str, dict[int, dict]]:
    by_iid = defaultdict(dict)
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                iid = row["instance_id"]; k = int(row["patch_idx"])
                ex = by_iid[iid].get(k)
                if ex is None:
                    by_iid[iid][k] = row
                    continue
                merged = {int(v.get("vote_idx")): v for v in ex.get("binary_votes", [])}
                for v in row.get("binary_votes", []) or []:
                    try: merged[int(v.get("vote_idx"))] = v
                    except: continue
                ex["binary_votes"] = [merged[i] for i in sorted(merged)]
                if ex.get("patch_resolves") is None and row.get("patch_resolves") is not None:
                    ex["patch_resolves"] = row.get("patch_resolves")
    return by_iid


def load_comparator(path: Path):
    """Returns {iid: {(a, b): row}}."""
    out = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            a, b = int(row["patch_a_idx"]), int(row["patch_b_idx"])
            if a > b:
                a, b = b, a
            out[row["instance_id"]][(a, b)] = row
    return out


def winner_for_pair(row, vote_subset):
    counts = Counter()
    conf = Counter()
    by_idx = {int(v.get("vote_idx")): v for v in (row.get("votes") or [])}
    for vi in vote_subset:
        vote = by_idx.get(vi)
        if not vote:
            continue
        items = [(vote.get("primary") or {}).get("parsed")]
        if vote.get("swap"):
            items.append((vote.get("swap") or {}).get("parsed_normalized"))
        for parsed in items:
            if not parsed:
                continue
            w = parsed.get("winner")
            if w not in ("A", "B", "TIE"):
                continue
            c = int(parsed.get("confidence") or 1)
            counts[w] += 1
            conf[w] += c
    if not counts:
        return "TIE", 1
    winner = sorted(counts, key=lambda w: (-counts[w], -conf[w], w))[0]
    return winner, max(1, round(conf[winner] / counts[winner]))


def copeland(iid, candidates, cmp_rows, vote_subset):
    if len(candidates) < 2:
        return candidates[0] if candidates else None
    cand = sorted(candidates)
    scores = {k: 0.0 for k in cand}
    for i, a in enumerate(cand):
        for b in cand[i+1:]:
            row = cmp_rows.get(iid, {}).get((a, b))
            if not row:
                return None
            winner, c = winner_for_pair(row, vote_subset)
            if winner == "A":
                scores[a] += c; scores[b] -= c
            elif winner == "B":
                scores[b] += c; scores[a] -= c
    return sorted(scores, key=lambda k: (-scores[k], k))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary-votes", required=True, nargs="+")
    p.add_argument("--comparator-votes", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--k-list", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-votes", type=int, default=10)
    p.add_argument("--label", default="R10_K07_dynamic")
    args = p.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    binary = load_binary(args.binary_votes)
    cmp_rows = load_comparator(Path(args.comparator_votes))

    # Restrict iids to those with comparator coverage AND binary coverage
    iids = sorted(iid for iid in cmp_rows.keys() if set(k_list).issubset(binary.get(iid, {})))
    n_pairs_required = len(k_list) * (len(k_list) - 1) // 2
    iids = [iid for iid in iids if len(cmp_rows[iid]) == n_pairs_required]
    print(f"Iids with full comparator + binary coverage: {len(iids)}")

    # Selector reference rate
    def selector_rate():
        from itertools import islice
        from collections import Counter
        res = tot = 0
        for iid in iids:
            rows = binary[iid]
            scored = sorted(
                ((k, sum(int(v.get("confidence") or 0) for v in rows[k].get("binary_votes", []) if v.get("resolves") is True)) for k in k_list),
                key=lambda x: (-x[1], x[0])
            )
            k_pick = scored[0][0]
            r = rows[k_pick].get("patch_resolves")
            if r in (True, False):
                tot += 1
                if r: res += 1
        return res, tot, res/tot if tot else 0.0
    sel_res, sel_tot, sel_rate = selector_rate()
    print(f"Selector (no comparator) on these iids: {sel_res}/{sel_tot} = {sel_rate:.4f}")

    # Comparator m-sweep
    means = []; stds = []
    for n in range(1, args.max_votes + 1):
        subs = list(itertools.combinations(range(args.max_votes), n))
        if len(subs) > 200:
            random.seed(42 + n)
            subs = random.sample(subs, 200)
        rates = []
        for sub in subs:
            res = tot = 0
            for iid in iids:
                k_pick = copeland(iid, k_list, cmp_rows, set(sub))
                if k_pick is None:
                    continue
                r = binary[iid][k_pick].get("patch_resolves")
                if r in (True, False):
                    tot += 1
                    if r: res += 1
            rates.append(res/tot if tot else 0.0)
        m = sum(rates) / len(rates)
        s = (sum((r-m)**2 for r in rates) / len(rates)) ** 0.5
        means.append(m); stds.append(s)
        print(f"  m={n}: mean={m:.4f} std={s:.4f} subsets={len(subs)}")

    # Pool oracle on these iids
    res_orc = tot_orc = 0
    for iid in iids:
        rows = binary[iid]
        if any(rows[k].get("patch_resolves") in (True, False) for k in k_list):
            tot_orc += 1
            if any(rows[k].get("patch_resolves") is True for k in k_list):
                res_orc += 1
    oracle_rate = res_orc / tot_orc * 100 if tot_orc else 0

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = list(range(1, args.max_votes + 1))
    ax.errorbar(xs, [m*100 for m in means], yerr=[s*100 for s in stds],
                fmt="o-", color="#c0392b", lw=2.8, ms=8, capsize=4, label="Comparator (Copeland)")
    ax.axhline(sel_rate * 100, color="#2980b9", linestyle="--", lw=1.8,
               label=f"Selector only = {sel_rate*100:.1f}%")
    ax.axhline(oracle_rate, color="#7f8c8d", linestyle=":", lw=1.5,
               label=f"Pool oracle = {oracle_rate:.1f}%")
    ax.axhline(80.5, color="#34495e", linestyle="-.", lw=1.5, alpha=0.65,
               label="16-candidate oracle = 80.5%")
    ax.set_xlabel("Number of comparator votes per pair (M)", fontsize=12)
    ax.set_ylabel("Resolve rate (%)", fontsize=12)
    ax.set_title(f"Layer 3 — Comparator count vs resolve rate\n(D3-enriched anchored, K={','.join(map(str, k_list))}, n_iids={len(iids)})", fontsize=12)
    ax.set_xticks(xs)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, max(82, oracle_rate + 2))
    fig.tight_layout()
    fig.savefig(args.out_path)
    plt.close(fig)
    print(f"\n[plot3] wrote {args.out_path}")


if __name__ == "__main__":
    main()
