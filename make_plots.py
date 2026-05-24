#!/usr/bin/env python3
"""Generate the 4 priority plots for the paper.

Plot 1: Candidate count vs resolve rate (Layer 1, n = 1..8)
Plot 2: Critic count vs resolve rate (Layer 2, R = 1..10)
Plot 3: Comparator count vs resolve rate (Layer 3, n_M = 1..M)  [needs comparator data]
Plot 4: Critic yes-rate threshold τ vs resolve rate (gate sweep)

All metrics use the confidence-weighted selector and denominator C
(any-K-of-pool has eval truth — matches orchestra-gpt's 80.5 % framing).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["font.family"] = "DejaVu Sans"


def load_binary(paths: list[str]) -> dict[str, dict[int, dict]]:
    by_iid: dict[str, dict[int, dict]] = defaultdict(dict)
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


def conf_yes(row, vote_subset=None):
    votes = row.get("binary_votes") or []
    if vote_subset is not None:
        votes = [v for v in votes if int(v.get("vote_idx")) in vote_subset]
    return sum(int(v.get("confidence") or 0) for v in votes if v.get("resolves") is True)


def yes_rate(row, vote_subset=None):
    votes = row.get("binary_votes") or []
    if vote_subset is not None:
        votes = [v for v in votes if int(v.get("vote_idx")) in vote_subset]
    valid = sum(1 for v in votes if v.get("resolves") is not None)
    yes = sum(1 for v in votes if v.get("resolves") is True)
    return yes / valid if valid else 0.0


def plot1_proposer(by_iid, k_list, iids, out_path: Path, R: int, label: str):
    """Cumulative marginal-coverage pool sweep, confidence-weighted, denominator C."""
    def oracle_hits(pool):
        hits = set()
        for iid in iids:
            rows = by_iid[iid]
            if any(rows[k].get("patch_resolves") is True for k in pool):
                hits.add(iid)
        return hits

    remaining = list(k_list)
    ordered_pool = []
    covered = set()
    while remaining:
        best_k = max(
            remaining,
            key=lambda k: (len(oracle_hits(ordered_pool + [k]) - covered), -k),
        )
        ordered_pool.append(best_k)
        covered = oracle_hits(ordered_pool)
        remaining.remove(best_k)

    def rate_pool(pool):
        res = tot = 0
        for iid in iids:
            rows = by_iid[iid]
            if not any(rows[k].get("patch_resolves") in (True, False) for k in pool):
                continue
            tot += 1
            scored = sorted(((k, conf_yes(rows[k])) for k in pool), key=lambda x: (-x[1], x[0]))
            k_pick = scored[0][0]
            if rows[k_pick].get("patch_resolves") is True:
                res += 1
        return res, tot, res / tot if tot else 0.0

    fixed_rates = [rate_pool(ordered_pool[:n])[2] for n in range(1, len(k_list)+1)]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = list(range(1, len(k_list)+1))
    ax.plot(xs, [r*100 for r in fixed_rates], "o-", color="#c0392b", lw=2.8, ms=8,
            label="Selector")
    # Reference lines
    res_orc = tot_orc = 0
    for iid in iids:
        rows = by_iid[iid]
        if any(rows[k].get("patch_resolves") in (True, False) for k in k_list):
            tot_orc += 1
            if any(rows[k].get("patch_resolves") is True for k in k_list):
                res_orc += 1
    oracle_rate = res_orc / tot_orc * 100 if tot_orc else 0
    ax.axhline(oracle_rate, color="#7f8c8d", linestyle=":", lw=1.5,
               label=f"Pool oracle = {oracle_rate:.1f}%")
    ax.axhline(80.5, color="#34495e", linestyle="-.", lw=1.5, alpha=0.65,
               label="16-candidate oracle = 80.5%")

    ax.set_xlabel("Number of candidates in pool (n)", fontsize=12)
    ax.set_ylabel("Resolve rate (%)", fontsize=12)
    ax.set_title(f"Layer 1 — Candidate pool size vs resolve rate\n(confidence-weighted selector, R={R})", fontsize=12)
    ax.set_xticks(xs)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, max(82, oracle_rate + 2))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot1] {out_path}")


def plot2_critic(by_iid, k_list, iids, out_path: Path, R: int, label: str):
    """Voter-count ablation, mean over all C(R, n) subsets, denominator C, conf-weighted."""

    def rate_with_votes(vote_subset):
        res = tot = 0
        for iid in iids:
            rows = by_iid[iid]
            if not any(rows[k].get("patch_resolves") in (True, False) for k in k_list):
                continue
            tot += 1
            scored = sorted(((k, conf_yes(rows[k], vote_subset)) for k in k_list), key=lambda x: (-x[1], x[0]))
            k_pick = scored[0][0]
            if rows[k_pick].get("patch_resolves") is True:
                res += 1
        return res / tot if tot else 0.0

    conf_means = []; conf_stds = []
    for n in range(1, R+1):
        subs = list(itertools.combinations(range(R), n))
        # Subsample large n combinatorial counts for speed
        if len(subs) > 200:
            import random
            random.seed(42 + n)
            subs = random.sample(subs, 200)
        conf_rates = [rate_with_votes(set(s)) for s in subs]
        conf_means.append(sum(conf_rates)/len(conf_rates))
        conf_stds.append((sum((r-conf_means[-1])**2 for r in conf_rates)/len(conf_rates))**0.5)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = list(range(1, R+1))
    ax.errorbar(xs, [m*100 for m in conf_means], yerr=[s*100 for s in conf_stds],
                fmt="o-", color="#c0392b", lw=2.5, ms=8, capsize=4,
                label="Confidence-weighted")

    # Reference lines
    res_orc = tot_orc = 0
    for iid in iids:
        rows = by_iid[iid]
        if any(rows[k].get("patch_resolves") in (True, False) for k in k_list):
            tot_orc += 1
            if any(rows[k].get("patch_resolves") is True for k in k_list):
                res_orc += 1
    oracle_rate = res_orc / tot_orc * 100 if tot_orc else 0
    ax.axhline(oracle_rate, color="#7f8c8d", linestyle=":", lw=1.5,
               label=f"Pool oracle = {oracle_rate:.1f}%")
    ax.axhline(80.5, color="#34495e", linestyle="-.", lw=1.5, alpha=0.65,
               label="16-candidate oracle = 80.5%")

    ax.set_xlabel("Number of binary critics per patch (R)", fontsize=12)
    ax.set_ylabel("Resolve rate (%)", fontsize=12)
    ax.set_title("Layer 2 — Critic count vs resolve rate\n(mean ± std over vote subsets)", fontsize=12)
    ax.set_xticks(xs)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, max(82, oracle_rate + 2))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot2] {out_path}")


def plot4_threshold(by_iid, k_list, iids, out_path: Path, R: int, label: str):
    """Critic yes-rate threshold τ vs resolve rate (denominator C, conf-weighted fallback)."""
    thresholds = [i / 20 for i in range(0, 21)]  # 0.00, 0.05, ..., 1.00 — finer at R=10

    def fallback_pick(rows, candidates):
        scored = sorted(((k, conf_yes(rows[k])) for k in candidates), key=lambda x: (-x[1], x[0]))
        return scored[0][0]

    fallback_rates = []
    avg_cands = []
    survivor_pairs = []
    pct_zero_cand = []
    pct_one_cand = []
    for thr in thresholds:
        res = tot = 0
        sum_cands = 0
        n_inst = 0
        n_zero = 0
        n_one = 0
        sum_pairs = 0
        for iid in iids:
            rows = by_iid[iid]
            if not any(rows[k].get("patch_resolves") in (True, False) for k in k_list):
                continue
            n_inst += 1
            tot += 1
            survivors = [k for k in k_list if yes_rate(rows[k]) >= thr]
            if not survivors:
                n_zero += 1
                survivors = [fallback_pick(rows, k_list)]
            elif len(survivors) == 1:
                n_one += 1
            sum_cands += len(survivors)
            sum_pairs += len(survivors) * (len(survivors) - 1) // 2
            k_pick = fallback_pick(rows, survivors)
            if rows[k_pick].get("patch_resolves") is True:
                res += 1
        fallback_rates.append(res/tot if tot else 0.0)
        avg_cands.append(sum_cands / n_inst if n_inst else 0.0)
        survivor_pairs.append(sum_pairs)
        pct_zero_cand.append(n_zero / n_inst * 100 if n_inst else 0.0)
        pct_one_cand.append(n_one / n_inst * 100 if n_inst else 0.0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = thresholds
    ax.plot(xs, [r*100 for r in fallback_rates], "o-", color="#c0392b", lw=2.5, ms=7,
            label="Selector-only (no comparator)")

    # Mark optimum
    best_thr_idx = max(range(len(fallback_rates)), key=lambda i: fallback_rates[i])
    ax.axvline(thresholds[best_thr_idx], color="#e67e22", linestyle="--", alpha=0.6,
               label=f"Best τ = {thresholds[best_thr_idx]:.2f}")

    res_orc = tot_orc = 0
    for iid in iids:
        rows = by_iid[iid]
        if any(rows[k].get("patch_resolves") in (True, False) for k in k_list):
            tot_orc += 1
            if any(rows[k].get("patch_resolves") is True for k in k_list):
                res_orc += 1
    oracle_rate = res_orc / tot_orc * 100 if tot_orc else 0
    ax.axhline(oracle_rate, color="#7f8c8d", linestyle=":", lw=1.5,
               label=f"Pool oracle = {oracle_rate:.1f}%")

    ax.set_xlabel("Yes-rate threshold τ (fraction of R critics saying YES)", fontsize=12)
    ax.set_ylabel("Resolve rate (%)", fontsize=12)
    ax.set_title(f"Layer 3 gate — Threshold vs selector resolve rate\n(R={R}, confidence-weighted)", fontsize=12)
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, max(82, oracle_rate + 2))

    # Right pane: candidate count after the gate. Keep this as one line so the
    # filtering behavior is readable without a second axis.
    ax2.plot(xs, avg_cands, "o-", color="#16a085", lw=2.5, ms=7,
             label="Average survivors")
    ax2.set_xlabel("Yes-rate threshold τ", fontsize=12)
    ax2.set_ylabel("Average surviving candidates", fontsize=12, color="#16a085")
    ax2.set_title("Gate selectivity", fontsize=12)
    ax2.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="y", labelcolor="#16a085")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot4] {out_path}")


def plot3_comparator_placeholder(out_path: Path):
    """Placeholder — will populate once comparator data exists."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.text(0.5, 0.55, "Layer 3 (Comparator) — pending", ha="center", va="center",
            transform=ax.transAxes, fontsize=18, color="#c0392b", weight="bold")
    ax.text(0.5, 0.40,
            "Per the paper's Proposition 1, comparator edge requires\n"
            "per-candidate execution-derived (D3) signal that we have\n"
            "not yet wired into the classifier prompt.\n\n"
            "Next step: mine cached harness reports for per-(iid, K)\n"
            "post-patch FAIL_TO_PASS / PASS_TO_FAIL outcomes and add\n"
            "to dynamic-hints. Run small comparator pilot. Then plot.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Layer 3 — Comparator count vs resolve rate", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot3] {out_path} (placeholder)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary-votes", required=True, nargs="+")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="R5_K07_dynamic")
    p.add_argument("--k-list", default="0,1,2,3,4,5,6,7")
    p.add_argument("--R", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    by_iid = load_binary(args.binary_votes)
    iids = sorted(iid for iid, rows in by_iid.items() if set(k_list).issubset(rows))
    print(f"Loaded {len(iids)} complete instances, R={args.R}")

    plot1_proposer(by_iid, k_list, iids, out / f"plot1_proposer_{args.label}.png", args.R, args.label)
    plot2_critic(by_iid, k_list, iids, out / f"plot2_critic_{args.label}.png", args.R, args.label)
    plot4_threshold(by_iid, k_list, iids, out / f"plot4_threshold_{args.label}.png", args.R, args.label)
    plot3_comparator_placeholder(out / f"plot3_comparator_{args.label}.png")
    print(f"\nAll plots written to {out}/")


if __name__ == "__main__":
    main()
