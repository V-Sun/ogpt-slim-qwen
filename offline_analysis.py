"""Offline analysis on cached critic + comparator votes — zero API calls.

Reads:
  outputs/stage1_critics_full/critic_votes.jsonl
  outputs/stage2_comparator_pilot/comparator_votes.jsonl

Produces:
  outputs/stage1_critics_full/threshold_sweep.json   (τ sweep at M=15)
  outputs/stage1_critics_full/m_ablation.json        (M sweep at τ*)
  outputs/stage2_comparator_pilot/r_ablation.json    (R sweep at M*, τ* with Copeland)
  outputs/RUN_SUMMARY.md                             (final report)

Decision signals:
  τ*  : maximizes (resolved_instances / total_instances) on Stage 1
  M*  : minimum M that retains ≥98% of M=15's resolved instances at τ*
  R*  : minimum R that retains ≥98% of R=10's committee resolve rate
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
STAGE1 = REPO / "outputs" / "stage1_critics_full"
STAGE2 = REPO / "outputs" / "stage2_comparator_pilot"
SUMMARY = REPO / "outputs" / "RUN_SUMMARY.md"

GLOBAL_SEED = 42

# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def load_critics():
    """Yield (iid, patch_idx, patch_resolves, votes) from stage1 jsonl."""
    f = STAGE1 / "critic_votes.jsonl"
    if not f.exists():
        raise SystemExit(f"missing {f} — run Stage 1 first")
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            yield (
                r["instance_id"],
                int(r["patch_idx"]),
                r.get("patch_resolves"),
                r.get("critic_votes", []),
            )


def load_comparators():
    """Yield (iid, a_idx, b_idx, votes) from stage2 jsonl. Empty iter if missing."""
    f = STAGE2 / "comparator_votes.jsonl"
    if not f.exists():
        return
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            yield (
                r["instance_id"],
                int(r["patch_a_idx"]),
                int(r["patch_b_idx"]),
                r.get("comparator_votes", []),
            )


# ----------------------------------------------------------------------------
# Survivor logic
# ----------------------------------------------------------------------------


FLAG_KEYS = (
    "addresses_root_cause",
    "preserves_existing_behavior",
    "tests_actually_test_the_issue",
    "no_unrelated_changes",
)


def votes_to_flag_pass_rates(votes: list[dict], m_subset: list[int] | None = None
                             ) -> dict[str, float] | None:
    """For a single patch's M critic votes, return the per-flag fraction of
    True votes among non-abstaining critics in the subset. Returns None if
    every critic abstained (effective M=0)."""
    if m_subset is None:
        m_subset = list(range(len(votes)))
    counts = {k: [0, 0] for k in FLAG_KEYS}  # [true, total_non_abstain]
    for vi in m_subset:
        if vi >= len(votes):
            continue
        v = votes[vi]
        flags = v.get("flags")
        if flags is None:
            continue  # abstain
        for k in FLAG_KEYS:
            counts[k][1] += 1
            if flags.get(k) is True:
                counts[k][0] += 1
    rates = {}
    for k, (t, n) in counts.items():
        if n == 0:
            return None
        rates[k] = t / n
    return rates


def patch_survives(votes: list[dict], tau: float,
                   m_subset: list[int] | None = None) -> bool:
    rates = votes_to_flag_pass_rates(votes, m_subset)
    if rates is None:
        return False
    return all(rates[k] >= tau for k in FLAG_KEYS)


# ----------------------------------------------------------------------------
# Threshold sweep (Stage 1, fixed M=15)
# ----------------------------------------------------------------------------


def threshold_sweep() -> dict:
    by_iid: dict[str, list[tuple[int, bool | None, list[dict]]]] = defaultdict(list)
    for iid, k, resolves, votes in load_critics():
        by_iid[iid].append((k, resolves, votes))
    taus = [round(0.05 * i, 2) for i in range(1, 21)]
    sweep = []
    best = None
    for tau in taus:
        n_inst = 0
        n_inst_with_survivor = 0
        n_inst_resolved = 0     # at least one surviving patch resolves
        survivor_counts = []
        survivor_resolve_count = 0
        n_survivors_total = 0
        for iid, recs in by_iid.items():
            n_inst += 1
            survivors = [
                (k, resolves) for (k, resolves, votes) in recs
                if patch_survives(votes, tau)
            ]
            survivor_counts.append(len(survivors))
            n_survivors_total += len(survivors)
            for _, r in survivors:
                if r:
                    survivor_resolve_count += 1
            if survivors:
                n_inst_with_survivor += 1
                if any(r for _, r in survivors):
                    n_inst_resolved += 1
        rec = {
            "tau": tau,
            "n_instances": n_inst,
            "n_instances_with_survivor": n_inst_with_survivor,
            "n_instances_resolved": n_inst_resolved,
            "resolve_rate_of_500": n_inst_resolved / max(1, n_inst),
            "survivor_resolve_fraction": (
                survivor_resolve_count / max(1, n_survivors_total)
            ),
            "median_survivors_per_instance": (
                statistics.median(survivor_counts) if survivor_counts else 0
            ),
            "mean_survivors_per_instance": (
                sum(survivor_counts) / max(1, len(survivor_counts))
            ),
            "survivor_count_histogram": dict(Counter(survivor_counts)),
        }
        sweep.append(rec)
        if best is None or rec["resolve_rate_of_500"] > best["resolve_rate_of_500"]:
            best = rec
    out = STAGE1 / "threshold_sweep.json"
    out.write_text(json.dumps({"sweep": sweep, "tau_star": best["tau"]}, indent=2))
    print(f"[threshold] τ*={best['tau']}  resolved={best['n_instances_resolved']}"
          f"/{best['n_instances']} = {best['resolve_rate_of_500']:.3f}  "
          f"survivor frac={best['survivor_resolve_fraction']:.3f}  "
          f"median surv={best['median_survivors_per_instance']}")
    return {"sweep": sweep, "tau_star": best["tau"], "best": best}


# ----------------------------------------------------------------------------
# M ablation at τ*
# ----------------------------------------------------------------------------


def m_ablation(tau_star: float, m_values=(1, 3, 5, 7, 9, 11, 13, 15),
               n_subsets: int = 100) -> dict:
    by_iid: dict[str, list[tuple[int, bool | None, list[dict]]]] = defaultdict(list)
    for iid, k, resolves, votes in load_critics():
        by_iid[iid].append((k, resolves, votes))
    rng = random.Random(GLOBAL_SEED)
    results = {}
    for M in m_values:
        per_subset = []
        for s in range(n_subsets):
            n_inst_resolved = 0
            n_inst = 0
            for iid, recs in by_iid.items():
                n_inst += 1
                # Common M-subset across all patches in this instance
                if M >= 15:
                    subset = list(range(15))
                else:
                    subset = rng.sample(range(15), M)
                survivors = [
                    (k, resolves) for (k, resolves, votes) in recs
                    if patch_survives(votes, tau_star, m_subset=subset)
                ]
                if any(r for _, r in survivors):
                    n_inst_resolved += 1
            per_subset.append(n_inst_resolved / max(1, n_inst))
        per_subset.sort()
        lo = per_subset[max(0, int(0.025 * n_subsets) - 1)]
        hi = per_subset[min(n_subsets - 1, int(0.975 * n_subsets))]
        med = per_subset[n_subsets // 2]
        results[M] = {
            "median_resolve_rate": med,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_subsets": n_subsets,
        }
        print(f"[M ablation] M={M:2d}  median={med:.3f}  CI95=[{lo:.3f},{hi:.3f}]")
    # Find smallest M retaining ≥98% of M=15
    target = results[max(m_values)]["median_resolve_rate"] * 0.98
    M_star = max(m_values)
    for M in sorted(m_values):
        if results[M]["median_resolve_rate"] >= target:
            M_star = M
            break
    out = STAGE1 / "m_ablation.json"
    out.write_text(json.dumps({"tau_star": tau_star, "results": results,
                               "m_star": M_star}, indent=2))
    return {"results": results, "m_star": M_star}


# ----------------------------------------------------------------------------
# Copeland tournament (Stage 2)
# ----------------------------------------------------------------------------


def copeland_winner(survivors: list[int],
                    matchup_winners: dict[tuple[int, int], str]) -> int | None:
    """Given a list of patch indices and matchup winners (per matchup pair as
    'A','B','TIE'), return the patch_idx with the most matchup wins. Ties
    broken by lower idx."""
    if not survivors:
        return None
    if len(survivors) == 1:
        return survivors[0]
    score = {p: 0 for p in survivors}
    for a in survivors:
        for b in survivors:
            if a >= b:
                continue
            w = matchup_winners.get((a, b))
            if w == "A":
                score[a] += 1
            elif w == "B":
                score[b] += 1
            # TIE: both 0
    # Return survivor with max score, tie-break by index
    return max(survivors, key=lambda p: (score[p], -p))


def aggregate_comparator_subset(votes: list[dict], r_subset: list[int]) -> str:
    """Aggregate R comparator votes (from votes list) by majority of
    matchup_winner. Returns 'A', 'B', or 'TIE'."""
    counts = Counter()
    for vi in r_subset:
        if vi >= len(votes):
            continue
        w = votes[vi].get("matchup_winner")
        if w in ("A", "B", "TIE"):
            counts[w] += 1
    if not counts:
        return "TIE"
    top = counts.most_common(1)[0][0]
    return top


# ----------------------------------------------------------------------------
# R ablation (Stage 2 pilot, at M*, τ*)
# ----------------------------------------------------------------------------


def r_ablation(tau_star: float, m_star: int,
               r_values=(1, 3, 5, 7, 10), n_subsets: int = 50) -> dict:
    # Build critic survivors for each pilot instance at (M*, τ*)
    by_iid_critics: dict[str, list[tuple[int, bool | None, list[dict]]]] = defaultdict(list)
    for iid, k, resolves, votes in load_critics():
        by_iid_critics[iid].append((k, resolves, votes))
    by_iid_cmp: dict[str, dict[tuple[int, int], list[dict]]] = defaultdict(dict)
    for iid, a, b, votes in load_comparators():
        by_iid_cmp[iid][(a, b)] = votes

    if not by_iid_cmp:
        out = STAGE2 / "r_ablation.json"
        out.write_text(json.dumps({"note": "no comparator data"}, indent=2))
        print("[R ablation] no comparator data — skipping")
        return {"results": {}, "r_star": None}

    pilot_iids = sorted(by_iid_cmp.keys())
    rng = random.Random(GLOBAL_SEED)
    results = {}
    # Determine survivors once per (iid, M-subset) — but for R-ablation we
    # use M=M* with the FIXED first-M-subset (not bootstrap). The brief says
    # "fixed M*, τ*" so we use a single M-subset of size M*.
    if m_star >= 15:
        m_subset = list(range(15))
    else:
        m_subset = sorted(rng.sample(range(15), m_star))
    survivors_by_iid: dict[str, list[tuple[int, bool | None]]] = {}
    for iid in pilot_iids:
        recs = by_iid_critics.get(iid, [])
        survivors = [
            (k, resolves) for (k, resolves, votes) in recs
            if patch_survives(votes, tau_star, m_subset=m_subset)
        ]
        survivors_by_iid[iid] = survivors

    for R in r_values:
        per_subset_resolve_rate = []
        for s in range(n_subsets):
            n_resolved = 0
            for iid in pilot_iids:
                survivors = survivors_by_iid[iid]
                if not survivors:
                    continue
                if len(survivors) == 1:
                    if survivors[0][1]:
                        n_resolved += 1
                    continue
                # Build matchup_winners for survivor pairs using R comparators
                if R >= 10:
                    r_subset = list(range(10))
                else:
                    r_subset = sorted(rng.sample(range(10), R))
                pair_winner: dict[tuple[int, int], str] = {}
                survivor_idxs = sorted(p for p, _ in survivors)
                for i, a in enumerate(survivor_idxs):
                    for b in survivor_idxs[i + 1:]:
                        votes = by_iid_cmp.get(iid, {}).get((a, b))
                        if votes is None:
                            pair_winner[(a, b)] = "TIE"
                        else:
                            pair_winner[(a, b)] = aggregate_comparator_subset(
                                votes, r_subset)
                winner_idx = copeland_winner(survivor_idxs, pair_winner)
                # Lookup whether winner resolves
                resolves_lookup = {p: r for p, r in survivors}
                if winner_idx is not None and resolves_lookup.get(winner_idx):
                    n_resolved += 1
            per_subset_resolve_rate.append(n_resolved / max(1, len(pilot_iids)))
        per_subset_resolve_rate.sort()
        lo = per_subset_resolve_rate[max(0, int(0.025 * n_subsets) - 1)]
        hi = per_subset_resolve_rate[min(n_subsets - 1, int(0.975 * n_subsets))]
        med = per_subset_resolve_rate[n_subsets // 2]
        results[R] = {
            "median_resolve_rate": med,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_subsets": n_subsets,
        }
        print(f"[R ablation] R={R:2d}  median={med:.3f}  CI95=[{lo:.3f},{hi:.3f}]")

    target = results[max(r_values)]["median_resolve_rate"] * 0.98
    R_star = max(r_values)
    for R in sorted(r_values):
        if results[R]["median_resolve_rate"] >= target:
            R_star = R
            break
    out = STAGE2 / "r_ablation.json"
    out.write_text(json.dumps({"tau_star": tau_star, "m_star": m_star,
                               "results": results, "r_star": R_star},
                              indent=2))
    return {"results": results, "r_star": R_star}


# ----------------------------------------------------------------------------
# Summary writer
# ----------------------------------------------------------------------------


def baseline_oracle_on_pilot() -> tuple[int, int]:
    """Oracle resolve on the pilot 50 instances (≥1 of K=8 resolves)."""
    by_iid: dict[str, list[bool]] = defaultdict(list)
    for iid, k, resolves, _ in load_critics():
        by_iid[iid].append(bool(resolves))
    pilot = sorted(by_iid.keys())
    # Pilot subset is whichever instances are in stage2_comparator_pilot
    cmp_iids = set()
    for iid, *_ in load_comparators():
        cmp_iids.add(iid)
    if cmp_iids:
        pilot = sorted(cmp_iids)
    n = len(pilot)
    n_resolved = sum(1 for iid in pilot if any(by_iid.get(iid, [])))
    return n_resolved, n


def summarize_usage() -> dict:
    """Aggregate token-spend across both stages from usage.jsonl."""
    out = {"critic": {"calls": 0, "input": 0, "output": 0, "reasoning": 0,
                      "cost_usd": 0.0},
           "comparator": {"calls": 0, "input": 0, "output": 0, "reasoning": 0,
                          "cost_usd": 0.0}}
    for stage_dir in (STAGE1, STAGE2):
        u = stage_dir / "usage.jsonl"
        if not u.exists():
            continue
        with u.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                kind = r.get("kind")
                if kind not in out:
                    continue
                out[kind]["calls"] += 1
                out[kind]["input"] += int(r.get("input_tokens", 0))
                out[kind]["output"] += int(r.get("output_tokens", 0))
                out[kind]["reasoning"] += int(r.get("reasoning_tokens", 0))
                out[kind]["cost_usd"] += float(r.get("cost_usd", 0.0))
    return out


def write_summary(threshold_res: dict, m_res: dict, r_res: dict) -> None:
    usage = summarize_usage()
    pilot_resolved, pilot_n = baseline_oracle_on_pilot()
    best = threshold_res["best"]
    tau_star = threshold_res["tau_star"]
    m_star = m_res.get("m_star")
    r_star = r_res.get("r_star")

    lines = [
        "# Ceiling experiment — RUN_SUMMARY",
        "",
        f"_Generated: {Path(__file__).name}_",
        "",
        "## API call totals",
        "",
        f"- Critic calls: {usage['critic']['calls']:,}",
        f"  - input tokens: {usage['critic']['input']:,}",
        f"  - output tokens: {usage['critic']['output']:,} "
        f"(of which reasoning: {usage['critic']['reasoning']:,})",
        f"  - cost: ${usage['critic']['cost_usd']:.2f}",
        f"- Comparator calls: {usage['comparator']['calls']:,}",
        f"  - input tokens: {usage['comparator']['input']:,}",
        f"  - output tokens: {usage['comparator']['output']:,} "
        f"(of which reasoning: {usage['comparator']['reasoning']:,})",
        f"  - cost: ${usage['comparator']['cost_usd']:.2f}",
        f"- **Total cost: "
        f"${usage['critic']['cost_usd'] + usage['comparator']['cost_usd']:.2f}** "
        f"(budget cap: $1800)",
        "",
        "## Stage 1 — threshold sweep",
        "",
        f"- τ* = **{tau_star}**",
        f"- Resolve rate of 500: "
        f"**{best['resolve_rate_of_500']:.3f}** "
        f"({best['n_instances_resolved']} / {best['n_instances']})",
        f"- Survivor resolve fraction: "
        f"**{best['survivor_resolve_fraction']:.3f}**",
        f"- Median survivors per instance: "
        f"**{best['median_survivors_per_instance']}**",
        f"- Mean survivors per instance: "
        f"**{best['mean_survivors_per_instance']:.2f}**",
        f"- Survivor-count histogram: `{best['survivor_count_histogram']}`",
        "",
        "## Stage 1 — M ablation (at τ*)",
        "",
        "| M | median resolve rate | CI95 lo | CI95 hi |",
        "|---|---|---|---|",
    ]
    for M, rec in sorted((m_res.get("results") or {}).items()):
        lines.append(f"| {M} | {rec['median_resolve_rate']:.3f} | "
                     f"{rec['ci95_lo']:.3f} | {rec['ci95_hi']:.3f} |")
    lines += [
        "",
        f"- M* (smallest M retaining ≥98% of M=15): **{m_star}**",
        "",
        "## Stage 2 — R ablation (50 pilot instances, M*, τ*)",
        "",
    ]
    if r_res.get("results"):
        lines += [
            "| R | median resolve rate | CI95 lo | CI95 hi |",
            "|---|---|---|---|",
        ]
        for R, rec in sorted(r_res["results"].items()):
            lines.append(f"| {R} | {rec['median_resolve_rate']:.3f} | "
                         f"{rec['ci95_lo']:.3f} | {rec['ci95_hi']:.3f} |")
        lines.append("")
        lines.append(f"- R* (smallest R retaining ≥98% of R=10): **{r_star}**")
    else:
        lines.append("_No comparator data — Stage 2 skipped or not yet run._")
    lines += [
        "",
        "## Baselines (for context)",
        "",
        f"- Oracle resolve on pilot ({pilot_n} instances): "
        f"**{pilot_resolved}** ({pilot_resolved / max(1, pilot_n):.3f}) "
        "— ≥1 of K=8 resolves",
        "",
        "## Decision signal",
        "",
        f"- Median survivors per instance at τ* = "
        f"**{best['median_survivors_per_instance']}**",
        "  - 1 → comparators unnecessary (the gate alone picks a unique patch)",
        "  - ≥3 → comparators are doing real work; full Stage 2 worth funding",
        "",
        "## Recommendations for next session",
    ]
    median_surv = best["median_survivors_per_instance"]
    if median_surv >= 3:
        lines += [
            "- Median survivors ≥3 — fund full Stage 2 (run comparators on all 500).",
            f"- At R*={r_star} per matchup, projected cost ≈ "
            f"${(usage['comparator']['cost_usd'] / max(1, pilot_n)) * 500:.0f}"
            " for full coverage.",
        ]
    elif median_surv == 1:
        lines += [
            "- Median survivors = 1 — gate alone selects a unique patch; "
            "comparators add no expected value.",
            "- Skip the full Stage 2 expansion.",
        ]
    else:
        lines += [
            "- Median survivors = 2 — comparator value is borderline; "
            "review per-instance survivor distribution before deciding.",
        ]
    SUMMARY.write_text("\n".join(lines))
    print(f"[summary] wrote {SUMMARY}")


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true",
                   help="Run threshold sweep + M ablation + R ablation + summary")
    p.add_argument("--threshold", action="store_true")
    p.add_argument("--m-ablation", action="store_true")
    p.add_argument("--r-ablation", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    if not any((args.all, args.threshold, args.m_ablation,
                args.r_ablation, args.summary)):
        p.print_help()
        return 2

    threshold_res = m_res = r_res = {}
    if args.threshold or args.all:
        threshold_res = threshold_sweep()
    if args.m_ablation or args.all:
        if not threshold_res:
            t = json.loads((STAGE1 / "threshold_sweep.json").read_text())
            threshold_res = {
                "tau_star": t["tau_star"],
                "best": next((r for r in t["sweep"] if r["tau"] == t["tau_star"]),
                             None),
            }
        m_res = m_ablation(threshold_res["tau_star"])
    if args.r_ablation or args.all:
        if not threshold_res:
            t = json.loads((STAGE1 / "threshold_sweep.json").read_text())
            threshold_res = {
                "tau_star": t["tau_star"],
                "best": next((r for r in t["sweep"] if r["tau"] == t["tau_star"]),
                             None),
            }
        if not m_res:
            m = json.loads((STAGE1 / "m_ablation.json").read_text())
            m_res = {"results": m["results"], "m_star": m["m_star"]}
        r_res = r_ablation(threshold_res["tau_star"], m_res["m_star"])
    if args.summary or args.all:
        write_summary(threshold_res, m_res, r_res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
