#!/usr/bin/env python3
"""Generate paper plots for techniques that did not improve selection.

These figures intentionally focus on negative/flat ablations:
  1. Prompt-diversity committee vs simple baselines.
  2. Critic prompt diversity M-sweep saturation.
  3. Comparator prompt diversity R-sweep flatline.
  4. Threshold gate tradeoff: either no filtering or lost coverage.
  5. Extra binary voters beyond the first few do not help.

All inputs are cached/offline artifacts under outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "negative_results_plots"

mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 220
mpl.rcParams["font.family"] = "DejaVu Sans"


def pct(x: float) -> float:
    return 100.0 * x


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(path)


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def plot_committee_bars() -> None:
    """Pilot comparison of committee/prompt diversity variants."""
    labels = [
        "K0 baseline",
        "Legacy\ncommittee",
        "Prompt-diverse\ncommittee",
        "Binary,\nno hints",
        "Binary,\nstatic hints",
    ]
    rates = [66.0, 64.0, 64.0, 66.0, 68.0]
    costs = [0.0, 30.87, 276.33, 7.85, 10.13]
    colors = ["#7f8c8d", "#c0392b", "#c0392b", "#d35400", "#2980b9"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = range(len(labels))
    bars = ax.bar(xs, rates, color=colors, alpha=0.9)
    ax.axhline(66.0, color="#34495e", linestyle="--", lw=1.4, label="K0 pilot baseline")
    ax.axhline(74.0, color="#7f8c8d", linestyle=":", lw=1.5, label="8-candidate pilot oracle")
    for b, rate, cost in zip(bars, rates, costs):
        ax.text(b.get_x() + b.get_width() / 2, rate + 0.35,
                f"{rate:.0f}%\n${cost:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("Pilot resolve rate (%)")
    ax.set_title("Prompt-diverse committee did not beat simple baselines")
    ax.set_ylim(60, 76)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "plot_negative_1_committee_prompt_diversity.png")


def plot_critic_m_saturation() -> None:
    data = load_json("outputs/stage1_critics_full/m_ablation.json")
    results = data["results"]
    xs = [int(k) for k in sorted(results, key=int)]
    med = [pct(results[str(x)]["median_resolve_rate"]) for x in xs]
    lo = [pct(results[str(x)]["ci95_lo"]) for x in xs]
    hi = [pct(results[str(x)]["ci95_hi"]) for x in xs]
    yerr = [[m - l for m, l in zip(med, lo)], [h - m for h, m in zip(hi, med)]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(xs, med, yerr=yerr, fmt="o-", color="#c0392b", lw=2.5,
                ms=7, capsize=4, label="Any-survivor coverage")
    ax.axhline(med[-1], color="#7f8c8d", linestyle=":", lw=1.4,
               label=f"Saturation = {med[-1]:.1f}%")
    ax.set_xlabel("Number of prompt-diverse critics per patch (M)")
    ax.set_ylabel("Any-survivor resolve ceiling (%)")
    ax.set_title("More prompt-diverse critics saturated after M=3-5")
    ax.set_xticks(xs)
    ax.set_ylim(72, 78.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "plot_negative_2_prompt_diverse_critic_m.png")


def plot_comparator_r_flatline() -> None:
    data = load_json("outputs/stage2_comparator_pilot/r_ablation.json")
    results = data["results"]
    xs = [int(k) for k in sorted(results, key=int)]
    med = [pct(results[str(x)]["median_resolve_rate"]) for x in xs]
    lo = [pct(results[str(x)]["ci95_lo"]) for x in xs]
    hi = [pct(results[str(x)]["ci95_hi"]) for x in xs]
    yerr = [[m - l for m, l in zip(med, lo)], [h - m for h, m in zip(hi, med)]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(xs, med, yerr=yerr, fmt="o-", color="#c0392b", lw=2.5,
                ms=7, capsize=4, label="Prompt-diverse comparator")
    ax.axhline(66.0, color="#34495e", linestyle="--", lw=1.4, label="K0 pilot baseline")
    ax.axhline(74.0, color="#7f8c8d", linestyle=":", lw=1.5, label="Pilot oracle")
    ax.set_xlabel("Comparator votes per matchup (R)")
    ax.set_ylabel("Pilot resolve rate (%)")
    ax.set_title("Comparator prompt diversity was flat below baseline")
    ax.set_xticks(xs)
    ax.set_ylim(60, 76)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "plot_negative_3_prompt_diverse_comparator_r.png")


def plot_threshold_gate_failure() -> None:
    data = load_json("outputs/stage1_critics_full/threshold_sweep.json")
    sweep = data["sweep"]
    xs = [r["tau"] for r in sweep]
    resolve = [pct(r["resolve_rate_of_500"]) for r in sweep]
    survivors = [r["mean_survivors_per_instance"] for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.plot(xs, resolve, "o-", color="#c0392b", lw=2.5, ms=6)
    ax1.axhline(78.5, color="#7f8c8d", linestyle=":", lw=1.5, label="8-candidate oracle")
    ax1.set_xlabel("Critic threshold")
    ax1.set_ylabel("Any-survivor resolve ceiling (%)")
    ax1.set_title("Strict thresholds lost resolving patches")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower left", fontsize=9)

    ax2.plot(xs, survivors, "o-", color="#16a085", lw=2.5, ms=6)
    ax2.axhline(8, color="#7f8c8d", linestyle=":", lw=1.2)
    ax2.set_xlabel("Critic threshold")
    ax2.set_ylabel("Mean surviving candidates")
    ax2.set_title("Loose thresholds barely filtered")
    ax2.set_ylim(0, 8.4)
    ax2.grid(True, alpha=0.25)
    save(fig, "plot_negative_4_threshold_gate_tradeoff.png")


def plot_extra_binary_voters() -> None:
    data = load_json("outputs/r10_K07_dynamic_high/voter_ablation.json")
    summary = data["summary"]["selectors"]
    xs = [int(k) for k in sorted(summary["majority"], key=int)]
    majority = [pct(summary["majority"][str(x)]["mean_rate"]) for x in xs]
    conf = [pct(summary["confidence_weighted"][str(x)]["mean_rate"]) for x in xs]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs, majority, "o-", color="#c0392b", lw=2.5, ms=7, label="Majority")
    ax.plot(xs, conf, "s--", color="#2980b9", lw=2.2, ms=6, label="Confidence-weighted")
    ax.set_xlabel("Binary critic votes per patch")
    ax.set_ylabel("Resolve rate (%)")
    ax.set_title("Extra binary voters saturated and then regressed")
    ax.set_xticks(xs)
    ax.set_ylim(75.0, 77.6)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "plot_negative_5_extra_binary_voters.png")


def plot_negative_summary_matrix() -> None:
    labels = [
        "Prompt-diverse\ncommittee",
        "Legacy\ncommittee",
        "No-hint\nbinary",
        "Comparator\nR=10",
        "Threshold\ngate",
        "Extra voters\nR=10",
    ]
    # Delta vs the relevant baseline/headline for each attempted lever.
    # Negative/zero means the added complexity did not buy resolve.
    deltas = [
        -2.0,  # 64 vs K0 pilot 66
        -2.0,  # 64 vs K0 pilot 66
        0.0,   # 66 vs K0 pilot 66
        0.0,   # R=10 comparator same as R=1 at 64
        0.0,   # thresholded binary fallback flat at 75.5
        -0.5,  # R10 conf-weighted below R5-ish peak in final R10 sweep
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = range(len(labels))
    colors = ["#c0392b" if d < 0 else "#7f8c8d" for d in deltas]
    bars = ax.bar(xs, deltas, color=colors, alpha=0.9)
    ax.axhline(0, color="#2c3e50", lw=1.2)
    for b, d in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width() / 2,
                d - 0.12 if d < 0 else d + 0.08,
                f"{d:+.1f} pp", ha="center",
                va="top" if d < 0 else "bottom", fontsize=9)
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("Resolve-rate lift over baseline/headline (pp)")
    ax.set_title("Complexity that did not improve the selector")
    ax.set_ylim(-3.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "plot_negative_0_summary_matrix.png")


def main() -> None:
    plot_negative_summary_matrix()
    plot_committee_bars()
    plot_critic_m_saturation()
    plot_comparator_r_flatline()
    plot_threshold_gate_failure()
    plot_extra_binary_voters()


if __name__ == "__main__":
    main()
