"""Generate RUN_SUMMARY_FULL500.md from the cached full-500 binary-hints output.

Compares against the 50-pilot result (68% with hints, 66% without hints) and
writes a paper-framing recommendation based on the full-500 number.

Reads:
  outputs/direct_binary_hints_full500/binary_votes.jsonl
  outputs/direct_binary_hints_full500/usage.jsonl
  outputs/direct_binary_hints_pilot/binary_votes.jsonl  (for comparison)
  outputs/direct_binary_pilot/binary_votes.jsonl        (no-hints pilot)
  outputs/stage1_critics_full/critic_votes.jsonl        (resolves data)

Writes to stdout (caller redirects to outputs/RUN_SUMMARY_FULL500.md).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
FULL500_DIR = REPO / "outputs" / "direct_binary_hints_full500"
PILOT_HINTS_DIR = REPO / "outputs" / "direct_binary_hints_pilot"
PILOT_NOHINTS_DIR = REPO / "outputs" / "direct_binary_pilot"

K_PATCHES = 8
R_VOTES = 5

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def load_binary_cache(d: Path) -> tuple[list[dict], dict]:
    jsonl = d / "binary_votes.jsonl"
    if not jsonl.exists():
        return [], {}
    rows = []
    by_iid = defaultdict(list)
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append(r)
            by_iid[r["instance_id"]].append(r)
    return rows, by_iid


def load_usage_totals(d: Path) -> tuple[int, float, int, int, int]:
    """Returns (n_calls, cost_usd, input_tokens, output_tokens, reasoning_tokens)."""
    jsonl = d / "usage.jsonl"
    if not jsonl.exists():
        return (0, 0.0, 0, 0, 0)
    n = 0
    cost = 0.0
    inp = 0
    out = 0
    rea = 0
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            cost += float(r.get("cost_usd", 0))
            inp += int(r.get("input_tokens", 0))
            out += int(r.get("output_tokens", 0))
            rea += int(r.get("reasoning_tokens", 0))
    return n, cost, inp, out, rea


def compute_resolve_rate(by_iid: dict) -> tuple[int, int, int, dict, int]:
    """Returns (n_resolved_majority, n_resolved_confidence, n_inst,
                yes_count_histogram, n_abstains)."""
    n_resolved_maj = 0
    n_resolved_conf = 0
    yes_counts = []
    abstain_count = 0
    iids = sorted(by_iid.keys())
    for iid in iids:
        rows = sorted(by_iid[iid], key=lambda r: r["patch_idx"])
        scores_maj = []
        scores_conf = []
        for row in rows:
            yes = sum(1 for v in row["binary_votes"] if v.get("resolves") is True)
            conf_yes = sum(int(v.get("confidence", 0))
                           for v in row["binary_votes"]
                           if v.get("resolves") is True)
            abst = sum(1 for v in row["binary_votes"]
                       if v.get("resolves") is None)
            abstain_count += abst
            yes_counts.append(yes)
            scores_maj.append((row["patch_idx"], yes,
                               bool(row.get("patch_resolves"))))
            scores_conf.append((row["patch_idx"], conf_yes,
                                bool(row.get("patch_resolves"))))
        scores_maj.sort(key=lambda x: (-x[1], x[0]))
        if scores_maj and scores_maj[0][2]:
            n_resolved_maj += 1
        scores_conf.sort(key=lambda x: (-x[1], x[0]))
        if scores_conf and scores_conf[0][2]:
            n_resolved_conf += 1
    hist = dict(Counter(yes_counts))
    return n_resolved_maj, n_resolved_conf, len(iids), hist, abstain_count


def load_resolves_data() -> tuple[dict, list]:
    """Load patch_resolves from the existing critic cache.
    {iid: {k: bool}} and sorted iid list."""
    f = REPO / "outputs" / "stage1_critics_full" / "critic_votes.jsonl"
    resolves: dict = {}
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            resolves.setdefault(r["instance_id"], {})[r["patch_idx"]] = bool(r.get("patch_resolves"))
    return resolves, sorted(resolves.keys())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    rows_full, by_iid_full = load_binary_cache(FULL500_DIR)
    rows_pilot_h, by_iid_pilot_h = load_binary_cache(PILOT_HINTS_DIR)
    rows_pilot_nh, by_iid_pilot_nh = load_binary_cache(PILOT_NOHINTS_DIR)

    if not rows_full:
        print("# RUN_SUMMARY_FULL500 — INCOMPLETE")
        print()
        print("No data in outputs/direct_binary_hints_full500/. The full run did not "
              "produce a cache. Check outputs/overnight_full500.log for errors.")
        return 0

    # Aggregate full-500
    n_calls, cost, inp, out, rea = load_usage_totals(FULL500_DIR)
    nm, nc, n_inst, hist, abstains = compute_resolve_rate(by_iid_full)

    # Pilot baselines (50 instances)
    nm_pilot_h, nc_pilot_h, n_pilot_h, hist_pilot_h, _ = compute_resolve_rate(by_iid_pilot_h)
    nm_pilot_nh, nc_pilot_nh, n_pilot_nh, hist_pilot_nh, _ = compute_resolve_rate(by_iid_pilot_nh)

    # Resolves data for K=0 baseline + oracle
    resolves, all_iids = load_resolves_data()
    iids_in_run = sorted(by_iid_full.keys())
    n_k0 = sum(1 for iid in iids_in_run if resolves.get(iid, {}).get(0, False))
    n_oracle = sum(1 for iid in iids_in_run
                   if any(resolves.get(iid, {}).get(k, False) for k in range(K_PATCHES)))

    # Output Markdown
    P = print
    P("# RUN_SUMMARY_FULL500 — Direct-binary + Execution Hints, Full SWE-Bench Verified")
    P()
    P("_Generated: generate_full500_summary.py_")
    P()
    P("## Headline")
    P()
    rate = nm / max(1, n_inst)
    pilot_rate = nm_pilot_h / max(1, n_pilot_h)
    P(f"Direct-binary classifier with FAIL_TO_PASS hints + patch file paths, "
      f"single committed prompt, R=5 votes per patch (most-yes, tiebreak lowest k_index).")
    P()
    P(f"- **Full-{n_inst} resolve rate: {nm}/{n_inst} = {rate:.3f} ({rate:.1%})**")
    P(f"- 50-pilot resolve rate (same setup): {nm_pilot_h}/{n_pilot_h} = "
      f"{pilot_rate:.3f} ({pilot_rate:.1%})")
    P(f"- Δ pilot → full: {(rate - pilot_rate):+.3f} ({(rate - pilot_rate) * 100:+.1f}pp)")
    P()
    if abs(rate - pilot_rate) <= 0.02:
        P("_Pilot generalized cleanly to full set (Δ ≤ 2pp)._")
    elif rate > pilot_rate:
        P("_Full set is **above** pilot — pilot was an unlucky-hard sample._")
    else:
        P("_Full set is **below** pilot — pilot was a lucky-easy sample. The headline "
          "should now be the full number, not the pilot._")
    P()

    P("## Cost + wall-clock")
    P()
    P(f"| Metric | Value |")
    P(f"|---|---:|")
    P(f"| API calls | {n_calls:,} |")
    P(f"| Input tokens | {inp:,} |")
    P(f"| Output tokens (incl reasoning) | {out:,} |")
    P(f"| Reasoning tokens | {rea:,} |")
    P(f"| **Total cost** | **${cost:.2f}** |")
    P(f"| Budget cap | $150.00 |")
    P(f"| Headroom under cap | ${150 - cost:.2f} |")
    P(f"| Abstains (R=5 vote slots that returned null) | {abstains} of {n_inst * K_PATCHES * R_VOTES:,} = "
      f"{abstains / max(1, n_inst * K_PATCHES * R_VOTES):.1%} |")
    P()

    P("## Comparison vs all prior experiments")
    P()
    P(f"All numbers on the **same 50-instance pilot** unless otherwise noted.")
    P()
    P("| Approach | Resolve rate | Cost | Notes |")
    P("|---|---:|---:|---|")
    P(f"| K=0 baseline (full {n_inst}) | {n_k0}/{n_inst} = {n_k0 / max(1, n_inst):.3f} | $0 | trivial |")
    P(f"| Oracle (any of K=0..7 resolves on full {n_inst}) | "
      f"{n_oracle}/{n_inst} = {n_oracle / max(1, n_inst):.3f} | $0 | ceiling |")
    P(f"| Diversity comparator (K=8/M=15/R=10, pilot) | 32/50 = 0.640 | $276 | most expensive, 2pp BELOW K=0 |")
    P(f"| Legacy single-prompt (K=5/M=3/R=3, pilot) | 32/50 = 0.640 | $30.87 | tied diversity exactly |")
    P(f"| Direct-binary (K=8/R=5, **no hints**, pilot) | "
      f"{nm_pilot_nh}/{n_pilot_nh} = {nm_pilot_nh / max(1, n_pilot_nh):.3f} | $7.85 | ties baseline |")
    P(f"| Direct-binary + **hints** (pilot) | "
      f"{nm_pilot_h}/{n_pilot_h} = {nm_pilot_h / max(1, n_pilot_h):.3f} | $10.13 | first lever that exceeded K=0 |")
    P(f"| **Direct-binary + hints (full {n_inst})** | "
      f"**{nm}/{n_inst} = {rate:.3f}** | **${cost:.2f}** | **headline** |")
    P()

    P("## Per-patch yes-count distribution: hints vs no-hints (pilot 50, then full)")
    P()
    P("Histogram of how many of R=5 votes said `resolves=True` per patch:")
    P()
    P("| Yes count | No-hints pilot (k×n=400) | With-hints pilot (k×n=400) | "
      f"With-hints full ({n_inst} × {K_PATCHES} = {n_inst * K_PATCHES}) |")
    P("|---:|---:|---:|---:|")
    for c in range(R_VOTES + 1):
        nh = hist_pilot_nh.get(c, 0)
        wh = hist_pilot_h.get(c, 0)
        full = hist.get(c, 0)
        P(f"| {c} | {nh} | {wh} | {full} |")
    P()
    n_unanimous_full = hist.get(R_VOTES, 0)
    n_total_patches_full = sum(hist.values())
    P(f"Unanimous-yes patches: {n_unanimous_full} / {n_total_patches_full} = "
      f"{n_unanimous_full / max(1, n_total_patches_full):.1%}")
    P()

    P("## Decision tree result")
    P()
    if rate >= 0.70:
        P(f"**≥70% on full {n_inst}** — grounding context is the lever. "
          f"This is the paper's positive contribution.")
    elif rate >= 0.67:
        P(f"**67-69% on full {n_inst}** — small but real grounding effect, "
          f"replicates the pilot. Worth writing up as the chosen approach.")
    elif rate >= 0.64:
        P(f"**64-66% on full {n_inst}** — grounding effect didn't hold at scale. "
          f"Pilot was probably an unlucky-easy sample. Writeup is a negative result.")
    else:
        P(f"**<64% on full {n_inst}** — unexpectedly low. "
          f"Investigate parse-fail rate or model issues before writing up.")
    P()

    P("## Recommendations for paper framing")
    P()
    if rate >= pilot_rate - 0.02:
        # Held up
        P("**Paper has a positive but narrow story.** Direct-binary with grounding "
          f"(FAIL_TO_PASS test names + file paths) reaches {rate:.1%} on full SWE-Bench Verified, "
          f"+{(rate - n_k0/max(1, n_inst))*100:.1f}pp over K=0 baseline and "
          f"{(n_oracle / max(1, n_inst) - rate)*100:.1f}pp under oracle.")
        P()
        P("**Recommended framing:**")
        P()
        P("1. **The negative finding is the central story**: weak-model committees "
          "(critics + comparators) on SWE-bench fail with or without diversity. "
          "Aggregation cannot recover what individual judgments lack — "
          "we ran 6 experiments at scales from $7 to $276 and none beat K=0 baseline "
          "by more than 2pp.")
        P()
        P("2. **Grounding context is the only lever that worked**: adding the "
          "FAIL_TO_PASS test name + a few file paths from the patch raised the "
          "cheapest classifier (single-prompt, R=5 binary judgments, $10) "
          f"from 64% to {rate:.1%}.")
        P()
        P("3. **The model has a strong yes-bias** (89-91% all-flags-true "
          "regardless of patch correctness in the critic pool; reduced but "
          "not eliminated by hints). This is the underlying reason aggregation "
          "fails on this task.")
        P()
        P("4. **Practical recommendation** for SWE-bench-style patch ranking with "
          "weak models: don't bother with committee structures; pay for the "
          "test-name and file-path grounding instead.")
    else:
        # Pilot didn't generalize
        P("**Pilot's 68% did not generalize to full 500.** The grounding lever "
          "appears not to be a real signal — pilot was a lucky sample.")
        P()
        P("**Recommended framing:**")
        P()
        P("1. **Negative result paper**: weak-model committees fail across ALL "
          "configurations tested — comparators, critics, direct-binary, and even "
          "execution-hints grounding. Sample-size effects on the 50-instance "
          "pilot misled us into thinking grounding helped.")
        P()
        P("2. The paper's contribution is the **systematic null result** plus "
          "the diagnostic finding that the model has a structural yes-bias "
          "uncorrelated with patch correctness.")
        P()
        P("3. Future work needs a stronger model or a different signal source "
          "(test execution traces, runtime behavior) — text+grounding alone is "
          "not enough.")
    P()

    P("## Cached artifacts (for further offline analysis)")
    P()
    P(f"| Path | Rows |")
    P(f"|---|---:|")
    P(f"| `outputs/direct_binary_hints_full500/binary_votes.jsonl` | "
      f"{sum(len(v) for v in by_iid_full.values()):,} |")
    P(f"| `outputs/direct_binary_hints_full500/usage.jsonl` | {n_calls:,} |")
    P()
    P("All caches are append-only JSONL — re-aggregate with different rules "
      "(confidence-weighted, abstain-as-no, etc.) without new API calls.")


if __name__ == "__main__":
    sys.exit(main() or 0)
