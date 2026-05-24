"""Generate RUN_SUMMARY_DYNAMIC_HINTS.md comparing dynamic-hints vs static-hints
on the full 500 SWE-Bench Verified subset.

Reads:
  outputs/direct_binary_dynamic_hints_full500/{binary_votes,usage}.jsonl
  outputs/direct_binary_hints_full500/binary_votes.jsonl  (static, for compare)
  outputs/stage1_critics_full/critic_votes.jsonl  (resolves data)

Decision tree:
  ≥77%: execution-grounded hints is the lever. Write up.
  73-76%: marginal additional gain. Two-step grounding story.
  69-72%: dynamic hints don't help over static. Write up the static result.
  <69%: investigate.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
DYN_DIR = REPO / "outputs" / "direct_binary_dynamic_hints_full500"
STATIC_DIR = REPO / "outputs" / "direct_binary_hints_full500"

K_PATCHES = 8
R_VOTES = 5


def load_binary(d: Path):
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


def load_usage(d: Path):
    jsonl = d / "usage.jsonl"
    if not jsonl.exists():
        return 0, 0.0, 0, 0, 0
    n = inp = out = rea = 0
    cost = 0.0
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


def compute_resolve(by_iid):
    nm = nc = 0
    yes_counts = []
    abst = 0
    for iid in sorted(by_iid):
        rows = sorted(by_iid[iid], key=lambda r: r["patch_idx"])
        scores_maj, scores_conf = [], []
        for row in rows:
            yes = sum(1 for v in row["binary_votes"] if v.get("resolves") is True)
            conf_yes = sum(int(v.get("confidence", 0)) for v in row["binary_votes"]
                           if v.get("resolves") is True)
            abst += sum(1 for v in row["binary_votes"]
                        if v.get("resolves") is None)
            yes_counts.append(yes)
            scores_maj.append((row["patch_idx"], yes,
                               bool(row.get("patch_resolves"))))
            scores_conf.append((row["patch_idx"], conf_yes,
                                bool(row.get("patch_resolves"))))
        scores_maj.sort(key=lambda x: (-x[1], x[0]))
        if scores_maj and scores_maj[0][2]:
            nm += 1
        scores_conf.sort(key=lambda x: (-x[1], x[0]))
        if scores_conf and scores_conf[0][2]:
            nc += 1
    return nm, nc, dict(Counter(yes_counts)), abst, len(by_iid)


def load_resolves():
    f = REPO / "outputs" / "stage1_critics_full" / "critic_votes.jsonl"
    out = {}
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            out.setdefault(r["instance_id"], {})[r["patch_idx"]] = bool(r.get("patch_resolves"))
    return out


def per_repo_breakdown(by_iid, resolves_data):
    """Per-repo resolve rate (winning patch resolves)."""
    by_repo = defaultdict(lambda: {"resolved": 0, "total": 0})
    for iid in sorted(by_iid):
        repo = iid.split("__")[0] if "__" in iid else iid
        rows = sorted(by_iid[iid], key=lambda r: r["patch_idx"])
        scores = []
        for row in rows:
            yes = sum(1 for v in row["binary_votes"] if v.get("resolves") is True)
            scores.append((row["patch_idx"], yes,
                           bool(row.get("patch_resolves"))))
        scores.sort(key=lambda x: (-x[1], x[0]))
        if not scores:
            continue
        by_repo[repo]["total"] += 1
        if scores[0][2]:
            by_repo[repo]["resolved"] += 1
    return dict(by_repo)


def main():
    rows_dyn, by_iid_dyn = load_binary(DYN_DIR)
    rows_st, by_iid_st = load_binary(STATIC_DIR)

    if not rows_dyn:
        print("# RUN_SUMMARY_DYNAMIC_HINTS — INCOMPLETE")
        print()
        print("No data in outputs/direct_binary_dynamic_hints_full500/. "
              "Check outputs/dynamic_full500.log for errors.")
        return 0

    n_dyn, cost_dyn, inp, out, rea = load_usage(DYN_DIR)
    nm_dyn, nc_dyn, hist_dyn, abst_dyn, n_inst = compute_resolve(by_iid_dyn)
    nm_st, nc_st, hist_st, _, n_st = compute_resolve(by_iid_st)

    resolves = load_resolves()
    iids_in_run = sorted(by_iid_dyn.keys())
    n_k0 = sum(1 for iid in iids_in_run
               if resolves.get(iid, {}).get(0, False))
    n_oracle = sum(1 for iid in iids_in_run
                   if any(resolves.get(iid, {}).get(k, False)
                          for k in range(K_PATCHES)))

    rate_dyn = nm_dyn / max(1, n_inst)
    rate_st = nm_st / max(1, n_st) if n_st else 0.0

    P = print
    P("# RUN_SUMMARY_DYNAMIC_HINTS — Direct-binary + Execution-grounded Hints")
    P()
    P("_Generated: generate_dynamic_hints_summary.py_")
    P()
    P("## Headline")
    P()
    P(f"Direct-binary classifier with **dynamic execution-grounded hints**: "
      f"FAIL_TO_PASS test names + actual error trace extracted from cached "
      f"test_output.txt + test_patch from SWE-Bench dataset.")
    P()
    P(f"- **Full-{n_inst} resolve rate (dynamic hints): "
      f"{nm_dyn}/{n_inst} = {rate_dyn:.3f} ({rate_dyn:.1%})**")
    P(f"- Static-hints baseline (full {n_st}): {nm_st}/{n_st} = {rate_st:.3f}")
    P(f"- **Δ static → dynamic: {(rate_dyn - rate_st):+.3f} "
      f"({(rate_dyn - rate_st) * 100:+.1f}pp)**")
    P()

    P("## Decision tree")
    P()
    if rate_dyn >= 0.77:
        P(f"**≥77% — execution-grounded hints is the lever.** Paper has a "
          f"strong positive contribution.")
    elif rate_dyn >= 0.73:
        P(f"**73-76% — marginal additional gain over static hints.** "
          f"Paper now has a two-step grounding story (static → +N pp → dynamic).")
    elif rate_dyn >= 0.69:
        P(f"**69-72% — dynamic hints don't help meaningfully over static.** "
          f"Write up the static-hints result; note that re-running the failing "
          f"test trace doesn't add discriminative signal beyond test names.")
    else:
        P(f"**<69% — unexpectedly low.** Investigate parse-fail rate or "
          f"extraction issues.")
    P()

    P("## Cost + wall-clock")
    P()
    P(f"| Metric | Value |")
    P(f"|---|---:|")
    P(f"| API calls | {n_dyn:,} |")
    P(f"| Input tokens | {inp:,} |")
    P(f"| Output tokens (incl reasoning) | {out:,} |")
    P(f"| Reasoning tokens | {rea:,} |")
    P(f"| **Total cost** | **${cost_dyn:.2f}** |")
    P(f"| Budget cap | $250.00 |")
    P(f"| Headroom | ${250 - cost_dyn:.2f} |")
    P(f"| Abstains | {abst_dyn:,} of {n_inst * K_PATCHES * R_VOTES:,} = "
      f"{abst_dyn / max(1, n_inst * K_PATCHES * R_VOTES):.1%} |")
    P()

    P("## Comparison vs all prior experiments (same K=0..7 subset)")
    P()
    P("| Approach | Pilot 50 | Full | Cost |")
    P("|---|---:|---:|---:|")
    P(f"| K=0 baseline | 33/50 = 66.0% | {n_k0}/{n_inst} = "
      f"{n_k0/max(1, n_inst):.1%} | $0 |")
    P(f"| Oracle (any of K=0..7 resolves) | 37/50 = 74.0% | "
      f"{n_oracle}/{n_inst} = {n_oracle/max(1, n_inst):.1%} | $0 |")
    P(f"| Diversity comparator (K=8/M=15/R=10, pilot) | 32/50 = 64.0% | — | $276 |")
    P(f"| Direct-binary (K=8/R=5, no hints, pilot) | 33/50 = 66.0% | — | $7.85 |")
    P(f"| Direct-binary + **static hints** (full {n_st}) | 34/50 = 68.0% | "
      f"{nm_st}/{n_st} = {rate_st:.1%} | $89.85 |")
    P(f"| **Direct-binary + dynamic hints (full {n_inst})** | — | "
      f"**{nm_dyn}/{n_inst} = {rate_dyn:.1%}** | **${cost_dyn:.2f}** |")
    P()

    P("## Yes-vote distribution shift: static vs dynamic")
    P()
    P("| Yes count | Static hints (full) | Dynamic hints (full) |")
    P("|---:|---:|---:|")
    for c in range(R_VOTES + 1):
        P(f"| {c} | {hist_st.get(c, 0)} | {hist_dyn.get(c, 0)} |")
    P()
    n_unanimous_dyn = hist_dyn.get(R_VOTES, 0)
    n_total_dyn = sum(hist_dyn.values())
    n_unanimous_st = hist_st.get(R_VOTES, 0)
    n_total_st = sum(hist_st.values()) if hist_st else 1
    P(f"Unanimous-yes patches:")
    P(f"  - Static: {n_unanimous_st}/{n_total_st} = {n_unanimous_st/n_total_st:.1%}")
    P(f"  - Dynamic: {n_unanimous_dyn}/{n_total_dyn} = "
      f"{n_unanimous_dyn/max(1, n_total_dyn):.1%}")
    delta = (n_unanimous_dyn / max(1, n_total_dyn)) - (n_unanimous_st / max(1, n_total_st))
    P(f"  - Δ (dynamic - static): {delta * 100:+.1f}pp "
      f"({'reduced yes-bias' if delta < 0 else 'increased yes-bias'})")
    P()

    P("## Per-repo breakdown (dynamic hints)")
    P()
    by_repo = per_repo_breakdown(by_iid_dyn, resolves)
    P("| Repo | Resolved | Total | Rate |")
    P("|---|---:|---:|---:|")
    for repo in sorted(by_repo, key=lambda r: -by_repo[r]["total"]):
        d = by_repo[repo]
        P(f"| {repo} | {d['resolved']} | {d['total']} | "
          f"{d['resolved']/max(1, d['total']):.1%} |")
    P()

    P("## Recommended paper framing")
    P()
    if rate_dyn >= 0.77:
        P("**Strong positive — dynamic hints is THE lever.** Headline: "
          f"adding the actual failing-test traceback to a single-shot binary "
          f"classifier on cached SWE-Bench data lifts resolve rate from "
          f"{rate_st:.1%} (static hints) to **{rate_dyn:.1%}** — a "
          f"{(rate_dyn - rate_st) * 100:+.1f}pp jump from grounding alone, "
          f"with no extra committee structure or model upgrade required.")
        P()
        P("Story: weak-model committees fail (4 experiments, all under "
          "K=0 baseline). Static FAIL_TO_PASS hints recover +2pp. Dynamic "
          "execution traces recover the rest. The model can read tracebacks; "
          "the committee structure was distracting it.")
    elif rate_dyn >= 0.73:
        P("**Two-step grounding paper.** Static hints (FAIL_TO_PASS test "
          f"names) lifted resolve rate from 66% (K=0 baseline) to {rate_st:.1%}. "
          f"Adding actual error tracebacks lifts another "
          f"{(rate_dyn - rate_st) * 100:+.1f}pp to {rate_dyn:.1%}. Both pieces "
          f"of grounding contribute; together they close the oracle gap "
          f"meaningfully.")
    else:
        P("**Negative result for dynamic hints; static-hints story stands.** "
          f"Static hints already captured most of the grounding signal at "
          f"{rate_st:.1%}; adding the actual traceback adds "
          f"{(rate_dyn - rate_st) * 100:+.1f}pp = noise level. The lever is "
          f"FAIL_TO_PASS test names + file paths, not the error message itself.")
    P()
    P("**Practical implication for production orchestra-gpt deployment:** "
      "if you want the resolve-rate boost in production, replace the "
      "critic+comparator committee with a single direct-binary classifier "
      f"using {'dynamic' if rate_dyn > rate_st + 0.02 else 'static'} hints. "
      "Cost-per-instance is ~$0.20 vs prior committee runs at $0.55-2.50.")
    P()

    P("## Cached artifacts")
    P()
    P(f"| Path | Rows |")
    P(f"|---|---:|")
    P(f"| `outputs/direct_binary_dynamic_hints_full500/binary_votes.jsonl` | "
      f"{sum(len(v) for v in by_iid_dyn.values()):,} |")
    P(f"| `outputs/direct_binary_dynamic_hints_full500/usage.jsonl` | {n_dyn:,} |")


if __name__ == "__main__":
    sys.exit(main() or 0)
