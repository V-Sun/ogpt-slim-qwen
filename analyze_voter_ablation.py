#!/usr/bin/env python3
"""Offline voter-count ablation for direct-binary selector caches.

This answers: how much does each extra binary voter add?

For R=5 saved votes, compute selector resolve rate using n=1..5 voters per
patch. The main table averages across every subset of n vote indices, so n=2
uses all 10 possible two-voter subsets. This avoids over-interpreting the
arbitrary order in which vote_idx values were assigned.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_k_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("empty --k-list")
    return out


def load_rows(paths: list[Path]) -> dict[str, dict[int, dict]]:
    by_iid: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                iid = row["instance_id"]
                k = int(row["patch_idx"])
                existing = by_iid[iid].get(k)
                if existing is None:
                    by_iid[iid][k] = row
                    continue
                merged_votes = {int(v.get("vote_idx")): v for v in existing.get("binary_votes", [])}
                for v in row.get("binary_votes", []) or []:
                    try:
                        merged_votes[int(v.get("vote_idx"))] = v
                    except Exception:
                        continue
                existing["binary_votes"] = [merged_votes[i] for i in sorted(merged_votes)]
                if existing.get("patch_resolves") is None and row.get("patch_resolves") is not None:
                    existing["patch_resolves"] = row.get("patch_resolves")
    return by_iid


def complete_iids(by_iid: dict[str, dict[int, dict]], k_list: list[int]) -> list[str]:
    want = set(k_list)
    return sorted(iid for iid, rows in by_iid.items() if want.issubset(rows))


def vote_maps(row: dict) -> dict[int, dict]:
    out = {}
    for v in row.get("binary_votes") or []:
        try:
            out[int(v.get("vote_idx"))] = v
        except Exception:
            continue
    return out


def score_row(row: dict, vote_subset: tuple[int, ...]) -> tuple[int, int, int]:
    """Return (yes_count, confidence_yes_sum, abstain_count)."""
    votes = vote_maps(row)
    yes = 0
    conf = 0
    abst = 0
    for vi in vote_subset:
        v = votes.get(vi)
        if not v or v.get("resolves") is None:
            abst += 1
            continue
        if v.get("resolves") is True:
            yes += 1
            conf += int(v.get("confidence") or 0)
    return yes, conf, abst


def evaluate_subset(
    by_iid: dict[str, dict[int, dict]],
    iids: list[str],
    k_list: list[int],
    vote_subset: tuple[int, ...],
) -> dict:
    hits_majority = hits_conf = 0
    total_majority = total_conf = 0
    abstains = 0
    selected_majority = Counter()
    selected_conf = Counter()

    for iid in iids:
        maj_scores = []
        conf_scores = []
        for k in k_list:
            row = by_iid[iid][k]
            yes, conf, abst = score_row(row, vote_subset)
            abstains += abst
            resolves = row.get("patch_resolves")
            maj_scores.append((k, yes, resolves))
            conf_scores.append((k, conf, resolves))

        maj_scores.sort(key=lambda x: (-x[1], x[0]))
        conf_scores.sort(key=lambda x: (-x[1], x[0]))

        k, _score, resolves = maj_scores[0]
        selected_majority[k] += 1
        if resolves in (True, False):
            total_majority += 1
            hits_majority += int(bool(resolves))

        k, _score, resolves = conf_scores[0]
        selected_conf[k] += 1
        if resolves in (True, False):
            total_conf += 1
            hits_conf += int(bool(resolves))

    return {
        "vote_subset": list(vote_subset),
        "majority": {
            "resolved": hits_majority,
            "total": total_majority,
            "rate": hits_majority / total_majority if total_majority else None,
            "selected_k_counts": dict(selected_majority),
        },
        "confidence_weighted": {
            "resolved": hits_conf,
            "total": total_conf,
            "rate": hits_conf / total_conf if total_conf else None,
            "selected_k_counts": dict(selected_conf),
        },
        "abstains": abstains,
    }


def summarize_n(results: list[dict], selector: str) -> dict:
    rates = [r[selector]["rate"] for r in results if r[selector]["rate"] is not None]
    if not rates:
        return {"mean_rate": None, "min_rate": None, "max_rate": None}
    return {
        "mean_rate": sum(rates) / len(rates),
        "min_rate": min(rates),
        "max_rate": max(rates),
        "num_vote_subsets": len(rates),
        "mean_resolved": sum(r[selector]["resolved"] for r in results) / len(results),
        "mean_total": sum(r[selector]["total"] for r in results) / len(results),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary-votes", required=True, nargs="+",
                   help="One or more binary_votes.jsonl files. Multiple files are merged by vote_idx.")
    p.add_argument("--output-json", default="", help="Optional machine-readable output")
    p.add_argument("--k-list", default=",".join(map(str, DEFAULT_K_LIST)))
    p.add_argument("--max-votes", type=int, default=5)
    args = p.parse_args()

    paths = [Path(p) for p in args.binary_votes]
    k_list = parse_k_list(args.k_list)
    by_iid = load_rows(paths)
    iids = complete_iids(by_iid, k_list)

    all_results = {}
    summary = {
        "binary_votes": [str(p) for p in paths],
        "k_list": k_list,
        "complete_instances": len(iids),
        "rows_loaded": sum(len(v) for v in by_iid.values()),
        "selectors": {"majority": {}, "confidence_weighted": {}},
    }

    for n in range(1, args.max_votes + 1):
        subsets = list(itertools.combinations(range(args.max_votes), n))
        n_results = [evaluate_subset(by_iid, iids, k_list, subset) for subset in subsets]
        all_results[str(n)] = n_results
        for selector in ("majority", "confidence_weighted"):
            summary["selectors"][selector][str(n)] = summarize_n(n_results, selector)

    for selector in ("majority", "confidence_weighted"):
        prev = None
        for n in range(1, args.max_votes + 1):
            rec = summary["selectors"][selector][str(n)]
            rate = rec["mean_rate"]
            rec["delta_from_previous"] = None if prev is None or rate is None else rate - prev
            if rate is not None:
                prev = rate

    out = {"summary": summary, "by_vote_subset": all_results}
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")

    print("Voter ablation for:")
    for path in paths:
        print(f"  {path}")
    print(f"Complete instances with all K rows: {len(iids)}")
    print()
    for selector in ("majority", "confidence_weighted"):
        print(selector)
        print("votes\tmean_rate\tmin\tmax\tdelta")
        for n in range(1, args.max_votes + 1):
            rec = summary["selectors"][selector][str(n)]
            rate = rec["mean_rate"]
            delta = rec["delta_from_previous"]
            if rate is None:
                print(f"{n}\tNA\tNA\tNA\tNA")
            else:
                d = "NA" if delta is None else f"{delta:+.4f}"
                print(f"{n}\t{rate:.4f}\t{rec['min_rate']:.4f}\t{rec['max_rate']:.4f}\t{d}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
