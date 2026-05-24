#!/usr/bin/env python3
"""Layer-1 ablation: vary the proposer pool size n_K within greedy-8.

For each n_K in 1..len(K_LIST):
  - For every C(K, n_K) subset S:
    - For each instance: select the best patch among S using R=10 binary
      votes (yes-count or confidence_yes), tiebreak by lowest k.
    - Score resolve rate over instances with truth labels.
  - Average / min / max across all subsets at that n_K.

Mirrors `analyze_voter_ablation.py`'s subset-averaging pattern but the
varied dimension is the K-pool, not the vote-pool. Offline. No API.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_k_list(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
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
                merged = {int(v.get("vote_idx")): v for v in existing.get("binary_votes", [])}
                for v in row.get("binary_votes", []) or []:
                    try:
                        merged[int(v.get("vote_idx"))] = v
                    except Exception:
                        continue
                existing["binary_votes"] = [merged[i] for i in sorted(merged)]
                if existing.get("patch_resolves") is None and row.get("patch_resolves") is not None:
                    existing["patch_resolves"] = row.get("patch_resolves")
    return by_iid


def complete_iids(by_iid: dict[str, dict[int, dict]], k_list: list[int]) -> list[str]:
    want = set(k_list)
    return sorted(iid for iid, rows in by_iid.items() if want.issubset(rows))


def vote_summary(row: dict) -> tuple[int, int]:
    yes = 0
    conf = 0
    for v in row.get("binary_votes") or []:
        if v.get("resolves") is True:
            yes += 1
            conf += int(v.get("confidence") or 0)
    return yes, conf


def evaluate_subset(
    by_iid: dict[str, dict[int, dict]],
    iids: list[str],
    k_subset: tuple[int, ...],
    summaries: dict[str, dict[int, tuple[int, int]]],
) -> dict:
    hits_maj = total_maj = 0
    hits_conf = total_conf = 0
    sel_maj = Counter()
    sel_conf = Counter()
    for iid in iids:
        s = summaries[iid]
        maj_rank = sorted(((k, s[k][0]) for k in k_subset), key=lambda x: (-x[1], x[0]))
        conf_rank = sorted(((k, s[k][1]) for k in k_subset), key=lambda x: (-x[1], x[0]))
        k_maj = maj_rank[0][0]
        k_conf = conf_rank[0][0]
        sel_maj[k_maj] += 1
        sel_conf[k_conf] += 1
        r_maj = by_iid[iid][k_maj].get("patch_resolves")
        r_conf = by_iid[iid][k_conf].get("patch_resolves")
        if r_maj in (True, False):
            total_maj += 1
            hits_maj += int(bool(r_maj))
        if r_conf in (True, False):
            total_conf += 1
            hits_conf += int(bool(r_conf))
    return {
        "k_subset": list(k_subset),
        "majority": {
            "resolved": hits_maj,
            "total": total_maj,
            "rate": hits_maj / total_maj if total_maj else None,
            "selected_k_counts": dict(sel_maj),
        },
        "confidence_weighted": {
            "resolved": hits_conf,
            "total": total_conf,
            "rate": hits_conf / total_conf if total_conf else None,
            "selected_k_counts": dict(sel_conf),
        },
    }


def summarize_n(results: list[dict], selector: str) -> dict:
    rates = [r[selector]["rate"] for r in results if r[selector]["rate"] is not None]
    if not rates:
        return {"mean_rate": None, "min_rate": None, "max_rate": None, "num_subsets": 0}
    return {
        "mean_rate": sum(rates) / len(rates),
        "min_rate": min(rates),
        "max_rate": max(rates),
        "num_subsets": len(rates),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary-votes", required=True, nargs="+",
                   help="binary_votes.jsonl files; merged by (iid, k, vote_idx)")
    p.add_argument("--output-json", default="")
    p.add_argument("--k-list", default=",".join(map(str, DEFAULT_K_LIST)))
    args = p.parse_args()

    k_list = parse_k_list(args.k_list)
    by_iid = load_rows([Path(p) for p in args.binary_votes])
    iids = complete_iids(by_iid, k_list)

    summaries = {iid: {k: vote_summary(by_iid[iid][k]) for k in k_list} for iid in iids}

    summary = {
        "binary_votes": args.binary_votes,
        "k_list": k_list,
        "complete_instances": len(iids),
        "selectors": {"majority": {}, "confidence_weighted": {}},
    }
    all_results = {}
    for n_k in range(1, len(k_list) + 1):
        subsets = list(itertools.combinations(k_list, n_k))
        n_results = [evaluate_subset(by_iid, iids, sub, summaries) for sub in subsets]
        all_results[str(n_k)] = n_results
        for selector in ("majority", "confidence_weighted"):
            summary["selectors"][selector][str(n_k)] = summarize_n(n_results, selector)

    for selector in ("majority", "confidence_weighted"):
        prev = None
        for n_k in range(1, len(k_list) + 1):
            rec = summary["selectors"][selector][str(n_k)]
            rate = rec["mean_rate"]
            rec["delta_from_previous"] = None if prev is None or rate is None else rate - prev
            if rate is not None:
                prev = rate

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({"summary": summary, "by_k_subset": all_results}, f, indent=2, sort_keys=True)
            f.write("\n")

    print("K-sweep ablation (Layer 1):")
    for path in args.binary_votes:
        print(f"  {path}")
    print(f"Complete instances: {len(iids)}")
    print(f"K-pool: {k_list}")
    print()
    for selector in ("majority", "confidence_weighted"):
        print(selector)
        print("n_K\tmean_rate\tmin\tmax\tnum_subsets\tdelta")
        for n_k in range(1, len(k_list) + 1):
            rec = summary["selectors"][selector][str(n_k)]
            rate = rec["mean_rate"]
            delta = rec["delta_from_previous"]
            if rate is None:
                print(f"{n_k}\tNA\tNA\tNA\t{rec['num_subsets']}\tNA")
            else:
                d = "NA" if delta is None else f"{delta:+.4f}"
                print(f"{n_k}\t{rate:.4f}\t{rec['min_rate']:.4f}\t{rec['max_rate']:.4f}\t{rec['num_subsets']}\t{d}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
