#!/usr/bin/env python3
"""Threshold-gated comparator analysis.

Layering evaluated:
  1. Greedy-8 proposer patches.
  2. Binary voters gate patches by yes_votes / available_votes >= threshold.
  3. Anchored comparator tournament chooses among gated survivors.

This script is offline. It can run before comparator votes exist to summarize
gate sizes and fallback binary-selector rates. Once comparator votes exist, it
also reports gated comparator resolve rates for each threshold.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
DEFAULT_THRESHOLDS = [i / 10 for i in range(0, 11)]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def load_binary(paths: list[Path]) -> dict[str, dict[int, dict]]:
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
                votes = {int(v.get("vote_idx")): v for v in existing.get("binary_votes", [])}
                for v in row.get("binary_votes", []) or []:
                    try:
                        votes[int(v.get("vote_idx"))] = v
                    except Exception:
                        continue
                existing["binary_votes"] = [votes[i] for i in sorted(votes)]
                if existing.get("patch_resolves") is None and row.get("patch_resolves") is not None:
                    existing["patch_resolves"] = row.get("patch_resolves")
    return by_iid


def binary_scores(row: dict) -> dict:
    votes = row.get("binary_votes") or []
    yes = sum(1 for v in votes if v.get("resolves") is True)
    valid = sum(1 for v in votes if v.get("resolves") is not None)
    conf_yes = sum(int(v.get("confidence") or 0) for v in votes if v.get("resolves") is True)
    return {
        "yes": yes,
        "valid": valid,
        "yes_rate": yes / valid if valid else 0.0,
        "conf_yes": conf_yes,
    }


def complete_iids(binary: dict[str, dict[int, dict]], k_list: list[int]) -> list[str]:
    want = set(k_list)
    return sorted(iid for iid, rows in binary.items() if want.issubset(rows))


def load_comparator(path: Path) -> dict[str, dict[tuple[int, int], dict]]:
    out: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    if not path or not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            a = int(row["patch_a_idx"])
            b = int(row["patch_b_idx"])
            if a > b:
                a, b = b, a
            out[row["instance_id"]][(a, b)] = row
    return out


def representative_winner(row: dict) -> tuple[str, int]:
    counts = Counter()
    conf = Counter()
    for vote in row.get("votes", []):
        parsed_items = [(vote.get("primary") or {}).get("parsed")]
        if vote.get("swap"):
            parsed_items.append((vote.get("swap") or {}).get("parsed_normalized"))
        for parsed in parsed_items:
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
    n = counts[winner]
    return winner, max(1, round(conf[winner] / n))


def select_binary(rows: dict[int, dict], candidates: list[int], mode: str) -> int:
    scored = []
    for k in candidates:
        s = binary_scores(rows[k])
        if mode == "confidence":
            score = s["conf_yes"]
        else:
            score = s["yes"]
        scored.append((k, score))
    return sorted(scored, key=lambda x: (-x[1], x[0]))[0][0]


def select_comparator(
    iid: str,
    candidates: list[int],
    cmp_rows: dict[str, dict[tuple[int, int], dict]],
    mode: str,
) -> int | None:
    if len(candidates) < 2:
        return candidates[0] if candidates else None
    cand = sorted(candidates)
    scores = {k: 0.0 for k in cand}
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            row = cmp_rows.get(iid, {}).get((a, b))
            if not row:
                return None
            winner, conf = representative_winner(row)
            weight = conf if mode == "confidence_weighted" else 1
            if winner == "A":
                scores[a] += weight
                scores[b] -= weight
            elif winner == "B":
                scores[b] += weight
                scores[a] -= weight
    return sorted(scores, key=lambda k: (-scores[k], k))[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary-votes", required=True, nargs="+")
    p.add_argument("--comparator-votes", default="")
    p.add_argument("--output-json", default="")
    p.add_argument("--k-list", default=",".join(map(str, DEFAULT_K_LIST)))
    p.add_argument("--thresholds", default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    p.add_argument("--fallback", choices=("majority", "confidence"), default="confidence")
    args = p.parse_args()

    k_list = parse_csv_ints(args.k_list)
    thresholds = parse_csv_floats(args.thresholds)
    binary = load_binary([Path(p) for p in args.binary_votes])
    cmp_rows = load_comparator(Path(args.comparator_votes)) if args.comparator_votes else {}
    iids = complete_iids(binary, k_list)

    summary = {
        "binary_votes": args.binary_votes,
        "comparator_votes": args.comparator_votes,
        "k_list": k_list,
        "complete_instances": len(iids),
        "thresholds": {},
    }

    for threshold in thresholds:
        bucket = {
            "instances": 0,
            "avg_candidates": 0.0,
            "zero_candidate_instances": 0,
            "one_candidate_instances": 0,
            "multi_candidate_instances": 0,
            "fallback_binary": {"resolved": 0, "total": 0, "rate": None},
            "comparator_copeland": {"resolved": 0, "total": 0, "rate": None, "missing_pair_instances": 0},
            "comparator_confidence_weighted": {"resolved": 0, "total": 0, "rate": None, "missing_pair_instances": 0},
            "pair_rows_needed_if_run_at_threshold": 0,
        }
        for iid in iids:
            rows = binary[iid]
            candidates = [k for k in k_list if binary_scores(rows[k])["yes_rate"] >= threshold]
            if not candidates:
                candidates = [select_binary(rows, k_list, args.fallback)]
                bucket["zero_candidate_instances"] += 1
            elif len(candidates) == 1:
                bucket["one_candidate_instances"] += 1
            else:
                bucket["multi_candidate_instances"] += 1
            bucket["instances"] += 1
            bucket["avg_candidates"] += len(candidates)
            bucket["pair_rows_needed_if_run_at_threshold"] += len(candidates) * (len(candidates) - 1) // 2

            fb = select_binary(rows, candidates, args.fallback)
            if rows[fb].get("patch_resolves") in (True, False):
                bucket["fallback_binary"]["total"] += 1
                bucket["fallback_binary"]["resolved"] += int(bool(rows[fb].get("patch_resolves")))

            for mode, key in (("copeland", "comparator_copeland"),
                              ("confidence_weighted", "comparator_confidence_weighted")):
                chosen = select_comparator(iid, candidates, cmp_rows, mode) if cmp_rows else None
                if chosen is None:
                    bucket[key]["missing_pair_instances"] += 1
                    continue
                if rows[chosen].get("patch_resolves") in (True, False):
                    bucket[key]["total"] += 1
                    bucket[key]["resolved"] += int(bool(rows[chosen].get("patch_resolves")))

        if bucket["instances"]:
            bucket["avg_candidates"] /= bucket["instances"]
        for key in ("fallback_binary", "comparator_copeland", "comparator_confidence_weighted"):
            total = bucket[key]["total"]
            bucket[key]["rate"] = bucket[key]["resolved"] / total if total else None
        summary["thresholds"][str(threshold)] = bucket

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"Complete instances: {len(iids)}")
    print("thr\tavg_cand\tpairs\tfallback\tcmp_copeland\tcmp_conf\tmissing_cmp")
    for threshold in thresholds:
        b = summary["thresholds"][str(threshold)]
        fb = b["fallback_binary"]["rate"]
        cc = b["comparator_copeland"]["rate"]
        cw = b["comparator_confidence_weighted"]["rate"]
        miss = b["comparator_copeland"]["missing_pair_instances"]
        fb_s = f"{fb:.4f}" if fb is not None else "NA"
        cc_s = f"{cc:.4f}" if cc is not None else "NA"
        cw_s = f"{cw:.4f}" if cw is not None else "NA"
        print(
            f"{threshold:.2f}\t{b['avg_candidates']:.2f}\t"
            f"{b['pair_rows_needed_if_run_at_threshold']}\t"
            f"{fb_s}\t{cc_s}\t{cw_s}\t{miss}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
