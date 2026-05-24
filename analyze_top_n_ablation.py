#!/usr/bin/env python3
"""Top-N selectors → comparator ablation.

Question answered: among R=10 binary-vote selectors, if we pick the
top-n patches per instance by yes-rate and only run Copeland over those
n patches' C(n, 2) pairs, how does resolve rate vary with n?

Inputs:
  - merged R=10 binary_votes.jsonl path(s) — to rank patches by yes-rate
  - comparator votes path — pairwise_dynamic_comparator_votes.jsonl from
    anchored_dynamic_comparator.py (M=10, position-swap optional)

For n_top in 2..len(K_LIST):
  - For each instance that has all C(n_top, 2) cached pairs among its
    top-n_top patches:
    - Aggregate Copeland with confidence weighting (matches
      `analyze_gated_comparator_thresholds.py:select_comparator`)
    - Pick winner; score resolve.

Offline. No API calls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_k_list(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise SystemExit("empty --k-list")
    return out


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


def yes_score(row: dict) -> tuple[int, int, int]:
    """Return (yes_count, conf_yes, valid_count)."""
    yes = conf = valid = 0
    for v in row.get("binary_votes") or []:
        r = v.get("resolves")
        if r is None:
            continue
        valid += 1
        if r is True:
            yes += 1
            conf += int(v.get("confidence") or 0)
    return yes, conf, valid


def load_comparator(path: Path) -> dict[str, dict[tuple[int, int], dict]]:
    out: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
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
    for vote in row.get("votes") or []:
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
    n = counts[winner]
    return winner, max(1, round(conf[winner] / n))


def copeland(iid: str, candidates: list[int],
             cmp_rows: dict[str, dict[tuple[int, int], dict]],
             mode: str = "confidence_weighted") -> int | None:
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
    p.add_argument("--comparator-votes", required=True)
    p.add_argument("--k-list", default=",".join(map(str, DEFAULT_K_LIST)))
    p.add_argument("--output-json", default="")
    p.add_argument("--rank-by", choices=("majority", "confidence"), default="confidence",
                   help="Score used to rank patches before picking top-n")
    p.add_argument("--copeland-mode", choices=("majority", "confidence_weighted"),
                   default="confidence_weighted")
    args = p.parse_args()

    k_list = parse_k_list(args.k_list)
    binary = load_binary([Path(p) for p in args.binary_votes])
    cmp_rows = load_comparator(Path(args.comparator_votes))
    iids = sorted(iid for iid, rows in binary.items() if set(k_list).issubset(rows))

    rankings = {}
    for iid in iids:
        scored = []
        for k in k_list:
            yes, conf, _ = yes_score(binary[iid][k])
            score = conf if args.rank_by == "confidence" else yes
            scored.append((k, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        rankings[iid] = [k for k, _ in scored]

    summary = {
        "binary_votes": args.binary_votes,
        "comparator_votes": args.comparator_votes,
        "k_list": k_list,
        "rank_by": args.rank_by,
        "copeland_mode": args.copeland_mode,
        "complete_instances": len(iids),
        "by_n_top": {},
    }

    for n_top in range(2, len(k_list) + 1):
        bucket = {
            "instances_with_all_pairs": 0,
            "missing_pair_instances": 0,
            "resolved": 0,
            "total": 0,
            "rate": None,
        }
        for iid in iids:
            top = rankings[iid][:n_top]
            chosen = copeland(iid, top, cmp_rows, mode=args.copeland_mode)
            if chosen is None:
                bucket["missing_pair_instances"] += 1
                continue
            bucket["instances_with_all_pairs"] += 1
            r = binary[iid][chosen].get("patch_resolves")
            if r in (True, False):
                bucket["total"] += 1
                bucket["resolved"] += int(bool(r))
        bucket["rate"] = bucket["resolved"] / bucket["total"] if bucket["total"] else None
        summary["by_n_top"][str(n_top)] = bucket

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"Top-N selectors → comparator ablation")
    print(f"  binary_votes: {args.binary_votes}")
    print(f"  comparator_votes: {args.comparator_votes}")
    print(f"  rank_by: {args.rank_by}, copeland: {args.copeland_mode}")
    print(f"  complete_instances: {len(iids)}")
    print()
    print("n_top\tinst_with_all_pairs\tmissing\tresolved\ttotal\trate")
    for n_top in range(2, len(k_list) + 1):
        b = summary["by_n_top"][str(n_top)]
        rate = f"{b['rate']:.4f}" if b["rate"] is not None else "NA"
        print(f"{n_top}\t{b['instances_with_all_pairs']}\t{b['missing_pair_instances']}\t"
              f"{b['resolved']}\t{b['total']}\t{rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
