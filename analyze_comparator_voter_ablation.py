#!/usr/bin/env python3
"""Layer-3 voter-count ablation for the anchored comparator.

For each m_votes in 1..M, take all C(M, m_votes) subsets of vote_idx
values and recompute Copeland on the cached pairwise comparator votes.
Mirrors `analyze_voter_ablation.py`'s subset-averaging pattern but the
varied dimension is comparator votes per pair, not binary selector votes.

Inputs:
  - merged R=10 binary_votes.jsonl path(s) — for τ-gate computation and
    fallback selection
  - comparator votes path — pairwise_dynamic_comparator_votes.jsonl

The ablation runs at a single τ (default 0.5 yes-rate gate) so it
isolates the comparator-vote dimension; pass --thresholds to sweep.

Offline. No API calls.
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


def parse_floats(raw: str) -> list[float]:
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


def yes_rate(row: dict) -> float:
    yes = valid = 0
    for v in row.get("binary_votes") or []:
        if v.get("resolves") is None:
            continue
        valid += 1
        if v.get("resolves") is True:
            yes += 1
    return yes / valid if valid else 0.0


def conf_yes(row: dict) -> int:
    return sum(int(v.get("confidence") or 0)
               for v in row.get("binary_votes") or []
               if v.get("resolves") is True)


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


def winner_for_subset(row: dict, vote_subset: tuple[int, ...]) -> tuple[str, int]:
    """Aggregate the chosen vote_idx subset into one (winner, confidence)."""
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
    n = counts[winner]
    return winner, max(1, round(conf[winner] / n))


def copeland_subset(iid: str, candidates: list[int],
                    cmp_rows: dict[str, dict[tuple[int, int], dict]],
                    vote_subset: tuple[int, ...],
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
            winner, conf = winner_for_subset(row, vote_subset)
            weight = conf if mode == "confidence_weighted" else 1
            if winner == "A":
                scores[a] += weight
                scores[b] -= weight
            elif winner == "B":
                scores[b] += weight
                scores[a] -= weight
    return sorted(scores, key=lambda k: (-scores[k], k))[0]


def evaluate(
    iids: list[str],
    binary: dict[str, dict[int, dict]],
    cmp_rows: dict[str, dict[tuple[int, int], dict]],
    k_list: list[int],
    thr: float,
    vote_subset: tuple[int, ...],
    mode: str = "confidence_weighted",
) -> dict:
    resolved = total = 0
    missing = 0
    for iid in iids:
        rows = binary[iid]
        cands = [k for k in k_list if yes_rate(rows[k]) >= thr]
        if not cands:
            cands = [sorted(((k, conf_yes(rows[k])) for k in k_list),
                            key=lambda x: (-x[1], x[0]))[0][0]]
        chosen = copeland_subset(iid, cands, cmp_rows, vote_subset, mode=mode)
        if chosen is None:
            missing += 1
            continue
        r = rows[chosen].get("patch_resolves")
        if r in (True, False):
            total += 1
            resolved += int(bool(r))
    return {
        "vote_subset": list(vote_subset),
        "resolved": resolved,
        "total": total,
        "rate": resolved / total if total else None,
        "missing_pair_instances": missing,
    }


def summarize_n(results: list[dict]) -> dict:
    rates = [r["rate"] for r in results if r["rate"] is not None]
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
    p.add_argument("--binary-votes", required=True, nargs="+")
    p.add_argument("--comparator-votes", required=True)
    p.add_argument("--k-list", default=",".join(map(str, DEFAULT_K_LIST)))
    p.add_argument("--max-votes", type=int, default=10,
                   help="Number of comparator votes per pair available in cache (M)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="τ yes-rate gate to apply before Copeland")
    p.add_argument("--output-json", default="")
    p.add_argument("--copeland-mode", choices=("majority", "confidence_weighted"),
                   default="confidence_weighted")
    args = p.parse_args()

    k_list = parse_k_list(args.k_list)
    binary = load_binary([Path(p) for p in args.binary_votes])
    cmp_rows = load_comparator(Path(args.comparator_votes))
    iids = sorted(iid for iid, rows in binary.items() if set(k_list).issubset(rows))

    summary = {
        "binary_votes": args.binary_votes,
        "comparator_votes": args.comparator_votes,
        "k_list": k_list,
        "threshold": args.threshold,
        "max_votes": args.max_votes,
        "copeland_mode": args.copeland_mode,
        "complete_instances": len(iids),
        "by_n_votes": {},
    }
    all_results = {}
    for n in range(1, args.max_votes + 1):
        subsets = list(itertools.combinations(range(args.max_votes), n))
        # cap subset enumeration when it would explode (n=5 from 10 → 252 sets, fine;
        # all 1023 subsets across n=1..10 is OK since each loops over instances quickly)
        n_results = [evaluate(iids, binary, cmp_rows, k_list, args.threshold, sub,
                              mode=args.copeland_mode) for sub in subsets]
        all_results[str(n)] = n_results
        summary["by_n_votes"][str(n)] = summarize_n(n_results)

    prev = None
    for n in range(1, args.max_votes + 1):
        rec = summary["by_n_votes"][str(n)]
        rate = rec["mean_rate"]
        rec["delta_from_previous"] = None if prev is None or rate is None else rate - prev
        if rate is not None:
            prev = rate

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({"summary": summary, "by_vote_subset": all_results}, f,
                      indent=2, sort_keys=True)
            f.write("\n")

    print("Comparator voter-count ablation (Layer 3):")
    print(f"  binary_votes: {args.binary_votes}")
    print(f"  comparator_votes: {args.comparator_votes}")
    print(f"  τ={args.threshold}, M={args.max_votes}, copeland={args.copeland_mode}")
    print(f"  complete_instances: {len(iids)}")
    print()
    print("m_votes\tmean_rate\tmin\tmax\tnum_subsets\tdelta")
    for n in range(1, args.max_votes + 1):
        rec = summary["by_n_votes"][str(n)]
        rate = rec["mean_rate"]
        delta = rec["delta_from_previous"]
        if rate is None:
            print(f"{n}\tNA\tNA\tNA\t{rec['num_subsets']}\tNA")
        else:
            d = "NA" if delta is None else f"{delta:+.4f}"
            print(f"{n}\t{rate:.4f}\t{rec['min_rate']:.4f}\t{rec['max_rate']:.4f}\t"
                  f"{rec['num_subsets']}\t{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
