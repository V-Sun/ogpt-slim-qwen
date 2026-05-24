#!/usr/bin/env python3
"""Offline aggregation for tracks 7 & 8 — $0, no API calls.

Reads pairwise_votes.jsonl from blind_comparator runs and binary_votes.jsonl
from blind_selector runs, then applies all aggregation rules and hybrid configs.

Aggregation rules (track 7):
  - copeland        Round-robin Copeland (default)
  - bracket         Single-elimination (seeded by patch_idx)
  - king_of_hill    Sequential king-of-the-hill chain
  - swiss           3-round Swiss pairing by current standing

Hybrid (track 8):
  - selector top-N (N=2,3,4) → comparator Copeland on survivors

Usage:
    python3 aggregate_and_report.py \
        --selector-votes outputs/blind_sel_*/binary_votes.jsonl \
        --comparator-votes outputs/blind_cmp_*/pairwise_votes.jsonl \
        --label "nano_pilot"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
K_LIST  = list(range(8))
DENOM   = 500


# ── Vote loading ──────────────────────────────────────────────────────────────

def load_selector(paths: list[Path]) -> dict[str, dict[int, list[bool]]]:
    """Return {iid: {k: [bool ...]}}."""
    out: dict[str, dict[int, list[bool]]] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r["instance_id"]
            k   = int(r["patch_idx"])
            bv  = [v.get("resolves") is True for v in (r.get("binary_votes") or [])]
            out.setdefault(iid, {})[k] = bv
    return out


def load_comparator(paths: list[Path]) -> dict[str, dict[tuple[int,int], list[str]]]:
    """Return {iid: {(a,b): [winner ...]}} where winner in {A,B,TIE,None}."""
    out: dict[str, dict[tuple[int,int], list[str]]] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.open():
            if not line.strip():
                continue
            r    = json.loads(line)
            iid  = r["instance_id"]
            a, b = int(r["patch_a_idx"]), int(r["patch_b_idx"])
            winners = []
            for v in (r.get("votes") or []):
                fwd = (v.get("primary") or {}).get("parsed") or {}
                w   = str(fwd.get("winner") or "").upper()
                # If position_swap: average fwd + secondary
                sec_p = (v.get("secondary") or {}).get("parsed") or {}
                sw    = str(sec_p.get("winner") or "").upper()
                if sw in ("A", "B", "TIE") and w in ("A", "B", "TIE"):
                    # agree → use fwd; disagree → TIE
                    winners.append(w if w == sw else "TIE")
                elif w in ("A", "B", "TIE"):
                    winners.append(w)
                else:
                    winners.append(None)
            out.setdefault(iid, {})[(a, b)] = winners
    return out


def load_truth(selector: dict, comparator: dict) -> dict[str, dict[int, bool]]:
    """Extract ground truth from patch_resolves in selector rows if present."""
    truth: dict[str, dict[int, bool]] = {}
    for iid, ks in selector.items():
        pass  # truth not embedded in blind selector rows
    return truth


def load_oracle_truth() -> dict[str, dict[int, bool]]:
    """Load truth from orchestra-gpt eval reports."""
    truth: dict[str, dict[int, bool]] = defaultdict(dict)
    base = ROOT.parent / "orchestra-gpt" / "logs" / "run_evaluation"
    # canonical K -> original K mapping
    new_to_old = {0:7, 1:3, 2:8, 3:9, 4:5, 5:15, 6:1, 7:12}
    for new_k, orig_k in new_to_old.items():
        for rep in base.glob(f"oracle_*_proposer_{orig_k}/*/*/report.json"):
            try:
                j   = json.load(rep.open())
                iid = rep.parent.name
                rec = j.get(iid, j) if isinstance(j, dict) else {}
                truth[iid][new_k] = bool(rec.get("resolved", False))
            except Exception:
                pass
    return dict(truth)


# ── Head-to-head ─────────────────────────────────────────────────────────────

def h2h(cmp: dict[tuple[int,int], list[str]], a: int, b: int, n: int
        ) -> tuple[int, int]:
    """Return (a_wins, b_wins) from first n votes."""
    vlist = cmp.get((a, b)) or cmp.get((b, a)) or []
    swapped = (b, a) in cmp and (a, b) not in cmp
    aw = bw = 0
    for w in vlist[:n]:
        if w is None:
            continue
        if not swapped:
            if w == "A": aw += 1
            elif w == "B": bw += 1
        else:
            if w == "A": bw += 1
            elif w == "B": aw += 1
    return aw, bw


# ── Aggregation rules ─────────────────────────────────────────────────────────

def copeland(cmp: dict, pool: list[int], n: int) -> int:
    scores = {k: 0 for k in pool}
    for a, b in combinations(pool, 2):
        aw, bw = h2h(cmp, a, b, n)
        if aw > bw:   scores[a] += 1; scores[b] -= 1
        elif bw > aw: scores[b] += 1; scores[a] -= 1
    return max(pool, key=lambda k: (scores[k], -k))


def bracket(pool: list[int], cmp: dict, n: int) -> int:
    """Single-elimination seeded by patch_idx (0 is first seed)."""
    survivors = sorted(pool)
    while len(survivors) > 1:
        next_round = []
        for i in range(0, len(survivors) - 1, 2):
            a, b = survivors[i], survivors[i+1]
            aw, bw = h2h(cmp, a, b, n)
            next_round.append(a if aw >= bw else b)
        if len(survivors) % 2 == 1:
            next_round.append(survivors[-1])
        survivors = next_round
    return survivors[0] if survivors else pool[0]


def king_of_hill(pool: list[int], cmp: dict, n: int) -> int:
    """Sequential king-of-the-hill — challenger must strictly win to dethrone."""
    king = pool[0]
    for challenger in pool[1:]:
        aw, bw = h2h(cmp, king, challenger, n)
        if bw > aw:  # challenger wins outright
            king = challenger
    return king


def swiss_system(pool: list[int], cmp: dict, n: int, rounds: int = 3) -> int:
    """Swiss pairing: 3 rounds, pair by current standing, Copeland tiebreak."""
    scores = {k: 0 for k in pool}
    paired_history: set[frozenset] = set()

    for _ in range(rounds):
        ordered = sorted(pool, key=lambda k: (-scores[k], k))
        paired_this_round: set[int] = set()
        for i in range(len(ordered)):
            if ordered[i] in paired_this_round:
                continue
            for j in range(i + 1, len(ordered)):
                if ordered[j] in paired_this_round:
                    continue
                pair = frozenset([ordered[i], ordered[j]])
                if pair in paired_history:
                    continue
                a, b = ordered[i], ordered[j]
                aw, bw = h2h(cmp, a, b, n)
                if aw > bw:   scores[a] += 1
                elif bw > aw: scores[b] += 1
                paired_history.add(pair)
                paired_this_round.add(a)
                paired_this_round.add(b)
                break

    return max(pool, key=lambda k: (scores[k], -k))


# ── Selector filter (track 8) ─────────────────────────────────────────────────

def selector_top_n(sel: dict[int, list[bool]], pool: list[int],
                   n_sel_votes: int, top_n: int) -> list[int]:
    """Return top-N patches by yes-vote count; fallback to all if too few."""
    scored = sorted(pool, key=lambda k: (-sum(sel.get(k, [])[:n_sel_votes]), k))
    survivors = scored[:top_n]
    return survivors if survivors else pool


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(selections: dict[str, int], truth: dict[str, dict[int, bool]],
             label: str, denom: int = DENOM) -> dict:
    resolved = total = 0
    for iid, chosen_k in selections.items():
        if iid in truth and chosen_k in truth[iid]:
            total += 1
            if truth[iid][chosen_k]:
                resolved += 1
    rate = resolved / denom
    print(f"  {label:50s}  {resolved:3d}/{denom}  {rate*100:.1f}%  "
          f"(scored {total}/{len(selections)} instances)")
    return {"label": label, "resolved": resolved, "denom": denom,
            "rate": rate, "scored": total}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selector-votes",   nargs="*", default=[])
    p.add_argument("--comparator-votes", nargs="*", default=[])
    p.add_argument("--label",            default="run")
    p.add_argument("--n-sel-votes",      type=int, default=5)
    p.add_argument("--n-cmp-votes",      type=int, default=5)
    p.add_argument("--out",              default=None)
    args = p.parse_args()

    sel_paths = [Path(x) for x in args.selector_votes]
    cmp_paths = [Path(x) for x in args.comparator_votes]

    sel = load_selector(sel_paths)
    cmp = load_comparator(cmp_paths)

    iids      = sorted(set(sel) | set(cmp))
    truth     = load_oracle_truth()
    pilot_iids = iids  # use whatever was provided

    print(f"\n[aggregate] label={args.label}  instances={len(iids)}")
    print(f"  selector instances: {len(sel)}  comparator instances: {len(cmp)}")
    print(f"  truth loaded for {len(truth)} instances")
    print()
    print(f"  {'Method':50s}  {'Resolved':>10}  Rate")
    print(f"  {'-'*50}  {'-'*10}  ------")

    results = []

    # ── Track 7: aggregation rules on comparator only ───────────────────────
    for rule_name, rule_fn in [
        ("Copeland",         lambda cmp, pool, n: copeland(cmp, pool, n)),
        ("Bracket",          lambda cmp, pool, n: bracket(pool, cmp, n)),
        ("King-of-the-hill", lambda cmp, pool, n: king_of_hill(pool, cmp, n)),
        ("Swiss (3 rounds)", lambda cmp, pool, n: swiss_system(pool, cmp, n)),
    ]:
        if not cmp:
            continue
        sels: dict[str, int] = {}
        for iid in iids:
            pool = [k for k in K_LIST if any((iid, k) == (iid, ka) or
                    (k == ka or k == kb) for (ka, kb) in (cmp.get(iid) or {}))]
            pool = K_LIST  # use full pool if data exists
            if iid in cmp:
                try:
                    sels[iid] = rule_fn(cmp[iid], K_LIST, args.n_cmp_votes)
                except Exception:
                    sels[iid] = K_LIST[0]
            else:
                sels[iid] = K_LIST[0]
        r = evaluate(sels, truth, f"Comparator {rule_name}")
        results.append(r)

    # ── Track 8: hybrid selector top-N + comparator Copeland ────────────────
    if sel and cmp:
        for top_n in (2, 3, 4):
            sels = {}
            for iid in iids:
                pool = selector_top_n(sel.get(iid, {}), K_LIST,
                                      args.n_sel_votes, top_n)
                if len(pool) == 1:
                    sels[iid] = pool[0]
                elif iid in cmp:
                    try:
                        sels[iid] = copeland(cmp.get(iid, {}), pool, args.n_cmp_votes)
                    except Exception:
                        sels[iid] = pool[0]
                else:
                    sels[iid] = pool[0]
            r = evaluate(sels, truth, f"Hybrid top-{top_n} + Copeland")
            results.append(r)

    # ── Selector-only baseline ───────────────────────────────────────────────
    if sel:
        sels = {}
        for iid, ks in sel.items():
            sels[iid] = max(K_LIST,
                            key=lambda k: (sum(ks.get(k, [])[:args.n_sel_votes]), -k))
        r = evaluate(sels, truth, "Selector only (most-yes)")
        results.append(r)

    # ── Oracle ceiling on pilot instances ────────────────────────────────────
    oracle_res = sum(1 for iid in iids
                     if iid in truth and any(truth[iid].get(k) for k in K_LIST))
    print(f"\n  Oracle (≥1 of 8 resolves):  {oracle_res}/{DENOM} = {oracle_res/DENOM*100:.1f}%")

    # ── Write RUN_SUMMARY.md row ─────────────────────────────────────────────
    summary_path = ROOT / "outputs" / "RUN_SUMMARY.md"
    with summary_path.open("a") as f:
        for r in results:
            f.write(f"| {args.label} | {r['label']} | {r['resolved']}/{r['denom']} "
                    f"| {r['rate']*100:.1f}% |\n")
    print(f"\n[aggregate] appended {len(results)} rows to {summary_path}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
