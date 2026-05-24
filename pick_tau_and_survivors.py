#!/usr/bin/env python3
"""Step 3 helper: pick τ for the Layer-3 comparator and write the
instance-filter file consumed by `anchored_dynamic_comparator.py`.

Reads the threshold-gated analysis output (from
`analyze_gated_comparator_thresholds.py --output-json`) and projects
comparator cost at each τ:

    cost ≈ #τ-instances × C(K, 2) × M × position_swap_factor × $/call

where #τ-instances = instances with ≥2 candidates surviving the τ gate.
At τ values where #candidates < 2 the comparator has nothing to choose
between — those instances are skipped.

Picks the smallest τ whose projected cost ≤ budget × safety_factor, and
writes a JSON list of the surviving instance_ids (the format consumed by
`anchored_dynamic_comparator.py --instances-path`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gated-thresholds-json", required=True,
                   help="Output of analyze_gated_comparator_thresholds.py --output-json")
    p.add_argument("--binary-votes", required=True, nargs="+",
                   help="binary_votes.jsonl files; needed to enumerate τ-survivor instance_ids")
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--m-votes", type=int, default=10)
    p.add_argument("--position-swap", action="store_true")
    p.add_argument("--cost-per-call-usd", type=float, default=0.012)
    p.add_argument("--safety-factor", type=float, default=0.9,
                   help="Project at safety_factor x budget. Default 0.9 (10pct buffer)")
    p.add_argument("--k-list", default="0,1,2,3,4,5,6,7")
    p.add_argument("--output-survivors", required=True,
                   help="Write list of surviving instance_ids here")
    args = p.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    n_pairs_per_instance = len(k_list) * (len(k_list) - 1) // 2
    swap_factor = 2 if args.position_swap else 1
    calls_per_instance = n_pairs_per_instance * args.m_votes * swap_factor
    cost_per_instance = calls_per_instance * args.cost_per_call_usd
    target_budget = args.budget_usd * args.safety_factor

    print(f"K-pool size: {len(k_list)}; pairs/instance: {n_pairs_per_instance}")
    print(f"M={args.m_votes}, position_swap={'on' if args.position_swap else 'off'} → "
          f"{calls_per_instance} calls/instance")
    print(f"$/call={args.cost_per_call_usd:.4f} → ${cost_per_instance:.2f}/instance")
    print(f"Budget cap=${args.budget_usd:.2f}, target=${target_budget:.2f} ({args.safety_factor*100:.0f}%)")
    print()

    gated = json.loads(Path(args.gated_thresholds_json).read_text())
    thresholds = sorted(gated["thresholds"].keys(), key=float)

    print("τ\tsurv_inst\tprojected_$\tunder_budget?")
    feasible = []
    for tau_str in thresholds:
        b = gated["thresholds"][tau_str]
        # surviving instances = those with ≥2 candidates (comparator has work to do)
        n_inst = b.get("multi_candidate_instances", 0)
        proj = n_inst * cost_per_instance
        ok = proj <= target_budget
        print(f"{float(tau_str):.2f}\t{n_inst}\t${proj:.2f}\t{'yes' if ok else 'no'}")
        if ok and n_inst > 0:
            feasible.append((float(tau_str), n_inst, proj))

    if not feasible:
        print("\nNo τ fits the budget. Drop M, drop position-swap, or raise budget.")
        return 2

    chosen_tau, n_chosen, chosen_cost = min(feasible, key=lambda x: x[0])
    print(f"\nChosen τ = {chosen_tau:.2f} → {n_chosen} instances, ~${chosen_cost:.2f}")

    # Enumerate the actual instance_ids that pass τ with ≥2 candidates.
    # This requires the binary cache to identify per-iid candidates.
    from collections import defaultdict
    binary = defaultdict(dict)
    for path in args.binary_votes:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                binary[row["instance_id"]][int(row["patch_idx"])] = row

    def yes_rate(row):
        yes = valid = 0
        for v in row.get("binary_votes") or []:
            if v.get("resolves") is None:
                continue
            valid += 1
            if v.get("resolves") is True:
                yes += 1
        return yes / valid if valid else 0.0

    survivors = []
    for iid, rows in binary.items():
        if not set(k_list).issubset(rows):
            continue
        cands = [k for k in k_list if yes_rate(rows[k]) >= chosen_tau]
        if len(cands) >= 2:
            survivors.append(iid)

    survivors.sort()
    out = Path(args.output_survivors)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(survivors, indent=2) + "\n")
    print(f"Wrote {len(survivors)} survivors → {out}")
    print(f"Discrepancy vs gated_thresholds report: {abs(len(survivors) - n_chosen)} (should be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
