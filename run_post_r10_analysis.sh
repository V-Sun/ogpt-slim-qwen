#!/usr/bin/env bash
# Auto-fire post-R=10-sidecar analysis pipeline.
# Runs all offline ablations on the merged R=10 cache (K=0..7, dynamic, high reasoning),
# then re-runs the worst-anchored sweep with significance, and writes a final summary doc.
set -euo pipefail

REPO=/home/vsun/ogpt-slim-qwen
PRIMARY="$REPO/outputs/direct_binary_dynamic_hints_full500/binary_votes.jsonl"
SIDECAR="$REPO/outputs/direct_binary_dynamic_hints_full500_R10ext/binary_votes.jsonl"
OUT="$REPO/outputs/r10_K07_dynamic_high"
KLIST="0,1,2,3,4,5,6,7"
mkdir -p "$OUT"

cd "$REPO"

if [ ! -f "$SIDECAR" ]; then
  echo "FATAL: sidecar not found at $SIDECAR"
  exit 1
fi
SIDECAR_ROWS=$(wc -l < "$SIDECAR")
PRIMARY_ROWS=$(wc -l < "$PRIMARY")
echo "[r10] primary R=5 rows: $PRIMARY_ROWS"
echo "[r10] sidecar R=10ext rows: $SIDECAR_ROWS"

BV="--binary-votes $PRIMARY $SIDECAR"

echo "[r10] firing all 4 ablations + K-sweep in parallel"
python3 analyze_voter_ablation.py $BV --k-list "$KLIST" --max-votes 10 \
  --output-json "$OUT/voter_ablation.json" > "$OUT/voter_ablation.log" 2>&1 &
A=$!
python3 analyze_gated_comparator_thresholds.py $BV --k-list "$KLIST" \
  --output-json "$OUT/gated_thresholds.json" > "$OUT/gated_thresholds.log" 2>&1 &
B=$!
python3 analyze_k_sweep_ablation.py $BV --k-list "$KLIST" \
  --output-json "$OUT/k_sweep.json" > "$OUT/k_sweep.log" 2>&1 &
C=$!
wait $A $B $C
echo "[r10] all 3 parallel ablations done"

echo "[r10] running worst-anchored sweep with McNemar"
python3 << PY > "$OUT/worst_anchored.log" 2>&1
import json, math
from collections import defaultdict
from pathlib import Path

PATHS = ["$PRIMARY", "$SIDECAR"]
K_LIST = [0,1,2,3,4,5,6,7]
OUT = Path("$OUT/worst_anchored.json")

by_iid = defaultdict(dict)
for p in PATHS:
    for line in open(p):
        if not line.strip(): continue
        r = json.loads(line)
        iid = r["instance_id"]; k = int(r["patch_idx"])
        existing = by_iid[iid].get(k)
        if existing is None:
            by_iid[iid][k] = r
            continue
        merged = {int(v.get("vote_idx")): v for v in existing.get("binary_votes", [])}
        for v in r.get("binary_votes", []) or []:
            try: merged[int(v.get("vote_idx"))] = v
            except: continue
        existing["binary_votes"] = [merged[i] for i in sorted(merged)]
        if existing.get("patch_resolves") is None and r.get("patch_resolves") is not None:
            existing["patch_resolves"] = r.get("patch_resolves")

iids = sorted(iid for iid, rows in by_iid.items() if set(K_LIST).issubset(rows))

# Verify R=10
sample_votes = len((by_iid[iids[0]][K_LIST[0]] or {}).get("binary_votes", []))
print(f"complete instances: {len(iids)}, votes per (iid,k): {sample_votes}")

def conf_yes(row):
    return sum(int(v.get("confidence") or 0) for v in row.get("binary_votes") or [] if v.get("resolves") is True)
def yes_count(row):
    return sum(1 for v in row.get("binary_votes") or [] if v.get("resolves") is True)

solo = {}
for k in K_LIST:
    res = tot = 0
    for iid in iids:
        r = by_iid[iid][k].get("patch_resolves")
        if r in (True, False):
            tot += 1; res += int(bool(r))
    solo[k] = (res, tot, res/tot if tot else 0.0)
order = sorted(K_LIST, key=lambda k: (solo[k][2], k))
print(f"order worst→best: {order}")

def outcome(pool, mode):
    out = {}
    for iid in iids:
        rows = by_iid[iid]
        if mode == "conf":
            scored = sorted(((k, conf_yes(rows[k])) for k in pool), key=lambda x: (-x[1], x[0]))
        else:
            scored = sorted(((k, yes_count(rows[k])) for k in pool), key=lambda x: (-x[1], x[0]))
        k_pick = scored[0][0]
        r = rows[k_pick].get("patch_resolves")
        out[iid] = bool(r) if r in (True, False) else None
    return out

def mcn_p(a, b):
    if a + b == 0: return 1.0
    stat = (abs(b - a) - 1) ** 2 / (a + b)
    return math.erfc(math.sqrt(stat)/math.sqrt(2))

result = {
    "binary_votes": PATHS, "k_list": K_LIST, "complete_instances": len(iids),
    "solo_per_k": {str(k): {"resolved": solo[k][0], "total": solo[k][1], "rate": solo[k][2]} for k in K_LIST},
    "order_worst_first": order,
    "by_n": {},
}
for sel_name, mode in [("confidence_weighted", "conf"), ("majority", "maj")]:
    base = outcome([order[0]], mode)
    rows = []
    for n in range(1, len(K_LIST)+1):
        cur = outcome(order[:n], mode)
        res = tot = a = b = 0
        for iid in iids:
            c = cur[iid]; o = base[iid]
            if c in (True, False):
                tot += 1; res += int(c)
            if o is None or c is None: continue
            if o and not c: a += 1
            if not o and c: b += 1
        rate = res/tot if tot else 0.0
        p = mcn_p(a, b) if (a + b) > 0 else 1.0
        rows.append({"n": n, "pool": order[:n], "rate": rate, "resolved": res,
                     "total": tot, "vs_worst_pp": (rate - solo[order[0]][2])*100,
                     "mcnemar_b_gains": b, "mcnemar_a_regressions": a, "mcnemar_p_2tail": p})
    result["by_n"][sel_name] = rows

OUT.write_text(json.dumps(result, indent=2) + "\n")
print(f"\nworst K = {order[0]}, solo rate = {solo[order[0]][2]:.4f}")
print()
print("CONFIDENCE-WEIGHTED:")
print(f"{'n':>3}  {'pool':<28} {'rate':>8}  {'res/tot':>10}  {'Δworst':>8}  {'b':>3}  {'a':>3}  {'p':>10}")
for r in result["by_n"]["confidence_weighted"]:
    pool_s = "{" + ",".join(map(str, r["pool"])) + "}"
    print(f"{r['n']:>3}  {pool_s:<28} {r['rate']:>8.4f}  {r['resolved']:>4}/{r['total']:<5}  {r['vs_worst_pp']:>+7.2f}  {r['mcnemar_b_gains']:>3}  {r['mcnemar_a_regressions']:>3}  {r['mcnemar_p_2tail']:>10.6f}")
print()
print("MAJORITY:")
print(f"{'n':>3}  {'pool':<28} {'rate':>8}  {'res/tot':>10}  {'Δworst':>8}  {'b':>3}  {'a':>3}  {'p':>10}")
for r in result["by_n"]["majority"]:
    pool_s = "{" + ",".join(map(str, r["pool"])) + "}"
    print(f"{r['n']:>3}  {pool_s:<28} {r['rate']:>8.4f}  {r['resolved']:>4}/{r['total']:<5}  {r['vs_worst_pp']:>+7.2f}  {r['mcnemar_b_gains']:>3}  {r['mcnemar_a_regressions']:>3}  {r['mcnemar_p_2tail']:>10.6f}")
PY

echo "[r10] worst-anchored sweep done"
echo
echo "============== voter ablation (n=1..10) =============="
cat "$OUT/voter_ablation.log"
echo
echo "============== K-sweep ablation =============="
cat "$OUT/k_sweep.log"
echo
echo "============== gated thresholds =============="
cat "$OUT/gated_thresholds.log"
echo
echo "============== worst-anchored + McNemar =============="
cat "$OUT/worst_anchored.log"

echo
echo "[r10] all R=10 analyses written to: $OUT"
ls -la "$OUT"
