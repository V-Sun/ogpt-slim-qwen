#!/usr/bin/env bash
# Step 2 driver: fire all 4 offline ablations in parallel against the
# merged R=10 binary-votes cache. Inputs:
#   $1 = primary binary_votes.jsonl (vote_idx 0..4)
#   $2 = sidecar binary_votes.jsonl (vote_idx 5..9)
#   $3 = output dir for analysis JSONs
# Each analysis prints its summary table to its own log; consolidated
# console output goes to $3/all_step2.log.
set -euo pipefail

PRIMARY="${1:?primary binary_votes.jsonl path required}"
SIDECAR="${2:?sidecar binary_votes.jsonl path required}"
OUT="${3:?output dir required}"
mkdir -p "$OUT"

REPO=/home/vsun/ogpt-slim-qwen
BINARY="--binary-votes $PRIMARY $SIDECAR"
KLIST="1,3,5,7,8,9,12,15"

echo "[step2] kicking off 4 offline ablations in parallel against R=10 cache"
echo "  primary:  $PRIMARY"
echo "  sidecar:  $SIDECAR"
echo "  out_dir:  $OUT"
echo

python3 "$REPO/analyze_voter_ablation.py" $BINARY --k-list "$KLIST" \
  --max-votes 10 --output-json "$OUT/voter_ablation.json" \
  > "$OUT/voter_ablation.log" 2>&1 &
PID2A=$!

python3 "$REPO/analyze_gated_comparator_thresholds.py" $BINARY --k-list "$KLIST" \
  --output-json "$OUT/gated_thresholds.json" \
  > "$OUT/gated_thresholds.log" 2>&1 &
PID2B=$!

python3 "$REPO/analyze_greedy8_bridge.py" \
  > "$OUT/greedy8_bridge.log" 2>&1 &
PID2C=$!

python3 "$REPO/analyze_k_sweep_ablation.py" $BINARY --k-list "$KLIST" \
  --output-json "$OUT/k_sweep_ablation.json" \
  > "$OUT/k_sweep_ablation.log" 2>&1 &
PID2D=$!

echo "[step2] PIDs 2a=$PID2A 2b=$PID2B 2c=$PID2C 2d=$PID2D"
wait $PID2A $PID2B $PID2C $PID2D
echo "[step2] all 4 ablations done"

echo
echo "============ 2a voter ablation ============"
cat "$OUT/voter_ablation.log"
echo
echo "============ 2b gated thresholds ============"
cat "$OUT/gated_thresholds.log"
echo
echo "============ 2c greedy-8 bridge ============"
cat "$OUT/greedy8_bridge.log"
echo
echo "============ 2d K-sweep ablation ============"
cat "$OUT/k_sweep_ablation.log"
