#!/bin/bash
# Chain: comparator-with-hints smoke (5 inst) → auto-promote to 50 pilot → analyze.
# Round-robin C(8,2)=28 matchups × R=5 × 2 swaps. Single committed prompt.
# Hints: FAIL_TO_PASS + error trace + file paths.
# Budget cap: $150 (in comparator_dynamic_hints.py).
set -u
cd /home/vsun/ogpt-slim-qwen
mkdir -p outputs

echo "========================================================================"
echo "[$(date)] COMPARATOR-HINTS CHAIN BEGIN"
echo "========================================================================"

# --- STEP 1: smoke ---
echo "[$(date)] === STEP 1: 5-instance smoke ==="
rm -rf outputs/comparator_dynamic_hints_smoke
mkdir -p outputs/comparator_dynamic_hints_smoke

python3 -u comparator_dynamic_hints.py --smoke 5 \
    > outputs/cmp_dyn_smoke.log 2>&1
SMOKE_EXIT=$?
echo "[$(date)] smoke exit code: $SMOKE_EXIT"
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "[$(date)] FATAL: smoke failed."
    tail -40 outputs/cmp_dyn_smoke.log
    exit 1
fi

DONE_LINE=$(grep -E '^\[done\] calls=' outputs/cmp_dyn_smoke.log | tail -1)
SMOKE_COST=$(echo "$DONE_LINE" | grep -oE 'cost=\$[0-9.]+' | tr -d 'cos=$')
SMOKE_ABS=$(echo "$DONE_LINE" | grep -oE 'abstains_cmp=[0-9]+' | tr -d 'abstains_cmp=')
echo "[$(date)] smoke: cost=\$${SMOKE_COST} abstains=${SMOKE_ABS}"

# Smoke = 5 × 28 × 5 × 2 = 1400 calls. At $0.005/call, ~$7.
# Allow up to $20 for headroom and abstains < 100.
HEALTHY=$(awk -v c="$SMOKE_COST" -v a="$SMOKE_ABS" \
    'BEGIN { print ((c+0)<20.0 && (a+0)<100) ? "yes" : "no" }')
if [ "$HEALTHY" != "yes" ]; then
    echo "[$(date)] UNHEALTHY: cost=\$${SMOKE_COST} abstains=${SMOKE_ABS} — aborting."
    exit 1
fi
echo "[$(date)] smoke healthy. Auto-promoting to 50 pilot."

# --- STEP 2: pilot (50 instances) ---
echo "[$(date)] === STEP 2: 50 pilot ==="
python3 -u comparator_dynamic_hints.py --pilot \
    > outputs/cmp_dyn_pilot.log 2>&1
PILOT_EXIT=$?
echo "[$(date)] pilot exit code: $PILOT_EXIT"

if [ $PILOT_EXIT -ne 0 ]; then
    echo "[$(date)] WARNING: pilot exited non-zero. Cache may be partial."
fi

# Surface pilot decision
echo "[$(date)] === STEP 3: pilot decision ==="
grep -A5 'DECISION' outputs/cmp_dyn_pilot.log | tail -10

echo "========================================================================"
echo "[$(date)] COMPARATOR-HINTS CHAIN END"
echo "========================================================================"
