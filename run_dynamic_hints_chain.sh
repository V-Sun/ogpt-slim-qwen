#!/bin/bash
# Chain: dynamic-hints smoke (5 inst) → auto-promote → full 500 → RUN_SUMMARY.
# Budget cap: $250 (in direct_binary_dynamic_hints.py).
# Hints: FAIL_TO_PASS names + error trace from cached test_output.txt + test_patch.
# (No source-line extraction — user approved skipping that.)
set -u
cd /home/vsun/ogpt-slim-qwen
mkdir -p outputs

echo "========================================================================"
echo "[$(date)] DYNAMIC-HINTS CHAIN BEGIN"
echo "========================================================================"

# --- STEP 1: 5-instance smoke ---
echo "[$(date)] === STEP 1: 5-instance smoke ==="
rm -rf outputs/direct_binary_dynamic_hints_smoke
mkdir -p outputs/direct_binary_dynamic_hints_smoke

python3 -u direct_binary_dynamic_hints.py --smoke 5 \
    > outputs/dynamic_smoke.log 2>&1
SMOKE_EXIT=$?
echo "[$(date)] smoke exit code: $SMOKE_EXIT"

if [ $SMOKE_EXIT -ne 0 ]; then
    echo "[$(date)] FATAL: smoke failed. Aborting."
    tail -40 outputs/dynamic_smoke.log
    exit 1
fi

# Health check on smoke
DONE_LINE=$(grep -E '^\[done\] calls=' outputs/dynamic_smoke.log | tail -1)
SMOKE_COST=$(echo "$DONE_LINE" | grep -oE 'cost=\$[0-9.]+' | tr -d 'cos=$')
SMOKE_ABS=$(echo "$DONE_LINE" | grep -oE 'abstains=[0-9]+' | tr -d 'abstains=')
SMOKE_TRACE_COUNT=$(grep -E 'extracted error traces for [0-9]+/' outputs/dynamic_smoke.log | tail -1 | grep -oE '[0-9]+/[0-9]+' | head -1)
echo "[$(date)] smoke summary: cost=\$${SMOKE_COST} abstains=${SMOKE_ABS} traces=${SMOKE_TRACE_COUNT}"

if [ -z "$SMOKE_COST" ] || [ -z "$SMOKE_ABS" ]; then
    echo "[$(date)] FATAL: cannot parse smoke output. Aborting."
    tail -40 outputs/dynamic_smoke.log
    exit 1
fi

# Smoke is healthy iff cost reasonable + abstains low.
# (Smoke = 5 inst × 8 patches × 5 votes = 200 calls expected.
#  Cost projection at $0.005/call ~ $1.00; allow $3 for headroom.)
HEALTHY=$(awk -v c="$SMOKE_COST" -v a="$SMOKE_ABS" \
    'BEGIN { print ((c+0)<3.0 && (a+0)<25) ? "yes" : "no" }')

if [ "$HEALTHY" != "yes" ]; then
    echo "[$(date)] UNHEALTHY: cost=\$${SMOKE_COST} abstains=${SMOKE_ABS} — aborting."
    exit 1
fi
echo "[$(date)] smoke healthy. Auto-promoting to full 500."

# --- STEP 2: Full 500 ---
echo "[$(date)] === STEP 2: full 500 instances ==="
python3 -u direct_binary_dynamic_hints.py --all \
    > outputs/dynamic_full500.log 2>&1
FULL_EXIT=$?
echo "[$(date)] full500 exit code: $FULL_EXIT"

if [ $FULL_EXIT -ne 0 ]; then
    echo "[$(date)] WARNING: full run exited non-zero. Cache may be partial; "
    echo "          continuing to summary generation anyway."
fi

# --- STEP 3: Generate RUN_SUMMARY ---
echo "[$(date)] === STEP 3: RUN_SUMMARY_DYNAMIC_HINTS.md ==="
python3 generate_dynamic_hints_summary.py \
    > outputs/RUN_SUMMARY_DYNAMIC_HINTS.md \
    2> outputs/dynamic_summary_errors.log
echo "[$(date)] Summary written to outputs/RUN_SUMMARY_DYNAMIC_HINTS.md"

echo "========================================================================"
echo "[$(date)] DYNAMIC-HINTS CHAIN END"
echo "========================================================================"
