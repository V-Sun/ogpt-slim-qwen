#!/bin/bash
# Overnight chain: smoke → auto-promote → full 500 → RUN_SUMMARY.md
# User-approved: skip replication, scale direct-binary + hints to full 500.
# Hard budget cap $150 (in direct_binary_hints.py).
set -u
cd /home/vsun/ogpt-slim-qwen

mkdir -p outputs

echo "========================================================================"
echo "[$(date)] OVERNIGHT FULL-500 RUN BEGIN"
echo "========================================================================"

# --- STEP 1: Fresh smoke test (1 instance) ---
echo "[$(date)] === STEP 1: smoke (1 instance) ==="
# Force fresh smoke by wiping the directory
rm -rf outputs/direct_binary_hints_smoke
mkdir -p outputs/direct_binary_hints_smoke

python3 -u direct_binary_hints.py --smoke 1 > outputs/overnight_smoke.log 2>&1
SMOKE_EXIT=$?
echo "[$(date)] smoke exit code: $SMOKE_EXIT"

if [ $SMOKE_EXIT -ne 0 ]; then
    echo "[$(date)] FATAL: smoke failed. Aborting."
    tail -30 outputs/overnight_smoke.log
    exit 1
fi

# Health check on smoke
DONE_LINE=$(grep -E '^\[done\] calls=' outputs/overnight_smoke.log | tail -1)
SMOKE_COST=$(echo "$DONE_LINE" | grep -oE 'cost=\$[0-9.]+' | tr -d 'cos=$')
SMOKE_ABS=$(echo "$DONE_LINE" | grep -oE 'abstains=[0-9]+' | tr -d 'abstains=')
echo "[$(date)] smoke cost=\$${SMOKE_COST}  abstains=${SMOKE_ABS}"

if [ -z "$SMOKE_COST" ] || [ -z "$SMOKE_ABS" ]; then
    echo "[$(date)] FATAL: cannot parse smoke output. Aborting."
    tail -30 outputs/overnight_smoke.log
    exit 1
fi

HEALTHY=$(awk -v c="$SMOKE_COST" -v a="$SMOKE_ABS" \
    'BEGIN { print ((c+0)<0.30 && (a+0)<10) ? "yes" : "no" }')

if [ "$HEALTHY" != "yes" ]; then
    echo "[$(date)] UNHEALTHY: cost=\$${SMOKE_COST} abstains=${SMOKE_ABS} — aborting."
    exit 1
fi
echo "[$(date)] smoke healthy. Auto-promoting to full 500."

# --- STEP 2: Full 500 ---
echo "[$(date)] === STEP 2: full 500 instances ==="
python3 -u direct_binary_hints.py --all > outputs/overnight_full500.log 2>&1
FULL_EXIT=$?
echo "[$(date)] full500 exit code: $FULL_EXIT"

if [ $FULL_EXIT -ne 0 ]; then
    echo "[$(date)] WARNING: full run exited non-zero. Cache may be partial; "
    echo "          continuing to summary generation anyway."
fi

# --- STEP 3: Generate RUN_SUMMARY ---
echo "[$(date)] === STEP 3: RUN_SUMMARY_FULL500.md ==="
python3 generate_full500_summary.py > outputs/RUN_SUMMARY_FULL500.md 2> outputs/summary_gen_errors.log
echo "[$(date)] Summary written to outputs/RUN_SUMMARY_FULL500.md"

echo "========================================================================"
echo "[$(date)] OVERNIGHT FULL-500 RUN END"
echo "========================================================================"
