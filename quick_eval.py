#!/usr/bin/env python3
"""Quick evaluation of specific predictions to see if they pass tests."""

import json
import subprocess
import tempfile
from pathlib import Path

# Extract the 3 tasks with solutions from xhigh run
predictions_file = "outputs/orchestra_xhigh_nano_100tasks/predictions.jsonl"

with open(predictions_file) as f:
    all_preds = [json.loads(line) for line in f if line.strip()]

# Filter to only tasks with solutions
with_solutions = [p for p in all_preds if p.get('winner_k_index', -1) >= 0]

print(f"Found {len(with_solutions)} tasks with solutions:")
for p in with_solutions:
    print(f"  - {p['instance_id']}")

# Create a temp file with just these predictions
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    temp_pred_file = f.name
    for pred in with_solutions:
        f.write(json.dumps(pred) + '\n')

print(f"\nCreated temp predictions file: {temp_pred_file}")
print(f"Now run evaluation manually with:")
print(f"  python3 -m swebench.harness.run_evaluation \\")
print(f"    --predictions_path {temp_pred_file} \\")
print(f"    --swe_bench_tasks princeton-nlp/SWE-Bench_Verified \\")
print(f"    --log_dir outputs/quick_eval_results \\")
print(f"    --testbed outputs/quick_eval_testbed \\")
print(f"    --timeout 900")
