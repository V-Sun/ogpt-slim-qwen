#!/usr/bin/env python3
"""Evaluate predictions from old runs using SWE-bench test harness."""

import json
import subprocess
import sys
from pathlib import Path

def evaluate_predictions(predictions_file: str):
    """Run SWE-bench evaluation on a predictions file."""
    predictions_path = Path(predictions_file)

    if not predictions_path.exists():
        print(f"ERROR: {predictions_file} does not exist")
        return None

    # Count predictions
    with open(predictions_path) as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Evaluating: {predictions_file}")
    print(f"Total predictions: {len(predictions)}")
    print(f"{'='*60}\n")

    if len(predictions) == 0:
        print("No predictions to evaluate")
        return None

    # Show first few instance_ids
    print("Instance IDs:")
    for i, pred in enumerate(predictions[:5], 1):
        print(f"  {i}. {pred.get('instance_id', 'unknown')}")
    if len(predictions) > 5:
        print(f"  ... and {len(predictions) - 5} more")

    # Run evaluation
    output_dir = predictions_path.parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    print(f"\nRunning evaluation (this may take a while)...")
    print(f"Output directory: {output_dir}")

    try:
        result = subprocess.run([
            "python3", "-m", "swebench.harness.run_evaluation",
            "--predictions_path", str(predictions_path),
            "--swe_bench_tasks", "princeton-nlp/SWE-Bench_Verified",
            "--log_dir", str(output_dir),
            "--testbed", str(output_dir / "testbed"),
            "--skip_existing", "false",
            "--timeout", "900",
            "--verbose"
        ], capture_output=True, text=True, timeout=3600)

        print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

        # Parse results
        results_file = output_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            resolved = sum(1 for r in results.values() if r.get("resolved", False))
            total = len(results)

            print(f"\n{'='*60}")
            print(f"RESULTS: {resolved}/{total} tasks resolved ({resolved/total*100:.1f}%)")
            print(f"{'='*60}\n")

            return {"resolved": resolved, "total": total, "rate": resolved/total}
        else:
            print(f"Results file not found at {results_file}")
            return None

    except subprocess.TimeoutExpired:
        print("Evaluation timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    runs_to_evaluate = [
        "outputs/orchestra_high_reasoning/predictions.jsonl",
        "outputs/orchestra_xhigh_nano_100tasks/predictions.jsonl",
    ]

    if len(sys.argv) > 1:
        runs_to_evaluate = sys.argv[1:]

    results_summary = {}
    for predictions_file in runs_to_evaluate:
        result = evaluate_predictions(predictions_file)
        if result:
            results_summary[predictions_file] = result

    print(f"\n{'='*60}")
    print("SUMMARY OF ALL EVALUATIONS")
    print(f"{'='*60}\n")
    for run, result in results_summary.items():
        print(f"{run}:")
        print(f"  Resolved: {result['resolved']}/{result['total']} ({result['rate']*100:.1f}%)")
