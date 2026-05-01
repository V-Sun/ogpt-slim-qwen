"""CLI to run Orchestra-GPT committee on N SWE-bench tasks.

Usage:
    python orchestra/run.py NUM_TASKS [K] [M] [R] [RUN_ID]

Example:
    python orchestra/run.py 5 3 1 1     # Smoke: 5 tasks, K=3, M=1, R=1
    python orchestra/run.py 50 10 5 3   # Paper config: 50 tasks, K=10, M=5, R=3
    python orchestra/run.py 500 16 0 0  # Proposer-only ablation, all 500 tasks
"""

import json
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from orchestra.committee import run_committee
from config import (
    MODEL, DATASET, SPLIT,
    MODEL_PROVIDER, MODEL_NAME,
    GENERATOR_MODEL, REVIEWER_MODEL, COMPARATOR_MODEL,
    OPENAI_BASE_URL, AZURE_API_BASE, AZURE_DEPLOYMENT, AZURE_REASONING_EFFORT,
)


def run_orchestra(num_tasks: int = 5, k: int = 10, m: int = 5, r: int = 5,
                   run_id: str = None):
    """Run committee on num_tasks SWE-bench instances.

    Args:
        num_tasks: Number of tasks to run.
        k: Number of proposers per task.
        m: Number of critics per proposal. M=0 skips reviewer stage AND suppresses
            predictions.jsonl writes — feeding proposer 0's patch (deterministic
            tiebreak) to the SWE-bench harness would silently return a K=1 result
            even though K proposers ran. Use predictions_per_proposer.jsonl with
            a downstream selection script (e.g., oracle eval) instead.
        r: Number of comparisons per pair (use odd for majority). R=0 skips comparator.
        run_id: Output run identifier.

    Returns:
        (preds_file_path_or_None, run_id) tuple. preds_file_path is None when M=0.
    """
    if run_id is None:
        run_id = f"orchestra_k{k}_m{m}_r{r}_{num_tasks}tasks"

    output_dir = f"outputs/{run_id}"
    preds_file = f"{output_dir}/predictions.jsonl"
    proposers_file = f"{output_dir}/predictions_per_proposer.jsonl"
    done_dir = Path(output_dir) / ".done"
    os.makedirs(output_dir, exist_ok=True)
    done_dir.mkdir(exist_ok=True)

    suppress_predictions = (m == 0)

    log_file = f"{output_dir}/run.log"
    error_log_file = f"{output_dir}/error.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    error_logger.addHandler(error_handler)

    # Bordered run-config banner so any pasted log fragment is self-describing.
    # Must NOT print API keys.
    logger.info("=" * 60)
    logger.info("Run config")
    logger.info("=" * 60)
    logger.info(f"  K / M / R         = {k} / {m} / {r}")
    logger.info(f"  RUN_ID            = {run_id}")
    logger.info(f"  NUM_TASKS         = {num_tasks}")
    logger.info(f"  DATASET           = {DATASET} (split={SPLIT})")
    logger.info(f"  MODEL_PROVIDER    = {MODEL_PROVIDER}")
    logger.info(f"  MODEL_NAME        = {MODEL_NAME}")
    if (GENERATOR_MODEL != MODEL_NAME
            or REVIEWER_MODEL != MODEL_NAME
            or COMPARATOR_MODEL != MODEL_NAME):
        logger.info(f"  GENERATOR_MODEL   = {GENERATOR_MODEL}")
        logger.info(f"  REVIEWER_MODEL    = {REVIEWER_MODEL}")
        logger.info(f"  COMPARATOR_MODEL  = {COMPARATOR_MODEL}")
    if MODEL_PROVIDER == "openai_compatible":
        logger.info(f"  OPENAI_BASE_URL   = {OPENAI_BASE_URL}")
    else:
        logger.info(f"  AZURE_API_BASE    = {AZURE_API_BASE}")
        logger.info(f"  AZURE_DEPLOYMENT  = {AZURE_DEPLOYMENT}")
        if AZURE_REASONING_EFFORT:
            logger.info(f"  AZURE_REASONING_EFFORT = {AZURE_REASONING_EFFORT}")
    logger.info("=" * 60)

    if suppress_predictions:
        logger.warning(
            "M=0: skipping predictions.jsonl. Use predictions_per_proposer.jsonl "
            "with a downstream selection script (e.g., oracle eval) to compute "
            "any aggregate number — the proposer-0-as-winner row would be a silent "
            "K=1 result if scored directly."
        )

    logger.info(f"\nLoading {DATASET}...")
    dataset = load_dataset(DATASET, split=SPLIT)
    instances = list(dataset)[:num_tasks]

    # Resumability: completion is tracked differently for M=0 vs M>0.
    #   M>0: predictions.jsonl is the marker (one row per finished instance).
    #   M=0: per-instance .done/<instance_id> marker file written atomically AFTER
    #        all K proposer rows have been flushed. Partial rows from a crashed
    #        run leave orphan rows in predictions_per_proposer.jsonl but do NOT
    #        block re-execution; the next run re-processes the instance.
    completed = set()
    if suppress_predictions:
        if done_dir.exists():
            completed = {p.name for p in done_dir.iterdir() if p.is_file()}
    else:
        if os.path.exists(preds_file):
            with open(preds_file) as f:
                for line in f:
                    if line.strip():
                        completed.add(json.loads(line)["instance_id"])
    if completed:
        logger.info(f"Already completed: {len(completed)}/{num_tasks}")

    task_concurrency = int(os.getenv("ORCHESTRA_TASK_CONCURRENCY", "1"))
    logger.info(f"Task concurrency: {task_concurrency}")

    results = []
    failed_tasks = []
    write_lock = threading.Lock()

    pending = [(i, inst) for i, inst in enumerate(instances)
               if inst["instance_id"] not in completed]

    def process_instance(args):
        i, instance = args
        logger.info(f"\n{'#'*60}")
        logger.info(f"Task {i+1}/{num_tasks}: {instance['instance_id']}")
        logger.info(f"{'#'*60}")
        try:
            result, proposals = run_committee(instance, k, m, r, MODEL, output_dir)
            logger.info(f"✓ Task {i+1}/{num_tasks} completed: {instance['instance_id']}")
            return result, proposals, None
        except Exception as e:
            error_msg = f"✗ Task {i+1}/{num_tasks} FAILED: {instance['instance_id']}"
            logger.error(error_msg)
            logger.error(f"Error: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            error_logger.error(f"\n{'='*60}")
            error_logger.error(error_msg)
            error_logger.error(f"Error: {str(e)}")
            error_logger.error(f"Full traceback:\n{traceback.format_exc()}")
            error_logger.error(f"{'='*60}\n")
            failed_entry = {
                "instance_id": instance["instance_id"],
                "task_number": i+1,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            empty_result = {
                "instance_id": instance["instance_id"],
                "model_name_or_path": f"orchestra_k{k}_m{m}_r{r}",
                "model_patch": "",
                "error": str(e),
                "failed": True
            }
            return empty_result, [], failed_entry

    out = open(preds_file, "a") if not suppress_predictions else None
    try:
        with open(proposers_file, "a") as prop_out:
            with ThreadPoolExecutor(max_workers=task_concurrency) as executor:
                futures = {executor.submit(process_instance, args): args for args in pending}
                for future in as_completed(futures):
                    result, proposals, failure = future.result()
                    with write_lock:
                        if failure:
                            failed_tasks.append(failure)
                        else:
                            results.append(result)
                        if out is not None:
                            out.write(json.dumps(result) + "\n")
                            out.flush()
                        # Per-proposer view: one row per (instance, k_index).
                        # Always written so K=N/M=0 runs preserve all candidates.
                        for p in proposals:
                            prop_row = {
                                "instance_id": p.get("instance_id", result["instance_id"]),
                                "model_name_or_path": f"orchestra_k{k}_m{m}_r{r}_proposer_{p.get('k_index', -1)}",
                                "model_patch": p.get("model_patch", ""),
                                "k_index": p.get("k_index", -1),
                                "nonempty": p.get("nonempty", False),
                                "cost": p.get("cost", 0.0),
                                "steps": p.get("steps", 0),
                                "exit_status": p.get("exit_status", "unknown"),
                            }
                            prop_out.write(json.dumps(prop_row) + "\n")
                        prop_out.flush()
                        if suppress_predictions and not failure:
                            # Atomic completion marker for M=0 resumability.
                            # Only written AFTER all K rows for this instance have been flushed.
                            os.fsync(prop_out.fileno())
                            (done_dir / result["instance_id"]).touch()
    finally:
        if out is not None:
            out.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"All tasks complete!")
    logger.info(f"Successful: {len(results)}/{num_tasks}")
    if failed_tasks:
        logger.warning(f"Failed: {len(failed_tasks)}/{num_tasks}")
        logger.warning(f"Failed tasks: {[t['instance_id'] for t in failed_tasks]}")
        with open(f"{output_dir}/failed_tasks.json", "w") as f:
            json.dump(failed_tasks, f, indent=2)
        logger.info(f"Failed tasks details: {output_dir}/failed_tasks.json")

    logger.info(f"Per-proposer file: {proposers_file}")
    if not suppress_predictions:
        logger.info(f"Predictions: {preds_file}")
    logger.info(f"Log file: {log_file}")
    if failed_tasks:
        logger.info(f"Error log: {error_log_file}")

    if suppress_predictions:
        logger.info(
            f"\nM=0 run: no predictions.jsonl was written. "
            f"Score per-proposer candidates from {proposers_file} via a "
            f"downstream oracle / pass@K selection script."
        )
    else:
        logger.info(f"\nNow score with SWE-bench harness:")
        logger.info(f"  python -m swebench.harness.run_evaluation \\")
        logger.info(f"    --dataset_name {DATASET} \\")
        logger.info(f"    --predictions_path {preds_file} \\")
        logger.info(f"    --max_workers 2 \\")
        logger.info(f"    --run_id {run_id}")

    return (None if suppress_predictions else preds_file), run_id


if __name__ == "__main__":
    num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    r = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    run_id = sys.argv[5] if len(sys.argv) > 5 else None
    run_orchestra(num_tasks, k, m, r, run_id)
