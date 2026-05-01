"""Independent proposer agents that generate patch candidates.

Each proposer runs mini-swe-agent with a different system prompt to reduce
correlation ρ. From paper Theorem 4: independent contexts required for committee advantage.
"""

import sys
import os
import threading
from pathlib import Path
import concurrent.futures
import time
import logging
import subprocess

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "mini-swe-agent" / "src"))

from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.config import get_config_from_spec, builtin_config_dir
from orchestra.llm import proposer_model_config


PROPOSER_TIMEOUT_SECONDS = int(os.getenv("ORCHESTRA_PROPOSER_TIMEOUT_SECONDS", "600"))
PROPOSER_CONCURRENCY = int(os.getenv("ORCHESTRA_PROPOSER_CONCURRENCY", "10"))

logger = logging.getLogger(__name__)


def _environment_class() -> str:
    """Mini-swe-agent environment_class. "docker" (default), "singularity", or "local".

    Setting this to anything other than "docker" disables every Docker shell-out
    in this module (image pre-pull, container cleanup, image rmi) and injects
    the chosen class into the proposer's agent config so mini-swe-agent's
    `get_sb_environment` builds the right environment.
    """
    return os.getenv("ORCHESTRA_ENVIRONMENT_CLASS", "docker").lower()


def _deep_merge(base: dict, overlay: dict) -> dict:
    """In-place deep-merge `overlay` into `base`. Mutates and returns `base`.

    Dicts merge recursively; everything else (lists, scalars) is replaced.
    """
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _maybe_overlay_proposer_config(config: dict) -> dict:
    """Deep-merge a YAML overlay onto the loaded base config if requested.

    Set `ORCHESTRA_PROPOSER_CONFIG_OVERLAY=path/to/file.yaml` (e.g.
    `configs/qwen_local.yaml`) to layer extra `agent`, `model`, or
    `environment` settings on top of the mini-swe-agent benchmark base.
    """
    overlay_path = os.getenv("ORCHESTRA_PROPOSER_CONFIG_OVERLAY", "").strip()
    if not overlay_path:
        return config
    overlay_file = Path(overlay_path)
    if not overlay_file.is_absolute():
        overlay_file = Path(__file__).parent.parent / overlay_file
    if not overlay_file.exists():
        logger.warning(f"ORCHESTRA_PROPOSER_CONFIG_OVERLAY={overlay_path} not found; ignoring")
        return config
    with overlay_file.open() as f:
        overlay = yaml.safe_load(f) or {}
    if not isinstance(overlay, dict):
        logger.warning(f"Overlay {overlay_path} is not a YAML mapping; ignoring")
        return config
    return _deep_merge(config, overlay)

# Per-instance locks so parallel proposers on the same task serialize their
# image pulls.  Kept keyed by instance_id (not image name) — identical in practice
# for Verified/Pro since each instance has exactly one image.
_image_pull_locks: dict[str, threading.Lock] = {}
_image_pull_locks_guard = threading.Lock()


def _image_for_instance(instance_id: str, is_pro: bool) -> str:
    # Pro images live under swebench_pro/sweb.eval.x86_64.<slug>:latest,
    # Verified under swebench/sweb.eval.x86_64.<slug>:latest.  Using docker.io
    # default namespace so inspect/pull names match mini-swe-agent's.
    repo = "swebench_pro" if is_pro else "swebench"
    return f"docker.io/{repo}/sweb.eval.x86_64.{instance_id.replace('__','_1776_')}:latest"


def _ensure_image_pulled(instance_id: str, is_pro: bool = False,
                          max_attempts: int = 5, backoff: float = 5.0) -> None:
    """Docker pull an instance image if missing, coordinating across proposer threads.

    This exists because concurrent `docker run` on a missing image — which
    implicitly triggers pulls — cascades to exit-status-125 failures on this
    daemon.  Serializing the pull (via a per-instance lock) avoids the
    thundering-herd pathology.  The lock is released after the first pull
    succeeds; subsequent callers fast-path through a `docker image inspect`.
    """
    with _image_pull_locks_guard:
        lock = _image_pull_locks.setdefault(instance_id, threading.Lock())

    with lock:
        image = _image_for_instance(instance_id, is_pro=is_pro)
        # Fast path: image already local
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=30,
        )
        if inspect.returncode == 0:
            return

        for attempt in range(1, max_attempts + 1):
            logger.info(f"    [pull] {instance_id} attempt {attempt}/{max_attempts}: {image}")
            out = subprocess.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=900,
            )
            if out.returncode == 0:
                logger.info(f"    [pull] {instance_id} ok (attempt {attempt})")
                return
            err = (out.stderr or out.stdout or "")[-500:]
            # Rate-limit → long sleep; other transient → short sleep
            if any(s in err.lower() for s in ("toomanyrequests", "pull rate limit", "rate exceeded")):
                wait = 180.0
            else:
                wait = backoff * attempt
            logger.warning(f"    [pull] {instance_id} failed (rc={out.returncode}); "
                           f"sleeping {wait:.0f}s. stderr tail: {err!r}")
            time.sleep(wait)

        logger.error(f"    [pull] {instance_id} giving up after {max_attempts} attempts")


def cleanup_stale_containers(instance_id: str):
    """Clean up stale Docker containers from a completed task.

    Multiple shard workers can be running at once.  Do not remove every
    minisweagent container here; only remove containers whose image matches the
    completed SWE-bench instance.

    Args:
        instance_id: Task instance ID to clean up containers for
    """
    try:
        docker_instance = instance_id.replace("__", "_1776_")

        # Get all minisweagent containers with images so we can target only the
        # completed instance.
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Image}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            logger.warning(f"Failed to list containers: {result.stderr}")
            return

        removed_count = 0

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            container, image = parts
            if "minisweagent-" not in container:
                continue
            if docker_instance not in image:
                continue

            try:
                subprocess.run(
                    ["docker", "rm", "-f", container],
                    capture_output=True,
                    timeout=30
                )
                removed_count += 1
                logger.debug(f"Removed container: {container}")
            except Exception as e:
                logger.debug(f"Failed to remove container {container}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} stale container(s) for {instance_id}")

    except Exception as e:
        logger.warning(f"Container cleanup failed: {e}")


# For now, use identical prompts for all proposers
# Diversity will come from random sampling/temperature in future iterations
# This ensures all proposers use the full, working SWE-bench template


def run_single_proposer(instance: dict, k_index: int, model_name: str,
                         output_dir: str) -> dict:
    """Run one mini-swe-agent instance for one SWE-bench task.

    Args:
        instance: SWE-bench instance dict with problem_statement, instance_id, etc.
        k_index: Proposer index (0 to K-1)
        model_name: Model string like "Qwen/Qwen3.6-35B-A3B" or "azure/<your-deployment>"
        output_dir: Where to save trajectory files

    Returns:
        dict with instance_id, k_index, model_patch, cost, etc.
    """
    instance_id = instance["instance_id"]
    print(f"    Proposer {k_index} starting...")

    try:
        # When Responses-API reasoning is active (xhigh/high), match the
        # known-good `baseline_nano_xhigh_500` recipe: backticks system
        # prompt + parallel_tool_calls=False.  Verified by comparing saved
        # trajectories — with parallel_tool_calls=True many turns produce
        # reasoning_tokens=0 (the model skips reasoning), so effective
        # effort is well below the requested level.  Opt-out via
        # ORCHESTRA_PROPOSER_BASE_CONFIG=swebench (tool-use default).
        from orchestra.llm import should_use_responses_api
        base_cfg_name = os.getenv(
            "ORCHESTRA_PROPOSER_BASE_CONFIG",
            "swebench_backticks" if should_use_responses_api() else "swebench",
        )
        config_path = builtin_config_dir / "benchmarks" / f"{base_cfg_name}.yaml"
        config = get_config_from_spec(str(config_path))

        # Optional YAML overlay (e.g. configs/qwen_local.yaml) layered on top
        # of the mini-swe-agent benchmark base before our code-side overrides.
        config = _maybe_overlay_proposer_config(config)

        # Set model while preserving benchmark YAML defaults.
        config["model"] = proposer_model_config(model_name, config.get("model", {}))

        # Force parallel_tool_calls=False under reasoning to avoid the
        # zero-reasoning-on-some-turns pathology.
        if should_use_responses_api():
            config["model"].setdefault("model_kwargs", {})
            config["model"]["model_kwargs"]["parallel_tool_calls"] = False

        # Override agent cost_limit. The benchmark YAMLs default to $3 which
        # cuts xhigh proposers short on hard tasks (a few hundred steps × ~$0.01
        # per call).  Set to a very large number unless explicitly overridden.
        cost_limit_override = float(os.getenv("ORCHESTRA_PROPOSER_COST_LIMIT", "1000000"))
        config.setdefault("agent", {})["cost_limit"] = cost_limit_override

        # Inject environment_class so mini-swe-agent's get_sb_environment routes
        # to docker, singularity, or local. Precedence: $ORCHESTRA_ENVIRONMENT_CLASS
        # (when set) > YAML overlay > base YAML. Both mini-swe-agent benchmark
        # YAMLs hardcode "docker", so the env var must override unconditionally.
        env_config = config.setdefault("environment", {})
        explicit_env_class = os.getenv("ORCHESTRA_ENVIRONMENT_CLASS", "").strip().lower()
        if explicit_env_class:
            env_config["environment_class"] = explicit_env_class
        env_class = env_config.get("environment_class", "docker").lower()

        # Pro images use /app as repo root (sweagent Pro images bake code at
        # /app rather than the swebench Verified default /testbed).
        # Pro images also set ENTRYPOINT=["/bin/bash"], which breaks
        # mini-swe-agent's `docker run ... <image> sleep 2h` keepalive
        # (bash treats "sleep" as a script name). Override with --entrypoint "".
        # The run_args here are docker-specific, so only apply when env_class is docker.
        is_pro = "before_repo_set_cmd" in instance
        if is_pro:
            env_config["cwd"] = "/app"
            if env_class == "docker":
                env_config["run_args"] = ["--rm", "--entrypoint", ""]
            else:
                logger.warning(
                    f"Pro instance {instance_id} with environment_class={env_class}: "
                    "skipping docker-specific run_args; pro image setup may not work."
                )

        # Create model and environment.
        model = get_model(config=config.get("model", {}))

        # Pre-pull the docker image with a per-instance lock so k proposers
        # on the same task don't race 16 concurrent pulls against the docker
        # daemon. Skip when running under a non-docker environment class
        # (singularity caches images on first `docker://` resolve).
        if env_class == "docker":
            _ensure_image_pulled(instance_id, is_pro=is_pro)

        env = get_sb_environment(config, instance)

        # Pro: reset to base_commit and plant the fail_to_pass test file from
        # fix_commit. mini-swe-agent's get_sb_environment has an
        # env_startup_command hook (swebench.py:103) but passes a raw string
        # to env.execute() which expects {"command": ...}, so we invoke it
        # ourselves. Without this, Pro images leave the agent at fix_commit
        # with no task to solve.
        if is_pro:
            out = env.execute({"command": instance["before_repo_set_cmd"]})
            if out.get("returncode", 0) != 0:
                raise RuntimeError(
                    f"before_repo_set_cmd failed (rc={out.get('returncode')}): "
                    f"{out.get('output','')[:500]}"
                )

        # Run agent
        agent = DefaultAgent(model, env, **config.get("agent", {}))
        info = agent.run(instance["problem_statement"])

        # Extract result
        patch = info.get("submission", "")
        exit_status = info.get("exit_status", "unknown")

        # Save trajectory
        traj_dir = Path(output_dir) / instance_id
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file = traj_dir / f"proposer_{k_index}.traj.json"
        agent.save(
            traj_file,
            {
                "instance_id": instance_id,
                "k_index": k_index,
                "info": {"exit_status": exit_status, "submission": patch},
            },
        )

        result = {
            "instance_id": instance_id,
            "k_index": k_index,
            "model_patch": patch,
            "nonempty": bool(patch.strip()),
            "cost": agent.cost,
            "steps": agent.n_calls,
            "exit_status": exit_status,
        }

        print(f"    Proposer {k_index} done: {len(patch)} chars, ${agent.cost:.2f}")
        return result

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"    Proposer {k_index} failed: {e}")
        print(f"    Traceback:\n{tb}")
        return {
            "instance_id": instance_id,
            "k_index": k_index,
            "model_patch": "",
            "nonempty": False,
            "cost": 0.0,
            "steps": 0,
            "exit_status": f"error_{type(e).__name__}",
            "error": str(e),
            "traceback": tb,
        }


def run_k_proposers(instance: dict, k: int, model_name: str,
                     output_dir: str) -> list[dict]:
    """Run K proposers in parallel for one instance.

    Args:
        instance: SWE-bench instance
        k: Number of proposers to run
        model_name: Model string
        output_dir: Output directory

    Returns:
        List of K proposal dicts
    """
    instance_id = instance["instance_id"]
    logger.info(f"  [Proposers] Running {k} agents (max {PROPOSER_CONCURRENCY} concurrent) for {instance_id}...")
    logger.info(f"  [Proposers] Timeout: {PROPOSER_TIMEOUT_SECONDS}s per task")

    # Initialize all K proposers as pending to ensure we always have K results
    proposals_by_index: dict[int, dict] = {
        i: {
            "instance_id": instance_id,
            "k_index": i,
            "model_patch": "",
            "nonempty": False,
            "cost": 0.0,
            "steps": 0,
            "exit_status": "not_started",
            "error": "Proposer never started or was cancelled",
        }
        for i in range(k)
    }

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(k, PROPOSER_CONCURRENCY))
    futures = {}
    for i in range(k):
        try:
            future = executor.submit(run_single_proposer, instance, i, model_name, output_dir)
            futures[future] = i
            logger.debug(f"    Proposer {i} submitted to executor")
        except Exception as e:
            logger.error(f"    Proposer {i} failed to submit: {e}")
            proposals_by_index[i]["exit_status"] = f"submit_error_{type(e).__name__}"
            proposals_by_index[i]["error"] = str(e)

    deadline = time.monotonic() + PROPOSER_TIMEOUT_SECONDS
    start_time = time.monotonic()
    completed_count = 0

    try:
        while futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"  [Proposers] Deadline reached after {PROPOSER_TIMEOUT_SECONDS}s")
                break

            done, _ = concurrent.futures.wait(
                futures,
                timeout=min(5.0, remaining),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                k_index = futures.pop(future)
                try:
                    result = future.result()
                    proposals_by_index[k_index] = result
                    completed_count += 1
                    elapsed = time.monotonic() - start_time
                    logger.info(f"    Proposer {k_index} completed ({completed_count}/{k}) after {elapsed:.1f}s")
                except Exception as e:
                    logger.error(f"    Proposer {k_index} failed with exception: {e}")
                    proposals_by_index[k_index] = {
                        "instance_id": instance_id,
                        "k_index": k_index,
                        "model_patch": "",
                        "nonempty": False,
                        "cost": 0.0,
                        "steps": 0,
                        "exit_status": f"error_{type(e).__name__}",
                        "error": str(e),
                    }
                    completed_count += 1

        if futures:
            timed_out = sorted(futures.values())
            logger.warning(
                f"  [Proposers] Timeout after {PROPOSER_TIMEOUT_SECONDS}s; "
                f"{len(timed_out)} proposers still running: {timed_out}"
            )
            for future, k_index in list(futures.items()):
                future.cancel()
                proposals_by_index[k_index] = {
                    "instance_id": instance_id,
                    "k_index": k_index,
                    "model_patch": "",
                    "nonempty": False,
                    "cost": 0.0,
                    "steps": 0,
                    "exit_status": "timeout",
                    "error": f"Timed out after {PROPOSER_TIMEOUT_SECONDS}s",
                }
                logger.warning(f"    Proposer {k_index} marked as timed out")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Build final list - should always have exactly K proposals
    proposals = [proposals_by_index[i] for i in range(k)]

    # Verify we have all K proposals
    if len(proposals) != k:
        logger.error(f"  [Proposers] BUG: Expected {k} proposals, got {len(proposals)}")

    nonempty = sum(1 for p in proposals if p["nonempty"])
    total_cost = sum(p["cost"] for p in proposals)
    elapsed_total = time.monotonic() - start_time

    logger.info(f"  [Proposers] Complete: {nonempty}/{k} non-empty, ${total_cost:.2f} total, {elapsed_total:.1f}s elapsed")

    # Log status breakdown
    status_counts = {}
    for p in proposals:
        status = p.get("exit_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    logger.info(f"  [Proposers] Status breakdown: {status_counts}")

    # Docker-only post-task cleanup. Singularity/local environments don't
    # have lingering containers or a daemon-managed image cache.
    # Env var (if set) overrides; otherwise default to docker (matches the
    # base YAML default and preserves prior behavior).
    cleanup_env_class = os.getenv("ORCHESTRA_ENVIRONMENT_CLASS", "docker").strip().lower() or "docker"
    if cleanup_env_class == "docker":
        logger.info(f"  [Proposers] Cleaning up Docker containers...")
        cleanup_stale_containers(instance_id)
        # Disk pressure: containerd content store lives on /, so each image (~1GB)
        # can't accumulate. Free the image once all proposers for this task finish.
        # The image will be re-pulled if a future run needs it.
        if os.getenv("ORCHESTRA_RMI_AFTER_TASK", "1") == "1":
            try:
                is_pro = "before_repo_set_cmd" in instance
                image = _image_for_instance(instance_id, is_pro=is_pro)
                rm = subprocess.run(
                    ["docker", "rmi", "-f", image],
                    capture_output=True, text=True, timeout=60,
                )
                if rm.returncode == 0:
                    logger.info(f"  [Proposers] rmi {image} → ok")
                else:
                    logger.debug(f"  [Proposers] rmi {image} → rc={rm.returncode}: {rm.stderr.strip()[:200]}")
            except Exception as exc:
                logger.warning(f"  [Proposers] rmi failed: {exc}")

    return proposals
