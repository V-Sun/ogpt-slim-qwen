"""Full Π_{k,m,r} committee protocol orchestrator.

Wires together: proposers (Generator pool, K) → critics (Reviewer pool, M) →
comparators (Comparator pool, R-round Copeland tournament with position-swap).

Stages are skipped cleanly when the parameter is 0:
  - M=0: no reviewer LLM calls; all nonempty proposals advance.
  - R=0: no comparator LLM calls; lowest k_index among survivors wins (deterministic).
"""

import logging
import random

from orchestra.proposers import run_k_proposers
from orchestra.critics import filter_proposals
from orchestra.comparators import copeland_winner
from config import MODEL, GENERATOR_MODEL, REVIEWER_MODEL, COMPARATOR_MODEL

logger = logging.getLogger(__name__)


def run_committee(instance: dict, k: int, m: int, r: int,
                   model_name: str, output_dir: str) -> tuple[dict, list[dict]]:
    """Full committee protocol Π_{k,m,r} for one SWE-bench instance.

    Args:
        instance: SWE-bench instance dict with problem_statement, instance_id, etc.
        k: Number of proposers.
        m: Number of critics per proposal. M=0 skips the reviewer stage entirely.
        r: Number of comparisons per pair (use odd for majority). R=0 skips the
            comparator stage; the lowest-indexed survivor wins by deterministic
            tiebreak.
        model_name: Model string. If this matches the configured `MODEL`, the
            per-role overrides (GENERATOR_MODEL, REVIEWER_MODEL, COMPARATOR_MODEL)
            from config are used; otherwise model_name overrides all three.
        output_dir: Where to save intermediate files.

    Returns:
        (winner_result, proposals) tuple.
        winner_result is the prediction dict for the SWE-bench harness:
            instance_id, model_name_or_path, model_patch (winning patch),
            committee metadata (k_proposals, critics_filtered, winner_k_index, etc.).
        proposals is the list of every proposer's output (always K entries),
            so callers can persist a per-proposer view of the run.
    """
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]

    if model_name == MODEL:
        gen_model, rev_model, cmp_model = GENERATOR_MODEL, REVIEWER_MODEL, COMPARATOR_MODEL
    else:
        gen_model = rev_model = cmp_model = model_name

    logger.info(f"\n{'='*60}")
    logger.info(f"[Committee Π_({k},{m},{r})] {instance_id}")
    logger.info(f"{'='*60}")

    # Stage 1: K proposers generate candidates independently
    logger.info(f"[Committee] Stage 1: Running {k} proposers...")
    proposals = run_k_proposers(instance, k, gen_model, output_dir)

    nonempty_count = sum(1 for p in proposals if p["nonempty"])
    logger.info(f"[Committee] Stage 1 complete: {nonempty_count}/{k} nonempty proposals")

    def _empty_result(filtered_count: int) -> dict:
        return {
            "instance_id": instance_id,
            "model_name_or_path": f"orchestra_k{k}_m{m}_r{r}",
            "model_patch": "",
            "k_proposals": k,
            "k_nonempty": nonempty_count,
            "critics_filtered": filtered_count,
            "winner_k_index": -1,
            "winner_critic_avg": 0.0,
        }

    if nonempty_count == 0:
        logger.warning(f"[Committee] All {k} proposers returned empty patches")
        return _empty_result(0), proposals

    # Stage 2: M critics filter unsound proposals (skip entirely when M=0)
    if m > 0:
        logger.info(f"[Committee] Stage 2: Running {m} critics to filter proposals...")
        try:
            survivors, filtered = filter_proposals(
                proposals, problem_statement, m, rev_model
            )
            logger.info(f"[Committee] Stage 2 complete: {len(survivors)} survivors, {len(filtered)} filtered")
        except Exception as e:
            logger.error(f"[Committee] Stage 2 FAILED: Critics failed with error: {e}")
            logger.warning(f"[Committee] Falling back: Using all {nonempty_count} nonempty proposals as survivors")
            survivors = [p for p in proposals if p["nonempty"]]
            filtered = []
    else:
        logger.info(f"[Committee] Stage 2 skipped (M=0): all nonempty proposals advance")
        survivors = [p for p in proposals if p["nonempty"]]
        filtered = []

    if not survivors:
        logger.warning(f"[Committee] All proposals filtered by critics")
        return _empty_result(len(filtered)), proposals

    # Stage 3: R-round Copeland tournament with position-swap debiasing
    # Skip when R=0 or only one survivor remains; pick lowest k_index deterministically.
    if r > 0 and len(survivors) > 1:
        logger.info(f"[Committee] Stage 3: Running tournament with {len(survivors)} survivors and {r} rounds...")
        try:
            winner = copeland_winner(survivors, problem_statement, r, cmp_model)
            logger.info(f"[Committee] Stage 3 complete: Proposer {winner['k_index']} won")
        except Exception as e:
            logger.error(f"[Committee] Stage 3 FAILED: Tournament failed with error: {e}")
            logger.warning(f"[Committee] Falling back: Randomly selecting winner from {len(survivors)} survivors")
            winner = random.choice(survivors)
            winner["critic_avg"] = 0.0
            logger.info(f"[Committee] Fallback winner: Proposer {winner['k_index']}")
    else:
        # R=0 or single survivor: deterministic tiebreak on k_index ascending.
        winner = min(survivors, key=lambda p: p["k_index"])
        winner.setdefault("critic_avg", 0.0)
        why = "single survivor" if len(survivors) == 1 else "R=0, deterministic tiebreak"
        logger.info(f"[Committee] Stage 3 skipped ({why}): proposer {winner['k_index']} wins")

    result = {
        "instance_id": instance_id,
        "model_name_or_path": f"orchestra_k{k}_m{m}_r{r}",
        "model_patch": winner["model_patch"],
        "k_proposals": k,
        "k_nonempty": nonempty_count,
        "critics_filtered": len(filtered),
        "survivors_count": len(survivors),
        "winner_k_index": winner["k_index"],
        "winner_critic_avg": winner.get("critic_avg", 0.0),
        "winner_cost": winner.get("cost", 0.0),
        "winner_steps": winner.get("steps", 0),
    }

    logger.info(f"\n[Committee] RESULT:")
    logger.info(f"  Proposer {winner['k_index']} won with critic_avg={winner.get('critic_avg', 0):.1f}")
    logger.info(f"  Patch: {len(winner['model_patch'])} chars")
    logger.info(f"  Stats: {nonempty_count}/{k} proposed, {len(filtered)} filtered, "
          f"{len(survivors)} in tournament")

    return result, proposals
