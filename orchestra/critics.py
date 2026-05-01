"""Critic agents that evaluate and filter patch proposals.

From paper: critics are necessary — without them, step-level decomposition
is worse than pass@K. Critics convert decomposition from liability to advantage.
"""

import json
import re
import os
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestra.llm import call_text_model
from orchestra.rate_limiter import get_api_semaphore

# Environment variables are set in config.py

CRITIC_PROMPT_TEMPLATE = """You are a code review expert evaluating a patch for a GitHub issue.

ISSUE:
{problem_statement}

PROPOSED PATCH:
{patch}

Evaluate this patch on these criteria:
1. Does it address the specific issue described?
2. Is the diff syntactically valid (proper +/- lines, correct file paths)?
3. Does it make a targeted fix or does it make unrelated changes?
4. Does this patch introduce signature mismatches, missing imports, or type inconsistencies that would prevent compilation or cause runtime errors?

Respond in this EXACT JSON format — JSON only, no markdown fences, no extra text:
{{
  "score": <integer 0-10>,
  "sound": <true if score >= 6, false otherwise>,
  "reasoning": "<one sentence explaining your score>",
  "red_flags": ["<specific problem if any, or empty list>"]
}}

Example of a well-formed response:
{{"score": 7, "sound": true, "reasoning": "Patch correctly fixes the off-by-one in the loop bounds and imports are consistent.", "red_flags": []}}

Example of a response flagging a compilation issue:
{{"score": 2, "sound": false, "reasoning": "Patch calls foo(x, y) but the updated signature is foo(x), causing a TypeError at runtime.", "red_flags": ["signature_mismatch"]}}"""


def critic_evaluate(problem_statement: str, patch: str, model_name: str) -> dict:
    """One critic evaluates one patch.

    Args:
        problem_statement: The GitHub issue description
        patch: The proposed patch (diff format)
        model_name: Model string like "Qwen/Qwen3.6-35B-A3B" or "azure/<your-deployment>"

    Returns:
        dict with score (0-10), sound (bool), reasoning, red_flags
    """
    if not patch or not patch.strip():
        return {
            "score": 0,
            "sound": False,
            "reasoning": "empty patch provided",
            "red_flags": ["no_patch"],
        }

    prompt = CRITIC_PROMPT_TEMPLATE.format(
        problem_statement=problem_statement[:2000],
        patch=patch[:8000],
    )
    # xhigh reasoning regularly burns 1-3k hidden tokens before emitting the
    # ~150-token JSON body. A 1600 cap starved the visible output ~1-in-4
    # calls in the smoke (empty "" response → JSONDecodeError). 6144 gives
    # plenty of headroom without materially changing latency or cost.
    max_out = int(os.getenv("ORCHESTRA_CRITIC_MAX_OUTPUT_TOKENS", "6144"))

    try:
        semaphore = get_api_semaphore()
        with semaphore:
            text = call_text_model(model_name, prompt, max_output_tokens=max_out)
        text = (text or "").strip()

        # Empty response: xhigh reasoning may have starved the visible output
        # despite the raised cap, or the Responses API returned only hidden
        # reasoning. Filter (do not approve) and flag so we can distinguish
        # from genuine low-score rejections.
        if not text:
            return {
                "score": 0,
                "sound": False,
                "reasoning": "critic returned empty text (reasoning starved output?)",
                "red_flags": ["empty_response"],
            }

        # Azure content-filter refusal. The patch is benign code review
        # material but the filter occasionally trips. Distinct from parse
        # error so we can track refusal rate separately.
        lowered = text.lower()
        if ("i'm sorry" in lowered or "i am sorry" in lowered) and "cannot" in lowered:
            return {
                "score": 0,
                "sound": False,
                "reasoning": f"azure content-filter refusal: {text[:120]}",
                "red_flags": ["refusal"],
            }

        # Strip markdown fences if present
        text = re.sub(r"```json\s*|\s*```", "", text).strip()

        # Tolerate a preamble like "Here is my review:" by grabbing the first
        # balanced JSON object in the response.
        if not text.startswith("{"):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)

        result = json.loads(text)
        result["sound"] = result.get("score", 0) >= 6
        return result

    except json.JSONDecodeError as e:
        print(f"      Critic JSON parse error: {e} | head={text[:120]!r}")
        return {
            "score": 0,
            "sound": False,  # Format drift should filter, not pass through
            "reasoning": f"critic parse error: {str(e)[:50]}",
            "red_flags": ["parse_error"],
        }
    except Exception as e:
        print(f"      Critic error: {e}")
        return {
            "score": 0,
            "sound": False,  # Errors should filter, not pass through
            "reasoning": f"critic error: {str(e)[:50]}",
            "red_flags": ["critic_error"],
        }


def run_m_critics(proposal: dict, problem_statement: str,
                   m: int, model_name: str) -> dict:
    """Run m critics on one proposal in parallel.

    Conservative filtering: proposal is filtered if ANY critic marks it unsound.
    From paper: edge β ≥ α0 under D1 detection condition.

    Args:
        proposal: Proposal dict from proposer
        problem_statement: GitHub issue text
        m: Number of critics to run
        model_name: Model string

    Returns:
        Proposal dict with critic_scores, critic_avg, critic_sound, filtered added
    """
    if not proposal["nonempty"]:
        # Empty patch — automatically filtered
        proposal["critic_scores"] = []
        proposal["critic_avg"] = 0.0
        proposal["critic_sound"] = False
        proposal["filtered"] = True
        return proposal

    # Run m critics IN PARALLEL (rate limiting handled by semaphore in critic_evaluate)
    scores = []
    # Use large worker pool - actual concurrency limited by global semaphore
    with ThreadPoolExecutor(max_workers=min(m, 100)) as executor:
        futures = [
            executor.submit(
                critic_evaluate,
                problem_statement,
                proposal["model_patch"],
                model_name,
            )
            for _ in range(m)
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                scores.append(result)
            except Exception as e:
                print(f"      Critic thread error: {e}")
                # Add neutral score on thread error
                scores.append({
                    "score": 5,
                    "sound": True,
                    "reasoning": f"thread error: {str(e)[:50]}",
                    "red_flags": [],
                })

    # Red-flag filter (Option C): a proposal is filtered only when EVERY
    # critic returned a structural failure (empty/refusal/parse_error/
    # critic_error) — i.e. the patch was unevaluable, not low-scoring.
    # Score-based rejection is delegated to the tournament's pairwise
    # comparator. This preserves the filter → tournament pipeline while
    # avoiding poorly-calibrated thresholds on score.
    STRUCTURAL_FLAGS = {"empty_response", "refusal", "parse_error", "critic_error"}
    unevaluable = [
        s for s in scores if set(s.get("red_flags") or []) & STRUCTURAL_FLAGS
    ]
    all_unevaluable = scores and len(unevaluable) == len(scores)
    avg_score = sum(s["score"] for s in scores) / len(scores) if scores else 0

    proposal["critic_scores"] = scores
    proposal["critic_avg"] = avg_score
    # critic_sound is retained for downstream logging but no longer gates
    # selection. True iff at least one critic produced a usable score.
    proposal["critic_sound"] = not all_unevaluable
    proposal["filtered"] = bool(all_unevaluable)

    return proposal


def filter_proposals(proposals: list[dict], problem_statement: str,
                      m: int, model_name: str) -> tuple[list, list]:
    """Run critics on all proposals and filter in parallel.

    Args:
        proposals: List of proposal dicts from proposers
        problem_statement: GitHub issue
        m: Number of critics per proposal
        model_name: Model string

    Returns:
        (survivors, filtered_out) tuple

    Safety: If ALL proposals get filtered, keep the highest-scoring one anyway.
    We need at least one candidate for the comparator stage.
    """
    print(f"  [Critics] Running {m} critics per proposal in parallel...")

    # Only evaluate non-empty proposals
    nonempty_proposals = [p for p in proposals if p["nonempty"]]

    if not nonempty_proposals:
        print(f"  [Critics] No non-empty proposals to evaluate")
        return [], proposals

    # Evaluate each proposal IN PARALLEL (rate limiting handled by semaphore)
    evaluated = []
    # Use large worker pool - actual concurrency limited by global semaphore
    with ThreadPoolExecutor(max_workers=min(len(nonempty_proposals), 100)) as executor:
        futures = {
            executor.submit(run_m_critics, p, problem_statement, m, model_name): p
            for p in nonempty_proposals
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                evaluated.append(result)
            except Exception as e:
                proposal = futures[future]
                print(f"  [Critics] Error evaluating proposal {proposal.get('k_index', '?')}: {e}")
                # Mark as filtered on error
                proposal["critic_scores"] = []
                proposal["critic_avg"] = 0.0
                proposal["critic_sound"] = False
                proposal["filtered"] = True
                evaluated.append(proposal)

    # Separate survivors and filtered
    survivors = [p for p in evaluated if not p["filtered"]]
    filtered = [p for p in evaluated if p["filtered"]]

    # Safety: if every proposal was unevaluable (all M critics errored for
    # every candidate), pass them through anyway — tournament is the decider.
    # Under Option C we don't fall back to a score-based "best" since critic
    # scores aren't trusted for ranking.
    if not survivors and evaluated:
        print(f"  [Critics] All proposals unevaluable — passing through; tournament decides")
        for p in evaluated:
            p["filtered"] = False
        survivors = evaluated
        filtered = []

    print(f"  [Critics] Result: {len(survivors)} survivors, {len(filtered)} filtered (structural-flag-only)")

    return survivors, filtered
