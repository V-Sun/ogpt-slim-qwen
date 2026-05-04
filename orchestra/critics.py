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

FLAG_NAMES = (
    "addresses_root_cause",
    "preserves_existing_behavior",
    "tests_actually_test_the_issue",
    "no_unrelated_changes",
)
FLAG_PASS_THRESHOLD = 0.8  # per-flag agreement gate across the M critics


CRITIC_PROMPT_TEMPLATE = """You are evaluating whether a patch resolves a GitHub issue. Apply ONE test:

  A correct patch is the smallest change that makes the failing tests pass
  without breaking the passing tests. Verbosity is not virtue. A 3-line fix
  is correct if it resolves the issue; a 50-line fix is incorrect if it
  doesn't.

Do not penalize a patch for being short. Do not reward a patch for being
long. Length correlates with neither correctness nor robustness.

ISSUE:
{problem_statement}

PROPOSED PATCH:
{patch}

ADVERSARIAL PROBE (required, exactly one):
Identify the specific failure mode the issue describes. State the smallest
input or call sequence that, under the original code, would exhibit that
failure. Then state what the patched code would do on that same input.
The probe must target the failure described in the ISSUE — do not invent
edge cases the issue does not mention.

STRUCTURAL FLAGS:
Set each of these to true or false. A patch is sound iff all flags are true.

  - addresses_root_cause: the patch modifies the code path that produces
    the failure described in the issue, not a downstream symptom.
  - preserves_existing_behavior: nothing in the patch changes behavior on
    inputs unrelated to the failure mode (no signature changes that break
    other callers, no import removals, no broad refactors).
  - tests_actually_test_the_issue: under the patched code, the adversarial
    probe above produces the issue's described correct behavior.
  - no_unrelated_changes: every hunk in the diff is justified by the issue.
    Cosmetic edits, unrelated bug "fixes," and out-of-scope refactors fail
    this flag.

Respond in this EXACT JSON format — JSON only, no markdown fences, no extra text:
{{
  "flags": {{
    "addresses_root_cause": <true|false>,
    "preserves_existing_behavior": <true|false>,
    "tests_actually_test_the_issue": <true|false>,
    "no_unrelated_changes": <true|false>
  }},
  "probe": "<one sentence: input X under original code produces Y; under patch produces Z>",
  "sound": <true iff all four flags are true>,
  "reasoning": "<one sentence justifying the lowest-set flag, or 'all flags pass' if sound>",
  "red_flags": ["<flag_name>", ...],
  "score": <10 if sound else 0-5 reflecting how many flags failed>
}}

Example of a sound patch:
{{"flags": {{"addresses_root_cause": true, "preserves_existing_behavior": true, "tests_actually_test_the_issue": true, "no_unrelated_changes": true}}, "probe": "calling foo([]) raised IndexError under original code; patch returns [] cleanly.", "sound": true, "reasoning": "all flags pass", "red_flags": [], "score": 10}}

Example of an unsound patch (over-broad refactor):
{{"flags": {{"addresses_root_cause": true, "preserves_existing_behavior": false, "tests_actually_test_the_issue": true, "no_unrelated_changes": false}}, "probe": "calling parse('') returned None under original code; patch returns empty dict.", "sound": false, "reasoning": "patch also rewrites unrelated serialize() function, breaking its callers.", "red_flags": ["preserves_existing_behavior", "no_unrelated_changes"], "score": 4}}"""


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

        # New per-flag schema: derive sound from AND of all required flags.
        # Reject as parse-error if the flags dict is missing any required name
        # — this protects against the old 0-10-only schema slipping through
        # and being silently treated as "sound" with no flag voting data.
        flags = result.get("flags") if isinstance(result, dict) else None
        if not isinstance(flags, dict) or not all(name in flags for name in FLAG_NAMES):
            return {
                "score": 0,
                "sound": False,
                "reasoning": "critic response missing required flags dict",
                "red_flags": ["parse_error"],
                "flags": {},
            }
        # Coerce to bools (some models emit "true"/"false" strings).
        coerced = {n: bool(flags[n]) if not isinstance(flags[n], str)
                   else flags[n].strip().lower() == "true"
                   for n in FLAG_NAMES}
        result["flags"] = coerced
        result["sound"] = all(coerced.values())
        # red_flags becomes the list of failed flag names; preserves any
        # non-flag entries the model added (e.g. "signature_mismatch").
        failed = [n for n, v in coerced.items() if not v]
        existing_extras = [
            r for r in (result.get("red_flags") or [])
            if isinstance(r, str) and r not in FLAG_NAMES
        ]
        result["red_flags"] = failed + existing_extras
        # Normalize score: if sound, 10; else proportional to flag failures.
        result["score"] = 10 if result["sound"] else max(0, 5 - len(failed))
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

    # Per-flag 80% aggregation: for each named structural flag, count how many
    # critics voted true and require >= FLAG_PASS_THRESHOLD * M agreement.
    # Proposal survives iff EVERY flag clears its threshold. Critics that hit
    # a structural failure (empty/refusal/parse_error/critic_error) implicitly
    # vote false on every flag, since they produced no usable signal — this
    # is the conservative interpretation and biases toward filtering when
    # critic responses degrade.
    #
    # Safety net: if every critic returned a structural failure (the proposal
    # is unevaluable rather than low-quality), defer the decision to the
    # tournament — see filter_proposals fall-through.
    STRUCTURAL_FLAGS = {"empty_response", "refusal", "parse_error", "critic_error"}
    unevaluable = [
        s for s in scores if set(s.get("red_flags") or []) & STRUCTURAL_FLAGS
    ]
    all_unevaluable = bool(scores) and len(unevaluable) == len(scores)

    flag_votes = {name: 0 for name in FLAG_NAMES}
    for s in scores:
        flags = s.get("flags") or {}
        for name in FLAG_NAMES:
            if bool(flags.get(name, False)):
                flag_votes[name] += 1

    denom = max(len(scores), 1)
    flag_pass = {
        name: (flag_votes[name] / denom) >= FLAG_PASS_THRESHOLD
        for name in FLAG_NAMES
    }
    avg_score = sum(s["score"] for s in scores) / len(scores) if scores else 0

    proposal["critic_scores"] = scores
    proposal["critic_avg"] = avg_score
    proposal["critic_flag_votes"] = flag_votes      # raw vote count per flag
    proposal["critic_flag_pass"] = flag_pass        # which flags cleared 80%
    proposal["critic_sound"] = all(flag_pass.values()) and not all_unevaluable

    # Filter unless either (a) all flags pass per-critic 80% gate, or
    # (b) every critic was structurally unevaluable (let tournament decide
    # via filter_proposals' fallthrough).
    if all_unevaluable:
        proposal["filtered"] = False
        proposal["filter_reason"] = "all_unevaluable_pass_through"
    elif all(flag_pass.values()):
        proposal["filtered"] = False
        proposal["filter_reason"] = "all_flags_pass_threshold"
    else:
        failed_flags = [n for n, ok in flag_pass.items() if not ok]
        proposal["filtered"] = True
        proposal["filter_reason"] = f"flags_below_threshold:{','.join(failed_flags)}"

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

    # Safety: if EVERY proposal failed the critic gate (whether due to
    # structural critic failures or per-flag agreement falling below the
    # 80% threshold for at least one flag on every proposal), pass them
    # all through to the tournament. We need a candidate for the comparator
    # stage; the tournament's pairwise comparison is the residual signal.
    if not survivors and evaluated:
        print(f"  [Critics] No proposal cleared the per-flag gate — passing all through; tournament decides")
        for p in evaluated:
            p["filtered"] = False
            p["filter_reason"] = "all_filtered_pass_through"
        survivors = evaluated
        filtered = []

    print(f"  [Critics] Result: {len(survivors)} survivors, {len(filtered)} filtered (structural-flag-only)")

    return survivors, filtered
