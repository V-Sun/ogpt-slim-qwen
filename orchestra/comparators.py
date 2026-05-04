"""Pairwise comparisons and Copeland winner selection.

From paper: comparator edge σ ≥ α0/2 under D1 condition.
Position-swap debiasing: run A vs B AND B vs A to cancel lead bias.
"""

import json
import re
import os
import random
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestra.llm import call_text_model
from orchestra.rate_limiter import get_api_semaphore

# Environment variables are set in config.py

COMPARATOR_PROMPT_TEMPLATE = """You are deciding which of two patches is more likely to make the failing
tests pass without breaking the passing tests. That is the only question
you are answering. Aesthetics, ambition, and visible effort are not relevant.

Do NOT select a patch because it makes more changes, addresses more cases,
or appears more thorough. The patch that minimally resolves the stated
issue is preferred. A 4kb patch that fixes the bug beats an 11kb patch
that doesn't.

ISSUE:
{problem_statement}

PATCH A:
{patch_a}

PATCH B:
{patch_b}

REQUIRED STRUCTURAL COMPARISON (fill these in BEFORE deciding):

  1. Failing test hypothesis: state the smallest hypothesis about what is
     wrong, derived from the issue text alone. One sentence. This is the
     ground truth you compare both patches against.

  2. A_changes: list the specific lines/functions Patch A modifies.
  3. B_changes: list the specific lines/functions Patch B modifies.

  4. A_consistent_with_hypothesis: do A's changes plausibly cause the
     failing test in the issue to start passing? (true/false + one-line
     justification)
  5. B_consistent_with_hypothesis: same question for B.

  6. A_collateral: does A change behavior on inputs unrelated to the
     failure mode? (true/false + one-line justification)
  7. B_collateral: same question for B.

The decision falls out of this comparison; do not pull a winner from prior.
If exactly one patch is consistent with the hypothesis and the other is
not, that one wins. If both are consistent, prefer the one with less
collateral. If both fail the hypothesis, output TIE. If they are
functionally equivalent (same lines changed differently, or different
lines that produce the same behavior), output TIE.

Respond in this EXACT JSON format — JSON only, no markdown fences, no extra text:
{{
  "hypothesis": "<one sentence>",
  "a_changes": "<files/functions modified by A>",
  "b_changes": "<files/functions modified by B>",
  "a_consistent": <true|false>,
  "b_consistent": <true|false>,
  "a_collateral": <true|false>,
  "b_collateral": <true|false>,
  "winner": "A" | "B" | "TIE",
  "confidence": <integer 1-5>,
  "reasoning": "<one sentence: why the winner falls out of the structural comparison>"
}}

Example where minimal correct beats ambitious broken:
{{"hypothesis": "splitter.split() returns wrong order when n=0.", "a_changes": "fixes the boundary check in splitter.py:42 (3 lines).", "b_changes": "rewrites split() in splitter.py and adds a new helper class SplitContext with retry logic (87 lines).", "a_consistent": true, "b_consistent": false, "a_collateral": false, "b_collateral": true, "winner": "A", "confidence": 5, "reasoning": "A directly fixes the n=0 boundary; B's rewrite changes signatures other callers depend on and does not address the n=0 case."}}

Example of a tie:
{{"hypothesis": "format_date crashes on tz-naive datetimes.", "a_changes": "guards format_date() with isinstance check (4 lines).", "b_changes": "guards the same call site in render_template() instead (5 lines).", "a_consistent": true, "b_consistent": true, "a_collateral": false, "b_collateral": false, "winner": "TIE", "confidence": 3, "reasoning": "Both patches resolve the failure mode in different but equivalent locations; no structural reason to prefer one."}}"""


def compare_two_once(patch_a: str, patch_b: str, problem_statement: str,
                      model_name: str, label_a: str, label_b: str) -> str:
    """Single comparison between two patches.

    Args:
        patch_a: First patch
        patch_b: Second patch
        problem_statement: GitHub issue
        model_name: Model string
        label_a: Label for first patch ("A" or "B")
        label_b: Label for second patch ("A" or "B")

    Returns:
        Winner label (label_a, label_b, or "TIE")
    """
    prompt = COMPARATOR_PROMPT_TEMPLATE.format(
        problem_statement=problem_statement[:2000],
        patch_a=patch_a[:8000],
        patch_b=patch_b[:8000],
    )
    # Same starvation bug as critics, but worse: the comparator prompt holds
    # two patches (not one) plus a longer "reasoning + analysis" template,
    # so xhigh reasoning + visible JSON can exceed 6k tokens. 1400 → 6/6
    # empties in validation; 6144 → still 3/6 failures; 8192 holds steady.
    max_out = int(os.getenv("ORCHESTRA_COMPARATOR_MAX_OUTPUT_TOKENS", "8192"))

    text = ""
    try:
        semaphore = get_api_semaphore()
        with semaphore:
            text = call_text_model(model_name, prompt, max_output_tokens=max_out)
        text = (text or "").strip()

        # Empty / refusal: neither outcome can be trusted — fall back to
        # unbiased random tiebreak (which is what the bare-except branch
        # below also does, but we log the specific signal so we can track it).
        if not text:
            print(f"        Comparator empty response")
            return random.choice([label_a, label_b])
        lowered = text.lower()
        if ("i'm sorry" in lowered or "i am sorry" in lowered) and "cannot" in lowered:
            print(f"        Comparator refusal")
            return random.choice([label_a, label_b])

        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        if not text.startswith("{"):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)

        result = json.loads(text)
        raw_winner = result.get("winner", "A")

        if raw_winner == "TIE":
            return "TIE"
        return label_a if raw_winner == "A" else label_b

    except Exception as e:
        print(f"        Comparator error: {e} | head={text[:120]!r}" if text else f"        Comparator error: {e}")
        return random.choice([label_a, label_b])


def compare_two(patch_a: str, patch_b: str, problem_statement: str,
                 model_name: str) -> str:
    """Compare two patches with position-swap debiasing.

    Runs twice: A vs B, then B vs A (positions swapped).
    This cancels lead bias (LLMs favor whichever option comes first).

    Args:
        patch_a: First patch
        patch_b: Second patch
        problem_statement: GitHub issue
        model_name: Model string

    Returns:
        "A" if patch_a wins, "B" if patch_b wins
    """
    # Round 1: A vs B (A is first)
    winner_1 = compare_two_once(
        patch_a, patch_b, problem_statement, model_name, "A", "B"
    )

    # Round 2: B vs A (positions swapped — debiases lead preference)
    winner_2 = compare_two_once(
        patch_b, patch_a, problem_statement, model_name, "B", "A"
    )
    # compare_two_once already returns the correct label, no need to remap

    # Both rounds agree (including both TIE) → take that result
    if winner_1 == winner_2:
        return winner_1
    # One round is TIE, the other picked a winner → go with the non-tie pick
    if winner_1 == "TIE":
        return winner_2
    if winner_2 == "TIE":
        return winner_1
    # Both rounds disagree and neither is TIE → random tie-break (unbiased)
    return random.choice(["A", "B"])


def copeland_winner(survivors: list[dict], problem_statement: str,
                     r: int, model_name: str) -> dict:
    """Run round-robin tournament with r comparisons per pair.

    Copeland winner = candidate with most pairwise wins.

    From paper: Copeland winner is sound if no B2 event occurs
    (no unsound proposal beats sound one in pairwise comparison).

    Args:
        survivors: List of surviving proposals (post-critic filtering)
        problem_statement: GitHub issue
        r: Number of comparisons per pair (use odd number for majority)
        model_name: Model string

    Returns:
        Winning proposal dict
    """
    if len(survivors) == 1:
        print(f"  [Tournament] Only 1 survivor — automatic winner")
        return survivors[0]

    n = len(survivors)
    wins = [0.0] * n

    print(f"  [Tournament] Running round-robin on {n} survivors, r={r}...")

    # Parallelize all comparisons
    comparison_tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            for round_idx in range(r):
                comparison_tasks.append((i, j, round_idx))

    # Run all comparisons in parallel (rate-limited by semaphore)
    comparison_results = {}
    with ThreadPoolExecutor(max_workers=min(len(comparison_tasks), 100)) as executor:
        future_to_task = {
            executor.submit(
                compare_two,
                survivors[i]["model_patch"],
                survivors[j]["model_patch"],
                problem_statement,
                model_name,
            ): (i, j, round_idx)
            for i, j, round_idx in comparison_tasks
        }

        for future in as_completed(future_to_task):
            i, j, round_idx = future_to_task[future]
            try:
                winner_label = future.result()
                key = (i, j)
                if key not in comparison_results:
                    comparison_results[key] = 0
                if winner_label == "TIE":
                    tie_key = (i, j, "ties")
                    comparison_results[tie_key] = comparison_results.get(tie_key, 0) + 1
                elif winner_label == "A":
                    comparison_results[key] += 1
                # "B" wins are implicit: decisive rounds where A didn't win
            except Exception as e:
                print(f"      Comparison error ({i} vs {j}): {e}")

    # Tally wins based on majority votes; TIE gives 0.5 to each side
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            i_wins = comparison_results.get(key, 0)
            ties = comparison_results.get((i, j, "ties"), 0)
            decisive = r - ties

            if ties == r:
                # All rounds tied
                wins[i] += 0.5
                wins[j] += 0.5
            elif i_wins > decisive // 2:
                wins[i] += 1
            else:
                wins[j] += 1

    # Copeland winner: most pairwise wins (random tie-break if multiple max)
    max_wins = max(wins)
    best_indices = [i for i, w in enumerate(wins) if w == max_wins]
    best_idx = random.choice(best_indices)  # Random selection among tied winners
    winner = survivors[best_idx]

    print(f"  [Tournament] Winner is proposer {winner['k_index']} "
          f"with {wins[best_idx]}/{n-1} pairwise wins")

    return winner
