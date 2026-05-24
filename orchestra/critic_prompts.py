"""Diversity pool for critic prompts (M ≤ 15 random draws with replacement).

Eight variants distributed across five LLM-as-judge ensemble families:
  1. Persona diversity   (CODERPE-inspired)
  2. Focus diversity     (SE-Jury strategy decomposition)
  3. Reasoning style     (Crowd Comparative Reasoning)
  4. Adversarial framing (Auto-Prompt Ensemble)
  5. Temperature variation (CCR baseline)

Every variant emits the SAME JSON schema as orchestra/critics.py so the
existing parser (which derives sound/score/red_flags from flags) works
without branching:

  {
    "flags": {
      "addresses_root_cause": <bool>,
      "preserves_existing_behavior": <bool>,
      "tests_actually_test_the_issue": <bool>,
      "no_unrelated_changes": <bool>
    },
    "probe": "<one sentence>",
    "reasoning": "<one sentence>"
  }

Templates use {problem_statement} and {patch} placeholders, filled at call
time. The diversity randomizer (random.Random(42)) picks one variant per
critic vote.
"""


CRITIC_FLAG_INSTRUCTIONS = """
STRUCTURAL FLAGS:
Set each of these to true or false.

  - addresses_root_cause: the patch modifies the code path that produces
    the failure described in the issue, not a downstream symptom.
  - preserves_existing_behavior: nothing in the patch changes behavior on
    inputs unrelated to the failure mode.
  - tests_actually_test_the_issue: under the patched code, the adversarial
    probe below produces the issue's described correct behavior.
  - no_unrelated_changes: every hunk in the diff is justified by the issue.
    Cosmetic edits, unrelated bug "fixes," and out-of-scope refactors fail.

ADVERSARIAL PROBE (required, exactly one):
Identify the smallest input or call sequence that, under the original code,
exhibits the failure described in the issue. State what the patched code
does on that same input. Do not invent edge cases the issue does not mention.

Respond in this EXACT JSON format — JSON only, no markdown fences:
{{
  "flags": {{
    "addresses_root_cause": <true|false>,
    "preserves_existing_behavior": <true|false>,
    "tests_actually_test_the_issue": <true|false>,
    "no_unrelated_changes": <true|false>
  }},
  "probe": "<one sentence: input X under original code produces Y; under patch produces Z>",
  "reasoning": "<one sentence>"
}}

ISSUE:
{problem_statement}

PROPOSED PATCH:
{patch}
"""


CRITIC_POOL = [
    # === FAMILY 1: PERSONA DIVERSITY (CODERPE-inspired) ===
    {
        "name": "senior_engineer",
        "temperature": 0.7,
        "template": (
            "You are a senior software engineer reviewing a patch for a bug report. "
            "You have shipped production code for fifteen years and you have seen every flavor of "
            "drive-by 'fix' that introduces three new bugs. Your reputation rests on catching them. "
            "Apply this single test: a correct patch is the smallest change that makes the failing "
            "tests pass without breaking the passing tests. Verbosity is not virtue.\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },
    {
        "name": "test_driven_dev",
        "temperature": 0.7,
        "template": (
            "You are a test-driven development practitioner. The only question that matters to you "
            "is: 'does this patch make the failing test pass and leave the passing tests intact?' "
            "You do not care about code style. You do not care about hypothetical edge cases the "
            "issue does not describe. You care about test outcomes.\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },
    {
        "name": "minimalist_reviewer",
        "temperature": 0.7,
        "template": (
            "You are a minimalist code reviewer. Your guiding principle: the best patch changes the "
            "fewest lines. A 3-line fix that resolves the issue is correct. A 50-line refactor that "
            "resolves the same issue is suspect — it is doing extra work that may regress something. "
            "You actively penalize scope creep, defensive additions, and unrelated 'improvements.'\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },

    # === FAMILY 2: FOCUS DIVERSITY (SE-Jury strategy decomposition) ===
    {
        "name": "regression_focused",
        "temperature": 0.7,
        "template": (
            "You are evaluating a patch with one specific concern: regression risk. Your job is to "
            "determine whether this patch could break behavior on inputs unrelated to the failure "
            "mode described in the issue. Look for signature changes, modified return types, "
            "removed branches, and changes to shared utilities. The patch may correctly fix the "
            "reported bug while breaking other code paths — that is what you are looking for.\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },
    {
        "name": "root_cause_focused",
        "temperature": 0.7,
        "template": (
            "You are evaluating whether a patch addresses the root cause of the reported issue or "
            "merely papers over a downstream symptom. A patch that catches an exception at the API "
            "boundary when the bug is in a deep utility function is suppressing the symptom, not "
            "fixing the cause. Trace the failure path from the issue's described behavior back to "
            "the originating code, and determine whether the patch modifies that originating code.\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },

    # === FAMILY 3: REASONING STYLE DIVERSITY ===
    {
        "name": "counterexample_first",
        "temperature": 0.7,
        "template": (
            "You are evaluating a patch by trying to break it. Before deciding whether the patch is "
            "sound, attempt to construct a counterexample: an input or call sequence where the "
            "patched code produces wrong behavior. If you can construct one that is consistent with "
            "the issue's description, the patch fails. If you genuinely cannot construct one after "
            "trying, the patch likely passes.\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },

    # === FAMILY 4: ADVERSARIAL FRAMINGS (Auto-Prompt Ensemble) ===
    {
        "name": "assume_signature_break",
        "temperature": 0.7,
        "template": (
            "Assume this patch has a hidden signature mismatch or import error that would break "
            "downstream callers. Your job is to find evidence for this assumption. Examine every "
            "modified function signature, every removed import, every changed type annotation, and "
            "every callsite that the patch touches. Only declare the patch sound if you find no "
            "evidence after a thorough adversarial search.\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },

    # === FAMILY 5: TEMPERATURE VARIATION (CCR baseline) ===
    {
        "name": "structured_baseline_low_temp",
        "temperature": 0.2,
        "template": (
            "You are evaluating whether a patch resolves a GitHub issue. Apply ONE test: a correct "
            "patch is the smallest change that makes the failing tests pass without breaking the "
            "passing tests. Verbosity is not virtue.\n\n"
            + CRITIC_FLAG_INSTRUCTIONS
        ),
    },
]
