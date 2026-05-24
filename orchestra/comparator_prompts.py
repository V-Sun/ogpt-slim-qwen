"""Diversity pool for comparator prompts (R ≤ 10 random draws with replacement).

Five variants distributed across LLM-as-judge ensemble families:
  1. Persona — minimalist
  2. Focus  — test outcome
  3. Adversarial — assume one is broken
  4. Reasoning style — root cause first
  5. Temperature variation — same prompt, low temp

Every variant emits the SAME JSON schema as orchestra/comparators.py so the
existing parser works without branching:

  {
    "hypothesis": "<one sentence>",
    "a_changes": "<files/functions modified by A>",
    "b_changes": "<files/functions modified by B>",
    "a_consistent": <bool>,
    "b_consistent": <bool>,
    "a_collateral": <bool>,
    "b_collateral": <bool>,
    "winner": "A" | "B" | "TIE",
    "confidence": <int 1-5>,
    "reasoning": "<one sentence>"
  }

Templates use {problem_statement}, {patch_a}, {patch_b} placeholders, filled
at call time. Each comparator vote is TWO API calls (position-swap A=A,B=B
then A=B,B=A); the comparator's matchup verdict is the original letter only
if both calls agree on that letter, else TIE.
"""


COMPARATOR_STRUCTURED_INSTRUCTIONS = """
REQUIRED STRUCTURAL COMPARISON (fill these in BEFORE deciding):

  1. Failing test hypothesis: smallest hypothesis about what is wrong, derived
     from the issue text alone. One sentence. This is the ground truth you
     compare both patches against.

  2. A_changes: specific lines/functions Patch A modifies.
  3. B_changes: specific lines/functions Patch B modifies.

  4. A_consistent_with_hypothesis: do A's changes plausibly cause the failing
     test to start passing? (true/false)
  5. B_consistent_with_hypothesis: same for B.

  6. A_collateral: does A change behavior on inputs unrelated to the failure
     mode? (true/false)
  7. B_collateral: same for B.

The decision falls out of this comparison; do not pull a winner from prior.
If exactly one patch is consistent with the hypothesis, that one wins.
If both are consistent, prefer the one with less collateral.
If both fail the hypothesis, output TIE.
If they are functionally equivalent, output TIE.

Respond in this EXACT JSON format — JSON only, no markdown fences:
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
  "reasoning": "<one sentence>"
}}

ISSUE:
{problem_statement}

PATCH A:
{patch_a}

PATCH B:
{patch_b}
"""


COMPARATOR_POOL = [
    # === FAMILY 1: PERSONA — minimalist ===
    {
        "name": "minimal_correct",
        "temperature": 0.7,
        "template": (
            "You are deciding which of two patches more minimally and correctly resolves a GitHub "
            "issue. A 4kb patch that fixes the bug beats an 11kb patch that doesn't. Aesthetics, "
            "ambition, and visible effort are not relevant. Do NOT select a patch because it makes "
            "more changes, addresses more cases, or appears more thorough.\n\n"
            + COMPARATOR_STRUCTURED_INSTRUCTIONS
        ),
    },

    # === FAMILY 2: FOCUS — test pass/fail framing ===
    {
        "name": "test_outcome_focused",
        "temperature": 0.7,
        "template": (
            "You are deciding which of two patches is more likely to make the failing tests pass "
            "without breaking the passing tests. That is the only question. The patch with cleaner "
            "code, better naming, or more comments does not win on those merits — it wins only if "
            "it is more likely to produce the correct test outcome.\n\n"
            + COMPARATOR_STRUCTURED_INSTRUCTIONS
        ),
    },

    # === FAMILY 3: ADVERSARIAL — assume one is broken ===
    {
        "name": "assume_one_broken",
        "temperature": 0.7,
        "template": (
            "Assume one of these two patches is broken in a non-obvious way: a signature mismatch, "
            "a missing import, a regression on a related code path, or a misread of the issue. Your "
            "job is to identify which one is broken. If you cannot, output TIE. Apply heavy "
            "skepticism to the more ambitious patch first.\n\n"
            + COMPARATOR_STRUCTURED_INSTRUCTIONS
        ),
    },

    # === FAMILY 4: REASONING STYLE — root cause first ===
    {
        "name": "root_cause_first",
        "temperature": 0.7,
        "template": (
            "Before evaluating either patch, derive the root cause of the issue from the issue text "
            "alone. State it as one sentence. Then evaluate each patch by whether it modifies the "
            "code that contains that root cause, vs whether it papers over a symptom further "
            "downstream. The patch that addresses the root cause wins, even if its diff is larger "
            "than the symptom-suppressing patch.\n\n"
            + COMPARATOR_STRUCTURED_INSTRUCTIONS
        ),
    },

    # === FAMILY 5: TEMPERATURE — same prompt, low temp ===
    {
        "name": "structured_baseline_low_temp",
        "temperature": 0.2,
        "template": (
            "You are deciding which of two patches more minimally and correctly resolves a GitHub "
            "issue. A 4kb patch that fixes the bug beats an 11kb patch that doesn't. Aesthetics, "
            "ambition, and visible effort are not relevant.\n\n"
            + COMPARATOR_STRUCTURED_INSTRUCTIONS
        ),
    },
]
