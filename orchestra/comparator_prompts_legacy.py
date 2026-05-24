"""Single-prompt comparator pool — uses the committed COMPARATOR_PROMPT_TEMPLATE
from orchestra/comparators.py.

Loads via regex on source (see critic_prompts_legacy.py for rationale).
"""

import os
import re

_COMPARATORS_PATH = os.path.join(os.path.dirname(__file__), "comparators.py")
with open(_COMPARATORS_PATH) as _f:
    _src = _f.read()
_match = re.search(r'COMPARATOR_PROMPT_TEMPLATE\s*=\s*"""(.*?)"""', _src, re.DOTALL)
if not _match:
    raise RuntimeError(
        f"Could not extract COMPARATOR_PROMPT_TEMPLATE from {_COMPARATORS_PATH}"
    )
COMPARATOR_PROMPT_TEMPLATE = _match.group(1)

COMPARATOR_POOL = [
    {
        "name": "legacy_committed",
        "temperature": 0.7,
        "template": COMPARATOR_PROMPT_TEMPLATE,
    },
]
