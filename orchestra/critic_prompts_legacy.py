"""Single-prompt critic pool — uses the committed CRITIC_PROMPT_TEMPLATE
from orchestra/critics.py (the prompt that produced the prior K=5/M=3/R=3
=73.9% baseline).

Loads the template by reading orchestra/critics.py source and extracting the
triple-quoted constant via regex — avoids the `from orchestra.critics import …`
chain that pulls in `from config import MODEL_PROVIDER`, which conflicts when
the orchestra-gpt config module is swapped in for credential resolution.
"""

import os
import re

_CRITICS_PATH = os.path.join(os.path.dirname(__file__), "critics.py")
with open(_CRITICS_PATH) as _f:
    _src = _f.read()
_match = re.search(r'CRITIC_PROMPT_TEMPLATE\s*=\s*"""(.*?)"""', _src, re.DOTALL)
if not _match:
    raise RuntimeError(
        f"Could not extract CRITIC_PROMPT_TEMPLATE from {_CRITICS_PATH}"
    )
CRITIC_PROMPT_TEMPLATE = _match.group(1)

CRITIC_POOL = [
    {
        "name": "legacy_committed",
        "temperature": 0.7,
        "template": CRITIC_PROMPT_TEMPLATE,
    },
]
