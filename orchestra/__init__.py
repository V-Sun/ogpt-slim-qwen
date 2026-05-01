"""Orchestra-GPT: Multi-agent committee system for SWE-bench

Implements the Π_{k,m,r} protocol:
- K proposers generate independent patches
- m critics filter unsound proposals
- r-round tournament selects Copeland winner
"""

__version__ = "1.0.0"
