"""Ceiling experiment: K=8 patches, M=15 critic votes/patch, R=10 comparator votes/matchup.

Stage 1 (all 500 instances): critic votes only. Cached at
    outputs/stage1_critics_full/critic_votes.jsonl (1 row per (iid, patch_idx))

Stage 2 (50 random pilot instances): comparator votes only. Cached at
    outputs/stage2_comparator_pilot/comparator_votes.jsonl (1 row per (iid, a, b))

Per-call cost telemetry:
    outputs/stage{1,2}_*/usage.jsonl (1 row per API call)

Both stages share asyncio.Semaphore(N) — default N=10. Reasoning effort is
forced to "high". $1800 hard-halt across both stages.

Run modes:
    python3 run_ceiling.py --smoke 5    # First 5 instances, writes to outputs/smoke/
    python3 run_ceiling.py --full       # Stage 1 (500) + Stage 2 (50 pilot)
    python3 run_ceiling.py --stage1     # Just Stage 1
    python3 run_ceiling.py --stage2     # Just Stage 2 (requires Stage 1 done for pilot iids)

Resumability: both JSONL outputs are append-only and idempotent. On restart,
existing (iid, patch_idx) and (iid, a, b) keys are skipped. The token usage
and resume keys persist across restarts so $1800 budget enforcement works.

Required env (refuses to start if missing):
    AZURE_API_KEY
    AZURE_API_BASE
    AZURE_DEPLOYMENT       (e.g. "gpt-5.4-nano")
    AZURE_REASONING_EFFORT (forced to "high" if unset)
    AZURE_API_VERSION      (default: 2025-03-01-preview when reasoning is on)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------
# Constants & paths
# ----------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent          # /home/vsun/ogpt-slim-qwen
DATA_REPO = Path("/home/vsun/orchestra-gpt")    # patches + resolve bools live here

PATCH_DIR = DATA_REPO / "outputs" / "oracle_preds_v3"
EVAL_RUN_BASE = DATA_REPO / "logs" / "run_evaluation"

# Experiment scale
K_PATCHES = 8                  # use proposers 0..7
M_CRITIC_VOTES = 15            # critic votes per patch
R_COMPARATOR_VOTES = 10        # comparator votes per matchup
PILOT_INSTANCE_COUNT = 50      # Stage 2 instance subset
GLOBAL_SEED = 42               # for prompt-pool draws and pilot-instance sampling

# Concurrency / budget / timeouts
# CLAUDE.md cap of ≤48 is Docker-scoped (workloads that spawn containers).
# This script is pure async HTTP — at sem=100 it used 0.79 GB RAM and 0.19
# cores out of 1160 GB / 256 cores (i.e. <0.1% of the box). Host is not
# remotely the limit; Azure deployment TPM/RPM is. User authorized
# CONCURRENCY=256 on 2026-05-04 to find the actual Azure ceiling
# ("lets do it until we take under 250 gb ram and less than 48 cores" —
# host can't get near those limits at any reasonable sem). Script's
# exponential 429 backoff handles sustained-throttle scenarios.
CONCURRENCY = 256
HARD_BUDGET_USD = 1800.00
PROGRESS_EVERY = 50            # write progress.txt every N completed instances

# Pricing (Azure gpt-5.4-nano, $/1M tokens). reasoning_tokens are billed as output.
PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60

# JSON-parse retry policy
PARSE_RETRY_MAX = 3
PARSE_RETRY_TEMP_BUMP = 0.2

# Output token caps (validated upstream in critics.py / comparators.py)
CRITIC_MAX_OUTPUT = 6144
COMPARATOR_MAX_OUTPUT = 8192

# Stage output paths
SMOKE_DIR = REPO / "outputs" / "smoke"
STAGE1_DIR = REPO / "outputs" / "stage1_critics_full"
STAGE2_DIR = REPO / "outputs" / "stage2_comparator_pilot"

# Per-instance smoke-test cost halt thresholds (20% over the budgeted target)
SMOKE_CRITIC_HALT_USD = 1.80      # budget target $1.50/inst
SMOKE_COMPARATOR_HALT_USD = 4.80  # budget target $4.00/inst

# ----------------------------------------------------------------------------
# Pool imports
# ----------------------------------------------------------------------------

sys.path.insert(0, str(REPO))
from orchestra.critic_prompts import CRITIC_POOL  # noqa: E402
from orchestra.comparator_prompts import COMPARATOR_POOL  # noqa: E402

# ----------------------------------------------------------------------------
# Env / client setup
# ----------------------------------------------------------------------------


CONFIG_FALLBACK_PATH = "/home/vsun/orchestra-gpt"


def _try_load_from_config_fallback() -> bool:
    """If Azure env vars aren't already set, attempt to import them from
    /home/vsun/orchestra-gpt/config.py — that file's import side-effect sets
    os.environ for AZURE_API_KEY / AZURE_API_BASE etc. (lines 15-17). This
    avoids requiring the user to manually export the same values that are
    already hardcoded in the existing repo's config. Returns True if the
    fallback set at least one variable.

    NOTE: config.py:12 picks AZURE_API_VERSION based on AZURE_REASONING_EFFORT
    at import time. We force it BEFORE importing so config.py picks the
    Responses-API-compatible version (2025-03-01-preview)."""
    if all(os.getenv(k) for k in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_DEPLOYMENT")):
        return False
    # Pre-set so config.py's import-time logic picks the right API version.
    os.environ.setdefault("AZURE_REASONING_EFFORT", "high")
    try:
        sys.path.insert(0, CONFIG_FALLBACK_PATH)
        import config as _orchestra_gpt_config
        # importing config.py sets os.environ for AZURE_API_KEY/AZURE_API_BASE/
        # AZURE_API_VERSION (config.py:15-17), but AZURE_DEPLOYMENT goes to
        # AZURE_DEPLOYMENT_NAME (line 18). Pull the deployment from the module
        # attr directly.
        if not os.getenv("AZURE_DEPLOYMENT") and getattr(
            _orchestra_gpt_config, "AZURE_DEPLOYMENT", None
        ):
            os.environ["AZURE_DEPLOYMENT"] = _orchestra_gpt_config.AZURE_DEPLOYMENT
        # Force the Responses-API-compatible version regardless of how
        # config.py decided — we always use Responses API in this script.
        os.environ["AZURE_API_VERSION"] = "2025-03-01-preview"
        sys.path.pop(0)
        return True
    except Exception as e:
        sys.stderr.write(
            f"WARNING: tried to fall back to {CONFIG_FALLBACK_PATH}/config.py "
            f"but import failed: {e}\n"
        )
        return False


def _require_env() -> dict:
    """Refuse to start if Azure creds are missing.

    Resolution order:
      1. Direct env vars (AZURE_API_KEY, AZURE_API_BASE, AZURE_DEPLOYMENT).
      2. Fallback: import /home/vsun/orchestra-gpt/config.py whose import
         side-effect populates os.environ. Authorized by user (explicit).
    """
    if not all(os.getenv(k) for k in ("AZURE_API_KEY", "AZURE_API_BASE",
                                       "AZURE_DEPLOYMENT")):
        if _try_load_from_config_fallback():
            print("[env] loaded Azure creds from "
                  f"{CONFIG_FALLBACK_PATH}/config.py fallback")
    missing = []
    for k in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_DEPLOYMENT"):
        if not os.getenv(k):
            missing.append(k)
    if missing:
        sys.stderr.write(
            f"FATAL: missing required env vars: {missing}.\n"
            f"export AZURE_API_KEY=... AZURE_API_BASE=... AZURE_DEPLOYMENT=...\n"
            f"(or ensure {CONFIG_FALLBACK_PATH}/config.py is importable)\n"
        )
        sys.exit(2)
    # Force reasoning=high (brief: 'high. Not xhigh. Not medium.')
    os.environ["AZURE_REASONING_EFFORT"] = "high"
    os.environ.setdefault("AZURE_API_VERSION", "2025-03-01-preview")
    return {
        "api_key": os.environ["AZURE_API_KEY"],
        "azure_endpoint": os.environ["AZURE_API_BASE"],
        "api_version": os.environ["AZURE_API_VERSION"],
        "deployment": os.environ["AZURE_DEPLOYMENT"],
    }


def _make_client(env: dict):
    """Build the AsyncAzureOpenAI client. Imported lazily to make --help work
    even without the openai package set up correctly."""
    from openai import AsyncAzureOpenAI
    return AsyncAzureOpenAI(
        api_key=env["api_key"],
        azure_endpoint=env["azure_endpoint"],
        api_version=env["api_version"],
    )


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------


def load_patches() -> tuple[dict[str, list[str]], list[str]]:
    """Returns (patches_by_iid_then_k, sorted_instance_ids).

    patches_by_iid_then_k[iid] is a list of length K_PATCHES; each element
    is the model_patch string (possibly empty)."""
    by_iid: dict[str, list[str]] = {}
    for k in range(K_PATCHES):
        f = PATCH_DIR / f"proposer_{k}.jsonl"
        if not f.exists():
            sys.stderr.write(f"FATAL: missing patches file {f}\n")
            sys.exit(2)
        with f.open() as fh:
            for line in fh:
                row = json.loads(line)
                iid = row["instance_id"]
                if iid not in by_iid:
                    by_iid[iid] = [""] * K_PATCHES
                by_iid[iid][k] = row.get("model_patch", "") or ""

    iids = sorted(by_iid.keys())
    return by_iid, iids


def load_resolve_bools() -> dict[str, dict[int, bool]]:
    """Aggregate resolve booleans across all oracle_*_proposer_K eval runs.
    Returns {iid: {k: bool}} (missing means never evaluated)."""
    resolved: dict[str, dict[int, bool]] = {}
    for k in range(K_PATCHES):
        for run_dir in EVAL_RUN_BASE.glob(f"oracle_*_proposer_{k}"):
            for model_dir in run_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for inst_dir in model_dir.iterdir():
                    if not inst_dir.is_dir():
                        continue
                    rep = inst_dir / "report.json"
                    if not rep.exists():
                        continue
                    try:
                        with rep.open() as fh:
                            j = json.load(fh)
                    except Exception:
                        continue
                    iid = inst_dir.name
                    rec = j.get(iid, j) if isinstance(j, dict) else {}
                    if not isinstance(rec, dict):
                        continue
                    val = bool(rec.get("resolved", False))
                    # Last write wins per (iid, k); state_snapshot.py uses
                    # same semantics.
                    resolved.setdefault(iid, {})[k] = val
    return resolved


def load_problem_statements(iids: list[str]) -> dict[str, str]:
    """Load problem_statement for each instance from SWE-Bench_Verified.

    We use the local datasets cache; no network call needed if the dataset
    has been hydrated before. If unavailable, fall back to empty strings
    (which will degrade critic/comparator quality but not break parsing)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
        wanted = set(iids)
        out: dict[str, str] = {}
        for inst in ds:
            iid = inst["instance_id"]
            if iid in wanted:
                out[iid] = inst.get("problem_statement", "") or ""
        return out
    except Exception as e:
        sys.stderr.write(
            f"WARNING: could not load problem statements ({e}). "
            f"Falling back to empty strings.\n"
        )
        return {iid: "" for iid in iids}


# ----------------------------------------------------------------------------
# Random draw helpers (per-call deterministic, reproducible)
# ----------------------------------------------------------------------------


def _draw_seed(*parts) -> int:
    """Stable hash of arbitrary parts → int seed for random.Random."""
    h = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return int(h[:16], 16) ^ GLOBAL_SEED


def draw_critic(iid: str, patch_idx: int, vote_idx: int) -> dict:
    rng = random.Random(_draw_seed("crit", iid, patch_idx, vote_idx))
    return rng.choice(CRITIC_POOL)


def draw_comparator(iid: str, a_idx: int, b_idx: int, vote_idx: int) -> dict:
    rng = random.Random(_draw_seed("cmp", iid, a_idx, b_idx, vote_idx))
    return rng.choice(COMPARATOR_POOL)


def pilot_instance_subset(iids: list[str]) -> list[str]:
    rng = random.Random(_draw_seed("pilot"))
    if len(iids) <= PILOT_INSTANCE_COUNT:
        return list(iids)
    return sorted(rng.sample(iids, PILOT_INSTANCE_COUNT))


# ----------------------------------------------------------------------------
# Resume state
# ----------------------------------------------------------------------------


def load_done_critic_keys(jsonl: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not jsonl.exists():
        return done
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["instance_id"], int(r["patch_idx"])))
    return done


def load_done_comparator_keys(jsonl: Path) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not jsonl.exists():
        return done
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["instance_id"], int(r["patch_a_idx"]), int(r["patch_b_idx"])))
    return done


# ----------------------------------------------------------------------------
# API call wrappers (async)
# ----------------------------------------------------------------------------


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int          # includes reasoning
    reasoning_tokens: int        # subset of output_tokens
    visible_tokens: int          # output - reasoning
    error: str | None = None


def _extract_text(response) -> str:
    """Pull assistant text out of a Responses API object."""
    obj = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    chunks: list[str] = []
    for item in obj.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                chunks.append(c.get("text", ""))
    return "".join(chunks)


def _extract_usage(response) -> tuple[int, int, int]:
    """Returns (input_tokens, output_tokens, reasoning_tokens)."""
    obj = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    u = obj.get("usage", {}) or {}
    inp = int(u.get("input_tokens", 0))
    out = int(u.get("output_tokens", 0))
    rt = 0
    details = u.get("output_tokens_details") or {}
    if isinstance(details, dict):
        rt = int(details.get("reasoning_tokens", 0))
    return inp, out, rt


async def _api_call(
    client,
    deployment: str,
    prompt: str,
    *,
    temperature: float,  # informational only — see note below
    max_output_tokens: int,
    sem: asyncio.Semaphore,
    timeout_s: int = 600,
) -> CallResult:
    """One Responses API call, retried on transient errors. Returns CallResult.

    Azure rejects `temperature` on Responses-API calls with reasoning enabled
    (per /home/vsun/orchestra-gpt/azure_api_guide.ipynb). The pool-entry
    `temperature` is therefore preserved for logging/reproducibility only —
    not passed to the API. Stochasticity comes from the reasoning model's
    native sampling instead."""
    backoff = 1.0
    max_attempts = 5
    last_err: str | None = None
    for attempt in range(max_attempts):
        async with sem:
            try:
                response = await asyncio.wait_for(
                    client.responses.create(
                        model=deployment,
                        input=[{"role": "user", "content": prompt}],
                        max_output_tokens=max_output_tokens,
                        reasoning={"effort": os.environ.get("AZURE_REASONING_EFFORT", "high")},
                    ),
                    timeout=timeout_s,
                )
                text = _extract_text(response)
                inp, out, rt = _extract_usage(response)
                return CallResult(
                    text=text,
                    input_tokens=inp,
                    output_tokens=out,
                    reasoning_tokens=rt,
                    visible_tokens=max(0, out - rt),
                )
            except Exception as e:
                msg = str(e)
                last_err = f"{type(e).__name__}: {msg[:300]}"
                low = msg.lower()
                is_429 = "429" in low or "rate" in low
                is_5xx = any(c in low for c in ("500", "502", "503", "504"))
                is_conn = "connection" in low or "timeout" in low or "timed out" in low
                if attempt < max_attempts - 1 and (is_429 or is_5xx or is_conn):
                    delay = min(60.0, backoff * (2**attempt))
                    await asyncio.sleep(delay)
                    continue
                break
    return CallResult(text="", input_tokens=0, output_tokens=0,
                      reasoning_tokens=0, visible_tokens=0, error=last_err)


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # ```json ... ```
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t


def parse_critic(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(_strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    flags = obj.get("flags")
    if not isinstance(flags, dict):
        return None
    needed = {
        "addresses_root_cause",
        "preserves_existing_behavior",
        "tests_actually_test_the_issue",
        "no_unrelated_changes",
    }
    if not needed.issubset(flags):
        return None
    coerced = {k: bool(flags.get(k)) for k in needed}
    return {
        "flags": coerced,
        "probe": (obj.get("probe") or "")[:1000],
        "reasoning": (obj.get("reasoning") or "")[:1000],
    }


def parse_comparator(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(_strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "winner" not in obj:
        return None
    winner = obj.get("winner")
    if winner not in ("A", "B", "TIE"):
        return None
    return {
        "hypothesis": (obj.get("hypothesis") or "")[:500],
        "a_changes": (obj.get("a_changes") or "")[:500],
        "b_changes": (obj.get("b_changes") or "")[:500],
        "a_consistent": bool(obj.get("a_consistent", False)),
        "b_consistent": bool(obj.get("b_consistent", False)),
        "a_collateral": bool(obj.get("a_collateral", False)),
        "b_collateral": bool(obj.get("b_collateral", False)),
        "winner": winner,
        "confidence": int(obj.get("confidence", 0) or 0),
        "reasoning": (obj.get("reasoning") or "")[:500],
    }


# ----------------------------------------------------------------------------
# Cost / budget / progress
# ----------------------------------------------------------------------------


@dataclass
class Budget:
    spent_usd: float = 0.0
    n_calls: int = 0
    n_critic: int = 0
    n_comparator: int = 0
    abstains_critic: int = 0
    abstains_comparator: int = 0
    halted: bool = False
    # Real-time throughput tracking — issued counts calls that have entered
    # _api_call; completion_times records when each call finishes.
    issued: int = 0
    completion_times: deque = field(default_factory=lambda: deque(maxlen=400))

    def add(self, kind: str, result: CallResult) -> None:
        cost = (
            result.input_tokens * PRICE_INPUT_PER_M / 1e6
            + result.output_tokens * PRICE_OUTPUT_PER_M / 1e6
        )
        self.spent_usd += cost
        self.n_calls += 1
        self.completion_times.append(time.time())
        if kind == "critic":
            self.n_critic += 1
        else:
            self.n_comparator += 1

    def check_halt(self) -> bool:
        if self.spent_usd >= HARD_BUDGET_USD:
            self.halted = True
        return self.halted

    def in_flight(self) -> int:
        return max(0, self.issued - self.n_calls)

    def recent_rate_per_min(self, window_s: float = 60.0) -> float:
        """Calls/min completed in the last `window_s` seconds."""
        if not self.completion_times:
            return 0.0
        now = time.time()
        # Deque is append-only ordered; walk from the right until older than window.
        n = 0
        for t in reversed(self.completion_times):
            if now - t > window_s:
                break
            n += 1
        return n * 60.0 / window_s


async def heartbeat_task(budget: Budget, interval_s: float = 30.0) -> None:
    """Periodic stdout line so we can see in-flight + throughput separately
    from the per-call usage.jsonl log (which only writes on completion)."""
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        rate = budget.recent_rate_per_min()
        print(
            f"[heartbeat] issued={budget.issued} completed={budget.n_calls} "
            f"in_flight={budget.in_flight()} errors=? "
            f"rate={rate:.1f}/min cost=${budget.spent_usd:.3f} "
            f"abstains: crit={budget.abstains_critic} cmp={budget.abstains_comparator}",
            flush=True,
        )


def write_progress(out_dir: Path, budget: Budget, instances_done: int,
                   total_instances: int, t_start: float, stage: str) -> None:
    elapsed = time.time() - t_start
    rate = instances_done / max(elapsed, 1e-6)
    eta_s = (total_instances - instances_done) / rate if rate > 0 else float("inf")
    progress = out_dir / "progress.txt"
    with progress.open("w") as fh:
        fh.write(
            f"stage={stage}\n"
            f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"instances_done={instances_done}/{total_instances}\n"
            f"calls_made={budget.n_calls} (critic={budget.n_critic}, "
            f"comparator={budget.n_comparator})\n"
            f"abstains=critic:{budget.abstains_critic}, "
            f"comparator:{budget.abstains_comparator}\n"
            f"running_cost_usd={budget.spent_usd:.4f}/${HARD_BUDGET_USD:.2f}\n"
            f"elapsed_s={elapsed:.1f}\n"
            f"eta_s={eta_s:.1f}\n"
        )


def append_usage(out_dir: Path, kind: str, key: dict, result: CallResult,
                 prompt_name: str, temperature: float) -> None:
    f = out_dir / "usage.jsonl"
    cost = (
        result.input_tokens * PRICE_INPUT_PER_M / 1e6
        + result.output_tokens * PRICE_OUTPUT_PER_M / 1e6
    )
    rec = {
        "ts": time.time(),
        "kind": kind,
        "prompt_name": prompt_name,
        "temperature": temperature,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "visible_tokens": result.visible_tokens,
        "cost_usd": cost,
        "error": result.error,
        **key,
    }
    with f.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------
# Stage 1: critic ceiling
# ----------------------------------------------------------------------------


async def _do_one_critic_vote(
    client, deployment: str, sem, problem: str, patch: str,
    iid: str, patch_idx: int, vote_idx: int, out_dir: Path, budget: Budget,
) -> dict:
    variant = draw_critic(iid, patch_idx, vote_idx)
    base_temp = float(variant["temperature"])
    prompt = variant["template"].format(problem_statement=problem[:2000],
                                        patch=patch[:8000])
    parsed = None
    last_text = ""
    last_result: CallResult | None = None
    for attempt in range(PARSE_RETRY_MAX):
        temp = min(1.5, base_temp + attempt * PARSE_RETRY_TEMP_BUMP)
        budget.issued += 1
        result = await _api_call(client, deployment, prompt,
                                 temperature=temp,
                                 max_output_tokens=CRITIC_MAX_OUTPUT,
                                 sem=sem)
        budget.add("critic", result)
        append_usage(out_dir, "critic",
                     {"iid": iid, "patch_idx": patch_idx, "vote_idx": vote_idx,
                      "attempt": attempt},
                     result, variant["name"], temp)
        last_result = result
        last_text = result.text
        if result.error:
            continue
        parsed = parse_critic(result.text)
        if parsed is not None:
            return {
                "critic_idx": vote_idx,
                "prompt_name": variant["name"],
                "temperature": base_temp,
                "flags": parsed["flags"],
                "probe": parsed["probe"],
                "reasoning": parsed["reasoning"],
                "raw_response": result.text[:4000],
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_tokens": result.reasoning_tokens,
            }
    # Abstain — log and return null flags
    budget.abstains_critic += 1
    return {
        "critic_idx": vote_idx,
        "prompt_name": variant["name"],
        "temperature": base_temp,
        "flags": None,
        "probe": "",
        "reasoning": "",
        "raw_response": last_text[:4000],
        "input_tokens": last_result.input_tokens if last_result else 0,
        "output_tokens": last_result.output_tokens if last_result else 0,
        "reasoning_tokens": last_result.reasoning_tokens if last_result else 0,
        "abstain_reason": (last_result.error if last_result and last_result.error
                           else "parse_failed"),
    }


async def stage1_run(
    client, deployment: str, instances: list[str],
    patches_by_iid: dict[str, list[str]],
    resolve_by_iid: dict[str, dict[int, bool]],
    problems_by_iid: dict[str, str],
    out_dir: Path, budget: Budget, total_label: str,
) -> None:
    """Streaming Stage 1: build coroutines for ALL (iid, patch, vote) triples
    upfront and process completions via asyncio.as_completed. Eliminates the
    per-instance drain that capped effective concurrency at the slowest patch
    in the current instance — workers now pull continuously across instance
    boundaries, keeping sem=CONCURRENCY saturated until the very end."""
    out_dir.mkdir(parents=True, exist_ok=True)
    crit_jsonl = out_dir / "critic_votes.jsonl"
    done_keys = load_done_critic_keys(crit_jsonl)
    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.time()
    total_inst = len(instances)
    print(f"[stage1] resuming with {len(done_keys)} critic-rows already cached")

    # Pre-write empty-patch rows synchronously (no API calls needed) and build
    # the global coroutine list for non-empty patches.
    task_keys: list[tuple[str, int, int]] = []  # (iid, k, vi)
    coros: list = []
    pk_resolves: dict[tuple[str, int], Any] = {}      # (iid, k) -> bool|None
    instance_pending_pks: dict[str, int] = {}          # iid -> count of pending patches
    for iid in instances:
        problem = problems_by_iid.get(iid, "")
        patches = patches_by_iid.get(iid, [""] * K_PATCHES)
        for k in range(K_PATCHES):
            if (iid, k) in done_keys:
                continue
            patch = patches[k] or ""
            patch_resolves = resolve_by_iid.get(iid, {}).get(k, None)
            if not patch.strip():
                row = {
                    "instance_id": iid,
                    "patch_idx": k,
                    "patch_resolves": patch_resolves,
                    "critic_votes": [
                        {"critic_idx": vi, "prompt_name": None,
                         "temperature": None, "flags": None,
                         "abstain_reason": "empty_patch"}
                        for vi in range(M_CRITIC_VOTES)
                    ],
                }
                with crit_jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done_keys.add((iid, k))
                continue
            pk_resolves[(iid, k)] = patch_resolves
            instance_pending_pks[iid] = instance_pending_pks.get(iid, 0) + 1
            for vi in range(M_CRITIC_VOTES):
                task_keys.append((iid, k, vi))
                coros.append(
                    _do_one_critic_vote(
                        client, deployment, sem, problem, patch,
                        iid, k, vi, out_dir, budget,
                    )
                )

    # Total instances that need at least one API call (others were either
    # empty-patch-only or already cached).
    instances_with_work = len(instance_pending_pks)
    instances_no_work = total_inst - instances_with_work
    instances_done = instances_no_work  # already complete
    print(f"[stage1] {len(coros)} critic API calls queued across "
          f"{instances_with_work} instances ({instances_no_work} fully cached)")

    if not coros:
        write_progress(out_dir, budget, instances_done, total_inst,
                       t_start, "stage1")
        return

    async def _keyed(key, coro):
        return key, await coro
    futures = [asyncio.ensure_future(_keyed(k, c))
               for k, c in zip(task_keys, coros)]

    # Per-(iid, k) accumulator: vi -> vote_dict. Row writes when len == M.
    pending_results: dict[tuple[str, int], dict[int, dict]] = {}
    halted = False
    completions = 0
    for fut in asyncio.as_completed(futures):
        if budget.check_halt() and not halted:
            print(f"[stage1] HALT: budget ${budget.spent_usd:.2f} ≥ ${HARD_BUDGET_USD}")
            halted = True
            for f in futures:
                if not f.done():
                    f.cancel()
        try:
            (iid, k, vi), vote = await fut
        except asyncio.CancelledError:
            continue
        bucket = pending_results.setdefault((iid, k), {})
        bucket[vi] = vote
        if len(bucket) == M_CRITIC_VOTES:
            votes = [bucket[v] for v in range(M_CRITIC_VOTES)]
            row = {
                "instance_id": iid,
                "patch_idx": k,
                "patch_resolves": pk_resolves[(iid, k)],
                "critic_votes": votes,
            }
            with crit_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_keys.add((iid, k))
            del pending_results[(iid, k)]
            instance_pending_pks[iid] -= 1
            if instance_pending_pks[iid] == 0:
                instances_done += 1
                if instances_done % PROGRESS_EVERY == 0 or instances_done == total_inst:
                    write_progress(out_dir, budget, instances_done, total_inst,
                                   t_start, "stage1")
                    print(f"[stage1] {instances_done}/{total_inst}  "
                          f"calls={budget.n_calls}  spent=${budget.spent_usd:.2f}  "
                          f"abstains={budget.abstains_critic}")
        completions += 1

    write_progress(out_dir, budget, instances_done, total_inst, t_start, "stage1")


# ----------------------------------------------------------------------------
# Lazy survivor computation (between Stage 1 and Stage 2)
# ----------------------------------------------------------------------------


FLAG_KEYS = (
    "addresses_root_cause",
    "preserves_existing_behavior",
    "tests_actually_test_the_issue",
    "no_unrelated_changes",
)


def _patch_survives(votes, tau):
    counts = {k: [0, 0] for k in FLAG_KEYS}
    for v in votes:
        flags = v.get("flags")
        if flags is None:
            continue
        for k in FLAG_KEYS:
            counts[k][1] += 1
            if flags.get(k) is True:
                counts[k][0] += 1
    for k, (t, n) in counts.items():
        if n == 0:
            return False
        if t / n < tau:
            return False
    return True


def compute_lazy_survivors(critic_jsonl: Path, target_iids: list[str]
                           ) -> tuple[dict[str, set[int]], float]:
    """After Stage 1 cache is populated, sweep τ on the cached critic votes to
    find τ* (max resolve rate over instances), then return per-instance
    survivor sets at τ*.

    Returns ({iid: {patch_idx, ...}}, tau_star). If critic_jsonl is missing or
    empty, returns ({}, 0.0) — caller falls back to full round-robin.
    """
    if not critic_jsonl.exists():
        return {}, 0.0
    by_iid: dict[str, list[tuple[int, bool | None, list[dict]]]] = {}
    with critic_jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            iid = r["instance_id"]
            by_iid.setdefault(iid, []).append(
                (int(r["patch_idx"]), r.get("patch_resolves"), r.get("critic_votes", []))
            )
    if not by_iid:
        return {}, 0.0
    taus = [round(0.05 * i, 2) for i in range(1, 21)]
    best_tau = 0.5
    best_resolved = -1
    for tau in taus:
        n_resolved = 0
        for iid, recs in by_iid.items():
            for k, resolves, votes in recs:
                if resolves and _patch_survives(votes, tau):
                    n_resolved += 1
                    break
        if n_resolved > best_resolved:
            best_resolved = n_resolved
            best_tau = tau
    survivors: dict[str, set[int]] = {}
    for iid in target_iids:
        if iid not in by_iid:
            continue
        survivors[iid] = {
            k for (k, _, votes) in by_iid[iid]
            if _patch_survives(votes, best_tau)
        }
    return survivors, best_tau


# ----------------------------------------------------------------------------
# Stage 2: comparator ceiling (on pilot subset)
# ----------------------------------------------------------------------------


async def _one_comparator_call(
    client, deployment: str, sem, problem: str, patch_a: str, patch_b: str,
    iid: str, a_idx: int, b_idx: int, vote_idx: int, swap: int,
    out_dir: Path, budget: Budget,
) -> tuple[str | None, CallResult]:
    """One comparator API call (one position-swap). Returns (winner-letter, raw).

    swap=1: A=patch_a, B=patch_b → returned letter is "A" or "B" or "TIE"
    swap=2: A=patch_b, B=patch_a → returned "A" actually means patch_b won;
            caller maps back.
    """
    variant = draw_comparator(iid, a_idx, b_idx, vote_idx)
    base_temp = float(variant["temperature"])
    if swap == 1:
        first, second = patch_a, patch_b
    else:
        first, second = patch_b, patch_a
    prompt = variant["template"].format(
        problem_statement=problem[:2000],
        patch_a=first[:8000],
        patch_b=second[:8000],
    )
    parsed = None
    last_result: CallResult | None = None
    for attempt in range(PARSE_RETRY_MAX):
        temp = min(1.5, base_temp + attempt * PARSE_RETRY_TEMP_BUMP)
        budget.issued += 1
        result = await _api_call(client, deployment, prompt,
                                 temperature=temp,
                                 max_output_tokens=COMPARATOR_MAX_OUTPUT,
                                 sem=sem)
        budget.add("comparator", result)
        append_usage(out_dir, "comparator",
                     {"iid": iid, "a_idx": a_idx, "b_idx": b_idx,
                      "vote_idx": vote_idx, "swap": swap, "attempt": attempt},
                     result, variant["name"], temp)
        last_result = result
        if result.error:
            continue
        parsed = parse_comparator(result.text)
        if parsed is not None:
            return parsed["winner"], result
    budget.abstains_comparator += 1
    return None, last_result or CallResult("", 0, 0, 0, 0, "no_call")


async def _do_one_comparator_vote(
    client, deployment: str, sem, problem: str, patch_a: str, patch_b: str,
    iid: str, a_idx: int, b_idx: int, vote_idx: int,
    out_dir: Path, budget: Budget,
) -> dict:
    variant = draw_comparator(iid, a_idx, b_idx, vote_idx)
    # Two position-swap calls
    w1, _ = await _one_comparator_call(
        client, deployment, sem, problem, patch_a, patch_b,
        iid, a_idx, b_idx, vote_idx, 1, out_dir, budget,
    )
    w2_raw, _ = await _one_comparator_call(
        client, deployment, sem, problem, patch_a, patch_b,
        iid, a_idx, b_idx, vote_idx, 2, out_dir, budget,
    )
    # Map swap-2's winner: in swap-2 A=patch_b, B=patch_a, so:
    #   "A" → patch_b → caller's "B"
    #   "B" → patch_a → caller's "A"
    #   "TIE" → "TIE"
    if w2_raw is None:
        w2 = None
    elif w2_raw == "A":
        w2 = "B"
    elif w2_raw == "B":
        w2 = "A"
    else:
        w2 = "TIE"
    # Comparator picks A only if BOTH calls say A; same for B; else TIE.
    if w1 == "A" and w2 == "A":
        winner = "A"
    elif w1 == "B" and w2 == "B":
        winner = "B"
    else:
        winner = "TIE"
    return {
        "comparator_idx": vote_idx,
        "prompt_name": variant["name"],
        "temperature": float(variant["temperature"]),
        "position_swap_1": w1,
        "position_swap_2": w2,
        "matchup_winner": winner,
    }


async def stage2_run(
    client, deployment: str, pilot_iids: list[str],
    patches_by_iid: dict[str, list[str]],
    problems_by_iid: dict[str, str],
    out_dir: Path, budget: Budget,
    survivors_by_iid: dict[str, set[int]] | None = None,
) -> None:
    """Run Stage 2 comparator ceiling. If survivors_by_iid is provided,
    only run matchups where both patch indices survived the Stage 1 gate
    (lazy round-robin) — saves substantial cost. If None, runs all
    C(K,2) matchups (full round-robin)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_jsonl = out_dir / "comparator_votes.jsonl"
    done_keys = load_done_comparator_keys(cmp_jsonl)
    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.time()
    total_inst = len(pilot_iids)
    lazy = survivors_by_iid is not None
    if lazy:
        n_match = sum(
            len(list(itertools.combinations(sorted(survivors_by_iid.get(i, set())), 2)))
            for i in pilot_iids
        )
        print(f"[stage2] LAZY mode: {n_match} survivor-pair matchups across "
              f"{total_inst} pilot instances "
              f"(vs {total_inst * len(list(itertools.combinations(range(K_PATCHES), 2)))} "
              f"if full round-robin)")
    print(f"[stage2] resuming with {len(done_keys)} comparator-rows already cached")

    # Streaming refactor: build coroutines for ALL (iid, a, b, vi) tuples
    # upfront and process via asyncio.as_completed. Eliminates per-instance
    # drain barriers; workers stay saturated across instance boundaries.
    task_keys: list[tuple[str, int, int, int]] = []  # (iid, a, b, vi)
    coros: list = []
    matchup_pending: dict[tuple[str, int, int], int] = {}  # (iid, a, b) -> R count
    instance_pending_matchups: dict[str, int] = {}
    for iid in pilot_iids:
        problem = problems_by_iid.get(iid, "")
        patches = patches_by_iid.get(iid, [""] * K_PATCHES)
        if lazy:
            survivors = sorted(survivors_by_iid.get(iid, set()))
            pairs = list(itertools.combinations(survivors, 2))
        else:
            pairs = list(itertools.combinations(range(K_PATCHES), 2))
        for a, b in pairs:
            if (iid, a, b) in done_keys:
                continue
            pa, pb = patches[a] or "", patches[b] or ""
            if not pa.strip() or not pb.strip():
                row = {
                    "instance_id": iid,
                    "patch_a_idx": a,
                    "patch_b_idx": b,
                    "comparator_votes": [
                        {"comparator_idx": vi, "prompt_name": None,
                         "matchup_winner": "TIE",
                         "abstain_reason": "empty_patch"}
                        for vi in range(R_COMPARATOR_VOTES)
                    ],
                }
                with cmp_jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done_keys.add((iid, a, b))
                continue
            matchup_pending[(iid, a, b)] = R_COMPARATOR_VOTES
            instance_pending_matchups[iid] = (
                instance_pending_matchups.get(iid, 0) + 1
            )
            for vi in range(R_COMPARATOR_VOTES):
                task_keys.append((iid, a, b, vi))
                coros.append(
                    _do_one_comparator_vote(
                        client, deployment, sem, problem, pa, pb,
                        iid, a, b, vi, out_dir, budget,
                    )
                )

    instances_with_work = len(instance_pending_matchups)
    instances_no_work = total_inst - instances_with_work
    instances_done = instances_no_work
    print(f"[stage2] {len(coros)} comparator API calls queued across "
          f"{instances_with_work} instances ({instances_no_work} "
          f"fully cached or no survivor pairs)")

    if not coros:
        write_progress(out_dir, budget, instances_done, total_inst,
                       t_start, "stage2")
        return

    async def _keyed(key, coro):
        return key, await coro
    futures = [asyncio.ensure_future(_keyed(k, c))
               for k, c in zip(task_keys, coros)]

    pending_results: dict[tuple[str, int, int], dict[int, dict]] = {}
    halted = False
    for fut in asyncio.as_completed(futures):
        if budget.check_halt() and not halted:
            print(f"[stage2] HALT: budget ${budget.spent_usd:.2f} ≥ ${HARD_BUDGET_USD}")
            halted = True
            for f in futures:
                if not f.done():
                    f.cancel()
        try:
            (iid, a, b, vi), vote = await fut
        except asyncio.CancelledError:
            continue
        bucket = pending_results.setdefault((iid, a, b), {})
        bucket[vi] = vote
        if len(bucket) == R_COMPARATOR_VOTES:
            votes = [bucket[v] for v in range(R_COMPARATOR_VOTES)]
            row = {
                "instance_id": iid,
                "patch_a_idx": a,
                "patch_b_idx": b,
                "comparator_votes": votes,
            }
            with cmp_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_keys.add((iid, a, b))
            del pending_results[(iid, a, b)]
            instance_pending_matchups[iid] -= 1
            if instance_pending_matchups[iid] == 0:
                instances_done += 1
                pe = max(1, PROGRESS_EVERY // 5)
                if instances_done % pe == 0 or instances_done == total_inst:
                    write_progress(out_dir, budget, instances_done, total_inst,
                                   t_start, "stage2")
                    print(f"[stage2] {instances_done}/{total_inst}  "
                          f"calls={budget.n_calls}  spent=${budget.spent_usd:.2f}  "
                          f"abstains_cmp={budget.abstains_comparator}")

    write_progress(out_dir, budget, instances_done, total_inst, t_start, "stage2")


# ----------------------------------------------------------------------------
# Smoke test cost-per-instance halt
# ----------------------------------------------------------------------------


def smoke_cost_check(out_dir: Path, n_smoke: int) -> tuple[bool, str]:
    """Read usage.jsonl and check $/instance against the smoke halts."""
    usage = out_dir / "usage.jsonl"
    if not usage.exists():
        return True, "no usage data"
    crit_total = 0.0
    cmp_total = 0.0
    with usage.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "critic":
                crit_total += r.get("cost_usd", 0.0)
            elif r.get("kind") == "comparator":
                cmp_total += r.get("cost_usd", 0.0)
    crit_per = crit_total / max(1, n_smoke)
    cmp_per = cmp_total / max(1, n_smoke)
    msg = (f"smoke per-instance: critic ${crit_per:.3f} "
           f"(halt>${SMOKE_CRITIC_HALT_USD}) | "
           f"comparator ${cmp_per:.3f} (halt>${SMOKE_COMPARATOR_HALT_USD})")
    if crit_per > SMOKE_CRITIC_HALT_USD or cmp_per > SMOKE_COMPARATOR_HALT_USD:
        return False, msg + " — HALT"
    return True, msg


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


async def amain(args) -> int:
    env = _require_env()
    client = _make_client(env)

    print("[load] patches…")
    patches_by_iid, all_iids = load_patches()
    print(f"[load] {len(all_iids)} instances, K={K_PATCHES} per instance")

    print("[load] resolve booleans…")
    resolve_by_iid = load_resolve_bools()
    n_eval = sum(1 for iid in all_iids if iid in resolve_by_iid)
    print(f"[load] resolve bools available for {n_eval}/{len(all_iids)} instances")

    print("[load] problem statements…")
    problems_by_iid = load_problem_statements(all_iids)
    n_prob = sum(1 for iid in all_iids if problems_by_iid.get(iid))
    print(f"[load] problem statements for {n_prob}/{len(all_iids)} instances")

    pilot = pilot_instance_subset(all_iids)
    print(f"[load] pilot subset (Stage 2): {len(pilot)} instances")

    budget = Budget()
    hb = asyncio.create_task(heartbeat_task(budget, interval_s=30.0))

    if args.smoke:
        n = int(args.smoke)
        smoke_iids = all_iids[:n]
        print(f"[smoke] {n} instances: {smoke_iids[:3]}...")
        await stage1_run(client, env["deployment"], smoke_iids,
                         patches_by_iid, resolve_by_iid, problems_by_iid,
                         SMOKE_DIR, budget, "smoke-stage1")
        ok, msg = smoke_cost_check(SMOKE_DIR, n)
        print(f"[smoke] cost check: {msg}")
        if not ok:
            print("[smoke] critic per-instance exceeds halt threshold; "
                  "skipping comparator smoke. Inspect outputs/smoke/.")
            hb.cancel()
            return 1
        survivors, tau_smoke = compute_lazy_survivors(
            SMOKE_DIR / "critic_votes.jsonl", smoke_iids
        )
        n_pairs = sum(
            len(list(itertools.combinations(sorted(survivors.get(i, set())), 2)))
            for i in smoke_iids
        )
        print(f"[smoke] lazy τ*={tau_smoke}, "
              f"survivor-pair matchups: {n_pairs} (across {n} smoke instances)")
        await stage2_run(client, env["deployment"], smoke_iids,
                         patches_by_iid, problems_by_iid,
                         SMOKE_DIR, budget, survivors_by_iid=survivors)
        ok, msg = smoke_cost_check(SMOKE_DIR, n)
        print(f"[smoke] post-comparator cost check: {msg}")
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        return 0 if ok else 1

    if args.full or args.stage1:
        await stage1_run(client, env["deployment"], all_iids,
                         patches_by_iid, resolve_by_iid, problems_by_iid,
                         STAGE1_DIR, budget, "stage1-full")

    if args.full or args.stage2:
        if not budget.halted:
            survivors, tau_used = compute_lazy_survivors(
                STAGE1_DIR / "critic_votes.jsonl", pilot
            )
            n_pairs = sum(
                len(list(itertools.combinations(sorted(survivors.get(i, set())), 2)))
                for i in pilot
            )
            print(f"[stage2-lazy] τ*={tau_used} on Stage 1 cache, "
                  f"{n_pairs} survivor-pair matchups across {len(pilot)} pilot")
            await stage2_run(client, env["deployment"], pilot,
                             patches_by_iid, problems_by_iid,
                             STAGE2_DIR, budget,
                             survivors_by_iid=survivors)

    hb.cancel()
    try:
        await hb
    except asyncio.CancelledError:
        pass
    print(f"[done] calls={budget.n_calls} spent=${budget.spent_usd:.2f} "
          f"abstains: critic={budget.abstains_critic} "
          f"comparator={budget.abstains_comparator}")
    return 0


def main() -> int:
    # All globals we mutate from CLI args, declared once up front.
    global CRITIC_POOL, COMPARATOR_POOL
    global K_PATCHES, M_CRITIC_VOTES, R_COMPARATOR_VOTES
    global SMOKE_DIR, STAGE1_DIR, STAGE2_DIR

    # Parse --legacy-prompts BEFORE we import pools so we can swap them.
    # Legacy pool files load templates via regex on their source siblings
    # (critics.py / comparators.py) WITHOUT triggering `from config import …`,
    # so no module-cache conflict.
    if "--legacy-prompts" in sys.argv:
        from orchestra.critic_prompts_legacy import CRITIC_POOL as _CP
        from orchestra.comparator_prompts_legacy import COMPARATOR_POOL as _MP
        CRITIC_POOL = _CP
        COMPARATOR_POOL = _MP
        print(f"[cfg] using LEGACY single-prompt pools "
              f"(critic={CRITIC_POOL[0]['name']}, comparator={COMPARATOR_POOL[0]['name']})")

    p = argparse.ArgumentParser()
    p.add_argument("--smoke", type=int, default=0,
                   help="Smoke test on first N instances (writes to outputs/smoke/)")
    p.add_argument("--full", action="store_true",
                   help="Run Stage 1 (500) + Stage 2 (50 pilot)")
    p.add_argument("--stage1", action="store_true",
                   help="Run Stage 1 only")
    p.add_argument("--stage2", action="store_true",
                   help="Run Stage 2 only")
    p.add_argument("--legacy-prompts", action="store_true",
                   help="Use single committed prompts from orchestra/critics.py "
                        "+ comparators.py (no diversity pool). Pre-parsed above.")
    p.add_argument("--k", type=int, default=K_PATCHES,
                   help=f"Patches per instance to use (default {K_PATCHES}, max 8)")
    p.add_argument("--m", type=int, default=M_CRITIC_VOTES,
                   help=f"Critic votes per patch (default {M_CRITIC_VOTES})")
    p.add_argument("--r", type=int, default=R_COMPARATOR_VOTES,
                   help=f"Comparator votes per matchup (default {R_COMPARATOR_VOTES})")
    p.add_argument("--output-suffix", default="",
                   help="Suffix appended to output dir names (e.g. '_legacy_k5m3r3')")
    args = p.parse_args()
    if not (args.smoke or args.full or args.stage1 or args.stage2):
        p.print_help()
        return 2
    # Override module-level globals with CLI args
    K_PATCHES = args.k
    M_CRITIC_VOTES = args.m
    R_COMPARATOR_VOTES = args.r
    if args.output_suffix:
        SMOKE_DIR = REPO / "outputs" / f"smoke{args.output_suffix}"
        STAGE1_DIR = REPO / "outputs" / f"stage1_critics_full{args.output_suffix}"
        STAGE2_DIR = REPO / "outputs" / f"stage2_comparator_pilot{args.output_suffix}"
    print(f"[cfg] K={K_PATCHES} M={M_CRITIC_VOTES} R={R_COMPARATOR_VOTES} "
          f"out_suffix='{args.output_suffix}'")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
