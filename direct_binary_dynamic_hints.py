"""Direct-binary classifier with dynamic hints.

Augments the static-hints prompt with:
  - FAIL_TO_PASS test names (same as static hints)
  - Optional legacy cached-eval trace extraction when explicitly enabled
  - TEST_PATCH from SWE-Bench dataset (shows the test code itself)
  - Files modified by patch (from diff parse)

Default mode is pre-eval blind: the model never reads candidate harness reports
or cached post-patch test outputs before selecting a patch.

Test-output extraction strategy:
  Legacy/non-blind mode: for each instance, find ANY proposer eval dir where the FAIL_TO_PASS test
  is in `tests_status.FAIL_TO_PASS.failure` (i.e., the patch didn't fix the
  bug, so the test fails with essentially the unpatched-code error). Read
  test_output.txt and pull the FAIL/ERROR block for the target test.
  This is not valid for blind patch selection claims unless the trace was
  produced independently before candidate selection.

Source-line extraction was deferred — would require git checkout per
(repo, base_commit), out of scope for this run. Test-patch serves as a
strong proxy for "the code that defines what must pass."

Caches:
  outputs/direct_binary_dynamic_hints_smoke/ (5-instance pre-flight)
  outputs/direct_binary_dynamic_hints_full500/ (full run)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ceiling as rc

REPO = Path(__file__).resolve().parent
SMOKE_DIR = REPO / "outputs" / "direct_binary_dynamic_hints_smoke"
FULL500_DIR = REPO / "outputs" / "direct_binary_dynamic_hints_full500"

EVAL_BASE = Path("/home/vsun/orchestra-gpt/logs/run_evaluation")

K_PATCHES = 8
GREEDY8_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
R_VOTES = 5
VOTE_START_INDEX = 0
CONCURRENCY = rc.CONCURRENCY
PARSE_RETRY_MAX = rc.PARSE_RETRY_MAX
PARSE_RETRY_TEMP_BUMP = rc.PARSE_RETRY_TEMP_BUMP
BINARY_MAX_OUTPUT = 4096
HARD_BUDGET_USD = 250.00  # Galanti-approved expanded budget

# ----------------------------------------------------------------------------
# Hint-loading
# ----------------------------------------------------------------------------


def load_dataset_fields(iids: list[str]) -> dict[str, dict]:
    """Load per-instance fields needed for dynamic hints: FAIL_TO_PASS,
    test_patch, base_commit. Returns {iid: {ftp, test_patch, base_commit, repo}}."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    except Exception as e:
        sys.stderr.write(f"FATAL: dataset load failed: {e}\n")
        sys.exit(2)
    wanted = set(iids)
    out: dict[str, dict] = {}
    for inst in ds:
        iid = inst["instance_id"]
        if iid not in wanted:
            continue
        ftp_raw = inst.get("FAIL_TO_PASS", "[]")
        try:
            ftp = json.loads(ftp_raw) if isinstance(ftp_raw, str) else list(ftp_raw)
        except Exception:
            ftp = []
        out[iid] = {
            "ftp": [n for n in ftp if isinstance(n, str)],
            "test_patch": inst.get("test_patch", "") or "",
            "base_commit": inst.get("base_commit", "") or "",
            "repo": inst.get("repo", "") or "",
        }
    return out


# ----------------------------------------------------------------------------
# Test-output error extraction
# ----------------------------------------------------------------------------


_FAIL_BLOCK_RE = re.compile(
    r'^(=+\s*\n(?:FAIL|ERROR):\s*(.+?)\n[-=]+\n)(.*?)(?=\n[-=]{20,}\n|\Z)',
    re.MULTILINE | re.DOTALL,
)
# pytest-style alternative
_PYTEST_FAIL_RE = re.compile(
    r'^_+\s+(.+?)\s+_+\n(.*?)(?=\n^_+|\Z)',
    re.MULTILINE | re.DOTALL,
)


def _index_failing_eval_dirs() -> dict[str, list[Path]]:
    """Build {iid: [eval_dir, ...]} where each eval_dir has a non-resolving
    report and a test_output.txt that may contain the failing test trace.
    Caches in-memory for the lifetime of one run."""
    by_iid: dict[str, list[Path]] = defaultdict(list)
    for rep in EVAL_BASE.glob("oracle_*_proposer_*/*/*/report.json"):
        try:
            with rep.open() as fh:
                j = json.load(fh)
        except Exception:
            continue
        iid = rep.parent.name
        rec = j.get(iid, j) if isinstance(j, dict) else {}
        if not isinstance(rec, dict):
            continue
        if rec.get("resolved", True):
            continue  # We want NON-resolving evals (test still fails)
        ts = rec.get("tests_status", {}) or {}
        failures = (ts.get("FAIL_TO_PASS", {}) or {}).get("failure", []) or []
        if not failures:
            continue
        test_out = rep.parent / "test_output.txt"
        if not test_out.exists():
            continue
        by_iid[iid].append(rep.parent)
    return by_iid


def extract_test_failures(test_out_path: Path, target_tests: list[str],
                          max_chars: int = 3000) -> str:
    """Extract the FAIL/ERROR blocks for the target tests from a
    test_output.txt file. Returns a concatenated string capped at max_chars."""
    try:
        text = test_out_path.read_text(errors="replace")
    except Exception:
        return ""
    # Narrow to the test-output region between markers, if present.
    start = text.find(">>>>> Start Test Output")
    end = text.find(">>>>> End Test Output")
    if start != -1 and end != -1 and end > start:
        text = text[start:end]
    blocks = []
    # unittest-style: ====...\nFAIL: <name>\n----...\n<traceback>
    for m in _FAIL_BLOCK_RE.finditer(text):
        header, name, body = m.group(1), m.group(2).strip(), m.group(3)
        # Match if any target test name is a substring of the block name
        if any(t.split("::")[-1].split("(")[0].strip() in name or
               name.split("(")[0].strip() in t
               for t in target_tests):
            block = (header + body).strip()
            blocks.append(block)
    # pytest-style: ___ test_name ___ ... (collect short blocks too)
    if not blocks:
        for m in _PYTEST_FAIL_RE.finditer(text):
            name, body = m.group(1).strip(), m.group(2)
            if any(t.split("::")[-1].split("(")[0].strip() in name or
                   name in t for t in target_tests):
                block = f"___ {name} ___\n{body}".strip()
                blocks.append(block)
    if not blocks:
        # Fallback: grep for traceback blocks generically; cap to last N lines
        # to avoid the huge conda boilerplate at the start.
        tail = "\n".join(text.splitlines()[-200:])
        if "Traceback" in tail or "FAIL" in tail or "ERROR" in tail:
            blocks.append(tail.strip())
    combined = "\n\n".join(blocks)
    if len(combined) > max_chars:
        # Keep the head (where the error type + assertion typically live)
        # and tail (where pytest summary often is).
        head = combined[: max_chars - 500]
        tail = combined[-500:]
        combined = head + "\n... [trimmed " + str(len(combined) - max_chars) + " chars] ...\n" + tail
    return combined


def get_dynamic_hint_for_iid(iid: str, target_tests: list[str],
                             eval_index: dict) -> tuple[str, Path | None]:
    """Returns (extracted_error_block, source_eval_dir) or ('', None)."""
    candidates = eval_index.get(iid, [])
    for d in candidates:
        out = extract_test_failures(d / "test_output.txt", target_tests)
        if out:
            return out, d
    return "", None


# ----------------------------------------------------------------------------
# Patch file-list parsing (same as static hints)
# ----------------------------------------------------------------------------


_DIFF_FILE_RE = re.compile(r'^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)', re.MULTILINE)


def patch_files(patch_text: str, max_files: int = 5) -> list[str]:
    if not patch_text:
        return []
    seen, seen_set = [], set()
    for m in _DIFF_FILE_RE.finditer(patch_text):
        path = m.group(1)
        if path in ("/dev/null", "") or path in seen_set:
            continue
        seen.append(path)
        seen_set.add(path)
        if len(seen) >= max_files:
            break
    return seen


# ----------------------------------------------------------------------------
# Augmented prompt
# ----------------------------------------------------------------------------


DYNAMIC_BINARY_PROMPT = """You are evaluating a single patch against the GitHub issue it claims to fix.
Output yes if and only if running the patched code passes the failing test
and does not break any test that previously passed.

Below you have:
  - The issue text
  - The exact failing test names (FAIL_TO_PASS)
  - The actual error trace from running the failing test on the (essentially)
    unpatched code — this is what the test outputs when the bug is present
  - The test source code that defines what "passing" means (test_patch)
  - The list of files the proposed patch modifies
  - The proposed patch itself

Your job: determine whether the proposed patch's edits resolve the actual
error shown in the trace. Trace the error frame from the trace; check the
patch modifies the code path producing that error; check the patch's logic
would make the failing assertion pass.

ISSUE:
{problem_statement}

FAILING TESTS (must pass after patch):
{fail_to_pass}

ACTUAL ERROR TRACE FROM RUNNING THE FAILING TEST (unpatched / non-resolving run):
{error_trace}

TEST SOURCE (test_patch — defines what a passing patch must achieve):
{test_patch}

FILES MODIFIED BY PROPOSED PATCH:
{patch_files}

PROPOSED PATCH:
{patch}

Respond with EXACTLY this JSON, no markdown fences, no extra text:
{{
  "resolves": <true|false>,
  "reasoning": "<one sentence: link the trace's error origin to the patched code>",
  "confidence": <integer 1-5>
}}
"""


def parse_binary(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(rc._strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "resolves" not in obj or obj["resolves"] not in (True, False):
        return None
    return {
        "resolves": bool(obj["resolves"]),
        "reasoning": (obj.get("reasoning") or "")[:500],
        "confidence": int(obj.get("confidence", 0) or 0),
    }


# ----------------------------------------------------------------------------
# Per-vote driver (mostly identical to direct_binary_hints.py)
# ----------------------------------------------------------------------------


async def _do_one_vote(
    client, deployment, sem,
    problem, patch, ftp_str, error_trace, test_patch_str, files_str,
    iid, k, vote_idx, out_dir, budget,
):
    base_temp = 0.7
    prompt = DYNAMIC_BINARY_PROMPT.format(
        problem_statement=problem[:2000],
        fail_to_pass=ftp_str,
        error_trace=error_trace[:3000],
        test_patch=test_patch_str[:2500],
        patch_files=files_str,
        patch=patch[:8000],
    )
    last_text = ""
    last_result = None
    for attempt in range(PARSE_RETRY_MAX):
        temp = min(1.5, base_temp + attempt * PARSE_RETRY_TEMP_BUMP)
        budget.issued += 1
        result = await rc._api_call(
            client, deployment, prompt,
            temperature=temp, max_output_tokens=BINARY_MAX_OUTPUT, sem=sem,
        )
        budget.add("critic", result)
        rc.append_usage(out_dir, "critic",
                        {"iid": iid, "patch_idx": k, "vote_idx": vote_idx,
                         "attempt": attempt},
                        result, "direct_binary_dynamic_hints", temp)
        last_result, last_text = result, result.text
        if result.error:
            continue
        parsed = parse_binary(result.text)
        if parsed is not None:
            return {
                "vote_idx": vote_idx,
                "temperature": base_temp,
                "resolves": parsed["resolves"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "raw_response": result.text[:2000],
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_tokens": result.reasoning_tokens,
            }
    budget.abstains_critic += 1
    return {
        "vote_idx": vote_idx,
        "temperature": base_temp,
        "resolves": None,
        "confidence": 0,
        "reasoning": "",
        "raw_response": last_text[:2000],
        "abstain_reason": (last_result.error if last_result and last_result.error
                           else "parse_failed"),
        "input_tokens": last_result.input_tokens if last_result else 0,
        "output_tokens": last_result.output_tokens if last_result else 0,
        "reasoning_tokens": last_result.reasoning_tokens if last_result else 0,
    }


def load_done_keys(jsonl: Path) -> set[tuple[str, int]]:
    done = set()
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


def parse_k_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except ValueError:
            raise SystemExit(f"FATAL: invalid --k-list entry: {part!r}")
        if k < 0:
            raise SystemExit(f"FATAL: invalid negative K index: {k}")
        out.append(k)
    if not out:
        raise SystemExit("FATAL: --k-list must contain at least one K index")
    if len(set(out)) != len(out):
        raise SystemExit(f"FATAL: duplicate K index in --k-list: {raw}")
    return out


def load_patches_subset(k_list: list[int]) -> tuple[dict[str, dict[int, str]], list[str]]:
    """Load arbitrary proposer indices. Rows keep actual proposer K as patch_idx."""
    by_iid: dict[str, dict[int, str]] = {}
    patch_dir = rc.PATCH_DIR
    for k in k_list:
        f = patch_dir / f"proposer_{k}.jsonl"
        if not f.exists():
            sys.stderr.write(f"FATAL: missing patches file {f}\n")
            sys.exit(2)
        with f.open() as fh:
            for line in fh:
                row = json.loads(line)
                iid = row["instance_id"]
                by_iid.setdefault(iid, {})[k] = row.get("model_patch", "") or ""
    return by_iid, sorted(by_iid.keys())


def load_resolve_bools_subset(k_list: list[int]) -> dict[str, dict[int, bool]]:
    """Load oracle resolved flags for arbitrary proposer indices."""
    resolved: dict[str, dict[int, bool]] = {}
    for k in k_list:
        for run_dir in rc.EVAL_RUN_BASE.glob(f"oracle_*_proposer_{k}"):
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
                    resolved.setdefault(iid, {})[k] = bool(rec.get("resolved", False))
    return resolved


async def run_dynamic_hints(
    client, deployment, instances, patches_by_iid, resolve_by_iid,
    problems_by_iid, dataset_fields, eval_index, out_dir, budget,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "binary_votes.jsonl"
    done_keys = load_done_keys(out_jsonl)
    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.time()

    # Pre-extract dynamic hint per instance (do this on the main thread; it's I/O-bound but cheap)
    print(f"[hints] extracting error traces for {len(instances)} instances…")
    hint_cache: dict[str, tuple[str, str, str]] = {}
    n_with_trace = 0
    for iid in instances:
        df = dataset_fields.get(iid, {})
        ftp = df.get("ftp", [])
        ftp_str = "\n".join(f"  - {n}" for n in ftp[:10]) or "  (none)"
        trace, _src = get_dynamic_hint_for_iid(iid, ftp, eval_index)
        if trace:
            n_with_trace += 1
        else:
            trace = "(no failing-test trace found in cached eval reports)"
        test_patch_str = (df.get("test_patch") or "")[:2500] or "(no test_patch in dataset)"
        hint_cache[iid] = (ftp_str, trace, test_patch_str)
    print(f"[hints] extracted error traces for {n_with_trace}/{len(instances)} instances")

    task_keys = []
    coros = []
    pk_resolves = {}
    for iid in instances:
        ftp_str, trace, test_patch_str = hint_cache[iid]
        problem = problems_by_iid.get(iid, "")
        patches = patches_by_iid.get(iid, [""] * K_PATCHES)
        for k in range(K_PATCHES):
            if (iid, k) in done_keys:
                continue
            patch = patches[k] or ""
            patch_resolves = resolve_by_iid.get(iid, {}).get(k, None)
            pk_resolves[(iid, k)] = patch_resolves
            files = patch_files(patch, max_files=5)
            files_str = "\n".join(f"  - {p}" for p in files) or "  (no patch / parse failed)"
            if not patch.strip():
                row = {
                    "instance_id": iid, "patch_idx": k,
                    "patch_resolves": patch_resolves,
                    "binary_votes": [
                        {"vote_idx": v, "resolves": None, "confidence": 0,
                         "reasoning": "", "abstain_reason": "empty_patch"}
                        for v in range(VOTE_START_INDEX, VOTE_START_INDEX + R_VOTES)
                    ],
                }
                with out_jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done_keys.add((iid, k))
                continue
            for vi in range(VOTE_START_INDEX, VOTE_START_INDEX + R_VOTES):
                task_keys.append((iid, k, vi))
                coros.append(_do_one_vote(
                    client, deployment, sem, problem, patch,
                    ftp_str, trace, test_patch_str, files_str,
                    iid, k, vi, out_dir, budget,
                ))

    print(f"[binary-dynamic] {len(coros)} API calls queued across "
          f"{len(instances)} instances")
    if not coros:
        return

    async def _keyed(key, coro):
        return key, await coro
    futures = [asyncio.ensure_future(_keyed(k, c)) for k, c in zip(task_keys, coros)]
    pending = {}
    halted = False
    completed = 0
    for fut in asyncio.as_completed(futures):
        if budget.check_halt() and not halted:
            print(f"[binary-dynamic] HALT: budget ${budget.spent_usd:.2f} "
                  f"≥ ${HARD_BUDGET_USD}")
            halted = True
            for f in futures:
                if not f.done():
                    f.cancel()
        try:
            (iid, k, vi), vote = await fut
        except asyncio.CancelledError:
            continue
        bucket = pending.setdefault((iid, k), {})
        bucket[vi] = vote
        if len(bucket) == R_VOTES:
            votes = [bucket[v] for v in range(VOTE_START_INDEX, VOTE_START_INDEX + R_VOTES)]
            row = {
                "instance_id": iid, "patch_idx": k,
                "patch_resolves": pk_resolves[(iid, k)],
                "binary_votes": votes,
            }
            with out_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_keys.add((iid, k))
            del pending[(iid, k)]
        completed += 1

    elapsed = time.time() - t_start
    print(f"[binary-dynamic] done: {completed} vote-events, "
          f"{budget.n_calls} API calls, ${budget.spent_usd:.2f} cost, "
          f"{elapsed:.0f}s wall-clock, abstains={budget.abstains_critic}")


async def run_dynamic_hints_subset(
    client, deployment, instances, patches_by_iid, resolve_by_iid,
    problems_by_iid, dataset_fields, eval_index, k_list, out_dir, budget,
):
    """Run dynamic-hints classifier on actual proposer indices in k_list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "binary_votes.jsonl"
    done_keys = load_done_keys(out_jsonl)
    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.time()

    print(f"[hints] extracting error traces for {len(instances)} instances…")
    hint_cache: dict[str, tuple[str, str, str]] = {}
    n_with_trace = 0
    for iid in instances:
        df = dataset_fields.get(iid, {})
        ftp = df.get("ftp", [])
        ftp_str = "\n".join(f"  - {n}" for n in ftp[:10]) or "  (none)"
        trace, _src = get_dynamic_hint_for_iid(iid, ftp, eval_index)
        if trace:
            n_with_trace += 1
        else:
            trace = "(no failing-test trace found in cached eval reports)"
        test_patch_str = (df.get("test_patch") or "")[:2500] or "(no test_patch in dataset)"
        hint_cache[iid] = (ftp_str, trace, test_patch_str)
    print(f"[hints] extracted error traces for {n_with_trace}/{len(instances)} instances")

    task_keys = []
    coros = []
    pk_resolves = {}
    for iid in instances:
        ftp_str, trace, test_patch_str = hint_cache[iid]
        problem = problems_by_iid.get(iid, "")
        patches = patches_by_iid.get(iid, {})
        for k in k_list:
            if (iid, k) in done_keys:
                continue
            patch = patches.get(k, "") or ""
            patch_resolves = resolve_by_iid.get(iid, {}).get(k, None)
            pk_resolves[(iid, k)] = patch_resolves
            files = patch_files(patch, max_files=5)
            files_str = "\n".join(f"  - {p}" for p in files) or "  (no patch / parse failed)"
            if not patch.strip():
                row = {
                    "instance_id": iid, "patch_idx": k,
                    "patch_resolves": patch_resolves,
                    "binary_votes": [
                        {"vote_idx": v, "resolves": None, "confidence": 0,
                         "reasoning": "", "abstain_reason": "empty_patch"}
                        for v in range(VOTE_START_INDEX, VOTE_START_INDEX + R_VOTES)
                    ],
                }
                with out_jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done_keys.add((iid, k))
                continue
            for vi in range(VOTE_START_INDEX, VOTE_START_INDEX + R_VOTES):
                task_keys.append((iid, k, vi))
                coros.append(_do_one_vote(
                    client, deployment, sem, problem, patch,
                    ftp_str, trace, test_patch_str, files_str,
                    iid, k, vi, out_dir, budget,
                ))

    print(f"[binary-dynamic] {len(coros)} API calls queued across "
          f"{len(instances)} instances (K={k_list})")
    if not coros:
        return

    async def _keyed(key, coro):
        return key, await coro
    futures = [asyncio.ensure_future(_keyed(k, c)) for k, c in zip(task_keys, coros)]
    pending = {}
    halted = False
    completed = 0
    for fut in asyncio.as_completed(futures):
        if budget.check_halt() and not halted:
            print(f"[binary-dynamic] HALT: budget ${budget.spent_usd:.2f} "
                  f"≥ ${HARD_BUDGET_USD}")
            halted = True
            for f in futures:
                if not f.done():
                    f.cancel()
        try:
            (iid, k, vi), vote = await fut
        except asyncio.CancelledError:
            continue
        bucket = pending.setdefault((iid, k), {})
        bucket[vi] = vote
        if len(bucket) == R_VOTES:
            votes = [bucket[v] for v in range(VOTE_START_INDEX, VOTE_START_INDEX + R_VOTES)]
            row = {
                "instance_id": iid, "patch_idx": k,
                "patch_resolves": pk_resolves[(iid, k)],
                "binary_votes": votes,
            }
            with out_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_keys.add((iid, k))
            del pending[(iid, k)]
        completed += 1

    elapsed = time.time() - t_start
    print(f"[binary-dynamic] done: {completed} vote-events, "
          f"{budget.n_calls} API calls, ${budget.spent_usd:.2f} cost, "
          f"{elapsed:.0f}s wall-clock, abstains={budget.abstains_critic}")


def analyze(out_dir: Path, label: str = "direct-binary + dynamic hints"):
    jsonl = out_dir / "binary_votes.jsonl"
    if not jsonl.exists():
        print(f"[analyze] no cache at {jsonl}")
        return 0.0
    by_iid: dict[str, list[dict]] = defaultdict(list)
    with jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_iid[r["instance_id"]].append(r)
    iids = sorted(by_iid.keys())
    n_inst = len(iids)
    n_resolved_maj = 0
    n_resolved_conf = 0
    yes_counts = []
    abstain_count = 0
    from collections import Counter
    for iid in iids:
        rows = sorted(by_iid[iid], key=lambda r: r["patch_idx"])
        scores_maj, scores_conf = [], []
        for row in rows:
            yes = sum(1 for v in row["binary_votes"] if v.get("resolves") is True)
            conf_yes = sum(int(v.get("confidence", 0))
                           for v in row["binary_votes"]
                           if v.get("resolves") is True)
            abst = sum(1 for v in row["binary_votes"]
                       if v.get("resolves") is None)
            abstain_count += abst
            yes_counts.append(yes)
            scores_maj.append((row["patch_idx"], yes, bool(row.get("patch_resolves"))))
            scores_conf.append((row["patch_idx"], conf_yes, bool(row.get("patch_resolves"))))
        scores_maj.sort(key=lambda x: (-x[1], x[0]))
        if scores_maj and scores_maj[0][2]:
            n_resolved_maj += 1
        scores_conf.sort(key=lambda x: (-x[1], x[0]))
        if scores_conf and scores_conf[0][2]:
            n_resolved_conf += 1
    hist = Counter(yes_counts)
    print(f"\n=== {label} analysis on {n_inst} instances ===")
    print(f"Per-patch yes-count distribution (out of R={R_VOTES}):")
    for c in range(R_VOTES + 1):
        print(f"  {c} yes votes: {hist.get(c, 0):>4d} patches")
    print(f"Total abstains: {abstain_count}")
    print()
    rate = n_resolved_maj / max(1, n_inst)
    print(f"Resolve rate (most-yes, tiebreak lowest-k):  {n_resolved_maj}/{n_inst} = {rate:.3f}")
    print(f"Resolve rate (confidence-weighted):          {n_resolved_conf}/{n_inst} = {n_resolved_conf/max(1, n_inst):.3f}")
    print()
    print("=" * 64)
    print(f"DECISION (most-yes resolve rate: {rate:.1%})")
    print("=" * 64)
    if rate >= 0.77:
        print(f"  ≥77% → execution-grounded hints is the lever. Write up.")
    elif rate >= 0.73:
        print(f"  73-76% → marginal additional gain. Two-step grounding story.")
    elif rate >= 0.69:
        print(f"  69-72% → dynamic hints don't help over static. Write up the static result.")
    else:
        print(f"  <69% → unexpectedly low; investigate parse-fail or extraction issues.")
    return rate


async def amain(args):
    env = rc._require_env()
    if args.reasoning_effort:
        os.environ["AZURE_REASONING_EFFORT"] = args.reasoning_effort
        print(f"[env] AZURE_REASONING_EFFORT={args.reasoning_effort}")
    if args.concurrency:
        global CONCURRENCY
        CONCURRENCY = int(args.concurrency)
        print(f"[env] CONCURRENCY={CONCURRENCY}")
    if args.max_output_tokens:
        global BINARY_MAX_OUTPUT
        BINARY_MAX_OUTPUT = int(args.max_output_tokens)
        print(f"[env] BINARY_MAX_OUTPUT={BINARY_MAX_OUTPUT}")
    if args.votes_per_patch:
        global R_VOTES
        R_VOTES = int(args.votes_per_patch)
        print(f"[env] R_VOTES={R_VOTES}")
    if args.vote_start_index:
        global VOTE_START_INDEX
        VOTE_START_INDEX = int(args.vote_start_index)
        print(f"[env] VOTE_START_INDEX={VOTE_START_INDEX}")
    client = rc._make_client(env)

    k_list = parse_k_list(args.k_list)
    print(f"[load] patches for proposer K-list {k_list}…")
    patches_by_iid, all_iids = load_patches_subset(k_list)
    print(f"[load] {len(all_iids)} instances, K-list={k_list}")
    print("[load] resolve booleans…")
    resolve_by_iid = load_resolve_bools_subset(k_list)
    print("[load] problem statements…")
    problems_by_iid = rc.load_problem_statements(all_iids)
    print("[load] dataset fields (FAIL_TO_PASS, test_patch, base_commit)…")
    dataset_fields = load_dataset_fields(all_iids)
    print(f"[load] dataset fields for {len(dataset_fields)}/{len(all_iids)}")
    eval_index = {}
    if args.allow_cached_eval_traces:
        print("[load] LEGACY/NON-BLIND: indexing failing eval dirs for error trace extraction…")
        eval_index = _index_failing_eval_dirs()
        print(f"[load] failing-eval-dir index covers {len(eval_index)} iids")
    else:
        print("[load] pre-eval blind mode: not reading cached eval reports/test_output.txt")

    pilot = rc.pilot_instance_subset(all_iids)
    budget = rc.Budget()
    hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))

    if args.smoke:
        smoke_iids = pilot[:int(args.smoke)]
        print(f"[smoke] {len(smoke_iids)} instances: {smoke_iids}")
        out_dir = Path(args.output_dir) if args.output_dir else SMOKE_DIR
        await run_dynamic_hints_subset(client, env["deployment"], smoke_iids,
                                       patches_by_iid, resolve_by_iid, problems_by_iid,
                                       dataset_fields, eval_index, k_list, out_dir, budget)
        analyze(out_dir, label=f"dynamic-hints smoke ({len(smoke_iids)} inst)")
    elif args.all:
        print(f"[all] {len(all_iids)} instances")
        out_dir = Path(args.output_dir) if args.output_dir else FULL500_DIR
        await run_dynamic_hints_subset(client, env["deployment"], all_iids,
                                       patches_by_iid, resolve_by_iid, problems_by_iid,
                                       dataset_fields, eval_index, k_list, out_dir, budget)
        analyze(out_dir, label="direct-binary + dynamic hints (full 500)")
    else:
        print("specify --smoke N or --all")
        return 2

    hb.cancel()
    try:
        await hb
    except asyncio.CancelledError:
        pass
    print(f"[done] calls={budget.n_calls} cost=${budget.spent_usd:.2f} "
          f"abstains={budget.abstains_critic}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", type=int, default=0,
                   help="Pre-flight smoke on first N pilot instances")
    p.add_argument("--all", action="store_true",
                   help="Full 500 run")
    p.add_argument("--output-dir", default="",
                   help="Override output directory for binary_votes.jsonl")
    p.add_argument("--reasoning-effort", default="",
                   choices=("", "low", "medium", "high", "xhigh"),
                   help="Override AZURE_REASONING_EFFORT after run_ceiling env setup")
    p.add_argument("--concurrency", type=int, default=0,
                   help="Override async API concurrency for this run")
    p.add_argument("--max-output-tokens", type=int, default=0,
                   help="Override per-call max_output_tokens for this run")
    p.add_argument("--votes-per-patch", type=int, default=0,
                   help="Number of binary votes to collect per patch in this run")
    p.add_argument("--vote-start-index", type=int, default=0,
                   help="First vote_idx to use. Use 5 with --votes-per-patch 5 "
                        "to extend an R=5 cache to R=10 in a sidecar output dir.")
    p.add_argument("--k-list", default=",".join(str(k) for k in GREEDY8_K_LIST),
                   help="Comma-separated proposer indices. Defaults to greedy-8 "
                        "0,1,2,3,4,5,6,7.")
    p.add_argument("--allow-cached-eval-traces", action="store_true",
                   help="LEGACY/NON-BLIND: mine cached eval report test_output.txt before selection.")
    args = p.parse_args()
    if not (args.smoke or args.all):
        p.print_help()
        return 2
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
