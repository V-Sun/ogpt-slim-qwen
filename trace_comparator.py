#!/usr/bin/env python3
"""Trace-aware pairwise comparator.

Extends the blind comparator with rich proposer context extracted from
.traj.json trajectory files: agent THOUGHT reasoning, bash commands run,
and key outputs. Pre-eval blind — no post-patch harness results.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ceiling as rc
import direct_binary_dynamic_hints as dyn

REPO = Path(__file__).resolve().parent
DATA_REPO = Path("/home/vsun/orchestra-gpt")
DEFAULT_PATCHES_DIR = DATA_REPO / "outputs" / "oracle_preds_v3"
DEFAULT_REPORTS_DIR = DATA_REPO / "logs" / "run_evaluation"
DEFAULT_OUTPUT_DIR = REPO / "outputs" / "trace_comparator_greedy8"
TRAJ_SHARD_DIRS = sorted((DATA_REPO / "outputs").glob("propsonly_xhigh_500_shard_*"))

K_PATCHES = 8
GREEDY8_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_OUTPUT_TOKENS = 4096
CONCURRENCY = int(os.getenv("TRACE_COMPARATOR_CONCURRENCY", "256"))

# Max chars for each trace section in the prompt
MAX_THOUGHTS = 1200
MAX_COMMANDS = 600
MAX_ERRORS = 600


# ---------------------------------------------------------------------------
# Trajectory index + extraction
# ---------------------------------------------------------------------------

def build_traj_index() -> dict[tuple[str, int], Path]:
    """Map (instance_id, k_index) -> .traj.json path across all shard dirs."""
    index: dict[tuple[str, int], Path] = {}
    for shard_dir in TRAJ_SHARD_DIRS:
        for task_dir in shard_dir.iterdir():
            if not task_dir.is_dir():
                continue
            iid = task_dir.name
            for tf in task_dir.glob("proposer_*.traj.json"):
                try:
                    k = int(tf.stem.replace("proposer_", "").replace(".traj", ""))
                except ValueError:
                    continue
                index[(iid, k)] = tf
    return index


def extract_trace_context(traj_path: Path, max_steps: int = 40) -> dict[str, str]:
    """Extract THOUGHT reasoning, key commands, and errors from a trajectory."""
    try:
        d = json.load(traj_path.open())
    except Exception:
        return {"thoughts": "", "commands": "", "errors": ""}

    msgs = d.get("messages", [])
    thoughts: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    step = 0

    for m in msgs:
        if step >= max_steps:
            break
        # Responses API output messages
        if "output" in m and isinstance(m.get("output"), list):
            step += 1
            for item in m["output"]:
                t = item.get("type", "")
                if t == "message":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            text = block.get("text", "")
                            if text.startswith("THOUGHT:"):
                                thoughts.append(text[8:].strip())
                elif t == "function_call":
                    try:
                        args = json.loads(item.get("arguments", "{}"))
                        cmd = args.get("command", "").strip()
                        if cmd:
                            commands.append(cmd)
                    except Exception:
                        pass
        # Tool result messages
        elif "extra" in m or m.get("type") == "function_call_output":
            raw = (m.get("extra") or {}).get("raw_output", "") or m.get("output", "")
            if isinstance(raw, str) and raw.strip():
                # Capture stderr / tracebacks as errors
                low = raw.lower()
                if any(kw in low for kw in ("error", "traceback", "exception", "failed", "cannot")):
                    errors.append(raw.strip()[:400])

    def _compact(items: list[str], max_chars: int) -> str:
        out = "\n".join(items)
        if len(out) <= max_chars:
            return out
        head = max_chars // 2
        tail = max_chars - head - 60
        return out[:head] + f"\n... [trimmed] ...\n" + out[-tail:]

    # Filter commands to exploration ones (skip pure edits)
    explore_cmds = [c for c in commands if any(
        c.startswith(p) for p in ("grep", "cat", "sed", "find", "ls", "python", "git")
    )]
    edit_cmds = [c for c in commands if c not in explore_cmds]

    return {
        "thoughts": _compact(thoughts, MAX_THOUGHTS),
        "commands": _compact(explore_cmds[:30], MAX_COMMANDS),
        "edit_summary": f"{len(edit_cmds)} edit/patch commands applied",
        "errors": _compact(errors[:5], MAX_ERRORS),
    }


def format_trace(ctx: dict[str, str], patch_label: str) -> str:
    parts = []
    if ctx.get("thoughts"):
        parts.append(f"Agent reasoning (THOUGHT steps):\n{ctx['thoughts']}")
    if ctx.get("commands"):
        parts.append(f"Files/code explored:\n{ctx['commands']}")
    if ctx.get("errors"):
        parts.append(f"Errors encountered during exploration:\n{ctx['errors']}")
    if ctx.get("edit_summary"):
        parts.append(ctx["edit_summary"])
    return "\n\n".join(parts) if parts else f"(no trace available for {patch_label})"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are comparing two candidate patches for a SWE-bench issue.

Your job is to choose which patch is more likely to pass the hidden SWE-bench tests.
You have access to both the final patch diffs AND the agent's working context: the
reasoning steps, files explored, and errors encountered while producing each patch.

Do NOT choose based on code style or patch length.
Do NOT call TIE unless both patches are genuinely equivalent after reading the agent context.
A patch that modifies code unrelated to the failing test cannot be correct.

Issue:
{issue_text}

FAIL_TO_PASS tests (must pass after correct patch):
{fail_to_pass_tests}

Test source code (defines what the correct patch must achieve):
{test_patch}

════════════════════════════════════════
PATCH A
════════════════════════════════════════
Changed files:
{patch_a_changed_files}

Diff:
{patch_a_diff}

Agent context (how Patch A was produced):
{patch_a_trace}

════════════════════════════════════════
PATCH B
════════════════════════════════════════
Changed files:
{patch_b_changed_files}

Diff:
{patch_b_diff}

Agent context (how Patch B was produced):
{patch_b_trace}

════════════════════════════════════════
Decision rules:
- Use the agent context to understand WHY each patch makes its changes.
- An agent that found and directly modified the relevant code path is more
  likely to have the correct fix than one that patched unrelated code.
- If one agent's reasoning clearly identifies the root cause and the other
  does not, that is the decisive factor.
- Use TIE ONLY if both agents found the same root cause with equally valid fixes.

Return ONLY valid JSON:
{{
  "winner": "A" | "B" | "TIE",
  "a_fix_evidence": "one sentence: does A's agent context support the fix?",
  "a_failure_risk": "one sentence: A's main failure risk",
  "b_fix_evidence": "one sentence: does B's agent context support the fix?",
  "b_failure_risk": "one sentence: B's main failure risk",
  "decisive_factor": "one sentence: main reason for the winner",
  "confidence": 1
}}
"""

_DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)", re.MULTILINE)


def patch_files(patch_text: str, max_files: int = 12) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _DIFF_FILE_RE.finditer(patch_text or ""):
        path = m.group(1)
        if path in ("", "/dev/null") or path in seen_set:
            continue
        seen.append(path)
        seen_set.add(path)
        if len(seen) >= max_files:
            break
    return seen


def compact(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 80
    return text[:head] + f"\n... [trimmed {len(text)-max_chars} chars] ...\n" + text[-tail:]


def build_prompt(
    iid: str,
    a: int,
    b: int,
    patches: dict[int, str],
    fields: dict[str, Any],
    trace_a: dict[str, str],
    trace_b: dict[str, str],
) -> str:
    ftp = fields.get("fail_to_pass", [])
    ftp_str = "\n".join(f"- {x}" for x in ftp[:20]) or "(none provided)"
    pa, pb = patches.get(a, "") or "", patches.get(b, "") or ""
    test_patch_str = compact(fields.get("test_patch", "") or "", 2500) or "(not available)"
    return PROMPT_TEMPLATE.format(
        issue_text=compact(fields.get("issue_text", ""), 2000),
        fail_to_pass_tests=ftp_str,
        test_patch=test_patch_str,
        patch_a_changed_files="\n".join(f"- {x}" for x in patch_files(pa)) or "(none parsed)",
        patch_a_diff=compact(pa, 5000),
        patch_a_trace=format_trace(trace_a, "A"),
        patch_b_changed_files="\n".join(f"- {x}" for x in patch_files(pb)) or "(none parsed)",
        patch_b_diff=compact(pb, 5000),
        patch_b_trace=format_trace(trace_b, "B"),
    )


# ---------------------------------------------------------------------------
# Response parsing (same as v1/v2)
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(rc._strip_fences(text))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("winner") not in ("A", "B", "TIE"):
        return None
    try:
        conf = int(obj.get("confidence", 1))
    except Exception:
        conf = 1
    conf = max(1, min(5, conf))
    return {
        "winner": obj["winner"],
        "a_fix_evidence": str(obj.get("a_fix_evidence", ""))[:500],
        "a_failure_risk": str(obj.get("a_failure_risk", ""))[:500],
        "b_fix_evidence": str(obj.get("b_fix_evidence", ""))[:500],
        "b_failure_risk": str(obj.get("b_failure_risk", ""))[:500],
        "decisive_factor": str(obj.get("decisive_factor", ""))[:500],
        "confidence": conf,
    }


def done_keys(votes_path: Path) -> set[tuple[str, int, int]]:
    out: set[tuple[str, int, int]] = set()
    if not votes_path.exists():
        return out
    with votes_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            out.add((row["instance_id"], int(row["patch_a_idx"]), int(row["patch_b_idx"])))
    return out


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def one_call(client, deployment, sem, prompt, temperature, out_dir, meta, budget):
    budget.issued += 1
    result = await rc._api_call(
        client, deployment, prompt,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        sem=sem,
    )
    budget.add("comparator", result)
    rc.append_usage(out_dir, "comparator", meta, result, "trace_comparator", temperature)
    parsed = None if result.error else parse_response(result.text)
    if parsed is None:
        budget.abstains_comparator += 1
    return {
        "meta": meta,
        "raw_response": result.text,
        "parsed": parsed,
        "parse_error": result.error or (None if parsed is not None else "parse_failed"),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
    }


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def run_pairs(args, instances, patches_by_iid, fields_by_iid, traj_index, truth) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    votes_path = out_dir / "pairwise_trace_comparator_votes.jsonl"
    done = done_keys(votes_path)

    env = rc._require_env()
    client = rc._make_client(env)
    deployment = args.model.split("/", 1)[1] if args.model.startswith("azure/") else args.model
    if not deployment:
        deployment = env["deployment"]
    sem = asyncio.Semaphore(CONCURRENCY)
    budget = rc.Budget()
    hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))

    # Pre-extract all traces (I/O-bound, do up front)
    print(f"[trace] extracting trajectories for {len(instances)} instances × {len(args.k_indices)} K's...")
    trace_cache: dict[tuple[str, int], dict[str, str]] = {}
    n_found = 0
    for iid in instances:
        for k in args.k_indices:
            tf = traj_index.get((iid, k))
            if tf and tf.exists():
                trace_cache[(iid, k)] = extract_trace_context(tf)
                n_found += 1
            else:
                trace_cache[(iid, k)] = {}
    print(f"[trace] found {n_found}/{len(instances)*len(args.k_indices)} trajectories")

    async def run_one(iid, a, b, prompt_ab):
        calls = []
        for vote_idx in range(args.votes_per_pair):
            primary = await one_call(
                client, deployment, sem, prompt_ab, args.temperature, out_dir,
                {"iid": iid, "patch_a_idx": a, "patch_b_idx": b, "vote_idx": vote_idx},
                budget,
            )
            calls.append({"vote_idx": vote_idx, "primary": primary})
        return iid, a, b, calls

    coros = []
    for iid in instances:
        fields = fields_by_iid.get(iid, {})
        patches = patches_by_iid.get(iid, {})
        for a, b in itertools.combinations(args.k_indices, 2):
            if (iid, a, b) in done:
                continue
            if not (patches.get(a, "") or "").strip() or not (patches.get(b, "") or "").strip():
                row = {"instance_id": iid, "patch_a_idx": a, "patch_b_idx": b,
                       "votes": [], "skip_reason": "empty_patch"}
                with votes_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                continue
            trace_a = trace_cache.get((iid, a), {})
            trace_b = trace_cache.get((iid, b), {})
            prompt_ab = build_prompt(iid, a, b, patches, fields, trace_a, trace_b)
            coros.append(run_one(iid, a, b, prompt_ab))

    print(f"[trace] queued {len(coros)} pair rows, votes_per_pair={args.votes_per_pair}")
    completed = 0
    for fut in asyncio.as_completed([asyncio.create_task(c) for c in coros]):
        iid, a, b, calls = await fut
        # Attach truth for offline scoring
        truth_a = truth.get(iid, {}).get(a)
        truth_b = truth.get(iid, {}).get(b)
        row = {
            "instance_id": iid,
            "patch_a_idx": a,
            "patch_b_idx": b,
            "patch_a_resolves": truth_a,
            "patch_b_resolves": truth_b,
            "votes": calls,
        }
        with votes_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        completed += 1
        if completed % 100 == 0:
            print(f"[trace] rows {completed}/{len(coros)} cost=${budget.spent_usd:.2f}")

    hb.cancel()
    try:
        await hb
    except asyncio.CancelledError:
        pass
    print(f"[trace] done rows={completed} calls={budget.n_calls} cost=${budget.spent_usd:.2f} "
          f"abstains={budget.abstains_comparator}")

    # Quick resolve rate summary
    summarize(votes_path, args.k_indices, truth, out_dir)


def summarize(votes_path: Path, k_list: list[int], truth: dict, out_dir: Path) -> None:
    from collections import Counter as C
    rows_by_iid: dict[str, list] = defaultdict(list)
    with votes_path.open() as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            rows_by_iid[row["instance_id"]].append(row)

    scores = {iid: {k: 0.0 for k in k_list} for iid in rows_by_iid}
    for iid, rows in rows_by_iid.items():
        for row in rows:
            a, b = int(row["patch_a_idx"]), int(row["patch_b_idx"])
            counts = C()
            for vote in row.get("votes", []):
                p = (vote.get("primary") or {}).get("parsed")
                if p:
                    counts[p["winner"]] += 1
            winner = counts.most_common(1)[0][0] if counts else "TIE"
            if winner == "A":
                scores[iid][a] += 1; scores[iid][b] -= 1
            elif winner == "B":
                scores[iid][b] += 1; scores[iid][a] -= 1

    resolved = total = 0
    for iid, sc in scores.items():
        best_k = max(k_list, key=lambda k: (-sc.get(k, 0), k))
        if best_k in truth.get(iid, {}):
            total += 1
            if truth[iid][best_k]:
                resolved += 1

    rate = resolved / total if total else 0.0
    summary = {"resolved": resolved, "total": total, "resolve_rate": rate,
               "instances": len(rows_by_iid)}
    print(f"\n[trace] Copeland resolve rate: {resolved}/{total} = {rate:.4f}")
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_patches(patches_dir: Path, k_list: list[int]) -> tuple[dict[str, dict[int, str]], list[str]]:
    by_iid: dict[str, dict[int, str]] = {}
    for k in k_list:
        f = patches_dir / f"proposer_{k}.jsonl"
        if not f.exists():
            raise FileNotFoundError(f)
        with f.open() as fh:
            for line in fh:
                if not line.strip(): continue
                row = json.loads(line)
                iid = row["instance_id"]
                by_iid.setdefault(iid, {})[k] = row.get("model_patch", "") or ""
    return by_iid, sorted(by_iid)


def load_dataset_fields(iids: list[str]) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset
    wanted = set(iids)
    out: dict[str, dict[str, Any]] = {}
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
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
            "issue_text": inst.get("problem_statement", "") or "",
            "fail_to_pass": [x for x in ftp if isinstance(x, str)],
            "test_patch": inst.get("test_patch", "") or "",
        }
    return out


def load_truth(reports_dir: Path, k_list: list[int]) -> dict[str, dict[int, bool]]:
    truth: dict[str, dict[int, bool]] = {}
    for k in k_list:
        for rep in reports_dir.glob(f"oracle_*_proposer_{k}/*/*/report.json"):
            try:
                obj = json.load(rep.open())
            except Exception:
                continue
            iid = rep.parent.name
            rec = obj.get(iid, obj) if isinstance(obj, dict) else {}
            if isinstance(rec, dict):
                truth.setdefault(iid, {})[k] = bool(rec.get("resolved", False))
    return truth


def parse_k_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part: continue
        k = int(part)
        out.append(k)
    if not out:
        raise SystemExit("FATAL: --k-list must contain at least one K index")
    return out


def read_instances_path(path: str) -> list[str]:
    if not path:
        return []
    p = Path(path)
    obj = json.load(p.open())
    if isinstance(obj, list):
        return [str(x["instance_id"] if isinstance(x, dict) else x) for x in obj]
    return [str(k) for k in obj]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--instances-path", default="")
    p.add_argument("--patches-dir", default=str(DEFAULT_PATCHES_DIR))
    p.add_argument("--harness-reports-dir", default=str(DEFAULT_REPORTS_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--model", default=os.getenv("AZURE_DEPLOYMENT", "gpt-5.4-nano"))
    p.add_argument("--votes-per-pair", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--k-list", default=",".join(str(k) for k in GREEDY8_K_LIST))
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--max-instances", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.k_indices = parse_k_list(args.k_list)
    if args.concurrency:
        global CONCURRENCY
        CONCURRENCY = args.concurrency

    patches_dir = Path(args.patches_dir)
    reports_dir = Path(args.harness_reports_dir)

    print(f"[load] patches from {patches_dir}")
    patches_by_iid, all_iids = load_patches(patches_dir, args.k_indices)

    requested = read_instances_path(args.instances_path)
    instances = [iid for iid in requested if iid in set(all_iids)] if requested else list(all_iids)
    if args.max_instances:
        instances = instances[:args.max_instances]
    print(f"[load] {len(instances)} instances, K={args.k_indices}")

    print("[load] dataset fields...")
    fields_by_iid = load_dataset_fields(instances)
    print("[load] eval truth...")
    truth = load_truth(reports_dir, args.k_indices)
    print("[load] building trajectory index...")
    traj_index = build_traj_index()
    print(f"[load] trajectory index: {len(traj_index)} entries")

    asyncio.run(run_pairs(args, instances, patches_by_iid, fields_by_iid, traj_index, truth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
