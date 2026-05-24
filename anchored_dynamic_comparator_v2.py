#!/usr/bin/env python3
"""Anchored pairwise comparator v2 — adds test_patch + stronger tie-breaking.

Blind (pre-eval): sees issue text, FAIL_TO_PASS test names, test_patch source,
candidate diffs, and optionally a baseline trace. No post-patch harness outcomes.
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
DEFAULT_OUTPUT_DIR = REPO / "outputs" / "blind_comparator_v2_fixable43"

K_PATCHES = 8
GREEDY8_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_OUTPUT_TOKENS = 4096
CONCURRENCY = int(os.getenv("ANCHORED_COMPARATOR_CONCURRENCY", "128"))


PROMPT_TEMPLATE = """You are comparing two candidate patches for a SWE-bench issue.

Your job is to choose which patch is more likely to pass the hidden SWE-bench tests.
The chosen patch will then be sent to the test harness for evaluation. You do NOT
have access to the test outcomes for either candidate — you must judge from the
patch text, the issue description, the test source code, and the failure behavior
of the unpatched code only.

Do NOT choose based on which patch looks cleaner, shorter, more conventional, or more confident.
Do NOT answer TIE just because both patches look plausible.
You must reason about which patch is more likely to fix the failure described below.

Most plausible patches are wrong. A patch that does not modify the code path
exercised by the FAIL_TO_PASS test CANNOT pass that test — prefer the patch
that directly addresses the tested code path.

Issue:
{issue_text}

FAIL_TO_PASS tests (these must pass after a correct patch):
{fail_to_pass_tests}

Test source code (the exact test that must pass — use this to determine what
behavior the correct patch must produce):
{test_patch}

Failure behavior of the unpatched code (baseline — both candidates address this):
{baseline_trace_or_summary}

Patch A changed files:
{patch_a_changed_files}

Patch A diff:
{patch_a_diff}

Patch B changed files:
{patch_b_changed_files}

Patch B diff:
{patch_b_diff}

Decision rules:
- Read the test source. Identify exactly what the test calls and what it asserts.
- A patch wins if it modifies the code path the test exercises AND produces the
  asserted outcome. A patch that touches unrelated code cannot win.
- If one patch modifies the relevant code path and the other does not, that is
  the decisive factor — do not call TIE.
- Use TIE ONLY if both patches modify the same relevant code path with equally
  plausible fixes. If there is any meaningful difference, pick a winner.

Return ONLY valid JSON:
{{
  "winner": "A" | "B" | "TIE",
  "a_fix_evidence": "short sentence about why A might fix the failure",
  "a_failure_risk": "short sentence about A's failure modes",
  "b_fix_evidence": "short sentence about why B might fix the failure",
  "b_failure_risk": "short sentence about B's failure modes",
  "decisive_factor": "short sentence explaining the main reason for the winner",
  "confidence": 1
}}
"""


_DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)", re.MULTILINE)


def strip_fences(text: str) -> str:
    return rc._strip_fences(text)


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
    return text[:head] + f"\n... [trimmed {len(text) - max_chars} chars] ...\n" + text[-tail:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--instances-path", default="", help="Optional ids file: txt, json list, or jsonl rows with instance_id")
    p.add_argument("--patches-dir", default=str(DEFAULT_PATCHES_DIR))
    p.add_argument("--harness-reports-dir", default=str(DEFAULT_REPORTS_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--model", default=os.getenv("AZURE_DEPLOYMENT", "gpt-5.4-nano"))
    p.add_argument("--max-instances", type=int, default=0)
    p.add_argument("--pilot", action="store_true", help="Use 50 instances unless --max-instances is provided")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--votes-per-pair", type=int, default=1)
    p.add_argument("--position-swap", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tie-policy", default="margin", choices=("margin", "zero"))
    p.add_argument("--summary-only", action="store_true", help="Do not call API; rebuild tournament outputs and summary")
    p.add_argument("--allow-cached-eval-traces", action="store_true",
                   help="LEGACY/NON-BLIND: mine cached eval reports for baseline-like traces before selection.")
    p.add_argument("--k-list", default=",".join(str(k) for k in GREEDY8_K_LIST),
                   help="Comma-separated proposer indices. Defaults to greedy-8 "
                        "0,1,2,3,4,5,6,7.")
    return p.parse_args()


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


def read_instances_path(path: str) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == ".json":
        obj = json.load(p.open())
        if isinstance(obj, list):
            return [str(x["instance_id"] if isinstance(x, dict) else x) for x in obj]
        if isinstance(obj, dict):
            return [str(k) for k in obj]
    out: list[str] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                out.append(str(row.get("instance_id") or row.get("iid")))
            else:
                out.append(line)
    return [x for x in out if x]


def load_patches(patches_dir: Path, k_list: list[int]) -> tuple[dict[str, dict[int, str]], list[str]]:
    by_iid: dict[str, dict[int, str]] = {}
    for k in k_list:
        f = patches_dir / f"proposer_{k}.jsonl"
        if not f.exists():
            raise FileNotFoundError(f)
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
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
            "repo": inst.get("repo", "") or "",
            "base_commit": inst.get("base_commit", "") or "",
            "test_patch": inst.get("test_patch", "") or "",
        }
    return out


def load_truth(harness_reports_dir: Path, k_list: list[int]) -> dict[str, dict[int, bool]]:
    truth: dict[str, dict[int, bool]] = {}
    for k in k_list:
        for rep in harness_reports_dir.glob(f"oracle_*_proposer_{k}/*/*/report.json"):
            try:
                obj = json.load(rep.open())
            except Exception:
                continue
            iid = rep.parent.name
            rec = obj.get(iid, obj) if isinstance(obj, dict) else {}
            if isinstance(rec, dict):
                truth.setdefault(iid, {})[k] = bool(rec.get("resolved", False))
    return truth


def load_report_index(harness_reports_dir: Path, k_list: list[int]) -> dict[tuple[str, int], Path]:
    index: dict[tuple[str, int], Path] = {}
    for k in k_list:
        for rep in harness_reports_dir.glob(f"oracle_*_proposer_{k}/*/*/report.json"):
            index[(rep.parent.name, k)] = rep
    return index


def report_summary(report_path: Path | None, iid: str, ftp: list[str]) -> str:
    if not report_path or not report_path.exists():
        return "(no post-patch harness report cached for this patch)"
    try:
        obj = json.load(report_path.open())
    except Exception as e:
        return f"(could not parse harness report: {e})"
    rec = obj.get(iid, obj) if isinstance(obj, dict) else {}
    if not isinstance(rec, dict):
        rec = {}
    # LEAKAGE FIX (2026-05-05): do NOT expose `resolved` in the prompt — that's
    # the SWE-bench oracle label. Surface only structured per-test outcomes
    # (FAIL_TO_PASS / PASS_TO_PASS / PASS_TO_FAIL), the apply-clean bool, and the
    # raw test_output excerpt. Those are signals an independent verifier could
    # legally produce at deployment time; `resolved` is the aggregated gold judgment.
    status = rec.get("tests_status") or {}
    applied = rec.get("patch_successfully_applied")
    parts = [f"patch_applied={applied}"] if applied is not None else []
    for group in ("FAIL_TO_PASS", "PASS_TO_PASS", "PASS_TO_FAIL"):
        g = status.get(group) or {}
        if isinstance(g, dict):
            for bucket in ("success", "failure", "error"):
                vals = g.get(bucket) or []
                if vals:
                    shown = ", ".join(map(str, vals[:8]))
                    more = f" (+{len(vals)-8} more)" if len(vals) > 8 else ""
                    parts.append(f"{group}.{bucket}: {shown}{more}")
    test_out = report_path.parent / "test_output.txt"
    trace = dyn.extract_test_failures(test_out, ftp, max_chars=1800) if test_out.exists() else ""
    if trace:
        parts.append("trace excerpt:\n" + trace)
    return compact("\n".join(parts), 2600)


def baseline_trace(iid: str, ftp: list[str], eval_index: dict[str, list[Path]]) -> str:
    if not eval_index:
        return "(no baseline trace provided)"
    trace, src = dyn.get_dynamic_hint_for_iid(iid, ftp, eval_index)
    if trace:
        return compact(f"source={src}\n{trace}", 2600)
    return "(no cached baseline-like failing trace found; use issue text and FAIL_TO_PASS names)"


def build_prompt(
    iid: str,
    a: int,
    b: int,
    patches: dict[int, str],
    fields: dict[str, Any],
    base_trace: str,
    patch_behaviors: dict[int, str] | None = None,
) -> str:
    ftp = fields.get("fail_to_pass", [])
    ftp_str = "\n".join(f"- {x}" for x in ftp[:20]) or "(none provided)"
    pa, pb = patches.get(a, "") or "", patches.get(b, "") or ""
    test_patch_str = compact(fields.get("test_patch", "") or "", 3000) or "(not available)"
    return PROMPT_TEMPLATE.format(
        issue_text=compact(fields.get("issue_text", ""), 2500),
        fail_to_pass_tests=ftp_str,
        test_patch=test_patch_str,
        baseline_trace_or_summary=base_trace,
        patch_a_changed_files="\n".join(f"- {x}" for x in patch_files(pa)) or "(none parsed)",
        patch_a_diff=compact(pa, 7000),
        patch_b_changed_files="\n".join(f"- {x}" for x in patch_files(pb)) or "(none parsed)",
        patch_b_diff=compact(pb, 7000),
    )


def parse_response(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(strip_fences(text))
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


async def one_call(client, deployment, sem, prompt, temperature, output_dir, meta, budget: rc.Budget) -> dict[str, Any]:
    budget.issued += 1
    result = await rc._api_call(
        client,
        deployment,
        prompt,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        sem=sem,
    )
    budget.add("comparator", result)
    rc.append_usage(output_dir, "comparator", meta, result, "anchored_dynamic_comparator", temperature)
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


def normalize_swap(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if parsed is None:
        return None
    out = dict(parsed)
    if out["winner"] == "A":
        out["winner"] = "B"
    elif out["winner"] == "B":
        out["winner"] = "A"
    out["a_fix_evidence"], out["b_fix_evidence"] = out["b_fix_evidence"], out["a_fix_evidence"]
    out["a_failure_risk"], out["b_failure_risk"] = out["b_failure_risk"], out["a_failure_risk"]
    return out


async def run_pairs(args, instances, patches_by_iid, fields_by_iid, eval_index, report_index) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    votes_path = out_dir / "pairwise_dynamic_comparator_votes.jsonl"
    done = done_keys(votes_path) if args.resume else set()

    env = rc._require_env()
    client = rc._make_client(env)
    deployment = args.model.split("/", 1)[1] if args.model.startswith("azure/") else args.model
    if not deployment:
        deployment = env["deployment"]
    sem = asyncio.Semaphore(CONCURRENCY)
    budget = rc.Budget()
    hb = asyncio.create_task(rc.heartbeat_task(budget, interval_s=30.0))

    async def run_one(iid: str, a: int, b: int, prompt_ab: str, prompt_ba: str | None):
        calls = []
        for vote_idx in range(args.votes_per_pair):
            primary = await one_call(
                client, deployment, sem, prompt_ab, args.temperature, out_dir,
                {"iid": iid, "patch_a_idx": a, "patch_b_idx": b, "vote_idx": vote_idx, "swap": 0},
                budget,
            )
            swap = None
            if prompt_ba is not None:
                swap = await one_call(
                    client, deployment, sem, prompt_ba, args.temperature, out_dir,
                    {"iid": iid, "patch_a_idx": a, "patch_b_idx": b, "vote_idx": vote_idx, "swap": 1},
                    budget,
                )
                swap["parsed_normalized"] = normalize_swap(swap.get("parsed"))
            calls.append({"vote_idx": vote_idx, "primary": primary, "swap": swap})
        return iid, a, b, calls

    coros = []
    for iid in instances:
        fields = fields_by_iid.get(iid, {})
        ftp = fields.get("fail_to_pass", [])
        base = baseline_trace(iid, ftp, eval_index)
        patches = patches_by_iid.get(iid, {})
        for a, b in itertools.combinations(args.k_indices, 2):
            if (iid, a, b) in done:
                continue
            if not (patches.get(a, "") or "").strip() or not (patches.get(b, "") or "").strip():
                row = {
                    "instance_id": iid,
                    "patch_a_idx": a,
                    "patch_b_idx": b,
                    "votes": [],
                    "skip_reason": "empty_patch",
                }
                with votes_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                continue
            prompt_ab = build_prompt(iid, a, b, patches, fields, base)
            prompt_ba = build_prompt(iid, b, a, patches, fields, base) if args.position_swap else None
            coros.append(run_one(iid, a, b, prompt_ab, prompt_ba))

    print(f"[anchored] queued {len(coros)} pair rows, votes_per_pair={args.votes_per_pair}, position_swap={args.position_swap}")
    completed = 0
    for fut in asyncio.as_completed([asyncio.create_task(c) for c in coros]):
        iid, a, b, calls = await fut
        row = {
            "instance_id": iid,
            "patch_a_idx": a,
            "patch_b_idx": b,
            "votes": calls,
        }
        with votes_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        completed += 1
        if completed % 50 == 0:
            print(f"[anchored] rows {completed}/{len(coros)} cost=${budget.spent_usd:.2f}")

    hb.cancel()
    try:
        await hb
    except asyncio.CancelledError:
        pass
    print(f"[anchored] done rows={completed} calls={budget.n_calls} cost=${budget.spent_usd:.2f} parse_failures={budget.abstains_comparator}")


def representative_winner(row: dict[str, Any]) -> tuple[str, int]:
    counts = Counter()
    conf_sum = Counter()
    for vote in row.get("votes", []):
        parsed = (vote.get("primary") or {}).get("parsed")
        if parsed:
            counts[parsed["winner"]] += 1
            conf_sum[parsed["winner"]] += int(parsed.get("confidence") or 1)
        swap = vote.get("swap")
        if swap:
            norm = swap.get("parsed_normalized")
            if norm:
                counts[norm["winner"]] += 1
                conf_sum[norm["winner"]] += int(norm.get("confidence") or 1)
    if not counts:
        return "TIE", 1
    best = sorted(("A", "B", "TIE"), key=lambda w: (-counts[w], -conf_sum[w], w))[0]
    avg_conf = round(conf_sum[best] / max(1, counts[best]))
    return best, max(1, min(5, avg_conf))


def behavior_flags(summary: str) -> dict[str, float]:
    # LEAKAGE FIX (2026-05-05): drop resolved_reward — that field was scraped
    # from the same `resolved=...` prompt line that we now omit. Keeping the
    # reward keyed off the (now-absent) text would be a no-op, but explicit
    # zero documents that the aggregator never had legitimate access to the
    # oracle label.
    s = (summary or "").lower()
    has_ftp_fail = "fail_to_pass.failure" in s or "fail_to_pass.error" in s
    has_ptp_fail = "pass_to_pass.failure" in s or "pass_to_pass.error" in s
    has_ptf = "pass_to_fail.failure" in s or "pass_to_fail.error" in s
    return {
        "resolved_reward": 0.0,  # NEVER use oracle label for selection
        "same_failure_penalty": -1.5 if has_ftp_fail else 0.0,
        "new_failure_penalty": -1.0 if (has_ptp_fail or has_ptf) else 0.0,
    }


def select_from_scores(scores: dict[int, float], k_list: list[int]) -> int:
    return min(k_list, key=lambda k: (-scores.get(k, 0.0), k))


def infer_tie_margin(row: dict[str, Any]) -> str:
    """When the model returned TIE, use its own decisive text only if it
    clearly names one side. Otherwise keep TIE."""
    text_parts = []
    confs = []
    for vote in row.get("votes", []):
        for parsed in ((vote.get("primary") or {}).get("parsed"),
                       (vote.get("swap") or {}).get("parsed_normalized")):
            if not parsed:
                continue
            confs.append(int(parsed.get("confidence") or 1))
            text_parts.append(str(parsed.get("decisive_factor") or ""))
    if confs and max(confs) < 4:
        return "TIE"
    text = " ".join(text_parts).lower()
    a_hits = sum(1 for pat in ("patch a", " a ", "a has", "a shows", "a fixes") if pat in text)
    b_hits = sum(1 for pat in ("patch b", " b ", "b has", "b shows", "b fixes") if pat in text)
    if a_hits > b_hits:
        return "A"
    if b_hits > a_hits:
        return "B"
    return "TIE"


def aggregate_instance(
    rows: list[dict[str, Any]],
    patch_behaviors: dict[int, str],
    k_list: list[int],
    *,
    tie_policy: str = "margin",
) -> tuple[dict[str, int], dict[str, Any]]:
    methods = ["copeland", "forced_margin", "confidence_weighted", "anti_tie"]
    if patch_behaviors:
        methods.append("execution_priority")
    scores = {m: {k: 0.0 for k in k_list} for m in methods}
    dist = Counter()

    for row in rows:
        a, b = int(row["patch_a_idx"]), int(row["patch_b_idx"])
        winner, conf = representative_winner(row)
        forced_winner = infer_tie_margin(row) if winner == "TIE" and tie_policy == "margin" else winner
        dist[winner] += 1
        if winner == "A":
            scores["copeland"][a] += 1; scores["copeland"][b] -= 1
            scores["confidence_weighted"][a] += conf; scores["confidence_weighted"][b] -= conf
            scores["anti_tie"][a] += 1; scores["anti_tie"][b] -= 1
        elif winner == "B":
            scores["copeland"][b] += 1; scores["copeland"][a] -= 1
            scores["confidence_weighted"][b] += conf; scores["confidence_weighted"][a] -= conf
            scores["anti_tie"][b] += 1; scores["anti_tie"][a] -= 1
        else:
            scores["anti_tie"][a] -= 0.25; scores["anti_tie"][b] -= 0.25
        if forced_winner == "A":
            scores["forced_margin"][a] += 1; scores["forced_margin"][b] -= 1
        elif forced_winner == "B":
            scores["forced_margin"][b] += 1; scores["forced_margin"][a] -= 1

    if "execution_priority" in scores:
        for k in k_list:
            f = behavior_flags(patch_behaviors.get(k, ""))
            scores["execution_priority"][k] = (
                scores["confidence_weighted"][k]
                + f["resolved_reward"]
                + f["same_failure_penalty"]
                + f["new_failure_penalty"]
            )

    selected = {m: select_from_scores(scores[m], k_list) for m in methods}
    return selected, {"scores": scores, "winner_distribution": dict(dist)}


def load_direct_binary_selector(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    by_iid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            votes = row.get("binary_votes") or []
            yes = sum(1 for v in votes if v.get("resolves") is True)
            by_iid[row["instance_id"]].append((int(row["patch_idx"]), yes))
    for iid, vals in by_iid.items():
        out[iid] = sorted(vals, key=lambda x: (-x[1], x[0]))[0][0]
    return out


def summarize(args, instances, fields_by_iid, truth, report_index) -> None:
    out_dir = Path(args.output_dir)
    votes_path = out_dir / "pairwise_dynamic_comparator_votes.jsonl"
    result_path = out_dir / "per_instance_tournament_results.jsonl"
    summary_path = out_dir / "summary.json"
    rows_by_iid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_failures = 0
    raw_votes = 0
    if votes_path.exists():
        with votes_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_by_iid[row["instance_id"]].append(row)
                for vote in row.get("votes", []):
                    raw_votes += 1
                    if (vote.get("primary") or {}).get("parsed") is None:
                        parse_failures += 1
                    if vote.get("swap") and (vote["swap"].get("parsed_normalized") is None):
                        parse_failures += 1

    db_selectors = {
        "direct_binary_hints_full500": load_direct_binary_selector(REPO / "outputs/direct_binary_hints_full500/binary_votes.jsonl"),
        "direct_binary_dynamic_hints_full500": load_direct_binary_selector(REPO / "outputs/direct_binary_dynamic_hints_full500/binary_votes.jsonl"),
        "direct_binary_dynamic_hints_xhigh_greedy8_full500": load_direct_binary_selector(REPO / "outputs/direct_binary_dynamic_hints_xhigh_greedy8_full500/binary_votes.jsonl"),
    }

    method_hits = Counter()
    method_total = Counter()
    selected_k_sum = Counter()
    win_by_k = Counter()
    selected_rows = []
    tie_count = 0
    matchup_count = 0
    baseline_hits = baseline_total = 0
    db_hits = Counter()
    db_total = Counter()

    with result_path.open("w") as out:
        for iid in sorted(rows_by_iid):
            fields = fields_by_iid.get(iid, {})
            ftp = fields.get("fail_to_pass", [])
            patch_behaviors = {}
            if args.allow_cached_eval_traces:
                patch_behaviors = {k: report_summary(report_index.get((iid, k)), iid, ftp) for k in args.k_indices}
            selected, detail = aggregate_instance(
                rows_by_iid[iid], patch_behaviors, args.k_indices, tie_policy=args.tie_policy
            )
            for w, n in Counter(detail["winner_distribution"]).items():
                if w == "TIE":
                    tie_count += n
                matchup_count += n
            k_wlt = {str(k): {"wins": 0, "losses": 0, "ties": 0} for k in args.k_indices}
            for row in rows_by_iid[iid]:
                a, b = int(row["patch_a_idx"]), int(row["patch_b_idx"])
                w, _conf = representative_winner(row)
                if w == "A":
                    k_wlt[str(a)]["wins"] += 1
                    k_wlt[str(b)]["losses"] += 1
                elif w == "B":
                    k_wlt[str(b)]["wins"] += 1
                    k_wlt[str(a)]["losses"] += 1
                else:
                    k_wlt[str(a)]["ties"] += 1
                    k_wlt[str(b)]["ties"] += 1
            rec = {
                "instance_id": iid,
                "selected": selected,
                "selected_patch_ids": {
                    method: f"{iid}:proposer_{k}" for method, k in selected.items()
                },
                "details": detail,
                "win_loss_tie_by_k_index": k_wlt,
                "truth": truth.get(iid, {}),
            }
            out.write(json.dumps(rec) + "\n")
            selected_rows.append(rec)
            for method, k in selected.items():
                selected_k_sum[method] += k
                if k in truth.get(iid, {}):
                    method_total[method] += 1
                    if truth[iid][k]:
                        method_hits[method] += 1
                selected_key = f"{method}:k{k}"
                win_by_k[selected_key] += 1
            for k, vals in k_wlt.items():
                for bucket, val in vals.items():
                    win_by_k[f"k{k}:{bucket}"] += val
            baseline_k = args.k_indices[0]
            if baseline_k in truth.get(iid, {}):
                baseline_total += 1
                if truth[iid][baseline_k]:
                    baseline_hits += 1
            for name, sel in db_selectors.items():
                if iid in sel and sel[iid] in truth.get(iid, {}):
                    db_total[name] += 1
                    if truth[iid][sel[iid]]:
                        db_hits[name] += 1

    methods = ["copeland", "forced_margin", "confidence_weighted", "anti_tie"]
    if args.allow_cached_eval_traces:
        methods.append("execution_priority")
    summary = {
        "total_instances": len(rows_by_iid),
        "pair_rows": sum(len(v) for v in rows_by_iid.values()),
        "raw_vote_groups": raw_votes,
        "tie_rate": tie_count / matchup_count if matchup_count else 0.0,
        "parse_failure_count": parse_failures,
        "k_list": args.k_indices,
        "win_loss_tie_distribution_by_k_index": dict(win_by_k),
        "aggregation_methods": {},
        "first_k_baseline": {
            "k_index": args.k_indices[0],
            "resolved": baseline_hits,
            "total": baseline_total,
            "rate": baseline_hits / baseline_total if baseline_total else None,
        },
        "direct_binary_comparison": {},
    }
    for method in methods:
        total = method_total[method]
        hits = method_hits[method]
        summary["aggregation_methods"][method] = {
            "resolved": hits,
            "total_with_truth": total,
            "resolve_rate": hits / total if total else None,
            "average_selected_k_index": selected_k_sum[method] / len(rows_by_iid) if rows_by_iid else None,
        }
    for name in db_selectors:
        total = db_total[name]
        hits = db_hits[name]
        summary["direct_binary_comparison"][name] = {
            "found": bool(db_selectors[name]),
            "resolved": hits,
            "total_with_truth": total,
            "resolve_rate": hits / total if total else None,
        }

    usage = out_dir / "usage.jsonl"
    if usage.exists():
        total_cost = 0.0
        n = 0
        with usage.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                total_cost += float(row.get("cost_usd") or 0.0)
                n += 1
        summary["cost_estimate"] = {"usage_rows": n, "estimated_cost_usd": total_cost}
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def prepare_instances(args, all_iids: list[str]) -> list[str]:
    requested = read_instances_path(args.instances_path)
    if requested:
        iids = [iid for iid in requested if iid in set(all_iids)]
    else:
        iids = list(all_iids)
    if args.pilot and not args.max_instances:
        iids = rc.pilot_instance_subset(iids)
    if args.max_instances:
        iids = iids[: args.max_instances]
    return iids


def dry_run(args, instances, patches_by_iid, fields_by_iid, eval_index, report_index) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for iid in instances[:2]:
        fields = fields_by_iid.get(iid, {})
        ftp = fields.get("fail_to_pass", [])
        base = baseline_trace(iid, ftp, eval_index)
        patches = patches_by_iid[iid]
        for a, b in list(itertools.combinations(args.k_indices, 2))[:2]:
            samples.append({
                "instance_id": iid,
                "patch_a_idx": a,
                "patch_b_idx": b,
                "prompt": build_prompt(iid, a, b, patches, fields, base),
            })
    path = out_dir / "sample_prompts.json"
    with path.open("w") as f:
        json.dump(samples, f, indent=2)
        f.write("\n")
    print(f"[dry-run] wrote {len(samples)} prompts to {path}")


def main() -> int:
    args = parse_args()
    args.k_indices = parse_k_list(args.k_list)
    out_dir = Path(args.output_dir)
    patches_dir = Path(args.patches_dir)
    reports_dir = Path(args.harness_reports_dir)

    print(f"[load] patches from {patches_dir}")
    print(f"[load] proposer K-list {args.k_indices}")
    patches_by_iid, all_iids = load_patches(patches_dir, args.k_indices)
    instances = prepare_instances(args, all_iids)
    print(f"[load] selected {len(instances)} instances")
    fields_by_iid = load_dataset_fields(instances)
    truth = load_truth(reports_dir, args.k_indices)
    report_index = {}
    eval_index = {}
    if args.allow_cached_eval_traces:
        report_index = load_report_index(reports_dir, args.k_indices)
        old_eval_base = dyn.EVAL_BASE
        dyn.EVAL_BASE = reports_dir
        try:
            eval_index = dyn._index_failing_eval_dirs()
        finally:
            dyn.EVAL_BASE = old_eval_base

    if args.dry_run:
        dry_run(args, instances, patches_by_iid, fields_by_iid, eval_index, report_index)
        summarize(args, instances, fields_by_iid, truth, report_index)
        return 0

    if args.summary_only:
        summarize(args, instances, fields_by_iid, truth, report_index)
        return 0

    asyncio.run(run_pairs(args, instances, patches_by_iid, fields_by_iid, eval_index, report_index))
    summarize(args, instances, fields_by_iid, truth, report_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
