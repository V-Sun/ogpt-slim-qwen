#!/usr/bin/env python3
"""Mine cached SWE-bench harness reports to build a per-(iid, K) D3 signal cache.

For each proposer K and each instance, surfaces:
  - Did the patch apply cleanly?
  - How many FAIL_TO_PASS tests turned green vs stayed red?
  - Did the patch introduce new failures (PASS_TO_FAIL)?
  - Did pre-existing PASS_TO_PASS regress?
  - Short (truncated) excerpt of the post-patch test_output.txt for the
    target failing test, if findable.

Output: outputs/d3_cache.json with the structure
  {
    "<instance_id>": {
      "<K>": {
        "applied": bool,
        "resolved": bool|null,
        "ftp_pass": int, "ftp_fail": int, "ftp_pass_names": [...] capped,
        "ptf_introduced": int, "ptf_names": [...] capped,
        "ptp_regressed": int,
        "raw_excerpt": str  # short test_output.txt slice
      }
    }
  }

This is pure cache mining — no API calls, no compute beyond filesystem reads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ORCHESTRA = Path("/home/vsun/orchestra-gpt")
# Union across all per-proposer oracle eval runs that exist; most recent wins on collisions.
RUN_PATTERNS = [
    "oracle_now_20260502_114638_proposer_{k}",
    "oracle_par_20260502_004945_proposer_{k}",
    "oracle_par_20260501_230950_proposer_{k}",
    "oracle_redo_20260502_142226_proposer_{k}",
    "oracle_redo_20260502_125223_proposer_{k}",
    "oracle_finish_20260502_165542_proposer_{k}",
    "oracle_finish_20260502_174258_finalize_proposer_{k}",
    "oracle_v3_20260504_003040_proposer_{k}",
    "oracle_combined_20260501_222722_proposer_{k}",
    "oracle_per_k_20260501_225603_proposer_{k}",
    "oracle_merged_20260501_221853_proposer_{k}",
    "oracle_incr_20260502_004945_proposer_{k}",
    "oracle_parallel_20260501_230950_proposer_{k}",
    "oracle_k{k}",
]
INNER_PREFIX = "propsonly_xhigh_proposer_"

MAX_FTP_NAMES = 5
MAX_EXCERPT_CHARS = 1500


def extract_excerpt(test_output_path: Path, fail_to_pass: list[str]) -> str:
    """Pull a slice of test_output.txt around the FAIL_TO_PASS test names."""
    if not test_output_path.exists():
        return ""
    try:
        text = test_output_path.read_text(errors="ignore")
    except Exception:
        return ""
    if not fail_to_pass:
        return text[-MAX_EXCERPT_CHARS:].strip()
    for name in fail_to_pass:
        # Match the leaf method name in pytest output (e.g. "test_x" in "module.Class.test_x")
        leaf = name.split(".")[-1].split(" ")[0]
        idx = text.find(leaf)
        if idx >= 0:
            head = max(0, idx - 200)
            tail = min(len(text), idx + MAX_EXCERPT_CHARS - 200)
            return text[head:tail].strip()
    return text[-MAX_EXCERPT_CHARS:].strip()


def mine_run_dir(run_dir: Path) -> dict[str, dict]:
    """Return {iid: report_dict} for one proposer's eval run."""
    # Find inner dir with reports (could be propsonly_xhigh_proposer_K, or just direct iid dirs)
    inner_dirs = list(run_dir.glob(f"{INNER_PREFIX}*"))
    if inner_dirs:
        inner = inner_dirs[0]
    else:
        # Try direct iid dirs (e.g. oracle_k0/<iid>/)
        inner = run_dir
    out: dict[str, dict] = {}
    candidates = list(inner.iterdir()) if inner.exists() else []
    for iid_dir in candidates:
        if not iid_dir.is_dir():
            continue
        rep = iid_dir / "report.json"
        test_out = iid_dir / "test_output.txt"
        if not rep.exists():
            continue
        try:
            data = json.loads(rep.read_text())
        except Exception:
            continue
        # report.json wraps under instance_id key
        entry = data.get(iid_dir.name) or next(iter(data.values()), {})
        if not entry:
            continue
        ftp = entry.get("tests_status", {}).get("FAIL_TO_PASS", {})
        ptp = entry.get("tests_status", {}).get("PASS_TO_PASS", {})
        ptf = entry.get("tests_status", {}).get("PASS_TO_FAIL", {})
        ftp_pass = ftp.get("success", []) or []
        ftp_fail = ftp.get("failure", []) or []
        ptp_pass = ptp.get("success", []) or []
        ptp_fail = ptp.get("failure", []) or []
        ptf_pass = ptf.get("success", []) or []
        ptf_fail = ptf.get("failure", []) or []
        excerpt = extract_excerpt(test_out, ftp_fail or ftp_pass)
        out[iid_dir.name] = {
            "applied": bool(entry.get("patch_successfully_applied", False)),
            "resolved": entry.get("resolved"),
            "ftp_total": len(ftp_pass) + len(ftp_fail),
            "ftp_pass": len(ftp_pass),
            "ftp_fail": len(ftp_fail),
            "ftp_pass_names": ftp_pass[:MAX_FTP_NAMES],
            "ftp_fail_names": ftp_fail[:MAX_FTP_NAMES],
            "ptf_introduced": len(ptf_fail),
            "ptf_names": ptf_fail[:MAX_FTP_NAMES],
            "ptp_regressed": len(ptp_fail),
            "ptp_total": len(ptp_pass) + len(ptp_fail),
            "raw_excerpt": excerpt[:MAX_EXCERPT_CHARS],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orchestra", default=str(ORCHESTRA))
    p.add_argument("--out", default="/home/vsun/ogpt-slim-qwen/outputs/d3_cache.json")
    p.add_argument("--k-list", default="0,1,2,3,4,5,6,7")
    args = p.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    base = Path(args.orchestra) / "logs" / "run_evaluation"

    cache: dict[str, dict[str, dict]] = defaultdict(dict)
    summary = {"k_list": k_list, "iids_per_k": {}, "missing_per_k": {}}
    for k in k_list:
        merged: dict[str, dict] = {}
        for pat in RUN_PATTERNS:
            run_dir = base / pat.format(k=k)
            if not run_dir.exists():
                continue
            per_iid = mine_run_dir(run_dir)
            for iid, entry in per_iid.items():
                # Prefer entries with successful patch_applied OR resolved=True; else first-seen
                ex = merged.get(iid)
                if ex is None:
                    merged[iid] = entry
                elif (entry.get("resolved") is True and ex.get("resolved") is not True):
                    merged[iid] = entry
                elif (entry.get("applied") and not ex.get("applied")):
                    merged[iid] = entry
        summary["iids_per_k"][k] = len(merged)
        for iid, entry in merged.items():
            cache[iid][str(k)] = entry
        print(f"[d3] K={k}: {len(merged)} iids cached (union across patterns)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=1) + "\n")
    print(f"\n[d3] wrote cache for {len(cache)} iids to {out_path}")

    # Sample
    if cache:
        sample_iid = next(iter(cache.keys()))
        print(f"\nSample for {sample_iid}:")
        for k, entry in cache[sample_iid].items():
            print(f"  K={k}: applied={entry['applied']} resolved={entry['resolved']} "
                  f"ftp_pass={entry['ftp_pass']}/{entry['ftp_total']} "
                  f"ptf={entry['ptf_introduced']} excerpt_len={len(entry['raw_excerpt'])}")


if __name__ == "__main__":
    main()
