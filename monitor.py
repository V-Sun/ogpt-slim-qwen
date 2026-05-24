#!/usr/bin/env python3
"""Live monitor for active comparator/selector runs. Updates every 5 seconds."""
import json, os, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parent

RUNS = [
    {
        "label": "v2 nano comparator (R=5)",
        "votes_file": "outputs/blind_comparator_v2_nano_full/pairwise_dynamic_comparator_votes.jsonl",
        "log_glob":   "logs/blind_comparator_v2_nano_full_R5_*.log",
        "total_pairs": 9589,
        "votes_per_pair": 5,
    },
    {
        "label": "v2 mini comparator (M=3)",
        "votes_file": "outputs/blind_comparator_v2_mini_m3_full/pairwise_dynamic_comparator_votes.jsonl",
        "log_glob":   "logs/blind_comparator_v2_mini_m3_*.log",
        "total_pairs": 10630,
        "votes_per_pair": 3,
    },
    {
        "label": "blind selector nano (R=5)",
        "votes_file": "outputs/blind_selector_greedy8_R5_xhigh/binary_votes.jsonl",
        "log_glob":   "logs/blind_selector_high_*.log",
        "total_pairs": 3610,   # 3952 - 342 done at start
        "votes_per_pair": 5,
        "done_offset": 342,    # rows already done before this run started
    },
    {
        "label": "mini selector (R=3)",
        "votes_file": "outputs/blind_selector_mini_greedy8_R3/binary_votes.jsonl",
        "log_glob":   "logs/blind_selector_mini_*.log",
        "total_pairs": 3952,
        "votes_per_pair": 3,
    },
    {
        "label": "trace comparator nano (R=3)",
        "votes_file": "outputs/trace_comparator_nano_greedy8/pairwise_trace_comparator_votes.jsonl",
        "log_glob":   "logs/trace_comparator_nano_*.log",
        "total_pairs": 13073,
        "votes_per_pair": 3,
        "done_offset": 759,   # rows written by earlier partial run
    },
]

RATE_RE     = re.compile(r'rate=([\d.]+)/min')
COST_RE     = re.compile(r'cost=\$([\d.]+)')
INFLIGHT_RE = re.compile(r'in_flight=(\d+)')
COMPLETED_RE= re.compile(r'completed=(\d+)')
ISSUED_RE   = re.compile(r'issued=(\d+)')


def latest_log(glob: str) -> Path | None:
    # Sort by filename (contains YYYYMMDD_HHMMSS timestamp) so we always get
    # the most recently launched run, not the most recently touched file.
    files = sorted(REPO.glob(glob), key=lambda p: p.name, reverse=True)
    return files[0] if files else None


def tail_heartbeat(log_path: Path, n: int = 50) -> str:
    """Read last n lines of log file efficiently."""
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(4096, size)
            f.seek(-chunk, 2)
            return f.read().decode(errors="replace")
    except Exception:
        return ""


def parse_heartbeat(log_path: Path) -> dict:
    text = tail_heartbeat(log_path)
    lines = [l for l in text.splitlines() if "heartbeat" in l.lower() or "rate=" in l]
    if not lines:
        return {}
    last = lines[-1]
    rate_m = RATE_RE.search(last)
    cost_m = COST_RE.search(last)
    inf_m  = INFLIGHT_RE.search(last)
    return {
        "rate":      float(rate_m.group(1))  if rate_m  else 0.0,
        "cost":      float(cost_m.group(1))  if cost_m  else 0.0,
        "in_flight": int(inf_m.group(1))     if inf_m   else 0,
        "completed": int(COMPLETED_RE.search(last).group(1)) if COMPLETED_RE.search(last) else 0,
        "issued":    int(ISSUED_RE.search(last).group(1))    if ISSUED_RE.search(last)    else 0,
    }


def count_rows(path: Path) -> tuple[int, int]:
    """Return (total_rows, real_vote_rows) — excludes skip_reason rows."""
    total = real = 0
    if not path.exists():
        return 0, 0
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    r = json.loads(line)
                    if not r.get("skip_reason") and (r.get("votes") or r.get("binary_votes")):
                        real += 1
                except Exception:
                    pass
    except Exception:
        pass
    return total, real


def fmt_eta(minutes: float) -> str:
    if minutes <= 0 or minutes > 9999:
        return "—"
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def bar(fraction: float, width: int = 24) -> str:
    filled = int(fraction * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def render():
    os.system("clear")
    print(f"  Orchestra-GPT Run Monitor  —  {time.strftime('%H:%M:%S')}\n")

    for run in RUNS:
        vf   = REPO / run["votes_file"]
        log  = latest_log(run["log_glob"])
        hb   = parse_heartbeat(log) if log else {}

        total_rows, real_rows = count_rows(vf)
        done_offset = run.get("done_offset", 0)
        real_rows = max(0, real_rows - done_offset)

        total_pairs   = run["total_pairs"]
        votes_per     = run["votes_per_pair"]
        total_calls   = total_pairs * votes_per

        rate          = hb.get("rate", 0.0)
        cost          = hb.get("cost", 0.0)
        calls_done    = hb.get("completed", 0)  # actual API calls completed per heartbeat
        remaining_calls = max(0, total_calls - calls_done)
        pct           = min(calls_done / total_calls, 1.0) if total_calls else 0.0
        eta_min       = remaining_calls / rate if rate > 0 else 0.0

        label = run["label"]
        print(f"  {label}")
        flushed_rows = max(0, real_rows - done_offset)
        print(f"  {bar(pct)} {pct*100:5.1f}%  calls: {calls_done:,}/{total_calls:,}  rows flushed: {flushed_rows}")
        print(f"  rate: {rate:.0f}/min   ETA: {fmt_eta(eta_min)}   cost: ${cost:.2f}")
        if not log:
            print("  [no log found]")
        print()

    print("  (ctrl-c to exit)")


if __name__ == "__main__":
    try:
        while True:
            render()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nExiting.")
