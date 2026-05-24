# Running Critic + Comparator Rounds on Your K=8 Proposers

This bundle gives you the **updated test_patch-aware** critic and comparator scripts
(the ones that pass `test_patch` + `FAIL_TO_PASS` names to the LLM judge as
benchmark metadata, not as eval results — pre-eval blind per the experiment
guardrail).

## Assumptions

You already have:
- 8 proposer trajectories per task (one per K = 0..7)
- Harness eval reports (you don't strictly need these — only used by some analysis scripts)
- Predictions JSONL per K (one row per `{instance_id, model_patch, model_name_or_path}`)

## The pipeline

```
                                              ┌──────────────────────┐
                                              │  K=8 proposer        │
                                              │  predictions         │
                                              └────────┬─────────────┘
                                                       │
                              ┌────────────────────────┴────────────────┐
                              ▼                                         ▼
                  direct_binary_dynamic_hints.py            anchored_dynamic_comparator_v2.py
                  (CRITIC: binary YES/NO per patch          (COMPARATOR: pairwise A-vs-B
                   with test_patch + FAIL_TO_PASS)           with test_patch + FAIL_TO_PASS)
                              │                                         │
                              ▼                                         ▼
                       binary_votes.jsonl                       pairwise_votes.jsonl
                              │                                         │
                              └─────────────────┬───────────────────────┘
                                                ▼
                                       aggregate_and_report.py
                                       (Copeland / bracket / KOTH / hybrid)
                                                │
                                                ▼
                                       chosen_patch per task
                                                │
                                                ▼
                                       swebench harness eval
                                       → final accuracy %
```

## Quick start

### Path overrides

The scripts hard-code `/home/vsun/orchestra-gpt` as the data repo. Change these:

```python
# In each script's header:
DATA_REPO            = Path("/your/path/to/your-proposers-repo")
DEFAULT_PATCHES_DIR  = DATA_REPO / "outputs" / "your_oracle_preds_dir"
DEFAULT_REPORTS_DIR  = DATA_REPO / "logs" / "run_evaluation"
```

Or just symlink your data dir to mirror that layout — usually fastest.

### Predictions format expected

Each `proposer_K.jsonl` row:
```json
{"instance_id": "...", "model_patch": "diff --git ...", "model_name_or_path": "your-model"}
```

Place these at `DATA_REPO / outputs / oracle_preds_v3 / proposer_{0..7}.jsonl` (or
update `DEFAULT_PATCHES_DIR`).

### Env setup

Default targets Azure OpenAI:
```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="..."        # your Azure endpoint
export AZURE_API_VERSION="2024-12-01-preview"
```

To use a local vLLM endpoint instead (e.g. your own Qwen running OpenAI-compatible):
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
# Pass --model openai/your-model-name
```

### Step 1: Run critic (binary votes)

```bash
python3 direct_binary_dynamic_hints.py \
  --model azure/gpt-5.4-nano \
  --mode full \
  --votes 5 \
  --concurrency 64
```

Outputs `outputs/direct_binary_hints_full500/binary_votes.jsonl` — one row per
`(instance_id, patch_idx)` with R=5 votes.

Modes: `smoke` (5 instances), `pilot` (50), `full` (all 500).

### Step 2: Run comparator (pairwise votes)

```bash
python3 anchored_dynamic_comparator_v2.py \
  --model azure/gpt-5.4-nano \
  --mode full \
  --votes 5 \
  --concurrency 64
```

Outputs `outputs/blind_comparator_v2_fixable43/pairwise_votes.jsonl` — one row
per `(instance_id, A, B)` with R=5 votes (and position-swap by default).

### Step 3: Aggregate (offline, $0)

```bash
python3 aggregate_and_report.py \
  --selector-votes outputs/direct_binary_hints_full500/binary_votes.jsonl \
  --comparator-votes outputs/blind_comparator_v2_fixable43/pairwise_votes.jsonl \
  --output outputs/aggregated_results
```

This evaluates **all aggregation rules** (Copeland, bracket, KOTH, Swiss) plus
hybrid configs (selector top-N → comparator on survivors) and reports the best.

### Step 4: Choose tau (selection threshold)

```bash
python3 pick_tau_and_survivors.py \
  --binary-votes outputs/direct_binary_hints_full500/binary_votes.jsonl \
  --target-survivors 5
```

Returns the threshold τ that admits the target number of patches per task into
the comparator round.

## File map (what's in this bundle)

### Core runners (the things you actually invoke)
- `direct_binary_dynamic_hints.py` — **critic**, R=N binary votes per patch, sees test_patch + FAIL_TO_PASS
- `anchored_dynamic_comparator_v2.py` — **comparator**, R=N pairwise votes, sees test_patch + FAIL_TO_PASS
- `aggregate_and_report.py` — offline aggregation, picks chosen patch
- `pick_tau_and_survivors.py` — selector threshold tuning

### Earlier blind variants (no test_patch — pre-guardrail baseline)
- `blind_selector.py` — binary, issue + patch text only
- `blind_comparator.py` — pairwise, issue + patches only

### Earlier hint variants (text hints only, no test_patch)
- `direct_binary.py` — base binary
- `direct_binary_hints.py` — binary + FAIL_TO_PASS names + changed files (no test_patch)
- `comparator_dynamic_hints.py` — pairwise with dynamic hints
- `anchored_dynamic_comparator.py` — older anchored comparator
- `anchored_judgment.py` — Likert-scale judgment baseline

### Shared utility
- `run_ceiling.py` — instance loader + ceiling computation utilities (imported by all)
- `extract_d3_cache.py` — extracts test_patch + FAIL_TO_PASS from SWE-Bench dataset cache

### Analysis
- `analyze_comparator_pilot.py`, `analyze_comparator_voter_ablation.py`,
  `analyze_dynamic_R10.py`, `analyze_gated_comparator_thresholds.py`,
  `analyze_greedy8_bridge.py`, `analyze_greedy8_v2.py`,
  `analyze_k_sweep_ablation.py`, `analyze_top_n_ablation.py`,
  `analyze_voter_ablation.py`, `analyze_xhigh_greedy8.py` — various post-hoc
  ablations and analyses
- `generate_dynamic_hints_summary.py`, `generate_full500_summary.py` — final-report generators
- `offline_analysis.py` — generic offline aggregation utilities
- `king_of_the_hill.py` — KOTH-specific runner
- `monitor.py` — live-progress monitor for long runs

### Orchestration shells (your home-machine workflow; adapt freely)
- `run_overnight_full500.sh` — full 500 sweep
- `run_comparator_hints_chain.sh` — chains critic → comparator → aggregate
- `run_dynamic_hints_chain.sh` — variant chain
- `run_post_r10_analysis.sh` — post-R10 analysis
- `run_step2_offline_ablations.sh` — offline-only ablation sweep

### Tracers (debugging individual decisions)
- `trace_selector.py`, `trace_comparator.py` — replay & inspect one vote in detail

### Plots
- `make_plots.py`, `make_plot3.py`, `make_negative_results_plots.py`

### Prompt templates (used by the older orchestra/ code path; reference quality)
- `orchestra/comparator_prompts.py`, `orchestra/critic_prompts.py`

## Notes from past pain points

1. **Position swap matters.** The comparator runs each pair (A, B) twice — once
   as `A vs B`, once as `B vs A`. Single-order judgments have a strong lead
   bias. De-swapping is automatic.

2. **Vote count.** R=5 per decision is the sweet spot. R=15 caps the
   improvement; R=3 is noisy.

3. **Cost cap.** Scripts have a `BUDGET_CAP` constant (default $400). Bump it
   for full runs.

4. **Concurrency.** `--concurrency 64` is safe against Azure TPM limits for
   nano. For local vLLM, you can go higher (limited only by your max_num_seqs).

5. **Aggregation winner.** In our runs, **Copeland with selector-top-3
   pre-filter** (`hybrid_sel_top3_comparator`) consistently beat plain
   Copeland by 2-4 pts.
