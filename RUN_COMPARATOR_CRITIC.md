# Running Critic + Comparator Rounds on Your K=8 Proposers

This is the **updated test_patch-aware** critic + comparator pipeline (passes
`test_patch` + `FAIL_TO_PASS` test names to the LLM judge as benchmark
metadata, not as eval results — pre-eval blind per the experiment guardrail).

## Assumptions

You already have:
- 8 proposer trajectories per task (one per K = 0..7)
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

## File map

That's it — 6 files:

| file | role |
|---|---|
| `direct_binary_dynamic_hints.py` | **CRITIC** — binary YES/NO votes per patch with `test_patch` + `FAIL_TO_PASS` |
| `anchored_dynamic_comparator_v2.py` | **COMPARATOR** — pairwise A-vs-B votes with `test_patch` + `FAIL_TO_PASS` |
| `aggregate_and_report.py` | Offline aggregation, all rules (Copeland / bracket / KOTH / Swiss) + hybrid configs |
| `pick_tau_and_survivors.py` | Selector threshold tuning (pick τ to admit N survivors per task) |
| `run_ceiling.py` | Shared utility (instance loader, ceiling calc) — imported by both runners |
| `orchestra/comparator_prompts.py` + `orchestra/critic_prompts.py` | Refined prompt templates (optional — runners have inline prompts already) |

## Quick start

### Path overrides

The runners hard-code `/home/vsun/orchestra-gpt` as the data repo. Change these
near the top of each script:

```python
DATA_REPO            = Path("/your/path/to/your-proposers-repo")
DEFAULT_PATCHES_DIR  = DATA_REPO / "outputs" / "your_oracle_preds_dir"
DEFAULT_REPORTS_DIR  = DATA_REPO / "logs" / "run_evaluation"
```

Or symlink to mirror that layout.

### Predictions format expected

Each `proposer_K.jsonl` row:
```json
{"instance_id": "...", "model_patch": "diff --git ...", "model_name_or_path": "your-model"}
```

Place these at `DATA_REPO / outputs / oracle_preds_v3 / proposer_{0..7}.jsonl`
(or update `DEFAULT_PATCHES_DIR`).

### Env setup

Default targets Azure OpenAI:
```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="..."             # your Azure endpoint
export AZURE_API_VERSION="2024-12-01-preview"
```

To use a local vLLM endpoint instead (e.g. your own model running OpenAI-compatible):
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
per `(instance_id, A, B)` with R=5 votes (position-swap on by default).

### Step 3: Aggregate (offline, $0)

```bash
python3 aggregate_and_report.py \
  --selector-votes outputs/direct_binary_hints_full500/binary_votes.jsonl \
  --comparator-votes outputs/blind_comparator_v2_fixable43/pairwise_votes.jsonl \
  --output outputs/aggregated_results
```

Evaluates all aggregation rules (Copeland, bracket, KOTH, Swiss) + hybrid
configs (selector top-N → comparator on survivors).

### Step 4: Choose tau (selection threshold) — optional

```bash
python3 pick_tau_and_survivors.py \
  --binary-votes outputs/direct_binary_hints_full500/binary_votes.jsonl \
  --target-survivors 5
```

Returns the τ that admits the target number of patches per task.

## Tips from past pain points

1. **Position swap matters.** The comparator runs each pair as `A vs B` AND
   `B vs A` and de-swaps automatically — single-order judgments have strong
   lead bias.

2. **R=5 votes per decision** is the sweet spot. R=15 caps the improvement;
   R=3 is noisy.

3. **`BUDGET_CAP`** constant (default $400) is a hard stop. Bump for full runs.

4. **Concurrency.** `--concurrency 64` is safe for Azure TPM. Local vLLM can
   go higher (limited only by `max_num_seqs`).

5. **Winning config.** In our experiments, **Copeland with selector-top-3
   pre-filter** (`hybrid_sel_top3_comparator`) consistently beat plain Copeland
   by 2–4 pts.
