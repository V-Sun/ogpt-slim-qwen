# Orchestra-GPT (slim, Qwen-ready)

## What this is

Orchestra-GPT runs a three-tier committee protocol on SWE-bench Verified instances. For each task it spawns K independent generator agents (mini-swe-agent), routes each candidate patch through M reviewer LLM calls that vote on four named structural flags (with a per-flag 80% agreement gate), and resolves the survivors via an R-round Copeland tournament that performs a structured pairwise comparison with position-swap debiasing. The committee returns a single winning patch ready for the SWE-bench evaluation harness.

The runtime is model-agnostic. The default configuration targets a local vLLM server speaking the OpenAI-compatible API; the same code path also handles Azure OpenAI (including the Responses API with reasoning effort) by flipping a single environment variable.

## Hardware

- `Qwen/Qwen3.6-35B-A3B` (base, default): roughly 35B total parameters / 3B active per token, 256K native context. Fits in about 46 GB of VRAM at FP8 across two GPUs with `tensor-parallel-size=2`.
- `Qwen/Qwen3-Coder-Next` (alternative): same VRAM profile as the base model. Recommended for maximum SWE-bench performance with native tool calling.
- `Qwen/Qwen2.5-Coder-32B-Instruct` (lighter fallback): single high-VRAM GPU (~24 GB at AWQ/INT4 quant, more at FP8). Drop in if you do not have two GPUs available.

## Quickstart

```bash
# 1. Install vLLM and orchestra dependencies (in two separate venvs is fine).
pip install vllm
pip install -r requirements.txt

# 2. Launch a vLLM server. Pick one of the two commands below.

#    Base Qwen (default):
vllm serve Qwen/Qwen3.6-35B-A3B --port 8000 \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --reasoning-parser qwen3

#    Qwen-Coder (alternative):
vllm serve Qwen/Qwen3-Coder-Next --port 8000 \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder

# 3. Configure the orchestrator.
cp .env.example .env
# Edit .env: confirm OPENAI_BASE_URL, OPENAI_API_KEY (vLLM accepts "EMPTY"),
# and MODEL_NAME match what you launched in step 2.

# 4. Run the committee on one SWE-bench Verified task as a smoke test.
python orchestra/run.py 1 2 1 1 smoke_test
# Output: outputs/smoke_test/predictions.jsonl

# 5. Score with the SWE-bench harness.
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path outputs/smoke_test/predictions.jsonl \
  --max_workers 4 --run_id smoke_test
```

## Configuration

Every environment variable the pipeline reads, with defaults and meaning, is listed in `.env.example`. The most important ones:

| Var | Default | Meaning |
|---|---|---|
| `MODEL_PROVIDER` | `openai_compatible` | `openai_compatible` for vLLM/SGLang/Ollama, `azure_openai` for Azure OpenAI |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | URL of your OpenAI-compatible server |
| `OPENAI_API_KEY` | `EMPTY` | API key; vLLM ignores this |
| `MODEL_NAME` | `Qwen/Qwen3.6-35B-A3B` | Model id served by your endpoint |
| `GENERATOR_MODEL`, `REVIEWER_MODEL`, `COMPARATOR_MODEL` | `MODEL_NAME` | Per-role overrides; mix and match for ablations |
| `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_DEPLOYMENT`, `AZURE_API_VERSION` | unset | Used only when `MODEL_PROVIDER=azure_openai` |
| `AZURE_REASONING_EFFORT` | `""` | Set to `low`/`medium`/`high` to enable Azure Responses API with reasoning |
| `ORCHESTRA_API_TIMEOUT_SECONDS` | `600` | Per-call SDK timeout |
| `ORCHESTRA_PROPOSER_TIMEOUT_SECONDS` | `3600` | Wall-clock deadline across the K-proposer pool |
| `ORCHESTRA_PROPOSER_CONCURRENCY` | `10` | Max parallel proposers |
| `ORCHESTRA_TASK_CONCURRENCY` | `1` | Number of SWE-bench tasks to run in parallel |
| `ORCHESTRA_ENVIRONMENT_CLASS` | `docker` | `docker`, `singularity`, or `local`. Set to `singularity` on Apptainer-based HPC clusters; orchestra skips its Docker shell-outs and mini-swe-agent builds the right environment automatically (image is auto-prefixed with `docker://`). |
| `ORCHESTRA_PROPOSER_CONFIG_OVERLAY` | unset | Path to a YAML file that gets deep-merged on top of the mini-swe-agent benchmark base config before our code-side overrides. `configs/qwen_local.yaml` is a ready-made overlay with Qwen sampling defaults. |

## Running the committee

```bash
# python orchestra/run.py NUM_TASKS K M R [RUN_ID]
```

`NUM_TASKS` is how many instances of `princeton-nlp/SWE-Bench_Verified` to process (the script picks the first N in dataset order). `K`, `M`, `R` are the committee sizes. `RUN_ID` names the output directory (omit for an auto-generated name). Outputs:

- `outputs/<RUN_ID>/predictions.jsonl` — one row per instance, the winning patch only. Feed this to the SWE-bench harness.
- `outputs/<RUN_ID>/predictions_per_proposer.jsonl` — one row per `(instance_id, k_index)`. Always written, regardless of M and R. Useful for proposer-only runs and oracle / pass@K analysis.
- `outputs/<RUN_ID>/<instance_id>/proposer_{k}.traj.json` — full mini-swe-agent rollout per proposer per instance. Preserved automatically.

Both `predictions.jsonl` and the SWE-bench harness are resumable: rerun with the same `RUN_ID` to pick up where you left off.

### Run configurations

Three configurations the slim repo is hardened to support out of the box:

```bash
# Proposer-only ablation (K=16, M=0, R=0)
# 16 independent proposers per instance, no review, no comparator.
# Output: predictions_per_proposer.jsonl ONLY (16 rows per instance).
# predictions.jsonl is intentionally NOT written when M=0 — the harness
# would otherwise score proposer 0's patch and silently report a K=1
# number even though K proposers ran. Use a downstream oracle / pass@K
# selection script to compute aggregate metrics.
python orchestra/run.py 500 16 0 0 qwen_proposers_k16

# Single-agent baseline (K=1, M=0, R=0)
# One proposer per instance, no committee logic. Same M=0 rule applies:
# only predictions_per_proposer.jsonl is written (1 row per instance).
# To score with the SWE-bench harness, treat predictions_per_proposer.jsonl
# as the predictions file directly — the row shape is harness-compatible.
python orchestra/run.py 500 1 0 0 qwen_baseline_k1

# Full three-tier committee (K=10, M=5, R=3)
# Generators -> Reviewers (PASS@80% structural-flag filter) -> Copeland tournament
# with position-swap debiasing. Both predictions.jsonl (winner per instance)
# and predictions_per_proposer.jsonl (all K candidates) are written.
python orchestra/run.py 500 10 5 3 qwen_committee
```

Setting `M=0` skips the reviewer LLM calls entirely AND suppresses `predictions.jsonl`; `predictions_per_proposer.jsonl` is the only output (plus the trajectory files). Setting `R=0` skips the comparator LLM calls; if you also have `M>0`, the survivor with the lowest `k_index` wins by deterministic tiebreak and predictions.jsonl is still written.

Resumability for M=0 runs uses an atomic per-instance marker file `outputs/<RUN_ID>/.done/<instance_id>` written only after all K proposer rows have been flushed and `fsync`'d. Crashed runs leave orphan rows in `predictions_per_proposer.jsonl` (dedupe by `(instance_id, k_index)` if needed) but the instance is NOT marked done, so a rerun re-executes it cleanly.

### Running on Singularity / Apptainer (HPC)

On clusters without a Docker daemon (e.g., MIT ORCD), set two env vars before launching:

```bash
export ORCHESTRA_ENVIRONMENT_CLASS=singularity
export ORCHESTRA_PROPOSER_CONFIG_OVERLAY=configs/qwen_local.yaml   # optional, for Qwen defaults
python orchestra/run.py 500 16 0 0 qwen_proposers_k16
```

`ORCHESTRA_ENVIRONMENT_CLASS=singularity` does two things: orchestra skips every Docker shell-out (`docker image inspect`, `docker pull`, `docker ps`/`rm`, `docker rmi`), and the value gets injected into the agent config so mini-swe-agent's `get_sb_environment` builds a Singularity environment instead of a Docker one. Mini-swe-agent auto-prefixes the SWE-bench image name with `docker://` so Apptainer can pull from Docker Hub on first use.

The Docker code path is unchanged when the env var is unset or set to `docker`.

## Evaluating on SWE-bench Verified

The committee produces patches; the SWE-bench harness scores them by checking each patch out, applying it, and running the per-instance test suite inside a Docker sandbox. The harness is a separate package with heavy dependencies, so it is not in `requirements.txt`.

**Prerequisites**

- Docker daemon running, with the current user in the `docker` group (or use `sudo`).
- Disk space for the per-instance images. SWE-bench Verified has 500 tasks; each image is roughly 1 GB. The harness pulls images on demand, so the working set is bounded by `--max_workers`. Plan for at least 50 GB free if you set `--max_workers 4`. The orchestrator already removes images after each task by default (`ORCHESTRA_RMI_AFTER_TASK=1`), but the harness has its own cache.
- Python 3.10+ (same env as the orchestrator is fine).

**Install the harness**

```bash
pip install swebench
```

**Score a predictions file**

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path outputs/my_run_full/predictions.jsonl \
  --max_workers 4 \
  --run_id my_run_full
```

This launches up to 4 Docker sandboxes in parallel, each running one instance's test suite against the corresponding patch. Expect on the order of an hour per 50 tasks at `--max_workers 4` on a recent x86 machine; the bottleneck is test execution and container churn, not network. Logs and per-instance reports land under `logs/run_evaluation/<RUN_ID>/<MODEL_NAME_OR_PATH>/<INSTANCE_ID>/`. Each instance directory has a `report.json` with the resolved bit.

**Read the resolved count**

```bash
python -c '
import json, glob
reports = glob.glob("logs/run_evaluation/my_run_full/*/*/report.json")
resolved = sum(
    1 for r in reports
    if any(v.get("resolved") for v in json.load(open(r)).values() if isinstance(v, dict))
)
print(f"{resolved}/{len(reports)} resolved ({100*resolved/max(1,len(reports)):.1f}%)")
'
```

**Helpers in this repo**

- `evaluate_predictions.py PATH/TO/predictions.jsonl` runs the harness and parses results in one step.
- `quick_eval.py` filters a predictions file down to the rows that actually contain a non-empty patch (saves time when most rows are empty).

**Resumability and partial runs**

Both `python orchestra/run.py` and the SWE-bench harness skip already-completed rows on rerun (the harness checks for an existing `report.json` per instance). For long runs, `tmux` or `screen` is recommended; both stages can be safely interrupted and resumed.

## Choosing a model

The base `Qwen/Qwen3.6-35B-A3B` is the default because it offers strong general agentic reasoning. With about 35B total parameters and 3B active per token, plus 256K native context, it runs comfortably across two GPUs at FP8 with tensor-parallel size 2. It is a reasonable starting point for most SWE-bench experiments and ablations.

`Qwen/Qwen3-Coder-Next` is recommended when the goal is maximum SWE-bench performance, especially with native tool calling. The VRAM profile is essentially identical to the base model, so switching is a drop-in change: update `MODEL_NAME` in `.env` and relaunch vLLM with the coder-flavored command shown in the quickstart. No code changes required.

## Known issues

An earlier Azure-hosted GPT-5.2 deployment we evaluated produced valid patches that the harness or downstream tooling rejected at high token counts (around 54K), while the same logical patch succeeded when the response was shorter (around 429 tokens). The bug was sensitive to total token length rather than patch content.

Qwen3-Coder-Next supports 256K context natively, so this failure mode may not surface, may surface differently, or may surface more often depending on prompt length. We recommend starting with `--max-model-len 32768` until you have verified parser behavior on your stack, then raising it if the workload demands.

## Architecture

The committee runs three stages per SWE-bench instance:

The Generator pool spawns K mini-swe-agent instances in parallel, each operating in its own Docker sandbox with the SWE-bench instance's repository. Each generator receives a different system prompt to reduce inter-proposer correlation. Generators produce candidate patches by exploring the repository, running tests, and iterating until they submit a diff or hit the step limit.

The Reviewer pool runs M reviewer LLM calls per non-empty patch. Each reviewer returns a structured JSON containing four named structural flags plus an adversarial probe. The flags are:

- `addresses_root_cause` — the patch modifies the code path producing the issue, not a downstream symptom.
- `preserves_existing_behavior` — nothing in the patch changes behavior on inputs unrelated to the failure mode.
- `tests_actually_test_the_issue` — under the patched code, the reviewer's adversarial probe produces the issue's correct behavior.
- `no_unrelated_changes` — every hunk in the diff is justified by the issue.

A reviewer's `sound` verdict is the AND of its four flag votes. The Reviewer pool aggregates **per-flag, not per-reviewer**: for each named flag we count the number of reviewers that voted `true`, divide by M (reviewers that returned a structural failure — parse error, refusal, empty — implicitly vote `false`), and require the ratio to clear the per-flag threshold (`FLAG_PASS_THRESHOLD = 0.8` by default). A patch survives the Reviewer pool iff **every flag clears the 80% threshold across the M reviewers**. Patches where at least one flag falls below threshold are filtered. This makes the structural flags meaningful: a single named failure mode that 21% of reviewers see is enough to block a patch, even if its 0–10 score average would have looked sound under the old logic.

Two safety nets sit on top of the per-flag gate. (1) If every reviewer for a given proposal returned a structural failure, the proposal passes through to the tournament rather than being filtered — we cannot trust a flag aggregation when no reviewer produced usable signal. (2) If every proposal across a task is filtered, all candidates pass through to the tournament so the comparator can still produce a winner.

The Comparator pool runs an R-round Copeland tournament across the survivors. Each pairwise judgment is structured: the comparator must first commit to a single-sentence hypothesis about what the failing test is testing, list the lines or functions each patch modifies, and judge each patch's consistency with the hypothesis and its collateral impact (changes to behavior outside the failure mode) before reporting `winner ∈ {A, B, TIE}`. The structured fields force the verdict to fall out of the comparison rather than be pulled from prior, which addresses the documented effort-bias failure mode: ambitious-but-broken patches no longer beat minimal-correct patches on aesthetics. Each unordered pair is judged R times with position swap (A vs B, then B vs A) to cancel lead bias; ties contribute 0.5 to each side in Copeland aggregation. The overall winner is the patch with the highest Copeland score across all pairings; ties at the top are broken by random selection.

The three pools are decoupled: the generator pool is bounded by Docker concurrency, the reviewer and comparator pools by API throughput. Per-role model overrides (`GENERATOR_MODEL`, `REVIEWER_MODEL`, `COMPARATOR_MODEL`) allow mixing different Qwen variants per role for ablation studies without changing code.
