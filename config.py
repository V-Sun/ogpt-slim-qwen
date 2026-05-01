import os

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
# "openai_compatible" (vLLM, SGLang, Ollama, ...) or "azure_openai"
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai_compatible")

# ---------------------------------------------------------------------------
# Azure path (used when MODEL_PROVIDER == "azure_openai")
# ---------------------------------------------------------------------------
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_REASONING_EFFORT = os.getenv("AZURE_REASONING_EFFORT", "")
AZURE_API_VERSION = os.getenv(
    "AZURE_API_VERSION",
    "2025-03-01-preview" if AZURE_REASONING_EFFORT else "2024-12-01-preview",
)

# ---------------------------------------------------------------------------
# OpenAI-compatible path (used when MODEL_PROVIDER == "openai_compatible")
# Default targets a local vLLM server.
# ---------------------------------------------------------------------------
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# ---------------------------------------------------------------------------
# Model name resolution. Per-role overrides fall through to MODEL_NAME.
# Default is base Qwen; swap MODEL_NAME=Qwen/Qwen3-Coder-Next for max
# SWE-bench performance with native tool calling.
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", MODEL_NAME)
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", MODEL_NAME)
COMPARATOR_MODEL = os.getenv("COMPARATOR_MODEL", MODEL_NAME)

# ---------------------------------------------------------------------------
# Back-compat constant the rest of the codebase imports.
# - azure_openai: "azure/<deployment>" so LiteLLM (under mini-swe-agent) routes to Azure.
# - openai_compatible: bare model id (vLLM-served), passed through to LiteLLM with
#   custom_llm_provider=openai and api_base=$OPENAI_BASE_URL via configs/qwen_local.yaml.
# ---------------------------------------------------------------------------
if MODEL_PROVIDER == "azure_openai" and AZURE_DEPLOYMENT:
    MODEL = f"azure/{AZURE_DEPLOYMENT}"
else:
    MODEL = MODEL_NAME

ORCHESTRA_USE_RESPONSES_API = os.getenv(
    "ORCHESTRA_USE_RESPONSES_API", ""
).lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# SWE-bench dataset
# ---------------------------------------------------------------------------
DATASET = os.getenv("SWEBENCH_DATASET", "princeton-nlp/SWE-Bench_Verified")
SPLIT = os.getenv("SWEBENCH_SPLIT", "test")
