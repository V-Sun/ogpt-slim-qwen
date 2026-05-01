"""Shared LLM call helpers for Orchestra.

Critics and comparators call `call_text_model()`; the proposer pool (mini-swe-agent)
gets its config dict via `proposer_model_config()`. Both honor `MODEL_PROVIDER`:

  - openai_compatible: any vLLM/SGLang/Ollama endpoint speaking the OpenAI API.
  - azure_openai:      Azure OpenAI (Chat Completions or Responses API).
"""

import os
import time
import logging
import signal
import threading
from functools import lru_cache

from openai import OpenAI, AzureOpenAI

from config import (
    MODEL_PROVIDER,
    AZURE_API_KEY,
    AZURE_API_BASE,
    AZURE_API_VERSION,
    AZURE_REASONING_EFFORT,
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    ORCHESTRA_USE_RESPONSES_API,
)

logger = logging.getLogger(__name__)


class OrchestraPerCallTimeout(TimeoutError):
    """Raised when an SDK call exceeds the external wall clock."""


def per_call_timeout_seconds() -> int:
    return int(os.getenv("ORCHESTRA_PER_CALL_TIMEOUT_SECONDS", "300"))


def call_with_wall_clock_guard(fn, *args, **kwargs):
    """Run a blocking SDK call with an external wall-clock timeout.

    Protects scale runs from socket reads that outlive the SDK's own timeout.
    In non-main threads, signal timers are unavailable, so we fall back to the
    SDK timeout and avoid installing a process-wide alarm.
    """
    timeout = per_call_timeout_seconds()
    if timeout <= 0 or threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs)

    def _handle_timeout(signum, frame):
        raise OrchestraPerCallTimeout(
            f"SDK call exceeded wall-clock timeout of {timeout}s"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@lru_cache(maxsize=1)
def get_client():
    """Return a provider-appropriate OpenAI SDK client."""
    if MODEL_PROVIDER == "azure_openai":
        return AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )
    return OpenAI(api_key=OPENAI_API_KEY or "EMPTY", base_url=OPENAI_BASE_URL)


def should_use_responses_api() -> bool:
    """Responses API only meaningful on Azure with reasoning effort."""
    return MODEL_PROVIDER == "azure_openai" and (
        ORCHESTRA_USE_RESPONSES_API or bool(AZURE_REASONING_EFFORT)
    )


def reasoning_kwargs() -> dict:
    if MODEL_PROVIDER == "azure_openai" and AZURE_REASONING_EFFORT:
        return {"reasoning": {"effort": AZURE_REASONING_EFFORT}}
    return {}


def response_text(response) -> str:
    """Extract assistant text from a Chat Completion or Responses API object."""
    if hasattr(response, "choices"):
        content = response.choices[0].message.content
        return content or ""

    obj = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    chunks: list[str] = []
    for item in obj.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def _sdk_model_id(model_name: str) -> str:
    """Strip a provider prefix for the SDK call.

    AzureOpenAI takes the deployment name (no `azure/` prefix); openai-compatible
    servers take the served model id (e.g., `Qwen/Qwen3.6-35B-A3B`, no prefix).
    """
    if "/" not in model_name:
        return model_name
    head, rest = model_name.split("/", 1)
    if head == "azure":
        return rest
    return model_name


def call_text_model(
    model_name: str,
    prompt: str,
    max_output_tokens: int,
    max_retries: int = 3,
    initial_delay: float = 1.0,
):
    """Call Chat Completions (or Responses API for Azure+reasoning) with retries."""
    client = get_client()
    timeout = int(os.getenv("ORCHESTRA_API_TIMEOUT_SECONDS", "600"))
    use_responses = should_use_responses_api()
    sdk_model = _sdk_model_id(model_name)

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            if use_responses:
                logger.debug(
                    f"API call attempt {attempt + 1}/{max_retries + 1} "
                    f"(Responses API, reasoning={AZURE_REASONING_EFFORT}, timeout={timeout}s)"
                )
                response = call_with_wall_clock_guard(
                    client.responses.create,
                    model=sdk_model,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    **reasoning_kwargs(),
                )
            else:
                logger.debug(
                    f"API call attempt {attempt + 1}/{max_retries + 1} "
                    f"(Chat Completions, timeout={timeout}s)"
                )
                response = call_with_wall_clock_guard(
                    client.chat.completions.create,
                    model=sdk_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_output_tokens,
                    timeout=timeout,
                )

            result = response_text(response)
            if attempt > 0:
                logger.info(f"API call succeeded after {attempt + 1} attempts")
            return result

        except Exception as e:
            last_exception = e
            error_type = type(e).__name__
            error_msg = str(e).lower()

            is_timeout = "timeout" in error_msg or "timed out" in error_msg
            is_rate_limit = "rate" in error_msg or "429" in error_msg
            is_server_error = "500" in error_msg or "502" in error_msg or "503" in error_msg
            is_connection = "connection" in error_msg or "network" in error_msg
            is_retryable = is_timeout or is_rate_limit or is_server_error or is_connection

            if attempt < max_retries and is_retryable:
                delay = initial_delay * (2 ** attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{error_type}: {e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue

            if attempt < max_retries:
                logger.error(f"API call failed with non-retryable error: {error_type}: {e}")
            else:
                logger.error(f"API call failed after {max_retries + 1} attempts: {error_type}: {e}")
            raise

    if last_exception:
        raise last_exception
    raise RuntimeError("API call failed with no exception recorded")


def proposer_model_config(model_name: str, base_config: dict) -> dict:
    """Build mini-swe-agent's model config dict, provider-aware.

    Mini-swe-agent uses LiteLLM internally; LiteLLM transparently handles both
    Azure (via the `azure/` prefix on the model name) and OpenAI-compatible
    endpoints (via `custom_llm_provider=openai` + `api_base`).
    """
    model_config = dict(base_config)
    model_config["model_name"] = model_name

    model_kwargs = dict(model_config.get("model_kwargs", {}))

    if MODEL_PROVIDER == "azure_openai":
        model_kwargs["api_key"] = AZURE_API_KEY
        model_kwargs["api_base"] = AZURE_API_BASE
        model_kwargs["api_version"] = AZURE_API_VERSION
        model_kwargs["drop_params"] = False
    else:
        # openai_compatible: route LiteLLM through OpenAI Chat Completions
        # against a custom base URL.
        model_kwargs["api_base"] = OPENAI_BASE_URL
        model_kwargs["api_key"] = OPENAI_API_KEY or "EMPTY"
        model_kwargs["custom_llm_provider"] = "openai"
        model_kwargs.setdefault("drop_params", False)

    model_kwargs.setdefault(
        "timeout",
        int(os.getenv("ORCHESTRA_API_TIMEOUT_SECONDS", "600")),
    )

    # Zombie-thread mitigation: proposers run in a ThreadPoolExecutor with a
    # wall deadline. Python can't kill threads, so past-deadline proposers keep
    # looping and each model call drags the next one through LiteLLM's retry
    # budget. Drop retries to 0.
    model_kwargs.setdefault(
        "num_retries",
        int(os.getenv("ORCHESTRA_LITELLM_RETRIES", "0")),
    )

    if should_use_responses_api():
        model_config["model_class"] = "litellm_response"
        model_kwargs.pop("temperature", None)
        model_kwargs.update(reasoning_kwargs())
        model_kwargs.setdefault(
            "max_output_tokens",
            int(os.getenv("ORCHESTRA_MAX_OUTPUT_TOKENS", "8192")),
        )

    model_config["model_kwargs"] = model_kwargs
    return model_config
