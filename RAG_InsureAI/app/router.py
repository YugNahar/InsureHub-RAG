"""
Router — LLM routing with auto-fallback.

Priority order:
  0. Groq (generation only) — set FORCE_BACKEND=groq AND GROQ_API_KEY. An
     explicit opt-in toggle, not just having the key present, so testing
     against a bigger model never silently overrides the stored vLLM
     config below — flip FORCE_BACKEND back off (or unset it) and vLLM is
     immediately active again with zero other changes.
  1. vLLM server        — set VLLM_HOST (VLLM_MODEL is validated/auto-detected)
  2. OpenAI             — set OPENAI_API_KEY  (model: gpt-4o-mini or OPENAI_MODEL)
  3. Anthropic          — set ANTHROPIC_API_KEY (model: claude-haiku-4-5-20251001 or ANTHROPIC_MODEL)

If none are configured the server still starts and the health/retrieval endpoints
work normally.  Only answer generation and RAGAS scoring are unavailable.
"""
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# ── Read env vars (no crash at import time) ────────────────────────────────────
FORCE_BACKEND = os.getenv("FORCE_BACKEND", "").strip().lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

VLLM_HOST  = os.getenv("VLLM_HOST", "").strip().rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "").strip()
VLLM_API_KEY = (
    os.getenv("VLLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "EMPTY"
)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()

# Validated model — set once after checking against /v1/models, then cached.
# Takes priority over VLLM_MODEL so a wrong env var never causes a 404.
_resolved_model: str = ""


# ── Model discovery ────────────────────────────────────────────────────────────

def list_vllm_models() -> list:
    """Query VLLM_HOST/v1/models. Returns [] if unreachable."""
    if not VLLM_HOST:
        return []
    try:
        url = f"{VLLM_HOST}/v1/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {VLLM_API_KEY}"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception as exc:
        logger.warning("[router] list_vllm_models failed: %s", exc)
        return []


def set_model_override(model_id: str) -> None:
    """Override the resolved model at runtime (persists until server restart)."""
    global _resolved_model
    _resolved_model = model_id.strip()
    logger.info("[router] model set to: %s", _resolved_model)


def _resolve_vllm_model() -> str:
    """
    Return the model to use, guaranteed to exist on the server.

    1. If already resolved (cached), return immediately.
    2. Query /v1/models from the server.
    3. If VLLM_MODEL matches one of the available models, use it.
    4. Otherwise pick the first available model and log a warning.
    """
    global _resolved_model

    if _resolved_model:
        return _resolved_model

    available = list_vllm_models()
    if not available:
        # Server unreachable — fall back to env var and let the call fail naturally
        return VLLM_MODEL

    if VLLM_MODEL and VLLM_MODEL in available:
        _resolved_model = VLLM_MODEL
    else:
        _resolved_model = available[0]
        if VLLM_MODEL:
            logger.warning(
                "[router] VLLM_MODEL='%s' not found on server. "
                "Auto-switching to '%s'. Available: %s",
                VLLM_MODEL, _resolved_model, available,
            )
        else:
            logger.info("[router] VLLM_MODEL not set, auto-detected: %s", _resolved_model)

    return _resolved_model


# ── Backend detection ──────────────────────────────────────────────────────────

def _active_backend() -> str:
    if FORCE_BACKEND == "groq" and GROQ_API_KEY:
        return "groq"
    if VLLM_HOST:
        return "vllm"
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "none"


def get_active_model_info() -> dict:
    backend = _active_backend()
    if backend == "groq":
        return {"backend": "groq", "model": GROQ_MODEL}
    if backend == "vllm":
        return {
            "backend": "vllm",
            "model": _resolved_model or VLLM_MODEL or "(auto-detect on first call)",
            "host": VLLM_HOST,
        }
    if backend == "openai":
        return {"backend": "openai", "model": OPENAI_MODEL}
    if backend == "anthropic":
        return {"backend": "anthropic", "model": ANTHROPIC_MODEL}
    return {"backend": "none", "model": None}


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_insurance_llm(temperature: float = 0, max_tokens: int = 0):
    """
    Return a LangChain chat model.

    max_tokens=0 means use the per-backend default (400 for vLLM answers,
    80 for RAGAS/judge calls). Pass an explicit value to override.
    """
    backend = _active_backend()

    if backend == "groq":
        from langchain_openai import ChatOpenAI

        _mt = max_tokens if max_tokens > 0 else 500
        logger.debug("[LLM] Groq model=%s max_tokens=%d", GROQ_MODEL, _mt)
        return ChatOpenAI(
            model=GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=_mt,
            timeout=60,
            max_retries=1,
        )

    if backend == "vllm":
        from langchain_openai import ChatOpenAI

        model = _resolve_vllm_model()
        if not model:
            raise RuntimeError(
                f"VLLM_HOST is set ({VLLM_HOST}) but no models are available. "
                "Check that the vLLM server is running."
            )
        _default = int(os.getenv("VLLM_MAX_TOKENS", "1024"))
        _mt = max_tokens if max_tokens > 0 else _default
        logger.debug("[LLM] vLLM model=%s max_tokens=%d", model, _mt)
        return ChatOpenAI(
            model=model,
            base_url=f"{VLLM_HOST}/v1",
            api_key=VLLM_API_KEY,
            temperature=temperature,
            max_tokens=_mt,
            timeout=60,
            max_retries=1,
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        _mt = max_tokens if max_tokens > 0 else 500
        logger.debug("[LLM] OpenAI model=%s max_tokens=%d", OPENAI_MODEL, _mt)
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=_mt,
            timeout=60,
            max_retries=1,
        )

    if backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic anthropic"
            )
        _mt = max_tokens if max_tokens > 0 else 500
        logger.debug("[LLM] Anthropic model=%s max_tokens=%d", ANTHROPIC_MODEL, _mt)
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=_mt,
            timeout=60,
        )

    raise RuntimeError(
        "No LLM is configured. Set one of:\n"
        "  • VLLM_HOST               — self-hosted vLLM (model auto-detected)\n"
        "  • OPENAI_API_KEY          — OpenAI (gpt-4o-mini by default)\n"
        "  • ANTHROPIC_API_KEY       — Anthropic (claude-haiku-4-5-20251001 by default)\n"
        "Export the variable(s) before starting the server."
    )


def get_general_llm(temperature: float = 0.3):
    return get_insurance_llm(temperature=temperature)
