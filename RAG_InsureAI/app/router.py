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

# Groq announced 2026-06-17 that llama-3.3-70b-versatile is deprecated,
# shutdown date 2026-08-16 (confirmed via console.groq.com/docs/
# deprecations) — still answering as of 2026-08-17 but past its stated
# end-of-life. Default moved to openai/gpt-oss-120b, Groq's recommended
# replacement, for both this (Ava's live chat + structured extraction —
# see travel_bot/services/llm_service.py) and GROQ_CLASSIFICATION_MODEL
# below. Unlike the old model it's a REASONING model — confirmed live
# both use cases still work correctly (Ava's TravelInsuranceDetails/
# CompanionTraveller function-calling extraction: correct fields, no
# hallucination on empty input) as long as reasoning_effort is set and
# max_tokens leaves headroom for reasoning tokens on top of the actual
# output — see GROQ_CLASSIFICATION_REASONING_EFFORT below and
# get_classification_llm's own comment for why that matters.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

GROQ_CLASSIFICATION_MODEL = os.getenv("GROQ_CLASSIFICATION_MODEL", "openai/gpt-oss-120b").strip()
# "low" still burns real tokens on internal reasoning before the actual
# answer (confirmed live: 11-13 reasoning tokens even for a one-word
# classification prompt) — callers need enough max_tokens headroom for
# that on top of the answer itself, or the response comes back empty.
GROQ_CLASSIFICATION_REASONING_EFFORT = os.getenv("GROQ_CLASSIFICATION_REASONING_EFFORT", "low").strip()

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

# ── Runtime backend override (Super Admin toggle) ──────────────────────────
# Lets Super Admin switch backends (or swap in a manual API key) without a
# redeploy — persisted to disk so it survives server restarts. Falls back to
# the FORCE_BACKEND/env-based auto-detection below when mode is "auto" (or
# never configured).
_ROUTER_STATE_DIR = os.getenv("API_STATE_DIR", os.path.join(os.path.dirname(__file__), "state"))
_ROUTER_SETTINGS_PATH = os.path.join(_ROUTER_STATE_DIR, "router_settings.json")

_runtime_mode = "auto"          # "auto" | "vllm" | "groq" | "manual"
_runtime_manual_api_key = ""
_runtime_manual_base_url = "https://api.groq.com/openai/v1"
_runtime_manual_model = GROQ_MODEL


def _load_router_settings() -> None:
    global _runtime_mode, _runtime_manual_api_key, _runtime_manual_base_url, _runtime_manual_model
    try:
        with open(_ROUTER_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _runtime_mode = data.get("mode", "auto")
        _runtime_manual_api_key = data.get("manual_api_key", "")
        _runtime_manual_base_url = data.get("manual_base_url") or "https://api.groq.com/openai/v1"
        _runtime_manual_model = data.get("manual_model") or GROQ_MODEL
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[router] failed to load router_settings.json: %s", exc)


def _save_router_settings() -> None:
    os.makedirs(_ROUTER_STATE_DIR, exist_ok=True)
    data = {
        "mode": _runtime_mode,
        "manual_api_key": _runtime_manual_api_key,
        "manual_base_url": _runtime_manual_base_url,
        "manual_model": _runtime_manual_model,
    }
    tmp_path = f"{_ROUTER_SETTINGS_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, _ROUTER_SETTINGS_PATH)


_load_router_settings()


def get_backend_settings() -> dict:
    """Current runtime backend settings for the Super Admin panel.
    manual_api_key is never returned in full — only whether one is set."""
    return {
        "mode": _runtime_mode,
        "manual_api_key_set": bool(_runtime_manual_api_key),
        "manual_base_url": _runtime_manual_base_url,
        "manual_model": _runtime_manual_model,
        "effective_backend": _active_backend(),
    }


def set_backend_settings(
    mode: str,
    manual_api_key: str | None = None,
    manual_base_url: str | None = None,
    manual_model: str | None = None,
) -> dict:
    """Update runtime backend settings from the Super Admin panel.

    mode: "auto" | "vllm" | "groq" | "manual".
    manual_api_key: only overwritten when a non-empty value is passed, so
    re-saving other fields doesn't require re-entering the key every time.
    """
    global _runtime_mode, _runtime_manual_api_key, _runtime_manual_base_url, _runtime_manual_model
    if mode not in {"auto", "vllm", "groq", "manual"}:
        raise ValueError(f"Invalid backend mode: {mode!r}")
    _runtime_mode = mode
    if manual_api_key:
        _runtime_manual_api_key = manual_api_key.strip()
    if manual_base_url is not None and manual_base_url.strip():
        _runtime_manual_base_url = manual_base_url.strip()
    if manual_model is not None and manual_model.strip():
        _runtime_manual_model = manual_model.strip()
    _save_router_settings()
    logger.info("[router] backend mode set to: %s", _runtime_mode)
    return get_backend_settings()

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
    if _runtime_mode == "manual" and _runtime_manual_api_key:
        return "manual"
    if _runtime_mode == "vllm" and VLLM_HOST:
        return "vllm"
    if _runtime_mode == "groq" and GROQ_API_KEY:
        return "groq"
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
    if backend == "manual":
        return {"backend": "manual", "model": _runtime_manual_model, "base_url": _runtime_manual_base_url}
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

        # GROQ_MODEL is a reasoning model (see its own comment above) —
        # same reasoning_effort/token-floor treatment as
        # get_classification_llm, so this dormant opt-in path
        # (FORCE_BACKEND=groq, normally left unset) doesn't silently
        # return empty answers if someone flips it on for testing.
        _mt = max(max_tokens, 150) if max_tokens > 0 else 500
        logger.debug(
            "[LLM] Groq model=%s max_tokens=%d reasoning_effort=%s",
            GROQ_MODEL, _mt, GROQ_CLASSIFICATION_REASONING_EFFORT,
        )
        return ChatOpenAI(
            model=GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=_mt,
            timeout=60,
            max_retries=1,
            reasoning_effort=GROQ_CLASSIFICATION_REASONING_EFFORT,
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

    if backend == "manual":
        from langchain_openai import ChatOpenAI
        _mt = max_tokens if max_tokens > 0 else 500
        logger.debug("[LLM] Manual model=%s base_url=%s max_tokens=%d", _runtime_manual_model, _runtime_manual_base_url, _mt)
        return ChatOpenAI(
            model=_runtime_manual_model,
            base_url=_runtime_manual_base_url,
            api_key=_runtime_manual_api_key,
            temperature=temperature,
            max_tokens=_mt,
            timeout=60,
            max_retries=1,
        )

    raise RuntimeError(
        "No LLM is configured. Set one of:\n"
        "  • VLLM_HOST               — self-hosted vLLM (model auto-detected)\n"
        "  • OPENAI_API_KEY          — OpenAI (gpt-4o-mini by default)\n"
        "  • ANTHROPIC_API_KEY       — Anthropic (claude-haiku-4-5-20251001 by default)\n"
        "Export the variable(s) before starting the server."
    )


# ── Classification-only Groq route (metadata_tagger.py's policy_type/
# doc_type/query_type calls, never Layla's live RAG answer generation) ──────
# Deliberately INDEPENDENT of _active_backend()/_runtime_mode above, which
# governs the live-answer backend the Super Admin panel toggles — Groq was
# tried there four separate times and made grounding quality worse each
# time (open-ended synthesis with strict citation/formatting rules is a
# different task than closed classification, and that finding does not
# transfer here). Confirmed live 2026-08-10: a synthetic off-vocab test
# (drone insurance, outside all 16 known types) had the local vLLM model
# confidently (85%) force-fit the content into "motor" on every retry,
# while llama-3.3-70b-versatile correctly and consistently answered
# "general" (3/3) -- and still correctly identified real motor/health/
# liability content as those same types (6/6), i.e. it isn't just
# defaulting to "general" indiscriminately. Classification/tagging is a
# background, non-latency-critical task (already async at ingestion —
# see api.py's _reclassify_chunks_with_llm), so the extra network hop to
# Groq costs nothing a user would notice, unlike a live answer stream.
def get_classification_llm(temperature: float = 0, max_tokens: int = 0):
    """Dedicated LLM for metadata_tagger.py's classification tasks
    (policy_type, doc_type, candidate_type, query-type-for-retrieval-
    routing) — prefers Groq whenever GROQ_API_KEY is configured,
    regardless of what get_insurance_llm()'s live-answer backend is set
    to. Falls back to get_insurance_llm() when Groq isn't configured, so
    this still works in any environment without it.

    Uses GROQ_CLASSIFICATION_MODEL, not GROQ_MODEL — the latter is Ava's
    live-chat model and deliberately untouched by this swap (see the
    comment above GROQ_CLASSIFICATION_MODEL's definition). That model is
    a reasoning model, unlike the old one: the 150-token floor below
    isn't headroom for a longer ANSWER, it's headroom for the internal
    reasoning tokens the model spends before it even starts the actual
    label — confirmed live, a bare max_tokens=6-60 (fine for the old
    non-reasoning model) came back completely empty, every token spent
    on reasoning that got cut off before reaching a real answer."""
    if GROQ_API_KEY:
        from langchain_openai import ChatOpenAI

        _mt = max(max_tokens, 150) if max_tokens > 0 else 500
        logger.debug(
            "[LLM] Classification via Groq model=%s max_tokens=%d reasoning_effort=%s",
            GROQ_CLASSIFICATION_MODEL, _mt, GROQ_CLASSIFICATION_REASONING_EFFORT,
        )
        return ChatOpenAI(
            model=GROQ_CLASSIFICATION_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=_mt,
            timeout=60,
            max_retries=1,
            reasoning_effort=GROQ_CLASSIFICATION_REASONING_EFFORT,
        )
    return get_insurance_llm(temperature=temperature, max_tokens=max_tokens)


def get_general_llm(temperature: float = 0.3):
    return get_insurance_llm(temperature=temperature)
