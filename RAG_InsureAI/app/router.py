"""
Router — LLM routing based on query type.
"""
import os
from langchain_openai import ChatOpenAI

VLLM_HOST  = os.environ["VLLM_HOST"]
VLLM_MODEL = os.environ["VLLM_MODEL"]
VLLM_API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"

def get_insurance_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=f"{VLLM_HOST}/v1",
        api_key=VLLM_API_KEY,
        temperature=temperature,
        max_tokens=600,
        timeout=120,
        max_retries=2,
    )

def get_general_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=f"{VLLM_HOST}/v1",
        api_key=VLLM_API_KEY,
        temperature=temperature,
        max_tokens=600,
        timeout=120,
        max_retries=2,
    )

def get_active_model_info() -> dict:
    return {"model": VLLM_MODEL, "backend": VLLM_HOST}
