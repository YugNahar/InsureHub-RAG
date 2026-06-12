"""
Router — LLM routing based on query type.
"""
import os
from langchain_openai import ChatOpenAI

VLLM_HOST  = os.environ["VLLM_HOST"]
VLLM_MODEL = os.environ["VLLM_MODEL"]

def get_insurance_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=f"{VLLM_HOST}/v1",
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=600,
        timeout=120,
        max_retries=2,
    )

def get_general_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=f"{VLLM_HOST}/v1",
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=600,
        timeout=120,
        max_retries=2,
    )

def get_active_model_info() -> dict:
    return {"model": VLLM_MODEL, "backend": VLLM_HOST}
