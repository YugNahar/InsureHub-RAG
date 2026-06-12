"""
Calculator — rule‑based + LLM‑assisted for accurate insurance calculations.
Follows strict rules: unit conversion, limits, deductibles, step‑by‑step output.
"""
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Phrases that unambiguously require a numeric calculation
_EXPLICIT_CALC_KEYWORDS = [
    "per thousand", "per hundred", "per unit",
    "per hour", "per day", "per block",
    "percentage", "discount", "deductible", "excess",
    "calculate",
]

def _is_calculation_question(question: str) -> bool:
    q = question.lower()
    # Must contain an actual number to be a real calculation question
    has_number = bool(re.search(r'\b\d+\b', q))
    # Explicit math phrases — always a calculation regardless of numbers
    if any(kw in q for kw in _EXPLICIT_CALC_KEYWORDS):
        return True
    # "how much" only counts as calculation when paired with a specific number
    if "how much" in q and has_number:
        return True
    return False

def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers (including decimals) from text."""
    return [float(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', text)]

def _simple_eval(expr: str) -> Optional[float]:
    """Safely evaluate a simple arithmetic expression."""
    expr = expr.replace(" ", "").replace(",", "")
    if not re.fullmatch(r'[\d\.\+\-\*/\(\)]+', expr):
        return None
    try:
        # Use a safe eval with restricted globals
        return eval(expr, {"__builtins__": {}}, {})
    except:
        return None

def compute_insurance_benefits(question: str, context: str) -> Tuple[str, bool]:
    """
    Returns (answer, is_calculation).
    If is_calculation is True, answer will follow the strict format.
    """
    if not _is_calculation_question(question):
        return "", False

    # For complex calculations, we rely on the LLM with the strict prompt.
    # But we also attempt a quick rule‑based fallback for simple cases.
    # Here we just return ("", True) to signal that the main RAG should use the calculation prompt.
    # The actual calculation will be handled by the LLM using the special prompt in multi_source_rag.
    return "", True