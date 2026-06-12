"""
Validator — strict grounding and conflict checks with numerical verification.
"""
import re
from typing import Tuple, List, Set


def extract_numerical_claims(answer: str) -> Set[str]:
    """Extract all numbers (including decimals) from answer."""
    cleaned = re.sub(r'[^\d\.\-]', ' ', answer)
    numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', cleaned))
    return numbers


def validate_grounding(answer: str, context: str) -> Tuple[bool, List[str]]:
    """Check that every numerical claim in the answer appears verbatim in the context."""
    answer_nums = extract_numerical_claims(answer)
    context_lower = context.lower()
    missing = [num for num in answer_nums if num not in context_lower]
    return len(missing) == 0, missing


def detect_conflict(chunks) -> Tuple[bool, List[str]]:
    """Detect if chunks come from multiple insurers (potential conflict)."""
    insurers = set()
    for chunk in chunks:
        insurer = chunk.metadata.get("insurer") if hasattr(chunk, "metadata") else None
        if insurer and insurer != "UNKNOWN":
            insurers.add(insurer)
    has_conflict = len(insurers) > 1
    return has_conflict, list(insurers)


def validate_calculation(answer: str, context: str) -> Tuple[bool, str]:
    """Check that any calculation result appears grounded."""
    calc_pattern = r'(?:=|\bresult:?\s*)(\d+(?:\.\d+)?)'
    matches = re.findall(calc_pattern, answer, re.IGNORECASE)
    if not matches:
        return True, ""
    context_nums = extract_numerical_claims(context)
    for m in matches:
        if m not in context_nums:
            return False, f"Calculation result '{m}' not found in source context."
    return True, ""