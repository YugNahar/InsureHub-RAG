"""
Metadata tagger — tags documents and queries with insurer/policy metadata.
"""
import re

_INSURER_PATTERNS = {
    "RAK": ["rak", "rak insurance", "rak national"],
    "AIG": ["aig", "american international"],
    "GIG": ["gig", "gulf insurance"],
    "LIVA": ["liva", "liva insurance"],
    "AXA": ["axa"],
    "ZURICH": ["zurich"],
    "ALLIANZ": ["allianz"],
}

_POLICY_PATTERNS = {
    "travel": ["travel", "trip", "flight", "baggage", "hajj", "umrah"],
    "health": ["health", "medical", "hospital", "clinical"],
    "life": ["life", "death", "accidental death", "term"],
    "motor": ["motor", "vehicle", "car", "auto"],
    "home": ["home", "property", "building", "contents"],
}


def tag_document(filename: str, preview: str) -> dict:
    """Return metadata tags for a document based on filename and content preview."""
    text = (filename + " " + preview).lower()

    insurer = "UNKNOWN"
    for name, patterns in _INSURER_PATTERNS.items():
        if any(p in text for p in patterns):
            insurer = name
            break

    policy_type = "general"
    for ptype, patterns in _POLICY_PATTERNS.items():
        if any(p in text for p in patterns):
            policy_type = ptype
            break

    return {"insurer": insurer, "policy_type": policy_type}


def classify_query(question: str) -> dict:
    """Classify a query to help route to the right documents."""
    q = question.lower()

    insurer = None
    for name, patterns in _INSURER_PATTERNS.items():
        if any(p in q for p in patterns):
            insurer = name
            break

    policy_type = None
    for ptype, patterns in _POLICY_PATTERNS.items():
        if any(p in q for p in patterns):
            policy_type = ptype
            break

    return {"insurer": insurer, "policy_type": policy_type}
