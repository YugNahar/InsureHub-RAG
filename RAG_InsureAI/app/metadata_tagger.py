"""
Metadata tagger — tags documents and queries with insurer/policy metadata.

  - classify_document_type(): pre-classifies as policy_document, reference_handbook,
    regulatory, or general BEFORE schema application.
  - tag_document(): accepts doc_type hint, skips keyword matching for non-policy docs.
  - classify_chunk_intent(): LLM-assisted per-chunk section labeller.
      * Fast path: regex keyword scoring (no LLM cost).
      * LLM path: triggered when regex is ambiguous ("general") OR for
        YouTube/conversational chunks where regex rarely fires.
      * Regex patterns serve as few-shot examples in the LLM prompt so the
        model understands each label — even for text outside any regex.
      * Graceful fallback to regex result if LLM unavailable or fails.
  - classify_chunk_policy_type(): LLM-assisted per-chunk policy type classifier.
      * Fast path: regex keyword scoring.
      * LLM path: triggered when regex is ambiguous OR for YouTube/conversational
        chunks where colloquial language rarely matches exact regex phrases.
      * Regex patterns serve as few-shot examples in the LLM prompt.
      * Graceful fallback to regex result if LLM unavailable or fails.
  - classify_query(): mirrors tag_document() logic, returns policy_type.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Pattern dictionaries ───────────────────────────────────────────────────────
# Each entry maps a canonical name to a list of lowercase match strings.
# Longer / more specific patterns are listed first so they get hit before short
# ones (matters for the hit-count approach).

_INSURER_PATTERNS: dict[str, list[str]] = {
    "RAK":     ["rak insurance", "rak national", "rak travel", "rak"],
    "AIG":     ["american international group", "aig"],
    "GIG":     ["gulf insurance group", "gulf insurance", "gig"],
    "LIVA":    ["liva insurance", "liva"],
    "AXA":     ["axa insurance", "axa"],
    "ZURICH":  ["zurich insurance", "zurich"],
    "ALLIANZ": ["allianz insurance", "allianz"],
}

_POLICY_PATTERNS: dict[str, list[str]] = {
    # Patterns are ordered most-specific → least-specific within each type.
    # Short bare words (life, car, home) are intentionally excluded — they
    # appear in generic insurance text and cause false-positive tagging.
    "travel":            ["travel insurance", "trip cancellation", "flight delay",
                          "baggage loss", "baggage delay", "baggage",
                          "hajj insurance", "umrah insurance", "outbound travel"],
    "health":            ["health insurance", "medical insurance", "hospitalisation",
                          "hospitalization", "medical expense", "clinical",
                          "group health", "mediclaim", "critical illness",
                          "cashless treatment", "pre-existing disease"],
    "life":              ["life insurance", "term life", "whole life",
                          "accidental death benefit", "death benefit",
                          "life assurance", "sum assured", "endowment plan",
                          "ulip", "unit linked", "money back plan",
                          "annuity", "pension plan", "lic policy"],
    "motor":             ["motor insurance", "vehicle insurance", "car insurance",
                          "auto insurance", "motor vehicle", "comprehensive motor",
                          "third party motor", "own damage", "ncb", "no claim bonus",
                          "road accident", "traffic accident"],
    "home":              ["home insurance", "property insurance", "building insurance",
                          "contents insurance", "household insurance",
                          "houseowners policy", "householders policy"],
    "personal_accident": ["personal accident", "pa insurance", "accidental injury",
                          "accidental disability", "permanent disability",
                          "temporary disability", "accidental dismemberment",
                          "group personal accident"],
    "fire":              ["fire insurance", "fire policy", "fire damage",
                          "standard fire", "special perils", "fire and allied perils",
                          "fire brigade", "consequential loss"],
    "marine":            ["marine insurance", "marine cargo", "marine hull",
                          "cargo insurance", "shipping insurance",
                          "inland transit", "import cargo", "export cargo",
                          "bill of lading", "marine policy"],
    "liability":         ["liability insurance", "public liability", "product liability",
                          "professional indemnity", "errors and omissions",
                          "directors and officers", "d&o insurance",
                          "employer liability", "third party liability"],
    "commercial":        ["commercial insurance", "business insurance",
                          "trade insurance", "commercial property",
                          "business interruption", "shop insurance",
                          "office insurance", "industrial all risk"],
    "crop":              ["crop insurance", "agriculture insurance",
                          "pradhan mantri fasal bima", "pmfby",
                          "weather based crop", "kharif", "rabi crop"],
    "cyber":             ["cyber insurance", "cyber risk", "data breach",
                          "cyber attack", "ransomware", "cyber liability",
                          "information security", "data protection insurance"],
}

# ── Document-type classifier patterns ─────────────────────────────────────────
_HANDBOOK_SIGNALS: list[str] = [
    "insurance laws", "insurance law", "insurance act",
    "principles of insurance", "utmost good faith",
    "subrogation", "contribution principle",
    "indemnity principle", "insurable interest",
    "case law", " v. ", " vs. ", " vs ",
    "lic v.", "supreme court", "high court", "judgment", "judgement",
    "chapter ", "unit ", "module ",
    "irda", "irdai", "irda act", "insurance regulatory",
    "insurance development authority",
    "section 64", "section 2", "section 3", "schedule i", "schedule ii",
    "first schedule", "second schedule",
    "gazette notification", "gazette of india",
    "reinsurance", "micro insurance", "micro-insurance",
    "marine insurance", "fire insurance", "motor vehicles act",
    "history of insurance", "evolution of insurance",
    "legislative history", "insurance ombudsman",
    "study material", "reference book", "textbook", "handbook",
    "module i", "module ii", "unit i", "unit ii",
    "examination", "syllabus", "institute of insurance",
]

_REGULATORY_SIGNALS: list[str] = [
    "irda regulation", "irdai regulation", "irda circular",
    "irdai circular", "insurance regulatory and development authority",
    "regulation no.", "notification no.", "f. no.",
    "gazette notification", "official gazette",
    "ministry of finance", "government of india",
]


def classify_document_type(filename: str, preview: str, extra_text: str = "") -> str:
    """
    Classify a document as one of four types BEFORE applying any schema.

    Types:
      "policy_document"    — An actual insurance policy issued to a customer.
      "reference_handbook" — Legal textbook, study guide, or handbook.
      "regulatory"         — IRDA/IRDAI regulations, circulars, gazette notifications.
      "general"            — Anything else (resumes, spreadsheets, YouTube, etc.).
    """
    text = (filename + " " + preview + " " + extra_text).lower()

    reg_hits = sum(1 for sig in _REGULATORY_SIGNALS if sig in text)
    if reg_hits >= 2:
        return "regulatory"

    handbook_hits = sum(1 for sig in _HANDBOOK_SIGNALS if sig in text)
    if handbook_hits >= 3:
        return "reference_handbook"

    policy_signals = [
        "policy number", "policy no", "policy no.", "policy id",
        "certificate of insurance", "policy schedule",
        "insured name", "policyholder", "policy holder",
        "sum insured", "sum assured",
        "premium amount", "annual premium",
        "policy period", "policy term",
        "commencement date", "inception date",
        "renewal date", "expiry date",
    ]
    policy_hits = sum(1 for sig in policy_signals if sig in text)
    if policy_hits >= 2:
        return "policy_document"

    if handbook_hits >= 1:
        return "reference_handbook"

    return "general"


def _count_hits(text: str, patterns: list[str]) -> int:
    """Return total number of pattern occurrences in text (not just a binary hit)."""
    return sum(text.count(p) for p in patterns)


def tag_document(
    filename: str,
    preview: str,
    *,
    extra_text: str = "",
    doc_type: Optional[str] = None,
    llm: Any = None,
) -> dict:
    """
    Return metadata tags for a document.

    Performs regex-based scoring for insurer and policy type. If regex is not
    confident or it is a non-policy document, calls the LLM (if available) for
    refinement.
    """
    if doc_type is None:
        doc_type = classify_document_type(filename, preview, extra_text)

    text = (filename + " " + preview + " " + extra_text).lower()

    # ── Insurer scoring ──────────────────────────────────────────────────────
    insurer_hits: dict[str, int] = {}
    for name, patterns in _INSURER_PATTERNS.items():
        hits = _count_hits(text, patterns)
        if hits > 0:
            insurer_hits[name] = hits

    if insurer_hits:
        total = sum(insurer_hits.values())
        best_insurer = max(insurer_hits, key=insurer_hits.__getitem__)
        insurer_confidence = round(insurer_hits[best_insurer] / total, 3)
        all_insurers = sorted(insurer_hits, key=insurer_hits.__getitem__, reverse=True)
    else:
        best_insurer = "UNKNOWN"
        insurer_confidence = 0.0
        all_insurers = []

    # ── Policy type scoring ──────────────────────────────────────────────────
    policy_hits: dict[str, int] = {}
    for ptype, patterns in _POLICY_PATTERNS.items():
        hits = _count_hits(text, patterns)
        if hits > 0:
            policy_hits[ptype] = hits

    if policy_hits:
        total = sum(policy_hits.values())
        best_policy = max(policy_hits, key=policy_hits.__getitem__)
        policy_confidence = round(policy_hits[best_policy] / total, 3)
        all_policy_types = sorted(policy_hits, key=policy_hits.__getitem__, reverse=True)
    else:
        best_policy = "general"
        policy_confidence = 0.0
        all_policy_types = []

    need_insurer_llm = best_insurer == "UNKNOWN" or insurer_confidence < 0.7
    need_policy_llm = best_policy == "general" or policy_confidence < 0.7
    is_non_policy_doc = doc_type != "policy_document"

    # Skip expensive LLM calls for handbooks/regulatory docs — they will always
    # produce UNKNOWN insurer + general policy type, wasting 60–120 s per upload.
    if is_non_policy_doc:
        llm = None

    if llm is not None:
        # LLM insurer refinement
        if need_insurer_llm or is_non_policy_doc:
            try:
                valid_insurers = list(_INSURER_PATTERNS.keys())
                prompt = f"""You are an insurance document classifier. Identify the insurer of the document.
Available insurers: {', '.join(valid_insurers)}, UNKNOWN.

Decide based on the filename and the text preview.
Filename: {filename}
Preview (first 1200 chars): {preview[:1200]}

Reply with ONLY the insurer name (one of: {', '.join(valid_insurers)}, UNKNOWN).
No punctuation. No explanation."""
                response = llm.invoke(prompt)
                raw = (response.content if hasattr(response, "content") else str(response)).strip().upper()
                label = re.split(r"[\s\n,.:;()]", raw)[0].strip()
                if label in valid_insurers or label == "UNKNOWN":
                    logger.info("[DOC_METADATA] LLM insurer: %s (regex was: %s)", label, best_insurer)
                    best_insurer = label
                    if label != "UNKNOWN":
                        insurer_confidence = 1.0
                        if label not in all_insurers:
                            all_insurers = [label] + all_insurers
                    else:
                        insurer_confidence = 0.0
            except Exception as exc:
                logger.warning("[DOC_METADATA] LLM insurer failed: %s", exc)

        # LLM policy type refinement
        if need_policy_llm or is_non_policy_doc:
            try:
                valid_policies = list(_POLICY_PATTERNS.keys())
                prompt = f"""You are an insurance document classifier. Identify the policy type of the document.
Available policy types: {', '.join(valid_policies)}, general.

Decide based on the filename, the document type, and the text preview.
Filename: {filename}
Doc Type context: {doc_type}
Preview (first 1200 chars): {preview[:1200]}

Reply with ONLY the policy type label (one of: {', '.join(valid_policies)}, general).
No punctuation. No explanation."""
                response = llm.invoke(prompt)
                raw = (response.content if hasattr(response, "content") else str(response)).strip().lower()
                label = re.split(r"[\s\n,.:;()]", raw)[0].strip()
                if label in valid_policies or label == "general":
                    logger.info("[DOC_METADATA] LLM policy_type: %s (regex was: %s)", label, best_policy)
                    best_policy = label
                    if label != "general":
                        policy_confidence = 1.0
                        if label not in all_policy_types:
                            all_policy_types = [label] + all_policy_types
                    else:
                        policy_confidence = 0.0
            except Exception as exc:
                logger.warning("[DOC_METADATA] LLM policy_type failed: %s", exc)

    return {
        "doc_type": doc_type,
        "insurer": best_insurer,
        "policy_type": best_policy,
        "insurer_confidence": insurer_confidence,
        "policy_type_confidence": policy_confidence,
        "all_insurers": all_insurers,
        "all_policy_types": all_policy_types,
    }


def classify_query(question: str, llm: Any = None) -> dict:
    """
    Classify a query to help route to the right documents.
    Queries are always treated as "policy_document" intent for matching purposes.
    """
    return tag_document(filename="", preview=question, doc_type="policy_document", llm=llm)


def build_metadata_filter(
    query_meta: dict,
    routed_sources: Optional[list[str]] = None,
    *,
    insurer_confidence_threshold: float = 0.65,
    policy_confidence_threshold: float = 0.65,
) -> Optional[dict]:
    """
    Build a TurboVec-compatible metadata filter from classified query metadata.

    Design decisions:
    - Confidence threshold raised to 0.65 (from 0.4) so weak keyword matches
      don't aggressively narrow the candidate pool.
    - Policy-type filter always includes "general" chunks via $or so that
      cross-topic handbook sections are never excluded. A "health" query will
      match chunks tagged health OR general, preventing the filter from hiding
      general-purpose sections that contain relevant health information.
    - Insurer filter only applied when a specific insurer is confidently
      detected (not UNKNOWN).
    - routed_sources (from summary Stage-1 search) bypasses all other filters
      since the source list is already the ground-truth narrowing.
    """
    if routed_sources:
        unique = list(dict.fromkeys(routed_sources))
        if len(unique) == 1:
            return {"source": {"$eq": unique[0]}}
        return {"$or": [{"source": {"$eq": s}} for s in unique]}

    conditions: list[dict] = []

    # ── Insurer filter ────────────────────────────────────────────────────────
    insurer      = query_meta.get("insurer")
    insurer_conf = query_meta.get("insurer_confidence", 0.0)
    all_insurers = query_meta.get("all_insurers", [])

    if insurer and insurer != "UNKNOWN" and insurer_conf >= insurer_confidence_threshold:
        candidates = list(dict.fromkeys([*all_insurers, insurer, "UNKNOWN"]))
        conditions.append({"insurer": {"$in": candidates}})

    # ── Policy-type filter ────────────────────────────────────────────────────
    policy_type      = query_meta.get("policy_type")
    policy_conf      = query_meta.get("policy_type_confidence", 0.0)
    all_policy_types = query_meta.get("all_policy_types", [])

    if policy_type and policy_type != "general" and policy_conf >= policy_confidence_threshold:
        # Always include "general" chunks — they contain cross-topic content
        # in handbooks and reference docs that is relevant to any specific query.
        candidates = list(dict.fromkeys([*all_policy_types, policy_type, "general"]))
        conditions.append({"policy_type": {"$in": candidates}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ══════════════════════════════════════════════════════════════════════════════
# LLM-ASSISTED CHUNK INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
# Regex patterns with human-readable examples used as few-shot hints in the
# LLM prompt.  The intent is: regex gives fast keyword signals; LLM handles
# everything that falls outside those signals (colloquial, YouTube, handbook).

_CHUNK_INTENT_LABELS: dict[str, dict] = {
    "benefits": {
        "desc": "What the policy pays for: coverage, sum insured, payout amounts, bonuses.",
        "keywords": ["benefit", "coverage", "covers", "sum insured", "payout", "compensation",
                     "maturity", "cashback", "reimbursement payable", "maximum benefit"],
        "regex": [r"\bbenefit\b", r"\bcoverage\b", r"\bcovers?\b", r"\bsum insured\b",
                  r"\bpayout\b", r"\bcompensation\b", r"\bmaturity\b", r"\bindemnity\b"],
    },
    "exclusions": {
        "desc": "What is NOT covered: excluded conditions, voids, exceptions.",
        "keywords": ["exclusion", "not covered", "excluded", "shall not", "not payable",
                     "void", "except", "exception", "waiver"],
        "regex": [r"\bexclusion\b", r"\bnot cover", r"\bnot include", r"\bexclud",
                  r"\bexcept\b", r"\bnot payable\b", r"\bvoid\b"],
    },
    "premiums": {
        "desc": "Premium amounts, payment modes, renewal, lapsing.",
        "keywords": ["premium", "installment", "payment mode", "renewal", "lapse", "due date"],
        "regex": [r"\bpremium\b", r"\binstallment\b", r"\brenewal\b", r"\blapse\b"],
    },
    "claims": {
        "desc": "How to file a claim, reimbursement process, TPA, cashless hospitals.",
        "keywords": ["claim", "settlement", "reimbursement", "cashless", "TPA", "network hospital",
                     "intimation", "documents required", "file a claim"],
        "regex": [r"\bclaim\b", r"\bsettlement\b", r"\breimbursement\b",
                  r"\bcashless\b", r"\btpa\b", r"\bnetwork hospital\b"],
    },
    "eligibility": {
        "desc": "Who can buy/enrol: age limits, entry age, insured person criteria.",
        "keywords": ["eligible", "eligibility", "minimum age", "maximum age", "entry age",
                     "who can", "requirement", "qualify"],
        "regex": [r"\beligib\b", r"\bminimum age\b", r"\bmaximum age\b", r"\bentry age\b"],
    },
    "definitions": {
        "desc": "What terms mean: 'means', 'defined as', glossary.",
        "keywords": ["means", "defined as", "shall mean", "refers to", "interpretation",
                     "glossary", "definition"],
        "regex": [r"\bdefin\b", r"\bmeans?\b", r"\bshall mean\b", r"\brefers? to\b",
                  r"\bglossary\b"],
    },
    "principles": {
        "desc": "Fundamental insurance principles: utmost good faith, subrogation, indemnity.",
        "keywords": ["utmost good faith", "uberrima fide", "subrogation", "contribution",
                     "insurable interest", "indemnity principle", "proximate cause"],
        "regex": [r"\butmost good faith\b", r"\bsubrogation\b", r"\bcontribution\b",
                  r"\binsurable interest\b", r"\bprinciple of\b"],
    },
    "case_law": {
        "desc": "Court cases, judgments, legal precedents.",
        "keywords": ["v.", "court", "judgment", "held", "appeal", "petitioner", "AIR", "SCC"],
        "regex": [r"\bv\.\b", r"\bjudgment\b", r"\bsupreme court\b",
                  r"\bhigh court\b", r"\bheld that\b"],
    },
    "legislation": {
        "desc": "Acts, sections, regulations, gazette notifications.",
        "keywords": ["act", "section", "clause", "regulation", "gazette", "IRDA",
                     "notification", "statute", "amendment"],
        "regex": [r"\bact\b", r"\bsection \d", r"\bregulation\b", r"\birdai?\b",
                  r"\bgazette\b"],
    },
    "types_of_insurance": {
        "desc": "Classification or overview of insurance types.",
        "keywords": ["types of insurance", "classification", "life insurance", "motor insurance",
                     "health insurance", "general insurance", "marine insurance"],
        "regex": [r"\btypes of insurance\b", r"\bclassification\b", r"\bgeneral insurance\b"],
    },
    "history": {
        "desc": "History, evolution, or origin of insurance.",
        "keywords": ["history", "evolution", "origin", "nationalised", "established",
                     "founded", "1938", "1956", "1972"],
        "regex": [r"\bhistory\b", r"\bevolution\b", r"\borigin\b", r"\bnationaliz"],
    },
    "how_to": {
        "desc": "Tips, steps, or guides on how to do something (common in video content).",
        "keywords": ["how to", "steps", "tips", "guide", "compare", "opt for",
                     "advice", "recommend", "should you", "ways to"],
        "regex": [r"\bhow to\b", r"\bsteps?\b", r"\btips?\b", r"\bguide\b",
                  r"\bcompare\b", r"\brecommend\b"],
    },
    "chapter": {
        "desc": "Introduction, overview, or summary of a chapter/unit.",
        "keywords": ["introduction", "chapter", "unit", "overview", "background", "summary"],
        "regex": [r"\bchapter\b", r"\bunit\b", r"\bintroduction\b",
                  r"\boverview\b", r"\bsummary\b"],
    },
}

_VALID_INTENT_LABELS = set(_CHUNK_INTENT_LABELS.keys()) | {"general"}


def _regex_section_score(text: str) -> dict[str, int]:
    """Return hit-count per label using regex patterns only (fast path)."""
    t = text.lower()
    return {
        label: sum(1 for p in info["regex"] if re.search(p, t))
        for label, info in _CHUNK_INTENT_LABELS.items()
    }


def _build_intent_prompt(text: str, doc_type: str, regex_scores: dict[str, int]) -> str:
    """
    Build the LLM classification prompt for chunk intent/section.

    Regex scores are surfaced as 'keyword signals' so the model knows what
    the regex already found — without being restricted to just those signals.
    The few-shot label descriptions tell the model what each label means for
    text that has no regex hits at all (e.g. conversational YouTube content).
    """
    top_regex = sorted(regex_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    regex_hint = ", ".join(
        f"{lbl}({score})" for lbl, score in top_regex if score > 0
    ) or "none"

    label_list = "\n".join(
        f"  - {lbl}: {info['desc']}\n"
        f"    Example keywords: {', '.join(info['keywords'][:5])}"
        for lbl, info in _CHUNK_INTENT_LABELS.items()
    )

    return f"""You are an insurance document section classifier.

Classify the TEXT below into exactly ONE of these labels:
{label_list}
  - general: content that doesn't clearly fit any label above

Document type context: {doc_type}
Regex keyword signals (hints only, may be empty or wrong for conversational text): {regex_hint}

IMPORTANT:
- The regex signals are hints based on keyword matching — they can be empty or misleading
  for conversational or YouTube-style text. Read the FULL MEANING of the text.
- Even if regex signals are empty, pick the most appropriate label based on content.
- Conversational or video-style text (e.g. "how to get cheap insurance") → "how_to"
- Text explaining what a policy covers → "benefits"
- Text about what is not covered → "exclusions"
- Text about filing a claim → "claims"
- Reply with ONLY the label name, nothing else. No explanation, no punctuation.

TEXT (first 600 chars):
{text[:600]}

LABEL:"""


def classify_chunk_intent(
    text: str,
    doc_type: str = "general",
    llm: Any = None,
    *,
    force_llm: bool = False,
) -> str:
    """
    Classify the section/intent of a document chunk using a regex+LLM hybrid.

    Strategy
    --------
    1. Run fast regex scoring across all intent labels.
    2. If regex finds a clear winner (≥2 hits, ≥2× ahead of runner-up) AND
       force_llm is False → return regex result immediately (no LLM call).
    3. Otherwise (ambiguous / no hits / force_llm=True):
       a. If an LLM is provided → call it with the regex signals as few-shot
          hints in the prompt.
       b. If no LLM → return the best regex guess or "general".

    Parameters
    ----------
    text      : The chunk text to classify.
    doc_type  : Document type ("policy_document", "reference_handbook",
                "regulatory", "general", "youtube" …). Passed to prompt.
    llm       : Optional LangChain LLM instance. If None, only regex is used.
    force_llm : If True, always call LLM even when regex is confident
                (useful for YouTube/conversational chunks).

    Returns
    -------
    A label string from _VALID_INTENT_LABELS.
    """
    regex_scores = _regex_section_score(text)
    best_label = max(regex_scores, key=regex_scores.__getitem__)
    best_score = regex_scores[best_label]

    sorted_scores = sorted(regex_scores.values(), reverse=True)
    runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0

    # Regex is confident if: ≥2 hits AND at least 2× ahead of runner-up
    regex_confident = best_score >= 2 and best_score >= (runner_up * 2 + 1)

    if regex_confident and not force_llm:
        logger.debug("[INTENT] regex confident → %s (score=%d)", best_label, best_score)
        return best_label

    # ── LLM path ──────────────────────────────────────────────────────────────
    if llm is None:
        result = best_label if best_score >= 1 else "general"
        logger.debug("[INTENT] no LLM, regex fallback → %s", result)
        return result

    try:
        prompt = _build_intent_prompt(text, doc_type, regex_scores)
        response = llm.invoke(prompt)
        raw = (response.content if hasattr(response, "content") else str(response)).strip().lower()
        # Clean: take first word/token only (model sometimes adds punctuation)
        label = re.split(r"[\s\n,.:;]", raw)[0].strip()
        if label in _VALID_INTENT_LABELS:
            logger.info("[INTENT] LLM → %s (regex was: %s/%d)", label, best_label, best_score)
            return label
        logger.warning("[INTENT] LLM returned unknown label '%s', using regex fallback", label)
    except Exception as exc:
        logger.warning("[INTENT] LLM call failed: %s — using regex fallback", exc)

    return best_label if best_score >= 1 else "general"


def classify_chunk_intent_batch(
    texts: list[str],
    doc_type: str = "general",
    llm: Any = None,
    *,
    force_llm_for_youtube: bool = True,
    source_type: str = "",
) -> list[str]:
    """
    Classify a batch of chunks efficiently.

    For YouTube/conversational chunks (source_type contains 'whisper' or
    'youtube'), force_llm=True so the LLM handles colloquial text.
    For regular document chunks, regex fast-path is used first; LLM only
    called for ambiguous ones.
    """
    is_youtube = "whisper" in source_type or "youtube" in source_type.lower()
    force = is_youtube and force_llm_for_youtube
    return [
        classify_chunk_intent(t, doc_type=doc_type, llm=llm, force_llm=force)
        for t in texts
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LLM-ASSISTED CHUNK POLICY TYPE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
# The existing tag_document() short-circuits to policy_type="general" for
# non-policy documents (handbooks, YouTube, etc.) which is correct at the
# DOCUMENT level.  But at the CHUNK level, a "how to get cheap car insurance"
# video is clearly "motor", and a handbook chapter on health claims is "health".
#
# This classifier uses a regex fast-path + LLM fallback, exactly like
# classify_chunk_intent().  The key difference vs the old implementation:
#
#   OLD: LLM was only called if regex was confident AND force_llm=True, or
#        if regex was ambiguous AND an LLM existed.  But "regex confident" was
#        broken — it excluded "general" from being the dominant type, but the
#        fallback WITHOUT an LLM still returned "general" for all-zero scores.
#
#   NEW: When regex scores are all zero (common for YouTube/handbook/colloquial
#        text), we ALWAYS call the LLM if available — because zero regex hits
#        means the fast path has no useful information.  The regex patterns are
#        passed to the LLM as few-shot examples so the model can generalise to
#        synonyms and paraphrases it would not have matched by keyword alone.

_POLICY_TYPE_HINTS: dict[str, dict] = {
    "motor": {
        "desc": (
            "Car, bike, vehicle, auto insurance. Covers own-damage, third-party "
            "liability, road accidents, traffic incidents, driving-related topics."
        ),
        "keywords": [
            "car insurance", "motor insurance", "vehicle insurance", "auto insurance",
            "motor vehicle", "comprehensive motor", "third party liability",
            "own damage", "road accident", "traffic", "driving", "bike insurance",
            "two-wheeler", "automobile", "collision", "fender bender",
        ],
        "regex": [
            r"\bcar insurance\b", r"\bmotor insurance\b", r"\bvehicle insurance\b",
            r"\bauto insurance\b", r"\bmotor vehicle\b", r"\bcomprehensive motor\b",
            r"\bthird.?party\b", r"\bown damage\b", r"\bdriving\b",
            r"\bbike insurance\b", r"\btwo.?wheeler\b", r"\bautomobile\b",
        ],
    },
    "health": {
        "desc": (
            "Medical, hospital, health coverage, clinical treatment. Covers "
            "hospitalisation, OPD, IPD, cashless treatment, doctor visits, "
            "medicine costs, surgery, emergency medical care."
        ),
        "keywords": [
            "health insurance", "medical insurance", "hospitalization", "hospital",
            "medical expense", "clinical", "OPD", "IPD", "cashless treatment",
            "doctor", "surgery", "medicine", "treatment", "illness", "disease",
            "pre-existing", "maternity", "dental", "vision", "pharmacy",
        ],
        "regex": [
            r"\bhealth insurance\b", r"\bmedical insurance\b", r"\bhospitali[sz]ation\b",
            r"\bhospital\b", r"\bmedical expense\b", r"\bclinical\b",
            r"\bdoctor\b", r"\bsurgery\b", r"\billness\b", r"\btreatment\b",
            r"\bpre.?existing\b", r"\bmaternity\b",
        ],
    },
    "life": {
        "desc": (
            "Life cover, term life, whole life, death benefit, sum assured. "
            "Covers death, terminal illness, critical illness riders, annuity, "
            "pension, retirement savings with life component."
        ),
        "keywords": [
            "life insurance", "term life", "whole life", "death benefit",
            "sum assured", "life assurance", "accidental death", "critical illness",
            "terminal illness", "annuity", "pension", "retirement plan",
            "endowment", "unit-linked", "ULIP", "nominee", "beneficiary",
        ],
        "regex": [
            r"\blife insurance\b", r"\bterm life\b", r"\bwhole life\b",
            r"\bdeath benefit\b", r"\bsum assured\b", r"\blife assurance\b",
            r"\bcritical illness\b", r"\bannuity\b", r"\bpension\b",
            r"\bendowment\b", r"\bulip\b",
        ],
    },
    "travel": {
        "desc": (
            "Travel, trip, flight delay, baggage loss/delay, trip cancellation, "
            "Hajj/Umrah insurance, outbound travel, passport loss, emergency "
            "overseas medical, travel accident."
        ),
        "keywords": [
            "travel insurance", "trip cancellation", "flight delay", "baggage",
            "baggage loss", "baggage delay", "hajj insurance", "outbound",
            "passport loss", "overseas medical", "travel accident",
            "holiday insurance", "vacation", "abroad", "international travel",
        ],
        "regex": [
            r"\btravel insurance\b", r"\btrip cancellation\b", r"\bflight delay\b",
            r"\bbaggage\b", r"\bhajj insurance\b", r"\bumrah insurance\b",
            r"\bpassport loss\b", r"\boverseas\b", r"\bholiday insurance\b",
            r"\babroad\b",
        ],
    },
    "home": {
        "desc": (
            "Home, property, building, contents, household insurance. Covers "
            "fire, flood, theft, structural damage, personal belongings inside "
            "the home."
        ),
        "keywords": [
            "home insurance", "property insurance", "building insurance",
            "contents insurance", "household insurance",
            "flood damage", "theft at home", "structural damage", "landlord",
            "houseowners", "householders",
        ],
        "regex": [
            r"\bhome insurance\b", r"\bproperty insurance\b", r"\bbuilding insurance\b",
            r"\bcontents insurance\b", r"\bhousehold insurance\b",
            r"\bflood\b", r"\btheft\b", r"\blandlord\b",
            r"\bhouseowners\b", r"\bhouseholders\b",
        ],
    },
    "personal_accident": {
        "desc": (
            "Personal accident cover. Covers accidental injury, death, permanent or "
            "temporary disability, dismemberment. Distinct from life insurance."
        ),
        "keywords": [
            "personal accident", "pa insurance", "accidental injury",
            "accidental disability", "permanent disability", "temporary disability",
            "accidental dismemberment", "group personal accident",
            "accidental death", "ptd", "ttd",
        ],
        "regex": [
            r"\bpersonal accident\b", r"\bpa insurance\b", r"\baccidental injur\b",
            r"\baccidental disabilit\b", r"\bpermanent disabilit\b",
            r"\btemporary disabilit\b", r"\bdismemberment\b",
        ],
    },
    "fire": {
        "desc": (
            "Fire insurance and allied perils. Covers fire damage, lightning, explosion, "
            "flood (in industrial context), riots, strikes, consequential loss."
        ),
        "keywords": [
            "fire insurance", "fire policy", "fire damage", "standard fire",
            "special perils", "fire and allied perils", "consequential loss",
            "fire brigade", "fire loss", "burning",
        ],
        "regex": [
            r"\bfire insurance\b", r"\bfire policy\b", r"\bfire damage\b",
            r"\bstandard fire\b", r"\bspecial perils\b", r"\bconsequential loss\b",
        ],
    },
    "marine": {
        "desc": (
            "Marine cargo and hull insurance. Covers goods in transit, shipping, "
            "import/export cargo, inland transit, vessel damage."
        ),
        "keywords": [
            "marine insurance", "marine cargo", "marine hull", "cargo insurance",
            "shipping insurance", "inland transit", "import cargo", "export cargo",
            "bill of lading", "marine policy", "goods in transit",
        ],
        "regex": [
            r"\bmarine insurance\b", r"\bmarine cargo\b", r"\bmarine hull\b",
            r"\bcargo insurance\b", r"\bshipping insurance\b", r"\binland transit\b",
            r"\bgoods in transit\b", r"\bbill of lading\b",
        ],
    },
    "liability": {
        "desc": (
            "Liability insurance. Covers public liability, product liability, "
            "professional indemnity, D&O, employer liability, errors and omissions."
        ),
        "keywords": [
            "liability insurance", "public liability", "product liability",
            "professional indemnity", "errors and omissions", "e&o",
            "directors and officers", "d&o insurance", "employer liability",
            "third party liability",
        ],
        "regex": [
            r"\bliability insurance\b", r"\bpublic liability\b", r"\bproduct liability\b",
            r"\bprofessional indemnity\b", r"\berrors and omissions\b",
            r"\bd&o insurance\b", r"\bdirectors and officers\b",
        ],
    },
    "commercial": {
        "desc": (
            "Commercial and business insurance. Covers business property, "
            "business interruption, shop/office insurance, industrial all-risk."
        ),
        "keywords": [
            "commercial insurance", "business insurance", "trade insurance",
            "commercial property", "business interruption", "shop insurance",
            "office insurance", "industrial all risk", "sme insurance",
        ],
        "regex": [
            r"\bcommercial insurance\b", r"\bbusiness insurance\b",
            r"\bbusiness interruption\b", r"\bshop insurance\b",
            r"\boffice insurance\b", r"\bindustrial all.?risk\b",
        ],
    },
    "crop": {
        "desc": (
            "Crop and agricultural insurance. Covers kharif/rabi crops, "
            "weather-based insurance, PMFBY, pradhan mantri fasal bima."
        ),
        "keywords": [
            "crop insurance", "agriculture insurance", "pmfby",
            "pradhan mantri fasal bima", "weather based crop",
            "kharif", "rabi crop", "farm insurance",
        ],
        "regex": [
            r"\bcrop insurance\b", r"\bagriculture insurance\b", r"\bpmfby\b",
            r"\bfasal bima\b", r"\bkharif\b", r"\brabi crop\b",
        ],
    },
    "cyber": {
        "desc": (
            "Cyber insurance. Covers data breach, cyber attacks, ransomware, "
            "cyber liability, information security, digital risk."
        ),
        "keywords": [
            "cyber insurance", "cyber risk", "data breach", "cyber attack",
            "ransomware", "cyber liability", "information security",
            "data protection insurance", "hacking", "phishing",
        ],
        "regex": [
            r"\bcyber insurance\b", r"\bcyber risk\b", r"\bdata breach\b",
            r"\bcyber attack\b", r"\bransomware\b", r"\bcyber liability\b",
        ],
    },
}

# "general" is kept as a valid output but NOT in the hints dict —
# the LLM is told to return "general" only when no other type fits.
_VALID_POLICY_TYPES = set(_POLICY_TYPE_HINTS.keys()) | {"general"}


def _regex_policy_score(text: str) -> dict[str, int]:
    """
    Return hit-count per policy type using regex (fast path).
    Only scores the four specific types — "general" is the fallback, not scored.
    """
    t = text.lower()
    return {
        ptype: sum(1 for p in info["regex"] if re.search(p, t))
        for ptype, info in _POLICY_TYPE_HINTS.items()
    }


def _build_policy_type_prompt(text: str, regex_scores: dict[str, int]) -> str:
    """
    Build LLM prompt for policy_type classification with regex as few-shot hints.

    The prompt is designed so the LLM can identify the correct policy type
    even when the text uses synonyms, colloquial language, or paraphrases
    that don't appear in our regex patterns.
    """
    top_regex = sorted(regex_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    regex_hint = ", ".join(
        f"{pt}({score})" for pt, score in top_regex if score > 0
    ) or "none (text may use synonyms or colloquial language)"

    label_list = "\n".join(
        f"  - {pt}: {info['desc']}\n"
        f"    Example keywords: {', '.join(info['keywords'][:6])}"
        for pt, info in _POLICY_TYPE_HINTS.items()
    )

    return f"""You are an insurance content classifier. Your job is to identify the POLICY TYPE of a text.

Available policy types:
{label_list}
  - general: text covers multiple types, is generic about insurance, or the type cannot be determined

Regex keyword signals found in this text (these are HINTS only — the text may use synonyms
or colloquial language that regex missed, so use your full understanding):
  {regex_hint}

DECISION RULES (apply in order):
1. If the text is about cars, vehicles, auto, driving, road accidents → motor
2. If the text is about hospitals, doctors, medicine, illness, medical bills → health
3. If the text is about death benefit, life cover, term plan, sum assured for life → life
4. If the text is about travel, flights, baggage, trip cancellation, overseas → travel
5. If the text is about home, property, building contents, fire at home, flood → home
6. If the text covers multiple types with no dominant one, or is truly generic → general

Reply with ONLY the policy type label (one word from: motor, health, life, travel, home, general).
No explanation. No punctuation. Just the label.

TEXT (first 500 chars):
{text[:500]}

POLICY TYPE:"""


def classify_chunk_policy_type(
    text: str,
    llm: Any = None,
    *,
    force_llm: bool = False,
) -> str:
    """
    Identify the policy type of a chunk using regex + optional LLM.

    Works for ALL document types including YouTube transcripts and handbooks —
    unlike tag_document() which returns 'general' for non-policy documents.

    Key fix vs original implementation:
    - When all regex scores are 0 (common for colloquial/YouTube text), we
      ALWAYS call the LLM if available rather than defaulting to "general".
    - "general" is not in the regex scoring dict so it can't win the max()
      race and become a misleading "best" type.
    - LLM prompt includes richer descriptions and more keyword examples so
      the model generalises correctly even without regex hits.

    Returns one of: 'motor', 'health', 'life', 'travel', 'home', 'general'
    """
    regex_scores = _regex_policy_score(text)

    # Find the best non-zero regex hit
    positive_scores = {k: v for k, v in regex_scores.items() if v > 0}

    if positive_scores:
        best_type = max(positive_scores, key=positive_scores.__getitem__)
        best_score = positive_scores[best_type]

        sorted_vals = sorted(positive_scores.values(), reverse=True)
        runner_up = sorted_vals[1] if len(sorted_vals) > 1 else 0

        # Regex is confident: ≥2 hits AND 2× runner-up
        regex_confident = best_score >= 2 and best_score >= (runner_up * 2 + 1)
    else:
        best_type = "general"
        best_score = 0
        regex_confident = False

    # If regex is confident and we're not forcing LLM, trust it
    if regex_confident and not force_llm:
        logger.debug("[POLICY_TYPE] regex confident → %s (score=%d)", best_type, best_score)
        return best_type

    # ── LLM path ──────────────────────────────────────────────────────────────
    # Called when:
    #   (a) regex found nothing (best_score == 0) — LLM must decide from meaning
    #   (b) regex is ambiguous (multiple types close in score)
    #   (c) force_llm=True (always use LLM, e.g. for YouTube chunks)
    if llm is None:
        # No LLM available — only trust the regex result when it actually
        # cleared the same regex_confident bar used above (>=2 hits AND
        # 2x the runner-up). A single incidental keyword match (best_score=1,
        # e.g. one passing mention of "life insurer" in an agent-licensing
        # paragraph) used to win outright here, which is what let a whole
        # multi-topic reference handbook's chunks get stamped with whatever
        # type its first incidental keyword happened to be — confirmed live
        # against this KB: 276/402 chunks tagged "life", including a chunk
        # that was actually about marine insurance law. Falling back to
        # "general" for anything below the confidence bar is honest about
        # what regex-only classification can actually tell without an LLM.
        result = best_type if regex_confident else "general"
        logger.debug("[POLICY_TYPE] no LLM, regex fallback → %s", result)
        return result

    try:
        prompt = _build_policy_type_prompt(text, regex_scores)
        response = llm.invoke(prompt)
        raw = (response.content if hasattr(response, "content") else str(response)).strip().lower()
        # Take first token only — model sometimes adds punctuation or explanation
        label = re.split(r"[\s\n,.:;()]", raw)[0].strip()
        if label in _VALID_POLICY_TYPES:
            logger.info(
                "[POLICY_TYPE] LLM → %s (regex was: %s/%d, force=%s)",
                label, best_type, best_score, force_llm,
            )
            return label
        # LLM returned something unexpected — fall back to regex or general
        logger.warning(
            "[POLICY_TYPE] LLM returned unknown label '%s', using regex fallback (best=%s/%d)",
            label, best_type, best_score,
        )
    except Exception as exc:
        logger.warning("[POLICY_TYPE] LLM call failed: %s — using regex fallback", exc)

    return best_type if best_score >= 1 else "general"


def classify_chunk_policy_type_batch(
    texts: list[str],
    llm: Any = None,
    *,
    force_llm_for_youtube: bool = True,
    source_type: str = "",
) -> list[str]:
    """
    Classify policy types for a batch of chunks.

    For YouTube/conversational chunks (source_type contains 'whisper' or
    'youtube'), force_llm=True so the LLM handles colloquial/informal text.
    For regular document chunks, regex fast-path is tried first.
    """
    is_youtube = "whisper" in source_type or "youtube" in source_type.lower()
    force = is_youtube and force_llm_for_youtube
    return [
        classify_chunk_policy_type(t, llm=llm, force_llm=force)
        for t in texts
    ]