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
import time
import uuid
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
                          "bill of lading", "marine policy", "transit insurance"],
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


def classify_query_policy_type(query: str) -> str:
    """
    Classify a short query's policy type using the same regex signals as
    classify_chunk_policy_type(), but with a confidence bar suited to
    short text instead of a full chunk.

    classify_chunk_policy_type() requires >=2 distinct pattern hits (and
    2x the runner-up) before trusting a regex result without an LLM --
    right for a 200-500 word chunk, where a single incidental keyword is
    weak evidence of the chunk's overall topic. A 5-10 word QUERY behaves
    completely differently: confirmed empirically that realistic queries
    ("Explain crop insurance in detail", "What is home insurance") score
    exactly ONE hit for their correct type and nothing else -- there's
    rarely room for a second distinct phrase in a single question.
    Requiring the same >=2 bar here would mean this classifier never
    fires for realistic queries at all.

    A single hit is trusted here as long as it's the sole top-scoring
    category (no tie) -- a comparison query mentioning two types by name
    ("how is liability insurance different from motor insurance") will
    score both at 1 and correctly fall back to "general" (unfiltered)
    rather than guessing one side of the comparison.
    """
    scores = _regex_policy_score(query)
    positive = {k: v for k, v in scores.items() if v > 0}
    if not positive:
        return "general"
    best_score = max(positive.values())
    tied = [k for k, v in positive.items() if v == best_score]
    if len(tied) > 1:
        return "general"
    return tied[0]


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


def _regex_section_score(text: str, heading: str = "") -> dict[str, int]:
    """Return hit-count per label using regex patterns only (fast path).

    A heading word is a much stronger signal than the same word buried in
    the body — confirmed live: a section headed "Common Exclusions" whose
    own bullet list never uses the words "excluded"/"not covered"/"except"
    (every item was phrased as "X, unless Y has been declared...") scored
    ZERO on the exclusions pattern set from body text alone, even though
    the heading itself would have scored a clean, unambiguous match.
    Counted 3x so a single heading hit alone (3) can already beat a
    same-scoring runner-up from body text under the existing "≥2x ahead"
    confidence rule, without needing 2 separate heading hits.
    """
    t = text.lower()
    h = heading.lower()
    return {
        label: sum(1 for p in info["regex"] if re.search(p, t))
        + sum(3 for p in info["regex"] if h and re.search(p, h))
        for label, info in _CHUNK_INTENT_LABELS.items()
    }


def _build_intent_prompt(text: str, doc_type: str, regex_scores: dict[str, int], heading: str = "") -> str:
    """
    Build the LLM classification prompt for chunk intent/section.

    Regex scores are surfaced as 'keyword signals' so the model knows what
    the regex already found — without being restricted to just those signals.
    The few-shot label descriptions tell the model what each label means for
    text that has no regex hits at all (e.g. conversational YouTube content).

    heading, when known, is the actual document heading this section falls
    under (e.g. "Common Exclusions") — confirmed live: without it, a
    section phrased entirely as "X, unless Y has been declared..." (no
    literal "excluded"/"not covered" anywhere in the body) has nothing in
    the first-600-chars TEXT block to signal it's an exclusions list, and
    both the regex AND the LLM (reading only the body) can misclassify it
    as "general". The heading alone usually settles it.
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

    heading_line = f"Section heading: {heading}\n" if heading else ""

    return f"""You are an insurance document section classifier.

Classify the TEXT below into exactly ONE of these labels:
{label_list}
  - general: content that doesn't clearly fit any label above

Document type context: {doc_type}
{heading_line}Regex keyword signals (hints only, may be empty or wrong for conversational text): {regex_hint}

IMPORTANT:
- The regex signals are hints based on keyword matching — they can be empty or misleading
  for conversational or YouTube-style text. Read the FULL MEANING of the text.
- Even if regex signals are empty, pick the most appropriate label based on content.
- If a section heading is given above, weigh it heavily — it is the document's own
  label for this content and is often the clearest signal available, especially
  when the body text itself never repeats the heading's own words (e.g. a heading
  "Common Exclusions" followed by a bullet list phrased entirely as "X, unless Y
  has been declared..." with no literal "excluded"/"not covered" anywhere in the
  body — that is still an exclusions list).
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
    heading: str = "",
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
    heading   : The section's own detected heading text (e.g. "Common
                Exclusions"), when known — see _regex_section_score and
                _build_intent_prompt docstrings for why this matters: a
                section's body can be a bare list with none of the
                category's keywords ever repeated, and the heading alone
                is often the only unambiguous signal available.

    Returns
    -------
    A label string from _VALID_INTENT_LABELS.
    """
    regex_scores = _regex_section_score(text, heading)
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
        prompt = _build_intent_prompt(text, doc_type, regex_scores, heading)
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


def _build_intent_batch_prompt(items: list[tuple[str, dict, str, str]]) -> str:
    """
    Combined-prompt builder for classify_chunk_intents_batch() — one call
    classifying N sections instead of N separate calls. Reuses the exact
    per-label descriptions/rules from _build_intent_prompt (just hoisted
    out so they appear once instead of once per section) so batched output
    matches what the single-section prompt would have produced.

    items: list of (text, regex_scores, doc_type, heading) for sections
    that need an LLM call — the regex-confident sections never reach here
    at all. heading is the section's own detected document heading (may
    be empty) — see _build_intent_prompt's docstring for why it matters:
    a section's body can be a bare list that never repeats the category's
    own keywords, and the heading is often the only unambiguous signal.
    """
    label_list = "\n".join(
        f"  - {lbl}: {info['desc']}\n"
        f"    Example keywords: {', '.join(info['keywords'][:5])}"
        for lbl, info in _CHUNK_INTENT_LABELS.items()
    )

    blocks = []
    for i, (text, regex_scores, doc_type, heading) in enumerate(items, start=1):
        top_regex = sorted(regex_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        regex_hint = ", ".join(
            f"{lbl}({score})" for lbl, score in top_regex if score > 0
        ) or "none"
        heading_line = f"Section heading: {heading}\n" if heading else ""
        blocks.append(
            f"\n=== SECTION {i} ===\n"
            f"Document type context: {doc_type}\n"
            f"{heading_line}"
            f"Regex keyword signals (hints only, may be empty or wrong for conversational text): {regex_hint}\n"
            f"TEXT (first 600 chars):\n{text[:600]}"
        )

    return f"""You are an insurance document section classifier.

Classify EACH of the {len(items)} sections below into exactly ONE of these labels:
{label_list}
  - general: content that doesn't clearly fit any label above

IMPORTANT:
- The regex signals are hints based on keyword matching — they can be empty or misleading
  for conversational or YouTube-style text. Read the FULL MEANING of each section's text.
- Even if regex signals are empty, pick the most appropriate label based on content.
- If a section heading is given, weigh it heavily — it is the document's own label for
  that content and is often the clearest signal available, especially when the body text
  itself never repeats the heading's own words (e.g. a heading "Common Exclusions"
  followed by a bullet list phrased entirely as "X, unless Y has been declared..." with
  no literal "excluded"/"not covered" anywhere in the body — that is still an exclusions list).
- Conversational or video-style text (e.g. "how to get cheap insurance") → "how_to"
- Text explaining what a policy covers → "benefits"
- Text about what is not covered → "exclusions"
- Text about filing a claim → "claims"
- Judge each section entirely independently — do not let one section's content
  influence another section's label.
{"".join(blocks)}

Reply with EXACTLY {len(items)} lines, one per section, in order, in this format:
SECTION 1: <label>
SECTION 2: <label>
...
No explanation, no extra text, no punctuation after the label."""


def classify_chunk_intents_batch(
    texts: list[str],
    doc_types: list[str] | None = None,
    llm: Any = None,
    *,
    force_llm_flags: list[bool] | None = None,
    doc_type: str = "general",
    headings: list[str] | None = None,
) -> list[str]:
    """
    Batched version of classify_chunk_intent() — classifies every section
    of a document in ONE LLM call instead of one call per section.

    Exactly the same regex-fast-path gating as classify_chunk_intent(): a
    section the regex already resolves confidently (and isn't force_llm)
    never touches the LLM at all, batched or not. Batching only combines
    the genuinely ambiguous sections into a single request instead of N —
    see verify_and_enrich_sections_batch()'s docstring for why this
    matters (an 8-section document previously fired 8+ separate Groq
    requests back-to-back with no throttling, reliably tripping Groq's
    per-minute rate limit).

    texts: section texts, in the order results should be returned.
    doc_types: per-section doc_type context (defaults to `doc_type` for
    every section when not given — matches effective_doc_type varying per
    chunk in mixed-source documents).
    force_llm_flags: per-section force_llm (e.g. is_youtube), same meaning
    as classify_chunk_intent's force_llm parameter.
    headings: per-section detected document heading (e.g. "Common
    Exclusions"), when known — SectionChunker already computes and stores
    this in each chunk's metadata["section_heading"], but this function
    used to never receive it, so a section whose body never repeats its
    own category's keywords (e.g. an exclusions list phrased entirely as
    "X, unless Y has been declared...", no literal "excluded"/"not
    covered" anywhere) had nothing to classify against but a misleadingly
    empty regex signal and 600 characters of body text — see
    _build_intent_prompt's docstring for the confirmed live case.
    """
    n = len(texts)
    doc_types = doc_types or [doc_type] * n
    force_llm_flags = force_llm_flags or [False] * n
    headings = headings or [""] * n

    labels: list[str] = [""] * n
    llm_needed: list[int] = []
    per_item_regex: dict[int, dict] = {}

    for i, text in enumerate(texts):
        regex_scores = _regex_section_score(text, headings[i])
        best_label = max(regex_scores, key=regex_scores.__getitem__)
        best_score = regex_scores[best_label]
        sorted_scores = sorted(regex_scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0
        regex_confident = best_score >= 2 and best_score >= (runner_up * 2 + 1)

        if regex_confident and not force_llm_flags[i]:
            labels[i] = best_label
        elif llm is None:
            labels[i] = best_label if best_score >= 1 else "general"
        else:
            llm_needed.append(i)
            per_item_regex[i] = regex_scores

    def _regex_fallback(idx: int) -> str:
        scores = per_item_regex.get(idx) or {}
        if not scores:
            return "general"
        best = max(scores, key=scores.__getitem__)
        return best if scores[best] >= 1 else "general"

    if llm_needed:
        try:
            items = [(texts[i], per_item_regex[i], doc_types[i], headings[i]) for i in llm_needed]
            prompt = _build_intent_batch_prompt(items)
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            found = dict(re.findall(r"SECTION\s+(\d+)\s*:\s*(\S+)", raw, re.IGNORECASE))
            if len(found) != len(llm_needed):
                logger.warning(
                    "[INTENT] batch classify returned %d label(s), expected %d — "
                    "falling back to regex for unmatched sections",
                    len(found), len(llm_needed),
                )
            for pos, idx in enumerate(llm_needed, start=1):
                raw_label = (found.get(str(pos), "") or "").strip().lower()
                label = re.split(r"[\s\n,.:;]", raw_label)[0].strip() if raw_label else ""
                labels[idx] = label if label in _VALID_INTENT_LABELS else _regex_fallback(idx)
        except Exception as exc:
            logger.warning("[INTENT] batch LLM call failed: %s — using regex fallback", exc)
            for idx in llm_needed:
                labels[idx] = _regex_fallback(idx)

    return labels


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
            # Narrowed from bare \bthird.?party\b (2026-07-16) — "third party"
            # alone is a general legal/insurance concept spanning liability,
            # professional indemnity, and general insurance principles, not
            # exclusively motor. Confirmed live: the heading "THIRD PARTY
            # ADMINISTRATORS-HEALTH" classified as motor purely because
            # "third party" hit here with zero competing signal, even though
            # "HEALTH" is literally in the same heading (health's list has
            # no standalone fallback for it — see below). Requiring an
            # immediate qualifier keeps the genuinely motor-specific
            # phrasings ("third party liability", "third-party insurance")
            # while dropping the bare, type-agnostic mention.
            r"\bthird.?party\s+(?:liability|insurance|cover(?:age)?)\b",
            r"\bown damage\b", r"\bdriving\b",
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
            # Bare \bhealth\b / \bmedical\b were tried and reverted the same
            # day (2026-07-16): fixed a heading-only edge case ("THIRD PARTY
            # ADMINISTRATORS-HEALTH") but caused a worse regression in body
            # text — a personal-accident section explicitly contrasting
            # itself against health cover ("Unlike a health plan, this kind
            # of cover doesn't reimburse hospital bills directly") scored
            # health=3 (health + hospital + medical, all present specifically
            # BECAUSE the text was distinguishing itself from health
            # insurance) and confidently misclassified. The original heading
            # case doesn't actually need this — its real body text already
            # contains "health insurance" as a full phrase (confirmed
            # directly against the live KB chunk), which the existing
            # \bhealth insurance\b entry above already catches once the
            # heading check falls through to the body.
        ],
    },
    "life": {
        "desc": (
            "Life cover, term life, whole life, death benefit, sum assured. "
            "Covers death, terminal illness, critical illness riders, annuity, "
            "pension, retirement savings with life component."
        ),
        "keywords": [
            "life insurance", "term insurance", "term life", "whole life", "death benefit",
            "sum assured", "life assurance", "accidental death", "critical illness",
            "terminal illness", "annuity", "pension", "retirement plan",
            "endowment", "unit-linked", "ULIP", "nominee", "beneficiary",
        ],
        "regex": [
            r"\blife insurance\b", r"\bterm insurance\b", r"\bterm life\b", r"\bwhole life\b",
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
            "Home, property, building, contents, household insurance for a "
            "RESIDENTIAL dwelling someone lives in — fire, flood, theft, "
            "structural damage, personal belongings inside the home. NOT for "
            "harm the policyholder CAUSES to someone else's property or a "
            "third party (a neighbour's property damaged by a leak from the "
            "insured's factory, a passerby injured outside the insured's "
            "shop) — that is liability insurance, even though the word "
            "'property' appears in both. NOT for a BUSINESS's own building, "
            "warehouse, shop, or office, even though the underlying risk "
            "(fire, flood, theft, a burst pipe) sounds identical — a "
            "business's own premises is commercial insurance, not home, "
            "regardless of how similar the covered perils are worded."
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
            "temporary disability, dismemberment. Distinct from life insurance. "
            "Users commonly say just \"personal insurance\" for this type."
        ),
        "keywords": [
            "personal accident", "pa insurance", "accidental injury",
            "accidental disability", "permanent disability", "temporary disability",
            "accidental dismemberment", "group personal accident",
            "accidental death", "ptd", "ttd", "personal insurance",
        ],
        "regex": [
            r"\bpersonal accident\b", r"\bpa insurance\b", r"\baccidental injur\b",
            r"\baccidental disabilit\b", r"\bpermanent disabilit\b",
            r"\btemporary disabilit\b", r"\bdismemberment\b",
            # Confirmed live: "What is personal insurance?" scored zero
            # regex hits for every type (the phrase is one word short of
            # "personal accident"), fell through to the LLM fallback, which
            # apparently didn't reliably map it here either — the correctly
            # generated answer about personal accident insurance then got
            # discarded as "cross-topic contamination" relative to whatever
            # type the LLM guessed instead. "Personal insurance" is the
            # natural everyday shorthand for this category (the KB's own
            # "main products of general insurance" list names it "Personal
            # accident insurance"), so it's added directly here rather than
            # left to per-call LLM inference.
            r"\bpersonal insurance\b",
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
            r"\bgoods in transit\b", r"\bbill of lading\b", r"\btransit insurance\b",
            # Bare words added 2026-07-23 — every entry above is a full
            # phrase, so a short natural query like "do you cover overseas
            # shipping of goods?" scored ZERO for marine and lost outright
            # to travel's bare \boverseas\b (confirmed live). None of
            # these three collide with any other type's keyword/regex list
            # (checked) — they're distinctively marine/shipping vocabulary,
            # unlike home's \btheft\b/\bflood\b or health's \btreatment\b,
            # which stay untouched here since narrowing THOSE has already
            # caused a worse regression once (see bare \bhealth\b revert
            # note above). This only ADDS a competing signal so marine can
            # win or at least tie (-> safe "general") instead of losing by
            # default to an unrelated type that happened to have a bare
            # word and marine didn't.
            r"\bcargo\b", r"\bvessel\b", r"\bshipping\b",
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
            "Commercial and business insurance. Covers business PROPERTY (the "
            "building/premises itself), business interruption, shop/office "
            "insurance, industrial all-risk. NOT a catch-all for anything a "
            "business happens to buy — goods in transit is marine, a company "
            "car is motor, a professional being sued for bad advice is "
            "liability, even though all three are 'business' contexts. Only "
            "use commercial when the cover is about the business's own "
            "premises/property/operations continuity, with no more specific "
            "type (marine, motor, liability, fire, etc.) actually fitting."
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
            # Bare \bcrops?\b added 2026-07-23 — every entry above requires
            # a full phrase, so "is flood damage to my crops covered?"
            # scored zero for crop and lost outright to home's bare
            # \bflood\b (confirmed live). Doesn't collide with any other
            # type's list. Only adds a competing signal — worst case is a
            # tie against home that safely falls back to "general" instead
            # of confidently answering "home" for a crop question.
            r"\bcrops?\b",
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


def get_active_vocab() -> dict[str, dict]:
    """
    _POLICY_TYPE_HINTS unioned with any promoted types from
    candidate_vocab.get_active_vocab_extra() — this is what makes "add a
    13th type" a data write instead of a code change. The original 12
    (extensively tuned, with inline comments explaining specific regex
    decisions) stay exactly as hardcoded above; only genuinely new,
    promoted labels ever come from the JSON side.

    Re-read on every call rather than cached — promotion is a rare, manual
    admin action and the file is tiny, so correctness immediately after a
    promotion (no stale in-memory copy, no cross-process cache invalidation
    to coordinate between the api and eval containers) matters far more
    than shaving a microsecond dict-merge off the classification path.
    """
    try:
        from candidate_vocab import get_active_vocab_extra
        extra = get_active_vocab_extra()
    except Exception as exc:
        logger.debug("[POLICY_TYPE] active-vocab-extra unavailable (%s)", exc)
        extra = {}
    return {**_POLICY_TYPE_HINTS, **extra} if extra else _POLICY_TYPE_HINTS


def _valid_policy_types() -> set[str]:
    return set(get_active_vocab().keys()) | {"general"}


def _regex_policy_score(text: str) -> dict[str, int]:
    """
    Return hit-count per policy type using regex (fast path).
    Only scores the four specific types — "general" is the fallback, not scored.
    """
    t = text.lower()
    return {
        ptype: sum(1 for p in info["regex"] if re.search(p, t))
        for ptype, info in get_active_vocab().items()
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

    _vocab = get_active_vocab()
    label_list = "\n".join(
        f"  - {pt}: {info['desc']}\n"
        f"    Example keywords: {', '.join(info['keywords'][:6])}"
        for pt, info in _vocab.items()
    )
    all_labels = ", ".join(list(_vocab.keys()) + ["general"])

    return f"""You are an insurance content classifier. Your job is to identify the POLICY TYPE of a text.

Available policy types:
{label_list}
  - general: text covers multiple types, is generic about insurance, or the type cannot be determined

Regex keyword signals found in this text (these are HINTS only — the text may use synonyms
or colloquial language that regex missed, so use your full understanding):
  {regex_hint}

STEP 1 — read the ENTIRE text first, start to finish, before deciding anything. Note
EVERY distinct insurance type it discusses, not just the first one you recognize —
a paragraph a third of the way through can introduce a completely different type
than the opening paragraph, and you must weigh it equally.

STEP 2 — decide:
- If the text discusses 2 OR MORE clearly different insurance types (e.g. one section
  about hospital/medical cover, a later section about vehicle/driving cover) with no
  single type covering the clear majority of the text → answer "general". Do NOT pick
  whichever type happens to appear first or takes up the most space if a genuinely
  different type is also substantively discussed — "general" is correct here even
  though it means picking no single winner.
- Only if the text is consistently about ONE type throughout (synonyms, examples, and
  elaboration of that same type are fine — that's still one type) → answer that
  specific type's label.
- WATCH FOR EXCLUSIONS AND CROSS-REFERENCES: a type mentioned only to say the text does
  NOT cover it ("does not cover", "excluded", "not covered", "unless"), or mentioned only
  as a comparison point to a DIFFERENT product ("as in X insurance", "unlike X", "whereas
  in X insurance..."), is evidence AGAINST that type, never for it — the words are
  present, but the sentence is denying or comparing, not describing the text's own
  subject. The type a text is ABOUT is the one whose own rules, benefits, or claims
  procedure it actually states.
  Example: "Exclusions in luggage insurance ... does not cover ... motorised vehicles ...
  electric vehicles for which motor liability insurance is required." is TRAVEL insurance
  (a real KB chunk once mistagged "motor" for exactly this reason) — "motor" appears
  purely to name what luggage cover excludes.
  Example: "...in General Insurance the cover is granted normally for one year and in
  Fire Insurance the preamble states..." is LIABILITY insurance, not fire (another real
  KB chunk mistagged this way) — a passing cross-reference used only to illustrate a
  general point about renewal notices, not the text's actual subject.

Reply with ONLY the policy type label (one word from: {all_labels}).
No explanation. No punctuation. Just the label.

FULL TEXT:
{text[:4000]}

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

    # Regex is a HINT for the LLM prompt below (see _build_policy_type_prompt,
    # which explicitly tells the model these are "hints only ... use your
    # full understanding"), not a decision authority in its own right — it
    # should never get to NAME the chunk on its own when an LLM is available
    # to actually judge the meaning. This used to short-circuit here
    # whenever regex hit >=2 confident keywords, entirely skipping the LLM
    # even when one was passed in. Real KB chunks can rack up 2+ incidental
    # keyword hits for the WRONG type (e.g. a chunk about crop-insurance
    # government schemes happens to pattern-match another type's regex)
    # while genuinely on-topic content doesn't always contain the exact
    # hardcoded keyword phrases regex looks for — regex is a narrow,
    # brittle proxy for "what is this chunk about," the LLM's semantic
    # reading is the actually reliable signal once it's available to ask.
    # No longer returns early here at all — falls straight through to the
    # LLM path below whenever an LLM is available, regardless of regex
    # confidence; the `llm is None` branch there is the ONLY place that
    # still trusts a confident regex result outright, since there's
    # nothing else to consult in that case.

    # ── LLM path ──────────────────────────────────────────────────────────────
    # Called whenever an LLM is available, regardless of regex confidence —
    # regex_scores are still passed in as hint context (see prompt builder).
    # Falls through to here when:
    #   (a) regex found nothing (best_score == 0) — LLM must decide from meaning
    #   (b) regex is ambiguous (multiple types close in score)
    #   (c) regex WAS confident but an LLM is available to double-check it
    #   (d) force_llm=True (always use LLM, e.g. for YouTube chunks)
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
        if label in _valid_policy_types():
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


# ── Section-level: heading-first pass + LLM verify/enrich ───────────────────────
# Two-role LLM usage, once per SECTION rather than once per chunk (see
# SectionChunker.split_documents in rag.py, which now groups chunks by
# section_id before calling this): a fast, free first-pass guess, then a
# targeted LLM pass that VERIFIES that guess (cheaper and more reliable than
# re-classifying from scratch — a yes/no-with-confidence question is a much
# narrower ask than "pick 1 of 12 types") and separately EXTRACTS metadata
# fields the structural/regex layer has no way to determine at all.
def regex_first_pass_policy_type(section_heading: str, section_text: str) -> str:
    """
    Step 1 (fast, free, no LLM): guess policy_type from the section HEADING
    first, falling back to the section BODY only if the heading itself
    doesn't confidently resolve.

    The heading is checked with classify_query_policy_type() rather than
    the chunk-tuned confidence bar below — a heading is short, deliberately
    topic-labeling text ("MOTOR INSURANCE", "FIRE INSURANCE"), the same
    shape as a user query, not a 200-500 word chunk where a single
    incidental keyword is weak evidence. When it works, it's a much
    cleaner signal than scanning the full body for keyword hits. It won't
    work for a paraphrased or abstract heading ("DIGITAL RISK COVER" for a
    cyber-insurance section) — those fall through to the body-text check,
    and if THAT also comes up empty, to "general", which
    verify_and_enrich_section_metadata() below still gets a chance to
    correct via its own semantic reading. Neither this function nor that
    one is authoritative alone; this is just the free, fast opening guess.
    """
    if section_heading:
        heading_type = classify_query_policy_type(section_heading)
        if heading_type != "general":
            return heading_type

    regex_scores = _regex_policy_score(section_text)
    positive_scores = {k: v for k, v in regex_scores.items() if v > 0}
    if not positive_scores:
        return "general"
    best_type = max(positive_scores, key=positive_scores.__getitem__)
    best_score = positive_scores[best_type]
    sorted_vals = sorted(positive_scores.values(), reverse=True)
    runner_up = sorted_vals[1] if len(sorted_vals) > 1 else 0
    regex_confident = best_score >= 2 and best_score >= (runner_up * 2 + 1)
    return best_type if regex_confident else "general"


# Confidence (0-100) the LLM must report before its verification is trusted
# to OVERRIDE the first-pass type. Below this, a "no, I think this is wrong"
# verdict is logged but not acted on — a single low-confidence LLM wobble
# shouldn't flip a tag the first pass may have already gotten right.
_VERIFY_OVERRIDE_THRESHOLD = 70

_ENRICHMENT_FIELDS = ("language", "jurisdiction", "document_version", "effective_date", "coverage_category")

# Common everyday synonyms for our canonical type labels. Confirmed live:
# asked the LLM to verify a wrongly-"health"-tagged section that was
# unambiguously about car/collision/no-claim-bonus content — it correctly
# recognized the mislabel (VERIFIED=no, confidence=85) but replied
# CORRECTED_TYPE=auto instead of "motor", the exact label from the "Available
# policy types" list handed to it in the same prompt. "auto" isn't in
# _VALID_POLICY_TYPES, so the strict membership check silently discarded an
# otherwise-correct correction and the wrong "health" tag stuck. Small models
# don't perfectly follow "reply with exactly one of these words" instructions
# even when the list is right there — normalizing a short list of the most
# likely everyday synonyms before the membership check is a safety net for
# that gap, not a replacement for the prompt's own instruction.
_TYPE_SYNONYMS: dict[str, str] = {
    "auto": "motor", "automobile": "motor", "car": "motor", "vehicle": "motor",
    "car insurance": "motor", "vehicle insurance": "motor",
    "property": "home", "household": "home", "house": "home", "homeowners": "home",
    "medical": "health",
    "accident": "personal_accident", "pa": "personal_accident",
    "trip": "travel",
    "agriculture": "crop", "agricultural": "crop", "farm": "crop",
    "business": "commercial",
    "shipping": "marine", "cargo": "marine",
}


def _normalize_policy_type(label: str) -> str:
    return _TYPE_SYNONYMS.get(label, label)


def derive_document_topic_prior(chunk_texts: list[str], filename: str = "") -> tuple[str, float]:
    """
    Compute a document-level policy_type prior from a sample of the
    document's OWN chunk texts, so per-chunk classification isn't done in
    total isolation from what document it actually lives in.

    Confirmed live (2026-07-31, plan_policy_type_tagging.md): re-tagging a
    travel guide's chunks using ONLY each chunk's own isolated text flipped
    several genuinely travel-specific chunks to "health" — a travel-medical
    benefit for "acute toothache that has begun during the journey", the
    policy's own exclusions list, its claims-filing process — because the
    classifier had no way to know these lived in a travel guide, so an
    incidental "treatment"/"toothache" mention won outright against a
    document it never got to see. The SAME re-tag also correctly fixed 5
    known mistags (e.g. a "health"-tagged chunk that was actually the LIFE
    proposal form's health questionnaire) — and in every one of those
    cases the CORRECT answer was the chunk's own document's dominant
    topic. A document-level prior helps both failure modes at once: it
    gives the classifier grounds to move A chunk BACK toward its
    document's real subject (the genuine-fix cases) while raising the bar
    for flipping AWAY from it on a single incidental mention (the
    regression cases).

    Two signals, filename checked first:

    1. Filename — highest-precision signal when it names its own topic
       (e.g. "...travelinsuranceguide.pdf"). Checked as a bare substring
       against each type's canonical name after stripping the leading
       hash/ID prefix this KB's uploads use and lowercasing — deliberately
       NOT word-boundary-anchored, since these filenames have no spaces
       between words ("travelinsuranceguide"). Only used when it's an
       UNAMBIGUOUS single-type match; a filename naming 2+ types (or
       none) falls through to signal 2.

    2. Raw regex keyword-score SUM across every chunk of the document —
       deliberately NOT regex_first_pass_policy_type()'s per-chunk result
       (which requires each INDIVIDUAL chunk to independently clear a
       strict confidence bar before it can even count towards the vote).
       Confirmed live that gate under-counts badly at the document level:
       a liability-insurance module chunked at ~12 chunks scored a
       confident regex_first_pass hit on almost none of them individually
       (liability's regex list is long, specific phrases — "public
       liability", "professional indemnity" — that don't always repeat
       within one ~400-word chunk), yet the RAW score total across all 12
       chunks combined was a clean, wide-margin liability win (17 vs a
       9-point runner-up). Summing raw hits first, THEN judging dominance
       once at the document level, recovers signal the per-chunk gate
       discards.

    Confirmed live this can still genuinely tie: a travel guide scored
    health=27 and travel=27 in raw sum (travel insurance content
    legitimately discusses medical treatment abroad at length — that's
    not a bug in the regex list, the underlying content really does use
    both vocabularies heavily). This is exactly the shape the filename
    check exists to resolve before ever reaching the tie-prone body-text
    signal.

    Returns ("general", 0.0) when neither signal clearly resolves a
    single type — the correct, expected answer for a genuinely
    multi-topic document (a large multi-subject reference textbook must
    land here, not get flattened to whichever type happened to score one
    point higher).
    """
    if filename:
        # "5e9acf857576_travelinsuranceguide.pdf" -> "travelinsuranceguide"
        cleaned = re.sub(r"^[0-9a-f]{8,}_", "", filename.lower())
        cleaned = re.sub(r"\.[a-z0-9]+$", "", cleaned)
        matched = [t for t in get_active_vocab() if t in cleaned]
        if len(matched) == 1:
            return matched[0], 1.0

    totals: dict[str, int] = {}
    for text in chunk_texts:
        for ptype, score in _regex_policy_score(text).items():
            if score:
                totals[ptype] = totals.get(ptype, 0) + score
    if not totals:
        return "general", 0.0
    best_type = max(totals, key=totals.__getitem__)
    best_score = totals[best_type]
    sorted_vals = sorted(totals.values(), reverse=True)
    runner_up = sorted_vals[1] if len(sorted_vals) > 1 else 0
    # Three bars, all tuned against this KB's actual documents (see
    # plan_policy_type_tagging.md Phase 1 verification for the full
    # per-document score table this was calibrated against):
    #
    # 1. best_score >= 10 — minimum absolute evidence; a 2-chunk document
    #    scoring 1 hit for "fire" shouldn't "dominate" on no real evidence.
    #
    # 2. best_score >= runner_up * 1.5 — the winner must clearly lead, not
    #    just plurality-win.
    #
    # 3. runner_up <= 15 — an ABSOLUTE cap on the second-place score, not
    #    just relative to the winner. This is the one that actually
    #    separates genuinely single-topic documents from broad reference
    #    material: a first version of this function used bars 1-2 alone
    #    and confidently called the 256-chunk multi-subject law textbook
    #    "life" (194 vs runner-up marine=103 — clears a 1.5x margin on
    #    sheer document length despite marine ALSO having overwhelming,
    #    genuine representation) and, separately, a 64-chunk regulatory/
    #    corporate-governance handbook "life" (61 vs marine=22) — manual
    #    read of that second document confirmed it discusses IRDA capital
    #    filings, ULIP product mechanics, IAIS/IASB accounting standards,
    #    AND a marine cargo-ship damage example, not life insurance
    #    specifically. Every CONFIRMED single-topic module in this KB has
    #    a runner-up under 12 (nothing else in the document accumulates
    #    much evidence at all); both confirmed multi-topic documents have
    #    a runner-up over 20 (a second topic is ALSO substantively
    #    present, just discussed less than the first). 15 sits with real
    #    headroom on both sides of that gap.
    # 4. breadth (count of types scoring >= 3) <= 6 — catches a shape bars
    #    1-3 miss: a document that discusses MANY types roughly evenly,
    #    where the winner still clears bars 1-3 on volume alone even
    #    though nothing actually dominates. Confirmed live with a
    #    synthetic 10-section reference PDF (one clearly-written section
    #    per type: motor/health/life/travel/home/marine/cyber/liability/
    #    fire/crop) — health won 11 vs runner-up motor=5, clearing bars
    #    1-3 (11>=10, 11>=5*1.5, 5<=15) despite the document having no
    #    real dominant topic at all; 9 types scored >=3 there. Every
    #    CONFIRMED single-topic module in this KB has at most 4 types
    #    scoring >=3 (nothing else accumulates much evidence); both
    #    confirmed multi-topic references have 10-11. 6 sits with real
    #    headroom on both sides of that gap, same discipline as bar 3.
    breadth = sum(1 for v in totals.values() if v >= 3)
    if best_score >= 10 and best_score >= runner_up * 1.5 and runner_up <= 15 and breadth <= 6:
        margin = best_score / (best_score + runner_up) if runner_up else 1.0
        return best_type, margin
    return "general", 0.0


def _verify_enrich_label_list() -> str:
    # Full descriptions, not just bare names — confirmed live this prompt
    # used to hand the model only a comma-separated name list (unlike
    # _build_policy_type_prompt's richer label_list), and a text unambiguously
    # about marine cargo (ship, vessel, bill of lading, sea/air/inland
    # transit) got confidently (95%) confirmed as "commercial" instead of
    # corrected to "marine" — with no description in front of it, the model
    # had nothing to weigh the surface-plausible-but-wrong label against.
    # Matches the existing richer prompt's format for the same reason it
    # exists there: names alone from a whole-industry vocabulary don't
    # reliably disambiguate close pairs.
    return "\n".join(
        f"  - {pt}: {info['desc']}"
        for pt, info in get_active_vocab().items()
    )


def _verify_enrich_step1_fields(assigned_type: str) -> tuple[str, str, str]:
    """Returns (step1_instructions, reply_fields, confidence_desc) for ONE
    section — factored out of _build_verify_and_enrich_prompt so the single
    -section and batched prompt builders share byte-identical wording."""
    # "general" isn't a real topic to verify — it means the first pass found
    # no confident single type, not "this text is about general insurance."
    # Asking "does this discuss general insurance?" is a question the model
    # will just agree with (confirmed live: VERIFIED=yes, confidence=80, on
    # text that was unambiguously about cyber insurance) since nothing in
    # the text actively contradicts "general." The real question when the
    # first pass came up empty is the opposite framing: actively look for
    # ONE specific type before accepting "general" as the answer.
    #
    # The IDENTIFY branch used to reuse the VERIFY branch's VERIFIED=<yes/no>
    # + CORRECTED_TYPE=<type/"same"> fields, but that pairing has no real
    # "no" case to verify against when the baseline is already "general" —
    # confirmed live the model settled into a self-contradictory pattern,
    # VERIFIED=no paired with CORRECTED_TYPE=same, on multiple different
    # unambiguous texts (cyber: hacker/ransomware/data-breach content, home:
    # house/building/contents/burglary content) across repeated identical
    # calls. "no" was being used to hedge ("I'm not fully confirming general")
    # without committing to naming an alternative, and since CORRECTED_TYPE
    # was "same" the override correctly never fired — but that meant the
    # IDENTIFY path silently never promoted anything out of "general" for
    # these texts at all. A single unambiguous field removes the room for
    # that contradiction: there's nothing to say "no" to.
    if assigned_type == "general":
        step1 = """STEP 1 — IDENTIFY: an initial keyword pass found no confident single type
for this text. Read it and decide: does it discuss ONE particular insurance
type clearly enough to name it specifically? Don't default to "general" just
because the text doesn't use an exact textbook phrase — judge the actual
subject matter (e.g. text about hackers, data breaches, and ransomware is
about cyber insurance even if it never says the words "cyber insurance").
- Judge by what TRIGGERS a claim and what the policy PAYS FOR — not by
  incidental nouns mentioned only in a passing illustrative example. A
  passage about a professional being sued for negligent advice is liability
  insurance even if its example happens to be "a software consultant whose
  faulty code crashes a client's system" — the trigger is a negligence
  claim over professional judgment, not a hack or data breach, so it is
  NOT cyber insurance just because the example mentions software.
- If a single type clearly applies, put its label in IDENTIFIED_TYPE below.
- If the text genuinely discusses multiple different types with no single
  dominant one, or truly can't be pinned to any specific type, put "general"
  in IDENTIFIED_TYPE — that is a correct, expected answer here, not a
  failure to identify something.
- SAME ANSWER — "general" — when the text IS confidently, singularly about
  ONE clear, coherent insurance product, but that product is genuinely
  DIFFERENT from every type listed below (e.g. drone/UAV insurance, wedding
  insurance, event-cancellation insurance — real, specific products that
  just aren't in this particular list). This is a DIFFERENT reason for
  "general" than the multi-topic/can't-pin-down case above, but the same
  correct answer here: "general" is also how you flag "this needs its own
  type that isn't in my list" so a separate step can identify what that new
  type actually is. Do NOT force-fit it into whichever listed type sounds
  closest just because it shares some surface vocabulary (e.g. a drone
  crashing is NOT motor insurance just because "collision" and "crash" also
  appear in motor policies) — that vocabulary overlap is real but shallow;
  the actual product, its triggers, and what it pays for are unrelated."""
        reply_fields = "IDENTIFIED_TYPE=<the single policy type that clearly applies, or \"general\" if none does>"
        confidence_desc = "how confident you are in this identification"
    else:
        step1 = f"""STEP 1 — VERIFY: this text has been provisionally tagged policy_type="{assigned_type}"
(see DOCUMENT CONTEXT below if shown, for why). Does this text genuinely,
primarily discuss "{assigned_type}" insurance?
- If yes, confirm it.
- If no, state the ONE type it actually discusses instead — or "general" if it
  genuinely discusses multiple different types with no single dominant one.
- Also answer "general" (not a guess at the closest listed type) if the text
  is confidently about ONE clear, coherent insurance product that is simply
  DIFFERENT from every type in the label list below — e.g. drone/UAV
  insurance, wedding insurance, event-cancellation insurance. The
  provisional "{assigned_type}" tag came from a keyword match, not a real
  read of the content, so don't let it anchor you toward confirming a
  closest-sounding type that shares surface vocabulary (e.g. "collision" or
  "crash" appearing) without actually being that product."""
        reply_fields = ("VERIFIED=<yes/no>\n"
                         "CORRECTED_TYPE=<the correct type if VERIFIED=no, otherwise write \"same\">")
        confidence_desc = "how confident you are in this verification"

    return step1, reply_fields, confidence_desc


def _verify_enrich_doc_context(doc_prior: str) -> str:
    # Confirmed live (2026-07-31): classifying a chunk with zero visibility
    # into its own document flips genuinely on-topic chunks to the wrong
    # type — a travel guide's own "acute toothache during the journey" travel-
    # medical benefit, read in isolation, looks like generic health content
    # and got reclassified "health". A document-level prior fixes this
    # without becoming a rubber stamp: it's a real signal to weigh, not a
    # verdict — a chunk substantively about a DIFFERENT type (explains that
    # type's own rules/benefits/procedures, not just an incidental word)
    # should still be classified as that other type. See
    # derive_document_topic_prior()'s own docstring for the confirmed
    # failure case and plan_policy_type_tagging.md for the fuller writeup.
    if not doc_prior or doc_prior == "general":
        return ""
    return f"""
- DOCUMENT CONTEXT: this text is one excerpt from a document whose OTHER
  sections are, on the whole, genuinely about **{doc_prior} insurance** — most
  chunks from this document are {doc_prior}. Default to {doc_prior} unless
  THIS excerpt gives you a clear, specific reason not to.
  - Generic insurance vocabulary is NOT that reason. Words like "treatment",
    "hospital", "illness", "claim", "exclusion", "benefit", or "claims
    procedure" appear inside every type of policy, including {doc_prior}
    itself — a {doc_prior} policy's OWN benefits, exclusions, and claims
    process will naturally use this vocabulary. Seeing these words is not
    evidence the excerpt is some OTHER type; it is what {doc_prior}
    insurance documents are made of. The earlier instruction to classify an
    excerpt by "that other type's own rules, benefits, or claims procedure"
    means a SELF-CONTAINED product with its own separate rules — not any
    sentence that happens to mention treatment or a claim.
  - A medical/treatment passage framed around a JOURNEY, TRIP, or being
    ABROAD/ON HOLIDAY (e.g. "treatment that began during the journey",
    "medical expenses incurred abroad") is the travel policy's OWN medical
    cover, not standalone health insurance — travel policies routinely
    include medical benefits as part of what they are; that does not make
    them health insurance.
  - A depreciation or age-deduction schedule listing PERSONAL BELONGINGS
    (electronics, phones, bicycles, sports gear, clothing, bags) — with no
    mention of a building, dwelling, or burglary — describes what a person
    carries or owns, not a standalone home/property insurance policy. Words
    like "household items" or "property is repaired" appearing in such a
    schedule are describing loss/damage to items a person had WITH them,
    which is {doc_prior}'s own belongings cover if {doc_prior} routinely
    covers what people carry (e.g. travel baggage, personal effects) — not
    evidence of a separate home-insurance document.
  - Only override {doc_prior} if the excerpt names or clearly describes a
    DIFFERENT, SELF-CONTAINED insurance product that could not plausibly be
    a clause of a {doc_prior} policy — e.g. it discusses a named motor
    own-damage cover, a life-insurance sum-assured/nomination mechanism, or
    a standalone health policy's hospital network/room-rent cap as its own
    subject, with no connection to {doc_prior}. A single sentence reusing
    shared insurance vocabulary is not enough to override."""


_VERIFY_ENRICH_WATCH_RULES = """- WATCH FOR CONTRAST: text naming a DIFFERENT type specifically to distinguish
  itself from it ("unlike a health plan...", "not like ongoing treatment
  cover...", "irrelevant here in a way it would matter for a scheme built
  around X...") is telling you it is NOT that other type — the other type's
  vocabulary appearing in a sentence like that is evidence AGAINST that type,
  not for it, even though the literal words are present.
- WATCH FOR EXCLUSIONS: the same applies to a type mentioned only to say the
  text does NOT cover it ("does not cover", "excluded", "not covered",
  "unless") — that is also evidence AGAINST that type, not for it. The type
  a text is ABOUT is the one whose own rules, benefits, or claims procedure
  it actually states, not any type it names while denying or comparing.
  Example: "Exclusions in luggage insurance ... does not cover ... motorised
  vehicles ... electric vehicles for which motor liability insurance is
  required." is TRAVEL insurance (a real KB chunk once mistagged "motor" for
  exactly this reason) — "motor" appears purely to name what luggage cover
  excludes.
  Example: "...in General Insurance the cover is granted normally for one
  year and in Fire Insurance the preamble states..." is LIABILITY insurance,
  not fire (another real KB chunk mistagged this way) — a passing
  cross-reference used only to illustrate a general point about renewal
  notices, not the text's actual subject."""

_VERIFY_ENRICH_STEP2_BLOCK = """STEP 2 — EXTRACT (from the text only — never guess or invent a value): for each
field below, give the value ONLY if it is explicitly stated in the text,
otherwise write exactly "unknown".
- language: the language the text is written in
- jurisdiction: a specific country, state, or region the text is legally scoped to
- document_version: an edition, version number, or year explicitly stated for this document
- effective_date: an effective or commencement date explicitly stated
- coverage_category: a specific named cover variant (e.g. "comprehensive", "third-party", "family floater") if the text is about ONE specific variant"""


def _build_verify_and_enrich_prompt(text: str, assigned_type: str, doc_prior: str = "") -> str:
    label_list = _verify_enrich_label_list()
    step1, reply_fields, confidence_desc = _verify_enrich_step1_fields(assigned_type)
    doc_context = _verify_enrich_doc_context(doc_prior)

    # Investigating (2026-08): a full 414-chunk retag pass consistently
    # misclassified the SAME ~7 travel-guide chunks the SAME wrong way,
    # with zero connection/exception errors, while an isolated run of just
    # those chunks (identical code, identical prompt) always came back
    # correct. The vLLM server's own /metrics shows a 66% prefix-cache hit
    # rate server-wide (prefix_cache_hits_total / prefix_cache_queries_total)
    # on a shared, actively-used deployment — every one of these prompts
    # shares a long, mostly-fixed instructional prefix across hundreds of
    # calls in one run, which is exactly the shape automatic prefix caching
    # targets. This nonce forces every call's prompt to be byte-unique, so
    # vLLM can never find a matching cached prefix to (correctly or
    # incorrectly) reuse across chunks — trading away a cache-hit speed
    # benefit this offline batch workload doesn't need, in exchange for
    # ruling out cross-request cache-state bleed as a cause. Not confirmed
    # as THE root cause yet — this is the test, not a verified fix.
    _cache_buster = f"[ref:{uuid.uuid4().hex[:12]}]\n\n"
    return _cache_buster + f"""You are verifying and enriching metadata for a section of an insurance document.

{step1}
{_VERIFY_ENRICH_WATCH_RULES}{doc_context}

{_VERIFY_ENRICH_STEP2_BLOCK}

Available policy types:
{label_list}
  - general: text covers multiple types, is generic about insurance, or the type cannot be determined

Use the EXACT label word shown above (e.g. "motor", not "auto" or "car"; "home",
not "property" or "household") — these are fixed labels an automated system
matches on, not free-text description.

TEXT:
{text[:4000]}

Reply in EXACTLY this format, one line per field, no explanation, no extra text:
{reply_fields}
CONFIDENCE=<a number 0-100, {confidence_desc}>
LANGUAGE=<value or unknown>
JURISDICTION=<value or unknown>
DOCUMENT_VERSION=<value or unknown>
EFFECTIVE_DATE=<value or unknown>
COVERAGE_CATEGORY=<value or unknown>"""


def _build_verify_and_enrich_batch_prompt(sections: list[tuple[str, str]], doc_prior: str = "") -> str:
    """
    Combined-prompt builder for verify_and_enrich_sections_batch() — verifies
    and enriches ALL of a document's sections in one call instead of one call
    per section. Reuses _verify_enrich_step1_fields/_verify_enrich_doc_context
    and the same WATCH/STEP2 wording as _build_verify_and_enrich_prompt
    verbatim — hoisted OUT of the per-section loop so shared pieces (label
    list, WATCH rules, doc context, STEP 2 rules) appear once instead of once
    per section. This isn't just tidiness: confirmed live 2026-08-19 that
    repeating them per section blew an 8-section batch past Groq's TPM
    (tokens-per-minute) limit outright (12698 tokens requested vs an 8000
    cap) even though it was still only ONE request — see
    verify_and_enrich_sections_batch()'s token-budget sub-batching for the
    other half of that fix.

    sections: list of (text, assigned_type) tuples, in the order results
    should be returned.
    """
    label_list = _verify_enrich_label_list()
    doc_context = _verify_enrich_doc_context(doc_prior)

    section_blocks = []
    for i, (text, assigned_type) in enumerate(sections, start=1):
        step1, reply_fields, confidence_desc = _verify_enrich_step1_fields(assigned_type)
        section_blocks.append(f"""
=== SECTION {i} ===
{step1}

TEXT:
{text[:4000]}

Reply for SECTION {i} in EXACTLY this format, one line per field, no explanation:
{reply_fields}
CONFIDENCE=<a number 0-100, {confidence_desc}>
LANGUAGE=<value or unknown>
JURISDICTION=<value or unknown>
DOCUMENT_VERSION=<value or unknown>
EFFECTIVE_DATE=<value or unknown>
COVERAGE_CATEGORY=<value or unknown>""")

    _cache_buster = f"[ref:{uuid.uuid4().hex[:12]}]\n\n"
    return _cache_buster + f"""You are verifying and enriching metadata for {len(sections)} DIFFERENT sections
of the SAME insurance document. Apply the reasoning below to EACH section
entirely independently — judge every section only by its own TEXT, never by
another section's content. These rules apply to EVERY section below:

{_VERIFY_ENRICH_WATCH_RULES}{doc_context}

{_VERIFY_ENRICH_STEP2_BLOCK}

Available policy types:
{label_list}
  - general: text covers multiple types, is generic about insurance, or the type cannot be determined

Use the EXACT label word shown above (e.g. "motor", not "auto" or "car"; "home",
not "property" or "household") — these are fixed labels an automated system
matches on, not free-text description.
{"".join(section_blocks)}

Reply with EXACTLY {len(sections)} blocks, one per section, each starting with
its own "=== SECTION i ===" marker exactly as shown above, in order, no extra
commentary before, after, or between blocks."""


def verify_and_enrich_section_metadata(
    text: str,
    assigned_type: str,
    llm: Any = None,
    *,
    doc_prior: str = "",
) -> dict:
    """
    Step 2: LLM verify + enrich, once per section.

    Returns a dict with "policy_type" (the verified/corrected type) plus
    the _ENRICHMENT_FIELDS, each either an extracted value or "unknown".
    "unknown" is the CORRECT, expected answer when the text doesn't state
    a field explicitly (most sections in a fixed textbook-style KB won't
    name a jurisdiction or effective date) — it is not a failure.

    doc_prior: the document's own dominant policy_type (see
    derive_document_topic_prior()), or "" / "general" for a genuinely
    multi-topic document. Passed straight through to the prompt as
    context, not used for any code-level override here — see
    _build_verify_and_enrich_prompt for how it's weighed.

    Falls back to {"policy_type": assigned_type, ...all "unknown"} when no
    LLM is available or the call fails; callers already have assigned_type
    from regex_first_pass_policy_type() as that fallback baseline.
    """
    result: dict = {"policy_type": assigned_type}
    result.update({field: "unknown" for field in _ENRICHMENT_FIELDS})

    if llm is None:
        return result

    try:
        prompt = _build_verify_and_enrich_prompt(text, assigned_type, doc_prior)
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        result = _parse_verify_and_enrich_reply(raw, assigned_type)
    except Exception as exc:
        logger.warning("[POLICY_TYPE] verify/enrich LLM call failed: %s — keeping first-pass assignment", exc)

    return result


def _parse_verify_and_enrich_reply(raw: str, assigned_type: str) -> dict:
    """
    Parse ONE section's reply block (whatever text followed its own
    "=== SECTION i ===" marker in a batched call, or the whole response for
    a single-section call) into the same {"policy_type": ..., **enrichment
    fields} shape verify_and_enrich_section_metadata() returns. Factored out
    of that function so verify_and_enrich_sections_batch() parses each of
    its N reply blocks with identical logic.
    """
    result: dict = {"policy_type": assigned_type}
    result.update({field: "unknown" for field in _ENRICHMENT_FIELDS})

    def _field(name: str, default: str = "") -> str:
        m = re.search(rf"{name}\s*=\s*(.+)", raw, re.IGNORECASE)
        return m.group(1).strip().strip('"').strip() if m else default

    try:
        confidence = float(re.sub(r"[^\d.]", "", _field("CONFIDENCE", "0")) or 0)
    except ValueError:
        confidence = 0.0

    # Two distinct reply shapes matching the two prompt branches in
    # _verify_enrich_step1_fields — see that function for why IDENTIFY gets
    # its own single field instead of reusing VERIFIED/CORRECTED_TYPE.
    if assigned_type == "general":
        identified = _normalize_policy_type(_field("IDENTIFIED_TYPE", "general").lower())
        if identified != "general" and identified in _valid_policy_types():
            if confidence >= _VERIFY_OVERRIDE_THRESHOLD:
                logger.info(
                    "[POLICY_TYPE] identified: general -> %r (confidence=%.0f)",
                    identified, confidence,
                )
                result["policy_type"] = identified
            else:
                logger.debug(
                    "[POLICY_TYPE] identification suggested general -> %r but confidence %.0f "
                    "< threshold %d — keeping general",
                    identified, confidence, _VERIFY_OVERRIDE_THRESHOLD,
                )
    else:
        verified = _field("VERIFIED", "yes").lower().startswith("y")
        corrected = _normalize_policy_type(_field("CORRECTED_TYPE", "same").lower())
        if (
            not verified
            and corrected not in ("same", "", assigned_type.lower())
            and corrected in _valid_policy_types()
        ):
            if confidence >= _VERIFY_OVERRIDE_THRESHOLD:
                logger.info(
                    "[POLICY_TYPE] verification override: %r -> %r (confidence=%.0f)",
                    assigned_type, corrected, confidence,
                )
                result["policy_type"] = corrected
            else:
                logger.debug(
                    "[POLICY_TYPE] verification suggested %r -> %r but confidence %.0f "
                    "< threshold %d — keeping first-pass %r",
                    assigned_type, corrected, confidence, _VERIFY_OVERRIDE_THRESHOLD, assigned_type,
                )

    for field in _ENRICHMENT_FIELDS:
        value = _field(field.upper(), "unknown").lower()
        if value and value != "unknown":
            result[field] = value

    return result


# A single combined batch call collapses REQUEST count, but if enough
# sections (or long enough ones) are packed into it, the request's own
# TOKEN count can exceed Groq's TPM (tokens-per-minute) cap outright —
# confirmed live 2026-08-19: an 8-section combined call measured 12698
# prompt tokens against gpt-oss-120b's 8000 TPM limit on this account and
# Groq rejected it wholesale (413), rather than throttling or queuing it.
# These constants keep each sub-batch's estimated PROMPT size well under
# that ceiling, leaving headroom both for the COMPLETION tokens the same
# call will also consume (see _batch_max_tokens below — TPM counts
# prompt+completion together) and for whatever else concurrently shares
# the same account's TPM budget (summary generation, doc-type
# classification, etc.).
_BATCH_TOKEN_BUDGET = 3000
_BATCH_SECTION_OVERHEAD_TOKENS = 300  # rough cost of one section's step1 + reply-format lines, excluding its TEXT
_BATCH_GROUP_PACING_SECONDS = 3       # gap between sub-batch calls so Groq's rolling TPM window can recover
# Hard cap on sections per group, independent of the token-budget grouping
# above. The token budget bounds PROMPT size, but a group of many small
# sections can still need a large COMPLETION (one full reply block per
# section) even when their combined prompt text is tiny — confirmed live
# 2026-08-19: a 9-section group (all short text, one group under the old
# prompt-only budget) came back with only 5 of 9 reply blocks, the
# completion cut off mid-batch by max_tokens sized for a single section's
# reply. Capping section COUNT per group bounds completion size directly.
_MAX_SECTIONS_PER_GROUP = 4
# GROQ_CLASSIFICATION_MODEL is a reasoning model — see get_classification_
# llm()'s docstring in router.py: max_tokens has to cover BOTH the model's
# internal reasoning tokens (a fixed-ish floor, not headroom for the
# visible answer) AND the actual per-section reply blocks, which scale
# with how many sections are in this call. A single-section call's
# existing default (500) already covers the floor for a n=1 case, but a
# batched call needs floor + (n_sections * a real per-block budget).
_BATCH_REASONING_FLOOR_TOKENS = 500
_BATCH_COMPLETION_TOKENS_PER_SECTION = 220


def _batch_max_tokens(n_sections: int) -> int:
    return _BATCH_REASONING_FLOOR_TOKENS + _BATCH_COMPLETION_TOKENS_PER_SECTION * n_sections


def _estimate_tokens(s: str) -> int:
    """Rough chars/4 estimate — good enough to size sub-batches conservatively,
    not meant to match Groq's real tokenizer exactly."""
    return max(1, len(s) // 4)


def _group_sections_by_token_budget(
    sections: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    """
    Split `sections` into ordered groups whose estimated combined prompt
    size stays under _BATCH_TOKEN_BUDGET AND whose section COUNT stays
    under _MAX_SECTIONS_PER_GROUP (bounds the completion side too — see
    that constant's comment), so verify_and_enrich_sections_batch can send
    each group as its own request instead of risking one oversized
    combined call. A single section that alone exceeds the token budget
    still gets its own group (nothing left to split further) rather than
    being dropped.
    """
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_tokens = 0
    for text, assigned_type in sections:
        section_tokens = _estimate_tokens(text[:4000]) + _BATCH_SECTION_OVERHEAD_TOKENS
        if current and (
            current_tokens + section_tokens > _BATCH_TOKEN_BUDGET
            or len(current) >= _MAX_SECTIONS_PER_GROUP
        ):
            groups.append(current)
            current, current_tokens = [], 0
        current.append((text, assigned_type))
        current_tokens += section_tokens
    if current:
        groups.append(current)
    return groups


def verify_and_enrich_sections_batch(
    sections: list[tuple[str, str]],
    llm: Any = None,
    *,
    doc_prior: str = "",
) -> list[dict]:
    """
    Batched version of verify_and_enrich_section_metadata() — verifies and
    enriches every section of a document in as few LLM calls as possible
    instead of one call per section.

    Exists because the per-section version is called unconditionally for
    every section during ingestion (never regex-gated, unlike
    classify_chunk_intent) — a document with N sections previously made N
    separate Groq requests back-to-back with no throttling, which reliably
    tripped Groq's per-minute REQUEST rate limit on multi-section documents
    (confirmed live 2026-08-19: an 8-section PDF produced a burst of 429s
    with 18-22s retry backoffs during ingestion).

    Sections are grouped by _group_sections_by_token_budget() rather than
    always sent as one call — a single oversized combined request can trip
    Groq's separate TOKEN rate limit instead (see _BATCH_TOKEN_BUDGET's
    comment), so this keeps each request's own size bounded and paces
    consecutive sub-batch calls _BATCH_GROUP_PACING_SECONDS apart. This
    still collapses what used to be N requests down to a small, roughly
    constant number regardless of section count (typically 1-2 for a
    normal document), and runs entirely inside a background thread so the
    added pacing delay costs nothing user-facing.

    sections: list of (text, assigned_type) tuples, in the order results
    should be returned. Falls back to {"policy_type": assigned_type, ...all
    "unknown"} for any section whose sub-batch call fails or whose reply
    can't be parsed — same fail-safe behavior as the single-section
    function, just applied per sub-batch rather than one call at a time.
    """
    results = [
        {"policy_type": assigned_type, **{f: "unknown" for f in _ENRICHMENT_FIELDS}}
        for _, assigned_type in sections
    ]
    if llm is None or not sections:
        return results

    groups = _group_sections_by_token_budget(sections)
    offset = 0
    for gi, group in enumerate(groups):
        if gi > 0:
            time.sleep(_BATCH_GROUP_PACING_SECONDS)
        try:
            prompt = _build_verify_and_enrich_batch_prompt(group, doc_prior)
            # Override max_tokens for this call via .bind() rather than
            # reconstructing the LLM — the caller's llm instance is shared
            # across other classification calls (classify_candidate_type
            # etc.) that need the smaller single-item default. A combined
            # multi-section reply needs more room than that default covers
            # (see _batch_max_tokens' comment) — confirmed live 2026-08-19:
            # without this override, a 9-section reply came back truncated
            # to 5 of 9 blocks under the single-section max_tokens default.
            response = llm.bind(max_tokens=_batch_max_tokens(len(group))).invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)

            blocks = re.split(r"===\s*SECTION\s+\d+\s*===", raw)[1:]
            if len(blocks) != len(group):
                logger.warning(
                    "[POLICY_TYPE] batch verify/enrich sub-batch %d/%d returned %d block(s), "
                    "expected %d — keeping first-pass assignment for this sub-batch",
                    gi + 1, len(groups), len(blocks), len(group),
                )
            else:
                for j, (block, (_, assigned_type)) in enumerate(zip(blocks, group)):
                    results[offset + j] = _parse_verify_and_enrich_reply(block, assigned_type)

        except Exception as exc:
            logger.warning(
                "[POLICY_TYPE] batch verify/enrich sub-batch %d/%d LLM call failed: %s — "
                "keeping first-pass assignment for this sub-batch",
                gi + 1, len(groups), exc,
            )

        offset += len(group)

    return results


# Generic insurance vocabulary excluded from candidate-hint keywords — these
# words appear in text about EVERY policy type, so keeping them would make
# match_candidate_vocab() fire on almost anything instead of the words that
# actually distinguish one novel topic from another.
_CANDIDATE_STOPWORDS = frozenset({
    "this", "that", "these", "those", "with", "from", "have", "will",
    "shall", "would", "could", "should", "which", "their", "there",
    "policy", "policies", "insurance", "insurer", "insured", "cover",
    "covered", "coverage", "claim", "claims", "amount", "period", "under",
    "such", "also", "only", "than", "when", "where", "into", "your", "each",
    "hereby", "herein", "whereas", "provided",
    # Confirmed live (2026-08-06): "premium" and "risk"/"risks" are as
    # universal to insurance text as "policy"/"cover"/"claim" above, yet
    # were missing here — they dominated the frequency-ranked top-8 for
    # almost every candidate in candidate_vocab.json regardless of that
    # candidate's actual topic (present in 6+ of 17 unrelated candidate
    # entries), and got promoted live into micro_insurance's active regex
    # list, where they silently inflated its score across totally
    # unrelated documents (a life-insurance module, a motor module, a
    # liability module) enough to break derive_document_topic_prior()'s
    # dominance bars for those documents. See policy_type_retag.py's
    # 2026-08-06 dry-run for the full regression this caused.
    "premium", "premiums", "risk", "risks",
    # Confirmed live (2026-08-10, off-vocab regression test): a real
    # candidate entry ("subrogation" -- itself a legal doctrine, not a
    # product, since removed from candidate_vocab.json) had accumulated
    # exactly this same class of universal insurance-domain words as its
    # stored keywords, and a genuinely unrelated synthetic drone-insurance
    # chunk cheap-matched it purely on 2+ of them co-occurring, well
    # before match_candidate_vocab()'s LLM step ever ran. "loss" and
    # "compensation" are the basis of every insurance claim regardless of
    # product; "indemnity", "principle", and "cause" are equally generic
    # legal/theoretical vocabulary (principle of indemnity, proximate
    # cause) rather than anything distinctive of one specific product.
    "loss", "losses", "compensation", "indemnity", "principle", "principles",
    "cause", "causes",
})

# General-English function/filler words, on top of the domain list above.
# A plain "not an insurance word" filter still lets ordinary connective and
# document-boilerplate words through — confirmed live: a document's opening
# ("Understanding Pet Insurance: A Complete Guide...") produced keywords
# "understanding", "complete", "guide", "right", "choosing" purely because
# they appeared early, and one of those (a later chunk's "plan", from
# "wellness plan") went on to falsely match a completely unrelated document
# ("...a specialized protection plan for...") that happened to share the
# same ordinary scaffolding word. Frequency ranking below is the primary
# defense; this list is the cheap first-pass filter in front of it.
_GENERAL_STOPWORDS = frozenset({
    "about", "after", "before", "between", "during", "over", "onto",
    "upon", "while", "then", "than", "some", "many", "most", "more",
    "much", "every", "both", "other", "another", "same", "different",
    "various", "based", "typically", "generally", "usually", "often",
    "always", "never", "sometimes", "specifically", "particularly",
    "especially", "understanding", "complete", "guide", "right",
    "choosing", "prepared", "apply", "history", "enrollment", "advisors",
    "because", "however", "therefore", "although", "though", "since",
    "being", "been", "were", "does", "doing", "done", "having", "here",
    "what", "when", "why", "how", "who", "whom", "whose", "plan", "plans",
    # Confirmed live (2026-08-06) alongside the _CANDIDATE_STOPWORDS
    # additions above — same failure, plain-English filler/pronoun words
    # this list's net was supposed to catch but didn't: "they" (pronoun,
    # "their" was already here but not this form), "people"/"income"/
    # "poor" (generic-content words, same shape as "understanding"/
    # "complete"/"guide" already excluded above) — all four ranked into
    # micro_insurance's promoted keyword list purely from appearing often
    # in a small sample, with zero topic-specificity.
    "they", "them", "people", "income", "poor",
})


def _is_duplicate_of_existing_type(label: str, active_types: set) -> bool:
    """
    True if `label` (e.g. "crop_insurance", "personal_accident_cover") is
    just a rephrasing of a type ALREADY in the active vocabulary, not a
    genuinely new one. classify_candidate_type() is only ever reached
    after the CLOSED-vocabulary classifier already said "general" for
    this text — so when the open-ended LLM (which has zero knowledge of
    the 12-name hardcoded list) confidently names an EXISTING type back,
    that isn't evidence of a new type at all. It's evidence the closed
    classifier missed real, already-covered content and this text fell
    through to the open-ended fallback by mistake.
    Confirmed live 2026-07-23: candidate_vocab.json had accumulated
    "crop_insurance" (15), "life_insurance" (21), "health_insurance" (6),
    "marine_insurance" (6), "home_insurance" (5), "personal_accident_cover"
    (8), "marine_cargo_insurance" (5), "group_health_insurance" (5),
    "unit_linked_life_insurance" (1) — all duplicates of hardcoded types,
    none of them a real gap. Left unguarded, an automatic promotion step
    would have created parallel, conflicting entries for types that
    already exist.

    Two checks, either one is sufficient:
    - Word-level: split on "_" and normalize each word (reusing
      _normalize_policy_type's existing synonym table) since the LLM's
      answer is always "<topic> insurance"/"<topic> cover" style, never
      the bare internal key. Catches direct synonym renamings
      ("auto_insurance" -> "motor" via the synonym table) instantly with
      no extra work.
    - Classifier-level (added 2026-07-23): run the label's own natural
      phrase back through classify_query_policy_type() — reuses the same
      tuned regex/phrase list every real query is judged against, instead
      of a second hand-maintained word list that inevitably drifts out of
      sync with it. Confirmed live this catches real gaps the word-level
      check missed: "term_insurance" (4 guesses, all the identical query
      "What are the exclusions in term insurance?") and
      "transit_insurance" (8 guesses, all "Explain transit insurance in
      detail") both slipped through the word-level check — "term" and
      "transit" aren't in the synonym table — but their natural phrases
      ("term insurance", "transit insurance") directly match life's
      \\bterm insurance\\b and marine's \\btransit insurance\\b regex
      respectively, i.e. the closed classifier already names them
      correctly and always has. Neither is a real gap.
    """
    for word in label.split("_"):
        if _normalize_policy_type(word) in active_types:
            return True
    return classify_query_policy_type(label.replace("_", " ")) != "general"


def _extract_candidate_keywords(text: str) -> list[str]:
    """
    Pick the words that actually distinguish THIS text, ranked by how often
    they repeat within it — not just whichever non-stopword words happen to
    appear first. A word mentioned several times is far more likely to be
    the text's real subject than a one-off word from opening boilerplate,
    which is exactly the failure mode the stopword list alone can't catch
    (there's no fixed list of every possible boilerplate word). Ties
    (equal frequency) break by first-occurrence order for determinism.
    """
    tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for i, t in enumerate(tokens):
        if t in _CANDIDATE_STOPWORDS or t in _GENERAL_STOPWORDS:
            continue
        counts[t] = counts.get(t, 0) + 1
        first_seen.setdefault(t, i)
    ranked = sorted(counts, key=lambda t: (-counts[t], first_seen[t]))
    return ranked[:8]


def classify_candidate_type(
    text: str, llm: Any = None, *, source: str = "", source_type: str = "chunk",
    skip_cheap_match: bool = False,
) -> Optional[str]:
    """
    Mode-A handling (open-vocabulary fallback). Called only after the
    closed-vocabulary classifiers (regex + constrained LLM) both land on
    "general" for this text — never changes the OFFICIAL policy_type;
    callers always keep that as "general" no matter what this returns. It
    only ever produces a candidate label: free-text, logged, used purely
    for the reranking candidate-match bypass and as raw material for a
    future manual promotion review.

    Step 3 (cheap): check the already-discovered candidate vocabulary's
    keyword hints before paying for another LLM call — catches repeat
    instances of a novel topic that's already been seen once.
    Step 4 (LLM, no vocabulary constraint): only if step 3 also misses and
    an LLM is available. Returns None (no candidate at all) if the model's
    answer normalizes to a degenerate non-answer ("general", "unclear",
    etc.) — a real absence of a guess, not an empty-string label, so two
    unrelated "no idea" cases can never collide in the candidate-match
    bypass downstream.

    skip_cheap_match: force step 4 unconditionally, bypassing step 3
    entirely. For a one-time bulk re-classification pass (e.g. backfilling
    candidate_policy_type across many pre-existing chunks in one run), the
    normal per-request cost tradeoff that step 3 exists for doesn't apply —
    confirmed live this backfill scenario is exactly where keyword-overlap
    matching runs away: one early classification seeds generic domain
    vocabulary ("risk", "insurable", "interest", "assessment" — all common
    across broad insurance-law/textbook content, not distinctive to any one
    topic), which then cheap-matches most of the REST of the same run
    before the LLM ever gets a chance to independently disagree.
    """
    from candidate_vocab import match_candidate_vocab, normalize_candidate_label, upsert_candidate

    hit = None if skip_cheap_match else match_candidate_vocab(text)
    if hit:
        # Deliberately pass [] here, not _extract_candidate_keywords(text) —
        # a cheap keyword match was never verified by the LLM, so growing
        # the label's keyword set from it is how one over-broad match
        # compounds into an even broader one next time. Only a fresh LLM
        # classification (below) should ever widen the keyword list;
        # guess_count/last_seen still update either way.
        upsert_candidate(hit, [], source, source_type)
        return hit

    if llm is None:
        return None

    try:
        prompt = f"""This text is an insurance-related passage. Read it and decide: does it
describe a SPECIFIC, NAMEABLE insurance PRODUCT — something a customer
could actually go buy a policy for — in 1-3 words (e.g. "pet insurance",
"aviation insurance", "directors and officers liability")?

- If yes, reply with ONLY that name, lowercase, 1-3 words, nothing else.
- If the text is genuinely generic, covers multiple unrelated types, or
  isn't about one specific insurance product at all, reply with exactly:
  general
- Also reply "general" (not a guess) if the text is about a LEGAL DOCTRINE,
  insurance MECHANISM/CONCEPT, RISK-MANAGEMENT THEORY, or REGULATORY TOPIC
  that applies ACROSS many different products rather than naming one
  specific product itself — e.g. subrogation, coinsurance, total loss,
  indemnity, licensing/regulatory compliance, pure vs. speculative risk.
  These describe how insurance works or is regulated in general, not a
  product a customer buys, even though they're genuinely insurance-related
  and may be the clear main subject of the passage.

TEXT:
{text[:2000]}

ANSWER:"""
        response = llm.invoke(prompt)
        raw = (response.content if hasattr(response, "content") else str(response)).strip()
    except Exception as exc:
        logger.debug("[CANDIDATE_TYPE] open-ended LLM call failed: %s", exc)
        return None

    label = normalize_candidate_label(raw)
    if label is None:
        return None

    if _is_duplicate_of_existing_type(label, get_active_vocab().keys()):
        # Not a new type — evidence the closed classifier missed real
        # existing-type content. Surfaced distinctly so it isn't silently
        # lost, but never enters the candidate vocabulary (would corrupt
        # any promotion step's counts) and never overrides the "general"
        # official tag either — the closed-classifier miss is a real, but
        # separate, problem this function isn't responsible for fixing.
        logger.info(
            "[CANDIDATE_TYPE] closed-classifier miss suspected: open-ended "
            "guess %r duplicates an existing type (source=%r) — check "
            "regex/prompt coverage for that type instead of promoting this",
            label, source[:60],
        )
        return None

    upsert_candidate(label, _extract_candidate_keywords(text), source, source_type)
    logger.info("[CANDIDATE_TYPE] open-ended guess: %r (from %r, source=%r)", label, raw[:60], source[:60])
    return label


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