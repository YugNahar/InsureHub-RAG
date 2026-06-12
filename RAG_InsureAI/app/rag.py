"""
Insurance RAG Pipeline — ChromaDB Vector Edition with HyDE, Hybrid Search, and Citation Enforcement.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from typing import Optional

import pandas as pd
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from document_loader import load_document, load_url, extract_urls
from metadata_tagger import tag_document
from validator import detect_conflict, validate_grounding, validate_calculation
from calculator import compute_insurance_benefits
from router import get_insurance_llm, get_general_llm, get_active_model_info, VLLM_HOST, VLLM_MODEL
from prompt_template import (
    SCENARIO_PROMPT,
    INFORMATIONAL_PROMPT,
    COMPARISON_PROMPT,
    GENERAL_PROMPT,
    RAG_PROMPT,
)
from vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
RETRIEVE_K = 12
RERANK_K = 6
MAX_CONTEXT_CHARS = 6000
SUMMARY_MAX_CHARS = 20000

# ══════════════════════════════════════════════════════════════════════════════
# SECTION DETECTION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_SECTION_PATTERNS: dict[str, list[str]] = {
    "definitions": [
        r"\bdefin", r"\bmeans?\b", r"\bshall mean\b", r"\brefers? to\b",
        r"\binterpretation\b", r"\bglossary\b",
    ],
    "eligibility": [
        r"\beligib", r"\bminimum age\b", r"\bmaximum age\b", r"\bage limit\b",
        r"\bentry age\b", r"\binsured person\b", r"\bwho (can|may|is)\b",
        r"\bqualif", r"\brequirement\b", r"\bage of\b",
    ],
    "benefits": [
        r"\bbenefit\b", r"\bcoverage\b", r"\bcovers?\b", r"\bcompensation\b",
        r"\breimbursement\b", r"\bpayable\b", r"\blimit\b",
        r"\bsum insured\b", r"\bpayout\b", r"\bindemnity\b",
        r"\bmaximum benefit\b", r"\bschedule of benefit\b",
    ],
    "exclusions": [
        r"\bexclusion\b", r"\bnot cover", r"\bnot include", r"\bexclud",
        r"\bexcept\b", r"\bnot payable\b", r"\bvoid\b",
    ],
    "claims": [
        r"\bclaim\b", r"\bnotif", r"\bprocedure\b",
        r"\bsubmit\b", r"\bdocuments? required\b", r"\bfile a claim\b",
    ],
    "flight_delay": [
        r"\bflight delay\b", r"\btrip delay\b", r"\bdeparture delay\b",
        r"\bconsecutive hours?\b", r"\bhours?\s+delay\b", r"\bdelay benefit\b",
        r"\bdelay compensation\b", r"\btravel delay\b", r"\bflight delay benefit\b",
    ],
    "medical": [
        r"\bmedical expense", r"\bhospital\b", r"\bemergency medical\b",
        r"\bmedical treatment\b", r"\bmedical evacuation\b", r"\bmedical benefit\b",
    ],
    "baggage": [
        r"\bbaggage\b", r"\bluggage\b", r"\bpersonal effects\b",
        r"\bbaggage loss\b", r"\bbaggage delay\b", r"\bbaggage benefit\b",
    ],
}

def _detect_section(text: str) -> str:
    t = text.lower()
    scores = {s: sum(1 for p in pats if re.search(p, t)) for s, pats in _SECTION_PATTERNS.items()}
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else "general"

# ══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_INTENT_MAP = [
    (["hajj", "umrah", "pilgrimage", "mecca"], ["medical", "benefits", "eligibility", "definitions"]),
    (["flight delay", "trip delay", "delay benefit", "hours delay", "departure delay", "delayed flight", "hours delayed"], ["flight_delay", "benefits"]),
    (["medical", "hospital", "emergency treatment", "medical expense", "medical evacuation", "repatriation"], ["medical", "benefits"]),
    (["baggage", "luggage", "personal effects", "baggage loss", "baggage delay", "lost baggage"], ["baggage", "benefits"]),
    (["minimum age", "max age", "age limit", "how old", "entry age", "insured age", "age of insured"], ["eligibility", "definitions"]),
    (["eligib", "who can", "qualify", "qualification", "insured person"], ["eligibility", "definitions"]),
    (["exclusion", "not cover", "not included", "excluded", "except", "what is not", "not payable"], ["exclusions"]),
    (["claim", "file claim", "how to claim", "claim procedure", "notification", "submit claim", "report"], ["claims", "general"]),
    (["benefit", "coverage", "covered", "cover", "compensation", "limit", "reimburs", "payable", "sum insured"], ["benefits"]),
    (["definition", "what is", "what does", "mean", "define", "covered trip", "what counts"], ["definitions", "benefits"]),
    (["premium", "price", "cost", "rate", "how much does"], ["benefits", "general"]),
    (["duration", "how long", "trip length", "maximum trip", "days allowed", "consecutive"], ["eligibility", "benefits", "general"]),
    (["deductible", "excess", "self-insured", "out of pocket"], ["benefits", "definitions"]),
]

def _detect_intent(query: str) -> list[str]:
    q = query.lower()
    sections = []
    for keywords, sec_list in _INTENT_MAP:
        if any(_query_contains_term(q, kw) for kw in keywords):
            for s in sec_list:
                if s not in sections:
                    sections.append(s)
    return sections or ["benefits", "eligibility", "definitions", "exclusions", "claims", "general"]

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT ROUTING (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_DOCUMENT_ROUTING_MAP = [
    (["hajj", "umrah", "pilgrimage", "mecca", "rak travel", "outbound", "rak_travel"], "RAK_Travel_Outbound"),
    (["aig"], "AIG"),
    (["gig"], "GIG"),
    (["liva"], "LIVA"),
    (["rak"], "RAK"),
]

# ══════════════════════════════════════════════════════════════════════════════
# PLAN TIER DETECTION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_PLAN_TIERS = {
    "platinum": ["platinum", "plat"],
    "gold": ["gold"],
    "silver": ["silver"],
    "prime": ["prime"],
    "enhanced": ["enhanced"],
    "basic": ["basic"],
    "standard": ["standard"],
    "comprehensive": ["comprehensive", "comp"],
}

def _detect_plan_tier(query: str) -> Optional[str]:
    q = query.lower()
    for tier, kws in _PLAN_TIERS.items():
        if any(kw in q for kw in kws):
            return tier
    return None

def _query_contains_term(query_lower: str, term: str) -> bool:
    q = re.sub(r"[_\-]", " ", query_lower)
    t = re.sub(r"[_\-]", " ", term.lower()).strip()
    if not t:
        return False
    if " " in t:
        return t in q
    return re.search(rf"\b{re.escape(t)}\b", q) is not None

def _route_to_documents(query: str, available_sources: list[str]) -> Optional[list[str]]:
    q = query.lower()
    matched_sources = []
    for keywords, tag in _DOCUMENT_ROUTING_MAP:
        if not any(_query_contains_term(q, kw) for kw in keywords):
            continue
        tag_lower = tag.lower().replace("_", " ").replace("-", " ")
        matched = [s for s in available_sources if tag_lower in s.lower().replace("_", " ").replace("-", " ")]
        for src in matched:
            if src not in matched_sources:
                matched_sources.append(src)
    if matched_sources:
        logger.info("[DOC ROUTER] Routed to %s", matched_sources)
        return matched_sources
    return None

# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL LOGIC DETECTOR (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_CONDITION_TRIGGERS = [
    r"\bonly if\b", r"\bunless\b", r"\bprovided that\b", r"\bsubject to\b",
    r"\bin the event\b", r"\bprovided\b", r"\bexcept\b", r"\bin case\b",
    r"\bif and only\b", r"\bcontingent\b", r"\bconditional\b",
]

def _extract_condition_hint(chunks: list[Document]) -> Optional[str]:
    conditions_found = []
    for chunk in chunks:
        text = chunk.page_content
        for pat in _CONDITION_TRIGGERS:
            for sent in re.split(r'[.\n]', text):
                if re.search(pat, sent, re.IGNORECASE) and len(sent.strip()) > 20:
                    conditions_found.append(sent.strip())
                    break
    if conditions_found:
        unique = list(dict.fromkeys(conditions_found))[:4]
        return "CONDITIONAL CLAUSES FOUND — handle with 'Covered only if …':\n" + "\n".join(f"  • {c}" for c in unique)
    return None

# ══════════════════════════════════════════════════════════════════════════════
# REGEX EXTRACTION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_AGE_RE = re.compile(r'\b(\d+\s*(?:days?|years?|months?))\b', re.IGNORECASE)
_MONEY_RE = re.compile(r'(?<!\d)((?:USD|US\$|\$|QAR|AED|SAR|OMR|BHD|KWD)\s*[\d,]+(?:\.\d+)?)(?!\d)', re.IGNORECASE)
_PLAIN_AMOUNT_RE = re.compile(r'(?<![,\d\$])(\d{1,3}(?:,\d{3})+)(?!\d)')
_DURATION_RE = re.compile(r'\b(\d+\s*(?:consecutive\s+)?days?)\b', re.IGNORECASE)
_PERCENT_RE = re.compile(r'\b(\d+(?:\.\d+)?\s*%)\b')

def _find_amounts(text: str) -> list[str]:
    with_prefix = _MONEY_RE.findall(text)
    if with_prefix:
        return list(dict.fromkeys(with_prefix))
    plain = _PLAIN_AMOUNT_RE.findall(text)
    return list(dict.fromkeys(plain))

def _try_regex_extract(query: str, chunks: list[Document]) -> Optional[str]:
    combined = "\n".join(c.page_content for c in chunks)
    q = query.lower()
    if any(w in q for w in ["minimum age", "max age", "age limit", "how old", "entry age"]):
        matches = list(dict.fromkeys(_AGE_RE.findall(combined)))
        if matches:
            return f"AGE VALUES FOUND IN POLICY: {', '.join(matches[:5])}"
    if any(w in q for w in ["how much", "amount", "limit", "maximum", "coverage amount", "usd", "compensation", "benefit amount", "sum insured"]):
        matches = _find_amounts(combined)
        if matches:
            return f"MONETARY AMOUNTS FOUND IN POLICY: {', '.join(matches[:8])}"
    if any(w in q for w in ["how long", "duration", "maximum trip", "trip length", "days allowed"]):
        matches = list(dict.fromkeys(_DURATION_RE.findall(combined)))
        if matches:
            return f"DURATION VALUES FOUND IN POLICY: {', '.join(matches[:5])}"
    return None

def _try_direct_answer(query: str, chunks: list[Document]) -> Optional[str]:
    q = query.lower()
    # Age queries
    if any(w in q for w in ["minimum age", "max age", "age limit", "how old", "entry age", "minimum entry", "age of insured", "age requirement", "how young"]):
        hits = []
        for chunk in chunks:
            sec = chunk.metadata.get("section", "")
            if sec not in ("eligibility", "definitions", "general"):
                continue
            vals = list(dict.fromkeys(_AGE_RE.findall(chunk.page_content)))
            if vals:
                src = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "?")
                for val in vals:
                    for sent in re.split(r'[.\n]', chunk.page_content):
                        if val.lower() in sent.lower() and len(sent.strip()) > 10:
                            hits.append(f"- **{val}** — _{sent.strip()}_ [Source: {src}, Page {page}]")
                            break
                    else:
                        hits.append(f"- **{val}** [Source: {src}, Page {page}]")
        if hits:
            return "**Age values found in policy documents:**\n\n" + "\n".join(hits[:8]) + "\n\n**Confidence:** High"
    # Emergency Medical
    if any(w in q for w in ["emergency medical", "medical coverage", "medical expenses", "medical and related", "medical benefit", "medical limit"]):
        tier = _detect_plan_tier(query)
        _TIER_COL = {"platinum": 0, "gold": 1, "silver": 2, "prime": 0, "enhanced": 1, "basic": 2}
        col_idx = _TIER_COL.get(tier) if tier else None
        _SKIP_KEYWORDS = ["personal accident", "accidental death", "permanent disability", "baggage", "luggage", "trip delay", "flight delay", "cancellation", "passport", "hijacking", "bail bond", "emergency family travel", "family travel", "emergency departure", "missed departure"]
        _MED_PRIORITY = ["medical expenses", "medical expense", "medical and related"]
        hits_primary = []
        hits_secondary = []
        for chunk in chunks:
            text = chunk.page_content
            src = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            for line in text.split("\n"):
                l_low = line.lower()
                if not any(kw in l_low for kw in ["medical", "hospital"]):
                    continue
                if any(skip in l_low for skip in _SKIP_KEYWORDS):
                    continue
                vals = _find_amounts(line)
                if not vals:
                    continue
                if col_idx is not None and len(vals) > col_idx:
                    chosen = vals[col_idx]
                    tier_label = ["Platinum", "Gold", "Silver"][col_idx]
                    entry = f"- **{chosen}** ({tier_label}) — _{line.strip()}_ [Source: {src}, Page {page}]"
                else:
                    entry = f"- **{vals[0]}** — _{line.strip()}_ [Source: {src}, Page {page}]"
                if any(kw in l_low for kw in _MED_PRIORITY):
                    hits_primary.append(entry)
                else:
                    hits_secondary.append(entry)
        hits = hits_primary + hits_secondary
        if hits:
            tier_note = f" ({tier.title()} plan)" if tier else ""
            return f"**Emergency Medical coverage limits{tier_note}:**\n\n" + "\n".join(list(dict.fromkeys(hits))[:6]) + "\n\n**Confidence:** High"
    # Monetary limits
    if any(w in q for w in ["how much", "maximum amount", "coverage amount", "benefit amount", "sum insured", "limit for", "payout", "compensation amount", "maximum coverage", "medical coverage", "emergency medical coverage", "maximum emergency", "maximum medical", "what is the maximum", "coverage limit", "coverage under", "covered under"]):
        tier = _detect_plan_tier(query)
        hits_tier = []
        hits_all = []
        for chunk in chunks:
            sec = chunk.metadata.get("section", "")
            if sec not in ("benefits", "eligibility", "medical", "general", "flight_delay", "baggage"):
                continue
            text = chunk.page_content
            vals = _find_amounts(text)
            if not vals:
                continue
            src = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            for val in vals[:6]:
                for sent in re.split(r'[.\n]', text):
                    s_norm = sent.replace(",", "").lower()
                    v_norm = val.replace(",", "")
                    if v_norm.lower() in s_norm and len(sent.strip()) > 10:
                        entry = f"- **{val}** — _{sent.strip()}_ [Source: {src}, Page {page}]"
                        if tier and tier in sent.lower():
                            hits_tier.append(entry)
                        else:
                            hits_all.append(entry)
                        break
                else:
                    hits_all.append(f"- **{val}** [Source: {src}, Page {page}]")
        hits = hits_tier + hits_all
        if hits:
            tier_note = f" ({tier.title()} plan)" if tier else ""
            return f"**Coverage amounts / limits found in policy documents{tier_note}:**\n\n" + "\n".join(hits[:8]) + "\n\n**Confidence:** High"
    # Flight delay
    if any(w in q for w in ["flight delay", "trip delay", "delay benefit", "hours delay", "departure delay", "delayed"]):
        hits = []
        for chunk in chunks:
            text = chunk.page_content
            t_low = text.lower()
            if not any(kw in t_low for kw in ["delay", "departure", "hour"]):
                continue
            src = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            hour_vals = re.findall(r'\b(\d+)\s*(?:consecutive\s+)?hours?\b', text, re.IGNORECASE)
            money_vals = _find_amounts(text)
            for sent in re.split(r'[.\n]', text):
                s_low = sent.lower()
                has_hour = any(str(h) in sent for h in hour_vals)
                has_money = any(m.replace(",", "") in sent.replace(",", "") for m in money_vals)
                if (has_hour or has_money) and len(sent.strip()) > 15:
                    hits.append(f"- _{sent.strip()}_ [Source: {src}, Page {page}]")
        if hits:
            return "**Flight/Trip delay terms found in policy:**\n\n" + "\n".join(list(dict.fromkeys(hits))[:6]) + "\n\n**Confidence:** High"
    # Hajj/Umrah
    if any(w in q for w in ["hajj", "umrah", "pilgrimage"]):
        hits = []
        for chunk in chunks:
            text = chunk.page_content
            money_vals = _find_amounts(text)
            if not money_vals:
                continue
            src = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            for sent in re.split(r'[.\n]', text):
                if any(m.replace(",", "") in sent.replace(",", "") for m in money_vals) and len(sent.strip()) > 15:
                    hits.append(f"- _{sent.strip()}_ [Source: {src}, Page {page}]")
        if hits:
            return "**Coverage limits found for Hajj/Umrah:**\n\n" + "\n".join(list(dict.fromkeys(hits))[:8]) + "\n\n**Confidence:** High"
    # Duration
    if any(w in q for w in ["how long", "maximum trip", "trip duration", "days allowed", "trip length", "maximum days", "consecutive days"]):
        hits = []
        for chunk in chunks:
            vals = list(dict.fromkeys(_DURATION_RE.findall(chunk.page_content)))
            if vals:
                src = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "?")
                for val in vals[:4]:
                    for sent in re.split(r'[.\n]', chunk.page_content):
                        if val.lower() in sent.lower() and len(sent.strip()) > 10:
                            hits.append(f"- **{val}** — _{sent.strip()}_ [Source: {src}, Page {page}]")
                            break
                    else:
                        hits.append(f"- **{val}** [Source: {src}, Page {page}]")
        if hits:
            return "**Duration values found in policy documents:**\n\n" + "\n".join(hits[:8]) + "\n\n**Confidence:** High"
    return None

# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD EXTRACTION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are", "be",
    "for", "on", "at", "by", "with", "from", "this", "that", "which",
    "as", "it", "its", "not", "but", "if", "when", "where", "who",
    "will", "shall", "may", "can", "under", "above", "below", "per",
    "any", "all", "each", "such", "no", "yes", "has", "have", "had",
    "been", "was", "were", "does", "did", "do", "been", "being",
})

def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b', text.lower())
    return list(dict.fromkeys(t for t in tokens if t not in _STOPWORDS))[:40]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION-AWARE CHUNKER (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class SectionChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
        )
    def split_documents(self, docs: list[Document]) -> list[Document]:
        chunks = []
        for doc in docs:
            raw = self._splitter.split_documents([doc])
            for chunk in raw:
                chunk.metadata["section"] = _detect_section(chunk.page_content)
                chunk.metadata["keywords"] = _extract_keywords(chunk.page_content)
            chunks.extend(raw)
        return chunks

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def _build_structured_context(chunks: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    for chunk in chunks:
        section = chunk.metadata.get("section", "general").title()
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Section: {section} | Source: {source} | Page: {page}]\n{chunk.page_content}")
    full = "\n\n".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars] + "... (truncated)"
    return full

def _sources_from_chunks(chunks: list[Document]) -> list[str]:
    seen, result = set(), []
    for doc in chunks:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        key = f"{src}_{page}"
        if key not in seen:
            seen.add(key)
            result.append(f"{src} (page {page})" if page and page != "?" else src)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# QUERY CLASSIFICATION HELPERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_ALL_DOCS_EXPLICIT = [
    "from all documents", "from all files", "from all resumes",
    "across all documents", "across all files", "across all resumes",
    "all documents", "all resumes", "all files",
    "from each document", "from each file", "from each resume",
    "each document", "each resume", "each file",
    "every document", "every resume", "every file",
    "extract from all", "extract all",
    "list all candidates", "list all resumes", "list all documents",
    "summary of all", "compare all",
]
_FIELD_MAP = [
    (["name", "candidate", "person", "insured", "policyholder", "holder"], "name"),
    (["email", "mail", "e-mail"], "email"),
    (["phone", "contact", "mobile"], "phone_number"),
    (["experience", "exp", "year"], "experience"),
    (["skill", "technology", "tech stack"], "skills"),
    (["education", "degree", "qualification"], "education"),
    (["company", "employer", "organisation", "organization", "worked at"], "current_company"),
    (["role", "designation", "position", "title", "job"], "designation"),
    (["policy number", "policy no", "policy id", "policy"], "policy_number"),
    (["covered", "coverage", "what is covered", "benefits", "benefit"], "coverage"),
    (["premium", "amount", "premium amount"], "premium"),
    (["sum insured", "sum assured", "coverage amount"], "sum_insured"),
    (["policy type", "plan type", "plan name"], "policy_type"),
    (["insurer", "insurance company", "provider"], "insurer"),
    (["expiry", "expiry date", "valid till", "end date"], "expiry_date"),
    (["start date", "commencement", "issue date", "inception"], "start_date"),
    (["nominee", "beneficiary"], "nominee"),
    (["exclusion", "not covered", "excluded"], "exclusions"),
    (["claim", "claim process", "claim procedure"], "claim_process"),
]
_COMPARISON_PHRASES = ["compare", "comparison", "vs", "versus", "difference between", "which is better", "which offers", "which insurer", "which policy", "all insurers", "all policies", "both", "each insurer", "across policies", "across insurers", "between"]
_PERSONAL_QUERY_WORDS = ["my flight", "my baggage", "my claim", "my policy", "my trip", "my luggage", "my travel", "my delay", " my ", "i was", "i am ", "i have", "i need", "i got", "i lost", "i missed", "i paid"]
_INFORMATIONAL_PHRASES = ["what is", "what are", "what does", "what do", "what's", "how much is", "how much does", "how much do", "how much can", "describe", "explain", "tell me about", "what coverage", "what benefit", "what limit", "what excess", "what deductible", "does it cover", "is there coverage", "is there a benefit", "list the", "show me the", "what type", "what kind", "under rak", "under aig", "under gig", "under liva"]
_SCENARIO_WORDS = ["hours delayed", "hour delay", "days delayed", "day delay", "missed my", "missed the", "lost my", "stolen", "trip cost", "total cost", "paid for", "booked", "i was delayed", "my flight was", "my baggage was", "calculate", "how much will", "how much would", "how much should", "how much can i", "how much do i"]

def _is_all_docs_query(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in _ALL_DOCS_EXPLICIT)

def _is_comparison_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _COMPARISON_PHRASES)

def _is_personal_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _PERSONAL_QUERY_WORDS)

def _is_informational_query(question: str) -> bool:
    if _is_personal_query(question):
        return False
    q = question.lower()
    return any(p in q for p in _INFORMATIONAL_PHRASES)

def _is_scenario_query(question: str) -> bool:
    q = question.lower()
    return _is_personal_query(question) or any(w in q for w in _SCENARIO_WORDS)

def _fields_from_question(question: str) -> list[str]:
    q = question.lower().replace("_", " ")
    fields = []
    for keywords, field_name in _FIELD_MAP:
        if any(kw in q for kw in keywords):
            if field_name not in fields:
                fields.append(field_name)
    return fields or ["name", "policy_number", "coverage", "premium"]

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH HELPERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def wait_for_vllm(retries: int = 20, delay: int = 3) -> bool:
    for _ in range(retries):
        try:
            r = requests.get(f"{VLLM_HOST}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def list_vllm_models() -> list[str]:
    try:
        r = requests.get(f"{VLLM_HOST}/v1/models", timeout=5)
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE — ChromaDB Vector Edition with HyDE, Hybrid Search, Citation
# ══════════════════════════════════════════════════════════════════════════════
class RAGPipeline:
    def __init__(self):
        self._vector_store = ChromaVectorStore()
        self._chunker = SectionChunker(chunk_size=600, chunk_overlap=80)

    # ── Document ingestion ─────────────────────────────────────────────────
    def add_document(self, uploaded_file) -> int:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            raw_docs = load_document(tmp_path, uploaded_file.name)
            chunks = self._chunker.split_documents(raw_docs)
            self._vector_store.delete_by_source(uploaded_file.name)
            preview = raw_docs[0].page_content[:600] if raw_docs else ""
            doc_tags = tag_document(uploaded_file.name, preview)
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
                chunk.metadata.update(doc_tags)
            self._vector_store.add_documents(chunks)
            return len(chunks)
        finally:
            os.unlink(tmp_path)

    def add_url(self, url: str) -> int:
        docs = load_url(url)
        chunks = self._chunker.split_documents(docs)
        self._vector_store.delete_by_source(url)
        for chunk in chunks:
            chunk.metadata["source"] = url
        self._vector_store.add_documents(chunks)
        return len(chunks)

    # ── Document management ─────────────────────────────────────────────────
    def list_documents(self) -> list[str]:
        return self._vector_store.list_sources()

    def clear_documents(self) -> None:
        self._vector_store.delete_all()

    def remove_document(self, doc_name: str) -> None:
        self._vector_store.delete_by_source(doc_name)

    def get_document_tags(self, doc_name: str) -> dict:
        results = self._vector_store.collection.get(where={"source": doc_name}, limit=1, include=["metadatas"])
        if results["metadatas"]:
            meta = results["metadatas"][0]
            return {"insurer": meta.get("insurer", "UNKNOWN"), "policy_type": meta.get("policy_type", "general")}
        return {"insurer": "UNKNOWN", "policy_type": "general"}

    def get_full_content(self, source: str) -> str:
        results = self._vector_store.collection.get(where={"source": source}, include=["documents"])
        return "\n\n".join(results["documents"])

    def summarize_url(self, url: str) -> tuple[str, list[str]]:
        full_text = self.get_full_content(url)
        if not full_text.strip():
            return "No content found for this URL.", []
        if len(full_text) > SUMMARY_MAX_CHARS:
            full_text = full_text[:SUMMARY_MAX_CHARS] + "... (truncated)"
        from prompt_template import URL_SUMMARY_PROMPT
        try:
            prompt = URL_SUMMARY_PROMPT.format(context=full_text, question="Summarize this content.")
        except Exception:
            prompt = f"Please provide a comprehensive summary of the following web page content.\nInclude all key points, names, numbers, dates, and important details.\n\nContent:\n{full_text}\n\nDetailed Summary:"
        llm = get_insurance_llm(temperature=0.3)
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, [url]

    # ── Query entry point (backward compat) ────────────────────────────────
    def query(self, question: str, model: str, allowed_docs: Optional[list[str]] = None) -> tuple[str, list[str], Optional[pd.DataFrame]]:
        if _is_all_docs_query(question) and allowed_docs:
            return self._extract_all_docs(question, model, allowed_docs)
        answer, sources = self._rag_query(question, model, allowed_docs=allowed_docs)
        return answer, sources, None

    # ── Main knowledge-base Q&A pipeline (with HyDE and citation) ──────────
    def _expand_query(self, question: str) -> list[str]:
        """Generate query variations using HyDE."""
        hyde_prompt = ChatPromptTemplate.from_template(
            "Write a detailed hypothetical answer to the following question. "
            "Use insurance policy language. Do NOT use any real facts, just plausible text.\n\nQuestion: {question}\n\nHypothetical answer:"
        )
        llm = get_insurance_llm(temperature=0.5)
        chain = hyde_prompt | llm | StrOutputParser()
        try:
            hypo = chain.invoke({"question": question})
            return [question, hypo[:500]]
        except Exception:
            return [question]

    def knowledge_query(self, question: str) -> tuple[str, bool, list[str]]:
        if self._vector_store.count() == 0:
            return "EMPTY_KB", True, []

        # ── URL handling (improved) ────────────────────────────────────────
        urls = extract_urls(question)
        if urls:
            q_lower = question.lower()
            if any(p in q_lower for p in ["full text", "raw text"]):
                full_text = self.get_full_content(urls[0])
                return full_text, False, [urls[0]]
            # For summary, use the improved loader
            from document_loader import load_url_advanced
            docs = load_url_advanced(urls[0])
            if docs:
                context = docs[0].page_content
                answer = self._summarize_with_citations(context, question)
                return answer, False, [urls[0]]

        # ── Query expansion (HyDE) ─────────────────────────────────────────
        expanded_queries = self._expand_query(question)
        all_chunks = []
        for q in expanded_queries:
            chunks = self._vector_store.search(q, top_k=6, use_hybrid=True, use_reranker=True)
            all_chunks.extend(chunks)
        # deduplicate by content hash
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            h = hash(c.page_content[:200])
            if h not in seen:
                seen.add(h)
                unique_chunks.append(c)
        chunks = unique_chunks[:12]

        if not chunks:
            return "Not mentioned in documents.", False, []

        sources = _sources_from_chunks(chunks)

        # ── Build context with forced citations ────────────────────────────
        context = _build_structured_context(chunks, max_chars=MAX_CONTEXT_CHARS)

        citation_prompt = f"""You are an Insurance Policy Analyst. Answer based ONLY on the CONTEXT below.

RULES (strictly enforced):
1. For every fact, number, condition, or limit, you MUST cite the exact source and page number like this: [Source: document_name, Page X].
2. If a piece of information is not present in the context, say "Not mentioned in documents."
3. Do not invent any information. If you are unsure, say "Not mentioned in documents."
4. Format your answer using markdown: headings, bullet points, bold for key numbers.
5. If the question asks for a calculation, show step‑by‑step using only numbers from context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""

        llm = get_insurance_llm(temperature=0)
        response = llm.invoke(citation_prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Post‑processing: warn if no citations found
        if "[Source:" not in answer and "Not mentioned" not in answer:
            answer += "\n\n⚠️ **Warning:** The above answer could not be verified with explicit citations. Please verify against the original documents."

        # Grounding validation
        grounded, missing = validate_grounding(answer, context)
        if not grounded and missing:
            missing_values = ", ".join(str(m) for m in missing)
            answer += f"\n\n⚠️ Warning: These figures could not be verified in the source documents: {missing_values}. Please cross-check against the original policy document."

        return answer, False, sources

    def _summarize_with_citations(self, content: str, question: str) -> str:
        prompt = f"""Summarize the following web page content in a detailed, structured way (like Perplexity). Use headings, bullet points, and include all important facts (dates, numbers, names). Do not add external knowledge.

Content:
{content[:6000]}

Question: {question}

Detailed summary:"""
        llm = get_insurance_llm(temperature=0.3)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    # ── URL / general queries ──────────────────────────────────────────────
    def general_query(self, question: str) -> str:
        llm = get_general_llm(temperature=0.7)
        response = llm.invoke(GENERAL_PROMPT.format(question=question))
        return response.content if hasattr(response, "content") else str(response)

    def _rag_query(self, question: str, model: str, allowed_docs: Optional[list[str]] = None) -> tuple[str, list[str]]:
        llm = get_insurance_llm(temperature=0)
        filter_meta = None
        if allowed_docs:
            if len(allowed_docs) == 1:
                filter_meta = {"source": allowed_docs[0]}
            else:
                filter_meta = {"source": {"$in": allowed_docs}}
        chunks = self._vector_store.search(question, top_k=5, filter_metadata=filter_meta)
        context = _build_structured_context(chunks, max_chars=MAX_CONTEXT_CHARS)
        sources = _sources_from_chunks(chunks)
        response = llm.invoke(RAG_PROMPT.format(context=context, question=question))
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, sources

    # ── Bulk structured extraction ─────────────────────────────────────────
    def _extract_all_docs(self, question: str, model: str, doc_names: list[str]) -> tuple[str, list[str], Optional[pd.DataFrame]]:
        llm = get_insurance_llm(temperature=0)
        fields = _fields_from_question(question)
        _FIELD_HINTS = {
            "name": "extract the full name of the insured person or policyholder.",
            "policy_number": 'look for "Policy No", "Policy Number", "Policy ID".',
            "coverage": 'look for "Sum Insured", "Coverage", "Benefits", "What is Covered".',
            "sum_insured": 'look for "Sum Insured", "Sum Assured", "Coverage Amount".',
            "insurer": "look for the insurance company name.",
            "policy_type": 'look for "Plan Name", "Policy Type", "Product Name".',
            "expiry_date": 'look for "Valid Till", "Expiry Date", "Policy End Date".',
            "start_date": 'look for "Inception Date", "Commencement Date".',
            "premium": 'look for "Premium Amount", "Annual Premium".',
            "nominee": 'look for "Nominee Name", "Beneficiary".',
            "exclusions": 'look for "Exclusions", "Not Covered".',
            "experience": "look for total years of work experience.",
            "skills": "look for technical skills, tools, programming languages.",
            "education": "look for degree, university, graduation year.",
            "current_company": "look for the most recent employer.",
            "designation": "look for current job title or most recent role.",
        }
        rows = []
        for doc_name in doc_names:
            results = self._vector_store.collection.get(where={"source": doc_name}, limit=50, include=["documents"])
            raw_chunks = results["documents"]
            context = "\n\n".join(raw_chunks)[:6000]
            hints = "\n".join(f"- For {f}: {_FIELD_HINTS[f]}" for f in fields if f in _FIELD_HINTS)
            fields_str = ", ".join(f'"{f}"' for f in fields)
            prompt = f"Extract data from this document. Reply with ONLY a single JSON object using these EXACT keys: {fields_str}\nRules:\n- Use null if a field is not found.\n{hints}\n- One value per field (string). If multiple, join with \", \".\n- No explanation. No extra keys. Just the JSON.\n\nDocument ({doc_name}):\n{context}\n\nJSON:"
            raw = llm.invoke(prompt)
            parsed = self._parse_json(raw.content if hasattr(raw, "content") else str(raw))
            for f in fields:
                parsed.setdefault(f, None)
            parsed = {f: parsed.get(f) for f in fields}
            parsed["file"] = doc_name
            rows.append(parsed)
        if not rows:
            return "No data extracted.", doc_names, None
        df = pd.DataFrame(rows)
        cols = ["file"] + [c for c in df.columns if c != "file"]
        df = df[cols]
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        return f"Extracted from {len(rows)} document(s). Download Excel above.", doc_names, df

    def _find_doc_by_name_in_query(self, question: str, allowed_docs: Optional[list[str]] = None) -> Optional[str]:
        q_words = set(question.lower().split())
        doc_names = allowed_docs if allowed_docs else self.list_documents()
        best_doc, best_score = None, 0
        for doc_name in doc_names:
            stem = re.sub(r'[_\-.]', ' ', doc_name)
            stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
            doc_words = set(w.lower() for w in stem.split() if w.lower() not in {"resume", "cv", "pdf", "updated", "doc"})
            matches = len(q_words & doc_words)
            if matches > best_score:
                best_score = matches
                best_doc = doc_name
        return best_doc if best_score >= 2 else None

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return {}