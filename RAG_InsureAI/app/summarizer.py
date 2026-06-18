"""
LLM-based document summarizer.

Generates a 200-300 word structured summary for any ingested source
(document, YouTube transcript, or webpage) that captures:
  - Insurer name and product type
  - Key coverages and exclusions
  - Notable limits, amounts, and claim procedures
  - Source type for easy identification

Falls back to a metadata-driven template when the LLM is unavailable.
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are an insurance document analyst. Read the following content excerpts and write a \
concise 200-300 word summary suitable for a retrieval index.

The summary MUST cover (where information is available):
1. Document type (policy document / handbook / regulatory guide / video / webpage)
2. Insurer name
3. Primary insurance product(s): motor / health / life / travel / home / general
4. Key coverages and benefits
5. Important exclusions or conditions
6. Notable limits, amounts, or claim procedures

Content:
{content}

Write only the summary text. No headings, no preamble, no trailing notes."""

_YOUTUBE_SUMMARY_PROMPT = """\
You are an insurance content analyst. The text below is a transcript from a YouTube video \
or audio recording about insurance. It may be informal, conversational, and lack punctuation.

Summarise the video in 200-300 words for a retrieval index. Cover:
1. Type: "YouTube video" or "audio recording"
2. Insurer or brand mentioned (if any)
3. Main insurance topics discussed (health / motor / life / travel / home / general etc.)
4. Key facts, figures, rules, or tips explained in the video
5. Specific products, plans, or claim procedures mentioned

Transcript excerpt:
{content}

Write only the summary text. No headings, no preamble."""


def generate_summary(
    docs: List[Document],
    source: str,
    doc_meta: Dict[str, Any],
    llm: Any,
    max_input_chars: int = 8000,
) -> str:
    """
    Build a summary string from the first *max_input_chars* of *docs*.

    Falls back to a template-based summary when *llm* is None or raises.
    YouTube/Whisper content uses a transcript-specific prompt.
    """
    doc_type = doc_meta.get("doc_type", "document")
    first_meta = docs[0].metadata if docs else {}
    source_type = first_meta.get("source_type", "")
    is_youtube = (
        doc_type == "youtube"
        or "youtube" in source_type.lower()
        or "whisper" in source_type.lower()
    )

    prompt_template = _YOUTUBE_SUMMARY_PROMPT if is_youtube else _SUMMARY_PROMPT

    # Sample content spread across document pages/chunks
    parts: List[str] = []
    budget = max_input_chars
    for doc in docs:
        snippet = doc.page_content[:budget]
        parts.append(snippet)
        budget -= len(snippet)
        if budget <= 0:
            break
    content = "\n\n---\n\n".join(parts)

    if llm is not None:
        try:
            resp = llm.invoke(prompt_template.format(content=content))
            text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            if len(text) > 60:
                return text
        except Exception as exc:
            logger.warning(
                "[Summarizer] LLM summary failed for %s: %s — using template fallback",
                source, exc,
            )

    # Template fallback (no LLM or LLM produced empty output)
    insurer     = doc_meta.get("insurer", "UNKNOWN")
    policy_type = doc_meta.get("policy_type", "general")
    n_docs      = len(docs)
    type_label  = "YouTube video transcript" if is_youtube else doc_type
    return (
        f"Source: {source}. "
        f"Type: {type_label}. "
        f"Insurer: {insurer}. "
        f"Policy coverage: {policy_type}. "
        f"Contains {n_docs} section(s). "
        "Detailed content is available in the vector store for chunk-level retrieval."
    )
