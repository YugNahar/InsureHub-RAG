"""
PDF -> Markdown conversion, preserving real section-heading structure.

MarkdownNodeParser (llama_index.core.node_parser) only accepts Markdown
text with real '#'/'##'/... header lines — it cannot read a PDF directly.
This module is the missing conversion step: it uses pymupdf4llm, which
derives heading LEVELS from the PDF's own font-size/bold metadata rather
than guessing from already-flattened plain text, so headings come from
the document's actual visual structure.

This is a genuinely different signal than this project's existing
pypdf/pdfplumber extraction path in document_loader.py — that path pulls
plain text only, with no heading information at all, which is the root
cause of the heading-carryover bug this conversion was built to fix (see
the section-aware SRG matching and section_llm_retag work from this same
session).

2026-09-01: wired into document_loader.py's _load_pdf() as the FIRST
attempt for every PDF load, not just an opt-in path anymore — see
load_pdf_pages_as_markdown() below, which returns the SAME
{page_num: text} shape _load_pdf() already builds from pypdf/pdfplumber,
so every existing per-page repair step (rupee-symbol repair, list-marker
corruption detection, repeating-furniture stripping) keeps working
unchanged on markdown text instead of plain text. _load_pdf() falls back
to the original pypdf/pdfplumber path automatically if this raises for
any reason (missing dependency, a genuinely unreadable file, or any
other pymupdf4llm failure) — this conversion is strictly additive, never
a hard requirement for ingestion to work at all.
"""
import logging
import re

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF file to Markdown text with real heading structure.

    Pages with no extractable text layer (scanned/image pages) fall back
    to OCR automatically — pymupdf4llm handles this internally, no
    special-casing needed here. Confirmed live on a 275-page mixed
    text/scanned document: 10 of 275 pages needed the OCR fallback and
    conversion still completed correctly.

    Raises whatever pymupdf4llm raises on a genuinely unreadable/corrupt
    file — callers should treat this the same as any other document-load
    failure, not silently swallow it here.
    """
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(pdf_path)
    logger.info(
        "[pdf_to_markdown] converted %s -> %d chars, %d header lines",
        pdf_path, len(md), len(re.findall(r"^#{1,6}\s+.+$", md, re.MULTILINE)),
    )
    return md


def load_pdf_pages_as_markdown(pdf_path: str) -> dict[int, str]:
    """
    Convert a PDF to markdown PER PAGE, returning {0-indexed page_num: text}
    — the same shape _load_pdf()'s pypdf path already builds, so it can
    be dropped straight into the existing per-page repair pipeline
    (_strip_repeating_page_furniture, _repair_rupee_symbol_corruption,
    _detect_list_marker_corruption) with no changes to those functions.

    Uses pymupdf4llm's own page_chunks=True mode rather than splitting
    the combined-document convert_pdf_to_markdown() output ourselves —
    page boundaries are already resolved correctly by the library (each
    chunk carries its own metadata["page_number"], 1-indexed), so there's
    no need to re-detect them from page-break markers in plain text.

    Raises whatever pymupdf4llm raises — same contract as
    convert_pdf_to_markdown(), callers handle fallback.
    """
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages: dict[int, str] = {}
    for chunk in chunks:
        page_num_1indexed = chunk.get("metadata", {}).get("page_number")
        text = (chunk.get("text") or "").strip()
        if page_num_1indexed and text:
            pages[page_num_1indexed - 1] = text
    logger.info(
        "[pdf_to_markdown] converted %s -> %d/%d page(s) with text",
        pdf_path, len(pages), len(chunks),
    )
    return pages
