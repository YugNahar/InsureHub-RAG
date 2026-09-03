"""
Reconstruct '9.3fixed2_9.3 INSURANCE LAW AND PRACTICE.pdf' as a real PDF
from its own already-ingested chunk text — no source file exists for this
document (confirmed: never persisted, not found anywhere on disk).

Renders headings with real bold/larger font (not markdown syntax) so the
new pymupdf4llm-based ingestion path (document_loader.py's _load_pdf)
can correctly re-derive heading structure from genuine font metadata when
this reconstructed file is re-uploaded — same reconstruct-and-reupload
approach already used once this session for the bullet-list chunking fix,
adapted here to also preserve BOLD numbered sub-item markers (e.g.
"**1) Term insurance**") as real bold text, since that's exactly the
signal the new _SUBHEADING_RE detector in semantic_chunker.py looks for.
"""
import json
import re
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

META_PATH = "/app/app/turbovec_data/documents/insurance_docs_meta.ndjson"
SOURCE = "9.3fixed2_9.3 INSURANCE LAW AND PRACTICE.pdf"
OUT_PDF = "/tmp/reconstructed_93fixed2.pdf"

_PICTURE_BLOCK_RE = re.compile(
    r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
    re.DOTALL,
)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*([^*\n]{1,120}?)\*\*")


def clean_body_text(text: str) -> str:
    """Strip picture-text blocks (garbled diagram-derived text, not real
    prose — confirmed the real prose elsewhere already covers the same
    info in proper sentences), normalize <br> tags to plain breaks, then
    XML-escape for reportlab's markup parser, THEN re-apply bold markup
    for any **...** spans found in the ORIGINAL text (escaping first,
    then adding our own controlled <b> tags, so a stray '<' or '&' in
    the source text can never be mistaken for one of our own tags).
    """
    text = _PICTURE_BLOCK_RE.sub(" ", text)
    text = _BR_RE.sub(" ", text)

    # Record bold spans (by their inner text) before escaping, so we can
    # re-find and wrap them after — the ** markers themselves are not
    # XML-unsafe, but doing this in one pass avoids any escaping/order bugs.
    bold_spans = set(m.group(1).strip() for m in _BOLD_RE.finditer(text))
    text = _BOLD_RE.sub(lambda m: m.group(1), text)  # drop ** markers, keep inner text

    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    for span in sorted(bold_spans, key=len, reverse=True):
        if not span:
            continue
        escaped_span = (
            span.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        if escaped_span in text:
            text = text.replace(escaped_span, f"<b>{escaped_span}</b>", 1)

    return text.strip()


def clean_heading_text(heading: str) -> str:
    h = _BOLD_RE.sub(lambda m: m.group(1), heading)
    h = h.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return h.strip()


def load_chunks():
    chunks = []
    with open(META_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("metadata", {}) or {}
            if meta.get("source") == SOURCE:
                chunks.append(d)
    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return chunks


def group_by_heading(chunks):
    """Merge consecutive chunks sharing the same section_heading into one
    (heading, body_text) pair, stripping each chunk's duplicated
    heading-as-first-line before joining.
    """
    groups = []
    cur_heading = None
    cur_parts = []
    for c in chunks:
        heading = c["metadata"].get("section_heading", "") or ""
        text = c.get("text", "") or ""
        if heading and text.startswith(heading):
            text = text[len(heading):].lstrip("\n").lstrip()
        if heading != cur_heading:
            if cur_parts:
                groups.append((cur_heading, "\n\n".join(cur_parts)))
            cur_heading = heading
            cur_parts = [text] if text else []
        else:
            if text:
                cur_parts.append(text)
    if cur_parts:
        groups.append((cur_heading, "\n\n".join(cur_parts)))
    return groups


def build_pdf(groups, out_path):
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontSize=14, spaceBefore=14, spaceAfter=8, alignment=TA_LEFT,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    story = []
    for heading, body in groups:
        if heading:
            story.append(Paragraph(clean_heading_text(heading), heading_style))
        for para in body.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            cleaned = clean_body_text(para)
            if cleaned:
                story.append(Paragraph(cleaned, body_style))
        story.append(Spacer(1, 4))

    doc.build(story)


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks for '{SOURCE}'")
    groups = group_by_heading(chunks)
    print(f"Grouped into {len(groups)} sections")
    build_pdf(groups, OUT_PDF)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
