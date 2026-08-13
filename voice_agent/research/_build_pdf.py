"""One-off markdown -> PDF renderer for the payment-troubleshooting doc.
Handles headings, tables, code blocks, bullet lists, and links well enough
for this specific document — not a general-purpose converter."""
import re
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_LEFT

SRC = "/Users/lavishdevoura/Downloads/InsureHub-RAG-main/voice_agent/research/payment_completion_troubleshooting.md"
OUT = "/Users/lavishdevoura/Downloads/InsureHub-RAG-main/voice_agent/research/payment_completion_troubleshooting.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=14, spaceBefore=6, textColor=colors.HexColor("#1a1a2e")))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceAfter=10, spaceBefore=16, textColor=colors.HexColor("#16213e")))
styles.add(ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#0f3460")))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8, alignment=TA_LEFT))
styles.add(ParagraphStyle("ScriptBlock", parent=styles["Code"], fontSize=8.5, leading=11, backColor=colors.HexColor("#f4f4f8"), borderPadding=8, borderColor=colors.HexColor("#d0d0e0"), borderWidth=0.5))
styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#555555"), spaceAfter=6))

def inline(text):
    # bold **x**, links [t](u), inline code `x`
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#1a5fb4">\1</link>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="9">\1</font>', text)
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    return text

def parse_table(lines, i):
    rows = []
    while i < len(lines) and lines[i].strip().startswith('|'):
        row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
        if not re.match(r'^:?-+:?$', row[0]):
            rows.append(row)
        i += 1
    return rows, i

def build():
    with open(SRC) as f:
        lines = f.read().split('\n')

    story = []
    i = 0
    in_code = False
    code_buf = []
    list_buf = []

    def flush_list():
        nonlocal list_buf
        if list_buf:
            items = [ListItem(Paragraph(inline(t), styles["Body"]), leftIndent=14) for t in list_buf]
            story.append(ListFlowable(items, bulletType='bullet', start='•', leftIndent=18))
            story.append(Spacer(1, 6))
            list_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('```'):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                flush_list()
                story.append(Preformatted('\n'.join(code_buf), styles["ScriptBlock"]))
                story.append(Spacer(1, 8))
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not stripped:
            flush_list()
            i += 1
            continue

        if stripped.startswith('# '):
            flush_list()
            story.append(Paragraph(inline(stripped[2:]), styles["H1"]))
        elif stripped.startswith('## '):
            flush_list()
            story.append(Paragraph(inline(stripped[3:]), styles["H2"]))
        elif stripped.startswith('### '):
            flush_list()
            story.append(Paragraph(inline(stripped[4:]), styles["H3"]))
        elif stripped.startswith('---'):
            flush_list()
            story.append(Spacer(1, 4))
        elif stripped.startswith('|'):
            flush_list()
            rows, i = parse_table(lines, i)
            if rows:
                tbl_data = [[Paragraph(inline(c), styles["Body"]) for c in r] for r in rows]
                col_count = len(rows[0])
                avail_w = 6.3 * inch
                t = Table(tbl_data, colWidths=[avail_w / col_count] * col_count, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#c0c0d0")),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
                story.append(Spacer(1, 10))
            continue
        elif stripped.startswith('- '):
            list_buf.append(stripped[2:])
        elif re.match(r'^\*\*[A-Za-z ,]+:\*\*', stripped):
            flush_list()
            story.append(Paragraph(inline(stripped), styles["Meta"]))
        else:
            flush_list()
            story.append(Paragraph(inline(stripped), styles["Body"]))
        i += 1

    flush_list()

    doc = SimpleDocTemplate(
        OUT, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        title="Payment-Completion Troubleshooting — Voice Agent Knowledge Base Addition",
    )
    doc.build(story)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    build()
