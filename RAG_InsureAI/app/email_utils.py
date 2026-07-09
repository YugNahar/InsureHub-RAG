"""
Email + PDF utilities for offline escalation.
When no agent is online and the AI can't answer, this module:
  1. Generates a PDF transcript with the unanswerable query highlighted
  2. Uses the LLM to compose a professional email body
  3. Sends the email + PDF to every agent listed in the agent-emails database
"""
import logging
import os
import re
import smtplib
from datetime import datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

logger = logging.getLogger(__name__)

GMAIL_SENDER    = os.getenv("GMAIL_SENDER", "")
GMAIL_PASSWORD  = os.getenv("GMAIL_APP_PASSWORD", "")
# Legacy single-recipient fallback — only used if the agent-emails database
# file itself doesn't exist yet (first-time setup). Once the file exists, it
# is the sole source of truth, even if that means it's empty.
AGENT_EMAIL     = os.getenv("AGENT_EMAIL", "lavishdevpura6@gmail.com")
VLLM_HOST       = os.getenv("VLLM_HOST", "")
VLLM_MODEL      = os.getenv("VLLM_MODEL", "")

# Excel "database" of agent recipient emails — one email per row under an
# "email" header in the first sheet. Edit this file (add/remove rows) to
# change who gets escalation emails; no restart or code change needed, since
# it's re-read from disk on every send rather than cached in memory.
AGENT_EMAILS_FILE = os.getenv(
    "AGENT_EMAILS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_emails.xlsx"),
)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_agent_emails() -> List[str]:
    """Read the current list of agent recipient emails from AGENT_EMAILS_FILE.

    Re-reads the file from disk on every call (no caching) so edits — adding
    or removing a row — take effect on the very next email sent, without
    needing a restart. Returns [] if the file is missing, unreadable, or has
    no valid email rows; callers decide how to handle an empty list.
    """
    if not os.path.exists(AGENT_EMAILS_FILE):
        return []
    try:
        import openpyxl
        wb = openpyxl.load_workbook(AGENT_EMAILS_FILE, read_only=True, data_only=True)
        ws = wb.active
        emails: List[str] = []
        seen = set()
        for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
            cell = row[0]
            if cell is None:
                continue
            email = str(cell).strip()
            if not email or not _EMAIL_RE.match(email):
                continue
            key = email.lower()
            if key in seen:
                continue
            seen.add(key)
            emails.append(email)
        return emails
    except Exception:
        logger.exception("Failed to read agent-emails database at %s", AGENT_EMAILS_FILE)
        return []


def _safe(text: str) -> str:
    """Strip characters outside latin-1 so fpdf2's built-in Helvetica never errors."""
    return (
        str(text)
        .replace("—", "-").replace("–", "-")   # em/en dash
        .replace("‘", "'").replace("’", "'")   # curly single quotes
        .replace("“", '"').replace("”", '"')   # curly double quotes
        .replace("…", "...")                         # ellipsis
        .replace("â", "'")              # mangled UTF-8
        .encode("latin-1", errors="replace").decode("latin-1")
    )


# ── PDF ───────────────────────────────────────────────────────────────────────

def generate_pdf(session_id: str, history, unanswerable_query: str) -> bytes:
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 13)
            self.set_fill_color(79, 70, 229)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, "  InsureHub - Unresolved Support Request", ln=True, fill=True)
            self.set_text_color(0, 0, 0)
            self.ln(2)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Meta
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(100, 100, 100)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 6, _safe(f"Session: #{session_id}   |   Generated: {ts}"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Highlighted unanswerable query box
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(254, 243, 199)
    pdf.set_draw_color(217, 119, 6)
    pdf.set_line_width(0.5)
    pdf.cell(0, 7, "  QUERY THAT REQUIRES AGENT ATTENTION", ln=True, fill=True, border=1)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_fill_color(255, 251, 235)
    pdf.multi_cell(0, 7, _safe(f'  "{unanswerable_query}"'), fill=True, border="LRB")
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(8)

    # Transcript
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "FULL CONVERSATION TRANSCRIPT", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    uq_lower = unanswerable_query.strip().lower()
    for msg in history:
        if msg.role == "system":
            continue
        if msg.role == "user":
            is_unanswered = msg.content.strip().lower() == uq_lower
            if is_unanswered:
                pdf.set_fill_color(254, 226, 226)
                prefix = "[USER - UNANSWERED]"
            else:
                pdf.set_fill_color(239, 246, 255)
                prefix = "[USER]"
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(30, 64, 175)
            pdf.multi_cell(0, 7, _safe(f"  {prefix}  {msg.content}"), fill=True)
        elif msg.role in ("ai", "agent"):
            pdf.set_fill_color(249, 250, 251)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(55, 65, 81)
            label = "[LAYLA (AI)]" if msg.role == "ai" else "[AGENT]"
            pdf.multi_cell(0, 6, _safe(f"  {label}  {msg.content}"), fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    pdf.ln(4)
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "Please review the highlighted query and follow up with the user at your earliest convenience.", ln=True)

    return bytes(pdf.output())


# ── LLM email composition ─────────────────────────────────────────────────────

def compose_email_body(session_id: str, history, unanswerable_query: str) -> str:
    """Use the VLLM to write a professional HTML email body. Falls back to template."""
    if not VLLM_HOST:
        return _template_email(session_id, unanswerable_query, history)

    history_lines = []
    for m in history:
        if m.role == "user":
            history_lines.append(f"User: {m.content}")
        elif m.role == "ai":
            history_lines.append(f"AI: {m.content[:300]}")
    conversation_text = "\n".join(history_lines[-20:])   # last 20 turns

    prompt = f"""You are writing a professional support escalation email from an AI insurance assistant.

CONTEXT:
- Platform: InsureHub AI Insurance Advisor ("Layla")
- Session: #{session_id}
- A user asked a question Layla could not answer
- No human agent was available so this email is auto-generated
- A PDF transcript is attached

THE UNANSWERABLE QUESTION (must be highlighted in the email):
"{unanswerable_query}"

CONVERSATION SUMMARY (last few turns):
{conversation_text}

Write a concise professional HTML email body to the support agent. Requirements:
- Open with a brief intro (1-2 sentences)
- Include the unanswerable question in bold/highlighted HTML
- Give a 2-3 sentence summary of the conversation context
- Mention the PDF transcript is attached
- End with a polite request to follow up with the user promptly
- Keep it under 250 words
- HTML only — no subject line, no To/From headers, just the body content
- Use simple tags: <p>, <b>, <span style="background:#fef3c7;padding:2px 6px">, <ul>, <li>"""

    try:
        from openai import OpenAI
        client = OpenAI(base_url=f"{VLLM_HOST}/v1", api_key="dummy")
        resp = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.4,
        )
        body = resp.choices[0].message.content.strip()
        # Ensure it's wrapped in basic HTML if not already
        if not body.strip().startswith("<"):
            body = f"<p>{body}</p>"
        return body
    except Exception as e:
        logger.warning("LLM email composition failed: %s — using template", e)
        return _template_email(session_id, unanswerable_query, history)


def _template_email(session_id: str, unanswerable_query: str, history) -> str:
    msg_count = sum(1 for m in history if m.role == "user")
    return f"""
<p>Hello,</p>

<p>A user on <b>InsureHub</b> needed help that our AI assistant Layla could not provide.
No agent was online at the time, so this email is being sent automatically.</p>

<p><b>Unanswerable Query:</b></p>
<p style="background:#fef3c7;padding:10px 14px;border-left:4px solid #d97706;border-radius:4px">
  &ldquo;{unanswerable_query}&rdquo;
</p>

<p>The user had {msg_count} message(s) in this session (Session ID: <b>#{session_id}</b>).
The full conversation transcript is attached as a PDF for your review.</p>

<p>Please follow up with the user at your earliest convenience.</p>

<p>Best regards,<br>
<b>InsureHub AI System</b></p>
"""


# ── Email dispatch ────────────────────────────────────────────────────────────

def send_escalation_email(session_id: str, history, unanswerable_query: str) -> bool:
    """
    Generate PDF + compose email body with LLM + send via Gmail SMTP to every
    agent listed in the agent-emails database (AGENT_EMAILS_FILE).

    No per-agent credentials are needed — the system sends FROM its own
    GMAIL_SENDER account TO every address the database currently lists, so
    adding or removing a row there changes who receives escalations without
    touching any code or credentials.

    Returns True on success, False on failure or if there are no recipients
    (logs the reason either way).
    """
    if not GMAIL_SENDER or not GMAIL_PASSWORD:
        logger.warning(
            "Email escalation skipped — GMAIL_SENDER / GMAIL_APP_PASSWORD not set. "
            "Add these to your .env to enable email notifications."
        )
        return False

    recipients = _load_agent_emails()
    if not recipients and not os.path.exists(AGENT_EMAILS_FILE):
        # Database file doesn't exist yet — fall back to the legacy single
        # recipient so a fresh install isn't silently mute before anyone's
        # created the file.
        recipients = [AGENT_EMAIL] if AGENT_EMAIL else []
    if not recipients:
        logger.warning(
            "Email escalation skipped for session %s — agent-emails database "
            "at %s has no valid recipients.", session_id, AGENT_EMAILS_FILE,
        )
        return False

    try:
        pdf_bytes = generate_pdf(session_id, history, unanswerable_query)
        body_html = compose_email_body(session_id, history, unanswerable_query)

        msg = MIMEMultipart("mixed")
        msg["From"]    = GMAIL_SENDER
        # Real recipients go in Bcc (the SMTP envelope below, not a visible
        # header) so agents don't see each other's addresses. "To" needs
        # *some* address to be a well-formed message — use the sender's own.
        msg["To"]      = GMAIL_SENDER
        msg["Subject"] = f"[InsureHub] User Needs Help — Session #{session_id}"

        # HTML body
        alt = MIMEMultipart("alternative")
        plain = f"A user (session #{session_id}) had an unanswerable question: {unanswerable_query}\nSee attached PDF for full transcript."
        alt.attach(MIMEText(plain, "plain"))
        alt.attach(MIMEText(_email_wrapper(body_html), "html"))
        msg.attach(alt)

        # PDF attachment
        part = MIMEApplication(pdf_bytes, _subtype="pdf")
        part.add_header("Content-Disposition", "attachment",
                        filename=f"insurehub_session_{session_id}.pdf")
        msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_SENDER, recipients, msg.as_string())

        logger.info(
            "Escalation email sent for session %s → %d agent(s): %s",
            session_id, len(recipients), ", ".join(recipients),
        )
        return True

    except Exception:
        logger.exception("Failed to send escalation email for session %s", session_id)
        return False


def _email_wrapper(body: str) -> str:
    return f"""<!DOCTYPE html><html><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
color:#1f2937;max-width:600px;margin:0 auto;padding:20px">
<div style="background:#4f46e5;color:white;padding:14px 20px;border-radius:8px 8px 0 0">
  <b>🛡 InsureHub — Support Escalation</b>
</div>
<div style="background:#f9fafb;padding:20px;border:1px solid #e5e7eb;border-radius:0 0 8px 8px">
{body}
</div>
<p style="font-size:11px;color:#9ca3af;margin-top:16px">
This is an automated message from the InsureHub AI system.
</p>
</body></html>"""
