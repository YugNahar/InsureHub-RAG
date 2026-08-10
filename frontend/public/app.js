/* =========================================================================
   Layla — InsureHub production chat frontend — app.js
   Single persistent chat window, no sidebar, no multi-conversation
   switching — matches the real production Layla product (confirmed
   against the prior React client, frontend/src/App.tsx, which never had
   a conversation-history sidebar either). session_id is generated once
   and never rotates — even across "Clear chat" (see clearChat()) — so the
   human-agent dashboard and super-admin always see one continuous session
   per user. "Clear chat" empties the visible thread client-side and resets
   server-side conversational/routing state via API.resetConversation() on
   that SAME session_id; the full transcript is never deleted server-side.
   ========================================================================= */

'use strict';

/* =========================================================================
   1. CONFIG
   ========================================================================= */

// Opt INTO mock mode for offline frontend-only dev (set
// window.INSUREHUB_MOCK_MODE = true in a <script> before this file loads).
// Production default is false — this file is served by the same FastAPI
// container as /ask-stream, so it talks to the real backend out of the box.
const MOCK_MODE = window.INSUREHUB_MOCK_MODE ?? false;
const API_BASE_URL = window.INSUREHUB_API_BASE_URL ?? '';

const STORAGE_KEY = 'insurehub_ui_state_v1';

/* =========================================================================
   2. MOCK DATA (offline dev only — see MOCK_MODE above)
   ========================================================================= */

const MOCK_RESPONSES = [
  {
    keywords: ['deductible', 'premium'],
    content:
      "Quick one to untangle.\n\n" +
      "Your **premium** is what you pay to keep the policy active — monthly or yearly, your call. Your **deductible** is what *you* cover first on a claim before we step in.\n\n" +
      "- Higher deductible → lower premium, bigger bill if you claim\n" +
      "- Lower deductible → higher premium, smaller bill if you claim\n\n" +
      "You pick your deductible tier when you set up cover.",
  },
  {
    keywords: ['water', 'damage', 'flood', 'leak', 'home', 'house'],
    content:
      "Here's how a water damage claim usually goes:\n\n" +
      "1. **Stop further damage** if it's safe — shut off the source, move valuables\n" +
      "2. **Photograph everything** before cleanup starts\n" +
      "3. **File the claim** within the notice window in your policy\n" +
      "4. An adjuster typically inspects within a few business days\n\n" +
      "One thing to check: **gradual leaks** are often excluded, while **sudden and accidental** damage usually isn't.",
  },
];

const MOCK_FALLBACK_RESPONSE = {
  content:
    "Happy to help with that.\n\n" +
    "Coverage details vary by plan and by the endorsements attached to your specific policy. The general rule carriers apply: a loss needs to be **sudden, accidental, and named or unexcluded** under your policy to qualify.\n\n" +
    "Tell me a bit more about your situation and I can point to the exact clause.",
};

function pickMockResponse(query) {
  const q = (query || '').toLowerCase();
  return MOCK_RESPONSES.find((r) => r.keywords.some((k) => q.includes(k))) || MOCK_FALLBACK_RESPONSE;
}

function msg(role, content, timestamp) {
  return { id: uid(), role, content, timestamp, feedback: null };
}

// The 11 fields Ava's intake form collects — see travel_bot/routers/chat.py
// REQUIRED_FIELDS. Mirrored here (not fetched) because the shape is fixed
// and small; field_options (the 3 dropdowns' choices) DOES come from the
// backend's ui signal, not hardcoded, since FIELD_OPTIONS_MAP is the single
// source of truth there.
const AVA_FORM_FIELDS = [
  'first_name', 'last_name', 'email', 'mobile_number',
  'coverage_type', 'destination', 'departure', 'start_date', 'end_date',
  'plan_type', 'cover_type', 'date_of_birth',
];

// role: 'ava_form' message — a special card rendered inline in the thread
// (see buildAvaFormCard). formState is the single source of truth for its
// field values: renderThread() fully rebuilds the DOM from state.messages
// on every call (including on every streamed token of any concurrent
// reply), so anything living only in the live DOM would be silently lost
// on the next unrelated render. Every input writes here on change, and the
// card is re-rendered FROM this object every time — never the other way
// around.
function buildAvaFormMessage(ui, collectedFields) {
  const values = {};
  for (const f of AVA_FORM_FIELDS) values[f] = (collectedFields && collectedFields[f]) || '';
  const coverageTypeLocks = (ui && ui.coverage_type_locks) || {};
  // Pre-fill immediately if the document/extraction already knew a coverage
  // type before the form was even shown — same "don't ask what's already
  // determined" principle as picking coverage_type itself.
  const initialLocks = coverageTypeLocks[values.coverage_type] || {};
  for (const [field, value] of Object.entries(initialLocks)) {
    if (!values[field]) values[field] = value;
  }

  // Try to decompose an already-known mobile_number (a resumed session, or
  // one a document upload pre-filled) back into country + national number,
  // so the phone country picker shows something sensible instead of
  // reverting to blank the moment the form renders.
  let mobileCountry = '';
  let mobileNational = '';
  if (values.mobile_number) {
    const raw = values.mobile_number.replace(/[\s-]/g, '');
    const digits = raw.startsWith('+') ? raw.slice(1) : raw;
    const match = Object.entries(COUNTRY_DIAL_CODES)
      .sort((a, b) => b[1].code.length - a[1].code.length)
      .find(([, entry]) => digits.startsWith(entry.code));
    if (match) {
      mobileCountry = match[0];
      mobileNational = digits.slice(match[1].code.length);
    } else {
      mobileNational = digits;
    }
  }

  return {
    id: uid(),
    role: 'ava_form',
    content: '',
    timestamp: Date.now(),
    feedback: null,
    formState: {
      fieldOptions: (ui && ui.field_options) || {},
      coverageTypeLocks,
      values,
      mobileCountry,
      mobileNational,
      fieldErrors: {},
      additionalTravellers: [],
      marketingConsent: false,
      submitted: false,
      quoteReply: null,
      error: null,
    },
  };
}

/* =========================================================================
   3. API MODULE — real backend by default, mock is the opt-in
   ========================================================================= */

const API = {
  baseUrl: API_BASE_URL,

  async checkHealth() {
    if (MOCK_MODE) return true;
    try {
      const res = await fetch(`${this.baseUrl}/health`, { method: 'GET' });
      return res.ok;
    } catch {
      return false;
    }
  },

  // ws:// or wss:// mirroring the page's own protocol — same-origin as
  // /ask-stream, just a different scheme for the persistent connection.
  wsUrl() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const base = this.baseUrl || `${window.location.protocol}//${window.location.host}`;
    return base.replace(/^http/, proto.replace(':', ''));
  },

  // Polling fallback for human-agent messages — used alongside the
  // WebSocket (which only carries out-of-band signals; this call owns
  // actual message delivery, see the polling loop below for why).
  async pollSession(sessionId, after) {
    if (MOCK_MODE) return null;
    try {
      const res = await fetch(`${this.baseUrl}/session/${sessionId}/poll?after=${after}`);
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  },

  async requestHandoff(sessionId) {
    if (MOCK_MODE) return;
    try {
      await fetch(`${this.baseUrl}/session/${sessionId}/request-handoff`, { method: 'POST' });
    } catch { /* best-effort HTTP fallback when WS is down */ }
  },

  async cancelHandoff(sessionId) {
    if (MOCK_MODE) return;
    try {
      await fetch(`${this.baseUrl}/session/${sessionId}/cancel-handoff`, { method: 'POST' });
    } catch { /* best-effort HTTP fallback when WS is down */ }
  },

  // Resets server-side conversation/agent-routing state on the SAME
  // session_id (never deletes the session or its transcript — see
  // clearChat()) so the human-agent dashboard / super-admin keep seeing
  // one continuous session per user across a "Clear chat".
  async resetConversation(sessionId) {
    if (MOCK_MODE) return;
    try {
      await fetch(`${this.baseUrl}/conversation/reset/${sessionId}`, { method: 'POST' });
    } catch { /* best-effort — worst case stale agent-routing state lingers until the next reset */ }
  },

  // Async generator yielding { type: 'chunk', text } | { type: 'done', ... }.
  async *streamAsk(query, sessionId, signal) {
    if (MOCK_MODE) {
      yield* mockStreamAsk(query, signal);
      return;
    }

    // Real RAG_InsureAI FastAPI /ask-stream contract: body is
    // {question, session_id}; response is plain streamed answer text with
    // a final '\n\n{"sources": [...], "done": true, ...}' JSON blob
    // appended at the end — not NDJSON/SSE. Same parsing already proven
    // in the production Layla client (frontend lib/api.ts) and the
    // Shadow-DOM widget build.
    const res = await fetch(`${this.baseUrl}/ask-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: query, session_id: sessionId }),
      signal,
    });
    if (!res.ok || !res.body) {
      throw new Error(`Request failed (${res.status || 'network error'})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const jsonStart = buffer.lastIndexOf('\n\n{"sources"');
      if (jsonStart !== -1) {
        const textPart = buffer.slice(0, jsonStart);
        const jsonPart = buffer.slice(jsonStart + 2);
        if (textPart) yield { type: 'chunk', text: textPart };
        try {
          const meta = JSON.parse(jsonPart);
          // correctedText: the backend re-checks the fully-streamed answer
          // (trimming a mid-sentence cutoff, dropping an ungrounded/cross-
          // topic-contaminated point, stripping a Rule4 fallback line) and
          // sends the corrected version here — the streamed text the user
          // already saw is the PRE-correction version. Same field the
          // proven production client (frontend/src/App.tsx) replaces the
          // message content with; dropping it here would silently show
          // users the uncorrected answer even when the backend caught and
          // fixed a problem with it.
          yield {
            type: 'done',
            correctedText: meta.corrected_text || null,
            needsHuman: !!meta.needs_human,
            offlineEscalated: !!meta.offline_escalated,
            ui: meta.ui || null,
            collectedFields: meta.collected_fields || null,
          };
        } catch {
          yield { type: 'done' };
        }
        return;
      }
      if (buffer) { yield { type: 'chunk', text: buffer }; buffer = ''; }
    }
    if (buffer) yield { type: 'chunk', text: buffer };
    yield { type: 'done' };
  },
};

async function* mockStreamAsk(query, signal) {
  const demo = pickMockResponse(query);
  await sleep(650 + Math.random() * 500, signal);
  const tokens = demo.content.split(/(\s+)/);
  for (const t of tokens) {
    if (signal?.aborted) return;
    yield { type: 'chunk', text: t };
    if (t.trim()) await sleep(16 + Math.random() * 34, signal);
  }
  yield { type: 'done' };
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => { clearTimeout(t); reject(new DOMException('aborted', 'AbortError')); });
  });
}

/* =========================================================================
   4. MARKDOWN RENDERING
   ========================================================================= */

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split('\n');
  let html = '';
  let inList = false;
  const closeList = () => { if (inList) { html += '</ul>'; inList = false; } };

  for (const line of lines) {
    if (/^#{1,3}\s+/.test(line)) {
      closeList();
      const level = line.match(/^#{1,3}/)[0].length;
      html += `<h${level}>${inlineMd(line.replace(/^#{1,3}\s+/, ''))}</h${level}>`;
    } else if (/^\d+\.\s+/.test(line)) {
      closeList();
      html += `<p>${inlineMd(line)}</p>`;
    } else if (/^[-*]\s+/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inlineMd(line.replace(/^[-*]\s+/, ''))}</li>`;
    } else if (line.trim() === '') {
      closeList();
    } else {
      closeList();
      html += `<p>${inlineMd(line)}</p>`;
    }
  }
  closeList();
  return html;
}

function inlineMd(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>');
}

// Real HTML <table> for Ava's quote-comparison replies — table is
// {columns: [name, ...], rows: [{label, values: [...]}]} from
// travel_bot/routers/chat.py's format_compare_quotes. Wrapped in a
// horizontally-scrollable container since column names (insurer + plan)
// can be long and this renders inside a narrow chat bubble.
function buildComparisonTable(table) {
  const wrap = document.createElement('div');
  wrap.className = 'compare-table-wrap';

  const el = document.createElement('table');
  el.className = 'compare-table';

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  headRow.appendChild(document.createElement('th'));
  for (const col of table.columns) {
    const th = document.createElement('th');
    th.textContent = col;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  el.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (const row of table.rows) {
    const tr = document.createElement('tr');
    const labelCell = document.createElement('th');
    labelCell.scope = 'row';
    labelCell.textContent = row.label;
    tr.appendChild(labelCell);
    for (const val of row.values) {
      const td = document.createElement('td');
      td.textContent = val;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  el.appendChild(tbody);

  wrap.appendChild(el);
  return wrap;
}

// "Email me this quote" — attached to any assistant message with
// isQuoteList set (see the SSE done-handler and submitAvaForm). The quote
// data itself is never sent from here — /api/chat/email-quote re-reads it
// from the session's own state server-side, so this only ever needs to
// know WHO to send it to. UI-only state (is the input row open, was it
// already sent) lives on the message object itself, not the DOM — a
// concurrent renderThread() rebuild (e.g. from a background stream chunk)
// would otherwise silently reset an open input row, same lesson as the
// Ava form card's own formState.
function buildEmailQuoteButton(m) {
  const wrap = document.createElement('div');
  wrap.className = 'email-quote';

  if (m.emailSentTo) {
    const sentEl = document.createElement('div');
    sentEl.className = 'email-quote__sent';
    sentEl.textContent = `✅ Sent to ${m.emailSentTo}`;
    wrap.appendChild(sentEl);
    return wrap;
  }

  if (!m.emailRowOpen) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'email-quote__btn';
    btn.textContent = '📧 Email me this quote';
    btn.addEventListener('click', () => {
      m.emailRowOpen = true;
      saveState();
      rerenderEmailQuoteButton(m);
    });
    wrap.appendChild(btn);
    return wrap;
  }

  const row = document.createElement('div');
  row.className = 'email-quote__row';

  const input = document.createElement('input');
  input.type = 'email';
  input.className = 'email-quote__input';
  input.placeholder = 'you@example.com';
  input.value = m.emailDraft || getDefaultAvaEmail();
  input.addEventListener('input', () => { m.emailDraft = input.value; });

  const sendBtn = document.createElement('button');
  sendBtn.type = 'button';
  sendBtn.className = 'email-quote__send';
  sendBtn.textContent = 'Send';

  const errEl = document.createElement('span');
  errEl.className = 'email-quote__error';
  errEl.hidden = true;

  sendBtn.addEventListener('click', () => sendQuoteEmail(m, input.value, errEl, sendBtn));

  row.appendChild(input);
  row.appendChild(sendBtn);
  wrap.appendChild(row);
  wrap.appendChild(errEl);
  return wrap;
}

function rerenderEmailQuoteButton(m) {
  const container = document.querySelector(`.msg[data-id="${m.id}"] .email-quote`);
  if (!container) return;
  container.replaceWith(buildEmailQuoteButton(m));
}

// Pre-fills the email input from the most recent Ava form's own email
// field, if any — still asks the user (the field stays editable, matching
// "ask the mail from the user"), just conveniently defaults instead of
// starting blank.
function getDefaultAvaEmail() {
  for (let i = state.messages.length - 1; i >= 0; i--) {
    const cand = state.messages[i];
    if (cand.role === 'ava_form' && cand.formState && cand.formState.values && cand.formState.values.email) {
      return cand.formState.values.email;
    }
  }
  return '';
}

async function sendQuoteEmail(m, email, errEl, sendBtn) {
  if (!isValidEmail(email)) {
    errEl.textContent = 'Enter a valid email address.';
    errEl.hidden = false;
    return;
  }
  errEl.hidden = true;
  sendBtn.disabled = true;
  sendBtn.textContent = 'Sending…';

  try {
    const res = await fetch(`${API.baseUrl}/api/chat/email-quote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, email }),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = data.detail;
      errEl.textContent = (detail && typeof detail === 'object' && detail.message)
        || (typeof detail === 'string' && detail)
        || 'Something went wrong sending the email.';
      errEl.hidden = false;
      sendBtn.disabled = false;
      sendBtn.textContent = 'Send';
      return;
    }
    m.emailSentTo = email;
    m.emailRowOpen = false;
    saveState();
    rerenderEmailQuoteButton(m);
    pushSystemMessage(`✅ Sent your quote to ${email}.`);
  } catch (err) {
    errEl.textContent = 'Network error — please try again.';
    errEl.hidden = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send';
  }
}

/* =========================================================================
   5. STATE + PERSISTENCE
   ========================================================================= */

// Single persistent chat — no conversation list, no switching. sessionId
// is generated once and kept forever (including across "Clear chat"), so
// backend-side conversation context stays tied to the same session.
const state = {
  sessionId: null,
  messages: [],
};

function uid() { return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`; }

function loadState() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch { /* ignore corrupt state */ }

  if (saved && typeof saved.sessionId === 'string' && Array.isArray(saved.messages)) {
    state.sessionId = saved.sessionId;
    state.messages = saved.messages;
  } else {
    state.sessionId = uid();
    state.messages = [];
  }
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      sessionId: state.sessionId,
      messages: state.messages,
    }));
  } catch { /* storage full/unavailable — non-fatal, app still works this session */ }
}

/* =========================================================================
   6. DOM REFS
   ========================================================================= */

const $ = (sel) => document.querySelector(sel);
const el = {
  clearChatBtn: $('#clearChatBtn'),
  statusDot: $('#statusDot'),
  statusText: $('#statusText'),
  thread: $('#thread'),
  jumpLatestBtn: $('#jumpLatestBtn'),
  composerForm: $('#composerForm'),
  composerInput: $('#composer-input'),
  sendBtn: $('#sendBtn'),
  srAnnouncer: $('#srAnnouncer'),
};

let userScrolledAway = false;
let activeAbortController = null;

// Set by clearChat(), awaited at the top of sendMessage() — closes a real
// race: API.resetConversation() used to be fire-and-forget, so a user who
// cleared the chat and immediately sent a new message (paste + enter, or
// just typing fast) could have that message reach the backend BEFORE the
// reset did, landing on the still-stale active_agent and silently going
// to whatever specialist agent the session was stuck on. Confirmed live.
let pendingResetPromise = null;

// Human-agent handoff state — not persisted to localStorage; re-derived
// from the server on every poll tick / page load, since the backend is
// the source of truth for whether an agent is actually connected.
let chatMode = 'ai'; // 'ai' | 'waiting' | 'human'
let agentName = null;
let pollSeen = 0;
let ws = null;
let pollIntervalId = null;

/* =========================================================================
   7. RENDERING — thread
   ========================================================================= */

function renderThread() {
  el.thread.innerHTML = '';

  if (state.messages.length === 0) {
    el.thread.appendChild(buildEmptyState());
    return;
  }

  const inner = document.createElement('div');
  inner.className = 'thread-inner';
  for (const m of state.messages) {
    inner.appendChild(buildMessageEl(m));
  }
  el.thread.appendChild(inner);
  scrollToBottom(true);
}

const EXAMPLE_QUESTIONS = [
  "What's the difference between a deductible and a premium?",
  'How do I file a claim after water damage?',
  'Does my travel policy cover a cancelled flight?',
  'What happens if I miss a premium payment?',
  'Is accidental damage to my phone covered under home insurance?',
];

function buildEmptyState() {
  const wrap = document.createElement('div');
  wrap.className = 'empty-state';
  wrap.innerHTML = `
    <div class="empty-state__icon" aria-hidden="true">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 2L3 6.5V11C3 16.2 6.8 20.9 12 22C17.2 20.9 21 16.2 21 11V6.5L12 2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8.5 12.2L10.8 14.5L15.5 9.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </div>
    <h1>Hi, I'm Layla 👋</h1>
    <p>Ask about a policy, a claim, or coverage — I'm here to help.</p>
    <div class="example-grid"></div>
  `;
  const grid = wrap.querySelector('.example-grid');
  for (const q of EXAMPLE_QUESTIONS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'example-chip';
    btn.textContent = q;
    btn.addEventListener('click', () => sendMessage(q));
    grid.appendChild(btn);
  }
  return wrap;
}

function buildMessageEl(m) {
  const wrap = document.createElement('div');
  wrap.className = `msg msg--${m.role}`;
  wrap.dataset.id = m.id;

  const avatar = document.createElement('div');
  avatar.className = 'msg__avatar';
  avatar.setAttribute('aria-hidden', 'true');
  avatar.textContent = m.role === 'user' ? 'You' : m.role === 'agent' ? (m.agentName || 'A')[0].toUpperCase() : m.role === 'ava_form' ? 'A' : 'L';

  const col = document.createElement('div');
  col.className = 'msg__col';

  const bubble = document.createElement('div');
  bubble.className = m.role === 'ava_form' ? 'msg__bubble msg__bubble--form' : 'msg__bubble';
  if (m.role === 'ava_form') {
    bubble.appendChild(buildAvaFormCard(m));
  } else {
    bubble.innerHTML = (m.role === 'assistant' || m.role === 'agent')
      ? renderMarkdown(m.content)
      : escapeHtml(m.content);
    if (m.comparisonTable) {
      bubble.appendChild(buildComparisonTable(m.comparisonTable));
    }
    if (m.isQuoteList) {
      bubble.appendChild(buildEmailQuoteButton(m));
    }
  }

  if (m.role === 'system') {
    // Centered notice pill (handoff transitions) — no timestamp, no avatar.
    col.appendChild(bubble);
  } else {
    const meta = document.createElement('div');
    meta.className = 'msg__meta';
    meta.textContent = m.role === 'agent' && m.agentName
      ? `${m.agentName} · ${formatTime(m.timestamp)}`
      : formatTime(m.timestamp);
    if (m.role === 'agent' && m.answersQuestion) {
      const answersLine = document.createElement('span');
      answersLine.className = 'msg__meta--answers';
      answersLine.textContent = `Replying to: "${m.answersQuestion}"`;
      meta.appendChild(answersLine);
    }
    col.appendChild(meta);
    col.appendChild(bubble);
  }

  if (m.role === 'assistant' && m.content) {
    col.appendChild(buildMessageControls(m));
  }

  wrap.appendChild(avatar);
  wrap.appendChild(col);
  return wrap;
}

/* =========================================================================
   6b. AVA INTAKE FORM CARD
   ========================================================================= */

const AVA_FIELD_LABELS = {
  first_name: 'First name',
  last_name: 'Last name',
  email: 'Email',
  mobile_number: 'Mobile number',
  destination: 'Destination country',
  departure: "Country you're travelling from",
  start_date: 'Trip start date',
  end_date: 'Trip end date',
  date_of_birth: 'Date of birth',
  coverage_type: 'Coverage type',
  plan_type: 'Plan type',
  cover_type: "Who's insured",
};

const AVA_FIELD_INPUT_TYPES = {
  email: 'email',
  mobile_number: 'tel',
  start_date: 'date',
  end_date: 'date',
  date_of_birth: 'date',
};

const AVA_MULTI_TRAVELLER_COVER_TYPES = ['Group', 'Family'];

// Country -> {code, digits}: dial code (no '+') and the set of national
// significant number lengths that country's mobile numbers actually use.
// Best-effort, not a full numbering-plan library (e.g. libphonenumber) —
// most countries have one common mobile length, a few genuinely allow more
// than one (Indonesia, Germany, Brazil). Mirrors the dial codes
// quote_service.py's _KNOWN_DIAL_CODES recognizes server-side, so a number
// composed here always splits correctly on the backend too.
// Digit-length ranges below are ITU/national-numbering-plan best-effort,
// not authoritative for every one of the ~195 countries — where the exact
// national significant number length wasn't confidently known, a wider
// plausible range is used instead of a fabricated precise value (still
// catches obviously-wrong lengths, matches this validator's own documented
// "better to accept than to wrongly block" philosophy below).
const _digitRange = (min, max) => Array.from({ length: max - min + 1 }, (_, i) => min + i);

const COUNTRY_DIAL_CODES = {
  // Gulf / Middle East
  'United Arab Emirates': { code: '971', digits: [9] },
  'Saudi Arabia': { code: '966', digits: [9] },
  'Qatar': { code: '974', digits: [8] },
  'Kuwait': { code: '965', digits: [8] },
  'Bahrain': { code: '973', digits: [8] },
  'Oman': { code: '968', digits: [8] },
  'Jordan': { code: '962', digits: [9] },
  'Lebanon': { code: '961', digits: [7, 8] },
  'Iraq': { code: '964', digits: [10] },
  'Israel': { code: '972', digits: [9] },
  'Palestine': { code: '970', digits: [9] },
  'Syria': { code: '963', digits: [9] },
  'Yemen': { code: '967', digits: [9] },
  'Iran': { code: '98', digits: [10] },
  // South Asia
  'India': { code: '91', digits: [10] },
  'Pakistan': { code: '92', digits: [10] },
  'Bangladesh': { code: '880', digits: [10] },
  'Sri Lanka': { code: '94', digits: [9] },
  'Nepal': { code: '977', digits: [10] },
  'Bhutan': { code: '975', digits: [7, 8] },
  'Maldives': { code: '960', digits: [7] },
  'Afghanistan': { code: '93', digits: [9] },
  // Southeast Asia
  'Philippines': { code: '63', digits: [10] },
  'Indonesia': { code: '62', digits: [9, 10, 11, 12] },
  'Malaysia': { code: '60', digits: [9, 10] },
  'Singapore': { code: '65', digits: [8] },
  'Thailand': { code: '66', digits: [9] },
  'Vietnam': { code: '84', digits: [9, 10] },
  'Myanmar': { code: '95', digits: _digitRange(7, 10) },
  'Cambodia': { code: '855', digits: [8, 9] },
  'Laos': { code: '856', digits: [8, 9, 10] },
  'Brunei': { code: '673', digits: [7] },
  'Timor-Leste': { code: '670', digits: [7, 8] },
  // East Asia
  'China': { code: '86', digits: [11] },
  'Hong Kong': { code: '852', digits: [8] },
  'Macau': { code: '853', digits: [8] },
  'Japan': { code: '81', digits: [10] },
  'South Korea': { code: '82', digits: [9, 10] },
  'North Korea': { code: '850', digits: _digitRange(6, 8) },
  'Taiwan': { code: '886', digits: [9] },
  'Mongolia': { code: '976', digits: [8] },
  // Central Asia / Caucasus
  'Kazakhstan': { code: '7', digits: [10] },
  'Uzbekistan': { code: '998', digits: [9] },
  'Turkmenistan': { code: '993', digits: [8] },
  'Kyrgyzstan': { code: '996', digits: [9] },
  'Tajikistan': { code: '992', digits: [9] },
  'Armenia': { code: '374', digits: [8] },
  'Azerbaijan': { code: '994', digits: [9] },
  'Georgia': { code: '995', digits: [9] },
  // Europe
  'United Kingdom': { code: '44', digits: [10] },
  'Ireland': { code: '353', digits: [9] },
  'France': { code: '33', digits: [9] },
  'Germany': { code: '49', digits: [10, 11] },
  'Italy': { code: '39', digits: [9, 10] },
  'Spain': { code: '34', digits: [9] },
  'Portugal': { code: '351', digits: [9] },
  'Netherlands': { code: '31', digits: [9] },
  'Belgium': { code: '32', digits: [9] },
  'Austria': { code: '43', digits: [10, 11] },
  'Greece': { code: '30', digits: [10] },
  'Finland': { code: '358', digits: [9, 10] },
  'Switzerland': { code: '41', digits: [9] },
  'Norway': { code: '47', digits: [8] },
  'Sweden': { code: '46', digits: [9] },
  'Denmark': { code: '45', digits: [8] },
  'Poland': { code: '48', digits: [9] },
  'Turkey': { code: '90', digits: [10] },
  'Russia': { code: '7', digits: [10] },
  'Ukraine': { code: '380', digits: [9] },
  'Belarus': { code: '375', digits: [9] },
  'Moldova': { code: '373', digits: [8] },
  'Romania': { code: '40', digits: [9] },
  'Bulgaria': { code: '359', digits: [8, 9] },
  'Serbia': { code: '381', digits: [8, 9] },
  'Croatia': { code: '385', digits: [8, 9] },
  'Bosnia and Herzegovina': { code: '387', digits: [8] },
  'Montenegro': { code: '382', digits: [8] },
  'North Macedonia': { code: '389', digits: [8] },
  'Kosovo': { code: '383', digits: [8] },
  'Albania': { code: '355', digits: [9] },
  'Slovenia': { code: '386', digits: [8] },
  'Slovakia': { code: '421', digits: [9] },
  'Czech Republic': { code: '420', digits: [9] },
  'Hungary': { code: '36', digits: [8, 9] },
  'Iceland': { code: '354', digits: [7] },
  'Estonia': { code: '372', digits: [7, 8] },
  'Latvia': { code: '371', digits: [8] },
  'Lithuania': { code: '370', digits: [8] },
  'Luxembourg': { code: '352', digits: [9] },
  'Malta': { code: '356', digits: [8] },
  'Cyprus': { code: '357', digits: [8] },
  'Monaco': { code: '377', digits: [8, 9] },
  'San Marino': { code: '378', digits: _digitRange(6, 10) },
  'Vatican City': { code: '379', digits: _digitRange(6, 10) },
  'Liechtenstein': { code: '423', digits: [7] },
  'Andorra': { code: '376', digits: [6] },
  // Americas
  'United States': { code: '1', digits: [10] },
  'Canada': { code: '1', digits: [10] },
  'Mexico': { code: '52', digits: [10] },
  'Brazil': { code: '55', digits: [10, 11] },
  'Argentina': { code: '54', digits: [10, 11] },
  'Chile': { code: '56', digits: [9] },
  'Colombia': { code: '57', digits: [10] },
  'Peru': { code: '51', digits: [9] },
  'Venezuela': { code: '58', digits: [10] },
  'Ecuador': { code: '593', digits: [8, 9] },
  'Bolivia': { code: '591', digits: [8] },
  'Paraguay': { code: '595', digits: [9] },
  'Uruguay': { code: '598', digits: [8] },
  'Guyana': { code: '592', digits: [7] },
  'Suriname': { code: '597', digits: [6, 7] },
  'Guatemala': { code: '502', digits: [8] },
  'Belize': { code: '501', digits: [7] },
  'Honduras': { code: '504', digits: [8] },
  'El Salvador': { code: '503', digits: [8] },
  'Nicaragua': { code: '505', digits: [8] },
  'Costa Rica': { code: '506', digits: [8] },
  'Panama': { code: '507', digits: [7, 8] },
  'Cuba': { code: '53', digits: [8] },
  'Jamaica': { code: '1', digits: [10] },
  'Haiti': { code: '509', digits: [8] },
  'Dominican Republic': { code: '1', digits: [10] },
  'Bahamas': { code: '1', digits: [10] },
  'Trinidad and Tobago': { code: '1', digits: [10] },
  'Barbados': { code: '1', digits: [10] },
  'Antigua and Barbuda': { code: '1', digits: [10] },
  'Dominica': { code: '1', digits: [10] },
  'Grenada': { code: '1', digits: [10] },
  'Saint Kitts and Nevis': { code: '1', digits: [10] },
  'Saint Lucia': { code: '1', digits: [10] },
  'Saint Vincent and the Grenadines': { code: '1', digits: [10] },
  // Africa
  'Egypt': { code: '20', digits: [10] },
  'Morocco': { code: '212', digits: [9] },
  'Tunisia': { code: '216', digits: [8] },
  'Algeria': { code: '213', digits: [9] },
  'Libya': { code: '218', digits: [9] },
  'Sudan': { code: '249', digits: [9] },
  'South Sudan': { code: '211', digits: [9] },
  'South Africa': { code: '27', digits: [9] },
  'Nigeria': { code: '234', digits: [10] },
  'Kenya': { code: '254', digits: [9] },
  'Ghana': { code: '233', digits: [9] },
  'Ethiopia': { code: '251', digits: [9] },
  'Tanzania': { code: '255', digits: [9] },
  'Uganda': { code: '256', digits: [9] },
  'Rwanda': { code: '250', digits: [9] },
  'Burundi': { code: '257', digits: [8] },
  'Angola': { code: '244', digits: [9] },
  'Zambia': { code: '260', digits: [9] },
  'Zimbabwe': { code: '263', digits: _digitRange(7, 9) },
  'Malawi': { code: '265', digits: _digitRange(7, 9) },
  'Mozambique': { code: '258', digits: [9] },
  'Namibia': { code: '264', digits: _digitRange(7, 9) },
  'Botswana': { code: '267', digits: [7, 8] },
  'Eswatini': { code: '268', digits: [8] },
  'Lesotho': { code: '266', digits: [8] },
  'Madagascar': { code: '261', digits: [9] },
  'Mauritius': { code: '230', digits: [7, 8] },
  'Seychelles': { code: '248', digits: [6, 7] },
  'Comoros': { code: '269', digits: [7] },
  'Djibouti': { code: '253', digits: [8] },
  'Eritrea': { code: '291', digits: [7] },
  'Somalia': { code: '252', digits: _digitRange(7, 9) },
  'Cameroon': { code: '237', digits: [9] },
  'Central African Republic': { code: '236', digits: [8] },
  'Chad': { code: '235', digits: [8] },
  'Congo': { code: '242', digits: [9] },
  'DR Congo': { code: '243', digits: [9] },
  'Gabon': { code: '241', digits: [8, 9] },
  'Equatorial Guinea': { code: '240', digits: [9] },
  'Sao Tome and Principe': { code: '239', digits: [7] },
  'Senegal': { code: '221', digits: [9] },
  'Gambia': { code: '220', digits: [7] },
  'Guinea': { code: '224', digits: [9] },
  'Guinea-Bissau': { code: '245', digits: [7] },
  'Mali': { code: '223', digits: [8] },
  'Mauritania': { code: '222', digits: [8] },
  'Niger': { code: '227', digits: [8] },
  'Burkina Faso': { code: '226', digits: [8] },
  'Ivory Coast': { code: '225', digits: [8, 10] },
  'Sierra Leone': { code: '232', digits: [8] },
  'Liberia': { code: '231', digits: _digitRange(7, 9) },
  'Togo': { code: '228', digits: [8] },
  'Benin': { code: '229', digits: [8] },
  'Cape Verde': { code: '238', digits: [7] },
  // Oceania
  'Australia': { code: '61', digits: [9] },
  'New Zealand': { code: '64', digits: [8, 9] },
  'Fiji': { code: '679', digits: [7] },
  'Papua New Guinea': { code: '675', digits: [7, 8] },
  'Samoa': { code: '685', digits: _digitRange(5, 7) },
  'Tonga': { code: '676', digits: [5, 7] },
  'Vanuatu': { code: '678', digits: [5, 7] },
  'Solomon Islands': { code: '677', digits: [5, 7] },
  'Kiribati': { code: '686', digits: [5, 8] },
  'Micronesia': { code: '691', digits: [7] },
  'Marshall Islands': { code: '692', digits: [7] },
  'Palau': { code: '680', digits: [7] },
  'Nauru': { code: '674', digits: [7] },
  'Tuvalu': { code: '688', digits: [5, 6] },
};

// A document-upload pre-fill's mobile_number arrives as one E.164-shaped
// string ("+919876543210"), but the phone widget itself renders from a
// {country, national} split (fs.mobileCountry/fs.mobileNational), not
// mobile_number directly — without this, a successfully-extracted number
// still shows an empty country picker. Longest-code-first avoids a short
// code false-matching a number that actually belongs to a longer one.
function decomposeMobileNumber(fullNumber) {
  const digits = (fullNumber || '').replace(/^\+/, '').replace(/\D/g, '');
  if (!digits) return null;
  const candidates = Object.entries(COUNTRY_DIAL_CODES)
    .filter(([, entry]) => digits.startsWith(entry.code))
    .sort((a, b) => b[1].code.length - a[1].code.length);
  if (!candidates.length) return null;
  const [country, entry] = candidates[0];
  return { country, national: digits.slice(entry.code.length) };
}

// Deliberately permissive — accepts any real email shape (business,
// personal, education, subdomains, plus-addressing) rather than a strict
// RFC 5322 pattern, which tends to reject valid real-world addresses more
// often than it catches genuinely malformed ones. The final label must
// still look like a real TLD (letters only, 2+ chars) — the previous
// `[^\s@]+$` tail accepted "user@test.1", "user@test.c", and even a bare
// trailing dot like "user@test.com.", since dots are allowed inside that
// class too. This can't catch a *real but misspelled* domain
// ("gmial.com") — no regex can, only DNS/MX lookup or a verification-link
// send can prove a domain is real and reachable.
function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[a-zA-Z]{2,}$/.test((email || '').trim());
}

// National-number-only validation against the selected phone country (a
// missing/unrecognized country just skips the length check — better to
// accept than to block on a country we don't have data for).
function isValidPhoneForCountry(nationalNumber, countryName) {
  const digits = (nationalNumber || '').replace(/\D/g, '');
  if (!digits) return false;
  const entry = COUNTRY_DIAL_CODES[countryName];
  if (!entry) return /^\d{6,14}$/.test(digits);
  return entry.digits.includes(digits.length);
}

// Re-renders just this one form card in place from its formState — never
// the whole thread. Structural changes (a dropdown that toggles the
// companion section, an upload/submit result) need this; a plain text/date
// keystroke does not and must NOT call it (renderThread()-style full
// rebuilds steal input focus — fine for a background render the user isn't
// looking at, not fine for the field they're actively typing into).
function rerenderAvaFormCard(m) {
  const bubble = document.querySelector(`.msg[data-id="${m.id}"] .msg__bubble--form`);
  if (!bubble) return;
  bubble.innerHTML = '';
  bubble.appendChild(buildAvaFormCard(m));
}

function buildAvaFormCard(m) {
  const fs = m.formState;
  const card = document.createElement('div');
  card.className = 'ava-form';

  if (fs.submitted) {
    const title = document.createElement('div');
    title.className = 'ava-form__summary-title';
    title.textContent = '✅ Details submitted';
    card.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'ava-form__summary-grid';
    for (const f of AVA_FORM_FIELDS) {
      const row = document.createElement('div');
      row.className = 'ava-form__summary-row';
      const label = document.createElement('span');
      label.className = 'ava-form__summary-label';
      label.textContent = `${AVA_FIELD_LABELS[f]}:`;
      const value = document.createElement('span');
      value.textContent = fs.values[f] || '—';
      row.appendChild(label);
      row.appendChild(value);
      grid.appendChild(row);
    }
    card.appendChild(grid);

    const editBtn = document.createElement('button');
    editBtn.type = 'button';
    editBtn.className = 'ava-form__edit-btn';
    editBtn.textContent = '✏️ Edit details';
    editBtn.addEventListener('click', () => {
      fs.submitted = false;
      fs.editing = true;
      saveState();
      rerenderAvaFormCard(m);
    });
    card.appendChild(editBtn);

    return card;
  }

  if (fs.error) {
    const errBox = document.createElement('div');
    errBox.className = 'ava-form__error';
    errBox.textContent = fs.error;
    card.appendChild(errBox);
  }

  const intro = document.createElement('p');
  intro.className = 'ava-form__intro';
  intro.textContent = 'Fill in your trip details, or upload a document below and I\'ll pre-fill what I can.';
  card.appendChild(intro);

  // ── Document upload ──────────────────────────────────────────────────
  const uploadRow = document.createElement('div');
  uploadRow.className = 'ava-form__upload-row';
  const uploadBtn = document.createElement('button');
  uploadBtn.type = 'button';
  uploadBtn.className = 'ava-form__upload-btn';
  uploadBtn.textContent = '📄 Upload a document to pre-fill';
  const uploadInput = document.createElement('input');
  uploadInput.type = 'file';
  uploadInput.accept = '.pdf,image/*';
  uploadInput.style.display = 'none';
  const uploadStatus = document.createElement('span');
  uploadStatus.className = 'ava-form__upload-status';
  uploadStatus.textContent = fs.uploadStatus || '';
  uploadBtn.addEventListener('click', () => uploadInput.click());
  uploadInput.addEventListener('change', () => {
    const file = uploadInput.files && uploadInput.files[0];
    if (file) handleAvaFormUpload(m, file);
  });
  uploadRow.appendChild(uploadBtn);
  uploadRow.appendChild(uploadInput);
  uploadRow.appendChild(uploadStatus);
  card.appendChild(uploadRow);

  // ── Core fields ───────────────────────────────────────────────────────
  const grid = document.createElement('div');
  grid.className = 'ava-form__grid';
  const CORE_FIELDS = ['first_name', 'last_name', 'email', 'date_of_birth', 'destination', 'departure', 'start_date', 'end_date'];
  for (const f of CORE_FIELDS) {
    grid.appendChild(buildAvaFormField(m, f));
    // Mobile number gets its own country-code selector right after email —
    // matches the order REQUIRED_FIELDS/FIELD_LABELS already use server-side.
    if (f === 'email') grid.appendChild(buildAvaFormPhoneField(m));
  }
  card.appendChild(grid);

  // ── Dropdowns ─────────────────────────────────────────────────────────
  const dropdownGrid = document.createElement('div');
  dropdownGrid.className = 'ava-form__grid';
  for (const f of ['coverage_type', 'plan_type', 'cover_type']) {
    dropdownGrid.appendChild(buildAvaFormDropdown(m, f));
  }
  card.appendChild(dropdownGrid);

  // ── Companion travellers (Group/Family only) ────────────────────────
  if (AVA_MULTI_TRAVELLER_COVER_TYPES.includes(fs.values.cover_type)) {
    card.appendChild(buildAvaCompanionSection(m));
  }

  // ── Marketing consent ─────────────────────────────────────────────────
  const consentRow = document.createElement('label');
  consentRow.className = 'ava-form__consent';
  const consentBox = document.createElement('input');
  consentBox.type = 'checkbox';
  consentBox.checked = !!fs.marketingConsent;
  consentBox.addEventListener('change', () => { fs.marketingConsent = consentBox.checked; saveState(); });
  consentRow.appendChild(consentBox);
  consentRow.appendChild(document.createTextNode(' I\'m okay being contacted about relevant offers.'));
  card.appendChild(consentRow);

  // ── Submit ────────────────────────────────────────────────────────────
  const submitBtn = document.createElement('button');
  submitBtn.type = 'button';
  submitBtn.className = 'ava-form__submit';
  submitBtn.textContent = fs.submitting
    ? (fs.editing ? 'Updating…' : 'Getting your quote…')
    : (fs.editing ? 'Update & get new quote' : 'Get my quote');
  submitBtn.disabled = !!fs.submitting;
  submitBtn.addEventListener('click', () => submitAvaForm(m));
  card.appendChild(submitBtn);

  return card;
}

// yyyy-mm-dd in the browser's LOCAL date, not UTC — new Date().toISOString()
// would roll back to yesterday for anyone west of UTC in the evening,
// wrongly excluding today from "present and future" / "present and past".
function _todayDateStr() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function buildAvaFormField(m, field) {
  const fs = m.formState;
  const wrap = document.createElement('label');
  wrap.className = 'ava-form__field';
  const labelText = document.createElement('span');
  labelText.className = 'ava-form__label';
  labelText.textContent = AVA_FIELD_LABELS[field];
  const input = document.createElement('input');
  input.type = AVA_FIELD_INPUT_TYPES[field] || 'text';
  input.value = fs.values[field] || '';
  input.dataset.field = field;

  // Calendar bounds: date of birth can't be in the future; a trip can't
  // start (or end) in the past. end_date's min also follows the currently
  // chosen start_date (updated live below) so an end date before the trip
  // even starts isn't selectable either.
  if (field === 'date_of_birth') {
    input.max = _todayDateStr();
  } else if (field === 'start_date') {
    input.min = _todayDateStr();
  } else if (field === 'end_date') {
    input.min = fs.values.start_date || _todayDateStr();
  }
  // destination/departure can be locked to a single value by the chosen
  // coverage_type (e.g. "UAE Inbound" always means destination="United Arab
  // Emirates") — see COVERAGE_TYPE_LOCKS server-side. Disabling it here
  // instead of leaving it editable-but-prefilled avoids a round trip to the
  // exact same validation error the backend would otherwise return.
  const lockedValue = (fs.coverageTypeLocks[fs.values.coverage_type] || {})[field];
  if (lockedValue) {
    input.value = lockedValue;
    input.disabled = true;
    labelText.textContent = `${AVA_FIELD_LABELS[field]} (set by coverage type)`;
  }
  input.addEventListener('input', () => {
    fs.values[field] = input.value;
    saveState();
    if (field === 'start_date') {
      const endInput = wrap.parentElement && wrap.parentElement.querySelector('input[data-field="end_date"]');
      if (endInput) {
        endInput.min = input.value || _todayDateStr();
        if (endInput.value && endInput.value < endInput.min) {
          endInput.value = '';
          fs.values.end_date = '';
          saveState();
        }
      }
    }
  });
  wrap.appendChild(labelText);
  wrap.appendChild(input);

  if (field === 'email') {
    const errEl = document.createElement('span');
    errEl.className = 'ava-form__field-error';
    errEl.hidden = true;
    const showEmailError = () => {
      const bad = input.value.trim() && !isValidEmail(input.value);
      input.classList.toggle('is-invalid', bad);
      errEl.hidden = !bad;
      errEl.textContent = bad ? 'Enter a valid email address.' : '';
    };
    input.addEventListener('blur', showEmailError);
    if (input.value) showEmailError();
    wrap.appendChild(errEl);
  }

  return wrap;
}

// Phone number gets its own [country <select>] + [national number] pair
// instead of one free-text input — the dial code needs to be known both to
// compose a backend-valid mobile_number ("+<code><national>", see
// quote_service.py's _split_mobile_number) and to validate the number's
// length against what that country's numbers actually look like.
// A plain text input + filtered dropdown list — a native <select> has no
// search/type-ahead beyond jumping to the first matching letter, which
// doesn't scale to ~70 countries. `names` is the full option list; typing
// filters it by substring (case-insensitive); Up/Down move a highlighted
// row, Enter/click selects it, Escape or losing focus closes the list.
// Reverts to the last valid selection on blur if what's typed doesn't
// exactly match an option, so the field can never end up "selected" on
// text that isn't actually a real country.
function buildCountrySearchCombobox(names, selectedName, labelFor, onSelect) {
  const wrap = document.createElement('div');
  wrap.className = 'country-search';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'country-search__input';
  input.placeholder = 'Search country…';
  input.autocomplete = 'off';
  input.value = selectedName ? labelFor(selectedName) : '';

  const list = document.createElement('div');
  list.className = 'country-search__list';
  list.hidden = true;

  let filtered = names;
  let activeIndex = -1;

  function renderList() {
    list.innerHTML = '';
    if (filtered.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'country-search__empty';
      empty.textContent = 'No matching country';
      list.appendChild(empty);
      return;
    }
    filtered.forEach((name, i) => {
      const item = document.createElement('div');
      item.className = 'country-search__item' + (i === activeIndex ? ' is-active' : '');
      item.textContent = labelFor(name);
      // mousedown (not click) + preventDefault so this fires BEFORE the
      // input's own blur handler would otherwise close the list first.
      item.addEventListener('mousedown', (e) => {
        e.preventDefault();
        pick(name);
      });
      list.appendChild(item);
    });
  }

  function openList() {
    const query = input.value.trim().toLowerCase();
    filtered = query ? names.filter((n) => n.toLowerCase().includes(query)) : names;
    activeIndex = -1;
    renderList();
    list.hidden = false;
  }

  function pick(name) {
    input.value = labelFor(name);
    list.hidden = true;
    onSelect(name);
  }

  input.addEventListener('focus', openList);
  input.addEventListener('input', openList);
  input.addEventListener('keydown', (e) => {
    if (list.hidden && e.key !== 'Escape') openList();
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, filtered.length - 1);
      renderList();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      renderList();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filtered[activeIndex]) pick(filtered[activeIndex]);
    } else if (e.key === 'Escape') {
      list.hidden = true;
    }
  });
  input.addEventListener('blur', () => {
    list.hidden = true;
    const typedMatch = names.find((n) => labelFor(n).toLowerCase() === input.value.trim().toLowerCase());
    if (typedMatch) {
      if (typedMatch !== selectedName) pick(typedMatch);
    } else {
      input.value = selectedName ? labelFor(selectedName) : '';
    }
  });

  wrap.appendChild(input);
  wrap.appendChild(list);
  return wrap;
}

function buildAvaFormPhoneField(m) {
  const fs = m.formState;
  const wrap = document.createElement('div');
  wrap.className = 'ava-form__field ava-form__phone-field';
  const labelText = document.createElement('span');
  labelText.className = 'ava-form__label';
  labelText.textContent = 'Mobile number';
  wrap.appendChild(labelText);

  const row = document.createElement('div');
  row.className = 'ava-form__phone-row';

  const syncMobileNumber = () => {
    const entry = COUNTRY_DIAL_CODES[fs.mobileCountry];
    fs.values.mobile_number = entry ? `+${entry.code}${fs.mobileNational}` : fs.mobileNational;
    saveState();
  };
  const showPhoneError = () => {
    const bad = fs.mobileNational && fs.mobileCountry && !isValidPhoneForCountry(fs.mobileNational, fs.mobileCountry);
    numberInput.classList.toggle('is-invalid', bad);
    errEl.hidden = !bad;
    errEl.textContent = bad ? `Doesn't look like a valid ${fs.mobileCountry} number.` : '';
  };

  const countryNames = Object.keys(COUNTRY_DIAL_CODES).sort();
  const countryCombobox = buildCountrySearchCombobox(
    countryNames,
    fs.mobileCountry,
    (name) => `${name} (+${COUNTRY_DIAL_CODES[name].code})`,
    (name) => {
      fs.mobileCountry = name;
      syncMobileNumber();
      showPhoneError();
    },
  );

  const numberInput = document.createElement('input');
  numberInput.type = 'tel';
  numberInput.placeholder = 'e.g. 501234567';
  numberInput.value = fs.mobileNational || '';

  const errEl = document.createElement('span');
  errEl.className = 'ava-form__field-error';
  errEl.hidden = true;

  numberInput.addEventListener('input', () => {
    fs.mobileNational = numberInput.value.replace(/[^\d]/g, '');
    numberInput.value = fs.mobileNational;
    syncMobileNumber();
  });
  numberInput.addEventListener('blur', showPhoneError);
  if (fs.mobileNational) showPhoneError();

  row.appendChild(countryCombobox);
  row.appendChild(numberInput);
  wrap.appendChild(row);
  wrap.appendChild(errEl);
  return wrap;
}

function buildAvaFormDropdown(m, field) {
  const fs = m.formState;
  const wrap = document.createElement('label');
  wrap.className = 'ava-form__field';
  const labelText = document.createElement('span');
  labelText.className = 'ava-form__label';
  labelText.textContent = AVA_FIELD_LABELS[field];
  const select = document.createElement('select');
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = 'Select…';
  select.appendChild(blank);
  for (const opt of fs.fieldOptions[field] || []) {
    const optEl = document.createElement('option');
    optEl.value = opt;
    optEl.textContent = opt;
    if (fs.values[field] === opt) optEl.selected = true;
    select.appendChild(optEl);
  }
  select.addEventListener('change', () => {
    fs.values[field] = select.value;
    if (field === 'coverage_type') {
      // e.g. "UAE Inbound" always means destination="United Arab Emirates" —
      // fill it in now rather than let the user type/pick something the
      // backend would just reject (see COVERAGE_TYPE_LOCKS server-side).
      const locked = fs.coverageTypeLocks[select.value] || {};
      for (const [lockedField, value] of Object.entries(locked)) {
        fs.values[lockedField] = value;
      }
    }
    saveState();
    // cover_type changes what's rendered (the companion section);
    // coverage_type changes whether destination/departure are locked —
    // both need a structural re-render. The other dropdown (plan_type)
    // doesn't affect anything else shown.
    if (field === 'cover_type' || field === 'coverage_type') rerenderAvaFormCard(m);
  });
  wrap.appendChild(labelText);
  wrap.appendChild(select);
  return wrap;
}

function buildAvaCompanionSection(m) {
  const fs = m.formState;
  const section = document.createElement('div');
  section.className = 'ava-form__companions';

  const heading = document.createElement('div');
  heading.className = 'ava-form__label';
  heading.textContent = `Travelling with others? Add at least 1 companion (${fs.values.cover_type} cover needs 2+ travellers total).`;
  section.appendChild(heading);

  fs.additionalTravellers.forEach((t, idx) => {
    const row = document.createElement('div');
    row.className = 'ava-form__companion-row';

    const first = document.createElement('input');
    first.type = 'text';
    first.placeholder = 'First name';
    first.value = t.first_name || '';
    first.addEventListener('input', () => { t.first_name = first.value; saveState(); });

    const last = document.createElement('input');
    last.type = 'text';
    last.placeholder = 'Last name';
    last.value = t.last_name || '';
    last.addEventListener('input', () => { t.last_name = last.value; saveState(); });

    const dob = document.createElement('input');
    dob.type = 'date';
    dob.value = t.date_of_birth || '';
    dob.addEventListener('input', () => { t.date_of_birth = dob.value; saveState(); });

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'ava-form__companion-remove';
    removeBtn.textContent = '✕';
    removeBtn.setAttribute('aria-label', 'Remove companion');
    removeBtn.addEventListener('click', () => {
      fs.additionalTravellers.splice(idx, 1);
      saveState();
      rerenderAvaFormCard(m);
    });

    row.appendChild(first);
    row.appendChild(last);
    row.appendChild(dob);
    row.appendChild(removeBtn);
    section.appendChild(row);
  });

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'ava-form__companion-add';
  addBtn.textContent = '+ Add companion';
  addBtn.addEventListener('click', () => {
    fs.additionalTravellers.push({ first_name: '', last_name: '', date_of_birth: '' });
    saveState();
    rerenderAvaFormCard(m);
  });
  section.appendChild(addBtn);

  return section;
}

async function handleAvaFormUpload(m, file) {
  const fs = m.formState;
  fs.uploadStatus = `Reading ${file.name}…`;
  rerenderAvaFormCard(m);

  const formData = new FormData();
  formData.append('session_id', state.sessionId);
  formData.append('file', file);

  try {
    const res = await fetch(`${API.baseUrl}/api/chat/upload-document`, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');

    // Only fill fields still empty — never clobber something the user
    // already typed, matching the backend's own merge_extracted_fields
    // "never overwrite" rule.
    const collected = data.collected_fields || {};
    let filledCount = 0;
    for (const f of AVA_FORM_FIELDS) {
      if (!fs.values[f] && collected[f]) {
        fs.values[f] = collected[f];
        filledCount += 1;
      }
    }

    // mobile_number lands as one string, but the phone widget renders
    // from a country+national split — without this, a correctly extracted
    // number still shows an empty, unfilled-looking country picker.
    if (!fs.mobileCountry && !fs.mobileNational && fs.values.mobile_number) {
      const decomposed = decomposeMobileNumber(fs.values.mobile_number);
      if (decomposed) {
        fs.mobileCountry = decomposed.country;
        fs.mobileNational = decomposed.national;
      }
    }

    fs.uploadStatus = filledCount > 0 ? `✓ Pre-filled ${filledCount} field(s) from ${file.name}` : `Uploaded ${file.name} — nothing new to pre-fill`;
  } catch (err) {
    fs.uploadStatus = `Couldn't read ${file.name} — fill in your details manually.`;
  }
  saveState();
  rerenderAvaFormCard(m);
}

async function submitAvaForm(m) {
  const fs = m.formState;

  // Same checks the two inline field validators run on blur — repeated here
  // so a user who never blurred a field (e.g. tabbed straight to submit)
  // can't skip them, and so there's one clear message instead of two silent
  // inline errors they might not scroll back up to see.
  if (fs.values.email && !isValidEmail(fs.values.email)) {
    fs.error = 'Please enter a valid email address.';
    rerenderAvaFormCard(m);
    return;
  }
  if (!fs.mobileCountry || !fs.mobileNational) {
    fs.error = 'Please select your mobile number\'s country and enter the number.';
    rerenderAvaFormCard(m);
    return;
  }
  if (!isValidPhoneForCountry(fs.mobileNational, fs.mobileCountry)) {
    fs.error = `That doesn't look like a valid ${fs.mobileCountry} mobile number.`;
    rerenderAvaFormCard(m);
    return;
  }

  fs.submitting = true;
  fs.error = null;
  rerenderAvaFormCard(m);

  const payload = {
    session_id: state.sessionId,
    ...fs.values,
    additional_travellers: fs.additionalTravellers.filter((t) => t.first_name && t.date_of_birth),
    marketing_consent: !!fs.marketingConsent,
  };

  try {
    const res = await fetch(`${API.baseUrl}/api/chat/submit-intake-form`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    fs.submitting = false;

    if (!res.ok) {
      const detail = data.detail;
      if (detail && typeof detail === 'object') {
        if (detail.error === 'missing_fields') {
          fs.error = `Please fill in: ${detail.fields.map((f) => AVA_FIELD_LABELS[f] || f).join(', ')}.`;
        } else if (detail.message) {
          fs.error = detail.message;
        } else {
          fs.error = 'Something didn\'t validate — please check your details.';
        }
      } else {
        fs.error = (typeof detail === 'string' && detail) || 'Something went wrong submitting your details.';
      }
      saveState();
      rerenderAvaFormCard(m);
      return;
    }

    const wasEditing = fs.editing;
    fs.submitted = true;
    fs.editing = false;
    fs.error = null;
    saveState();
    rerenderAvaFormCard(m);

    // An edit corrects a mistake — it should update the SAME quote message
    // in place, not leave the old (wrong) quote sitting above a new one.
    // m.quoteMessageId tracks which message that is; only set on an edit
    // (first-ever submit always creates a fresh message).
    const existingQuoteMsg = wasEditing && m.quoteMessageId
      ? state.messages.find((x) => x.id === m.quoteMessageId)
      : null;

    if (existingQuoteMsg) {
      existingQuoteMsg.content = data.response || '';
      existingQuoteMsg.isQuoteList = !!(data.ui && data.ui.type === 'quote_list');
      existingQuoteMsg.timestamp = Date.now();
      // The quote just changed — any email-button state from the old one
      // (sent/open/draft) no longer matches what's on screen.
      existingQuoteMsg.emailSentTo = null;
      existingQuoteMsg.emailRowOpen = false;
      existingQuoteMsg.emailDraft = null;
    } else {
      const quoteMsg = msg('assistant', data.response || '', Date.now());
      quoteMsg.isQuoteList = !!(data.ui && data.ui.type === 'quote_list');
      state.messages.push(quoteMsg);
      m.quoteMessageId = quoteMsg.id;
    }
    saveState();
    renderThread();
    if (!userScrolledAway) scrollToBottom();
  } catch (err) {
    fs.submitting = false;
    fs.error = 'Network error submitting your details — please try again.';
    saveState();
    rerenderAvaFormCard(m);
  }
}

function buildMessageControls(m) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-controls';
  wrap.innerHTML = `
    <button type="button" class="ctrl-copy" aria-label="Copy response" title="Copy">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><rect x="8" y="8" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.7"/><path d="M4 16V6C4 4.9 4.9 4 6 4H16" stroke="currentColor" stroke-width="1.7"/></svg>
    </button>
    <button type="button" class="ctrl-up" aria-label="Good response" aria-pressed="false" title="Good response">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M7 11V21H4V11H7ZM7 11L11 3C12.1 3 13 3.9 13 5V9H18.3C19.5 9 20.4 10.1 20.1 11.3L18.6 18.3C18.4 19.3 17.5 20 16.5 20H10C8.3 20 7 18.7 7 17" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>
    </button>
    <button type="button" class="ctrl-down" aria-label="Poor response" aria-pressed="false" title="Poor response">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" style="transform:rotate(180deg)"><path d="M7 11V21H4V11H7ZM7 11L11 3C12.1 3 13 3.9 13 5V9H18.3C19.5 9 20.4 10.1 20.1 11.3L18.6 18.3C18.4 19.3 17.5 20 16.5 20H10C8.3 20 7 18.7 7 17" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>
    </button>
    <button type="button" class="ctrl-regen" aria-label="Regenerate response" title="Regenerate">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M20 11A8 8 0 1 0 18.5 15.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><path d="M20 5V11H14" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
  `;

  wrap.querySelector('.ctrl-copy').addEventListener('click', (e) => copyMessage(m, e.currentTarget));
  wrap.querySelector('.ctrl-up').addEventListener('click', (e) => setFeedback(m, 'up', e.currentTarget));
  wrap.querySelector('.ctrl-down').addEventListener('click', (e) => setFeedback(m, 'down', e.currentTarget));
  wrap.querySelector('.ctrl-regen').addEventListener('click', () => regenerateResponse(m));
  return wrap;
}

function copyMessage(m, btn) {
  const plain = m.content.replace(/[*#]/g, '');
  navigator.clipboard?.writeText(plain).then(() => {
    const original = btn.innerHTML;
    btn.classList.add('copied-flash');
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M5 12L10 17L19 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    announce('Response copied to clipboard.');
    setTimeout(() => { btn.innerHTML = original; btn.classList.remove('copied-flash'); }, 1400);
  });
}

function setFeedback(m, value, btn) {
  const msgEl = btn.closest('.msg');
  const controls = btn.closest('.msg-controls');
  m.feedback = m.feedback === value ? null : value;

  controls.querySelector('.ctrl-up').classList.toggle('is-active up', m.feedback === 'up');
  controls.querySelector('.ctrl-up').setAttribute('aria-pressed', String(m.feedback === 'up'));
  controls.querySelector('.ctrl-down').classList.toggle('is-active down', m.feedback === 'down');
  controls.querySelector('.ctrl-down').setAttribute('aria-pressed', String(m.feedback === 'down'));

  msgEl.querySelector('.feedback-chips')?.remove();
  msgEl.querySelector('.feedback-thanks')?.remove();

  if (m.feedback === 'down') {
    const chips = document.createElement('div');
    chips.className = 'feedback-chips';
    chips.innerHTML = ['Inaccurate', 'Missing sources', 'Hard to understand', 'Other']
      .map((label) => `<button type="button" class="feedback-chip">${label}</button>`).join('');
    chips.addEventListener('click', (e) => {
      const chip = e.target.closest('.feedback-chip');
      if (!chip) return;
      chips.remove();
      const thanks = document.createElement('p');
      thanks.className = 'feedback-thanks';
      thanks.textContent = 'Thanks — noted for review.';
      controls.insertAdjacentElement('afterend', thanks);
      announce('Feedback submitted.');
    });
    controls.insertAdjacentElement('afterend', chips);
  } else if (m.feedback === 'up') {
    announce('Marked as a good response.');
  }
  saveState();
}

async function regenerateResponse(assistantMsg) {
  const idx = state.messages.findIndex((mm) => mm.id === assistantMsg.id);
  const priorUser = [...state.messages.slice(0, idx)].reverse().find((mm) => mm.role === 'user');
  if (!priorUser) return;

  assistantMsg.content = '';
  assistantMsg.feedback = null;
  renderThread();
  await streamInto(assistantMsg, priorUser.content);
}

/* =========================================================================
   8. STREAMING
   ========================================================================= */

async function sendMessage(text) {
  if (pendingResetPromise) await pendingResetPromise;

  const value = (text ?? el.composerInput.value).trim();
  if (!value) return;

  const userMsg = msg('user', value, Date.now());
  state.messages.push(userMsg);

  el.composerInput.value = '';
  autoResizeTextarea();
  renderThread();
  saveState();

  // In human mode, a live agent is handling this conversation — send
  // straight over the WebSocket instead of /ask-stream. The agent's reply
  // comes back through the polling loop (see section 8b), not here.
  if (chatMode === 'human' && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'message', content: value }));
    return;
  }

  const assistantMsg = msg('assistant', '', Date.now());
  state.messages.push(assistantMsg);
  renderThread();

  await streamInto(assistantMsg, value);
}

async function streamInto(assistantMsg, queryText) {
  setComposerLoading(true);
  activeAbortController = new AbortController();

  const msgEl = () => document.querySelector(`.msg[data-id="${assistantMsg.id}"]`);
  let bubble = msgEl()?.querySelector('.msg__bubble');
  showTypingIndicator(bubble);

  let firstChunkArrived = false;
  let raw = '';

  try {
    for await (const evt of API.streamAsk(queryText, state.sessionId, activeAbortController.signal)) {
      if (evt.type === 'chunk') {
        if (!firstChunkArrived) {
          firstChunkArrived = true;
          bubble = msgEl()?.querySelector('.msg__bubble');
          if (bubble) bubble.innerHTML = '';
        }
        raw += evt.text;
        assistantMsg.content = raw;
        if (bubble) {
          bubble.innerHTML = renderMarkdown(raw) + '<span class="streaming-cursor" aria-hidden="true"></span>';
        }
        if (!userScrolledAway) scrollToBottom();
      } else if (evt.type === 'done') {
        if (evt.correctedText) raw = evt.correctedText;
        if (evt.needsHuman) requestHandoff();
        if (evt.ui && evt.ui.type === 'ava_form') {
          state.messages.push(buildAvaFormMessage(evt.ui, evt.collectedFields));
        }
        // Ava's quote-comparison reply ("compare 1 and 2") — the chat bubble
        // has no markdown table renderer (a plain-text pipe table used to
        // show up as literal '|'/'---' characters), so the backend sends
        // the comparison as structured data instead of text; attach it to
        // THIS message (not a new one, unlike ava_form) so buildMessageEl
        // can render a real <table> below the "Comparing X and Y:" intro.
        // Only travel_bot/routers/chat.py ever sends this ui.type, so this
        // is unreachable from Layla's plain RAG replies.
        if (evt.ui && evt.ui.type === 'comparison_table') {
          assistantMsg.comparisonTable = evt.ui.table;
        }
        // Ava just showed a quote list (initial fetch or a sort-by re-list)
        // — same ui-marker mechanism as comparison_table, just a plain
        // marker with no payload since the "email me this quote" button
        // re-reads the quotes from session state server-side rather than
        // trusting whatever's in this message.
        if (evt.ui && evt.ui.type === 'quote_list') {
          assistantMsg.isQuoteList = true;
        }
        if (evt.offlineEscalated) {
          // Was pushed as role 'assistant' — rendered as a second full AI
          // bubble (avatar, timestamp, copy/feedback controls) right under
          // the actual answer, which read as two separate replies to one
          // question. This is a status notice, not a reply — same category
          // as the "You're now connected with an agent" / "no agents
          // available" notices the WS-polling path already renders via
          // pushSystemMessage() (see section 8b) as a centered pill with no
          // avatar. Reuse that exact path for consistency.
          pushSystemMessage('No agents are available right now. Your question has been emailed to our support team and someone will reach out to you soon.');
        }
      }
    }

    assistantMsg.content = raw;
    renderThread();
    if (!userScrolledAway) scrollToBottom();
  } catch (err) {
    if (err.name === 'AbortError') return;
    renderErrorInto(assistantMsg, queryText, err);
  } finally {
    setComposerLoading(false);
    activeAbortController = null;
    saveState();
  }
}

function showTypingIndicator(bubble) {
  if (!bubble) return;
  bubble.innerHTML = '<span class="typing-dots" aria-label="Assistant is composing a response"><span></span><span></span><span></span></span>';
}

function renderErrorInto(assistantMsg, queryText, err) {
  state.messages = state.messages.filter((m) => m.id !== assistantMsg.id);
  renderThread();

  const inner = el.thread.querySelector('.thread-inner') || el.thread;
  const card = document.createElement('div');
  card.className = 'error-card';
  card.innerHTML = `
    <span class="error-card__icon" aria-hidden="true">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M12 8V13" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="16.3" r="1" fill="currentColor"/></svg>
    </span>
    <div class="error-card__body">
      <p class="error-card__title">Couldn't reach the assistant</p>
      <p class="error-card__desc">${escapeHtml(err.message || 'The request failed before a response came back.')} Check that the backend is running, then try again.</p>
      <button type="button" class="retry-btn">Retry</button>
    </div>
  `;
  card.querySelector('.retry-btn').addEventListener('click', () => {
    card.remove();
    const assistantMsg2 = msg('assistant', '', Date.now());
    state.messages.push(assistantMsg2);
    renderThread();
    streamInto(assistantMsg2, queryText);
  });
  inner.appendChild(card);
  scrollToBottom(true);
}

function setComposerLoading(loading) {
  el.sendBtn.classList.toggle('is-loading', loading);
  el.sendBtn.disabled = loading;
}

/* =========================================================================
   8b. HUMAN AGENT HANDOFF
   Mirrors the proven React client's architecture (frontend/src/App.tsx):
   a WebSocket carries out-of-band signals only (agent_joined,
   handoff_timeout, etc.) while a 1s polling loop owns actual message
   delivery. Confirmed live this frontend previously had NEITHER — an
   agent's reply was visible in the agent dashboard (backend-side) but had
   no delivery path to the user's browser at all.
   ========================================================================= */

function connectAgentChannel() {
  if (state.sessionId == null || MOCK_MODE) return;

  // Sync pollSeen to the backend's current total BEFORE opening the
  // interval, so a returning visitor doesn't replay messages already
  // shown from localStorage.
  API.pollSession(state.sessionId, 0).then((data) => {
    if (data && typeof data.total === 'number') pollSeen = data.total;
  }).finally(() => {
    openAgentSocket();
    startPolling();
  });
}

function openAgentSocket() {
  try {
    const socket = new WebSocket(`${API.wsUrl()}/ws/user/${state.sessionId}`);
    ws = socket;
    socket.onmessage = handleWsMessage;
    socket.onclose = () => {
      if (ws === socket) ws = null;
      // Reconnect once after 3s so the backend can deliver anything it
      // buffered while we were disconnected (session.pending_ws_message).
      setTimeout(() => {
        if (!ws && state.sessionId) openAgentSocket();
      }, 3000);
    };
  } catch {
    // WebSocket unavailable entirely (e.g. blocked by a host page's CSP) —
    // polling alone still delivers messages and status transitions.
  }
}

let pollInFlight = false;

async function pollNow() {
  // Guards against overlapping calls (the 1s interval tick racing a
  // WS-triggered nudge) reading the same pollSeen cursor and rendering
  // the same message twice.
  if (pollInFlight || state.sessionId == null) return;
  pollInFlight = true;
  try {
    const data = await API.pollSession(state.sessionId, pollSeen);
    if (data) applyPollResult(data);
  } finally {
    pollInFlight = false;
  }
}

function startPolling() {
  if (pollIntervalId) return;
  pollIntervalId = setInterval(pollNow, 1000);
}

function applyPollResult(data) {
  // Sync chatMode with server status — the source of truth.
  if (data.status === 'human' && chatMode !== 'human') {
    chatMode = 'human';
    agentName = data.agent_name || 'Agent';
    pushSystemMessage(`You're now connected with ${agentName}. They can see your conversation and will help you directly.`);
  }
  if (data.status === 'waiting' && chatMode === 'ai') {
    chatMode = 'waiting';
  }
  if (data.status === 'ai' && chatMode !== 'ai') {
    const wasWaiting = chatMode === 'waiting';
    const wasHuman = chatMode === 'human';
    chatMode = 'ai';
    agentName = null;
    if (wasWaiting) {
      pushSystemMessage('No agents are available right now. Your question has been emailed to our support team — someone will reach out to you soon.');
    } else if (wasHuman) {
      // Fallback for when the WS "agent_left" event was missed (socket
      // disconnected at the exact moment the agent released/reassigned) —
      // polling is the source of truth here, same as the wasWaiting case
      // above, so this can never double-fire with the WS-delivered message
      // (that path already flips chatMode to 'ai' before the next poll tick).
      pushSystemMessage('The agent has stepped away. Layla is back to help!');
    }
  }

  let delivered = false;
  for (const m of data.messages || []) {
    if (m.role === 'agent') {
      state.messages.push({
        ...msg('agent', m.content, Date.parse(m.timestamp) || Date.now()),
        agentName: data.agent_name || 'Agent',
        answersQuestion: m.answers_question || null,
      });
      delivered = true;
    }
  }
  if (delivered) {
    renderThread();
    if (!userScrolledAway) scrollToBottom();
    saveState();
  }
  if (typeof data.total === 'number') pollSeen = data.total;
}

function pushSystemMessage(text) {
  state.messages.push(msg('system', text, Date.now()));
  renderThread();
  if (!userScrolledAway) scrollToBottom();
  saveState();
}

function handleWsMessage(ev) {
  let payload;
  try { payload = JSON.parse(ev.data); } catch { return; }
  if (payload.type === 'agent_message') {
    // The reply itself is still rendered ONLY by the poll response
    // (pollNow's pollSeen cursor is the single source of truth, so the
    // same reply can never be shown twice) — this just wakes polling up
    // immediately instead of waiting for the next 1s interval tick, which
    // otherwise can stall up to ~60s if the tab is backgrounded (browsers
    // throttle setInterval there, but WS message delivery isn't throttled
    // the same way).
    pollNow();
    return;
  }
  if (payload.type === 'agent_joined') {
    if (chatMode !== 'human') {
      chatMode = 'human';
      agentName = payload.agent_name;
      pushSystemMessage(`You're now connected with ${agentName}. They can see your conversation and will help you directly.`);
    }
  } else if (payload.type === 'agent_left') {
    chatMode = 'ai';
    agentName = null;
    pushSystemMessage(payload.message || "You're back with Layla.");
  } else if (payload.type === 'waiting') {
    pushSystemMessage(payload.message);
  } else if (payload.type === 'handoff_timeout') {
    chatMode = 'ai';
    agentName = null;
    pushSystemMessage(payload.message || 'No agents available. Our team has been notified by email.');
  }
}

function requestHandoff() {
  // Guard: only trigger once per AI session — don't re-trigger if already
  // waiting for an agent or already in a live human session.
  if (chatMode !== 'ai') return;
  chatMode = 'waiting';
  pushSystemMessage("I'm finding a human agent who can help you better. One moment…");
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'request_handoff' }));
  } else {
    API.requestHandoff(state.sessionId);
  }
}

function cancelHandoff() {
  if (chatMode !== 'waiting') return;
  chatMode = 'ai';
  if (ws && ws.readyState === WebSocket.OPEN) {
    // WS reply (type: "handoff_timeout") delivers the confirmation message.
    ws.send(JSON.stringify({ type: 'cancel_handoff' }));
  } else {
    // No WS to deliver the server's reply — push the confirmation locally
    // so Cancel doesn't look unresponsive while the next poll tick catches up.
    API.cancelHandoff(state.sessionId);
    pushSystemMessage("No problem! I've sent your question to our support team — someone will follow up with you soon. You can keep chatting with me in the meantime! 😊");
  }
}

/* =========================================================================
   9. SCROLL BEHAVIOR
   ========================================================================= */

function scrollToBottom(instant) {
  el.thread.scrollTo({ top: el.thread.scrollHeight, behavior: instant ? 'auto' : 'smooth' });
  userScrolledAway = false;
  el.jumpLatestBtn.hidden = true;
}

function handleThreadScroll() {
  const distanceFromBottom = el.thread.scrollHeight - el.thread.scrollTop - el.thread.clientHeight;
  userScrolledAway = distanceFromBottom > 80;
  el.jumpLatestBtn.hidden = !userScrolledAway;
}

/* =========================================================================
   10. COMPOSER
   ========================================================================= */

function autoResizeTextarea() {
  el.composerInput.style.height = 'auto';
  el.composerInput.style.height = `${Math.min(el.composerInput.scrollHeight, 160)}px`;
}

/* =========================================================================
   11. CLEAR CHAT
   ========================================================================= */

// session_id stays fixed across a clear — deliberate, explicit requirement:
// the human-agent dashboard and super-admin should keep seeing ONE
// continuous session per user, not a new dashboard entry every time
// someone clicks Clear chat. Only the VISIBLE thread is erased client-side;
// nothing is deleted server-side — API.resetConversation() resets
// server-side state on that SAME session_id instead (RAG conversational
// memory, and agent_router's active_agent back to "layla" so a user who
// was mid-conversation with a specialist agent, e.g. Ava, doesn't stay
// silently stuck talking to it after "clearing"), and appends a "— New
// Chat started —" marker so admins can see exactly where the reset
// happened in the still-intact full transcript.
//
// pendingResetPromise: sendMessage() awaits this before doing anything, so
// a message sent right after clearing (paste + enter, or just typing
// fast) can never reach the backend before the reset does — see that
// variable's own comment for the race this closes.
function clearChat() {
  activeAbortController?.abort();
  state.messages = [];
  saveState();
  renderThread();
  announce('Chat cleared.');
  el.composerInput.focus();
  pendingResetPromise = API.resetConversation(state.sessionId).finally(() => {
    pendingResetPromise = null;
  });
}

/* =========================================================================
   12. STATUS / HEALTH CHECK
   ========================================================================= */

async function refreshStatus() {
  el.statusDot.className = 'status-dot checking';
  el.statusText.textContent = 'Checking…';
  const ok = await API.checkHealth();
  el.statusDot.className = `status-dot ${ok ? 'online' : 'offline'}`;
  el.statusText.textContent = MOCK_MODE ? 'Demo mode' : (ok ? 'Connected' : 'Offline');
}

/* =========================================================================
   13. MISC HELPERS
   ========================================================================= */

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function announce(text) {
  el.srAnnouncer.textContent = '';
  requestAnimationFrame(() => { el.srAnnouncer.textContent = text; });
}

/* =========================================================================
   14. EVENT WIRING
   ========================================================================= */

function wireEvents() {
  el.composerForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
  });

  el.composerInput.addEventListener('input', autoResizeTextarea);
  el.composerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  el.clearChatBtn.addEventListener('click', clearChat);

  el.thread.addEventListener('scroll', handleThreadScroll);
  el.jumpLatestBtn.addEventListener('click', () => scrollToBottom());

  window.addEventListener('beforeunload', saveState);
}

/* =========================================================================
   15. INIT
   ========================================================================= */

function init() {
  loadState();
  renderThread();
  autoResizeTextarea();
  wireEvents();
  refreshStatus();
  connectAgentChannel();

  setInterval(refreshStatus, 30000);
}

document.addEventListener('DOMContentLoaded', init);
