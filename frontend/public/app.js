/* =========================================================================
   Layla — InsureHub production chat frontend — app.js
   Single persistent chat window, no sidebar, no multi-conversation
   switching — matches the real production Layla product (confirmed
   against the prior React client, frontend/src/App.tsx, which never had
   a conversation-history sidebar either). "Clear chat" empties the
   visible thread only; session_id is generated once and never rotates,
   so backend-side conversation context (multi_source_rag.py's
   _get_conversation_history(session_id)) is untouched by a clear.
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
  avatar.textContent = m.role === 'user' ? 'You' : m.role === 'agent' ? (m.agentName || 'A')[0].toUpperCase() : 'L';

  const col = document.createElement('div');
  col.className = 'msg__col';

  const bubble = document.createElement('div');
  bubble.className = 'msg__bubble';
  bubble.innerHTML = (m.role === 'assistant' || m.role === 'agent')
    ? renderMarkdown(m.content)
    : escapeHtml(m.content);

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

function startPolling() {
  if (pollIntervalId) return;
  pollIntervalId = setInterval(async () => {
    const data = await API.pollSession(state.sessionId, pollSeen);
    if (data) applyPollResult(data);
  }, 1000);
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
    chatMode = 'ai';
    agentName = null;
    if (wasWaiting) {
      pushSystemMessage('No agents are available right now. Your question has been emailed to our support team — someone will reach out to you soon.');
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
  // agent_message is intentionally NOT handled here — polling owns
  // message delivery (see this section's header comment) so the same
  // reply can never be shown twice.
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

// Empties the visible thread only — session_id is untouched, so this is
// NOT a new conversation from the backend's point of view (see file
// header). Matches the user's explicit "same session id, just cleared"
// requirement, not a "new chat window" model.
function clearChat() {
  activeAbortController?.abort();
  state.messages = [];
  saveState();
  renderThread();
  announce('Chat cleared.');
  el.composerInput.focus();
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
