/* =========================================================================
   InsureHub Assistant — app.js
   No framework, no build step. Organized top to bottom as:
   config → mock data → API module → markdown/citation rendering →
   state/persistence → DOM rendering → event wiring → init.
   ========================================================================= */

'use strict';

/* =========================================================================
   1. CONFIG
   ========================================================================= */

// Flip this to false once the FastAPI backend is reachable and you want
// this UI talking to it instead of canned demo data. Everything backend-
// related lives inside the API object below — this is the one line you
// change to swap from demo to real.
let MOCK_MODE = true;

const STORAGE_KEY = 'insurehub_ui_state_v1';

/* =========================================================================
   2. MOCK DATA
   ========================================================================= */

const SOURCE_TYPE_META = {
  document: { label: 'Document', color: 'var(--doc-color)' },
  video: { label: 'Video', color: 'var(--video-color)' },
  webpage: { label: 'Webpage', color: 'var(--web-color)' },
};

// A handful of canned Q&A pairs used when the user types a fresh question
// in mock mode. Picked by loose keyword match so the demo feels a little
// responsive rather than always returning the same text; falls back to a
// generic mixed-source answer for anything unmatched.
const MOCK_RESPONSES = [
  {
    keywords: ['deductible', 'premium'],
    content:
      "Good question — these two get mixed up a lot.\n\n" +
      "A **premium** is what you pay the insurer regularly (usually monthly or annually) just to keep the policy active {{cite:1}}. A **deductible** is different: it's the amount *you* pay out of pocket on a claim before your coverage kicks in {{cite:2}}.\n\n" +
      "- Higher deductible → lower premium, but more upfront cost if you claim\n" +
      "- Lower deductible → higher premium, but less upfront cost if you claim\n\n" +
      "Most policies let you choose your deductible tier when you set up or renew coverage {{cite:3}}.",
    sources: [
      { type: 'document', title: 'Homeowners Policy Wording, Section 4', excerpt: 'The premium is the consideration payable by the insured to the insurer for the coverage bound under this policy, payable in accordance with the schedule selected at inception.', page: 12, score: 0.91 },
      { type: 'document', title: 'Homeowners Policy Wording, Section 4', excerpt: 'The deductible represents the portion of any covered loss that remains the responsibility of the insured before indemnification under this policy applies.', page: 13, score: 0.88 },
      { type: 'webpage', title: 'Choosing the Right Deductible — InsureHub Learn', url: 'learn.insurehub.example/deductible-tiers', excerpt: 'Most carriers offer three or four deductible tiers at renewal, typically ranging from $250 to $2,500 depending on policy type.', score: 0.74 },
    ],
  },
  {
    keywords: ['water', 'damage', 'flood', 'leak'],
    content:
      "Here's the general process for a water damage claim:\n\n" +
      "1. **Stop further damage** if it's safe to do so — shut off the water source, move valuables\n" +
      "2. **Document everything** with photos and video before cleanup starts {{cite:1}}\n" +
      "3. **File the claim** through the app or your agent within the notice window in your policy {{cite:2}}\n" +
      "4. An adjuster will typically inspect within a few business days {{cite:1}}\n\n" +
      "One thing worth checking: **gradual leaks** (like a slow pipe leak over months) are often excluded, while **sudden and accidental** water damage usually is covered {{cite:3}}. Worth confirming which category your situation falls under before you file.",
    sources: [
      { type: 'document', title: 'Claims Handbook — Property Damage', excerpt: 'Policyholders should photograph and, where practicable, video the affected area prior to commencing any remediation or repair work.', page: 27, score: 0.93 },
      { type: 'document', title: 'Homeowners Policy Wording, Section 9', excerpt: 'Notice of loss must be provided to the insurer within a reasonable time, and in no event later than 60 days from the date of discovery.', page: 31, score: 0.85 },
      { type: 'video', title: 'Water Damage Claims: What\'s Actually Covered', excerpt: 'The big distinction adjusters look for is sudden versus gradual. A burst pipe on a Tuesday afternoon is very different, coverage-wise, from a slow leak that\'s been going on for six months.', timestamp: '3:41', score: 0.80 },
    ],
  },
  {
    keywords: ['rental', 'car', 'rent a car'],
    content:
      "It depends on which coverage you're asking about.\n\n" +
      "If you mean **renting a car and something happens to the rental itself**, that's usually only covered if you've added rental car coverage as an endorsement to your auto policy, or if your credit card provides it separately {{cite:1}}.\n\n" +
      "If you mean **driving a rental as a temporary replacement while your own car is being repaired**, most policies extend your existing liability and collision coverage to a rental automatically for a limited period {{cite:2}} — but it's worth double-checking your specific declarations page, since limits vary.",
    sources: [
      { type: 'webpage', title: 'Rental Car Coverage Explained — InsureHub Learn', url: 'learn.insurehub.example/rental-car-coverage', excerpt: 'Rental reimbursement and physical damage to a rental vehicle are typically separate, optional endorsements — not included by default on most standard auto policies.', score: 0.86 },
      { type: 'document', title: 'Auto Policy Wording, Section 6', excerpt: 'Coverage under this policy extends to a temporary substitute automobile while the insured\'s covered auto is out of service due to breakdown, repair, or loss, for up to 30 days.', page: 19, score: 0.82 },
    ],
  },
];

const MOCK_FALLBACK_RESPONSE = {
  content:
    "I can help with that. Based on what's in your policy documentation and our reference material:\n\n" +
    "Coverage details vary quite a bit by policy type and endorsements, so the honest answer is *it depends on your specific plan* {{cite:1}}. That said, the general principle carriers apply is that losses need to be **sudden, accidental, and named or unexcluded** under your policy to qualify {{cite:2}}.\n\n" +
    "If you can tell me a bit more about the specific situation, I can point to the exact clause that applies.",
  sources: [
    { type: 'document', title: 'General Policy Provisions', excerpt: 'Coverage terms, limits, and exclusions vary by product line and by the specific endorsements attached to an individual policy at binding.', page: 4, score: 0.71 },
    { type: 'webpage', title: 'How Claims Eligibility Works — InsureHub Learn', url: 'learn.insurehub.example/claims-eligibility', excerpt: 'Insurers generally distinguish between sudden and accidental losses (typically covered) and gradual or expected wear (typically excluded).', score: 0.68 },
  ],
};

function pickMockResponse(query) {
  const q = (query || '').toLowerCase();
  const found = MOCK_RESPONSES.find((r) => r.keywords.some((k) => q.includes(k)));
  return found || MOCK_FALLBACK_RESPONSE;
}

// Pre-seeded conversation history so the sidebar isn't empty on first load.
function buildSeedConversations() {
  const now = Date.now();
  return [
    {
      id: 'seed-1',
      title: 'Water damage claim process',
      createdAt: now - 1000 * 60 * 60 * 26,
      messages: [
        msg('user', 'How do I file a claim after water damage?', now - 1000 * 60 * 60 * 26),
        msg('assistant', MOCK_RESPONSES[1].content, now - 1000 * 60 * 60 * 26 + 4000, MOCK_RESPONSES[1].sources),
      ],
    },
    {
      id: 'seed-2',
      title: 'Deductible vs premium explained',
      createdAt: now - 1000 * 60 * 60 * 3,
      messages: [
        msg('user', "What's the difference between a deductible and a premium?", now - 1000 * 60 * 60 * 3),
        msg('assistant', MOCK_RESPONSES[0].content, now - 1000 * 60 * 60 * 3 + 4000, MOCK_RESPONSES[0].sources),
      ],
    },
    {
      id: 'seed-3',
      title: 'Rental car coverage question',
      createdAt: now - 1000 * 60 * 20,
      messages: [
        msg('user', 'Does my policy cover a rental car?', now - 1000 * 60 * 20),
        msg('assistant', MOCK_RESPONSES[2].content, now - 1000 * 60 * 20 + 4000, MOCK_RESPONSES[2].sources),
      ],
    },
  ];
}

function msg(role, content, timestamp, sources) {
  return {
    id: uid(),
    role,
    content,
    timestamp,
    sources: sources || null,
    feedback: null, // null | 'up' | 'down'
  };
}

/* =========================================================================
   3. API MODULE — the only place that knows about the backend
   ========================================================================= */

const API = {
  baseUrl: '', // same-origin; set to e.g. 'http://localhost:8501' if needed

  async checkHealth() {
    if (MOCK_MODE) return true;
    try {
      const res = await fetch(`${this.baseUrl}/health`, { method: 'GET' });
      return res.ok;
    } catch {
      return false;
    }
  },

  // Async generator yielding { type: 'chunk', text } | { type: 'sources', sources }.
  // Swapping MOCK_MODE = false routes every call through the real branch below —
  // that's the one spot to touch when the backend is ready.
  async *streamAsk(query, conversationId, signal) {
    if (MOCK_MODE) {
      yield* mockStreamAsk(query, signal);
      return;
    }

    // ================= REAL BACKEND CALL =================
    // Contract assumed: POST /ask { query, conversation_id } -> a streamed
    // response body of newline-delimited JSON, each line either
    // {"answer_chunk": "..."} or a final {"sources": [...]}. NDJSON rather
    // than SSE framing since it needs no extra parsing beyond a newline
    // split — easy to swap for EventSource here if the backend prefers SSE.
    const res = await fetch(`${this.baseUrl}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, conversation_id: conversationId }),
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
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let evt;
        try { evt = JSON.parse(line); } catch { continue; }
        if (evt.answer_chunk !== undefined) yield { type: 'chunk', text: evt.answer_chunk };
        else if (evt.sources) yield { type: 'sources', sources: evt.sources };
      }
    }
  },
};

async function* mockStreamAsk(query, signal) {
  const demo = pickMockResponse(query);
  // "Thinking" delay before the first token — the caller uses this gap to
  // show the typing indicator, then swaps to real content the instant the
  // first chunk arrives.
  await sleep(650 + Math.random() * 500, signal);
  // Split on whitespace boundaries but keep the whitespace tokens so a
  // simple join reproduces the original text exactly.
  const tokens = demo.content.split(/(\s+)/);
  for (const t of tokens) {
    if (signal?.aborted) return;
    yield { type: 'chunk', text: t };
    if (t.trim()) await sleep(16 + Math.random() * 34, signal);
  }
  yield { type: 'sources', sources: demo.sources };
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => { clearTimeout(t); reject(new DOMException('aborted', 'AbortError')); });
  });
}

/* =========================================================================
   4. MARKDOWN + CITATION RENDERING
   ========================================================================= */

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Deliberately small markdown subset — bold, inline code, bullet lists,
// h1-h3, paragraphs — matching what the assistant's prompt actually
// produces. HTML-escapes first so nothing retrieved from a document can
// inject markup into the page.
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
      html += `<p>${inlineMd(line)}</p>`; // numbered steps render as plain lines; kept simple on purpose
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

// {{cite:N}} tokens survive escapeHtml untouched (no special chars), so
// this runs as a final pass over the rendered HTML string, swapping each
// token for an interactive marker tied to sources[N-1]. Kept separate from
// renderMarkdown so citations are only "finalized" once — during live
// streaming we show the raw growing text and skip this pass entirely,
// running it once when the stream completes.
function injectCitations(html, sources, msgId) {
  if (!sources || !sources.length) return html;
  return html.replace(/\{\{cite:(\d+)\}\}/g, (_, n) => {
    const idx = parseInt(n, 10) - 1;
    const src = sources[idx];
    if (!src) return '';
    const meta = SOURCE_TYPE_META[src.type] || SOURCE_TYPE_META.document;
    const detail = sourceDetailText(src);
    return (
      `<button type="button" class="cite-marker" style="--type-color:${meta.color}" ` +
      `data-msg-id="${msgId}" data-cite-index="${idx}" aria-label="Source ${n}: ${escapeAttr(src.title)}">` +
      `${n}<span class="cite-tooltip" role="tooltip">` +
      `<span class="cite-tooltip__title">${escapeHtml(src.title)}</span>` +
      `${escapeHtml(truncate(src.excerpt, 110))}` +
      `</span></button>`
    );
  });
}

function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }
function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

function sourceDetailText(src) {
  if (src.type === 'document') return `Page ${src.page ?? '—'}`;
  if (src.type === 'video') return src.timestamp ?? '—';
  if (src.type === 'webpage') return domainFromUrl(src.url);
  return '';
}
function domainFromUrl(url) {
  if (!url) return '';
  try { return new URL(url.startsWith('http') ? url : `https://${url}`).hostname.replace(/^www\./, ''); }
  catch { return url; }
}

function sourceTypeIcon(type) {
  const icons = {
    document: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M7 3H14L19 8V19C19 20.1 18.1 21 17 21H7C5.9 21 5 20.1 5 19V5C5 3.9 5.9 3 7 3Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M14 3V8H19" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
    video: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M10 8.5L16 12L10 15.5V8.5Z" fill="currentColor"/></svg>',
    webpage: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M3 12H21M12 3C14.5 5.7 15.8 9 15.8 12C15.8 15 14.5 18.3 12 21C9.5 18.3 8.2 15 8.2 12C8.2 9 9.5 5.7 12 3Z" stroke="currentColor" stroke-width="1.6"/></svg>',
  };
  return icons[type] || icons.document;
}

/* =========================================================================
   5. STATE + PERSISTENCE
   ========================================================================= */

const state = {
  conversations: [],
  currentId: null,
  theme: null, // null = follow OS, else 'light' | 'dark'
  sidebarCollapsed: false,
};

function uid() { return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`; }

function loadState() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch { /* ignore corrupt state */ }

  if (saved && Array.isArray(saved.conversations) && saved.conversations.length) {
    state.conversations = saved.conversations;
    state.currentId = saved.currentId || saved.conversations[0].id;
    state.theme = saved.theme ?? null;
    state.sidebarCollapsed = !!saved.sidebarCollapsed;
  } else {
    state.conversations = buildSeedConversations();
    state.currentId = null; // land on the empty/welcome state on first visit
    state.theme = null;
    state.sidebarCollapsed = false;
  }
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      conversations: state.conversations,
      currentId: state.currentId,
      theme: state.theme,
      sidebarCollapsed: state.sidebarCollapsed,
    }));
  } catch { /* storage full/unavailable — non-fatal, app still works this session */ }
}

function currentConversation() {
  return state.conversations.find((c) => c.id === state.currentId) || null;
}

/* =========================================================================
   6. DOM REFS
   ========================================================================= */

const $ = (sel) => document.querySelector(sel);
const el = {
  app: $('#app'),
  sidebar: $('#sidebar'),
  scrim: $('#scrim'),
  historyList: $('#historyList'),
  historySearch: $('#historySearch'),
  newChatBtn: $('#newChatBtn'),
  sidebarCollapseBtn: $('#sidebarCollapseBtn'),
  menuBtn: $('#menuBtn'),
  conversationTitle: $('#conversationTitle'),
  statusDot: $('#statusDot'),
  statusText: $('#statusText'),
  themeToggleBtn: $('#themeToggleBtn'),
  thread: $('#thread'),
  jumpLatestBtn: $('#jumpLatestBtn'),
  composerForm: $('#composerForm'),
  composerInput: $('#composer-input'),
  sendBtn: $('#sendBtn'),
  srAnnouncer: $('#srAnnouncer'),
};

let userScrolledAway = false;
let activeAbortController = null;

/* =========================================================================
   7. RENDERING — sidebar
   ========================================================================= */

function renderSidebar(filterText) {
  const q = (filterText || '').trim().toLowerCase();
  const list = [...state.conversations]
    .sort((a, b) => b.createdAt - a.createdAt)
    .filter((c) => !q || c.title.toLowerCase().includes(q));

  el.historyList.innerHTML = '';

  if (!list.length) {
    const p = document.createElement('p');
    p.className = 'history-empty';
    p.textContent = q ? 'No conversations match.' : 'No conversations yet.';
    el.historyList.appendChild(p);
    return;
  }

  for (const conv of list) {
    const item = document.createElement('div');
    item.className = 'history-item' + (conv.id === state.currentId ? ' is-active' : '');
    item.dataset.id = conv.id;
    item.setAttribute('role', 'button');
    item.tabIndex = 0;
    item.innerHTML = `
      <span class="history-item__title">${escapeHtml(conv.title)}</span>
      <button type="button" class="icon-btn history-item__menu-btn" aria-label="Conversation options" aria-haspopup="true">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><circle cx="5" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="19" cy="12" r="1.6" fill="currentColor"/></svg>
      </button>
    `;
    el.historyList.appendChild(item);
  }
}

function openConversationMenu(anchorBtn, convId) {
  closeAnyOpenMenu();
  const menu = document.createElement('div');
  menu.className = 'history-item__menu';
  menu.innerHTML = `
    <button type="button" data-action="rename">Rename</button>
    <button type="button" data-action="delete" class="danger">Delete</button>
  `;
  anchorBtn.closest('.history-item').appendChild(menu);
  anchorBtn.classList.add('is-open');

  menu.addEventListener('click', (e) => {
    const action = e.target.closest('button')?.dataset.action;
    if (action === 'rename') startRenameConversation(convId);
    if (action === 'delete') deleteConversation(convId);
    closeAnyOpenMenu();
  });

  // Close on outside click / Escape
  setTimeout(() => {
    document.addEventListener('click', closeAnyOpenMenu, { once: true });
  }, 0);
}

function closeAnyOpenMenu() {
  document.querySelectorAll('.history-item__menu').forEach((m) => m.remove());
  document.querySelectorAll('.history-item__menu-btn.is-open').forEach((b) => b.classList.remove('is-open'));
}

function startRenameConversation(convId) {
  const conv = state.conversations.find((c) => c.id === convId);
  if (!conv) return;
  const item = el.historyList.querySelector(`.history-item[data-id="${convId}"]`);
  if (!item) return;
  const titleSpan = item.querySelector('.history-item__title');
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'history-item__title-input';
  input.value = conv.title;
  titleSpan.replaceWith(input);
  input.focus();
  input.select();

  const commit = () => {
    const val = input.value.trim();
    if (val) conv.title = val;
    saveState();
    renderSidebar(el.historySearch.value);
    if (convId === state.currentId) el.conversationTitle.textContent = conv.title;
  };
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    if (e.key === 'Escape') { renderSidebar(el.historySearch.value); }
  });
  input.addEventListener('blur', commit);
}

function deleteConversation(convId) {
  state.conversations = state.conversations.filter((c) => c.id !== convId);
  if (state.currentId === convId) {
    state.currentId = null;
  }
  saveState();
  renderSidebar(el.historySearch.value);
  renderThread();
  updateHeaderTitle();
  announce('Conversation deleted.');
}

function switchConversation(id) {
  state.currentId = id;
  saveState();
  renderSidebar(el.historySearch.value);
  renderThread();
  updateHeaderTitle();
  closeMobileSidebar();
}

function updateHeaderTitle() {
  const conv = currentConversation();
  el.conversationTitle.textContent = conv ? conv.title : 'New conversation';
}

/* =========================================================================
   8. RENDERING — thread
   ========================================================================= */

function renderThread() {
  const conv = currentConversation();
  el.thread.innerHTML = '';

  if (!conv || conv.messages.length === 0) {
    el.thread.appendChild(buildEmptyState());
    return;
  }

  const inner = document.createElement('div');
  inner.className = 'thread-inner';
  for (const m of conv.messages) {
    inner.appendChild(buildMessageEl(m));
  }
  el.thread.appendChild(inner);
  scrollToBottom(true);
}

const EXAMPLE_QUESTIONS = [
  "What's the difference between a deductible and a premium?",
  'How do I file a claim after water damage?',
  'Does my policy cover a rental car?',
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
    <h1>What can I help you with?</h1>
    <p>Ask about a policy, a claim, or coverage — every answer links back to exactly where it came from.</p>
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
  avatar.textContent = m.role === 'user' ? 'You' : 'IH';

  const col = document.createElement('div');
  col.className = 'msg__col';

  const meta = document.createElement('div');
  meta.className = 'msg__meta';
  meta.textContent = formatTime(m.timestamp);

  const bubble = document.createElement('div');
  bubble.className = 'msg__bubble';
  bubble.innerHTML = m.role === 'assistant'
    ? injectCitations(renderMarkdown(m.content), m.sources, m.id)
    : escapeHtml(m.content);

  col.appendChild(meta);
  col.appendChild(bubble);

  // Controls (especially Regenerate) are gated on non-empty content — while
  // a response is still streaming (content === ''), the finalize step in
  // streamInto() hasn't run yet, so a click here could start a second
  // concurrent stream into the same message and interleave garbled text.
  if (m.role === 'assistant' && m.content) {
    if (m.sources && m.sources.length) {
      col.appendChild(buildSourcesToggle(m));
      col.appendChild(buildSourcesPanel(m));
    }
    col.appendChild(buildMessageControls(m));
  }

  wrap.appendChild(avatar);
  wrap.appendChild(col);
  return wrap;
}

function buildSourcesToggle(m) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'sources-toggle';
  btn.setAttribute('aria-expanded', 'false');
  btn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    Show ${m.sources.length} source${m.sources.length === 1 ? '' : 's'}
  `;
  btn.addEventListener('click', () => toggleSourcesPanel(m.id));
  return btn;
}

function buildSourcesPanel(m) {
  const panel = document.createElement('div');
  panel.className = 'sources-panel';
  panel.id = `sources-${m.id}`;

  const innerWrap = document.createElement('div');
  innerWrap.className = 'sources-panel__inner';
  const grid = document.createElement('div');
  grid.className = 'sources-grid';

  m.sources.forEach((src, i) => grid.appendChild(buildSourceCard(src, i, m.id)));

  innerWrap.appendChild(grid);
  panel.appendChild(innerWrap);
  return panel;
}

function buildSourceCard(src, index, msgId) {
  const meta = SOURCE_TYPE_META[src.type] || SOURCE_TYPE_META.document;
  const pct = Math.round((src.score ?? 0) * 100);
  const card = document.createElement('div');
  card.className = 'source-card';
  card.style.setProperty('--type-color', meta.color);
  card.id = `source-${msgId}-${index}`;
  card.innerHTML = `
    <div class="source-card__head">
      <span class="source-card__icon" aria-hidden="true">${sourceTypeIcon(src.type)}</span>
      <span class="source-card__num">${index + 1}</span>
      <span class="source-card__title" title="${escapeAttr(src.title)}">${escapeHtml(src.title)}</span>
    </div>
    <p class="source-card__excerpt">${escapeHtml(src.excerpt)}</p>
    <div class="source-card__foot">
      <span class="source-card__detail">${escapeHtml(meta.label)} · ${escapeHtml(sourceDetailText(src))}</span>
      <span class="relevance" title="Relevance score">
        <span class="relevance__track"><span class="relevance__fill" style="width:${pct}%"></span></span>
        <span class="relevance__pct">${pct}%</span>
      </span>
    </div>
  `;
  return card;
}

function toggleSourcesPanel(msgId, forceOpen) {
  const panel = document.getElementById(`sources-${msgId}`);
  const btn = document.querySelector(`.msg[data-id="${msgId}"] .sources-toggle`);
  if (!panel || !btn) return;
  const open = forceOpen ?? !panel.classList.contains('is-open');
  panel.classList.toggle('is-open', open);
  btn.classList.toggle('is-open', open);
  btn.setAttribute('aria-expanded', String(open));
  const count = panel.querySelectorAll('.source-card').length;
  btn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    ${open ? 'Hide' : 'Show'} ${count} source${count === 1 ? '' : 's'}
  `;
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
  const plain = m.content.replace(/\{\{cite:\d+\}\}/g, '').replace(/[*#]/g, '');
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

  // Remove any existing chip row / thanks note first
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
  const conv = currentConversation();
  if (!conv) return;
  const idx = conv.messages.findIndex((mm) => mm.id === assistantMsg.id);
  const priorUser = [...conv.messages.slice(0, idx)].reverse().find((mm) => mm.role === 'user');
  if (!priorUser) return;

  // Reset this message's content and replay the stream into it in place.
  assistantMsg.content = '';
  assistantMsg.sources = null;
  assistantMsg.feedback = null;
  renderThread();
  await streamInto(assistantMsg, priorUser.content, conv);
}

/* =========================================================================
   9. STREAMING
   ========================================================================= */

async function sendMessage(text) {
  const value = (text ?? el.composerInput.value).trim();
  if (!value) return;

  let conv = currentConversation();
  if (!conv) {
    conv = { id: uid(), title: truncate(value, 44), createdAt: Date.now(), messages: [] };
    state.conversations.push(conv);
    state.currentId = conv.id;
  } else if (conv.messages.length === 0) {
    // First message in an existing-but-empty conversation still auto-names it.
    conv.title = truncate(value, 44);
  }

  const userMsg = msg('user', value, Date.now());
  conv.messages.push(userMsg);

  el.composerInput.value = '';
  autoResizeTextarea();
  updateHeaderTitle();
  renderSidebar(el.historySearch.value);
  renderThread();

  const assistantMsg = msg('assistant', '', Date.now(), null);
  conv.messages.push(assistantMsg);
  renderThread();

  await streamInto(assistantMsg, value, conv);
}

async function streamInto(assistantMsg, queryText, conv) {
  setComposerLoading(true);
  activeAbortController = new AbortController();

  const msgEl = () => document.querySelector(`.msg[data-id="${assistantMsg.id}"]`);
  let bubble = msgEl()?.querySelector('.msg__bubble');
  showTypingIndicator(bubble);

  let firstChunkArrived = false;
  let raw = '';

  try {
    for await (const evt of API.streamAsk(queryText, conv.id, activeAbortController.signal)) {
      if (evt.type === 'chunk') {
        if (!firstChunkArrived) {
          firstChunkArrived = true;
          bubble = msgEl()?.querySelector('.msg__bubble');
          if (bubble) bubble.innerHTML = '';
        }
        raw += evt.text;
        assistantMsg.content = raw;
        if (bubble) {
          // Re-render markdown from the growing buffer each chunk; cheap at
          // this text size, and keeps formatting correct mid-stream instead
          // of just appending a raw text node.
          bubble.innerHTML = renderMarkdown(raw) + '<span class="streaming-cursor" aria-hidden="true"></span>';
        }
        if (!userScrolledAway) scrollToBottom();
      } else if (evt.type === 'sources') {
        assistantMsg.sources = evt.sources;
      }
    }

    // Finalize: real citation markers + controls + sources panel.
    assistantMsg.content = raw;
    renderThread();
    if (!userScrolledAway) scrollToBottom();
  } catch (err) {
    if (err.name === 'AbortError') return;
    renderErrorInto(assistantMsg, conv, queryText, err);
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

function renderErrorInto(assistantMsg, conv, queryText, err) {
  const conv2 = currentConversation();
  conv2.messages = conv2.messages.filter((m) => m.id !== assistantMsg.id);
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
    const assistantMsg2 = msg('assistant', '', Date.now(), null);
    conv2.messages.push(assistantMsg2);
    renderThread();
    streamInto(assistantMsg2, queryText, conv2);
  });
  inner.appendChild(card);
  scrollToBottom(true);
}

function setComposerLoading(loading) {
  el.sendBtn.classList.toggle('is-loading', loading);
  el.sendBtn.disabled = loading;
}

/* =========================================================================
   10. SCROLL BEHAVIOR
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
   11. CITATION MARKER INTERACTION (event delegation)
   ========================================================================= */

function handleThreadClick(e) {
  const marker = e.target.closest('.cite-marker');
  if (!marker) return;
  const msgId = marker.dataset.msgId;
  const idx = marker.dataset.citeIndex;
  toggleSourcesPanel(msgId, true);
  const card = document.getElementById(`source-${msgId}-${idx}`);
  if (card) {
    setTimeout(() => {
      card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      card.classList.add('is-highlighted');
      setTimeout(() => card.classList.remove('is-highlighted'), 1600);
      announce(`Highlighted source ${parseInt(idx, 10) + 1}.`);
    }, 60);
  }
}

/* =========================================================================
   12. COMPOSER
   ========================================================================= */

function autoResizeTextarea() {
  el.composerInput.style.height = 'auto';
  el.composerInput.style.height = `${Math.min(el.composerInput.scrollHeight, 160)}px`;
}

/* =========================================================================
   13. THEME
   ========================================================================= */

function applyTheme() {
  if (state.theme === 'light' || state.theme === 'dark') {
    document.documentElement.setAttribute('data-theme', state.theme);
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  const isDark = state.theme === 'dark' || (state.theme === null && window.matchMedia('(prefers-color-scheme: dark)').matches);
  el.themeToggleBtn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
}

function toggleTheme() {
  const currentlyDark = document.documentElement.getAttribute('data-theme') === 'dark'
    || (state.theme === null && window.matchMedia('(prefers-color-scheme: dark)').matches);
  state.theme = currentlyDark ? 'light' : 'dark';
  applyTheme();
  saveState();
}

/* =========================================================================
   14. SIDEBAR OPEN/COLLAPSE (responsive)
   ========================================================================= */

function toggleSidebarCollapse() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  el.app.classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
  saveState();
}

function openMobileSidebar() {
  el.app.classList.add('sidebar-open');
  el.scrim.hidden = false;
  el.menuBtn.setAttribute('aria-expanded', 'true');
  el.historySearch.focus();
}
function closeMobileSidebar() {
  el.app.classList.remove('sidebar-open');
  el.scrim.hidden = true;
  el.menuBtn.setAttribute('aria-expanded', 'false');
}

/* =========================================================================
   15. STATUS / HEALTH CHECK
   ========================================================================= */

async function refreshStatus() {
  el.statusDot.className = 'status-dot checking';
  el.statusText.textContent = 'Checking…';
  const ok = await API.checkHealth();
  el.statusDot.className = `status-dot ${ok ? 'online' : 'offline'}`;
  el.statusText.textContent = MOCK_MODE ? 'Demo mode' : (ok ? 'Connected' : 'Offline');
}

/* =========================================================================
   16. MISC HELPERS
   ========================================================================= */

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function announce(text) {
  el.srAnnouncer.textContent = '';
  requestAnimationFrame(() => { el.srAnnouncer.textContent = text; });
}

/* =========================================================================
   17. EVENT WIRING
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

  el.newChatBtn.addEventListener('click', () => {
    state.currentId = null;
    saveState();
    renderSidebar(el.historySearch.value);
    renderThread();
    updateHeaderTitle();
    closeMobileSidebar();
    el.composerInput.focus();
  });

  el.historySearch.addEventListener('input', () => renderSidebar(el.historySearch.value));

  el.historyList.addEventListener('click', (e) => {
    const menuBtn = e.target.closest('.history-item__menu-btn');
    const item = e.target.closest('.history-item');
    if (menuBtn) {
      e.stopPropagation();
      openConversationMenu(menuBtn, item.dataset.id);
      return;
    }
    if (item) switchConversation(item.dataset.id);
  });
  el.historyList.addEventListener('keydown', (e) => {
    const item = e.target.closest('.history-item');
    if (item && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      switchConversation(item.dataset.id);
    }
  });

  el.sidebarCollapseBtn.addEventListener('click', toggleSidebarCollapse);
  el.menuBtn.addEventListener('click', () => {
    if (el.app.classList.contains('sidebar-open')) closeMobileSidebar();
    else openMobileSidebar();
  });
  el.scrim.addEventListener('click', closeMobileSidebar);

  el.themeToggleBtn.addEventListener('click', toggleTheme);

  el.thread.addEventListener('scroll', handleThreadScroll);
  el.thread.addEventListener('click', handleThreadClick);
  el.jumpLatestBtn.addEventListener('click', () => scrollToBottom());

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    // Close whichever "panel" is currently open, most specific first.
    const openMenu = document.querySelector('.history-item__menu');
    if (openMenu) { closeAnyOpenMenu(); return; }
    if (el.app.classList.contains('sidebar-open')) { closeMobileSidebar(); return; }
    const openPanel = document.querySelector('.sources-panel.is-open');
    if (openPanel) {
      const msgId = openPanel.id.replace('sources-', '');
      toggleSourcesPanel(msgId, false);
    }
  });

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (state.theme === null) applyTheme();
  });

  window.addEventListener('beforeunload', saveState);
}

/* =========================================================================
   18. INIT
   ========================================================================= */

function init() {
  loadState();
  applyTheme();
  el.app.classList.toggle('sidebar-collapsed', state.sidebarCollapsed);

  renderSidebar();
  renderThread();
  updateHeaderTitle();
  autoResizeTextarea();
  wireEvents();
  refreshStatus();

  // Backend reachability isn't going to change on its own in mock mode,
  // but this keeps the status pill honest if MOCK_MODE is flipped off
  // later without a page reload during development.
  setInterval(refreshStatus, 30000);
}

document.addEventListener('DOMContentLoaded', init);
