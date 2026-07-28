/**
 * RAG_InsureAI chat widget — self-contained embed for the InsureHub site.
 *
 * Usage on the host page:
 *   <div id="raginsureai-widget-root"></div>
 *   <script src="raginsureai-widget.js"></script>
 *
 * Mounts into a Shadow DOM so none of this CSS/JS touches the host page,
 * and none of the host page's styles (resets, global selectors) leak in.
 * Hooks the EXISTING robot-icon trigger button
 * (`button[aria-label="Open Chatbot"]`, confirmed live on the deployed
 * site) rather than adding a second button.
 *
 * ---------------------------------------------------------------------
 * THEME VALUES — provenance
 * ---------------------------------------------------------------------
 * Every color/font/radius/spacing value below was read directly off the
 * live site (insure-hub-react-*.azurewebsites.net) via getComputedStyle
 * and its real `--theme-*` CSS custom properties — not eyeballed from a
 * screenshot. Specifics worth flagging:
 *
 *  - The "terracotta/burnt-orange" accent described from the screenshot
 *    is actually the site's `--theme-color-danger: #DC2626` (a clean
 *    red, reused for the "Insure." tagline flourish) — there is no
 *    separate brand-accent token. Used here for the send button / user
 *    bubbles / active states per the spec's own instruction to treat it
 *    as the action color, but flagging the mismatch with "terracotta"
 *    since it's a cooler red, not an orange-brown.
 *  - `font-family: Inter, system-ui, sans-serif` is DECLARED on the
 *    live site, but no Inter font file is actually loaded there
 *    (`document.fonts` is empty, no <link> to a font host) — it's
 *    silently falling back to system-ui everywhere already. Matching
 *    that exactly (same stack, no font fetch of our own) is what makes
 *    this widget render in the literal same typeface as the rest of
 *    the page, rather than introducing a font InsureHub doesn't
 *    actually use yet.
 *  - Card radius (`rounded-xl` = 12px), 2px borders, and the selected-
 *    state treatment (primary-color border + `bg-theme-primary/5` tint
 *    + soft primary-tinted shadow) are copied from the live CAR/HOME/
 *    TRAVEL/HEALTH category cards' computed styles.
 *  - The trigger-button stack is `fixed bottom-4 right-4 md:bottom-6
 *    md:right-6 z-40`, two 48px (mobile) / 56px (md:768px+) circular
 *    buttons with an 8px/12px gap — so total stack height is 16+48+8+48
 *    =120px (mobile) / 24+56+12+56=148px (desktop). The panel's bottom
 *    offset below is derived from those exact numbers so it clears the
 *    WhatsApp button instead of guessing a round number.
 */
(function () {
  'use strict';

  /* =========================================================================
     CONFIG
     ========================================================================= */

  // Ships defaulting to mock (per spec) so the widget is demoable with
  // zero setup — but for local testing against a real running backend,
  // set these two globals in a <script> BEFORE this file loads (see
  // preview.html) rather than hand-editing this deliverable file back
  // and forth.
  const MOCK_MODE = window.RAGINSUREAI_MOCK_MODE ?? true;
  const TRIGGER_SELECTOR = 'button[aria-label="Open Chatbot"]';
  const API_BASE_URL = window.RAGINSUREAI_API_BASE_URL ?? '';
  const STORAGE_KEY = 'raginsureai_widget_state_v1';

  /* =========================================================================
     THEME TOKENS — real values, see file header for provenance
     ========================================================================= */

  const T = {
    fontSans: `"Inter", system-ui, sans-serif`, // declared, not loaded — see header
    canvas: '#F8FAFC',
    surface: '#FFFFFF',
    surfaceMuted: '#F3F4F6',
    surfaceSoft: '#E5E7EB',
    primary: '#51A3FF',
    primaryAlt: '#3B82F6',
    primaryHover: '#1D4ED8',
    primarySoft: '#DBEAFE',
    accent: '#93C5FD', // hero background on the host site
    text: '#111827',
    textMuted: '#4B5563',
    textSoft: '#6B7280',
    textInverse: '#FFFFFF',
    border: '#E5E7EB',
    borderStrong: '#D1D5DB',
    action: '#DC2626', // site's --theme-color-danger, reused as brand action color — see header
    actionHover: '#B91C1C',
    actionSoft: '#FEF2F2',
    success: '#16A34A',
    shadow: 'rgba(17, 24, 39, 0.08)',
    radiusCard: '12px', // matches host's rounded-xl
    radiusPanel: '20px',
    radiusPill: '999px',
    // See file header: 120px mobile / 148px desktop clears the existing
    // WhatsApp + robot button stack instead of overlapping it.
    stackClearanceMobile: '120px',
    stackClearanceDesktop: '148px',
    sideMarginMobile: '16px',
    sideMarginDesktop: '24px',
  };

  // Source-type colors kept distinct from the host palette (documents/
  // video/webpage need to be tellable apart at a glance) but pulled from
  // the same family so they still read as "this site", not a stranger's
  // palette: primary blue for documents, the action red for video,
  // and the host's own --theme-color-purple-strong for webpages.
  const SOURCE_TYPE = {
    document: { label: 'Document', color: T.primaryAlt },
    video: { label: 'Video', color: T.action },
    webpage: { label: 'Webpage', color: '#7C3AED' },
  };

  /* =========================================================================
     MOCK DATA
     ========================================================================= */

  const MOCK_RESPONSES = [
    {
      keywords: ['deductible', 'premium'],
      content:
        "Quick one to untangle.\n\n" +
        "Your **premium** is what you pay to keep the policy active — monthly or yearly, your call {{cite:1}}. Your **deductible** is what *you* cover first on a claim before we step in {{cite:2}}.\n\n" +
        "- Higher deductible → lower premium, bigger bill if you claim\n" +
        "- Lower deductible → higher premium, smaller bill if you claim\n\n" +
        "You pick your deductible tier when you set up cover {{cite:3}}.",
      sources: [
        { type: 'document', title: 'Home Policy Wording — Section 4', excerpt: 'The premium is the amount payable by the insured for the coverage bound under this policy, per the schedule selected at inception.', page: 12, score: 0.91 },
        { type: 'document', title: 'Home Policy Wording — Section 4', excerpt: 'The deductible is the portion of any covered loss that remains the responsibility of the insured before indemnification applies.', page: 13, score: 0.88 },
        { type: 'webpage', title: 'Choosing Your Deductible — InsureHub', url: 'insurehub.example/learn/deductible-tiers', excerpt: 'Most plans offer three or four deductible tiers at renewal, from around ₹2,500 to ₹25,000 depending on policy type.', score: 0.74 },
      ],
    },
    {
      keywords: ['water', 'damage', 'flood', 'leak', 'home', 'house'],
      content:
        "Here's how a water damage claim usually goes:\n\n" +
        "1. **Stop further damage** if it's safe — shut off the source, move valuables\n" +
        "2. **Photograph everything** before cleanup starts {{cite:1}}\n" +
        "3. **File the claim** within the notice window in your policy {{cite:2}}\n" +
        "4. An adjuster typically inspects within a few business days {{cite:1}}\n\n" +
        "One thing to check: **gradual leaks** (a slow drip over months) are often excluded, while **sudden and accidental** damage usually isn't {{cite:3}}.",
      sources: [
        { type: 'document', title: 'Claims Handbook — Property Damage', excerpt: 'Policyholders should photograph, and where practicable video, the affected area before remediation or repair begins.', page: 27, score: 0.93 },
        { type: 'document', title: 'Home Policy Wording — Section 9', excerpt: 'Notice of loss must be given within a reasonable time, and in no event later than 60 days from discovery.', page: 31, score: 0.85 },
        { type: 'video', title: "Water Damage Claims: What's Actually Covered", excerpt: "The big distinction adjusters look for is sudden versus gradual — a burst pipe is very different, coverage-wise, from a slow leak going on for months.", timestamp: '3:41', score: 0.80 },
      ],
    },
    {
      keywords: ['travel', 'trip', 'flight', 'abroad'],
      content:
        "Depends what stage of the trip you mean.\n\n" +
        "**Before you fly**, trip cancellation cover kicks in for covered reasons like sudden illness or a family emergency {{cite:1}}. **While you're away**, medical, baggage, and delay cover apply per your plan's limits {{cite:2}}.\n\n" +
        "Worth checking your specific plan tier — cover for adventure activities or pre-existing conditions usually needs an add-on {{cite:1}}.",
      sources: [
        { type: 'document', title: 'Travel Policy Wording — Section 2', excerpt: 'Trip cancellation benefits apply where cancellation is caused by a covered reason occurring after the policy effective date and before departure.', page: 6, score: 0.87 },
        { type: 'webpage', title: 'What Travel Insurance Actually Covers — InsureHub', url: 'insurehub.example/learn/travel-coverage', excerpt: 'Medical, baggage delay, and trip interruption are the three most-claimed benefits — each has its own sub-limit shown on your certificate.', score: 0.79 },
      ],
    },
    {
      keywords: ['car', 'vehicle', 'accident', 'rental'],
      content:
        "For car cover, it comes down to which policy layer applies.\n\n" +
        "**Third-party liability** is mandatory and covers damage or injury you cause to others {{cite:1}}. **Own-damage cover** (optional, add it if you want it) pays for repairs to your own vehicle {{cite:2}}.\n\n" +
        "A rental car during repairs is usually covered automatically for a limited period — worth confirming the exact number of days on your declarations page {{cite:2}}.",
      sources: [
        { type: 'document', title: 'Motor Policy Wording — Section 3', excerpt: 'This policy provides liability coverage for bodily injury and property damage caused to third parties arising from use of the insured vehicle.', page: 9, score: 0.89 },
        { type: 'document', title: 'Motor Policy Wording — Section 6', excerpt: 'Coverage extends to a temporary substitute automobile while the covered vehicle is out of service for repair, for up to 30 days.', page: 19, score: 0.82 },
      ],
    },
  ];

  const MOCK_FALLBACK = {
    content:
      "Happy to help with that.\n\n" +
      "Coverage details vary by plan and by the endorsements attached to your specific policy {{cite:1}}. The general rule carriers apply: a loss needs to be **sudden, accidental, and named or unexcluded** under your policy to qualify {{cite:2}}.\n\n" +
      "Tell me a bit more about your situation and I can point to the exact clause.",
    sources: [
      { type: 'document', title: 'General Policy Provisions', excerpt: 'Coverage terms, limits, and exclusions vary by product line and by the specific endorsements attached at binding.', page: 4, score: 0.71 },
      { type: 'webpage', title: 'How Claims Eligibility Works — InsureHub', url: 'insurehub.example/learn/claims-eligibility', excerpt: 'Insurers generally distinguish sudden and accidental losses (typically covered) from gradual or expected wear (typically excluded).', score: 0.68 },
    ],
  };

  function pickMockResponse(query) {
    const q = (query || '').toLowerCase();
    return MOCK_RESPONSES.find((r) => r.keywords.some((k) => q.includes(k))) || MOCK_FALLBACK;
  }

  const EXAMPLE_QUESTIONS = [
    "What's the difference between a deductible and a premium?",
    'How do I file a claim after water damage?',
    'Does my travel policy cover a cancelled flight?',
    'Is a rental car covered while mine is being repaired?',
    'What happens if I miss a premium payment?',
  ];

  /* =========================================================================
     API MODULE — the only place that knows about the backend
     ========================================================================= */

  const API = {
    async checkHealth() {
      if (MOCK_MODE) return true;
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        return res.ok;
      } catch {
        return false;
      }
    },

    async *streamAsk(query, conversationId, signal) {
      if (MOCK_MODE) {
        yield* mockStreamAsk(query, signal);
        return;
      }
      // ================= REAL BACKEND CALL =================
      // This is the RAG_InsureAI FastAPI backend's actual /ask-stream
      // contract (confirmed against api.py and the existing Layla web
      // client's lib/api.ts — NOT the generic placeholder described in
      // the original widget spec). Body is {question, session_id}, and
      // the response is plain streamed answer text with a final
      // '\n\n{"sources": [...], "done": true, ...}' JSON blob appended
      // at the very end — not NDJSON. `sources` here also carry a
      // different shape than the mock data (source/page/similarity
      // rather than type/title/excerpt/score) — see the mapping below.
      const res = await fetch(`${API_BASE_URL}/ask-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, session_id: conversationId }),
        signal,
      });
      if (!res.ok || !res.body) throw new Error(`Request failed (${res.status || 'network error'})`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Same detection the real Layla client uses: the answer text
        // itself won't ever contain this exact literal substring, so a
        // plain lastIndexOf is reliable without needing real JSON
        // streaming framing.
        const jsonStart = buffer.lastIndexOf('\n\n{"sources"');
        if (jsonStart !== -1) {
          const textPart = buffer.slice(0, jsonStart);
          const jsonPart = buffer.slice(jsonStart + 2);
          if (textPart) yield { type: 'chunk', text: textPart };
          try {
            const meta = JSON.parse(jsonPart);
            yield { type: 'sources', sources: mapBackendSources(meta.sources) };
          } catch {
            yield { type: 'sources', sources: [] };
          }
          return;
        }
        if (buffer) { yield { type: 'chunk', text: buffer }; buffer = ''; }
      }
      if (buffer) yield { type: 'chunk', text: buffer };
      yield { type: 'sources', sources: [] };
    },
  };

  // The backend's real source objects (source filename/url, page,
  // similarity, source_type: "document"|"video"|"webpage") don't match
  // the mock shape (type/title/excerpt/score) — normalize here so
  // rendering code only ever has to deal with one shape. Assumed field
  // names based on this codebase's established metadata conventions
  // (page/timestamp/source_type/similarity) — confirm against a real
  // response if the sources panel renders blank once MOCK_MODE is off.
  function mapBackendSources(raw) {
    if (!Array.isArray(raw)) return [];
    return raw.map((s) => ({
      type: s.source_type || s.type || 'document',
      title: s.title || s.source || s.document_name || 'Source',
      excerpt: s.excerpt || s.text || s.content || '',
      page: s.page,
      timestamp: s.timestamp,
      url: s.url || s.source,
      score: typeof s.similarity === 'number' ? s.similarity : (typeof s.score === 'number' ? s.score : 0),
    }));
  }

  async function* mockStreamAsk(query, signal) {
    const demo = pickMockResponse(query);
    await sleep(600 + Math.random() * 450, signal);
    const tokens = demo.content.split(/(\s+)/);
    for (const t of tokens) {
      if (signal?.aborted) return;
      yield { type: 'chunk', text: t };
      if (t.trim()) await sleep(15 + Math.random() * 32, signal);
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
     MARKDOWN + CITATIONS (same small hand-rolled parser as the dev preview)
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
    return s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>');
  }

  function injectCitations(html, sources, msgId) {
    if (!sources || !sources.length) return html;
    return html.replace(/\{\{cite:(\d+)\}\}/g, (_, n) => {
      const idx = parseInt(n, 10) - 1;
      const src = sources[idx];
      if (!src) return '';
      const meta = SOURCE_TYPE[src.type] || SOURCE_TYPE.document;
      return (
        `<button type="button" class="rih-cite" style="--type-color:${meta.color}" ` +
        `data-msg-id="${msgId}" data-cite-index="${idx}" aria-label="Source ${n}: ${escapeAttr(src.title)}">` +
        `${n}<span class="rih-cite__tip" role="tooltip">` +
        `<b>${escapeHtml(src.title)}</b>${escapeHtml(truncate(src.excerpt, 100))}</span></button>`
      );
    });
  }

  function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }
  function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  function sourceDetail(src) {
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

  function sourceIcon(type) {
    // Same outline weight/style as the host's own car/home/plane/heart icons.
    const icons = {
      document: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M7 3H14L19 8V19C19 20.1 18.1 21 17 21H7C5.9 21 5 20.1 5 19V5C5 3.9 5.9 3 7 3Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M14 3V8H19" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
      video: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M10 8.5L16 12L10 15.5V8.5Z" fill="currentColor"/></svg>',
      webpage: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M3 12H21M12 3C14.5 5.7 15.8 9 15.8 12C15.8 15 14.5 18.3 12 21C9.5 18.3 8.2 15 8.2 12C8.2 9 9.5 5.7 12 3Z" stroke="currentColor" stroke-width="1.6"/></svg>',
    };
    return icons[type] || icons.document;
  }

  function uid() { return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`; }
  function msg(role, content, timestamp, sources) {
    return { id: uid(), role, content, timestamp, sources: sources || null, feedback: null };
  }

  /* =========================================================================
     CSS — injected via <style> template string (no <link>/@import; those
     don't reliably work inside a shadow root)
     ========================================================================= */

  const CSS_TEXT = `
    :host { all: initial; }
    * { box-sizing: border-box; }

    .rih-root {
      font-family: ${T.fontSans};
      font-size: 14px;
      color: ${T.text};
      line-height: 1.5;
    }
    button { cursor: pointer; font-family: inherit; }
    ul { margin: 0; padding: 0; list-style: none; }

    .rih-root :focus-visible {
      outline: 2px solid ${T.primaryAlt};
      outline-offset: 2px;
      border-radius: 4px;
    }

    @media (prefers-reduced-motion: reduce) {
      .rih-root, .rih-root * {
        animation-duration: 0.001ms !important;
        transition-duration: 0.001ms !important;
      }
    }

    /* ---- Panel shell ---- */
    .rih-panel {
      position: fixed;
      right: ${T.sideMarginMobile};
      bottom: ${T.stackClearanceMobile};
      width: min(400px, calc(100vw - 32px));
      height: min(620px, 72vh);
      background: ${T.surface};
      border-radius: ${T.radiusPanel};
      box-shadow: 0 24px 60px rgba(17,24,39,0.22), 0 4px 14px rgba(17,24,39,0.10);
      border: 1px solid ${T.border};
      display: flex;
      flex-direction: column;
      overflow: hidden;
      z-index: 2147483000; /* one step above the host's own z-40 stack */
      transform-origin: bottom right;
      transform: scale(0.92) translateY(10px);
      opacity: 0;
      pointer-events: none;
      transition: transform 260ms cubic-bezier(0.2,0.8,0.3,1), opacity 220ms ease-out;
    }
    .rih-panel.is-open {
      transform: scale(1) translateY(0);
      opacity: 1;
      pointer-events: auto;
    }
    @media (min-width: 768px) {
      .rih-panel { right: ${T.sideMarginDesktop}; bottom: ${T.stackClearanceDesktop}; }
    }
    /* Near-fullscreen sheet below ~600px per spec — a 380-420px panel
       doesn't work at phone width. */
    @media (max-width: 600px) {
      .rih-panel {
        right: 8px; left: 8px; bottom: 8px; top: 8px;
        width: auto; height: auto;
        border-radius: 18px;
        transform-origin: bottom center;
        transform: scale(0.96) translateY(16px);
      }
      .rih-panel.is-open { transform: scale(1) translateY(0); }
    }

    /* ---- Header ---- */
    .rih-header {
      display: flex; align-items: center; gap: 10px;
      padding: 14px 14px 14px 16px;
      border-bottom: 1px solid ${T.border};
      flex-shrink: 0;
    }
    .rih-header__avatar {
      width: 32px; height: 32px; border-radius: 10px;
      background: linear-gradient(155deg, ${T.primary}, ${T.primaryHover});
      display: flex; align-items: center; justify-content: center;
      color: #fff; flex-shrink: 0;
    }
    .rih-header__meta { flex: 1; min-width: 0; }
    .rih-header__name { font-weight: 700; font-size: 0.95rem; color: ${T.text}; }
    .rih-header__status { font-size: 0.72rem; color: ${T.textSoft}; display: flex; align-items: center; gap: 5px; }
    .rih-status-dot { width: 6px; height: 6px; border-radius: 50%; background: ${T.success}; flex-shrink: 0; }
    .rih-status-dot.offline { background: ${T.action}; }
    .rih-icon-btn {
      width: 30px; height: 30px; border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      background: none; border: none; color: ${T.textMuted};
      transition: background 120ms ease;
    }
    .rih-icon-btn:hover { background: ${T.surfaceMuted}; color: ${T.text}; }

    /* ---- Thread ---- */
    .rih-thread { flex: 1; overflow-y: auto; padding: 16px 14px; scroll-behavior: smooth; }
    @media (prefers-reduced-motion: reduce) { .rih-thread { scroll-behavior: auto; } }
    .rih-thread::-webkit-scrollbar { width: 7px; }
    .rih-thread::-webkit-scrollbar-thumb { background: ${T.borderStrong}; border-radius: 8px; }

    .rih-empty { text-align: center; padding: 12px 6px 4px; }
    .rih-empty__icon {
      width: 42px; height: 42px; margin: 0 auto 12px; border-radius: 13px;
      background: linear-gradient(155deg, ${T.primary}, ${T.primaryHover});
      display: flex; align-items: center; justify-content: center; color: #fff;
    }
    .rih-empty h2 { font-size: 1rem; margin: 0 0 6px; font-weight: 700; color: ${T.text}; }
    .rih-empty p { font-size: 0.82rem; color: ${T.textSoft}; margin: 0 0 16px; line-height: 1.5; }
    .rih-examples { display: flex; flex-direction: column; gap: 7px; text-align: left; }
    .rih-example {
      background: ${T.surface}; border: 1.5px solid ${T.border}; border-radius: 12px;
      padding: 10px 12px; font-size: 0.8rem; color: ${T.text}; text-align: left;
      transition: border-color 120ms ease, transform 120ms ease;
    }
    .rih-example:hover { border-color: ${T.primary}; transform: translateY(-1px); }

    .rih-msg { display: flex; gap: 8px; margin-bottom: 16px; animation: rih-in 320ms cubic-bezier(0.2,0.7,0.3,1); }
    @keyframes rih-in { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
    .rih-msg--user { flex-direction: row-reverse; }
    .rih-msg__avatar {
      width: 24px; height: 24px; border-radius: 7px; flex-shrink: 0; margin-top: 2px;
      display: flex; align-items: center; justify-content: center; font-size: 0.62rem; font-weight: 700;
    }
    .rih-msg--assistant .rih-msg__avatar { background: linear-gradient(155deg, ${T.primary}, ${T.primaryHover}); color: #fff; }
    .rih-msg--user .rih-msg__avatar { background: ${T.actionSoft}; color: ${T.action}; }
    .rih-msg__col { flex: 1; min-width: 0; display: flex; flex-direction: column; }
    .rih-msg--user .rih-msg__col { align-items: flex-end; }
    .rih-bubble { border-radius: 14px; padding: 10px 13px; font-size: 0.84rem; line-height: 1.55; max-width: 100%; }
    .rih-msg--assistant .rih-bubble { background: ${T.surfaceMuted}; color: ${T.text}; border-top-left-radius: 4px; }
    .rih-msg--user .rih-bubble { background: ${T.action}; color: #fff; border-top-right-radius: 4px; }
    .rih-bubble p { margin: 0 0 8px; }
    .rih-bubble p:last-child { margin-bottom: 0; }
    .rih-bubble ul { margin: 3px 0 8px; padding-left: 18px; list-style: disc; }
    .rih-bubble li { margin-bottom: 3px; }
    .rih-bubble strong { font-weight: 700; }
    .rih-bubble code { background: rgba(0,0,0,0.06); border-radius: 4px; padding: 1px 4px; font-size: 0.88em; }

    .rih-cite {
      display: inline-flex; align-items: center; justify-content: center;
      width: 15px; height: 15px; border-radius: 50%; font-size: 0.6rem; font-weight: 700;
      vertical-align: super; margin-left: 1px; cursor: pointer; position: relative;
      background: ${T.surface}; border: 1.4px solid var(--type-color, ${T.primaryAlt});
      color: var(--type-color, ${T.primaryAlt});
      transition: background 120ms ease, transform 120ms ease;
    }
    .rih-cite:hover, .rih-cite:focus-visible { background: var(--type-color); color: #fff; transform: translateY(-1px); }
    .rih-cite__tip {
      position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
      width: 200px; background: ${T.text}; color: #fff; padding: 8px 10px; border-radius: 8px;
      font-size: 0.68rem; font-weight: 400; line-height: 1.4; opacity: 0; pointer-events: none;
      transition: opacity 120ms ease; z-index: 5;
    }
    .rih-cite:hover .rih-cite__tip { opacity: 1; }
    .rih-cite__tip b { display: block; margin-bottom: 2px; font-weight: 700; }
    @media (hover: none) { .rih-cite__tip { display: none; } }

    .rih-typing { display: inline-flex; gap: 3px; padding: 2px; }
    .rih-typing span { width: 5px; height: 5px; border-radius: 50%; background: ${T.textSoft}; animation: rih-bounce 1.1s ease-in-out infinite; }
    .rih-typing span:nth-child(2) { animation-delay: 0.15s; }
    .rih-typing span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes rih-bounce { 0%,60%,100% { transform: translateY(0); opacity: 0.5; } 30% { transform: translateY(-3px); opacity: 1; } }
    .rih-cursor { display: inline-block; width: 2px; height: 0.9em; background: ${T.primaryAlt}; margin-left: 2px; vertical-align: text-bottom; animation: rih-blink 0.9s steps(1) infinite; }
    @keyframes rih-blink { 50% { opacity: 0; } }

    .rih-controls { display: flex; gap: 2px; margin-top: 5px; opacity: 0.7; }
    .rih-controls button {
      width: 24px; height: 24px; border-radius: 6px; background: none; border: none; color: ${T.textSoft};
      display: flex; align-items: center; justify-content: center; transition: background 120ms ease, color 120ms ease;
    }
    .rih-controls button:hover { background: ${T.surfaceMuted}; color: ${T.text}; }
    .rih-controls button.is-active.up { color: ${T.success}; }
    .rih-controls button.is-active.down { color: ${T.action}; }

    .rih-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
    .rih-chip { font-size: 0.68rem; padding: 4px 9px; border-radius: 999px; border: 1px solid ${T.border}; background: ${T.surface}; color: ${T.textSoft}; }
    .rih-chip:hover { border-color: ${T.action}; color: ${T.action}; }
    .rih-thanks { font-size: 0.68rem; color: ${T.textSoft}; font-style: italic; margin-top: 5px; }

    .rih-sources-toggle {
      display: inline-flex; align-items: center; gap: 5px; margin-top: 8px;
      font-size: 0.76rem; font-weight: 650; color: ${T.primaryAlt}; padding: 3px;
    }
    .rih-sources-toggle svg { transition: transform 120ms ease; }
    .rih-sources-toggle.is-open svg { transform: rotate(180deg); }
    .rih-sources-panel { display: grid; grid-template-rows: 0fr; transition: grid-template-rows 260ms ease; }
    .rih-sources-panel.is-open { grid-template-rows: 1fr; }
    .rih-sources-panel__inner { overflow: hidden; }
    .rih-sources-grid { display: flex; flex-direction: column; gap: 7px; margin-top: 8px; }

    .rih-source-card {
      background: ${T.surface}; border: 2px solid ${T.border}; border-left: 3px solid var(--type-color, ${T.primaryAlt});
      border-radius: ${T.radiusCard}; padding: 9px 10px;
      transition: box-shadow 120ms ease, border-color 200ms ease;
      scroll-margin-top: 20px;
    }
    .rih-source-card.is-highlighted { border-color: var(--type-color); box-shadow: 0 0 0 3px color-mix(in srgb, var(--type-color) 20%, transparent); }
    .rih-source-card__head { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }
    .rih-source-card__icon { width: 19px; height: 19px; border-radius: 5px; display: flex; align-items: center; justify-content: center; background: color-mix(in srgb, var(--type-color) 15%, transparent); color: var(--type-color); flex-shrink: 0; }
    .rih-source-card__num { font-size: 0.65rem; font-weight: 700; color: var(--type-color); }
    .rih-source-card__title { font-size: 0.76rem; font-weight: 650; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .rih-source-card__excerpt { font-size: 0.74rem; color: ${T.textSoft}; line-height: 1.45; margin: 0 0 6px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .rih-source-card__foot { display: flex; align-items: center; justify-content: space-between; gap: 6px; }
    .rih-source-card__detail { font-size: 0.66rem; color: ${T.textSoft}; }
    .rih-relevance { display: flex; align-items: center; gap: 4px; }
    .rih-relevance__track { width: 28px; height: 3px; border-radius: 3px; background: ${T.border}; overflow: hidden; }
    .rih-relevance__fill { height: 100%; background: var(--type-color); }
    .rih-relevance__pct { font-size: 0.62rem; color: ${T.textSoft}; }

    .rih-error { border: 1px solid color-mix(in srgb, ${T.action} 35%, ${T.border}); background: ${T.actionSoft}; border-radius: 12px; padding: 11px 12px; display: flex; gap: 9px; }
    .rih-error__title { font-weight: 650; font-size: 0.82rem; margin: 0 0 3px; }
    .rih-error__desc { font-size: 0.76rem; color: ${T.textMuted}; margin: 0 0 8px; line-height: 1.4; }
    .rih-retry { background: ${T.action}; color: #fff; padding: 6px 12px; border-radius: 8px; font-size: 0.76rem; font-weight: 600; border: none; }

    /* ---- Composer ---- */
    .rih-composer { border-top: 1px solid ${T.border}; padding: 10px 12px 10px; flex-shrink: 0; }
    .rih-composer__box {
      display: flex; align-items: flex-end; gap: 6px;
      background: ${T.canvas}; border: 1.5px solid ${T.border}; border-radius: 16px; padding: 6px 6px 6px 12px;
      transition: border-color 120ms ease;
    }
    .rih-composer__box:focus-within { border-color: ${T.primary}; }
    .rih-input {
      flex: 1; border: none; background: none; resize: none; max-height: 100px;
      font-family: inherit; font-size: 0.84rem; line-height: 1.4; padding: 6px 0; color: ${T.text};
    }
    .rih-input:focus { outline: none; }
    .rih-input::placeholder { color: ${T.textSoft}; }
    .rih-send {
      flex-shrink: 0; width: 32px; height: 32px; border-radius: 9px; background: ${T.action}; color: #fff;
      display: flex; align-items: center; justify-content: center; border: none; position: relative;
      transition: background 120ms ease;
    }
    .rih-send:hover:not(:disabled) { background: ${T.actionHover}; }
    .rih-send:disabled { background: ${T.borderStrong}; cursor: not-allowed; }
    .rih-send__icon { transition: opacity 120ms ease; }
    .rih-send.is-loading .rih-send__icon { opacity: 0; }
    .rih-send__spinner { display: none; position: absolute; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff; border-radius: 50%; animation: rih-spin 0.7s linear infinite; }
    .rih-send.is-loading .rih-send__spinner { display: block; }
    @keyframes rih-spin { to { transform: rotate(360deg); } }
    .rih-disclaimer { font-size: 0.66rem; color: ${T.textSoft}; text-align: center; margin: 7px 2px 0; }

    .rih-sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); }
  `;

  /* =========================================================================
     WIDGET CLASS
     ========================================================================= */

  class RagInsureAIWidget {
    constructor() {
      this.messages = [];
      this.isOpen = false;
      this.userScrolledAway = false;
      this.activeAbort = null;
      this.conversationId = uid();
      this._restoreState();
      this._buildDom();
      this._wireEvents();
      this._hookTrigger();
      this._checkHealth();
    }

    _restoreState() {
      try {
        const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || 'null');
        if (saved && Array.isArray(saved.messages)) {
          this.messages = saved.messages;
          this.conversationId = saved.conversationId || this.conversationId;
        }
      } catch { /* ignore corrupt state */ }
    }

    _saveState() {
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ messages: this.messages, conversationId: this.conversationId }));
      } catch { /* non-fatal */ }
    }

    _buildDom() {
      // Host element lives in the LIGHT DOM — `all: initial` here (not just
      // inside the shadow root) matters because inherited CSS properties
      // (font-family, color, line-height...) cross the shadow boundary even
      // though style RULES don't. Without this, the host page's own body
      // font/color would silently bleed into everything inside.
      const mountPoint = document.getElementById('raginsureai-widget-root') || document.body;
      this.hostEl = document.createElement('div');
      this.hostEl.id = 'raginsureai-widget-host';
      this.hostEl.style.all = 'initial';
      this.hostEl.style.position = 'fixed';
      this.hostEl.style.zIndex = '2147483000';
      mountPoint.appendChild(this.hostEl);

      this.shadow = this.hostEl.attachShadow({ mode: 'open' });
      const style = document.createElement('style');
      style.textContent = CSS_TEXT;
      this.shadow.appendChild(style);

      const root = document.createElement('div');
      root.className = 'rih-root';
      root.innerHTML = `
        <div class="rih-panel" role="dialog" aria-modal="false" aria-label="InsureHub Assistant" aria-hidden="true">
          <div class="rih-header">
            <div class="rih-header__avatar" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 2L3 6.5V11C3 16.2 6.8 20.9 12 22C17.2 20.9 21 16.2 21 11V6.5L12 2Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M8.5 12.2L10.8 14.5L15.5 9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="rih-header__meta">
              <div class="rih-header__name">InsureHub Assistant</div>
              <div class="rih-header__status"><span class="rih-status-dot" data-role="status-dot"></span><span data-role="status-text">Online</span></div>
            </div>
            <button type="button" class="rih-icon-btn" data-action="new-chat" aria-label="New conversation" title="New conversation">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 5V19M5 12H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </button>
            <button type="button" class="rih-icon-btn" data-action="minimize" aria-label="Minimize">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M5 12H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </button>
            <button type="button" class="rih-icon-btn" data-action="close" aria-label="Close">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M6 6L18 18M6 18L18 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </button>
          </div>
          <div class="rih-thread" data-role="thread" role="log" aria-live="polite" aria-atomic="false"></div>
          <div class="rih-composer">
            <form data-role="composer-form">
              <div class="rih-composer__box">
                <textarea class="rih-input" data-role="input" rows="1" placeholder="Ask about your policy or a claim…" aria-label="Message"></textarea>
                <button type="submit" class="rih-send" data-role="send" aria-label="Send message">
                  <svg class="rih-send__icon" width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M4 12L20 4L14 20L11 13L4 12Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/></svg>
                  <span class="rih-send__spinner"></span>
                </button>
              </div>
            </form>
            <p class="rih-disclaimer">General information, not advice from a licensed agent.</p>
          </div>
        </div>
        <div class="rih-sr-only" data-role="announcer" role="status" aria-live="polite"></div>
      `;
      this.shadow.appendChild(root);

      this.panelEl = this.shadow.querySelector('.rih-panel');
      this.threadEl = this.shadow.querySelector('[data-role="thread"]');
      this.formEl = this.shadow.querySelector('[data-role="composer-form"]');
      this.inputEl = this.shadow.querySelector('[data-role="input"]');
      this.sendEl = this.shadow.querySelector('[data-role="send"]');
      this.statusDotEl = this.shadow.querySelector('[data-role="status-dot"]');
      this.statusTextEl = this.shadow.querySelector('[data-role="status-text"]');
      this.announcerEl = this.shadow.querySelector('[data-role="announcer"]');

      this._renderThread();
    }

    _hookTrigger() {
      const attach = () => {
        const btn = document.querySelector(TRIGGER_SELECTOR);
        if (!btn) return false;
        // addEventListener (not overwriting .onclick) so whatever the host
        // page already wired to this button keeps working alongside us.
        btn.addEventListener('click', () => this.toggle());
        return true;
      };
      if (!attach()) {
        // Button may not exist yet if this script loads before React
        // mounts it — poll briefly rather than failing silently.
        let tries = 0;
        const iv = setInterval(() => {
          tries += 1;
          if (attach() || tries > 40) clearInterval(iv);
        }, 250);
      }
    }

    _wireEvents() {
      this.formEl.addEventListener('submit', (e) => { e.preventDefault(); this.sendMessage(); });
      this.inputEl.addEventListener('input', () => this._autoResize());
      this.inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }
      });

      this.shadow.querySelector('[data-action="close"]').addEventListener('click', () => this.close());
      this.shadow.querySelector('[data-action="minimize"]').addEventListener('click', () => this.close());
      this.shadow.querySelector('[data-action="new-chat"]').addEventListener('click', () => this.newChat());

      this.threadEl.addEventListener('scroll', () => {
        const d = this.threadEl.scrollHeight - this.threadEl.scrollTop - this.threadEl.clientHeight;
        this.userScrolledAway = d > 60;
      });
      this.threadEl.addEventListener('click', (e) => this._handleThreadClick(e));

      // Escape / click-outside must attach to `document`, not the shadow
      // root — keyboard and click events bubble out of shadow DOM
      // normally, so this still reaches them from the host page.
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.isOpen) this.close();
      });
      document.addEventListener('click', (e) => {
        if (!this.isOpen) return;
        const path = e.composedPath();
        const clickedTrigger = path.some((el) => el.matches?.(TRIGGER_SELECTOR));
        if (!path.includes(this.hostEl) && !clickedTrigger) this.close();
      });
    }

    async _checkHealth() {
      const ok = await API.checkHealth();
      this.statusDotEl.classList.toggle('offline', !ok);
      this.statusTextEl.textContent = MOCK_MODE ? 'Demo mode' : (ok ? 'Online' : 'Offline');
    }

    open() {
      this.isOpen = true;
      this.panelEl.classList.add('is-open');
      this.panelEl.setAttribute('aria-hidden', 'false');
      setTimeout(() => this.inputEl.focus(), 260);
      this._scrollToBottom(true);
    }
    close() {
      this.isOpen = false;
      this.panelEl.classList.remove('is-open');
      this.panelEl.setAttribute('aria-hidden', 'true');
    }
    toggle() { this.isOpen ? this.close() : this.open(); }

    newChat() {
      // Abort any response still streaming into the OLD conversation first —
      // otherwise a late chunk could land in a message object that's about
      // to be thrown away, or worse, get appended after the empty state has
      // already rendered.
      this.activeAbort?.abort();
      this.messages = [];
      this.conversationId = uid();
      this.userScrolledAway = false;
      this.inputEl.value = '';
      this._autoResize();
      this._renderThread();
      this._saveState(); // overwrite sessionStorage so a reload doesn't resurrect the old thread
      this._announce('Started a new conversation.');
      this.inputEl.focus();
    }

    _autoResize() {
      this.inputEl.style.height = 'auto';
      this.inputEl.style.height = `${Math.min(this.inputEl.scrollHeight, 100)}px`;
    }

    _announce(text) {
      this.announcerEl.textContent = '';
      requestAnimationFrame(() => { this.announcerEl.textContent = text; });
    }

    _scrollToBottom(instant) {
      this.threadEl.scrollTo({ top: this.threadEl.scrollHeight, behavior: instant ? 'auto' : 'smooth' });
      this.userScrolledAway = false;
    }

    /* ---------------- rendering ---------------- */

    _renderThread() {
      this.threadEl.innerHTML = '';
      if (!this.messages.length) {
        this.threadEl.appendChild(this._buildEmptyState());
        return;
      }
      for (const m of this.messages) this.threadEl.appendChild(this._buildMessageEl(m));
    }

    _buildEmptyState() {
      const wrap = document.createElement('div');
      wrap.className = 'rih-empty';
      wrap.innerHTML = `
        <div class="rih-empty__icon" aria-hidden="true"><svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 2L3 6.5V11C3 16.2 6.8 20.9 12 22C17.2 20.9 21 16.2 21 11V6.5L12 2Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg></div>
        <h2>What can I help with?</h2>
        <p>Ask about a policy, a claim, or coverage — every answer links back to where it came from.</p>
        <div class="rih-examples"></div>
      `;
      const grid = wrap.querySelector('.rih-examples');
      for (const q of EXAMPLE_QUESTIONS) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'rih-example';
        btn.textContent = q;
        btn.addEventListener('click', () => this.sendMessage(q));
        grid.appendChild(btn);
      }
      return wrap;
    }

    _buildMessageEl(m) {
      const wrap = document.createElement('div');
      wrap.className = `rih-msg rih-msg--${m.role}`;
      wrap.dataset.id = m.id;

      const avatar = document.createElement('div');
      avatar.className = 'rih-msg__avatar';
      avatar.textContent = m.role === 'user' ? 'You' : 'IH';

      const col = document.createElement('div');
      col.className = 'rih-msg__col';

      const bubble = document.createElement('div');
      bubble.className = 'rih-bubble';
      bubble.innerHTML = m.role === 'assistant'
        ? injectCitations(renderMarkdown(m.content), m.sources, m.id)
        : escapeHtml(m.content);
      col.appendChild(bubble);

      if (m.role === 'assistant' && m.content) {
        if (m.sources && m.sources.length) {
          col.appendChild(this._buildSourcesToggle(m));
          col.appendChild(this._buildSourcesPanel(m));
        }
        col.appendChild(this._buildControls(m));
      }

      wrap.appendChild(avatar);
      wrap.appendChild(col);
      return wrap;
    }

    _buildSourcesToggle(m) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'rih-sources-toggle';
      btn.setAttribute('aria-expanded', 'false');
      btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>Show ${m.sources.length} source${m.sources.length === 1 ? '' : 's'}`;
      btn.addEventListener('click', () => this._toggleSources(m.id));
      return btn;
    }

    _buildSourcesPanel(m) {
      const panel = document.createElement('div');
      panel.className = 'rih-sources-panel';
      panel.id = `rih-sources-${m.id}`;
      const inner = document.createElement('div');
      inner.className = 'rih-sources-panel__inner';
      const grid = document.createElement('div');
      grid.className = 'rih-sources-grid';
      m.sources.forEach((src, i) => grid.appendChild(this._buildSourceCard(src, i, m.id)));
      inner.appendChild(grid);
      panel.appendChild(inner);
      return panel;
    }

    _buildSourceCard(src, index, msgId) {
      const meta = SOURCE_TYPE[src.type] || SOURCE_TYPE.document;
      const pct = Math.round((src.score ?? 0) * 100);
      const card = document.createElement('div');
      card.className = 'rih-source-card';
      card.style.setProperty('--type-color', meta.color);
      card.id = `rih-source-${msgId}-${index}`;
      card.innerHTML = `
        <div class="rih-source-card__head">
          <span class="rih-source-card__icon">${sourceIcon(src.type)}</span>
          <span class="rih-source-card__num">${index + 1}</span>
          <span class="rih-source-card__title" title="${escapeAttr(src.title)}">${escapeHtml(src.title)}</span>
        </div>
        <p class="rih-source-card__excerpt">${escapeHtml(src.excerpt)}</p>
        <div class="rih-source-card__foot">
          <span class="rih-source-card__detail">${escapeHtml(meta.label)} · ${escapeHtml(sourceDetail(src))}</span>
          <span class="rih-relevance"><span class="rih-relevance__track"><span class="rih-relevance__fill" style="width:${pct}%"></span></span><span class="rih-relevance__pct">${pct}%</span></span>
        </div>
      `;
      return card;
    }

    _toggleSources(msgId, forceOpen) {
      const panel = this.shadow.querySelector(`#rih-sources-${msgId}`);
      const btn = this.shadow.querySelector(`.rih-msg[data-id="${msgId}"] .rih-sources-toggle`);
      if (!panel || !btn) return;
      const open = forceOpen ?? !panel.classList.contains('is-open');
      panel.classList.toggle('is-open', open);
      btn.classList.toggle('is-open', open);
      btn.setAttribute('aria-expanded', String(open));
      const count = panel.querySelectorAll('.rih-source-card').length;
      btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>${open ? 'Hide' : 'Show'} ${count} source${count === 1 ? '' : 's'}`;
    }

    _buildControls(m) {
      const wrap = document.createElement('div');
      wrap.className = 'rih-controls';
      wrap.innerHTML = `
        <button type="button" class="rih-ctrl-copy" aria-label="Copy response" title="Copy"><svg width="14" height="14" viewBox="0 0 24 24" fill="none"><rect x="8" y="8" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.7"/><path d="M4 16V6C4 4.9 4.9 4 6 4H16" stroke="currentColor" stroke-width="1.7"/></svg></button>
        <button type="button" class="rih-ctrl-up" aria-label="Good response" aria-pressed="false" title="Good response"><svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M7 11V21H4V11H7ZM7 11L11 3C12.1 3 13 3.9 13 5V9H18.3C19.5 9 20.4 10.1 20.1 11.3L18.6 18.3C18.4 19.3 17.5 20 16.5 20H10C8.3 20 7 18.7 7 17" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg></button>
        <button type="button" class="rih-ctrl-down" aria-label="Poor response" aria-pressed="false" title="Poor response"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="transform:rotate(180deg)"><path d="M7 11V21H4V11H7ZM7 11L11 3C12.1 3 13 3.9 13 5V9H18.3C19.5 9 20.4 10.1 20.1 11.3L18.6 18.3C18.4 19.3 17.5 20 16.5 20H10C8.3 20 7 18.7 7 17" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg></button>
        <button type="button" class="rih-ctrl-regen" aria-label="Regenerate response" title="Regenerate"><svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M20 11A8 8 0 1 0 18.5 15.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><path d="M20 5V11H14" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
      `;
      wrap.querySelector('.rih-ctrl-copy').addEventListener('click', (e) => this._copyMessage(m, e.currentTarget));
      wrap.querySelector('.rih-ctrl-up').addEventListener('click', (e) => this._setFeedback(m, 'up', e.currentTarget));
      wrap.querySelector('.rih-ctrl-down').addEventListener('click', (e) => this._setFeedback(m, 'down', e.currentTarget));
      wrap.querySelector('.rih-ctrl-regen').addEventListener('click', () => this._regenerate(m));
      return wrap;
    }

    _copyMessage(m, btn) {
      const plain = m.content.replace(/\{\{cite:\d+\}\}/g, '').replace(/[*#]/g, '');
      navigator.clipboard?.writeText(plain).then(() => {
        this._announce('Response copied.');
        const original = btn.innerHTML;
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 12L10 17L19 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        setTimeout(() => { btn.innerHTML = original; }, 1300);
      });
    }

    _setFeedback(m, value, btn) {
      const msgEl = btn.closest('.rih-msg');
      const controls = btn.closest('.rih-controls');
      m.feedback = m.feedback === value ? null : value;
      controls.querySelector('.rih-ctrl-up').classList.toggle('is-active up', m.feedback === 'up');
      controls.querySelector('.rih-ctrl-down').classList.toggle('is-active down', m.feedback === 'down');
      msgEl.querySelector('.rih-chips')?.remove();
      msgEl.querySelector('.rih-thanks')?.remove();
      if (m.feedback === 'down') {
        const chips = document.createElement('div');
        chips.className = 'rih-chips';
        chips.innerHTML = ['Inaccurate', 'Missing sources', 'Hard to understand', 'Other']
          .map((l) => `<button type="button" class="rih-chip">${l}</button>`).join('');
        chips.addEventListener('click', (e) => {
          if (!e.target.closest('.rih-chip')) return;
          chips.remove();
          const t = document.createElement('p');
          t.className = 'rih-thanks';
          t.textContent = 'Thanks — noted.';
          controls.insertAdjacentElement('afterend', t);
          this._announce('Feedback submitted.');
        });
        controls.insertAdjacentElement('afterend', chips);
      }
      this._saveState();
    }

    async _regenerate(assistantMsg) {
      const idx = this.messages.findIndex((mm) => mm.id === assistantMsg.id);
      const priorUser = [...this.messages.slice(0, idx)].reverse().find((mm) => mm.role === 'user');
      if (!priorUser) return;
      assistantMsg.content = '';
      assistantMsg.sources = null;
      assistantMsg.feedback = null;
      this._renderThread();
      await this._streamInto(assistantMsg, priorUser.content);
    }

    _handleThreadClick(e) {
      const marker = e.target.closest('.rih-cite');
      if (!marker) return;
      const msgId = marker.dataset.msgId;
      const idx = marker.dataset.citeIndex;
      this._toggleSources(msgId, true);
      const card = this.shadow.querySelector(`#rih-source-${msgId}-${idx}`);
      if (card) {
        setTimeout(() => {
          card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
          card.classList.add('is-highlighted');
          setTimeout(() => card.classList.remove('is-highlighted'), 1500);
          this._announce(`Highlighted source ${parseInt(idx, 10) + 1}.`);
        }, 50);
      }
    }

    /* ---------------- sending / streaming ---------------- */

    async sendMessage(text) {
      const value = (text ?? this.inputEl.value).trim();
      if (!value) return;
      const userMsg = msg('user', value, Date.now());
      this.messages.push(userMsg);
      this.inputEl.value = '';
      this._autoResize();
      this._renderThread();
      this._scrollToBottom(true);

      const assistantMsg = msg('assistant', '', Date.now(), null);
      this.messages.push(assistantMsg);
      this._renderThread();
      await this._streamInto(assistantMsg, value);
    }

    async _streamInto(assistantMsg, queryText) {
      this.sendEl.classList.add('is-loading');
      this.sendEl.disabled = true;
      this.activeAbort = new AbortController();

      const findBubble = () => this.shadow.querySelector(`.rih-msg[data-id="${assistantMsg.id}"] .rih-bubble`);
      let bubble = findBubble();
      if (bubble) bubble.innerHTML = '<span class="rih-typing"><span></span><span></span><span></span></span>';

      let firstChunk = true;
      let raw = '';
      try {
        for await (const evt of API.streamAsk(queryText, this.conversationId, this.activeAbort.signal)) {
          if (evt.type === 'chunk') {
            if (firstChunk) { firstChunk = false; bubble = findBubble(); if (bubble) bubble.innerHTML = ''; }
            raw += evt.text;
            assistantMsg.content = raw;
            if (bubble) bubble.innerHTML = renderMarkdown(raw) + '<span class="rih-cursor"></span>';
            if (!this.userScrolledAway) this._scrollToBottom();
          } else if (evt.type === 'sources') {
            assistantMsg.sources = evt.sources;
          }
        }
        assistantMsg.content = raw;
        this._renderThread();
        if (!this.userScrolledAway) this._scrollToBottom();
      } catch (err) {
        if (err.name === 'AbortError') return;
        this._renderError(assistantMsg, queryText, err);
      } finally {
        this.sendEl.classList.remove('is-loading');
        this.sendEl.disabled = false;
        this.activeAbort = null;
        this._saveState();
      }
    }

    _renderError(assistantMsg, queryText, err) {
      this.messages = this.messages.filter((m) => m.id !== assistantMsg.id);
      this._renderThread();
      const card = document.createElement('div');
      card.className = 'rih-error';
      card.innerHTML = `
        <div>
          <p class="rih-error__title">Couldn't reach the assistant</p>
          <p class="rih-error__desc">${escapeHtml(err.message || 'The request failed.')} Check the connection and try again.</p>
          <button type="button" class="rih-retry">Retry</button>
        </div>
      `;
      card.querySelector('.rih-retry').addEventListener('click', () => {
        card.remove();
        const retryMsg = msg('assistant', '', Date.now(), null);
        this.messages.push(retryMsg);
        this._renderThread();
        this._streamInto(retryMsg, queryText);
      });
      this.threadEl.appendChild(card);
      this._scrollToBottom(true);
    }
  }

  /* =========================================================================
     INIT
     ========================================================================= */

  function init() {
    if (window.__raginsureaiWidget) return; // guard against double-init
    window.__raginsureaiWidget = new RagInsureAIWidget();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
