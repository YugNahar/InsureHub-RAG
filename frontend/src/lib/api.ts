// When served from the backend directly (any *.trycloudflare.com tunnel or localhost)
// use relative "" so chat works even after a tunnel restart — no hardcoded URL to go stale.
// When served from Vercel (different origin), use the baked VITE_API_BASE_URL to reach
// the tunnel backend. The restart script updates that env var and redeploys Vercel.
function _resolveApiUrl(): string {
  const baked =
    (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ||
    (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ||
    "";
  if (typeof window !== "undefined") {
    const h = window.location.hostname;
    if (h === "localhost" || h === "127.0.0.1" || h.endsWith(".trycloudflare.com")) {
      return "";
    }
  }
  return baked;
}
export const API_URL: string = _resolveApiUrl();

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem("insurehub_token");
}

export function setToken(token: string) {
  window.localStorage.setItem("insurehub_token", token);
}

export function clearToken() {
  window.localStorage.removeItem("insurehub_token");
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed (${res.status})`);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

export interface StreamMeta {
  sources: string[];
  needs_human: boolean;
  offline_escalated: boolean;
  corrected_text?: string;
  suggested_questions?: string[];
}

/** Derive WebSocket base URL from the current page origin when no API_URL is set. */
export function getWsUrl(): string {
  if (!API_URL) {
    const proto = typeof location !== "undefined" && location.protocol === "https:" ? "wss" : "ws";
    const host  = typeof location !== "undefined" ? location.host : "localhost:8501";
    return `${proto}://${host}`;
  }
  return API_URL.replace(/^https/, "wss").replace(/^http(?!s)/, "ws");
}

/**
 * Stream a POST to /ask-stream.
 * Calls onToken for each text chunk, then onDone with sources + needs_human flag.
 */
export async function apiStream(
  question: string,
  sessionId: string,
  onToken: (token: string) => void,
  onDone: (meta: StreamMeta) => void,
  onError: (msg: string) => void,
) {
  const headers = new Headers({ "Content-Type": "application/json" });
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${API_URL}/ask-stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({ question, session_id: sessionId }),
    });
  } catch {
    onError("Cannot reach the server. Is the backend running?");
    return;
  }

  if (!res.ok || !res.body) {
    onError(`Server error (${res.status})`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // The backend appends a final JSON line: {"sources":[...],"done":true}
    const jsonStart = buffer.lastIndexOf('\n\n{"sources"');
    if (jsonStart !== -1) {
      const textPart = buffer.slice(0, jsonStart);
      const jsonPart = buffer.slice(jsonStart + 2);
      if (textPart) onToken(textPart);
      try {
        const meta = JSON.parse(jsonPart);
        onDone({
          sources: meta.sources ?? [],
          needs_human: meta.needs_human ?? false,
          offline_escalated: meta.offline_escalated ?? false,
          corrected_text: meta.corrected_text,
          suggested_questions: meta.suggested_questions,
        });
      } catch {
        onDone({ sources: [], needs_human: false, offline_escalated: false });
      }
      return;
    }

    onToken(buffer);
    buffer = "";
  }

  if (buffer) onToken(buffer);
  onDone({ sources: [], needs_human: false, offline_escalated: false });
}
