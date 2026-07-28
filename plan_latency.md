# Plan — Cut RAG answer latency toward the 5–6s target

**Author:** Opus 4.8 (planning only — no code here)
**Implementer:** Sonnet (phase by phase)
**Reviewer:** Opus 4.8 (after each phase)
**Status:** all decisions settled (see §0) — ready to build
**Scope:** `RAG_InsureAI/app/` (`turbovec_store.py`, `context_compressor.py`, `multi_source_rag.py`) + deployment config (`.env`, `docker-compose.yml`/Dockerfile). No frontend changes for latency itself.

**Named `plan_latency.md`, NOT `plan.md`, on purpose:** the existing `plan.md` is the cross-topic-contamination plan and its Phase 1 is mid-execution. Do not overwrite it.

---

## 0. Decisions — ALL SETTLED (do not re-litigate)

1. **Groq generation: REJECTED.** Groq refuses genuine, answerable queries (four prior attempts, memory `groq_4th_attempt_confirms_avoid`). Generation never routes to Groq.
2. **App-server GPU: YES.** The dev machine is a Mac (CPU-only — why every §1 measurement is CPU), but the **production server running the `insurehub_api` container has a GPU.** The reranker, embedder, and compressor move onto it.
3. **Answer generation → GPU: YES.** Generation **relocates off the slow remote vLLM host** (`123.253.124.14:7000`, ~7–8 tok/s) **onto a vLLM instance on the GPU server.** This is the decisive change: a 7B AWQ model on a real GPU runs at tens of tok/s (vs 7–8), so the 40–65s generation phase collapses.
4. **Output-length / prompt shaping: YES**, as a secondary quality+latency polish.

**What this means for the target:** with **both** machines on GPU, the two dominant costs fall together — the CPU reranker/compressor **dead air before the first token** (6–11s + 5–15s spikes) *and* the **generation phase** (40–65s). This is what makes 5–6s reachable for **detailed** answers, not just brief ones. Everything below is sequenced around that.

**Hard constraint on verification:** the GPU only exists on the server. **None of the GPU wins can be measured on the dev Mac.** Every GPU change must be (a) provably a no-op / still-correct on CPU so the Mac keeps working, and (b) measured for speed **on the server deploy**. Do not "verify" GPU latency locally and assume.

---

## 1. Evidence map (measured + code)

### 1a. Measured — live TIMING logs (detailed, `Explain X insurance in detail`)

> ⚠️ Contention caveat: captured while the 56-case contamination sweep hammered the same single remote vLLM host and the same 6 CPU cores. Structure/proportions are real; absolute totals are an upper bound. **Phase 0 re-baselines cleanly first.**

| query | total | retrieval | grounding | llm | promptbuild | other |
|---|---|---|---|---|---|---|
| engineering …detail | 67961 | 7046 | 1008 | **58388** | 106 | 1519 |
| liability …detail | 61216 | 6731 | 1015 | **47320** | 5557 | 6150 |
| life …detail | 65184 | 7706 | 1433 | 40497 | **14954** | 15548 |
| marine …detail | 59459 | 11295 | 1515 | 33362 | **12217** | 13287 |
| crop …detail | 73841 | **18766** | 999 | 41343 | 11197 | 12733 |
| motor …detail | 48054 | 8333 | 557 | 38306 | 218 | 858 |
| health …detail | 48335 | 10448 | 973 | 30949 | 5033 | 5965 |

Brief (`What is X?`): total **17000–31000ms**, `llm` 6800–18000, `retrieval` 6100–7100, `promptbuild` 150–5200.

**Reading (all four costs now GPU-addressable):**
- `llm` 30–58s — remote vLLM at 7–8 tok/s → **fixed by Decision 3 (GPU-hosted vLLM).**
- `retrieval` flat 6–11s — CPU cross-encoder rerank → **fixed by Decision 2 (app-server GPU).**
- `promptbuild` bimodal, 5–15s cold — CPU sentence-embedding in the compressor → **fixed by Decision 2.**
- `grounding` 0.3–1.5s — vLLM YES/NO call → also faster once generation vLLM is on GPU (same host).
- (`other = total − retrieval − grounding − llm`, so `promptbuild`/`preprocess`/`postllm` live inside `other`, `:7746`.)

### 1b. Code facts

| # | Fact | Location |
|---|------|----------|
| A | Generation streams via `VLLM_HOST`; model auto-resolved from `/v1/models`; detailed cap 512 tok, brief 300 | `router.py:_resolve_vllm_model/get_insurance_llm`; `multi_source_rag.py:5819,5831`; `.env` |
| B | App container loads BGE embedder + `bge-reranker-base` with **no explicit device** → CPU even where a GPU exists | `turbovec_store.py:83` (`SentenceTransformer(...)`), `:111` (`CrossEncoder(RERANKER_MODEL_NAME)`) |
| C | The **entire** app→generation coupling is the `VLLM_HOST`/`VLLM_MODEL` env + auto-resolve — so relocating generation to a GPU-hosted vLLM is **mostly ops (stand up vLLM on the GPU box, repoint `VLLM_HOST`), not app-code** | `router.py:30-36,157-189`; streaming uses the same host at `:5819` |
| D | Reranker: merged pool ~20–40 candidates × up to 2 windows = 40–80 forward passes/query | `_retrieve_all_sources_combined :3520`, `rerank_documents :990`, `_RERANK_MAX_WINDOWS=2 :165` |
| E | Compressor fires only when `sum(chunk chars) > budget`; then CPU-embeds sentences; FIFO cache(3000) keyed by chunk text; cold/churn = the 5–15s spikes; **reuses the shared embedder → inherits its device automatically** | `context_compressor.py:172,182,293–309`; embedder from `_get_shared_embed_model` |
| F | Streaming is real end-to-end → **TTFT = preprocess+retrieval+grounding+promptbuild** | `:5832 stream:True`; frontend `app.js streamAsk` |
| G | Contamination-plan Phase 1 (just landed, uncommitted) window-trims generation context at `:5491` — **overlaps the compressor**, may make its expensive branch rarely fire now | `multi_source_rag.py:5474–5491` |
| H | `context_compressor.py` has an **uncommitted** dead-code removal (`compress()` + `_MIN_COMPRESSED_CHARS`) — confirm intended | `git diff HEAD context_compressor.py` |

### 1c. Compressor "drops important parts" = correctness bug, tracked separately from speed

Two prior fixes exist for this shape — top-sentence-guarantee (`:337–342`), metadata-prefix sentence enrichment (`:280`), both from the "broken hand" exclusion-clause drops. Remaining drop risk **and its interaction with window-scoping (fact G)** is a Phase-2 correctness question to *measure*, not assume.

---

## 2. Design principles

1. **Measure clean before touching anything** — §1a is contention-inflated; a warm single request on an idle backend is the only trustworthy baseline (Phase 0).
2. **GPU wins are server-only.** Device changes must be a no-op / still-correct on the CPU Mac and measured for speed on the server. Budget a server deploy+measure step for every GPU phase.
3. **Separate the two machines when reasoning.** Decision 2 (app-server GPU) moves TTFT; Decision 3 (GPU vLLM) moves the generation phase. Report both `TTFT` and `total`.
4. **Never trade correctness for speed silently** — every change re-runs the contamination corpus (`contamination_corpus_runner.py`) **and** a grounding/recall spot-set. Faster+wronger is a failure.
5. **One change at a time, re-measured.**

---

## 3. The plan — phased

### Phase 0 — Clean latency baseline (build/run FIRST)

- Fixed 6-query set (2 brief, 2 detailed, 1 follow-up, 1 refusal), **idle** backend, cache purged (`query_kv_cache.json`), each query **warm** (run twice, take the 2nd).
- Capture the existing TIMING line **plus an explicit TTFT** (timestamp at first streamed token — add if not present).
- Baseline **twice**: once on the **CPU Mac** (honest "before" for reranker/compressor) and, once the GPU server is available, once there **pre-change** — so the GPU improvement is measured against the server's own CPU baseline, not the Mac's.

**Exit:** committed clean per-phase + TTFT baseline table.

### Phase 1 — Move all compute to GPU (the decisive phase; verified on the server)

**1a. App-server GPU for embedder + reranker (+ compressor for free).**
- Add device auto-detect (`cuda` if available else `cpu`) at the two shared-model load sites: `SentenceTransformer(..., device=...)` (`turbovec_store.py:83`) and `CrossEncoder(RERANKER_MODEL_NAME, device=...)` (`:111`). The compressor reuses the shared embedder (fact E) → GPU automatically, no separate change.
- Env-overridable (`MODEL_DEVICE`), **default auto-detect** so a fresh GPU deploy just works and the Mac stays CPU.
- Ensure the container image ships **CUDA-enabled torch** and the runtime exposes the GPU (`--gpus all` / compose `deploy.resources.reservations.devices`). Flag for the server deploy; remember a `docker compose up` recreate can wipe `docker cp`'d files (memory `feedback_verify_local_file_matches_committed`).

**1b. Relocate generation to a GPU-hosted vLLM (mostly ops, per fact C).**
- Stand up vLLM serving the 7B AWQ model (or a better-fitting model if GPU headroom allows) on the GPU server; repoint `VLLM_HOST` (and `VLLM_MODEL` if changed) at it. The app already routes generation, grounding, and query-classification through this env + `_resolve_vllm_model()` — **little to no app-code change**, but verify: model auto-resolves, streaming works, and (if a fresh vLLM) guided-decoding/tool-call flags are set consistently with what the code expects (the old host rejected tool-calling — a fresh GPU vLLM is a chance to configure this cleanly).
- Because grounding and preprocess LLM calls hit the same host, they speed up too.

**Verification (on the server):** re-run the Phase-0 set — expect `retrieval`, cold `promptbuild`, `grounding`, and `llm` to all drop sharply. Re-run the **contamination corpus** on the GPU stack — GPU vs CPU inference should give near-identical rankings/answers; confirm no borderline reranking flip changed a retrieval outcome, and no generation-quality regression from any model change.

**Exit:** on the GPU server, TTFT → ~1–3s and detailed `total` down to single-digit / low-double-digit seconds, with **zero** contamination-corpus regression; on the Mac, unchanged (CPU, still correct).

### Phase 2 — Code polish (secondary; do after the GPU baseline shows what's actually left)

Only pursue the pieces the Phase-1 GPU numbers show still matter — GPU may already dissolve most of this.
- **2a.** Confirm the uncommitted compressor dead-code removal (fact H) is intended.
- **2b. Compressor × windowing (fact G):** measure how often `compress_to_budget` still trips `total > max_total_chars` now that generation context is window-trimmed. If rare post-windowing, simplify rather than add machinery.
- **2c. Compressor correctness (the "dropping important parts" report):** on a grounded-accuracy spot-set, verify the answer-bearing sentence survives, including the window-scoping interaction. Distinct exit criterion from speed — this one matters **regardless of GPU** (it's about *what* gets kept, not *how fast*).
- **2d. Reranker pool width** (`_doc_top_k`/`_media_top_k`, `:4444/:4453`): trim only if the GPU numbers still show reranking on the critical path, gated on the contamination corpus + recall spot-set. **Do not touch `_RERANK_MAX_WINDOWS` 2→1** unless recall is explicitly re-verified.

**Exit:** compressor correctness confirmed (no dropped answer sentences); any further trim passes the contamination corpus; no change kept unless it moves the GPU baseline.

### Phase 3 — Output-length / prompt shaping (user-approved)

Even with fast GPU generation, tighter answers are faster and better:
- Detailed cap `VLLM_MAX_TOKENS=512` (`:5831`) — with GPU generation, latency pressure eases, so this becomes mostly about **quality/conciseness** (cut padding/filler, tighten point count) rather than a hard latency lever. Keep answers genuinely complete; avoid the mid-generation truncation the `:5825` comment warns about.
- Measure on the Phase-0 detailed queries: no truncated answers, no quality regression.

**Exit:** detailed answers tighter/complete, `total` unchanged-or-better, no quality regression.

### Phase 4 — Lock it in

- Fold Phase-0 into a repeatable latency command (sibling to `contamination_corpus_runner.py`), runnable on Mac (CPU) and server (GPU), emitting per-phase + TTFT.
- Document knobs (device flag, GPU vLLM host/model, pool width, compressor strategy, token caps) with measured before/after.
- Memory note: latency is a measured per-phase budget on a fixed set; regressions must move the number; the GPU stack is the production target and the Mac is CPU-only.

---

## 4. Files likely touched (for Sonnet)

| File | Change |
|------|--------|
| `turbovec_store.py` | **1a:** device auto-detect for `SentenceTransformer` (`:83`) + `CrossEncoder` (`:111`), env-overridable, default auto. **2d:** pool width only if still needed |
| `.env` / `docker-compose.yml` / Dockerfile | **1a:** `MODEL_DEVICE`, CUDA torch in image, GPU exposed to container. **1b:** `VLLM_HOST`/`VLLM_MODEL` → GPU-hosted vLLM |
| `context_compressor.py` | **2a/2b/2c:** confirm dead-code removal; measure/simplify post-windowing; verify no dropped answer sentences |
| `multi_source_rag.py` | **3:** token caps `:5831` + prompt tightening; **0/4:** TTFT instrumentation |
| new: latency runner under `RAG_InsureAI/` | Phase 0/4 harness |

---

## 5. Decision points

All primary decisions are settled (§0). Remaining tuning choices, decide from measured data during the phases:
1. **Model choice on the GPU vLLM (Phase 1b):** keep `Qwen2.5-7B-Instruct-AWQ`, or use GPU headroom for a stronger model? Bigger = better answers but slower/more memory. *Recommendation: keep the current model first (isolates the GPU speedup from a model change), evaluate an upgrade separately against the contamination corpus.*
2. **Output length aggressiveness (Phase 3):** *Recommendation: trim padding first; only lower the hard cap if answers still complete.*
3. **Reranker pool trim (Phase 2d):** only if GPU numbers still show it on the critical path; gate on the corpus.

---

## 6. Risks & honest tradeoffs

- **GPU wins are unverifiable on the dev Mac** — device auto-detect must be provably a no-op on CPU, and every speed claim measured on the server. Budget the server deploy+measure into each GPU phase.
- **CUDA vs CPU inference differs numerically** — tiny score deltas could flip a borderline reranking. Low risk, but re-run the contamination corpus on the GPU stack, don't assume identical to CPU.
- **Relocating the vLLM host (1b) is an ops change with app-visible failure modes** — model auto-resolve, streaming SSE format, guided-decoding/tool-call config. Verify generation, grounding, and query-classification all work against the new host, not just that it answers once. Keep the old `VLLM_HOST` as a documented fallback until the GPU host is proven.
- **A model change on the new host is a separate variable** — don't bundle it with the relocation; prove the speedup on the same model first, then evaluate any model upgrade on its own.
- **Compressor correctness (2c) matters regardless of GPU** — the "broken hand" drop history is about *what* is kept; GPU makes it faster, not more correct. Verify the answer sentence survives.
- **Contention makes measurement lie** — §1a is inflated; Phase 0's clean baseline is mandatory before judging any change.

---

## 7. Orchestration

- Sonnet implements phase by phase, **Phase 0 first**. Explicit exit criteria per phase; do not advance until met.
- Phase 1 is verified for **speed on the server** and for **still-correct on the Mac (CPU)**. Every phase re-runs the contamination corpus + a grounded-accuracy spot-set, deploys via `docker cp`/`docker restart insurehub_api` (+ `frontend/dist` sync if any frontend touch), purges `query_kv_cache.json` before re-measuring.
- Definition of done: on the GPU server, brief answers a few seconds, detailed answers near the 5–6s target (both TTFT and generation now on GPU), TTFT low across all types; zero contamination/grounding regression; compressor no longer drops answer-bearing sentences; latency regression command wired and documented.
