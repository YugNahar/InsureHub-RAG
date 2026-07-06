# InsureAI RAG Agent — Tech Stack Overview

## What It Does

A document-aware Q&A agent built for insurance policy analysis. Upload policy PDFs, Word files, spreadsheets, or URLs and ask natural language questions. The agent retrieves relevant chunks, grounds the answer in the source documents through a multi-layer validation pipeline, and cites the exact page and source for every fact.

---

## Architecture

```text
User / Frontend
      │
      ▼
FastAPI (REST API — port 8501)
      │
      ├── Document Ingestion → TurboVec Index (Vector Store)
      │                         └── 4-bit quantization, ~4GB RAM, no GPU
      │
      └── Query Pipeline
              ├── LLM Query Contextualization (resolves pronouns/references via _contextualize_query)
              ├── Hybrid Search (Dense via TurboVec + BM25)
              ├── Cross-Encoder Reranking
              ├── Lexical Coverage Checks (_context_covers_query, _enumeration_query_covered, _quoted_comparison_covered)
              ├── Semantic Grounding Check (_verify_grounding — LLM-based)
              └── vLLM / Groq (LLM Backend)
```

---

## Core Components

### LLM Backend (Dual-backend support)

| Component            | Detail                                                       |
| -------------------- | ------------------------------------------------------------ |
| **vLLM**             | High-throughput LLM inference server hosted remotely         |
| **Groq**             | Cloud LLM API — alternative to vLLM, no GPU needed on your side |
| **Default Model**    | `Qwen/Qwen2.5-7B-Instruct-AWQ` (vLLM) / `llama-3.3-70b-versatile` (Groq) |
| **Interface**        | OpenAI-compatible REST API (`/v1/chat/completions`)          |
| **Force Backend**    | `FORCE_BACKEND` env var pins to `vllm` or `groq`; auto-detected if unset |
| **Fallback**         | Short auxiliary calls (topic extraction, grounding, contextualization) use the **same** backend as the main generation via `_backend_completion()` |
| **LangChain**        | `langchain>=0.3.0`, `langchain-core>=0.3.0`, `langchain-openai>=0.2.0`, `langchain-anthropic>=0.2.0` |

### Vector Store & Retrieval

| Component                      | Detail                                                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **TurboVec (TurboQuantIndex)** | Ultra-fast ANN vector index with 4-bit quantization — ~4GB RAM instead of ~31GB. No GPU required. ARM-compatible. Air-gap safe. |
| **IdMapIndex**                 | TurboVec wrapper for O(1) external ID mapping — enables stable document IDs through deletes                                     |
| **BAAI/bge-base-en-v1.5**      | Sentence embedding model — converts text to vectors for semantic search (`sentence-transformers>=3.0.0`)                        |
| **BM25 (rank-bm25)**           | Keyword-based retrieval — complements dense search for exact term matching                                                      |
| **Hybrid Search**              | Merges dense (TurboVec semantic) + BM25 (keyword) results for better recall                                                     |
| **Cross-Encoder Reranker**     | `BAAI/bge-reranker-base` — re-scores candidates to surface the most relevant chunks                                             |

### Query Pipeline (Multi-layer Grounding)

| Stage                          | Detail                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------- |
| **Short Followup Handling**    | `_is_short_followup()` detects 1-4 word continuations; merges with last assistant turn |
| **Query Contextualization**    | `_contextualize_query()` — LLM-based pronoun/reference resolution on every turn. Regex fast-path skips LLM call for structurally standalone questions |
| **Keyword Follow-up Detection** | `_is_likely_followup()` — legacy keyword-based classifier (kept for side-by-side logging via `[CTX]` line) |
| **Intent Extraction**          | `_extract_intent_topics()` — fast LLM call extracting discriminating topic words for coverage checking |
| **Typo Correction**            | `_correct_typos()` — fuzzy-matches insurance-domain terms via rapidfuzz before retrieval |
| **Compound Word Re-joining**   | `_join_split_compounds()` — re-joins split prefixes (re, co, un, under, over, non) before typo correction |
| **Hybrid Search**              | Dense (TurboVec) + BM25 (keyword) merge                                          |
| **Cross-Encoder Reranking**    | BAAI/bge-reranker-base re-scores candidates                                      |
| **Context Compression**        | `ContextCompressor` — similarity-based sentence trimming when prompt exceeds budget |
| **LLM Topic Coverage**         | `_context_covers_query()` — AND-logic lexical check: every topic word must appear in at least one chunk |
| **Named Entity Enumeration**   | `_enumeration_query_covered()` — "which insurers" questions require a company name in the same chunk as topic words |
| **Quoted Comparison Check**    | `_quoted_comparison_covered()` — every quoted term must appear in retrieved context |
| **Semantic Grounding**         | `_verify_grounding()` — LLM is asked whether the specific question is answerable from only the provided context (fail-safe to False) |
| **KV Cache**                   | `kv_cache.py` — semantic cache with intent-aware keying (detailed/simple/example flags); related entries provide supplementary context |
| **Stage-1 Summary Boost**      | Summary-store search identifies which documents are relevant; fetches extra chunks from those documents before reranking |
| **Suggestion Generation**      | `_generate_suggestions()` — LLM produces follow-up chip questions grounded in the answer text; verified against context before serving |

### Conversation Intelligence

| Feature                     | Detail                                                                          |
| --------------------------- | ------------------------------------------------------------------------------- |
| **SMALL_TALK intent**       | Detects greetings, thanks, and chit-chat; answers directly without retrieval    |
| **User-statement fast path** | "I have a health plan" → acknowledges warmly, no retrieval needed               |
| **PURE_CONV fast path**     | "yes", "ok", "thanks" → direct conversational reply, skips entire pipeline      |
| **Three-way prompt selection** | Chooses between `STRICT_GROUNDED`, `DETAILED_GROUNDED`, and `CONVERSATIONAL_RAG` based on query type and user preference |
| **Modifier-signal injection** | Detects example/detail/simple modifiers on follow-ups and injects targeted instructions into the LLM prompt |
| **Off-topic detection**     | Identifies out-of-domain questions and responds with a graceful fallback         |
| **Rule 4 fallback strip**   | Detects when the LLM appends a canned "I don't have that info" after real content; strips it or discards the whole answer based on reranker confidence |
| **Truncation detection**    | If the stream hits max_tokens mid-sentence, trims to the last complete sentence |
| **Hard sentence cap**       | Conversational mode enforces a 4-sentence maximum regardless of model compliance |

### API Layer

| Component           | Detail                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **FastAPI**         | REST API framework (`fastapi>=0.115.0`) — exposes all endpoints                                                                  |
| **Uvicorn**         | ASGI server running FastAPI (`uvicorn[standard]>=0.30.0`)                                                                      |
| **aiohttp**         | Async HTTP client for URL fetching and direct LLM streaming (bypasses LangChain's buffered astream) (`aiohttp>=3.10.0`)          |
| **Streaming**       | `/ask-stream` streams tokens directly from vLLM/Groq SSE endpoint — first word in <1 second                                     |
| **Async Job Queue** | `/upload` and `/ingest-url` return a `job_id` immediately; status polled via `GET /upload/{job_id}`                             |
| **Auth Middleware** | `python-jose>=3.3.0` JWT tokens + `passlib[bcrypt]>=1.7.4` — `/auth/login` issues tokens, `require_auth` dependency guards protected endpoints |

### Document Ingestion

| Format                      | Library Used                                                                        |
| --------------------------- | ----------------------------------------------------------------------------------- |
| **PDF**                     | `pdfplumber>=0.11.4` (fast, text-based) → `pypdf>=5.0.0` fallback → `Docling` OCR for scanned PDFs |
| **Word (.docx/.doc)**       | `python-docx>=1.1.0` → Docling fallback                                            |
| **Excel (.xlsx/.xls)**      | `openpyxl>=3.1.2` / `pandas>=2.2.0`                                                |
| **PowerPoint (.pptx/.ppt)** | `python-pptx>=0.6.23`                                                               |
| **CSV**                     | `pandas>=2.2.0`                                                                     |
| **Plain Text / EML**        | Built-in Python                                                                     |
| **Web URLs**                | `trafilatura>=2.0.0` + `readability-lxml` + `BeautifulSoup>=4.12.0` for HTML extraction |
| **YouTube**                 | `youtube-transcript-api>=0.6.0` + `yt-dlp>=2024.12.0` for video transcript and audio download |

### Human Handoff

| Component          | Detail                                                                                |
| ------------------ | ------------------------------------------------------------------------------------- |
| **Agent Dashboard** | Admin interface for live session monitoring and takeover                               |
| **WebSocket**       | Real-time bidirectional communication between user and agent during handoff            |
| **Session Management** | Track active chat sessions, transfer control, and restore context on takeover        |

### Multi-source RAG

| Source              | Library / Method                                                                     |
| ------------------- | ------------------------------------------------------------------------------------ |
| **Documents**        | PDF, Word, Excel, PowerPoint, CSV via `pdfplumber`, `python-docx`, `pandas`, etc.    |
| **Webpages**         | `trafilatura` + `readability-lxml` + `BeautifulSoup` for HTML extraction             |
| **YouTube**          | `yt-dlp` + `youtube-transcript-api` for video transcript ingestion                   |

### Semantic Chunking

| Component             | Detail                                                                               |
| --------------------- | ------------------------------------------------------------------------------------ |
| **SemanticChunker**   | `semantic_chunker.py` — splits documents into meaningful chunks using embedding similarity breaks, not fixed token windows |
| **ContextCompressor** | `context_compressor.py` — compresses retrieved chunks to fit within the LLM's context window while preserving relevant sentences |

### Voice Transcription

| Component          | Detail                                                                                |
| ------------------ | ------------------------------------------------------------------------------------- |
| **OpenAI Whisper** | Local speech-to-text model (base) — transcribes `.webm`, `.wav`, `.mp3`, `.m4a` audio |

### Infrastructure

| Component          | Detail                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **Docker Compose** | Two services: `api` (port 8501) and `eval` (port 8002)                                          |
| **Remote GPU**     | vLLM LLM server deployed on a remote GPU server for inference                                    |
| **Volumes**        | TurboVec data, Whisper cache, HuggingFace cache, upload temp storage                            |
| **Python 3.11+**   | Base runtime                                                                                    |
| **Auth**           | JWT-based login via `/auth` — protects upload and delete endpoints                              |
| **WebSockets**     | `websockets>=13.0` for real-time bidirectional communication                                    |

### Evaluation

| Component          | Detail                                                                                |
| ------------------ | ------------------------------------------------------------------------------------- |
| **eval_api.py**    | Standalone evaluation server (port 8002) with RAGAS-based metrics                     |
| **eval_frontend.html** | Browser-based evaluation UI for comparing answers against ground truth             |

---

## Key Design Decisions

* **No cloud LLM dependency** — vLLM runs on a private GPU server; no data leaves your infrastructure to a third-party AI provider.
* **Groq as fallback/alternative** — When a local GPU isn't available, Groq's cloud API provides fast inference with the same OpenAI-compatible interface.
* **TurboVec as primary vector store** — TurboQuantIndex provides 4-bit quantized dense ANN storage, reducing memory from ~31GB to ~4GB with no GPU required and full ARM compatibility. Ideal for air-gapped and privacy-sensitive deployments.
* **Hybrid retrieval** — pure semantic search misses exact policy numbers and clause references; BM25 catches those.
* **Reranking** — retrieval returns many candidates; the cross-encoder reranker picks the most relevant ones before sending to the LLM, reducing hallucination.
* **Multi-layer grounding** — No single check is trusted alone. Lexical keyword overlap, named-entity verification, quoted-term comparison, and an LLM-based semantic grounding check all must pass before an answer is served.
* **LLM-based query contextualization** — Replaced keyword follow-up detection (`_is_likely_followup`) with a single `_contextualize_query()` call on every turn. A regex fast-path avoids the LLM call entirely for structurally standalone questions (no pronoun/reference tokens).
* **Split locking** — `asyncio.Lock` for async endpoint handlers, `threading.RLock` for background thread operations, preventing deadlocks in concurrent access.
* **Lazy Whisper load** — Whisper model is loaded on the first transcription request only, keeping initial startup fast.
* **Protected admin endpoints** — upload, delete, and ingest endpoints require a JWT token.
* **Public chatbot access** — the public `/ask` endpoint remains open so the Layla chatbot can operate without authentication.
* **Direct SSE streaming** — Bypasses LangChain's buffered `astream()` to call the backend's `/v1/chat/completions` endpoint with `stream=True` directly, enabling the first word to appear in <1 second.
* **Rule 4 safety strip** — When the LLM appends "Honestly, I don't have that specific info..." after generating real content, the system strips the fallback only when reranker confidence is high enough (>0.05) to trust the leading content.