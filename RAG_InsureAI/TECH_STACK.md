# InsureAI RAG Agent — Tech Stack Overview

## What It Does

A document-aware Q&A agent built for insurance policy analysis. Upload policy PDFs, Word files, spreadsheets, or URLs and ask natural language questions. The agent retrieves relevant chunks, grounds the answer in the source documents, and cites the exact page and source for every fact.

---

## Architecture

```text
User / Frontend
      │
      ▼
FastAPI (REST API — port 8502)
      │
      ├── Document Ingestion → TurboVec Index (Vector Store)
      │                         └── 4-bit quantization, ~4GB RAM, no GPU
      │
      └── Query Pipeline
              ├── HyDE Query Expansion
              ├── Hybrid Search (Dense via TurboVec + BM25)
              ├── Cross-Encoder Reranking
              └── vLLM (LLM Server — remote)
```

---

## Core Components

### LLM Backend

| Component            | Detail                                                       |
| -------------------- | ------------------------------------------------------------ |
| **vLLM**             | High-throughput LLM inference server hosted remotely         |
| **Model**            | `Qwen/Qwen2.5-7B-Instruct-AWQ` (quantized, fast)             |
| **Interface**        | OpenAI-compatible REST API (`/v1/chat/completions`)          |
| **LangChain OpenAI** | Python client that talks to vLLM using the OpenAI SDK format |

### Vector Store & Retrieval

| Component                      | Detail                                                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **TurboVec (TurboQuantIndex)** | Ultra-fast ANN vector index with 4-bit quantization — ~4GB RAM instead of ~31GB. No GPU required. ARM-compatible. Air-gap safe. |
| **IdMapIndex**                 | TurboVec wrapper for O(1) external ID mapping — enables stable document IDs through deletes                                     |
| **BAAI/bge-base-en-v1.5**      | Sentence embedding model — converts text to vectors for semantic search                                                         |
| **BM25 (rank-bm25)**           | Keyword-based retrieval — complements dense search for exact term matching                                                      |
| **Hybrid Search**              | Merges dense (TurboVec semantic) + BM25 (keyword) results for better recall                                                     |
| **Cross-Encoder Reranker**     | `BAAI/bge-reranker-base` — re-scores candidates to surface the most relevant chunks                                             |

### Query Pipeline

| Technique                | Detail                                                                          |
| ------------------------ | ------------------------------------------------------------------------------- |
| **HyDE**                 | Hypothetical Document Embeddings — generates a fake answer to improve retrieval |
| **Section Detection**    | Classifies each chunk (benefits, exclusions, claims, eligibility, etc.)         |
| **Intent Detection**     | Routes query to the right document sections based on keywords                   |
| **Document Routing**     | Narrows search to specific insurer documents (AIG, GIG, LIVA, RAK, etc.)        |
| **Grounding Validation** | Checks that figures in the LLM answer actually appear in the retrieved context  |
| **Citation Enforcement** | Every fact in the answer must include `[Source: document, Page X]`              |

### API Layer

| Component           | Detail                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **FastAPI**         | REST API framework — exposes all endpoints                                                                                      |
| **Uvicorn**         | ASGI server running FastAPI                                                                                                     |
| **aiohttp**         | Async HTTP client for URL fetching                                                                                              |
| **Streaming**       | `/ask-stream` and `/ask-url` stream answers token by token                                                                      |
| **Async Job Queue** | `/upload` and `/ingest-url` return a `job_id` immediately; status polled via `GET /upload/{job_id}`                             |
| **Auth Middleware** | `python-jose` JWT tokens + `passlib` bcrypt — `/auth/login` issues tokens, `require_auth` dependency guards protected endpoints |

### Document Ingestion

| Format                      | Library Used                                                                        |
| --------------------------- | ----------------------------------------------------------------------------------- |
| **PDF**                     | `pdfplumber` (fast, text-based) → `pypdf` fallback → `Docling` OCR for scanned PDFs |
| **Word (.docx/.doc)**       | `python-docx` → Docling fallback                                                    |
| **Excel (.xlsx/.xls)**      | `openpyxl` / `pandas`                                                               |
| **PowerPoint (.pptx/.ppt)** | `python-pptx`                                                                       |
| **CSV**                     | `pandas`                                                                            |
| **Plain Text / EML**        | Built-in Python                                                                     |
| **Web URLs**                | Jina Reader API → `readability-lxml` → `trafilatura` + `BeautifulSoup`              |
| **YouTube**                 | `youtube-transcript-api` — pulls transcript directly                                |

### Voice Transcription

| Component          | Detail                                                                                |
| ------------------ | ------------------------------------------------------------------------------------- |
| **OpenAI Whisper** | Local speech-to-text model (base) — transcribes `.webm`, `.wav`, `.mp3`, `.m4a` audio |

### Infrastructure

| Component          | Detail                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **Docker**         | Entire app runs in a container                                                                  |
| **Docker Compose** | Orchestrates the API container with volumes for TurboVec data, Whisper cache, HuggingFace cache |
| **Python 3.11**    | Base runtime                                                                                    |
| **Auth**           | JWT-based login via `/auth` — protects upload and delete endpoints                              |

---

## Key Design Decisions

* **No cloud LLM dependency** — vLLM runs on a private GPU server; no data leaves your infrastructure to a third-party AI provider.
* **TurboVec over ChromaDB** — switched to TurboQuantIndex for the dense ANN store. 4-bit quantization brings memory from ~31GB down to ~4GB, with no GPU required and full ARM compatibility. Ideal for air-gapped and privacy-sensitive deployments.
* **Hybrid retrieval** — pure semantic search misses exact policy numbers and clause references; BM25 catches those.
* **Reranking** — retrieval returns many candidates; the cross-encoder reranker picks the most relevant ones before sending to the LLM, reducing hallucination.
* **Grounding check** — after the LLM answers, the system verifies that every number in the answer exists in the retrieved context. Unverified figures trigger a warning.
* **TTL job cache** — background ingest jobs auto-expire after 1 hour to prevent memory leaks.
* **Split locking** — `asyncio.Lock` for async endpoint handlers, `threading.RLock` for background thread operations, preventing deadlocks in concurrent access.
* **Lazy Whisper load** — Whisper model is loaded on the first transcription request only, keeping initial startup fast.
* **Protected admin endpoints** — upload, delete, and ingest endpoints require a JWT token.
* **Public chatbot access** — the public `/ask` endpoint remains open so the Layla chatbot can operate without authentication.
