# InsureAI RAG Agent — Tech Stack Overview

## What It Does
A document-aware Q&A agent built for insurance policy analysis. Upload policy PDFs, Word files, spreadsheets, or URLs and ask natural language questions. The agent retrieves relevant chunks, grounds the answer in the source documents, and cites the exact page and source for every fact.

---

## Architecture

```
User / Frontend
      │
      ▼
FastAPI (REST API — port 8502)
      │
      ├── Document Ingestion → ChromaDB (Vector Store)
      │
      └── Query Pipeline
              ├── HyDE Query Expansion
              ├── Hybrid Search (Dense + BM25)
              ├── Cross-Encoder Reranking
              └── vLLM (LLM Server — remote)
```

---

## Core Components

### LLM Backend
| Component | Detail |
|---|---|
| **vLLM** | High-throughput LLM inference server hosted remotely |
| **Model** | `Qwen/Qwen2.5-3B-Instruct-AWQ` (quantized, fast) |
| **Interface** | OpenAI-compatible REST API (`/v1/chat/completions`) |
| **LangChain OpenAI** | Python client that talks to vLLM using the OpenAI SDK format |

### Vector Store & Retrieval
| Component | Detail |
|---|---|
| **ChromaDB** | Persistent local vector database storing document chunks |
| **BAAI/bge-base-en-v1.5** | Sentence embedding model — converts text to vectors for semantic search |
| **BM25 (rank-bm25)** | Keyword-based retrieval — complements dense search for exact term matching |
| **Hybrid Search** | Merges dense (semantic) + BM25 (keyword) results for better recall |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-base` — re-scores candidates to surface the most relevant chunks |

### Query Pipeline
| Technique | Detail |
|---|---|
| **HyDE** | Hypothetical Document Embeddings — generates a fake answer to improve retrieval |
| **Section Detection** | Classifies each chunk (benefits, exclusions, claims, eligibility, etc.) |
| **Intent Detection** | Routes query to the right document sections based on keywords |
| **Document Routing** | Narrows search to specific insurer documents (AIG, GIG, LIVA, RAK, etc.) |
| **Grounding Validation** | Checks that figures in the LLM answer actually appear in the retrieved context |
| **Citation Enforcement** | Every fact in the answer must include `[Source: document, Page X]` |

### API Layer
| Component | Detail |
|---|---|
| **FastAPI** | REST API framework — exposes all endpoints |
| **Uvicorn** | ASGI server running FastAPI |
| **aiohttp** | Async HTTP client for URL fetching |
| **Streaming** | `/ask-stream` and `/ask-url` stream answers token by token |
| **Async Job Queue** | `/upload` and `/ingest-url` return a `job_id` immediately; status polled via `GET /upload/{job_id}` |

### Document Ingestion
| Format | Library Used |
|---|---|
| **PDF** | `pdfplumber` (fast, text-based) → `pypdf` fallback → `Docling` OCR for scanned PDFs |
| **Word (.docx/.doc)** | `python-docx` → Docling fallback |
| **Excel (.xlsx/.xls)** | `openpyxl` / `pandas` |
| **PowerPoint (.pptx/.ppt)** | `python-pptx` |
| **CSV** | `pandas` |
| **Plain Text / EML** | Built-in Python |
| **Web URLs** | Jina Reader API → `readability-lxml` → `trafilatura` + `BeautifulSoup` |
| **YouTube** | `youtube-transcript-api` — pulls transcript directly |

### Voice Transcription
| Component | Detail |
|---|---|
| **OpenAI Whisper** | Local speech-to-text model (base) — transcribes `.webm`, `.wav`, `.mp3`, `.m4a` audio |

### Infrastructure
| Component | Detail |
|---|---|
| **Docker** | Entire app runs in a container |
| **Docker Compose** | Orchestrates the API container with volumes for ChromaDB, Whisper cache, HuggingFace cache |
| **Python 3.11** | Base runtime |

---

## Key Design Decisions

- **No cloud LLM dependency** — vLLM runs on a private GPU server; no data leaves your infrastructure to a third-party AI provider.
- **Hybrid retrieval** — pure semantic search misses exact policy numbers and clause references; BM25 catches those.
- **Reranking** — retrieval returns many candidates; the cross-encoder reranker picks the most relevant ones before sending to the LLM, reducing hallucination.
- **Grounding check** — after the LLM answers, the system verifies that every number in the answer exists in the retrieved context. Unverified figures trigger a warning.
- **TTL job cache** — background ingest jobs auto-expire after 1 hour to prevent memory leaks.
- **Lifespan startup** — Whisper model is pre-loaded at startup so the first transcription request is instant.
