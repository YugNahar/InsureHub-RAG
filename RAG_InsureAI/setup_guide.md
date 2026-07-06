# InsureAI — Setup Guide

This guide walks you through setting up InsureAI from scratch on your machine. Follow the steps in order and you should have everything running within 15–20 minutes (excluding first-time model downloads).

---

## What You're Setting Up

InsureAI is a local Q&A agent built for insurance documents. You upload your policy PDFs, Word files, or even YouTube links and URLs, and then ask it questions in plain English. It finds the relevant parts of your documents and answers based only on what's actually in them — no hallucinations, no guessing.

The system has two main pieces:
- **The app** — runs locally in Docker on your machine (port 8501)
- **The LLM server** — a remote vLLM or Groq server that does the actual text generation (you need this to be running separately)

---

## Before You Start

Make sure the following are installed on your machine:

**Docker Desktop**
Download from docker.com. During installation, make sure WSL2 integration is enabled (Windows will prompt you). After install, open Docker Desktop and wait for it to fully start before continuing.

**Git** (optional, only needed if you're cloning from a repo)
Download from git-scm.com.

You'll also need at least 8 GB of RAM free and around 10 GB of disk space for the Docker image and model caches.

**Python 3.11+** (for local development without Docker)
Download from python.org. Verify with `python --version`.

---

## Step 1 — Get the Project Files

If you received the project as a ZIP, just extract it somewhere on your machine. If it's on a git repo:

```
git clone <repo-url> AIAgent
cd AIAgent
```

The folder should look like this once you have it:

```
AIAgent/
  app/
  docker-compose.yml
  Dockerfile
  requirements.txt
  render.yaml
  setup_guide.md
  TECH_STACK.md
```

---

## Step 1.5 — Create Your `.env` File

Create a file named `.env` in the project root (same directory as `docker-compose.yml`).

Example:

```env
# ── LLM Backend ────────────────────────────────────────────────────────────
# Either VLLM_HOST (for self-hosted vLLM) or GROQ_API_KEY (for Groq cloud)
# If both are set, FORCE_BACKEND=vllm or FORCE_BACKEND=groq picks which to use.
# If neither is set, defaults to groq if GROQ_API_KEY is present, else vllm.

VLLM_HOST=http://<your-vllm-server-ip>:7000
VLLM_API_KEY=not-needed-for-local
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
VLLM_MODEL_BRIEF=Qwen/Qwen2.5-7B-Instruct-AWQ
VLLM_MAX_TOKENS=600
VLLM_MAX_TOKENS_BRIEF=300

GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# ── Forced backend (optional) — uncomment to pin: "vllm" or "groq"
# FORCE_BACKEND=groq

# ── Embeddings & Re-ranking ─────────────────────────────────────────────────
EMBED_MODEL=BAAI/bge-base-en-v1.5
RERANK_MODEL=BAAI/bge-reranker-base

# ── Admin Authentication ────────────────────────────────────────────────────
ADMIN_USERNAME=admin
ADMIN_PASSWORD=yourpassword

AUTH_SECRET_KEY=pick-any-long-random-string
AUTH_TOKEN_EXPIRE_MINUTES=480
```

**Important:**

* Never commit this file to Git.
* Use a long random string for `AUTH_SECRET_KEY`.
* Change the default admin username and password before deploying to production.
* If using Groq, the `GROQ_MODEL` must be a model that supports function calling/JSON mode for structured extraction (e.g. `llama-3.3-70b-versatile` or `mixtral-8x7b-32768`).

---

## Step 2 — Configure the Backend Connection

Open `docker-compose.yml` and verify that the API service loads values from the `.env` file:

```yaml
environment:
  - VLLM_HOST=${VLLM_HOST}
  - VLLM_MODEL=${VLLM_MODEL}
  - VLLM_API_KEY=${VLLM_API_KEY}
  - GROQ_API_KEY=${GROQ_API_KEY}
  - GROQ_MODEL=${GROQ_MODEL}
  - FORCE_BACKEND=${FORCE_BACKEND}
  - EMBED_MODEL=${EMBED_MODEL}
  - RERANK_MODEL=${RERANK_MODEL}

  - ADMIN_USERNAME=${ADMIN_USERNAME}
  - ADMIN_PASSWORD=${ADMIN_PASSWORD}
  - AUTH_SECRET_KEY=${AUTH_SECRET_KEY}
  - AUTH_TOKEN_EXPIRE_MINUTES=${AUTH_TOKEN_EXPIRE_MINUTES}
```

Before continuing, verify the chosen backend is reachable:

**For vLLM:**
```bash
curl $VLLM_HOST/v1/models
```

**For Groq:**
```bash
curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
```

The API container can start without the LLM server, but document questions will fail until the model server becomes reachable.

---

## Step 3 — Build and Start

Open a terminal, navigate to the project folder, and run:

```
docker compose up -d --build
```

The first time you run this it will take a while — it needs to download the base Python image, install all the packages (including LangChain 0.3.x, FastAPI 0.115+, and sentence-transformers 3.x), and on first startup it will also pull the embedding and reranker models from HuggingFace. Subsequent starts are much faster.

Once it finishes, verify everything is up:

```
docker compose ps
```

You should see the `rag_api` container listed with status `Up`. Then hit the health endpoint:

```
curl http://localhost:8501/health
```

Expected response:
```json
{"status": "ok", "chunks": 0}
```

The `chunks` count will be 0 until you upload some documents, which is fine.

---

## Step 4 — Upload Your Documents

You can upload documents through the frontend UI, or directly via the API.

**Via API (curl):**

```
curl -X POST http://localhost:8501/upload \
  -F "file=@/path/to/your/policy.pdf"
```

This returns immediately with a `job_id`. The actual processing happens in the background. Check progress with:

```
curl http://localhost:8501/upload/<job_id>
```

When `status` says `done`, the document is in the knowledge base and ready to query.

**Supported file types:**
PDF, Word (.docx/.doc), Excel (.xlsx/.xls), PowerPoint (.pptx/.ppt), CSV, plain text, .txt, and .eml files.

---

## Step 5 — Ask Questions

```
curl -X POST http://localhost:8501/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the hospitalization limit?", "session_id": "my-session"}'
```

The `session_id` is optional — if you include the same one across multiple questions, the system remembers the conversation context. Leave it out or use `"default"` if you don't need that.

For streaming (token-by-token) responses:

```
curl -X POST http://localhost:8501/ask-stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the deductible?", "session_id": "my-session"}'
```

---

## Adding Videos and Webpages

Beyond documents, you can also feed it YouTube video transcripts and webpages.

**Add a YouTube video:**
```
curl -X POST http://localhost:8501/upload-video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=..."}'
```

**Add a webpage:**
```
curl -X POST http://localhost:8501/upload-webpage \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/policy-page"}'
```

Once added, these are included in every `/ask` query alongside your documents.

---

## Managing the Knowledge Base

```
# See what's in the knowledge base
curl http://localhost:8501/docs

# Remove a specific document
curl -X DELETE "http://localhost:8501/docs/policy.pdf"

# Wipe everything and start fresh
curl -X DELETE http://localhost:8501/docs

# List uploaded videos
curl http://localhost:8501/videos

# Remove a video
curl -X DELETE "http://localhost:8501/videos/https://youtube.com/watch?v=..."
```

---

## Checking the API Docs

The full list of endpoints with request/response schemas is available at:

```
http://localhost:8501/swagger
```

Open that in a browser and you can try every endpoint interactively without writing any curl commands.

---

## Day-to-Day Commands

```bash
# Start the app (after first setup)
docker compose up -d

# Stop the app
docker compose down

# Restart after changing a Python file (hot-reload handles most changes automatically)
docker compose restart api

# Rebuild after changing requirements.txt or Dockerfile
docker compose up -d --build

# Watch live logs
docker compose logs -f api
```

---

## Server Deployment

For production or remote GPU server deployments, start only the API service:

```bash
docker compose up -d api
```

This is useful when:
- The `eval` service is not needed in production
- You want to minimize resource usage on the deployment server
- The vLLM server runs on a separate remote GPU host

To start all services (API + eval), use the standard:

```bash
docker compose up -d
```

If you ran `docker compose down -v`, that deletes the volumes including your document store. Use `docker compose down` (without `-v`) to preserve data between restarts.

**Voice transcription isn't working**
Whisper downloads its model on first use — make sure the container has internet access. Supported audio formats are `.webm`, `.wav`, `.mp3`, and `.m4a`.

---

## Local Development (Without Docker)

For development or testing without Docker, you can run the API directly:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8501 --reload
```

The `--reload` flag enables auto-reload on code changes. Note: you'll need to set environment variables manually or use a `.env` file with `python-dotenv`.

---

## Data Storage

All persistent data lives in Docker named volumes, not inside the container image:

| What | Where it's stored |
|------|------------------|
| Uploaded documents (TurboVec index + metadata) | `turbovec_data` volume |
| Embedding + reranker models | `hf_cache` volume |
| Whisper model | `whisper_cache` volume |
| Temporary file uploads | `upload_data` volume |

These survive container restarts and rebuilds. They're only lost if you explicitly run `docker compose down -v`.

---

## Troubleshooting

**"I don't have that in my knowledge base" for a question that should be answerable**
This can happen if the query contextualization over-injects topic context. First verify the question is standalone (no pronouns like "it", "that", "their"). If it is, the system should return it unchanged. Check the logs for `[CTX]` entries to see if the query was being rewritten when it shouldn't.

**The LLM seems to be answering from general knowledge**
The system applies multiple grounding checks: lexical keyword overlap, named-entity enumeration verification, quoted-term comparison, and a semantic LLM-based grounding check (`_verify_grounding`). If any of these fail, the answer is refused. Check the application logs for coverage-check results.

**Reranker gate is blocking legitimate content**
The reranker gate threshold is set very low (0.0005 by default) to only block pure noise. If legitimate chunks are being dropped, check `RERANK_GATE_THRESHOLD` in the environment — the default should rarely need adjustment.

**Package version conflicts**
If you encounter dependency conflicts during `docker compose up --build`, ensure you're using Python 3.11 in the Docker base image. The current requirements target Python 3.11+ with updated package versions.

**Whisper model download fails**
Ensure the container has internet access on first run. Whisper downloads its model (~140MB for the base model) on first transcription request. If behind a proxy, set `HTTP_PROXY`/`HTTPS_PROXY` environment variables in docker-compose.yml.

**Import errors after updating requirements**
If you see module import errors after updating dependencies, rebuild the Docker image from scratch:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```
