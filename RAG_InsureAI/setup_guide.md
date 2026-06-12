# InsureAI — Setup Guide

This guide walks you through setting up InsureAI from scratch on your machine. Follow the steps in order and you should have everything running within 15–20 minutes (excluding first-time model downloads).

---

## What You're Setting Up

InsureAI is a local Q&A agent built for insurance documents. You upload your policy PDFs, Word files, or even YouTube links and URLs, and then ask it questions in plain English. It finds the relevant parts of your documents and answers based only on what's actually in them — no hallucinations, no guessing.

The system has two main pieces:
- **The app** — runs locally in Docker on your machine (port 8502)
- **The LLM server** — a remote vLLM server that does the actual text generation (you need this to be running separately)

---

## Before You Start

Make sure the following are installed on your machine:

**Docker Desktop**
Download from docker.com. During installation, make sure WSL2 integration is enabled (Windows will prompt you). After install, open Docker Desktop and wait for it to fully start before continuing.

**Git** (optional, only needed if you're cloning from a repo)
Download from git-scm.com.

You'll also need at least 8 GB of RAM free and around 10 GB of disk space for the Docker image and model caches.

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
```

---

## Step 2 — Point It at Your vLLM Server

Open `docker-compose.yml` in a text editor. Find the `environment` section under the `api` service and update the vLLM host if yours is different:

```yaml
environment:
  - VLLM_HOST=http://123.253.124.14:7000
  - VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
  - EMBED_MODEL=BAAI/bge-base-en-v1.5
```

Before going further, confirm your vLLM server is actually reachable:

```
curl http://123.253.124.14:7000/v1/models
```

If you get back a JSON response listing models, you're good. If it times out or refuses the connection, the server is either down or blocked by a firewall — sort that out before continuing, otherwise the app will start but won't be able to answer any questions.

---

## Step 3 — Build and Start

Open a terminal, navigate to the project folder, and run:

```
docker compose up -d --build
```

The first time you run this it will take a while — it needs to download the base Python image, install all the packages, and on first startup it will also pull the embedding and reranker models from HuggingFace. Subsequent starts are much faster.

Once it finishes, verify everything is up:

```
docker compose ps
```

You should see the `rag_api` container listed with status `Up`. Then hit the health endpoint:

```
curl http://localhost:8502/health
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
curl -X POST http://localhost:8502/upload \
  -F "file=@/path/to/your/policy.pdf"
```

This returns immediately with a `job_id`. The actual processing happens in the background. Check progress with:

```
curl http://localhost:8502/upload/<job_id>
```

When `status` says `done`, the document is in the knowledge base and ready to query.

**Supported file types:**
PDF, Word (.docx/.doc), Excel (.xlsx/.xls), PowerPoint (.pptx/.ppt), CSV, plain text, and .eml files.

---

## Step 5 — Ask Questions

```
curl -X POST http://localhost:8502/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the hospitalization limit?", "session_id": "my-session"}'
```

The `session_id` is optional — if you include the same one across multiple questions, the system remembers the conversation context. Leave it out or use `"default"` if you don't need that.

---

## Adding Videos and Webpages

Beyond documents, you can also feed it YouTube video transcripts and webpages.

**Add a YouTube video:**
```
curl -X POST http://localhost:8502/upload-video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=..."}'
```

**Add a webpage:**
```
curl -X POST http://localhost:8502/upload-webpage \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/policy-page"}'
```

Once added, these are included in every `/ask` query alongside your documents.

---

## Managing the Knowledge Base

```
# See what's in the knowledge base
curl http://localhost:8502/docs

# Remove a specific document
curl -X DELETE "http://localhost:8502/docs/policy.pdf"

# Wipe everything and start fresh
curl -X DELETE http://localhost:8502/docs

# List uploaded videos
curl http://localhost:8502/videos

# Remove a video
curl -X DELETE "http://localhost:8502/videos/https://youtube.com/watch?v=..."
```

---

## Checking the API Docs

The full list of endpoints with request/response schemas is available at:

```
http://localhost:8502/swagger
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

## A Note on Hot-Reload

The `app/` folder is mounted directly into the container as a volume. This means if you edit any Python file in `app/`, uvicorn picks it up automatically within a couple of seconds — no rebuild needed. The only time you need to rebuild is when you add new Python packages to `requirements.txt` or change the `Dockerfile`.

---

## Troubleshooting

**Container exits immediately after starting**
Run `docker compose logs api` to see the error. The most common cause is port 8502 already being in use by something else. Change the port mapping in `docker-compose.yml` if that's the case.

**Answers are very slow on first query**
Normal. The embedding and reranker models are loaded into memory on the first request. After that, things speed up significantly.

**Getting "could not connect to model server" errors**
The vLLM server is unreachable. Test with `curl http://<vllm-host>:7000/v1/models` and check that the host/port in `docker-compose.yml` is correct.

**ChromaDB is empty after I restarted**
If you ran `docker compose down -v`, that deletes the volumes including your document store. Use `docker compose down` (without `-v`) to preserve data between restarts.

**Voice transcription isn't working**
Whisper downloads its model on first use — make sure the container has internet access. Supported audio formats are `.webm`, `.wav`, `.mp3`, and `.m4a`.

---

## Data Storage

All persistent data lives in Docker named volumes, not inside the container image:

| What | Where it's stored |
|------|------------------|
| Uploaded documents (ChromaDB) | `chroma_data` volume |
| Embedding + reranker models | `hf_cache` volume |
| Whisper model | `whisper_cache` volume |
| Temporary file uploads | `upload_data` volume |

These survive container restarts and rebuilds. They're only lost if you explicitly run `docker compose down -v`.
