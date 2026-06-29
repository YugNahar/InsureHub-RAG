# InsureAI — Setup Guide

This guide walks you through setting up InsureAI from scratch on your machine. Follow the steps in order and you should have everything running within 15–20 minutes (excluding first-time model downloads).

---

## What You're Setting Up

InsureAI is a local Q&A agent built for insurance documents. You upload your policy PDFs, Word files, or even YouTube links and URLs, and then ask it questions in plain English. It finds the relevant parts of your documents and answers based only on what's actually in them — no hallucinations, no guessing.

The system has two main pieces:
- **The app** — runs locally in Docker on your machine (port 8501)
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
## Step 1.5 — Create Your `.env` File

Create a file named `.env` in the project root (same directory as `docker-compose.yml`).

Example:

```env
VLLM_HOST=http://<your-vllm-server-ip>:7000
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
EMBED_MODEL=BAAI/bge-base-en-v1.5

ADMIN_USERNAME=admin
ADMIN_PASSWORD=yourpassword

AUTH_SECRET_KEY=pick-any-long-random-string
AUTH_TOKEN_EXPIRE_MINUTES=480
```

**Important:**

* Never commit this file to Git.
* Use a long random string for `AUTH_SECRET_KEY`.
* Change the default admin username and password before deploying to production.

---

## Step 2 — Configure the vLLM Connection

Open `docker-compose.yml` and verify that the API service loads values from the `.env` file:

```yaml
environment:
  - VLLM_HOST=${VLLM_HOST}
  - VLLM_MODEL=${VLLM_MODEL}
  - EMBED_MODEL=${EMBED_MODEL}

  - ADMIN_USERNAME=${ADMIN_USERNAME}
  - ADMIN_PASSWORD=${ADMIN_PASSWORD}
  - AUTH_SECRET_KEY=${AUTH_SECRET_KEY}
  - AUTH_TOKEN_EXPIRE_MINUTES=${AUTH_TOKEN_EXPIRE_MINUTES}
```

Before continuing, verify the vLLM server is reachable:

```bash
curl $VLLM_HOST/v1/models
```

Expected response:

```json
{
  "data": [...]
}
```

If the request times out or returns a connection error, verify:

* The vLLM server is running.
* The host and port are correct.
* Firewall rules allow access to port `7000`.

The API container can start without the vLLM server, but document questions will fail until the model server becomes reachable.

```
```

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
PDF, Word (.docx/.doc), Excel (.xlsx/.xls), PowerPoint (.pptx/.ppt), CSV, plain text, and .eml files.

---

## Step 5 — Ask Questions

```
curl -X POST http://localhost:8501/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the hospitalization limit?", "session_id": "my-session"}'
```

The `session_id` is optional — if you include the same one across multiple questions, the system remembers the conversation context. Leave it out or use `"default"` if you don't need that.

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

## Data Storage

All persistent data lives in Docker named volumes, not inside the container image:

| What | Where it's stored |
|------|------------------|
| Uploaded documents (TurboVec index + metadata) | `turbovec_data` volume |
| Embedding + reranker models | `hf_cache` volume |
| Whisper model | `whisper_cache` volume |
| Temporary file uploads | `upload_data` volume |

These survive container restarts and rebuilds. They're only lost if you explicitly run `docker compose down -v`.