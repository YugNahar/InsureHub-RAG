FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only torch first to prevent pip from pulling CUDA packages
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# /app/app/  — Yug's production chat API modules (api.py, conversation_agent.py, etc.)
# /app/      — teammate's evaluation entry point (eval_api.py, eval_frontend.html)
COPY app/ ./app/
COPY eval_api.py eval_frontend.html ./

# Both services listen on different ports; docker-compose overrides CMD per service.
# Default CMD runs the eval service (backward-compatible with old single-service setup).
EXPOSE 8001 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail http://localhost:8001/eval/health || curl --fail http://localhost:8501/health || exit 1

CMD ["uvicorn", "eval_api:app", "--host", "0.0.0.0", "--port", "8001"]
