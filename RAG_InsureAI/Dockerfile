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

# Install torch first, before requirements.txt, so pip doesn't resolve a
# second/conflicting torch off PyPI's default index. CPU-only by default
# to keep the image small on hosts without a GPU (most deployments of
# this app so far); pass --build-arg TORCH_DEVICE=cuda for a CUDA build
# on a host with an NVIDIA GPU + nvidia-container-toolkit — see
# docker-compose.gpu.yml, which sets this automatically.
ARG TORCH_DEVICE=cpu
RUN if [ "$TORCH_DEVICE" = "cuda" ]; then \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

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
