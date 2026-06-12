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


COPY app/ .

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8502"]
