#!/usr/bin/env bash
# Run this script ONCE on the remote server (123.253.124.14) to deploy the backend.
# Usage: bash deploy-server.sh
set -e

REPO_URL="https://github.com/YugNahar/InsureHub-RAG.git"
REPO_DIR="$HOME/InsureHub-RAG"
COMPOSE_DIR="$REPO_DIR/RAG_InsureAI"

echo "==> Checking Docker..."
if ! command -v docker &>/dev/null; then
  echo "    Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  echo "    Docker installed. You may need to log out and back in once."
fi

if ! docker compose version &>/dev/null; then
  echo "    Installing Docker Compose plugin..."
  sudo apt-get install -y docker-compose-plugin 2>/dev/null || \
  sudo yum install -y docker-compose-plugin 2>/dev/null || true
fi

echo "==> Checking Node.js..."
if ! command -v node &>/dev/null; then
  echo "    Installing Node.js 20..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
  sudo apt-get install -y nodejs
fi

echo "==> Cloning / updating repo..."
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --rebase
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "==> Building frontend..."
cd "$REPO_DIR/frontend"
npm ci
npm run build
cd "$COMPOSE_DIR"

if [ ! -f "$COMPOSE_DIR/.env" ]; then
  echo "==> No .env found, writing default server environment..."
  cat > "$COMPOSE_DIR/.env" <<EOF
VLLM_HOST=http://localhost:7000
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
EMBED_MODEL=BAAI/bge-base-en-v1.5
ADMIN_USERNAME=admin
ADMIN_PASSWORD=insurehub2026
AUTH_SECRET_KEY=insurehub-rag-secret-2026
AUTH_TOKEN_EXPIRE_MINUTES=43200
EOF
else
  echo "==> .env already exists on server, leaving it untouched."
fi

# ── GPU detection ──────────────────────────────────────────────────────
# docker-compose.gpu.yml rebuilds torch against a CUDA wheel and reserves
# an NVIDIA device, but nothing in the deploy path referenced it — so a
# GPU host silently deployed the CPU image. That failure is invisible at
# runtime: torch.cuda.is_available() just returns False and every model
# quietly stays on CPU, so the deploy "succeeds" and buys nothing. This
# block exists to make that outcome impossible to get by accident.
#
# Override with FORCE_GPU=1 (skip probing, assume GPU) or FORCE_CPU=1.
COMPOSE_FILES=(-f docker-compose.yml)
GPU_MODE="cpu"

if [ "${FORCE_CPU:-}" = "1" ]; then
  echo "==> FORCE_CPU=1 — building CPU image."
elif [ "${FORCE_GPU:-}" = "1" ]; then
  COMPOSE_FILES+=(-f docker-compose.gpu.yml); GPU_MODE="cuda"
  echo "==> FORCE_GPU=1 — building CUDA image without probing."
elif command -v nvidia-smi &>/dev/null; then
  echo "==> NVIDIA GPU found on host — checking Docker can actually reach it..."
  # Host nvidia-smi working does NOT imply containers can use the GPU;
  # that needs nvidia-container-toolkit. Probe the real thing.
  if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    COMPOSE_FILES+=(-f docker-compose.gpu.yml); GPU_MODE="cuda"
    echo "    ✓ Docker has GPU access — building CUDA image."
  else
    echo "    ✗ nvidia-smi works on the host but Docker cannot reach the GPU."
    echo "      Install nvidia-container-toolkit and re-run to get GPU acceleration."
    echo "      Falling back to the CPU image."
  fi
else
  echo "==> No NVIDIA GPU detected — building CPU image."
fi

echo "==> Building and starting the backend (${GPU_MODE})..."
cd "$COMPOSE_DIR"
docker compose pull --ignore-pull-failures 2>/dev/null || true
docker compose "${COMPOSE_FILES[@]}" build api
docker compose "${COMPOSE_FILES[@]}" up -d api

echo "==> Waiting for the API to become healthy..."
for i in $(seq 1 20); do
  STATUS=$(docker inspect --format='{{.State.Health.Status}}' insurehub_api 2>/dev/null || echo "starting")
  [ "$STATUS" = "healthy" ] && break
  echo "    ($i/20) Status: $STATUS — waiting 10s..."
  sleep 10
done

echo ""
echo "==> Checking the API..."
curl -sf http://localhost:8501/health && echo " ✓ API is up!" || echo " ✗ API health check failed — check: docker logs insurehub_api"

# ── Phase 1a exit criterion (plan_latency.md): device=cuda observed ─────
# The models log their resolved device on load. Assert it matches what we
# built for, so a GPU deploy that silently fell back to CPU is reported
# loudly here instead of being discovered later as "the GPU changed nothing".
echo ""
echo "==> Verifying model device..."
DEVICE_LINE=$(docker logs insurehub_api 2>&1 | grep -m1 "SharedModels.*device=" || true)
if [ -z "$DEVICE_LINE" ]; then
  echo "    ? No model-load line yet (models load lazily on first query)."
  echo "      Check later with: docker logs insurehub_api 2>&1 | grep 'SharedModels'"
elif echo "$DEVICE_LINE" | grep -q "device=cuda"; then
  echo "    ✓ Models are on the GPU: $DEVICE_LINE"
elif [ "$GPU_MODE" = "cuda" ]; then
  echo "    ✗ BUILT FOR GPU BUT RUNNING ON CPU — this deploy bought nothing."
  echo "      $DEVICE_LINE"
  echo "      torch.cuda.is_available() is False inside the container. Check that"
  echo "      nvidia-container-toolkit is installed and the daemon was restarted."
else
  echo "    ✓ CPU build, models on CPU (expected): $DEVICE_LINE"
fi

echo ""
echo "================================================================"
echo " Backend is running at: http://$(curl -s ifconfig.me 2>/dev/null || echo "SERVER_IP"):8501"
echo ""
echo " Next step — update Vercel to point to the server IP:"
echo "   cd /path/to/insurehub-RAG-frontend/insurehub-your-ai-insurance-advisor"
echo "   vercel env rm VITE_API_BASE_URL production --yes"
echo "   vercel env rm VITE_API_URL production --yes"
echo "   echo 'http://$(curl -s ifconfig.me 2>/dev/null || echo "SERVER_IP"):8501' | vercel env add VITE_API_BASE_URL production"
echo "   echo 'http://$(curl -s ifconfig.me 2>/dev/null || echo "SERVER_IP"):8501' | vercel env add VITE_API_URL production"
echo "   vercel --prod"
echo "================================================================"

echo ""
echo "==> Setting up auto-restart on server reboot..."
# Create a systemd service so the container restarts if the server reboots.
# It must carry the SAME compose file list chosen above — otherwise a GPU
# host comes back on the CPU config after any reboot, silently undoing the
# deploy with no error anywhere.
COMPOSE_ARGS="${COMPOSE_FILES[*]}"
sudo tee /etc/systemd/system/insurehub.service > /dev/null <<UNIT
[Unit]
Description=InsureHub RAG Backend ($GPU_MODE)
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=$COMPOSE_DIR
ExecStart=/usr/bin/docker compose $COMPOSE_ARGS up api
ExecStop=/usr/bin/docker compose $COMPOSE_ARGS stop api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable insurehub
echo "==> Auto-restart service enabled. Backend will survive server reboots."
