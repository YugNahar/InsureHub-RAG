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

echo "==> Cloning / updating repo..."
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --rebase
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "==> Writing server environment overrides..."
cat > "$COMPOSE_DIR/.env" <<EOF
# On the server, vLLM runs locally so we use localhost
VLLM_HOST=http://localhost:7000
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
EMBED_MODEL=BAAI/bge-base-en-v1.5
ADMIN_USERNAME=admin
ADMIN_PASSWORD=insurehub2026
AUTH_SECRET_KEY=insurehub-rag-secret-2026
AUTH_TOKEN_EXPIRE_MINUTES=43200
EOF

echo "==> Building and starting the backend..."
cd "$COMPOSE_DIR"
docker compose pull --ignore-pull-failures 2>/dev/null || true
docker compose build api
docker compose up -d api

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
# Create a systemd service so the container restarts if the server reboots
sudo tee /etc/systemd/system/insurehub.service > /dev/null <<UNIT
[Unit]
Description=InsureHub RAG Backend
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=$COMPOSE_DIR
ExecStart=/usr/bin/docker compose up api
ExecStop=/usr/bin/docker compose stop api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable insurehub
echo "==> Auto-restart service enabled. Backend will survive server reboots."
