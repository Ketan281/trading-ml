#!/usr/bin/env bash
# One-shot VM bootstrap for the Trading-AI / Agentic OS.
# Adds swap, installs Docker, and brings the stack up. Idempotent — safe to re-run.
#
# Usage (run from inside the cloned repo, on a fresh Ubuntu 22.04 VM):
#   chmod +x deploy/setup.sh
#   ALLOWED_ORIGINS="https://your-frontend" PROFILE=micro ./deploy/setup.sh
#
# PROFILE=micro  → docker-compose.micro.yml  (1 GB AMD box: API only, no LLM)
# PROFILE=full   → docker-compose.yml         (A1 box: API + scheduler + Ollama)
set -euo pipefail

PROFILE="${PROFILE:-micro}"
SWAP_GB="${SWAP_GB:-4}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:3000}"

echo "==> Profile: $PROFILE | swap: ${SWAP_GB}G | origins: $ALLOWED_ORIGINS"

# ── 1. Swap (critical on the 1 GB micro) ──────────────
if ! swapon --show | grep -q '/swapfile'; then
  echo "==> Creating ${SWAP_GB}G swap"
  sudo fallocate -l "${SWAP_GB}G" /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB*1024))
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  # Prefer RAM, use swap only under pressure.
  sudo sysctl -w vm.swappiness=10
  grep -q 'vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
else
  echo "==> Swap already present"
fi

# ── 2. Docker + compose plugin ────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

# ── 3. Open the OS firewall for 80/443 (Oracle Ubuntu is restrictive) ──
echo "==> Opening ports 80/443 on the host"
sudo iptables -C INPUT -p tcp --dport 80  -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -p tcp --dport 80  -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
(sudo netfilter-persistent save 2>/dev/null) || (sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save) || true

# ── 4. Env + bring up ─────────────────────────────────
echo "ALLOWED_ORIGINS=${ALLOWED_ORIGINS}" > .env
COMPOSE="docker-compose.micro.yml"
[ "$PROFILE" = "full" ] && COMPOSE="docker-compose.yml"

echo "==> docker compose -f $COMPOSE up -d --build"
sudo docker compose -f "$COMPOSE" up -d --build

echo "==> Waiting for the API to come up..."
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    echo "==> API healthy ✔"; break
  fi; sleep 3
done

echo
echo "==> DONE. Verify NSE access from this box:"
echo "    curl -s https://ipinfo.io/country   (expect: IN)"
echo "    curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' \\"
echo "         -d '{\"q\":\"best banknifty intraday option today\",\"polish\":false}'"
echo
echo "Next: install nginx + certbot (see docs/DEPLOY.md) to put HTTPS in front."
[ "$PROFILE" = "full" ] && echo "Ollama: docker compose exec ollama ollama pull qwen2.5:3b"
