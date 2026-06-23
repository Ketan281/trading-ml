#!/bin/bash
# Setup script for Oracle Cloud 1GB micro instance
# Run: bash scripts/setup_server.sh

set -e

echo "=== Trading-AI Server Setup ==="
REPO="$HOME/trading-ml"

# Rule #20: Increase swap to 4GB, keep memory < 80%
echo "[1/6] Setting up 4GB swap..."
if [ -f /swapfile ]; then
    sudo swapoff /swapfile 2>/dev/null || true
    sudo rm /swapfile
fi
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Make persistent
grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
echo "  ✓ Swap: $(swapon --show | tail -1)"

# Rule #20: Tune swappiness for low-RAM server
sudo sysctl vm.swappiness=10
sudo sysctl vm.vfs_cache_pressure=50
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# Rule #26: Max concurrent workers = 2
echo "[2/6] Writing systemd service..."
sudo tee /etc/systemd/system/trading-ai.service > /dev/null << 'UNIT'
[Unit]
Description=Trading-AI Backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-ml
ExecStart=/home/ubuntu/trading-ml/venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 10
Restart=always
RestartSec=10

# Rule #10: Minimal logging
Environment=AOS_DISABLE_LLM=1
Environment=MALLOC_ARENA_MAX=2
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONUNBUFFERED=1
# Rule #26: Limit memory to prevent OOM kill of SSH
MemoryMax=800M
MemoryHigh=600M
CPUQuota=80%

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
echo "  ✓ Service configured"

# Rule #27, #28: Cron jobs for health monitoring and log archival
echo "[3/6] Setting up cron jobs..."
(crontab -l 2>/dev/null | grep -v 'health_monitor\|convert_to_parquet\|archive-logs'; echo "
# Health monitor every 5 min (rule #27)
*/5 * * * * /home/ubuntu/trading-ml/venv/bin/python /home/ubuntu/trading-ml/scripts/health_monitor.py >> /home/ubuntu/trading-ml/logs/health_cron.log 2>&1

# Archive old logs every Sunday 3am (rule #28)
0 3 * * 0 /home/ubuntu/trading-ml/venv/bin/python /home/ubuntu/trading-ml/scripts/health_monitor.py --archive-logs >> /home/ubuntu/trading-ml/logs/health_cron.log 2>&1

# Rule #12: Clean old intraday data monthly (keep 60 days)
0 4 1 * * find /home/ubuntu/trading-ml/data/intraday -mtime +60 -name '*.csv' -delete 2>/dev/null
") | crontab -
echo "  ✓ Cron jobs installed"

# Rule #7: Convert CSV to Parquet (one-time)
echo "[4/6] Checking for Parquet conversion..."
if [ ! -d "$REPO/data/historical_parquet" ] && [ -d "$REPO/data/historical" ]; then
    echo "  Converting CSV → Parquet (this takes a few minutes)..."
    cd "$REPO" && venv/bin/pip install pyarrow -q 2>/dev/null
    cd "$REPO" && venv/bin/python scripts/convert_to_parquet.py || echo "  (Parquet conversion skipped — pyarrow not available)"
fi

# Nginx rate limiting (rule #26)
echo "[5/6] Updating nginx..."
sudo tee /etc/nginx/sites-available/trading-ai > /dev/null << 'NGINX'
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    listen 80;
    server_name ketan-trading.duckdns.org;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name ketan-trading.duckdns.org;

    ssl_certificate /etc/letsencrypt/live/ketan-trading.duckdns.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ketan-trading.duckdns.org/privkey.pem;

    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
NGINX
sudo nginx -t && sudo systemctl reload nginx
echo "  ✓ Nginx updated with rate limiting"

# Rule #29: Direct systemd, no Docker
echo "[6/6] Starting service..."
sudo systemctl enable trading-ai
sudo systemctl restart trading-ai
echo "  ✓ Service started"

echo ""
echo "=== Setup Complete ==="
echo "  Service: sudo systemctl status trading-ai"
echo "  Logs:    sudo journalctl -u trading-ai -f"
echo "  Health:  curl http://localhost:8000/health/detail"
echo "  Cron:    crontab -l"
