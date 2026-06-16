# Deploy Guide — Oracle Cloud Always Free + nginx + SSL + CORS lockdown

Hosts the Trading-AI / Agentic OS for **free, always-on, with an Indian IP**
(so NSE live data works). Two run modes are given: **Docker** (easiest) and
**systemd** (lighter on RAM). Pick one.

> Paper-trading only. No broker keys live anywhere in this stack.

---

## 0. Why Oracle Always Free (recap)
- Free forever: ARM Ampere A1, up to **4 OCPU / 24 GB RAM**.
- Choose **Mumbai or Hyderabad** region → **Indian IP** → NSE/`jugaad_data`
  fetches succeed (the #1 reason other free hosts fail).
- Always-on → the 24/7 `aos/scheduler.py` actually runs.

---

## 1. Create the VM
1. Sign up at cloud.oracle.com → **Region: Mumbai** (or Hyderabad).
2. Compute → Instances → **Create Instance**.
   - Image: **Ubuntu 22.04** (Canonical).
   - Shape: **Ampere A1 (Always Free eligible)** — e.g. 2 OCPU / 12 GB (room
     for Ollama) or 1 OCPU / 6 GB (API-only, no LLM).
   - Add your SSH public key.
3. Networking → keep the auto-created VCN; note the **public IP**.

## 2. Open the firewall (two layers)
**Oracle Security List** (Console → VCN → Security Lists → default):
- Add Ingress rules: **TCP 80** and **TCP 443** from `0.0.0.0/0`.
  (Leave 8000 closed — only nginx is public.)

**On the VM** (Ubuntu's iptables is restrictive by default):
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 3. Base setup
```bash
ssh ubuntu@<PUBLIC_IP>
sudo apt update && sudo apt -y upgrade
sudo timedatectl set-timezone Asia/Kolkata
git clone <YOUR_REPO_URL> ~/trading-ai && cd ~/trading-ai
```

---

## 4a. Run mode: DOCKER (recommended)
```bash
# install docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# set the CORS origin to your frontend before starting
echo "ALLOWED_ORIGINS=https://<your-frontend-domain>" > .env   # compose reads it

docker compose up -d --build           # api + scheduler + ollama
docker compose exec ollama ollama pull qwen2.5:3b   # optional; skip if low RAM
docker compose ps
curl localhost:8000/health             # {"status":"ok"}
```
> Low RAM (≤6 GB)? Remove the `ollama` service from `docker-compose.yml` and
> call `/query` with `"polish": false` — deterministic answers, no LLM.

## 4b. Run mode: systemd (lighter)
```bash
sudo apt -y install python3-venv
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
deactivate

# edit the unit files: set User, WorkingDirectory, ALLOWED_ORIGINS
sudo cp deploy/trading-ai-api.service       /etc/systemd/system/
sudo cp deploy/trading-ai-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-ai-api trading-ai-scheduler
systemctl status trading-ai-api --no-pager
```

---

## 5. nginx reverse proxy
```bash
sudo apt -y install nginx
sudo cp deploy/nginx.conf /etc/nginx/sites-available/trading-ai
# edit: replace api.example.com with your domain
sudo ln -s /etc/nginx/sites-available/trading-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```
Point your domain's **A record** to `<PUBLIC_IP>` (any free DNS — e.g. a
Cloudflare-managed domain, or a free subdomain from DuckDNS).

## 6. Free SSL (Let's Encrypt)
```bash
sudo apt -y install certbot python3-certbot-nginx
sudo certbot --nginx -d api.example.com --redirect -m you@email.com --agree-tos -n
# auto-renew is installed; test it:
sudo certbot renew --dry-run
```
Now `https://api.example.com/query` is live with a valid cert and HTTP→HTTPS
redirect.

## 7. CORS lockdown (already wired)
`api/server.py` reads `ALLOWED_ORIGINS` (comma-separated). Set it to **exactly**
your frontend origin(s) — no `*`:
```bash
# docker: in .env
ALLOWED_ORIGINS=https://app.yourdomain.com
# systemd: in the unit's Environment= line, then: sudo systemctl restart trading-ai-api
```
The browser will then reject calls from any other origin. nginx additionally
404s `/docs`, `/redoc`, `/openapi.json` (no API surface leakage) and rate-limits.

---

## 8. Frontend (free)
Deploy a static page (Cloudflare Pages / Vercel / Netlify) that calls your API:
```js
fetch("https://api.example.com/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ q: "best banknifty intraday option today", polish: false })
}).then(r => r.json()).then(d => render(d.answer, d.data));
```
Set `ALLOWED_ORIGINS` to that site's origin.

---

## 9. First-run data + verification
```bash
# warm the cache and confirm NSE access from the VM
docker compose exec api python -c "from pipelines.options.chain_live_intel import fetch_chain; print(bool(fetch_chain('NIFTY')))"
# expect: True  (False ⇒ NSE blocking this IP — confirm region is Indian)

curl -s https://api.example.com/query -X POST -H 'Content-Type: application/json' \
  -d '{"q":"best banknifty intraday option today","polish":false}' | head
```

## 10. Operations
- Logs: `docker compose logs -f scheduler` or `journalctl -u trading-ai-scheduler -f`
  and `logs/aos/aos.log`.
- Backups: the whole `data/` dir (Trade Memory `data/aos/memory.db`, paper ledger,
  collectors). `tar czf backup.tgz data/ models/registry/`.
- Updates: `git pull && docker compose up -d --build`.
- Crash recovery is automatic (Restart=always / restart: unless-stopped; the
  scheduler catches up missed daily jobs from `last_run.json`).

---

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `fetch_chain` returns False / empty data | NSE blocking the IP → ensure the VM is in **Mumbai/Hyderabad** region |
| 502 from nginx | API not up → `docker compose ps` / `systemctl status trading-ai-api` |
| CORS error in browser | `ALLOWED_ORIGINS` doesn't match the frontend origin exactly (scheme+host+port) |
| OOM / killed | Drop the `ollama` service, use `"polish": false`; or use a larger Ampere shape |
| `/query` slow first call | Cold cache → the precompute job warms it on schedule; or hit `/options/NIFTY` once |
