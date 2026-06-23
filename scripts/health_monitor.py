"""
Health monitor for 1GB Oracle micro instance (rules #20, #27).

Monitors RAM, CPU, disk. Auto-kills stalled jobs if memory > 80%.
Run as cron: */5 * * * * python /home/ubuntu/trading-ml/scripts/health_monitor.py

Also archives old logs weekly (rule #28):
  0 3 * * 0 python /home/ubuntu/trading-ml/scripts/health_monitor.py --archive-logs
"""

import os
import sys
import json
import time
import glob
import shutil
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
HEALTH_LOG = os.path.join(LOG_DIR, "health.jsonl")
MEMORY_THRESHOLD = 80  # percent


def get_memory():
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split()
            mem[parts[0].rstrip(":")] = int(parts[1])
        total = mem.get("MemTotal", 1)
        available = mem.get("MemAvailable", total)
        used_pct = round((1 - available / total) * 100, 1)
        swap_total = mem.get("SwapTotal", 0)
        swap_free = mem.get("SwapFree", 0)
        swap_used_pct = round((1 - swap_free / max(swap_total, 1)) * 100, 1)
        return {
            "total_mb": round(total / 1024),
            "available_mb": round(available / 1024),
            "used_pct": used_pct,
            "swap_total_mb": round(swap_total / 1024),
            "swap_used_pct": swap_used_pct,
        }
    except Exception:
        return {"used_pct": 0, "error": "not linux"}


def get_cpu():
    try:
        load = os.getloadavg()
        return {"load_1m": round(load[0], 2), "load_5m": round(load[1], 2)}
    except Exception:
        return {"load_1m": 0}


def get_disk():
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize
        free = st.f_bfree * st.f_frsize
        return {
            "total_gb": round(total / 1e9, 1),
            "free_gb": round(free / 1e9, 1),
            "used_pct": round((1 - free / total) * 100, 1),
        }
    except Exception:
        return {"used_pct": 0}


def kill_stalled_jobs():
    """Kill python processes using too much memory."""
    try:
        result = subprocess.run(
            ["ps", "aux", "--sort=-rss"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines()[1:10]:
            parts = line.split()
            if len(parts) < 11:
                continue
            rss_kb = int(parts[5])
            pid = int(parts[1])
            cmd = " ".join(parts[10:])
            if rss_kb > 400_000 and "python" in cmd and "uvicorn" not in cmd:
                os.kill(pid, 9)
                return f"Killed PID {pid} ({rss_kb}KB): {cmd[:60]}"
    except Exception:
        pass
    return None


def archive_logs():
    """Compress and archive old logs (rule #28)."""
    archive_dir = os.path.join(LOG_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d")
    for f in glob.glob(os.path.join(LOG_DIR, "*.log")) + \
             glob.glob(os.path.join(LOG_DIR, "*.jsonl")):
        size = os.path.getsize(f)
        if size > 10 * 1024 * 1024:  # > 10MB
            dest = os.path.join(archive_dir, f"{os.path.basename(f)}.{now}")
            shutil.move(f, dest)
            subprocess.run(["gzip", dest], timeout=30)


def check():
    mem = get_memory()
    cpu = get_cpu()
    disk = get_disk()

    record = {
        "ts": datetime.now().isoformat(),
        "memory": mem,
        "cpu": cpu,
        "disk": disk,
    }

    killed = None
    if mem["used_pct"] > MEMORY_THRESHOLD:
        killed = kill_stalled_jobs()
        if killed:
            record["action"] = killed

    with open(HEALTH_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


if __name__ == "__main__":
    if "--archive-logs" in sys.argv:
        archive_logs()
        print("Logs archived")
    else:
        r = check()
        print(json.dumps(r, indent=2))
