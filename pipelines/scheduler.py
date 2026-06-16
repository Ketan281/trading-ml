import os
import sys
import time
import schedule
import logging
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.run_pipeline          import run_full_pipeline
from pipelines.combined_intelligence import run_combined_intelligence
from training.dataset_builder        import (load_memory,
                                             build_raw_dataset,
                                             build_finetune_dataset,
                                             build_reflection_dataset,
                                             save_datasets,
                                             print_stats)

# ── Logging Setup ─────────────────────────────────────
LOG_DIR = os.path.join(ROOT, "outputs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers= [
        logging.FileHandler(
            os.path.join(LOG_DIR, "scheduler.log")
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TradingAI")

# ── Install Schedule ──────────────────────────────────
def install_schedule():
    try:
        import schedule
    except ImportError:
        log.info("Installing schedule library...")
        os.system("pip install schedule")

# ── Morning Job — 9:00 AM ─────────────────────────────
def morning_job():
    log.info("=" * 55)
    log.info("  🌅 MORNING JOB STARTING")
    log.info("=" * 55)

    try:
        # Run full pipeline
        log.info("  Running full pipeline...")
        run_full_pipeline(skip_reflection=True)

        # Run combined intelligence
        log.info("  Running combined intelligence...")
        signals = run_combined_intelligence()

        # Log summary
        if signals:
            active = [s for s in signals
                      if not s.get("blocked")
                      and s.get("final_action")
                      in ["buy", "sell"]]
            log.info(f"  Active signals : {len(active)}")
            log.info(f"  Total analyzed : {len(signals)}")

            for s in active:
                log.info(
                    f"  🎯 {s['symbol']} → "
                    f"{s['final_action'].upper()} | "
                    f"Confidence: {s['fused_confidence']} | "
                    f"Entry: {s['entry_zone']}"
                )

        log.info("  ✅ Morning job complete")

    except Exception as e:
        log.error(f"  ❌ Morning job failed: {e}")

# ── Midday Job — 12:30 PM ─────────────────────────────
def midday_job():
    log.info("=" * 55)
    log.info("  ☀️  MIDDAY REFRESH")
    log.info("=" * 55)

    try:
        signals = run_combined_intelligence()

        if signals:
            strong = [s for s in signals
                      if s.get("fused_confidence", 0) > 0.75
                      and not s.get("blocked")]
            if strong:
                log.info(f"  ⚡ High confidence signals:")
                for s in strong:
                    log.info(
                        f"     {s['symbol']} → "
                        f"{s['final_action'].upper()} "
                        f"({s['fused_confidence']})"
                    )
            else:
                log.info("  No high confidence signals midday")

        log.info("  ✅ Midday refresh complete")

    except Exception as e:
        log.error(f"  ❌ Midday job failed: {e}")

# ── Evening Job — 4:00 PM ─────────────────────────────
def evening_job():
    log.info("=" * 55)
    log.info("  🌆 EVENING DATASET BUILD")
    log.info("=" * 55)

    try:
        analyses, outcomes, reflections = load_memory()

        if len(analyses) > 0:
            raw        = build_raw_dataset(
                             analyses, outcomes, reflections)
            finetune   = build_finetune_dataset(raw)
            reflection = build_reflection_dataset(raw)

            print_stats(raw, finetune, reflection)
            save_datasets(raw, finetune, reflection)

            log.info(
                f"  ✅ Dataset updated — "
                f"{len(raw)} records | "
                f"{len(finetune)} fine-tune entries"
            )
        else:
            log.info("  No data yet to build dataset")

    except Exception as e:
        log.error(f"  ❌ Evening job failed: {e}")

# ── Health Check ──────────────────────────────────────
def health_check():
    log.info(
        f"  💓 Health check OK — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

# ── Setup Schedule ────────────────────────────────────
def setup_schedule():
    # Morning analysis — before market open
    schedule.every().day.at("09:00").do(morning_job)

    # Midday refresh — during market hours
    schedule.every().day.at("12:30").do(midday_job)

    # Evening dataset build — after market close
    schedule.every().day.at("16:00").do(evening_job)

    # Hourly health check
    schedule.every().hour.do(health_check)

    log.info("\n  ✅ Schedule configured:")
    log.info("     09:00 → Morning Analysis")
    log.info("     12:30 → Midday Refresh")
    log.info("     16:00 → Evening Dataset Build")
    log.info("     Every hour → Health Check\n")

# ── Run Once Immediately ──────────────────────────────
def run_now():
    log.info("\n" + "=" * 55)
    log.info("  🚀 Running immediate analysis...")
    log.info("=" * 55)
    morning_job()

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "🔥" * 27)
    print("   TRADING AI — DAILY SCHEDULER")
    print("🔥" * 27)

    # Install schedule if needed
    try:
        import schedule
    except ImportError:
        os.system(
            f"{sys.executable} -m pip install schedule"
        )
        import schedule

    log.info(f"  Scheduler starting...")
    log.info(
        f"  Time: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Setup schedule
    setup_schedule()

    # Run immediately on start
    run_now()

    # Keep running
    log.info("\n  ⏰ Scheduler is running...")
    log.info("  Press Ctrl+C to stop\n")

    while True:
        schedule.run_pending()
        time.sleep(60)