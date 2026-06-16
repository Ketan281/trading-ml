import os
import sys
import json
import requests
from datetime import datetime, date, timedelta

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Known Event Calendar ──────────────────────────────
# Add important dates manually here
# Format: "YYYY-MM-DD": {"event": "...", "impact": "..."}

KNOWN_EVENTS = {
    # RBI Policy Meetings 2026
    "2026-04-09": {
        "event"     : "RBI Monetary Policy",
        "type"      : "rbi_policy",
        "impact"    : "extreme",
        "direction" : "unknown",
        "note"      : "Avoid options buying — IV crush risk"
    },
    "2026-06-06": {
        "event"     : "RBI Monetary Policy",
        "type"      : "rbi_policy",
        "impact"    : "extreme",
        "direction" : "unknown",
        "note"      : "Avoid options buying — IV crush risk"
    },
    "2026-08-06": {
        "event"     : "RBI Monetary Policy",
        "type"      : "rbi_policy",
        "impact"    : "extreme",
        "direction" : "unknown",
        "note"      : "Avoid options buying — IV crush risk"
    },
    "2026-10-08": {
        "event"     : "RBI Monetary Policy",
        "type"      : "rbi_policy",
        "impact"    : "extreme",
        "direction" : "unknown",
        "note"      : "Avoid options buying — IV crush risk"
    },
    "2026-12-04": {
        "event"     : "RBI Monetary Policy",
        "type"      : "rbi_policy",
        "impact"    : "extreme",
        "direction" : "unknown",
        "note"      : "Avoid options buying — IV crush risk"
    },

    # US Fed Meetings 2026
    "2026-01-29": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },
    "2026-03-19": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },
    "2026-05-07": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },
    "2026-06-17": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },
    "2026-07-29": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },
    "2026-09-16": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },
    "2026-11-05": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },
    "2026-12-16": {
        "event"     : "US Fed FOMC Meeting",
        "type"      : "fed_meeting",
        "impact"    : "high",
        "direction" : "unknown",
        "note"      : "Global risk-off possible"
    },

    # India Budget
    "2026-02-01": {
        "event"     : "India Union Budget",
        "type"      : "budget",
        "impact"    : "extreme",
        "direction" : "unknown",
        "note"      : "Highest volatility day of year"
    },

    # NSE Holidays 2026
    "2026-01-26": {
        "event"     : "Republic Day",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-03-25": {
        "event"     : "Holi",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-04-02": {
        "event"     : "Ram Navami",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-04-03": {
        "event"     : "Good Friday",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-04-14": {
        "event"     : "Dr Ambedkar Jayanti",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-05-01": {
        "event"     : "Maharashtra Day",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-08-15": {
        "event"     : "Independence Day",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-10-02": {
        "event"     : "Gandhi Jayanti",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
    "2026-11-14": {
        "event"     : "Diwali Laxmi Puja",
        "type"      : "holiday",
        "impact"    : "none",
        "direction" : "neutral",
        "note"      : "Market closed"
    },
}

# ── Results Season Detection ──────────────────────────
# Q1 results  → July-August
# Q2 results  → October-November
# Q3 results  → January-February
# Q4 results  → April-May

RESULTS_SEASONS = [
    {"months": [7, 8],    "quarter": "Q1",
     "impact": "elevated"},
    {"months": [10, 11],  "quarter": "Q2",
     "impact": "elevated"},
    {"months": [1, 2],    "quarter": "Q3",
     "impact": "elevated"},
    {"months": [4, 5],    "quarter": "Q4",
     "impact": "elevated"}
]

# ── Event Impact Mapping ──────────────────────────────
EVENT_STRATEGY_MAP = {
    "rbi_policy": {
        "avoid"      : [
            "naked_options",
            "long_straddle",
            "long_strangle"
        ],
        "prefer"     : [
            "iron_condor",
            "spread_strategies",
            "wait_for_announcement"
        ],
        "position_size": "quarter",
        "note"       : "IV will crush after announcement"
    },
    "fed_meeting": {
        "avoid"      : [
            "overnight_positions",
            "high_leverage"
        ],
        "prefer"     : [
            "defined_risk_strategies",
            "hedged_positions"
        ],
        "position_size": "half",
        "note"       : "Global markets may gap"
    },
    "budget": {
        "avoid"      : [
            "all_directional_bets",
            "naked_options"
        ],
        "prefer"     : [
            "iron_condor",
            "wait_for_clarity"
        ],
        "position_size": "avoid",
        "note"       : "Extreme unpredictable moves"
    },
    "holiday": {
        "avoid"      : ["all_trades"],
        "prefer"     : ["no_trading"],
        "position_size": "avoid",
        "note"       : "Market closed"
    },
    "results_season": {
        "avoid"      : [
            "sector_concentration",
            "naked_options_in_results_stocks"
        ],
        "prefer"     : [
            "index_options_over_stock_options",
            "spread_strategies"
        ],
        "position_size": "half",
        "note"       : "Individual stock volatility high"
    }
}

# ── Check Today's Events ──────────────────────────────
def check_today_events():
    today     = date.today()
    today_str = str(today)
    events    = []

    # Check known events
    if today_str in KNOWN_EVENTS:
        event = KNOWN_EVENTS[today_str].copy()
        event["date"]       = today_str
        event["days_away"]  = 0
        event["is_today"]   = True
        events.append(event)

    return events

# ── Check Upcoming Events (next N days) ───────────────
def check_upcoming_events(days_ahead=7):
    today    = date.today()
    upcoming = []

    for i in range(1, days_ahead + 1):
        check_date = today + timedelta(days=i)
        date_str   = str(check_date)

        if date_str in KNOWN_EVENTS:
            event = KNOWN_EVENTS[date_str].copy()
            event["date"]      = date_str
            event["days_away"] = i
            event["is_today"]  = False
            upcoming.append(event)

    return upcoming

# ── Check Results Season ──────────────────────────────
def check_results_season():
    today = date.today()
    month = today.month

    for season in RESULTS_SEASONS:
        if month in season["months"]:
            return {
                "in_results_season": True,
                "quarter"          : season["quarter"],
                "impact"           : season["impact"],
                "note"             : (
                    f"{season['quarter']} results season "
                    f"— stock specific volatility high"
                )
            }

    return {
        "in_results_season": False,
        "quarter"          : None,
        "impact"           : "normal",
        "note"             : "Not in results season"
    }

# ── Check Pre-Event Window ────────────────────────────
def check_pre_event_window(days_ahead=2):
    upcoming = check_upcoming_events(days_ahead)

    if not upcoming:
        return None

    # Get highest impact upcoming event
    impact_order = {
        "extreme": 4,
        "high"   : 3,
        "medium" : 2,
        "low"    : 1,
        "none"   : 0
    }

    upcoming.sort(
        key=lambda x: impact_order.get(
            x.get("impact", "low"), 0
        ),
        reverse=True
    )

    return upcoming[0] if upcoming else None

# ── Gap Analysis ──────────────────────────────────────
def analyze_market_gap(prev_close, current_open):
    if not prev_close or not current_open:
        return None

    gap_pct = (
        (current_open - prev_close)
        / prev_close * 100
    )

    if gap_pct > 1.5:
        gap_type   = "large_gap_up"
        gap_impact = "high"
        gap_signal = "bullish_but_fade_risk"
    elif gap_pct > 0.5:
        gap_type   = "small_gap_up"
        gap_impact = "medium"
        gap_signal = "mild_bullish"
    elif gap_pct < -1.5:
        gap_type   = "large_gap_down"
        gap_impact = "high"
        gap_signal = "bearish_but_bounce_risk"
    elif gap_pct < -0.5:
        gap_type   = "small_gap_down"
        gap_impact = "medium"
        gap_signal = "mild_bearish"
    else:
        gap_type   = "flat_open"
        gap_impact = "low"
        gap_signal = "neutral"

    return {
        "prev_close"  : prev_close,
        "current_open": current_open,
        "gap_pct"     : round(gap_pct, 2),
        "gap_type"    : gap_type,
        "gap_impact"  : gap_impact,
        "gap_signal"  : gap_signal
    }

# ── Build Event Context Block ─────────────────────────
def build_event_context():
    today_events  = check_today_events()
    upcoming      = check_upcoming_events(7)
    results       = check_results_season()
    pre_event     = check_pre_event_window(2)

    # Determine overall event risk
    event_risk    = "normal"
    avoid_list    = []
    prefer_list   = []
    position_size = "full"
    alerts        = []
    is_event_day  = False

    # Process today's events
    for event in today_events:
        is_event_day = True
        etype        = event.get("type", "unknown")
        impact       = event.get("impact", "normal")

        if impact in ["extreme", "high"]:
            event_risk = "extreme" \
                         if impact == "extreme" \
                         else "high"

        strategy = EVENT_STRATEGY_MAP.get(etype, {})
        avoid_list.extend(
            strategy.get("avoid", [])
        )
        prefer_list.extend(
            strategy.get("prefer", [])
        )

        ps = strategy.get("position_size", "full")
        if ps in ["avoid", "quarter"]:
            position_size = ps
        elif ps == "half" and \
                position_size == "full":
            position_size = "half"

        alerts.append(
            f"TODAY: {event['event']} — "
            f"{event.get('note', '')}"
        )

    # Process pre-event window
    if pre_event and not is_event_day:
        etype  = pre_event.get("type", "unknown")
        impact = pre_event.get("impact", "normal")
        days   = pre_event.get("days_away", 0)

        if impact == "extreme":
            event_risk    = "elevated"
            position_size = "half"
            alerts.append(
                f"⚠ {pre_event['event']} in "
                f"{days} days — reduce exposure"
            )
        elif impact == "high":
            alerts.append(
                f"ℹ {pre_event['event']} in "
                f"{days} days — be cautious"
            )

    # Results season
    if results["in_results_season"]:
        alerts.append(
            f"📊 {results['note']}"
        )
        if event_risk == "normal":
            event_risk = "elevated"

    # Build context string for AI prompt
    lines = []
    lines.append(
        "── EVENT AWARENESS CONTEXT ─────────────────"
    )
    lines.append(
        f"Event Risk Level  : {event_risk.upper()}"
    )
    lines.append(
        f"Is Event Day      : "
        f"{'YES' if is_event_day else 'NO'}"
    )
    lines.append(
        f"Results Season    : "
        f"{'YES — ' + results['quarter'] if results['in_results_season'] else 'NO'}"
    )

    if today_events:
        lines.append(f"\nToday's Events:")
        for e in today_events:
            lines.append(
                f"  🔴 {e['event']} "
                f"[{e['impact'].upper()}]"
            )
            lines.append(
                f"     → {e.get('note', '')}"
            )

    if upcoming:
        lines.append(f"\nUpcoming Events (7 days):")
        for e in upcoming[:3]:
            lines.append(
                f"  📅 {e['date']} — "
                f"{e['event']} "
                f"[{e['impact'].upper()}] "
                f"({e['days_away']}d away)"
            )

    if avoid_list:
        lines.append(
            f"\nAvoid Today      : "
            f"{', '.join(set(avoid_list))}"
        )
    if prefer_list:
        lines.append(
            f"Prefer Today     : "
            f"{', '.join(set(prefer_list))}"
        )

    lines.append(
        f"Position Size    : {position_size.upper()}"
    )

    if alerts:
        lines.append(f"\n⚠ EVENT ALERTS:")
        for alert in alerts:
            lines.append(f"  → {alert}")

    lines.append(
        "── END EVENT CONTEXT ───────────────────────"
    )

    context_str = "\n".join(lines)

    return {
        "context_str"   : context_str,
        "event_risk"    : event_risk,
        "is_event_day"  : is_event_day,
        "today_events"  : today_events,
        "upcoming"      : upcoming,
        "results_season": results,
        "avoid_list"    : list(set(avoid_list)),
        "prefer_list"   : list(set(prefer_list)),
        "position_size" : position_size,
        "alerts"        : alerts
    }

# ── Apply Event Adjustments ───────────────────────────
def apply_event_adjustments(decision,
                              event_context):
    if not decision or not event_context:
        return decision

    event_risk    = event_context.get(
        "event_risk", "normal"
    )
    is_event_day  = event_context.get(
        "is_event_day", False
    )
    position_size = event_context.get(
        "position_size", "full"
    )
    alerts        = event_context.get("alerts", [])
    avoid_list    = event_context.get(
        "avoid_list", []
    )

    confidence    = float(
        decision.get("confidence", 0.5)
    )
    warnings      = decision.get("warnings", [])

    # Event day — extreme caution
    if is_event_day:
        today_events = event_context.get(
            "today_events", []
        )
        for event in today_events:
            if event.get("impact") == "extreme":
                # Budget or RBI — avoid trading
                decision["action"]        = "avoid"
                decision["confidence"]    = 0.0
                decision["position_size"] = "avoid"
                warnings.append(
                    f"EVENT DAY: "
                    f"{event['event']} — "
                    f"avoid all trades"
                )
            elif event.get("impact") == "high":
                # High impact — reduce
                decision["confidence"] = max(
                    0.0, confidence - 0.25
                )
                decision["position_size"] = "quarter"
                warnings.append(
                    f"HIGH IMPACT EVENT: "
                    f"{event['event']}"
                )

    # Pre-event window
    elif event_risk == "elevated":
        decision["confidence"] = max(
            0.0, confidence - 0.10
        )
        if position_size == "half" and \
                decision.get(
                    "position_size"
                ) == "full":
            decision["position_size"] = "half"
        warnings.append(
            "Pre-event window — reduced exposure"
        )

    # Override position size
    if position_size == "avoid":
        decision["position_size"] = "avoid"
    elif position_size == "quarter" and \
            decision.get(
                "position_size"
            ) not in ["avoid"]:
        decision["position_size"] = "quarter"
    elif position_size == "half" and \
            decision.get(
                "position_size"
            ) == "full":
        decision["position_size"] = "half"

    # Add event alerts to warnings
    for alert in alerts:
        if alert not in warnings:
            warnings.append(
                f"[EVENT] {alert}"
            )

    # Add event context note
    decision["event_context"] = {
        "risk"         : event_risk,
        "is_event_day" : is_event_day,
        "alerts"       : alerts
    }

    decision["warnings"] = warnings
    return decision

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Event Awareness Layer")
    print(f"  Date: {date.today()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Build full event context
    print("\n  🗓 Checking event calendar...")
    ctx = build_event_context()

    # Print context
    print(f"\n{ctx['context_str']}")

    # Print summary
    print(f"\n  {'─' * 50}")
    print(f"  EVENT SUMMARY")
    print(f"  {'─' * 50}")
    print(f"  Event Risk     : "
          f"{ctx['event_risk'].upper()}")
    print(f"  Is Event Day   : "
          f"{'🔴 YES' if ctx['is_event_day'] else '✅ NO'}")
    print(f"  Results Season : "
          f"{'📊 YES — ' + ctx['results_season']['quarter'] if ctx['results_season']['in_results_season'] else '✅ NO'}")
    print(f"  Position Size  : "
          f"{ctx['position_size'].upper()}")

    if ctx["today_events"]:
        print(f"\n  🔴 TODAY'S EVENTS:")
        for e in ctx["today_events"]:
            print(f"     {e['event']} "
                  f"[{e['impact'].upper()}]")
            print(f"     → {e.get('note', '')}")

    if ctx["upcoming"]:
        print(f"\n  📅 UPCOMING EVENTS:")
        for e in ctx["upcoming"][:5]:
            print(
                f"     {e['date']} — "
                f"{e['event']} "
                f"({e['days_away']}d away) "
                f"[{e['impact'].upper()}]"
            )

    if ctx["alerts"]:
        print(f"\n  ⚠ ALERTS:")
        for alert in ctx["alerts"]:
            print(f"     → {alert}")

    if ctx["avoid_list"]:
        print(f"\n  ❌ Avoid Today:")
        for a in ctx["avoid_list"]:
            print(f"     → {a}")

    if ctx["prefer_list"]:
        print(f"\n  ✅ Prefer Today:")
        for p in ctx["prefer_list"]:
            print(f"     → {p}")

    # Save event context
    path = os.path.join(
        OUTPUT_DIR,
        f"event_context_"
        f"{datetime.now().strftime('%Y%m%d')}.json"
    )
    with open(path, "w") as f:
        save_ctx = {
            k: v for k, v in ctx.items()
            if k != "context_str"
        }
        json.dump(save_ctx, f,
                  indent=2, default=str)

    print(f"\n  ✅ Event context saved → {path}")
    print("\n  ✅ Event Awareness complete!")