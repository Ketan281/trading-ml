import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET_DIR = os.path.join(ROOT, "data",
                            "datasets")

# ── Validation Checks ─────────────────────────────────

# ── Check 1 — File Integrity ──────────────────────────
def check_file_integrity(window_dir):
    required = [
        "train.jsonl",
        "test.jsonl",
        "stats.json"
    ]
    missing  = []
    for f in required:
        if not os.path.exists(
            os.path.join(window_dir, f)
        ):
            missing.append(f)

    return {
        "passed" : len(missing) == 0,
        "missing": missing
    }

# ── Check 2 — JSONL Validity ──────────────────────────
def check_jsonl_validity(path, max_check=100):
    errors    = []
    valid     = 0
    total     = 0

    try:
        with open(path) as f:
            for i, line in enumerate(f):
                total += 1
                if i >= max_check:
                    break
                try:
                    entry = json.loads(
                        line.strip()
                    )
                    if "instruction" not in entry:
                        errors.append(
                            f"Line {i}: missing instruction"
                        )
                    elif "output" not in entry:
                        errors.append(
                            f"Line {i}: missing output"
                        )
                    else:
                        valid += 1
                except json.JSONDecodeError as e:
                    errors.append(
                        f"Line {i}: JSON error {e}"
                    )

    except Exception as e:
        errors.append(f"File error: {e}")

    return {
        "passed"      : len(errors) == 0,
        "total_lines" : total,
        "valid_lines" : valid,
        "errors"      : errors[:5]
    }

# ── Check 3 — Output JSON Validity ───────────────────
def check_output_validity(path,
                           max_check=200):
    required_keys = [
        "symbol", "date",
        "market_condition", "action",
        "strategy", "confidence",
        "risk_level", "reasoning"
    ]

    errors    = []
    valid     = 0
    total     = 0
    actions   = Counter()
    conditions = Counter()
    risks     = Counter()
    conf_vals = []

    try:
        with open(path) as f:
            for i, line in enumerate(f):
                total += 1
                if i >= max_check:
                    break

                try:
                    entry  = json.loads(
                        line.strip()
                    )
                    output = json.loads(
                        entry.get("output", "{}")
                    )

                    # Check required keys
                    missing = [
                        k for k in required_keys
                        if k not in output
                    ]
                    if missing:
                        errors.append(
                            f"Line {i}: missing "
                            f"keys {missing}"
                        )
                        continue

                    # Check value ranges
                    conf = float(
                        output.get("confidence", 0)
                    )
                    if not 0 <= conf <= 1:
                        errors.append(
                            f"Line {i}: invalid "
                            f"confidence {conf}"
                        )

                    # Check action validity
                    valid_actions = [
                        "buy", "sell", "hold",
                        "avoid", "reduce_exposure"
                    ]
                    action = output.get("action")
                    if action not in valid_actions:
                        errors.append(
                            f"Line {i}: invalid "
                            f"action {action}"
                        )

                    # Check reasoning
                    reasoning = output.get(
                        "reasoning", []
                    )
                    if len(reasoning) < 2:
                        errors.append(
                            f"Line {i}: insufficient "
                            f"reasoning"
                        )

                    # Collect stats
                    actions[action]           += 1
                    conditions[
                        output.get(
                            "market_condition",
                            "unknown"
                        )
                    ]                         += 1
                    risks[
                        output.get(
                            "risk_level", "medium"
                        )
                    ]                         += 1
                    conf_vals.append(conf)
                    valid += 1

                except Exception as e:
                    errors.append(
                        f"Line {i}: parse error {e}"
                    )

    except Exception as e:
        errors.append(f"File error: {e}")

    return {
        "passed"     : len(errors) == 0,
        "total"      : total,
        "valid"      : valid,
        "errors"     : errors[:5],
        "actions"    : dict(actions),
        "conditions" : dict(conditions),
        "risks"      : dict(risks),
        "avg_conf"   : round(
            np.mean(conf_vals), 3
        ) if conf_vals else 0
    }

# ── Check 4 — Class Balance ───────────────────────────
def check_class_balance(actions):
    if not actions:
        return {
            "passed": False,
            "reason": "No actions found"
        }

    total  = sum(actions.values())
    counts = list(actions.values())
    max_c  = max(counts)
    min_c  = min(counts)

    # Check imbalance ratio
    ratio  = max_c / min_c if min_c > 0 else 999

    # Good balance = ratio < 5
    passed = ratio < 5

    pcts = {
        k: round(v/total*100, 1)
        for k, v in actions.items()
    }

    return {
        "passed"        : passed,
        "ratio"         : round(ratio, 2),
        "distribution"  : pcts,
        "recommendation": (
            "Good balance" if passed
            else f"Imbalanced — ratio {ratio:.1f}x, "
                 f"consider resampling"
        )
    }

# ── Check 5 — Prompt Quality ──────────────────────────
def check_prompt_quality(path,
                          max_check=50):
    issues      = []
    avg_lengths = []
    min_length  = 500   # Minimum prompt chars
    max_length  = 3000  # Maximum prompt chars

    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_check:
                    break
                try:
                    entry  = json.loads(
                        line.strip()
                    )
                    prompt = entry.get(
                        "instruction", ""
                    )
                    length = len(prompt)
                    avg_lengths.append(length)

                    if length < min_length:
                        issues.append(
                            f"Line {i}: prompt too "
                            f"short ({length} chars)"
                        )
                    elif length > max_length:
                        issues.append(
                            f"Line {i}: prompt too "
                            f"long ({length} chars)"
                        )

                    # Check key sections present
                    required_sections = [
                        "TREND ANALYSIS",
                        "MOMENTUM",
                        "VOLATILITY",
                        "Return ONLY valid JSON"
                    ]
                    for section in required_sections:
                        if section not in prompt:
                            issues.append(
                                f"Line {i}: missing "
                                f"section '{section}'"
                            )
                            break

                except Exception as e:
                    issues.append(
                        f"Line {i}: {e}"
                    )

    except Exception as e:
        issues.append(f"File error: {e}")

    avg_len = round(
        np.mean(avg_lengths), 0
    ) if avg_lengths else 0

    return {
        "passed"    : len(issues) == 0,
        "avg_length": avg_len,
        "min_ok"    : avg_len >= min_length,
        "max_ok"    : avg_len <= max_length,
        "issues"    : issues[:5]
    }

# ── Check 6 — Year Coverage ───────────────────────────
def check_year_coverage(path,
                         expected_start,
                         expected_end):
    years_found = set()

    try:
        with open(path) as f:
            for line in f:
                try:
                    entry  = json.loads(
                        line.strip()
                    )
                    output = json.loads(
                        entry.get("output", "{}")
                    )
                    date   = output.get(
                        "date", ""
                    )
                    if date:
                        year = int(date[:4])
                        years_found.add(year)
                except Exception:
                    pass

    except Exception:
        pass

    expected_years = set(
        range(expected_start, expected_end + 1)
    )
    missing_years  = expected_years - years_found

    return {
        "passed"       : len(missing_years) == 0,
        "years_found"  : sorted(years_found),
        "missing_years": sorted(missing_years),
        "coverage_pct" : round(
            len(years_found) /
            len(expected_years) * 100, 1
        ) if expected_years else 0
    }

# ── Validate Single Window ────────────────────────────
def validate_window(window_dir,
                     train_end, test_year):
    window_name = (
        f"2014-{train_end} → Test {test_year}"
    )
    print(f"\n  {'─' * 55}")
    print(f"  Validating: {window_name}")
    print(f"  {'─' * 55}")

    results   = {}
    all_passed = True

    # Check 1 — File integrity
    check1 = check_file_integrity(window_dir)
    results["file_integrity"] = check1
    status = "✅" if check1["passed"] else "❌"
    print(f"  {status} File Integrity    : "
          f"{'PASS' if check1['passed'] else 'FAIL'}"
          f" {check1.get('missing', [])}")
    if not check1["passed"]:
        all_passed = False

    # Check 2 — JSONL validity (train)
    train_path = os.path.join(
        window_dir, "train.jsonl"
    )
    check2 = check_jsonl_validity(train_path)
    results["train_jsonl"] = check2
    status = "✅" if check2["passed"] else "❌"
    print(
        f"  {status} Train JSONL       : "
        f"{'PASS' if check2['passed'] else 'FAIL'}"
        f" ({check2['valid_lines']}/"
        f"{check2['total_lines']} valid)"
    )
    if not check2["passed"]:
        all_passed = False

    # Check 3 — Output validity
    check3 = check_output_validity(train_path)
    results["output_validity"] = check3
    status = "✅" if check3["passed"] else "❌"
    print(
        f"  {status} Output Validity   : "
        f"{'PASS' if check3['passed'] else 'FAIL'}"
        f" (avg conf: {check3['avg_conf']})"
    )
    if not check3["passed"]:
        all_passed = False

    # Check 4 — Class balance
    check4 = check_class_balance(
        check3.get("actions", {})
    )
    results["class_balance"] = check4
    status = "✅" if check4["passed"] else "⚠️"
    print(
        f"  {status} Class Balance     : "
        f"{'PASS' if check4['passed'] else 'WARN'}"
        f" (ratio: {check4.get('ratio', 0)}x)"
    )

    # Check 5 — Prompt quality
    check5 = check_prompt_quality(train_path)
    results["prompt_quality"] = check5
    status = "✅" if check5["passed"] else "⚠️"
    print(
        f"  {status} Prompt Quality    : "
        f"{'PASS' if check5['passed'] else 'WARN'}"
        f" (avg len: {check5['avg_length']} chars)"
    )

    # Check 6 — Year coverage
    check6 = check_year_coverage(
        train_path, 2014, train_end
    )
    results["year_coverage"] = check6
    status = "✅" if check6["passed"] else "⚠️"
    print(
        f"  {status} Year Coverage     : "
        f"{check6['coverage_pct']}% "
        f"({check6['years_found']})"
    )

    # Print action distribution
    actions = check3.get("actions", {})
    if actions:
        total = sum(actions.values())
        print(f"\n  Action Distribution:")
        for action, count in sorted(
            actions.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = round(count/total*100, 1)
            bar = "█" * int(pct / 3)
            print(
                f"     {action:<20}: "
                f"{bar:<15} {pct}%"
            )

    # Print risk distribution
    risks = check3.get("risks", {})
    if risks:
        total = sum(risks.values())
        print(f"\n  Risk Distribution:")
        for risk, count in sorted(
            risks.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = round(count/total*100, 1)
            print(
                f"     {risk:<10}: "
                f"{count:>5} ({pct}%)"
            )

    # Window verdict
    verdict    = "✅ PASS" if all_passed \
                 else "❌ FAIL"
    print(f"\n  Window Verdict: {verdict}")

    return {
        "window"    : window_name,
        "passed"    : all_passed,
        "checks"    : results
    }

# ── Validate All Windows ──────────────────────────────
def validate_all_windows():
    print("=" * 60)
    print("  Trading AI — Dataset Validator")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    windows = [
        {"train_end": 2018, "test_year": 2019},
        {"train_end": 2019, "test_year": 2020},
        {"train_end": 2020, "test_year": 2021},
        {"train_end": 2021, "test_year": 2022},
        {"train_end": 2022, "test_year": 2023},
        {"train_end": 2023, "test_year": 2024},
    ]

    all_results = []
    passed      = 0
    failed      = 0

    for w in windows:
        train_end = w["train_end"]
        test_year = w["test_year"]

        window_dir = os.path.join(
            DATASET_DIR,
            f"window_2014_{train_end}"
            f"_test_{test_year}"
        )

        if not os.path.exists(window_dir):
            print(
                f"\n  ⚠ Window not found: "
                f"2014-{train_end} → {test_year}"
            )
            print(
                f"  Run build_walk_forward"
                f"_dataset.py first"
            )
            failed += 1
            continue

        result = validate_window(
            window_dir, train_end, test_year
        )
        all_results.append(result)

        if result["passed"]:
            passed += 1
        else:
            failed += 1

    # Overall summary
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  {'WINDOW':<28} {'STATUS'}"
    )
    print("  " + "─" * 40)

    for r in all_results:
        icon   = "✅" if r["passed"] else "❌"
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {icon} {r['window']:<26} {status}"
        )

    print("  " + "─" * 40)
    print(f"  Passed : {passed}/{len(windows)}")
    print(f"  Failed : {failed}/{len(windows)}")

    # Dataset size summary
    print(f"\n  Dataset Sizes:")
    total_train = 0
    total_test  = 0

    for w in windows:
        train_end = w["train_end"]
        test_year = w["test_year"]
        wdir = os.path.join(
            DATASET_DIR,
            f"window_2014_{train_end}"
            f"_test_{test_year}"
        )
        stats_path = os.path.join(
            wdir, "stats.json"
        )

        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)

            tc = stats.get("train_count", 0)
            ec = stats.get("test_count",  0)
            total_train += tc
            total_test  += ec

            print(
                f"     2014-{train_end} → "
                f"Test {test_year}: "
                f"Train={tc:,} | "
                f"Test={ec:,}"
            )

    print(
        f"\n  Total Train Samples: "
        f"{total_train:,}"
    )
    print(
        f"  Total Test Samples : "
        f"{total_test:,}"
    )

    # Overall verdict
    if passed == len(windows):
        print(f"\n  {'=' * 60}")
        print(
            f"  ✅ ALL WINDOWS VALID — "
            f"Ready for fine-tuning!"
        )
        print(f"  {'=' * 60}")
        print(
            f"\n  Next step: "
            f"Upload to Google Colab and fine-tune"
        )
        print(
            f"  Command: "
            f"python training/prepare_colab.py"
        )
    else:
        print(f"\n  {'=' * 60}")
        print(
            f"  ⚠ {failed} windows need attention"
        )
        print(
            f"  Re-run build_walk_forward"
            f"_dataset.py to fix"
        )
        print(f"  {'=' * 60}")

    # Save validation report
    report_path = os.path.join(
        DATASET_DIR, "validation_report.json"
    )
    with open(report_path, "w") as f:
        json.dump({
            "timestamp"  : datetime.now().isoformat(),
            "total"      : len(windows),
            "passed"     : passed,
            "failed"     : failed,
            "total_train": total_train,
            "total_test" : total_test,
            "results"    : [
                {
                    "window": r["window"],
                    "passed": r["passed"]
                }
                for r in all_results
            ]
        }, f, indent=2)

    print(
        f"\n  ✅ Report saved → {report_path}"
    )
    return all_results

# ── Quick Stats ───────────────────────────────────────
def print_quick_stats():
    print(f"\n{'=' * 60}")
    print(f"  QUICK DATASET STATS")
    print(f"{'=' * 60}")

    # Load first window sample
    first_dir = os.path.join(
        DATASET_DIR,
        "window_2014_2018_test_2019"
    )
    train_path = os.path.join(
        first_dir, "train.jsonl"
    )

    if not os.path.exists(train_path):
        print("  ⚠ No dataset found")
        return

    # Count and sample
    samples = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            try:
                samples.append(
                    json.loads(line.strip())
                )
            except Exception:
                pass

    if samples:
        sample = samples[0]
        prompt = sample.get("instruction", "")
        output = sample.get("output", "{}")

        print(f"\n  Sample Entry #1:")
        print(f"  Prompt length : {len(prompt)} chars")
        print(
            f"  Output length : {len(output)} chars"
        )

        try:
            out = json.loads(output)
            print(f"\n  Sample Output:")
            print(
                f"     Action     : "
                f"{out.get('action')}"
            )
            print(
                f"     Condition  : "
                f"{out.get('market_condition')}"
            )
            print(
                f"     Strategy   : "
                f"{out.get('strategy')}"
            )
            print(
                f"     Confidence : "
                f"{out.get('confidence')}"
            )
            print(
                f"     Risk       : "
                f"{out.get('risk_level')}"
            )
            print(f"\n  Reasoning:")
            for r in out.get("reasoning", []):
                print(f"     → {r}")
        except Exception as e:
            print(f"  ⚠ Could not parse: {e}")

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    validate_all_windows()
    print_quick_stats()