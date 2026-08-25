"""Verify the optimised _quarterly_noi_provider against Charlene's baseline.

Usage (from the repo root):
    .venv\\Scripts\\python scripts/verify_quarterly_noi.py          # baseline mode
    .venv\\Scripts\\python scripts/verify_quarterly_noi.py --local   # old-vs-new on local data

Baseline mode:  Loads Charlene's baseline CSV and compares the new provider
    against the captured values.  Requires the local ISBS data to cover the
    same periods as the baseline (e.g. through July 2026).

Local mode (--local):  Runs BOTH the old chart pipeline and the new direct
    provider on the same local data (all baseline vcodes x 6 quarters of
    complete data).  This proves equivalence without needing Azure-level data.
    Exits 0 only if every row matches.
"""
import csv
import os
import sys

# Repo root on path so config, one_pager etc. are importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

BASELINE = os.path.join(
    os.path.expanduser("~"), "Downloads",
    "debt_yield_baseline_TIAA_20260824-200432_CLEAN.csv",
)


def main_baseline():
    """Compare the new provider against Charlene's captured baseline CSV."""
    if not os.path.exists(BASELINE):
        print(f"Baseline not found: {BASELINE}")
        return 1

    with open(BASELINE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Baseline: {len(rows)} rows from {BASELINE}")

    from flask_app import create_app
    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        data = data_service.get_data()

        from flask_app.services.portfolio_snapshot_freeze import (
            _quarterly_noi_provider,
        )
        provider = _quarterly_noi_provider(data)

        passed, failed, total = 0, 0, 0
        for row in rows:
            vcode = row["vcode"]
            quarter = row["quarter"]
            expected_none = row["quarter_noi_is_None"] == "True"
            expected_noi_str = row["quarter_noi"]
            total += 1

            actual = provider(vcode, quarter)

            if expected_none:
                if actual is None:
                    passed += 1
                else:
                    failed += 1
                    print(f"  FAIL {vcode} {quarter}: expected None, "
                          f"got {actual}")
            else:
                if actual is None:
                    failed += 1
                    print(f"  FAIL {vcode} {quarter}: expected "
                          f"{expected_noi_str}, got None")
                else:
                    expected = float(expected_noi_str)
                    if abs(actual - expected) < 0.005:
                        passed += 1
                    else:
                        failed += 1
                        diff = actual - expected
                        print(f"  FAIL {vcode} {quarter}: expected "
                              f"{expected}, got {actual} (diff={diff})")

        print(f"\n{'=' * 70}")
        print(f"RESULT: {passed}/{total} passed, {failed} failed")
        if failed == 0:
            print("ALL ROWS MATCH — safe to ship")
        else:
            print("MISMATCHES FOUND — do NOT ship")
        print(f"{'=' * 70}")
        return 0 if failed == 0 else 1


def main_local():
    """Compare old chart pipeline vs new direct provider on local data.

    Proves equivalence without needing Azure-level ISBS data.  Tests all
    baseline vcodes across 6 quarters of complete local data.
    """
    import pandas as pd

    if not os.path.exists(BASELINE):
        print(f"Baseline not found (need vcode list): {BASELINE}")
        return 1

    with open(BASELINE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    all_vcodes = sorted(set(r["vcode"] for r in rows))

    from flask_app import create_app
    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        data = data_service.get_data()

        from flask_app.services.financials_service import (
            get_performance_chart_data,
        )
        from flask_app.services.portfolio_snapshot_freeze import (
            _quarterly_noi_provider,
        )
        new_provider = _quarterly_noi_provider(data)

        quarters = ["2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1",
                     "2024-Q4", "2024-Q3"]

        passed, failed, total = 0, 0, 0
        for vcode in all_vcodes:
            for quarter in quarters:
                total += 1
                yr = int(quarter.split("-Q")[0])
                qn = int(quarter.split("Q")[1])
                q_end = (pd.Timestamp(year=yr, month=qn * 3, day=1)
                         + pd.offsets.MonthEnd(0))

                # OLD: chart pipeline (the current production implementation)
                old_val = None
                try:
                    chart = get_performance_chart_data(
                        data["isbs_raw"], data["occupancy_raw"], vcode,
                        freq="Quarterly", periods=12,
                        period_end=str(q_end.date()),
                    ) or {}
                    for lbl, actual in zip(chart.get("periods") or [],
                                           chart.get("actual_noi") or []):
                        if lbl == f"Q{qn} {yr}" and actual is not None:
                            old_val = float(actual)
                            break
                except Exception:
                    old_val = None

                # NEW: direct ISBS
                new_val = new_provider(vcode, quarter)

                match = (
                    (old_val is None and new_val is None)
                    or (old_val is not None and new_val is not None
                        and abs(old_val - new_val) < 0.005)
                )
                if match:
                    passed += 1
                else:
                    failed += 1
                    print(f"  DIFF {vcode} {quarter}: "
                          f"old={old_val}, new={new_val}")

        print(f"\n{'=' * 70}")
        print(f"RESULT: {passed}/{total} matched, {failed} mismatched")
        print(f"Tested {len(all_vcodes)} vcodes x {len(quarters)} quarters")
        if failed == 0:
            print("ALL ROWS MATCH — new provider is equivalent to chart pipeline")
        else:
            print("MISMATCHES FOUND — do NOT ship")
        print(f"{'=' * 70}")
        return 0 if failed == 0 else 1


def main():
    if "--local" in sys.argv:
        return main_local()
    return main_baseline()


if __name__ == "__main__":
    raise SystemExit(main())
