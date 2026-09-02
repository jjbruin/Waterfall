"""Guardrail: a mid-month sale is not modelled as a month-end sale.

``compute_deal_analysis`` used to apply ``month_end()`` at the moment it parsed
the sale date, so the real date was gone before anything could use it. A 4 Sep
sale became 30 Sep everywhere: pref accrued 26 extra days and the loan was
carried an extra 26 days.

Two dates are now kept, and they are used for different things — see the note
at the top of the sale-date block in compute.py:

    sale_actual   when the deal closes. Pref stops here; the loan is repaid
                  here; the closing period settles here.
    sale_me       the month end it falls in. The forecast, the cash schedule
                  and the terminal-NOI window stay on this monthly grid,
                  because the operating data is not accurate enough to split a
                  month (author's instruction, Sep 2 2026).

And the terminal-NOI window now starts the month AFTER the month of sale,
rather than beginning with the sale month itself — the seller keeps the whole
month of sale on the cash schedule, so pricing the exit on a window that also
contains it counted that month twice.

    python scripts/sale_date_month_end_check.py

Uses 30 Bearfoot with a 4 Sep 2026 sale override and OPMCCORD's $300,000
contribution of 13 Aug 2026 — the case that surfaced it. The contribution is
injected because the local accounting snapshot ends 2026-06-02.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKS: list = []


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


def main() -> int:
    import pandas as pd
    from flask_app import create_app

    app = create_app()
    with app.app_context():
        from flask_app.services import data_service, compute_service
        from flask_app.config import Config
        from compute import total_loan_balance_at
        from planned_loans import twelve_month_noi_after_date
        from utils import add_months, month_end

        data = dict(data_service.get_data())
        acct = data["acct"]
        base = acct[(acct["InvestmentID"].astype(str).str.upper() == "30BEAR")
                    & (acct["InvestorID"].astype(str).str.upper() == "OPMCCORD")
                    & (acct["Typename"].astype(str)
                       .str.contains("Contribution", na=False))]
        proto = base.iloc[-1].copy()
        proto["EffectiveDate"] = dt.date(2026, 8, 13)
        proto["Amt"] = -300000.0
        data["acct"] = pd.concat([acct, pd.DataFrame([proto])], ignore_index=True)

        res = compute_service.get_cached_deal_result(
            "P0000001", Config.DEFAULT_START_YEAR, Config.DEFAULT_HORIZON_YEARS,
            Config.PRO_YR_BASE_DEFAULT, data, force=True,
            actuals_through="2026-08-31", sale_date_override="2026-09-04")

        print("\n1. Pref stops on the SALE DATE, not the month end")
        cf = res["cf_alloc"]
        pref = cf[(cf["PropCode"].astype(str).str.upper() == "OPMCCORD")
                  & (cf["vState"].astype(str) == "Pref")]
        chk("OPMCCORD has exactly one pref row (the closing period)",
            len(pref) == 1, f"{len(pref)} rows")
        if len(pref):
            row = pref.iloc[0]
            chk("dated 2026-09-04, the actual sale",
                str(row["event_date"])[:10] == "2026-09-04",
                str(row["event_date"])[:10])
            # 300,000 x 9% x 22/365, 13 Aug -> 4 Sep
            want = 300000 * 0.09 * 22 / 365
            chk(f"and is {want:,.2f} — 22 days, not 48",
                abs(row["Allocated"] - want) < 0.5,
                f"got {row['Allocated']:,.2f}; the month end gave "
                f"{300000 * 0.09 * 48 / 365:,.2f}")

        print("\n2. The monthly grid is untouched")
        chk("sale_me is still the month end", str(res.get("sale_me")) == "2026-09-30",
            str(res.get("sale_me")))
        chk("the closing settlement is reported in the diagnostics",
            any("Closing period settles on the sale date" in m
                for m in (res.get("debug_msgs") or [])))

        print("\n3. The loan is repaid on the sale date")
        ls = res.get("loan_sched")
        if ls is None or ls.empty:
            chk("loan schedule present", False)
        else:
            at_sale = total_loan_balance_at(ls, dt.date(2026, 9, 4))
            at_me = total_loan_balance_at(ls, dt.date(2026, 9, 30))
            chk("the balance at the sale exceeds the month-end balance",
                at_sale > at_me,
                f"${at_sale:,.2f} at 4 Sep vs ${at_me:,.2f} at 30 Sep — an extra "
                f"${at_sale - at_me:,.2f} of amortisation was being credited")

        print("\n4. Terminal NOI starts the month AFTER the month of sale")
        fc = res.get("fc_deal_modeled")
        sale_me = res.get("sale_me")
        if fc is None or fc.empty or sale_me is None:
            chk("forecast present", False)
        else:
            old = twelve_month_noi_after_date(fc, sale_me)
            new = twelve_month_noi_after_date(fc, add_months(month_end(sale_me), 1))
            got = (res.get("sale_dbg") or {}).get("NOI_12m_After_Sale")
            chk("the window the model used starts in October, not September",
                got is not None and abs(got - new) < 1.0,
                f"model {got:,.0f} vs Oct-start {new:,.0f} "
                f"(Sep-start would be {old:,.0f})")
            chk("the two windows genuinely differ", abs(new - old) > 1.0,
                f"{new - old:+,.0f} — the sale month is no longer counted twice")

        print("\n5. A month-end sale is unaffected")
        res2 = compute_service.get_cached_deal_result(
            "P0000001", Config.DEFAULT_START_YEAR, Config.DEFAULT_HORIZON_YEARS,
            Config.PRO_YR_BASE_DEFAULT, data, force=True,
            actuals_through="2026-08-31", sale_date_override="2026-09-30")
        cf2 = res2["cf_alloc"]
        p2 = cf2[(cf2["PropCode"].astype(str).str.upper() == "OPMCCORD")
                 & (cf2["vState"].astype(str) == "Pref")]
        chk("a 30 Sep sale still settles on 30 Sep",
            len(p2) == 1 and str(p2.iloc[0]["event_date"])[:10] == "2026-09-30",
            str(p2.iloc[0]["event_date"])[:10] if len(p2) else "no pref row")
        if len(p2):
            want48 = 300000 * 0.09 * 48 / 365
            chk(f"and accrues the full {want48:,.2f} — 48 days is correct there",
                abs(p2.iloc[0]["Allocated"] - want48) < 0.5,
                f"got {p2.iloc[0]['Allocated']:,.2f}")

    passed = sum(CHECKS)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
