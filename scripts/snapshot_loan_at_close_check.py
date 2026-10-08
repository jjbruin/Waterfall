"""Guardrail -- the Loan tab's AT CLOSE rule (portfolio_snapshot_loan, AT_CLOSE_LABEL).

A deal with no complete calendar quarter of actual NOI yet -- from its
acquisition quarter through the report quarter -- shows its Loan ratios at close:
DSCR = NOI at close / 12 months of modeled debt service, Debt Yield = NOI at close
/ debt, LTV = debt / purchase price where no valuation exists. From its first
complete quarter it reads actuals.

Runs the real /bundle LOCALLY (test client, local SQLite):

    DB_PATH=C:/path/waterfall.db .venv/Scripts/python.exe \
        scripts/snapshot_loan_at_close_check.py

  A  every AT CLOSE row recomputes independently: NOI = the One Pager's
     noi.at_close (NOT deals.Close_Rev - Close_Exp), debt service = the first 12
     months of valuation_debt_service.monthly_schedule, LTV = debt / price
  B  BOTH DIRECTIONS: every AT CLOSE row has no complete quarter since its
     acquisition; every non-dev, unsold row with debt, no complete quarter and
     a positive NOI at close IS an AT CLOSE row; no row with a complete quarter is
  C  Presidential Arms (the case that prompted the rule) is AT CLOSE at 26Q2 and
     no longer prints the 3.81x the partial first statement produced
  D  with the window switched off, Presidential is not AT CLOSE (not vacuous)
"""
from __future__ import annotations

import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask_app import create_app                                  # noqa: E402
import flask_app.services.portfolio_snapshot_loan as L            # noqa: E402

FAILS: list = []
PRESIDENTIAL = "P0000119"


def chk(label, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def rows_of(x):
    if isinstance(x, dict):
        if "vcode" in x and "ytd_dscr" in x:
            yield x
        for v in x.values():
            yield from rows_of(v)
    elif isinstance(x, list):
        for v in x:
            yield from rows_of(v)


def main():
    app = create_app()
    c = app.test_client()
    j = c.post("/auth/login", json={"username": "admin", "password": "admin"}).get_json()
    tok = j.get("token") or j.get("access_token")

    with app.app_context():
        from flask_app.services import data_service
        from flask_app.services.portfolio_snapshot_freeze import _quarterly_noi_provider
        from flask_app.services.valuation_debt_service import monthly_schedule
        data = data_service.get_data()
        q_noi = _quarterly_noi_provider(data)

        def loan(q):
            r = c.get(f"/api/portfolio-snapshot/bundle?investor=TGAM&quarter={q}",
                      headers={"Authorization": "Bearer " + tok})
            assert r.status_code == 200, r.status_code
            return list(rows_of(r.get_json()["subtabs"]["loan"]["groups"]))

        def op_at_close(vc, q):
            d = c.get(f"/api/financials/{vc}/one-pager?quarter={q}",
                      headers={"Authorization": "Bearer " + tok}).get_json() or {}
            return ((d.get("property_performance") or {}).get("noi") or {}).get("at_close")

        for q in ("2026-Q2", "2026-Q3"):
            rows = loan(q)
            ac = [r for r in rows if r.get("at_close")]
            bad_a = []
            for r in ac:
                a = r["at_close"]
                acq = date.fromisoformat(a["acquired"])
                sched = (monthly_schedule(r["vcode"], acq, date(acq.year + 1, acq.month, 1),
                                          data).get("rows") or [])[:12]
                ds = sum(x["interest"] + x["principal"] for x in sched) if len(sched) == 12 else None
                noi = op_at_close(r["vcode"], q)
                ok = (abs(a["noi_at_close"] - noi) < 1
                      and abs(r["debt_yield"] - noi / r["debt"]) < 1e-12
                      and (ds is None and r["ytd_dscr"] is None
                           or abs(r["ytd_dscr"] - noi / ds) < 1e-12)
                      and (not a["ltv_from_price"]
                           or abs(r["ltv"] - r["debt"] / a["purchase_price"]) < 1e-12))
                if not ok:
                    bad_a.append(r["vcode"])
            chk(f"A {q}: every AT CLOSE row recomputes from the One Pager, the debt "
                "service engine and the purchase price", not bad_a, bad_a)

            inv = data["inv"]
            missing, wrong = [], []
            for r in rows:
                acq, _ = L._deal_acquisition(inv, r["vcode"])
                qs = L._quarters_from(acq, q) if acq else []
                no_quarter = bool(qs) and all(q_noi(r["vcode"], x) is None for x in qs)
                recent = bool(acq) and L._is_recent_acquisition(acq, q)
                if r.get("at_close") and not recent:
                    wrong.append(r["vcode"])
                eligible = (no_quarter and recent
                            and not r.get("is_dev") and not r.get("debt_free")
                            and not r.get("kept_despite_sold") and r.get("debt")
                            and r["vcode"] not in L.PROJECTED_YE_NOI_FALLBACK
                            and (op_at_close(r["vcode"], q) or 0) > 0)
                if eligible and not r.get("at_close"):
                    missing.append(r["vcode"])
                if r.get("at_close") and not no_quarter:
                    wrong.append(r["vcode"])
            chk(f"B {q}: every eligible deal is AT CLOSE", not missing, missing)
            chk(f"B {q}: no deal with a complete quarter is AT CLOSE", not wrong, wrong)

        # E -- NEW deals only. Crowne Plaza (acquired 2021, no actuals feed ever)
        # is the case that set AT_CLOSE_MAX_MONTHS; assert the boundary both ways.
        rec = L._is_recent_acquisition
        chk("E: the window admits new deals (Presidential 5/13/26, Hanestowne "
            "3/18/26 at 26Q2 and 26Q3)",
            rec(date(2026, 5, 13), "2026-Q2") and rec(date(2026, 3, 18), "2026-Q2")
            and rec(date(2026, 3, 18), "2026-Q3"))
        chk("E: the window refuses old deals (Crowne Plaza 11/18/2021) and "
            "deals not yet acquired",
            not rec(date(2021, 11, 18), "2026-Q2")
            and not rec(date(2026, 7, 30), "2026-Q2")
            and not rec(date(2025, 6, 30), "2026-Q2"))

        p = [r for r in loan("2026-Q2") if r["vcode"] == PRESIDENTIAL]
        chk("C: Presidential Arms is AT CLOSE at 26Q2 and no longer reads 3.81x",
            bool(p) and p[0].get("at_close") and round(p[0]["ytd_dscr"], 2) != 3.81,
            p[0].get("ytd_dscr") if p else None)

        saved = L._quarters_from
        L._quarters_from = lambda start, quarter: []
        try:
            p0 = [r for r in loan("2026-Q2") if r["vcode"] == PRESIDENTIAL]
        finally:
            L._quarters_from = saved
        chk("D: with the window off, Presidential is not AT CLOSE (check is not vacuous)",
            bool(p0) and not p0[0].get("at_close"))

    print(f"\n{len(FAILS)} failing")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
