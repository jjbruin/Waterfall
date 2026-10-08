"""Guardrail -- the Loan tab's Debt Yield basis says what was actually divided.

Giant 7 (PROJECTED_YE_NOI_FALLBACK) divides Projected YE NOI -- with no 2026
actuals, the full-year 2026 budget -- yet its `debt_yield_basis` read
"single-quarter Interim IS NOI x 4 / debt" like every other row. The field is
not on the page; it is what the assistant's field trace quotes as the basis.

Runs the real /bundle LOCALLY (test client, local SQLite):

    DB_PATH=C:/path/waterfall.db .venv/Scripts/python.exe \
        scripts/snapshot_debt_yield_basis_check.py

Both directions, and non-vacuous:
  A  Giant 7: basis names Projected YE NOI and the budget, and its numerator
     (annualised_noi) is that Projected YE figure, not quarter NOI x 4
  B  every other row with a Debt Yield keeps the single-quarter basis, and its
     numerator really is quarter NOI x 4
  C  re-injecting the old fixed text makes A fail
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask_app import create_app                                  # noqa: E402
import flask_app.services.portfolio_snapshot_loan as L            # noqa: E402

FAILS: list = []
GIANT7 = "P0000019"


def chk(label, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def rows_of(x):
    if isinstance(x, dict):
        if "vcode" in x and "debt_yield_basis" in x:
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

    def loan(q):
        r = c.get(f"/api/portfolio-snapshot/bundle?investor=TGAM&quarter={q}",
                  headers={"Authorization": "Bearer " + tok})
        assert r.status_code == 200, r.status_code
        return list(rows_of(r.get_json()["subtabs"]["loan"]["groups"]))

    def giant7_ok(rows):
        g = [r for r in rows if r["vcode"] == GIANT7]
        return bool(g) and g[0]["debt_yield_basis"].startswith("Projected YE NOI") \
            and "budget" in g[0]["debt_yield_basis"] \
            and g[0]["debt_yield"] is not None \
            and abs(g[0]["annualised_noi"] / g[0]["debt"] - g[0]["debt_yield"]) < 1e-12 \
            and g[0]["quarter_noi"] is None, (g[0] if g else {})

    for q in ("2026-Q2", "2026-Q3"):
        rows = loan(q)
        ok, g = giant7_ok(rows)
        chk(f"A {q}: Giant 7 basis names the Projected YE budget and matches its numerator",
            ok, {k: g.get(k) for k in ("debt_yield_basis", "annualised_noi", "quarter_noi")})
        bad = [r["vcode"] for r in rows
               if r["vcode"] != GIANT7 and r.get("debt_yield") is not None
               and (r["debt_yield_basis"] != L.DY_BASIS_QUARTER
                    or abs((r["quarter_noi"] or 0) * 4 - (r["annualised_noi"] or 0)) > 1e-6)]
        chk(f"B {q}: every other Debt Yield is single-quarter x 4, and says so", not bad, bad)

    src = open(L.__file__, encoding="utf-8").read()
    chk("C: the basis is set per row, not as one fixed literal (check is not vacuous)",
        '"debt_yield_basis": dy_basis' in src
        and '"debt_yield_basis": "single-quarter' not in src)
    # Behavioural: with the fallback switched off, Giant 7 must fail A.
    saved = L.PROJECTED_YE_NOI_FALLBACK
    try:
        L.PROJECTED_YE_NOI_FALLBACK = frozenset()
        ok, _ = giant7_ok(loan("2026-Q2"))
    finally:
        L.PROJECTED_YE_NOI_FALLBACK = saved
    chk("C: with the fallback off, Giant 7 no longer passes A (A is testing the fallback)", not ok)

    print(f"\n{len(FAILS)} failing")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
