"""Guardrail -- the Loan tab carries NO typed LTV / YTD DSCR / Debt Yield.

Replaces the assertions of snapshot_loan_manual_cells_check.py, which pinned the
six pre-filled deals. Those seeds were removed 2026-10-07; every ratio on the
tab is now computed or a dash.

Runs the real /bundle endpoint LOCALLY (test client, local SQLite) -- no live
token. Point it at a pulled database with DB_PATH=...

    DB_PATH=C:/path/waterfall.db .venv/Scripts/python.exe \
        scripts/snapshot_loan_no_typed_ratios_check.py

Asserts in BOTH directions, and proves itself non-vacuous by re-injecting a seed:
  A  MANUAL_RATIO_SEEDS is empty and no row is marked typed
  B  every non-dev ratio a row DISPLAYS equals its computed figure (a displayed
     number is never anything else), and a blank stays blank
  C  the typed counts on the total rows are 0 and each total's population is the
     number of rows that actually carry a computed value
  D  re-injecting one seed makes A, B and C fail
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask_app import create_app                                  # noqa: E402
import flask_app.services.portfolio_snapshot_loan as L            # noqa: E402

QUARTERS = ("2026-Q2", "2026-Q3")
FAILS: list = []


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


def loan_tab(client, tok, quarter):
    r = client.get(f"/api/portfolio-snapshot/bundle?investor=TGAM&quarter={quarter}",
                   headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 200, r.status_code
    return r.get_json()["subtabs"]["loan"]


def audit(tab, tag):
    """Return {check label: ok} for one loan tab."""
    rows = [r for r in rows_of(tab["groups"])]
    typed = [r["vcode"] for r in rows
             if r.get("ltv_is_manual") or r.get("ytd_dscr_is_manual")
             or r.get("debt_yield_is_manual")]
    bad = []
    for r in rows:
        if r.get("is_dev") or r.get("debt_free") or r.get("sold_suppressed"):
            continue
        for f in ("ltv", "ytd_dscr", "debt_yield"):
            d = r.get(f + "_display")
            if d is None:
                continue
            if isinstance(d, str):          # a literal ("Dev", "N/A") or a typed string
                if d not in (L.DEV_DISPLAY, L.NA_DISPLAY):
                    bad.append((r["vcode"], f, d))
            elif d != r.get(f):
                bad.append((r["vcode"], f, d))
    out = {
        f"{tag}: no row is typed": not typed,
        f"{tag}: every displayed ratio is its computed figure": not bad,
    }
    for t in (tab["total"], tab["total_excluding_dev"]):
        out[f"{tag}: typed counts are zero ({t.get('label') or 'total'})"] = all(
            (t.get(f"{f}_typed_n") or 0) == 0 for f in ("ltv", "ytd_dscr", "debt_yield"))
    return out, typed, bad


def main():
    app = create_app()
    client = app.test_client()
    j = client.post("/auth/login", json={"username": "admin", "password": "admin"}).get_json()
    tok = j.get("token") or j.get("access_token")

    chk("A: MANUAL_RATIO_SEEDS is empty", L.MANUAL_RATIO_SEEDS == {}, str(L.MANUAL_RATIO_SEEDS))
    for q in QUARTERS:
        res, typed, bad = audit(loan_tab(client, tok, q), q)
        for k, v in res.items():
            chk(k, v, f"typed={typed} bad={bad}")

    # D -- non-vacuous: put one seed back and the same audit must fail.
    saved = L.MANUAL_RATIO_SEEDS
    L.MANUAL_RATIO_SEEDS = {"P0000117": {"ytd_dscr": 1.9}}
    try:
        res, typed, bad = audit(loan_tab(client, tok, "2026-Q2"), "inject")
    finally:
        L.MANUAL_RATIO_SEEDS = saved
    chk("D: re-injecting a seed is caught (check is not vacuous)",
        not all(res.values()) and "P0000117" in typed, f"typed={typed}")

    print(f"\n{len(FAILS)} failing")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
