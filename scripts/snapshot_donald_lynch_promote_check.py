"""Guardrail -- Donald Lynch (MCCORD) is reported under P0000073, the vcode with data.

One-off, see REPORT_VCODE_PROMOTE in portfolio_snapshot_service. Runs the real
/bundle LOCALLY (test client, local SQLite):

    DB_PATH=C:/path/waterfall.db .venv/Scripts/python.exe \
        scripts/snapshot_donald_lynch_promote_check.py

Both directions, and non-vacuous by emptying the promotion:
  A  Declaration (DCXVIA) 26Q2: the Donald Lynch row is P0000073, P0000049 is
     on no subtab, and its debt / partner equity / total cap are real (> 0)
  B  26Q3: still on the page, marked (Sold), stack read at 2026-Q2, debt n/a
  C  TIAA (TGAM) never shows Donald Lynch -- no route, so the one-off must not
     add it
  D  with REPORT_VCODE_PROMOTE emptied the row reverts to P0000049 with zero
     total cap -- i.e. A fails, so A is testing something
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask_app import create_app                                  # noqa: E402
import flask_app.services.portfolio_snapshot_service as S         # noqa: E402

FAILS: list = []


def chk(label, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def rows(tab):
    def walk(x):
        if isinstance(x, dict):
            if "vcode" in x and "name" in x:
                yield x
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)
    return [r for r in walk(tab.get("groups") or {})]


def donald(bundle, tab):
    return [r for r in rows(bundle["subtabs"][tab]) if r["vcode"] in ("P0000049", "P0000073")]


def main():
    app = create_app()
    c = app.test_client()
    j = c.post("/auth/login", json={"username": "admin", "password": "admin"}).get_json()
    tok = j.get("token") or j.get("access_token")

    def get(inv, q):
        r = c.get(f"/api/portfolio-snapshot/bundle?investor={inv}&quarter={q}",
                  headers={"Authorization": "Bearer " + tok})
        assert r.status_code == 200, r.status_code
        return r.get_json()

    q2 = get("DCXVIA", "2026-Q2")
    fin = donald(q2, "financial")
    chk("A: one Donald Lynch row on Financial, vcode P0000073",
        [r["vcode"] for r in fin] == ["P0000073"], [r["vcode"] for r in fin])
    chk("A: P0000049 appears on no subtab",
        not any(r["vcode"] == "P0000049" for t in ("financial", "loan", "operating")
                for r in donald(q2, t)))
    f = fin[0] if fin else {}
    chk("A: debt, partner equity and total cap are real at 26Q2",
        all((f.get(k) or 0) > 0 for k in ("debt", "ptr_equity", "total_cap")),
        {k: f.get(k) for k in ("debt", "ptr_equity", "total_cap")})
    ln = donald(q2, "loan")
    chk("A: Loan row carries a computed DSCR", bool(ln) and (ln[0].get("ytd_dscr") or 0) > 0)

    q3 = get("DCXVIA", "2026-Q3")
    f3 = (donald(q3, "financial") or [{}])[0]
    chk("B: 26Q3 kept, marked (Sold), stack at 2026-Q2, debt n/a",
        f3.get("vcode") == "P0000073" and f3.get("sold_label") == "(Sold)"
        and f3.get("stack_quarter") == "2026-Q2" and f3.get("debt_display") == "n/a",
        {k: f3.get(k) for k in ("vcode", "sold_label", "stack_quarter", "debt_display")})

    for q in ("2026-Q2", "2026-Q3"):
        t = get("TGAM", q)
        chk(f"C: TIAA {q} has no Donald Lynch row",
            not any(donald(t, tab) for tab in ("financial", "loan", "operating")))

    saved = dict(S.REPORT_VCODE_PROMOTE)
    S.REPORT_VCODE_PROMOTE.clear()
    try:
        f0 = (donald(get("DCXVIA", "2026-Q2"), "financial") or [{}])[0]
    finally:
        S.REPORT_VCODE_PROMOTE.update(saved)
    chk("D: without the promotion the row reverts to the empty stub (check is not vacuous)",
        f0.get("vcode") == "P0000049" and not (f0.get("total_cap") or 0),
        {k: f0.get(k) for k in ("vcode", "total_cap")})

    print(f"\n{len(FAILS)} failing")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
