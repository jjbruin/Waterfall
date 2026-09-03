"""Guardrail: a deal reported after its sale carries no loan.

The asset and its facility went together, so Rate, Maturity, Debt, YTD DSCR,
LTV and Debt Yield all describe something that no longer exists. Every one
reads an em dash, and none of them feeds a subtotal.

This closes a real disagreement between two pages of the same report: East
Manchester's FINANCIAL row said Debt n/a (SOLD_NA_CELLS, v409) while its LOAN
row printed 9,641,912 — and summed that figure into the Individual Investments
and Portfolio Debt totals, and its 48.9% LTV into the weighted average.

Keyed on the sale via ``kept_despite_sold``, not on a vcode.

    python scripts/snapshot_loan_sold_suppression_check.py

The local snapshot carries Sale_Date 12/01/2030 for East Manchester where live
has 6/25/2026, so the sold gate never fires here; the check injects the live
date. City West (foreclosed 8/30/2025) fires without help and is the control
that the injection is not doing the work.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKS: list = []
COLS = ("rate_display", "maturity_display", "debt_display",
        "ytd_dscr_display", "ltv_display", "debt_yield_display")


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


def _loan(inject_sale: bool):
    from flask_app.services import data_service
    from flask_app.services.portfolio_snapshot_freeze import build_subtab
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    d = dict(data_service.get_data())
    inv = d["inv"].copy()
    if inject_sale:
        inv.loc[inv["vcode"].astype(str).str.upper() == "P0000017",
                "Sale_Date"] = "6/25/2026"
    d["inv"] = inv
    resolved = resolve_investor_deals("TGAM", "2026-Q2",
                                      d.get("relationships_raw"), inv)
    ln = build_subtab("loan", "TGAM", "2026-Q2", d, resolved)
    rows = {r["vcode"].upper(): r
            for b in (ln.get("groups") or {}).values() for r in b}
    rows.update({r["vcode"].upper(): r for r in (ln.get("ownership_flagged") or [])})
    return ln, rows


def main() -> int:
    from flask_app import create_app
    app = create_app()
    with app.app_context():
        ln, rows = _loan(inject_sale=True)
        em, cw = rows.get("P0000017"), rows.get("PCITWES")

        print("\n1. Every Loan column reads an em dash on a sold deal")
        for vc, r, nm in (("P0000017", em, "East Manchester"),
                          ("PCITWES", cw, "City West")):
            if r is None:
                chk(f"{nm} is on the page", False)
                continue
            chk(f"{nm}: flagged kept_despite_sold", r.get("kept_despite_sold") is True)
            blank = [c for c in COLS if r.get(c) is not None]
            chk(f"{nm}: all six columns blank", not blank,
                "still showing: " + ", ".join(f"{c}={r.get(c)}" for c in blank))

        print("\n2. The underlying figures are NOT destroyed")
        chk("East Manchester's debt is still 9,641,912 underneath",
            em is not None and abs((em.get("debt") or 0) - 9_641_912.0) < 1.0,
            f"{em.get('debt') if em else None} — display only, every audit still sees it")
        chk("and its computed LTV survives too",
            em is not None and em.get("ltv") is not None,
            str(em.get("ltv") if em else None))

        print("\n3. Nothing suppressed feeds a subtotal")
        subs = ln.get("subtotals") or {}
        ind = subs.get("Individual Investments") or {}
        chk("Individual Investments Debt excludes the 9,641,912",
            abs((ind.get("debt") or 0) - 211_873_179.0) < 1.0,
            f"{ind.get('debt'):,.0f} — was 221,515,091 with the sold balance in it")
        chk("and its weighted LTV excludes the sold deal's 48.9%",
            ind.get("ltv") is not None and abs(ind["ltv"] - 0.6825072890774447) < 1e-9,
            f"{ind.get('ltv'):.4f} — was 0.6741")
        chk("the sold deals are not counted in the LTV population",
            ind.get("ltv_n") == 4, f"ltv_n = {ind.get('ltv_n')}")

        print("\n4. No other deal moves")
        base_ln, base_rows = _loan(inject_sale=False)
        base_subs = base_ln.get("subtotals") or {}
        moved = []
        for g, st in subs.items():
            b = base_subs.get(g) or {}
            if g == "Individual Investments":
                continue          # the one group holding both sold deals
            for k in ("debt", "ltv", "ytd_dscr", "debt_yield"):
                if (st.get(k) or 0) != (b.get(k) or 0):
                    moved.append(f"{g}.{k}")
        chk("every other fund subtotal is identical", not moved, ", ".join(moved))
        g7 = rows.get("P0000019") or {}
        chk("Giant 7, a live deal, is untouched",
            g7.get("rate_display") == "3.9% fixed"
            and g7.get("debt_display") is not None,
            f"{g7.get('rate_display')} / debt {g7.get('debt_display')}")

        print("\n5. City West fires WITHOUT the injection")
        # It was foreclosed 8/30/2025, which the local snapshot does carry, so
        # this proves the rule keys on the sale and not on the test's fixture.
        cw_base = base_rows.get("PCITWES") or {}
        chk("City West is suppressed on unmodified local data",
            cw_base.get("kept_despite_sold") is True
            and all(cw_base.get(c) is None for c in COLS),
            "the injection only supplies East Manchester's live sale date")

    passed = sum(CHECKS)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
