"""Guardrail: Giant 7 drives DSCR and Debt Yield from its Projected YE NOI.

Prompt 4 of UPDATED_PortSnapshot_Debug_4 (Sep 2 2026).

Giant 7 (P0000019) is under PSA to sell and its NOI feed stopped in Nov 2025, so
there is no complete quarter of actual NOI and no YTD DSCR: both cells read an
em dash. Its One Pager still carries a Projected YE NOI (~9.2M), and that now
drives the two cells until the deal closes.

A PER-DEAL HARDCODE — ``PROJECTED_YE_NOI_FALLBACK`` — and the second on this
page after ``MANUAL_RATIO_SEEDS``. It is a FALLBACK rather than an override: it
fires only where the real input is missing, so the moment the feed resumes the
computed figures win and it becomes dead weight. One deletion to retire, and the
weekday `retire-manual-ratio-seeds` reminder tracks it alongside the seeds.

    python scripts/snapshot_giant7_ye_fallback_check.py

Giant 7's Projected YE NOI is absent from this snapshot too (its feed stopped
before the snapshot was taken), so the live value is injected to prove the path
fires — and the injection is stated rather than hidden.
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
        from flask_app.services import data_service
        from flask_app.services.portfolio_snapshot_freeze import build_subtab
        from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
        from flask_app.services import portfolio_snapshot_operating as OP
        from flask_app.services import portfolio_snapshot_loan as L
        import one_pager

        d = data_service.get_data()
        resolved = resolve_investor_deals("TGAM", "2026-Q2",
                                          d.get("relationships_raw"), d["inv"])

        print("\n4. Giant 7's fallback fires only where the real input is missing")
        chk("the fallback is scoped to one vcode",
            set(L.PROJECTED_YE_NOI_FALLBACK) == {"P0000019"},
            str(sorted(L.PROJECTED_YE_NOI_FALLBACK)))

        # Giant 7 has no Projected YE NOI in this snapshot either, so inject the
        # live one (~9.2M) to prove the path. Nothing else is touched.
        real = one_pager.get_property_performance

        def patched(vcode, *a, **k):
            perf = real(vcode, *a, **k)
            if str(vcode).strip().upper() == "P0000019":
                perf.setdefault("noi", {})["actual_ye"] = 9_200_000.0
                perf.setdefault("dscr", {})["actual_ye"] = 1.42
            return perf

        one_pager.get_property_performance = patched
        try:
            ln = build_subtab("loan", "TGAM", "2026-Q2", d, resolved)
        finally:
            one_pager.get_property_performance = real
        rows = {r["vcode"].upper(): r
                for b in (ln.get("groups") or {}).values() for r in b}
        g = rows.get("P0000019") or {}
        chk("YTD DSCR populates from the Projected YE ratio",
            g.get("ytd_dscr_display") is not None,
            f"{g.get('ytd_dscr_display')} — the One Pager's dscr.actual_ye, which "
            f"is already Projected YE NOI over debt service")
        dy, debt = g.get("debt_yield_display"), g.get("debt")
        chk("Debt Yield populates from Projected YE NOI over Debt",
            dy is not None and debt
            and abs(dy - 9_200_000.0 / debt) < 1e-9,
            f"{dy} vs 9,200,000 / {debt:,.0f} = {9_200_000.0 / debt:.4f}"
            if debt else str(dy))
        chk("and the row says the figure is temporary",
            any("TEMPORARY" in f for f in (g.get("flags") or [])),
            "; ".join(f for f in (g.get("flags") or []) if "TEMPORARY" in f)[:110])

        print("\n5. No other deal's basis moves")
        base = build_subtab("loan", "TGAM", "2026-Q2", d, resolved)
        brows = {r["vcode"].upper(): r
                 for b in (base.get("groups") or {}).values() for r in b}
        moved = [vc for vc, r in rows.items() if vc != "P0000019"
                 and (r.get("ytd_dscr_display") != brows.get(vc, {}).get("ytd_dscr_display")
                      or r.get("debt_yield_display") != brows.get(vc, {}).get("debt_yield_display"))]
        chk("every other deal's DSCR and Debt Yield are identical",
            not moved, str(moved))
        chk("a deal with a real quarter of NOI still uses it, not any fallback",
            all(not any("TEMPORARY" in f for f in (r.get("flags") or []))
                for vc, r in rows.items() if vc != "P0000019"))

    passed = sum(CHECKS)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
