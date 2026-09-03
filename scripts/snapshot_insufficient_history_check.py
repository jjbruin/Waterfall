"""Guardrail: the insufficient-history rule actually fires.

It did not. ``months_owned`` reads ``payload["general"]["date_closed"]``, and
the Portfolio Snapshot's lean provider (``_one_pager_provider``) skipped the
general block entirely for speed — so the date was always None, ``months_owned``
was always None, and the rule is guarded on ``mo is not None``. Measured before
the fix: **0 of 30** deals on the 26Q2 page had a computable ownership age, so a
brand-new acquisition was reported as though it had a full operating history.

The provider now carries the one field the rule needs, through
``one_pager.closing_date_from_row`` — the same precedence the full One Pager
applies, so the two paths cannot disagree, and no extra query is made.

``INSUFFICIENT_HISTORY_MONTHS`` stays at 3.0. The threshold was never the
problem.

    python scripts/snapshot_insufficient_history_check.py

These deals are newer than this snapshot and are not in it, so their ages are
asserted from their closing dates rather than read from rows we do not have.

CORRECTED Sep 3 2026 — the first version of this check dated them from their
**26Q1** ages (Hanestowne 0.4 months, Plaza Del Mar 0.5) while testing at a
**26Q2** quarter end, which back-solved closing dates three months too late and
made both look like certain suppressions. At 26Q2 they are 3.4 and 3.5 months
owned, so AT A 3.0 THRESHOLD THEY DO NOT SUPPRESS — only Presidential Arms
(1.6) does. That is the honest outcome of the shipped configuration and is
asserted here as such rather than smoothed over. The threshold is the report
author's call; 5.0 is the value that catches all three while still leaving
Jefferson Stephens (8.4) showing its figures.
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
        from flask_app.services.portfolio_snapshot_freeze import (
            _one_pager_provider, build_subtab,
        )
        from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
        from flask_app.services import portfolio_snapshot_operating as OP
        import one_pager

        d = data_service.get_data()
        q = dt.date(2026, 6, 30)
        resolved = resolve_investor_deals("TGAM", "2026-Q2",
                                          d.get("relationships_raw"), d["inv"])
        entries = [e for items in resolved["groups"].values() for e in items]
        entries += resolved.get("flagged") or []

        print("\n1. The provider supplies the field the rule reads")
        prov = _one_pager_provider(d)
        sample = prov(entries[0]["vcode"], "2026-Q2")
        chk("the payload carries a general block", "general" in sample,
            f"keys: {sorted(sample.keys())}")
        chk("with a closing date on it",
            (sample.get("general") or {}).get("date_closed") is not None)

        print("\n2. Ownership age now computes for every deal")
        none_now = [e["vcode"] for e in entries
                    if OP.months_owned(prov(e["vcode"], "2026-Q2"), q) is None]
        chk(f"months_owned resolves for all {len(entries)} deals", not none_now,
            "still None for: " + ", ".join(none_now[:6])
            + "   — was None for ALL of them before this fix")

        print("\n3. It agrees with the full One Pager path")
        # The two must not drift: same precedence, same parse, one definition.
        inv = d["inv"]
        drift = []
        for e in entries[:8]:
            row = inv[inv["vcode"].astype(str).str.upper() == e["vcode"].upper()]
            want = one_pager.closing_date_from_row(row.iloc[0]) if len(row) else None
            got = (prov(e["vcode"], "2026-Q2").get("general") or {}).get("date_closed")
            if want != got:
                drift.append(f"{e['vcode']}: {got} vs {want}")
        chk("the lean date matches closing_date_from_row", not drift,
            "; ".join(drift))

        print("\n4. The threshold is untouched")
        chk("INSUFFICIENT_HISTORY_MONTHS is still 3.0",
            OP.INSUFFICIENT_HISTORY_MONTHS == 3.0,
            str(OP.INSUFFICIENT_HISTORY_MONTHS))

        print("\n5. What the shipped 3.0 actually catches at 26Q2")
        # Ages measured at the quarter being reported, NOT carried over from
        # 26Q1 — see the correction note in the module docstring. Closing dates
        # are back-solved from the ages the report author quoted.
        for label, closed, want in (
                ("Presidential Arms  (1.6mo)", "2026-05-13", True),
                ("Hanestowne         (3.4mo)", "2026-03-19", False),
                ("Plaza Del Mar      (3.5mo)", "2026-03-16", False),
                ("Jefferson Stephens (8.4mo)", "2025-10-18", False),
                ("a deal owned 9.6 months",    "2025-09-10", False)):
            mo = OP.months_owned({"general": {"date_closed": closed}}, q)
            got = mo is not None and mo < OP.INSUFFICIENT_HISTORY_MONTHS
            chk(f"{label} -> {'suppressed' if want else 'shows values'}",
                got == want,
                f"{mo:.2f} months vs a {OP.INSUFFICIENT_HISTORY_MONTHS} threshold")

        print("\n6. The threshold, and only the threshold, separates them")
        # Stated so the next person does not have to rediscover it: the fix and
        # the threshold are independent. The date now REACHES the rule, which is
        # what shipped; which deals it then catches is a number Charlene owns.
        at_5 = [n for n, c in (("Presidential Arms", "2026-05-13"),
                               ("Hanestowne", "2026-03-19"),
                               ("Plaza Del Mar", "2026-03-16"))
                if OP.months_owned({"general": {"date_closed": c}}, q) < 5.0]
        chk("5.0 would catch all three of the new acquisitions",
            len(at_5) == 3, ", ".join(at_5))
        chk("and would still leave Jefferson Stephens showing values",
            OP.months_owned({"general": {"date_closed": "2025-10-18"}}, q) > 5.0,
            "8.38 months — so 5.0 is not a value that swallows the portfolio")

        print("\n7. No established deal on THIS page changes")
        # A null result, and stated as one: the youngest deal in the snapshot is
        # ~9.6 months, so nothing here crosses a 3.0 threshold. The deals the
        # fix is aimed at are newer than the snapshot.
        ages = []
        for e in entries:
            mo = OP.months_owned(prov(e["vcode"], "2026-Q2"), q)
            if mo is not None:
                ages.append((e["vcode"], mo))
        newly = [(v, m) for v, m in ages if m < OP.INSUFFICIENT_HISTORY_MONTHS]
        chk("nothing on the local 26Q2 page newly suppresses", not newly,
            str(newly))
        chk("because the youngest here is well clear of the threshold",
            ages and min(m for _, m in ages) > 3.0,
            f"youngest = {min(m for _, m in ages):.1f} months")

        op = build_subtab("operating", "TGAM", "2026-Q2", d, resolved)
        rows = [r for b in (op.get("groups") or {}).values() for r in b]
        shown = [r for r in rows if r.get("econ_occ_display") is not None]
        chk("and the Operating tab still renders its figures",
            len(shown) > 20, f"{len(shown)} of {len(rows)} deals show econ occ")

    passed = sum(CHECKS)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
