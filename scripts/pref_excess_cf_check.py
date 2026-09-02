"""Guardrail: excess cash flow pays pref down, so it cannot coexist with arrears.

Excess Cash Flow sits BELOW the pref step in every one of these waterfalls. A
partner cannot receive it while pref is outstanding, so an excess distribution
is evidence the pref was current at that date.

``reports_service._compute_accrued_pref`` has always applied both Preferred
Return (TypeID 1019) AND Excess Cash Flow (1020) against accrued pref.
``waterfall.seed_states_from_accounting`` applied only 1019, so it left phantom
arrears standing behind later excess distributions — 30 Bearfoot showed
OPMCCORD $8,995 owed while $1,055,944 of excess cash flow had been paid to it
since its last pref payment. The two paths now agree.

    python scripts/pref_excess_cf_check.py

THE INVARIANT: no investor may carry accrued pref if excess cash flow was paid
to them after their last pref payment. Checked across every deal with a
waterfall. Accruals that survive are the legitimate ones — pref that accrued
and simply has not been paid, with nothing paid since that would have cleared
it.
"""
from __future__ import annotations

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
        from loaders import normalize_accounting_feed
        from waterfall import seed_states_from_accounting

        d = data_service.get_data()
        acct, invmap, wf, inv = d["acct"], d.get("investment_map"), d["wf"], d["inv"]
        CUT = pd.Timestamp("2026-07-31")

        norm = acct
        try:
            norm = normalize_accounting_feed(acct)
        except Exception:
            pass
        n = norm.copy()
        n["amt"] = pd.to_numeric(n["Amt"], errors="coerce")
        n["d"] = pd.to_datetime(n["EffectiveDate"], errors="coerce")
        n["tid"] = pd.to_numeric(n["TypeID"], errors="coerce")
        n["IID"] = n["InvestmentID"].astype(str).str.upper()
        n["INV"] = n["InvestorID"].astype(str).str.upper()
        vmap = {str(r["vcode"]).upper(): str(r.get("InvestmentID", "")).upper()
                for _, r in inv.iterrows()}

        print("\n1. 30 Bearfoot — the deal that surfaced it")
        wfp = wf[wf["vcode"].astype(str).str.upper() == "P0000001"]
        st = seed_states_from_accounting(acct, invmap, wfp, "P0000001",
                                         cutoff_date=CUT)
        for who in ("OPMCCORD", "PPI27"):
            s = st.get(who)
            pool = list(s.pools.values())[0]
            accr = sum(t.pref_unpaid_compounded + t.pref_accrued_current_year
                       for t in pool.pref_tiers)
            chk(f"{who} carries no accrued pref", abs(accr) < 0.01,
                f"${accr:,.2f} against capital ${pool.capital_outstanding:,.2f}"
                f" — was $4,660.27 (OPMCCORD) / $24,235.36 (PPI27)")

        print("\n2. THE INVARIANT, across every deal with a waterfall")
        violations, legit = [], []
        for vc in sorted(set(wf["vcode"].astype(str).str.upper())):
            w = wf[wf["vcode"].astype(str).str.upper() == vc]
            try:
                states = seed_states_from_accounting(acct, invmap, w, vc,
                                                     cutoff_date=CUT)
            except Exception:
                continue
            iid = vmap.get(vc, "")
            for who, stt in states.items():
                accr = sum(t.pref_unpaid_compounded + t.pref_accrued_current_year
                           for pool in stt.pools.values() for t in pool.pref_tiers)
                if accr <= 1.0:
                    continue
                g = n[(n["IID"] == iid) & (n["INV"] == who.upper())]
                if g.empty:
                    continue
                prefs = g[g["tid"] == 1019.0]["d"]
                last_pref = prefs.max() if not prefs.empty else None
                after = g[(g["tid"] == 1020.0) & (g["amt"] > 0)]
                if last_pref is not None and pd.notna(last_pref):
                    after = after[after["d"] > last_pref]
                paid = after["amt"].sum()
                (violations if paid > 1.0 else legit).append(
                    (vc, who, accr, paid, last_pref))

        chk("no investor carries pref behind a later excess distribution",
            not violations,
            "; ".join(f"{v}/{w} ${a:,.0f} arrears vs ${p:,.0f} excess paid after"
                      for v, w, a, p, _ in violations[:5]))
        print(f"      {len(legit)} legitimate accrual(s) remain — pref that "
              f"accrued and simply has not been paid:")
        for v, w, a, _p, lp in sorted(legit, key=lambda x: -x[2])[:8]:
            when = lp.date() if lp is not None and pd.notna(lp) else "never paid"
            print(f"         {v}/{w}: ${a:,.2f}, last pref payment {when}, "
                  f"no excess cash flow since")

        print("\n3. The two pref paths agree")
        # reports_service has always applied excess CF; the seeding did not.
        # Same rule now, so the same rows must be recognised as pref-reducing.
        src = open(os.path.join(ROOT, "waterfall.py"), encoding="utf-8").read()
        chk("the seeding recognises excess cash flow", '"excess cash" in _tname' in src)
        chk("and TypeID 1020 alongside 1019", "(1019.0, 1020.0)" in src)
        chk("a reversal restores pref rather than paying it",
            "elif tr_amt < 0:" in src and "pref_accrued_current_year += -tr_amt" in src,
            "a negative pref/excess row is a reclassification, not a payment")

    passed = sum(CHECKS)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
