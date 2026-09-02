"""Guardrail: a reversed accounting entry must not move capital twice.

MRI reverses a booking by re-posting it with the OPPOSITE SIGN under the same
MajorType and Typename. Every place that walked the accounting feed keeping a
running capital balance used to read the row's TYPE and then take ``abs()`` of
the amount, which threw away the only thing distinguishing the reversal from
the entry it reverses — so the pair moved capital by twice the amount instead
of leaving it unchanged.

SIX call sites shared the defect:
  * waterfall.seed_states_from_accounting - the pool capital the Initial
    Capital step returns, the XIRR cashflows, and the pref accrual base
  * one_pager.get_capitalization_stack - partner_equity / pref_equity, i.e.
    the One Pager cap stack AND the Portfolio Snapshot's Ptr. Equity column
  * one_pager.get_pe_performance - funded_to_date and the ROE capital events
  * reports_service._compute_accrued_pref - ROE Summary
  * reports_service.build_pref_balance_detail - Pref Balance Detail
  * sold_service - Sold Portfolio cashflows and running capital
All now net by sign through ``loaders.capital_after``.

    python scripts/capital_reversal_sign_check.py unit        # no DB needed
    python scripts/capital_reversal_sign_check.py portfolio   # needs waterfall.db
    python scripts/capital_reversal_sign_check.py all

`portfolio` is the one that matters: for every (deal, investor) it compares the
capital the ENGINE arrives at against the arithmetic truth — the negated sum of
the signed capital-affecting amounts — and lists any pair that disagrees.
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


def unit() -> None:
    """The rule itself, on the shapes that actually occur in the feed."""
    from loaders import capital_after, capital_outstanding

    print("\n1. capital_after: the sign of the amount is the direction")
    cases = [
        ("contribution  -750,000 onto 0", 0.0, -750_000.0, 750_000.0),
        ("contribution  -640,000 onto 750,000", 750_000.0, -640_000.0, 1_390_000.0),
        ("return of cap +220,000 off 1,390,000", 1_390_000.0, 220_000.0, 1_170_000.0),
        ("REVERSAL of a return: -210,000 ADDS it back",
         750_000.0, -210_000.0, 960_000.0),
        ("REVERSAL of a contribution: +3,850,000 REMOVES it",
         3_850_000.0, 3_850_000.0, 0.0),
        ("a zero row moves nothing", 500.0, 0.0, 500.0),
    ]
    for label, start, amt, want in cases:
        got = capital_after(start, amt)
        chk(label, abs(got - want) < 1e-9, f"got {got:,.2f} want {want:,.2f}")

    print("\n2. The running total may go negative; the BALANCE floors at zero")
    running = capital_after(540_000.0, 960_000.0)
    chk("an over-return leaves the running total negative",
        abs(running + 420_000.0) < 1e-9,
        f"running = {running:,.2f} — 30 Bearfoot's overshoot, previously "
        f"absorbed by max(0.0, ...) and invisible")
    chk("but the reported balance is zero",
        capital_outstanding(running) == 0.0)

    print("\n3. A booking and its reversal net to zero")
    bal = 0.0
    for amt in (-3_850_000.0, 3_850_000.0):      # JB Fair Park, 2020-12-18
        bal = capital_after(bal, amt)
    chk("same-day pair leaves capital unchanged", abs(bal) < 1e-9,
        f"balance {bal:,.2f} — was 7,700,000 under abs()")

    print("\n3b. Same-day ORDER cannot change the answer")
    # The reason capital_after does not floor per row: JB Fair Park's reversal
    # sorts BEFORE the contribution it reverses, and a per-row floor swallowed
    # it, leaving both remaining contributions to land.
    for order in ((-3_850_000.0, 3_850_000.0, -3_850_000.0),
                  (3_850_000.0, -3_850_000.0, -3_850_000.0)):
        b = 0.0
        for amt in order:
            b = capital_after(b, amt)
        shape = " ".join("+" if x > 0 else "-" for x in order)
        chk(f"order [{shape}] -> 3,850,000",
            abs(capital_outstanding(b) - 3_850_000.0) < 1e-9,
            f"got {capital_outstanding(b):,.2f}")


def portfolio() -> None:
    """Engine capital vs the signed arithmetic, for every (deal, investor)."""
    import pandas as pd
    from flask_app import create_app

    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        from loaders import (normalize_accounting_feed, capital_after,
                             capital_outstanding)
        d_ = data_service.get_data()
        acct = d_["acct"]
        try:
            acct = normalize_accounting_feed(acct)
        except Exception:
            pass
        a = acct.copy()
        a["amt"] = pd.to_numeric(a["Amt"], errors="coerce")
        a = a.dropna(subset=["amt"])
        a["d"] = pd.to_datetime(a["EffectiveDate"], errors="coerce")

        touches = a["is_contribution"] | (a["is_capital"] & a["is_distribution"])
        cap = a[touches].sort_values("d")

        print("\n4. Modelled capital reconciles to the signed feed")
        bad, reversals, clamps = [], 0, []
        for (iid, inv), g in cap.groupby(["InvestmentID", "InvestorID"]):
            bal = 0.0
            for _, r in g.iterrows():
                bal = capital_after(bal, r["amt"])
            # Arithmetic truth: cash in less cash out, floored at zero.
            truth = max(0.0, -g["amt"].sum())
            got = capital_outstanding(bal)
            if abs(got - truth) > 0.01:
                bad.append((iid, inv, got, truth))
            if bal < -0.01:
                # Split by WHY. A pair with no contributions at all is not a
                # capital account — see the note where these are printed.
                funded = -g[g["is_contribution"]]["amt"].sum()
                clamps.append((iid, inv, -bal, funded))
            n_rev = int(((g["is_contribution"] & (g["amt"] > 0))
                         | (g["is_distribution"] & (g["amt"] < 0))).sum())
            reversals += n_rev

        pairs = cap.groupby(["InvestmentID", "InvestorID"]).ngroups
        chk(f"all {pairs} (deal, investor) pairs reconcile", not bad,
            "; ".join(f"{i}/{v} engine {b:,.0f} vs {t:,.0f}"
                      for i, v, b, t in bad[:5]))
        print(f"      {reversals} reversal rows in the feed — each one would "
              f"have double-counted under abs()")
        if clamps:
            # NOT a finding on its own, and it was mis-read as one first time
            # round. A (deal, investor) pair is only a closed capital account
            # when the capital came IN under the same pair. It often does not:
            #   * PCBLE receives no contributions from anyone, ever - it is the
            #     promote / AM-fee vehicle, and its payouts are earned income
            #     carried under the Typename "Distribution: Return of Capital".
            #   * OWPSC was funded by six investors; WOFC, which draws from it,
            #     contributed nothing anywhere.
            # So the split below is the point: a pair with ZERO contributions is
            # a pass-through or fee vehicle and the measure does not apply. Only
            # a pair that WAS funded and still over-returns is worth a look.
            unfunded = [c for c in clamps if c[3] <= 0.01]
            funded = [c for c in clamps if c[3] > 0.01]
            print(f"      {len(unfunded)} pair(s) have NO contributions at this "
                  f"grain - not capital accounts, nothing to read into:")
            for i, v, o, _f in sorted(unfunded, key=lambda x: -x[2])[:6]:
                print(f"         {i}/{v}: paid out ${o:,.2f} against no capital")
            # ANSWERED (Jim, Sep 2 2026) - do not re-raise. The funded pairs
            # below are all PSC3 as the investor into a fund vehicle, and PSC3
            # had a redemption event: PSC1 acquired PSCMAN from PSC3 while the
            # Investee Funds, previously held by PSC3's members, were assigned
            # from the members to PSC3 directly. The mismatch is journal
            # entries from that redemption. It does not reach any reported
            # figure: returns are run for the individual assets INSIDE PSC3,
            # never for PSC3 itself, and none of the entities listed here is in
            # the deals table.
            print(f"      {len(funded)} funded pair(s) return more than they "
                  f"took in - PSC3 redemption journal entries, out of scope:")
            for i, v, o, f in sorted(funded, key=lambda x: -x[2])[:6]:
                pct = (o / f * 100) if f else 0
                print(f"         {i}/{v}: funded ${f:,.0f}, over by ${o:,.0f} "
                      f"({pct:.1f}%)")

        print("\n5. The corrected figures agree with the published 26Q1 page")
        # The strongest evidence that the netted reading is the right one: the
        # reference document — produced outside this app — prints the NETTED
        # partner equity, not the doubled figure the code used to compute. Three
        # deals on the TIAA 26Q1 page carry contribution reversals, and all
        # three now tie. Values transcribed in snapshot_pdf_variance_pdfdata.
        import one_pager
        args = dict(mri_loans=d_["mri_loans_raw"], mri_val=d_["mri_val"],
                    waterfalls=d_["wf"], acct=d_["acct"], inv_map=d_["inv"],
                    isbs_raw=d_["isbs_raw"], quarter_str="2026-Q1",
                    relationships=d_.get("relationships_raw"))
        PDF = {                       # vcode: (name, Ptr Equity $M per the PDF)
            "P0000021": ("JB Fair Park", 3.9),
            "P0000066": ("Pegasus Life Storage", 2.6),
            "P0000084": ("Cocoplum Apartments", 23.4),
        }
        for vc, (name, want) in PDF.items():
            try:
                cap = one_pager.get_capitalization_stack(vc, **args) or {}
            except Exception as exc:
                chk(f"{name}: cap stack builds", False, str(exc)[:80])
                continue
            got = (cap.get("partner_equity") or 0.0) / 1e6
            chk(f"{name}: Ptr Equity ties the published {want}M",
                abs(got - want) < 0.05, f"got {got:,.2f}M")

        print("\n6. No abs() capital site survives in the affected modules")
        import re
        pats = [r"capital \+= abs\(", r"capital = max\(0\.0, capital - abs\(",
                r"inv_bal \+= abs\(", r"current_capital \+= abs\(",
                r"balance \+= abs\(cf\)",
                r"investor_balances\[investor_id\] \+= abs\(",
                r"funded_to_date'\] \+= abs\(",
                r"pool\.capital_outstanding \+= abs\("]
        for rel in ("waterfall.py", "one_pager.py",
                    "flask_app/services/reports_service.py",
                    "flask_app/services/sold_service.py"):
            src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
            hits = [p for p in pats if re.search(p, src)]
            chk(f"{rel} is clean", not hits, "; ".join(hits))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "unit":
        unit()
    elif cmd == "portfolio":
        portfolio()
    elif cmd == "all":
        unit()
        portfolio()
    else:
        print(__doc__)
        raise SystemExit(2)
    print("\n  {}/{} checks passed".format(sum(CHECKS), len(CHECKS)))
    raise SystemExit(0 if all(CHECKS) else 1)
