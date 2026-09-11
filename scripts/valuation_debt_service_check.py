#!/usr/bin/env python
"""Guardrail: modeled debt service in the valuation comparison.

The appraiser's Argus download is unlevered, so the Valuation column showed 0 interest,
0 principal and a blank DSCR. This lays the deal's OWN modeled debt service into the
Budget and Valuation columns. Driven against real loans and the real comparison, because
every assertion here is about a number the team will act on.

WHAT EACH RULE IS DEFENDING:

  * 5190, NOT 7030. The comparison's Interest row reads
    config.IS_ACCOUNTS['DEBT_SERVICE']['Interest'] == ['5190']. The AM forecast writes
    interest to list(INTEREST_ACCTS)[0], which yields 7030 out of that set. Layering in
    7030 would put interest on a row the comparison does not display — the figure would
    be there and invisible.

  * THE ESTIMATE COLUMN IS NOT TOUCHED. It means actuals plus budget for the remaining
    months, and its interest is interest actually paid. A modeled figure is not an
    improvement on a reported one.

  * BALLOONS ARE EXCLUDED, matching compute.py. A balloon is repaid from sale proceeds;
    counting it as operating debt service would collapse DSCR in the maturity year.

  * CHILD PROPERTIES COUNT. Loans aggregate up from properties to the parent deal, so a
    portfolio deal whose debt sits on its children must not model zero.

  * "NO LOANS" IS NOT ZERO. A deal we cannot model reports unavailable with a reason. A
    0 that means "could not model" is indistinguishable from a deal with no debt, and
    the second is a real and different thing.

Run:  .venv/Scripts/python.exe scripts/valuation_debt_service_check.py
"""
from __future__ import annotations

import os
import sys
from datetime import date

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    import config
    from flask_app import create_app
    from flask_app.db import get_engine
    from flask_app.services import data_service as ds
    from flask_app.services import valuation_debt_service as vds
    from flask_app.services import valuation_service as vs

    app = create_app()
    with app.app_context():
        engine = get_engine()
        data = ds.load_all(db_path="waterfall.db")

        print("1. The account the comparison actually reads")
        chk("interest goes to 5190", vds.INTEREST_ACCOUNT == "5190",
            vds.INTEREST_ACCOUNT)
        chk("which is what IS_ACCOUNTS['DEBT_SERVICE']['Interest'] contains",
            config.IS_ACCOUNTS["DEBT_SERVICE"]["Interest"] == [vds.INTEREST_ACCOUNT],
            f"{config.IS_ACCOUNTS['DEBT_SERVICE']['Interest']}")
        chk("and NOT the 7030 the AM forecast happens to pick out of INTEREST_ACCTS",
            list(config.INTEREST_ACCTS)[0] != int(vds.INTEREST_ACCOUNT),
            "the sets now agree — re-check whether this divergence is still intended")
        chk("principal goes to 7060", vds.PRINCIPAL_ACCOUNT == "7060")
        chk("which is what PRINCIPAL_ACCTS holds",
            {int(vds.PRINCIPAL_ACCOUNT)} == config.PRINCIPAL_ACCTS,
            f"{config.PRINCIPAL_ACCTS}")

        print("\n2. A real deal's schedule, from its real loan terms")
        ml = data["mri_loans_raw"]
        col = next(c for c in ml.columns if c.lower() == "vcode")
        top = ml[col].astype(str).str.strip().value_counts()
        vcode = top.index[0]
        sch = vds.for_year(vcode, 2026, data)
        chk(f"{vcode} models", sch["available"] is True, f"{sch['notes']}")
        chk("twelve months", len(sch["rows"]) == 12, f"got {len(sch['rows'])}")
        chk("interest is positive", (sch["interest"] or 0) > 0, f"{sch['interest']}")
        chk("months sum to the annual total",
            abs(sum(r["interest"] for r in sch["rows"]) - sch["interest"]) < 0.05
            and abs(sum(r["principal"] for r in sch["rows"]) - sch["principal"]) < 0.05)
        chk("every loan on the deal is counted",
            sch["loan_count"] == int(top.iloc[0]), f"{sch['loan_count']} vs {top.iloc[0]}")
        print(f"        {vcode} 2026: interest {sch['interest']:,.0f}  "
              f"principal {sch['principal']:,.0f}  ({sch['loan_count']} loans)")

        print("\n3. It ties to the engine Deal Analysis uses")
        import loans as L
        import pandas as pd
        sub = ml[ml[col].astype(str).str.strip().str.lower() == vcode.lower()]
        objs = L.build_loans_from_mri_loans(sub)
        raw = pd.concat([L.amortize_monthly_schedule(o, date(2026, 1, 1), date(2026, 12, 31))
                         for o in objs], ignore_index=True)
        chk("interest matches the raw amortization to the cent",
            abs(float(raw["interest"].sum()) - sch["interest"]) < 0.01,
            f"{float(raw['interest'].sum()):,.2f} vs {sch['interest']:,.2f}")
        # Principal may differ ONLY by an excluded balloon, and by exactly that.
        chk("principal differs from raw by exactly the excluded balloon",
            abs((float(raw["principal"].sum()) - sch["principal"])
                - sch["balloon_excluded"]) < 0.01,
            f"raw {float(raw['principal'].sum()):,.2f} - modeled {sch['principal']:,.2f} "
            f"vs balloon {sch['balloon_excluded']:,.2f}")

        print("\n4. Unmodelable is reported as unavailable, never as zero")
        gone = vds.for_year("P9999999", 2026, data)
        chk("a deal with no loans is unavailable", gone["available"] is False)
        chk("interest is None, not 0", gone["interest"] is None, f"{gone['interest']}")
        chk("and it says why", bool(gone["notes"]) and "no loans" in gone["notes"][0].lower(),
            f"{gone['notes']}")
        old = vds.for_year(vcode, 1990, data)
        chk("a year outside every loan's life is unavailable, with a reason",
            old["available"] is False and bool(old["notes"]), f"{old}")

        print("\n5. The comparison substitutes Budget and Valuation, and says so")
        with engine.connect() as conn:
            rec = conn.execute(text("""
                SELECT r.id, r.vcode FROM valuation_records r
                JOIN valuation_cycles c ON c.id = r.cycle_id
                ORDER BY r.id
            """)).fetchall()
        if not rec:
            print("  (no valuation_records locally — skipping the integration checks)")
            print(f"\nPASS={PASS} FAIL={FAIL}")
            return 1 if FAIL else 0

        # Prefer a record whose deal actually has loans, so the substitution is exercised.
        with_loans = {c.strip().lower() for c in ml[col].astype(str)}
        pick = next((r for r in rec if str(r[1]).strip().lower() in with_loans), rec[0])
        rid, rvcode = int(pick[0]), str(pick[1])
        br = vs.get_budget_review(engine, rid, data)
        dsinfo = br["debt_service"]
        print(f"        record {rid} = {rvcode}, source={dsinfo['source']}, "
              f"loans={dsinfo['loan_count']}")

        rows = {r["account"]: r for r in br["rows"]}
        chk("the payload declares where the debt rows came from",
            dsinfo["source"] in ("modeled", "file"))
        # The Valuation column is levered ONLY when there is an Argus forecast to lever.
        # Modeled debt against a zero NOI turned a blank DSCR into a hard 0.00, which
        # reads as "cannot cover its debt" when it means "no appraiser forecast loaded".
        expected_cols = (["budget", "valuation"] if br["has_argus"] else ["budget"]) \
            if dsinfo["source"] == "modeled" else []
        chk("it touches Budget, and Valuation only when an Argus forecast exists",
            dsinfo["applies_to"] == expected_cols,
            f"applies_to={dsinfo['applies_to']} has_argus={br['has_argus']}")
        chk("and keeps what the file itself said, so nothing is substituted silently",
            set(dsinfo["as_stated_in_source"]) == {
                "interest_budget", "principal_budget",
                "interest_valuation", "principal_valuation"})

        if dsinfo["source"] == "modeled":
            exp = vds.for_year(rvcode, br["budget_year"], data)
            chk("Budget interest is the modeled figure",
                abs(rows["Interest Expense"]["budget"] - exp["interest"]) < 0.01,
                f"{rows['Interest Expense']['budget']} vs {exp['interest']}")
            dscr = next(r for r in br["rows"] if r["account"] == "DSCR")
            if br["has_argus"]:
                chk("Valuation interest is the SAME figure — same loans, same year",
                    abs(rows["Interest Expense"]["valuation"]
                        - rows["Interest Expense"]["budget"]) < 0.01,
                    f"{rows['Interest Expense']}")
                chk("Valuation principal likewise",
                    abs(rows["Principal Payments"]["valuation"]
                        - rows["Principal Payments"]["budget"]) < 0.01,
                    f"{rows['Principal Payments']}")
                chk("the Valuation DSCR is now a real ratio, which was the point",
                    dscr["valuation"] is not None and dscr["valuation"] > 0, f"{dscr}")
            else:
                chk("with no Argus import the Valuation debt rows stay at the file's 0",
                    rows["Interest Expense"]["valuation"] == 0
                    and rows["Principal Payments"]["valuation"] == 0,
                    f"{rows['Interest Expense']} / {rows['Principal Payments']}")
                chk("and the Valuation DSCR stays BLANK rather than becoming 0.00",
                    dscr["valuation"] is None, f"{dscr}")
            # Estimate means actuals-plus-budget-remainder, and its interest was actually
            # paid. If it had been swapped for the model it would now equal the modeled
            # figure exactly — which is the one thing it must not do.
            est_i = rows["Interest Expense"]["estimate"]
            chk("the ESTIMATE column is untouched — it is actuals, not a model",
                est_i is not None and abs(est_i - exp["interest"]) > 0.01,
                f"estimate {est_i} vs modeled {exp['interest']} — identical means it "
                f"was substituted")
            chk("Total Debt Service ties to its two components",
                abs(rows["Total Debt Service"]["budget"]
                    - (rows["Interest Expense"]["budget"]
                       + rows["Principal Payments"]["budget"])) < 0.01)
            print(f"        DSCR  est {_f(dscr['estimate'])}  "
                  f"bud {_f(dscr['budget'])}  val {_f(dscr['valuation'])}")
        else:
            chk("a record whose deal cannot be modeled leaves the file's figures alone",
                rows["Interest Expense"]["budget"]
                == dsinfo["as_stated_in_source"]["interest_budget"])

        # ── 6 ──────────────────────────────────────────────────────────────────
        # The branch that IS the point of this work: an UNLEVERED appraiser forecast.
        # There are no Argus imports in the local database, so without this the
        # Valuation half was never executed and could not be claimed to work. A
        # synthetic unlevered forecast is injected, the record is pointed at it, and
        # both are put back in a finally.
        print("\n6. An UNLEVERED appraiser forecast gets levered")
        import pandas as pd
        from flask_app.services import argus_service

        by = br["budget_year"]
        fake_fc = pd.DataFrame([
            # Revenue and expenses only — exactly what Argus exports: no 5190, no 7060.
            {"vcode": rvcode, "event_date": date(by, mth, 28),
             "vAccount": acct, "mAmount_norm": amt}
            for mth in range(1, 13)
            for acct, amt in (("4010", 100000.0), ("5090", -20000.0), ("5110", -5000.0))
        ])
        chk("the synthetic forecast is unlevered, like a real Argus export",
            not fake_fc["vAccount"].isin(["5190", "7060", "7030"]).any())

        original = argus_service.get_forecast_df_by_id
        argus_service.get_forecast_df_by_id = lambda *a, **k: fake_fc
        try:
            with engine.begin() as conn:
                conn.execute(text("UPDATE valuation_records SET argus_import_id = 999999 "
                                  "WHERE id = :i"), {"i": rid})
            lv = vs.get_budget_review(engine, rid, data)
        finally:
            argus_service.get_forecast_df_by_id = original
            with engine.begin() as conn:
                conn.execute(text("UPDATE valuation_records SET argus_import_id = NULL "
                                  "WHERE id = :i"), {"i": rid})

        lrows = {r["account"]: r for r in lv["rows"]}
        ldsi = lv["debt_service"]
        exp = vds.for_year(rvcode, by, data)
        chk("the comparison sees an Argus forecast", lv["has_argus"] is True)
        chk("and NOW levers the Valuation column too",
            ldsi["applies_to"] == ["budget", "valuation"], f"{ldsi['applies_to']}")
        chk("Valuation interest is the modeled figure, not the file's 0",
            abs(lrows["Interest Expense"]["valuation"] - exp["interest"]) < 0.01,
            f"{lrows['Interest Expense']['valuation']} vs {exp['interest']}")
        chk("Valuation principal likewise",
            abs(lrows["Principal Payments"]["valuation"] - exp["principal"]) < 0.01,
            f"{lrows['Principal Payments']['valuation']} vs {exp['principal']}")
        chk("Budget and Valuation debt service agree — same loans, same year",
            abs(lrows["Total Debt Service"]["valuation"]
                - lrows["Total Debt Service"]["budget"]) < 0.01,
            f"{lrows['Total Debt Service']}")
        chk("the file said 0 for both, and that is still recorded",
            ldsi["as_stated_in_source"]["interest_valuation"] == 0
            and ldsi["as_stated_in_source"]["principal_valuation"] == 0,
            f"{ldsi['as_stated_in_source']}")
        ldscr = next(r for r in lv["rows"] if r["account"] == "DSCR")
        chk("the Valuation DSCR is a real ratio — the blank column was the complaint",
            ldscr["valuation"] is not None and ldscr["valuation"] > 0, f"{ldscr}")
        noi_v = next(r for r in lv["rows"]
                     if r["account"] == "Net Operating Income")["valuation"]
        chk("and it is NOI over the modeled debt service, to the cent",
            abs(ldscr["valuation"]
                - noi_v / lrows["Total Debt Service"]["valuation"]) < 1e-9,
            f"{ldscr['valuation']} vs {noi_v} / "
            f"{lrows['Total Debt Service']['valuation']}")
        print(f"        NOI {noi_v:,.0f} / DS "
              f"{lrows['Total Debt Service']['valuation']:,.0f} = "
              f"DSCR {ldscr['valuation']:.2f}")

        with engine.connect() as conn:
            left = conn.execute(text("SELECT argus_import_id FROM valuation_records "
                                     "WHERE id = :i"), {"i": rid}).scalar()
        chk("the record was put back the way it was found", left is None, f"{left}")

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


def _f(v):
    return "—" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    sys.exit(main())
