#!/usr/bin/env python
"""Guardrail: ISBS supplements are protected, and the app's row beats MRI's.

TWO RULES, both learned the hard way.

1. PROTECT WHAT THE APP WRITES — AND ONLY THAT. The CSV import runs
   `to_sql(if_exists="replace")`, which DROPS the table, so one
   `ISBS_Budget_IS_Supplements.csv` upload would destroy every budget the team had
   imported and vetted, with no error, exactly as a single MRI_Capital_Calls.csv upload
   destroyed every app-entered capital call on Sep 10 2026. The budget supplement is the
   SOURCE OF RECORD for unapproved budgets — MRI does not receive one until it is
   analysed and approved — so between import and approval the app holds the only copy.

   THE OTHER FOUR ARE DELIBERATELY NOT PROTECTED, and this reversed on Sep 11 2026 after
   the first version protected all five. Ownership runs the other way for them: a CSV is
   their source of record and `replace` is their DESIGNED refresh. `isbs_uw_supplements`
   has no app write path at all, so protecting it did not make its 56 rows safe — it made
   them unchangeable, and they feed the One Pager's underwritten PE ROE (7073 capital
   events). Protection without a write path is a lockout, not a safeguard.

2. WHERE BOTH EXIST, THE APP'S ROW WINS. Supplements are appended, not merged, so once
   MRI loads an approved budget the same (vcode, dtEntry, vSource, vAccount) appears
   twice and every consumer double-counts. An NOI that silently doubles is worse than
   either version being wrong. The app's copy is kept because the app is where the work
   is done — a partner budget is imported, questioned against the appraiser's starting
   points, and re-imported until final, and MRI only sees it afterwards.

Run:  python scripts/isbs_supplement_precedence_check.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

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
    logging.disable(logging.INFO)

    import database
    from flask_app.services import data_service as ds

    # The app writes exactly one supplement table today (budget_import_validate.commit).
    # Protect that one; the rest are CSV-owned and must stay loadable.
    APP_WRITTEN = {"isbs_budget_is_supplements"}
    CSV_OWNED = set(ds._ISBS_SUPPLEMENTS) - APP_WRITTEN

    print("1. The table the APP writes is protected from CSV import")
    for t in sorted(APP_WRITTEN):
        chk(f"{t} is in PROTECTED_TABLES", t in database.PROTECTED_TABLES,
            "a CSV upload would DROP this table and destroy vetted budgets")
        # Signature is (table_name, df) — short-circuits before any DB work.
        res = database.import_csv_dataframe(t, pd.DataFrame([{"vcode": "X"}]))
        chk(f"import_csv_dataframe('{t}') returns protected",
            isinstance(res, dict) and res.get("status") == "protected", f"got {res}")

    print("\n2. The CSV-OWNED supplements stay loadable — protection without a write "
          "path is a lockout")
    for t in sorted(CSV_OWNED):
        chk(f"{t} is NOT protected", t not in database.PROTECTED_TABLES,
            "its CSV is its only load path; protecting it freezes the table forever")
    chk("in particular isbs_uw_supplements, which feeds One Pager PE ROE (7073)",
        "isbs_uw_supplements" not in database.PROTECTED_TABLES)
    chk("and no supplement table is protected unless the app writes it",
        {t for t in ds._ISBS_SUPPLEMENTS if t in database.PROTECTED_TABLES} == APP_WRITTEN,
        f"{sorted(t for t in ds._ISBS_SUPPLEMENTS if t in database.PROTECTED_TABLES)}")

    print("\n3. None of them is in QUERY_REGISTRY — an MRI refresh must not own them")
    from flask_app.services import mri_service
    targets = {v.get("target_table") for v in mri_service.QUERY_REGISTRY.values()}
    for t in sorted(ds._ISBS_SUPPLEMENTS):
        chk(f"{t} is not an MRI target", t not in targets)

    KEY = ["vcode", "dtEntry", "vSource", "vAccount"]

    # THIS CALLS THE SHIPPED FUNCTION. It used to be a local replica of the
    # shadowing rule, and the replica is why the case/date-drift defect in
    # section 7 survived every run of this file: the replica compared the key
    # with `.astype(str)`, the engine compared it with `.astype(str)`, the two
    # agreed perfectly, and both were wrong. A test that reimplements the thing
    # it is testing can only ever confirm that the author made the same mistake
    # twice.
    #
    # `_append_isbs_supplements` opens by loading each supplement table through
    # its adapter; with no database behind it every one of those raises and is
    # swallowed by the function's own `except: pass`, so nothing is appended and
    # the frame passed in — which already carries `_is_supplement` — goes
    # straight into the supersede block. That block is what these checks are
    # about, and it is now the real one.
    def shadow(df):
        return ds._append_isbs_supplements(
            df.copy(), {"db_path": "__no_such_db_for_guardrail__"}
        ).reset_index(drop=True)

    cols = ["vcode", "dtEntry", "vSource", "vAccount", "mAmount"]

    print("\n4. ISBS IS A JOURNAL — many rows per key must survive untouched")
    # This is the check that matters most. Writing the rule as
    # drop_duplicates(subset=KEY, keep='last') passes every other test here and is
    # catastrophically wrong: measured on the live snapshot it took isbs_raw from
    # 797,660 rows to 439,268, deleting 358,392 genuine MRI journal entries and roughly
    # halving every NOI. Four same-key MRI rows with NO supplement must all remain.
    journal = pd.DataFrame(
        [{**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "4010", 25.0])),
          "_is_supplement": False} for _ in range(4)])
    out = shadow(journal)
    chk("4 same-key MRI rows with no supplement ALL survive", len(out) == 4,
        f"got {len(out)} — this is the 358k-row deletion")
    chk("and they still sum to 100", out.mAmount.sum() == 100.0)

    print("\n5. A supplement replaces EVERY MRI row on its key, not just one")
    mixed = pd.concat([journal, pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-02-28", "Budget IS", "4010", 110.0])),
         "_is_supplement": False},
        {**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "4010", 999.0])),
         "_is_supplement": True},
    ])], ignore_index=True)
    out = shadow(mixed)
    jan = out[out.dtEntry == "2026-01-31"]
    chk("all 4 MRI Jan rows are gone", len(jan) == 1 and bool(jan.iloc[0]._is_supplement),
        f"got {len(jan)} Jan rows")
    chk("Jan totals the APP's 999, not MRI's 4x25", jan.mAmount.sum() == 999.0,
        f"got {jan.mAmount.sum()}")
    feb = out[out.dtEntry == "2026-02-28"]
    chk("an MRI row with no app counterpart is UNTOUCHED",
        len(feb) == 1 and float(feb.iloc[0].mAmount) == 110.0, f"got {feb.mAmount.tolist()}")

    print("\n6. A different account, source or property is NOT a match")
    other = pd.concat([journal, pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "5020", 50.0])), "_is_supplement": True},
        {**dict(zip(cols, ["p1", "2026-01-31", "Interim IS", "4010", 60.0])), "_is_supplement": True},
        {**dict(zip(cols, ["p2", "2026-01-31", "Budget IS", "4010", 70.0])), "_is_supplement": True},
    ])], ignore_index=True)
    out = shadow(other)
    chk("nothing is shadowed and every row survives", len(out) == len(other),
        f"{len(other)} in, {len(out)} out")

    print("\n7. THE KEY IS NORMALISED ON BOTH SIDES — case and date drift still match")
    # THE DEFECT THIS SECTION EXISTS FOR. `_append_isbs_supplements` runs INSIDE
    # `_assemble_isbs`, one line BEFORE `_normalize_isbs` lower-cases vcode and
    # parses dtEntry — so the supersede key saw MRI's raw spellings. MRI writes
    # 'p0000069' / '2026-01-31T00:00:00'; the app's budget import writes
    # 'P0000069' / '2026-01-31'. Four different strings, no key ever matched,
    # and BOTH rows reached isbs_raw where every consumer sums them.
    #
    # Measured on production: P0000069 Mount Prospect's 2026-Q2 budget NOI came
    # out at 2,685,375.22 against a true 1,342,687.61 — exactly double, which is
    # the hardest kind of wrong to see, because a doubled NOI is still a
    # plausible NOI.
    drift = pd.DataFrame([
        # MRI's spelling: lower-case vcode, timestamped date
        {**dict(zip(cols, ["p0000069", "2026-01-31T00:00:00", "Budget IS", "4010", 1_342_687.61])),
         "_is_supplement": False},
        # The app's spelling: the deal's own case, a bare date
        {**dict(zip(cols, ["P0000069", "2026-01-31", "Budget IS", "4010", 1_342_687.61])),
         "_is_supplement": True},
    ])
    out = shadow(drift)
    chk("case + date drift on the SAME fact is ONE key — the MRI row is dropped",
        len(out) == 1 and bool(out.iloc[0]._is_supplement),
        f"got {len(out)} rows — 2 means the pre-fix key, and the figure doubles")
    chk("...so the total is the budget, not twice the budget",
        abs(out.mAmount.sum() - 1_342_687.61) < 0.005,
        f"got {out.mAmount.sum():,.2f} against 1,342,687.61")

    # Each drift axis on its own, so a half-fix cannot pass.
    case_only = pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "4010", 10.0])), "_is_supplement": False},
        {**dict(zip(cols, ["P1", "2026-01-31", "Budget IS", "4010", 10.0])), "_is_supplement": True},
    ])
    chk("case drift alone is one key", len(shadow(case_only)) == 1,
        "vcode is not being lower-cased for the comparison")
    date_only = pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-01-31T00:00:00", "Budget IS", "4010", 10.0])), "_is_supplement": False},
        {**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "4010", 10.0])), "_is_supplement": True},
    ])
    chk("date drift alone is one key", len(shadow(date_only)) == 1,
        "dtEntry is not being parsed for the comparison")

    print("\n8. Normalising the key must not widen it, and must not touch the data")
    # The dangerous direction. Under-matching double-counts; OVER-matching
    # DELETES MRI rows. A different DAY, a different account and a different
    # source must still all miss, however the strings are spelled.
    wrong_day = pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-01-30T00:00:00", "Budget IS", "4010", 10.0])), "_is_supplement": False},
        {**dict(zip(cols, ["P1", "2026-01-31", "Budget IS", "4010", 99.0])), "_is_supplement": True},
    ])
    chk("a date one day apart is NOT the same key", len(shadow(wrong_day)) == 2,
        "normalisation has started matching different days")
    unparseable = pd.DataFrame([
        {**dict(zip(cols, ["p1", "not a date", "Budget IS", "4010", 10.0])), "_is_supplement": False},
        {**dict(zip(cols, ["p1", "also not a date", "Budget IS", "4010", 99.0])), "_is_supplement": True},
    ])
    chk("two DIFFERENT unparseable dates do not collapse onto one key",
        len(shadow(unparseable)) == 2,
        "every unparseable date became NaT, so one bad supplement row would "
        "shadow every bad MRI row — that deletes data rather than doubling it")
    same_unparseable = pd.DataFrame([
        {**dict(zip(cols, ["p1", "not a date", "Budget IS", "4010", 10.0])), "_is_supplement": False},
        {**dict(zip(cols, ["p1", "not a date", "Budget IS", "4010", 99.0])), "_is_supplement": True},
    ])
    chk("...but two IDENTICAL unparseable dates still match, as they did before",
        len(shadow(same_unparseable)) == 1)
    # The stored values are the loader's business, not the comparison's.
    src = pd.DataFrame([
        {**dict(zip(cols, ["P0000069", "2026-01-31T00:00:00", "Budget IS", "4010", 10.0])),
         "_is_supplement": False},
    ])
    out = shadow(src)
    chk("the comparison does NOT rewrite vcode or dtEntry — _normalize_isbs owns that",
        out.iloc[0].vcode == "P0000069" and out.iloc[0].dtEntry == "2026-01-31T00:00:00",
        f"got {out.iloc[0].vcode!r} / {out.iloc[0].dtEntry!r}")
    # And the journal rule from section 4 must survive the normalisation.
    drifted_journal = pd.DataFrame(
        [{**dict(zip(cols, ["p1", "2026-01-31T00:00:00", "Budget IS", "4010", 25.0])),
          "_is_supplement": False} for _ in range(4)])
    chk("4 same-key MRI rows STILL all survive with no supplement present",
        len(shadow(drifted_journal)) == 4,
        "the journal rule is the 358k-row deletion; normalisation must not reach it")

    print("\n9. A YTD-CUMULATIVE vSource is never superseded")
    # FOUND BY THE SECTION-7 FIX, NOT BEFORE IT. While the key never matched,
    # this could not happen; the moment it matched, three deals' underwritten
    # capital DOUBLED.
    #
    # `isbs_projected_is` account 7073 is a RUNNING TOTAL. Burton carries
    # 26,597,500 on 2025-06-30 and the same 26,597,500 every month to December —
    # one contribution, restated, not seven. `one_pager._get_uw_7073_signed`
    # reads the first month of a year as the periodic figure and every later
    # month as a difference from the one before, which is zero.
    #
    # The supplement row is PERIODIC: a different quantity on the same key.
    # Supersede it and MRI's 2025-06-30 row goes, which makes 2025-07-31 the
    # first month of the year — so its full cumulative is read as a SECOND
    # contribution, while the supplement still supplies the real one. Measured
    # on production: Burton -26,597,500 -> -53,195,000, Presidential Arms
    # -20,600,000 -> -41,200,000, Court of Deptford -8,751,184 -> -18,297,184,
    # all of which feed U/W ROE to Date and CoC Proj. Since Close.
    #
    # The genuine duplicate is ALREADY resolved downstream by the
    # (date, amount) dedupe at the end of `_get_uw_7073_signed`. There is
    # nothing for this layer to do and real harm in trying.
    chk("Interim IS and Projected IS are declared cumulative",
        ds._CUMULATIVE_VSOURCES == frozenset({"Interim IS", "Projected IS"}),
        f"got {ds._CUMULATIVE_VSOURCES}")
    cumulative = pd.DataFrame([
        {**dict(zip(cols, ["p0000109", "2025-06-30T00:00:00", "Projected IS", "7073", 26_597_500.0])),
         "_is_supplement": False},
        {**dict(zip(cols, ["p0000109", "2025-07-31T00:00:00", "Projected IS", "7073", 26_597_500.0])),
         "_is_supplement": False},
        {**dict(zip(cols, ["P0000109", "6/30/2025", "Projected IS", "7073", 26_597_500.0])),
         "_is_supplement": True},
    ])
    out = shadow(cumulative)
    chk("a Projected IS supplement does NOT drop the MRI row it matches",
        len(out) == 3 and (~out["_is_supplement"].astype(bool)).sum() == 2,
        f"got {len(out)} rows — dropping the 06-30 row makes 07-31 read as a "
        "second contribution and the capital doubles")
    # Both halves of the rule, so neither can be satisfied by doing nothing.
    periodic = pd.DataFrame([
        {**dict(zip(cols, ["p0000069", "2026-01-31T00:00:00", "Budget IS", "4010", 1_342_687.61])),
         "_is_supplement": False},
        {**dict(zip(cols, ["P0000069", "2026-01-31", "Budget IS", "4010", 1_342_687.61])),
         "_is_supplement": True},
    ])
    chk("...while a Budget IS supplement on the same drift still DOES",
        len(shadow(periodic)) == 1,
        "scoping by vSource must not switch the fix off for periodic sources")
    interim = pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-01-31T00:00:00", "Interim IS", "4010", 10.0])),
         "_is_supplement": False},
        {**dict(zip(cols, ["P1", "2026-01-31", "Interim IS", "4010", 10.0])),
         "_is_supplement": True},
    ])
    chk("Interim IS — also YTD cumulative — is likewise left alone",
        len(shadow(interim)) == 2)
    bs = pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-01-31T00:00:00", "Interim BS", "2150", 10.0])),
         "_is_supplement": False},
        {**dict(zip(cols, ["P1", "2026-01-31", "Interim BS", "2150", 10.0])),
         "_is_supplement": True},
    ])
    chk("Interim BS is a point-in-time balance, so it IS superseded",
        len(shadow(bs)) == 1)

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
