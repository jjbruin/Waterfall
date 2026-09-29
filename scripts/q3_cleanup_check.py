"""Guardrail for the Q3 cleanup phase 1 — three fixes, asserted both ways.

  A. Return of capital counts ONLY `Capital='Y'` rows, so a Realized Gain
     stops inflating it (East Manchester, open item 14).
  B. The paid-off filter drops a LOAN, not a ROW, and can be asked as at a
     date (open item 1).
  C. A debt-free deal carries no raw debt figure, so the subtotal and the
     printed cell cannot disagree (open item 3).

EVERY NARROWING IS ASSERTED IN BOTH DIRECTIONS. A check written only in the
excluding direction is satisfied by excluding everything — so each section also
pins what must SURVIVE. Run with ``--inject`` to restore each old behaviour and
confirm the checks actually fail; a guardrail that passes against the defect it
describes is worse than none.

    python scripts/q3_cleanup_check.py
    python scripts/q3_cleanup_check.py --inject
"""
from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import one_pager as OP                                       # noqa: E402
from flask_app.services import data_service as DS            # noqa: E402
from flask_app.services import portfolio_snapshot_loan as PL  # noqa: E402

PASS = FAIL = 0
FAILURES: list[str] = []


def chk(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok]   {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))


# ── Fixtures ──────────────────────────────────────────────────────────────

#: East Manchester's real 2026-06-25 rows, as production carries them. The two
#: Realized Gain rows are `Capital='N'`; the two Return of Capital rows are 'Y'.
EASTMA_ROWS = [
    ("PPI20",  "Distribution", "Distribution: Return of Capital", "Y", 3600000.00),
    ("PPI20",  "Distribution", "Distribution: Realized Gain",     "N", 1539662.37),
    ("OPVAST", "Distribution", "Distribution: Return of Capital", "Y", 2400000.00),
    ("OPVAST", "Distribution", "Distribution: Realized Gain",     "N", 2037020.76),
]


def eastma_acct() -> pd.DataFrame:
    rows = [{
        "InvestmentID": "EASTMA", "InvestorID": inv,
        "EffectiveDate": pd.Timestamp("2026-06-25"),
        "MajorType": major, "Typename": tname, "TypeName": tname,
        "Capital": cap, "is_capital": cap == "Y", "Amt": amt,
    } for inv, major, tname, cap, amt in EASTMA_ROWS]
    # A contribution so funded_to_date is real and the balance is checkable.
    rows.append({
        "InvestmentID": "EASTMA", "InvestorID": "PPI20",
        "EffectiveDate": pd.Timestamp("2019-01-01"),
        "MajorType": "Contribution", "Typename": "Contribution: Capital",
        "TypeName": "Contribution: Capital", "Capital": "Y",
        "is_capital": True, "Amt": -3600000.00,
    })
    return pd.DataFrame(rows)


def loans_fixture() -> pd.DataFrame:
    """The four live 'Paid Off' loans plus one that was never repaid."""
    r = [
        # East Manchester 257 — repaid 2026-06-25, but ALSO carries Origination
        # and Maturity rows. This is the loan the row-wise filter left alive.
        ("P0000017", 257, "Origination", "2025-10-17"),
        ("P0000017", 257, "Paid Off",    "2026-06-25"),
        ("P0000017", 257, "Maturity",    "2031-01-11"),
        # Nottingham 264 — repaid 2026-06-01, single row.
        ("P0000030", 264, "Paid Off",    "2026-06-01"),
        # Nottingham 337 — the replacement facility, never repaid.
        ("P0000030", 337, "Maturity",    "2031-06-01"),
        # Ascent 288 / 289 — repaid 2026-07-01, i.e. AFTER 26Q2 quarter end.
        ("P0000065", 288, "Paid Off",    "2026-07-01"),
        ("P0000065", 289, "Paid Off",    "2026-07-01"),
        ("P0000065", 328, "Maturity",    "2031-07-01"),
        # A loan with an UNDATED repayment — must drop whatever the as-of is.
        ("P0000099", 319, "Paid Off",    None),
    ]
    return pd.DataFrame([
        {"vCode": vc, "LoanID": lid, "vDateType": dt,
         "dtEvent": pd.Timestamp(ev) if ev else pd.NaT,
         "mOrigLoanAmt": 1000.0}
        for vc, lid, dt, ev in r
    ])


def _ids(df: pd.DataFrame) -> set:
    return set(zip(df["vCode"], df["LoanID"]))


# ── A. Return of capital ──────────────────────────────────────────────────

def section_a() -> None:
    print("\nA. Return of capital counts only Capital='Y'")

    # A1 — the predicate itself, every route to an answer.
    chk("is_capital=True is a return of capital",
        OP._is_return_of_capital({"is_capital": True}) is True)
    chk("is_capital=False is NOT",
        OP._is_return_of_capital({"is_capital": False}) is False)
    chk("Capital='Y' is read when is_capital is absent",
        OP._is_return_of_capital({"Capital": "y"}) is True)
    chk("Capital='N' is read when is_capital is absent",
        OP._is_return_of_capital({"Capital": "N"}) is False)
    chk("a row carrying NEITHER column is NOT a return of capital "
        "(never falls back to the Typename rule)",
        OP._is_return_of_capital({"Typename": "Distribution: Return of Capital"})
        is False)

    # A2 — the shipping path, on East Manchester's own rows.
    acct = eastma_acct()
    inv_map = pd.DataFrame([{"vcode": "P0000017", "InvestmentID": "EASTMA"}])
    pe = OP.get_pe_performance(
        "P0000017", "2026-Q2", acct, pd.DataFrame(), inv_map)
    roc = round(float(pe.get("return_of_capital") or 0.0), 2)

    chk("East Manchester PPI20 return of capital is 3,600,000.00",
        roc == 3600000.00, f"got {roc:,.2f}")
    chk("the 1,539,662.37 Realized Gain is NOT in it "
        "(the old figure was 5,139,662.37)",
        abs(roc - 5139662.37) > 0.005, f"got {roc:,.2f}")

    # A3 — BOTH DIRECTIONS. Excluding the gain must not also exclude the
    # capital: a rule that zeroed the field would satisfy A2's second check.
    chk("the Return of Capital row SURVIVES — the figure is not merely zeroed",
        roc > 0, f"got {roc:,.2f}")

    # A4 — the two definitions of capital outstanding now agree, which is why
    # financials_service could stop working around this one.
    funded = round(float(pe.get("funded_to_date") or 0.0), 2)
    chk("funded_to_date - return_of_capital is 0.00, matching the engine "
        "(it was -1,539,662.37)",
        round(funded - roc, 2) == 0.00, f"{funded:,.2f} - {roc:,.2f}")


# ── B. Paid-off filter ────────────────────────────────────────────────────

def section_b() -> None:
    print("\nB. The paid-off filter drops a loan, not a row")

    # SKIPPED ON THIS BRANCH, DELIBERATELY. The paid-off fix (30a1cec) is NOT
    # part of this release — it stays on fix/q3-cleanup-phase1 — so the
    # behaviour these checks pin does not exist here yet. Skipped rather than
    # deleted: the coverage returns by itself the day that commit ships, and a
    # deleted section would have to be remembered and rewritten instead.
    #
    # Detected from the SIGNATURE, not from a branch name or a flag: the fix is
    # what adds the `as_of` parameter, so its absence IS the absence of the fix.
    # That means this cannot skip by accident once the fix is present.
    import inspect
    if "as_of" not in inspect.signature(DS._filter_paid_off_loans).parameters:
        print("  SKIP  the paid-off fix (30a1cec) is not in this release — "
              "_filter_paid_off_loans takes no as_of parameter")
        return

    df = loans_fixture()

    # B1 — the defect: every row of 257 must go, not just its 'Paid Off' row.
    out = DS._filter_paid_off_loans(df, as_of=date(2026, 6, 30))
    left = _ids(out)
    chk("26Q2: East Manchester 257 is gone ENTIRELY (all 3 rows)",
        ("P0000017", 257) not in left
        and out[(out["vCode"] == "P0000017")].empty)
    chk("26Q2: Nottingham 264 (repaid 2026-06-01) is gone",
        ("P0000030", 264) not in left)
    chk("26Q2: Ascent 288/289 (repaid 2026-07-01) are KEPT — "
        "the repayment is after quarter end",
        ("P0000065", 288) in left and ("P0000065", 289) in left)

    # B2 — BOTH DIRECTIONS. Loans that were never repaid must survive; a filter
    # that dropped everything would pass every check above.
    chk("26Q2: Nottingham 337 survives (never repaid)",
        ("P0000030", 337) in left)
    chk("26Q2: Ascent 328 survives (never repaid)",
        ("P0000065", 328) in left)
    chk("26Q2: 257's three rows are the only P0000017 rows removed",
        len(out) == len(df) - 3 - 1 - 1,
        f"kept {len(out)} of {len(df)}")

    # B3 — the date genuinely moves the answer.
    q1 = _ids(DS._filter_paid_off_loans(df, as_of=date(2026, 3, 31)))
    chk("26Q1: Nottingham 264 is KEPT (repaid 2026-06-01, after Q1)",
        ("P0000030", 264) in q1)
    chk("26Q1: East Manchester 257 is KEPT (repaid 2026-06-25, after Q1)",
        ("P0000017", 257) in q1)
    q3 = _ids(DS._filter_paid_off_loans(df, as_of=date(2026, 9, 30)))
    chk("26Q3: Ascent 288/289 are gone (repaid 2026-07-01)",
        ("P0000065", 288) not in q3 and ("P0000065", 289) not in q3)
    chk("26Q3: East Manchester 257 and Nottingham 264 are gone",
        ("P0000017", 257) not in q3 and ("P0000030", 264) not in q3)

    # B4 — the conservative default used by load_all.
    none_ = _ids(DS._filter_paid_off_loans(df, as_of=None))
    chk("as_of=None drops every repaid loan, whatever its date",
        all(k not in none_ for k in
            (("P0000017", 257), ("P0000030", 264),
             ("P0000065", 288), ("P0000065", 289))))
    chk("as_of=None still keeps the loans that were never repaid",
        ("P0000030", 337) in none_ and ("P0000065", 328) in none_)

    # B5 — an undated repayment is acted on at every as-of.
    chk("an UNDATED 'Paid Off' drops the loan at every as-of",
        all(("P0000099", 319) not in s for s in (left, q1, q3, none_)))

    # B6 — a frame with no loan identity cannot group; it must still behave.
    noid = df.drop(columns=["LoanID"])
    out2 = DS._filter_paid_off_loans(noid, as_of=date(2026, 6, 30))
    chk("without LoanID it falls back to the row-wise drop rather than raising",
        len(out2) == len(noid) - 5
        and not (out2["vDateType"].str.lower() == "paid off").any())

    # B7 — a frame with no repayments at all is returned untouched.
    clean = df[df["vDateType"] != "Paid Off"].reset_index(drop=True)
    chk("a frame with no 'Paid Off' row is returned unchanged",
        len(DS._filter_paid_off_loans(clean, as_of=date(2026, 6, 30)))
        == len(clean))


# ── C. Debt-free deal carries no raw figure ───────────────────────────────

def _rows(raw_balance):
    """Two ordinary rows plus Pegasus, built through the SHIPPING rule.

    ``raw_balance`` is what ``resolve_debt`` handed the row builder; what the
    row actually carries is whatever ``debt_field`` decides, which is the thing
    under test.
    """
    return [
        {"vcode": "P0000018", "debt": PL.debt_field(45394000.0, False),
         "is_dev": False},
        {"vcode": "P0000019", "debt": PL.debt_field(95105179.0, False),
         "is_dev": False},
        {"vcode": "P0000066", "debt": PL.debt_field(raw_balance, True),
         "is_dev": False, "debt_free": True,
         "debt_display": PL.debt_field(raw_balance, True)},
    ]


def section_c() -> None:
    print("\nC. A debt-free deal carries no raw debt figure")
    base = 45394000.0 + 95105179.0

    chk("Pegasus is the only deal on the debt-free list",
        PL.DEBT_FREE_DEALS == {"P0000066"}, str(PL.DEBT_FREE_DEALS))
    chk("_debt_free identifies it, case- and space-insensitively",
        PL._debt_free(" p0000066 ") and not PL._debt_free("P0000018"))

    # C1 — the rule itself: the debt-free gate decides, not the balance.
    chk("debt_field returns None for a debt-free deal, whatever the balance",
        PL.debt_field(0.0, True) is None
        and PL.debt_field(25200000.0, True) is None)
    chk("debt_field passes a normal deal's figure through untouched",
        PL.debt_field(45394000.0, False) == 45394000.0
        and PL.debt_field(0.0, False) == 0.0)

    # C2 — NO FIGURE MOVES TODAY. Pegasus's balance is 0.0, so the row's total
    # is the same as it was. This is the check that says the fix is safe to
    # ship against the sent quarter.
    chk("today the total is unchanged — no printed figure moves",
        PL.loan_subtotal(_rows(0.0), "t")["debt"] == base,
        f"got {PL.loan_subtotal(_rows(0.0), 't')['debt']!r} vs {base!r}")

    # C3 — the latent path. Give Pegasus its stale 2024-09-30 balance: under
    # the old rule that 25.2M reached the total with no row accounting for it.
    stale = 25200000.0
    new_stale = PL.loan_subtotal(_rows(stale), "t")["debt"]
    chk("a stale 25,200,000 cannot leak into the total",
        new_stale == base, f"got {new_stale!r}")
    chk("and the cell beside it is blank, so the two agree by construction",
        _rows(stale)[2]["debt"] is None
        and _rows(stale)[2]["debt_display"] is None)

    # C4 — BOTH DIRECTIONS. Excluding the debt-free row must not exclude the
    # others; a subtotal of None would satisfy C3 on its own.
    chk("the other deals still reach the total in full",
        new_stale == base and base > 0)
    chk("deal_count still counts the debt-free deal — it is on the report",
        PL.loan_subtotal(_rows(None), "t")["deal_count"] == 3)


def main() -> int:
    inject = "--inject" in sys.argv
    if inject:
        print("=== INJECTION MODE: restoring the three old behaviours ===")
        print("    Every section below MUST fail.\n")

        # A: return of capital counts the Typename, not the flag.
        OP._is_return_of_capital = lambda row: (
            "return of capital" in str(row.get("TypeName", "")).lower()
            or "realized gain" in str(row.get("TypeName", "")).lower())

        # B: the row-wise filter.
        def _old(df, as_of=None):
            if df is None or df.empty:
                return df
            col = next((c for c in df.columns
                        if c.lower() == "vdatetype"), None)
            if col:
                df = df[df[col].astype(str).str.strip().str.lower()
                        != "paid off"].reset_index(drop=True)
            return df
        DS._filter_paid_off_loans = _old

        # C: the raw field kept the balance and only the DISPLAY was blanked.
        PL.debt_field = lambda debt, debt_free: debt

    for fn in (section_a, section_b, section_c):
        try:
            fn()
        except Exception as exc:                              # noqa: BLE001
            chk(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        for f in FAILURES:
            print(f"    - {f}")
    if inject:
        ok = FAIL > 0
        print("\nINJECTION RESULT:",
              "the guardrail catches the old behaviour" if ok
              else "*** VACUOUS — it passes against the defect ***")
        return 0 if ok else 1
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
