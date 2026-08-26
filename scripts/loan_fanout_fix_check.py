"""Guardrail for the Loan_Date fan-out fix — runs the app's REAL functions.

Drives, on identical input, the genuine committed code on both sides of
``data_service._collapse_loan_date_events``:

  * ``data_service._filter_paid_off_loans``   (real)  -> the BEFORE frame
  * ``data_service._collapse_loan_date_events`` (real) -> the AFTER frame
  * ``loaders.load_mri_loans``                (real)
  * ``loans.build_loans_from_mri_loans``      (real)  -> Debt Service, Surveillance
                                                         maturity, compute.py:791
  * ``loans.amortize_monthly_schedule``       (real)  -> interest / debt service
  * ``one_pager.get_capitalization_stack``    (real)  -> 1st/2nd Loan Terms + debt
  * ``dashboard_service.get_loan_maturity_data`` (real) -> maturities chart

Nothing is re-implemented; a replica proving itself is worth nothing.

Input is a cached snapshot of the live ``loans`` table (89 rows) so both sides see
byte-identical data.  Refresh it with ``--pull`` and a valid ``WF_TOKEN``.

    python scripts/loan_fanout_fix_check.py --pull     # refresh the snapshot
    python scripts/loan_fanout_fix_check.py            # run the checks

Exit code 0 = every assertion passed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from flask_app.services.data_service import (
    _collapse_loan_date_events, _filter_paid_off_loans)
from flask_app.services.dashboard_service import get_loan_maturity_data
from loaders import load_mri_loans
from loans import amortize_monthly_schedule, build_loans_from_mri_loans
from one_pager import get_capitalization_stack

SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "fixtures", "live_loans_snapshot.json")

# The 4 deals the fan-out reaches, and what each must look like afterwards.
EXPECT = {
    "P0000114": ("Jefferson Stephens", 1, 50_000_000, "2029-10-16"),
    "P0000017": ("East Manchester", 1, 10_000_000, "2035-10-17"),
    "P0000116": ("Plaza Del Mar", 1, 35_000_000, "2029-03-16"),
    "P0000119": ("Presidential Arms", 1, 49_490_000, "2036-05-13"),
}

results = []


def chk(label, ok, detail=""):
    results.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))


def pull():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import live_api as api
    d = api.get("/api/data/tables/loans/rows",
                params={"page": 1, "page_size": 500})
    if (d.get("total") or 0) > 500:
        raise SystemExit("loans table exceeds one page — narrow the pull")
    os.makedirs(os.path.dirname(SNAP), exist_ok=True)
    with open(SNAP, "w") as f:
        json.dump({"build": api.get("/api/data/version"),
                   "rows": d["rows"]}, f, indent=1)
    print(f"wrote {len(d['rows'])} rows to {SNAP}")


def loan_objects(df, vcode):
    """The real modelling path, exactly as compute.py:785-787 runs it."""
    ml = load_mri_loans(df.copy())
    ml = ml[ml["vCode"].astype(str) == str(vcode)]
    return build_loans_from_mri_loans(ml)


def cap_stack(df, vcode):
    """The real One Pager cap stack.

    isbs_raw=None forces the mOrigLoanAmt debt fallback for every deal, which is
    the path that doubles debt — so this exercises it rather than hiding behind
    ISBS balance-sheet debt.  Empty acct/waterfalls leave the equity legs at 0;
    only the loan legs are under test here.
    """
    empty = pd.DataFrame()
    return get_capitalization_stack(vcode, df.copy(), empty, empty, empty, empty,
                                   isbs_raw=None, quarter_str="2026-Q2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    args = ap.parse_args()
    if args.pull:
        pull()
        return 0

    if not os.path.exists(SNAP):
        raise SystemExit(f"no snapshot at {SNAP} — run with --pull first")
    snap = json.load(open(SNAP))
    raw = pd.DataFrame(snap["rows"])
    print(f"snapshot from live build {snap.get('build')}, {len(raw)} rows\n")

    # ---- the two frames, from the real functions -------------------------
    before = _filter_paid_off_loans(raw.copy())
    after = _collapse_loan_date_events(_filter_paid_off_loans(raw.copy()))

    print("=" * 78)
    print("A — frame shape")
    print("=" * 78)
    print(f"  raw rows {len(raw)} -> paid-off filtered {len(before)} "
          f"-> collapsed {len(after)}")
    chk("collapse removed exactly the 3 duplicate date-event rows",
        len(before) - len(after) == 3, f"{len(before)} -> {len(after)}")
    ids_before = set(zip(before["vCode"].astype(str).str.upper(), before["LoanID"]))
    ids_after = set(zip(after["vCode"].astype(str).str.upper(), after["LoanID"]))
    chk("no loan facility was lost", ids_before == ids_after,
        f"{len(ids_before)} facilities both sides")
    chk("one row per facility after collapse", len(after) == len(ids_after))

    # ---- per-deal, through the real functions ---------------------------
    print("\n" + "=" * 78)
    print("B — the 4 affected deals, via real load_mri_loans + build_loans_from_mri_loans")
    print("=" * 78)
    for vc, (name, n_exp, debt_exp, mat_exp) in EXPECT.items():
        objs_b, objs_a = loan_objects(before, vc), loan_objects(after, vc)
        cap_b, cap_a = cap_stack(before, vc), cap_stack(after, vc)
        mats_a = [o.maturity_date for o in objs_a if o.maturity_date]
        max_a = max(mats_a) if mats_a else None

        print(f"\n  {vc}  {name}")
        print(f"    BEFORE  loans={len(objs_b)}  debt={cap_b['debt']:>13,.0f}  "
              f"1st={cap_b['loan_terms_str']!r}")
        print(f"            2nd={cap_b['second_loan_terms_str']!r}")
        print(f"    AFTER   loans={len(objs_a)}  debt={cap_a['debt']:>13,.0f}  "
              f"1st={cap_a['loan_terms_str']!r}")
        print(f"            2nd={cap_a['second_loan_terms_str']!r}")
        print(f"            maturity={max_a}  (sale-date default)")

        chk(f"{vc} models exactly {n_exp} loan object", len(objs_a) == n_exp,
            f"was {len(objs_b)}")
        chk(f"{vc} debt = ${debt_exp:,}", abs(cap_a["debt"] - debt_exp) < 1,
            f"was ${cap_b['debt']:,.0f}")
        chk(f"{vc} maturity = {mat_exp}", str(max_a) == mat_exp)
        # get_capitalization_stack seeds second_loan_terms_str to 'N/A' and only
        # overwrites it when a genuine iloc[1] exists, so 'N/A' == no 2nd loan.
        chk(f"{vc} renders no 2nd loan",
            (cap_a["second_loan_terms_str"] or "N/A") == "N/A",
            repr(cap_a["second_loan_terms_str"]))
        chk(f"{vc} 1st Loan Terms carries the real maturity",
            mat_exp.split("-")[0] in (cap_a["loan_terms_str"] or ""))

    # East Manchester's sale-date default must leave the past
    objs = loan_objects(after, "P0000017")
    chk("P0000017 sale-date default is no longer in the past",
        objs and objs[0].maturity_date.year == 2035,
        f"maturity {objs[0].maturity_date}")

    # ---- unaffected deals must be untouched ------------------------------
    print("\n" + "=" * 78)
    print("C — the unaffected deals are byte-identical")
    print("=" * 78)
    touched = set(EXPECT)
    b_un = before[~before["vCode"].astype(str).str.upper().isin(touched)]
    a_un = after[~after["vCode"].astype(str).str.upper().isin(touched)]
    b_un = b_un.sort_values(["vCode", "LoanID"]).reset_index(drop=True)
    a_un = a_un.sort_values(["vCode", "LoanID"]).reset_index(drop=True)
    chk("unaffected row count unchanged", len(b_un) == len(a_un),
        f"{len(b_un)} rows")
    chk("unaffected frames identical (every column, every row)",
        b_un.equals(a_un[b_un.columns]))
    print(f"  {b_un['vCode'].nunique()} unaffected deals, {len(b_un)} rows, untouched")

    # every unaffected deal's Loan objects must match exactly
    same = True
    for vc in sorted(b_un["vCode"].astype(str).str.upper().unique()):
        ob, oa = loan_objects(before, vc), loan_objects(after, vc)
        if len(ob) != len(oa) or [
                (o.loan_id, o.orig_amount, o.maturity_date) for o in ob] != [
                (o.loan_id, o.orig_amount, o.maturity_date) for o in oa]:
            same = False
            print(f"    DIFFERS: {vc}")
    chk("every unaffected deal's Loan objects unchanged", same)

    # ---- interest / debt service must not move --------------------------
    print("\n" + "=" * 78)
    print("D — debt service: no month can stack two copies of one facility")
    print("=" * 78)
    print("  A phantom Loan object harms debt service two ways: it STACKS where its")
    print("  months collide with the real facility's (compute.py sums the schedule")
    print("  grouped by vcode/LoanID/event_date), and it ADDS months the real")
    print("  facility does not cover at all — e.g. P0000119's phantom, back-dated to")
    print("  2016 origination, billed Mar+Apr 2026 before the real loan closed.")
    print("  So the invariant is: after the fix every scheduled month belongs to the")
    print("  one real facility's life, and nothing stacks.\n")
    start, end = pd.Timestamp("2026-03-31").date(), pd.Timestamp("2029-02-28").date()
    for vc in EXPECT:
        def sched(df):
            s = [amortize_monthly_schedule(o, start, end) for o in loan_objects(df, vc)]
            s = [x for x in s if x is not None and not x.empty]
            if not s:
                return 0.0, set(), 0, 0
            allr = pd.concat(s)
            grouped = allr.groupby(["vcode", "LoanID", "event_date"], as_index=False).sum()
            return (float(grouped["interest"].sum()), set(allr["event_date"]),
                    len(allr), len(grouped))

        ib, mb, rb, gb = sched(before)
        ia, ma, ra, ga = sched(after)
        phantom_months = sorted(mb - ma)
        print(f"  {vc}: interest {ib:>14,.2f} -> {ia:>14,.2f}   "
              f"stacked before={rb - gb}  phantom months removed={len(phantom_months)}")
        if phantom_months:
            print(f"        {[str(m)[:10] for m in phantom_months]}")

        chk(f"{vc} AFTER has no stacked months", ra == ga)
        # Every scheduled month must fall inside the single real facility's life.
        objs = loan_objects(after, vc)
        chk(f"{vc} AFTER schedules only months the real facility covers",
            len(objs) == 1 and all(
                objs[0].orig_date <= pd.Timestamp(m).date() <= max(
                    objs[0].maturity_date, end) for m in ma))
        if ib == 0:
            chk(f"{vc} interest restored (facility was missing entirely)", ia > 0,
                f"0 -> {ia:,.2f}")
        elif phantom_months or (rb - gb):
            chk(f"{vc} interest corrected DOWN by the phantom months", ia < ib,
                f"removed {ib - ia:,.2f}")
        else:
            chk(f"{vc} interest unchanged (phantom fell outside the window)",
                abs(ib - ia) < 0.01)

    # ---- portfolio-level effects ----------------------------------------
    print("\n" + "=" * 78)
    print("E — portfolio debt and the maturities chart")
    print("=" * 78)
    amt_b = pd.to_numeric(before["mOrigLoanAmt"], errors="coerce").fillna(0).sum()
    amt_a = pd.to_numeric(after["mOrigLoanAmt"], errors="coerce").fillna(0).sum()
    print(f"  sum(mOrigLoanAmt) before {amt_b:,.0f} -> after {amt_a:,.0f}"
          f"   (removed {amt_b - amt_a:,.0f})")
    chk("phantom principal removed = 50M + 35M + 49.49M",
        abs((amt_b - amt_a) - 134_490_000) < 1)

    inv = pd.DataFrame({"vcode": sorted(set(after["vCode"].astype(str))),
                        "Investment_Name": "", "Portfolio_Name": None})
    mb = get_loan_maturity_data(before.copy(), inv, inv)
    ma = get_loan_maturity_data(after.copy(), inv, inv)

    def bucket(m):
        return {(r["year"], r["rate_type"]): r["amount"] for r in m["yearly"]}
    bb, ba = bucket(mb), bucket(ma)
    print("\n  maturity buckets that moved:")
    for k in sorted(set(bb) | set(ba)):
        if abs(bb.get(k, 0) - ba.get(k, 0)) > 0.5:
            print(f"    {k}: {bb.get(k, 0):>14,.0f} -> {ba.get(k, 0):>14,.0f}")
    # Every bucket movement must be exactly one of the four known corrections.
    # (This harness feeds get_loan_maturity_data every vcode, so the buckets are
    # larger than the live Dashboard's, which filters to active parent deals — the
    # $50M Jefferson Stephens phantom sat alone in the live 2025 Floating bucket.)
    expected_moves = {
        ("2025", "Floating"): -50_000_000,   # P0000114 phantom origination maturity
        ("2025", "Fixed"): -10_000_000,      # P0000017 misplaced out of 2025...
        ("2035", "Fixed"): +10_000_000,      # ...and into its real year
        ("2026", "Fixed"): -49_490_000,      # P0000119 phantom
        ("2026", "Floating"): -35_000_000,   # P0000116 phantom
    }
    actual_moves = {k: ba.get(k, 0) - bb.get(k, 0)
                    for k in set(bb) | set(ba) if abs(ba.get(k, 0) - bb.get(k, 0)) > 0.5}
    chk("every maturity-bucket movement is an expected correction",
        actual_moves == expected_moves,
        f"unexpected: { {k: v for k, v in actual_moves.items() if expected_moves.get(k) != v} }")
    chk("$50M Jefferson Stephens phantom left the 2025 Floating bucket",
        abs(actual_moves.get(("2025", "Floating"), 0) + 50_000_000) < 1)
    chk("chart total drops by the phantom principal",
        abs((sum(bb.values()) - sum(ba.values())) - 134_490_000) < 1)
    chk("no deal appears twice in the chart detail",
        len(ma["detail"]) == len(
            {(r["property"], r["loan_id"]) for r in ma["detail"]}),
        f"{len(ma['detail'])} detail rows")

    # ---- idempotence -----------------------------------------------------
    print("\n" + "=" * 78)
    print("F — collapsing twice changes nothing")
    print("=" * 78)
    twice = _collapse_loan_date_events(after.copy())
    chk("idempotent", twice.reset_index(drop=True).equals(after.reset_index(drop=True)))

    print("\n" + "=" * 78)
    failed = [l for l, ok in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for l in failed:
            print(f"  - {l}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
