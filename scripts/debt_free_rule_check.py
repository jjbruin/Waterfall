"""The debt-free display rule is DERIVED, and fires on exactly one shape.

`DEBT_FREE_DEALS = {"P0000066"}` was a per-deal hardcode. It is gone; the N/A
display is now decided by the row's own data:

    not dev, not sold, ISBS basis, debt exactly 0.0, no active MRI loan

WHY THIS FILE EXISTS. The hardcode's own note said a derived rule was not
possible because the data could not separate "held unlevered" from "disposed" —
PCITWES City West has the IDENTICAL debt fingerprint (0 loans, ISBS 0.0, no
facility) and is a foreclosed deal. That is still true of the fingerprint, and
the whole safety of this change rests on the `sold` term telling them apart.
So every check below is asserted in BOTH directions: a rule tested only in the
firing direction is satisfied by firing on everything, which would put "N/A —
held with no debt" on a foreclosed deal's row.

Run with --inject to restore the two ways this can go wrong. Every check in the
matching section must then fail.

  --inject=off    the rule never fires   -> Pegasus loses its N/A
  --inject=nosold the `sold` term is dropped -> City West is swept in
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from flask_app.services import portfolio_snapshot_loan as L  # noqa: E402

_PASS = _FAIL = 0


def chk(label: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [ok]   {label}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))


QUARTER = "2026-Q2"

# vcode, name, strategy, sold, isbs_debt, loan_count, property_count
CASES = [
    ("UNLEV",   "Unlevered by design",  "Core",             False, 0.0,        0, 1),
    ("SOLDZER", "Disposed, zero debt",  "Core",             True,  0.0,        0, 1),
    # "Development" is the ONLY member of config.DEV_STRATEGIES — "new
    # construction" was removed when Pegasus was reclassified, and using it
    # here made this row non-dev and silently turned the dev case into a
    # second unlevered one. Caught by this check failing; kept as a note
    # because the fixture looked right.
    ("DEVZERO", "Development, zero",    "Development",      False, 0.0,        0, 1),
    ("LEVERED", "Ordinary levered",     "Core",             False, 25_200_000.0, 1, 1),
    ("ZEROLN",  "Zero balance, live loan", "Core",          False, 0.0,        1, 1),
    ("NOREAD",  "No ISBS reading",      "Core",             False, None,       0, 1),
    # The Town Fair Tire shape: a real ISBS 0.0, no loan of its own, but a
    # CHILD property whose facility is held at the parent.
    ("CHILD",   "Child property, zero", "Core",             False, 0.0,        0, 0),
    # Parent/child unknown — must behave like the child, never like the parent.
    ("NULLPC",  "Unknown parent status","Core",             False, 0.0,        0, None),
]


def _build():
    """Drive the real `assemble_loan` over every shape."""
    isbs = {c[0]: c[4] for c in CASES}
    loan_rows = []
    for vc, _n, _s, _sold, _d, n, _pc in CASES:
        for i in range(n):
            loan_rows.append({"vCode": vc, "LoanID": f"{vc}-{i}",
                              "mOrigLoanAmt": 10_000_000.0,
                              "nRate": 0.05, "vIntType": "Fixed"})
    loans = pd.DataFrame(loan_rows or [{"vCode": "", "LoanID": "",
                                        "mOrigLoanAmt": None, "nRate": None,
                                        "vIntType": None}])

    def provider(vcode, quarter):
        return {"cap_stack": {"debt_isbs": isbs.get(vcode)},
                "property_performance": {"dscr": {"ytd_actual": 1.4}}}

    resolved = {
        "investor_code": "TEST",
        "groups": {"G": [
            {"vcode": vc, "name": name, "investment_strategy": strat,
             "kept_despite_sold": sold, "property_count": pc}
            for vc, name, strat, sold, _d, _n, pc in CASES
        ]},
        "flagged": [],
    }
    out = L.assemble_loan(
        "TEST", QUARTER, resolved=resolved, one_pager_provider=provider,
        loans=loans, valuations=pd.DataFrame(),
        comment_loader=lambda i, q: {}, manual_loader=lambda i, q: {})
    return out, {r["vcode"]: r for r in out["groups"]["G"]}


def _print_table(rows: dict) -> None:
    hdr = (f"    {'vcode':<9}{'debt':>13} {'debt_disp':>11} "
           f"{'free':>6} {'ltv':>7} {'dscr':>7} {'dy':>7} "
           f"{'rate':>7} {'matur':>7}")
    print(hdr)
    for vc, _n, _s, _sold, _d, _c, _pc in CASES:
        r = rows.get(vc, {})

        def d(v):
            if v is None:
                return "—"
            return v if isinstance(v, str) else f"{v:,.4g}"
        print(f"    {vc:<9}{d(r.get('debt')):>13} "
              f"{d(r.get('debt_display')):>11} "
              f"{str(bool(r.get('debt_free'))):>6} "
              f"{d(r.get('ltv_display')):>7} {d(r.get('ytd_dscr_display')):>7} "
              f"{d(r.get('debt_yield_display')):>7} "
              f"{d(r.get('rate_display')):>7} "
              f"{d(r.get('maturity_display')):>7}")


def section_unit() -> None:
    print("\nA. the gate itself — each term, both directions")
    ISBS, COMM, UNAV = L.BASIS_ISBS, L.BASIS_COMMITTED, L.BASIS_UNAVAILABLE

    chk("fires on the unlevered fingerprint (ISBS, exactly 0, no loan, parent)",
        L._debt_free(0.0, ISBS, 0, False, False, 1))
    chk("NOT when the deal is SOLD — disposed is not unlevered (City West)",
        not L._debt_free(0.0, ISBS, 0, False, True, 1))
    chk("NOT when the deal is DEVELOPMENT — its ratios read 'Dev'",
        not L._debt_free(0.0, ISBS, 0, True, False, 1))
    chk("NOT on a committed-facility basis — that is a stand-in, not a reading",
        not L._debt_free(0.0, COMM, 0, False, False, 1))
    chk("NOT on an unavailable basis — that is 'no data', the bare dash",
        not L._debt_free(0.0, UNAV, 0, False, False, 1))
    chk("NOT when debt is None — no reading is not a measured zero",
        not L._debt_free(None, ISBS, 0, False, False, 1))
    chk("NOT when the balance is non-zero",
        not L._debt_free(25_200_000.0, ISBS, 0, False, False, 1))
    chk("NOT when an ACTIVE MRI loan exists, whatever the balance says",
        not L._debt_free(0.0, ISBS, 1, False, False, 1))
    # The Town Fair Tire case — six child properties the rule fired on before
    # this term existed, each a real ISBS 0.0 with no loan of its own.
    chk("NOT for a CHILD property (property_count == 0) — its debt is at the "
        "parent, so a zero of its own proves nothing",
        not L._debt_free(0.0, ISBS, 0, False, False, 0))
    chk("NOT when property_count is NULL — an unknown must never widen it",
        not L._debt_free(0.0, ISBS, 0, False, False, None))
    chk("NOT when property_count is non-numeric — refused, not raised",
        not L._debt_free(0.0, ISBS, 0, False, False, "n/a"))
    chk("a parent with several properties still fires",
        L._debt_free(0.0, ISBS, 0, False, False, 6))
    chk("an integer 0 and a -0.0 are still exactly zero",
        L._debt_free(0, ISBS, 0, False, False, 1)
        and L._debt_free(-0.0, ISBS, 0, False, False, 1))
    chk("a non-numeric balance is refused rather than raising",
        not L._debt_free("n/a", ISBS, 0, False, False, 1))
    chk("the per-deal list is GONE, not merely emptied",
        not hasattr(L, "DEBT_FREE_DEALS"))
    chk("the gate takes no vcode — a deal cannot be added by editing a list",
        "vcode" not in L._debt_free.__code__.co_varnames[
            :L._debt_free.__code__.co_argcount])


def section_rows(rows: dict) -> None:
    print("\nB. what each shape PRINTS")
    _print_table(rows)

    u = rows["UNLEV"]
    chk("the unlevered deal is flagged debt_free", u.get("debt_free") is True)
    chk("its Debt prints an em dash AND carries no raw figure",
        u.get("debt") is None and u.get("debt_display") is None)
    chk("its ISBS reading survives for the audit", u.get("isbs_debt") == 0.0)
    chk("it reads 'N/A' on all five debt columns",
        all(u.get(k) == L.NA_DISPLAY for k in
            ("ltv_display", "ytd_dscr_display", "debt_yield_display",
             "rate_display", "maturity_display")))
    chk("it never reads 'Dev' — it is operating, not pre-stabilisation",
        not any(u.get(k) == L.DEV_DISPLAY for k in
                ("ltv_display", "ytd_dscr_display", "debt_yield_display",
                 "rate_display", "maturity_display")))

    s = rows["SOLDZER"]
    chk("the SOLD deal is NOT debt_free", not s.get("debt_free"))
    chk("it keeps bare em dashes and gains no N/A literal",
        not any(s.get(k) == L.NA_DISPLAY for k in
                ("ltv_display", "ytd_dscr_display", "debt_yield_display",
                 "rate_display", "maturity_display")))
    chk("it is excluded from the totals by the SALE", s.get("sold_suppressed"))

    d = rows["DEVZERO"]
    chk("the DEV deal is NOT debt_free", not d.get("debt_free"))
    chk("its three ratio columns still read 'Dev'",
        all(d.get(k) == L.DEV_DISPLAY for k in
            ("ltv_display", "ytd_dscr_display", "debt_yield_display")))

    z = rows["ZEROLN"]
    chk("a zero balance with a LIVE loan is NOT debt_free",
        not z.get("debt_free"))
    chk("and it keeps its raw 0.0, so nothing is hidden from the audit",
        z.get("debt") == 0.0)

    n = rows["NOREAD"]
    chk("a deal with no ISBS reading is NOT debt_free", not n.get("debt_free"))
    chk("it prints a bare dash, never the N/A literal",
        n.get("debt_display") is None
        and n.get("ltv_display") != L.NA_DISPLAY)

    # The six Town Fair Tire properties, end to end.
    ch = rows["CHILD"]
    chk("a CHILD property with a real 0.0 and no loan is NOT debt_free",
        not ch.get("debt_free"))
    chk("it keeps its raw 0.0 and gains no N/A literal",
        ch.get("debt") == 0.0
        and not any(ch.get(k) == L.NA_DISPLAY for k in
                    ("ltv_display", "ytd_dscr_display", "debt_yield_display",
                     "rate_display", "maturity_display")))
    nz = rows["NULLPC"]
    chk("an UNKNOWN parent status behaves like a child, not like a parent",
        not nz.get("debt_free")
        and not any(nz.get(k) == L.NA_DISPLAY for k in
                    ("ltv_display", "ytd_dscr_display", "debt_yield_display")))

    lv = rows["LEVERED"]
    chk("an ordinary levered deal is untouched",
        not lv.get("debt_free") and lv.get("debt") == 25_200_000.0)


def section_totals(rows: dict) -> None:
    print("\nC. the subtotal foots to what the page shows")
    all_rows = list(rows.values())
    total = L.loan_subtotal(all_rows, "t")["debt"]
    # Only LEVERED and ZEROLN carry a summable figure: UNLEV is None by the
    # debt-free rule, SOLDZER is excluded by the sale, NOREAD has no reading,
    # DEVZERO resolves to its committed facility... which is None here (no
    # loan rows), so it contributes nothing either.
    chk("the debt-free row contributes nothing to the total",
        total == 25_200_000.0, f"got {total!r}")
    chk("and the other rows still reach it in full — not a blanket zero",
        total > 0)
    chk("every row is still COUNTED — they are all on the report",
        L.loan_subtotal(all_rows, "t")["deal_count"] == len(CASES))


def section_consumers(out: dict, rows: dict) -> None:
    """The keys SnapshotLoan.vue reads. A renamed or dropped field here is
    silent — the chip just stops rendering and the tooltip goes blank."""
    print("\nD. the screen's contract is unchanged")
    chk("the row still publishes `debt_free` (SnapshotLoan.vue:293 tooltip "
        "'Held with no debt')",
        rows["UNLEV"].get("debt_free") is True)
    chk("the diagnostics still count debt-free deals (the chip at :221)",
        (out.get("diagnostics") or {}).get("debt_free") == 1)
    chk("`debt_basis` survives for the rows that are NOT debt free",
        rows["LEVERED"].get("debt_basis") == L.BASIS_ISBS)
    chk("the row still says WHY, in its flags",
        any("held with no debt" in str(f)
            for f in (rows["UNLEV"].get("flags") or [])))


# ── population mode ───────────────────────────────────────────────────────
#
# STRICTLY READ-ONLY. Every statement below is a SELECT; nothing writes, and
# `create_app()` is deliberately NOT called because on the PostgreSQL path it
# runs ensure_pg_tables() / _pg_fix_column_types(), i.e. it would fire schema
# migrations at production from whatever branch happens to be checked out.
#
# It answers the one question the fixtures cannot: over the REAL deals, at a
# real quarter, which rows does the new rule fire on that the old vcode list
# did not? Every input is taken from the SHIPPING function that computes it —
# no reimplementation — so a disagreement here is a real disagreement.

OLD_DEBT_FREE = {"P0000066"}          # what the deleted hardcode contained


def section_population(quarters: list) -> None:
    import pandas as _pd
    from flask import Flask
    import flask_app.db as _D
    from flask_app.services import data_service as _DS
    from flask_app.services.portfolio_snapshot_debt import (
        committed_facility as _cf, deal_loan_rows as _dlr,
        resolve_debt as _rd)
    from flask_app.services.portfolio_snapshot_service import (
        KEEP_DESPITE_SOLD, _deal_index, _quarter_end, is_acquired_as_of,
        is_sold_as_of)
    from compute import get_isbs_debt_balance as _gidb
    from config import is_dev_deal as _isdev

    app = Flask(__name__)
    app.config["DATABASE_URL"] = os.environ["DATABASE_URL"]
    app.app_context().push()
    eng = _D.get_engine()

    inv = _pd.read_sql("select * from deals", eng)
    # The SHIPPING loan pipeline, in the shipping order — a paid-off loan is
    # dropped and MRI's date-event fan-out collapsed, so `loan_count` means
    # "active facilities" exactly as it does on the page.
    loans_all = _pd.read_sql("select * from loans", eng)
    loans = _DS._collapse_loan_date_events(
        _DS._filter_paid_off_loans(loans_all))
    isbs = _DS._normalize_isbs(_pd.read_sql(
        'select vcode,"dtEntry","vSource","vAccount","mAmount" '
        'from isbs_interim_bs', eng))

    meta = _deal_index(inv)
    print(f"\nE. the REAL population — {len(meta)} deals, "
          f"{len(loans)} active loans, {len(isbs):,} Interim BS rows")

    overall_extra = []
    for q in quarters:
        qe = _quarter_end(q)
        fires, changed, excluded_sold = [], [], []
        for vc, m in sorted(meta.items()):
            sold_now = is_sold_as_of(m, qe)
            on_report = (is_acquired_as_of(m, qe)
                         and (not sold_now
                              or vc.upper() in KEEP_DESPITE_SOLD))
            if not on_report:
                continue
            sold = sold_now and vc.upper() in KEEP_DESPITE_SOLD
            dev = _isdev(m["investment_strategy"])
            lrows = _dlr(loans, vc)
            debt, basis = _rd({"debt_isbs": _gidb(isbs, vc, as_of_date=qe,
                                                  mri_loans=loans)},
                              dev, _cf(lrows))
            new = L._debt_free(debt, basis, len(lrows), dev, sold)
            old = vc.upper() in OLD_DEBT_FREE
            if new:
                fires.append((vc, m["name"], debt, basis, len(lrows),
                              dev, sold))
            if new != old:
                changed.append((vc, m["name"], old, new, debt, basis,
                                len(lrows), dev, sold))
            if sold and debt == 0 and not lrows and not dev:
                excluded_sold.append((vc, m["name"]))

        print(f"\n  {q} (quarter end {qe})")
        print(f"    rule fires on {len(fires)} deal(s):")
        for vc, nm, debt, basis, n, dev, sold in fires:
            print(f"      {vc:<9} {nm[:34]:<35} debt={debt!r:<6} "
                  f"loans={n} basis={basis[:24]}")
        print("    deals with the same DEBT fingerprint held out by the "
              f"`sold` term: {[v for v, _ in excluded_sold] or 'none'}")

        chk(f"{q}: the rule fires on Pegasus ONLY",
            [f[0].upper() for f in fires] == ["P0000066"],
            f"got {[f[0] for f in fires]}")
        chk(f"{q}: no deal's display changes vs the old vcode list",
            not changed,
            "; ".join(f"{c[0]} old={c[2]} new={c[3]} debt={c[4]!r} "
                      f"loans={c[6]} dev={c[7]} sold={c[8]}"
                      for c in changed))
        if changed:
            overall_extra.extend(changed)

    print("\n  BEFORE / AFTER, every deal the rule touches")
    print(f"    {'vcode':<9}{'deal':<35}{'before':<22}{'after':<22}")
    for q in quarters[:1]:
        pass
    print(f"    {'P0000066':<9}{'Pegasus Life Storage':<35}"
          f"{'dash + 5x N/A (vcode)':<22}{'dash + 5x N/A (data)':<22}")
    for c in overall_extra:
        print(f"    {c[0]:<9}{c[1][:34]:<35}"
              f"{'computed' if not c[2] else 'dash + N/A':<22}"
              f"{'dash + N/A' if c[3] else 'computed':<22}")
    if not overall_extra:
        print("    (no other deal — the only change is Pegasus's SOURCE)")


def main() -> int:
    mode = ""
    for a in sys.argv[1:]:
        if a.startswith("--inject"):
            mode = a.split("=", 1)[1] if "=" in a else "off"

    if mode == "off":
        print("=== INJECTION: the rule never fires ===")
        print("    Section A and B's debt-free checks MUST fail.\n")
        L._debt_free = lambda *a, **k: False
    elif mode == "nosold":
        print("=== INJECTION: the `sold` term is dropped ===")
        print("    The City West checks MUST fail.\n")
        _orig = L._debt_free
        L._debt_free = (lambda debt, basis, n, dev, sold, pc:
                        _orig(debt, basis, n, dev, False, pc))
    elif mode == "nochild":
        print("=== INJECTION: the parent term is dropped ===")
        print("    The Town Fair Tire (child / null) checks MUST fail.\n")
        _orig = L._debt_free
        L._debt_free = (lambda debt, basis, n, dev, sold, pc:
                        _orig(debt, basis, n, dev, sold, 1))

    if "--population" in sys.argv:
        qs = [a for a in sys.argv[1:] if not a.startswith("--")]
        section_population(qs or ["2026-Q1", "2026-Q2", "2026-Q3"])
        print(f"\n{_PASS} passed, {_FAIL} failed")
        return 1 if _FAIL else 0

    section_unit()
    out, rows = _build()
    section_rows(rows)
    section_totals(rows)
    section_consumers(out, rows)

    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
