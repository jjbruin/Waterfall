"""Guardrail: one PE investor recorded under two casings must be counted once.

Background
----------
The accounting feed carries the same InvestorID under two casings on nine deals
(e.g. PPI11 and ppi11 on Centre at Westbank).  Two comparisons disagreed on
case, and the disagreement silently doubled money:

  * ``get_deal_pe_investors()`` de-duplicated with a case-SENSITIVE ``unique()``,
    so the pair surfaced as two investors.
  * ``build_pref_balance_detail()`` filters the ledger case-INSENSITIVELY, so
    each of those "two" investors pulled the whole combined ledger.
  * ``_compute_accrued_from_pref_detail()`` summed them, so the One Pager's
    Preferred Equity "Accrued Balance" came out at exactly 2x the truth
    (Pontchartrain Landing was overstated by $1.27M).

A secondary defect: the PE and capitalization blocks matched vcode
case-sensitively, so a lowercase vcode rendered an all-zero section rather than
raising — a blank report that looks like real data.

This runs on synthetic frames only: no database, no network, no credentials.

    python scripts/pe_investor_case_guardrail.py
"""
from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import normalize_accounting_feed, load_waterfalls  # noqa: E402
from one_pager import get_pe_performance, get_capitalization_stack  # noqa: E402
from flask_app.services.reports_service import (  # noqa: E402
    get_deal_pe_investors,
    build_pref_balance_detail,
)

VCODE = "P0009999"
IID = "TESTIID"
REPORT_DATE = date(2026, 6, 30)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deal with one PE investor whose rows are split across two casings."""
    deals = pd.DataFrame([{
        "vcode": VCODE, "InvestmentID": IID, "Investment_Name": "Case Test Deal",
        "Portfolio_Name": None, "Property_Count": "1",
    }])
    # Half the contributions filed as PPI99, half as ppi99 — one real investor.
    rows = [
        (IID, "PPI99", "2024-01-01", "Contribution", "Contribution: Investments", -1_000_000.0, 1018),
        (IID, "ppi99", "2024-07-01", "Contribution", "Contribution: Investments", -1_000_000.0, 1018),
        (IID, "PPI99", "2025-06-30", "Distribution", "Distribution: Preferred Return", 50_000.0, 1019),
        (IID, "OPTEST", "2024-01-01", "Contribution", "Contribution: Investments", -500_000.0, 1018),
    ]
    acct = pd.DataFrame(
        [{"InvestmentID": a, "InvestorID": b, "EffectiveDate": c, "MajorType": d,
          "Typename": e, "Amt": f, "TypeID": g, "Capital": "N", "Partner": ""}
         for a, b, c, d, e, f, g in rows]
    )
    wf = pd.DataFrame([{
        "vcode": VCODE, "vmisc": "CF_WF", "iOrder": 1, "PropCode": "PPI99",
        "vState": "Pref", "FXRate": 1.0, "nPercent": 0.10, "mAmount": None,
        "vtranstype": None, "vAmtType": None, "vNotes": "Pref",
        "dteffective": "2024-01-01",
    }])
    return deals, normalize_accounting_feed(acct), load_waterfalls(wf)


def main() -> int:
    deals, acct, wf = build_frames()

    print("1. investor list collapses casing variants")
    investors = [i["investor_id"] for i in get_deal_pe_investors(VCODE, acct, deals)]
    non_op = [i for i in investors if not i.upper().startswith("OP")]
    check("one entry per real investor", len(non_op) == 1, f"got {investors}")
    check("canonical casing kept", non_op == ["PPI99"] if non_op else False, f"got {non_op}")

    print("\n2. the ledger filter really is case-insensitive (why dedupe is required)")
    hi = build_pref_balance_detail(VCODE, "PPI99", REPORT_DATE, acct, deals, wf_steps=wf)
    lo = build_pref_balance_detail(VCODE, "ppi99", REPORT_DATE, acct, deals, wf_steps=wf)
    a_hi = abs(hi["header"].get("accrued_pref", 0.0))
    a_lo = abs(lo["header"].get("accrued_pref", 0.0))
    check("both casings return the identical ledger", len(hi["rows"]) == len(lo["rows"]) and abs(a_hi - a_lo) < 1e-9,
          f"{len(hi['rows'])} vs {len(lo['rows'])} rows, {a_hi:,.2f} vs {a_lo:,.2f}")
    check("ledger spans both casings' contributions",
          abs(hi["header"].get("investment_balance", 0.0) - 2_000_000.0) < 1e-6,
          f"balance {hi['header'].get('investment_balance', 0.0):,.2f}")

    print("\n3. summing the investor list counts that ledger exactly once")
    total = sum(
        abs(build_pref_balance_detail(VCODE, i, REPORT_DATE, acct, deals, wf_steps=wf)
            ["header"].get("accrued_pref", 0.0))
        for i in non_op
    )
    check("sum equals the single-investor value", abs(total - a_hi) < 1e-9,
          f"sum {total:,.2f} vs single {a_hi:,.2f}")
    check("sum is not doubled", abs(total - 2 * a_hi) > 1e-6 or a_hi == 0,
          f"2x would be {2 * a_hi:,.2f}")

    print("\n4. vcode matching is case-insensitive (no silent all-zero section)")
    pe_u = get_pe_performance(VCODE, "2026-Q2", acct, wf, deals, isbs_raw=None)
    pe_l = get_pe_performance(VCODE.lower(), "2026-Q2", acct, wf, deals, isbs_raw=None)
    check("PE funded_to_date matches across casing",
          abs(pe_u["funded_to_date"] - pe_l["funded_to_date"]) < 1e-6,
          f"{pe_u['funded_to_date']:,.2f} vs {pe_l['funded_to_date']:,.2f}")
    check("PE funded_to_date is non-zero", pe_l["funded_to_date"] > 0,
          f"{pe_l['funded_to_date']:,.2f}")
    cs_u = get_capitalization_stack(VCODE, None, None, wf, acct, deals,
                                    isbs_raw=None, quarter_str="2026-Q2")
    cs_l = get_capitalization_stack(VCODE.lower(), None, None, wf, acct, deals,
                                    isbs_raw=None, quarter_str="2026-Q2")
    check("cap-stack pref equity matches across casing",
          abs(cs_u["pref_equity"] - cs_l["pref_equity"]) < 1e-6,
          f"{cs_u['pref_equity']:,.2f} vs {cs_l['pref_equity']:,.2f}")
    check("cap-stack pref equity is non-zero", cs_l["pref_equity"] > 0,
          f"{cs_l['pref_equity']:,.2f}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
