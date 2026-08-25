"""Portfolio Snapshot — the Debt column's basis, in one place.

WHY THIS MODULE EXISTS. The Financial and Loan subtabs both print a "Debt"
column and, until now, chose its source differently: Loan branched on the
development classification and used the committed facility for dev deals, while
Financial took the One Pager cap stack unconditionally. The same deal therefore
carried two different Debt figures on the same report — JB Fair Park 66.36 on
Financial against 77.37 on Loan, Jefferson Waters Creek 48.42 against 51.67.

The rule now lives here and nowhere else, so neither subtab owns it and they
cannot drift again.

THE RULE, and why it is the PDF's:

  Operating deals -> the One Pager's ISBS-derived balance (``cap_stack['debt']``,
  ISBS Interim BS as of quarter end). Fresh, quarter-scoped, and it reproduces
  the reference PDF — 22 of TIAA's 23 non-dev deals tie to the cent at 26Q1.

  Development deals -> ``mOrigLoanAmt`` summed over the deal's loans, the full
  committed facility. This is what PDF page 2 footnote (6) states: "Debt amount
  is current as of 03/31/2026 except for development deals, which reflects fully
  funded debt amount at construction completion."

  A dev deal's ISBS balance is not merely stale-ish, it can be dead. JB Fair
  Park carries a single 2022-12-31 "SENIOR FINANCING" row of 66,363,992 that the
  app keeps alive because an active MRI loan record exists, while the balance
  sheet balances without it (assets ~10.9M against equity ~10.8M) and accounts
  5190/7030/7060 have no rows at all — nothing has ever been serviced on it.

NOTE the honest limit of this change. Rebasing the four dev deals whose figures
actually move WIDENS the portfolio Debt gap against the PDF, from +49.6 to
+95.7, because the previous total was closer only through cancellation: three
deals were understated by a combined -35.1, offsetting overstatement elsewhere.
After the change 29 of 33 deals tie to the cent and the whole remaining gap is
four named data problems (see KNOWN_DEBT_RESIDUALS in
``scripts/snapshot_debt_basis_check.py``). Attributable beats small.

Total Cap is deliberately NOT recomputed from the chosen Debt — see
``resolve_debt``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

#: Basis labels, surfaced per row so a figure can always be traced to its source.
BASIS_ISBS = "ISBS Interim BS (as of quarter end)"
BASIS_COMMITTED = "mOrigLoanAmt (committed facility)"
BASIS_UNAVAILABLE = "unavailable"


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def deal_loan_rows(loans: Optional[pd.DataFrame], vcode: str,
                   child_vcodes: Optional[list] = None) -> pd.DataFrame:
    """The loan rows belonging to a deal, falling back to its children.

    Mirrors the One Pager's parent-inheritance rule: a portfolio parent holding
    no loans of its own (Burton) uses the loans sitting on its children.
    """
    if loans is None or getattr(loans, "empty", True):
        return pd.DataFrame()
    df = loans.copy()
    col = next((c for c in df.columns if c.lower() == "vcode"), None)
    if not col:
        return pd.DataFrame()
    df["_vc"] = df[col].astype(str).str.strip().str.upper()
    own = df[df["_vc"] == str(vcode).strip().upper()]
    if not own.empty:
        return own
    if child_vcodes:
        kids = {str(c).strip().upper() for c in child_vcodes}
        return df[df["_vc"].isin(kids)]
    return pd.DataFrame()


def committed_facility(loan_rows: Optional[pd.DataFrame]) -> Optional[float]:
    """Sum of ``mOrigLoanAmt`` over a deal's loan rows, or None.

    The column is spelled inconsistently across sources (``mOrigLoanAmt`` /
    ``mOriginLoanAmt``), so it is matched case-insensitively on both.
    """
    if loan_rows is None or getattr(loan_rows, "empty", True):
        return None
    col = next((c for c in loan_rows.columns
                if c.lower() in ("moriginloanamt", "morigloanamt")), None)
    if not col:
        return None
    vals = loan_rows[col].map(_num).dropna()
    return float(vals.sum()) if len(vals) else None


def resolve_debt(cap: Optional[dict], dev: bool,
                 committed: Optional[float]) -> tuple:
    """(debt, basis) for one deal — the single decision point.

    ``cap`` is the One Pager ``cap_stack``; ``committed`` is
    ``committed_facility()`` for the deal's loans. A dev deal with no facility
    on record returns (None, BASIS_UNAVAILABLE) rather than falling back to the
    ISBS balance: that balance is the very figure the dev branch exists to
    distrust, and an em dash is honest where a stale number is not. Pegasus Life
    Storage is that case — held debt free, ISBS 0.0, and the PDF prints a dash.

    Total Cap is NOT derived from this. It stays the One Pager's own
    ``cap_stack['total_cap']``, so for a rebased dev deal
    Debt + Total Pref + Ptr Equity no longer foots exactly to Total Cap. That is
    deliberate and matches the published page, which does not foot either
    (JB Fair Park: 48.98 + 14.3 + 3.9 = 67.2 against a printed Total Cap of
    67.1). Recomputing Total Cap would change a second metric to tidy up a
    presentation artefact the source document also carries.
    """
    isbs = _num((cap or {}).get("debt"))
    if not dev:
        return isbs, BASIS_ISBS
    if committed:
        return committed, BASIS_COMMITTED
    return None, BASIS_UNAVAILABLE
