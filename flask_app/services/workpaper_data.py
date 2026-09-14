"""The data behind a workpaper package — the tabs, from the app's MRI tables.

Every function here answers one tab of the example package, from gl_detail /
ia_transactions / commitments rather than from Spreadsheet Server.

HOW A BALANCE IS BUILT, AND WHY IT IS NOT "SUM EVERYTHING". The GL carries two
kinds of row, and the difference decides the arithmetic:

    BALFOR = 'B'  balance-forward. GHIS writes one per account at the first
                  period of a year, described "Opening Balance". It IS the
                  cumulative position at the start of that year.
    BALFOR = 'N'  a normal entry: activity within its period.

So a balance at period P is the year's opening row plus the year's activity to
P -- NOT a sum from the beginning of time, which would count the opening row
and the prior years it already represents. That also means the window in
MRI_GL_Detail.sql (PERIOD >= 202401) does not truncate any balance: each year
carries its own opening row.

    YTD beginning  = sum(AMT) where BALFOR='B' and PERIOD = YYYY01
    YTD change     = sum(AMT) where BALFOR='N' and PERIOD in YYYY01..YYYYMM
    YTD ending     = beginning + change
    QTD change     = sum(AMT) where BALFOR='N' and PERIOD in the quarter

CONFIRM WITH ACCOUNTING BEFORE ANYONE SIGNS ONE OF THESE. The model above is
read off the PPI Eastchase package's own data (its 202601 rows are BASIS 'A',
BALFOR 'B', "Opening Balance"; its activity is BASIS 'B', BALFOR 'N'). Four
BASIS codes exist portfolio-wide -- A, B, C and T -- and accounting's
Spreadsheet Server parameter is the range `A.B`, which excludes C and T. This
module defaults to the same range and takes it as a parameter, but what those
codes mean is a question for accounting, not an inference worth trusting.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from flask_app.db import get_engine

logger = logging.getLogger(__name__)

DEFAULT_BASES = ["A", "B"]


# ── Period helpers ───────────────────────────────────────────────────────

def _p(d: date) -> str:
    return f"{d.year:04d}{d.month:02d}"


def periods_for(period_end: str) -> Dict[str, Any]:
    """The period strings a package needs, from its quarter-end date."""
    pe = pd.to_datetime(period_end).date()
    year_start = date(pe.year, 1, 1)
    q_first_month = ((pe.month - 1) // 3) * 3 + 1
    return {
        "period_end": pe.isoformat(),
        "year": pe.year,
        "ytd_first": _p(year_start),
        "ytd_last": _p(pe),
        "qtd_first": _p(date(pe.year, q_first_month, 1)),
        "qtd_last": _p(pe),
        "prior_year_end": _p(date(pe.year - 1, 12, 1)),
    }


def _gl(entityid: str, bases: List[str], engine) -> pd.DataFrame:
    """Every GL row for one entity, on the requested bases."""
    with engine.connect() as conn:
        df = pd.read_sql(text(
            'SELECT * FROM gl_detail WHERE UPPER(TRIM("ENTITYID")) = :e'),
            conn, params={"e": entityid.strip().upper()})
    if df.empty:
        return df
    df["BASIS"] = df["BASIS"].astype(str).str.strip()
    df["BALFOR"] = df["BALFOR"].astype(str).str.strip()
    df["PERIOD"] = df["PERIOD"].astype(str).str.strip()
    df["AMT"] = pd.to_numeric(df["AMT"], errors="coerce").fillna(0.0)
    if bases:
        df = df[df["BASIS"].isin(bases)]
    return df


# ── Tabs ─────────────────────────────────────────────────────────────────

def trial_balance(entityid: str, period_end: str, bases: Optional[List[str]] = None,
                  engine=None) -> Dict[str, Any]:
    """Account-level trial balance with the FS tag beside each account."""
    engine = engine or get_engine()
    bases = bases if bases is not None else DEFAULT_BASES
    p = periods_for(period_end)
    gl = _gl(entityid, bases, engine)
    if gl.empty:
        return {"periods": p, "rows": [], "note": "No GL rows for this entity"}

    ytd = gl[(gl["PERIOD"] >= p["ytd_first"]) & (gl["PERIOD"] <= p["ytd_last"])]
    opening = ytd[(ytd["BALFOR"] == "B") & (ytd["PERIOD"] == p["ytd_first"])]
    activity = ytd[ytd["BALFOR"] == "N"]
    qtd = activity[(activity["PERIOD"] >= p["qtd_first"]) & (activity["PERIOD"] <= p["qtd_last"])]

    names = (gl.groupby("ACCTNUM")["ACCTNAME"].first().to_dict())
    beg = opening.groupby("ACCTNUM")["AMT"].sum()
    chg = activity.groupby("ACCTNUM")["AMT"].sum()
    qchg = qtd.groupby("ACCTNUM")["AMT"].sum()

    fs = {r["acctnum"]: r for r in _fs_map(engine)}
    rows = []
    for acct in sorted(set(beg.index) | set(chg.index)):
        b, c = float(beg.get(acct, 0.0)), float(chg.get(acct, 0.0))
        tag = fs.get(str(acct).strip(), {})
        rows.append({
            "acctnum": acct,
            "acctname": names.get(acct, ""),
            "fs_statement": tag.get("statement"),
            "fs_line": tag.get("fs_line"),
            "ytd_beginning": b,
            "ytd_change": c,
            "ytd_ending": b + c,
            "qtd_change": float(qchg.get(acct, 0.0)),
        })
    return {"periods": p, "bases": bases, "rows": rows,
            "unmapped": [r["acctnum"] for r in rows if not r["fs_line"]]}


def _fs_map(engine) -> List[dict]:
    from flask_app.services.workpaper_service import get_fs_map
    try:
        return get_fs_map(engine)
    except Exception:
        return []


def gl_detail(entityid: str, period_end: str, bases: Optional[List[str]] = None,
              full_year: bool = True, engine=None) -> List[dict]:
    """The GL Detail tab — entries for the year to the period end."""
    engine = engine or get_engine()
    bases = bases if bases is not None else DEFAULT_BASES
    p = periods_for(period_end)
    gl = _gl(entityid, bases, engine)
    if gl.empty:
        return []
    first = p["ytd_first"] if full_year else p["qtd_first"]
    gl = gl[(gl["PERIOD"] >= first) & (gl["PERIOD"] <= p["ytd_last"])]
    cols = ["ENTITYID", "PERIOD", "ENTRDATE", "ACCTNAME", "ACCTNUM", "BASIS",
            "BALFOR", "ITEM", "REF", "DESCRPN", "SEGMENTID", "RLTDENTITY",
            "RLTDENTITY_NAME", "AMT"]
    gl = gl[[c for c in cols if c in gl.columns]]
    return gl.sort_values(["PERIOD", "ACCTNUM"]).to_dict("records")


def account_summary(entityid: str, period_end: str, bases: Optional[List[str]] = None,
                    engine=None) -> List[dict]:
    """GL activity grouped by account — the shape every pivot tab in the
    example package takes (Cash, Intercompany, Accruals, Expenses, ...).
    One summary beats a dozen near-identical tabs until the CFO says which
    breakdowns they actually want as separate sheets."""
    engine = engine or get_engine()
    bases = bases if bases is not None else DEFAULT_BASES
    p = periods_for(period_end)
    gl = _gl(entityid, bases, engine)
    if gl.empty:
        return []
    gl = gl[(gl["PERIOD"] >= p["ytd_first"]) & (gl["PERIOD"] <= p["ytd_last"])
            & (gl["BALFOR"] == "N")]
    if gl.empty:
        return []
    g = gl.groupby(["ACCTNUM", "ACCTNAME"], dropna=False).agg(
        entries=("AMT", "size"), amount=("AMT", "sum")).reset_index()
    return g.sort_values("ACCTNUM").to_dict("records")


def _ia(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        try:
            df = pd.read_sql(text("SELECT * FROM ia_transactions"), conn)
        except Exception:
            logger.warning("ia_transactions not loaded", exc_info=True)
            return pd.DataFrame()
    if df.empty:
        return df
    for c in ("InvestmentID", "InvestorID"):
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    return df


def investor_detail(entityid: str, period_end: str, engine=None) -> List[dict]:
    """Who invested INTO this entity — the workpaper runs the IA query with
    InvestmentID = the entity and InvestorID = *."""
    engine = engine or get_engine()
    df = _ia(engine)
    if df.empty:
        return []
    cut = pd.to_datetime(period_end)
    df = df[(df["InvestmentID"] == entityid.strip().upper()) & (df["TransactionDate"] <= cut)]
    return _ia_rows(df, period_end)


def investment_detail(entityid: str, period_end: str, engine=None) -> List[dict]:
    """What this entity invested INTO — the same query with the filters
    swapped (InvestorID = the entity). Same rows, other side."""
    engine = engine or get_engine()
    df = _ia(engine)
    if df.empty:
        return []
    cut = pd.to_datetime(period_end)
    df = df[(df["InvestorID"] == entityid.strip().upper()) & (df["TransactionDate"] <= cut)]
    return _ia_rows(df, period_end)


def _ia_rows(df: pd.DataFrame, period_end: str) -> List[dict]:
    """Tag each row Beginning Balance vs Current Period, as the workbook does."""
    if df.empty:
        return []
    pe = pd.to_datetime(period_end)
    year_start = pd.Timestamp(pe.year, 1, 1)
    out = df.copy()
    out["Period"] = out["TransactionDate"].apply(
        lambda d: "Beginning Balance" if pd.notna(d) and d < year_start else "Current Period")
    out["TransactionDate"] = out["TransactionDate"].dt.strftime("%Y-%m-%d")
    if "EffectiveDate" in out.columns:
        out["EffectiveDate"] = pd.to_datetime(
            out["EffectiveDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    cols = ["InvestmentID", "InvestmentName", "InvestorID", "InvestorName",
            "TransactionDate", "EffectiveDate", "MajorType", "Typename",
            "Amount", "Period"]
    out = out[[c for c in cols if c in out.columns]]
    return out.sort_values(["InvestorID", "TransactionDate"]).to_dict("records")


def ia_rollforward(entityid: str, period_end: str, engine=None) -> Dict[str, Any]:
    """The IA Query RF tab: per investor, beginning balance + current period.

    Note this needs the NON-CASH rows the app's accounting_feed drops -- in
    the example package the entire current-period movement is MajorType
    'Other' ("Income/Loss before Fees"). ia_transactions keeps them, which is
    why the rollforward is reproducible at all.
    """
    engine = engine or get_engine()
    df = _ia(engine)
    if df.empty:
        return {"rows": [], "columns": []}
    cut = pd.to_datetime(period_end)
    pe = cut
    year_start = pd.Timestamp(pe.year, 1, 1)
    df = df[(df["InvestmentID"] == entityid.strip().upper()) & (df["TransactionDate"] <= cut)]
    if df.empty:
        return {"rows": [], "columns": []}
    df["bucket"] = df["TransactionDate"].apply(
        lambda d: "Beginning Balance" if pd.notna(d) and d < year_start else "Current Period")
    piv = df.pivot_table(index=["InvestorID", "InvestorName"], columns="bucket",
                         values="Amount", aggfunc="sum", fill_value=0.0).reset_index()
    for c in ("Beginning Balance", "Current Period"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv["Total"] = piv["Beginning Balance"] + piv["Current Period"]
    rows = piv.to_dict("records")
    total = {"InvestorID": "TOTAL", "InvestorName": "",
             "Beginning Balance": float(piv["Beginning Balance"].sum()),
             "Current Period": float(piv["Current Period"].sum()),
             "Total": float(piv["Total"].sum())}
    return {"rows": rows + [total],
            "columns": ["InvestorID", "InvestorName", "Beginning Balance",
                        "Current Period", "Total"]}


def commitment_rollforward(entityid: str, engine=None) -> List[dict]:
    """Commitments for the entity, from the IA_Commitment feed."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        try:
            df = pd.read_sql(text("SELECT * FROM commitments"), conn)
        except Exception:
            return []
    if df.empty:
        return []
    ent = entityid.strip().upper()
    cols = [c for c in df.columns if c.lower() in ("investmentid", "investorid")]
    if not cols:
        return []
    mask = False
    for c in cols:
        mask = mask | (df[c].astype(str).str.strip().str.upper() == ent)
    return df[mask].to_dict("records")


def financial_statements(entityid: str, period_end: str, bases: Optional[List[str]] = None,
                         engine=None) -> Dict[str, Any]:
    """Trial balance folded through the FS mapping.

    Returns one section per statement, each a list of FS lines with their
    contributing accounts. UNMAPPED ACCOUNTS ARE REPORTED, NOT DROPPED: a
    statement that silently omits an account is the one error an auditor will
    find and nobody else will.
    """
    tb = trial_balance(entityid, period_end, bases, engine)
    rows = tb["rows"]
    statements: Dict[str, Dict[str, Dict[str, Any]]] = {}
    unmapped = []
    for r in rows:
        if not r["fs_line"]:
            unmapped.append(r)
            continue
        stmt = r["fs_statement"] or "Unassigned statement"
        line = statements.setdefault(stmt, {}).setdefault(
            r["fs_line"], {"fs_line": r["fs_line"], "ytd_ending": 0.0,
                           "ytd_change": 0.0, "accounts": []})
        line["ytd_ending"] += r["ytd_ending"]
        line["ytd_change"] += r["ytd_change"]
        line["accounts"].append(r["acctnum"])
    return {
        "periods": tb["periods"],
        "statements": {k: list(v.values()) for k, v in statements.items()},
        "unmapped": unmapped,
        "unmapped_total": sum(r["ytd_ending"] for r in unmapped),
    }
