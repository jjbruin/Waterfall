"""Financial statement engine — one definition, any entity, any period.

Produces the Balance Sheet and Income Statement for ANY entity from the GL,
so the same numbers serve a workpaper package, a standalone entity statement,
and (later) a consolidation. The account-to-line mapping is maintained once
(wp_fs_map) rather than retyped into every workbook every quarter, which is
what the example package does today: 192 accounts tagged by hand, per entity,
per close.

WHICH STATEMENT AN ACCOUNT BELONGS TO IS MRI'S ANSWER, NOT OURS. GACC.TYPE
carries it:

    B  balance sheet   1,353 accounts
    C  cash              111
    I  income statement  698

Measured against production on Sep 14 2026, that classification is not a
guess -- it predicts the GL's own behaviour. Comparing each account's 2025
opening plus 2025 activity against its 2026 opening:

    TYPE C   98% tie      87 of 111 carry a 2026 opening
    TYPE B   90% tie   1,096 of 1,353 carry one
    TYPE I   11% tie       8 of   698 carry one

Balance-sheet and cash accounts roll forward; income accounts are closed to
equity at year end, so they have no opening balance to carry and it would be
wrong to give them one. THIS IS WHY NEITHER STATEMENT NEEDS LAST QUARTER'S
APPROVED WORKBOOK: the Balance Sheet's opening is the GL's own balance-forward
row, and the Income Statement is a period statement with no opening at all.
(Members' Capital and Cash Flow are the two that do reach back further -- they
are not built here yet.)

THE MAPPING SAYS WHERE ON THE STATEMENT, NOT WHICH STATEMENT. If a mapping row
claims a statement that disagrees with GACC.TYPE, the disagreement is reported
rather than resolved: MRI's classification wins for placement, and the
conflict is surfaced so somebody fixes the mapping.

SIGNS. The GL stores credits negative -- a capital contribution is -2,694,676,
a liability is negative. Statements present every section positive. So each
section carries a presentation sign, applied once at the end and reported in
the output, so a reader can always get back to the GL figure. Nothing is
silently flipped per account.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from flask_app.db import get_engine
from flask_app.services.workpaper_data import DEFAULT_BASES, periods_for

logger = logging.getLogger(__name__)

# GACC.TYPE -> the statement the account belongs on.
TYPE_STATEMENT = {"B": "balance_sheet", "C": "balance_sheet", "I": "income_statement"}

# Default presentation sign per section. Assets are debit-normal and already
# read positive; everything else is credit-normal and is negated for display.
SECTION_SIGN = {
    "Assets": 1,
    "Liabilities": -1,
    "Members' Capital": -1,
    "Income": -1,
    "Expenses": 1,
}

BALANCE_SHEET_SECTIONS = ["Assets", "Liabilities", "Members' Capital"]
INCOME_SECTIONS = ["Income", "Expenses"]


def account_types(engine=None) -> Dict[str, Dict[str, str]]:
    """ACCTNUM -> {type, name}. Empty if gl_accounts has not been loaded."""
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text('SELECT "ACCTNUM", "ACCTNAME", "TYPE" FROM gl_accounts'), conn)
    except Exception:
        logger.warning("gl_accounts not available", exc_info=True)
        return {}
    df["ACCTNUM"] = df["ACCTNUM"].astype(str).str.strip()
    return {r["ACCTNUM"]: {"type": str(r["TYPE"]).strip(),
                           "name": str(r["ACCTNAME"]).strip()}
            for _, r in df.iterrows()}


def _mapping(engine=None) -> Dict[str, dict]:
    from flask_app.services.workpaper_service import get_fs_map
    try:
        return {m["acctnum"]: m for m in get_fs_map(engine)}
    except Exception:
        return {}


def _balances(entityid: str, period_end: str, bases: List[str], engine) -> pd.DataFrame:
    """Per account: opening, YTD activity, closing, QTD activity."""
    p = periods_for(period_end)
    with engine.connect() as conn:
        gl = pd.read_sql(text(
            'SELECT "ACCTNUM", "ACCTNAME", "PERIOD", "BALFOR", "BASIS", "AMT" '
            'FROM gl_detail WHERE UPPER(TRIM("ENTITYID")) = :e'),
            conn, params={"e": entityid.strip().upper()})
    if gl.empty:
        return gl
    for c in ("ACCTNUM", "PERIOD", "BALFOR", "BASIS"):
        gl[c] = gl[c].astype(str).str.strip()
    gl["AMT"] = pd.to_numeric(gl["AMT"], errors="coerce").fillna(0.0)
    if bases:
        gl = gl[gl["BASIS"].isin(bases)]
    if gl.empty:
        return gl

    ytd = gl[(gl["PERIOD"] >= p["ytd_first"]) & (gl["PERIOD"] <= p["ytd_last"])]
    opening = ytd[(ytd["BALFOR"] == "B") & (ytd["PERIOD"] == p["ytd_first"])]
    activity = ytd[ytd["BALFOR"] == "N"]
    qtd = activity[(activity["PERIOD"] >= p["qtd_first"]) & (activity["PERIOD"] <= p["qtd_last"])]

    out = pd.DataFrame({
        "opening": opening.groupby("ACCTNUM")["AMT"].sum(),
        "ytd": activity.groupby("ACCTNUM")["AMT"].sum(),
        "qtd": qtd.groupby("ACCTNUM")["AMT"].sum(),
    }).fillna(0.0)
    out["closing"] = out["opening"] + out["ytd"]
    out["name"] = gl.groupby("ACCTNUM")["ACCTNAME"].first()
    return out.reset_index()


def build(entityid: str, period_end: str, statement: str = "both",
          bases: Optional[List[str]] = None, engine=None) -> Dict[str, Any]:
    """Balance Sheet and/or Income Statement for one entity and period."""
    engine = engine or get_engine()
    bases = bases if bases is not None else DEFAULT_BASES
    p = periods_for(period_end)
    bal = _balances(entityid, period_end, bases, engine)
    types = account_types(engine)
    mapping = _mapping(engine)

    if bal.empty:
        return {"entity": entityid, "periods": p, "bases": bases,
                "balance_sheet": None, "income_statement": None,
                "note": "No GL rows for this entity on these bases"}

    unmapped: List[dict] = []
    untyped: List[dict] = []
    conflicts: List[dict] = []
    sections: Dict[str, Dict[str, dict]] = {}

    for _, r in bal.iterrows():
        acct = r["ACCTNUM"]
        meta = types.get(acct)
        m = mapping.get(acct)
        row = {"acctnum": acct, "acctname": r["name"],
               "opening": float(r["opening"]), "ytd": float(r["ytd"]),
               "qtd": float(r["qtd"]), "closing": float(r["closing"])}

        if not meta or meta["type"] not in TYPE_STATEMENT:
            # No GACC row, or a type outside B/C/I. Never guessed onto a
            # statement -- an account on the wrong statement is worse than an
            # account visibly missing from both.
            untyped.append({**row, "type": (meta or {}).get("type")})
            continue
        stmt = TYPE_STATEMENT[meta["type"]]

        if not m or not m.get("fs_line"):
            unmapped.append({**row, "statement": stmt})
            continue

        section = (m.get("statement") or "").strip()
        valid = BALANCE_SHEET_SECTIONS if stmt == "balance_sheet" else INCOME_SECTIONS
        if section not in valid:
            # The mapping put this account in a section that does not belong
            # to the statement MRI says it is on. Report; do not resolve.
            conflicts.append({**row, "mapped_section": section or None,
                              "gacc_type": meta["type"], "statement_by_type": stmt})
            continue

        key = (stmt, section, m["fs_line"])
        line = sections.setdefault(stmt, {}).setdefault(
            key, {"section": section, "fs_line": m["fs_line"],
                  "sort_order": m.get("sort_order", 0),
                  "opening": 0.0, "ytd": 0.0, "qtd": 0.0, "closing": 0.0,
                  "accounts": []})
        for f in ("opening", "ytd", "qtd", "closing"):
            line[f] += row[f]
        line["accounts"].append({"acctnum": acct, "acctname": row["acctname"],
                                 "closing": row["closing"], "ytd": row["ytd"]})

    def render(stmt_key: str, section_names: List[str], value_field: str) -> dict:
        lines = sorted(sections.get(stmt_key, {}).values(),
                       key=lambda l: (section_names.index(l["section"])
                                      if l["section"] in section_names else 99,
                                      l["sort_order"], l["fs_line"]))
        out_sections = []
        for name in section_names:
            sec_lines = [l for l in lines if l["section"] == name]
            if not sec_lines:
                continue
            sign = SECTION_SIGN.get(name, 1)
            rendered = [{
                "fs_line": l["fs_line"],
                "amount": sign * l[value_field],
                "gl_amount": l[value_field],
                "accounts": l["accounts"],
            } for l in sec_lines]
            out_sections.append({
                "section": name, "presentation_sign": sign,
                "lines": rendered,
                "total": sum(x["amount"] for x in rendered),
                "gl_total": sum(x["gl_amount"] for x in rendered),
            })
        return {"sections": out_sections,
                "gl_total": sum(s["gl_total"] for s in out_sections)}

    result: Dict[str, Any] = {
        "entity": entityid, "periods": p, "bases": bases,
        "unmapped": unmapped, "untyped": untyped, "conflicts": conflicts,
        "unmapped_total": sum(r["closing"] for r in unmapped),
    }

    if statement in ("both", "balance_sheet"):
        bs = render("balance_sheet", BALANCE_SHEET_SECTIONS, "closing")
        # THE TIE-OUT. In GL signs a complete balance sheet sums to zero:
        # debits and credits net. A non-zero total is the amount that is
        # unmapped, misclassified or genuinely out of balance -- it is the
        # statement's own error bar and is reported beside it, not hidden.
        bs["out_of_balance"] = bs["gl_total"]
        bs["balanced"] = abs(bs["gl_total"]) < 0.01
        result["balance_sheet"] = bs

    if statement in ("both", "income_statement"):
        inc = render("income_statement", INCOME_SECTIONS, "ytd")
        inc["net_income"] = -inc["gl_total"]
        result["income_statement"] = inc

    return result


def seed_mapping_from_names(engine=None) -> List[dict]:
    """A starting mapping proposal, from each account's own name.

    NOT applied automatically. The example package maps 192 accounts to 56
    lines by hand; proposing the obvious ones saves that typing, but a
    proposal presented as a decision is how a wrong line ends up in a
    statement nobody re-checked. The screen shows these as suggestions.
    """
    engine = engine or get_engine()
    types = account_types(engine)
    props = []
    for acct, meta in sorted(types.items()):
        t = meta["type"]
        if t not in TYPE_STATEMENT:
            continue
        name = meta["name"]
        low = name.lower()
        if t == "C":
            section, line = "Assets", "Cash and cash equivalents"
        elif t == "I":
            section = "Expenses" if ("expense" in low or "fee" in low) else "Income"
            line = name.split(":")[0].strip()
        else:
            if any(w in low for w in ("payable", "accrued", "due to", "liabilit")):
                section = "Liabilities"
            elif any(w in low for w in ("capital", "contribution", "distribution",
                                        "retained", "members", "equity")):
                section = "Members' Capital"
            else:
                section = "Assets"
            line = name.split(":")[0].strip()
        props.append({"acctnum": acct, "acctname": name, "gacc_type": t,
                      "statement": section, "fs_line": line, "suggested": True})
    return props
