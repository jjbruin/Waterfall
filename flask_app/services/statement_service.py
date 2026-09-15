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
Members' Capital and Cash Flow are built below and reach further back, but
still only into the GL and the IA subledger -- not into a prior workbook.

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


def line_sort_key(section_names: List[str], line: dict):
    """Where a line sits on the statement.

    ORDER IS PART OF BEING A STATEMENT. A financial statement runs
    most-liquid first -- cash, receivables, prepaids, investments, other --
    not in whatever order the mapping happened to be written. The ranks live
    in fs_line_seed.LINE_ORDER; a caption with no rank sorts after the ranked
    ones, alphabetically, so adding a caption never disturbs the rest.
    """
    from flask_app.services import fs_line_seed as seed
    section = line["section"]
    return (section_names.index(section) if section in section_names else 99,
            seed.LINE_ORDER.get(line["fs_line"], 9999), line["fs_line"])


def is_dormant(line: dict) -> bool:
    """No balance AND no movement -- nothing happened on this line.

    Zero WITH movement is not dormant: a balance that went out and came back
    is a fact about the period, and hiding it would make the statement
    disagree with the trial balance behind it.
    """
    return (abs(line.get("closing", 0.0) or 0.0) < 0.005
            and abs(line.get("ytd", 0.0) or 0.0) < 0.005)


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
                       key=lambda l: line_sort_key(section_names, l))
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
                "dormant": is_dormant(l),
            } for l in sec_lines]
            out_sections.append({
                "section": name, "presentation_sign": sign,
                "lines": rendered,
                "dormant_count": sum(1 for x in rendered if x["dormant"]),
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

    # The income statement is built even when only the balance sheet was
    # asked for, because the balance sheet needs the period's result.
    inc = render("income_statement", INCOME_SECTIONS, "ytd")
    inc["net_income"] = -inc["gl_total"]

    if statement in ("both", "balance_sheet"):
        bs = render("balance_sheet", BALANCE_SHEET_SECTIONS, "closing")

        # THE PERIOD'S RESULT BELONGS IN MEMBERS' CAPITAL, and the balance
        # sheet does not balance without it. Income accounts are closed to
        # equity at YEAR END, so at any date before that the equity accounts
        # hold opening capital plus capital movements and NOT the year's
        # profit or loss -- which is why every entity came out of balance by
        # exactly its net income when the mapping was first applied
        # (PPIECH -11,745.08, AMB6 -16,282.49, Sep 14 2026). The figure is the
        # income statement's own total, so the two statements cannot disagree.
        if abs(inc["gl_total"]) > 0.005:
            period_line = {
                "fs_line": "Net increase (decrease) in members' capital "
                           "resulting from operations",
                "amount": SECTION_SIGN["Members' Capital"] * inc["gl_total"],
                "gl_amount": inc["gl_total"],
                "accounts": [],
                "from_income_statement": True,
            }
            sec = next((s for s in bs["sections"]
                        if s["section"] == "Members' Capital"), None)
            if sec is None:
                sec = {"section": "Members' Capital",
                       "presentation_sign": SECTION_SIGN["Members' Capital"],
                       "lines": [], "total": 0.0, "gl_total": 0.0}
                bs["sections"].append(sec)
            sec["lines"].append(period_line)
            sec["total"] += period_line["amount"]
            sec["gl_total"] += period_line["gl_amount"]
            bs["gl_total"] += period_line["gl_amount"]

        # THE TIE-OUT. In GL signs a complete balance sheet sums to zero:
        # debits and credits net. A non-zero total is the amount that is
        # unmapped, misclassified or genuinely out of balance -- it is the
        # statement's own error bar and is reported beside it, not hidden.
        bs["out_of_balance"] = bs["gl_total"]
        bs["balanced"] = abs(bs["gl_total"]) < 0.01

        # A LINE FACING THE WRONG WAY IS A MAPPING ERROR, AND IT STILL
        # BALANCES. A negative asset or a negative liability nets out
        # correctly, so the tie-out cannot catch it -- only a reader can, and
        # only if they notice a minus sign in a column of positives.
        #
        # The case that found this: the example package tags MR22000002,
        # named "Other Liabilities", to the asset line "Due from Manager".
        # Harmless in that workbook because the account is zero for PPIECH;
        # on AMB6 the same mapping put -629,125.04 into assets. The app
        # inherited accounting's own tagging error and had no way to say so.
        bs["sign_anomalies"] = [
            {"section": sec["section"], "fs_line": l["fs_line"],
             "amount": l["amount"],
             "accounts": [a["acctnum"] for a in l["accounts"]],
             "account_names": [a["acctname"] for a in l["accounts"]]}
            for sec in bs["sections"] for l in sec["lines"]
            if l["amount"] < -0.005 and not l.get("from_income_statement")
        ]
        result["balance_sheet"] = bs

    if statement in ("both", "income_statement"):
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


# ── Members' Capital ─────────────────────────────────────────────────────
# Per member, from the IA subledger: the GL carries one equity balance for the
# entity, the members' split lives in ia_transactions. Rows follow the example
# package's own statement; the Typename -> row map below is the default, drawn
# from the 28 types actually present in production, and is the first thing to
# put in front of the CFO.

MC_ROWS = [
    "Capital contributions",
    "Capital distributions",
    "Net investment income (loss)",
    "Management fee",
    "Net realized gain (loss)",
    "Net change in unrealized gain (loss)",
    "Carried interest allocation",
    "Unrealized carried interest allocation",
    "Transfer of ownership",
]

MC_TYPENAME_ROW = {
    "Income/Loss before Fees": "Net investment income (loss)",
    "Management Fee": "Management fee",
    "Realized Gain/Loss": "Net realized gain (loss)",
    "Unrealized Gain/Loss": "Net change in unrealized gain (loss)",
    "Realized Carried Interest Allocation": "Carried interest allocation",
    "Unrealized Carried Interest Allocation": "Unrealized carried interest allocation",
    "Transfer of Ownership - Capital": "Transfer of ownership",
    "Transfer of Ownership - P&L": "Transfer of ownership",
}


def _mc_row(major_type: str, typename: str) -> str:
    """Which movement row a transaction belongs on.

    MajorType decides for contributions and distributions -- all nine
    contribution types and all eleven distribution types are capital movement
    whatever their sub-type. Only MajorType 'Other' needs the sub-type,
    because that is where income, fees, gains and carried interest live.
    Anything unrecognised lands on "Other movement" rather than being dropped.
    """
    mt = (major_type or "").strip()
    if mt == "Contribution":
        return "Capital contributions"
    if mt == "Distribution":
        return "Capital distributions"
    return MC_TYPENAME_ROW.get((typename or "").strip(), "Other movement")


def build_members_capital(entityid: str, period_end: str, engine=None) -> Dict[str, Any]:
    """Statement of Changes in Members' Capital, per member.

    Opening comes from the subledger itself -- every transaction before the
    year start -- NOT from last quarter's workbook. The example package says
    "Use Prior Quarter FS" because an accountant in Excel cannot query the
    subledger; the app can.

    TIE-OUT: the members' total must equal the entity's GL equity. The GL is
    the control account and ia_transactions is the subledger; a difference
    means one of the two is wrong, and it is reported rather than reconciled
    away.
    """
    engine = engine or get_engine()
    p = periods_for(period_end)
    ent = entityid.strip().upper()
    with engine.connect() as conn:
        try:
            ia = pd.read_sql(text("SELECT * FROM ia_transactions"), conn)
        except Exception:
            logger.warning("ia_transactions not loaded", exc_info=True)
            return {"entity": entityid, "periods": p, "members": [], "rows": [],
                    "note": "ia_transactions not loaded"}
    if ia.empty:
        return {"entity": entityid, "periods": p, "members": [], "rows": [],
                "note": "No investor activity"}

    ia["InvestmentID"] = ia["InvestmentID"].astype(str).str.strip().str.upper()
    ia["Amount"] = pd.to_numeric(ia["Amount"], errors="coerce").fillna(0.0)
    ia["TransactionDate"] = pd.to_datetime(ia["TransactionDate"], errors="coerce")
    ia = ia[ia["InvestmentID"] == ent]
    if ia.empty:
        return {"entity": entityid, "periods": p, "members": [], "rows": [],
                "note": "No investor activity for this entity"}

    year_start = pd.Timestamp(p["year"], 1, 1)
    cut = pd.Timestamp(p["period_end"])
    ia = ia[ia["TransactionDate"] <= cut]
    ia["row"] = [_mc_row(m, t) for m, t in zip(ia["MajorType"], ia["Typename"])]

    members = (ia.groupby(["InvestorID", "InvestorName"], dropna=False)["Amount"]
                 .sum().reset_index()[["InvestorID", "InvestorName"]]
                 .to_dict("records"))

    opening = ia[ia["TransactionDate"] < year_start].groupby("InvestorID")["Amount"].sum()
    period = ia[ia["TransactionDate"] >= year_start]

    rows = []
    rows.append({"label": "Members' Capital, January 1, %d" % p["year"], "kind": "opening",
                 "by_member": {m["InvestorID"]: float(opening.get(m["InvestorID"], 0.0))
                               for m in members}})
    present = [r for r in MC_ROWS + ["Other movement"]
               if not period.empty and r in set(period["row"])]
    for label in present:
        sub = period[period["row"] == label].groupby("InvestorID")["Amount"].sum()
        rows.append({"label": label, "kind": "movement",
                     "by_member": {m["InvestorID"]: float(sub.get(m["InvestorID"], 0.0))
                                   for m in members}})
    closing = ia.groupby("InvestorID")["Amount"].sum()
    rows.append({"label": "Members' Capital, %s" % p["period_end"], "kind": "closing",
                 "by_member": {m["InvestorID"]: float(closing.get(m["InvestorID"], 0.0))
                               for m in members}})
    for r in rows:
        r["total"] = sum(r["by_member"].values())

    # Control-account tie-out against the GL's equity section.
    gl_equity = None
    try:
        st = build(entityid, period_end, "balance_sheet", engine=engine)
        bs = st.get("balance_sheet") or {}
        sec = next((x for x in bs.get("sections", []) if x["section"] == "Members' Capital"), None)
        if sec:
            gl_equity = -sec["gl_total"]   # GL credit-negative -> positive capital
    except Exception:
        logger.warning("could not read GL equity for the tie-out", exc_info=True)

    subledger_total = rows[-1]["total"]
    return {
        "entity": entityid, "periods": p, "members": members, "rows": rows,
        "subledger_total": subledger_total,
        "gl_equity": gl_equity,
        "difference": None if gl_equity is None else subledger_total - gl_equity,
        "ties": None if gl_equity is None else abs(subledger_total - gl_equity) < 0.01,
    }


# ── Cash Flow ────────────────────────────────────────────────────────────
# Built on an identity, not on judgement. Every period's entries balance, so
# the change in cash is exactly the negative of the change in everything else:
#
#     Delta cash = -SUM(Delta of every non-cash account)
#
# Each non-cash account therefore contributes (minus its movement) to cash, and
# the only open question is WHICH ACTIVITY it is: operating, investing or
# financing. That makes the statement a classification rather than a
# reconstruction, and its total is arithmetically guaranteed to equal the
# movement in the cash accounts. Which is exactly why the tie-out is worth
# printing: if it fails, either the GL did not balance or an account went
# unclassified, and both are worth knowing before anybody signs.

CF_OPERATING, CF_INVESTING, CF_FINANCING = "Operating", "Investing", "Financing"


def _cf_category(section, mapped):
    """Where an account's movement belongs.

    The mapping may say. Otherwise equity is financing and everything else is
    operating. INVESTING IS NEVER GUESSED -- an account is investing only when
    somebody says so, and every account that fell to the default is listed on
    the statement so the gap is visible rather than implied.
    """
    if mapped in (CF_OPERATING, CF_INVESTING, CF_FINANCING):
        return mapped
    return CF_FINANCING if section == "Members' Capital" else CF_OPERATING


def build_cash_flow(entityid: str, period_end: str, bases: Optional[List[str]] = None,
                    engine=None) -> Dict[str, Any]:
    """Statement of Cash Flows, indirect method, from account movements."""
    engine = engine or get_engine()
    bases = bases if bases is not None else DEFAULT_BASES
    p = periods_for(period_end)
    bal = _balances(entityid, period_end, bases, engine)
    if bal.empty:
        return {"entity": entityid, "periods": p, "sections": [],
                "note": "No GL rows for this entity"}
    types = account_types(engine)
    mapping = _mapping(engine)

    buckets = {CF_OPERATING: {}, CF_INVESTING: {}, CF_FINANCING: {}}
    net_income = 0.0
    cash_movement = 0.0
    defaulted = []
    unclassified = []

    for _, r in bal.iterrows():
        acct, ytd = r["ACCTNUM"], float(r["ytd"])
        meta = types.get(acct)
        if not meta:
            unclassified.append({"acctnum": acct, "acctname": r["name"], "ytd": ytd})
            continue
        t = meta["type"]
        if t == "C":
            cash_movement += ytd
            continue
        if t == "I":
            # Income accounts roll into one line: the period's result.
            net_income += -ytd
            continue
        if t != "B":
            unclassified.append({"acctnum": acct, "acctname": r["name"], "ytd": ytd})
            continue

        m = mapping.get(acct) or {}
        section = (m.get("statement") or "").strip()
        explicit = (m.get("cf_category") or "").strip() or None
        cat = _cf_category(section, explicit)
        if not explicit:
            defaulted.append({"acctnum": acct, "acctname": r["name"],
                              "section": section or None, "category": cat, "ytd": ytd})
        line_name = m.get("fs_line") or r["name"]
        line = buckets[cat].setdefault(line_name, {"fs_line": line_name, "amount": 0.0,
                                                   "accounts": []})
        line["amount"] += -ytd          # the account's contribution to cash
        line["accounts"].append(acct)

    sections = []
    op_lines = [{"fs_line": "Net increase (decrease) in members' capital from operations",
                 "amount": net_income, "accounts": []}]
    op_lines += sorted(buckets[CF_OPERATING].values(), key=lambda l: l["fs_line"])
    for name, lines in ((CF_OPERATING, op_lines),
                        (CF_INVESTING, sorted(buckets[CF_INVESTING].values(),
                                              key=lambda l: l["fs_line"])),
                        (CF_FINANCING, sorted(buckets[CF_FINANCING].values(),
                                              key=lambda l: l["fs_line"]))):
        if not lines:
            continue
        sections.append({"section": "Cash flows from %s activities" % name.lower(),
                         "category": name, "lines": lines,
                         "total": sum(l["amount"] for l in lines)})

    computed = sum(s["total"] for s in sections)
    return {
        "entity": entityid, "periods": p, "bases": bases, "sections": sections,
        "net_change_computed": computed,
        "net_change_actual": cash_movement,
        "difference": computed - cash_movement,
        "ties": abs(computed - cash_movement) < 0.01,
        "defaulted_accounts": defaulted,
        "unclassified": unclassified,
    }


# ── Schedule of Investments ──────────────────────────────────────────────
# There is no special data path here, despite appearances. The example
# package's SOI makes three Spreadsheet Server calls and every one is an
# ordinary GL account balance -- MR14000001, MR14000002, MR14000003, LTD at
# the period, same entity and basis as its Trial Balance tab. It fetched them
# directly instead of referencing that tab: a spreadsheet convenience, not
# another source. Cost = Purchase + Return of Capital; Fair Value = Cost +
# Unrealized; % = Fair Value over members' capital.
#
# Two cells genuinely are not in the GL -- the investment's NAME and the
# MEMBERSHIP INTEREST -- and both are in `relationships` (MRI_IA_Relationship),
# which carries InvestmentID, InvestorID, OwnershipPct and Name.
#
# ROLES COME FROM MRI'S OWN ACCOUNT NAMES, not from hardcoded numbers. The
# chart contains exactly one investment family and MRI names it
# "Investment: <role>". Keying on the name rather than on MR14000001 means a
# renumbered chart still works, and ANY "Investment:" account whose role is
# not recognised is listed on the statement as unassigned rather than being
# swallowed into a total.
#
# SPLITTING BY INVESTMENT. An entity holding several investments has its GL
# lines tagged with RLTDENTITY, the related entity -- populated on 3,097 of
# 4,409 investment rows portfolio-wide. Lines that carry it are attributed;
# lines that do not are reported as unallocated rather than spread on a guess.

SOI_ROLES = {
    "investment: purchase": "cost",
    "investment: return of capital": "cost",
    "investment: unrealized gain/loss": "unrealized",
    "investment: realized gain/loss": "realized",
}


def _soi_role(acctname: str) -> Optional[str]:
    return SOI_ROLES.get((acctname or "").strip().lower())


def build_schedule_of_investments(entityid: str, period_end: str,
                                  bases: Optional[List[str]] = None,
                                  engine=None) -> Dict[str, Any]:
    """Schedule of Investments: cost, fair value and % of members' capital."""
    engine = engine or get_engine()
    bases = bases if bases is not None else DEFAULT_BASES
    p = periods_for(period_end)
    ent = entityid.strip().upper()
    types = account_types(engine)

    # What this entity holds, and how much of it.
    #
    # MEMBERSHIP INTEREST IS DERIVED FROM COMMITTED AMOUNTS, not read from a
    # stored percentage. Sep 14 2026: PPIECH's commitment to EASTCH is
    # 29,390,000 of 44,085,000 committed in total -- 66.67%, which is the
    # 0.6667 the example workbook's SOI carries. `relationships` says PPIECH
    # holds 100% of EASTCH and the operating partner 0%, and `commitments`
    # carries CapitalPercent 0.00 on both EASTCH rows. So on that investment
    # the amounts were right and BOTH percentage fields were wrong or unset,
    # while the amounts needed no maintenance to be correct.
    #
    # A stored percentage is right only once somebody fills it in and silently
    # wrong until then. Deriving costs nothing where the field IS populated:
    # the three commitments into PPIECH carry 15.31 / 16.64 / 68.05 and their
    # implied shares match to the cent.
    #
    # BOTH FIGURES ARE REPORTED. Accounting is mid-update on this table, so a
    # disagreement is news, not noise -- the statement shows the derived
    # interest, the relationships figure beside it, and flags the line when
    # they differ rather than quietly picking one.
    with engine.connect() as conn:
        try:
            rel = pd.read_sql(text("SELECT * FROM relationships"), conn)
        except Exception:
            logger.warning("relationships not loaded", exc_info=True)
            rel = pd.DataFrame()
        try:
            com = pd.read_sql(text("SELECT * FROM commitments"), conn)
        except Exception:
            logger.warning("commitments not loaded", exc_info=True)
            com = pd.DataFrame()

    rel_pct, rel_name = {}, {}
    if not rel.empty:
        for c in ("InvestmentID", "InvestorID"):
            rel[c] = rel[c].astype(str).str.strip().str.upper()
        held = rel[(rel["InvestorID"] == ent) & (rel["EndDate"].isna()
                   if "EndDate" in rel.columns else True)]
        for _, r in held.iterrows():
            rel_pct[r["InvestmentID"]] = (float(r["OwnershipPct"])
                                          if pd.notna(r.get("OwnershipPct")) else None)
            rel_name[r["InvestmentID"]] = str(r.get("Name") or "").strip() or None

    derived_pct, committed = {}, {}
    if not com.empty:
        com["EntityID"] = com["EntityID"].astype(str).str.strip().str.upper()
        com["InvestorID"] = com["InvestorID"].astype(str).str.strip().str.upper()
        com["Amount"] = pd.to_numeric(com["Amount"], errors="coerce").fillna(0.0)
        mine = com[com["InvestorID"] == ent]
        for _, r in mine.iterrows():
            inv_id = r["EntityID"]
            total = com[com["EntityID"] == inv_id]["Amount"].sum()
            derived_pct[inv_id] = (100.0 * r["Amount"] / total) if total else None
            committed[inv_id] = float(r["Amount"])

    holdings = []
    for inv_id in sorted(set(rel_pct) | set(derived_pct)):
        d, s = derived_pct.get(inv_id), rel_pct.get(inv_id)
        holdings.append({
            "investment_id": inv_id,
            "name": rel_name.get(inv_id) or inv_id,
            "ownership_pct": d if d is not None else s,
            "ownership_pct_source": "commitments" if d is not None else "relationships",
            "ownership_pct_derived": d,
            "ownership_pct_relationships": s,
            "committed_amount": committed.get(inv_id),
            "ownership_disagrees": (d is not None and s is not None
                                    and abs(d - s) > 0.01),
        })

    # The investment accounts' balances, by related entity where tagged.
    with engine.connect() as conn:
        gl = pd.read_sql(text(
            'SELECT "ACCTNUM", "ACCTNAME", "PERIOD", "BALFOR", "BASIS", '
            '"RLTDENTITY", "AMT" FROM gl_detail WHERE UPPER(TRIM("ENTITYID")) = :e'),
            conn, params={"e": ent})
    if gl.empty:
        return {"entity": entityid, "periods": p, "lines": [], "holdings": holdings,
                "note": "No GL rows for this entity"}
    for c in ("ACCTNUM", "ACCTNAME", "PERIOD", "BALFOR", "BASIS"):
        gl[c] = gl[c].astype(str).str.strip()
    gl["RLTDENTITY"] = gl["RLTDENTITY"].astype(str).str.strip().str.upper()
    gl.loc[gl["RLTDENTITY"].isin(("", "NONE", "NAN")), "RLTDENTITY"] = ""
    gl["AMT"] = pd.to_numeric(gl["AMT"], errors="coerce").fillna(0.0)
    if bases:
        gl = gl[gl["BASIS"].isin(bases)]

    gl["role"] = [_soi_role(n) for n in gl["ACCTNAME"]]
    inv = gl[gl["ACCTNAME"].str.lower().str.startswith("investment:")]
    unassigned = sorted({a for a, r in zip(inv["ACCTNUM"], inv["role"]) if r is None})

    # Balance at the period end = the year's opening plus its activity.
    inv = inv[(inv["PERIOD"] >= p["ytd_first"]) & (inv["PERIOD"] <= p["ytd_last"])]
    opening = inv[(inv["BALFOR"] == "B") & (inv["PERIOD"] == p["ytd_first"])]
    activity = inv[inv["BALFOR"] == "N"]
    bal = pd.concat([opening, activity])
    if bal.empty:
        return {"entity": entityid, "periods": p, "lines": [], "holdings": holdings,
                "unassigned_accounts": unassigned,
                "note": "No investment account balances for this entity"}

    by = bal.groupby(["RLTDENTITY", "role"])["AMT"].sum().unstack(fill_value=0.0)
    for col in ("cost", "unrealized", "realized"):
        if col not in by.columns:
            by[col] = 0.0

    members_capital = None
    try:
        mc = build_members_capital(entityid, period_end, engine=engine)
        members_capital = mc.get("subledger_total")
    except Exception:
        logger.warning("members' capital unavailable for the SOI percentage", exc_info=True)

    name_by_id = {h["investment_id"]: h for h in holdings}
    lines, unallocated = [], None
    for rltd, row in by.iterrows():
        cost = float(row["cost"])
        unreal = float(row["unrealized"])
        fv = cost + unreal
        h = name_by_id.get(rltd, {}) if rltd else {}
        entry = {
            "related_entity": rltd or None,
            "name": h.get("name"),
            "ownership_pct": h.get("ownership_pct"),
            "ownership_pct_source": h.get("ownership_pct_source"),
            "ownership_pct_derived": h.get("ownership_pct_derived"),
            "ownership_pct_relationships": h.get("ownership_pct_relationships"),
            "committed_amount": h.get("committed_amount"),
            "ownership_disagrees": h.get("ownership_disagrees", False),
            "cost": cost, "unrealized": unreal, "fair_value": fv,
            "realized": float(row["realized"]),
            "pct_of_members_capital": (fv / members_capital
                                       if members_capital else None),
        }
        if rltd:
            lines.append(entry)
        else:
            unallocated = entry

    # ONE INVESTMENT, NO TAG: attribute it rather than showing an anonymous
    # line. An entity holding exactly one thing has no ambiguity to resolve;
    # anything more and the untagged amount stays visibly unallocated.
    if unallocated and not lines and len(holdings) == 1:
        h = holdings[0]
        unallocated.update({"related_entity": h["investment_id"], "name": h["name"],
                            "ownership_pct": h["ownership_pct"],
                            "attributed_by": "sole holding, GL rows carry no RLTDENTITY"})
        lines, unallocated = [unallocated], None

    total_fv = sum(l["fair_value"] for l in lines) + (unallocated["fair_value"]
                                                      if unallocated else 0.0)
    gl_total = float(by["cost"].sum() + by["unrealized"].sum())
    return {
        "entity": entityid, "periods": p, "bases": bases,
        "holdings": holdings, "lines": lines,
        "unallocated": unallocated,
        "unassigned_accounts": unassigned,
        "members_capital": members_capital,
        "total_cost": sum(l["cost"] for l in lines) + (unallocated["cost"]
                                                       if unallocated else 0.0),
        "total_fair_value": total_fv,
        "gl_investment_balance": gl_total,
        "ties": abs(total_fv - gl_total) < 0.01,
        "difference": total_fv - gl_total,
    }


# ── Consolidated mapping ─────────────────────────────────────────────────
# seed_mapping_from_names() gives every account its own caption, which on the
# real chart produced 361 statement lines: a balance sheet with 163 asset
# lines is a trial balance with a title. A presentable statement needs the
# accounts grouped, and the grouping should speak the vocabulary the reviewers
# already use -- so the 56 lines accounting tags in the PPI Eastchase package
# are the target, not captions invented here.
#
# Two passes:
#   1. ACCOUNTING'S OWN TAG wins wherever it exists (192 accounts).
#   2. Everything else is routed INTO THE SAME 56 LINES by name. The keywords
#      below are drawn from those line names; nothing coins a new caption.
#
# WHAT WILL NOT ROUTE GOES TO "Other ...", VISIBLY. Every real statement has
# an Other assets / Other expenses / Other income line and accounting's own
# mapping uses all three -- 8, 17 and 6 accounts. An account landing there is
# not hidden: consolidated_mapping() reports how many did and which, so the
# CFO can pull the material ones out into lines of their own.

# keyword -> canonical line. Order matters: the first match wins, so the more
# specific phrases come first.
_ROUTES = [
    # assets
    ("restricted cash", "Restricted cash"),
    ("cash", "Cash and cash equivalents"),
    ("due from manager", "Due from Manager"),
    ("due from", "Due from affiliates"),
    ("intercompany", "Due from affiliates"),
    ("subscription receivable", "Subscription receivable"),
    ("management fee receivable", "Management fee receivable"),
    ("interest receivable", "Interest receivable"),
    ("income receivable", "Income receivable"),
    ("note receivable", "Note receivable"),
    ("deal cost receivable", "Deal cost receivable"),
    ("gst", "GST/HST receivable"),
    ("receivable", "Accounts receivable"),
    ("prepaid contribution", "Prepaid contribution"),
    ("prepaid", "Prepaid expenses"),
    ("investment: unrealized", "Investment unrealized gain/loss"),
    ("investment", "Investment cost"),
    # liabilities
    ("due to manager", "Due to manager"),
    ("due to", "Due to manager"),
    ("accrued", "Accrued expenses"),
    ("distribution payable", "Distribution payable"),
    ("interest payable", "Interest payable"),
    ("income tax payable", "Income tax payable"),
    ("tax payable", "Income tax payable"),
    ("note payable", "Note payable"),
    ("line of credit", "Line of credit"),
    ("payable", "Accounts payable"),
    # members' capital
    ("contribution", "Capital contributions"),
    ("distribution", "Capital distributions"),
    ("retained earnings", "Retained earnings"),
    ("equity pickup", "Retained earnings"),
    ("member", "Capital contributions"),
    # income
    ("management fee income", "Management fee income"),
    ("transaction fee", "Transaction fee income"),
    ("carried interest", "Carried interest income"),
    ("kicker", "Kicker income"),
    ("dividend", "Dividend income"),
    ("interest income", "Interest income"),
    ("investment income", "Investment income"),
    ("unrealized", "Unrealized Gain/Loss"),
    ("realized", "Realized Gain/Loss"),
    # expenses
    ("payroll", "Payroll and related expenses"),
    ("salary", "Payroll and related expenses"),
    ("professional", "Professional fees"),
    ("legal", "Professional fees"),
    ("audit", "Professional fees"),
    ("accounting fee", "Professional fees"),
    ("advisory", "Advisory fees"),
    ("monitoring", "Monitoring fees"),
    ("service fee", "Service fees"),
    ("financing fee", "Financing fees"),
    ("management fee", "Management fees"),
    ("organizational", "Organizational costs"),
    ("syndication", "Syndication costs"),
    ("amortization", "Amortization expense"),
    ("depreciation", "Depreciation expense"),
    ("insurance", "Insurance expense"),
    ("interest expense", "Interest expense"),
    ("acquisition", "Acquisition expense"),
    ("broken deal", "Broken deal expense"),
    ("tax", "Tax expense"),
]

_OTHER_BY_SECTION = {
    "Assets": "Other assets",
    "Liabilities": "Other assets",          # replaced below by type check
    "Members' Capital": "Capital contributions",
    "Income": "Other income",
    "Expenses": "Other expenses",
}


def _route(name: str) -> Optional[str]:
    low = (name or "").strip().lower()
    for kw, line in _ROUTES:
        if kw in low:
            return line
    return None


def consolidated_mapping(engine=None) -> List[dict]:
    """Every account mapped to one of accounting's own 56 statement lines.

    Returns the proposal; nothing is written. Each row says where its line
    came from -- `accounting` for the 192 tagged in the example package,
    `routed` for a name match into the same vocabulary, `other` for the ones
    that fell to an Other line and are worth a second look.
    """
    from flask_app.services import fs_line_seed as seed

    engine = engine or get_engine()
    types = account_types(engine)
    out: List[dict] = []
    for acct, meta in sorted(types.items()):
        t = meta["type"]
        if t not in TYPE_STATEMENT:
            continue          # L and M are roll-up headers, not statement lines
        name = meta["name"]

        line = seed.ACCOUNT_LINE.get(acct)
        origin = "accounting" if line else None
        if not line:
            line = _route(name)
            origin = "routed" if line else None

        if line:
            section = seed.LINE_SECTION.get(line)
        else:
            section = None

        # A routed line has to belong to the statement MRI says the account is
        # on. A "Management fees" expense caption on a balance-sheet account
        # would be worse than no caption at all.
        stmt = TYPE_STATEMENT[t]
        valid = (BALANCE_SHEET_SECTIONS if stmt == "balance_sheet" else INCOME_SECTIONS)
        if section not in valid:
            section, line, origin = None, None, None

        if not line:
            # Fall back to the Other line for the statement this account is on.
            if stmt == "income_statement":
                low = name.lower()
                section = ("Expenses" if any(w in low for w in
                                             ("expense", "fee", "cost", "tax"))
                           else "Income")
                line = "Other expenses" if section == "Expenses" else "Other income"
            else:
                low = name.lower()
                if any(w in low for w in ("payable", "accrued", "due to", "liabilit")):
                    section, line = "Liabilities", "Accrued expenses"
                elif any(w in low for w in ("capital", "member", "equity",
                                            "retained", "distribution")):
                    section, line = "Members' Capital", "Capital contributions"
                else:
                    section, line = "Assets", "Other assets"
            origin = "other"

        out.append({"acctnum": acct, "acctname": name, "gacc_type": t,
                    "statement": section, "fs_line": line, "origin": origin})
    return out
