"""What an accountant needs in front of them to finish one step.

The checklist told a preparer WHAT to do and then left them to go and find the
numbers somewhere else -- which is how a step gets ticked because it was
worked last quarter rather than because it was checked this one. This module
answers, for a given step: what am I asserting, which figures prove it, and
what does the system already know is wrong?

ONE DEFINITION, SERVER SIDE. The step -> evidence mapping lives here rather
than in the view, because it is accounting knowledge, not layout: it decides
what "cash reconciled" means in terms of accounts and tie-outs. A second copy
in TypeScript would drift from this one the first time a step changed.

EVERY STEP CARRIES ITS CHECKS. A check is a named assertion with a pass/fail
the system can evaluate -- the balance sheet nets to zero, the subledger ties
to the control account, the cash-flow identity holds. They are shown whether
they pass or fail, because a preparer needs to see that a check RAN, and a
green check nobody can point to is worth as much as no check at all.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from flask_app.db import get_engine
from flask_app.services import statement_service as ss
from flask_app.services import workpaper_data as wd
from flask_app.services import workpaper_service as ws

logger = logging.getLogger(__name__)

MAX_ROWS = 300


def _check(label: str, ok: Optional[bool], value: str, detail: str = "") -> dict:
    return {"label": label,
            "status": "unknown" if ok is None else ("pass" if ok else "fail"),
            "value": value, "detail": detail}


def _money(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _table(title: str, columns: List[str], rows: List[dict], note: str = "") -> dict:
    return {"title": title, "columns": columns, "rows": rows[:MAX_ROWS],
            "row_count": len(rows), "truncated": len(rows) > MAX_ROWS, "note": note}


# ── Per-step evidence ────────────────────────────────────────────────────

def _accounts_of_type(entity, period_end, engine, types_wanted, name_filter=None):
    """Account balances for one family, for the reconciliation steps."""
    tb = wd.trial_balance(entity, period_end, engine=engine)
    meta = ss.account_types(engine)
    out = []
    for r in tb["rows"]:
        t = (meta.get(r["acctnum"], {}) or {}).get("type")
        if types_wanted and t not in types_wanted:
            continue
        if name_filter and name_filter.lower() not in (r["acctname"] or "").lower():
            continue
        out.append({**r, "type": t})
    return out


def step_evidence(package_id: int, step_key: str, engine=None) -> Dict[str, Any]:
    """Guidance, checks and tables for one step of one package."""
    engine = engine or get_engine()
    detail = ws.package_detail(package_id, engine)
    pkg = detail["package"]
    entity, period_end = pkg["entityid"], pkg["period_end"]
    step = next((s for s in detail["steps"] if s["key"] == step_key), None)
    if step is None:
        raise ValueError(f"Unknown step '{step_key}'")

    out: Dict[str, Any] = {"step": step, "entity": entity, "period_end": period_end,
                           "guidance": "", "checks": [], "tables": [],
                           "exhibit_slots": [], "statements": []}

    if step_key == "tb_load":
        tb = wd.trial_balance(entity, period_end, engine=engine)
        st = ss.build(entity, period_end, "balance_sheet", engine=engine)
        bs = st.get("balance_sheet") or {}
        out["guidance"] = ("Confirm the trial balance is complete for the period and that "
                           "every account carries a statement line. The balance sheet nets "
                           "to zero in GL signs when nothing is missing.")
        out["checks"] = [
            _check("Balance sheet nets to zero", bs.get("balanced"),
                   _money(bs.get("out_of_balance")),
                   "Whatever is left is unmapped, untyped or misclassified"),
            _check("Every account mapped to a statement line",
                   len(st.get("unmapped", [])) == 0,
                   f"{len(st.get('unmapped', []))} unmapped",
                   _money(st.get("unmapped_total")) + " not on any line"),
            _check("Every account carries a GACC type",
                   len(st.get("untyped", [])) == 0,
                   f"{len(st.get('untyped', []))} untyped"),
        ]
        out["tables"] = [
            _table("Trial balance", ["acctnum", "acctname", "fs_line",
                                     "ytd_beginning", "ytd_change", "ytd_ending"],
                   tb["rows"], f"Periods {tb['periods']['ytd_first']}–{tb['periods']['ytd_last']}"),
        ]
        if st.get("unmapped"):
            out["tables"].append(_table("Unmapped accounts",
                                        ["acctnum", "acctname", "closing"],
                                        st["unmapped"],
                                        "These are absent from every statement"))

    elif step_key == "gl_review":
        rows = wd.gl_detail(entity, period_end, engine=engine)
        summ = wd.account_summary(entity, period_end, engine=engine)
        out["guidance"] = ("Review the period's entries. The summary is the same data "
                           "grouped by account — the shape each pivot tab takes in the "
                           "legacy workbook.")
        out["checks"] = [_check("Entries present for the period", len(rows) > 0,
                                f"{len(rows)} rows")]
        out["tables"] = [
            _table("Activity by account", ["ACCTNUM", "ACCTNAME", "entries", "amount"], summ),
            _table("GL detail", ["PERIOD", "ENTRDATE", "ACCTNUM", "ACCTNAME",
                                 "DESCRPN", "RLTDENTITY", "AMT"], rows),
        ]

    elif step_key == "cash_rec":
        cash = _accounts_of_type(entity, period_end, engine, {"C"})
        cf = ss.build_cash_flow(entity, period_end, engine=engine)
        out["guidance"] = ("Agree the cash accounts to the bank. Attach the bank report "
                           "as the YTD Cash Support exhibit — the movement below is what "
                           "it has to support.")
        out["checks"] = [
            _check("Cash flow ties to the cash accounts", cf.get("ties"),
                   _money(cf.get("net_change_actual")),
                   f"computed {_money(cf.get('net_change_computed'))}"),
            _check("Cash accounts present", len(cash) > 0, f"{len(cash)} accounts"),
        ]
        out["tables"] = [_table("Cash accounts",
                                ["acctnum", "acctname", "ytd_beginning",
                                 "ytd_change", "ytd_ending"], cash)]
        out["exhibit_slots"] = ["cash_support"]

    elif step_key == "intercompany":
        rows = [r for r in _accounts_of_type(entity, period_end, engine, {"B"})
                if any(w in (r["acctname"] or "").lower()
                       for w in ("due to", "due from", "intercompany", "receivable"))]
        out["guidance"] = ("Agree intercompany and receivable balances to the counterparty. "
                           "The related-entity tag on each GL line says who the other side is.")
        out["checks"] = [_check("Intercompany accounts with a balance",
                                None, f"{len(rows)} accounts",
                                "Confirm each against the counterparty")]
        out["tables"] = [_table("Intercompany and receivables",
                                ["acctnum", "acctname", "ytd_beginning",
                                 "ytd_change", "ytd_ending"], rows)]

    elif step_key == "accruals":
        rows = [r for r in _accounts_of_type(entity, period_end, engine, {"B"})
                if any(w in (r["acctname"] or "").lower()
                       for w in ("accrued", "payable", "management fee"))]
        out["guidance"] = ("Confirm each accrual is still required and correctly stated, "
                           "and that the management fee agrees to the agreement.")
        out["checks"] = [_check("Accrual accounts with a balance", None,
                                f"{len(rows)} accounts")]
        out["tables"] = [_table("Accruals and fees",
                                ["acctnum", "acctname", "ytd_beginning",
                                 "ytd_change", "ytd_ending"], rows)]

    elif step_key == "investments":
        soi = ss.build_schedule_of_investments(entity, period_end, engine=engine)
        out["guidance"] = ("Confirm cost, fair value and membership interest per "
                           "investment. Attach the valuation as the Valuation Support "
                           "exhibit — fair value is cost plus unrealised, so a change in "
                           "value must be posted before it appears here.")
        out["checks"] = [
            _check("Schedule ties to the GL investment accounts", soi.get("ties"),
                   _money(soi.get("total_fair_value")),
                   f"GL {_money(soi.get('gl_investment_balance'))}"),
            _check("All investment accounts recognised",
                   not soi.get("unassigned_accounts"),
                   ", ".join(soi.get("unassigned_accounts") or []) or "none unassigned"),
            _check("Membership interest agrees with relationships",
                   not any(l.get("ownership_disagrees") for l in soi.get("lines", [])),
                   "derived from committed amounts",
                   "Accounting is updating the commitments table"),
        ]
        out["tables"] = [_table("Schedule of investments",
                                ["name", "ownership_pct", "ownership_pct_relationships",
                                 "committed_amount", "cost", "unrealized", "fair_value"],
                                soi.get("lines", []))]
        out["exhibit_slots"] = ["valuation_support"]
        out["statements"] = ["soi"]

    elif step_key == "capital_activity":
        mc = ss.build_members_capital(entity, period_end, engine=engine)
        rf = wd.ia_rollforward(entity, period_end, engine=engine)
        com = wd.commitment_rollforward(entity, engine=engine)
        out["guidance"] = ("Agree each member's movement to the subledger, and the "
                           "members' total to the GL equity account.")
        out["checks"] = [
            _check("Subledger ties to GL members' capital", mc.get("ties"),
                   _money(mc.get("subledger_total")),
                   f"GL {_money(mc.get('gl_equity'))}"),
            _check("Members with activity", None, f"{len(mc.get('members', []))} members"),
        ]
        out["tables"] = [
            _table("Members' capital movement",
                   ["label"] + [m["InvestorID"] for m in mc.get("members", [])] + ["total"],
                   [{"label": r["label"], "total": r["total"],
                     **{m["InvestorID"]: r["by_member"].get(m["InvestorID"])
                        for m in mc.get("members", [])}} for r in mc.get("rows", [])]),
            _table("IA rollforward", rf.get("columns", []), rf.get("rows", [])),
            _table("Commitments", list(com[0].keys()) if com else [], com),
        ]
        out["statements"] = ["members_capital"]

    elif step_key == "exhibits":
        out["guidance"] = ("Attach every supporting document the package needs. Each "
                           "lands in a named place in the download: spreadsheets and "
                           "CSVs become tabs, images are embedded, anything else is "
                           "listed on the Exhibits tab and travels with the file.")
        have = {e["slot_key"] for e in detail["exhibits"]}
        out["checks"] = [_check("Exhibits attached", len(have) > 0,
                                f"{len(detail['exhibits'])} files across "
                                f"{len(have)} slots")]
        out["exhibit_slots"] = [s["key"] for s in ws.EXHIBIT_SLOTS]

    elif step_key in ("fs_draft", "preparer_signoff", "manager_review", "cfo_approval"):
        st = ss.build(entity, period_end, "both", engine=engine)
        bs = st.get("balance_sheet") or {}
        inc = st.get("income_statement") or {}
        mc = ss.build_members_capital(entity, period_end, engine=engine)
        cf = ss.build_cash_flow(entity, period_end, engine=engine)
        soi = ss.build_schedule_of_investments(entity, period_end, engine=engine)
        out["guidance"] = ("Every statement the package will contain, with the checks "
                           "the system can make. These are the same figures the "
                           "downloaded workbook carries.")
        out["checks"] = [
            _check("Balance sheet nets to zero", bs.get("balanced"),
                   _money(bs.get("out_of_balance"))),
            _check("Members' capital ties to the GL", mc.get("ties"),
                   _money(mc.get("subledger_total")),
                   f"GL {_money(mc.get('gl_equity'))}"),
            _check("Cash flow ties to the cash accounts", cf.get("ties"),
                   _money(cf.get("net_change_computed")),
                   f"actual {_money(cf.get('net_change_actual'))}"),
            _check("Schedule of investments ties to the GL", soi.get("ties"),
                   _money(soi.get("total_fair_value"))),
            _check("No unmapped accounts", len(st.get("unmapped", [])) == 0,
                   f"{len(st.get('unmapped', []))} unmapped",
                   _money(st.get("unmapped_total"))),
            _check("Net income", None, _money(inc.get("net_income"))),
        ]
        out["statements"] = ["balance_sheet", "income_statement", "soi",
                             "members_capital", "cash_flow"]
        if step_key in ("manager_review", "cfo_approval"):
            out["guidance"] = ("Review what the preparer has asserted. Every check below "
                               "is the system's own; a failing one is a reason to return "
                               "the package rather than approve it.")

    return out


def statements_summary(package_id: int, engine=None) -> Dict[str, Any]:
    """Every drafted statement for a package, with its tie-out.

    The point of putting this at the top of the screen is that the statements
    are the thing being validated -- the checklist is how you get there, not
    what you are producing.
    """
    engine = engine or get_engine()
    pkg = ws.package_detail(package_id, engine)["package"]
    entity, period_end = pkg["entityid"], pkg["period_end"]

    st = ss.build(entity, period_end, "both", engine=engine)
    mc = ss.build_members_capital(entity, period_end, engine=engine)
    cf = ss.build_cash_flow(entity, period_end, engine=engine)
    soi = ss.build_schedule_of_investments(entity, period_end, engine=engine)
    bs = st.get("balance_sheet") or {}
    inc = st.get("income_statement") or {}

    return {
        "entity": entity, "period_end": period_end,
        "balance_sheet": {
            "title": "Balance Sheet", "sections": bs.get("sections", []),
            "ties": bs.get("balanced"),
            "tie_label": "Nets to zero" if bs.get("balanced")
                         else f"Out of balance by {_money(bs.get('out_of_balance'))}",
        },
        "income_statement": {
            "title": "Income Statement", "sections": inc.get("sections", []),
            "ties": None, "tie_label": f"Net income {_money(inc.get('net_income'))}",
            "net_income": inc.get("net_income"),
        },
        "soi": {
            "title": "Schedule of Investments", "lines": soi.get("lines", []),
            "ties": soi.get("ties"),
            "tie_label": "Ties to the GL" if soi.get("ties")
                         else f"Differs from the GL by {_money(soi.get('difference'))}",
            "total_cost": soi.get("total_cost"),
            "total_fair_value": soi.get("total_fair_value"),
        },
        "members_capital": {
            "title": "Members' Capital", "rows": mc.get("rows", []),
            "members": mc.get("members", []), "ties": mc.get("ties"),
            "tie_label": "Ties to the GL" if mc.get("ties")
                         else f"Differs from the GL by {_money(mc.get('difference'))}",
        },
        "cash_flow": {
            "title": "Cash Flow", "sections": cf.get("sections", []),
            "ties": cf.get("ties"),
            "tie_label": "Ties to the cash accounts" if cf.get("ties")
                         else f"Differs by {_money(cf.get('difference'))}",
            "net_change": cf.get("net_change_computed"),
        },
        "unmapped_count": len(st.get("unmapped", [])),
        "unmapped_total": st.get("unmapped_total"),
    }
