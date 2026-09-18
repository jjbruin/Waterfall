"""The two portfolio summary tabs asset management keeps in Excel.

Jack Day: "The attached file has two tabs we'd like to recreate in the app. For the
current review period it'd be 2025 vs 2026 in all comparisons."

    2025_Val_Summary_1   pref balance, accrual, and the pref NAV against last year
    2025_Val_Summary_2   method, rates, value, debt and net proceeds against last year

NOTHING HERE IS CALCULATED. Every figure already exists, computed by an engine that has
been vetted:

    valuation_records        method, cap rate, exit cap, discount rate, direct cap NOI,
                             concluded value  (what the analyst concluded)
    valuation_nav_results    net proceeds, PSC NAV, OP NAV, and the pref figures

THE PREF BALANCES AND ACCRUALS ARE THE PREF BALANCE DETAIL REPORT'S OWN NUMBERS. The
chain is: `reports_service.build_pref_balance_detail` -> `valuation_nav_service._pref_walks`
-> stored in `valuation_nav_results.inputs_json` when the NAV is computed -> read here.
Nothing along that chain recalculates a pref balance, and neither does this.

Where a NAV has been computed, the stored figure is used, so the summary agrees with
the package that was published. Where it has not, the SAME engine is run at the cycle's
own as-of date. That is not a second answer: the pref walk is a function of the deal,
the investor and the cut-off date, so running it at the cycle date reproduces what the
NAV run would have stored (Jim, Sep 18 2026: "only difference should be the date the
analysis cuts off"). Every row says which of the two it is.

Without that fallback the tab is blank for any deal whose NAV has not been run yet --
83 of 84 records on the current cycle -- which is the whole report missing.

Both tabs lay two cycles side by side and nothing more. The restraint is the point:
a summary that carried its OWN pref arithmetic would be a second answer to a question
the app has already answered, and the two would eventually disagree. Asked before and
answered: "why are you trying to recreate a calculation engine that we have already
built and vetted?"

A NAV that has not been run still reports as missing. It never reads as zero -- a zero
NAV and an uncomputed one look identical on a summary page and only one is a fact.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _cycles(engine, cycle_id: int) -> Dict[str, Any]:
    """This cycle and the one for the year before it, if there is one."""
    with engine.connect() as conn:
        cur = conn.execute(text(
            "SELECT id, year, as_of_date FROM valuation_cycles WHERE id = :c"
        ), {"c": cycle_id}).fetchone()
        if not cur:
            raise ValueError(f"Valuation cycle {cycle_id} not found")
        prior = conn.execute(text(
            "SELECT id, year, as_of_date FROM valuation_cycles WHERE year = :y"
        ), {"y": int(cur[1]) - 1}).fetchone()
    return {
        "current": {"id": cur[0], "year": cur[1], "as_of": cur[2]},
        "prior": ({"id": prior[0], "year": prior[1], "as_of": prior[2]}
                  if prior else None),
    }


def _records(engine, cycle_id: int, data: Optional[dict] = None,
             as_of=None) -> Dict[str, Dict[str, Any]]:
    """Every record in a cycle, with its NAV result where one has been computed.

    Where one has not, the pref walk is run at `as_of` through the same engine, so the
    tab is populated rather than blank.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT r.id, r.vcode, r.method, r.concluded_value, r.cap_rate,
                   r.term_cap_rate, r.discount_rate, r.direct_cap_noi,
                   r.classification, r.status,
                   n.net_proceeds, n.psc_nav, n.op_nav, n.inputs_json, n.computed_at
              FROM valuation_records r
              LEFT JOIN valuation_nav_results n ON n.record_id = r.id
             WHERE r.cycle_id = :c
        """), {"c": cycle_id}).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pref_balance = pref_accrued = None
        op_balance = op_accrued = None
        pref_source = None
        debt = None
        if r[13]:
            try:
                inp = json.loads(r[13])
                debt = inp.get("debt")
                split = _split_pref(inp.get("pref") or {})
                pref_balance = split["pref_balance"]
                pref_accrued = split["pref_accrued"]
                op_balance, op_accrued = split["op_balance"], split["op_accrued"]
                if pref_balance is not None or pref_accrued is not None:
                    # Say where it came from, so nobody has to trust it blindly.
                    pref_source = "Pref Balance Detail, via the NAV run"
            except (ValueError, AttributeError) as e:
                logger.info("Unreadable NAV inputs for record %s: %s", r[0], e)
        pref_note = None
        if pref_balance is None and pref_accrued is None and data is not None and as_of:
            live = _live_pref(data, str(r[1]), as_of)
            pref_balance = live["pref_balance"]
            pref_accrued = live["pref_accrued"]
            op_balance, op_accrued = live.get("op_balance"), live.get("op_accrued")
            pref_source = live["pref_source"]
            pref_note = live["pref_note"]
        out[str(r[1])] = {
            "record_id": r[0], "vcode": str(r[1]), "method": r[2],
            "concluded_value": r[3], "cap_rate": r[4], "term_cap_rate": r[5],
            "discount_rate": r[6], "direct_cap_noi": r[7],
            "classification": r[8], "status": r[9],
            "net_proceeds": r[10], "psc_nav": r[11], "op_nav": r[12],
            "pref_balance": pref_balance, "pref_accrued": pref_accrued,
            "pref_source": pref_source, "pref_note": pref_note,
            "op_balance": op_balance, "op_accrued": op_accrued, "debt": debt,
            "nav_computed": r[14] is not None,
        }
    return out


def _delta(cur: Optional[float], prior: Optional[float]) -> Optional[float]:
    """The year-on-year move, or None when either side is not a number.

    None, not zero. "No change" and "we have not computed one side" are different
    answers and a summary that shows 0 for both is lying about one of them.
    """
    if cur is None or prior is None:
        return None
    return float(cur) - float(prior)


def _names(data: dict) -> Dict[str, Dict[str, Any]]:
    """Deal name and portfolio grouping, from the deals table the app already loads."""
    out: Dict[str, Dict[str, Any]] = {}
    inv = (data or {}).get("inv")
    if inv is None or not len(inv):
        return out
    cols = {c.lower(): c for c in inv.columns}
    vc = cols.get("vcode")
    if not vc:
        return out
    name_col = cols.get("deal_name") or cols.get("property_name") or cols.get("name")
    port_col = cols.get("portfolio_name")
    inv_col = cols.get("investmentid")
    for _, row in inv.iterrows():
        v = str(row[vc]).strip()
        out[v] = {
            "name": str(row[name_col]).strip() if name_col else v,
            "portfolio": (str(row[port_col]).strip()
                          if port_col and row.get(port_col) is not None else None),
            "investment_id": (str(row[inv_col]).strip()
                              if inv_col and row.get(inv_col) is not None else None),
        }
    return out


def _as_date(value) -> Optional[date]:
    """Coerce a cycle's as_of to a real date.

    The cycles table stores it as TEXT. Passing that string straight into the pref walk
    does not raise -- every transaction simply fails the date comparison and the walk
    returns a balance of 0.00, which reads as a real answer. Seven of the eight deals
    checked against the Excel came back zero for exactly this reason.
    """
    if value is None or isinstance(value, date):
        return value if isinstance(value, date) else None
    if isinstance(value, datetime):
        return value.date()
    try:
        return pd.to_datetime(str(value)).date()
    except Exception:
        logger.warning("Unusable cycle as_of date: %r", value)
        return None


def _is_psc_side(investor_code: str) -> bool:
    """The preferred side of the walk, as the NAV engine itself splits it.

    `valuation_nav_service._pref_walks` treats an investor code starting "OP" as the
    operating partner and everything else as PSC. The summary's "Pref Balance" column
    is the PREFERRED position -- summing both sides doubles it, and the doubling is
    invisible because it still looks like a plausible balance.
    """
    return not str(investor_code or "").upper().startswith("OP")


def _split_pref(walks_or_summaries) -> Dict[str, Any]:
    """PSC-side balance and accrual, with the OP side kept separately."""
    psc_bal = psc_acc = op_bal = op_acc = None
    for code, w in (walks_or_summaries or {}).items():
        h = (w.get("header") if isinstance(w, dict) and "header" in w else w) or {}
        bal, acc = h.get("investment_balance"), h.get("accrued_pref")
        if _is_psc_side(code):
            psc_bal = (psc_bal or 0) + bal if bal is not None else psc_bal
            psc_acc = (psc_acc or 0) + acc if acc is not None else psc_acc
        else:
            op_bal = (op_bal or 0) + bal if bal is not None else op_bal
            op_acc = (op_acc or 0) + acc if acc is not None else op_acc
    return {"pref_balance": psc_bal, "pref_accrued": psc_acc,
            "op_balance": op_bal, "op_accrued": op_acc}


def _live_pref(data: dict, vcode: str, as_of) -> Dict[str, Any]:
    """Run the pref walk at a given cut-off, through the engine the NAV run uses.

    `valuation_nav_service._pref_walks` calls `reports_service.build_pref_balance_detail`
    -- the vetted engine behind the Pref Balance Detail report. Calling it here is the
    same code path, not a parallel one.
    """
    from flask_app.services import valuation_nav_service as nav

    as_of = _as_date(as_of)
    if as_of is None:
        return {"pref_balance": None, "pref_accrued": None, "op_balance": None,
                "op_accrued": None, "pref_source": None,
                "pref_note": "cycle has no usable as-of date"}
    try:
        steps = nav._cap_wf_steps(data, vcode)
        if steps is None or steps.empty:
            return {"pref_balance": None, "pref_accrued": None,
                    "pref_source": None, "pref_note": "no Cap_WF waterfall configured"}
        walks = nav._pref_walks(None, data, vcode, as_of, steps)
    except Exception as e:
        logger.info("Live pref walk failed for %s: %s", vcode, e)
        return {"pref_balance": None, "pref_accrued": None,
                "pref_source": None, "pref_note": f"pref walk unavailable: {e}"}

    split = _split_pref(walks)
    if split["pref_balance"] is None and split["pref_accrued"] is None:
        return {**split, "pref_source": None,
                "pref_note": "no PSC pref steps on this deal"}
    return {**split, "pref_source": "Pref Balance Detail, run at the cycle date",
            "pref_note": None}


def _assemble(engine, cycle_id: int, data: dict) -> Dict[str, Any]:
    cyc = _cycles(engine, cycle_id)
    cur = _records(engine, cycle_id, data, cyc["current"]["as_of"])
    prior = (_records(engine, cyc["prior"]["id"], data, cyc["prior"]["as_of"])
             if cyc["prior"] else {})
    names = _names(data)
    return {"cycles": cyc, "current": cur, "prior": prior, "names": names}


def pref_summary(engine, cycle_id: int, data: dict) -> Dict[str, Any]:
    """Tab 1: pref balance, accrual, and the pref NAV against last year.

    "Pref NAV" is the PSC NAV the walk produced -- PSC holds the preferred equity, so
    the preferred position's NAV is what that walk allocates to it.
    """
    a = _assemble(engine, cycle_id, data)
    cy, py = a["cycles"]["current"], a["cycles"]["prior"]

    rows: List[Dict[str, Any]] = []
    for vcode, rec in sorted(a["current"].items()):
        p = a["prior"].get(vcode, {})
        meta = a["names"].get(vcode, {})
        bal, acc = rec["pref_balance"], rec["pref_accrued"]
        rows.append({
            "vcode": vcode,
            "investment_id": meta.get("investment_id"),
            "name": meta.get("name") or vcode,
            "portfolio": meta.get("portfolio"),
            "pref_balance": bal,
            "pref_accrued": acc,
            "pref_with_accrual": (None if bal is None and acc is None
                                  else (bal or 0) + (acc or 0)),
            "pref_nav": rec["psc_nav"],
            "prior_pref_nav": p.get("psc_nav"),
            "var_to_prior": _delta(rec["psc_nav"], p.get("psc_nav")),
            "pref_source": rec["pref_source"], "pref_note": rec["pref_note"],
            "nav_computed": rec["nav_computed"],
            "prior_nav_computed": bool(p.get("nav_computed")),
        })
    return {
        "tab": "pref_summary",
        "title": f"PSC — Property Valuation Analysis (as of {cy['as_of']})",
        "current_year": cy["year"],
        "prior_year": py["year"] if py else None,
        "rows": rows,
        "missing_nav": [r["vcode"] for r in rows if not r["nav_computed"]],
        "no_prior_cycle": py is None,
    }


def valuation_summary(engine, cycle_id: int, data: dict) -> Dict[str, Any]:
    """Tab 2: method, rates, value, debt and net proceeds against last year."""
    a = _assemble(engine, cycle_id, data)
    cy, py = a["cycles"]["current"], a["cycles"]["prior"]

    rows: List[Dict[str, Any]] = []
    for vcode, rec in sorted(a["current"].items()):
        p = a["prior"].get(vcode, {})
        meta = a["names"].get(vcode, {})
        rows.append({
            "vcode": vcode,
            "investment_id": meta.get("investment_id"),
            "name": meta.get("name") or vcode,
            "portfolio": meta.get("portfolio"),
            "prior_method": p.get("method"), "method": rec["method"],
            "prior_cap_rate": p.get("cap_rate"), "cap_rate": rec["cap_rate"],
            "prior_exit_cap": p.get("term_cap_rate"), "exit_cap": rec["term_cap_rate"],
            "prior_discount": p.get("discount_rate"), "discount": rec["discount_rate"],
            "prior_direct_cap_noi": p.get("direct_cap_noi"),
            "direct_cap_noi": rec["direct_cap_noi"],
            "prior_value": p.get("concluded_value"), "value": rec["concluded_value"],
            "var_to_prior_value": _delta(rec["concluded_value"],
                                         p.get("concluded_value")),
            "prior_debt": p.get("debt"), "debt": rec["debt"],
            "prior_net_proceeds": p.get("net_proceeds"),
            "net_proceeds": rec["net_proceeds"],
            "var_to_prior_proceeds": _delta(rec["net_proceeds"],
                                            p.get("net_proceeds")),
            "nav_computed": rec["nav_computed"],
        })
    return {
        "tab": "valuation_summary",
        "title": f"PSC — Property Valuation Analysis (as of {cy['as_of']})",
        "current_year": cy["year"],
        "prior_year": py["year"] if py else None,
        "rows": rows,
        "missing_nav": [r["vcode"] for r in rows if not r["nav_computed"]],
        "no_prior_cycle": py is None,
    }
