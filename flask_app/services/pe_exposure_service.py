"""PSC Preferred Equity exposure -- accounting's tracker, from our MRI copy.

Replaces ``PSC Preferred Equity Tracker - <date>.xlsx``, which accounting builds
from Spreadsheet Server (``GEXD("IA Query.edq", ...)``) with the ownership
percentages typed by hand. Jim, Oct 5 2026: "replace the spreadsheet server
links with references to our MRI accounting database ... calculate these
percentages using our commitments table and the ratios embedded flowing through
the ownership chain ... rely upon engines we have already created and tested."

ALMOST NOTHING HERE IS A NEW CALCULATION. Each figure comes from the engine that
already owns it, so this report cannot disagree with the screen beside it:

  | Figure                      | Engine                                                  |
  |-----------------------------|---------------------------------------------------------|
  | holding entity per deal     | committed_pref.deal_commitment_rows / row_in_effect      |
  | Cost                        | reports_service.build_pref_balance_detail (header        |
  |                             |   investment_balance -- the Pref Balance Detail figure), |
  |                             |   less realized losses from ia_transactions              |
  | Unrealized gain/loss        | ia_transactions, accounting's own IA query (NEW: no      |
  |                             |   engine owned it; ``unrealized_by_holding`` is it now)  |
  | FMV                         | Cost + Unrealized -- the tracker's own definition         |
  | Investor split              | ownership_chain_service.group_shares (commitments as of  |
  |                             |   the quarter, multiplied through the chain)             |
  | Future funding              | one_pager.get_pe_performance remaining_to_fund (the      |
  |                             |   One Pager's figure)                                    |
  | CAD -> USD                  | market_rates_service.rate_on("USDCAD", quarter end)      |

MEASURED AGAINST ACCOUNTING'S 26Q2 TRACKER on production data (Oct 5 2026):
Cost 56 of 57 holdings to the dollar (Nottingham: the tracker's Cost view omits
the $2,923,426.79 June contribution its own FMV view includes); FMV 57 of 57;
the seven investor percentages 52 of 54 (Brainerd, where the tracker typed a
funded-to-date split while its own side note gives the commitment split we
compute, and Bel Air, 0.4 points between two typed constants); Future Funding 7
of the tracker's 9 (Pontchartrain and Middle Island differ in MRI itself).

Cost is the tracker's definition -- contributions, return of capital, and
realized losses -- and the Pref Balance Detail engine reproduces it on every
holding. FMV adds the "Unrealized Gain/Loss" marks, which exist only in
``ia_transactions``: the app's ``accounting`` feed keeps no non-cash rows.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

#: The tracker's seven investor columns, in its order.
GROUPS = ("PSC", "KOC", "TIAA", "Clarion", "F&F", "Declaration", "Ambassadors")

#: Where the walk stops: an entity that is a holder in its own right. Taken from
#: accounting's classification (the tracker's Mapping tab: PSC, KOC Investor,
#: TIAA Investor, OW) and from the tracker's columns -- Declaration and Clarion
#: are third-party investors with columns of their own. PSC I is a stop: its own
#: members are PSC's business, not this report's.
STOPS: Dict[str, str] = {
    "PSC1": "PSC", "PSC2": "PSC", "OWPSC": "PSC", "PSCMAN": "PSC",
    "PSL1": "PSC", "PSS1": "PSC",
    "KCREIT": "KOC",
    "TGAM": "TIAA",
    "DCXVIA": "Declaration", "DCXVIB": "Declaration",
    "DIFPP": "Clarion",
}

#: Every other end investor is F&F -- the tracker's own treatment, Bruin Real
#: Estate and RBS262 included -- EXCEPT under an Ambassadors fund, where the
#: outside investors are the Ambassadors column and PSC I's stake is PSC.
DEFAULT_GROUP = "F&F"


def _context(entity: str) -> Optional[str]:
    return "Ambassadors" if entity.startswith("AMB") else None


def quarter_end(d: date) -> date:
    q_month = ((d.month - 1) // 3) * 3 + 3
    nxt = date(d.year + (q_month == 12), 1 if q_month == 12 else q_month + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def _norm(s) -> str:
    return str(s).strip().upper() if s is not None and not pd.isna(s) else ""


# ---------------------------------------------------------------- the pieces

def holdings(inv: pd.DataFrame, commitments: pd.DataFrame, as_of: date) -> List[dict]:
    """(deal, holding entity) pairs: every non-OP investor committed to a deal.

    The tracker's "GEXD Codes" tab, derived instead of typed. A holder whose
    commitment had not started by ``as_of`` is not yet a holder.
    """
    import committed_pref as cp
    from loaders import build_investmentid_to_vcode
    out, seen = [], set()
    if inv is None or inv.empty or commitments is None:
        return out
    # THE ENGINES' OWN vcode for an InvestmentID. InvestmentID is not unique
    # (Donald Lynch is MCCORD under P0000049 and P0000073), and the pref engine
    # reaches a deal's ledger through this same map -- asked under the other
    # vcode it finds no InvestmentID and answers 0.
    canonical = {_norm(k): str(v).strip() for k, v in build_investmentid_to_vcode(inv).items()}
    cols = {c.lower(): c for c in inv.columns}
    for r in inv.to_dict("records"):
        iid = _norm(r.get(cols.get("investmentid", "InvestmentID")))
        vcode = str(r.get(cols.get("vcode", "vcode")) or "").strip()
        if not iid or not vcode or canonical.get(iid, vcode) != vcode:
            continue
        rows = cp.deal_commitment_rows(commitments, iid)
        for holder in sorted({x["investor"] for x in rows}):
            chain = [x for x in rows if x["investor"] == holder]
            if not any(x["start"] is not None and x["start"] <= as_of for x in chain):
                continue
            if (iid, holder) in seen:          # InvestmentID is not unique to a vcode
                continue
            seen.add((iid, holder))
            live = cp.row_in_effect(chain, as_of)
            out.append({
                "vcode": vcode, "investment_id": iid, "holder": holder,
                "deal_name": str(r.get(cols.get("investment_name", "Investment_Name")) or iid).strip(),
                "currency": _norm(r.get(cols.get("currency", "Currency"))) or "USD",
                "committed": live["amount"] if live else None,
            })
    return out


def noncash_by_holding(engine, as_of: date) -> Dict[str, Dict[tuple, float]]:
    """The two non-cash terms the tracker uses, per (InvestmentID, InvestorID).

    ``unrealized``: every "Unrealized Gain/Loss" row (FMV only).
    ``realized_loss``: "Realized Gain/Loss" rows BELOW ZERO, row by row -- the
    tracker's ``SUMIFS(..., "<0")``. A written-off investment is closed this way
    (Adirondack -2,419,200, City West -5,925,000), and Cost counts it.

    FROM ``ia_transactions`` -- accounting's IA query, accounting's signs. The
    app's ``accounting`` feed keeps no non-cash rows, which is why the pref
    engine's balance carries a written-off deal at its full capital. Cut on
    TRANSACTION date, the date accounting's own query parameter cuts on: 1,767
    non-cash rows have no Effective Date at all (the 2021-22 year-end marks
    among them), so a cut on Effective Date would silently drop them.
    """
    with engine.connect() as c:
        rows = c.execute(text(
            'SELECT "InvestmentID", "InvestorID", "Typename", "Amount" FROM ia_transactions '
            'WHERE "Typename" IN (:u, :r) AND "TransactionDate" <= :d'),
            {"u": "Unrealized Gain/Loss", "r": "Realized Gain/Loss",
             "d": as_of.isoformat() + " 23:59:59"}).all()
    out: Dict[str, Dict[tuple, float]] = {"unrealized": {}, "realized_loss": {}}
    for a, i, t, amt in rows:
        amt = float(amt or 0.0)
        k = (_norm(a), _norm(i))
        if t == "Unrealized Gain/Loss":
            out["unrealized"][k] = out["unrealized"].get(k, 0.0) + amt
        elif amt < 0:
            out["realized_loss"][k] = out["realized_loss"].get(k, 0.0) + amt
    return out


def _cost(h: dict, as_of: date, acct, inv, wf_steps) -> Optional[float]:
    from flask_app.services.reports_service import build_pref_balance_detail
    try:
        d = build_pref_balance_detail(h["vcode"], h["holder"], as_of, acct, inv, wf_steps=wf_steps)
    except Exception:
        logger.warning("pe exposure: cost failed for %s/%s", h["vcode"], h["holder"], exc_info=True)
        return None
    v = (d.get("header") or {}).get("investment_balance")
    return None if v is None else float(v)


def _remaining_to_fund(vcode: str, investment_id: str, as_of: date, acct, wf, inv,
                       commitments) -> Optional[dict]:
    """The One Pager's remaining to fund, by the One Pager's own engine."""
    from one_pager import get_pe_performance
    from investment_metrics import _deal_accounting
    quarter = f"{as_of.year}-Q{(as_of.month - 1) // 3 + 1}"
    try:
        pe = get_pe_performance(vcode, quarter, _deal_accounting(acct, investment_id), wf, inv,
                                isbs_raw=None, deal_terms=None, commitments=commitments)
    except Exception:
        logger.warning("pe exposure: remaining to fund failed for %s", vcode, exc_info=True)
        return None
    if pe.get("remaining_to_fund") is None:
        return None
    return {"remaining_to_fund": float(pe["remaining_to_fund"]),
            "committed": pe.get("committed_pe"), "funded": pe.get("funded_to_date"),
            "basis": pe.get("committed_pe_basis"),
            "below_funded": bool(pe.get("committed_below_funded"))}


def _split(amount: Optional[float], shares: Dict[str, float]) -> Dict[str, Optional[float]]:
    if amount is None:
        return {g: None for g in GROUPS}
    return {g: amount * shares.get(g, 0.0) for g in GROUPS}


# ---------------------------------------------------------------- the report

def build(as_of: date, data: Optional[dict] = None, engine=None) -> dict:
    """The tracker for one quarter end: Cost and FMV by holding and investor group."""
    from flask_app.db import get_engine
    from flask_app.services import market_rates_service as mr
    from flask_app.services import ownership_chain_service as oc
    from loaders import load_waterfalls

    engine = engine or get_engine()
    if data is None:
        from flask_app.services.data_service import get_data
        data = get_data()
    acct, inv, wf = data.get("acct"), data.get("inv"), data.get("wf")
    commitments = data.get("commitments_raw")
    # ANY DATE, not only a quarter end: "Live" is today (Jim, Oct 5 2026). Every
    # engine here takes a date -- the pref balance, the IA cut, the commitments
    # in force, the published rate -- except the One Pager's remaining to fund,
    # which works by QUARTER; for a mid-quarter date it answers for the quarter
    # the date falls in, and the report says so (``future_funding_basis``).
    wf_steps = load_waterfalls(wf) if wf is not None else None

    fx = mr.rate_on(engine, "USDCAD", as_of)
    nc = noncash_by_holding(engine, as_of)
    unreal, rloss = nc["unrealized"], nc["realized_loss"]
    src = oc._Source(engine, as_of=as_of, with_balances=False)

    rows, notes = [], []
    for h in holdings(inv, commitments, as_of):
        balance = _cost(h, as_of, acct, inv, wf_steps)
        loss = rloss.get((h["investment_id"], h["holder"]), 0.0)
        u = unreal.get((h["investment_id"], h["holder"]), 0.0)
        # Cost = the Pref Balance Detail capital balance, less capital written
        # off. Each term is its own column so the arithmetic can be read.
        cost = None if balance is None else balance + loss
        fmv = None if cost is None else cost + u
        if (cost is None or abs(cost) < 0.5) and (fmv is None or abs(fmv) < 0.5) and abs(u) < 0.5:
            continue
        walk = oc.group_shares(h["holder"], as_of, STOPS, context=_context,
                               default_group=DEFAULT_GROUP, src=src)
        shares = walk["shares"]
        rate = None
        if h["currency"] != "USD":
            rate = fx["value"] if (fx and h["currency"] == "CAD") else None
        to_usd = (lambda v: v) if h["currency"] == "USD" else (
            (lambda v: None if v is None else v / rate) if rate else (lambda v: None))
        rows.append({
            **h,
            "balance_local": balance, "realized_loss_local": loss,
            "cost_local": cost, "unrealized_local": u, "fmv_local": fmv,
            "balance": to_usd(balance), "realized_loss": to_usd(loss),
            "cost": to_usd(cost), "unrealized": to_usd(u), "fmv": to_usd(fmv),
            "fx": ({"rate": rate, "unit": "CAD per USD", "date": fx["date"], "source": fx["source"]}
                   if rate else None),
            "fx_missing": h["currency"] != "USD" and rate is None,
            "shares": {g: shares.get(g, 0.0) for g in GROUPS},
            "share_total": sum(shares.values()),
            "cost_by_group": _split(to_usd(cost), shares),
            "fmv_by_group": _split(to_usd(fmv), shares),
            "routes": walk["routes"], "problems": walk["problems"],
        })

    rows.sort(key=lambda r: (r["deal_name"].lower(), r["holder"]))

    # FUTURE FUNDING: the One Pager's remaining to fund, per deal. Split by the
    # holder's groups when the deal has one holder; a deal with several (Pegasus)
    # is shown whole, since the deal-level figure does not say whose it is.
    holders_by_deal: Dict[str, list] = {}
    for h in holdings(inv, commitments, as_of):
        holders_by_deal.setdefault(h["vcode"], []).append(h)
    row_by_holder = {(r["vcode"], r["holder"]): r for r in rows}
    future = []
    for vcode, hs in holders_by_deal.items():
        rtf = _remaining_to_fund(vcode, hs[0]["investment_id"], as_of, acct, wf, inv, commitments)
        if not rtf or abs(rtf["remaining_to_fund"]) < 1.0:
            continue
        cur = hs[0]["currency"]
        amt = rtf["remaining_to_fund"]
        amt_usd = amt if cur == "USD" else (amt / fx["value"] if (fx and cur == "CAD") else None)
        shares = None
        if len(hs) == 1:
            r = row_by_holder.get((vcode, hs[0]["holder"]))
            shares = (r["shares"] if r else
                      oc.group_shares(hs[0]["holder"], as_of, STOPS, context=_context,
                                      default_group=DEFAULT_GROUP, src=src)["shares"])
        future.append({
            "vcode": vcode, "deal_name": hs[0]["deal_name"], "currency": cur,
            "holders": [x["holder"] for x in hs], **rtf,
            "remaining_to_fund_usd": amt_usd,
            "by_group": _split(amt_usd, shares) if shares else None,
        })
    future.sort(key=lambda r: r["deal_name"].lower())
    is_qe = as_of == quarter_end(as_of)
    q_label = f"{as_of.year}-Q{(as_of.month - 1) // 3 + 1}"

    def total(items, key):
        out = {g: 0.0 for g in GROUPS}
        for it in items:
            for g, v in (it.get(key) or {}).items():
                if v is not None:
                    out[g] += v
        return out

    usd_rows = [r for r in rows if not r["fx_missing"]]
    cost_tot, fmv_tot = total(usd_rows, "cost_by_group"), total(usd_rows, "fmv_by_group")
    fut_tot = total(future, "by_group")
    unsplit_future = sum(f["remaining_to_fund_usd"] or 0.0 for f in future if not f["by_group"])

    if any(r["fx_missing"] for r in rows):
        notes.append("No USD/CAD rate is stored for this quarter end; CAD holdings are shown "
                     "in CAD and left out of the USD totals. Refresh Market Rates.")
    for r in rows:
        if r["problems"]:
            notes.append(f"{r['deal_name']} ({r['holder']}): " + "; ".join(r["problems"]))
        elif abs(r["share_total"] - 1.0) > 0.0005:
            notes.append(f"{r['deal_name']} ({r['holder']}): investor shares total "
                         f"{r['share_total']:.2%}, not 100%")

    return {
        "as_of": as_of.isoformat(),
        "is_quarter_end": is_qe,
        "future_funding_basis": (None if is_qe else
                                 f"The One Pager's remaining to fund works by quarter, so this is "
                                 f"its {q_label} figure, not a figure as of {as_of.isoformat()}."),
        "groups": list(GROUPS),
        "fx": fx,
        "rows": rows,
        "future_funding": future,
        "totals": {
            "cost": sum(r["cost"] or 0.0 for r in usd_rows),
            "fmv": sum(r["fmv"] or 0.0 for r in usd_rows),
            "cost_by_group": cost_tot, "fmv_by_group": fmv_tot,
            "future_funding": sum(f["remaining_to_fund_usd"] or 0.0 for f in future),
            "future_by_group": fut_tot, "future_unsplit": unsplit_future,
            # TOTAL EQUITY INVESTED / COMMITTED -- the tracker's grand total:
            # current exposure plus what is committed and not yet funded.
            "grand_cost": sum(r["cost"] or 0.0 for r in usd_rows)
                          + sum(f["remaining_to_fund_usd"] or 0.0 for f in future),
            "grand_fmv": sum(r["fmv"] or 0.0 for r in usd_rows)
                         + sum(f["remaining_to_fund_usd"] or 0.0 for f in future),
            "grand_cost_by_group": {g: cost_tot[g] + fut_tot[g] for g in GROUPS},
            "grand_fmv_by_group": {g: fmv_tot[g] + fut_tot[g] for g in GROUPS},
        },
        "notes": notes,
    }


# ---------------------------------------------------------------- quarters & cache

#: A quarter is offered as the default once this many days have passed since it
#: ended -- the rule Investment Metrics records (DEFAULT_QUARTER_LAG_DAYS),
#: because a quarter that closed last week has no accounting behind it yet.
DEFAULT_QUARTER_LAG_DAYS = 45


def _previous_quarter_end(qe: date) -> date:
    first = date(qe.year, qe.month - 2, 1)
    return date.fromordinal(first.toordinal() - 1)


def quarter_options(today: Optional[date] = None, count: int = 8) -> dict:
    """The last ``count`` FINISHED quarter ends, newest first, and the default."""
    today = today or date.today()
    qe = quarter_end(today)
    if qe > today:
        qe = _previous_quarter_end(qe)
    ends = [qe]
    while len(ends) < count:
        ends.append(_previous_quarter_end(ends[-1]))
    default = next((e for e in ends if (today - e).days >= DEFAULT_QUARTER_LAG_DAYS), ends[-1])
    return {"quarters": [e.isoformat() for e in ends], "default": default.isoformat(),
            "live": today.isoformat(),
            "default_rule": f"the latest quarter ended at least {DEFAULT_QUARTER_LAG_DAYS} days ago"}


_CACHE: Dict[tuple, dict] = {}


def get_report(as_of: date, data: Optional[dict] = None, engine=None) -> dict:
    """``build`` with a cache keyed on the data it was built from.

    Keyed by OBJECT IDENTITY of the frames, as the Investment Metrics route
    cache is, and holding a reference to them so an id cannot be reused: a data
    reload creates new frames, so a refresh invalidates the entry by itself.
    """
    if data is None:
        from flask_app.services.data_service import get_data
        data = get_data()
    frames = (data.get("acct"), data.get("commitments_raw"), data.get("inv"))
    key = (as_of,) + tuple(id(f) for f in frames)
    hit = _CACHE.get(key)
    if hit is not None and all(a is b for a, b in zip(hit["_frames"], frames)):
        return hit["report"]
    report = build(as_of, data=data, engine=engine)
    if len(_CACHE) >= 6:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = {"report": report, "_frames": frames}
    return report


# ---------------------------------------------------------------- Excel

def to_excel(report: dict) -> bytes:
    """The tracker as a workbook: Cost and FMV (each with future funding below and a
    grand total, as accounting's tracker lays them out), Future Funding Detail,
    Ownership Routes, Sources."""
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    money = '#,##0;(#,##0);"-"'
    pct = '0.00%;(0.00%);"-"'
    hfont, hfill = Font(bold=True, color="FFFFFF"), PatternFill("solid", start_color="2F6F4F")
    bold, top = Font(bold=True), Border(top=Side(style="thin"))
    groups = report["groups"]
    fx = report.get("fx")
    fx_note = (f"CAD converted at {fx['value']:.4f} CAD per USD ({fx['source']}, {fx['date']})"
               if fx else "No USD/CAD rate stored for this quarter end: CAD holdings are left out of USD totals")

    wb = Workbook()

    def sheet(ws, title, lead_cols, group_key):
        """One exposure sheet laid out as accounting's tracker is: current
        exposure, its subtotal, future funding below it, its subtotal, and the
        grand total -- Total Equity Invested / Committed."""
        ws.append([title])
        ws["A1"].font = Font(bold=True, size=13)
        ws.append([f"As of {report['as_of']}.  {fx_note}."])
        ws.append([])
        ws.append(["Property", "Investment", "Holding entity", "Currency"]
                  + [c[0] for c in lead_cols] + groups + [g + " %" for g in groups])
        for c in ws[4]:
            c.font, c.fill = hfont, hfill
            c.alignment = Alignment(horizontal="center", wrap_text=True)
        n_lead, n_grp = len(lead_cols), len(groups)
        amt_col = 4 + n_lead                     # the Cost / FMV column (1-based)
        first_grp = amt_col + 1
        first_pct = first_grp + n_grp

        def section(label):
            ws.append([label])
            ws.cell(ws.max_row, 1).font = bold

        def total_row(label, rows, cols):
            vals = [label, "", "", ""]
            for ci in range(5, first_pct):
                col = get_column_letter(ci)
                vals.append("=" + "+".join(f"SUM({col}{a}:{col}{b})" for a, b in rows) if rows and ci in cols else None)
            ws.append(vals)
            r = ws.max_row
            for g in range(n_grp):           # the group % of a total: its $ over the total's $
                ws.cell(r, first_pct + g).value = (
                    f"=IF({get_column_letter(amt_col)}{r}=0,0,"
                    f"{get_column_letter(first_grp + g)}{r}/{get_column_letter(amt_col)}{r})")
            for c in ws[r]:
                c.font, c.border = bold, top
            return r

        section("Current exposure")
        a = ws.max_row + 1
        for r in report["rows"]:
            ws.append([r["deal_name"], r["investment_id"], r["holder"], r["currency"]]
                      + [r.get(c[1]) for c in lead_cols]
                      + [(r[group_key] or {}).get(g) for g in groups]
                      + [r["shares"].get(g) for g in groups])
        b = ws.max_row
        net = total_row("Net Invested Equity", [(a, b)], set(range(5, first_pct)))
        ws.append([])

        section("Future funding - committed, not yet funded (the One Pager's remaining to fund)")
        fa = ws.max_row + 1
        for f in report["future_funding"]:
            amt = f.get("remaining_to_fund_usd")
            by = f.get("by_group") or {}
            ws.append([f["deal_name"], "", ", ".join(f["holders"]), f["currency"]]
                      + [amt if i == n_lead - 1 else None for i in range(n_lead)]
                      + [by.get(g) if by else None for g in groups]
                      + [((by.get(g) or 0) / amt) if (by and amt) else None for g in groups])
        fb = ws.max_row
        fut = total_row("Total Future Funding", [(fa, fb)] if fb >= fa else [],
                        set(range(amt_col, first_pct)))
        ws.append([])
        grand = ["Total Equity Invested / Committed", "", "", ""]
        for ci in range(5, first_pct):
            col = get_column_letter(ci)
            grand.append(f"={col}{net}+{col}{fut}" if ci >= amt_col else None)
        ws.append(grand)
        g_row = ws.max_row
        for g in range(n_grp):
            ws.cell(g_row, first_pct + g).value = (
                f"=IF({get_column_letter(amt_col)}{g_row}=0,0,"
                f"{get_column_letter(first_grp + g)}{g_row}/{get_column_letter(amt_col)}{g_row})")
        for c in ws[g_row]:
            c.font = Font(bold=True, size=11)
            c.border = Border(top=Side(style="double"))

        for row in ws.iter_rows(min_row=5, max_row=ws.max_row):
            for c in row[4:first_pct - 1]:
                c.number_format = money
            for c in row[first_pct - 1:]:
                c.number_format = pct
        ws.freeze_panes = "E5"
        widths = [34, 11, 14, 9] + [15] * (n_lead + n_grp) + [10] * n_grp
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w

    ws = wb.active
    ws.title = "Cost"
    sheet(ws, "PSC Preferred Equity Exposure - Cost",
          [("Capital balance", "balance"), ("Realized loss", "realized_loss"), ("Cost", "cost")],
          "cost_by_group")
    sheet(wb.create_sheet("FMV"), "PSC Preferred Equity Exposure - Fair Market Value",
          [("Cost", "cost"), ("Unrealized gain/loss", "unrealized"), ("FMV", "fmv")],
          "fmv_by_group")

    ws3 = wb.create_sheet("Future Funding Detail")
    ws3.append(["Future funding - committed, not yet funded (the One Pager's remaining to fund)"])
    ws3["A1"].font = Font(bold=True, size=13)
    ws3.append([f"As of {report['as_of']}"])
    ws3.append([])
    ws3.append(["Property", "Holding entity", "Committed", "Funded", "Remaining to fund (USD)"]
               + groups + ["Basis"])
    for c in ws3[4]:
        c.font, c.fill = hfont, hfill
    for f in report["future_funding"]:
        flags = ("several holders, not split; " if not f.get("by_group") else "") \
            + ("committed below funded; " if f.get("below_funded") else "")
        ws3.append([f["deal_name"], ", ".join(f["holders"]), f.get("committed"), f.get("funded"),
                    f.get("remaining_to_fund_usd")]
                   + [(f.get("by_group") or {}).get(g) for g in groups]
                   + [flags + str(f.get("basis") or "")])
    for row in ws3.iter_rows(min_row=5, max_row=ws3.max_row):
        for c in row[2:5 + len(groups)]:
            c.number_format = money
    for i, w in enumerate([34, 16, 15, 15, 18] + [14] * len(groups) + [60], 1):
        ws3.column_dimensions[get_column_letter(i)].width = w

    ws4 = wb.create_sheet("Ownership Routes")
    ws4.append(["Every chain each percentage came from: commitments in force at the quarter end, "
                "each owner's share = its commitment over the entity's total, multiplied down"])
    ws4["A1"].font = bold
    ws4.append([])
    ws4.append(["Holding entity", "Route", "Share of holding", "Group"])
    for c in ws4[3]:
        c.font, c.fill = hfont, hfill
    for r in report["rows"]:
        for rt in r["routes"]:
            ws4.append([r["holder"], " > ".join(rt["path"]), rt["share"], rt["group"]])
    for row in ws4.iter_rows(min_row=4, max_row=ws4.max_row):
        row[2].number_format = "0.0000%"
    for i, w in enumerate([16, 70, 16, 14], 1):
        ws4.column_dimensions[get_column_letter(i)].width = w

    ws5 = wb.create_sheet("Sources")
    lines = [
        "Where every figure comes from",
        "",
        "Holding entity: the non-OP investors in MRI commitments (IA_Commitment) for the deal.",
        "Capital balance: the Pref Balance Detail report's investment balance - the same figure that report shows.",
        "Realized loss: MRI IA 'Realized Gain/Loss' rows below zero (accounting's IA query), through the quarter end.",
        "Cost = Capital balance + Realized loss.",
        "Unrealized gain/loss: MRI IA 'Unrealized Gain/Loss' rows (accounting's IA query), cut on transaction date.",
        "FMV = Cost + Unrealized gain/loss.",
        "Investor %: MRI commitments in force at the quarter end, multiplied through the ownership chain.",
        "  Stops: PSC = PSC1, PSC2, OWPSC, PSCMAN, PSL1, PSS1; KOC = KCREIT; TIAA = TGAM;",
        "  Declaration = DCXVIA, DCXVIB; Clarion = DIFPP; an AMB fund's outside investors = Ambassadors;",
        "  every other outside investor = F&F.",
        "Future funding: the One Pager's remaining to fund (committed pref less funded to date).",
        f"FX: {fx_note}.",
        "",
        "Notes",
    ] + (report.get("notes") or ["(none)"])
    for line in lines:
        ws5.append([line])
    ws5["A1"].font = Font(bold=True, size=13)
    ws5.column_dimensions["A"].width = 120

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
