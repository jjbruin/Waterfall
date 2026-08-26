"""
prospect_excel.py
Audit workbook for a Prospect Deal Analysis.

Built to be handed to a third party: every tab shows the inputs or the
supporting calculation behind the results, and the partner cash flow tab
carries live =XIRR formulas so an auditor's Excel recomputes the returns
from the same cash flows the model used.
"""

import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Assumption fields grouped the way an auditor reads them
_ASSUMPTION_GROUPS = [
    ("Acquisition", [
        ("purchase_price", "Purchase Price", "curr"),
        ("closing_cost_pct", "Closing Cost %", "pct"),
        ("capex_at_close", "CapEx at Close", "curr"),
        ("hold_years", "Hold (years)", "num"),
    ]),
    ("Debt", [
        ("debt_amount", "Debt Amount", "curr"),
        ("debt_rate", "Interest Rate", "pct"),
        ("debt_term_months", "Term (months)", "num"),
        ("io_months", "IO (months)", "num"),
        ("amort_months", "Amortization (months)", "num"),
        ("lender", "Lender", "text"),
        ("rate_type", "Rate Type", "text"),
        ("max_ltv", "Max LTV", "num"),
        ("min_dscr", "Min DSCR", "num"),
        ("min_debt_yield", "Min Debt Yield", "num"),
    ]),
    ("Equity & Waterfall", [
        ("psc_equity_pct", "PSC Equity %", "pct"),
        ("pref_rate", "Preferred Return", "pct"),
        ("promote_pct", "Promote %", "pct"),
    ]),
    ("Operations & Exit", [
        ("noi_year1", "NOI Year 1", "curr"),
        ("noi_growth_rate", "NOI Growth", "pct"),
        ("mgmt_fee_pct", "Management Fee %", "pct"),
        ("replacement_reserve_psf", "Replacement Reserve /SF", "num"),
        ("capex_reserve_psf", "CapEx Reserve /SF", "num"),
        ("exit_cap_rate", "Exit Cap Rate", "pct"),
        ("selling_cost_pct", "Selling Cost %", "pct"),
    ]),
]


def _fmt_cell(ws, row, col, value, styles, kind):
    c = ws.cell(row=row, column=col, value=value)
    if kind == "curr":
        c.number_format = styles["curr"]
    elif kind == "pct":
        c.number_format = styles["pct"]
    elif kind == "num":
        c.number_format = styles["num"]
    return c


def _section(ws, row, title, styles):
    c = ws.cell(row=row, column=1, value=title)
    c.font = styles["bold"]
    return row + 1


def generate_prospect_analysis_excel(
    result: Dict[str, Any],
    deal: Dict[str, Any],
    assumptions: Dict[str, Any],
    wf_steps: Optional[List[Dict[str, Any]]] = None,
    scenario: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Build the audit workbook. Returns xlsx bytes."""
    import json
    import pandas as pd
    from openpyxl import Workbook
    from flask_app.services.compute_service import (
        _excel_styles, _write_header_row, _autosize_columns,
        _build_forecast_table, _write_forecast_to_sheet,
    )

    s = _excel_styles()
    wb = Workbook()

    # ------------------------------------------------------------------
    # 1. Summary
    # ------------------------------------------------------------------
    ws = wb.active
    ws.title = "Summary"
    r = 1
    ws.cell(row=r, column=1, value=f"Deal Analysis — {deal.get('deal_name', '')}").font = s["bold"]
    r += 1
    for label, val in [
        ("vCode", deal.get("vcode")),
        ("Stage", deal.get("stage")),
        ("Scenario", (scenario or {}).get("name") or "Live assumptions"),
        ("Generated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
    ]:
        ws.cell(row=r, column=1, value=label)
        ws.cell(row=r, column=2, value=val)
        r += 1
    r += 1

    r = _section(ws, r, "Partner Returns", s)
    _write_header_row(ws, r, ["Partner", "Contributions", "CF Distributions",
                              "Capital Distributions", "Total Distributions",
                              "IRR", "ROE", "MOIC"], s)
    r += 1
    for p in (result.get("partner_results") or []):
        ws.cell(row=r, column=1, value=p.get("partner"))
        _fmt_cell(ws, r, 2, p.get("contributions"), s, "curr")
        _fmt_cell(ws, r, 3, p.get("cf_distributions"), s, "curr")
        _fmt_cell(ws, r, 4, p.get("cap_distributions"), s, "curr")
        _fmt_cell(ws, r, 5, p.get("total_distributions"), s, "curr")
        _fmt_cell(ws, r, 6, p.get("irr"), s, "pct")
        _fmt_cell(ws, r, 7, p.get("roe"), s, "pct")
        c = ws.cell(row=r, column=8, value=p.get("moic"))
        c.number_format = s["mult"]
        r += 1
    ds = result.get("deal_summary") or {}
    ws.cell(row=r, column=1, value="Deal Total").font = s["bold"]
    _fmt_cell(ws, r, 2, ds.get("total_contributions"), s, "curr").font = s["bold"]
    _fmt_cell(ws, r, 3, ds.get("total_cf_distributions"), s, "curr").font = s["bold"]
    _fmt_cell(ws, r, 4, ds.get("total_cap_distributions"), s, "curr").font = s["bold"]
    _fmt_cell(ws, r, 5, ds.get("total_distributions"), s, "curr").font = s["bold"]
    _fmt_cell(ws, r, 6, ds.get("deal_irr"), s, "pct").font = s["bold"]
    _fmt_cell(ws, r, 7, ds.get("deal_roe"), s, "pct").font = s["bold"]
    c = ws.cell(row=r, column=8, value=ds.get("deal_moic"))
    c.number_format = s["mult"]
    c.font = s["bold"]
    r += 2

    sd = result.get("sale_dbg") or {}
    if sd:
        r = _section(ws, r, "Sale Proceeds Calculation", s)
        for label, key, kind in [
            ("Sale Date", None, "text"),
            ("Terminal NOI (fwd 12 mo)", "NOI_12m_After_Sale", "curr"),
            ("Cap Rate", "CapRate_Sale", "pct"),
            ("Sale Price", "Implied_Value", "curr"),
            ("Less: Selling Costs", "Selling_Cost_Amount", "curr"),
            ("Less: Loan Payoff", "Less_Loan_Balances", "curr"),
            ("Net Sale Proceeds", "Net_Sale_Proceeds", "curr"),
        ]:
            ws.cell(row=r, column=1, value=label)
            if key is None:
                ws.cell(row=r, column=2, value=str(result.get("sale_me") or ""))
            else:
                _fmt_cell(ws, r, 2, sd.get(key), s, kind)
            r += 1
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 2. Assumptions (incl. capital budget + refi + scenario)
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Assumptions")
    r = 1
    for group, fields in _ASSUMPTION_GROUPS:
        r = _section(ws, r, group, s)
        for key, label, kind in fields:
            val = (assumptions or {}).get(key)
            if val in (None, ""):
                continue
            ws.cell(row=r, column=1, value=label)
            _fmt_cell(ws, r, 2, val, s, kind)
            r += 1
        r += 1

    def _json(blob):
        if not blob:
            return None
        try:
            return json.loads(blob) if isinstance(blob, str) else blob
        except (TypeError, ValueError):
            return None

    uses = _json((assumptions or {}).get("capital_uses_json"))
    if uses:
        r = _section(ws, r, "Capital Budget — Uses", s)
        for u in uses:
            amt = u.get("amount")
            if amt in (None, 0, ""):
                continue
            ws.cell(row=r, column=1, value=u.get("label") or u.get("id"))
            _fmt_cell(ws, r, 2, amt, s, "curr")
            r += 1
        r += 1

    srcs = _json((assumptions or {}).get("capital_sources_json"))
    if srcs:
        r = _section(ws, r, "Capital Budget — Debt Sources", s)
        for d in (srcs.get("debt") or []):
            amt = d.get("amount")
            if amt in (None, 0, ""):
                continue
            label = d.get("label") or d.get("id")
            if d.get("rate"):
                label += (f"  ({d['rate']*100:.2f}%"
                          + (f", {d.get('term_months')}mo" if d.get("term_months") else "")
                          + ", own loan)")
            ws.cell(row=r, column=1, value=label)
            _fmt_cell(ws, r, 2, amt, s, "curr")
            r += 1
        r += 1

    refi = _json((assumptions or {}).get("planned_refi_json"))
    if refi and refi.get("enabled"):
        r = _section(ws, r, "Planned Refinancing", s)
        for label, key, kind in [
            ("Refi Date", "refi_date", "text"), ("New Loan Amount", "loan_amount", "curr"),
            ("Rate", "rate", "pct"), ("Term (years)", "term_years", "num"),
            ("Amort (years)", "amort_years", "num"), ("IO (years)", "io_years", "num"),
            ("Closing Costs", "closing_costs", "curr"), ("Reserve Holdback", "holdback", "curr"),
        ]:
            if refi.get(key) in (None, "", 0) and key != "refi_date":
                continue
            ws.cell(row=r, column=1, value=label)
            _fmt_cell(ws, r, 2, refi.get(key), s, kind)
            r += 1
        r += 1

    if scenario:
        r = _section(ws, r, f"Scenario — {scenario.get('name')}", s)
        for k, v in (scenario.get("assumption_overrides") or {}).items():
            ws.cell(row=r, column=1, value=f"Override: {k}")
            ws.cell(row=r, column=2, value=v)
            r += 1
        for adj in (scenario.get("adjustments") or []):
            span = f"from {adj.get('start_date')}" + (f" to {adj.get('end_date')}" if adj.get("end_date") else "")
            rev = sum(float(v or 0) for v in (adj.get("revenue") or {}).values())
            exp = sum(float(v or 0) for v in (adj.get("expense") or {}).values())
            ws.cell(row=r, column=1, value=f"Adjustment: {adj.get('label')} ({span})")
            ws.cell(row=r, column=2, value=f"revenue {rev:+,.0f}/yr, expense {exp:+,.0f}/yr")
            r += 1
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 3. Annual Forecast (reuses the AM builder)
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Annual Forecast")
    try:
        start_year = int(str(result.get("model_start"))[:4])
        end_year = int(str(result.get("model_end_full"))[:4])
        built = _build_forecast_table(result, start_year, end_year - start_year + 1)
        if built:
            _t, wide, years = built
            _write_forecast_to_sheet(ws, wide, years, s)
        else:
            ws.cell(row=1, column=1, value="No forecast data")
    except Exception as e:
        logger.warning("Forecast sheet failed: %s", e)
        ws.cell(row=1, column=1, value=f"Forecast sheet unavailable: {e}")
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 4. Debt Service (loan summary + monthly amortization)
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Debt Service")
    ls = result.get("loan_sched")
    if ls is not None and not ls.empty:
        r = _section(ws, 1, "Loan Summary", s)
        _write_header_row(ws, r, ["Loan", "First Period", "Last Period", "Rate",
                                  "Total Interest", "Total Principal",
                                  "Curtailments", "Ending Balance"], s)
        r += 1
        for lid, grp in ls.groupby("LoanID"):
            g = grp.sort_values("event_date")
            ws.cell(row=r, column=1, value=str(lid))
            ws.cell(row=r, column=2, value=str(g.iloc[0]["event_date"])[:10])
            ws.cell(row=r, column=3, value=str(g.iloc[-1]["event_date"])[:10])
            _fmt_cell(ws, r, 4, float(g.iloc[0]["rate"]), s, "pct")
            _fmt_cell(ws, r, 5, float(g["interest"].sum()), s, "curr")
            _fmt_cell(ws, r, 6, float(g["principal"].sum()), s, "curr")
            cur = float(g["curtailment"].sum()) if "curtailment" in g.columns else 0.0
            _fmt_cell(ws, r, 7, cur, s, "curr")
            _fmt_cell(ws, r, 8, float(g.iloc[-1]["ending_balance"]), s, "curr")
            r += 1
        r += 1
        r = _section(ws, r, "Amortization Schedule (monthly)", s)
        cols = ["LoanID", "Period", "Rate", "Interest", "Principal", "Payment",
                "Curtailment", "Ending Balance"]
        _write_header_row(ws, r, cols, s)
        r += 1
        for _, x in ls.sort_values(["LoanID", "event_date"]).iterrows():
            ws.cell(row=r, column=1, value=str(x["LoanID"]))
            ws.cell(row=r, column=2, value=str(x["event_date"])[:10])
            _fmt_cell(ws, r, 3, float(x["rate"]), s, "pct")
            _fmt_cell(ws, r, 4, float(x["interest"]), s, "curr")
            _fmt_cell(ws, r, 5, float(x["principal"]), s, "curr")
            _fmt_cell(ws, r, 6, float(x["payment"]), s, "curr")
            _fmt_cell(ws, r, 7, float(x.get("curtailment", 0) or 0), s, "curr")
            _fmt_cell(ws, r, 8, float(x["ending_balance"]), s, "curr")
            r += 1
    else:
        ws.cell(row=1, column=1, value="No debt modelled")
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 5. Cash Management
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Cash Management")
    cs = result.get("cash_schedule")
    if cs is not None and not cs.empty:
        cols = [c for c in ["event_date", "beginning_cash", "capital_call",
                            "reserve_deposit", "operating_cf", "capex_paid",
                            "capex_unpaid", "deficit_covered", "distributable",
                            "ending_cash"] if c in cs.columns]
        _write_header_row(ws, 1, [c.replace("_", " ").title() for c in cols], s)
        r = 2
        for _, x in cs.iterrows():
            for ci, c in enumerate(cols, 1):
                v = x[c]
                if c == "event_date":
                    ws.cell(row=r, column=ci, value=str(v)[:10])
                else:
                    _fmt_cell(ws, r, ci, float(v or 0), s, "curr")
            r += 1
    else:
        ws.cell(row=1, column=1, value="No cash schedule")
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 6. Waterfall Steps (the rules the money followed)
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Waterfall Steps")
    _write_header_row(ws, 1, ["Waterfall", "Tie #", "Partner", "Step", "FX Rate",
                              "Rate %", "Amount", "Description"], s)
    r = 2
    for st in (wf_steps or []):
        ws.cell(row=r, column=1, value=st.get("vmisc"))
        ws.cell(row=r, column=2, value=st.get("iOrder"))
        ws.cell(row=r, column=3, value=st.get("PropCode"))
        ws.cell(row=r, column=4, value=st.get("vState"))
        ws.cell(row=r, column=5, value=st.get("FXRate"))
        ws.cell(row=r, column=6, value=st.get("nPercent"))
        _fmt_cell(ws, r, 7, st.get("mAmount"), s, "curr")
        ws.cell(row=r, column=8, value=st.get("vtranstype"))
        r += 1
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 7. Waterfall Allocations (every dollar, every step, every period)
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Waterfall Allocations")
    r = 1
    for name, key in [("CF Waterfall", "cf_alloc"), ("Capital Waterfall", "cap_alloc")]:
        alloc = result.get(key)
        if alloc is None or alloc.empty:
            continue
        r = _section(ws, r, name, s)
        _write_header_row(ws, r, ["Period", "Tie #", "Partner", "Step",
                                  "Allocated", "Remaining After"], s)
        r += 1
        for _, x in alloc.sort_values(["event_date", "iOrder"]).iterrows():
            if not float(x.get("Allocated", 0) or 0):
                continue
            ws.cell(row=r, column=1, value=str(x["event_date"])[:10])
            ws.cell(row=r, column=2, value=int(x.get("iOrder", 0) or 0))
            ws.cell(row=r, column=3, value=str(x.get("PropCode", "")))
            ws.cell(row=r, column=4, value=str(x.get("vState", "")))
            _fmt_cell(ws, r, 5, float(x["Allocated"]), s, "curr")
            _fmt_cell(ws, r, 6, float(x.get("RemainingAfter", 0) or 0), s, "curr")
            r += 1
        r += 1
    if r == 1:
        ws.cell(row=1, column=1, value="No allocations")
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 8. Partner Cash Flows — with live =XIRR so Excel audits the IRR
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Partner Cash Flows")
    col = 1
    for p in (result.get("partner_results") or []):
        cfs = p.get("combined_cashflows") or []
        ws.cell(row=1, column=col, value=p.get("partner")).font = s["bold"]
        for ci, h in ((col, "Date"), (col + 1, "Amount")):
            hc = ws.cell(row=2, column=ci, value=h)
            hc.font = s["header_font"]
            hc.fill = s["header_fill"]
        r = 3
        for d, amt in cfs:
            c = ws.cell(row=r, column=col, value=d)
            c.number_format = s["date"]
            _fmt_cell(ws, r, col + 1, float(amt), s, "curr")
            r += 1
        from openpyxl.utils import get_column_letter
        dcol, acol = get_column_letter(col), get_column_letter(col + 1)
        ws.cell(row=r + 1, column=col, value="Excel XIRR:").font = s["bold"]
        fc = ws.cell(row=r + 1, column=col + 1,
                     value=f"=XIRR({acol}3:{acol}{r - 1},{dcol}3:{dcol}{r - 1})")
        fc.number_format = s["pct"]
        ws.cell(row=r + 2, column=col, value="Model IRR:").font = s["bold"]
        _fmt_cell(ws, r + 2, col + 1, p.get("irr"), s, "pct")
        col += 3
    _autosize_columns(ws)

    # ------------------------------------------------------------------
    # 9. Diagnostics — the model's own account of what it did
    # ------------------------------------------------------------------
    ws = wb.create_sheet("Diagnostics")
    ws.cell(row=1, column=1, value="Model Diagnostics").font = s["bold"]
    r = 2
    for m in (result.get("debug_msgs") or []):
        ws.cell(row=r, column=1, value=str(m))
        r += 1
    ws.column_dimensions["A"].width = 140

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
