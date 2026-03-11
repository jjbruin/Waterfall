"""Compute service — wraps compute.py without Streamlit dependency.

Replaces get_cached_deal_result() which uses st.session_state + st.toast.
Also contains ROE/MOIC audit builders extracted from app.py.
"""

import copy
from io import BytesIO
from typing import Optional

import pandas as pd
import numpy as np

from compute import compute_deal_analysis, build_partner_results
from loaders import load_waterfalls, load_mri_loans

# Module-level deal result cache (replaces st.session_state['_deal_results'])
_deal_cache: dict = {}


def get_cached_deal_result(
    vcode: str,
    start_year: int,
    horizon_years: int,
    pro_yr_base: int,
    data: dict,
    force: bool = False,
) -> dict:
    """Compute or retrieve cached deal result.

    Args:
        vcode: Deal vcode.
        start_year: Forecast start year.
        horizon_years: Number of years to model.
        pro_yr_base: Pro-forma base year.
        data: Dict from data_service.load_all().
        force: If True, bypass cache.

    Returns:
        Full result dict from compute_deal_analysis().
    """
    cache_key = f"{vcode}|{start_year}|{horizon_years}|{pro_yr_base}"
    if not force and cache_key in _deal_cache:
        return _deal_cache[cache_key]

    inv = data["inv"]
    wf = data["wf"]
    acct = data["acct"]
    fc = data["fc"]
    coa = data["coa"]
    mri_loans_raw = data["mri_loans_raw"]
    mri_supp = data["mri_supp"]
    mri_val = data["mri_val"]
    relationships_raw = data["relationships_raw"]
    capital_calls_raw = data["capital_calls_raw"]
    isbs_raw = data["isbs_raw"]

    # Resolve InvestmentID for this vcode
    deal_row = inv[inv["vcode"] == vcode]
    if deal_row.empty:
        raise ValueError(f"Deal not found: {vcode}")
    deal_investment_id = deal_row.iloc[0].get("InvestmentID", vcode)

    # Sale date
    sale_date_raw = deal_row.iloc[0].get("Sale_Date", None)

    result = compute_deal_analysis(
        deal_vcode=vcode,
        deal_investment_id=deal_investment_id,
        sale_date_raw=sale_date_raw,
        inv=inv,
        wf=wf,
        acct=acct,
        fc=fc,
        coa=coa,
        mri_loans_raw=mri_loans_raw,
        mri_supp=mri_supp,
        mri_val=mri_val,
        relationships_raw=relationships_raw,
        capital_calls_raw=capital_calls_raw,
        isbs_raw=isbs_raw,
        start_year=start_year,
        horizon_years=horizon_years,
        pro_yr_base=pro_yr_base,
    )

    _deal_cache[cache_key] = result
    return result


def clear_cache(vcode: Optional[str] = None):
    """Clear deal computation cache.

    Args:
        vcode: If provided, clear only entries for this deal. Otherwise clear all.
    """
    if vcode:
        keys_to_remove = [k for k in _deal_cache if k.startswith(f"{vcode}|")]
        for k in keys_to_remove:
            del _deal_cache[k]
    else:
        _deal_cache.clear()


def get_all_cached_vcodes() -> list[str]:
    """Return list of vcodes currently cached."""
    return list({k.split("|")[0] for k in _deal_cache})


# ============================================================
# ROE / MOIC Audit Builders  (extracted from app.py)
# ============================================================

def build_roe_timeline(combined_cashflows, cf_only_distributions, end_date):
    """Replay calculate_roe logic to produce an audit trail.

    Returns (timeline_rows, cf_dist_rows, summary_dict) — all JSON-safe.
    """
    cf_by_date = {}
    for d, a in cf_only_distributions:
        if a > 0:
            cf_by_date[d] = cf_by_date.get(d, 0.0) + a

    events = []
    for d, amt in combined_cashflows:
        if amt < 0:
            events.append((d, -amt))
        elif amt > 0:
            cf_at_date = cf_by_date.get(d, 0.0)
            if cf_at_date > 0:
                consumed = min(cf_at_date, amt)
                cf_by_date[d] -= consumed
                cap_return = amt - consumed
            else:
                cap_return = amt
            if cap_return > 0.005:
                events.append((d, -cap_return))
    events = sorted(events, key=lambda x: x[0])

    if not events:
        return [], [], {}

    inception = events[0][0]
    rows = []
    current_balance = 0.0
    prev_date = inception
    total_weighted = 0.0

    for evt_date, change in events:
        days = (evt_date - prev_date).days
        weighted = current_balance * days
        if days > 0 and current_balance > 0:
            rows.append({
                'Date': prev_date.isoformat(), 'Event': '(holding period)',
                'Amount': None, 'Capital Balance': current_balance,
                'Days at Balance': days, 'Weighted Capital': weighted,
            })
        total_weighted += weighted
        current_balance = max(0, current_balance + change)
        orig_amt = -change
        if orig_amt < 0:
            event_label = 'Contribution'
        elif orig_amt > 0:
            event_label = 'Capital Return'
        else:
            event_label = 'Event'
        rows.append({
            'Date': evt_date.isoformat(), 'Event': event_label,
            'Amount': orig_amt, 'Capital Balance': current_balance,
            'Days at Balance': 0, 'Weighted Capital': 0.0,
        })
        prev_date = evt_date

    if end_date and prev_date < end_date:
        days = (end_date - prev_date).days
        weighted = current_balance * days
        if days > 0:
            rows.append({
                'Date': prev_date.isoformat(), 'Event': '(holding period)',
                'Amount': None, 'Capital Balance': current_balance,
                'Days at Balance': days, 'Weighted Capital': weighted,
            })
            total_weighted += weighted

    total_days = (end_date - inception).days if end_date else 0
    wac = total_weighted / total_days if total_days > 0 else 0.0
    years = total_days / 365.0 if total_days > 0 else 0.0

    cf_rows = [{'Date': d.isoformat(), 'Amount': a} for d, a in cf_only_distributions if a > 0]
    total_cf_dist = sum(r['Amount'] for r in cf_rows)
    roe = (total_cf_dist / wac) / years if wac > 0 and years > 0 else 0.0

    summary = {
        'inception': inception.isoformat(), 'end': end_date.isoformat() if end_date else None,
        'total_days': total_days, 'years': years,
        'total_cf_dist': total_cf_dist, 'wac': wac, 'roe': roe,
    }
    return rows, cf_rows, summary


def build_roe_audit(partner_results, deal_summary, sale_me):
    """Build full ROE audit data for all partners + deal level.

    Returns dict with 'partners' list and 'deal_level' dict.
    """
    partners = []
    for pr in partner_results:
        tl, cf, summary = build_roe_timeline(
            pr['combined_cashflows'], pr['cf_only_distributions'], sale_me)
        partners.append({
            'partner': pr['partner'],
            'timeline': tl,
            'cf_distributions': cf,
            'summary': summary,
        })

    # Deal-level
    all_cfs = deal_summary.get('all_combined_cashflows', [])
    all_cf_dist = []
    for pr in partner_results:
        all_cf_dist.extend(pr['cf_only_distributions'])
    tl, cf, summary = build_roe_timeline(all_cfs, all_cf_dist, sale_me)

    return {
        'partners': partners,
        'deal_level': {'timeline': tl, 'cf_distributions': cf, 'summary': summary},
    }


def build_moic_breakdown(cashflow_details, pr_dict):
    """Build MOIC audit breakdown from cashflow_details.

    Returns (breakdown_rows, summary_dict) — JSON-safe.
    """
    rows = []
    for cf in cashflow_details:
        amt = cf['Amount']
        desc = cf.get('Description', '')
        if amt < 0:
            cf_type = 'Contribution'
        elif 'Unrealized' in desc or 'NAV' in desc:
            cf_type = 'Terminal Value'
        elif 'Capital' in desc or 'Cap ' in desc or 'Refi' in desc or 'Sale' in desc:
            cf_type = 'Cap Distribution'
        else:
            cf_type = 'CF Distribution'
        rows.append({
            'Date': cf['Date'].isoformat() if hasattr(cf['Date'], 'isoformat') else str(cf['Date']),
            'Description': desc, 'Type': cf_type, 'Amount': amt,
        })

    summary = {
        'contributions': pr_dict.get('contributions', 0.0),
        'cf_dist': pr_dict.get('cf_distributions', 0.0),
        'cap_dist': pr_dict.get('cap_distributions', 0.0),
        'total_dist': pr_dict.get('total_distributions', 0.0),
        'unrealized': pr_dict.get('unrealized_nav', 0.0),
        'moic': pr_dict.get('moic', 0.0),
    }
    return rows, summary


def build_moic_audit(partner_results, deal_summary, sale_me):
    """Build full MOIC audit data for all partners + deal level.

    Returns dict with 'partners' list and 'deal_level' dict.
    """
    partners = []
    for pr in partner_results:
        breakdown, summary = build_moic_breakdown(pr.get('cashflow_details', []), pr)
        partners.append({
            'partner': pr['partner'],
            'breakdown': breakdown,
            'summary': summary,
        })

    return {
        'partners': partners,
        'deal_level': {
            'total_contributions': deal_summary.get('total_contributions', 0),
            'total_cf_distributions': deal_summary.get('total_cf_distributions', 0),
            'total_cap_distributions': deal_summary.get('total_cap_distributions', 0),
            'total_distributions': deal_summary.get('total_distributions', 0),
            'deal_moic': deal_summary.get('deal_moic', 0),
            'note': 'Deal MOIC uses realized distributions only (no unrealized NAV).',
        },
    }


# ============================================================
# Excel generators  (extracted from app.py)
# ============================================================

def generate_roe_audit_excel(partner_results, deal_summary, sale_me) -> bytes:
    """Generate formatted Excel workbook for ROE audit."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "ROE Audit"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    bold_font = Font(bold=True)
    top_border = Border(top=Side(style='medium'))
    curr_fmt = '$#,##0'
    pct_fmt = '0.00%'
    date_fmt = 'MM/DD/YYYY'
    num_fmt = '#,##0'

    row = 1

    def write_header(ws, r, cols):
        for ci, name in enumerate(cols, 1):
            c = ws.cell(row=r, column=ci, value=name)
            c.font = header_font
            c.fill = header_fill
            c.alignment = Alignment(horizontal="center")
        return r + 1

    def _write_roe_section(ws, row, tl_rows, cf_rows, summary, label):
        ws.cell(row=row, column=1, value=label).font = Font(bold=True, size=12)
        row += 1

        ws.cell(row=row, column=1, value="Capital Balance Timeline").font = bold_font
        row += 1
        tl_cols = ['Date', 'Event', 'Amount', 'Capital Balance', 'Days at Balance', 'Weighted Capital']
        row = write_header(ws, row, tl_cols)
        for tr in tl_rows:
            ws.cell(row=row, column=1, value=tr['Date']).number_format = date_fmt
            ws.cell(row=row, column=2, value=tr['Event'])
            if tr['Amount'] is not None:
                ws.cell(row=row, column=3, value=tr['Amount']).number_format = curr_fmt
            ws.cell(row=row, column=4, value=tr['Capital Balance']).number_format = curr_fmt
            ws.cell(row=row, column=5, value=tr['Days at Balance']).number_format = num_fmt
            ws.cell(row=row, column=6, value=tr['Weighted Capital']).number_format = curr_fmt
            row += 1
        ws.cell(row=row, column=1, value="Totals").font = bold_font
        ws.cell(row=row, column=1).border = top_border
        ws.cell(row=row, column=5, value=summary.get('total_days', 0)).font = bold_font
        ws.cell(row=row, column=5).number_format = num_fmt
        ws.cell(row=row, column=5).border = top_border
        ws.cell(row=row, column=6, value=summary.get('wac', 0)).font = bold_font
        ws.cell(row=row, column=6).number_format = curr_fmt
        ws.cell(row=row, column=6).border = top_border
        row += 2

        ws.cell(row=row, column=1, value="CF Distributions").font = bold_font
        row += 1
        row = write_header(ws, row, ['Date', 'Amount'])
        for cr in cf_rows:
            ws.cell(row=row, column=1, value=cr['Date']).number_format = date_fmt
            ws.cell(row=row, column=2, value=cr['Amount']).number_format = curr_fmt
            row += 1
        ws.cell(row=row, column=1, value="Total").font = bold_font
        ws.cell(row=row, column=1).border = top_border
        ws.cell(row=row, column=2, value=summary.get('total_cf_dist', 0)).font = bold_font
        ws.cell(row=row, column=2).number_format = curr_fmt
        ws.cell(row=row, column=2).border = top_border
        row += 2

        ws.cell(row=row, column=1, value="ROE Summary").font = bold_font
        row += 1
        row = write_header(ws, row, ['Inception', 'End Date', 'Total Days', 'Years',
                                      'CF Distributions', 'Wtd Avg Capital', 'ROE'])
        ws.cell(row=row, column=1, value=summary.get('inception')).number_format = date_fmt
        ws.cell(row=row, column=2, value=summary.get('end')).number_format = date_fmt
        ws.cell(row=row, column=3, value=summary.get('total_days', 0)).number_format = num_fmt
        ws.cell(row=row, column=4, value=round(summary.get('years', 0), 2))
        ws.cell(row=row, column=5, value=summary.get('total_cf_dist', 0)).number_format = curr_fmt
        ws.cell(row=row, column=6, value=summary.get('wac', 0)).number_format = curr_fmt
        ws.cell(row=row, column=7, value=summary.get('roe', 0)).number_format = pct_fmt
        row += 2
        return row

    # Per-partner sections
    for pr in partner_results:
        tl, cf, summary = build_roe_timeline(
            pr['combined_cashflows'], pr['cf_only_distributions'], sale_me)
        row = _write_roe_section(ws, row, tl, cf, summary, f"Partner: {pr['partner']}")

    # Deal-level
    all_cfs = deal_summary.get('all_combined_cashflows', [])
    all_cf_dist = []
    for pr in partner_results:
        all_cf_dist.extend(pr['cf_only_distributions'])
    tl, cf, summary = build_roe_timeline(all_cfs, all_cf_dist, sale_me)
    _write_roe_section(ws, row, tl, cf, summary, "Deal-Level ROE")

    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=0)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 30)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def generate_moic_audit_excel(partner_results, deal_summary, sale_me) -> bytes:
    """Generate formatted Excel workbook for MOIC audit."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "MOIC Audit"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    bold_font = Font(bold=True)
    top_border = Border(top=Side(style='medium'))
    curr_fmt = '$#,##0'
    mult_fmt = '0.00"x"'
    date_fmt = 'MM/DD/YYYY'

    row = 1

    def write_header(ws, r, cols):
        for ci, name in enumerate(cols, 1):
            c = ws.cell(row=r, column=ci, value=name)
            c.font = header_font
            c.fill = header_fill
            c.alignment = Alignment(horizontal="center")
        return r + 1

    for pr in partner_results:
        breakdown, summary = build_moic_breakdown(pr.get('cashflow_details', []), pr)
        ws.cell(row=row, column=1, value=f"Partner: {pr['partner']}").font = Font(bold=True, size=12)
        row += 1
        row = write_header(ws, row, ['Date', 'Description', 'Type', 'Amount'])
        for br in breakdown:
            ws.cell(row=row, column=1, value=br['Date']).number_format = date_fmt
            ws.cell(row=row, column=2, value=br['Description'])
            ws.cell(row=row, column=3, value=br['Type'])
            ws.cell(row=row, column=4, value=br['Amount']).number_format = curr_fmt
            row += 1
        row += 1

        ws.cell(row=row, column=1, value="MOIC Summary").font = bold_font
        row += 1
        row = write_header(ws, row, ['Contributions', 'CF Distributions', 'Cap Distributions',
                                      'Total Distributions', 'Unrealized NAV', 'MOIC'])
        ws.cell(row=row, column=1, value=summary['contributions']).number_format = curr_fmt
        ws.cell(row=row, column=2, value=summary['cf_dist']).number_format = curr_fmt
        ws.cell(row=row, column=3, value=summary['cap_dist']).number_format = curr_fmt
        ws.cell(row=row, column=4, value=summary['total_dist']).number_format = curr_fmt
        ws.cell(row=row, column=5, value=summary['unrealized']).number_format = curr_fmt
        ws.cell(row=row, column=6, value=summary['moic']).number_format = mult_fmt
        for ci in range(1, 7):
            ws.cell(row=row, column=ci).font = bold_font
        row += 2

    # Deal-level
    ws.cell(row=row, column=1, value="Deal-Level MOIC").font = Font(bold=True, size=12)
    row += 1
    ws.cell(row=row, column=1,
            value="Note: Deal MOIC uses realized distributions only (no unrealized NAV)."
            ).font = Font(italic=True)
    row += 1
    ds = deal_summary
    row = write_header(ws, row, ['Total Contributions', 'CF Distributions', 'Cap Distributions',
                                  'Total Distributions', 'MOIC'])
    ws.cell(row=row, column=1, value=ds.get('total_contributions', 0)).number_format = curr_fmt
    ws.cell(row=row, column=2, value=ds.get('total_cf_distributions', 0)).number_format = curr_fmt
    ws.cell(row=row, column=3, value=ds.get('total_cap_distributions', 0)).number_format = curr_fmt
    ws.cell(row=row, column=4, value=ds.get('total_distributions', 0)).number_format = curr_fmt
    ws.cell(row=row, column=5, value=ds.get('deal_moic', 0)).number_format = mult_fmt
    for ci in range(1, 6):
        ws.cell(row=row, column=ci).font = bold_font

    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=0)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 30)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
