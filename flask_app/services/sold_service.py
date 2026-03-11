"""Sold Portfolio service — historical returns from accounting.

Extracts pure logic from sold_portfolio_ui.py (no Streamlit dependency).
"""

import pandas as pd
import numpy as np
import io
from typing import Optional

from loaders import normalize_accounting_feed, build_investmentid_to_vcode
from metrics import xirr, calculate_roe


def compute_all_sold_returns(inv_sold: pd.DataFrame, acct: pd.DataFrame,
                             inv: pd.DataFrame) -> pd.DataFrame:
    """Compute pref-equity returns for all sold deals from accounting.

    Extracted from sold_portfolio_ui.py::_compute_all_sold_returns().
    Filters out OP partners, case-insensitive InvestorID grouping,
    computes IRR/ROE/MOIC per deal, plus portfolio total from combined pool.

    Returns DataFrame with columns: Investment Name, Acquisition Date, Sale Date,
    Total Contributions, Total Distributions, IRR, ROE, MOIC, _is_deal_total, vcode.
    """
    if "is_contribution" not in acct.columns:
        acct = normalize_accounting_feed(acct)

    inv_to_vcode = build_investmentid_to_vcode(inv)
    vcode_to_iids = {}
    for iid, vc in inv_to_vcode.items():
        vcode_to_iids.setdefault(vc, []).append(iid)

    all_rows = []
    portfolio_cashflows = []
    portfolio_capital_events = []
    portfolio_cf_distributions = []

    for _, deal_row in inv_sold.iterrows():
        deal_name = str(deal_row.get("Investment_Name", "")).strip()
        deal_vcode = str(deal_row.get("vcode", "")).strip()

        acq_raw = deal_row.get("Acquisition_Date", None)
        sale_raw = deal_row.get("Sale_Date", None)
        acq_date = pd.to_datetime(acq_raw, errors="coerce")
        sale_date = pd.to_datetime(sale_raw, errors="coerce")
        acq_str = acq_date.strftime("%m/%d/%Y") if pd.notna(acq_date) else ""
        sale_str = sale_date.strftime("%m/%d/%Y") if pd.notna(sale_date) else ""

        deal_iids = set(vcode_to_iids.get(deal_vcode, []))
        direct_iid = str(deal_row.get("InvestmentID", "")).strip()
        if direct_iid:
            deal_iids.add(direct_iid)

        if not deal_iids or acct is None or acct.empty:
            continue

        deal_acct = acct[acct["InvestmentID"].isin(deal_iids)].copy()
        if deal_acct.empty:
            continue

        deal_acct = deal_acct[~deal_acct["InvestorID"].str.upper().str.startswith("OP")].copy()
        if deal_acct.empty:
            continue

        deal_acct["_investor_key"] = deal_acct["InvestorID"].str.upper()

        pref_cashflows = []
        pref_capital_events = []
        pref_cf_distributions = []

        for _, grp in deal_acct.groupby("_investor_key"):
            _, _, _, _, _, cashflows, capital_events, cf_dists = (
                _compute_partner_metrics(grp)
            )
            pref_cashflows.extend(cashflows)
            pref_capital_events.extend(capital_events)
            pref_cf_distributions.extend(cf_dists)

        if not pref_cashflows:
            continue

        contribs = sum(abs(a) for _, a in pref_cashflows if a < 0)
        distribs = sum(a for _, a in pref_cashflows if a > 0)
        irr_val = xirr(pref_cashflows) if len(pref_cashflows) >= 2 else None

        start = min(d for d, _ in pref_cashflows)
        end = max(d for d, _ in pref_cashflows)
        roe_val = calculate_roe(pref_capital_events, pref_cf_distributions, start, end)
        moic_val = distribs / contribs if contribs > 0 else 0.0

        all_rows.append({
            "Investment Name": deal_name,
            "vcode": deal_vcode,
            "Acquisition Date": acq_str,
            "Sale Date": sale_str,
            "Total Contributions": contribs,
            "Total Distributions": distribs,
            "IRR": irr_val,
            "ROE": roe_val,
            "MOIC": moic_val,
            "_is_deal_total": False,
        })

        portfolio_cashflows.extend(pref_cashflows)
        portfolio_capital_events.extend(pref_capital_events)
        portfolio_cf_distributions.extend(pref_cf_distributions)

    if not all_rows:
        return pd.DataFrame()

    # Portfolio total row
    port_contribs = sum(abs(a) for _, a in portfolio_cashflows if a < 0)
    port_distribs = sum(a for _, a in portfolio_cashflows if a > 0)
    port_irr = xirr(portfolio_cashflows) if len(portfolio_cashflows) >= 2 else None

    port_start = min(d for d, _ in portfolio_cashflows)
    port_end = max(d for d, _ in portfolio_cashflows)
    port_roe = calculate_roe(portfolio_capital_events, portfolio_cf_distributions, port_start, port_end)
    port_moic = port_distribs / port_contribs if port_contribs > 0 else 0.0

    all_rows.append({
        "Investment Name": "Portfolio Total",
        "vcode": "",
        "Acquisition Date": "",
        "Sale Date": "",
        "Total Contributions": port_contribs,
        "Total Distributions": port_distribs,
        "IRR": port_irr,
        "ROE": port_roe,
        "MOIC": port_moic,
        "_is_deal_total": True,
    })

    return pd.DataFrame(all_rows)


def build_deal_detail(vcode: str, inv_sold: pd.DataFrame, acct: pd.DataFrame,
                      inv: pd.DataFrame) -> dict:
    """Build cashflow detail table for a single sold deal.

    Returns dict with:
      - rows: list of dicts (Date, InvestorID, MajorType, Typename, Capital, Amount,
              Cashflow (XIRR), Capital Balance)
      - summary: dict with IRR, ROE, MOIC, Total Contributions, Total Distributions
    """
    if "is_contribution" not in acct.columns:
        acct = normalize_accounting_feed(acct)

    inv_to_vcode = build_investmentid_to_vcode(inv)
    vcode_to_iids = {}
    for iid, vc in inv_to_vcode.items():
        vcode_to_iids.setdefault(vc, []).append(iid)

    # Find deal by vcode
    match = inv_sold[inv_sold["vcode"].astype(str).str.strip() == str(vcode)]
    if match.empty:
        return {"rows": [], "summary": {}}

    deal_row = match.iloc[0]
    deal_vcode = str(deal_row.get("vcode", "")).strip()

    deal_iids = set(vcode_to_iids.get(deal_vcode, []))
    direct_iid = str(deal_row.get("InvestmentID", "")).strip()
    if direct_iid:
        deal_iids.add(direct_iid)

    if not deal_iids:
        return {"rows": [], "summary": {}}

    deal_acct = acct[acct["InvestmentID"].isin(deal_iids)].copy()
    if deal_acct.empty:
        return {"rows": [], "summary": {}}

    # Pref equity only
    deal_acct = deal_acct[~deal_acct["InvestorID"].str.upper().str.startswith("OP")].copy()
    if deal_acct.empty:
        return {"rows": [], "summary": {}}

    # Build detail rows
    rows = []
    for _, r in deal_acct.iterrows():
        amt = float(r["Amt"])
        if r["is_contribution"]:
            cashflow = -abs(amt)
        elif r["is_distribution"]:
            cashflow = abs(amt)
        else:
            cashflow = 0.0

        rows.append({
            "Date": str(r["EffectiveDate"]),
            "InvestorID": r["InvestorID"],
            "MajorType": r["MajorType"],
            "Typename": r.get("Typename", ""),
            "Capital": r["Capital"],
            "Amount": amt,
            "Cashflow (XIRR)": cashflow,
        })

    if not rows:
        return {"rows": [], "summary": {}}

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    # Running capital balance
    balance = 0.0
    balances = []
    for _, r in df.iterrows():
        cf = r["Cashflow (XIRR)"]
        if cf < 0:
            balance += abs(cf)
        elif r["Capital"] == "Y" and cf > 0:
            balance = max(0.0, balance - cf)
        balances.append(balance)
    df["Capital Balance"] = balances

    # Compute summary metrics — use original EffectiveDate (date objects) for XIRR
    # The df "Date" column is stringified for JSON; rebuild cashflows from deal_acct
    cashflows = []
    for _, r in deal_acct.iterrows():
        amt = float(r["Amt"])
        if r["is_contribution"]:
            cashflows.append((r["EffectiveDate"], -abs(amt)))
        elif r["is_distribution"]:
            cashflows.append((r["EffectiveDate"], abs(amt)))
    contribs = sum(abs(cf) for _, cf in cashflows if cf < 0)
    distribs = sum(cf for _, cf in cashflows if cf > 0)
    irr_val = xirr(cashflows) if len(cashflows) >= 2 else None
    moic_val = distribs / contribs if contribs > 0 else 0.0

    # ROE from capital events and CF distributions (using date objects from deal_acct)
    capital_events = []
    cf_dists = []
    for _, r in deal_acct.iterrows():
        amt = float(r["Amt"])
        if r["is_contribution"]:
            cf = -abs(amt)
            capital_events.append((r["EffectiveDate"], cf))
        elif r["is_distribution"]:
            cf = abs(amt)
            if r["is_capital"]:
                capital_events.append((r["EffectiveDate"], cf))
            else:
                cf_dists.append((r["EffectiveDate"], cf))

    if cashflows:
        start = min(d for d, _ in cashflows)
        end = max(d for d, _ in cashflows)
        roe_val = calculate_roe(capital_events, cf_dists, start, end)
    else:
        roe_val = 0.0

    detail_rows = df.to_dict(orient="records")
    summary = {
        "Total Contributions": contribs,
        "Total Distributions": distribs,
        "IRR": irr_val,
        "ROE": roe_val,
        "MOIC": moic_val,
    }

    return {"rows": detail_rows, "summary": summary}


def generate_sold_excel(df: pd.DataFrame) -> bytes:
    """Generate formatted Excel for sold portfolio summary.

    Extracted from sold_portfolio_ui.py::_generate_sold_excel().
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "Sold Portfolio Returns"

    display_cols = [c for c in df.columns if c not in ("_is_deal_total", "vcode")]

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

    for col_idx, col_name in enumerate(display_cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    bold_font = Font(bold=True)
    top_border = Border(top=Side(style="medium"))

    currency_cols = {"Total Contributions", "Total Distributions"}
    pct_cols = {"IRR", "ROE"}
    moic_cols = {"MOIC"}

    for row_idx, (_, row) in enumerate(df.iterrows(), 2):
        is_total = bool(row.get("_is_deal_total", False))
        for col_idx, col_name in enumerate(display_cols, 1):
            val = row[col_name]
            cell = ws.cell(row=row_idx, column=col_idx)

            if col_name in currency_cols:
                cell.value = float(val) if pd.notna(val) else 0.0
                cell.number_format = "$#,##0"
            elif col_name in pct_cols:
                cell.value = float(val) if pd.notna(val) else None
                cell.number_format = "0.00%"
            elif col_name in moic_cols:
                cell.value = float(val) if pd.notna(val) else 0.0
                cell.number_format = '0.00"x"'
            else:
                cell.value = val

            if is_total:
                cell.font = bold_font
                cell.border = top_border

    for col_idx, col_name in enumerate(display_cols, 1):
        max_len = len(str(col_name))
        for row_idx in range(2, len(df) + 2):
            cv = ws.cell(row=row_idx, column=col_idx).value
            if cv is not None:
                max_len = max(max_len, len(str(cv)))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 4, 30)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def generate_detail_excel(detail_rows: list[dict], deal_name: str,
                          summary: dict) -> bytes:
    """Generate formatted Excel for deal activity detail.

    Extracted from sold_portfolio_ui.py::_generate_detail_excel().
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Activity Detail"

    if not detail_rows:
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    df = pd.DataFrame(detail_rows)
    cols = list(df.columns)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    for ci, col in enumerate(cols, 1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    currency_cols = {"Amount", "Cashflow (XIRR)", "Capital Balance"}
    date_cols = {"Date"}

    for ri, (_, row) in enumerate(df.iterrows(), 2):
        for ci, col in enumerate(cols, 1):
            val = row[col]
            cell = ws.cell(row=ri, column=ci)
            if col in date_cols:
                cell.value = val
                cell.number_format = "MM/DD/YYYY"
            elif col in currency_cols:
                cell.value = float(val) if pd.notna(val) else 0.0
                cell.number_format = "$#,##0.00"
            else:
                cell.value = val

    # Summary section below
    summary_row = len(df) + 3
    bold = Font(bold=True)
    ws.cell(row=summary_row, column=1, value="Computed Returns").font = bold

    if summary:
        labels = [
            ("Total Contributions", summary.get("Total Contributions", 0), "$#,##0"),
            ("Total Distributions", summary.get("Total Distributions", 0), "$#,##0"),
            ("IRR", summary.get("IRR"), "0.00%"),
            ("ROE", summary.get("ROE"), "0.00%"),
            ("MOIC", summary.get("MOIC", 0), '0.00"x"'),
        ]
        for i, (label, val, fmt) in enumerate(labels):
            r = summary_row + 1 + i
            ws.cell(row=r, column=1, value=label).font = bold
            cell = ws.cell(row=r, column=2)
            cell.value = float(val) if val is not None and pd.notna(val) else None
            cell.number_format = fmt

    for ci, col in enumerate(cols, 1):
        max_len = len(str(col))
        for ri in range(2, len(df) + 2):
            cv = ws.cell(row=ri, column=ci).value
            if cv is not None:
                max_len = max(max_len, len(str(cv)))
        ws.column_dimensions[ws.cell(row=1, column=ci).column_letter].width = min(max_len + 4, 30)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _compute_partner_metrics(partner_acct):
    """Compute metrics for a single partner from accounting data.

    Returns:
        (contributions, distributions, irr, roe, moic,
         cashflows, capital_events, cf_distributions)
    """
    cashflows = []
    capital_events = []
    cf_distributions = []

    for _, r in partner_acct.iterrows():
        d = r["EffectiveDate"]
        amt = float(r["Amt"])

        if r["is_contribution"]:
            cf = -abs(amt)
            cashflows.append((d, cf))
            capital_events.append((d, cf))
        elif r["is_distribution"]:
            cf = abs(amt)
            cashflows.append((d, cf))
            if r["is_capital"]:
                capital_events.append((d, cf))
            else:
                cf_distributions.append((d, cf))

    contributions = sum(abs(a) for _, a in cashflows if a < 0)
    distributions = sum(a for _, a in cashflows if a > 0)

    irr_val = xirr(cashflows) if len(cashflows) >= 2 else None

    if cashflows:
        start = min(d for d, _ in cashflows)
        end = max(d for d, _ in cashflows)
        roe_val = calculate_roe(capital_events, cf_distributions, start, end)
    else:
        roe_val = 0.0

    moic_val = distributions / contributions if contributions > 0 else 0.0

    return contributions, distributions, irr_val, roe_val, moic_val, cashflows, capital_events, cf_distributions


def get_sold_deals(inv: pd.DataFrame) -> pd.DataFrame:
    """Filter investment map to sold deals, excluding child properties."""
    if "Sale_Status" not in inv.columns:
        return pd.DataFrame()

    inv_sold = inv[inv["Sale_Status"].fillna("").str.upper() == "SOLD"].copy()

    if "Portfolio_Name" in inv_sold.columns:
        inv_sold["Investment_Name"] = inv_sold["Investment_Name"].fillna("").astype(str).str.strip()
        inv_sold["Portfolio_Name"] = inv_sold["Portfolio_Name"].fillna("").astype(str).str.strip()
        parent_names = set(inv.loc[
            inv["Sale_Status"].fillna("").str.upper() == "SOLD", "Investment_Name"
        ].fillna("").astype(str).str.strip())
        is_child = (
            inv_sold["Portfolio_Name"].isin(parent_names)
            & (inv_sold["Portfolio_Name"] != inv_sold["Investment_Name"])
            & (inv_sold["Portfolio_Name"] != "")
        )
        inv_sold = inv_sold[~is_child].copy()

    return inv_sold
