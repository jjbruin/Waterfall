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


def compute_net_waterfall_for_deal(deal_acct: pd.DataFrame, inv: pd.DataFrame,
                                   deal_vcode: str, assumptions: dict) -> dict:
    """Run simplified net waterfall for a single deal from investor perspective.

    Walks through accounting events chronologically, applying AM fee, expenses,
    pref accrual, and promote split to compute net-of-fees cashflows.
    """
    if "is_contribution" not in deal_acct.columns:
        deal_acct = normalize_accounting_feed(deal_acct)

    ownership_pct = assumptions["ownership_pct"]
    am_fee_pct = assumptions["am_fee_pct"]
    hurdle_rate = assumptions["hurdle_rate"]
    promote_pct = assumptions["promote_pct"]
    annual_expenses = assumptions["annual_expenses"]

    # Filter to pref equity (no OP)
    pref = deal_acct[~deal_acct["InvestorID"].str.upper().str.startswith("OP")].copy()
    if pref.empty:
        return {"net_irr": None, "net_roe": 0.0, "net_moic": 0.0,
                "net_contributions": 0.0, "net_distributions": 0.0,
                "waterfall_detail": [], "net_cashflows": []}

    # Build sorted event list
    events = []
    for _, r in pref.iterrows():
        amt = float(r["Amt"])
        events.append({
            "date": r["EffectiveDate"],
            "is_contribution": bool(r["is_contribution"]),
            "is_distribution": bool(r["is_distribution"]),
            "is_capital": bool(r["is_capital"]),
            "gross_amount": amt,
        })
    events.sort(key=lambda e: e["date"])

    capital_balance = 0.0
    pref_accrued = 0.0
    last_date = None
    net_cashflows = []
    waterfall_detail = []

    for ev in events:
        d = ev["date"]
        gross = ev["gross_amount"]
        scaled = gross * ownership_pct

        if ev["is_contribution"]:
            contrib = abs(scaled)
            capital_balance += contrib
            net_cashflows.append((d, -contrib))
            waterfall_detail.append({
                "Date": str(d),
                "Event": "Contribution",
                "Gross Amount": -abs(gross),
                "Ownership %": ownership_pct,
                "Scaled Amount": -contrib,
                "AM Fee": 0.0,
                "Expenses": 0.0,
                "Available": 0.0,
                "Pref Accrued": 0.0,
                "Pref Paid": 0.0,
                "Capital Returned": 0.0,
                "Excess": 0.0,
                "Promote (GP)": 0.0,
                "Net to Investor": -contrib,
                "Capital Balance": capital_balance,
                "Pref Balance": pref_accrued,
            })
            last_date = d
            continue

        if ev["is_distribution"]:
            scaled_dist = abs(scaled)

            # Accrue pref since last event
            if last_date is not None and capital_balance > 0:
                days = (d - last_date).days
                if days > 0:
                    pref_accrued += capital_balance * hurdle_rate * days / 365.0

            # Compute fees for this period
            if last_date is not None:
                days_period = max((d - last_date).days, 0)
            else:
                days_period = 0
            am_fee = capital_balance * am_fee_pct * days_period / 365.0 if capital_balance > 0 else 0.0
            expenses = annual_expenses * days_period / 365.0

            # Available after fees (floor at 0)
            total_fees = am_fee + expenses
            if total_fees > scaled_dist:
                am_fee = scaled_dist * (am_fee / total_fees) if total_fees > 0 else 0.0
                expenses = scaled_dist - am_fee
                total_fees = scaled_dist
            available = scaled_dist - total_fees

            # Pay accrued pref
            pref_paid = min(pref_accrued, available)
            pref_accrued -= pref_paid
            remaining = available - pref_paid

            # Return capital if capital event
            capital_returned = 0.0
            if ev["is_capital"] and remaining > 0:
                capital_returned = min(capital_balance, remaining)
                capital_balance -= capital_returned
                remaining -= capital_returned

            # Promote split on excess
            gp_promote = remaining * promote_pct
            investor_share = remaining - gp_promote

            net_to_investor = pref_paid + capital_returned + investor_share
            net_cashflows.append((d, net_to_investor))

            waterfall_detail.append({
                "Date": str(d),
                "Event": "Capital Distribution" if ev["is_capital"] else "CF Distribution",
                "Gross Amount": abs(gross),
                "Ownership %": ownership_pct,
                "Scaled Amount": scaled_dist,
                "AM Fee": am_fee,
                "Expenses": expenses,
                "Available": available,
                "Pref Accrued": pref_accrued + pref_paid,  # show pre-payment balance
                "Pref Paid": pref_paid,
                "Capital Returned": capital_returned,
                "Excess": remaining + gp_promote,
                "Promote (GP)": gp_promote,
                "Net to Investor": net_to_investor,
                "Capital Balance": capital_balance,
                "Pref Balance": pref_accrued,
            })
            last_date = d

    # Compute net metrics
    if len(net_cashflows) >= 2:
        net_irr = xirr(net_cashflows)
    else:
        net_irr = None

    net_contribs = sum(abs(a) for _, a in net_cashflows if a < 0)
    net_distribs = sum(a for _, a in net_cashflows if a > 0)
    net_moic = net_distribs / net_contribs if net_contribs > 0 else 0.0

    # ROE: capital events = contributions + capital returns; cf_dists = pref + profit share
    net_capital_events = []
    net_cf_dists = []
    for row in waterfall_detail:
        d = pd.to_datetime(row["Date"])
        if row["Event"] == "Contribution":
            net_capital_events.append((d, -abs(row["Scaled Amount"])))
        else:
            if row["Capital Returned"] > 0:
                net_capital_events.append((d, row["Capital Returned"]))
            cf_portion = row["Pref Paid"] + row["Net to Investor"] - row["Capital Returned"] - row["Pref Paid"]
            # Actually: net_to_investor = pref_paid + capital_returned + investor_share
            # cf_dist = pref_paid + investor_share = net_to_investor - capital_returned
            cf_dist = row["Net to Investor"] - row["Capital Returned"]
            if cf_dist > 0:
                net_cf_dists.append((d, cf_dist))

    if net_cashflows:
        start = min(d for d, _ in net_cashflows)
        end = max(d for d, _ in net_cashflows)
        net_roe = calculate_roe(net_capital_events, net_cf_dists, start, end)
    else:
        net_roe = 0.0

    return {
        "net_irr": net_irr,
        "net_roe": net_roe,
        "net_moic": net_moic,
        "net_contributions": net_contribs,
        "net_distributions": net_distribs,
        "waterfall_detail": waterfall_detail,
        "net_cashflows": net_cashflows,
    }


def compute_all_net_returns(inv_sold: pd.DataFrame, acct: pd.DataFrame,
                            inv: pd.DataFrame, assumptions: dict) -> dict:
    """Compute net returns for all sold deals with fee/promote waterfall.

    Returns dict with deal_results (list), portfolio metrics, and assumptions.
    """
    if "is_contribution" not in acct.columns:
        acct = normalize_accounting_feed(acct)

    inv_to_vcode = build_investmentid_to_vcode(inv)
    vcode_to_iids = {}
    for iid, vc in inv_to_vcode.items():
        vcode_to_iids.setdefault(vc, []).append(iid)

    deal_results = []
    portfolio_net_cashflows = []

    for _, deal_row in inv_sold.iterrows():
        deal_name = str(deal_row.get("Investment_Name", "")).strip()
        deal_vcode = str(deal_row.get("vcode", "")).strip()

        deal_iids = set(vcode_to_iids.get(deal_vcode, []))
        direct_iid = str(deal_row.get("InvestmentID", "")).strip()
        if direct_iid:
            deal_iids.add(direct_iid)

        if not deal_iids or acct.empty:
            continue

        deal_acct = acct[acct["InvestmentID"].isin(deal_iids)].copy()
        if deal_acct.empty:
            continue

        result = compute_net_waterfall_for_deal(deal_acct, inv, deal_vcode, assumptions)

        if not result["net_cashflows"]:
            continue

        deal_results.append({
            "vcode": deal_vcode,
            "Investment Name": deal_name,
            "Net IRR": result["net_irr"],
            "Net ROE": result["net_roe"],
            "Net MOIC": result["net_moic"],
            "Net Contributions": result["net_contributions"],
            "Net Distributions": result["net_distributions"],
            "waterfall_detail": result["waterfall_detail"],
        })
        portfolio_net_cashflows.extend(result["net_cashflows"])

    # Portfolio-level metrics from pooled cashflows
    if portfolio_net_cashflows:
        port_irr = xirr(portfolio_net_cashflows) if len(portfolio_net_cashflows) >= 2 else None
        port_contribs = sum(abs(a) for _, a in portfolio_net_cashflows if a < 0)
        port_distribs = sum(a for _, a in portfolio_net_cashflows if a > 0)
        port_moic = port_distribs / port_contribs if port_contribs > 0 else 0.0

        # Portfolio ROE from pooled net capital events / cf dists
        port_net_capital = []
        port_net_cf = []
        for dr in deal_results:
            for row in dr["waterfall_detail"]:
                d = pd.to_datetime(row["Date"])
                if row["Event"] == "Contribution":
                    port_net_capital.append((d, -abs(row["Scaled Amount"])))
                else:
                    if row["Capital Returned"] > 0:
                        port_net_capital.append((d, row["Capital Returned"]))
                    cf_dist = row["Net to Investor"] - row["Capital Returned"]
                    if cf_dist > 0:
                        port_net_cf.append((d, cf_dist))

        port_start = min(d for d, _ in portfolio_net_cashflows)
        port_end = max(d for d, _ in portfolio_net_cashflows)
        port_roe = calculate_roe(port_net_capital, port_net_cf, port_start, port_end)
    else:
        port_irr = None
        port_contribs = 0.0
        port_distribs = 0.0
        port_moic = 0.0
        port_roe = 0.0

    return {
        "deal_results": deal_results,
        "portfolio": {
            "Net IRR": port_irr,
            "Net ROE": port_roe,
            "Net MOIC": port_moic,
            "Net Contributions": port_contribs,
            "Net Distributions": port_distribs,
        },
        "assumptions": assumptions,
    }


def generate_net_returns_excel(gross_df: pd.DataFrame, net_result: dict) -> bytes:
    """Generate multi-sheet Excel with gross+net summary and per-deal waterfall detail."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    bold_font = Font(bold=True)
    top_border = Border(top=Side(style="medium"))

    # Build summary columns
    summary_cols = [
        "Investment Name", "Acquisition Date", "Sale Date",
        "Gross Contributions", "Gross Distributions", "Gross IRR", "Gross ROE", "Gross MOIC",
        "",  # spacer
        "Net Contributions", "Net Distributions", "Net IRR", "Net ROE", "Net MOIC",
    ]

    for ci, col in enumerate(summary_cols, 1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Spacer column narrow
    spacer_idx = 9
    ws.column_dimensions[ws.cell(row=1, column=spacer_idx).column_letter].width = 2

    # Map net results by vcode
    net_by_vcode = {dr["vcode"]: dr for dr in net_result["deal_results"]}
    portfolio_net = net_result["portfolio"]

    currency_cols = {4, 5, 10, 11}  # 1-indexed
    pct_cols = {6, 7, 12, 13}
    moic_cols = {8, 14}

    row_idx = 2
    for _, grow in gross_df.iterrows():
        is_total = bool(grow.get("_is_deal_total", False))
        vcode = str(grow.get("vcode", ""))

        if is_total:
            net_data = portfolio_net
            net_contribs = net_data.get("Net Contributions", 0)
            net_distribs = net_data.get("Net Distributions", 0)
            net_irr = net_data.get("Net IRR")
            net_roe = net_data.get("Net ROE", 0)
            net_moic = net_data.get("Net MOIC", 0)
        elif vcode in net_by_vcode:
            nd = net_by_vcode[vcode]
            net_contribs = nd.get("Net Contributions", 0)
            net_distribs = nd.get("Net Distributions", 0)
            net_irr = nd.get("Net IRR")
            net_roe = nd.get("Net ROE", 0)
            net_moic = nd.get("Net MOIC", 0)
        else:
            net_contribs = net_distribs = net_irr = net_roe = net_moic = None

        values = [
            grow.get("Investment Name", ""),
            grow.get("Acquisition Date", ""),
            grow.get("Sale Date", ""),
            grow.get("Total Contributions", 0),
            grow.get("Total Distributions", 0),
            grow.get("IRR"),
            grow.get("ROE"),
            grow.get("MOIC", 0),
            "",  # spacer
            net_contribs,
            net_distribs,
            net_irr,
            net_roe,
            net_moic,
        ]

        for ci, val in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=ci)
            if ci == spacer_idx:
                cell.value = ""
            elif ci in currency_cols:
                cell.value = float(val) if val is not None and pd.notna(val) else 0.0
                cell.number_format = "$#,##0"
            elif ci in pct_cols:
                cell.value = float(val) if val is not None and pd.notna(val) else None
                cell.number_format = "0.00%"
            elif ci in moic_cols:
                cell.value = float(val) if val is not None and pd.notna(val) else 0.0
                cell.number_format = '0.00"x"'
            else:
                cell.value = val

            if is_total:
                cell.font = bold_font
                cell.border = top_border

        row_idx += 1

    # Assumptions footnote
    assumptions = net_result.get("assumptions", {})
    foot_row = row_idx + 1
    ws.cell(row=foot_row, column=1, value="Net Returns Assumptions:").font = bold_font
    labels = [
        f"Ownership: {assumptions.get('ownership_pct', 0) * 100:.1f}%",
        f"AM Fee: {assumptions.get('am_fee_pct', 0) * 100:.2f}%",
        f"Hurdle Rate: {assumptions.get('hurdle_rate', 0) * 100:.2f}%",
        f"Promote: {assumptions.get('promote_pct', 0) * 100:.1f}%",
        f"Annual Expenses: ${assumptions.get('annual_expenses', 0):,.0f}",
    ]
    ws.cell(row=foot_row + 1, column=1, value="  ".join(labels))

    # Auto-size columns
    for ci in range(1, len(summary_cols) + 1):
        if ci == spacer_idx:
            continue
        max_len = len(str(summary_cols[ci - 1]))
        for ri in range(2, row_idx):
            cv = ws.cell(row=ri, column=ci).value
            if cv is not None:
                max_len = max(max_len, len(str(cv)))
        ws.column_dimensions[ws.cell(row=1, column=ci).column_letter].width = min(max_len + 4, 30)

    # Per-deal waterfall sheets
    used_names = set()
    for dr in net_result["deal_results"]:
        detail = dr.get("waterfall_detail", [])
        if not detail:
            continue

        sheet_name = dr["Investment Name"][:31]
        # Ensure uniqueness
        base = sheet_name
        counter = 2
        while sheet_name in used_names:
            sheet_name = base[:28] + f" ({counter})"
            counter += 1
        used_names.add(sheet_name)

        dws = wb.create_sheet(title=sheet_name)
        detail_cols = [
            "Date", "Event", "Gross Amount", "Ownership %", "Scaled Amount",
            "AM Fee", "Expenses", "Available", "Pref Accrued", "Pref Paid",
            "Capital Returned", "Excess", "Promote (GP)", "Net to Investor",
            "Capital Balance", "Pref Balance",
        ]

        for ci, col in enumerate(detail_cols, 1):
            cell = dws.cell(row=1, column=ci, value=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        currency_detail = {3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        pct_detail = {4}

        for ri, row in enumerate(detail, 2):
            for ci, col in enumerate(detail_cols, 1):
                val = row.get(col, "")
                cell = dws.cell(row=ri, column=ci)
                if ci in pct_detail:
                    cell.value = float(val) if val else 0.0
                    cell.number_format = "0.00%"
                elif ci in currency_detail:
                    cell.value = float(val) if val is not None else 0.0
                    cell.number_format = "$#,##0"
                else:
                    cell.value = val

        # Summary below detail
        sr = len(detail) + 3
        dws.cell(row=sr, column=1, value="Net Metrics").font = bold_font
        metrics = [
            ("Net IRR", dr.get("Net IRR"), "0.00%"),
            ("Net ROE", dr.get("Net ROE", 0), "0.00%"),
            ("Net MOIC", dr.get("Net MOIC", 0), '0.00"x"'),
            ("Net Contributions", dr.get("Net Contributions", 0), "$#,##0"),
            ("Net Distributions", dr.get("Net Distributions", 0), "$#,##0"),
        ]
        for i, (label, val, fmt) in enumerate(metrics):
            r = sr + 1 + i
            dws.cell(row=r, column=1, value=label).font = bold_font
            cell = dws.cell(row=r, column=2)
            cell.value = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
            cell.number_format = fmt

        # Auto-size detail columns
        for ci, col in enumerate(detail_cols, 1):
            max_len = len(col)
            for ri in range(2, len(detail) + 2):
                cv = dws.cell(row=ri, column=ci).value
                if cv is not None:
                    max_len = max(max_len, len(str(cv)))
            dws.column_dimensions[dws.cell(row=1, column=ci).column_letter].width = min(max_len + 4, 22)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


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
