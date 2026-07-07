"""Reports service — partner returns, population selectors, Excel generation.

Extracts pure logic from reports_ui.py (no Streamlit dependency).
"""

import logging
import pandas as pd
import io
from datetime import date
from typing import Optional
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from utils import normalize_columns

log = logging.getLogger(__name__)


def build_partner_returns(deal_result: dict, deal_name: str) -> list[dict]:
    """Build partner + deal-total rows from a compute result.

    Thin wrapper around partner_results/deal_summary from compute.py.
    Returns list of dicts with columns: Deal Name, Partner, Contributions,
    CF Distributions, Capital Distributions, IRR, ROE, MOIC, _is_deal_total.
    """
    rows = []
    partner_results = deal_result.get("partner_results", [])
    for pr in partner_results:
        rows.append({
            "Deal Name": deal_name,
            "Partner": pr["partner"],
            "Contributions": pr["contributions"],
            "CF Distributions": pr["cf_distributions"],
            "Capital Distributions": pr["cap_distributions"],
            "IRR": pr["irr"],
            "ROE": pr["roe"],
            "MOIC": pr["moic"],
            "_is_deal_total": False,
        })

    ds = deal_result.get("deal_summary", {})
    rows.append({
        "Deal Name": deal_name,
        "Partner": "DEAL TOTAL",
        "Contributions": ds.get("total_contributions", 0),
        "CF Distributions": ds.get("total_cf_distributions", 0),
        "Capital Distributions": ds.get("total_cap_distributions", 0),
        "IRR": ds.get("deal_irr", None),
        "ROE": ds.get("deal_roe", None),
        "MOIC": ds.get("deal_moic", None),
        "_is_deal_total": True,
    })

    return rows


def build_deal_lookup(inv: pd.DataFrame, wf: pd.DataFrame) -> dict:
    """Build deal lookup tables for population selectors.

    Returns dict with:
      - eligible: list of {vcode, name, label} for deals with waterfall definitions
      - vcode_to_label: dict mapping vcode -> display label
    """
    inv_disp = inv.copy()

    # Exclude sold deals
    if "Sale_Status" in inv_disp.columns:
        inv_disp = inv_disp[inv_disp["Sale_Status"].fillna("").str.upper() != "SOLD"].copy()

    inv_disp["Investment_Name"] = inv_disp["Investment_Name"].fillna("").astype(str)
    inv_disp["vcode"] = inv_disp["vcode"].astype(str)

    name_counts = inv_disp["Investment_Name"].value_counts()
    inv_disp["DealLabel"] = inv_disp.apply(
        lambda r: (
            f"{r['Investment_Name']} ({r['vcode']})"
            if name_counts.get(r["Investment_Name"], 0) > 1
            else r["Investment_Name"]
        ),
        axis=1,
    )

    # Exclude child properties
    if "Portfolio_Name" in inv_disp.columns:
        inv_disp["Portfolio_Name"] = inv_disp["Portfolio_Name"].fillna("").astype(str).str.strip()
        parent_names = set(inv_disp["Investment_Name"].str.strip())
        is_child = (
            inv_disp["Portfolio_Name"].isin(parent_names)
            & (inv_disp["Portfolio_Name"] != inv_disp["Investment_Name"].str.strip())
            & (inv_disp["Portfolio_Name"] != "")
        )
        inv_disp = inv_disp[~is_child].copy()

    # Waterfall normalisation
    wf_norm = wf.copy()
    normalize_columns(wf_norm)
    if "vCode" in wf_norm.columns and "vcode" not in wf_norm.columns:
        wf_norm = wf_norm.rename(columns={"vCode": "vcode"})
    wf_norm["vcode"] = wf_norm["vcode"].astype(str)

    wf_vcodes = set(wf_norm["vcode"].unique())
    eligible = inv_disp[inv_disp["vcode"].isin(wf_vcodes)]

    eligible_list = sorted(
        [
            {"vcode": r["vcode"], "name": r["Investment_Name"], "label": r["DealLabel"]}
            for _, r in eligible.iterrows()
        ],
        key=lambda x: x["label"].lower(),
    )

    vcode_to_label = {r["vcode"]: r["DealLabel"] for _, r in eligible.iterrows()}

    return {
        "eligible": eligible_list,
        "vcode_to_label": vcode_to_label,
        "wf_norm": wf_norm,
        "eligible_vcodes": wf_vcodes & set(inv_disp["vcode"]),
    }


def get_partner_deals(wf: pd.DataFrame, eligible_vcodes: set,
                      vcode_to_label: dict) -> dict[str, list[str]]:
    """Build partner -> list of deal vcodes from waterfall PropCodes.

    Returns dict mapping partner_id -> sorted list of vcodes.
    """
    partner_to_vcodes: dict[str, set[str]] = {}

    for _, r in wf.iterrows():
        vc = str(r.get("vcode", ""))
        pc = str(r.get("PropCode", "")).strip()
        if vc in eligible_vcodes and pc:
            partner_to_vcodes.setdefault(pc, set()).add(vc)

    return {
        partner: sorted(vcodes)
        for partner, vcodes in sorted(partner_to_vcodes.items(), key=lambda x: x[0].lower())
    }


def get_upstream_investor_deals(
    relationships_raw: pd.DataFrame, inv: pd.DataFrame,
    eligible_vcodes: set
) -> dict[str, dict]:
    """Build upstream_investor -> list of deal vcodes from ownership tree.

    Returns dict mapping investor_id -> {name, vcodes: [...]}.
    """
    if relationships_raw is None or relationships_raw.empty:
        return {}

    from ownership_tree import load_relationships, build_ownership_tree, get_ultimate_investors
    from loaders import build_investmentid_to_vcode

    relationships = load_relationships(relationships_raw)

    # Filter out ended relationships (matching Review Tracking behaviour)
    if "EndDate" in relationships.columns:
        end_col = relationships["EndDate"]
        is_empty = end_col.isna() | (end_col.astype(str).str.strip().isin(["", "NaT", "nan", "None"]))
        relationships = relationships[is_empty].copy()

    nodes = build_ownership_tree(relationships)
    inv_to_vcode = build_investmentid_to_vcode(inv)

    investor_to_vcodes: dict[str, set[str]] = {}
    investor_names: dict[str, str] = {}

    for inv_id, vc in inv_to_vcode.items():
        if str(vc) not in eligible_vcodes:
            continue
        if inv_id not in nodes:
            continue
        ultimate = get_ultimate_investors(inv_id, nodes, normalize=True)
        for investor_id, _ in ultimate:
            investor_to_vcodes.setdefault(investor_id, set()).add(str(vc))
            node = nodes.get(investor_id)
            if node and hasattr(node, "name") and node.name:
                investor_names[investor_id] = node.name

    # Also include direct investors from relationships
    for _, rel_row in relationships.iterrows():
        inv_id = str(rel_row.get("InvestmentID", "")).strip()
        investor_id = str(rel_row.get("InvestorID", "")).strip()
        vc = inv_to_vcode.get(inv_id)
        if vc and str(vc) in eligible_vcodes and investor_id:
            investor_to_vcodes.setdefault(investor_id, set()).add(str(vc))

    result = {}
    for iid in sorted(investor_to_vcodes.keys(), key=lambda x: x.lower()):
        name = investor_names.get(iid, "")
        result[iid] = {
            "name": name,
            "display": f"{iid} — {name}" if name and name != iid else iid,
            "vcodes": sorted(investor_to_vcodes[iid]),
        }

    return result


def generate_returns_excel(df: pd.DataFrame) -> bytes:
    """Generate formatted Excel for Projected Returns Summary.

    Extracted from reports_ui.py::_generate_excel().
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Projected Returns"

    display_cols = [c for c in df.columns if c != "_is_deal_total"]
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

    for col_idx, col_name in enumerate(display_cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    bold_font = Font(bold=True)
    top_border = Border(top=Side(style="medium"))

    currency_cols = {"Contributions", "CF Distributions", "Capital Distributions"}
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
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max(len(col_name) + 2, 14)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ROE Summary Report
# ---------------------------------------------------------------------------

def _compute_accrued_pref(deal_acct: pd.DataFrame, report_date: date,
                          pref_rates: dict) -> float:
    """Compute total accrued (unpaid) pref across all PE investors through report_date.

    Walks accounting chronologically per investor, accrues pref daily on capital
    balance at the waterfall pref rate, reduces accrued by pref payments (TypeID 1019).
    Compounds at year-end with 45-day grace period (matching seed_states logic).
    """
    if not pref_rates:
        return 0.0

    total_accrued = 0.0

    for investor_id, grp in deal_acct.groupby("InvestorID"):
        if investor_id.upper().startswith("OP"):
            continue
        pref_rate = pref_rates.get(investor_id, 0.0)
        if pref_rate <= 0:
            continue

        rows = grp.sort_values("EffectiveDate")
        capital = 0.0
        pref_compounded = 0.0
        pref_cy = 0.0
        prev_date = None

        # Detect TypeID column
        has_typeid = "TypeID" in rows.columns

        for _, r in rows.iterrows():
            evt_date = r["EffectiveDate"].date() if pd.notna(r["EffectiveDate"]) else None
            if evt_date is None:
                continue
            amt = float(r["Amt"])
            major = r["MajorType"].lower()
            tname = r["TypeName"].lower()

            # Accrue pref from prev_date to this event
            if prev_date is not None and capital > 0 and evt_date > prev_date:
                cur = prev_date
                while cur < evt_date:
                    year_end = date(cur.year, 12, 31)
                    next_stop = min(evt_date, year_end)
                    days = (next_stop - cur).days
                    if days > 0:
                        base = max(0.0, capital + pref_compounded)
                        pref_cy += base * pref_rate * (days / 365.0)
                    if next_stop == year_end and next_stop < evt_date:
                        days_past = (evt_date - year_end).days
                        if days_past >= 45:
                            pref_compounded += pref_cy
                            pref_cy = 0.0
                        cur = date(cur.year + 1, 1, 1)
                    else:
                        break

            # Apply pref payment (TypeID 1019 = Preferred Return distribution)
            is_pref_payment = False
            if has_typeid:
                try:
                    is_pref_payment = float(r.get("TypeID", 0)) == 1019.0
                except (ValueError, TypeError):
                    pass
            if not is_pref_payment:
                is_pref_payment = "preferred return" in tname or "pref return" in tname
            if is_pref_payment and "distri" in major:
                payment = abs(amt)
                # Reduce compounded first, then current year
                if pref_compounded > 0:
                    pay = min(payment, pref_compounded)
                    pref_compounded -= pay
                    payment -= pay
                if pref_cy > 0 and payment > 0:
                    pay = min(payment, pref_cy)
                    pref_cy -= pay

            # Update capital balance
            if "contrib" in major:
                capital += abs(amt)
            elif "distri" in major:
                if "return of capital" in tname or "realized gain" in tname:
                    capital = max(0.0, capital - abs(amt))

            prev_date = evt_date

        # Accrue from last transaction to report_date
        if prev_date is not None and capital > 0 and report_date > prev_date:
            cur = prev_date
            while cur < report_date:
                year_end = date(cur.year, 12, 31)
                next_stop = min(report_date, year_end)
                days = (next_stop - cur).days
                if days > 0:
                    base = max(0.0, capital + pref_compounded)
                    pref_cy += base * pref_rate * (days / 365.0)
                if next_stop == year_end and next_stop < report_date:
                    days_past = (report_date - year_end).days
                    if days_past >= 45:
                        pref_compounded += pref_cy
                        pref_cy = 0.0
                    cur = date(cur.year + 1, 1, 1)
                else:
                    break

        total_accrued += pref_compounded + pref_cy

    return total_accrued


def build_roe_summary_row(
    vcode: str,
    deal_name: str,
    acct: pd.DataFrame,
    inv_map: pd.DataFrame,
    report_date: date,
    wf_steps: Optional[pd.DataFrame] = None,
    seed_states: Optional[dict] = None,
) -> Optional[dict]:
    """Build one ROE summary row from accounting data through report_date.

    Returns dict with: Deal Name, Total Funded, Return of Capital,
    Current Balance, Wtd Avg Balance, CF Received, Accrued Pref, ITD ROE.
    """
    from loaders import build_investmentid_to_vcode
    from metrics import calculate_roe_detailed

    vcode_str = str(vcode).strip()

    inv_to_vcode = build_investmentid_to_vcode(inv_map)
    deal_iids = [iid for iid, vc in inv_to_vcode.items() if str(vc) == vcode_str]
    if not deal_iids:
        return None

    acct_norm = acct.copy()
    normalize_columns(acct_norm)
    acct_norm["InvestmentID"] = acct_norm["InvestmentID"].astype(str).str.strip()
    acct_norm["EffectiveDate"] = pd.to_datetime(acct_norm["EffectiveDate"], errors="coerce")

    deal_acct = acct_norm[
        (acct_norm["InvestmentID"].isin(deal_iids))
        & (acct_norm["EffectiveDate"].dt.date <= report_date)
    ].copy()

    if deal_acct.empty:
        return None

    deal_acct["MajorType"] = deal_acct["MajorType"].fillna("").astype(str).str.strip()
    deal_acct["Amt"] = pd.to_numeric(deal_acct["Amt"], errors="coerce").fillna(0.0)
    if "TypeName" not in deal_acct.columns and "Typename" in deal_acct.columns:
        deal_acct["TypeName"] = deal_acct["Typename"]
    elif "TypeName" not in deal_acct.columns:
        deal_acct["TypeName"] = ""
    deal_acct["TypeName"] = deal_acct["TypeName"].fillna("").astype(str).str.strip()
    deal_acct["InvestorID"] = deal_acct["InvestorID"].astype(str).str.strip()

    funded = 0.0
    roc = 0.0
    capital_events = []
    cf_distributions = []

    for _, row in deal_acct.iterrows():
        if row["InvestorID"].upper().startswith("OP"):
            continue
        major = row["MajorType"].lower()
        tname = row["TypeName"].lower()
        amt = float(row["Amt"])
        evt_date = row["EffectiveDate"].date() if pd.notna(row["EffectiveDate"]) else None
        if evt_date is None:
            continue

        if "contrib" in major:
            funded += abs(amt)
            capital_events.append((evt_date, -abs(amt)))
        elif "distri" in major:
            capital_events.append((evt_date, abs(amt)))
            if "return of capital" in tname or "realized gain" in tname:
                roc += abs(amt)
            else:
                cf_distributions.append((evt_date, abs(amt)))

    if not capital_events:
        return None

    inception = min(d for d, _ in capital_events)
    detail = calculate_roe_detailed(capital_events, cf_distributions, inception, report_date)

    current_balance = funded - roc

    # Compute accrued pref directly from accounting + waterfall pref rates
    pref_rates = {}
    if wf_steps is not None and not wf_steps.empty:
        wf_deal = wf_steps[wf_steps["vcode"] == vcode_str] if "vcode" in wf_steps.columns else wf_steps
        pref_rows = wf_deal[wf_deal["vState"] == "Pref"] if "vState" in wf_deal.columns else pd.DataFrame()
        rate_col = "nPercent_dec" if "nPercent_dec" in pref_rows.columns else "nPercent"
        for _, pr in pref_rows.iterrows():
            pc = str(pr.get("PropCode", "")).strip()
            r = float(pr[rate_col]) if pd.notna(pr.get(rate_col)) else 0.0
            if pc and r > 0 and pc not in pref_rates:
                pref_rates[pc] = r

    accrued = _compute_accrued_pref(deal_acct, report_date, pref_rates)

    return {
        "Deal Name": deal_name,
        "Total Funded": funded,
        "Return of Capital": roc,
        "Current Balance": current_balance,
        "Wtd Avg Balance": detail["weighted_avg_capital"],
        "CF Received": detail["total_cf_distributions"],
        "Accrued Pref": accrued,
        "ITD ROE": detail["roe"],
    }


def generate_roe_summary_excel(df: pd.DataFrame) -> bytes:
    """Generate formatted Excel for ROE Summary report."""
    wb = Workbook()
    ws = wb.active
    ws.title = "ROE Summary"

    display_cols = [c for c in df.columns if not c.startswith("_")]
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

    for col_idx, col_name in enumerate(display_cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    currency_cols = {"Total Funded", "Return of Capital", "Current Balance",
                     "Wtd Avg Balance", "CF Received", "Accrued Pref"}
    pct_cols = {"ITD ROE"}

    for row_idx, (_, row) in enumerate(df.iterrows(), 2):
        for col_idx, col_name in enumerate(display_cols, 1):
            val = row[col_name]
            cell = ws.cell(row=row_idx, column=col_idx)
            if col_name in currency_cols:
                cell.value = float(val) if pd.notna(val) else 0.0
                cell.number_format = "$#,##0"
            elif col_name in pct_cols:
                cell.value = float(val) if pd.notna(val) else None
                cell.number_format = "0.00%"
            else:
                cell.value = val

    for col_idx, col_name in enumerate(display_cols, 1):
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max(len(col_name) + 2, 16)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
