"""Reports service — partner returns, population selectors, Excel generation.

Extracts pure logic from reports_ui.py (no Streamlit dependency).
"""

import logging
import calendar
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
    balance at the waterfall pref rate, reduces accrued by pref payments (TypeID 1019)
    and excess CF distributions. Compounds at year-end (always, no grace period).
    Uses Act/Act day count convention (366 for leap years).
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
            type_id = None
            if has_typeid:
                try:
                    type_id = float(r.get("TypeID", 0))
                except (ValueError, TypeError):
                    pass

            # Accrue pref from prev_date to this event (Act/Act)
            if prev_date is not None and capital > 0 and evt_date > prev_date:
                cur = prev_date
                while cur < evt_date:
                    year_end = date(cur.year, 12, 31)
                    next_stop = min(evt_date, year_end)
                    days = (next_stop - cur).days
                    if days > 0:
                        diy = _days_in_year(cur.year)
                        base = max(0.0, capital + pref_compounded)
                        pref_cy += base * pref_rate * (days / diy)
                    if next_stop == year_end and next_stop < evt_date:
                        # Always compound at year-end
                        pref_compounded += pref_cy
                        pref_cy = 0.0
                        cur = date(cur.year + 1, 1, 1)
                    else:
                        break

            # Apply pref payment (TypeID 1019 = Preferred Return distribution)
            is_pref_payment = (type_id == 1019.0)
            if not is_pref_payment:
                is_pref_payment = "preferred return" in tname or "pref return" in tname
            if is_pref_payment and "distri" in major:
                payment = abs(amt)
                if pref_compounded > 0:
                    pay = min(payment, pref_compounded)
                    pref_compounded -= pay
                    payment -= pay
                if pref_cy > 0 and payment > 0:
                    pay = min(payment, pref_cy)
                    pref_cy -= pay

            # Excess CF also reduces accrued pref
            is_excess_cf = "excess cash" in tname or type_id == 1020.0
            if is_excess_cf and "distri" in major and (pref_compounded + pref_cy) > 0:
                payment = abs(amt)
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

        # Accrue from last transaction to report_date (Act/Act)
        if prev_date is not None and capital > 0 and report_date > prev_date:
            cur = prev_date
            while cur < report_date:
                year_end = date(cur.year, 12, 31)
                next_stop = min(report_date, year_end)
                days = (next_stop - cur).days
                if days > 0:
                    diy = _days_in_year(cur.year)
                    base = max(0.0, capital + pref_compounded)
                    pref_cy += base * pref_rate * (days / diy)
                if next_stop == year_end and next_stop < report_date:
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


# ---------------------------------------------------------------------------
# Pref Balance Detail Report
# ---------------------------------------------------------------------------

def _days_in_year(yr: int) -> int:
    """Return 366 for leap years, 365 otherwise (Act/Act convention)."""
    return 366 if calendar.isleap(yr) else 365


def get_deal_pe_investors(
    vcode: str,
    acct: pd.DataFrame,
    inv_map: pd.DataFrame,
) -> list[dict]:
    """Return list of PE investors for a deal (excludes OP partners)."""
    from loaders import build_investmentid_to_vcode

    vcode_str = str(vcode).strip()
    inv_to_vcode = build_investmentid_to_vcode(inv_map)
    deal_iids = [iid for iid, vc in inv_to_vcode.items() if str(vc) == vcode_str]
    if not deal_iids:
        return []

    acct_norm = acct.copy()
    normalize_columns(acct_norm)
    acct_norm["InvestorID"] = acct_norm["InvestorID"].astype(str).str.strip()
    acct_norm["InvestmentID"] = acct_norm["InvestmentID"].astype(str).str.strip()

    deal_acct = acct_norm[acct_norm["InvestmentID"].isin(deal_iids)]
    investors = sorted(deal_acct["InvestorID"].unique())
    return [{"investor_id": iid} for iid in investors]


def _quarter_ends_between(start: date, end: date) -> list[date]:
    """Return all quarter-end dates strictly between start and end."""
    qends = []
    yr = start.year
    while yr <= end.year + 1:
        for m in (3, 6, 9, 12):
            d = date(yr, m, calendar.monthrange(yr, m)[1])
            if start < d < end:
                qends.append(d)
        yr += 1
    return qends


def build_pref_balance_detail(
    vcode: str,
    investor_id: str,
    report_date: date,
    acct: pd.DataFrame,
    inv_map: pd.DataFrame,
    wf_steps: Optional[pd.DataFrame] = None,
) -> dict:
    """Build pref balance detail matching the PE_Pref_Balances Excel layout.

    Returns dict with 'header' (summary info) and 'rows' (transaction detail).
    Uses Act/Act day count convention (366 for leap years).
    """
    from loaders import build_investmentid_to_vcode

    vcode_str = str(vcode).strip()
    inv_to_vcode = build_investmentid_to_vcode(inv_map)
    deal_iids = [iid for iid, vc in inv_to_vcode.items() if str(vc) == vcode_str]

    # Get pref rate from waterfall
    pref_rate = 0.0
    if wf_steps is not None and not wf_steps.empty:
        wf_deal = wf_steps[wf_steps["vcode"] == vcode_str] if "vcode" in wf_steps.columns else wf_steps
        pref_rows = wf_deal[wf_deal["vState"] == "Pref"] if "vState" in wf_deal.columns else pd.DataFrame()
        rate_col = "nPercent_dec" if "nPercent_dec" in pref_rows.columns else "nPercent"
        # Find the rate for this investor
        for _, pr in pref_rows.iterrows():
            pc = str(pr.get("PropCode", "")).strip()
            r = float(pr[rate_col]) if pd.notna(pr.get(rate_col)) else 0.0
            if pc.upper() == investor_id.upper() and r > 0:
                pref_rate = r
                break
        # Fallback: any Pref rate for this deal
        if pref_rate <= 0:
            for _, pr in pref_rows.iterrows():
                r = float(pr[rate_col]) if pd.notna(pr.get(rate_col)) else 0.0
                if r > 0:
                    pref_rate = r
                    break

    # Normalize accounting data
    acct_norm = acct.copy()
    normalize_columns(acct_norm)
    acct_norm["InvestorID"] = acct_norm["InvestorID"].astype(str).str.strip()
    acct_norm["InvestmentID"] = acct_norm["InvestmentID"].astype(str).str.strip()
    acct_norm["EffectiveDate"] = pd.to_datetime(acct_norm["EffectiveDate"], errors="coerce")
    acct_norm["Amt"] = pd.to_numeric(acct_norm["Amt"], errors="coerce").fillna(0.0)
    if "TypeName" not in acct_norm.columns and "Typename" in acct_norm.columns:
        acct_norm["TypeName"] = acct_norm["Typename"]
    elif "TypeName" not in acct_norm.columns:
        acct_norm["TypeName"] = ""
    acct_norm["TypeName"] = acct_norm["TypeName"].fillna("").astype(str).str.strip()
    acct_norm["MajorType"] = acct_norm["MajorType"].fillna("").astype(str).str.strip()

    deal_acct = acct_norm[
        (acct_norm["InvestmentID"].isin(deal_iids))
        & (acct_norm["InvestorID"].str.upper() == investor_id.upper())
        & (acct_norm["EffectiveDate"].dt.date <= report_date)
    ].sort_values("EffectiveDate").copy()

    if deal_acct.empty:
        return {"header": {}, "rows": []}

    # Detect TypeID column
    has_typeid = "TypeID" in deal_acct.columns

    # Build event list: actual transactions + generated quarter-end markers
    events = []  # (date, amt, typename, major_type, is_generated, type_id)
    for _, r in deal_acct.iterrows():
        evt_date = r["EffectiveDate"].date()
        tname_raw = r["TypeName"]
        # Skip Acquisition Fee rows (not in Excel pref balance model)
        if "acquisition fee" in tname_raw.lower():
            continue
        tid = None
        if has_typeid:
            try:
                tid = float(r["TypeID"])
            except (ValueError, TypeError):
                pass
        events.append((
            evt_date, float(r["Amt"]), tname_raw, r["MajorType"],
            False, tid,
        ))

    # Insert generated quarter-end rows between events and at report_date
    if events:
        first_date = events[0][0]
        all_dates = set(e[0] for e in events)
        qends = _quarter_ends_between(first_date, report_date)
        for qe in qends:
            if qe not in all_dates:
                events.append((qe, 0, "Generated", "", True, None))
        # Add report_date if not already present and not a quarter-end already added
        added_dates = set(e[0] for e in events)
        if report_date not in added_dates:
            events.append((report_date, 0, "Generated", "", True, None))

    def _evt_sort_key(e):
        """Sort: date, then contributions first, pref return, excess CF, generated last."""
        dt, _, tn, mt, is_gen, _ = e
        tn_lower = tn.lower()
        if is_gen:
            order = 90
        elif "contrib" in mt.lower():
            order = 10
        elif "preferred return" in tn_lower or "pref return" in tn_lower:
            order = 20
        elif "return of capital" in tn_lower or "realized gain" in tn_lower:
            order = 25
        elif "excess cash" in tn_lower:
            order = 30
        else:
            order = 50
        return (dt, order)

    events.sort(key=_evt_sort_key)

    # Walk through events row-by-row matching Excel PE_Pref_Balances formulas.
    # All "balance" columns use negative sign convention (owed = negative).
    # Excel columns: F=InvBal, G=CompPref, H=Inv+Comp, I=Days, J=CurrDue,
    #   K=AccrPref, L=TotalDue, M=PrefPaid, N=Remaining
    # Key formulas:
    #   H(row) = F(prior) + G(prior)          — accrual base is PRIOR row's values
    #   J(row) = H(row) * rate / diy * days
    #   K(row) = N(prior)                     — carry forward prior Remaining
    #   L(row) = J + K                        — total owed this period
    #   M(row) = payment if pref/excess CF    — positive (reduces debt)
    #   N(row) = MIN(0, L + M)                — remaining after payment
    #   G(row) = at 12/31: N(row)             — all unpaid pref compounds
    #            else: MIN(0, G(prior) + M + J) if (M+J)>0, else G(prior)

    inv_id = deal_iids[0] if deal_iids else ""

    # State variables matching Excel row values (stored as positive magnitudes)
    inv_bal = 0.0       # F: Investment Balance (positive = capital outstanding)
    comp_pref = 0.0     # G: Compounded Pref (positive = compounded pref owed)
    remaining = 0.0     # N: Remaining accrual (positive = pref owed)
    prev_date = None
    result_rows = []

    for evt_date, amt, typename, major_type, is_generated, type_id in events:
        major = major_type.lower()
        tname = typename.lower()

        # --- Update capital balance FIRST (affects InvBal for this row) ---
        if "contrib" in major:
            inv_bal += abs(amt)
        elif "distri" in major:
            if "return of capital" in tname or "realized gain" in tname:
                inv_bal = max(0.0, inv_bal - abs(amt))

        # --- H: Inv+Comp = PRIOR row's InvBal + PRIOR row's CompPref ---
        # On the first row there is no prior, so Inv+Comp = 0 (matches Excel)
        if not result_rows:
            inv_plus_comp = 0.0
        else:
            prior = result_rows[-1]
            inv_plus_comp = abs(prior["Investment_Balance"]) + abs(prior["Compounded Pref"])

        # --- I: Days since last event ---
        days_since = (evt_date - prev_date).days if prev_date else 0

        # --- J: Current Due = Inv+Comp * rate / diy * days ---
        # Excel uses simple formula: total days × rate / days_in_year(event year)
        # No splitting across year boundaries — matches Excel exactly.
        curr_due = 0.0
        if prev_date is not None and inv_plus_comp > 0 and days_since > 0:
            diy = _days_in_year(evt_date.year)
            curr_due = inv_plus_comp * pref_rate * (days_since / diy)

        # --- K: Accrued Pref = prior row's Remaining ---
        accr_pref = remaining  # carry forward from prior row

        # --- L: Total Due = Current Due + Accrued Pref ---
        total_due = curr_due + accr_pref

        # --- M: Pref Paid (positive = payment that reduces owed balance) ---
        pref_paid = 0.0
        is_pref_payment = (type_id == 1019.0) or "preferred return" in tname or "pref return" in tname
        is_excess_cf = (type_id == 1020.0) or "excess cash" in tname
        if "distri" in major and (is_pref_payment or is_excess_cf):
            pref_paid = abs(amt)

        # --- N: Remaining = MIN(0, TotalDue + PrefPaid) ---
        # TotalDue is positive (owed), PrefPaid is positive (paid).
        # In Excel sign convention both are negative/positive respectively,
        # but we track magnitudes: remaining_owed = max(0, total_due - pref_paid)
        remaining = max(0.0, total_due - pref_paid)

        # --- G: Compounded Pref ---
        is_year_end = evt_date.month == 12 and evt_date.day == 31
        if is_year_end:
            # At 12/31, all remaining unpaid pref compounds
            comp_pref = remaining
        else:
            # If (PrefPaid + CurrDue) net reduces the owed balance (payment > accrual),
            # the excess reduces compounded pref. Otherwise comp_pref unchanged.
            # Excel: MIN(0, G_prior + M + J) when (M+J)>0, else G_prior
            # In our magnitude convention: payment exceeds accrual → reduce comp_pref
            net_payment = pref_paid - curr_due  # positive if payment > accrual
            if net_payment > 0:
                comp_pref = max(0.0, comp_pref - net_payment)

        # --- Sign convention for display: negative = owed/invested ---
        display_amt = -abs(amt) if "contrib" in major else (abs(amt) if amt != 0 else 0)

        row = {
            "InvestmentID": inv_id,
            "InvestorID": investor_id,
            "EffectiveDate": evt_date.isoformat(),
            "Amt": display_amt,
            "Typename": typename,
            "Investment_Balance": -inv_bal if inv_bal > 0 else 0,
            "Compounded Pref": -comp_pref if comp_pref > 0 else 0,
            "Inv + Comp": -inv_plus_comp if inv_plus_comp > 0 else 0,
            "DaysSinceLast": days_since,
            "Current Due": -curr_due if curr_due > 0 else 0,
            "Accrued Pref": -accr_pref if accr_pref > 0 else 0,
            "Total Due": -total_due if total_due > 0 else 0,
            "Pref Paid": pref_paid,
            "Remaining Accrual": -remaining if remaining > 0 else 0,
        }
        result_rows.append(row)
        prev_date = evt_date

    # Build header
    annual_pref_est = inv_bal * pref_rate if inv_bal > 0 and pref_rate > 0 else 0
    total_accrued = remaining

    header = {
        "vcode": vcode_str,
        "investor_id": investor_id,
        "investment_id": inv_id,
        "pref_rate": pref_rate,
        "report_date": report_date.isoformat(),
        "investment_balance": inv_bal,
        "accrued_pref": total_accrued,
        "total": inv_bal + total_accrued,
        "annual_pref_est": annual_pref_est,
    }

    return {"header": header, "rows": result_rows}


def generate_pref_balance_excel(header: dict, rows: list[dict]) -> bytes:
    """Generate formatted Excel for Pref Balance Detail report."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Pref Balance Detail"

    bold = Font(bold=True)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    curr_fmt = '$#,##0'
    pct_fmt = '0.00%'

    # Header section
    ws.cell(row=1, column=1, value="Property #:").font = bold
    ws.cell(row=1, column=2, value=header.get("vcode", ""))
    ws.cell(row=1, column=4, value="As of:").font = bold
    ws.cell(row=1, column=5, value=header.get("report_date", ""))

    ws.cell(row=2, column=1, value="Investment ID:").font = bold
    ws.cell(row=2, column=2, value=header.get("investment_id", ""))
    ws.cell(row=2, column=4, value="Investment Balance:").font = bold
    c = ws.cell(row=2, column=5, value=header.get("investment_balance", 0))
    c.number_format = curr_fmt
    ws.cell(row=2, column=7, value="Annual Pref Est").font = bold
    c = ws.cell(row=2, column=8, value=header.get("annual_pref_est", 0))
    c.number_format = curr_fmt

    ws.cell(row=3, column=1, value="Investor:").font = bold
    ws.cell(row=3, column=2, value=header.get("investor_id", ""))
    ws.cell(row=3, column=4, value="Accrued:").font = bold
    c = ws.cell(row=3, column=5, value=header.get("accrued_pref", 0))
    c.number_format = curr_fmt

    ws.cell(row=4, column=1, value="Pref Rate:").font = bold
    c = ws.cell(row=4, column=2, value=header.get("pref_rate", 0))
    c.number_format = pct_fmt
    ws.cell(row=4, column=4, value="Total:").font = bold
    c = ws.cell(row=4, column=5, value=header.get("total", 0))
    c.number_format = curr_fmt

    # Column headers at row 6
    columns = [
        "InvestmentID", "InvestorID", "EffectiveDate", "Amt", "Typename",
        "Investment_Balance", "Compounded Pref", "Inv + Comp",
        "DaysSinceLast", "Current Due", "Accrued Pref", "Total Due",
        "Pref Paid", "Remaining Accrual",
    ]
    for ci, col in enumerate(columns, 1):
        cell = ws.cell(row=6, column=ci, value=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    currency_cols = {"Amt", "Investment_Balance", "Compounded Pref", "Inv + Comp",
                     "Current Due", "Accrued Pref", "Total Due", "Pref Paid",
                     "Remaining Accrual"}

    for ri, row in enumerate(rows, 7):
        for ci, col in enumerate(columns, 1):
            val = row.get(col)
            cell = ws.cell(row=ri, column=ci)
            if col in currency_cols:
                cell.value = float(val) if val is not None else 0.0
                cell.number_format = curr_fmt
            elif col == "DaysSinceLast":
                cell.value = int(val) if val is not None else 0
            else:
                cell.value = val

    # Auto-width
    for ci, col in enumerate(columns, 1):
        ws.column_dimensions[ws.cell(row=6, column=ci).column_letter].width = max(len(col) + 4, 16)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
