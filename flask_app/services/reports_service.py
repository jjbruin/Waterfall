"""Reports service — partner returns, population selectors, Excel generation.

Extracts pure logic from reports_ui.py (no Streamlit dependency).
"""

import pandas as pd
import io
from typing import Optional
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill


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
    wf_norm.columns = [str(c).strip() for c in wf_norm.columns]
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
