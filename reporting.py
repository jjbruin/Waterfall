"""
reporting.py
Annual aggregation tables and formatting utilities
With integrated waterfall displays
"""

import pandas as pd
import streamlit as st
from typing import Optional

from config import (GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS, EXPENSE_ACCTS,
                    INTEREST_ACCTS, PRINCIPAL_ACCTS, CAPEX_ACCTS, OTHER_EXCLUDED_ACCTS)


def cashflows_monthly_fad(fc_deal_modeled_full: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly Funds Available for Distribution
    
    FAD = NOI + Interest + Principal + Excluded + Capex (all from mAmount_norm)
    Note: Interest, Principal, Capex should be negative (outflows)
    """
    f = fc_deal_modeled_full.copy()
    f["event_date"] = pd.to_datetime(f["event_date"]).dt.date
    f["me"] = f["event_date"].apply(lambda d: pd.Timestamp(d).to_period('M').to_timestamp('M').date())

    is_rev = f["vAccount"].isin(GROSS_REVENUE_ACCTS | CONTRA_REVENUE_ACCTS)
    is_exp = f["vAccount"].isin(EXPENSE_ACCTS)
    is_int = f["vAccount"].isin(INTEREST_ACCTS)
    is_prin = f["vAccount"].isin(PRINCIPAL_ACCTS)
    is_capex = f["vAccount"].isin(CAPEX_ACCTS)
    is_excl = f["vAccount"].isin(OTHER_EXCLUDED_ACCTS)

    f["rev"] = f["mAmount_norm"].where(is_rev, 0.0)
    f["exp"] = f["mAmount_norm"].where(is_exp, 0.0)
    f["noi"] = f["rev"] + f["exp"]
    f["int"] = f["mAmount_norm"].where(is_int, 0.0)
    f["prin"] = f["mAmount_norm"].where(is_prin, 0.0)
    f["capex"] = f["mAmount_norm"].where(is_capex, 0.0)
    f["excl"] = f["mAmount_norm"].where(is_excl, 0.0)

    g = f.groupby("me", as_index=False)[["noi", "int", "prin", "capex", "excl"]].sum()
    g["fad"] = g["noi"] + g["int"] + g["prin"] + g["capex"] + g["excl"]
    return g.rename(columns={"me": "event_date"})


def annual_aggregation_table(
    fc_deal_display: pd.DataFrame,
    start_year: int,
    horizon_years: int,
    proceeds_by_year: Optional[pd.Series] = None,
    cf_alloc: Optional[pd.DataFrame] = None,
    cap_alloc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create annual aggregation table with integrated waterfall details
    
    Args:
        fc_deal_display: Forecast data
        start_year: First year
        horizon_years: Number of years (e.g., 10)
        proceeds_by_year: Capital event proceeds by year
        cf_alloc: Cash flow waterfall allocations
        cap_alloc: Capital waterfall allocations
    
    Returns DataFrame with columns for each year
    """
    years = list(range(int(start_year), int(start_year) + int(horizon_years)))
    f = fc_deal_display[fc_deal_display["Year"].isin(years)].copy()

    def sum_where(mask: pd.Series) -> pd.Series:
        if f.empty:
            return pd.Series(dtype=float)
        return f.loc[mask].groupby("Year")["mAmount_norm"].sum()

    revenues = sum_where(f["vAccount"].isin(GROSS_REVENUE_ACCTS | CONTRA_REVENUE_ACCTS))
    expenses = sum_where(f["vAccount"].isin(EXPENSE_ACCTS))
    interest = sum_where(f["vAccount"].isin(INTEREST_ACCTS))
    principal = sum_where(f["vAccount"].isin(PRINCIPAL_ACCTS))
    capex = sum_where(f["vAccount"].isin(CAPEX_ACCTS))
    excluded_other = sum_where(f["vAccount"].isin(OTHER_EXCLUDED_ACCTS))

    out = pd.DataFrame({"Year": years}).set_index("Year")
    out["Revenues"] = revenues
    out["Expenses"] = expenses
    out["NOI"] = out["Revenues"].fillna(0.0) + out["Expenses"].fillna(0.0)
    out["Interest"] = interest
    out["Principal"] = principal
    out["Total Debt Service"] = out["Interest"].fillna(0.0) + out["Principal"].fillna(0.0)
    out["Excluded Accounts"] = excluded_other
    out["Capital Expenditures"] = capex
    out["Funds Available for Distribution"] = (
        out["NOI"].fillna(0.0) +
        out["Interest"].fillna(0.0) +
        out["Principal"].fillna(0.0) +
        out["Excluded Accounts"].fillna(0.0) +
        out["Capital Expenditures"].fillna(0.0)
    )

    tds_abs = out["Total Debt Service"].abs().replace(0, pd.NA)
    out["Debt Service Coverage Ratio"] = out["NOI"] / tds_abs

    # Add CF Waterfall rows
    if cf_alloc is not None and not cf_alloc.empty:
        out[""] = None  # Blank row separator
        out["CF Waterfall:"] = None  # Section header
        
        cf_alloc_copy = cf_alloc.copy()
        cf_alloc_copy["Year"] = pd.to_datetime(cf_alloc_copy["event_date"]).dt.year
        
        # Get unique steps in order
        step_cols = ['iOrder', 'vAmtType', 'PropCode', 'vState']
        available_cols = [c for c in step_cols if c in cf_alloc_copy.columns]
        
        if available_cols:
            cf_alloc_copy["StepKey"] = cf_alloc_copy.apply(
                lambda r: f"  {int(r.get('iOrder', 0)):>2} | {r.get('vAmtType', '')} | {r.get('PropCode', '')} | {r.get('vState', '')}",
                axis=1
            )
            
            # Pivot to get years as columns
            cf_pivot = cf_alloc_copy.pivot_table(
                index="StepKey",
                columns="Year",
                values="Allocated",
                aggfunc="sum",
                fill_value=0.0
            )
            
            # Sort by step order
            cf_alloc_copy_sorted = cf_alloc_copy.drop_duplicates("StepKey").sort_values("iOrder")
            step_order = cf_alloc_copy_sorted["StepKey"].tolist()
            cf_pivot = cf_pivot.reindex(step_order)
            
            # Add each step as a row
            for step_key in step_order:
                if step_key in cf_pivot.index:
                    for yr in years:
                        if yr in cf_pivot.columns:
                            out.loc[yr, step_key] = cf_pivot.loc[step_key, yr]
                        else:
                            out.loc[yr, step_key] = 0.0

    # Proceeds (capital events)
    if proceeds_by_year is None:
        proceeds_by_year = pd.Series(dtype=float)
    out["Proceeds from Sale or Refinancing"] = proceeds_by_year.reindex(out.index).fillna(0.0)

    # Add Capital Waterfall rows
    if cap_alloc is not None and not cap_alloc.empty:
        out[" "] = None  # Blank row separator (different key)
        out["Capital Waterfall:"] = None  # Section header
        
        cap_alloc_copy = cap_alloc.copy()
        cap_alloc_copy["Year"] = pd.to_datetime(cap_alloc_copy["event_date"]).dt.year
        
        # Get unique steps in order
        step_cols = ['iOrder', 'vAmtType', 'PropCode', 'vState']
        available_cols = [c for c in step_cols if c in cap_alloc_copy.columns]
        
        if available_cols:
            cap_alloc_copy["StepKey"] = cap_alloc_copy.apply(
                lambda r: f"  {int(r.get('iOrder', 0)):>2} | {r.get('vAmtType', '')} | {r.get('PropCode', '')} | {r.get('vState', '')}",
                axis=1
            )
            
            # Pivot to get years as columns
            cap_pivot = cap_alloc_copy.pivot_table(
                index="StepKey",
                columns="Year",
                values="Allocated",
                aggfunc="sum",
                fill_value=0.0
            )
            
            # Sort by step order
            cap_alloc_copy_sorted = cap_alloc_copy.drop_duplicates("StepKey").sort_values("iOrder")
            step_order = cap_alloc_copy_sorted["StepKey"].tolist()
            cap_pivot = cap_pivot.reindex(step_order)
            
            # Add each step as a row
            for step_key in step_order:
                if step_key in cap_pivot.index:
                    for yr in years:
                        if yr in cap_pivot.columns:
                            out.loc[yr, step_key] = cap_pivot.loc[step_key, yr]
                        else:
                            out.loc[yr, step_key] = 0.0

    # Add Partner Totals section
    if (cf_alloc is not None and not cf_alloc.empty) or (cap_alloc is not None and not cap_alloc.empty):
        out["  "] = None  # Blank row separator (different key)
        out["Partner Totals:"] = None  # Section header
        
        # Combine CF and Cap allocations
        all_alloc = []
        if cf_alloc is not None and not cf_alloc.empty:
            cf_copy = cf_alloc.copy()
            cf_copy["Year"] = pd.to_datetime(cf_copy["event_date"]).dt.year
            cf_copy["WaterfallType"] = "CF"
            all_alloc.append(cf_copy)
        
        if cap_alloc is not None and not cap_alloc.empty:
            cap_copy = cap_alloc.copy()
            cap_copy["Year"] = pd.to_datetime(cap_copy["event_date"]).dt.year
            cap_copy["WaterfallType"] = "Cap"
            all_alloc.append(cap_copy)
        
        if all_alloc:
            combined = pd.concat(all_alloc, ignore_index=True)
            
            # Get unique partners
            partners = sorted(combined["PropCode"].unique())
            
            for partner in partners:
                partner_data = combined[combined["PropCode"] == partner]
                partner_by_year = partner_data.groupby("Year")["Allocated"].sum()
                
                row_label = f"  {partner} Total"
                for yr in years:
                    if yr in partner_by_year.index:
                        out.loc[yr, row_label] = partner_by_year[yr]
                    else:
                        out.loc[yr, row_label] = 0.0
            
            # Grand total row
            total_by_year = combined.groupby("Year")["Allocated"].sum()
            for yr in years:
                if yr in total_by_year.index:
                    out.loc[yr, "  Total Distributions"] = total_by_year[yr]
                else:
                    out.loc[yr, "  Total Distributions"] = 0.0

    return out.reset_index().fillna(0.0)


def pivot_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert annual table to wide format (years as columns)"""
    wide = df.set_index("Year").T
    wide.index.name = "Line Item"
    
    # Define the order - base items first, then waterfall items will follow naturally
    base_order = [
        "Revenues", "Expenses", "NOI",
        "Interest", "Principal", "Total Debt Service",
        "Capital Expenditures", "Excluded Accounts",
        "Funds Available for Distribution", "Debt Service Coverage Ratio",
    ]
    
    # Get existing base items in order
    existing_base = [r for r in base_order if r in wide.index]
    
    # Get CF Waterfall section
    cf_header_idx = None
    cf_items = []
    if "CF Waterfall:" in wide.index:
        cf_header_idx = list(wide.index).index("CF Waterfall:")
        # Find items between CF Waterfall: and Proceeds or next section
        in_cf_section = False
        for item in wide.index:
            if item == "CF Waterfall:":
                in_cf_section = True
                cf_items.append(item)
            elif in_cf_section:
                if item in ["Proceeds from Sale or Refinancing", "Capital Waterfall:", " ", "  ", "Partner Totals:"]:
                    break
                if item not in ["", None] and item not in existing_base:
                    cf_items.append(item)
    
    # Proceeds row
    proceeds_items = []
    if "Proceeds from Sale or Refinancing" in wide.index:
        proceeds_items = ["Proceeds from Sale or Refinancing"]
    
    # Get Capital Waterfall section
    cap_items = []
    if "Capital Waterfall:" in wide.index:
        in_cap_section = False
        for item in wide.index:
            if item == "Capital Waterfall:":
                in_cap_section = True
                cap_items.append(item)
            elif in_cap_section:
                if item in ["Partner Totals:", "  "]:
                    break
                if item not in ["", " ", None] and item not in existing_base and item not in cf_items and item not in proceeds_items:
                    cap_items.append(item)
    
    # Get Partner Totals section
    partner_items = []
    if "Partner Totals:" in wide.index:
        in_partner_section = False
        for item in wide.index:
            if item == "Partner Totals:":
                in_partner_section = True
                partner_items.append(item)
            elif in_partner_section:
                if item not in ["", " ", "  ", None] and item not in existing_base and item not in cf_items and item not in proceeds_items and item not in cap_items:
                    partner_items.append(item)
    
    # Build final order
    final_order = existing_base
    
    # Add blank separator and CF Waterfall if exists
    if cf_items:
        if "" in wide.index:
            final_order.append("")
        final_order.extend(cf_items)
    
    # Add Proceeds
    final_order.extend(proceeds_items)
    
    # Add blank separator and Capital Waterfall if exists
    if cap_items:
        if " " in wide.index:
            final_order.append(" ")
        final_order.extend(cap_items)
    
    # Add blank separator and Partner Totals if exists
    if partner_items:
        if "  " in wide.index:
            final_order.append("  ")
        final_order.extend(partner_items)
    
    # Add any remaining items not yet included
    remainder = [r for r in wide.index if r not in final_order]
    final_order.extend(remainder)
    
    # Filter to only items that exist
    final_order = [r for r in final_order if r in wide.index]
    
    return wide.loc[final_order]


def style_annual_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply formatting to annual table with waterfall integration"""
    def money_fmt(x):
        if pd.isna(x) or x == "" or x is None:
            return ""
        try:
            return f"{float(x):,.0f}"
        except (ValueError, TypeError):
            return ""

    def dscr_fmt(x):
        if pd.isna(x) or x == "" or x is None:
            return ""
        try:
            return f"{float(x):,.2f}"
        except (ValueError, TypeError):
            return ""

    styler = df.style.format(money_fmt)

    # DSCR formatting
    if "Debt Service Coverage Ratio" in df.index:
        styler = styler.format(
            {col: dscr_fmt for col in df.columns},
            subset=pd.IndexSlice[["Debt Service Coverage Ratio"], :]
        )

    # Base styling
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left"), ("width", "280px")]},
            {"selector": "td", "props": [("text-align", "right"), ("width", "100px")]},
        ],
        overwrite=False,
    )

    # Expenses underline
    if "Expenses" in df.index:
        styler = styler.set_properties(subset=pd.IndexSlice[["Expenses"], :], **{"text-decoration": "underline"})

    # NOI bold with double underline
    if "NOI" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["NOI"], :],
            **{"border-bottom": "3px double black", "font-weight": "bold"}
        )

    # FAD styling
    if "Funds Available for Distribution" in df.index:
        fad_idx = list(df.index).index("Funds Available for Distribution")
        if fad_idx > 0:
            prev_row = df.index[fad_idx - 1]
            if prev_row not in ["", " ", "  ", None]:
                styler = styler.set_properties(subset=pd.IndexSlice[[prev_row], :], **{"border-bottom": "2px solid black"})
        styler = styler.set_properties(subset=pd.IndexSlice[["Funds Available for Distribution"], :], **{"font-weight": "bold"})

    # DSCR styling
    if "Debt Service Coverage Ratio" in df.index:
        styler = styler.set_properties(subset=pd.IndexSlice[["Debt Service Coverage Ratio"], :], **{"border-top": "1px solid #999"})

    # CF Waterfall header styling
    if "CF Waterfall:" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["CF Waterfall:"], :],
            **{"font-weight": "bold", "font-style": "italic", "border-top": "2px solid black"}
        )

    # Proceeds styling
    if "Proceeds from Sale or Refinancing" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Proceeds from Sale or Refinancing"], :],
            **{"border-top": "2px solid black", "font-weight": "bold"}
        )

    # Capital Waterfall header styling
    if "Capital Waterfall:" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Capital Waterfall:"], :],
            **{"font-weight": "bold", "font-style": "italic", "border-top": "2px solid black"}
        )

    # Partner Totals header styling
    if "Partner Totals:" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Partner Totals:"], :],
            **{"font-weight": "bold", "font-style": "italic", "border-top": "2px solid black"}
        )

    # Total Distributions styling
    if "  Total Distributions" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["  Total Distributions"], :],
            **{"font-weight": "bold", "border-top": "1px solid black"}
        )

    return styler


def pivot_waterfall_by_year(alloc_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot waterfall allocations to show years as columns"""
    if alloc_df is None or alloc_df.empty:
        return pd.DataFrame()

    df = alloc_df.copy()

    for c in ["iOrder", "vAmtType", "PropCode", "vtranstype", "nPercent", "FXRate", "Allocated", "event_date"]:
        if c not in df.columns:
            df[c] = ""

    df["Year"] = pd.to_datetime(df["event_date"]).dt.year

    df["StepLabel"] = df.apply(
        lambda r: (
            f"{int(r['iOrder']):>2} | "
            f"{r['vAmtType']} | "
            f"{r['PropCode']} | "
            f"{r['vtranstype']} | "
            f"n%={float(r['nPercent']):.4f} | "
            f"FX={float(r['FXRate']):.4f}"
        ),
        axis=1,
    )

    pv = (
        df.pivot_table(
            index=["iOrder", "StepLabel"],
            columns="Year",
            values="Allocated",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(level=0)
        .reset_index(level=0, drop=True)
        .reset_index()
        .rename(columns={"StepLabel": "Waterfall Step"})
    )

    return pv


def show_waterfall_matrix(title: str, alloc_df: pd.DataFrame):
    """Display waterfall allocations in Streamlit"""
    st.subheader(title)

    pv = pivot_waterfall_by_year(alloc_df)
    if pv.empty:
        st.info("No allocations to display.")
        return

    year_cols = [c for c in pv.columns if isinstance(c, int)]

    col_config = {}
    for y in year_cols:
        col_config[y] = st.column_config.NumberColumn(
            label=f"{y:>6}",
            format="%,.0f",
            width="small",
        )

    st.dataframe(
        pv,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )
