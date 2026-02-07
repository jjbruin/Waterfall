"""
one_pager_ui.py
Streamlit UI components for One Pager Investor Report

Provides:
- render_one_pager_section() main function
- Quarter selector dropdown
- Section renderers for General Info, Cap Stack, Property Performance, PE Performance
- Editable comments with save functionality
- NOI/Occupancy chart
- Print/Export buttons
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any, Optional

import altair as alt

from config import IS_ACCOUNTS
from one_pager import (
    get_available_quarters,
    get_general_information,
    get_capitalization_stack,
    get_property_performance,
    get_pe_performance,
    get_one_pager_comments,
    save_one_pager_comments,
    quarter_to_date_range,
)


def render_one_pager_section(
    vcode: str,
    inv_map: pd.DataFrame,
    isbs_df: pd.DataFrame,
    mri_loans: pd.DataFrame,
    mri_val: pd.DataFrame,
    waterfalls: pd.DataFrame,
    commitments: pd.DataFrame,
    acct: pd.DataFrame,
    occupancy_df: pd.DataFrame = None,
):
    """
    Render the complete One Pager Investor Report section

    Args:
        vcode: Deal vcode
        inv_map: Investment map DataFrame
        isbs_df: ISBS DataFrame with income statement data
        mri_loans: Loans DataFrame
        mri_val: Valuations DataFrame
        waterfalls: Waterfalls DataFrame
        commitments: Commitments DataFrame
        acct: Accounting feed DataFrame
        occupancy_df: Occupancy DataFrame (optional)
    """
    st.markdown("### One Pager Investor Report")

    # Quarter selector
    available_quarters = get_available_quarters(isbs_df)

    if not available_quarters:
        st.info("No quarterly data available. Upload ISBS_Download.csv with Interim IS data to generate One Pager reports.")
        return

    selected_quarter = st.selectbox(
        "Select Reporting Quarter",
        available_quarters,
        key=f"one_pager_quarter_{vcode}"
    )

    if not selected_quarter:
        return

    _, quarter_end = quarter_to_date_range(selected_quarter)

    # Get all data
    general_info = get_general_information(inv_map, vcode)
    cap_stack = get_capitalization_stack(vcode, mri_loans, mri_val, waterfalls, commitments, acct, inv_map)
    prop_perf = get_property_performance(vcode, selected_quarter, isbs_df, mri_val, occupancy_df)
    pe_perf = get_pe_performance(vcode, selected_quarter, acct, commitments, waterfalls, inv_map)
    comments = get_one_pager_comments(vcode, selected_quarter)

    # ============================================================
    # SECTION 1: GENERAL INFORMATION
    # ============================================================
    st.markdown("---")
    st.markdown("#### 1. GENERAL INFORMATION")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Partner:** {general_info['partner']}")
        st.markdown(f"**Asset Type:** {general_info['asset_type']}")
        st.markdown(f"**Location:** {general_info['location']}")

    with col2:
        units_sf = f"{general_info['units']:,} Units" if general_info['units'] else ""
        if general_info['sqft']:
            units_sf += f" | {general_info['sqft']:,} SF" if units_sf else f"{general_info['sqft']:,} SF"
        st.markdown(f"**Units | SF:** {units_sf}")

        date_closed = general_info['date_closed'].strftime('%m/%d/%Y') if general_info['date_closed'] else 'N/A'
        st.markdown(f"**Date Closed:** {date_closed}")

        st.markdown(f"**Year Built:** {general_info['year_built']}")

    with col3:
        st.markdown(f"**Investment Strategy:** {general_info['investment_strategy']}")
        anticipated_exit = general_info['anticipated_exit'].strftime('%m/%d/%Y') if general_info['anticipated_exit'] else 'N/A'
        st.markdown(f"**Anticipated Exit:** {anticipated_exit}")

    # ============================================================
    # SECTION 2: CAPITALIZATION / EXPOSURE / DEAL TERMS
    # ============================================================
    st.markdown("---")
    st.markdown("#### 2. CAPITALIZATION / EXPOSURE / DEAL TERMS")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Deal Terms**")
        st.markdown(f"- Purchase Price: ${cap_stack['purchase_price']:,.0f}" if cap_stack['purchase_price'] > 0 else "- Purchase Price: N/A")
        st.markdown(f"- P.E. Coupon: {cap_stack['pe_coupon']:.1%}" if cap_stack['pe_coupon'] > 0 else "- P.E. Coupon: N/A")
        st.markdown(f"- P.E. Participation: {cap_stack['pe_participation']:.1%}" if cap_stack['pe_participation'] > 0 else "- P.E. Participation: N/A")

    with col2:
        st.markdown("**Loan Terms**")
        maturity = cap_stack['loan_maturity'].strftime('%m/%d/%Y') if cap_stack['loan_maturity'] else 'N/A'
        st.markdown(f"- Maturity: {maturity}")
        st.markdown(f"- Rate: {cap_stack['loan_rate']:.2%}" if cap_stack['loan_rate'] > 0 else "- Rate: N/A")
        st.markdown(f"- Type: {cap_stack['loan_type']}" if cap_stack['loan_type'] else "- Type: N/A")

    with col3:
        st.markdown("**Current Values**")
        st.markdown(f"- Valuation: ${cap_stack['current_valuation']:,.0f}" if cap_stack['current_valuation'] > 0 else "- Valuation: N/A")
        st.markdown(f"- P.E. Exposure (Cap): {cap_stack['pe_exposure_on_cap']:.1%}" if cap_stack['pe_exposure_on_cap'] > 0 else "- P.E. Exposure (Cap): N/A")
        st.markdown(f"- P.E. Exposure (Value): {cap_stack['pe_exposure_on_value']:.1%}" if cap_stack['pe_exposure_on_value'] > 0 else "- P.E. Exposure (Value): N/A")

    # Capitalization Stack Table
    st.markdown("**Capitalization Stack**")
    cap_rows = [
        {'Layer': 'Senior Debt', 'Amount': cap_stack['debt'], '% of Cap': cap_stack['debt_pct']},
        {'Layer': 'Preferred Equity', 'Amount': cap_stack['pref_equity'], '% of Cap': cap_stack['pref_equity_pct']},
        {'Layer': 'Partner Equity', 'Amount': cap_stack['partner_equity'], '% of Cap': cap_stack['partner_equity_pct']},
        {'Layer': 'Total Capitalization', 'Amount': cap_stack['total_cap'], '% of Cap': 1.0},
    ]
    cap_df = pd.DataFrame(cap_rows)
    cap_df['Amount'] = cap_df['Amount'].apply(lambda x: f"${x:,.0f}")
    cap_df['% of Cap'] = cap_df['% of Cap'].apply(lambda x: f"{x:.1%}")

    st.dataframe(cap_df, use_container_width=True, hide_index=True)

    # ============================================================
    # SECTION 3: PROPERTY PERFORMANCE
    # ============================================================
    st.markdown("---")
    st.markdown(f"#### 3. PROPERTY PERFORMANCE (As of {quarter_end.strftime('%m/%d/%Y')})")

    # Performance table
    def format_money(val):
        if val is None or val == 0:
            return "-"
        return f"${val:,.0f}"

    def format_dscr(val):
        if val is None:
            return "-"
        return f"{val:.2f}x"

    def format_pct(val):
        if val is None:
            return "-"
        return f"{val:.1%}"

    def format_variance(val, is_pct=False):
        if val is None or val == 0:
            return "-"
        if is_pct:
            prefix = "+" if val > 0 else ""
            return f"{prefix}{val:.1%}"
        prefix = "+" if val > 0 else ""
        return f"{prefix}${val:,.0f}"

    perf_rows = []

    # Economic Occupancy
    econ_occ = prop_perf['economic_occ']
    perf_rows.append({
        'Metric': 'Economic Occupancy',
        'YTD Actual': format_pct(econ_occ['ytd_actual']),
        'YTD Budget': format_pct(econ_occ['ytd_budget']),
        'Variance': format_variance(econ_occ['variance'], is_pct=True) if econ_occ['variance'] else '-',
        'At Close': format_pct(econ_occ['at_close']),
        'Actual YE': format_pct(econ_occ['actual_ye']),
        'U/W YE': format_pct(econ_occ['uw_ye']),
    })

    # Revenue
    rev = prop_perf['revenue']
    perf_rows.append({
        'Metric': 'Revenue',
        'YTD Actual': format_money(rev['ytd_actual']),
        'YTD Budget': format_money(rev['ytd_budget']),
        'Variance': format_variance(rev['variance']),
        'At Close': format_money(rev['at_close']),
        'Actual YE': format_money(rev['actual_ye']),
        'U/W YE': format_money(rev['uw_ye']),
    })

    # Expenses
    exp = prop_perf['expenses']
    perf_rows.append({
        'Metric': 'Expenses',
        'YTD Actual': format_money(exp['ytd_actual']),
        'YTD Budget': format_money(exp['ytd_budget']),
        'Variance': format_variance(-exp['variance']),  # Negative variance is good for expenses
        'At Close': format_money(exp['at_close']),
        'Actual YE': format_money(exp['actual_ye']),
        'U/W YE': format_money(exp['uw_ye']),
    })

    # NOI
    noi = prop_perf['noi']
    perf_rows.append({
        'Metric': 'NOI',
        'YTD Actual': format_money(noi['ytd_actual']),
        'YTD Budget': format_money(noi['ytd_budget']),
        'Variance': format_variance(noi['variance']),
        'At Close': format_money(noi['at_close']),
        'Actual YE': format_money(noi['actual_ye']),
        'U/W YE': format_money(noi['uw_ye']),
    })

    # DSCR
    dscr = prop_perf['dscr']
    perf_rows.append({
        'Metric': 'DSCR',
        'YTD Actual': format_dscr(dscr['ytd_actual']),
        'YTD Budget': format_dscr(dscr['ytd_budget']),
        'Variance': f"{dscr['variance']:+.2f}x" if dscr['variance'] else '-',
        'At Close': format_dscr(dscr['at_close']),
        'Actual YE': format_dscr(dscr['actual_ye']),
        'U/W YE': format_dscr(dscr['uw_ye']),
    })

    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Economic Comments
    st.markdown("**Comments:**")
    econ_comments = st.text_area(
        "Property Performance Comments",
        value=comments.get('econ_comments', ''),
        height=100,
        key=f"econ_comments_{vcode}_{selected_quarter}",
        label_visibility="collapsed"
    )

    # ============================================================
    # SECTION 4: PREFERRED EQUITY PERFORMANCE
    # ============================================================
    st.markdown("---")
    st.markdown("#### 4. PREFERRED EQUITY PERFORMANCE")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Commitment & Funding**")
        st.metric("Committed P.E.", f"${pe_perf['committed_pe']:,.0f}")
        st.metric("Funded to Date", f"${pe_perf['funded_to_date']:,.0f}")
        st.metric("Remaining to Fund", f"${pe_perf['remaining_to_fund']:,.0f}")

    with col2:
        st.markdown("**Returns**")
        st.metric("Coupon", f"{pe_perf['coupon']:.1%}" if pe_perf['coupon'] > 0 else "N/A")
        st.metric("Participation", f"{pe_perf['participation']:.1%}" if pe_perf['participation'] > 0 else "N/A")
        st.metric("Return of Capital", f"${pe_perf['return_of_capital']:,.0f}")

    # Current balance row
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Current P.E. Balance", f"${pe_perf['current_pe_balance']:,.0f}")
    with col4:
        st.metric("Accrued Balance", f"${pe_perf['accrued_balance']:,.0f}")

    # Accrued Pref Comment
    st.markdown("**Accrued Pref Comment:**")
    accrued_pref_comment = st.text_area(
        "Accrued Pref Comment",
        value=comments.get('accrued_pref_comment', ''),
        height=80,
        key=f"accrued_pref_comment_{vcode}_{selected_quarter}",
        label_visibility="collapsed"
    )

    # ============================================================
    # SECTION 5: BUSINESS PLAN & UPDATES
    # ============================================================
    st.markdown("---")
    st.markdown("#### 5. BUSINESS PLAN & UPDATES")

    business_plan_comments = st.text_area(
        "Business Plan & Updates",
        value=comments.get('business_plan_comments', ''),
        height=150,
        key=f"business_plan_{vcode}_{selected_quarter}",
        label_visibility="collapsed"
    )

    # Save button
    if st.button("Save Comments", key=f"save_comments_{vcode}_{selected_quarter}"):
        success = save_one_pager_comments(
            vcode=vcode,
            reporting_period=selected_quarter,
            econ_comments=econ_comments,
            business_plan_comments=business_plan_comments,
            accrued_pref_comment=accrued_pref_comment
        )
        if success:
            st.success("Comments saved successfully!")
        else:
            st.error("Failed to save comments. Please ensure the database table exists.")

    # ============================================================
    # SECTION 6: OCCUPANCY vs NOI CHART (Trailing 12 Quarters)
    # ============================================================
    st.markdown("---")
    st.markdown("#### 6. OCCUPANCY vs NOI (Trailing 12 Quarters)")

    chart_data = _build_quarterly_noi_chart(vcode, isbs_df, occupancy_df, num_quarters=12)

    # ============================================================
    # EXPORT BUTTONS
    # ============================================================
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        # Export to CSV
        export_data = _build_export_data(
            general_info, cap_stack, prop_perf, pe_perf,
            selected_quarter, comments
        )
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv,
            file_name=f"one_pager_{vcode}_{selected_quarter}.csv",
            mime="text/csv",
            key=f"export_csv_{vcode}_{selected_quarter}"
        )

    with col2:
        # Print button
        if st.button("Print Report", key=f"print_{vcode}_{selected_quarter}"):
            _generate_print_html(
                vcode, selected_quarter, general_info, cap_stack,
                prop_perf, pe_perf, comments, chart_data
            )


def _build_quarterly_noi_chart(deal_vcode, isbs_raw, occupancy_raw, num_quarters=12):
    """Build and render a quarterly NOI + Occupancy chart (same style as performance chart).

    Returns a DataFrame with columns Quarter, Occupancy, NOI_Actual, NOI_UW
    for use by the print-report function.
    """
    empty_df = pd.DataFrame(columns=['Quarter', 'Occupancy', 'NOI_Actual', 'NOI_UW'])

    if isbs_raw is None or isbs_raw.empty:
        st.info("Insufficient data available for chart.")
        return empty_df

    # --- Prepare IS data ---
    isbs = isbs_raw.copy()
    isbs.columns = [str(c).strip() for c in isbs.columns]

    if 'vcode' in isbs.columns:
        isbs['vcode'] = isbs['vcode'].astype(str).str.strip().str.lower()
        isbs = isbs[isbs['vcode'] == str(deal_vcode).strip().lower()]

    if isbs.empty or 'dtEntry' not in isbs.columns:
        st.info("Insufficient data available for chart.")
        return empty_df

    try:
        isbs['dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
    except Exception:
        isbs['dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], errors='coerce')
    null_dates = isbs['dtEntry_parsed'].isna()
    if null_dates.any():
        isbs.loc[null_dates, 'dtEntry_parsed'] = pd.to_datetime(isbs.loc[null_dates, 'dtEntry'], errors='coerce')

    if 'vSource' in isbs.columns:
        isbs['vSource'] = isbs['vSource'].astype(str).str.strip()
    if 'vAccount' in isbs.columns:
        isbs['vAccount'] = isbs['vAccount'].astype(str).str.strip()
    if 'mAmount' in isbs.columns:
        isbs['mAmount'] = pd.to_numeric(isbs['mAmount'], errors='coerce').fillna(0)

    actual_data = isbs[isbs['vSource'] == 'Interim IS']
    uw_data = isbs[isbs['vSource'] == 'Projected IS']

    if actual_data.empty and uw_data.empty:
        st.info("Insufficient data available for chart.")
        return empty_df

    # Flatten revenue & expense account codes
    rev_accounts = [a for accts in IS_ACCOUNTS['REVENUES'].values() for a in accts]
    exp_accounts = [a for accts in IS_ACCOUNTS['EXPENSES'].values() for a in accts]

    def _compute_cumulative_noi(data, dates):
        noi_by_date = {}
        for dt in dates:
            period = data[data['dtEntry_parsed'] == dt]
            rev = period[period['vAccount'].isin(rev_accounts)]['mAmount'].sum()
            exp = period[period['vAccount'].isin(exp_accounts)]['mAmount'].sum()
            noi_by_date[dt] = (-rev) - exp
        return noi_by_date

    actual_dates = sorted(actual_data['dtEntry_parsed'].dropna().unique())
    uw_dates = sorted(uw_data['dtEntry_parsed'].dropna().unique())

    actual_cum = _compute_cumulative_noi(actual_data, actual_dates)
    uw_cum = _compute_cumulative_noi(uw_data, uw_dates)

    def _cumulative_to_periodic(cum_dict, sorted_dates):
        periodic = {}
        for i, dt in enumerate(sorted_dates):
            dt_ts = pd.Timestamp(dt)
            if dt_ts.month == 1:
                periodic[dt_ts] = cum_dict[dt]
            else:
                prior = None
                for j in range(i - 1, -1, -1):
                    p = pd.Timestamp(sorted_dates[j])
                    if p.year == dt_ts.year:
                        prior = sorted_dates[j]
                        break
                periodic[dt_ts] = cum_dict[dt] - cum_dict[prior] if prior is not None else cum_dict[dt]
        return periodic

    actual_periodic = _cumulative_to_periodic(actual_cum, actual_dates)
    uw_periodic = _cumulative_to_periodic(uw_cum, uw_dates)

    # Aggregate to quarterly (sum 3 months per quarter, only complete quarters)
    def _to_quarterly(periodic_dict):
        quarterly = {}
        month_counts = {}
        for dt, val in sorted(periodic_dict.items()):
            dt_ts = pd.Timestamp(dt)
            q_month = ((dt_ts.month - 1) // 3 + 1) * 3
            q_end = pd.Timestamp(year=dt_ts.year, month=q_month, day=1) + pd.offsets.MonthEnd(0)
            quarterly[q_end] = quarterly.get(q_end, 0) + val
            month_counts[q_end] = month_counts.get(q_end, 0) + 1
        return {k: v for k, v in quarterly.items() if month_counts.get(k, 0) == 3}

    actual_q = _to_quarterly(actual_periodic)
    uw_q = _to_quarterly(uw_periodic)

    all_q_ends = sorted(set(actual_q.keys()) | set(uw_q.keys()))
    if not all_q_ends:
        st.info("Not enough quarterly data for chart.")
        return empty_df

    # Take trailing N quarters
    display_dates = all_q_ends[-num_quarters:]

    # --- Occupancy data ---
    has_occupancy = False
    occ_by_date = {}
    if occupancy_raw is not None and not occupancy_raw.empty:
        occ = occupancy_raw.copy()
        occ.columns = [str(c).strip() for c in occ.columns]
        if 'vCode' in occ.columns:
            occ['vCode'] = occ['vCode'].astype(str).str.strip().str.lower()
            occ = occ[occ['vCode'] == str(deal_vcode).strip().lower()]

        occ_col = 'Occ%' if 'Occ%' in occ.columns else (
            'OccupancyPercent' if 'OccupancyPercent' in occ.columns else None)

        if not occ.empty and occ_col and 'dtReported' in occ.columns:
            occ['occ_val'] = pd.to_numeric(occ[occ_col], errors='coerce')
            try:
                occ['date_parsed'] = pd.to_datetime(
                    occ['dtReported'], unit='D', origin='1899-12-30', errors='coerce')
            except Exception:
                occ['date_parsed'] = pd.to_datetime(occ['dtReported'], errors='coerce')
            occ = occ.dropna(subset=['date_parsed', 'occ_val'])

            if not occ.empty:
                monthly_occ = {}
                for _, row in occ.iterrows():
                    me = pd.Timestamp(row['date_parsed']) + pd.offsets.MonthEnd(0)
                    monthly_occ[me] = row['occ_val']

                for dt in display_dates:
                    dt_ts = pd.Timestamp(dt)
                    q_start_month = ((dt_ts.month - 1) // 3) * 3 + 1
                    vals = []
                    for m in range(q_start_month, q_start_month + 3):
                        me = pd.Timestamp(year=dt_ts.year, month=m, day=1) + pd.offsets.MonthEnd(0)
                        if me in monthly_occ:
                            vals.append(monthly_occ[me])
                    if vals:
                        occ_by_date[dt_ts] = sum(vals) / len(vals)
                if occ_by_date:
                    has_occupancy = True

    # --- Build chart dataframe ---
    chart_rows = []
    for dt in display_dates:
        dt_ts = pd.Timestamp(dt)
        label = f"Q{(dt_ts.month - 1) // 3 + 1} {dt_ts.year}"
        chart_rows.append({
            'Quarter': label,
            'Actual NOI': actual_q.get(dt_ts, None),
            'Underwritten NOI': uw_q.get(dt_ts, None),
            'Occupancy': occ_by_date.get(dt_ts, None),
        })

    chart_df = pd.DataFrame(chart_rows)
    period_order = chart_df['Quarter'].tolist()

    # --- Altair chart ---
    noi_df = chart_df.melt(id_vars=['Quarter'], value_vars=['Actual NOI', 'Underwritten NOI'],
                           var_name='Series', value_name='NOI')
    noi_df = noi_df.dropna(subset=['NOI'])

    if noi_df.empty:
        st.info("Insufficient data available for chart.")
        # Still return the df for print compatibility
        return _to_print_df(chart_df)

    color_scale = alt.Scale(
        domain=['Actual NOI', 'Underwritten NOI'],
        range=['#1F4E79', '#ED7D31']
    )
    dash_scale = alt.Scale(
        domain=['Actual NOI', 'Underwritten NOI'],
        range=[[0], [5, 5]]
    )

    if has_occupancy:
        occ_df = chart_df[['Quarter', 'Occupancy']].dropna(subset=['Occupancy'])
        bars = alt.Chart(occ_df).mark_bar(color='#B4D4F0', opacity=0.6).encode(
            x=alt.X('Quarter:N', sort=period_order, title='Quarter'),
            y=alt.Y('Occupancy:Q', title='Occupancy %', scale=alt.Scale(domain=[0, 100])),
        )
        lines = alt.Chart(noi_df).mark_line(point=True).encode(
            x=alt.X('Quarter:N', sort=period_order),
            y=alt.Y('NOI:Q', title='NOI ($)'),
            color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash('Series:N', scale=dash_scale, legend=alt.Legend(title=None)),
        )
        chart = alt.layer(bars, lines).resolve_scale(y='independent').properties(height=300)
    else:
        chart = alt.Chart(noi_df).mark_line(point=True).encode(
            x=alt.X('Quarter:N', sort=period_order, title='Quarter'),
            y=alt.Y('NOI:Q', title='NOI ($)'),
            color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash('Series:N', scale=dash_scale, legend=alt.Legend(title=None)),
        ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    return _to_print_df(chart_df)


def _to_print_df(chart_df):
    """Convert internal chart df to the format expected by _generate_print_html."""
    return pd.DataFrame({
        'Quarter': chart_df['Quarter'],
        'Occupancy': chart_df['Occupancy'],
        'NOI_Actual': chart_df['Actual NOI'],
        'NOI_UW': chart_df['Underwritten NOI'],
    })


def _build_export_data(
    general_info: Dict,
    cap_stack: Dict,
    prop_perf: Dict,
    pe_perf: Dict,
    quarter: str,
    comments: Dict
) -> pd.DataFrame:
    """Build flat DataFrame for CSV export"""
    rows = []

    # General Info
    rows.append({'Section': 'General Information', 'Metric': 'Partner', 'Value': general_info['partner']})
    rows.append({'Section': 'General Information', 'Metric': 'Asset Type', 'Value': general_info['asset_type']})
    rows.append({'Section': 'General Information', 'Metric': 'Location', 'Value': general_info['location']})
    rows.append({'Section': 'General Information', 'Metric': 'Units', 'Value': general_info['units']})
    rows.append({'Section': 'General Information', 'Metric': 'Square Feet', 'Value': general_info['sqft']})

    # Cap Stack
    rows.append({'Section': 'Capitalization', 'Metric': 'Senior Debt', 'Value': cap_stack['debt']})
    rows.append({'Section': 'Capitalization', 'Metric': 'Preferred Equity', 'Value': cap_stack['pref_equity']})
    rows.append({'Section': 'Capitalization', 'Metric': 'Partner Equity', 'Value': cap_stack['partner_equity']})
    rows.append({'Section': 'Capitalization', 'Metric': 'Total Capitalization', 'Value': cap_stack['total_cap']})
    rows.append({'Section': 'Capitalization', 'Metric': 'Current Valuation', 'Value': cap_stack['current_valuation']})

    # Property Performance
    for metric, data in [('Revenue', prop_perf['revenue']), ('Expenses', prop_perf['expenses']),
                          ('NOI', prop_perf['noi'])]:
        rows.append({'Section': 'Property Performance', 'Metric': f'{metric} - YTD Actual', 'Value': data['ytd_actual']})
        rows.append({'Section': 'Property Performance', 'Metric': f'{metric} - YTD Budget', 'Value': data['ytd_budget']})
        rows.append({'Section': 'Property Performance', 'Metric': f'{metric} - Variance', 'Value': data['variance']})

    # PE Performance
    rows.append({'Section': 'PE Performance', 'Metric': 'Committed PE', 'Value': pe_perf['committed_pe']})
    rows.append({'Section': 'PE Performance', 'Metric': 'Funded to Date', 'Value': pe_perf['funded_to_date']})
    rows.append({'Section': 'PE Performance', 'Metric': 'Return of Capital', 'Value': pe_perf['return_of_capital']})
    rows.append({'Section': 'PE Performance', 'Metric': 'Current PE Balance', 'Value': pe_perf['current_pe_balance']})

    # Comments
    rows.append({'Section': 'Comments', 'Metric': 'Economic Comments', 'Value': comments.get('econ_comments', '')})
    rows.append({'Section': 'Comments', 'Metric': 'Business Plan', 'Value': comments.get('business_plan_comments', '')})

    return pd.DataFrame(rows)


def _generate_print_html(
    vcode: str,
    quarter: str,
    general_info: Dict,
    cap_stack: Dict,
    prop_perf: Dict,
    pe_perf: Dict,
    comments: Dict,
    chart_data: pd.DataFrame = None
):
    """Generate printable HTML report - fits on one page"""
    investment_name = general_info.get('investment_name', vcode)

    # Pre-compute DSCR values for template
    dscr_actual = f"{prop_perf['dscr']['ytd_actual']:.2f}x" if prop_perf['dscr']['ytd_actual'] else "-"
    dscr_budget = f"{prop_perf['dscr']['ytd_budget']:.2f}x" if prop_perf['dscr']['ytd_budget'] else "-"

    # Pre-compute date values
    date_closed = general_info['date_closed'].strftime('%m/%d/%Y') if general_info.get('date_closed') else 'N/A'
    anticipated_exit = general_info['anticipated_exit'].strftime('%m/%d/%Y') if general_info.get('anticipated_exit') else 'N/A'

    # Build chart HTML if data available
    chart_html = ""
    if chart_data is not None and not chart_data.empty:
        chart_rows = ""
        for _, row in chart_data.iterrows():
            noi_val = f"${row['NOI_Actual']:,.0f}" if pd.notna(row.get('NOI_Actual')) else "-"
            occ_val = f"{row['Occupancy']:.0f}%" if pd.notna(row.get('Occupancy')) else "-"
            chart_rows += f"<tr><td>{row['Quarter']}</td><td>{occ_val}</td><td>{noi_val}</td></tr>"
        chart_html = f"""
        <div class="section">
            <div class="section-title">6. TRAILING QUARTERS</div>
            <table class="small-table">
                <tr><th>Quarter</th><th>Occupancy</th><th>NOI</th></tr>
                {chart_rows}
            </table>
        </div>
        """

    html = f"""
    <html>
    <head>
        <title>{investment_name} - {quarter}</title>
        <style>
            @page {{ size: letter portrait; margin: 0.3in; }}
            @media print {{
                body {{ margin: 0; padding: 0.2in; font-family: Arial, sans-serif; font-size: 8pt; }}
                .header {{ text-align: center; margin-bottom: 8px; border-bottom: 2px solid #1e3a5f; padding-bottom: 5px; }}
                .header h1 {{ margin: 0; font-size: 14pt; color: #1e3a5f; }}
                .header h2 {{ margin: 2px 0 0 0; font-size: 10pt; color: #666; font-weight: normal; }}
                .content {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
                .section {{ margin-bottom: 6px; break-inside: avoid; }}
                .section-title {{ font-size: 9pt; font-weight: bold; border-bottom: 1px solid #333; margin-bottom: 4px; padding-bottom: 2px; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 7pt; }}
                .small-table {{ font-size: 6pt; }}
                th, td {{ border: 1px solid #ccc; padding: 2px 4px; text-align: left; }}
                th {{ background: #f0f0f0; font-weight: bold; }}
                .metric-row {{ display: flex; justify-content: space-between; margin: 1px 0; font-size: 7pt; }}
                .comments {{ background: #f9f9f9; padding: 4px; border: 1px solid #ddd; font-size: 7pt; min-height: 20px; max-height: 40px; overflow: hidden; }}
                .full-width {{ grid-column: 1 / -1; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{investment_name}</h1>
            <h2>{quarter} Quarterly Report</h2>
        </div>

        <div class="content">
            <div class="section">
                <div class="section-title">GENERAL INFORMATION</div>
                <div class="metric-row"><span>Partner:</span><span>{general_info['partner']}</span></div>
                <div class="metric-row"><span>Asset Type:</span><span>{general_info['asset_type']}</span></div>
                <div class="metric-row"><span>Location:</span><span>{general_info['location']}</span></div>
                <div class="metric-row"><span>Units | SF:</span><span>{general_info['units']:,} | {general_info['sqft']:,}</span></div>
                <div class="metric-row"><span>Date Closed:</span><span>{date_closed}</span></div>
                <div class="metric-row"><span>Anticipated Exit:</span><span>{anticipated_exit}</span></div>
            </div>

            <div class="section">
                <div class="section-title">CAPITALIZATION</div>
                <table>
                    <tr><th>Layer</th><th>Amount</th><th>%</th></tr>
                    <tr><td>Senior Debt</td><td>${cap_stack['debt']:,.0f}</td><td>{cap_stack['debt_pct']:.1%}</td></tr>
                    <tr><td>Pref Equity</td><td>${cap_stack['pref_equity']:,.0f}</td><td>{cap_stack['pref_equity_pct']:.1%}</td></tr>
                    <tr><td>Partner Equity</td><td>${cap_stack['partner_equity']:,.0f}</td><td>{cap_stack['partner_equity_pct']:.1%}</td></tr>
                    <tr style="font-weight: bold;"><td>Total</td><td>${cap_stack['total_cap']:,.0f}</td><td>100%</td></tr>
                </table>
                <div class="metric-row" style="margin-top: 4px;"><span>Valuation:</span><span>${cap_stack['current_valuation']:,.0f}</span></div>
                <div class="metric-row"><span>PE Coupon:</span><span>{cap_stack['pe_coupon']:.1%}</span></div>
            </div>

            <div class="section">
                <div class="section-title">PROPERTY PERFORMANCE</div>
                <table>
                    <tr><th>Metric</th><th>YTD Actual</th><th>Budget</th><th>Var</th></tr>
                    <tr><td>Revenue</td><td>${prop_perf['revenue']['ytd_actual']:,.0f}</td><td>${prop_perf['revenue']['ytd_budget']:,.0f}</td><td>${prop_perf['revenue']['variance']:,.0f}</td></tr>
                    <tr><td>Expenses</td><td>${prop_perf['expenses']['ytd_actual']:,.0f}</td><td>${prop_perf['expenses']['ytd_budget']:,.0f}</td><td>${prop_perf['expenses']['variance']:,.0f}</td></tr>
                    <tr style="font-weight: bold;"><td>NOI</td><td>${prop_perf['noi']['ytd_actual']:,.0f}</td><td>${prop_perf['noi']['ytd_budget']:,.0f}</td><td>${prop_perf['noi']['variance']:,.0f}</td></tr>
                    <tr><td>DSCR</td><td>{dscr_actual}</td><td>{dscr_budget}</td><td>-</td></tr>
                </table>
                <div class="comments" style="margin-top: 4px;">{comments.get('econ_comments', '')}</div>
            </div>

            <div class="section">
                <div class="section-title">PREFERRED EQUITY</div>
                <div class="metric-row"><span>Committed:</span><span>${pe_perf['committed_pe']:,.0f}</span></div>
                <div class="metric-row"><span>Funded:</span><span>${pe_perf['funded_to_date']:,.0f}</span></div>
                <div class="metric-row"><span>ROC:</span><span>${pe_perf['return_of_capital']:,.0f}</span></div>
                <div class="metric-row"><span>Current Balance:</span><span>${pe_perf['current_pe_balance']:,.0f}</span></div>
                <div class="metric-row"><span>Coupon:</span><span>{pe_perf['coupon']:.1%}</span></div>
                <div class="metric-row"><span>Participation:</span><span>{pe_perf['participation']:.1%}</span></div>
            </div>

            <div class="section full-width">
                <div class="section-title">BUSINESS PLAN & UPDATES</div>
                <div class="comments">{comments.get('business_plan_comments', '')}</div>
            </div>

            {chart_html}
        </div>

        <script>window.print();</script>
    </body>
    </html>
    """

    # Display in a component that opens print dialog
    st.components.v1.html(html, height=0, scrolling=False)
    st.info("Print dialog should open. If not, right-click and select 'Print'.")
