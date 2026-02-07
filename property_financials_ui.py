"""
property_financials_ui.py
Renders the Property Financials tab: Balance Sheet, Income Statement,
Tenant Roster / Lease Rollover, and One Pager.
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

from config import BS_ACCOUNTS, IS_ACCOUNTS
from one_pager_ui import render_one_pager_section


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def render_property_financials(deal_vcode, isbs_raw, fc_deal_modeled,
                               tenants_raw, inv, mri_loans_raw, mri_val,
                               wf, commitments_raw, acct, occupancy_raw):
    """Render the Property Financials tab contents."""

    _render_performance_chart(deal_vcode, isbs_raw, occupancy_raw)
    _render_income_statement(deal_vcode, isbs_raw, fc_deal_modeled)
    _render_balance_sheet(deal_vcode, isbs_raw)
    _render_tenant_roster(deal_vcode, tenants_raw, inv)
    _render_one_pager(deal_vcode, isbs_raw, inv, mri_loans_raw, mri_val,
                      wf, commitments_raw, acct, occupancy_raw)


# ============================================================
# PERFORMANCE CHART
# ============================================================

def _render_performance_chart(deal_vcode, isbs_raw, occupancy_raw):
    """Render a performance overview chart showing Actual vs U/W NOI and occupancy."""
    if isbs_raw is None or isbs_raw.empty:
        return

    st.subheader("Property Performance")

    # --- Prepare IS data (same pattern as _render_income_statement) ---
    isbs = isbs_raw.copy()
    isbs.columns = [str(c).strip() for c in isbs.columns]

    if 'vcode' in isbs.columns:
        isbs['vcode'] = isbs['vcode'].astype(str).str.strip().str.lower()
        isbs = isbs[isbs['vcode'] == str(deal_vcode).strip().lower()]

    if isbs.empty or 'dtEntry' not in isbs.columns:
        st.info("No income statement data available for performance chart.")
        return

    # Parse dates
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
        st.info("No income statement data available for performance chart.")
        return

    # Flatten revenue & expense account codes
    rev_accounts = []
    for acct_list in IS_ACCOUNTS['REVENUES'].values():
        rev_accounts.extend(acct_list)
    exp_accounts = []
    for acct_list in IS_ACCOUNTS['EXPENSES'].values():
        exp_accounts.extend(acct_list)

    def _compute_cumulative_noi(data, dates):
        """Compute cumulative YTD NOI for each date. Revenue is stored as credits
        (negative), so negate. Expenses are debits (positive)."""
        noi_by_date = {}
        for dt in dates:
            period = data[data['dtEntry_parsed'] == dt]
            rev = period[period['vAccount'].isin(rev_accounts)]['mAmount'].sum()
            exp = period[period['vAccount'].isin(exp_accounts)]['mAmount'].sum()
            noi_by_date[dt] = (-rev) - exp  # revenue flipped, minus expenses
        return noi_by_date

    # Get sorted unique dates for each source
    actual_dates = sorted(actual_data['dtEntry_parsed'].dropna().unique())
    uw_dates = sorted(uw_data['dtEntry_parsed'].dropna().unique())

    all_dates = sorted(set(actual_dates) | set(uw_dates))
    if not all_dates:
        st.info("No period dates found for performance chart.")
        return

    actual_cum = _compute_cumulative_noi(actual_data, actual_dates)
    uw_cum = _compute_cumulative_noi(uw_data, uw_dates)

    def _cumulative_to_periodic(cum_dict, sorted_dates):
        """Convert cumulative YTD balances to periodic (monthly) values.
        YTD resets each January, so Jan periodic = Jan cumulative."""
        periodic = {}
        for i, dt in enumerate(sorted_dates):
            dt_ts = pd.Timestamp(dt)
            if dt_ts.month == 1:
                # First month of year — YTD resets, periodic = cumulative
                periodic[dt_ts] = cum_dict[dt]
            else:
                # Find prior month in same year
                prior = None
                for j in range(i - 1, -1, -1):
                    p = pd.Timestamp(sorted_dates[j])
                    if p.year == dt_ts.year:
                        prior = sorted_dates[j]
                        break
                if prior is not None:
                    periodic[dt_ts] = cum_dict[dt] - cum_dict[prior]
                else:
                    periodic[dt_ts] = cum_dict[dt]
        return periodic

    actual_periodic = _cumulative_to_periodic(actual_cum, actual_dates)
    uw_periodic = _cumulative_to_periodic(uw_cum, uw_dates)

    # --- Controls ---
    freq_options = ["Monthly", "Quarterly", "Annually"]
    ctrl_cols = st.columns([1, 1, 2])
    with ctrl_cols[0]:
        frequency = st.selectbox("Frequency", freq_options, index=1, key="perf_freq")

    # Build period-end dates based on frequency
    def _aggregate_periodic(periodic_dict, freq):
        """Aggregate monthly periodic NOI to the selected frequency."""
        if not periodic_dict:
            return {}
        if freq == "Monthly":
            return periodic_dict
        elif freq == "Quarterly":
            # Sum 3 months ending at quarter-end (Mar, Jun, Sep, Dec)
            quarterly = {}
            for dt, val in sorted(periodic_dict.items()):
                dt_ts = pd.Timestamp(dt)
                # Find quarter-end for this month
                q_month = ((dt_ts.month - 1) // 3 + 1) * 3
                q_end = pd.Timestamp(year=dt_ts.year, month=q_month, day=1) + pd.offsets.MonthEnd(0)
                quarterly[q_end] = quarterly.get(q_end, 0) + val
            # Only keep quarters where all 3 months are present
            month_counts = {}
            for dt in periodic_dict:
                dt_ts = pd.Timestamp(dt)
                q_month = ((dt_ts.month - 1) // 3 + 1) * 3
                q_end = pd.Timestamp(year=dt_ts.year, month=q_month, day=1) + pd.offsets.MonthEnd(0)
                month_counts[q_end] = month_counts.get(q_end, 0) + 1
            return {k: v for k, v in quarterly.items() if month_counts.get(k, 0) == 3}
        else:  # Annually
            annual = {}
            for dt, val in sorted(periodic_dict.items()):
                yr_end = pd.Timestamp(year=pd.Timestamp(dt).year, month=12, day=31)
                annual[yr_end] = annual.get(yr_end, 0) + val
            # Only keep years with 12 months
            month_counts = {}
            for dt in periodic_dict:
                yr_end = pd.Timestamp(year=pd.Timestamp(dt).year, month=12, day=31)
                month_counts[yr_end] = month_counts.get(yr_end, 0) + 1
            return {k: v for k, v in annual.items() if month_counts.get(k, 0) == 12}

    actual_agg = _aggregate_periodic(actual_periodic, frequency)
    uw_agg = _aggregate_periodic(uw_periodic, frequency)

    all_period_ends = sorted(set(actual_agg.keys()) | set(uw_agg.keys()))
    if not all_period_ends:
        st.info("Not enough data for the selected frequency.")
        return

    period_end_labels = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in all_period_ends]

    with ctrl_cols[1]:
        default_pe_idx = len(period_end_labels) - 1
        period_end_sel = st.selectbox("Period End", period_end_labels,
                                      index=default_pe_idx, key="perf_period_end")

    # Filter to 12 periods ending at selected period end
    sel_idx = period_end_labels.index(period_end_sel)
    start_idx = max(0, sel_idx - 11)
    display_dates = all_period_ends[start_idx:sel_idx + 1]

    # --- Occupancy data ---
    # CSV has monthly rows with dtReported (Excel serial date), Qtr ("2021-Q4"),
    # and Occ% (always populated).  OccupancyPercent has many nulls so we prefer Occ%.
    has_occupancy = False
    occ_by_date = {}
    if occupancy_raw is not None and not occupancy_raw.empty:
        occ = occupancy_raw.copy()
        occ.columns = [str(c).strip() for c in occ.columns]
        if 'vCode' in occ.columns:
            occ['vCode'] = occ['vCode'].astype(str).str.strip().str.lower()
            occ = occ[occ['vCode'] == str(deal_vcode).strip().lower()]

        # Pick best occupancy column
        occ_col = 'Occ%' if 'Occ%' in occ.columns else (
            'OccupancyPercent' if 'OccupancyPercent' in occ.columns else None)

        if not occ.empty and occ_col and 'dtReported' in occ.columns:
            occ['occ_val'] = pd.to_numeric(occ[occ_col], errors='coerce')
            # Parse dtReported (Excel serial date)
            try:
                occ['date_parsed'] = pd.to_datetime(
                    occ['dtReported'], unit='D', origin='1899-12-30', errors='coerce')
            except Exception:
                occ['date_parsed'] = pd.to_datetime(occ['dtReported'], errors='coerce')
            occ = occ.dropna(subset=['date_parsed', 'occ_val'])

            if not occ.empty:
                # Build monthly lookup keyed by month-end
                monthly_occ = {}
                for _, row in occ.iterrows():
                    me = pd.Timestamp(row['date_parsed']) + pd.offsets.MonthEnd(0)
                    monthly_occ[me] = row['occ_val']

                # Align to chart periods
                for dt in display_dates:
                    dt_ts = pd.Timestamp(dt)
                    if frequency == "Monthly":
                        me = dt_ts + pd.offsets.MonthEnd(0)
                        if me in monthly_occ:
                            occ_by_date[dt_ts] = monthly_occ[me]
                    elif frequency == "Quarterly":
                        # Average the 3 months in the quarter
                        q_start_month = ((dt_ts.month - 1) // 3) * 3 + 1
                        vals = []
                        for m in range(q_start_month, q_start_month + 3):
                            me = pd.Timestamp(year=dt_ts.year, month=m, day=1) + pd.offsets.MonthEnd(0)
                            if me in monthly_occ:
                                vals.append(monthly_occ[me])
                        if vals:
                            occ_by_date[dt_ts] = sum(vals) / len(vals)
                    else:  # Annually — average all months in year
                        yr_vals = [v for k, v in monthly_occ.items()
                                   if pd.Timestamp(k).year == dt_ts.year]
                        if yr_vals:
                            occ_by_date[dt_ts] = sum(yr_vals) / len(yr_vals)
                if occ_by_date:
                    has_occupancy = True

    # --- Build chart dataframe ---
    chart_rows = []
    for dt in display_dates:
        dt_ts = pd.Timestamp(dt)
        label = dt_ts.strftime('%b %Y') if frequency == "Monthly" else (
            f"Q{(dt_ts.month - 1) // 3 + 1} {dt_ts.year}" if frequency == "Quarterly" else str(dt_ts.year)
        )
        chart_rows.append({
            'Period': label,
            '_sort': dt_ts,
            'Actual NOI': actual_agg.get(dt_ts, None),
            'Underwritten NOI': uw_agg.get(dt_ts, None),
            'Occupancy': occ_by_date.get(dt_ts, None),
        })

    chart_df = pd.DataFrame(chart_rows)
    period_order = chart_df['Period'].tolist()

    # --- Altair chart ---
    if has_occupancy:
        # Melt NOI columns for line chart
        noi_df = chart_df.melt(id_vars=['Period'], value_vars=['Actual NOI', 'Underwritten NOI'],
                               var_name='Series', value_name='NOI')
        noi_df = noi_df.dropna(subset=['NOI'])

        occ_df = chart_df[['Period', 'Occupancy']].dropna(subset=['Occupancy'])

        bars = alt.Chart(occ_df).mark_bar(color='#B4D4F0', opacity=0.6).encode(
            x=alt.X('Period:N', sort=period_order, title='Period'),
            y=alt.Y('Occupancy:Q', title='Occupancy %', scale=alt.Scale(domain=[0, 100])),
        )

        color_scale = alt.Scale(
            domain=['Actual NOI', 'Underwritten NOI'],
            range=['#1F4E79', '#ED7D31']
        )
        dash_scale = alt.Scale(
            domain=['Actual NOI', 'Underwritten NOI'],
            range=[[0], [5, 5]]
        )

        lines = alt.Chart(noi_df).mark_line(point=True).encode(
            x=alt.X('Period:N', sort=period_order),
            y=alt.Y('NOI:Q', title='NOI ($)'),
            color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash('Series:N', scale=dash_scale, legend=alt.Legend(title=None)),
        )

        chart = alt.layer(bars, lines).resolve_scale(y='independent').properties(height=300)
    else:
        # No occupancy — just NOI lines on single axis
        noi_df = chart_df.melt(id_vars=['Period'], value_vars=['Actual NOI', 'Underwritten NOI'],
                               var_name='Series', value_name='NOI')
        noi_df = noi_df.dropna(subset=['NOI'])

        if noi_df.empty:
            st.info("No NOI data available for the selected periods.")
            return

        color_scale = alt.Scale(
            domain=['Actual NOI', 'Underwritten NOI'],
            range=['#1F4E79', '#ED7D31']
        )
        dash_scale = alt.Scale(
            domain=['Actual NOI', 'Underwritten NOI'],
            range=[[0], [5, 5]]
        )

        chart = alt.Chart(noi_df).mark_line(point=True).encode(
            x=alt.X('Period:N', sort=period_order, title='Period'),
            y=alt.Y('NOI:Q', title='NOI ($)'),
            color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash('Series:N', scale=dash_scale, legend=alt.Legend(title=None)),
        ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)


# ============================================================
# BALANCE SHEET COMPARISON
# ============================================================

def _render_balance_sheet(deal_vcode, isbs_raw):
    if isbs_raw is None or isbs_raw.empty:
        return

    st.subheader("Balance Sheet Comparison")

    # Prepare ISBS data for the deal
    isbs_bs = isbs_raw.copy()
    isbs_bs.columns = [str(c).strip() for c in isbs_bs.columns]

    # Filter for deal
    if 'vcode' in isbs_bs.columns:
        isbs_bs['vcode'] = isbs_bs['vcode'].astype(str).str.strip().str.lower()
        isbs_bs = isbs_bs[isbs_bs['vcode'] == str(deal_vcode).strip().lower()]

    # Filter for actual reported balance sheets only (vSource = "Interim BS")
    if 'vSource' in isbs_bs.columns:
        isbs_bs['vSource'] = isbs_bs['vSource'].astype(str).str.strip()
        isbs_bs = isbs_bs[isbs_bs['vSource'] == 'Interim BS']

    if isbs_bs.empty or 'dtEntry' not in isbs_bs.columns:
        st.info("No balance sheet data available for this deal.")
        return

    # Parse dates
    try:
        isbs_bs['dtEntry_parsed'] = pd.to_datetime(isbs_bs['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
    except Exception:
        isbs_bs['dtEntry_parsed'] = pd.to_datetime(isbs_bs['dtEntry'], errors='coerce')
    null_dates = isbs_bs['dtEntry_parsed'].isna()
    if null_dates.any():
        isbs_bs.loc[null_dates, 'dtEntry_parsed'] = pd.to_datetime(isbs_bs.loc[null_dates, 'dtEntry'], errors='coerce')

    # Get available periods (only from actual reported balance sheets)
    available_periods = sorted(isbs_bs['dtEntry_parsed'].dropna().unique())

    if len(available_periods) < 1:
        st.info("No balance sheet periods available for this deal.")
        return

    # Convert to date strings for display
    period_options = [pd.Timestamp(p).strftime('%Y-%m-%d') for p in available_periods]

    # Default selections: most recent and prior year end
    most_recent_idx = len(period_options) - 1
    most_recent = period_options[most_recent_idx]

    # Find prior year end (December of previous year)
    most_recent_date = pd.Timestamp(available_periods[most_recent_idx])
    prior_year = most_recent_date.year - 1
    prior_year_end = None
    for i, p in enumerate(available_periods):
        p_date = pd.Timestamp(p)
        if p_date.year == prior_year and p_date.month == 12:
            prior_year_end = period_options[i]
            break
    if prior_year_end is None and len(period_options) > 1:
        prior_year_end = period_options[0]  # Fallback to earliest
    elif prior_year_end is None:
        prior_year_end = most_recent

    # ---------- Period selectors inside a form ----------
    with st.form("bs_settings"):
        col_left, col_right = st.columns(2)
        with col_left:
            period1 = st.selectbox("Prior Period", period_options,
                                   index=period_options.index(prior_year_end) if prior_year_end in period_options else 0,
                                   key="bs_period1")
        with col_right:
            period2 = st.selectbox("Current Period", period_options,
                                   index=most_recent_idx,
                                   key="bs_period2")
        bs_submitted = st.form_submit_button("Apply Settings")

    # Get balances for each period
    balances1 = _get_period_balances(isbs_bs, period1, BS_ACCOUNTS)
    balances2 = _get_period_balances(isbs_bs, period2, BS_ACCOUNTS)

    # Build display table
    rows = []

    def add_row(label, val1, val2, indent=0, is_header=False, is_total=False):
        variance = val2 - val1 if val1 is not None and val2 is not None else None
        var_pct = (variance / abs(val1) * 100) if val1 and val1 != 0 and variance is not None else None

        prefix = "  " * indent
        if is_header or is_total:
            label_display = f"**{prefix}{label}**"
        else:
            label_display = f"{prefix}{label}"

        rows.append({
            'Account': label_display,
            period1: f"${val1:,.0f}" if val1 is not None else "",
            period2: f"${val2:,.0f}" if val2 is not None else "",
            'Variance $': f"${variance:,.0f}" if variance is not None else "",
            'Variance %': f"{var_pct:.1f}%" if var_pct is not None else "",
        })

    # ASSETS
    add_row("ASSETS", None, None, is_header=True)
    total_current_assets_1 = 0
    total_current_assets_2 = 0

    # Current Assets
    add_row("Current Assets", None, None, indent=1, is_header=True)
    for category in BS_ACCOUNTS['ASSETS']['Current Assets'].keys():
        val1 = balances1.get('ASSETS', {}).get('Current Assets', {}).get(category, 0)
        val2 = balances2.get('ASSETS', {}).get('Current Assets', {}).get(category, 0)
        total_current_assets_1 += val1
        total_current_assets_2 += val2
        if val1 != 0 or val2 != 0:
            add_row(category, val1, val2, indent=2)
    add_row("Total Current Assets", total_current_assets_1, total_current_assets_2, indent=1, is_total=True)

    # Noncurrent Assets
    total_noncurrent_assets_1 = 0
    total_noncurrent_assets_2 = 0
    add_row("Noncurrent Assets", None, None, indent=1, is_header=True)
    for category in BS_ACCOUNTS['ASSETS']['Noncurrent Assets'].keys():
        val1 = balances1.get('ASSETS', {}).get('Noncurrent Assets', {}).get(category, 0)
        val2 = balances2.get('ASSETS', {}).get('Noncurrent Assets', {}).get(category, 0)
        total_noncurrent_assets_1 += val1
        total_noncurrent_assets_2 += val2
        if val1 != 0 or val2 != 0:
            add_row(category, val1, val2, indent=2)
    add_row("Total Noncurrent Assets", total_noncurrent_assets_1, total_noncurrent_assets_2, indent=1, is_total=True)

    total_assets_1 = total_current_assets_1 + total_noncurrent_assets_1
    total_assets_2 = total_current_assets_2 + total_noncurrent_assets_2
    add_row("TOTAL ASSETS", total_assets_1, total_assets_2, is_total=True)

    # Blank row
    rows.append({'Account': '', period1: '', period2: '', 'Variance $': '', 'Variance %': ''})

    # LIABILITIES
    add_row("LIABILITIES", None, None, is_header=True)
    total_current_liab_1 = 0
    total_current_liab_2 = 0

    # Current Liabilities
    add_row("Current Liabilities", None, None, indent=1, is_header=True)
    for category in BS_ACCOUNTS['LIABILITIES']['Current Liabilities'].keys():
        val1 = balances1.get('LIABILITIES', {}).get('Current Liabilities', {}).get(category, 0)
        val2 = balances2.get('LIABILITIES', {}).get('Current Liabilities', {}).get(category, 0)
        total_current_liab_1 += val1
        total_current_liab_2 += val2
        if val1 != 0 or val2 != 0:
            add_row(category, val1, val2, indent=2)
    add_row("Total Current Liabilities", total_current_liab_1, total_current_liab_2, indent=1, is_total=True)

    # Noncurrent Liabilities
    total_noncurrent_liab_1 = 0
    total_noncurrent_liab_2 = 0
    add_row("Noncurrent Liabilities", None, None, indent=1, is_header=True)
    for category in BS_ACCOUNTS['LIABILITIES']['Noncurrent Liabilities'].keys():
        val1 = balances1.get('LIABILITIES', {}).get('Noncurrent Liabilities', {}).get(category, 0)
        val2 = balances2.get('LIABILITIES', {}).get('Noncurrent Liabilities', {}).get(category, 0)
        total_noncurrent_liab_1 += val1
        total_noncurrent_liab_2 += val2
        if val1 != 0 or val2 != 0:
            add_row(category, val1, val2, indent=2)
    add_row("Total Noncurrent Liabilities", total_noncurrent_liab_1, total_noncurrent_liab_2, indent=1, is_total=True)

    total_liab_1 = total_current_liab_1 + total_noncurrent_liab_1
    total_liab_2 = total_current_liab_2 + total_noncurrent_liab_2
    add_row("TOTAL LIABILITIES", total_liab_1, total_liab_2, is_total=True)

    # Blank row
    rows.append({'Account': '', period1: '', period2: '', 'Variance $': '', 'Variance %': ''})

    # EQUITY
    add_row("EQUITY", None, None, is_header=True)
    total_equity_1 = 0
    total_equity_2 = 0
    for category in BS_ACCOUNTS['EQUITY']['Equity'].keys():
        val1 = balances1.get('EQUITY', {}).get('Equity', {}).get(category, 0)
        val2 = balances2.get('EQUITY', {}).get('Equity', {}).get(category, 0)
        total_equity_1 += val1
        total_equity_2 += val2
        if val1 != 0 or val2 != 0:
            add_row(category, val1, val2, indent=1)
    add_row("TOTAL EQUITY", total_equity_1, total_equity_2, is_total=True)

    # Blank row
    rows.append({'Account': '', period1: '', period2: '', 'Variance $': '', 'Variance %': ''})

    # TOTAL LIABILITIES + EQUITY
    total_liab_equity_1 = total_liab_1 + total_equity_1
    total_liab_equity_2 = total_liab_2 + total_equity_2
    add_row("TOTAL LIABILITIES + EQUITY", total_liab_equity_1, total_liab_equity_2, is_total=True)

    # Display the balance sheet
    bs_df = pd.DataFrame(rows)
    st.dataframe(bs_df, use_container_width=True, hide_index=True)

    # Balance check (Assets + Liabilities + Equity = 0 when balanced, since L&E are credit/negative)
    balance_check = total_assets_2 + total_liab_equity_2
    if abs(balance_check) > 1:
        st.warning(f"Balance sheet does not balance. Assets + Liabilities + Equity = ${balance_check:,.0f}")


def _get_period_balances(isbs_df, period_str, accounts_dict):
    """Get account balances for a specific period."""
    period_date = pd.to_datetime(period_str)
    period_data = isbs_df[isbs_df['dtEntry_parsed'] == period_date].copy()

    if period_data.empty:
        return {}

    # Normalize account column
    if 'vAccount' in period_data.columns:
        period_data['vAccount'] = period_data['vAccount'].astype(str).str.strip()
    if 'mAmount' in period_data.columns:
        period_data['mAmount'] = pd.to_numeric(period_data['mAmount'], errors='coerce').fillna(0)

    balances = {}
    for section, subsections in accounts_dict.items():
        balances[section] = {}
        for subsection, categories in subsections.items():
            balances[section][subsection] = {}
            for category, acct_list in categories.items():
                total = period_data[period_data['vAccount'].isin(acct_list)]['mAmount'].sum()
                balances[section][subsection][category] = total
    return balances


# ============================================================
# INCOME STATEMENT COMPARISON
# ============================================================

def _render_income_statement(deal_vcode, isbs_raw, fc_deal_modeled):
    if isbs_raw is None or isbs_raw.empty:
        return

    st.subheader("Income Statement Comparison")

    # Prepare ISBS data for the deal
    isbs_is = isbs_raw.copy()
    isbs_is.columns = [str(c).strip() for c in isbs_is.columns]

    # Filter for deal
    if 'vcode' in isbs_is.columns:
        isbs_is['vcode'] = isbs_is['vcode'].astype(str).str.strip().str.lower()
        isbs_is = isbs_is[isbs_is['vcode'] == str(deal_vcode).strip().lower()]

    if isbs_is.empty or 'dtEntry' not in isbs_is.columns:
        st.info("No income statement data available for this deal.")
        return

    # Parse dates
    try:
        isbs_is['dtEntry_parsed'] = pd.to_datetime(isbs_is['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
    except Exception:
        isbs_is['dtEntry_parsed'] = pd.to_datetime(isbs_is['dtEntry'], errors='coerce')
    null_dates = isbs_is['dtEntry_parsed'].isna()
    if null_dates.any():
        isbs_is.loc[null_dates, 'dtEntry_parsed'] = pd.to_datetime(isbs_is.loc[null_dates, 'dtEntry'], errors='coerce')

    # Normalize vSource
    if 'vSource' in isbs_is.columns:
        isbs_is['vSource'] = isbs_is['vSource'].astype(str).str.strip()

    # Normalize vAccount and mAmount
    if 'vAccount' in isbs_is.columns:
        isbs_is['vAccount'] = isbs_is['vAccount'].astype(str).str.strip()
    if 'mAmount' in isbs_is.columns:
        isbs_is['mAmount'] = pd.to_numeric(isbs_is['mAmount'], errors='coerce').fillna(0)

    # Get available actual periods (Interim IS only)
    actual_data = isbs_is[isbs_is['vSource'] == 'Interim IS']
    actual_periods = sorted(actual_data['dtEntry_parsed'].dropna().unique()) if not actual_data.empty else []

    # Get budget data
    budget_data = isbs_is[isbs_is['vSource'] == 'Budget IS']

    # Get underwriting data
    uw_data = isbs_is[isbs_is['vSource'] == 'Projected IS']

    if len(actual_periods) < 1:
        st.info("No income statement periods available for this deal.")
        return

    # Determine most recent actual period
    most_recent_actual = pd.Timestamp(actual_periods[-1])

    # Period type options
    period_types = [
        "TTM (Trailing Twelve Months)",
        "YTD (Year to Date)",
        "Current Year Estimate",
        "Full Year",
        "Custom Month",
    ]

    # Source options
    source_options = ["Actual", "Budget", "Underwriting", "Valuation"]

    # As-of date options (used for all period types)
    period_date_options = [pd.Timestamp(p).strftime('%Y-%m-%d') for p in actual_periods]

    # Default for column 2: prior year same month if available
    most_recent_ts = pd.Timestamp(actual_periods[-1])
    default_idx2 = len(period_date_options) - 1
    for i, p in enumerate(period_date_options):
        p_ts = pd.Timestamp(p)
        if p_ts.year == most_recent_ts.year - 1 and p_ts.month == most_recent_ts.month:
            default_idx2 = i
            break

    # ---------- All settings inside a form ----------
    with st.form("is_settings"):
        st.markdown("**Configure Comparison:**")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*Column 1*")
            period_type1 = st.selectbox("Period Type", period_types, index=0, key="is_period_type1")
            source1 = st.selectbox("Source", source_options, index=0, key="is_source1")
            ref_date1 = st.selectbox("As of Date", period_date_options,
                                     index=len(period_date_options) - 1, key="is_ref_date1")
            st.caption("For Full Year, the full year of the selected date is used.")

        with col2:
            st.markdown("*Column 2*")
            period_type2 = st.selectbox("Period Type", period_types, index=0, key="is_period_type2")
            source2 = st.selectbox("Source", source_options, index=0, key="is_source2")
            ref_date2 = st.selectbox("As of Date", period_date_options,
                                     index=default_idx2, key="is_ref_date2")
            st.caption("For Full Year, the full year of the selected date is used.")

        is_submitted = st.form_submit_button("Apply Settings")

    # Derive year from as-of date for Full Year mode
    year1 = pd.Timestamp(ref_date1).year
    year2 = pd.Timestamp(ref_date2).year

    # Calculate amounts for both columns
    amounts1 = _calculate_is_amounts(period_type1, source1, ref_date1, year1, IS_ACCOUNTS,
                                     actual_data, actual_periods, budget_data, uw_data, fc_deal_modeled)
    amounts2 = _calculate_is_amounts(period_type2, source2, ref_date2, year2, IS_ACCOUNTS,
                                     actual_data, actual_periods, budget_data, uw_data, fc_deal_modeled)

    # Build display table
    is_rows = []

    # Column labels
    col1_label = f"{source1} {period_type1.split(' ')[0]} {ref_date1[:7] if 'TTM' in period_type1 or 'YTD' in period_type1 else year1}"
    col2_label = f"{source2} {period_type2.split(' ')[0]} {ref_date2[:7] if 'TTM' in period_type2 or 'YTD' in period_type2 else year2}"

    def add_is_row(label, val1, val2, indent=0, is_header=False, is_total=False, is_calc=False):
        variance = val2 - val1 if val1 is not None and val2 is not None else None
        var_pct = (variance / abs(val1) * 100) if val1 and val1 != 0 and variance is not None else None

        prefix = "  " * indent
        if is_header or is_total or is_calc:
            label_display = f"**{prefix}{label}**"
        else:
            label_display = f"{prefix}{label}"

        is_rows.append({
            'Account': label_display,
            col1_label: f"${val1:,.0f}" if val1 is not None else "",
            col2_label: f"${val2:,.0f}" if val2 is not None else "",
            'Variance $': f"${variance:,.0f}" if variance is not None else "",
            'Variance %': f"{var_pct:.1f}%" if var_pct is not None else "",
        })

    # REVENUES (flip signs - stored as credits/negative in accounting)
    add_is_row("REVENUES", None, None, is_header=True)
    total_rev_1 = 0
    total_rev_2 = 0
    for category in IS_ACCOUNTS['REVENUES'].keys():
        val1 = -amounts1.get('REVENUES', {}).get(category, 0)  # Flip sign
        val2 = -amounts2.get('REVENUES', {}).get(category, 0)  # Flip sign
        total_rev_1 += val1
        total_rev_2 += val2
        if val1 != 0 or val2 != 0:
            add_is_row(category, val1, val2, indent=1)
    add_is_row("Total Revenues", total_rev_1, total_rev_2, is_total=True)

    # Blank row
    is_rows.append({'Account': '', col1_label: '', col2_label: '', 'Variance $': '', 'Variance %': ''})

    # EXPENSES
    add_is_row("EXPENSES", None, None, is_header=True)
    total_exp_1 = 0
    total_exp_2 = 0
    for category in IS_ACCOUNTS['EXPENSES'].keys():
        val1 = amounts1.get('EXPENSES', {}).get(category, 0)
        val2 = amounts2.get('EXPENSES', {}).get(category, 0)
        total_exp_1 += val1
        total_exp_2 += val2
        if val1 != 0 or val2 != 0:
            add_is_row(category, val1, val2, indent=1)
    add_is_row("Total Expenses", total_exp_1, total_exp_2, is_total=True)

    # Blank row
    is_rows.append({'Account': '', col1_label: '', col2_label: '', 'Variance $': '', 'Variance %': ''})

    # NOI
    noi_1 = total_rev_1 - total_exp_1
    noi_2 = total_rev_2 - total_exp_2
    add_is_row("NET OPERATING INCOME", noi_1, noi_2, is_calc=True)

    # Blank row
    is_rows.append({'Account': '', col1_label: '', col2_label: '', 'Variance $': '', 'Variance %': ''})

    # DEBT SERVICE
    add_is_row("DEBT SERVICE", None, None, is_header=True)
    interest_1 = amounts1.get('DEBT_SERVICE', {}).get('Interest', 0)
    interest_2 = amounts2.get('DEBT_SERVICE', {}).get('Interest', 0)
    principal_1 = amounts1.get('DEBT_SERVICE', {}).get('Principal', 0)
    principal_2 = amounts2.get('DEBT_SERVICE', {}).get('Principal', 0)

    if interest_1 != 0 or interest_2 != 0:
        add_is_row("Interest", interest_1, interest_2, indent=1)
    if principal_1 != 0 or principal_2 != 0:
        add_is_row("Principal", principal_1, principal_2, indent=1)

    total_ds_1 = abs(interest_1) + abs(principal_1)
    total_ds_2 = abs(interest_2) + abs(principal_2)
    add_is_row("Total Debt Service", total_ds_1, total_ds_2, is_total=True)

    # DSCR
    dscr_1 = noi_1 / total_ds_1 if total_ds_1 != 0 else None
    dscr_2 = noi_2 / total_ds_2 if total_ds_2 != 0 else None
    dscr_var = dscr_2 - dscr_1 if dscr_1 is not None and dscr_2 is not None else None

    is_rows.append({
        'Account': '**Debt Service Coverage Ratio**',
        col1_label: f"{dscr_1:.2f}x" if dscr_1 is not None else "",
        col2_label: f"{dscr_2:.2f}x" if dscr_2 is not None else "",
        'Variance $': f"{dscr_var:.2f}x" if dscr_var is not None else "",
        'Variance %': "",
    })

    # Blank row
    is_rows.append({'Account': '', col1_label: '', col2_label: '', 'Variance $': '', 'Variance %': ''})

    # OTHER BELOW THE LINE
    add_is_row("OTHER BELOW THE LINE", None, None, is_header=True)
    total_btl_1 = 0
    total_btl_2 = 0
    for category in IS_ACCOUNTS['OTHER_BTL'].keys():
        val1 = amounts1.get('OTHER_BTL', {}).get(category, 0)
        val2 = amounts2.get('OTHER_BTL', {}).get(category, 0)
        # Flip sign for Interest Income (stored as credit/negative)
        if category == 'Interest Income':
            val1 = -val1
            val2 = -val2
        total_btl_1 += val1
        total_btl_2 += val2
        if val1 != 0 or val2 != 0:
            add_is_row(category, val1, val2, indent=1)
    add_is_row("Total Other Below the Line", total_btl_1, total_btl_2, is_total=True)

    # Display the income statement
    is_df = pd.DataFrame(is_rows)
    st.dataframe(is_df, use_container_width=True, hide_index=True)


# ---- IS helper functions ----

def _get_cumulative_balances(data_df, as_of_date, accounts_dict):
    """Get cumulative account balances as of a specific date from Actuals."""
    period_data = data_df[data_df['dtEntry_parsed'] == as_of_date].copy()
    if period_data.empty:
        return {}
    balances = {}
    for section, categories in accounts_dict.items():
        balances[section] = {}
        for category, acct_list in categories.items():
            total = period_data[period_data['vAccount'].isin(acct_list)]['mAmount'].sum()
            balances[section][category] = total
    return balances


def _get_budget_sum(data_df, start_date, end_date, accounts_dict):
    """Sum budget amounts between start_date (exclusive) and end_date (inclusive)."""
    period_data = data_df[
        (data_df['dtEntry_parsed'] > start_date) &
        (data_df['dtEntry_parsed'] <= end_date)
    ].copy()
    if period_data.empty:
        return {}
    balances = {}
    for section, categories in accounts_dict.items():
        balances[section] = {}
        for category, acct_list in categories.items():
            total = period_data[period_data['vAccount'].isin(acct_list)]['mAmount'].sum()
            balances[section][category] = total
    return balances


def _subtract_balances(bal1, bal2, accounts_dict):
    """bal1 - bal2"""
    result = {}
    for section, categories in accounts_dict.items():
        result[section] = {}
        for category in categories.keys():
            v1 = bal1.get(section, {}).get(category, 0)
            v2 = bal2.get(section, {}).get(category, 0)
            result[section][category] = v1 - v2
    return result


def _add_balances(bal1, bal2, accounts_dict):
    """bal1 + bal2"""
    result = {}
    for section, categories in accounts_dict.items():
        result[section] = {}
        for category in categories.keys():
            v1 = bal1.get(section, {}).get(category, 0)
            v2 = bal2.get(section, {}).get(category, 0)
            result[section][category] = v1 + v2
    return result


def _get_valuation_sum(fc_df, start_date, end_date, accounts_dict):
    """Get valuation amounts from forecast_feed."""
    if fc_df is None or fc_df.empty:
        return {}
    period_data = fc_df[
        (fc_df['event_date'] > start_date) &
        (fc_df['event_date'] <= end_date)
    ].copy()
    if period_data.empty:
        return {}
    balances = {}
    for section, categories in accounts_dict.items():
        balances[section] = {}
        for category, acct_list in categories.items():
            total = period_data[period_data['vAccount'].isin(acct_list)]['mAmount_norm'].sum()
            balances[section][category] = total
    return balances


def _calculate_is_amounts(period_type, source, ref_date, year, accounts_dict,
                          actual_data, actual_periods, budget_data, uw_data, fc_deal_modeled):
    """Calculate income statement amounts based on period type and source."""
    ref_date = pd.Timestamp(ref_date)

    if source == "Actual":
        if period_type == "TTM (Trailing Twelve Months)":
            # TTM = current month + prior Dec - same month last year
            current_bal = _get_cumulative_balances(actual_data, ref_date, accounts_dict)
            # Find December of prior year
            dec_prior = None
            for p in actual_periods:
                p_ts = pd.Timestamp(p)
                if p_ts.year == ref_date.year - 1 and p_ts.month == 12:
                    dec_prior = p_ts
                    break
            # Find same month last year
            same_month_ly = None
            for p in actual_periods:
                p_ts = pd.Timestamp(p)
                if p_ts.year == ref_date.year - 1 and p_ts.month == ref_date.month:
                    same_month_ly = p_ts
                    break

            if dec_prior and same_month_ly:
                dec_bal = _get_cumulative_balances(actual_data, dec_prior, accounts_dict)
                ly_bal = _get_cumulative_balances(actual_data, same_month_ly, accounts_dict)
                # TTM = current + dec_prior - same_month_ly
                temp = _add_balances(current_bal, dec_bal, accounts_dict)
                return _subtract_balances(temp, ly_bal, accounts_dict)
            elif dec_prior:
                dec_bal = _get_cumulative_balances(actual_data, dec_prior, accounts_dict)
                return _add_balances(current_bal, dec_bal, accounts_dict)
            else:
                return current_bal

        elif period_type == "YTD (Year to Date)":
            # YTD = current month cumulative (since it resets each year)
            return _get_cumulative_balances(actual_data, ref_date, accounts_dict)

        elif period_type == "Full Year":
            # Full Year = December of that year
            dec_date = None
            for p in actual_periods:
                p_ts = pd.Timestamp(p)
                if p_ts.year == year and p_ts.month == 12:
                    dec_date = p_ts
                    break
            if dec_date:
                return _get_cumulative_balances(actual_data, dec_date, accounts_dict)
            return {}

        elif period_type == "Current Year Estimate":
            # YTD Actual + Budget for remainder
            ytd_bal = _get_cumulative_balances(actual_data, ref_date, accounts_dict)
            # Budget from ref_date to Dec 31
            dec_end = pd.Timestamp(f"{ref_date.year}-12-31")
            budget_remainder = _get_budget_sum(budget_data, ref_date, dec_end, accounts_dict)
            return _add_balances(ytd_bal, budget_remainder, accounts_dict)

        else:  # Custom Month
            return _get_cumulative_balances(actual_data, ref_date, accounts_dict)

    elif source == "Budget":
        if period_type == "TTM (Trailing Twelve Months)":
            # Sum 12 months of budget ending at ref_date
            start = ref_date - pd.DateOffset(months=12)
            return _get_budget_sum(budget_data, start, ref_date, accounts_dict)

        elif period_type == "YTD (Year to Date)":
            # Sum budget from Jan 1 to ref_date
            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
            return _get_budget_sum(budget_data, jan1, ref_date, accounts_dict)

        elif period_type == "Full Year":
            # Sum budget for full year
            jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
            dec31 = pd.Timestamp(f"{year}-12-31")
            return _get_budget_sum(budget_data, jan1, dec31, accounts_dict)

        elif period_type == "Current Year Estimate":
            # Full year budget
            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
            dec31 = pd.Timestamp(f"{ref_date.year}-12-31")
            return _get_budget_sum(budget_data, jan1, dec31, accounts_dict)

        else:  # Custom Month
            # Single month budget
            prior_month = ref_date - pd.DateOffset(months=1)
            return _get_budget_sum(budget_data, prior_month, ref_date, accounts_dict)

    elif source == "Underwriting":
        if period_type == "TTM (Trailing Twelve Months)":
            start = ref_date - pd.DateOffset(months=12)
            return _get_budget_sum(uw_data, start, ref_date, accounts_dict)

        elif period_type == "YTD (Year to Date)":
            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
            return _get_budget_sum(uw_data, jan1, ref_date, accounts_dict)

        elif period_type == "Full Year":
            jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
            dec31 = pd.Timestamp(f"{year}-12-31")
            return _get_budget_sum(uw_data, jan1, dec31, accounts_dict)

        elif period_type == "Current Year Estimate":
            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
            dec31 = pd.Timestamp(f"{ref_date.year}-12-31")
            return _get_budget_sum(uw_data, jan1, dec31, accounts_dict)

        else:  # Custom Month
            prior_month = ref_date - pd.DateOffset(months=1)
            return _get_budget_sum(uw_data, prior_month, ref_date, accounts_dict)

    elif source == "Valuation":
        if period_type == "TTM (Trailing Twelve Months)":
            start = ref_date - pd.DateOffset(months=12)
            return _get_valuation_sum(fc_deal_modeled, start.date(), ref_date.date(), accounts_dict)

        elif period_type == "YTD (Year to Date)":
            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
            return _get_valuation_sum(fc_deal_modeled, jan1.date(), ref_date.date(), accounts_dict)

        elif period_type == "Full Year":
            jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
            dec31 = pd.Timestamp(f"{year}-12-31")
            return _get_valuation_sum(fc_deal_modeled, jan1.date(), dec31.date(), accounts_dict)

        elif period_type == "Current Year Estimate":
            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
            dec31 = pd.Timestamp(f"{ref_date.year}-12-31")
            return _get_valuation_sum(fc_deal_modeled, jan1.date(), dec31.date(), accounts_dict)

        else:  # Custom Month
            prior_month = ref_date - pd.DateOffset(months=1)
            return _get_valuation_sum(fc_deal_modeled, prior_month.date(), ref_date.date(), accounts_dict)

    return {}


# ============================================================
# TENANT ROSTER (Commercial Properties Only)
# ============================================================

def _render_tenant_roster(deal_vcode, tenants_raw, inv):
    if tenants_raw is None or tenants_raw.empty:
        return

    # Filter tenants for current deal (Code column = vcode)
    tenants_raw_copy = tenants_raw.copy()
    tenants_raw_copy['Code'] = tenants_raw_copy['Code'].astype(str).str.strip()
    deal_tenants = tenants_raw_copy[tenants_raw_copy['Code'] == str(deal_vcode)].copy()

    if deal_tenants.empty:
        return

    st.subheader("Tenant Roster")

    # Clean and prepare data
    def clean_currency(val):
        """Convert currency string to float"""
        if pd.isna(val) or val == '':
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).replace('$', '').replace(',', '').strip())

    def clean_number(val):
        """Convert number string to float"""
        if pd.isna(val) or val == '':
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).replace(',', '').strip())

    # Parse numeric columns
    deal_tenants['SF_Leased'] = deal_tenants['SF Leased'].apply(clean_number)
    deal_tenants['Monthly_Rent'] = deal_tenants['Rent'].apply(clean_currency)
    deal_tenants['Annual_Rent'] = deal_tenants['Monthly_Rent'] * 12
    deal_tenants['Rentable_SF'] = deal_tenants['Rentable SF'].apply(clean_number)

    # Calculate RPSF (Annual Rent / SF Leased)
    deal_tenants['RPSF'] = deal_tenants.apply(
        lambda r: r['Annual_Rent'] / r['SF_Leased'] if r['SF_Leased'] > 0 else 0, axis=1
    )

    # Calculate % of GLA (SF Leased / Rentable SF)
    total_rentable_sf = deal_tenants['Rentable_SF'].iloc[0] if len(deal_tenants) > 0 else 0
    deal_tenants['Pct_GLA'] = deal_tenants['SF_Leased'] / total_rentable_sf if total_rentable_sf > 0 else 0

    # Calculate % of ABR (Annual Rent / Sum of all Annual Rent)
    total_annual_rent = deal_tenants['Annual_Rent'].sum()
    deal_tenants['Pct_ABR'] = deal_tenants['Annual_Rent'] / total_annual_rent if total_annual_rent > 0 else 0

    # Parse dates
    deal_tenants['Lease_Start'] = pd.to_datetime(deal_tenants['Lease Start'], errors='coerce')
    deal_tenants['Lease_End'] = pd.to_datetime(deal_tenants['Lease End'], errors='coerce')

    # Determine if lease expires within 2 years of current date
    two_years_from_now = pd.Timestamp(datetime.now()) + pd.DateOffset(years=2)
    deal_tenants['Expiring_Soon'] = (deal_tenants['Lease_End'] <= two_years_from_now) & (deal_tenants['Lease_End'] >= pd.Timestamp(datetime.now()))

    # Sort by SF Leased descending
    deal_tenants = deal_tenants.sort_values('SF_Leased', ascending=False).reset_index(drop=True)

    # Identify vacant vs occupied
    deal_tenants['Is_Vacant'] = deal_tenants['Tenant Name'].str.strip().str.lower() == 'vacant'

    # Calculate totals for non-vacant tenants
    occupied = deal_tenants[~deal_tenants['Is_Vacant']]
    total_occupied_sf = occupied['SF_Leased'].sum()
    total_sf_leased = deal_tenants['SF_Leased'].sum()
    occupancy_pct = total_occupied_sf / total_sf_leased if total_sf_leased > 0 else 0

    # Weighted average RPSF for leased (non-vacant) spaces
    weighted_rpsf = (occupied['Annual_Rent'].sum() / total_occupied_sf) if total_occupied_sf > 0 else 0

    # Build display dataframe with formatting
    display_rows = []
    for _, row in deal_tenants.iterrows():
        display_rows.append({
            'Tenant Name': row['Tenant Name'],
            'SF Leased': f"{row['SF_Leased']:,.0f}",
            'Lease Start': row['Lease_Start'].strftime('%m/%d/%Y') if pd.notna(row['Lease_Start']) else '',
            'Lease End': row['Lease_End'].strftime('%m/%d/%Y') if pd.notna(row['Lease_End']) else '',
            'Annual Rent': f"${row['Annual_Rent']:,.0f}",
            'RPSF': f"${row['RPSF']:,.2f}",
            '% of GLA': f"{row['Pct_GLA']:.1%}",
            '% of ABR': f"{row['Pct_ABR']:.1%}",
            '_expiring': row['Expiring_Soon'] and not row['Is_Vacant']
        })

    format_df = pd.DataFrame(display_rows)

    # Style function to highlight expiring leases
    def highlight_expiring(row):
        if row['_expiring']:
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)

    # Apply styling and display (hide the _expiring helper column)
    styled_df = format_df.style.apply(highlight_expiring, axis=1)

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            '_expiring': None  # Hide the helper column
        }
    )

    # Display totals
    st.markdown("---")
    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("Occupied SF", f"{total_occupied_sf:,.0f}")
    tcol2.metric("Occupancy %", f"{occupancy_pct:.1%}")
    tcol3.metric("Wtd Avg RPSF", f"${weighted_rpsf:,.2f}")

    # Legend
    st.caption("Yellow highlight indicates lease expiring within 2 years of current date")

    # ============================================================
    # LEASE ROLLOVER REPORT
    # ============================================================
    st.markdown("---")
    st.subheader("Lease Rollover Report")

    current_date = datetime.now()
    two_years_out = pd.Timestamp(current_date) + pd.DateOffset(years=2)

    # Get property info from investment_map
    prop_info = inv[inv['vcode'] == str(deal_vcode)].iloc[0] if not inv[inv['vcode'] == str(deal_vcode)].empty else {}
    prop_name = prop_info.get('Investment_Name', 'Unknown')
    partner = prop_info.get('Operating_Partner', 'N/A')
    location = f"{prop_info.get('City', '')}, {prop_info.get('State', '')}".strip(', ') or 'N/A'
    asset_type = prop_info.get('Asset_Type', 'N/A')
    strategy = prop_info.get('Lifecycle', 'N/A')
    acq_date = prop_info.get('Acquisition_Date', 'N/A')
    property_gla = clean_number(prop_info.get('Size_Sqf', 0)) or total_rentable_sf

    # Calculate metrics
    total_occupied_sf_rpt = occupied['SF_Leased'].sum()
    total_annual_rent_rpt = occupied['Annual_Rent'].sum()
    current_occupancy = total_occupied_sf_rpt / property_gla if property_gla > 0 else 0
    property_avg_rpsf = total_annual_rent_rpt / total_occupied_sf_rpt if total_occupied_sf_rpt > 0 else 0

    # 2-Year Lease Maturity Exposure (non-vacant, expiring within 2 years)
    expiring_2yr = deal_tenants[
        (deal_tenants['Expiring_Soon']) &
        (~deal_tenants['Is_Vacant'])
    ]
    exposure_gla = expiring_2yr['SF_Leased'].sum()
    exposure_abr = expiring_2yr['Annual_Rent'].sum()
    exposure_gla_pct = exposure_gla / property_gla if property_gla > 0 else 0
    exposure_abr_pct = exposure_abr / total_annual_rent_rpt if total_annual_rent_rpt > 0 else 0
    exposure_rpsf = exposure_abr / exposure_gla if exposure_gla > 0 else 0

    # Vacancy calculations
    vacant_tenants = deal_tenants[deal_tenants['Is_Vacant']]
    total_vacant_gla = vacant_tenants['SF_Leased'].sum()
    vacancy_loss = total_vacant_gla * property_avg_rpsf

    # Calculate remaining years for each tenant
    deal_tenants['Remain_Years'] = deal_tenants['Lease_End'].apply(
        lambda x: max(0, (x - pd.Timestamp(current_date)).days / 365) if pd.notna(x) else 0
    )

    # Group by lease end year for maturity profile
    deal_tenants['End_Year'] = deal_tenants.apply(
        lambda r: 'Vacant' if r['Is_Vacant'] else (str(r['Lease_End'].year) if pd.notna(r['Lease_End']) else 'Unknown'),
        axis=1
    )

    maturity_by_year = deal_tenants.groupby('End_Year').agg({
        'SF_Leased': 'sum',
        'Annual_Rent': 'sum'
    }).reset_index()
    maturity_by_year['Avg_RPSF'] = maturity_by_year.apply(
        lambda r: r['Annual_Rent'] / r['SF_Leased'] if r['SF_Leased'] > 0 else 0, axis=1
    )
    maturity_by_year['Pct_Revenue'] = maturity_by_year['Annual_Rent'] / total_annual_rent_rpt if total_annual_rent_rpt > 0 else 0

    # Top 10 tenants by revenue (non-vacant)
    top_10 = occupied.nlargest(10, 'Annual_Rent').copy()
    top_10['GLA_Pct'] = top_10['SF_Leased'] / property_gla if property_gla > 0 else 0
    top_10['Rent_Pct'] = top_10['Annual_Rent'] / total_annual_rent_rpt if total_annual_rent_rpt > 0 else 0
    top_10['Remain_Years'] = top_10['Lease_End'].apply(
        lambda x: max(0, (x - pd.Timestamp(current_date)).days / 365) if pd.notna(x) else 0
    )

    # Near term maturities (within 2 years, non-vacant, sorted by end date)
    near_term = expiring_2yr.sort_values('Lease_End').head(10).copy()
    near_term['Remain_Years'] = near_term['Lease_End'].apply(
        lambda x: max(0, (x - pd.Timestamp(current_date)).days / 365) if pd.notna(x) else 0
    )

    # Vacancies by size
    vacancies_by_size = vacant_tenants.nlargest(10, 'SF_Leased')[['Tenant Name', 'SF_Leased']].copy()

    # --- Display Header Section ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"""
        **Partner:** {partner}
        **Location:** {location}
        **Asset Type:** {asset_type}
        **Strategy:** {strategy}
        **Acquisition Date:** {acq_date}
        **GLA:** {property_gla:,.0f} sf
        """)
    with col_right:
        st.markdown(f"**Current Occupancy:** {current_occupancy:.1%}")
        st.markdown("**2 Year Lease Maturity Exposure:**")
        st.markdown(f"- GLA: {exposure_gla:,.0f} ({exposure_gla_pct:.1%})")
        st.markdown(f"- ABR: ${exposure_abr:,.0f} ({exposure_abr_pct:.1%})")
        st.markdown(f"- ABR/SF: ${exposure_rpsf:.2f}")
        st.markdown(f"**Property Avg ABR/SF:** ${property_avg_rpsf:.2f}")

    # --- Lease Maturity Profile Chart ---
    st.markdown("#### Lease Maturity Profile")

    # Prepare chart data - include all years in 10-year forecast
    chart_data = maturity_by_year.copy()
    chart_data = chart_data[chart_data['End_Year'] != 'Vacant']  # Exclude vacant for chart

    # Create full 10-year range from current year
    current_year_val = current_date.year
    all_years = [str(y) for y in range(current_year_val, current_year_val + 11)]

    # Build complete chart dataframe with all years
    chart_rows = []
    for yr in all_years:
        year_data = chart_data[chart_data['End_Year'] == yr]
        if not year_data.empty:
            chart_rows.append({
                'Year': yr,
                'Annual Rent': year_data['Annual_Rent'].iloc[0],
                'Avg RPSF': year_data['Avg_RPSF'].iloc[0],
                '% of Rent': year_data['Pct_Revenue'].iloc[0] * 100
            })
        else:
            chart_rows.append({
                'Year': yr,
                'Annual Rent': 0,
                'Avg RPSF': 0,
                '% of Rent': 0
            })

    chart_df = pd.DataFrame(chart_rows)

    import altair as alt

    # Create layered chart
    base = alt.Chart(chart_df).encode(
        x=alt.X('Year:N', title='Lease End Year', sort=all_years)
    )

    bars = base.mark_bar(color='#4472C4').encode(
        y=alt.Y('Annual Rent:Q', title='Annual Rent ($)')
    )

    max_rpsf = chart_df['Avg RPSF'].max()
    line = base.mark_line(color='#ED7D31', strokeWidth=2, point=True).encode(
        y=alt.Y('Avg RPSF:Q', title='Avg RPSF ($)', scale=alt.Scale(domain=[0, max(max_rpsf * 1.2, 1)]))
    )

    chart = alt.layer(bars, line).resolve_scale(y='independent').properties(height=250)
    st.altair_chart(chart, use_container_width=True)

    # --- Lease Maturity Table ---
    st.markdown("#### Lease Maturity by Year")
    maturity_display = maturity_by_year.copy()
    maturity_display.columns = ['Year', 'GLA', 'Gross Rent', 'Avg RPSF', '% of Rev']
    maturity_display['GLA'] = maturity_display['GLA'].apply(lambda x: f"{x:,.0f}")
    maturity_display['Gross Rent'] = maturity_display['Gross Rent'].apply(lambda x: f"${x:,.0f}")
    maturity_display['Avg RPSF'] = maturity_display['Avg RPSF'].apply(lambda x: f"${x:.2f}")
    maturity_display['% of Rev'] = maturity_display['% of Rev'].apply(lambda x: f"{x:.1%}")
    st.dataframe(maturity_display, use_container_width=True, hide_index=True)

    # --- Top 10 Tenants by Revenue ---
    st.markdown("#### Top 10 Tenants by Revenue")
    if not top_10.empty:
        top_10_display = pd.DataFrame({
            'Tenant Name': top_10['Tenant Name'],
            'GLA': top_10['SF_Leased'].apply(lambda x: f"{x:,.0f}"),
            '% GLA': top_10['GLA_Pct'].apply(lambda x: f"{x:.0%}"),
            'Annual Rent': top_10['Annual_Rent'].apply(lambda x: f"${x:,.0f}"),
            '% Rent': top_10['Rent_Pct'].apply(lambda x: f"{x:.0%}"),
            'RPSF': top_10['RPSF'].apply(lambda x: f"${x:.2f}"),
            'Start': top_10['Lease_Start'].apply(lambda x: x.strftime('%m/%d/%Y') if pd.notna(x) else ''),
            'End': top_10['Lease_End'].apply(lambda x: x.strftime('%m/%d/%Y') if pd.notna(x) else ''),
            'Remain': top_10['Remain_Years'].apply(lambda x: f"{x:.1f}")
        })
        st.dataframe(top_10_display, use_container_width=True, hide_index=True)

    # --- Comments Section ---
    st.markdown("#### Comments")
    rollover_comment_key = f"rollover_comment_{deal_vcode}"
    rollover_comment = st.text_area(
        "Add comments about lease rollover",
        value=st.session_state.get(rollover_comment_key, ""),
        key=rollover_comment_key,
        height=100
    )

    # --- Print Report Button ---
    st.markdown("---")
    if st.button("Print Lease Rollover Report", key="print_rollover"):
        # Generate print-friendly HTML
        print_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{prop_name} - Lease Rollover Report</title>
            <style>
                @page {{ size: letter landscape; margin: 0.5in; }}
                @media print {{
                    body {{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
                }}
                body {{ font-family: Arial, sans-serif; font-size: 9pt; margin: 0; padding: 10px; }}
                .header {{ display: flex; justify-content: space-between; border-bottom: 2px solid #000; padding-bottom: 5px; margin-bottom: 10px; }}
                .header h1 {{ margin: 0; font-size: 18pt; }}
                .header .date {{ font-size: 10pt; }}
                .info-section {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
                .info-left, .info-right {{ width: 48%; }}
                .info-row {{ margin: 2px 0; }}
                .info-label {{ font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 5px 0; font-size: 8pt; }}
                th, td {{ border: 1px solid #ccc; padding: 3px 5px; text-align: left; }}
                th {{ background-color: #f0f0f0; font-weight: bold; }}
                .section-title {{ font-weight: bold; font-size: 10pt; margin: 10px 0 5px 0; border-bottom: 1px solid #000; }}
                .highlight {{ background-color: #fff3cd !important; }}
                .two-col {{ display: flex; justify-content: space-between; }}
                .two-col > div {{ width: 48%; }}
                .vacancy-item {{ margin: 2px 0; }}
                .totals {{ margin-top: 10px; font-weight: bold; }}
                .comments {{ margin-top: 10px; border: 1px solid #ccc; padding: 5px; min-height: 50px; }}
                .right-align {{ text-align: right; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{prop_name}</h1>
                <div class="date">As of: {current_date.strftime('%m/%d/%y')}</div>
            </div>

            <div class="info-section">
                <div class="info-left">
                    <div class="info-row"><span class="info-label">Partner:</span> {partner}</div>
                    <div class="info-row"><span class="info-label">Location:</span> {location}</div>
                    <div class="info-row"><span class="info-label">Asset Type:</span> {asset_type}</div>
                    <div class="info-row"><span class="info-label">Strategy:</span> {strategy}</div>
                    <div class="info-row"><span class="info-label">Acquisition Date:</span> {acq_date}</div>
                    <div class="info-row"><span class="info-label">GLA:</span> {property_gla:,.0f} sf</div>
                </div>
                <div class="info-right">
                    <div class="info-row"><span class="info-label">Current Occupancy:</span> {current_occupancy:.1%}</div>
                    <div class="info-row"><span class="info-label">2 Year Lease Maturity Exposure:</span></div>
                    <div class="info-row">&nbsp;&nbsp;&nbsp;&nbsp;GLA: {exposure_gla:,.0f} ({exposure_gla_pct:.1%})</div>
                    <div class="info-row">&nbsp;&nbsp;&nbsp;&nbsp;ABR: ${exposure_abr:,.0f} ({exposure_abr_pct:.1%})</div>
                    <div class="info-row">&nbsp;&nbsp;&nbsp;&nbsp;ABR/SF: ${exposure_rpsf:.2f}</div>
                    <div class="info-row"><span class="info-label">Property Avg ABR/SF:</span> ${property_avg_rpsf:.2f}</div>
                </div>
            </div>

            <div class="section-title">Lease Maturity by Year</div>
            <table>
                <tr><th>Year</th><th class="right-align">GLA</th><th class="right-align">Gross Rent</th><th class="right-align">Avg RPSF</th><th class="right-align">% of Rev</th></tr>
        """

        for _, row in maturity_by_year.iterrows():
            print_html += f"""
                <tr>
                    <td>{row['End_Year']}</td>
                    <td class="right-align">{row['SF_Leased']:,.0f}</td>
                    <td class="right-align">${row['Annual_Rent']:,.0f}</td>
                    <td class="right-align">${row['Avg_RPSF']:.2f}</td>
                    <td class="right-align">{row['Pct_Revenue']:.1%}</td>
                </tr>
            """

        print_html += """
            </table>

            <div class="section-title">Top 10 Tenants by Revenue</div>
            <table>
                <tr><th>Tenant Name</th><th class="right-align">GLA</th><th class="right-align">%</th><th class="right-align">Annual Rent</th><th class="right-align">%</th><th class="right-align">RPSF</th><th>Start</th><th>End</th><th class="right-align">Remain</th></tr>
        """

        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            start_str = row['Lease_Start'].strftime('%m/%d/%Y') if pd.notna(row['Lease_Start']) else ''
            end_str = row['Lease_End'].strftime('%m/%d/%Y') if pd.notna(row['Lease_End']) else ''
            gla_pct = row['SF_Leased'] / property_gla if property_gla > 0 else 0
            rent_pct = row['Annual_Rent'] / total_annual_rent_rpt if total_annual_rent_rpt > 0 else 0
            remain = max(0, (row['Lease_End'] - pd.Timestamp(current_date)).days / 365) if pd.notna(row['Lease_End']) else 0
            print_html += f"""
                <tr>
                    <td>{i}) {row['Tenant Name']}</td>
                    <td class="right-align">{row['SF_Leased']:,.0f}</td>
                    <td class="right-align">{gla_pct:.0%}</td>
                    <td class="right-align">${row['Annual_Rent']:,.0f}</td>
                    <td class="right-align">{rent_pct:.0%}</td>
                    <td class="right-align">${row['RPSF']:.2f}</td>
                    <td>{start_str}</td>
                    <td>{end_str}</td>
                    <td class="right-align">{remain:.1f}</td>
                </tr>
            """

        print_html += f"""
            </table>

            <div class="section-title">Comments</div>
            <div class="comments">{rollover_comment or '&nbsp;'}</div>
        </body>
        </html>
        """

        # Display in new window via components
        import streamlit.components.v1 as components
        components.html(f"""
            <script>
                var printWindow = window.open('', '_blank');
                printWindow.document.write(`{print_html.replace('`', '\\`')}`);
                printWindow.document.close();
                printWindow.focus();
                printWindow.print();
            </script>
        """, height=0)


# ============================================================
# ONE PAGER INVESTOR REPORT
# ============================================================

def _render_one_pager(deal_vcode, isbs_raw, inv, mri_loans_raw, mri_val,
                      wf, commitments_raw, acct, occupancy_raw):
    if isbs_raw is None or isbs_raw.empty:
        return

    st.subheader("One Pager Investor Report")
    render_one_pager_section(
        vcode=deal_vcode,
        inv_map=inv,
        isbs_df=isbs_raw,
        mri_loans=mri_loans_raw,
        mri_val=mri_val,
        waterfalls=wf,
        commitments=commitments_raw,
        acct=acct,
        occupancy_df=occupancy_raw,
    )
