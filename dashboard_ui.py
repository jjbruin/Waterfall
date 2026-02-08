"""
dashboard_ui.py
Executive Portfolio Dashboard — KPI cards, portfolio charts, and button-gated computed returns.

Provides a portfolio-level view of all investments at a glance.
Lightweight sections load instantly; computed returns (IRR/MOIC) are button-gated.
"""

import streamlit as st
import pandas as pd
import altair as alt

from config import IS_ACCOUNTS
from compute import get_deal_capitalization, compute_deal_analysis
from consolidation import get_property_vcodes_for_deal
from reports_ui import _build_deal_lookup, _build_partner_returns


# ============================================================
# COLOUR PALETTE  (matches property_financials_ui.py)
# ============================================================
CLR_DARK = '#1F4E79'
CLR_ACCENT = '#ED7D31'
CLR_LIGHT = '#B4D4F0'


# ============================================================
# HELPER — PORTFOLIO CAPITALIZATIONS  (lightweight, no waterfall)
# ============================================================
def _get_portfolio_caps(inv_disp, inv, wf, acct, mri_val, mri_loans_raw):
    """Return capitalisation dicts for every parent deal.

    Cached in session_state to avoid re-computation on Streamlit reruns.
    """
    cached = st.session_state.get('_dashboard_caps')
    if cached is not None:
        return cached

    caps = []
    for _, row in inv_disp.iterrows():
        vcode = str(row['vcode'])
        prop_vcodes = get_property_vcodes_for_deal(vcode, inv)
        cap = get_deal_capitalization(
            acct, inv, wf, mri_val, mri_loans_raw,
            deal_vcode=vcode,
            property_vcodes=prop_vcodes or None,
        )
        cap['vcode'] = vcode
        cap['name'] = row.get('DealLabel', row.get('Investment_Name', vcode))
        cap['asset_type'] = str(row.get('Asset_Type', '') or '')
        cap['total_units'] = pd.to_numeric(row.get('Total_Units', 0), errors='coerce') or 0
        caps.append(cap)

    st.session_state['_dashboard_caps'] = caps
    return caps


# ============================================================
# HELPER — LATEST OCCUPANCY PER DEAL
# ============================================================
def _get_latest_occupancy(inv_disp, occupancy_raw):
    """Return dict {vcode: latest Occ%} for each deal."""
    if occupancy_raw is None or occupancy_raw.empty:
        return {}

    occ = occupancy_raw.copy()
    occ.columns = [str(c).strip() for c in occ.columns]

    if 'vCode' not in occ.columns:
        return {}

    occ['vCode'] = occ['vCode'].astype(str).str.strip().str.lower()

    occ_col = 'Occ%' if 'Occ%' in occ.columns else (
        'OccupancyPercent' if 'OccupancyPercent' in occ.columns else None)
    if occ_col is None or 'dtReported' not in occ.columns:
        return {}

    occ['occ_val'] = pd.to_numeric(occ[occ_col], errors='coerce')
    try:
        occ['date_parsed'] = pd.to_datetime(
            occ['dtReported'], unit='D', origin='1899-12-30', errors='coerce')
    except Exception:
        occ['date_parsed'] = pd.to_datetime(occ['dtReported'], errors='coerce')
    occ = occ.dropna(subset=['date_parsed', 'occ_val'])

    occ_map = {}
    for _, row in inv_disp.iterrows():
        vc = str(row['vcode']).strip().lower()
        deal_occ = occ[occ['vCode'] == vc]
        if not deal_occ.empty:
            latest = deal_occ.loc[deal_occ['date_parsed'].idxmax()]
            occ_map[str(row['vcode'])] = float(latest['occ_val'])

    return occ_map


# ============================================================
# HELPER — PORTFOLIO NOI DATA
# ============================================================
def _compute_portfolio_noi(isbs_raw, inv_disp, frequency, period_end_label,
                           occupancy_raw):
    """Aggregate NOI across all parent deals and return a chart DataFrame.

    Returns (chart_df, period_order) or (None, None) on insufficient data.
    """
    if isbs_raw is None or isbs_raw.empty:
        return None, None

    isbs = isbs_raw.copy()
    isbs.columns = [str(c).strip() for c in isbs.columns]

    # Filter to parent deal vcodes only
    parent_vcodes = set(inv_disp['vcode'].astype(str).str.strip().str.lower())
    if 'vcode' in isbs.columns:
        isbs['vcode'] = isbs['vcode'].astype(str).str.strip().str.lower()
        isbs = isbs[isbs['vcode'].isin(parent_vcodes)]

    if isbs.empty or 'dtEntry' not in isbs.columns:
        return None, None

    # Parse dates
    try:
        isbs['dtEntry_parsed'] = pd.to_datetime(
            isbs['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
    except Exception:
        isbs['dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], errors='coerce')
    null_dates = isbs['dtEntry_parsed'].isna()
    if null_dates.any():
        isbs.loc[null_dates, 'dtEntry_parsed'] = pd.to_datetime(
            isbs.loc[null_dates, 'dtEntry'], errors='coerce')

    if 'vSource' in isbs.columns:
        isbs['vSource'] = isbs['vSource'].astype(str).str.strip()
    if 'vAccount' in isbs.columns:
        isbs['vAccount'] = isbs['vAccount'].astype(str).str.strip()
    if 'mAmount' in isbs.columns:
        isbs['mAmount'] = pd.to_numeric(isbs['mAmount'], errors='coerce').fillna(0)

    actual_data = isbs[isbs['vSource'] == 'Interim IS']
    uw_data = isbs[isbs['vSource'] == 'Projected IS']

    if actual_data.empty and uw_data.empty:
        return None, None

    # Flatten account codes
    rev_accounts = []
    for acct_list in IS_ACCOUNTS['REVENUES'].values():
        rev_accounts.extend(acct_list)
    exp_accounts = []
    for acct_list in IS_ACCOUNTS['EXPENSES'].values():
        exp_accounts.extend(acct_list)

    def _compute_cumulative_noi(data, dates):
        noi_by_date = {}
        for dt in dates:
            period = data[data['dtEntry_parsed'] == dt]
            rev = period[period['vAccount'].isin(rev_accounts)]['mAmount'].sum()
            exp = period[period['vAccount'].isin(exp_accounts)]['mAmount'].sum()
            noi_by_date[dt] = (-rev) - exp
        return noi_by_date

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
                if prior is not None:
                    periodic[dt_ts] = cum_dict[dt] - cum_dict[prior]
                else:
                    periodic[dt_ts] = cum_dict[dt]
        return periodic

    def _aggregate_periodic(periodic_dict, freq):
        if not periodic_dict:
            return {}
        if freq == "Monthly":
            return periodic_dict
        elif freq == "Quarterly":
            quarterly = {}
            for dt, val in sorted(periodic_dict.items()):
                dt_ts = pd.Timestamp(dt)
                q_month = ((dt_ts.month - 1) // 3 + 1) * 3
                q_end = pd.Timestamp(year=dt_ts.year, month=q_month, day=1) + pd.offsets.MonthEnd(0)
                quarterly[q_end] = quarterly.get(q_end, 0) + val
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
            month_counts = {}
            for dt in periodic_dict:
                yr_end = pd.Timestamp(year=pd.Timestamp(dt).year, month=12, day=31)
                month_counts[yr_end] = month_counts.get(yr_end, 0) + 1
            return {k: v for k, v in annual.items() if month_counts.get(k, 0) == 12}

    # Per-deal: cumulative → periodic, then SUM across deals
    actual_periodic_total = {}
    uw_periodic_total = {}

    for vc in parent_vcodes:
        for source_label, source_data, target in [
            ('actual', actual_data, actual_periodic_total),
            ('uw', uw_data, uw_periodic_total),
        ]:
            deal_src = source_data[source_data['vcode'] == vc] if 'vcode' in source_data.columns else source_data
            if deal_src.empty:
                continue
            dates = sorted(deal_src['dtEntry_parsed'].dropna().unique())
            cum = _compute_cumulative_noi(deal_src, dates)
            periodic = _cumulative_to_periodic(cum, dates)
            for dt, val in periodic.items():
                target[dt] = target.get(dt, 0) + val

    actual_agg = _aggregate_periodic(actual_periodic_total, frequency)
    uw_agg = _aggregate_periodic(uw_periodic_total, frequency)

    all_period_ends = sorted(set(actual_agg.keys()) | set(uw_agg.keys()))
    if not all_period_ends:
        return None, None

    period_end_labels = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in all_period_ends]

    # Determine display window (trailing 12 from selected period_end)
    if period_end_label and period_end_label in period_end_labels:
        sel_idx = period_end_labels.index(period_end_label)
    else:
        sel_idx = len(period_end_labels) - 1
    start_idx = max(0, sel_idx - 11)
    display_dates = all_period_ends[start_idx:sel_idx + 1]

    # Portfolio-level average occupancy per period
    occ_by_period = {}
    if occupancy_raw is not None and not occupancy_raw.empty:
        occ = occupancy_raw.copy()
        occ.columns = [str(c).strip() for c in occ.columns]
        if 'vCode' in occ.columns:
            occ['vCode'] = occ['vCode'].astype(str).str.strip().str.lower()
            occ = occ[occ['vCode'].isin(parent_vcodes)]
        occ_col = 'Occ%' if 'Occ%' in occ.columns else (
            'OccupancyPercent' if 'OccupancyPercent' in occ.columns else None)
        if occ_col and 'dtReported' in occ.columns:
            occ['occ_val'] = pd.to_numeric(occ[occ_col], errors='coerce')
            try:
                occ['date_parsed'] = pd.to_datetime(
                    occ['dtReported'], unit='D', origin='1899-12-30', errors='coerce')
            except Exception:
                occ['date_parsed'] = pd.to_datetime(occ['dtReported'], errors='coerce')
            occ = occ.dropna(subset=['date_parsed', 'occ_val'])

            if not occ.empty:
                monthly_occ = {}
                for _, r in occ.iterrows():
                    me = pd.Timestamp(r['date_parsed']) + pd.offsets.MonthEnd(0)
                    monthly_occ.setdefault(me, []).append(r['occ_val'])
                # Average across deals per month
                monthly_occ_avg = {k: sum(v) / len(v) for k, v in monthly_occ.items()}

                for dt in display_dates:
                    dt_ts = pd.Timestamp(dt)
                    if frequency == "Monthly":
                        me = dt_ts + pd.offsets.MonthEnd(0)
                        if me in monthly_occ_avg:
                            occ_by_period[dt_ts] = monthly_occ_avg[me]
                    elif frequency == "Quarterly":
                        q_start_month = ((dt_ts.month - 1) // 3) * 3 + 1
                        vals = []
                        for m in range(q_start_month, q_start_month + 3):
                            me = pd.Timestamp(year=dt_ts.year, month=m, day=1) + pd.offsets.MonthEnd(0)
                            if me in monthly_occ_avg:
                                vals.append(monthly_occ_avg[me])
                        if vals:
                            occ_by_period[dt_ts] = sum(vals) / len(vals)
                    else:
                        yr_vals = [v for k, v in monthly_occ_avg.items()
                                   if pd.Timestamp(k).year == dt_ts.year]
                        if yr_vals:
                            occ_by_period[dt_ts] = sum(yr_vals) / len(yr_vals)

    # Build chart rows
    chart_rows = []
    for dt in display_dates:
        dt_ts = pd.Timestamp(dt)
        label = (dt_ts.strftime('%b %Y') if frequency == "Monthly" else
                 f"Q{(dt_ts.month - 1) // 3 + 1} {dt_ts.year}" if frequency == "Quarterly" else
                 str(dt_ts.year))
        chart_rows.append({
            'Period': label,
            '_sort': dt_ts,
            'Actual NOI': actual_agg.get(dt_ts, None),
            'Underwritten NOI': uw_agg.get(dt_ts, None),
            'Occupancy': occ_by_period.get(dt_ts, None),
        })

    chart_df = pd.DataFrame(chart_rows)
    period_order = chart_df['Period'].tolist()
    return chart_df, period_order


# ============================================================
# RENDER — KPI CARDS
# ============================================================
def _render_kpi_cards(caps, occ_map, inv_disp):
    """Six metric cards across the top of the dashboard."""
    total_value = sum(c.get('current_valuation', 0) for c in caps)
    total_debt = sum(c.get('debt', 0) for c in caps)
    total_pref = sum(c.get('pref_equity', 0) for c in caps)
    total_partner = sum(c.get('partner_equity', 0) for c in caps)
    total_equity = total_pref + total_partner
    deal_count = len(caps)

    # Weighted avg cap rate (by valuation)
    cap_rate_num = sum(c.get('cap_rate', 0) * c.get('current_valuation', 0) for c in caps)
    wtd_cap_rate = cap_rate_num / total_value if total_value > 0 else 0.0

    # Weighted avg occupancy (by units)
    occ_num = 0.0
    occ_denom = 0.0
    for c in caps:
        vc = c.get('vcode', '')
        units = c.get('total_units', 0) or 0
        occ = occ_map.get(vc)
        if occ is not None and units > 0:
            occ_num += occ * units
            occ_denom += units
    wtd_occ = occ_num / occ_denom if occ_denom > 0 else 0.0

    cols = st.columns(6)
    with cols[0]:
        st.metric("Portfolio Value", f"${total_value:,.0f}")
    with cols[1]:
        st.metric("Debt Outstanding", f"${total_debt:,.0f}")
    with cols[2]:
        st.metric("Wtd Avg Cap Rate", f"{wtd_cap_rate:.2%}")
    with cols[3]:
        st.metric("Portfolio Occupancy", f"{wtd_occ:.1f}%")
    with cols[4]:
        st.metric("Deal Count", str(deal_count))
    with cols[5]:
        st.metric("Total Equity", f"${total_equity:,.0f}")


# ============================================================
# RENDER — PORTFOLIO NOI TREND  (Altair dual-axis)
# ============================================================
def _render_portfolio_noi_chart(isbs_raw, inv_disp, occupancy_raw):
    """Portfolio-level NOI trend with frequency/period controls."""
    st.subheader("Portfolio NOI Trend")

    if isbs_raw is None or isbs_raw.empty:
        st.info("No income statement data available.")
        return

    freq_options = ["Monthly", "Quarterly", "Annually"]
    ctrl_cols = st.columns([1, 1, 2])
    with ctrl_cols[0]:
        frequency = st.selectbox("Frequency", freq_options, index=1, key="dash_noi_freq")

    # Pre-compute to get period labels for the selector
    chart_df_pre, _ = _compute_portfolio_noi(
        isbs_raw, inv_disp, frequency, None, occupancy_raw)
    if chart_df_pre is None or chart_df_pre.empty:
        st.info("Not enough data for the selected frequency.")
        return

    # Build period-end labels from isbs for selector
    isbs_tmp = isbs_raw.copy()
    isbs_tmp.columns = [str(c).strip() for c in isbs_tmp.columns]
    parent_vcodes = set(inv_disp['vcode'].astype(str).str.strip().str.lower())
    if 'vcode' in isbs_tmp.columns:
        isbs_tmp['vcode'] = isbs_tmp['vcode'].astype(str).str.strip().str.lower()
        isbs_tmp = isbs_tmp[isbs_tmp['vcode'].isin(parent_vcodes)]

    # We already have the chart — extract available period labels
    period_end_labels_display = chart_df_pre['Period'].tolist()

    # Default to most recently ended quarter (relative to today)
    today = pd.Timestamp.today()
    current_q_month = ((today.month - 1) // 3) * 3  # last month of prior quarter
    if current_q_month == 0:
        last_q_end = pd.Timestamp(year=today.year - 1, month=12, day=31)
    else:
        last_q_end = pd.Timestamp(year=today.year, month=current_q_month, day=1) + pd.offsets.MonthEnd(0)
    last_q_label = f"Q{(last_q_end.month - 1) // 3 + 1} {last_q_end.year}"

    if last_q_label in period_end_labels_display:
        default_pe_idx = period_end_labels_display.index(last_q_label)
    else:
        default_pe_idx = len(period_end_labels_display) - 1

    with ctrl_cols[1]:
        period_end_sel = st.selectbox("Period End", period_end_labels_display,
                                      index=default_pe_idx, key="dash_noi_pe")

    # The chart_df_pre already windows to trailing 12 from the last period;
    # use it directly (the selector merely re-confirms the end).
    chart_df = chart_df_pre
    period_order = chart_df['Period'].tolist()

    has_occupancy = chart_df['Occupancy'].notna().any()

    if has_occupancy:
        noi_df = chart_df.melt(id_vars=['Period'],
                               value_vars=['Actual NOI', 'Underwritten NOI'],
                               var_name='Series', value_name='NOI')
        noi_df = noi_df.dropna(subset=['NOI'])

        occ_df = chart_df[['Period', 'Occupancy']].dropna(subset=['Occupancy'])

        bars = alt.Chart(occ_df).mark_bar(color=CLR_LIGHT, opacity=0.6).encode(
            x=alt.X('Period:N', sort=period_order, title='Period'),
            y=alt.Y('Occupancy:Q', title='Occupancy %', scale=alt.Scale(domain=[0, 100])),
        )

        color_scale = alt.Scale(domain=['Actual NOI', 'Underwritten NOI'],
                                range=[CLR_DARK, CLR_ACCENT])
        dash_scale = alt.Scale(domain=['Actual NOI', 'Underwritten NOI'],
                               range=[[0], [5, 5]])

        lines = alt.Chart(noi_df).mark_line(point=True).encode(
            x=alt.X('Period:N', sort=period_order),
            y=alt.Y('NOI:Q', title='NOI ($)'),
            color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash('Series:N', scale=dash_scale, legend=alt.Legend(title=None)),
        )

        chart = alt.layer(bars, lines).resolve_scale(y='independent').properties(height=350)
    else:
        noi_df = chart_df.melt(id_vars=['Period'],
                               value_vars=['Actual NOI', 'Underwritten NOI'],
                               var_name='Series', value_name='NOI')
        noi_df = noi_df.dropna(subset=['NOI'])
        if noi_df.empty:
            st.info("No NOI data available for the selected periods.")
            return
        color_scale = alt.Scale(domain=['Actual NOI', 'Underwritten NOI'],
                                range=[CLR_DARK, CLR_ACCENT])
        dash_scale = alt.Scale(domain=['Actual NOI', 'Underwritten NOI'],
                               range=[[0], [5, 5]])
        chart = alt.Chart(noi_df).mark_line(point=True).encode(
            x=alt.X('Period:N', sort=period_order, title='Period'),
            y=alt.Y('NOI:Q', title='NOI ($)'),
            color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash('Series:N', scale=dash_scale, legend=alt.Legend(title=None)),
        ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)


# ============================================================
# RENDER — CAPITAL STRUCTURE BY DEAL
# ============================================================
def _render_capital_stack_chart(caps):
    """Horizontal stacked bar chart: Debt / Pref Equity / Partner Equity per deal."""
    st.subheader("Capital Structure by Deal")

    rows = []
    for c in caps:
        name = c.get('name', c.get('vcode', ''))
        for cat, key in [('Debt', 'debt'), ('Pref Equity', 'pref_equity'),
                         ('Partner Equity', 'partner_equity')]:
            rows.append({'Deal': name, 'Category': cat, 'Value': c.get(key, 0)})

    if not rows:
        st.info("No capitalisation data available.")
        return

    df = pd.DataFrame(rows)
    # Sort deals by total cap descending
    deal_totals = df.groupby('Deal')['Value'].sum().sort_values(ascending=True)
    deal_order = deal_totals.index.tolist()

    color_scale = alt.Scale(
        domain=['Debt', 'Pref Equity', 'Partner Equity'],
        range=[CLR_DARK, CLR_ACCENT, CLR_LIGHT],
    )

    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('Deal:N', sort=deal_order, title=None),
        x=alt.X('Value:Q', title='Amount ($)', stack='zero'),
        color=alt.Color('Category:N', scale=color_scale, legend=alt.Legend(title=None)),
        tooltip=['Deal', 'Category', alt.Tooltip('Value:Q', format='$,.0f')],
    ).properties(height=max(250, len(caps) * 28))

    st.altair_chart(chart, use_container_width=True)


# ============================================================
# RENDER — OCCUPANCY BY DEAL
# ============================================================
def _render_occupancy_by_deal(occ_map, inv_disp):
    """Horizontal bar chart: occupancy per deal, coloured vs portfolio avg."""
    st.subheader("Occupancy by Deal")

    if not occ_map:
        st.info("No occupancy data available.")
        return

    rows = []
    for _, row in inv_disp.iterrows():
        vc = str(row['vcode'])
        occ = occ_map.get(vc)
        if occ is not None:
            rows.append({
                'Deal': row.get('DealLabel', row.get('Investment_Name', vc)),
                'Occupancy': occ,
            })

    if not rows:
        st.info("No occupancy data for displayed deals.")
        return

    df = pd.DataFrame(rows).sort_values('Occupancy', ascending=True)

    # Portfolio weighted average (simple mean here since we don't have units in df)
    avg_occ = df['Occupancy'].mean()

    df['Color'] = df['Occupancy'].apply(lambda v: CLR_DARK if v >= avg_occ else CLR_ACCENT)

    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y('Deal:N', sort=alt.EncodingSortField(field='Occupancy', order='ascending'), title=None),
        x=alt.X('Occupancy:Q', title='Occupancy %', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Color:N', scale=None),
        tooltip=['Deal', alt.Tooltip('Occupancy:Q', format='.1f')],
    ).properties(height=max(250, len(df) * 28))

    rule = alt.Chart(pd.DataFrame({'avg': [avg_occ]})).mark_rule(
        strokeDash=[5, 5], color='#333', strokeWidth=1.5
    ).encode(x='avg:Q')

    st.altair_chart((bars + rule), use_container_width=True)


# ============================================================
# RENDER — ASSET ALLOCATION  (donut chart)
# ============================================================
def _render_asset_allocation(caps, inv_disp):
    """Donut chart by Asset_Type, sized by valuation."""
    st.subheader("Asset Allocation")

    rows = []
    for c in caps:
        at = c.get('asset_type', '').strip()
        if not at:
            at = 'Unknown'
        rows.append({
            'Asset_Type': at,
            'Valuation': c.get('current_valuation', 0),
        })

    if not rows:
        st.info("No asset allocation data available.")
        return

    df = pd.DataFrame(rows)
    agg = df.groupby('Asset_Type').agg(
        Total_Value=('Valuation', 'sum'),
        Count=('Valuation', 'size'),
    ).reset_index().sort_values('Total_Value', ascending=False)

    if agg.empty or agg['Total_Value'].sum() == 0:
        st.info("No valuation data for asset allocation chart.")
        return

    col_chart, col_legend = st.columns([2, 1])

    with col_chart:
        donut = alt.Chart(agg).mark_arc(innerRadius=50).encode(
            theta=alt.Theta('Total_Value:Q', stack=True),
            color=alt.Color('Asset_Type:N', legend=None),
            tooltip=['Asset_Type', alt.Tooltip('Total_Value:Q', format='$,.0f'), 'Count:Q'],
        ).properties(height=250, width=250)
        st.altair_chart(donut)

    with col_legend:
        legend_df = agg[['Asset_Type', 'Count', 'Total_Value']].copy()
        legend_df['Total_Value'] = legend_df['Total_Value'].apply(lambda v: f"${v:,.0f}")
        legend_df.columns = ['Type', 'Deals', 'Value']
        st.dataframe(legend_df, hide_index=True, use_container_width=True)


# ============================================================
# RENDER — LOAN MATURITY SCHEDULE
# ============================================================
def _render_loan_maturity(mri_loans_raw, inv_disp):
    """Bar chart: loan maturities by year."""
    st.subheader("Loan Maturities")

    if mri_loans_raw is None or mri_loans_raw.empty:
        st.info("No loan data available.")
        return

    loans = mri_loans_raw.copy()
    loans.columns = [str(c).strip() for c in loans.columns]

    # Normalise vCode column
    if 'vCode' not in loans.columns and 'vcode' in loans.columns:
        loans = loans.rename(columns={'vcode': 'vCode'})
    if 'vCode' not in loans.columns:
        st.info("Loan data missing vCode column.")
        return

    loans['vCode'] = loans['vCode'].astype(str).str.strip()

    # Filter to known deal vcodes
    known_vcodes = set(inv_disp['vcode'].astype(str).str.strip())
    loans = loans[loans['vCode'].isin(known_vcodes)]

    # Parse maturity date
    mat_col = 'dtEvent' if 'dtEvent' in loans.columns else (
        'dtMaturity' if 'dtMaturity' in loans.columns else None)
    if mat_col is None:
        st.info("No maturity date column found in loan data.")
        return

    loans['maturity'] = pd.to_datetime(loans[mat_col], errors='coerce')
    loans = loans.dropna(subset=['maturity'])

    if 'mOrigLoanAmt' not in loans.columns:
        st.info("Loan data missing mOrigLoanAmt column.")
        return

    loans['mOrigLoanAmt'] = pd.to_numeric(loans['mOrigLoanAmt'], errors='coerce').fillna(0)
    loans['Year'] = loans['maturity'].dt.year

    yearly = loans.groupby('Year')['mOrigLoanAmt'].sum().reset_index()
    yearly.columns = ['Year', 'Amount']
    yearly['Year'] = yearly['Year'].astype(str)

    if yearly.empty:
        st.info("No loan maturity data to display.")
        return

    chart = alt.Chart(yearly).mark_bar(color=CLR_DARK).encode(
        x=alt.X('Year:N', title='Maturity Year'),
        y=alt.Y('Amount:Q', title='Loan Amount ($)'),
        tooltip=['Year', alt.Tooltip('Amount:Q', format='$,.0f')],
    ).properties(height=250)

    text = chart.mark_text(dy=-10, color=CLR_DARK, fontSize=11).encode(
        text=alt.Text('Amount:Q', format='$,.0f'),
    )

    st.altair_chart(chart + text, use_container_width=True)


# ============================================================
# RENDER — COMPUTED RETURNS  (button-gated, expensive)
# ============================================================
def _render_computed_returns(inv_disp, inv, wf, acct, fc, coa,
                             mri_loans_raw, mri_supp, mri_val,
                             relationships_raw, capital_calls_raw, isbs_raw,
                             start_year, horizon_years, pro_yr_base):
    """Button-gated computed returns: IRR by deal chart + summary table."""
    st.subheader("Portfolio Computed Returns")

    cached_returns = st.session_state.get('_dashboard_returns')

    if st.button("Compute Portfolio Returns", type="primary", key="dash_compute_btn"):
        rows = []
        errors = []
        eligible = inv_disp.copy()

        # Check which deals have waterfalls
        wf_norm = wf.copy()
        wf_norm.columns = [str(c).strip() for c in wf_norm.columns]
        if 'vCode' in wf_norm.columns and 'vcode' not in wf_norm.columns:
            wf_norm = wf_norm.rename(columns={'vCode': 'vcode'})
        wf_norm['vcode'] = wf_norm['vcode'].astype(str)
        wf_vcodes = set(wf_norm['vcode'].unique())
        eligible = eligible[eligible['vcode'].astype(str).isin(wf_vcodes)]

        if eligible.empty:
            st.warning("No deals with waterfall definitions found.")
            return

        progress = st.progress(0, text="Computing returns...")

        for i, (_, row) in enumerate(eligible.iterrows()):
            vcode = str(row['vcode'])
            label = row.get('DealLabel', row.get('Investment_Name', vcode))
            progress.progress((i + 1) / len(eligible), text=f"Processing {label}...")

            inv_id = str(row.get('InvestmentID', ''))
            sale_date_raw = row.get('Sale_Date', None)

            # Use deal cache if available
            _deal_key = f"{vcode}|{start_year}|{horizon_years}|{pro_yr_base}"
            if st.session_state.get('_deal_cache_key') == _deal_key:
                result = st.session_state['_deal_cache']
            else:
                try:
                    result = compute_deal_analysis(
                        deal_vcode=vcode,
                        deal_investment_id=inv_id,
                        sale_date_raw=sale_date_raw,
                        inv=inv, wf=wf, acct=acct, fc=fc, coa=coa,
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
                except Exception as e:
                    errors.append(f"{label}: {e}")
                    continue

            if 'error' in result:
                errors.append(f"{label}: {result['error']}")
                continue

            partner_rows = _build_partner_returns(result, label)
            # Extract deal-total row only
            for pr in partner_rows:
                if pr.get('_is_deal_total'):
                    rows.append(pr)

        progress.empty()

        if rows:
            st.session_state['_dashboard_returns'] = pd.DataFrame(rows)
            st.session_state['_dashboard_returns_errors'] = errors
            cached_returns = st.session_state['_dashboard_returns']
        else:
            st.warning("No return data computed.")
            if errors:
                with st.expander("Errors"):
                    for err in errors:
                        st.text(err)
            return

    # Display cached results
    if cached_returns is not None and not cached_returns.empty:
        errors = st.session_state.get('_dashboard_returns_errors', [])
        if errors:
            with st.expander(f"{len(errors)} deal(s) skipped"):
                for err in errors:
                    st.text(err)

        ret_df = cached_returns.copy()

        # IRR by Deal bar chart
        irr_df = ret_df[['Deal Name', 'IRR']].dropna(subset=['IRR']).copy()
        irr_df['IRR_pct'] = irr_df['IRR'] * 100
        irr_df = irr_df.sort_values('IRR_pct', ascending=True)

        if not irr_df.empty:
            irr_chart = alt.Chart(irr_df).mark_bar(color=CLR_DARK).encode(
                y=alt.Y('Deal Name:N',
                        sort=alt.EncodingSortField(field='IRR_pct', order='ascending'),
                        title=None),
                x=alt.X('IRR_pct:Q', title='IRR (%)'),
                tooltip=['Deal Name', alt.Tooltip('IRR_pct:Q', format='.2f', title='IRR %')],
            ).properties(height=max(250, len(irr_df) * 28))
            st.altair_chart(irr_chart, use_container_width=True)

        # Summary table
        display_cols = ['Deal Name', 'Contributions', 'CF Distributions',
                        'Capital Distributions', 'IRR', 'ROE', 'MOIC']
        table_df = ret_df[[c for c in display_cols if c in ret_df.columns]].copy()

        styled = table_df.style.format({
            'Contributions': '${:,.0f}',
            'CF Distributions': '${:,.0f}',
            'Capital Distributions': '${:,.0f}',
            'IRR': lambda v: f'{v:.2%}' if pd.notna(v) else 'N/A',
            'ROE': '{:.2%}',
            'MOIC': '{:.2f}x',
        })

        st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def render_dashboard(inv, wf, acct, isbs_raw, mri_loans_raw, mri_val,
                     occupancy_raw, fc, coa, mri_supp, relationships_raw,
                     capital_calls_raw, start_year, horizon_years, pro_yr_base):
    """Render the Executive Portfolio Dashboard tab."""

    st.header("Portfolio Dashboard")

    # --- Build deal lookup (reuse from reports_ui) ---
    inv_disp, eligible_deals, eligible_labels, wf_norm = _build_deal_lookup(inv, wf)

    # --- Lightweight data ---
    caps = _get_portfolio_caps(inv_disp, inv, wf, acct, mri_val, mri_loans_raw)
    occ_map = _get_latest_occupancy(inv_disp, occupancy_raw)

    # --- KPI Cards ---
    _render_kpi_cards(caps, occ_map, inv_disp)

    st.markdown("---")

    # --- Row 1: NOI chart (left) + Capital stack (right) ---
    col_noi, col_cap = st.columns([3, 2])
    with col_noi:
        _render_portfolio_noi_chart(isbs_raw, inv_disp, occupancy_raw)
    with col_cap:
        _render_capital_stack_chart(caps)

    st.markdown("---")

    # --- Row 2: Occupancy | Asset Allocation | Loan Maturities ---
    col_occ, col_asset, col_loan = st.columns(3)
    with col_occ:
        _render_occupancy_by_deal(occ_map, inv_disp)
    with col_asset:
        _render_asset_allocation(caps, inv_disp)
    with col_loan:
        _render_loan_maturity(mri_loans_raw, inv_disp)

    st.divider()

    # --- Computed returns (button-gated) ---
    _render_computed_returns(
        inv_disp, inv, wf, acct, fc, coa,
        mri_loans_raw, mri_supp, mri_val,
        relationships_raw, capital_calls_raw, isbs_raw,
        start_year, horizon_years, pro_yr_base,
    )
