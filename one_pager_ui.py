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

from one_pager import (
    get_available_quarters,
    get_general_information,
    get_capitalization_stack,
    get_property_performance,
    get_pe_performance,
    get_noi_chart_data,
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
    # SECTION 6: OCCUPANCY vs NOI CHART
    # ============================================================
    st.markdown("---")
    st.markdown("#### 6. OCCUPANCY vs NOI (Trailing 10 Quarters)")

    chart_data = get_noi_chart_data(vcode, selected_quarter, isbs_df, occupancy_df)

    if not chart_data.empty and chart_data['NOI_Actual'].notna().any():
        try:
            import altair as alt

            # Prepare data for dual-axis chart
            chart_df = chart_data.copy()

            # Create base chart
            base = alt.Chart(chart_df).encode(
                x=alt.X('Quarter:N', title='Quarter', sort=None)
            )

            # Occupancy bars
            bars = base.mark_bar(color='#4A90A4', opacity=0.7).encode(
                y=alt.Y('Occupancy:Q', title='Occupancy %', scale=alt.Scale(domain=[0, 100])),
                tooltip=['Quarter', alt.Tooltip('Occupancy:Q', format='.1f')]
            )

            # NOI lines
            noi_actual_line = base.mark_line(color='#2E7D32', strokeWidth=2).encode(
                y=alt.Y('NOI_Actual:Q', title='NOI ($)', axis=alt.Axis(format='$,.0f')),
                tooltip=['Quarter', alt.Tooltip('NOI_Actual:Q', format='$,.0f', title='NOI Actual')]
            )

            noi_uw_line = base.mark_line(color='#FF8F00', strokeWidth=2, strokeDash=[5, 5]).encode(
                y=alt.Y('NOI_UW:Q'),
                tooltip=['Quarter', alt.Tooltip('NOI_UW:Q', format='$,.0f', title='NOI U/W')]
            )

            # Combine with dual axis
            chart = alt.layer(
                bars,
                noi_actual_line,
                noi_uw_line
            ).resolve_scale(
                y='independent'
            ).properties(
                width='container',
                height=300
            )

            st.altair_chart(chart, use_container_width=True)

            # Legend
            st.markdown("""
            <div style="display: flex; gap: 20px; font-size: 12px; color: #666;">
                <span><span style="background: #4A90A4; padding: 2px 10px; margin-right: 5px;"></span> Occupancy %</span>
                <span><span style="background: #2E7D32; padding: 2px 10px; margin-right: 5px;"></span> NOI Actual</span>
                <span><span style="border-top: 2px dashed #FF8F00; padding: 2px 10px; margin-right: 5px;"></span> NOI U/W</span>
            </div>
            """, unsafe_allow_html=True)

        except ImportError:
            # Fallback to simple Streamlit charts if altair not available
            st.warning("Install altair for dual-axis charts. Using simplified view.")

            # Show data as table instead
            display_df = chart_data.copy()
            display_df['NOI_Actual'] = display_df['NOI_Actual'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
            display_df['NOI_UW'] = display_df['NOI_UW'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
            display_df['Occupancy'] = display_df['Occupancy'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient data available for chart. Requires trailing quarter NOI data.")

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
                prop_perf, pe_perf, comments
            )


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
    comments: Dict
):
    """Generate printable HTML report"""
    html = f"""
    <html>
    <head>
        <title>One Pager - {vcode} - {quarter}</title>
        <style>
            @media print {{
                body {{ margin: 0.5in; font-family: Arial, sans-serif; font-size: 10pt; }}
                .section {{ margin-bottom: 15px; }}
                .section-title {{ font-size: 12pt; font-weight: bold; border-bottom: 2px solid #333; margin-bottom: 8px; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 9pt; }}
                th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
                th {{ background: #f0f0f0; }}
                .metric-row {{ display: flex; justify-content: space-between; margin: 3px 0; }}
                .comments {{ background: #f9f9f9; padding: 8px; border: 1px solid #ddd; min-height: 50px; }}
            }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center; margin-bottom: 5px;">One Pager Investor Report</h1>
        <h2 style="text-align: center; color: #666; margin-top: 0;">{vcode} - Q{quarter}</h2>

        <div class="section">
            <div class="section-title">1. GENERAL INFORMATION</div>
            <div class="metric-row"><span>Partner:</span><span>{general_info['partner']}</span></div>
            <div class="metric-row"><span>Asset Type:</span><span>{general_info['asset_type']}</span></div>
            <div class="metric-row"><span>Location:</span><span>{general_info['location']}</span></div>
            <div class="metric-row"><span>Units | SF:</span><span>{general_info['units']:,} | {general_info['sqft']:,}</span></div>
        </div>

        <div class="section">
            <div class="section-title">2. CAPITALIZATION</div>
            <table>
                <tr><th>Layer</th><th>Amount</th><th>% of Cap</th></tr>
                <tr><td>Senior Debt</td><td>${cap_stack['debt']:,.0f}</td><td>{cap_stack['debt_pct']:.1%}</td></tr>
                <tr><td>Preferred Equity</td><td>${cap_stack['pref_equity']:,.0f}</td><td>{cap_stack['pref_equity_pct']:.1%}</td></tr>
                <tr><td>Partner Equity</td><td>${cap_stack['partner_equity']:,.0f}</td><td>{cap_stack['partner_equity_pct']:.1%}</td></tr>
                <tr style="font-weight: bold;"><td>Total</td><td>${cap_stack['total_cap']:,.0f}</td><td>100%</td></tr>
            </table>
        </div>

        <div class="section">
            <div class="section-title">3. PROPERTY PERFORMANCE</div>
            <table>
                <tr><th>Metric</th><th>YTD Actual</th><th>YTD Budget</th><th>Variance</th></tr>
                <tr><td>Revenue</td><td>${prop_perf['revenue']['ytd_actual']:,.0f}</td><td>${prop_perf['revenue']['ytd_budget']:,.0f}</td><td>${prop_perf['revenue']['variance']:,.0f}</td></tr>
                <tr><td>Expenses</td><td>${prop_perf['expenses']['ytd_actual']:,.0f}</td><td>${prop_perf['expenses']['ytd_budget']:,.0f}</td><td>${prop_perf['expenses']['variance']:,.0f}</td></tr>
                <tr style="font-weight: bold;"><td>NOI</td><td>${prop_perf['noi']['ytd_actual']:,.0f}</td><td>${prop_perf['noi']['ytd_budget']:,.0f}</td><td>${prop_perf['noi']['variance']:,.0f}</td></tr>
            </table>
            <p><strong>Comments:</strong></p>
            <div class="comments">{comments.get('econ_comments', '')}</div>
        </div>

        <div class="section">
            <div class="section-title">4. PREFERRED EQUITY PERFORMANCE</div>
            <div class="metric-row"><span>Committed P.E.:</span><span>${pe_perf['committed_pe']:,.0f}</span></div>
            <div class="metric-row"><span>Funded to Date:</span><span>${pe_perf['funded_to_date']:,.0f}</span></div>
            <div class="metric-row"><span>Return of Capital:</span><span>${pe_perf['return_of_capital']:,.0f}</span></div>
            <div class="metric-row"><span>Current P.E. Balance:</span><span>${pe_perf['current_pe_balance']:,.0f}</span></div>
        </div>

        <div class="section">
            <div class="section-title">5. BUSINESS PLAN & UPDATES</div>
            <div class="comments">{comments.get('business_plan_comments', '')}</div>
        </div>

        <script>window.print();</script>
    </body>
    </html>
    """

    # Display in a component that opens print dialog
    st.components.v1.html(html, height=0, scrolling=False)
    st.info("Print dialog should open. If not, right-click and select 'Print'.")
