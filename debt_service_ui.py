"""
debt_service_ui.py
Debt Service display components for the Deal Analysis tab.

Renders:
- Loan Summary table
- Detailed Amortization Schedules (per-loan tables with summary metrics)
- Sale Proceeds Calculation (exit valuation & net proceeds)
"""

import streamlit as st
import pandas as pd


def render_debt_service(loans, loan_sched, sale_dbg):
    """Entry point for the Debt Service section of Deal Analysis.

    Parameters
    ----------
    loans : list[Loan]
        Loan objects from the compute result.
    loan_sched : pd.DataFrame
        Amortization schedule with columns: LoanID, event_date, rate,
        interest, principal, payment, ending_balance.
    sale_dbg : dict | None
        Sale valuation debug info (Sale_Date, NOI_12m_After_Sale,
        CapRate_Sale, Implied_Value, etc.).
    """
    if loans:
        st.divider()
        _render_loan_summary(loans)
        _render_amortization_schedules(loans, loan_sched)

    if sale_dbg:
        _render_sale_proceeds(sale_dbg)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _render_loan_summary(loans):
    """Loan Summary table showing key terms for each loan."""
    loan_summary = []
    for ln in loans:
        orig_str = ln.orig_date.strftime('%Y-%m-%d') if pd.notna(ln.orig_date) else 'N/A'
        mat_str = ln.maturity_date.strftime('%Y-%m-%d') if pd.notna(ln.maturity_date) else 'N/A'
        loan_summary.append({
            'Loan ID': ln.loan_id,
            'Type': 'Existing' if ln.loan_id != 'PLANNED_2ND' else 'Planned 2nd Mortgage',
            'Original Amount': f"${ln.orig_amount:,.0f}",
            'Origination': orig_str,
            'Maturity': mat_str,
            'Rate Type': ln.int_type,
            'Rate': f"{ln.rate_for_month():.2%}",
            'Term (months)': ln.loan_term_m,
            'Amort (months)': ln.amort_term_m,
            'IO Period (months)': ln.io_months
        })

    st.markdown("### Loan Summary")
    st.dataframe(pd.DataFrame(loan_summary), use_container_width=True, hide_index=True)


def _render_amortization_schedules(loans, loan_sched):
    """Detailed per-loan amortization tables inside an expander."""
    with st.expander("\U0001f4cb Detailed Amortization Schedules", expanded=False):
        for ln in loans:
            st.markdown(f"#### {ln.loan_id}")

            if not loan_sched.empty:
                ln_sched = loan_sched[loan_sched['LoanID'] == ln.loan_id].copy()

                if not ln_sched.empty:
                    ln_sched = ln_sched.sort_values('event_date')
                    ln_sched['beginning_balance'] = ln_sched['ending_balance'] + ln_sched['principal']

                    display_cols = ['event_date', 'rate', 'beginning_balance', 'interest', 'principal', 'payment', 'ending_balance']

                    st.dataframe(
                        ln_sched[display_cols].style.format({
                            'rate': '{:.2%}',
                            'beginning_balance': '${:,.0f}',
                            'interest': '${:,.0f}',
                            'principal': '${:,.0f}',
                            'payment': '${:,.0f}',
                            'ending_balance': '${:,.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Interest", f"${ln_sched['interest'].sum():,.0f}")
                    col2.metric("Total Principal", f"${ln_sched['principal'].sum():,.0f}")
                    col3.metric("Final Balance", f"${ln_sched['ending_balance'].iloc[-1]:,.0f}")
                else:
                    st.info(f"No schedule generated for {ln.loan_id}")

            st.divider()


def _render_sale_proceeds(sale_dbg):
    """Sale Proceeds Calculation expander showing exit valuation."""
    with st.expander("\U0001f4b0 Sale Proceeds Calculation", expanded=False):
        st.markdown("### Exit Valuation & Net Proceeds")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Valuation Inputs:**")
            st.write(f"Sale Date: {sale_dbg['Sale_Date']}")
            st.write(f"NOI (12 months after sale): ${sale_dbg['NOI_12m_After_Sale']:,.0f}")
            st.write(f"Exit Cap Rate: {sale_dbg['CapRate_Sale']:.2%}")

        with col2:
            st.markdown("**Calculation:**")
            st.write(f"Implied Value (NOI \u00f7 Cap Rate): ${sale_dbg['Implied_Value']:,.0f}")
            st.write(f"Less: Selling Costs (2%): (${sale_dbg['Less_Selling_Cost_2pct']:,.0f})")
            st.write(f"**Value Net of Costs:** ${sale_dbg['Value_Net_Selling_Cost']:,.0f}")

        st.divider()

        st.markdown("### Net Proceeds to Equity")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Value Net of Costs: ${sale_dbg['Value_Net_Selling_Cost']:,.0f}")
            st.write(f"Less: Loan Payoff: (${sale_dbg['Less_Loan_Balances']:,.0f})")

        with col2:
            st.markdown(f"### **Net Sale Proceeds: ${sale_dbg['Net_Sale_Proceeds']:,.0f}**")
            st.caption("(Remaining cash reserves will be added to CF waterfall at sale)")
