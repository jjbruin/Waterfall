"""
app.py
Main Streamlit application for waterfall model

Multi-layer architecture:
1. Deal-level operations and capital events
2. Deal waterfalls (OP partners vs PPI entities)
3. Fund aggregation (optional - if fund_deals.csv provided)
4. Investor waterfalls (optional - if investor_waterfalls.csv provided)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date

# Local imports
from config import *
from utils import *
from metrics import *
from models import *
from loaders import *
from loans import *
from waterfall import *
from planned_loans import *
from portfolio import *
from reporting import *
from ownership_tree import *
from capital_calls import *
from cash_management import *

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="Waterfall Model")
st.title("Multi-Layer Waterfall Model")


# ============================================================
# SIDEBAR - DATA SOURCE
# ============================================================
with st.sidebar:
    st.header("Data Source")
    
    CLOUD = is_streamlit_cloud()
    
    if CLOUD:
        mode = "Upload CSVs"
        st.info("Running on Streamlit Cloud – local folders disabled. Upload CSVs.")
    else:
        mode = st.radio("Load data from:", ["Local folder", "Upload CSVs"], index=0)
    
    folder = None
    uploads = {}
    
    if mode == "Local folder":
        folder = st.text_input("Data folder path", placeholder=r"C:\Path\To\Data")
        st.caption("**Required:** investment_map.csv, waterfalls.csv, coa.csv, accounting_feed.csv, forecast_feed.csv")
        st.caption("**Optional:** MRI_Loans.csv, MRI_Supp.csv, MRI_Val.csv")
        st.caption("**Portfolio:** fund_deals.csv, investor_waterfalls.csv, investor_accounting.csv")
    else:
        st.subheader("Required Files")
        uploads["investment_map"] = st.file_uploader("investment_map.csv", type="csv", key="inv_map")
        uploads["waterfalls"] = st.file_uploader("waterfalls.csv", type="csv", key="wf")
        uploads["coa"] = st.file_uploader("coa.csv", type="csv", key="coa")
        uploads["accounting_feed"] = st.file_uploader("accounting_feed.csv", type="csv", key="acct")
        uploads["forecast_feed"] = st.file_uploader("forecast_feed.csv", type="csv", key="fc")
        
        st.divider()
        st.subheader("Optional - Loans & Valuation")
        uploads["MRI_Loans"] = st.file_uploader("MRI_Loans.csv", type="csv", key="loans")
        uploads["MRI_Supp"] = st.file_uploader("MRI_Supp.csv", type="csv", key="supp")
        uploads["MRI_Val"] = st.file_uploader("MRI_Val.csv", type="csv", key="val")
        
        st.divider()
        st.subheader("Optional - Portfolio")
        uploads["fund_deals"] = st.file_uploader("fund_deals.csv", type="csv", key="fund_deals")
        uploads["investor_waterfalls"] = st.file_uploader("investor_waterfalls.csv", type="csv", key="inv_wf")
        uploads["investor_accounting"] = st.file_uploader("investor_accounting.csv", type="csv", key="inv_acct")
    

    st.divider()
    st.header("Report Settings")
    start_year = st.number_input("Start year", min_value=2000, max_value=2100, value=DEFAULT_START_YEAR, step=1)
    horizon_years = st.number_input("Horizon (years)", min_value=1, max_value=30, value=DEFAULT_HORIZON_YEARS, step=1)
    pro_yr_base = st.number_input("Pro_Yr base year", min_value=1900, max_value=2100, value=PRO_YR_BASE_DEFAULT, step=1)


# ============================================================
# DATA LOADING
# ============================================================
def load_inputs():
    """Load all CSV files"""
    if CLOUD and mode == "Local folder":
        st.error("Local folder mode disabled on Streamlit Cloud.")
        st.stop()
    
    # Create a cache key based on mode and folder/uploads
    if mode == "Local folder":
        cache_key = f"local_{folder}"
    else:
        # For uploads, create key from uploaded file names
        cache_key = f"upload_{len([k for k, v in uploads.items() if v is not None])}"
    
    # Check if data is already loaded in session state
    if 'data_cache_key' in st.session_state and st.session_state.data_cache_key == cache_key:
        if 'cached_data' in st.session_state:
            return st.session_state.cached_data
    
    if mode == "Local folder":
        if not folder:
            st.error("Enter data folder path.")
            st.stop()
        
        inv = pd.read_csv(f"{folder}/investment_map.csv")
        wf = pd.read_csv(f"{folder}/waterfalls.csv")
        coa = load_coa(pd.read_csv(f"{folder}/coa.csv"))
        acct = pd.read_csv(f"{folder}/accounting_feed.csv")
        fc = load_forecast(pd.read_csv(f"{folder}/forecast_feed.csv"), coa, int(pro_yr_base))
        
        # Optional files
        mri_loans_raw = pd.read_csv(f"{folder}/MRI_Loans.csv") if Path(f"{folder}/MRI_Loans.csv").exists() else pd.DataFrame()
        mri_supp = pd.read_csv(f"{folder}/MRI_Supp.csv") if Path(f"{folder}/MRI_Supp.csv").exists() else pd.DataFrame()
        mri_val = pd.read_csv(f"{folder}/MRI_Val.csv") if Path(f"{folder}/MRI_Val.csv").exists() else pd.DataFrame()
        
        # Portfolio files
        fund_deals_raw = pd.read_csv(f"{folder}/fund_deals.csv") if Path(f"{folder}/fund_deals.csv").exists() else pd.DataFrame()
        inv_wf_raw = pd.read_csv(f"{folder}/investor_waterfalls.csv") if Path(f"{folder}/investor_waterfalls.csv").exists() else pd.DataFrame()
        inv_acct_raw = pd.read_csv(f"{folder}/investor_accounting.csv") if Path(f"{folder}/investor_accounting.csv").exists() else pd.DataFrame()
    
    else:
        for k in ["investment_map", "waterfalls", "coa", "accounting_feed", "forecast_feed"]:
            if uploads.get(k) is None:
                st.warning(f"Upload {k}.csv")
                st.stop()
        
        inv = pd.read_csv(uploads["investment_map"])
        wf = pd.read_csv(uploads["waterfalls"])
        coa = load_coa(pd.read_csv(uploads["coa"]))
        acct = pd.read_csv(uploads["accounting_feed"])
        fc = load_forecast(pd.read_csv(uploads["forecast_feed"]), coa, int(pro_yr_base))
        
        mri_loans_raw = pd.read_csv(uploads["MRI_Loans"]) if uploads.get("MRI_Loans") else pd.DataFrame()
        mri_supp = pd.read_csv(uploads["MRI_Supp"]) if uploads.get("MRI_Supp") else pd.DataFrame()
        mri_val = pd.read_csv(uploads["MRI_Val"]) if uploads.get("MRI_Val") else pd.DataFrame()
        
        fund_deals_raw = pd.read_csv(uploads["fund_deals"]) if uploads.get("fund_deals") else pd.DataFrame()
        inv_wf_raw = pd.read_csv(uploads["investor_waterfalls"]) if uploads.get("investor_waterfalls") else pd.DataFrame()
        inv_acct_raw = pd.read_csv(uploads["investor_accounting"]) if uploads.get("investor_accounting") else pd.DataFrame()
    
    # Capital calls file
    capital_calls_raw = None
    if mode == "Local folder":
        cc_path = Path(f"{folder}/MRI_Capital_Calls.csv")
        if cc_path.exists():
            capital_calls_raw = pd.read_csv(cc_path)
    elif uploads.get("capital_calls"):
        capital_calls_raw = pd.read_csv(uploads["capital_calls"])
    
    # ISBS (balance sheet) file for cash balances
    isbs_raw = None
    if mode == "Local folder":
        isbs_path = Path(f"{folder}/ISBS_Download.csv")
        if isbs_path.exists():
            try:
                # Standard CSV file
                isbs_raw = pd.read_csv(isbs_path, encoding='utf-8')
            except Exception as e:
                print(f"Warning: Could not load ISBS file: {e}")
                isbs_raw = None
    elif uploads.get("isbs"):
        try:
            isbs_raw = pd.read_csv(uploads["isbs"], encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not load uploaded ISBS file: {e}")
            isbs_raw = None
    
    # Relationship file
    relationships_raw = None
    if mode == "Local folder":
        rel_path = Path(f"{folder}/MRI_IA_Relationship.csv")
        if rel_path.exists():
            relationships_raw = pd.read_csv(rel_path)
    elif uploads.get("relationships"):
        relationships_raw = pd.read_csv(uploads["relationships"])

    # Normalize investment map
    inv.columns = [str(c).strip() for c in inv.columns]
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)
    
    # Prepare return data
    result = (inv, wf, acct, fc, mri_loans_raw, mri_supp, mri_val, fund_deals_raw, inv_wf_raw, inv_acct_raw, relationships_raw, capital_calls_raw, isbs_raw)
    
    # Cache in session state
    st.session_state.cached_data = result
    st.session_state.data_cache_key = cache_key
    
    return result


# Load data with progress indicator
with st.spinner("Loading data..."):
    inv, wf, acct, fc, mri_loans_raw, mri_supp, mri_val, fund_deals_raw, inv_wf_raw, inv_acct_raw, relationships_raw, capital_calls_raw, isbs_raw = load_inputs()


# ============================================================
# DEAL SELECTION
# ============================================================
inv_disp = inv.copy()
if "Investment_Name" not in inv_disp.columns:
    st.error("investment_map.csv must include Investment_Name column")
    st.stop()

inv_disp["Investment_Name"] = inv_disp["Investment_Name"].fillna("").astype(str)
inv_disp["vcode"] = inv_disp["vcode"].astype(str)

name_counts = inv_disp["Investment_Name"].value_counts()
inv_disp["DealLabel"] = inv_disp.apply(
    lambda r: f"{r['Investment_Name']} ({r['vcode']})" if name_counts.get(r['Investment_Name'], 0) > 1 else r['Investment_Name'],
    axis=1
)

labels_sorted = sorted(inv_disp["DealLabel"].dropna().unique().tolist(), key=lambda x: x.lower())
selected_label = st.selectbox("Select Deal", labels_sorted, key="deal_selector")

selected_row = inv_disp[inv_disp["DealLabel"] == selected_label].iloc[0]
deal_vcode = str(selected_row["vcode"])

# Show a progress indicator while generating report
with st.spinner(f"Generating report for {selected_label}..."):
    pass  # Report generation happens below


# ============================================================
# DEAL HEADER
# ============================================================
st.markdown("### Deal Summary")
st.markdown(
    f"""
    <div style="padding:14px 16px;border:1px solid #e6e6e6;border-radius:12px;background:#fafafa;">
      <div style="font-size:20px;font-weight:700;line-height:1.2;">{selected_row.get('Investment_Name','')}</div>
      <div style="margin-top:6px;color:#555;">
        <span style="margin-right:14px;"><b>vCode:</b> {deal_vcode}</span>
        <span style="margin-right:14px;"><b>InvestmentID:</b> {selected_row.get('InvestmentID','')}</span>
        <span style="margin-right:14px;"><b>Operating Partner:</b> {selected_row.get('Operating_Partner','')}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Asset Type", selected_row.get("Asset_Type", "—") or "—")
c2.metric("Total Units", fmt_int(selected_row.get("Total_Units", "")))
c3.metric("Size (Sqf)", fmt_num(selected_row.get("Size_Sqf", "")))
c4.metric("Lifecycle", selected_row.get("Lifecycle", "—") or "—")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Acquisition Date", fmt_date(selected_row.get("Acquisition_Date", "")))


# ============================================================
# FORECAST & LOAN MODELING
# ============================================================
fc_deal_full = fc[fc["vcode"].astype(str) == str(deal_vcode)].copy()
if fc_deal_full.empty:
    st.error(f"No forecast rows for vcode {deal_vcode}")
    st.stop()

model_start = min(fc_deal_full["event_date"])
model_end_full = max(fc_deal_full["event_date"])

# Determine sale date
def horizon_end_date(start_yr: int, horizon_yrs: int) -> date:
    y = int(start_yr) + int(horizon_yrs) - 1
    return date(y, 12, 31)

sale_date_raw = selected_row.get("Sale_Date", None)
if sale_date_raw is None or (isinstance(sale_date_raw, float) and pd.isna(sale_date_raw)) or str(sale_date_raw).strip() == "":
    sale_date = month_end(horizon_end_date(int(start_year), int(horizon_years)))
else:
    sale_date = month_end(as_date(sale_date_raw))

if sale_date < month_end(model_start):
    sale_date = month_end(model_start)

# Convert to month-end for use throughout
sale_me = month_end(sale_date)

# Build loan schedules
debug_msgs = []
loan_sched = pd.DataFrame()
loans = []

if mri_loans_raw is not None and not mri_loans_raw.empty:
    mri_loans = load_mri_loans(mri_loans_raw)
    mri_loans = mri_loans[mri_loans["vCode"].astype(str) == str(deal_vcode)].copy()
    loans.extend(build_loans_from_mri_loans(mri_loans))
else:
    debug_msgs.append("MRI_Loans.csv not provided; existing loans will NOT be modeled.")

# Planned loan sizing
planned_dbg = None
planned_new_loan_amt = 0.0
planned_orig_date = None

if mri_supp is not None and not mri_supp.empty:
    ms = mri_supp.copy()
    ms.columns = [str(c).strip() for c in ms.columns]
    if "vCode" not in ms.columns and "vcode" in ms.columns:
        ms = ms.rename(columns={"vcode": "vCode"})
    ms["vCode"] = ms["vCode"].astype(str)
    
    if (ms["vCode"] == str(deal_vcode)).any():
        if mri_val is None or mri_val.empty:
            debug_msgs.append("MRI_Supp present, but MRI_Val missing – cannot size planned loan.")
        else:
            supp_row = ms[ms["vCode"] == str(deal_vcode)].iloc[0]
            try:
                planned_new_loan_amt, planned_dbg = size_planned_second_mortgage(inv, fc_deal_full, supp_row, mri_val)
                planned_orig_date = month_end(as_date(supp_row["Orig_Date"]))
                
                if planned_new_loan_amt > 0:
                    loans.append(planned_loan_as_loan_object(deal_vcode, supp_row, planned_new_loan_amt))
            except Exception as e:
                debug_msgs.append(f"Planned loan sizing failed: {e}")

# Generate loan schedules
if loans:
    schedules = []
    for ln in loans:
        s = amortize_monthly_schedule(ln, model_start, model_end_full)
        if not s.empty:
            schedules.append(s)
    if schedules:
        loan_sched = pd.concat(schedules, ignore_index=True)
else:
    debug_msgs.append("No loans to model for this deal.")

# Replace forecast debt service with modeled
fc_deal_modeled = fc_deal_full.copy()
fc_deal_modeled = fc_deal_modeled[~fc_deal_modeled["vAccount"].isin(INTEREST_ACCTS | PRINCIPAL_ACCTS)].copy()

if not loan_sched.empty:
    monthly = loan_sched.groupby(["vcode", "event_date"], as_index=False)[["interest", "principal"]].sum()
    monthly["vAccount_interest"] = list(INTEREST_ACCTS)[0]
    monthly["vAccount_principal"] = list(PRINCIPAL_ACCTS)[0]
    
    add_rows = []
    for _, r in monthly.iterrows():
        add_rows.append({
            "vcode": r["vcode"],
            "event_date": r["event_date"],
            "vAccount": r["vAccount_interest"],
            "mAmount_norm": -abs(r["interest"]),  # Negative (cash outflow)
        })
        add_rows.append({
            "vcode": r["vcode"],
            "event_date": r["event_date"],
            "vAccount": r["vAccount_principal"],
            "mAmount_norm": -abs(r["principal"]),  # Negative (cash outflow)
        })
    if add_rows:
        fc_deal_modeled = pd.concat([fc_deal_modeled, pd.DataFrame(add_rows)], ignore_index=True)

fc_deal_modeled["Year"] = fc_deal_modeled["event_date"].apply(lambda d: pd.Timestamp(d).year).astype("Int64")

# Capital events
capital_events = []
sale_dbg = None

if mri_val is not None and not mri_val.empty:
    try:
        noi_12_sale = twelve_month_noi_after_date(fc_deal_modeled, sale_me)
        proj_begin = min(fc_deal_modeled["event_date"])
        cap_rate_sale = projected_cap_rate_at_date(mri_val, str(deal_vcode), proj_begin, sale_me)
        
        value_sale = (noi_12_sale / cap_rate_sale) if cap_rate_sale != 0 else 0.0
        value_net_selling_cost = value_sale * (1.0 - SELLING_COST_RATE)
        
        loan_bal_sale = total_loan_balance_at(loan_sched, sale_me)
        
        sale_proceeds = max(0.0, value_net_selling_cost - loan_bal_sale)
        
        sale_dbg = {
            "Sale_Date": str(sale_me),
            "NOI_12m_After_Sale": noi_12_sale,
            "CapRate_Sale": cap_rate_sale,
            "Implied_Value": value_sale,
            "Less_Selling_Cost_2pct": value_sale * SELLING_COST_RATE,
            "Value_Net_Selling_Cost": value_net_selling_cost,
            "Less_Loan_Balances": loan_bal_sale,
            "Net_Sale_Proceeds": sale_proceeds,
        }
        
        capital_events.append({
            "vcode": str(deal_vcode),
            "event_date": sale_me,
            "event_type": "Proceeds from Sale",
            "amount": float(sale_proceeds),
        })
    except Exception as e:
        debug_msgs.append(f"Sale proceeds estimation failed: {e}")
else:
    debug_msgs.append("MRI_Val missing – cannot estimate sale proceeds.")

cap_events_df = pd.DataFrame(capital_events)
if not cap_events_df.empty:
    cap_events_df["Year"] = cap_events_df["event_date"].apply(lambda d: pd.Timestamp(d).year).astype("Int64")

# Zero out display after sale
fc_deal_display = fc_deal_modeled.copy()
after_sale_mask = fc_deal_display["event_date"] > sale_me
fc_deal_display.loc[after_sale_mask, "mAmount_norm"] = 0.0

# ============================================================
# ANNUAL OPERATING FORECAST (with integrated waterfalls)
# ============================================================
st.divider()
st.subheader("Annual Operating Forecast")

# Note: cf_alloc and cap_alloc will be passed after waterfalls run
# For now, create the table structure - we'll update it after waterfalls
proceeds_by_year = None
if not cap_events_df.empty:
    proceeds_by_year = cap_events_df.groupby("Year")["amount"].sum()

# Annual table will be displayed after waterfalls run (see below)

st.caption(
    f"Sale Date: {sale_me.isoformat()} — Operating cash flows and debt service display as 0 after this date. "
    f"Full forecast retained internally for NOI-forward valuation."
)

if debug_msgs:
    with st.expander("Diagnostics"):
        for m in debug_msgs:
            st.write("- " + m)

# ============================================================
# DEBT AMORTIZATION SCHEDULES DISPLAY
# ============================================================
if loans:
    st.divider()
    
    # Summary of all loans
    loan_summary = []
    for ln in loans:
        loan_summary.append({
            'Loan ID': ln.loan_id,
            'Type': 'Existing' if ln.loan_id != 'PLANNED_2ND' else 'Planned 2nd Mortgage',
            'Original Amount': f"${ln.orig_amount:,.0f}",
            'Origination': ln.orig_date.strftime('%Y-%m-%d'),
            'Maturity': ln.maturity_date.strftime('%Y-%m-%d'),
            'Rate Type': ln.int_type,
            'Rate': f"{ln.rate_for_month():.2%}",
            'Term (months)': ln.loan_term_m,
            'Amort (months)': ln.amort_term_m,
            'IO Period (months)': ln.io_months
        })
    
    st.markdown("### Loan Summary")
    st.dataframe(pd.DataFrame(loan_summary), use_container_width=True, hide_index=True)
    
    # Detailed amortization schedules
    with st.expander("📋 Detailed Amortization Schedules", expanded=False):
        for ln in loans:
            st.markdown(f"#### {ln.loan_id}")
            
            # Get this loan's schedule
            if not loan_sched.empty:
                ln_sched = loan_sched[loan_sched['LoanID'] == ln.loan_id].copy()
                
                if not ln_sched.empty:
                    # Add beginning balance column
                    ln_sched = ln_sched.sort_values('event_date')
                    ln_sched['beginning_balance'] = ln_sched['ending_balance'].shift(1).fillna(ln.orig_amount)
                    
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
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Interest", f"${ln_sched['interest'].sum():,.0f}")
                    col2.metric("Total Principal", f"${ln_sched['principal'].sum():,.0f}")
                    col3.metric("Final Balance", f"${ln_sched['ending_balance'].iloc[-1]:,.0f}")
                else:
                    st.info(f"No schedule generated for {ln.loan_id}")
            
            st.divider()

# ============================================================
# SALE PROCEEDS CALCULATION
# ============================================================
if 'sale_dbg' in locals() and sale_dbg:
    with st.expander("💰 Sale Proceeds Calculation", expanded=False):
        st.markdown("### Exit Valuation & Net Proceeds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Valuation Inputs:**")
            st.write(f"Sale Date: {sale_dbg['Sale_Date']}")
            st.write(f"NOI (12 months after sale): ${sale_dbg['NOI_12m_After_Sale']:,.0f}")
            st.write(f"Exit Cap Rate: {sale_dbg['CapRate_Sale']:.2%}")
        
        with col2:
            st.markdown("**Calculation:**")
            st.write(f"Implied Value (NOI ÷ Cap Rate): ${sale_dbg['Implied_Value']:,.0f}")
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

# Load and build capital calls schedule (but don't apply yet)
capital_calls = []
if capital_calls_raw is not None and not capital_calls_raw.empty:
    try:
        cc_df = load_capital_calls(capital_calls_raw)
        if cc_df is not None and not cc_df.empty:
            capital_calls = build_capital_call_schedule(cc_df, deal_vcode)
            
            # Only integrate with forecast if we have capital calls for this deal
            if capital_calls:
                # Integrate with forecast to show as cash inflow
                fc_deal_modeled = integrate_capital_calls_with_forecast(
                    fc_deal_modeled, capital_calls
                )
    except Exception as e:
        st.warning(f"Note: Could not process capital calls data. Continuing without capital calls. ({str(e)})")
        capital_calls = []


# ============================================================
# CASH MANAGEMENT & RESERVES
# ============================================================
st.divider()
st.header("Cash Management & Reserves")

# Load beginning cash balance from ISBS
beginning_cash = 0.0
if isbs_raw is not None and not isbs_raw.empty:
    try:
        beginning_cash = load_beginning_cash_balance(isbs_raw, deal_vcode, model_start)
        st.success(f"✅ Beginning cash balance loaded: ${beginning_cash:,.2f}")
    except Exception as e:
        st.warning(f"Could not load cash balance from ISBS file: {str(e)}")
else:
    st.info("ℹ️ No ISBS file provided - starting with $0 cash balance")

# Build cash flow schedule
cash_summary = {}  # Initialize outside try block
try:
    # Get FAD using existing function (already works in the app)
    fad_monthly = cashflows_monthly_fad(fc_deal_modeled)
    
    # Build cash schedule with pre-calculated FAD
    cash_schedule = build_cash_flow_schedule_from_fad(
        fad_monthly=fad_monthly,
        capital_calls=capital_calls,
        beginning_cash=beginning_cash,
        deal_vcode=deal_vcode
    )
    
    # Show cash schedule summary
    with st.expander("📊 Cash Flow Schedule", expanded=False):
        st.dataframe(
            cash_schedule[[
                'event_date', 'beginning_cash', 'capital_call', 'capex_paid', 
                'operating_cf', 'deficit_covered', 'distributable', 'ending_cash'
            ]].style.format({
                'beginning_cash': '${:,.0f}',
                'capital_call': '${:,.0f}',
                'capex_paid': '${:,.0f}',
                'operating_cf': '${:,.0f}',
                'deficit_covered': '${:,.0f}',
                'distributable': '${:,.0f}',
                'ending_cash': '${:,.0f}'
            }),
            use_container_width=True
        )
    
    # Show summary statistics
    cash_summary = summarize_cash_usage(cash_schedule)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beginning Cash", f"${cash_summary.get('beginning_cash', 0):,.0f}")
    col2.metric("Total CapEx Paid", f"${cash_summary.get('total_capex_paid', 0):,.0f}")
    col3.metric("Total Distributable", f"${cash_summary.get('total_distributable', 0):,.0f}")
    col4.metric("Ending Cash", f"${cash_summary.get('ending_cash', 0):,.0f}")
    
    # Flag if cash went negative
    if cash_summary.get('min_cash_balance', 0) < 0:
        st.warning(f"⚠️ Cash balance went negative (lowest: ${cash_summary.get('min_cash_balance', 0):,.0f}). Consider additional capital calls or reducing CapEx.")
    
except Exception as e:
    st.error(f"Error building cash flow schedule: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    # Create empty schedule as fallback using FAD
    try:
        fad_monthly_fallback = cashflows_monthly_fad(fc_deal_modeled)
        cash_schedule = fad_monthly_fallback.copy()
        cash_schedule['distributable'] = cash_schedule['fad']
        cash_schedule['beginning_cash'] = beginning_cash
        cash_schedule['ending_cash'] = beginning_cash
        cash_summary = {}
    except:
        # Ultimate fallback - empty dataframe
        cash_schedule = pd.DataFrame()
        cash_summary = {}





# ============================================================
# WATERFALLS
# ============================================================
st.divider()
st.header("Waterfalls & Investor Returns")

# Use distributable cash from cash schedule (not raw FAD)
# This already accounts for CapEx, deficits, and cash reserves
cf_period_cash = cash_schedule[['event_date', 'distributable']].copy()
cf_period_cash = cf_period_cash.rename(columns={'distributable': 'cash_available'})

# Get remaining cash at sale date (not last period which might be different)
remaining_cash_at_sale, _ = get_sale_period_total_cash(cash_schedule, sale_me)

# Add remaining cash to the CF waterfall at sale date
if remaining_cash_at_sale > 0:
    sale_mask = cf_period_cash['event_date'] == sale_me
    if sale_mask.any():
        cf_period_cash.loc[sale_mask, 'cash_available'] += remaining_cash_at_sale
    else:
        # Add sale date row if it doesn't exist
        cf_period_cash = pd.concat([
            cf_period_cash,
            pd.DataFrame([{'event_date': sale_me, 'cash_available': remaining_cash_at_sale}])
        ], ignore_index=True).sort_values('event_date')

# Capital events cash (from sales, refinances, etc.)
cap_period_cash = pd.DataFrame(columns=["event_date", "cash_available"])
if cap_events_df is not None and not cap_events_df.empty:
    ce = cap_events_df.copy()
    ce["event_date"] = pd.to_datetime(ce["event_date"]).dt.date
    ce["event_date"] = ce["event_date"].apply(month_end)
    cap_period_cash = ce.groupby("event_date", as_index=False)["amount"].sum().rename(columns={"amount": "cash_available"})

cf_period_cash = cf_period_cash[cf_period_cash["event_date"] <= sale_me].copy()
cap_period_cash = cap_period_cash[cap_period_cash["event_date"] <= sale_me].copy()

if capital_calls:
    st.subheader("Capital Calls Schedule")
    
    summary = capital_calls_summary_table(capital_calls)
    st.dataframe(
        summary,
        use_container_width=True
    )
    
    by_investor = capital_calls_by_investor(capital_calls)
    st.markdown("**By Investor:**")
    st.dataframe(
        by_investor,
        use_container_width=True
       )


# Load waterfalls
wf_steps = load_waterfalls(wf)

# Seed from accounting
seed_states = seed_states_from_accounting(acct, inv, wf_steps, deal_vcode)

# Apply capital calls to investor states if we have any
if capital_calls:
    try:
        seed_states = apply_capital_calls_to_states(capital_calls, seed_states)
    except Exception as e:
        st.warning(f"Note: Could not apply capital calls to investor states. ({str(e)})")

# Run Waterfalls
cf_alloc, cf_investors = run_waterfall(wf_steps, deal_vcode, "CF_WF", cf_period_cash, seed_states)
cap_alloc, cap_investors = run_waterfall(wf_steps, deal_vcode, "Cap_WF", cap_period_cash, seed_states)


# ============================================================
# ANNUAL OPERATING FORECAST WITH INTEGRATED WATERFALLS
# ============================================================
st.divider()
st.header("📊 Annual Operating Forecast & Waterfall Summary")

# Build the integrated annual table with waterfall details
annual_df_raw = annual_aggregation_table(
    fc_deal_display,
    int(start_year),
    int(horizon_years),
    proceeds_by_year=proceeds_by_year,
    cf_alloc=cf_alloc,
    cap_alloc=cap_alloc
)
annual_df = pivot_annual_table(annual_df_raw)
styled = style_annual_table(annual_df)
st.dataframe(styled, use_container_width=True)


# ============================================================
# PARTNER RETURNS SUMMARY
# ============================================================
st.divider()
st.header("🎯 Partner Returns")

# ============================================================
# PARTNER TOTALS & RETURNS
# ============================================================

# Combine CF and Cap allocations by partner
partner_totals = []

all_partners = set()
if not cf_alloc.empty:
    all_partners.update(cf_alloc['PropCode'].unique())
if not cap_alloc.empty:
    all_partners.update(cap_alloc['PropCode'].unique())

for partner in sorted(all_partners):
    # Cash flow distributions
    cf_dist = 0
    if not cf_alloc.empty:
        cf_dist = cf_alloc[cf_alloc['PropCode'] == partner]['Allocated'].sum()
    
    # Capital distributions
    cap_dist = 0
    if not cap_alloc.empty:
        cap_dist = cap_alloc[cap_alloc['PropCode'] == partner]['Allocated'].sum()
    
    # Get investor state
    state = seed_states.get(partner)
    
    if state:
        # Calculate metrics
        unrealized = state.capital_outstanding + state.pref_unpaid_compounded + state.pref_accrued_current_year
        
        from metrics import investor_metrics
        metrics = investor_metrics(state, sale_me, unrealized_nav=unrealized)
        
        partner_totals.append({
            'Partner': partner,
            'Cash Flow Distributions': cf_dist,
            'Capital Distributions': cap_dist,
            'Total Distributions': cf_dist + cap_dist,
            'Capital Outstanding': state.capital_outstanding,
            'Unpaid Pref (Compounded)': state.pref_unpaid_compounded,
            'Accrued Pref (Current Yr)': state.pref_accrued_current_year,
            'Total Capital + Pref': state.capital_outstanding + state.pref_unpaid_compounded + state.pref_accrued_current_year,
            'Total Contributions': metrics.get('TotalContributions', 0),
            'IRR': metrics.get('IRR'),
            'ROE': metrics.get('ROE', 0),
            'MOIC': metrics.get('MOIC', 0)
        })

if partner_totals:
    totals_df = pd.DataFrame(partner_totals)
    
    # Display summary table
    st.dataframe(
        totals_df.style.format({
            'Cash Flow Distributions': '${:,.0f}',
            'Capital Distributions': '${:,.0f}',
            'Total Distributions': '${:,.0f}',
            'Capital Outstanding': '${:,.0f}',
            'Unpaid Pref (Compounded)': '${:,.0f}',
            'Accrued Pref (Current Yr)': '${:,.0f}',
            'Total Capital + Pref': '${:,.0f}',
            'Total Contributions': '${:,.0f}',
            'IRR': '{:.2%}',
            'ROE': '{:.2%}',
            'MOIC': '{:.2f}x'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Summary metrics
    st.markdown("### Deal-Level Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_cf_dist = totals_df['Cash Flow Distributions'].sum()
    total_cap_dist = totals_df['Capital Distributions'].sum()
    total_dist = totals_df['Total Distributions'].sum()
    total_contrib = totals_df['Total Contributions'].sum()
    
    col1.metric("Total CF Distributions", f"${total_cf_dist:,.0f}")
    col2.metric("Total Capital Distributions", f"${total_cap_dist:,.0f}")
    col3.metric("Total All Distributions", f"${total_dist:,.0f}")
    col4.metric("Deal MOIC", f"{(total_dist / total_contrib if total_contrib > 0 else 0):.2f}x")
else:
    st.info("No partner data available")


if cap_alloc.empty:
    st.info("No Cap_WF steps found for this deal.")
else:
    with st.expander("Cap_WF (by year)"):
        show_waterfall_matrix("Cap_WF Allocations by Year", cap_alloc)
    
    st.markdown("**Cap_WF Investor Summary**")
    if not cap_investors.empty:
        out = cap_investors.copy()
        for c in ["CapitalOutstanding", "UnpaidPrefCompounded", "AccruedPrefCurrentYear",
                  "TotalDistributions", "TotalContributions"]:
            if c in out.columns:
                out[c] = out[c].map(lambda x: f"{x:,.0f}")
        if "IRR" in out.columns:
            out["IRR"] = out["IRR"].map(lambda r: "" if r is None else f"{r*100:,.2f}%")
        if "ROE" in out.columns:
            out["ROE"] = out["ROE"].map(lambda r: f"{r*100:,.2f}%")
        if "MOIC" in out.columns:
            out["MOIC"] = out["MOIC"].map(lambda r: f"{r:,.2f}x")
        st.dataframe(out, use_container_width=True)

# ============================================================
# OWNERSHIP TREE ANALYSIS
# ============================================================
st.divider()
st.header("Ownership Relationships & Waterfall Requirements")

if relationships_raw is not None:
    # Load and process relationships
    relationships = load_relationships(relationships_raw)
    
    st.success(f"✅ Loaded {len(relationships)} ownership relationships")
    
    # Build ownership tree
    with st.spinner("Building ownership tree..."):
        nodes = build_ownership_tree(relationships)
    
    st.info(f"📊 Identified {len(nodes)} unique entities in ownership structure")
    
    # Identify entities that are deals (have vcode mappings)
    deal_entities = set(inv["vcode"].unique())
    
    # Identify waterfall requirements
    waterfall_reqs = identify_waterfall_requirements(nodes, deal_entities)
    
    st.subheader(f"Waterfall Requirements: {len(waterfall_reqs)} entities need waterfalls")
    
    # Create summary table
    if waterfall_reqs:
        summary_df = create_waterfall_summary_df(waterfall_reqs)
        
        st.dataframe(
            summary_df,
            use_container_width=True,
            column_config={
                "EntityID": st.column_config.TextColumn("Entity ID", width="medium"),
                "EntityName": st.column_config.TextColumn("Entity Name", width="large"),
                "NumInvestors": st.column_config.NumberColumn("# Investors", width="small"),
                "IsDeal": st.column_config.TextColumn("Is Deal?", width="small"),
                "DirectInvestors": st.column_config.TextColumn("Direct Investors", width="large"),
                "UltimateInvestors": st.column_config.TextColumn("Ultimate Investors", width="large"),
            }
        )
        
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Waterfall Requirements CSV",
            data=csv,
            file_name="waterfall_requirements.csv",
            mime="text/csv"
        )
    
    # Entity selector
    st.subheader("Detailed Entity Analysis")
    
    entity_options = sorted(nodes.keys())
    selected_entity = st.selectbox(
        "Select entity to analyze:",
        entity_options,
        format_func=lambda x: f"{x} - {nodes[x].name}" if nodes[x].name else x
    )
    
    if selected_entity:
        node = nodes[selected_entity]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entity ID", node.entity_id)
        col2.metric("# Direct Investors", len(node.investors))
        col3.metric("# Investments", len(node.investments))
        col4.metric("Status", 
                   "🔄 Passthrough" if node.is_passthrough 
                   else "💧 Needs Waterfall" if node.needs_waterfall 
                   else "👤 Investor")
        
        if node.investors:
            st.markdown("**Direct Investors:**")
            inv_data = []
            for inv_id, pct in node.investors:
                inv_name = nodes[inv_id].name if inv_id in nodes else ""
                inv_data.append({
                    "Investor ID": inv_id,
                    "Name": inv_name,
                    "Ownership %": f"{pct*100:.4f}%"
                })
            st.dataframe(pd.DataFrame(inv_data), use_container_width=True, hide_index=True)
        
        ultimate = get_ultimate_investors(selected_entity, nodes)
        ultimate = consolidate_ultimate_investors(ultimate)
        ultimate = sorted(ultimate, key=lambda x: x[1], reverse=True)
        
        if ultimate:
            st.markdown("**Ultimate Beneficial Owners:**")
            ult_data = []
            for inv_id, eff_pct in ultimate:
                inv_name = nodes[inv_id].name if inv_id in nodes else ""
                ult_data.append({
                    "Investor ID": inv_id,
                    "Name": inv_name,
                    "Effective Ownership %": f"{eff_pct*100:.4f}%"
                })
            st.dataframe(pd.DataFrame(ult_data), use_container_width=True, hide_index=True)
        
        with st.expander("📊 View Ownership Tree"):
            tree_viz = visualize_ownership_tree(selected_entity, nodes, max_depth=5)
            st.code(tree_viz, language=None)

else:
    st.info("Upload MRI_IA_Relationship.csv to analyze multi-tiered ownership structures")


# ============================================================
# OPTIONAL DISPLAYS
# ============================================================
if loan_sched is not None and not loan_sched.empty:
    with st.expander("Loan Schedule (modeled)"):
        show = loan_sched.sort_values(["LoanID", "event_date"]).copy()
        show_disp = show.copy()
        show_disp["rate"] = (show_disp["rate"] * 100.0).map(lambda x: f"{x:,.2f}%")
        for c in ["interest", "principal", "payment", "ending_balance"]:
            show_disp[c] = show_disp[c].map(lambda x: f"{x:,.0f}")
        st.dataframe(
            show_disp[["LoanID", "event_date", "rate", "interest", "principal", "payment", "ending_balance"]],
            use_container_width=True
        )

if not cap_events_df.empty:
    with st.expander("Capital Events"):
        disp = cap_events_df.sort_values("event_date").copy()
        disp2 = disp.copy()
        disp2["amount"] = disp2["amount"].map(lambda x: f"{x:,.0f}")
        st.dataframe(disp2[["event_date", "event_type", "amount"]], use_container_width=True)

st.success("✅ Report generated successfully!")
