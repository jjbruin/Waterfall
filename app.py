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
import sqlite3
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
from consolidation import build_consolidated_forecast, get_sub_portfolio_summary, get_property_vcodes_for_deal, get_parent_deal_for_property
from compute import compute_deal_analysis, get_deal_capitalization
from property_financials_ui import render_property_financials
from reports_ui import render_reports

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="PSC Asset Management")
st.title("Peaceable Street Capital - Asset Management System")


# ============================================================
# CACHED DATA LOADER (defined before sidebar for .clear() access)
# ============================================================
@st.cache_data(show_spinner="Loading data from database...")
def _load_sqlite_data(db_path_str, pro_yr_base_int):
    """Load all data from SQLite database - cached by Streamlit.

    Replaces manual session_state caching for the primary data source.
    Cache invalidation: automatic when args change, or manual via .clear().
    """
    conn = sqlite3.connect(db_path_str)

    # Required tables
    inv = pd.read_sql('SELECT * FROM deals', conn)
    wf = pd.read_sql('SELECT * FROM waterfalls', conn)
    coa_raw = pd.read_sql('SELECT * FROM coa', conn)
    coa = load_coa(coa_raw)
    acct = pd.read_sql('SELECT * FROM accounting', conn)
    fc_raw = pd.read_sql('SELECT * FROM forecasts', conn)
    fc = load_forecast(fc_raw, coa, pro_yr_base_int)

    # Optional tables
    try:
        mri_loans_raw = pd.read_sql('SELECT * FROM loans', conn)
    except Exception:
        mri_loans_raw = pd.DataFrame()

    try:
        mri_val = pd.read_sql('SELECT * FROM valuations', conn)
    except Exception:
        mri_val = pd.DataFrame()

    try:
        relationships_raw = pd.read_sql('SELECT * FROM relationships', conn)
    except Exception:
        relationships_raw = None

    try:
        capital_calls_raw = pd.read_sql('SELECT * FROM capital_calls', conn)
    except Exception:
        capital_calls_raw = None

    try:
        isbs_raw = pd.read_sql('SELECT * FROM isbs', conn)
    except Exception:
        isbs_raw = None

    try:
        occupancy_raw = pd.read_sql('SELECT * FROM occupancy', conn)
    except Exception:
        occupancy_raw = None

    try:
        commitments_raw = pd.read_sql('SELECT * FROM commitments', conn)
    except Exception:
        commitments_raw = None

    try:
        tenants_raw = pd.read_sql('SELECT * FROM tenants', conn)
    except Exception:
        tenants_raw = None

    # Not typically in DB
    mri_supp = pd.DataFrame()
    fund_deals_raw = pd.DataFrame()
    inv_wf_raw = pd.DataFrame()
    inv_acct_raw = pd.DataFrame()

    conn.close()

    # Normalize investment map
    inv.columns = [str(c).strip() for c in inv.columns]
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)

    return (inv, wf, acct, fc, coa, mri_loans_raw, mri_supp, mri_val,
            fund_deals_raw, inv_wf_raw, inv_acct_raw, relationships_raw,
            capital_calls_raw, isbs_raw, occupancy_raw, commitments_raw, tenants_raw)


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
        mode = st.radio("Load data from:", ["SQLite Database", "Local folder", "Upload CSVs"], index=0)
    
    folder = None
    uploads = {}
    db_path = None

    if mode == "SQLite Database":
        db_path = st.text_input("Database path", value="waterfall.db", placeholder="waterfall.db")
        st.caption("Loads all data from SQLite database including relationships for sub-portfolio consolidation.")
        if st.button("🔄 Reload Data"):
            _load_sqlite_data.clear()
            # Clear deal computation cache too
            for k in list(st.session_state.keys()):
                if k.startswith('_deal_'):
                    del st.session_state[k]
            st.rerun()
    elif mode == "Local folder":
        folder = st.text_input("Data folder path", placeholder=r"C:\Path\To\Data")
        st.caption("**Required:** investment_map.csv, waterfalls.csv, coa.csv, accounting_feed.csv, forecast_feed.csv")
        st.caption("**Optional:** MRI_Loans.csv, MRI_Supp.csv, MRI_Val.csv, MRI_IA_Relationship.csv")
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
    """Load data from SQLite, CSV folder, or uploaded files"""
    if CLOUD and mode == "Local folder":
        st.error("Local folder mode disabled on Streamlit Cloud.")
        st.stop()

    # SQLite mode uses @st.cache_data for optimal caching
    if mode == "SQLite Database":
        if not db_path or not Path(db_path).exists():
            st.error(f"Database not found: {db_path}")
            st.stop()
        return _load_sqlite_data(db_path, int(pro_yr_base))

    # Folder and Upload modes use manual session_state caching
    if mode == "Local folder":
        cache_key = f"local_{folder}"
    else:
        cache_key = f"upload_{len([k for k, v in uploads.items() if v is not None])}"

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

        mri_loans_raw = pd.read_csv(f"{folder}/MRI_Loans.csv") if Path(f"{folder}/MRI_Loans.csv").exists() else pd.DataFrame()
        mri_supp = pd.read_csv(f"{folder}/MRI_Supp.csv") if Path(f"{folder}/MRI_Supp.csv").exists() else pd.DataFrame()
        mri_val = pd.read_csv(f"{folder}/MRI_Val.csv") if Path(f"{folder}/MRI_Val.csv").exists() else pd.DataFrame()

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

    # Optional tables for Local folder and Upload modes
    capital_calls_raw = None
    isbs_raw = None
    relationships_raw = None
    occupancy_raw = None
    commitments_raw = None
    tenants_raw = None

    if mode == "Local folder":
        def _try_csv(fname):
            fpath = Path(f"{folder}/{fname}")
            if fpath.exists():
                try:
                    return pd.read_csv(fpath)
                except Exception as e:
                    print(f"Warning: Could not load {fname}: {e}")
            return None

        capital_calls_raw = _try_csv("MRI_Capital_Calls.csv")
        isbs_raw = _try_csv("ISBS_Download.csv")
        relationships_raw = _try_csv("MRI_IA_Relationship.csv")
        occupancy_raw = _try_csv("MRI_Occupancy_Download.csv")
        commitments_raw = _try_csv("MRI_Commitments.csv")
        tenants_raw = _try_csv("Tenant_Report.csv")

    # Normalize investment map
    inv.columns = [str(c).strip() for c in inv.columns]
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)

    result = (inv, wf, acct, fc, coa, mri_loans_raw, mri_supp, mri_val,
              fund_deals_raw, inv_wf_raw, inv_acct_raw, relationships_raw,
              capital_calls_raw, isbs_raw, occupancy_raw, commitments_raw, tenants_raw)

    st.session_state.cached_data = result
    st.session_state.data_cache_key = cache_key
    return result


# Load data
inv, wf, acct, fc, coa, mri_loans_raw, mri_supp, mri_val, fund_deals_raw, inv_wf_raw, inv_acct_raw, relationships_raw, capital_calls_raw, isbs_raw, occupancy_raw, commitments_raw, tenants_raw = load_inputs()


# Create tabs for different sections - tabs at top level
tab_deal, tab_financials, tab_ownership, tab_reports = st.tabs(["Deal Analysis", "Property Financials", "Ownership & Partnerships", "Reports"])

with tab_deal:
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
    # ============================================================
    # DEAL COMPUTATION (cached to avoid recomputation on reruns)
    # ============================================================
    deal_investment_id = str(selected_row.get('InvestmentID', ''))
    sale_date_raw = selected_row.get("Sale_Date", None)

    _deal_key = f"{deal_vcode}|{start_year}|{horizon_years}|{pro_yr_base}"
    if st.session_state.get('_deal_cache_key') != _deal_key:
        st.session_state['_deal_cache'] = compute_deal_analysis(
            deal_vcode=deal_vcode,
            deal_investment_id=deal_investment_id,
            sale_date_raw=sale_date_raw,
            inv=inv, wf=wf, acct=acct, fc=fc, coa=coa,
            mri_loans_raw=mri_loans_raw,
            mri_supp=mri_supp,
            mri_val=mri_val,
            relationships_raw=relationships_raw,
            capital_calls_raw=capital_calls_raw,
            isbs_raw=isbs_raw,
            start_year=int(start_year),
            horizon_years=int(horizon_years),
            pro_yr_base=int(pro_yr_base),
        )
        st.session_state['_deal_cache_key'] = _deal_key

    _dc = st.session_state['_deal_cache']
    cap_data = _dc['cap_data']

    # ============================================================
    # DEAL HEADER - Compact Header + Data Table
    # ============================================================

    # Format the as_of date
    as_of_str = ""
    if cap_data['as_of_date']:
        as_of_str = pd.Timestamp(cap_data['as_of_date']).strftime('%m/%Y')

    # Build the Option B header with capitalization table
    st.markdown(
        f"""
        <style>
            .deal-header {{
                background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
                color: white;
                padding: 12px 16px;
                border-radius: 8px 8px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .deal-title {{
                font-size: 24px;
                font-weight: 700;
            }}
            .asset-badge {{
                background: rgba(255,255,255,0.2);
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 15px;
            }}
            .deal-table {{
                width: 100%;
                border-collapse: collapse;
                border: 1px solid #e0e0e0;
                border-top: none;
                border-radius: 0 0 8px 8px;
                overflow: hidden;
            }}
            .deal-table td {{
                padding: 4px 8px;
                font-size: 14px;
                border-bottom: 1px solid #f0f0f0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                height: 28px;
                vertical-align: middle;
            }}
            .deal-table tr:nth-child(even) {{
                background: #f9f9f9;
            }}
            .deal-table tr:nth-child(odd) {{
                background: #ffffff;
            }}
            .deal-table .label {{
                font-weight: 600;
                color: #444;
                width: 18%;
            }}
            .deal-table .value {{
                color: #222;
                width: 22%;
            }}
            .deal-table .spacer {{
                width: 2%;
                border-right: 2px solid #e0e0e0;
            }}
            .deal-table .cap-label {{
                font-weight: 600;
                color: #444;
                width: 28%;
            }}
            .deal-table .cap-value {{
                color: #222;
                text-align: right;
                width: 30%;
            }}
            .cap-header {{
                font-weight: 700;
                color: #1e3a5f;
                font-size: 15px;
                border-bottom: 2px solid #1e3a5f !important;
            }}

            /* Smaller metrics for Deal-Level Summary and Cash Management */
            [data-testid="stMetricValue"] {{
                font-size: 1.1rem !important;
            }}
            [data-testid="stMetricLabel"] {{
                font-size: 0.85rem !important;
            }}
            [data-testid="stMetricDelta"] {{
                font-size: 0.75rem !important;
            }}
        </style>
    
        <div class="deal-header">
            <span class="deal-title">{selected_row.get('Investment_Name','')}</span>
            <span class="asset-badge">{selected_row.get('Asset_Type', '—') or '—'}</span>
        </div>
        <table class="deal-table">
            <tr>
                <td class="label"></td>
                <td class="value"></td>
                <td class="spacer"></td>
                <td class="cap-header" colspan="2">Capitalization {f'(as of {as_of_str})' if as_of_str else ''}</td>
            </tr>
            <tr>
                <td class="label">vCode</td>
                <td class="value">{deal_vcode}</td>
                <td class="spacer"></td>
                <td class="cap-label">Debt</td>
                <td class="cap-value">${cap_data['debt']:,.0f}</td>
            </tr>
            <tr>
                <td class="label">Investment ID</td>
                <td class="value">{selected_row.get('InvestmentID','')}</td>
                <td class="spacer"></td>
                <td class="cap-label">Pref Equity</td>
                <td class="cap-value">${cap_data['pref_equity']:,.0f}</td>
            </tr>
            <tr>
                <td class="label">OP</td>
                <td class="value">{selected_row.get('Operating_Partner','')}</td>
                <td class="spacer"></td>
                <td class="cap-label">Ptr Equity</td>
                <td class="cap-value">${cap_data['partner_equity']:,.0f}</td>
            </tr>
            <tr>
                <td class="label">Units</td>
                <td class="value">{fmt_int(selected_row.get('Total_Units', ''))}</td>
                <td class="spacer"></td>
                <td class="cap-label" style="border-top: 1px solid #ccc;">Total Cap</td>
                <td class="cap-value" style="border-top: 1px solid #ccc; font-weight: 600;">${cap_data['total_cap']:,.0f}</td>
            </tr>
            <tr>
                <td class="label">Sqf</td>
                <td class="value">{fmt_num(selected_row.get('Size_Sqf', ''))}</td>
                <td class="spacer"></td>
                <td class="cap-label">Valuation</td>
                <td class="cap-value">${cap_data['current_valuation']:,.0f}</td>
            </tr>
            <tr>
                <td class="label">Lifecycle</td>
                <td class="value">{selected_row.get('Lifecycle', '—') or '—'}</td>
                <td class="spacer"></td>
                <td class="cap-label">Cap Rate</td>
                <td class="cap-value">{cap_data['cap_rate']:.2%}</td>
            </tr>
            <tr>
                <td class="label">Acq Date</td>
                <td class="value">{fmt_date(selected_row.get('Acquisition_Date', ''))}</td>
                <td class="spacer"></td>
                <td class="cap-label">PE Exp (Cap)</td>
                <td class="cap-value">{cap_data['pe_exposure_cap']:.1%}</td>
            </tr>
            <tr>
                <td class="label"></td>
                <td class="value"></td>
                <td class="spacer"></td>
                <td class="cap-label">PE Exp (Val)</td>
                <td class="cap-value">{cap_data['pe_exposure_value']:.1%}</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.write("")  # Add some spacing

    # ============================================================
    # UNPACK COMPUTATION RESULTS (from cached compute_deal_analysis)
    # ============================================================
    if 'error' in _dc:
        st.error(_dc['error'])
        st.stop()

    if _dc.get('sub_portfolio_msg'):
        st.info(_dc['sub_portfolio_msg'])

    # Unpack all computed results
    consolidation_info = _dc['consolidation_info']
    fc_deal_modeled = _dc['fc_deal_modeled']
    fc_deal_display = _dc['fc_deal_display']
    loan_sched = _dc['loan_sched']
    loans = _dc['loans']
    cf_alloc = _dc['cf_alloc']
    cf_investors = _dc['cf_investors']
    cap_alloc = _dc['cap_alloc']
    cap_investors = _dc['cap_investors']
    cap_events_df = _dc['cap_events_df']
    cash_schedule = _dc['cash_schedule']
    cash_summary = _dc['cash_summary']
    sale_dbg = _dc['sale_dbg']
    sale_me = _dc['sale_me']
    debug_msgs = _dc['debug_msgs']
    capital_calls = _dc['capital_calls']
    seed_states = _dc['seed_states']
    wf_steps = _dc['wf_steps']
    model_start = _dc['model_start']
    model_end_full = _dc['model_end_full']
    beginning_cash = _dc['beginning_cash']

    # ============================================================
    # 📊 ANNUAL OPERATING FORECAST & WATERFALL SUMMARY
    # ============================================================
    st.divider()
    st.header("📊 Annual Operating Forecast & Waterfall Summary")

    proceeds_by_year = None
    if not cap_events_df.empty:
        proceeds_by_year = cap_events_df.groupby("Year")["amount"].sum()

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
    
        # Detailed amortization schedules
        with st.expander("📋 Detailed Amortization Schedules", expanded=False):
            for ln in loans:
                st.markdown(f"#### {ln.loan_id}")
            
                # Get this loan's schedule
                if not loan_sched.empty:
                    ln_sched = loan_sched[loan_sched['LoanID'] == ln.loan_id].copy()
                
                    if not ln_sched.empty:
                        # Add beginning balance column (ending_balance + principal = beginning_balance)
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

    # ============================================================
    # CASH MANAGEMENT & RESERVES (Display)
    # ============================================================
    st.divider()
    st.header("Cash Management & Reserves")

    if beginning_cash > 0:
        st.success(f"✅ Beginning cash balance: ${beginning_cash:,.2f}")
    else:
        st.info("ℹ️ No ISBS file provided - starting with $0 cash balance")

    if not cash_schedule.empty:
        with st.expander("📊 Cash Flow Schedule", expanded=False):
            display_cols = [
                'event_date', 'beginning_cash', 'capital_call', 'capex_paid',
                'capex_unpaid', 'operating_cf', 'deficit_covered',
                'unpaid_shortfall', 'distributable', 'ending_cash'
            ]
            display_cols = [c for c in display_cols if c in cash_schedule.columns]
            fmt = {c: '${:,.0f}' for c in display_cols if c != 'event_date'}
            st.dataframe(
                cash_schedule[display_cols].style.format(fmt),
                use_container_width=True
            )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Beginning Cash", f"${cash_summary.get('beginning_cash', 0):,.0f}")
        col2.metric("Total CapEx Paid", f"${cash_summary.get('total_capex_paid', 0):,.0f}")
        col3.metric("Total Distributable", f"${cash_summary.get('total_distributable', 0):,.0f}")
        col4.metric("Ending Cash", f"${cash_summary.get('ending_cash', 0):,.0f}")

        if cash_summary.get('min_cash_balance', 0) < 0:
            st.warning(f"⚠️ Cash balance went negative (lowest: ${cash_summary.get('min_cash_balance', 0):,.0f}). Consider additional capital calls or reducing CapEx.")

    # ============================================================
    # CAPITAL CALLS SCHEDULE (Display)
    # ============================================================
    if capital_calls:
        st.divider()
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


    # ============================================================
    # PARTNER RETURNS SUMMARY
    # ============================================================
    st.divider()
    st.header("🎯 Partner Returns")

    # Check if this is a child property - returns are calculated at parent level
    parent_deal_info = get_parent_deal_for_property(deal_vcode, inv)
    _is_child_property = bool(parent_deal_info)
    if _is_child_property:
        parent_name = parent_deal_info.get('Investment_Name', parent_deal_info.get('vcode', 'parent'))
        st.info(f"📊 Partner Returns are calculated at the **{parent_name}** level. "
                f"Select **{parent_name}** from the deal dropdown above to view returns.")

    # ============================================================
    # PARTNER TOTALS & RETURNS (skip for child properties)
    # ============================================================
    if not _is_child_property:

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
          state = cf_investors.get(partner) or cap_investors.get(partner)
    
          if state:
              # Calculate metrics
              unrealized = state.total_capital_outstanding + state.total_pref_balance

              from metrics import investor_metrics
              metrics = investor_metrics(state, sale_me, unrealized_nav=unrealized)

              partner_totals.append({
                  'Partner': partner,
                  'Cash Flow Distributions': cf_dist,
                  'Capital Distributions': cap_dist,
                  'Total Distributions': cf_dist + cap_dist,
                  'Capital Outstanding': state.total_capital_outstanding,
                  'Unpaid Pref (Compounded)': state.pref_unpaid_compounded + state.add_pref_unpaid_compounded,
                  'Accrued Pref (Current Yr)': state.pref_accrued_current_year + state.add_pref_accrued_current_year,
                  'Total Capital + Pref': unrealized,
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

          # --- Total Accrued Pref Owed ---
          total_pref_owed = 0.0
          for partner in all_partners:
              state = cf_investors.get(partner) or cap_investors.get(partner)
              if state:
                  total_pref_owed += state.total_pref_balance
          if total_pref_owed > 0:
              st.metric("Total Accrued Pref Owed (All Partners)", f"${total_pref_owed:,.2f}")

          # --- Capital Pool Breakdown ---
          with st.expander("Capital Pool Breakdown"):
              pool_rows = []
              for partner in sorted(all_partners):
                  state = cf_investors.get(partner) or cap_investors.get(partner)
                  if not state:
                      continue
                  for pname, pool in state.pools.items():
                      if pool.capital_outstanding == 0 and not pool.pref_tiers:
                          continue
                      for tier in pool.pref_tiers:
                          pool_rows.append({
                              "Partner": partner,
                              "Pool": pname,
                              "Capital Outstanding": pool.capital_outstanding,
                              "Pref Tier": tier.tier_name,
                              "Pref Rate": tier.pref_rate,
                              "Compounded Pref": tier.pref_unpaid_compounded,
                              "Current Year Pref": tier.pref_accrued_current_year,
                              "Total Pref": tier.pref_unpaid_compounded + tier.pref_accrued_current_year,
                              "Cumulative Cap": pool.cumulative_cap if pool.cumulative_cap is not None else "",
                              "Cumulative Returned": pool.cumulative_returned if pool.cumulative_returned > 0 else "",
                          })
                      if not pool.pref_tiers and pool.capital_outstanding > 0:
                          pool_rows.append({
                              "Partner": partner,
                              "Pool": pname,
                              "Capital Outstanding": pool.capital_outstanding,
                              "Pref Tier": "(none)",
                              "Pref Rate": 0.0,
                              "Compounded Pref": 0.0,
                              "Current Year Pref": 0.0,
                              "Total Pref": 0.0,
                              "Cumulative Cap": pool.cumulative_cap if pool.cumulative_cap is not None else "",
                              "Cumulative Returned": pool.cumulative_returned if pool.cumulative_returned > 0 else "",
                          })
              if pool_rows:
                  pool_df = pd.DataFrame(pool_rows)
                  st.dataframe(
                      pool_df.style.format({
                          'Capital Outstanding': '${:,.0f}',
                          'Pref Rate': '{:.2%}',
                          'Compounded Pref': '${:,.2f}',
                          'Current Year Pref': '${:,.2f}',
                          'Total Pref': '${:,.2f}',
                      }),
                      use_container_width=True,
                      hide_index=True,
                  )
              else:
                  st.info("No capital pool data available")

          # --- Pool Validation: Waterfall Step -> Pool Routing ---
          with st.expander("Waterfall Step \u2192 Pool Routing"):
              from config import resolve_pool_and_action as _resolve_rpa
              routing_rows = []
              for wf_type_label in ["CF_WF", "Cap_WF"]:
                  _is_cap = (wf_type_label == "Cap_WF")
                  _steps = wf_steps[
                      (wf_steps["vcode"].astype(str) == str(deal_vcode)) &
                      (wf_steps["vmisc"] == wf_type_label)
                  ].sort_values("iOrder")
                  for _, _step in _steps.iterrows():
                      _vs = str(_step["vState"]).strip()
                      _vt = str(_step.get("vtranstype", "")).strip()
                      _pn, _act = _resolve_rpa(_vs, _vt, _is_cap)
                      routing_rows.append({
                          "Waterfall": wf_type_label,
                          "iOrder": int(_step["iOrder"]),
                          "vAmtType": _step.get("vAmtType", ""),
                          "PropCode": str(_step["PropCode"]),
                          "vState": _vs,
                          "vtranstype": _vt,
                          "Pool": _pn if _pn else f"({_vs})",
                          "Action": _act if _act else _vs,
                      })
              if routing_rows:
                  st.dataframe(pd.DataFrame(routing_rows), use_container_width=True, hide_index=True)
              else:
                  st.info("No waterfall steps found")

          # Summary metrics
          st.markdown("### Deal-Level Summary")

          total_cf_dist = totals_df['Cash Flow Distributions'].sum()
          total_cap_dist = totals_df['Capital Distributions'].sum()
          total_dist = totals_df['Total Distributions'].sum()
          total_contrib = totals_df['Total Contributions'].sum()
          deal_moic = total_dist / total_contrib if total_contrib > 0 else 0

          # Calculate deal-level XIRR by combining all investor cashflows
          from metrics import xirr
          all_cashflows = []
          for partner in all_partners:
              state = cf_investors.get(partner) or cap_investors.get(partner)
              if state and state.cashflows:
                  all_cashflows.extend(state.cashflows)

          deal_xirr = xirr(all_cashflows) if all_cashflows else None

          # Calculate deal-level ROE using proper formula:
          # ROE = (Total CF Distributions / Weighted Average Capital) / Years
          from metrics import calculate_roe

          # Collect all CF distributions and capital events from all investors
          all_cf_distributions = []
          all_capital_events = []
          for partner in all_partners:
              state = cf_investors.get(partner) or cap_investors.get(partner)
              if state:
                  if hasattr(state, 'cf_distributions') and state.cf_distributions:
                      all_cf_distributions.extend(state.cf_distributions)
                  if state.cashflows:
                      all_capital_events.extend(state.cashflows)

          # Find inception date (first contribution)
          contributions = [(d, a) for d, a in all_capital_events if a < 0]
          inception_date = min(d for d, _ in contributions) if contributions else None

          if inception_date and all_capital_events:
              deal_roe = calculate_roe(all_capital_events, all_cf_distributions, inception_date, sale_me)
          else:
              deal_roe = 0.0

          col1, col2, col3, col4, col5, col6 = st.columns(6)
          col1.metric("Total CF Distributions", f"${total_cf_dist:,.0f}")
          col2.metric("Total Capital Distributions", f"${total_cap_dist:,.0f}")
          col3.metric("Total All Distributions", f"${total_dist:,.0f}")
          col4.metric("Deal MOIC", f"{deal_moic:.2f}x")
          col5.metric("Deal XIRR", f"{deal_xirr:.2%}" if deal_xirr is not None else "N/A")
          col6.metric("Deal ROE", f"{deal_roe:.2%}")

          # ============================================================
          # PARTNER PREFERRED RETURN SCHEDULES
          # ============================================================
          st.markdown("### Partner Preferred Return Schedules")

          from waterfall import pref_rates_from_waterfall_steps
          from datetime import timedelta
          from calendar import monthrange

          def get_month_end(d):
              """Get last day of month for a given date"""
              _, last_day = monthrange(d.year, d.month)
              return date(d.year, d.month, last_day)

          def build_pref_schedule(partner, pref_rate, cf_alloc_df, cap_alloc_df, acct_df, inv_map_df, target_vcode, state, start_date, end_date):
              """
              Build detailed preferred return accrual schedule for a partner.

              Includes:
              - Historical distributions from accounting feed (reduces pref accrual)
              - Projected CF waterfall distributions (reduces pref accrual)
              - Capital events (Pref paid first, then capital returned)

              Distribution logic:
              - Cash distributions first reduce compounded pref, then current year accrual
              - Excess distributions beyond pref due are profit (don't reduce equity)
              - Capital returns (Initial) reduce equity balance
              - Accruals never go negative

              Columns:
              - Date, Typename, Amt, Equity Balance, Compounded Pref, Total Inv+Comp,
              - Days, Current Due, Prior Accrual, Total Due, Amount Paid, Remaining Accrual
              """
              rows = []

              # Collect all events: (date, iOrder, typename, amount, event_type)
              # event_type: 'contribution', 'cash_distribution', 'capital_return', 'profit_share'
              events = []

              # Get cashflows from investor state (contributions)
              if state and state.cashflows:
                  for d, amt in state.cashflows:
                      if amt < 0:
                          # Contributions have order 0 (first)
                          events.append((d, 0, "Contribution", abs(amt), 'contribution'))

              # Get historical distributions from accounting feed
              if acct_df is not None and not acct_df.empty and inv_map_df is not None:
                  from loaders import build_investmentid_to_vcode
                  inv_to_vcode = build_investmentid_to_vcode(inv_map_df)

                  # Filter accounting to this deal and partner
                  acct_filtered = acct_df.copy()
                  acct_filtered["vcode"] = acct_filtered["InvestmentID"].map(inv_to_vcode)
                  acct_filtered = acct_filtered[acct_filtered["vcode"].astype(str) == str(target_vcode)]
                  acct_filtered = acct_filtered[acct_filtered["InvestorID"].astype(str) == str(partner)]

                  for _, row in acct_filtered.iterrows():
                      d = row["EffectiveDate"]
                      if pd.isna(d):
                          continue
                      # Ensure date is a proper date object
                      if isinstance(d, str):
                          d = pd.to_datetime(d).date()
                      elif hasattr(d, 'date'):
                          d = d.date()
                      elif not isinstance(d, date):
                          continue

                      amt = float(row["Amt"])
                      major_type = str(row.get("MajorType", "")).strip()
                      typename = str(row.get("Typename", "")).strip()

                      # Skip contributions (already handled above)
                      if "contrib" in major_type.lower():
                          continue

                      # Handle distributions based on Typename text
                      if "distri" in major_type.lower() and amt > 0:
                          typename_lower = typename.lower()

                          if "return of capital" in typename_lower:
                              # Distribution: Return of Capital -> directly reduces investment balance
                              events.append((d, 100, typename, amt, 'capital_return'))
                          elif "preferred return" in typename_lower or "excess cash" in typename_lower or "income" in typename_lower:
                              # Distribution: Preferred Return, Excess Cash Flow, Income
                              # -> follows payment priority (Current Due, Remaining Accrual, Compounded Pref)
                              events.append((d, 50, typename, amt, 'cash_distribution'))
                          else:
                              # Other distributions - treat as cash distribution
                              events.append((d, 50, typename, amt, 'cash_distribution'))

              # Helper to ensure date is a date object
              def ensure_date(d):
                  if isinstance(d, str):
                      return pd.to_datetime(d).date()
                  elif hasattr(d, 'date'):
                      return d.date()
                  return d

              # Get projected CF waterfall distributions (forecast period)
              if not cf_alloc_df.empty:
                  partner_cf = cf_alloc_df[cf_alloc_df['PropCode'] == partner].copy()
                  for _, row in partner_cf.iterrows():
                      d = ensure_date(row['event_date'])
                      amt = row['Allocated']
                      vstate = str(row.get('vState', '')).strip()
                      iorder = int(row.get('iOrder', 999))
                      if amt > 0:
                          if vstate == 'Pref':
                              # Pref distributions reduce pref accrual
                              events.append((d, iorder, f"CF_WF {vstate}", amt, 'cash_distribution'))
                          elif vstate == 'Initial':
                              events.append((d, iorder, f"CF_WF {vstate}", amt, 'capital_return'))
                          elif vstate == 'Share':
                              # Share distributions also reduce pref first, then are profit
                              events.append((d, iorder, f"CF_WF {vstate}", amt, 'cash_distribution'))
                          else:
                              events.append((d, iorder, f"CF_WF {vstate}", amt, 'cash_distribution'))

              # Get Cap waterfall distributions
              if not cap_alloc_df.empty:
                  partner_cap = cap_alloc_df[cap_alloc_df['PropCode'] == partner].copy()
                  for _, row in partner_cap.iterrows():
                      d = ensure_date(row['event_date'])
                      amt = row['Allocated']
                      vstate = str(row.get('vState', '')).strip()
                      iorder = int(row.get('iOrder', 999))
                      if amt > 0:
                          if vstate == 'Pref':
                              events.append((d, iorder, f"Cap_WF {vstate}", amt, 'cash_distribution'))
                          elif vstate == 'Initial':
                              events.append((d, iorder, f"Cap_WF {vstate}", amt, 'capital_return'))
                          else:
                              events.append((d, iorder, f"Cap_WF {vstate}", amt, 'profit_share'))

              # Sort events by date, then by iOrder (to ensure Pref before Initial)
              events = sorted(events, key=lambda x: (x[0], x[1]))

              if not events:
                  return pd.DataFrame()

              # Generate month-end dates from start to end
              month_ends = []
              current = get_month_end(start_date)
              while current <= end_date:
                  month_ends.append(current)
                  if current.month == 12:
                      next_month = date(current.year + 1, 1, 1)
                  else:
                      next_month = date(current.year, current.month + 1, 1)
                  current = get_month_end(next_month)

              # Combine events with month ends for accrual checkpoints
              all_dates = sorted(set([e[0] for e in events] + month_ends))

              # Build event lookup by date, preserving order
              event_lookup = {}
              for d, iorder, typename, amt, etype in events:
                  if d not in event_lookup:
                      event_lookup[d] = []
                  event_lookup[d].append((iorder, typename, amt, etype))

              # Sort events within each day by iOrder
              for d in event_lookup:
                  event_lookup[d] = sorted(event_lookup[d], key=lambda x: x[0])

              # Initialize tracking variables (matching Excel logic)
              investment_balance = 0.0  # Cumulative: contributions - return of capital
              compounded_pref = 0.0     # Pref compounded from prior years (on 12/31)
              remaining_accrual = 0.0   # Carries forward between periods
              deferred_ye_accrual = 0.0 # 12/31 remaining accrual awaiting Q4 payment
              deferred_ye_year = 0      # Year of the deferred snapshot
              last_date = None

              # Act/365 Fixed day count convention — denominator is always 365
              def days_in_year(d):
                  return 365.0

              for d in all_dates:
                  # Calculate current due (accrual for this period)
                  if last_date is not None and pref_rate > 0 and investment_balance > 0:
                      days = (d - last_date).days
                      base = max(0.0, investment_balance + compounded_pref)
                      current_due = base * pref_rate * (days / days_in_year(d))
                  else:
                      days = (d - last_date).days if last_date else 0
                      current_due = 0.0

                  # Total Due = Current Due + Remaining Accrual + Deferred 12/31 balance
                  total_due = current_due + remaining_accrual + deferred_ye_accrual

                  # Process events on this date
                  day_events = event_lookup.get(d, [])

                  if day_events:
                      for iorder, typename, amt, etype in day_events:
                          pref_paid = 0.0

                          if etype == 'contribution':
                              # Contribution increases investment balance
                              # Accumulate this period's accrual into remaining
                              remaining_accrual += current_due
                              investment_balance += amt
                              rows.append({
                                  'Date': d,
                                  'Type': typename,
                                  'Amt': -amt,  # Show as negative (outflow from investor)
                                  'Equity Balance': investment_balance,
                                  'Compounded Pref': compounded_pref,
                                  'Total Inv+Comp': investment_balance + compounded_pref,
                                  'Days': days,
                                  'Current Due': current_due,
                                  'Prior Accrual': remaining_accrual + deferred_ye_accrual - current_due,
                                  'Total Due': total_due,
                                  'Amount Paid': 0.0,
                                  'Remaining Accrual': remaining_accrual + deferred_ye_accrual,
                              })

                          elif etype == 'cash_distribution':
                              # Cash distribution payment application order:
                              # 1. Deferred year-end balance (12/31 balance — Q4 payment covers this first)
                              # 2. Compounded Pref (prior years' compounded)
                              # 3. Current Due (this period's accrual)
                              # 4. Remaining Accrual (prior period's unpaid pref, same year)
                              # Excess beyond all pref is just profit (no equity reduction)

                              pref_paid = amt
                              payment_remaining = amt
                              pre_deferred = deferred_ye_accrual

                              # 1. Apply to deferred year-end balance first
                              deferred_paid = 0.0
                              if deferred_ye_accrual > 0:
                                  deferred_paid = min(payment_remaining, deferred_ye_accrual)
                                  payment_remaining -= deferred_paid
                                  deferred_ye_accrual -= deferred_paid

                              # 2. Apply to compounded pref
                              compounded_paid = min(payment_remaining, compounded_pref)
                              payment_remaining -= compounded_paid
                              compounded_pref -= compounded_paid

                              # 3. Apply to current due
                              current_due_paid = min(payment_remaining, current_due)
                              payment_remaining -= current_due_paid
                              new_current_due = current_due - current_due_paid

                              # 4. Apply to remaining accrual (prior period, same year)
                              prior_accrual_paid = min(payment_remaining, remaining_accrual)
                              payment_remaining -= prior_accrual_paid
                              new_remaining = remaining_accrual - prior_accrual_paid

                              # Update remaining accrual (current due not yet paid becomes remaining)
                              new_remaining = max(0.0, new_remaining + new_current_due)

                              rows.append({
                                  'Date': d,
                                  'Type': typename,
                                  'Amt': amt,
                                  'Equity Balance': investment_balance,
                                  'Compounded Pref': compounded_pref,
                                  'Total Inv+Comp': investment_balance + compounded_pref,
                                  'Days': days,
                                  'Current Due': current_due,
                                  'Prior Accrual': remaining_accrual + pre_deferred,
                                  'Total Due': total_due,
                                  'Amount Paid': pref_paid,
                                  'Remaining Accrual': new_remaining + deferred_ye_accrual,
                              })
                              remaining_accrual = new_remaining

                          elif etype == 'capital_return':
                              # Distribution: Return of Capital
                              # Directly reduces investment balance (does NOT pay pref first)
                              # Cannot return more capital than was invested

                              capital_returned = min(amt, investment_balance)
                              investment_balance = max(0.0, investment_balance - capital_returned)

                              # Remaining accrual carries forward (add current due)
                              new_remaining = remaining_accrual + current_due

                              rows.append({
                                  'Date': d,
                                  'Type': typename,
                                  'Amt': amt,
                                  'Equity Balance': investment_balance,
                                  'Compounded Pref': compounded_pref,
                                  'Total Inv+Comp': investment_balance + compounded_pref,
                                  'Days': days,
                                  'Current Due': current_due,
                                  'Prior Accrual': remaining_accrual + deferred_ye_accrual,
                                  'Total Due': total_due,
                                  'Amount Paid': 0.0,  # Capital return doesn't pay pref
                                  'Remaining Accrual': new_remaining + deferred_ye_accrual,
                              })
                              remaining_accrual = new_remaining

                          elif etype == 'profit_share':
                              # Profit share after pref is cleared - no pref reduction, no equity change
                              # Accumulate this period's accrual into remaining
                              remaining_accrual += current_due
                              rows.append({
                                  'Date': d,
                                  'Type': typename,
                                  'Amt': amt,
                                  'Equity Balance': investment_balance,
                                  'Compounded Pref': compounded_pref,
                                  'Total Inv+Comp': investment_balance + compounded_pref,
                                  'Days': days,
                                  'Current Due': current_due,
                                  'Prior Accrual': remaining_accrual + deferred_ye_accrual - current_due,
                                  'Total Due': total_due,
                                  'Amount Paid': 0.0,
                                  'Remaining Accrual': remaining_accrual + deferred_ye_accrual,
                              })

                          # Reset for subsequent events on same day
                          days = 0
                          current_due = 0.0
                          total_due = remaining_accrual + deferred_ye_accrual

                  else:
                      # Month-end accrual checkpoint (no event)
                      # Update remaining accrual with current due
                      # (deferred_ye_accrual is tracked separately)
                      remaining_accrual = current_due + remaining_accrual

                      rows.append({
                          'Date': d,
                          'Type': 'Accrual',
                          'Amt': 0.0,
                          'Equity Balance': investment_balance,
                          'Compounded Pref': compounded_pref,
                          'Total Inv+Comp': investment_balance + compounded_pref,
                          'Days': days,
                          'Current Due': current_due,
                          'Prior Accrual': remaining_accrual - current_due + deferred_ye_accrual,
                          'Total Due': total_due,
                          'Amount Paid': 0.0,
                          'Remaining Accrual': remaining_accrual + deferred_ye_accrual,
                      })

                  # Deferred year-end compounding:
                  # At 12/31, snapshot remaining accrual as deferred.  The next
                  # distribution (within 45 days) gets a chance to pay it down.
                  # Only the unpaid shortfall compounds as of 1/1.
                  if d.month == 12 and d.day == 31:
                      deferred_ye_accrual += remaining_accrual
                      remaining_accrual = 0.0
                      deferred_ye_year = d.year

                  elif deferred_ye_accrual > 0 and d.year > deferred_ye_year:
                      # In the new year with a pending deferred balance.
                      # Compound after the first distribution event, or by 2/15.
                      had_dist = bool(day_events) and any(
                          e[3] == 'cash_distribution' for e in day_events
                      )
                      if had_dist or d >= date(deferred_ye_year + 1, 2, 15):
                          shortfall = deferred_ye_accrual
                          if shortfall > 0:
                              compounded_pref += shortfall
                              rows.append({
                                  'Date': date(deferred_ye_year + 1, 1, 1),
                                  'Type': 'Deferred Compound',
                                  'Amt': shortfall,
                                  'Equity Balance': investment_balance,
                                  'Compounded Pref': compounded_pref,
                                  'Total Inv+Comp': investment_balance + compounded_pref,
                                  'Days': 0,
                                  'Current Due': 0.0,
                                  'Prior Accrual': shortfall,
                                  'Total Due': shortfall,
                                  'Amount Paid': 0.0,
                                  'Remaining Accrual': remaining_accrual,
                              })
                          deferred_ye_accrual = 0.0

                  last_date = d

              return pd.DataFrame(rows)

          # Get pref rates from waterfall steps
          pref_rates = pref_rates_from_waterfall_steps(wf_steps, deal_vcode)

          # Find inception date (first contribution)
          first_contrib_date = None
          for partner in sorted(all_partners):
              state = cf_investors.get(partner) or cap_investors.get(partner)
              if state and state.cashflows:
                  contribs = [d for d, a in state.cashflows if a < 0]
                  if contribs:
                      d = min(contribs)
                      if first_contrib_date is None or d < first_contrib_date:
                          first_contrib_date = d

          if first_contrib_date is None:
              first_contrib_date = date(start_year, 1, 1)

          # Create expander for each partner
          for partner in sorted(all_partners):
              state = cf_investors.get(partner) or cap_investors.get(partner)
              pref_rate_val = pref_rates.get(partner, 0.0)

              # Multi-tier: pref_rate_val may be a list of rates
              if isinstance(pref_rate_val, list):
                  rate_list = pref_rate_val
                  rate_label = " / ".join(f"{r:.2%}" for r in rate_list)
              else:
                  rate_list = [pref_rate_val] if pref_rate_val else []
                  rate_label = f"{pref_rate_val:.2%}" if pref_rate_val else "0.00%"

              with st.expander(f"📊 {partner} Preferred Return Schedule (Rate: {rate_label})"):
                  if not rate_list or all(r == 0 for r in rate_list):
                      st.warning(f"No pref rate found for {partner}")
                  else:
                      for tier_idx, pref_rate in enumerate(rate_list):
                          if len(rate_list) > 1:
                              st.markdown(f"**Tier {tier_idx + 1}: {pref_rate:.2%}** (initial pool)")

                          schedule_df = build_pref_schedule(
                              partner, pref_rate, cf_alloc, cap_alloc,
                              acct, inv, deal_vcode,
                              state, first_contrib_date, sale_me
                          )

                          if schedule_df.empty:
                              st.info("No events found for this partner")
                          else:
                              # Format the dataframe
                              st.dataframe(
                                  schedule_df.style.format({
                                      'Amt': '${:,.0f}',
                                      'Equity Balance': '${:,.0f}',
                                      'Compounded Pref': '${:,.0f}',
                                      'Total Inv+Comp': '${:,.0f}',
                                      'Days': '{:.0f}',
                                      'Current Due': '${:,.2f}',
                                      'Prior Accrual': '${:,.2f}',
                                      'Total Due': '${:,.2f}',
                                      'Amount Paid': '${:,.2f}',
                                      'Remaining Accrual': '${:,.2f}',
                                  }),
                                  use_container_width=True,
                                  hide_index=True
                              )

                              # Summary
                              total_accrued = schedule_df['Current Due'].sum()
                              total_paid = schedule_df['Amount Paid'].sum()
                              final_remaining = schedule_df['Remaining Accrual'].iloc[-1] if len(schedule_df) > 0 else 0
                              final_compounded = schedule_df['Compounded Pref'].iloc[-1] if len(schedule_df) > 0 else 0

                              col1, col2, col3, col4 = st.columns(4)
                              col1.metric("Total Pref Accrued", f"${total_accrued:,.0f}")
                              col2.metric("Total Pref Paid", f"${total_paid:,.0f}")
                              col3.metric("Final Compounded Pref", f"${final_compounded:,.0f}")
                              col4.metric("Final Remaining Accrual", f"${final_remaining:,.0f}")

      else:
          st.info("No partner data available")


      if cap_alloc.empty:
          st.info("No Cap_WF steps found for this deal.")
      else:
          st.markdown("**Investor Summary**")
          if cap_investors:
              # Convert investor states dict to DataFrame for display
              inv_rows = []
              for pc, stt in cap_investors.items():
                  unrealized = stt.total_capital_outstanding + stt.total_pref_balance
                  metrics = investor_metrics(stt, sale_me, unrealized_nav=unrealized)
                  inv_rows.append({
                      "PropCode": pc,
                      "CapitalOutstanding": stt.capital_outstanding,
                      "UnpaidPrefCompounded": stt.pref_unpaid_compounded,
                      "AccruedPrefCurrentYear": stt.pref_accrued_current_year,
                      "TotalDistributions": metrics.get('TotalDistributions', 0),
                      "TotalContributions": metrics.get('TotalContributions', 0),
                      "IRR": metrics.get('IRR'),
                      "ROE": metrics.get('ROE', 0),
                      "MOIC": metrics.get('MOIC', 0),
                  })
              out = pd.DataFrame(inv_rows)
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

      # ============================================================
      # XIRR CASH FLOWS TABLE
      # ============================================================
      # Build a table showing all cash flows used for XIRR calculation
      # Columns: Date, Description, Pref Equity, Ptr Equity, Deal
      xirr_cf_rows = []

      # Collect cashflows from all partners
      for partner in sorted(all_partners):
          state = cf_investors.get(partner) or cap_investors.get(partner)
          if state and state.cashflows:
              # Determine if this is Pref Equity (PPI) or Partner Equity (OP)
              is_pref_equity = not partner.upper().startswith("OP")

              for cf_date, cf_amount in state.cashflows:
                  # Determine description based on sign
                  if cf_amount < 0:
                      description = "Contribution"
                  else:
                      description = "Distribution"

                  xirr_cf_rows.append({
                      "Date": cf_date,
                      "Description": description,
                      "Partner": partner,
                      "is_pref": is_pref_equity,
                      "Amount": cf_amount
                  })

      if xirr_cf_rows:
          xirr_cf_df = pd.DataFrame(xirr_cf_rows)

          # Pivot to get Pref Equity vs Ptr Equity columns
          # Group by Date and Description, sum amounts by equity type
          grouped = xirr_cf_df.groupby(["Date", "Description", "is_pref"])["Amount"].sum().reset_index()

          # Pivot to separate Pref and Ptr columns
          pref_df = grouped[grouped["is_pref"] == True][["Date", "Description", "Amount"]].rename(columns={"Amount": "Pref Equity"})
          ptr_df = grouped[grouped["is_pref"] == False][["Date", "Description", "Amount"]].rename(columns={"Amount": "Ptr Equity"})

          # Merge on Date and Description
          final_df = pd.merge(pref_df, ptr_df, on=["Date", "Description"], how="outer").fillna(0)

          # Calculate Deal total
          final_df["Deal"] = final_df["Pref Equity"] + final_df["Ptr Equity"]

          # Sort by date ascending
          final_df = final_df.sort_values("Date").reset_index(drop=True)

          # Format for display
          display_df = final_df.copy()
          display_df["Pref Equity"] = display_df["Pref Equity"].map(lambda x: f"({abs(x):,.0f})" if x < 0 else f"{x:,.0f}" if x != 0 else "")
          display_df["Ptr Equity"] = display_df["Ptr Equity"].map(lambda x: f"({abs(x):,.0f})" if x < 0 else f"{x:,.0f}" if x != 0 else "")
          display_df["Deal"] = display_df["Deal"].map(lambda x: f"({abs(x):,.0f})" if x < 0 else f"{x:,.0f}" if x != 0 else "")

          with st.expander("XIRR Cash Flows"):
              st.dataframe(display_df[["Date", "Description", "Pref Equity", "Ptr Equity", "Deal"]], use_container_width=True)

with tab_financials:
    # Independent property selector (defaults to Deal Analysis selection)
    fin_label = st.selectbox("Select Property", labels_sorted,
                             index=labels_sorted.index(selected_label),
                             key="fin_deal_selector")
    fin_row = inv_disp[inv_disp["DealLabel"] == fin_label].iloc[0]
    fin_vcode = str(fin_row["vcode"])

    # fc_deal_modeled is only valid for the Deal Analysis selection
    fin_fc = fc_deal_modeled if fin_vcode == deal_vcode else None

    render_property_financials(
        deal_vcode=fin_vcode,
        isbs_raw=isbs_raw,
        fc_deal_modeled=fin_fc,
        tenants_raw=tenants_raw,
        inv=inv,
        mri_loans_raw=mri_loans_raw,
        mri_val=mri_val,
        wf=wf,
        commitments_raw=commitments_raw,
        acct=acct,
        occupancy_raw=occupancy_raw,
    )

with tab_ownership:
    # ============================================================
    # OWNERSHIP TREE ANALYSIS
    # ============================================================
    st.header("Ownership Relationships & Waterfall Requirements")

    if relationships_raw is not None:
        # Load and process relationships
        relationships = load_relationships(relationships_raw)

        st.success(f"Loaded {len(relationships)} ownership relationships")

        # Build ownership tree
        with st.spinner("Building ownership tree..."):
            nodes = build_ownership_tree(relationships)

        st.info(f"Identified {len(nodes)} unique entities in ownership structure")

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
                label="Download Waterfall Requirements CSV",
                data=csv,
                file_name="waterfall_requirements.csv",
                mime="text/csv"
            )

        # Use the selected deal for ownership analysis
        selected_entity = deal_vcode

        if selected_entity in nodes:
            node = nodes[selected_entity]

            ultimate = get_ultimate_investors(selected_entity, nodes, normalize=True)
            ultimate = consolidate_ultimate_investors(ultimate)
            ultimate = sorted(ultimate, key=lambda x: x[1], reverse=True)

            if ultimate:
                with st.expander("Ultimate Beneficial Owners"):
                    ult_data = []
                    total_pct = 0.0
                    for inv_id, eff_pct in ultimate:
                        inv_name = nodes[inv_id].name if inv_id in nodes else ""
                        total_pct += eff_pct
                        ult_data.append({
                            "Investor ID": inv_id,
                            "Name": inv_name,
                            "Effective Ownership %": f"{eff_pct*100:.4f}%"
                        })
                    # Add total row
                    ult_data.append({
                        "Investor ID": "",
                        "Name": "**TOTAL**",
                        "Effective Ownership %": f"{total_pct*100:.4f}%"
                    })
                    st.dataframe(pd.DataFrame(ult_data), use_container_width=True, hide_index=True)

            with st.expander("View Ownership Tree"):
                tree_viz = visualize_ownership_tree(selected_entity, nodes)
                st.code(tree_viz, language=None)
        else:
            st.info(f"No ownership relationships found for {deal_vcode}")

        # ============================================================
        # UPSTREAM WATERFALL ANALYSIS
        # ============================================================
        st.markdown("---")
        st.header("Upstream Waterfall Analysis")
        st.markdown("Trace cash flows from deal level through funds, JVs, and partnerships to ultimate beneficiaries.")

        # Get list of deals with forecasts for selection
        deals_with_forecasts = []
        if fc is not None and not fc.empty:
            fc_vcodes = fc["vcode"].astype(str).unique()
            for vc in fc_vcodes:
                deal_info = inv[inv["vcode"].astype(str) == vc]
                if not deal_info.empty:
                    deal_name = deal_info["Investment_Name"].iloc[0] if "Investment_Name" in deal_info.columns else vc
                    deals_with_forecasts.append((vc, f"{deal_name} ({vc})"))

        if deals_with_forecasts:
            # Deal selection for upstream analysis
            col1, col2 = st.columns([2, 1])
            with col1:
                upstream_deal_options = [label for _, label in deals_with_forecasts]
                upstream_deal_label = st.selectbox(
                    "Select Deal for Upstream Analysis",
                    upstream_deal_options,
                    key="upstream_deal_selector"
                )
                # Extract vcode from selection
                upstream_vcode = None
                for vc, label in deals_with_forecasts:
                    if label == upstream_deal_label:
                        upstream_vcode = vc
                        break

            with col2:
                analysis_amount = st.number_input(
                    "Distribution Amount ($)",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=10000,
                    key="upstream_analysis_amount"
                )

            # Run analysis button
            if st.button("Run Upstream Waterfall Analysis", key="run_upstream_btn"):
                if upstream_vcode:
                    with st.spinner("Processing upstream waterfalls..."):
                        try:
                            # Create test cash for analysis
                            from datetime import date as date_class
                            analysis_cash = pd.DataFrame([{
                                "event_date": date_class(2025, 12, 31),
                                "cash_available": float(analysis_amount)
                            }])

                            # Run deal-level waterfall
                            deal_alloc, deal_states = run_waterfall(
                                wf_steps=wf,
                                vcode=upstream_vcode,
                                wf_name="CF_WF",
                                period_cash=analysis_cash,
                                initial_states={},
                            )

                            if deal_alloc.empty:
                                st.warning(f"No waterfall defined for {upstream_vcode}")
                            else:
                                # Add vcode to allocations for tracking
                                deal_alloc["vcode"] = upstream_vcode

                                # Run recursive upstream waterfalls
                                upstream_alloc, entity_states, beneficiary_totals = run_recursive_upstream_waterfalls(
                                    deal_allocations=deal_alloc,
                                    wf_steps=wf,
                                    relationships=relationships,
                                    wf_type="CF_WF",
                                )

                                # Store results in session state
                                st.session_state.upstream_results = {
                                    "deal_alloc": deal_alloc,
                                    "upstream_alloc": upstream_alloc,
                                    "entity_states": entity_states,
                                    "beneficiary_totals": beneficiary_totals,
                                    "analysis_amount": analysis_amount,
                                    "vcode": upstream_vcode,
                                }
                                st.success("Upstream analysis complete!")

                        except Exception as e:
                            st.error(f"Error running upstream analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

            # Display results if available
            if "upstream_results" in st.session_state and st.session_state.upstream_results:
                results = st.session_state.upstream_results
                upstream_alloc = results["upstream_alloc"]
                beneficiary_totals = results["beneficiary_totals"]
                analysis_amount = results["analysis_amount"]

                if not upstream_alloc.empty:
                    # Summary metrics
                    st.subheader("Distribution Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Distributed", f"${analysis_amount:,.0f}")
                    with col2:
                        num_levels = upstream_alloc["Level"].max() + 1
                        st.metric("Ownership Levels", f"{num_levels}")
                    with col3:
                        num_beneficiaries = len(beneficiary_totals)
                        st.metric("Terminal Beneficiaries", f"{num_beneficiaries}")
                    with col4:
                        # Calculate OWPSC's share
                        owpsc_flow = calculate_entity_through_flow(upstream_alloc, "OWPSC")
                        owpsc_total = owpsc_flow.get("total_received", 0)
                        owpsc_pct = (owpsc_total / analysis_amount * 100) if analysis_amount > 0 else 0
                        st.metric("OWPSC Share", f"${owpsc_total:,.0f} ({owpsc_pct:.1f}%)")

                    # OWPSC Revenue Streams
                    with st.expander("OWPSC Revenue Streams", expanded=True):
                        ubo_revenue = calculate_ubo_revenue_streams(upstream_alloc, "OWPSC", "PSC1")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Revenue Breakdown:**")
                            rev_data = [
                                {"Stream": "GP Promotes (via PSC1)", "Amount": f"${ubo_revenue.get('gp_promotes', 0):,.2f}"},
                                {"Stream": "Fees", "Amount": f"${ubo_revenue.get('fees', 0):,.2f}"},
                                {"Stream": "Capital Income", "Amount": f"${ubo_revenue.get('capital_income', 0):,.2f}"},
                                {"Stream": "**Total Through-Flow**", "Amount": f"**${ubo_revenue.get('total_through_flow', 0):,.2f}**"},
                            ]
                            st.dataframe(pd.DataFrame(rev_data), hide_index=True, use_container_width=True)

                        with col2:
                            st.markdown("**PSC1 Sources (flows to OWPSC):**")
                            psc1_flow = calculate_entity_through_flow(upstream_alloc, "PSC1")
                            if psc1_flow.get("by_source"):
                                source_data = [
                                    {"Source": src, "Amount": f"${amt:,.2f}"}
                                    for src, amt in sorted(psc1_flow["by_source"].items(), key=lambda x: -x[1])
                                ]
                                st.dataframe(pd.DataFrame(source_data), hide_index=True, use_container_width=True)

                    # Flow by Level
                    with st.expander("Cash Flow by Level"):
                        for level in sorted(upstream_alloc["Level"].unique()):
                            level_data = upstream_alloc[upstream_alloc["Level"] == level]
                            level_total = level_data["Allocated"].sum()

                            level_name = "Deal Level" if level == 0 else f"Level {level}"
                            st.markdown(f"**{level_name}** (${level_total:,.0f} total)")

                            # Group by entity for cleaner display
                            level_summary = level_data.groupby(["Entity", "PropCode", "vState"]).agg({
                                "Allocated": "sum"
                            }).reset_index()
                            level_summary = level_summary[level_summary["Allocated"] > 0.01]
                            level_summary["Allocated"] = level_summary["Allocated"].apply(lambda x: f"${x:,.2f}")

                            st.dataframe(
                                level_summary.rename(columns={
                                    "Entity": "From",
                                    "PropCode": "To",
                                    "vState": "Type",
                                    "Allocated": "Amount"
                                }),
                                hide_index=True,
                                use_container_width=True
                            )

                    # Terminal Beneficiaries
                    with st.expander("Terminal Beneficiaries"):
                        ben_data = [
                            {
                                "Entity": entity_id,
                                "Cash Received": f"${total:,.2f}",
                                "% of Total": f"{(total/analysis_amount*100):.2f}%"
                            }
                            for entity_id, total in sorted(beneficiary_totals.items(), key=lambda x: -x[1])
                            if total > 0.01
                        ]
                        st.dataframe(pd.DataFrame(ben_data), hide_index=True, use_container_width=True)

                        # Download button
                        ben_df = pd.DataFrame([
                            {"EntityID": k, "CashReceived": v, "PctOfTotal": v/analysis_amount*100}
                            for k, v in beneficiary_totals.items()
                        ])
                        csv = ben_df.to_csv(index=False)
                        st.download_button(
                            label="Download Beneficiaries CSV",
                            data=csv,
                            file_name="terminal_beneficiaries.csv",
                            mime="text/csv"
                        )

                    # Full Allocation Detail
                    with st.expander("Full Allocation Detail"):
                        detail_cols = ["Level", "Entity", "Path", "PropCode", "vState", "Allocated", "vtranstype"]
                        available_cols = [c for c in detail_cols if c in upstream_alloc.columns]
                        detail_df = upstream_alloc[available_cols].copy()
                        detail_df = detail_df[detail_df["Allocated"] > 0.01]
                        detail_df["Allocated"] = detail_df["Allocated"].apply(lambda x: f"${x:,.2f}")

                        st.dataframe(detail_df, hide_index=True, use_container_width=True)

        else:
            st.info("No deals with forecasts available for upstream analysis.")

    else:
        st.info("Upload MRI_IA_Relationship.csv to analyze multi-tiered ownership structures")

with tab_reports:
    render_reports(
        inv=inv, wf=wf, acct=acct, fc=fc, coa=coa,
        mri_loans_raw=mri_loans_raw, mri_supp=mri_supp, mri_val=mri_val,
        relationships_raw=relationships_raw,
        capital_calls_raw=capital_calls_raw,
        isbs_raw=isbs_raw,
        start_year=int(start_year),
        horizon_years=int(horizon_years),
        pro_yr_base=int(pro_yr_base),
    )
