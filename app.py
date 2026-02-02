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
from consolidation import build_consolidated_forecast, get_sub_portfolio_summary

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="PSC Asset Management")
st.title("Peaceable Street Capital - Asset Management System")


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
            if 'cached_data' in st.session_state:
                del st.session_state.cached_data
            if 'data_cache_key' in st.session_state:
                del st.session_state.data_cache_key
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

    # Create a cache key based on mode and folder/uploads
    if mode == "SQLite Database":
        cache_key = f"db_{db_path}"
    elif mode == "Local folder":
        cache_key = f"local_{folder}"
    else:
        # For uploads, create key from uploaded file names
        cache_key = f"upload_{len([k for k, v in uploads.items() if v is not None])}"

    # Check if data is already loaded in session state
    if 'data_cache_key' in st.session_state and st.session_state.data_cache_key == cache_key:
        if 'cached_data' in st.session_state:
            return st.session_state.cached_data

    if mode == "SQLite Database":
        if not db_path or not Path(db_path).exists():
            st.error(f"Database not found: {db_path}")
            st.stop()

        conn = sqlite3.connect(db_path)

        # Load required tables
        inv = pd.read_sql('SELECT * FROM deals', conn)
        wf = pd.read_sql('SELECT * FROM waterfalls', conn)
        coa_raw = pd.read_sql('SELECT * FROM coa', conn)
        coa = load_coa(coa_raw)
        acct = pd.read_sql('SELECT * FROM accounting', conn)
        fc_raw = pd.read_sql('SELECT * FROM forecasts', conn)
        fc = load_forecast(fc_raw, coa, int(pro_yr_base))

        # Optional tables
        try:
            mri_loans_raw = pd.read_sql('SELECT * FROM loans', conn)
        except:
            mri_loans_raw = pd.DataFrame()

        try:
            mri_val = pd.read_sql('SELECT * FROM valuations', conn)
        except:
            mri_val = pd.DataFrame()

        # Relationships table (key for sub-portfolio consolidation)
        try:
            relationships_raw = pd.read_sql('SELECT * FROM relationships', conn)
        except:
            relationships_raw = None

        # Capital calls
        try:
            capital_calls_raw = pd.read_sql('SELECT * FROM capital_calls', conn)
        except:
            capital_calls_raw = None

        # ISBS (balance sheet for cash balances)
        try:
            isbs_raw = pd.read_sql('SELECT * FROM isbs', conn)
        except:
            isbs_raw = None

        # Occupancy data for One Pager
        try:
            occupancy_raw = pd.read_sql('SELECT * FROM occupancy', conn)
        except:
            occupancy_raw = None

        # Commitments for One Pager PE performance
        try:
            commitments_raw = pd.read_sql('SELECT * FROM commitments', conn)
        except:
            commitments_raw = None

        # Tenant roster for commercial properties
        try:
            tenants_raw = pd.read_sql('SELECT * FROM tenants', conn)
        except:
            tenants_raw = None

        # Not typically in DB
        mri_supp = pd.DataFrame()
        fund_deals_raw = pd.DataFrame()
        inv_wf_raw = pd.DataFrame()
        inv_acct_raw = pd.DataFrame()

        conn.close()

    elif mode == "Local folder":
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
    
    # Capital calls file (only for Local folder and Upload modes - SQLite loads it above)
    if mode == "Local folder":
        capital_calls_raw = None
        cc_path = Path(f"{folder}/MRI_Capital_Calls.csv")
        if cc_path.exists():
            capital_calls_raw = pd.read_csv(cc_path)
    elif mode == "Upload CSVs":
        capital_calls_raw = None
        if uploads.get("capital_calls"):
            capital_calls_raw = pd.read_csv(uploads["capital_calls"])

    # ISBS (balance sheet) file for cash balances (only for Local folder and Upload modes)
    if mode == "Local folder":
        isbs_raw = None
        isbs_path = Path(f"{folder}/ISBS_Download.csv")
        if isbs_path.exists():
            try:
                isbs_raw = pd.read_csv(isbs_path, encoding='utf-8')
            except Exception as e:
                print(f"Warning: Could not load ISBS file: {e}")
                isbs_raw = None
    elif mode == "Upload CSVs":
        isbs_raw = None
        if uploads.get("isbs"):
            try:
                isbs_raw = pd.read_csv(uploads["isbs"], encoding='utf-8')
            except Exception as e:
                print(f"Warning: Could not load uploaded ISBS file: {e}")
                isbs_raw = None
    
    # Relationship file (only for Local folder and Upload modes - SQLite loads it above)
    if mode == "Local folder":
        relationships_raw = None
        rel_path = Path(f"{folder}/MRI_IA_Relationship.csv")
        if rel_path.exists():
            relationships_raw = pd.read_csv(rel_path)
    elif mode == "Upload CSVs":
        relationships_raw = None
        if uploads.get("relationships"):
            relationships_raw = pd.read_csv(uploads["relationships"])

    # Occupancy file (for One Pager - only for Local folder and Upload modes)
    if mode == "Local folder":
        occupancy_raw = None
        occ_path = Path(f"{folder}/MRI_Occupancy_Download.csv")
        if occ_path.exists():
            try:
                occupancy_raw = pd.read_csv(occ_path)
            except Exception as e:
                print(f"Warning: Could not load Occupancy file: {e}")
                occupancy_raw = None
    elif mode == "Upload CSVs":
        occupancy_raw = None
        # Could add upload support here if needed

    # Commitments file (for One Pager - only for Local folder and Upload modes)
    if mode == "Local folder":
        commitments_raw = None
        comm_path = Path(f"{folder}/MRI_Commitments.csv")
        if comm_path.exists():
            try:
                commitments_raw = pd.read_csv(comm_path)
            except Exception as e:
                print(f"Warning: Could not load Commitments file: {e}")
                commitments_raw = None
    elif mode == "Upload CSVs":
        commitments_raw = None
        # Could add upload support here if needed

    # Tenant roster file (for commercial properties - only for Local folder and Upload modes)
    if mode == "Local folder":
        tenants_raw = None
        tenant_path = Path(f"{folder}/Tenant_Report.csv")
        if tenant_path.exists():
            try:
                tenants_raw = pd.read_csv(tenant_path)
            except Exception as e:
                print(f"Warning: Could not load Tenant Report file: {e}")
                tenants_raw = None
    elif mode == "Upload CSVs":
        tenants_raw = None
        # Could add upload support here if needed

    # Normalize investment map
    inv.columns = [str(c).strip() for c in inv.columns]
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)

    # Prepare return data
    result = (inv, wf, acct, fc, coa, mri_loans_raw, mri_supp, mri_val, fund_deals_raw, inv_wf_raw, inv_acct_raw, relationships_raw, capital_calls_raw, isbs_raw, occupancy_raw, commitments_raw, tenants_raw)

    # Cache in session state
    st.session_state.cached_data = result
    st.session_state.data_cache_key = cache_key

    return result


# Load data with progress indicator
with st.spinner("Loading data..."):
    inv, wf, acct, fc, coa, mri_loans_raw, mri_supp, mri_val, fund_deals_raw, inv_wf_raw, inv_acct_raw, relationships_raw, capital_calls_raw, isbs_raw, occupancy_raw, commitments_raw, tenants_raw = load_inputs()


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
# ============================================================
# DEAL HEADER - Option B: Compact Header + Data Table
# ============================================================

# Get capitalization data
def get_deal_capitalization(acct, inv, wf, mri_val, mri_loans, deal_vcode):
    """Calculate deal capitalization from accounting_feed.
    
    Equity = Contributions + Return of Capital distributions.
    InvestorID links to PropCode in waterfalls.
    InvestmentID links to vcode via investment_map.
    """
    from loaders import build_investmentid_to_vcode
    
    cap_data = {
        'as_of_date': None,
        'debt': 0.0,
        'pref_equity': 0.0,
        'partner_equity': 0.0,
        'total_cap': 0.0,
        'current_valuation': 0.0,
        'cap_rate': 0.0,
        'pe_exposure_cap': 0.0,
        'pe_exposure_value': 0.0,
    }
    
    try:
        inv_to_vcode = build_investmentid_to_vcode(inv)
        deal_investment_ids = [iid for iid, vc in inv_to_vcode.items() if str(vc) == str(deal_vcode)]
        acct_norm = acct.copy()
        acct_norm.columns = [str(c).strip() for c in acct_norm.columns]
        acct_norm["InvestmentID"] = acct_norm["InvestmentID"].astype(str).str.strip()
        deal_acct = acct_norm[acct_norm["InvestmentID"].isin(deal_investment_ids)].copy()

        if not deal_acct.empty:
            if 'EffectiveDate' in deal_acct.columns:
                deal_acct['EffectiveDate'] = pd.to_datetime(deal_acct['EffectiveDate'], errors='coerce')
                cap_data['as_of_date'] = deal_acct['EffectiveDate'].max()
            deal_acct["MajorType"] = deal_acct["MajorType"].fillna("").astype(str).str.strip()
            deal_acct["Amt"] = pd.to_numeric(deal_acct["Amt"], errors="coerce").fillna(0.0)
            if "TypeName" not in deal_acct.columns and "Typename" in deal_acct.columns:
                deal_acct["TypeName"] = deal_acct["Typename"]
            elif "TypeName" not in deal_acct.columns:
                deal_acct["TypeName"] = ""
            deal_acct["TypeName"] = deal_acct["TypeName"].fillna("").astype(str).str.strip()
            deal_acct["InvestorID"] = deal_acct["InvestorID"].astype(str).str.strip()
            # Calculate equity balances per investor
            # InvestorID starting with "OP" = Partner Equity, otherwise = Preferred Equity
            investor_balances = {}
            for _, row in deal_acct.iterrows():
                investor_id = row["InvestorID"]
                major_type = row["MajorType"].lower()
                type_name = row["TypeName"].lower()
                amt = float(row["Amt"])
                if investor_id not in investor_balances:
                    investor_balances[investor_id] = 0.0
                if "contrib" in major_type:
                    investor_balances[investor_id] += abs(amt)
                if "distri" in major_type and "return of capital" in type_name:
                    investor_balances[investor_id] -= abs(amt)
            for investor_id, balance in investor_balances.items():
                # InvestorID starting with "OP" indicates Operating Partner (Partner Equity)
                if investor_id.upper().startswith("OP"):
                    cap_data['partner_equity'] += max(0, balance)
                else:
                    cap_data['pref_equity'] += max(0, balance)
        # Get debt from MRI_Loans if available
        if mri_loans is not None and not mri_loans.empty:
            mri_loans_copy = mri_loans.copy()
            mri_loans_copy.columns = [str(col).strip() for col in mri_loans_copy.columns]
            if 'vCode' not in mri_loans_copy.columns and 'vcode' in mri_loans_copy.columns:
                mri_loans_copy = mri_loans_copy.rename(columns={'vcode': 'vCode'})
            if 'vCode' in mri_loans_copy.columns:
                mri_loans_copy['vCode'] = mri_loans_copy['vCode'].astype(str)
                deal_loans = mri_loans_copy[mri_loans_copy['vCode'] == str(deal_vcode)]
                if not deal_loans.empty and 'mOrigLoanAmt' in deal_loans.columns:
                    cap_data['debt'] = pd.to_numeric(deal_loans['mOrigLoanAmt'], errors='coerce').fillna(0).sum()
        cap_data['total_cap'] = cap_data['debt'] + cap_data['pref_equity'] + cap_data['partner_equity']
        if mri_val is not None and not mri_val.empty:
            mri_val_copy = mri_val.copy()
            mri_val_copy.columns = [str(c).strip() for c in mri_val_copy.columns]
            if 'vcode' not in mri_val_copy.columns and 'vCode' in mri_val_copy.columns:
                mri_val_copy = mri_val_copy.rename(columns={'vCode': 'vcode'})
            if 'vcode' in mri_val_copy.columns:
                mri_val_copy['vcode'] = mri_val_copy['vcode'].astype(str)
                val_deal = mri_val_copy[mri_val_copy['vcode'] == str(deal_vcode)]
                if not val_deal.empty:
                    if 'mIncomeCapConcludedValue' in val_deal.columns:
                        val = val_deal['mIncomeCapConcludedValue'].iloc[-1]
                        cap_data['current_valuation'] = float(val) if pd.notna(val) else 0.0
                    if 'fCapRate' in val_deal.columns:
                        rate = val_deal['fCapRate'].iloc[-1]
                        cap_data['cap_rate'] = float(rate) if pd.notna(rate) else 0.0
        senior_exposure = cap_data['debt'] + cap_data['pref_equity']
        if cap_data['total_cap'] > 0:
            cap_data['pe_exposure_cap'] = senior_exposure / cap_data['total_cap']
        if cap_data['current_valuation'] > 0:
            cap_data['pe_exposure_value'] = senior_exposure / cap_data['current_valuation']
    except Exception as e:
        import traceback
        print(f"Error in get_deal_capitalization: {e}")
        print(traceback.format_exc())

    return cap_data

# Calculate capitalization (loan_sched will be defined after loan modeling, so we'll update this later)
# For now, get what we can without loan data
cap_data = get_deal_capitalization(acct, inv, wf, mri_val, mri_loans_raw, deal_vcode)

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
# FORECAST & LOAN MODELING (with sub-portfolio consolidation)
# ============================================================
debug_msgs = []

# Check for sub-portfolio consolidation
deal_investment_id = str(selected_row.get('InvestmentID', ''))
consolidation_info = None

if relationships_raw is not None and not relationships_raw.empty:
    # Try consolidation for sub-portfolio deals
    consolidated_fc, consolidated_loans, consolidation_info = build_consolidated_forecast(
        deal_investment_id=deal_investment_id,
        deals=inv,
        relationships=relationships_raw,
        forecasts=fc,
        loans=mri_loans_raw if mri_loans_raw is not None else pd.DataFrame(),
        debug=True
    )

    if consolidation_info.get('is_sub_portfolio', False):
        fc_raw = consolidated_fc.copy()
        source = consolidation_info.get('forecast_source', 'unknown')

        # Check if data is already processed (deal_level returns already-processed data)
        if 'vAccountType' in fc_raw.columns and 'mAmount_norm' in fc_raw.columns:
            # Already processed, just use it
            fc_deal_full = fc_raw.copy()
        else:
            # Needs processing (properties_aggregated returns raw aggregated data)
            if 'vcode' in fc_raw.columns and 'Vcode' not in fc_raw.columns:
                fc_raw = fc_raw.rename(columns={'vcode': 'Vcode'})
            fc_deal_full = load_forecast(fc_raw, coa, int(pro_yr_base))

        prop_count = consolidation_info.get('property_count', 0)
        debug_msgs.append(f"Sub-portfolio deal: {prop_count} properties consolidated ({source})")
        st.info(f"📦 Sub-portfolio: {prop_count} properties consolidated from {source}")
    else:
        fc_deal_full = fc[fc["vcode"].astype(str) == str(deal_vcode)].copy()
        debug_msgs.append(f"Not a sub-portfolio deal (InvestmentID: {deal_investment_id})")
else:
    fc_deal_full = fc[fc["vcode"].astype(str) == str(deal_vcode)].copy()
    debug_msgs.append(f"No relationships data loaded - consolidation skipped")

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
loan_sched = pd.DataFrame()
loans = []

# Use consolidated loans for sub-portfolios, or filter by vcode for regular deals
if consolidation_info and consolidation_info.get('is_sub_portfolio', False) and not consolidated_loans.empty:
    mri_loans = load_mri_loans(consolidated_loans)
    loans.extend(build_loans_from_mri_loans(mri_loans))
    debug_msgs.append(f"Loans loaded: {len(loans)} from deal/property level")
elif mri_loans_raw is not None and not mri_loans_raw.empty:
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

# Add historical Contribution and Distribution: Return of Capital events
if 'deal_acct' in dir() and deal_acct is not None and not deal_acct.empty:
    for _, row in deal_acct.iterrows():
        major_type = str(row.get("MajorType", "")).strip().lower()
        type_name = str(row.get("TypeName", "")).strip()
        type_name_lower = type_name.lower()
        amt = float(row.get("Amt", 0))

        # Get the date
        eff_date = row.get("EffectiveDate", None)
        if eff_date is None:
            continue
        if isinstance(eff_date, str):
            try:
                eff_date = pd.to_datetime(eff_date).date()
            except:
                continue
        elif hasattr(eff_date, 'date'):
            eff_date = eff_date.date()
        elif not isinstance(eff_date, date):
            continue

        # Include Contributions (from MajorType)
        if "contrib" in major_type and amt != 0:
            capital_events.append({
                "vcode": str(deal_vcode),
                "event_date": eff_date,
                "event_type": "Contribution",
                "amount": float(abs(amt)),
            })
        # Include Distribution: Return of Capital (from TypeName)
        elif "return of capital" in type_name_lower and amt != 0:
            capital_events.append({
                "vcode": str(deal_vcode),
                "event_date": eff_date,
                "event_type": type_name,  # Use actual TypeName
                "amount": float(abs(amt)),
            })

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
        # Note: Sale proceeds are not added here as separate events because they flow
        # through the capital waterfall and will be captured as "Capital Distribution"
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
# CAPITAL CALLS, CASH MANAGEMENT & WATERFALLS (Computation)
# ============================================================
# Load and build capital calls schedule
capital_calls = []
if capital_calls_raw is not None and not capital_calls_raw.empty:
    try:
        cc_df = load_capital_calls(capital_calls_raw)
        if cc_df is not None and not cc_df.empty:
            capital_calls = build_capital_call_schedule(cc_df, deal_vcode)
            if capital_calls:
                fc_deal_modeled = integrate_capital_calls_with_forecast(
                    fc_deal_modeled, capital_calls
                )
    except Exception as e:
        debug_msgs.append(f"Could not process capital calls: {str(e)}")
        capital_calls = []

# Load beginning cash balance from ISBS
beginning_cash = 0.0
if isbs_raw is not None and not isbs_raw.empty:
    try:
        beginning_cash = load_beginning_cash_balance(isbs_raw, deal_vcode, model_start)
    except Exception as e:
        debug_msgs.append(f"Could not load cash balance: {str(e)}")

# Build cash flow schedule
cash_summary = {}
try:
    fad_monthly = cashflows_monthly_fad(fc_deal_modeled)
    cash_schedule = build_cash_flow_schedule_from_fad(
        fad_monthly=fad_monthly,
        capital_calls=capital_calls,
        beginning_cash=beginning_cash,
        deal_vcode=deal_vcode
    )
    cash_summary = summarize_cash_usage(cash_schedule)
except Exception as e:
    debug_msgs.append(f"Error building cash flow schedule: {str(e)}")
    try:
        fad_monthly_fallback = cashflows_monthly_fad(fc_deal_modeled)
        cash_schedule = fad_monthly_fallback.copy()
        cash_schedule['distributable'] = cash_schedule['fad']
        cash_schedule['beginning_cash'] = beginning_cash
        cash_schedule['ending_cash'] = beginning_cash
    except:
        cash_schedule = pd.DataFrame()

# Prepare waterfall inputs
cf_period_cash = cash_schedule[['event_date', 'distributable']].copy() if not cash_schedule.empty else pd.DataFrame(columns=['event_date', 'cash_available'])
if not cf_period_cash.empty:
    cf_period_cash = cf_period_cash.rename(columns={'distributable': 'cash_available'})

remaining_cash_at_sale, _ = get_sale_period_total_cash(cash_schedule, sale_me) if not cash_schedule.empty else (0, None)

if remaining_cash_at_sale > 0 and not cf_period_cash.empty:
    sale_mask = cf_period_cash['event_date'] == sale_me
    if sale_mask.any():
        cf_period_cash.loc[sale_mask, 'cash_available'] += remaining_cash_at_sale
    else:
        cf_period_cash = pd.concat([
            cf_period_cash,
            pd.DataFrame([{'event_date': sale_me, 'cash_available': remaining_cash_at_sale}])
        ], ignore_index=True).sort_values('event_date')

cap_period_cash = pd.DataFrame(columns=["event_date", "cash_available"])
if cap_events_df is not None and not cap_events_df.empty:
    ce = cap_events_df.copy()
    ce["event_date"] = pd.to_datetime(ce["event_date"]).dt.date
    ce["event_date"] = ce["event_date"].apply(month_end)
    cap_period_cash = ce.groupby("event_date", as_index=False)["amount"].sum().rename(columns={"amount": "cash_available"})

if not cf_period_cash.empty:
    cf_period_cash = cf_period_cash[cf_period_cash["event_date"] <= sale_me].copy()
cap_period_cash = cap_period_cash[cap_period_cash["event_date"] <= sale_me].copy()

# Add sale proceeds to cap_period_cash for capital waterfall distribution
# (not added to capital_events to avoid double-counting with distribution results)
if sale_dbg is not None and sale_dbg.get("Net_Sale_Proceeds", 0) > 0:
    sale_cash_entry = pd.DataFrame([{
        "event_date": sale_me,
        "cash_available": sale_dbg["Net_Sale_Proceeds"]
    }])
    if cap_period_cash.empty:
        cap_period_cash = sale_cash_entry
    else:
        # Check if sale_me already exists in cap_period_cash
        if sale_me in cap_period_cash["event_date"].values:
            cap_period_cash.loc[cap_period_cash["event_date"] == sale_me, "cash_available"] += sale_dbg["Net_Sale_Proceeds"]
        else:
            cap_period_cash = pd.concat([cap_period_cash, sale_cash_entry], ignore_index=True).sort_values("event_date")

# Load waterfalls and run
wf_steps = load_waterfalls(wf)
seed_states = seed_states_from_accounting(acct, inv, wf_steps, deal_vcode)

if capital_calls:
    try:
        seed_states = apply_capital_calls_to_states(capital_calls, seed_states)
    except Exception as e:
        debug_msgs.append(f"Could not apply capital calls to investor states: {str(e)}")

cf_alloc, cf_investors = run_waterfall(wf_steps, deal_vcode, "CF_WF", cf_period_cash, seed_states)
cap_alloc, cap_investors = run_waterfall(wf_steps, deal_vcode, "Cap_WF", cap_period_cash, seed_states)

# ============================================================
# ENHANCE CAPITAL EVENTS WITH CALLS AND DISTRIBUTIONS
# ============================================================
# Add future capital calls to capital events
if capital_calls:
    for call in capital_calls:
        call_date = call.get('call_date')
        if call_date is not None:
            if hasattr(call_date, 'date'):
                call_date = call_date.date()
            elif isinstance(call_date, str):
                call_date = pd.to_datetime(call_date).date()
            capital_events.append({
                "vcode": str(deal_vcode),
                "event_date": call_date,
                "event_type": "Capital Call",
                "amount": float(call.get('amount', 0)),
            })

# Add Capital waterfall distributions to capital events (refi/sale proceeds only)
# Note: We do NOT include CF waterfall distributions here per user requirement
if cap_alloc is not None and not cap_alloc.empty:
    # Group by date and sum allocations
    cap_by_date = cap_alloc.groupby('event_date')['Allocated'].sum().reset_index()
    for _, row in cap_by_date.iterrows():
        if row['Allocated'] > 0:
            evt_date = row['event_date']
            if hasattr(evt_date, 'date'):
                evt_date = evt_date.date()
            capital_events.append({
                "vcode": str(deal_vcode),
                "event_date": evt_date,
                "event_type": "Capital Distribution",
                "amount": float(row['Allocated']),
            })

# Rebuild cap_events_df with all events
cap_events_df = pd.DataFrame(capital_events)
if not cap_events_df.empty:
    cap_events_df["Year"] = cap_events_df["event_date"].apply(lambda d: pd.Timestamp(d).year).astype("Int64")
    # Sort by date
    cap_events_df = cap_events_df.sort_values("event_date").reset_index(drop=True)

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beginning Cash", f"${cash_summary.get('beginning_cash', 0):,.0f}")
    col2.metric("Total CapEx Paid", f"${cash_summary.get('total_capex_paid', 0):,.0f}")
    col3.metric("Total Distributable", f"${cash_summary.get('total_distributable', 0):,.0f}")
    col4.metric("Ending Cash", f"${cash_summary.get('ending_cash', 0):,.0f}")

    if cash_summary.get('min_cash_balance', 0) < 0:
        st.warning(f"⚠️ Cash balance went negative (lowest: ${cash_summary.get('min_cash_balance', 0):,.0f}). Consider additional capital calls or reducing CapEx.")

# ============================================================
# BALANCE SHEET COMPARISON
# ============================================================
if isbs_raw is not None and not isbs_raw.empty:
    with st.expander("📊 Balance Sheet Comparison"):
        # Define account classifications
        BS_ACCOUNTS = {
            'ASSETS': {
                'Current Assets': {
                    'Cash': ['1010', '1012'],
                    'Misc Current Assets': ['1040', '1070'],
                },
                'Noncurrent Assets': {
                    'Accounts Receivable': ['1020', '1025', '1030'],
                    'Lender Held Reserves & Escrows': ['1145', '1092', '1091'],
                    'Other Reserves & Escrows': ['1014', '1080', '1090', '1100', '1120', '1130', '1140'],
                    'Prepaid': ['1050', '1060', '1075', '1151'],
                    'Fixed Assets': ['1240', '1250', '1260', '1270', '1280', '1282', '1275'],
                    'Depreciation & Amortization': ['1230', '1290'],
                    'Other Assets': ['1150', '1224', '1220'],
                },
            },
            'LIABILITIES': {
                'Current Liabilities': {
                    'Accounts Payable': ['2010', '2012', '2015', '2020'],
                    'Accrued Interest Payable': ['2060'],
                    'Accrued Taxes Payable': ['2110'],
                    'Security Deposits': ['2090'],
                    'Prepaid Revenues': ['2080'],
                    'Other Accrued Liabilities': ['2115', '2120', '2124', '2130'],
                },
                'Noncurrent Liabilities': {
                    'Mortgages and Loans': ['2150', '2152', '2210'],
                    'Misc Long Term Liabilities': ['2300', '2310'],
                    'Deferred Developer/AM Fee': ['2230'],
                    'Notes Payable to GP': ['2280'],
                    'Notes Payable to LP': ['2290'],
                },
            },
            'EQUITY': {
                'Equity': {
                    'Equity': ['2520', '2530', '2534', '2536', '2540'],
                    'Partner Equity': ['2525'],
                    'PSC Pref Equity': ['2526'],
                    'Distributions-2527': ['2527'],
                    'Distributions-2528': ['2528'],
                    'Net Income': ['2550'],
                },
            },
        }

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

        if not isbs_bs.empty and 'dtEntry' in isbs_bs.columns:
            # Parse dates
            try:
                isbs_bs['dtEntry_parsed'] = pd.to_datetime(isbs_bs['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
            except:
                isbs_bs['dtEntry_parsed'] = pd.to_datetime(isbs_bs['dtEntry'], errors='coerce')
            null_dates = isbs_bs['dtEntry_parsed'].isna()
            if null_dates.any():
                isbs_bs.loc[null_dates, 'dtEntry_parsed'] = pd.to_datetime(isbs_bs.loc[null_dates, 'dtEntry'], errors='coerce')

            # Get available periods (only from actual reported balance sheets)
            available_periods = sorted(isbs_bs['dtEntry_parsed'].dropna().unique())

            if len(available_periods) >= 1:
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

                # Period selectors
                col_left, col_right = st.columns(2)
                with col_left:
                    period1 = st.selectbox("Prior Period", period_options, index=period_options.index(prior_year_end) if prior_year_end in period_options else 0, key="bs_period1")
                with col_right:
                    period2 = st.selectbox("Current Period", period_options, index=most_recent_idx, key="bs_period2")

                # Get balances for each period
                def get_period_balances(isbs_df, period_str, accounts_dict):
                    """Get account balances for a specific period"""
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

                balances1 = get_period_balances(isbs_bs, period1, BS_ACCOUNTS)
                balances2 = get_period_balances(isbs_bs, period2, BS_ACCOUNTS)

                # Build display table
                rows = []

                def add_row(label, val1, val2, indent=0, is_header=False, is_total=False):
                    variance = val2 - val1 if val1 is not None and val2 is not None else None
                    var_pct = (variance / abs(val1) * 100) if val1 and val1 != 0 and variance is not None else None

                    prefix = "  " * indent
                    if is_header:
                        label_display = f"**{prefix}{label}**"
                    elif is_total:
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
                    st.warning(f"⚠️ Balance sheet does not balance. Assets + Liabilities + Equity = ${balance_check:,.0f}")
            else:
                st.info("No balance sheet periods available for this deal.")
        else:
            st.info("No balance sheet data available for this deal.")

# ============================================================
# INCOME STATEMENT COMPARISON
# ============================================================
if isbs_raw is not None and not isbs_raw.empty:
    with st.expander("📊 Income Statement Comparison"):
        # Define income statement account classifications
        IS_ACCOUNTS = {
            'REVENUES': {
                'Rental Income': ['4010', '4012'],
                'Commercial': ['4020', '4041'],
                'Abated Apartments': ['4045'],
                'Vacancy': ['4040', '4043', '4030', '4042'],
                'RUBS': ['4070'],
                'RET': ['4091'],
                'INS': ['4092'],
                'CAM': ['4090', '4097', '4093', '4094', '4096', '4095'],
                'Other Income': ['4063', '4060', '4061', '4062', '4080', '4065'],
            },
            'EXPENSES': {
                'Real Estate Taxes': ['5090'],
                'Property & Liability Insurance': ['5110', '5114'],
                'Salary & Benefits': ['5018', '5010', '5016', '5012', '5014'],
                'Utilities': ['5051', '5053', '5050', '5052', '5054', '5055'],
                'Repairs & Maintenance': ['5060', '5067', '5063', '5069', '5061', '5064', '5065', '5068', '5070', '5066'],
                'Administrative': ['5020', '5022', '5021', '5023', '5025', '5026', '5080'],
                'Marketing & Advertising': ['5045'],
                'Legal & Professional': ['5087', '5085'],
                'Management Fee': ['5040'],
                'Other Expenses': ['5096', '5095', '5091', '5100'],
            },
            'DEBT_SERVICE': {
                'Interest': ['5190'],
                'Principal': ['2145', '2150', '2152', '2154', '2156'],
            },
            'OTHER_BTL': {
                'Interest Income': ['4050'],
                'Other (Income) Expenses': ['5220', '5210', '5195', '7065'],
                'Capital Expenditures': ['7050'],
                'Partnership Expenses': ['5120', '5130'],
                'Extraordinary Expenses': ['5400'],
            },
        }

        # Prepare ISBS data for the deal
        isbs_is = isbs_raw.copy()
        isbs_is.columns = [str(c).strip() for c in isbs_is.columns]

        # Filter for deal
        if 'vcode' in isbs_is.columns:
            isbs_is['vcode'] = isbs_is['vcode'].astype(str).str.strip().str.lower()
            isbs_is = isbs_is[isbs_is['vcode'] == str(deal_vcode).strip().lower()]

        # Parse dates
        if not isbs_is.empty and 'dtEntry' in isbs_is.columns:
            try:
                isbs_is['dtEntry_parsed'] = pd.to_datetime(isbs_is['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
            except:
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

            if len(actual_periods) >= 1:
                # Determine most recent actual period
                most_recent_actual = pd.Timestamp(actual_periods[-1])
                current_year = most_recent_actual.year
                current_month = most_recent_actual.month

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

                # Helper function to get cumulative balances for a specific date
                def get_cumulative_balances(data_df, as_of_date, accounts_dict):
                    """Get cumulative account balances as of a specific date from Actuals"""
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

                # Helper function to sum budget amounts over a date range
                def get_budget_sum(data_df, start_date, end_date, accounts_dict):
                    """Sum budget amounts between start_date (exclusive) and end_date (inclusive)"""
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

                # Helper to subtract two balance dicts
                def subtract_balances(bal1, bal2, accounts_dict):
                    """bal1 - bal2"""
                    result = {}
                    for section, categories in accounts_dict.items():
                        result[section] = {}
                        for category in categories.keys():
                            v1 = bal1.get(section, {}).get(category, 0)
                            v2 = bal2.get(section, {}).get(category, 0)
                            result[section][category] = v1 - v2
                    return result

                # Helper to add two balance dicts
                def add_balances(bal1, bal2, accounts_dict):
                    """bal1 + bal2"""
                    result = {}
                    for section, categories in accounts_dict.items():
                        result[section] = {}
                        for category in categories.keys():
                            v1 = bal1.get(section, {}).get(category, 0)
                            v2 = bal2.get(section, {}).get(category, 0)
                            result[section][category] = v1 + v2
                    return result

                # Helper to get valuation data for a period
                def get_valuation_sum(fc_df, start_date, end_date, accounts_dict):
                    """Get valuation amounts from forecast_feed"""
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

                # Function to calculate IS amounts based on period type and source
                def calculate_is_amounts(period_type, source, ref_date, year, accounts_dict):
                    """Calculate income statement amounts based on period type and source"""
                    ref_date = pd.Timestamp(ref_date)

                    if source == "Actual":
                        if period_type == "TTM (Trailing Twelve Months)":
                            # TTM = current month + prior Dec - same month last year
                            current_bal = get_cumulative_balances(actual_data, ref_date, accounts_dict)
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
                                dec_bal = get_cumulative_balances(actual_data, dec_prior, accounts_dict)
                                ly_bal = get_cumulative_balances(actual_data, same_month_ly, accounts_dict)
                                # TTM = current + dec_prior - same_month_ly
                                temp = add_balances(current_bal, dec_bal, accounts_dict)
                                return subtract_balances(temp, ly_bal, accounts_dict)
                            elif dec_prior:
                                dec_bal = get_cumulative_balances(actual_data, dec_prior, accounts_dict)
                                return add_balances(current_bal, dec_bal, accounts_dict)
                            else:
                                return current_bal

                        elif period_type == "YTD (Year to Date)":
                            # YTD = current month cumulative (since it resets each year)
                            return get_cumulative_balances(actual_data, ref_date, accounts_dict)

                        elif period_type == "Full Year":
                            # Full Year = December of that year
                            dec_date = None
                            for p in actual_periods:
                                p_ts = pd.Timestamp(p)
                                if p_ts.year == year and p_ts.month == 12:
                                    dec_date = p_ts
                                    break
                            if dec_date:
                                return get_cumulative_balances(actual_data, dec_date, accounts_dict)
                            return {}

                        elif period_type == "Current Year Estimate":
                            # YTD Actual + Budget for remainder
                            ytd_bal = get_cumulative_balances(actual_data, ref_date, accounts_dict)
                            # Budget from ref_date to Dec 31
                            dec_end = pd.Timestamp(f"{ref_date.year}-12-31")
                            budget_remainder = get_budget_sum(budget_data, ref_date, dec_end, accounts_dict)
                            return add_balances(ytd_bal, budget_remainder, accounts_dict)

                        else:  # Custom Month
                            return get_cumulative_balances(actual_data, ref_date, accounts_dict)

                    elif source == "Budget":
                        if period_type == "TTM (Trailing Twelve Months)":
                            # Sum 12 months of budget ending at ref_date
                            start = ref_date - pd.DateOffset(months=12)
                            return get_budget_sum(budget_data, start, ref_date, accounts_dict)

                        elif period_type == "YTD (Year to Date)":
                            # Sum budget from Jan 1 to ref_date
                            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
                            return get_budget_sum(budget_data, jan1, ref_date, accounts_dict)

                        elif period_type == "Full Year":
                            # Sum budget for full year
                            jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
                            dec31 = pd.Timestamp(f"{year}-12-31")
                            return get_budget_sum(budget_data, jan1, dec31, accounts_dict)

                        elif period_type == "Current Year Estimate":
                            # Full year budget
                            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
                            dec31 = pd.Timestamp(f"{ref_date.year}-12-31")
                            return get_budget_sum(budget_data, jan1, dec31, accounts_dict)

                        else:  # Custom Month
                            # Single month budget
                            prior_month = ref_date - pd.DateOffset(months=1)
                            return get_budget_sum(budget_data, prior_month, ref_date, accounts_dict)

                    elif source == "Underwriting":
                        if period_type == "TTM (Trailing Twelve Months)":
                            start = ref_date - pd.DateOffset(months=12)
                            return get_budget_sum(uw_data, start, ref_date, accounts_dict)

                        elif period_type == "YTD (Year to Date)":
                            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
                            return get_budget_sum(uw_data, jan1, ref_date, accounts_dict)

                        elif period_type == "Full Year":
                            jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
                            dec31 = pd.Timestamp(f"{year}-12-31")
                            return get_budget_sum(uw_data, jan1, dec31, accounts_dict)

                        elif period_type == "Current Year Estimate":
                            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
                            dec31 = pd.Timestamp(f"{ref_date.year}-12-31")
                            return get_budget_sum(uw_data, jan1, dec31, accounts_dict)

                        else:  # Custom Month
                            prior_month = ref_date - pd.DateOffset(months=1)
                            return get_budget_sum(uw_data, prior_month, ref_date, accounts_dict)

                    elif source == "Valuation":
                        if period_type == "TTM (Trailing Twelve Months)":
                            start = ref_date - pd.DateOffset(months=12)
                            return get_valuation_sum(fc_deal_modeled, start.date(), ref_date.date(), accounts_dict)

                        elif period_type == "YTD (Year to Date)":
                            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
                            return get_valuation_sum(fc_deal_modeled, jan1.date(), ref_date.date(), accounts_dict)

                        elif period_type == "Full Year":
                            jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
                            dec31 = pd.Timestamp(f"{year}-12-31")
                            return get_valuation_sum(fc_deal_modeled, jan1.date(), dec31.date(), accounts_dict)

                        elif period_type == "Current Year Estimate":
                            jan1 = pd.Timestamp(f"{ref_date.year}-01-01") - pd.DateOffset(days=1)
                            dec31 = pd.Timestamp(f"{ref_date.year}-12-31")
                            return get_valuation_sum(fc_deal_modeled, jan1.date(), dec31.date(), accounts_dict)

                        else:  # Custom Month
                            prior_month = ref_date - pd.DateOffset(months=1)
                            return get_valuation_sum(fc_deal_modeled, prior_month.date(), ref_date.date(), accounts_dict)

                    return {}

                # UI for period selection
                st.markdown("**Configure Comparison:**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("*Column 1*")
                    period_type1 = st.selectbox("Period Type", period_types, index=0, key="is_period_type1")
                    source1 = st.selectbox("Source", source_options, index=0, key="is_source1")

                    # Year/date selector based on period type
                    available_years1 = sorted(set(pd.Timestamp(p).year for p in actual_periods), reverse=True)
                    if period_type1 in ["TTM (Trailing Twelve Months)", "YTD (Year to Date)", "Custom Month", "Current Year Estimate"]:
                        period_options1 = [pd.Timestamp(p).strftime('%Y-%m-%d') for p in actual_periods]
                        ref_date1 = st.selectbox("As of Date", period_options1, index=len(period_options1)-1, key="is_ref_date1")
                        year1 = pd.Timestamp(ref_date1).year
                    else:  # Full Year
                        year1 = st.selectbox("Year", available_years1, index=0, key="is_year1")
                        # Find December of selected year or latest month
                        ref_date1 = None
                        for p in reversed(actual_periods):
                            if pd.Timestamp(p).year == year1:
                                ref_date1 = pd.Timestamp(p).strftime('%Y-%m-%d')
                                break
                        if not ref_date1:
                            ref_date1 = pd.Timestamp(actual_periods[-1]).strftime('%Y-%m-%d')

                with col2:
                    st.markdown("*Column 2*")
                    period_type2 = st.selectbox("Period Type", period_types, index=0, key="is_period_type2")
                    source2 = st.selectbox("Source", source_options, index=0, key="is_source2")

                    available_years2 = sorted(set(pd.Timestamp(p).year for p in actual_periods), reverse=True)
                    if period_type2 in ["TTM (Trailing Twelve Months)", "YTD (Year to Date)", "Custom Month", "Current Year Estimate"]:
                        period_options2 = [pd.Timestamp(p).strftime('%Y-%m-%d') for p in actual_periods]
                        # Default to prior year same month if available
                        default_idx2 = len(period_options2) - 1
                        most_recent_ts = pd.Timestamp(actual_periods[-1])
                        for i, p in enumerate(period_options2):
                            p_ts = pd.Timestamp(p)
                            if p_ts.year == most_recent_ts.year - 1 and p_ts.month == most_recent_ts.month:
                                default_idx2 = i
                                break
                        ref_date2 = st.selectbox("As of Date", period_options2, index=default_idx2, key="is_ref_date2")
                        year2 = pd.Timestamp(ref_date2).year
                    else:  # Full Year
                        year2 = st.selectbox("Year", available_years2, index=min(1, len(available_years2)-1), key="is_year2")
                        ref_date2 = None
                        for p in reversed(actual_periods):
                            if pd.Timestamp(p).year == year2:
                                ref_date2 = pd.Timestamp(p).strftime('%Y-%m-%d')
                                break
                        if not ref_date2:
                            ref_date2 = pd.Timestamp(actual_periods[-1]).strftime('%Y-%m-%d')

                # Calculate amounts for both columns
                amounts1 = calculate_is_amounts(period_type1, source1, ref_date1, year1, IS_ACCOUNTS)
                amounts2 = calculate_is_amounts(period_type2, source2, ref_date2, year2, IS_ACCOUNTS)

                # Build display table
                is_rows = []

                def add_is_row(label, val1, val2, indent=0, is_header=False, is_total=False, is_calc=False):
                    variance = val2 - val1 if val1 is not None and val2 is not None else None
                    var_pct = (variance / abs(val1) * 100) if val1 and val1 != 0 and variance is not None else None

                    prefix = "  " * indent
                    if is_header or is_total or is_calc:
                        label_display = f"**{prefix}{label}**"
                    else:
                        label_display = f"{prefix}{label}"

                    col1_label = f"{source1} {period_type1.split(' ')[0]} {ref_date1[:7] if 'TTM' in period_type1 or 'YTD' in period_type1 else year1}"
                    col2_label = f"{source2} {period_type2.split(' ')[0]} {ref_date2[:7] if 'TTM' in period_type2 or 'YTD' in period_type2 else year2}"

                    is_rows.append({
                        'Account': label_display,
                        col1_label: f"${val1:,.0f}" if val1 is not None else "",
                        col2_label: f"${val2:,.0f}" if val2 is not None else "",
                        'Variance $': f"${variance:,.0f}" if variance is not None else "",
                        'Variance %': f"{var_pct:.1f}%" if var_pct is not None else "",
                    })

                # Column labels for consistent naming
                col1_label = f"{source1} {period_type1.split(' ')[0]} {ref_date1[:7] if 'TTM' in period_type1 or 'YTD' in period_type1 else year1}"
                col2_label = f"{source2} {period_type2.split(' ')[0]} {ref_date2[:7] if 'TTM' in period_type2 or 'YTD' in period_type2 else year2}"

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

            else:
                st.info("No income statement periods available for this deal.")
        else:
            st.info("No income statement data available for this deal.")

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
    state = cf_investors.get(partner) or cap_investors.get(partner)
    
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
        last_date = None

        # Helper to check for leap year
        def days_in_year(d):
            year = d.year
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                return 366.0
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

            # Total Due = Current Due + Remaining Accrual from prior period
            total_due = current_due + remaining_accrual

            # Process events on this date
            day_events = event_lookup.get(d, [])

            if day_events:
                for iorder, typename, amt, etype in day_events:
                    pref_paid = 0.0

                    if etype == 'contribution':
                        # Contribution increases investment balance
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
                            'Prior Accrual': remaining_accrual - current_due if remaining_accrual > current_due else 0,
                            'Total Due': total_due,
                            'Amount Paid': 0.0,
                            'Remaining Accrual': remaining_accrual,
                        })

                    elif etype == 'cash_distribution':
                        # Cash distribution payment application order:
                        # 1. Current Due (this period's accrual)
                        # 2. Remaining Accrual (prior period's unpaid pref)
                        # 3. Compounded Pref (prior years' compounded)
                        # Excess beyond all pref is just profit (no equity reduction)

                        pref_paid = amt
                        payment_remaining = amt

                        # 1. Apply to current due
                        current_due_paid = min(payment_remaining, current_due)
                        payment_remaining -= current_due_paid
                        new_current_due = current_due - current_due_paid

                        # 2. Apply to remaining accrual (prior period)
                        prior_accrual_paid = min(payment_remaining, remaining_accrual)
                        payment_remaining -= prior_accrual_paid
                        new_remaining = remaining_accrual - prior_accrual_paid

                        # 3. Apply to compounded pref
                        compounded_paid = min(payment_remaining, compounded_pref)
                        payment_remaining -= compounded_paid
                        compounded_pref -= compounded_paid

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
                            'Prior Accrual': remaining_accrual,
                            'Total Due': total_due,
                            'Amount Paid': pref_paid,
                            'Remaining Accrual': new_remaining,
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
                            'Prior Accrual': remaining_accrual,
                            'Total Due': total_due,
                            'Amount Paid': 0.0,  # Capital return doesn't pay pref
                            'Remaining Accrual': new_remaining,
                        })
                        remaining_accrual = new_remaining

                    elif etype == 'profit_share':
                        # Profit share after pref is cleared - no pref reduction, no equity change
                        rows.append({
                            'Date': d,
                            'Type': typename,
                            'Amt': amt,
                            'Equity Balance': investment_balance,
                            'Compounded Pref': compounded_pref,
                            'Total Inv+Comp': investment_balance + compounded_pref,
                            'Days': days,
                            'Current Due': current_due,
                            'Prior Accrual': remaining_accrual,
                            'Total Due': total_due,
                            'Amount Paid': 0.0,
                            'Remaining Accrual': remaining_accrual,
                        })

                    # Reset for subsequent events on same day
                    days = 0
                    current_due = 0.0
                    total_due = remaining_accrual  # Next event's total due is just remaining

            else:
                # Month-end accrual checkpoint (no event)
                # Update remaining accrual with current due
                remaining_accrual = total_due

                rows.append({
                    'Date': d,
                    'Type': 'Accrual',
                    'Amt': 0.0,
                    'Equity Balance': investment_balance,
                    'Compounded Pref': compounded_pref,
                    'Total Inv+Comp': investment_balance + compounded_pref,
                    'Days': days,
                    'Current Due': current_due,
                    'Prior Accrual': remaining_accrual - current_due,
                    'Total Due': total_due,
                    'Amount Paid': 0.0,
                    'Remaining Accrual': remaining_accrual,
                })

            # Year-end compounding (12/31): Compounded Pref = Current Due + Remaining Accrual
            if d.month == 12 and d.day == 31 and remaining_accrual > 0:
                compounded_pref += remaining_accrual
                rows.append({
                    'Date': d,
                    'Type': 'Year-End Compound',
                    'Amt': remaining_accrual,
                    'Equity Balance': investment_balance,
                    'Compounded Pref': compounded_pref,
                    'Total Inv+Comp': investment_balance + compounded_pref,
                    'Days': 0,
                    'Current Due': 0.0,
                    'Prior Accrual': remaining_accrual,
                    'Total Due': remaining_accrual,
                    'Amount Paid': 0.0,
                    'Remaining Accrual': 0.0,
                })
                remaining_accrual = 0.0

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
        pref_rate = pref_rates.get(partner, 0.0)

        with st.expander(f"📊 {partner} Preferred Return Schedule (Rate: {pref_rate:.2%})"):
            if pref_rate == 0:
                st.warning(f"No pref rate found for {partner}")
            else:
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
            unrealized = stt.capital_outstanding + stt.pref_unpaid_compounded + stt.pref_accrued_current_year
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

# ============================================================
# TENANT ROSTER (Commercial Properties Only)
# ============================================================
if tenants_raw is not None and not tenants_raw.empty:
    # Filter tenants for current deal (Code column = vcode)
    tenants_raw['Code'] = tenants_raw['Code'].astype(str).str.strip()
    deal_tenants = tenants_raw[tenants_raw['Code'] == str(deal_vcode)].copy()

    if not deal_tenants.empty:
        with st.expander("Tenant Roster"):
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
            from datetime import datetime
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
            display_cols = ['Tenant Name', 'SF Leased', 'Lease Start', 'Lease End', 'Annual Rent', 'RPSF', '% of GLA', '% of ABR']
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
            col1, col2, col3 = st.columns(3)
            col1.metric("Occupied SF", f"{total_occupied_sf:,.0f}")
            col2.metric("Occupancy %", f"{occupancy_pct:.1%}")
            col3.metric("Wtd Avg RPSF", f"${weighted_rpsf:,.2f}")

            # Legend
            st.caption("⚠️ Yellow highlight indicates lease expiring within 2 years of current date")

# ============================================================
# ONE PAGER INVESTOR REPORT
# ============================================================
if isbs_raw is not None and not isbs_raw.empty:
    with st.expander("One Pager Investor Report"):
        from one_pager_ui import render_one_pager_section
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

# ============================================================
# OWNERSHIP TREE ANALYSIS
# ============================================================
st.divider()
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

else:
    st.info("Upload MRI_IA_Relationship.csv to analyze multi-tiered ownership structures")

st.success("Report generated successfully!")
