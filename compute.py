"""
compute.py
Deal-level computation engine - separated from display for caching.

All expensive per-deal computations are here so app.py can cache results
in session_state and avoid recomputing on every Streamlit rerun.
"""

import pandas as pd
import copy
from datetime import date
from typing import Optional, Dict, List

from config import (INTEREST_ACCTS, PRINCIPAL_ACCTS, SELLING_COST_RATE,
                    GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS, EXPENSE_ACCTS)
from utils import month_end, as_date
from metrics import investor_metrics, xirr, calculate_roe
from models import InvestorState
from loaders import (load_coa, load_forecast, load_mri_loans,
                     build_investmentid_to_vcode, normalize_accounting_feed,
                     load_waterfalls)
from loans import build_loans_from_mri_loans, amortize_monthly_schedule, total_loan_balance_at
from waterfall import run_waterfall, seed_states_from_accounting
from planned_loans import (size_planned_second_mortgage, planned_loan_as_loan_object,
                           twelve_month_noi_after_date, projected_cap_rate_at_date)
from reporting import cashflows_monthly_fad
from capital_calls import (load_capital_calls, build_capital_call_schedule,
                           integrate_capital_calls_with_forecast)
from cash_management import (load_beginning_cash_balance, build_cash_flow_schedule_from_fad,
                             summarize_cash_usage, get_sale_period_total_cash)
from consolidation import build_consolidated_forecast, get_property_vcodes_for_deal


def _horizon_end_date(start_yr: int, horizon_yrs: int) -> date:
    y = int(start_yr) + int(horizon_yrs) - 1
    return date(y, 12, 31)


def get_deal_capitalization(acct, inv, wf, mri_val, mri_loans, deal_vcode, property_vcodes=None):
    """Calculate deal capitalization from accounting_feed.

    Equity = Contributions + Return of Capital distributions.
    InvestorID links to PropCode in waterfalls.
    InvestmentID links to vcode via investment_map.

    For portfolio deals, property_vcodes should include all sub-property vcodes
    so that debt from those properties is aggregated.
    """
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
                all_vcodes = [str(deal_vcode)]
                if property_vcodes:
                    all_vcodes.extend([str(v) for v in property_vcodes])
                deal_loans = mri_loans_copy[mri_loans_copy['vCode'].isin(all_vcodes)]
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


def get_cached_deal_result(vcode, start_year, horizon_years, pro_yr_base, **kwargs):
    """Check shared multi-deal cache; compute and store on miss.

    All consumers (Deal Analysis, Dashboard, Reports) should call this
    instead of compute_deal_analysis() directly so that results computed
    anywhere are reused everywhere.

    The full result dict is stored — including cf_alloc, cap_alloc, and
    partner_results — so upstream waterfall analysis can pull actual
    allocations without re-running waterfalls.
    """
    import streamlit as st
    cache = st.session_state.setdefault('_deal_results', {})
    key = f"{vcode}|{start_year}|{horizon_years}|{pro_yr_base}"
    if key not in cache:
        st.toast(f"Computing {vcode}...")
        cache[key] = compute_deal_analysis(
            deal_vcode=vcode,
            start_year=start_year,
            horizon_years=horizon_years,
            pro_yr_base=pro_yr_base,
            **kwargs,
        )
    return cache[key]


def compute_deal_analysis(
    deal_vcode, deal_investment_id, sale_date_raw,
    inv, wf, acct, fc, coa,
    mri_loans_raw, mri_supp, mri_val,
    relationships_raw, capital_calls_raw, isbs_raw,
    start_year, horizon_years, pro_yr_base,
):
    """
    Compute all deal-level analysis results.

    Returns a dict with all computed results, or a dict with 'error' key
    if forecast data is missing (cap_data is still included for header display).
    """
    debug_msgs = []

    # --- Capitalization ---
    prop_vcodes_for_cap = get_property_vcodes_for_deal(deal_vcode, inv)
    cap_data = get_deal_capitalization(acct, inv, wf, mri_val, mri_loans_raw, deal_vcode, prop_vcodes_for_cap)

    # --- Forecast consolidation ---
    rels_for_consol = relationships_raw if relationships_raw is not None else pd.DataFrame()

    consolidated_fc, consolidated_loans, consolidation_info = build_consolidated_forecast(
        deal_investment_id=deal_investment_id,
        deals=inv,
        relationships=rels_for_consol,
        forecasts=fc,
        loans=mri_loans_raw if mri_loans_raw is not None else pd.DataFrame(),
        debug=True
    )

    sub_portfolio_msg = None
    if consolidation_info.get('is_sub_portfolio', False):
        fc_raw_local = consolidated_fc.copy()
        source = consolidation_info.get('forecast_source', 'unknown')

        if 'vAccountType' in fc_raw_local.columns and 'mAmount_norm' in fc_raw_local.columns:
            fc_deal_full = fc_raw_local.copy()
        else:
            if 'vcode' in fc_raw_local.columns and 'Vcode' not in fc_raw_local.columns:
                fc_raw_local = fc_raw_local.rename(columns={'vcode': 'Vcode'})
            fc_deal_full = load_forecast(fc_raw_local, coa, int(pro_yr_base))

        prop_count = consolidation_info.get('property_count', 0)
        debug_msgs.append(f"Sub-portfolio deal: {prop_count} properties consolidated ({source})")
        sub_portfolio_msg = f"📦 Sub-portfolio: {prop_count} properties consolidated from {source}"
    else:
        fc_deal_full = fc[fc["vcode"].astype(str) == str(deal_vcode)].copy()
        debug_msgs.append(f"Not a sub-portfolio deal (InvestmentID: {deal_investment_id})")

    if fc_deal_full.empty:
        return {
            'error': f"No forecast rows for vcode {deal_vcode}",
            'cap_data': cap_data,
            'prop_vcodes_for_cap': prop_vcodes_for_cap,
        }

    model_start = min(fc_deal_full["event_date"])
    model_end_full = max(fc_deal_full["event_date"])

    # --- Sale date ---
    if sale_date_raw is None or (isinstance(sale_date_raw, float) and pd.isna(sale_date_raw)) or str(sale_date_raw).strip() == "":
        sale_date = month_end(_horizon_end_date(int(start_year), int(horizon_years)))
    else:
        sale_date = month_end(as_date(sale_date_raw))

    if sale_date < month_end(model_start):
        sale_date = month_end(model_start)

    sale_me = month_end(sale_date)

    # --- Build loan schedules ---
    loan_sched = pd.DataFrame()
    loans = []

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

        supp_vcodes = [str(deal_vcode)]
        if consolidation_info and consolidation_info.get('is_sub_portfolio', False):
            supp_vcodes.extend(consolidation_info.get('property_vcodes', []))

        matching_supp = ms[ms["vCode"].isin(supp_vcodes)]
        if not matching_supp.empty:
            if mri_val is None or mri_val.empty:
                debug_msgs.append("MRI_Supp present, but MRI_Val missing – cannot size planned loan.")
            else:
                for _, supp_row in matching_supp.iterrows():
                    try:
                        row_amt, row_dbg = size_planned_second_mortgage(inv, fc_deal_full, supp_row, mri_val)
                        row_orig_date = month_end(as_date(supp_row["Orig_Date"]))

                        if row_amt > 0:
                            loans.append(planned_loan_as_loan_object(str(supp_row["vCode"]), supp_row, row_amt))
                            if planned_dbg is None:
                                planned_dbg = row_dbg
                                planned_new_loan_amt = row_amt
                                planned_orig_date = row_orig_date
                    except Exception as e:
                        debug_msgs.append(f"Planned loan sizing failed for {supp_row['vCode']}: {e}")

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

    # --- Replace forecast debt service with modeled ---
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
                "mAmount_norm": -abs(r["interest"]),
            })
            add_rows.append({
                "vcode": r["vcode"],
                "event_date": r["event_date"],
                "vAccount": r["vAccount_principal"],
                "mAmount_norm": -abs(r["principal"]),
            })
        if add_rows:
            fc_deal_modeled = pd.concat([fc_deal_modeled, pd.DataFrame(add_rows)], ignore_index=True)

    fc_deal_modeled["Year"] = fc_deal_modeled["event_date"].apply(lambda d: pd.Timestamp(d).year).astype("Int64")

    # --- Capital events ---
    capital_events = []
    sale_dbg = None

    # Note: deal_acct is not in scope here (same as original code where this block never executed)
    # Historical capital events from accounting are handled by seed_states_from_accounting

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

    # --- Capital calls, cash management ---
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
        except Exception:
            cash_schedule = pd.DataFrame()

    # --- Prepare waterfall inputs ---
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

    # Add sale proceeds to cap_period_cash
    if sale_dbg is not None and sale_dbg.get("Net_Sale_Proceeds", 0) > 0:
        sale_cash_entry = pd.DataFrame([{
            "event_date": sale_me,
            "cash_available": sale_dbg["Net_Sale_Proceeds"]
        }])
        if cap_period_cash.empty:
            cap_period_cash = sale_cash_entry
        else:
            if sale_me in cap_period_cash["event_date"].values:
                cap_period_cash.loc[cap_period_cash["event_date"] == sale_me, "cash_available"] += sale_dbg["Net_Sale_Proceeds"]
            else:
                cap_period_cash = pd.concat([cap_period_cash, sale_cash_entry], ignore_index=True).sort_values("event_date")

    # --- Run waterfalls ---
    wf_steps = load_waterfalls(wf)
    seed_states = seed_states_from_accounting(acct, inv, wf_steps, deal_vcode)

    cf_alloc, cf_investors = run_waterfall(wf_steps, deal_vcode, "CF_WF", cf_period_cash, copy.deepcopy(seed_states), capital_calls=capital_calls)
    cap_alloc, cap_investors = run_waterfall(wf_steps, deal_vcode, "Cap_WF", cap_period_cash, copy.deepcopy(seed_states))

    # --- Enhance capital events with calls and distributions ---
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

    if cap_alloc is not None and not cap_alloc.empty:
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
        cap_events_df = cap_events_df.sort_values("event_date").reset_index(drop=True)

    result = {
        'cap_data': cap_data,
        'prop_vcodes_for_cap': prop_vcodes_for_cap,
        'consolidation_info': consolidation_info,
        'sub_portfolio_msg': sub_portfolio_msg,
        'fc_deal_modeled': fc_deal_modeled,
        'fc_deal_display': fc_deal_display,
        'loan_sched': loan_sched,
        'loans': loans,
        'cf_alloc': cf_alloc,
        'cf_investors': cf_investors,
        'cap_alloc': cap_alloc,
        'cap_investors': cap_investors,
        'cap_events_df': cap_events_df,
        'cash_schedule': cash_schedule,
        'cash_summary': cash_summary,
        'sale_dbg': sale_dbg,
        'sale_me': sale_me,
        'debug_msgs': debug_msgs,
        'capital_calls': capital_calls,
        'seed_states': seed_states,
        'wf_steps': wf_steps,
        'model_start': model_start,
        'model_end_full': model_end_full,
        'beginning_cash': beginning_cash,
    }

    # --- Build partner results (single source of truth for all metrics) ---
    partner_results, deal_summary = build_partner_results(
        cf_investors=cf_investors,
        cap_investors=cap_investors,
        seed_states=seed_states,
        cf_alloc=cf_alloc,
        cap_alloc=cap_alloc,
        sale_me=sale_me,
    )
    result['partner_results'] = partner_results
    result['deal_summary'] = deal_summary
    return result


def build_partner_results(cf_investors, cap_investors, seed_states, cf_alloc, cap_alloc, sale_me):
    """Build partner-level and deal-level metrics from waterfall results.

    This is the single source of truth for IRR, ROE, MOIC, cashflow details,
    and pref/capital balances.  Called once per deal by compute_deal_analysis();
    all UI screens read from the returned dicts instead of reimplementing the
    combine-CF-and-Cap logic.

    Returns:
        (partner_results, deal_summary) where partner_results is a list of
        PartnerResult dicts and deal_summary is a dict of deal-level aggregates.
    """
    all_partners = set()
    if cf_alloc is not None and not cf_alloc.empty:
        all_partners.update(cf_alloc['PropCode'].unique())
    if cap_alloc is not None and not cap_alloc.empty:
        all_partners.update(cap_alloc['PropCode'].unique())

    if not all_partners:
        return [], {}

    partner_results = []
    total_contrib = 0.0
    total_cf_dist = 0.0
    total_cap_dist = 0.0

    all_irr_cashflows = []
    all_cashflows = []
    all_cf_distributions = []

    for partner in sorted(all_partners):
        # --- Distribution totals from allocation DataFrames ---
        cf_dist = 0.0
        if cf_alloc is not None and not cf_alloc.empty:
            cf_dist = cf_alloc[cf_alloc['PropCode'] == partner]['Allocated'].sum()

        cap_dist = 0.0
        if cap_alloc is not None and not cap_alloc.empty:
            cap_dist = cap_alloc[cap_alloc['PropCode'] == partner]['Allocated'].sum()

        cf_state = cf_investors.get(partner)
        cap_state = cap_investors.get(partner)
        state = cf_state or cap_state
        if not state:
            continue

        # --- Combined cashflows (seed once + CF new + Cap new) ---
        seed_st = seed_states.get(partner)
        seed_len = len(seed_st.cashflows) if seed_st else 0

        combined_cfs = []
        if seed_st and seed_st.cashflows:
            combined_cfs.extend(seed_st.cashflows)
        if cf_state and len(cf_state.cashflows) > seed_len:
            combined_cfs.extend(cf_state.cashflows[seed_len:])
        if cap_state and len(cap_state.cashflows) > seed_len:
            combined_cfs.extend(cap_state.cashflows[seed_len:])

        # --- Unrealized NAV ---
        unrealized = 0.0
        if cap_state:
            unrealized = cap_state.total_capital_outstanding + cap_state.total_pref_balance
        elif cf_state:
            unrealized = cf_state.total_capital_outstanding + cf_state.total_pref_balance

        # --- Investor metrics (IRR, ROE, MOIC) ---
        combined_state = InvestorState(propcode=partner)
        combined_state.cashflows = combined_cfs
        combined_state.cf_distributions = (
            cf_state.cf_distributions if cf_state and hasattr(cf_state, 'cf_distributions') else []
        )
        metrics = investor_metrics(combined_state, sale_me, unrealized_nav=unrealized)

        contrib = metrics.get('TotalContributions', 0.0)
        is_pref_equity = not partner.upper().startswith("OP")

        # --- Cashflow detail rows (for XIRR Cash Flows table) ---
        cashflow_details = []
        if seed_st and seed_st.cashflows:
            for cf_date, cf_amount in seed_st.cashflows:
                desc = "Contribution" if cf_amount < 0 else "Historical Distribution"
                cashflow_details.append({
                    "Date": cf_date, "Description": desc,
                    "Partner": partner, "is_pref": is_pref_equity,
                    "Amount": cf_amount,
                })
        if cf_state and len(cf_state.cashflows) > seed_len:
            for cf_date, cf_amount in cf_state.cashflows[seed_len:]:
                desc = "Capital Call" if cf_amount < 0 else "CF Distribution"
                cashflow_details.append({
                    "Date": cf_date, "Description": desc,
                    "Partner": partner, "is_pref": is_pref_equity,
                    "Amount": cf_amount,
                })
        if cap_state and len(cap_state.cashflows) > seed_len:
            for cf_date, cf_amount in cap_state.cashflows[seed_len:]:
                desc = "Capital Distribution" if cf_amount > 0 else "Cap WF Adjustment"
                cashflow_details.append({
                    "Date": cf_date, "Description": desc,
                    "Partner": partner, "is_pref": is_pref_equity,
                    "Amount": cf_amount,
                })
        if unrealized > 0 and sale_me:
            cashflow_details.append({
                "Date": sale_me, "Description": "Unrealized NAV",
                "Partner": partner, "is_pref": is_pref_equity,
                "Amount": unrealized,
            })

        # --- Pref balances ---
        pref_unpaid_compounded = state.pref_unpaid_compounded + state.add_pref_unpaid_compounded
        pref_accrued_current_year = state.pref_accrued_current_year + state.add_pref_accrued_current_year

        partner_results.append({
            'partner': partner,
            'is_pref_equity': is_pref_equity,
            'contributions': contrib,
            'cf_distributions': cf_dist,
            'cap_distributions': cap_dist,
            'total_distributions': cf_dist + cap_dist,
            'irr': metrics.get('IRR'),
            'roe': metrics.get('ROE', 0.0),
            'moic': metrics.get('MOIC', 0.0),
            'unrealized_nav': unrealized,
            'capital_outstanding': cap_state.total_capital_outstanding if cap_state else state.total_capital_outstanding,
            'pref_unpaid_compounded': pref_unpaid_compounded,
            'pref_accrued_current_year': pref_accrued_current_year,
            'combined_cashflows': combined_cfs,
            'cf_only_distributions': combined_state.cf_distributions,
            'cashflow_details': cashflow_details,
        })

        total_contrib += contrib
        total_cf_dist += cf_dist
        total_cap_dist += cap_dist

        # --- Accumulate deal-level cashflow streams ---
        # Seed (once)
        if seed_st and seed_st.cashflows:
            all_irr_cashflows.extend(seed_st.cashflows)
            all_cashflows.extend(seed_st.cashflows)
        # CF waterfall new
        if cf_state and len(cf_state.cashflows) > seed_len:
            all_irr_cashflows.extend(cf_state.cashflows[seed_len:])
            all_cashflows.extend(cf_state.cashflows[seed_len:])
        # Cap waterfall new
        if cap_state and len(cap_state.cashflows) > seed_len:
            all_irr_cashflows.extend(cap_state.cashflows[seed_len:])
        # Terminal unrealized
        if unrealized > 0 and sale_me:
            all_irr_cashflows.append((sale_me, unrealized))
        # CF distributions for ROE
        if cf_state and hasattr(cf_state, 'cf_distributions') and cf_state.cf_distributions:
            all_cf_distributions.extend(cf_state.cf_distributions)

    if not partner_results:
        return [], {}

    # --- Deal-level metrics ---
    deal_irr = xirr(all_irr_cashflows) if all_irr_cashflows else None

    contributions_list = [(d, a) for d, a in all_cashflows if a < 0]
    inception_date = min(d for d, _ in contributions_list) if contributions_list else None
    if inception_date and all_cashflows and sale_me:
        deal_roe = calculate_roe(all_cashflows, all_cf_distributions, inception_date, sale_me)
    else:
        deal_roe = 0.0

    total_dist = total_cf_dist + total_cap_dist
    deal_moic = total_dist / total_contrib if total_contrib > 0 else 0.0

    deal_summary = {
        'deal_irr': deal_irr,
        'deal_roe': deal_roe,
        'deal_moic': deal_moic,
        'total_contributions': total_contrib,
        'total_cf_distributions': total_cf_dist,
        'total_cap_distributions': total_cap_dist,
        'total_distributions': total_dist,
        'all_combined_cashflows': all_cashflows,
        'all_irr_cashflows': all_irr_cashflows,
    }

    return partner_results, deal_summary
