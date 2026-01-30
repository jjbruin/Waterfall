"""
waterfall.py
Waterfall calculation engine with corrected logic

KEY PRINCIPLES:
- CF Waterfall: Operating cash, does NOT reduce capital outstanding
- Capital Waterfall: Refi/sale proceeds, DOES reduce capital outstanding
- Pref accrues daily (Act/365), compounds annually on 12/31
- Historical accounting seeds initial state, does NOT modify pref
- Paired steps (same vAmtType) distribute together with Tag logic
- Tag partner receives proportional share based on lead partner's distribution
"""

from datetime import date
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.optimize import brentq

from models import InvestorState
from metrics import xirr, investor_metrics
from loaders import build_investmentid_to_vcode, normalize_accounting_feed


# ============================================================
# PREF ACCRUAL LOGIC
# ============================================================

def accrue_pref_to_date(ist: InvestorState, asof: date, annual_rate: float):
    """
    Accrue preferred return to a specific date
    
    Logic:
    - Daily accrual on base = capital_outstanding + pref_unpaid_compounded
    - On 12/31: compound current_year accrual into pref_unpaid_compounded
    - Act/365 day count
    
    Args:
        ist: InvestorState to update
        asof: Date to accrue through
        annual_rate: Pref rate as decimal (e.g., 0.08 for 8%)
    """
    if ist.last_accrual_date is None:
        ist.last_accrual_date = asof
        return

    if asof <= ist.last_accrual_date:
        return

    # Step through year boundaries
    cur = ist.last_accrual_date
    while cur < asof:
        year_end = date(cur.year, 12, 31)
        next_stop = min(asof, year_end)
        days = (next_stop - cur).days
        
        if days > 0 and annual_rate > 0:
            base = max(0.0, ist.capital_outstanding + ist.pref_unpaid_compounded)
            ist.pref_accrued_current_year += base * annual_rate * (days / 365.0)

        # Move to next day after year end if we hit year end
        if next_stop == year_end and next_stop < asof:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = next_stop
            break

    ist.last_accrual_date = asof


def compound_if_year_end(istates: Dict[str, InvestorState], period_date: date):
    """
    On 12/31, compound current-year accrued pref into unpaid compounded balance
    
    This happens AFTER distributions are processed for that period
    """
    if period_date.month == 12 and period_date.day == 31:
        for s in istates.values():
            if s.pref_accrued_current_year > 0:
                s.pref_unpaid_compounded += s.pref_accrued_current_year
                s.pref_accrued_current_year = 0.0


# ============================================================
# CASHFLOW RECORDING
# ============================================================

def apply_distribution(ist: InvestorState, d: date, amt: float):
    """Record a distribution (positive cashflow from investor perspective)"""
    if amt == 0:
        return
    ist.cashflows.append((d, float(amt)))


def apply_contribution(ist: InvestorState, d: date, amt: float):
    """Record a contribution (negative cashflow from investor perspective)"""
    if amt == 0:
        return
    ist.cashflows.append((d, -abs(float(amt))))
    ist.capital_outstanding += abs(float(amt))


# ============================================================
# WATERFALL PAYMENT STEPS
# ============================================================

def pay_pref(ist: InvestorState, d: date, available: float) -> float:
    """
    Pay preferred return: compounded unpaid first, then current-year accrued
    
    Returns amount paid
    """
    pay = 0.0
    if available <= 0:
        return 0.0

    # Pay compounded unpaid first
    if ist.pref_unpaid_compounded > 0:
        x = min(available, ist.pref_unpaid_compounded)
        ist.pref_unpaid_compounded -= x
        available -= x
        pay += x

    if available <= 0:
        if pay:
            apply_distribution(ist, d, pay)
        return pay

    # Pay current-year accrued next
    if ist.pref_accrued_current_year > 0:
        x = min(available, ist.pref_accrued_current_year)
        ist.pref_accrued_current_year -= x
        available -= x
        pay += x

    if pay:
        apply_distribution(ist, d, pay)
    return pay


def pay_initial_capital(ist: InvestorState, d: date, available: float) -> float:
    """Return initial capital until outstanding capital is 0"""
    if available <= 0 or ist.capital_outstanding <= 0:
        return 0.0
    x = min(available, ist.capital_outstanding)
    ist.capital_outstanding -= x
    apply_distribution(ist, d, x)
    return x


def irr_needed_distribution(ist: InvestorState, d: date, target_irr: float, max_cash: float) -> float:
    """
    Calculate distribution needed to achieve target IRR
    
    Uses binary search to find distribution amount that results in target IRR
    """
    if max_cash <= 0:
        return 0.0
    
    # Must have at least one negative cashflow (investment)
    if not ist.cashflows or min(a for _, a in ist.cashflows) >= 0:
        return 0.0

    def irr_of(x):
        cfs = ist.cashflows + [(d, float(x))]
        r = xirr(cfs)
        return r if r is not None else -999.0

    r0 = irr_of(0.0)
    if r0 >= target_irr:
        return 0.0

    r1 = irr_of(max_cash)
    if r1 < target_irr:
        # Even all cash doesn't reach target; take all
        return max_cash

    # Root find x such that irr(x) = target
    def f(x):
        rr = irr_of(x)
        return rr - target_irr

    try:
        x = float(brentq(f, 0.0, max_cash, xtol=0.01, maxiter=50))
        return max(0.0, min(max_cash, x))
    except Exception:
        return 0.0


def pay_default_interest(ist: InvestorState, d: date, available: float, 
                         default_balance: float, default_rate: float) -> float:
    """
    Pay interest accrued on default contributions
    
    Args:
        ist: InvestorState
        d: Distribution date
        available: Cash available for this step
        default_balance: Outstanding default balance (from mAmount)
        default_rate: Interest rate on defaults (from nPercent_dec)
    
    Returns:
        Amount paid
    """
    if available <= 0 or default_balance <= 0:
        return 0.0
    
    # Calculate interest owed (simplified - could be enhanced with actual accrual tracking)
    # For now, assume interest = balance * rate (annual)
    interest_owed = default_balance * default_rate
    
    x = min(available, interest_owed)
    if x > 0:
        apply_distribution(ist, d, x)
    return x


def pay_default_principal(ist: InvestorState, d: date, available: float, 
                          default_balance: float) -> float:
    """
    Pay principal on default contributions
    
    Args:
        ist: InvestorState
        d: Distribution date
        available: Cash available for this step
        default_balance: Outstanding default balance (from mAmount)
    
    Returns:
        Amount paid
    """
    if available <= 0 or default_balance <= 0:
        return 0.0
    
    x = min(available, default_balance)
    if x > 0:
        apply_distribution(ist, d, x)
    return x


# ============================================================
# WATERFALL PERIOD PROCESSING (WITH PAIRED STEP LOGIC)
# ============================================================

def run_waterfall_period(
    steps: pd.DataFrame,
    istates: Dict[str, InvestorState],
    period_date: date,
    cash_available: float,
    pref_rates: Dict[str, float],
    is_capital_waterfall: bool = False,
) -> Tuple[float, List[dict]]:
    """
    Process one period's waterfall distributions with paired step logic
    
    Paired steps are identified by matching vAmtType. Within a pair:
    - Non-Tag steps execute first and determine actual distribution
    - Tag steps receive proportional share: lead_dist / lead_fx * tag_fx
    
    Args:
        steps: Waterfall step definitions (already filtered to this deal/waterfall)
        istates: Investor states (modified in place)
        period_date: Distribution date
        cash_available: Cash to distribute
        pref_rates: Pref rates by PropCode (updated during processing)
        is_capital_waterfall: True for Cap_WF, False for CF_WF
    
    Returns:
        (remaining_cash, allocation_rows)
    """
    alloc_rows = []
    
    # Accrue pref to this date for all investors
    for pc, stt in istates.items():
        rate = float(pref_rates.get(pc, 0.0))
        accrue_pref_to_date(stt, period_date, rate)
    
    remaining = float(cash_available)
    
    # Group steps by vAmtType to identify paired steps
    steps = steps.sort_values("iOrder")
    
    # Track which steps we've processed
    processed_orders = set()
    
    for _, step in steps.iterrows():
        order = int(step["iOrder"])
        
        # Skip if already processed (as part of a pair)
        if order in processed_orders:
            continue
        
        amt_type = str(step.get("vAmtType", "")).strip()
        pc = str(step["PropCode"])
        state = str(step["vState"]).strip()
        fx = float(step["FXRate"])
        rate = float(step["nPercent_dec"])
        m_amount = float(step.get("mAmount", 0.0) or 0.0)
        
        # Ensure investor state exists
        if pc not in istates:
            istates[pc] = InvestorState(propcode=pc)
        
        stt = istates[pc]
        
        # Find paired steps (same vAmtType)
        if amt_type:
            paired_steps = steps[steps["vAmtType"].astype(str).str.strip() == amt_type].copy()
        else:
            paired_steps = steps[steps["iOrder"] == order].copy()
        
        # Separate Tag steps from lead steps
        lead_steps = paired_steps[paired_steps["vState"].astype(str).str.strip() != "Tag"]
        tag_steps = paired_steps[paired_steps["vState"].astype(str).str.strip() == "Tag"]
        
        # Calculate total FXRate for this level to determine available cash
        total_fx = paired_steps["FXRate"].sum()
        level_available = remaining  # Full remaining available at this level
        
        # Track distributions at this level for Tag calculation
        level_distributions = {}  # PropCode -> (allocated, fx_rate)
        
        # Process lead steps first (non-Tag)
        for _, lead in lead_steps.iterrows():
            lead_order = int(lead["iOrder"])
            if lead_order in processed_orders:
                continue
                
            lead_pc = str(lead["PropCode"])
            lead_state = str(lead["vState"]).strip()
            lead_fx = float(lead["FXRate"])
            lead_rate = float(lead["nPercent_dec"])
            lead_m_amount = float(lead.get("mAmount", 0.0) or 0.0)
            
            if lead_pc not in istates:
                istates[lead_pc] = InvestorState(propcode=lead_pc)
            
            lead_stt = istates[lead_pc]
            allocated = 0.0
            
            # Maximum this step can take (FXRate share of remaining)
            step_max = remaining * lead_fx if lead_fx > 0 else remaining
            
            # Process based on vState
            if lead_state == "Pref":
                pref_rates[lead_pc] = lead_rate
                allocated = pay_pref(lead_stt, period_date, step_max)
            
            elif lead_state == "Initial":
                if is_capital_waterfall:
                    allocated = pay_initial_capital(lead_stt, period_date, step_max)
            
            elif lead_state == "Share":
                allocated = step_max
                if allocated > 0:
                    apply_distribution(lead_stt, period_date, allocated)
            
            elif lead_state == "IRR":
                target_irr = lead_rate
                allocated = irr_needed_distribution(lead_stt, period_date, target_irr, step_max)
                if allocated > 0:
                    apply_distribution(lead_stt, period_date, allocated)
            
            elif lead_state in ("Def&Int", "DefInt"):
                # Combined default interest and principal
                # Interest first, then principal
                int_paid = pay_default_interest(lead_stt, period_date, step_max, lead_m_amount, lead_rate)
                prin_avail = step_max - int_paid
                prin_paid = pay_default_principal(lead_stt, period_date, prin_avail, lead_m_amount - int_paid)
                allocated = int_paid + prin_paid
            
            elif lead_state == "Def_Int":
                # Default interest only
                allocated = pay_default_interest(lead_stt, period_date, step_max, lead_m_amount, lead_rate)
            
            elif lead_state == "Default":
                # Default principal only
                allocated = pay_default_principal(lead_stt, period_date, step_max, lead_m_amount)
            
            elif lead_state == "Add":
                # Additional capital pref
                allocated = pay_pref(lead_stt, period_date, step_max)
            
            elif lead_state == "Amt":
                # Fixed amount
                allocated = min(step_max, abs(lead_m_amount)) if lead_m_amount != 0 else 0.0
                if allocated > 0:
                    apply_distribution(lead_stt, period_date, allocated)
            
            else:
                # Unknown state - no allocation
                allocated = 0.0
            
            # Track for Tag calculation
            level_distributions[lead_pc] = (allocated, lead_fx)
            
            # Record allocation
            alloc_rows.append({
                "event_date": period_date,
                "Year": period_date.year,
                "iOrder": lead_order,
                "vAmtType": amt_type,
                "PropCode": lead_pc,
                "vtranstype": lead.get("vtranstype", ""),
                "vState": lead_state,
                "FXRate": lead_fx,
                "nPercent": lead_rate,
                "Allocated": float(allocated),
                "RemainingAfter": float(remaining - allocated),
            })
            
            remaining -= allocated
            processed_orders.add(lead_order)
            
            if remaining <= 1e-9:
                remaining = 0.0
                break
        
        # Process Tag steps - they receive proportional share based on lead distributions
        for _, tag in tag_steps.iterrows():
            tag_order = int(tag["iOrder"])
            if tag_order in processed_orders:
                continue
            
            tag_pc = str(tag["PropCode"])
            tag_fx = float(tag["FXRate"])
            tag_rate = float(tag["nPercent_dec"])
            
            if tag_pc not in istates:
                istates[tag_pc] = InvestorState(propcode=tag_pc)
            
            tag_stt = istates[tag_pc]
            
            # Calculate Tag allocation based on lead partner's distribution
            # Tag gets: lead_allocated / lead_fx * tag_fx
            allocated = 0.0
            
            for lead_pc, (lead_alloc, lead_fx) in level_distributions.items():
                if lead_alloc > 0 and lead_fx > 0:
                    # Tag receives proportional share
                    tag_share = (lead_alloc / lead_fx) * tag_fx
                    # But limited by remaining cash
                    tag_share = min(tag_share, remaining)
                    allocated += tag_share
                    break  # Use first lead partner's distribution as reference
            
            if allocated > 0:
                apply_distribution(tag_stt, period_date, allocated)
            
            # Record allocation
            alloc_rows.append({
                "event_date": period_date,
                "Year": period_date.year,
                "iOrder": tag_order,
                "vAmtType": amt_type,
                "PropCode": tag_pc,
                "vtranstype": tag.get("vtranstype", ""),
                "vState": "Tag",
                "FXRate": tag_fx,
                "nPercent": tag_rate,
                "Allocated": float(allocated),
                "RemainingAfter": float(remaining - allocated),
            })
            
            remaining -= allocated
            processed_orders.add(tag_order)
            
            if remaining <= 1e-9:
                remaining = 0.0
                break
        
        # Mark all steps in this pair as processed
        for _, s in paired_steps.iterrows():
            processed_orders.add(int(s["iOrder"]))
        
        if remaining <= 1e-9:
            remaining = 0.0
            break
    
    # After all allocations, if 12/31 compound remaining pref
    compound_if_year_end(istates, period_date)
    
    return remaining, alloc_rows


# ============================================================
# FULL WATERFALL EXECUTION
# ============================================================

def run_waterfall(
    wf_steps: pd.DataFrame,
    vcode: str,
    wf_name: str,
    period_cash: pd.DataFrame,
    initial_states: Optional[Dict[str, InvestorState]] = None,
    show_structure_when_no_cash: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete waterfall for all periods
    
    Args:
        wf_steps: All waterfall step definitions
        vcode: Deal code
        wf_name: "CF_WF" or "Cap_WF"
        period_cash: DataFrame with columns [event_date, cash_available]
        initial_states: Optional pre-seeded investor states
        show_structure_when_no_cash: If True, create zero-allocation rows even when no cash
    
    Returns:
        (allocations_df, investor_summary_df)
    """
    # Filter steps to this deal and waterfall type
    steps = wf_steps[
        (wf_steps["vcode"].astype(str) == str(vcode)) & 
        (wf_steps["vmisc"] == wf_name)
    ].copy()
    steps = steps.sort_values("iOrder")
    
    if steps.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # If no cash periods, but we want to show structure, create a single zero-cash period
    if period_cash.empty or period_cash["cash_available"].sum() == 0:
        if show_structure_when_no_cash:
            period_cash = pd.DataFrame([{
                "event_date": date.today(),
                "cash_available": 0.0
            }])
        else:
            return pd.DataFrame(), pd.DataFrame()
    
    istates = initial_states if initial_states is not None else {}
    pref_rates = {}
    
    # Determine if this is capital waterfall
    is_cap_wf = (wf_name == "Cap_WF")
    
    all_rows = []
    for _, r in period_cash.sort_values("event_date").iterrows():
        d = r["event_date"]
        cash = float(r["cash_available"])
        
        _rem, rows = run_waterfall_period(
            steps, istates, d, cash, pref_rates, 
            is_capital_waterfall=is_cap_wf
        )
        all_rows.extend(rows)
    
    alloc_df = pd.DataFrame(all_rows)
    
    # Investor summary with metrics
    inv_rows = []
    for pc, stt in istates.items():
        # Calculate unrealized NAV
        unrealized = stt.capital_outstanding + stt.pref_unpaid_compounded + stt.pref_accrued_current_year
        
        # Get metrics
        as_of = period_cash['event_date'].max()
        metrics = investor_metrics(stt, as_of_date=as_of, unrealized_nav=unrealized)
        
        inv_rows.append({
            "PropCode": pc,
            "CapitalOutstanding": stt.capital_outstanding,
            "UnpaidPrefCompounded": stt.pref_unpaid_compounded,
            "AccruedPrefCurrentYear": stt.pref_accrued_current_year,
            "UnrealizedNAV": unrealized,
            "IRR": metrics['IRR'],
            "ROE": metrics['ROE'],
            "MOIC": metrics['MOIC'],
            "TotalContributions": metrics['TotalContributions'],
            "TotalDistributions": metrics['TotalDistributions'],
        })
    
    inv_sum = pd.DataFrame(inv_rows)
    if not inv_sum.empty:
        inv_sum = inv_sum.sort_values("PropCode")
    
    return alloc_df, inv_sum


# ============================================================
# HISTORICAL ACCOUNTING SEEDING (CORRECTED)
# ============================================================

def pref_rates_from_waterfall_steps(wf_steps: pd.DataFrame, vcode: str) -> dict:
    """Extract pref rates from waterfall Pref steps"""
    s = wf_steps[wf_steps["vcode"].astype(str) == str(vcode)].copy()
    if s.empty:
        return {}
    pref = s[s["vState"].astype(str).str.strip() == "Pref"].copy()
    if pref.empty:
        return {}
    pref = pref.sort_values(["vmisc", "iOrder"])
    out = {}
    for pc, grp in pref.groupby("PropCode"):
        out[str(pc)] = float(grp["nPercent_dec"].iloc[0])
    return out


def seed_states_from_accounting(
    acct_raw: pd.DataFrame,
    inv_map: pd.DataFrame,
    wf_steps: pd.DataFrame,
    target_vcode: str,
) -> Dict[str, InvestorState]:
    """
    Build InvestorState per PropCode from historical accounting
    
    KEY PRINCIPLE: Accounting shows RESULTS of past waterfalls, not mechanics
    We use it to:
    1. Seed cashflow history for XIRR
    2. Track capital balances
    3. DO NOT modify pref balances (forward model handles that)
    
    Args:
        acct_raw: accounting_feed.csv DataFrame
        inv_map: investment_map.csv DataFrame
        wf_steps: Waterfall definitions
        target_vcode: Deal to process
    
    Returns:
        Dict of InvestorState by PropCode
    """
    acct = normalize_accounting_feed(acct_raw)
    inv_to_vcode = build_investmentid_to_vcode(inv_map)
    
    # Filter to target deal
    acct["vcode"] = acct["InvestmentID"].map(inv_to_vcode)
    acct = acct[acct["vcode"].astype(str) == str(target_vcode)].copy()
    
    # Build states
    states: Dict[str, InvestorState] = {}
    acct = acct.sort_values(["EffectiveDate", "InvestorID"]).copy()
    
    for _, r in acct.iterrows():
        pc = str(r["InvestorID"])  # InvestorID == PropCode
        d = r["EffectiveDate"]
        if pd.isna(d):
            continue
        
        if pc not in states:
            states[pc] = InvestorState(propcode=pc)
            if r["is_contribution"]:
                states[pc].last_accrual_date = d
        
        stt = states[pc]
        amt = float(r["Amt"])
        is_capital = bool(r["is_capital"])
        is_contribution = bool(r["is_contribution"])
        
        # CONTRIBUTIONS: Increase capital and record cashflow
        if is_contribution:
            cf = amt if amt < 0 else -abs(amt)  # Ensure negative
            stt.cashflows.append((d, cf))
            stt.capital_outstanding += abs(cf)
            
            if stt.last_accrual_date is None:
                stt.last_accrual_date = d
        
        # DISTRIBUTIONS: Record cashflow, reduce capital if capital event
        elif r["is_distribution"]:
            cf = amt if amt > 0 else abs(amt)  # Ensure positive
            stt.cashflows.append((d, cf))
            
            # ONLY capital distributions reduce capital_outstanding
            if is_capital:
                capital_return = min(cf, max(0.0, stt.capital_outstanding))
                stt.capital_outstanding -= capital_return
            
            # CF distributions (is_capital=N) do NOT touch capital_outstanding
    
    return states


# ============================================================
# UTILITY: PIVOT WATERFALL BY YEAR
# ============================================================

def pivot_waterfall_by_year(alloc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot waterfall allocations to show years as columns
    
    Returns DataFrame with:
    - Rows: iOrder, vAmtType, PropCode, vtranstype, vState, FXRate, nPercent
    - Columns: Years
    - Values: Allocated amounts
    """
    if alloc_df.empty:
        return pd.DataFrame()
    
    # Create row identifier
    id_cols = ['iOrder', 'vAmtType', 'PropCode', 'vtranstype', 'vState', 'FXRate', 'nPercent']
    available_cols = [c for c in id_cols if c in alloc_df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    # Pivot
    pivot = alloc_df.pivot_table(
        index=available_cols,
        columns='Year',
        values='Allocated',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    return pivot
