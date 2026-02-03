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

    # Step through year boundaries, compounding at each 12/31
    cur = ist.last_accrual_date
    while cur < asof:
        year_end = date(cur.year, 12, 31)
        next_stop = min(asof, year_end)
        days = (next_stop - cur).days

        if days > 0 and annual_rate > 0:
            base = max(0.0, ist.capital_outstanding + ist.pref_unpaid_compounded)
            ist.pref_accrued_current_year += base * annual_rate * (days / 365.0)

        # At year end, compound current-year accrual into unpaid balance
        # so the next year's accrual base includes prior unpaid pref
        if next_stop == year_end and next_stop < asof:
            if ist.pref_accrued_current_year > 0:
                ist.pref_unpaid_compounded += ist.pref_accrued_current_year
                ist.pref_accrued_current_year = 0.0
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = next_stop
            break

    ist.last_accrual_date = asof


def accrue_add_pref_to_date(ist: InvestorState, asof: date, annual_rate: float):
    """
    Accrue preferred return on additional/special capital to a specific date.

    Mirrors accrue_pref_to_date but accrues on add_capital_outstanding +
    add_pref_unpaid_compounded.  Does NOT update last_accrual_date (the main
    accrual function handles that).
    """
    if ist.last_accrual_date is None or asof <= ist.last_accrual_date:
        return

    cur = ist.last_accrual_date
    while cur < asof:
        year_end = date(cur.year, 12, 31)
        next_stop = min(asof, year_end)
        days = (next_stop - cur).days

        if days > 0 and annual_rate > 0:
            base = max(0.0, ist.add_capital_outstanding + ist.add_pref_unpaid_compounded)
            ist.add_pref_accrued_current_year += base * annual_rate * (days / 365.0)

        # Compound at year-end so next year accrues on the correct base
        if next_stop == year_end and next_stop < asof:
            if ist.add_pref_accrued_current_year > 0:
                ist.add_pref_unpaid_compounded += ist.add_pref_accrued_current_year
                ist.add_pref_accrued_current_year = 0.0
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = next_stop
            break


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
            if s.add_pref_accrued_current_year > 0:
                s.add_pref_unpaid_compounded += s.add_pref_accrued_current_year
                s.add_pref_accrued_current_year = 0.0


# ============================================================
# CASHFLOW RECORDING
# ============================================================

def apply_distribution(ist: InvestorState, d: date, amt: float, is_cf_waterfall: bool = False):
    """Record a distribution (positive cashflow from investor perspective)

    Args:
        ist: Investor state to update
        d: Distribution date
        amt: Distribution amount
        is_cf_waterfall: If True, also record in cf_distributions for ROE calculation
    """
    if amt == 0:
        return
    ist.cashflows.append((d, float(amt)))
    if is_cf_waterfall:
        ist.cf_distributions.append((d, float(amt)))


def apply_contribution(ist: InvestorState, d: date, amt: float):
    """Record a contribution (negative cashflow from investor perspective)"""
    if amt == 0:
        return
    ist.cashflows.append((d, -abs(float(amt))))
    ist.capital_outstanding += abs(float(amt))


# ============================================================
# WATERFALL PAYMENT STEPS
# ============================================================

def pay_pref(ist: InvestorState, d: date, available: float, is_cf_waterfall: bool = False) -> float:
    """
    Pay preferred return: compounded unpaid first, then current-year accrued

    Args:
        is_cf_waterfall: If True, track in cf_distributions for ROE calculation

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
            apply_distribution(ist, d, pay, is_cf_waterfall)
        return pay

    # Pay current-year accrued next
    if ist.pref_accrued_current_year > 0:
        x = min(available, ist.pref_accrued_current_year)
        ist.pref_accrued_current_year -= x
        available -= x
        pay += x

    if pay:
        apply_distribution(ist, d, pay, is_cf_waterfall)
    return pay


def pay_initial_capital(ist: InvestorState, d: date, available: float, is_cf_waterfall: bool = False) -> float:
    """Return initial capital until outstanding capital is 0

    Note: Capital returns are NOT CF distributions (excluded from ROE numerator)
    """
    if available <= 0 or ist.capital_outstanding <= 0:
        return 0.0
    x = min(available, ist.capital_outstanding)
    ist.capital_outstanding -= x
    # Capital return is never a CF distribution - it's return of principal
    apply_distribution(ist, d, x, is_cf_waterfall=False)
    return x


def pay_add_pref(ist: InvestorState, d: date, available: float, is_cf_waterfall: bool = False) -> float:
    """
    Pay preferred return on additional/special capital.

    Mirrors pay_pref but draws from add_pref_unpaid_compounded then
    add_pref_accrued_current_year.

    Returns amount paid.
    """
    pay = 0.0
    if available <= 0:
        return 0.0

    if ist.add_pref_unpaid_compounded > 0:
        x = min(available, ist.add_pref_unpaid_compounded)
        ist.add_pref_unpaid_compounded -= x
        available -= x
        pay += x

    if available <= 0:
        if pay:
            apply_distribution(ist, d, pay, is_cf_waterfall)
        return pay

    if ist.add_pref_accrued_current_year > 0:
        x = min(available, ist.add_pref_accrued_current_year)
        ist.add_pref_accrued_current_year -= x
        available -= x
        pay += x

    if pay:
        apply_distribution(ist, d, pay, is_cf_waterfall)
    return pay


def pay_add_capital(ist: InvestorState, d: date, available: float, is_cf_waterfall: bool = False) -> float:
    """
    Return additional/special capital until add_capital_outstanding is 0.

    Mirrors pay_initial_capital but reduces add_capital_outstanding.
    Capital returns are NOT CF distributions.
    """
    if available <= 0 or ist.add_capital_outstanding <= 0:
        return 0.0
    x = min(available, ist.add_capital_outstanding)
    ist.add_capital_outstanding -= x
    apply_distribution(ist, d, x, is_cf_waterfall=False)
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
                         default_balance: float, default_rate: float,
                         is_cf_waterfall: bool = False) -> float:
    """
    Pay interest accrued on default contributions

    Args:
        ist: InvestorState
        d: Distribution date
        available: Cash available for this step
        default_balance: Outstanding default balance (from mAmount)
        default_rate: Interest rate on defaults (from nPercent_dec)
        is_cf_waterfall: If True, track in cf_distributions for ROE calculation

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
        # Interest is income, track as CF distribution
        apply_distribution(ist, d, x, is_cf_waterfall)
    return x


def pay_default_principal(ist: InvestorState, d: date, available: float,
                          default_balance: float, is_cf_waterfall: bool = False) -> float:
    """
    Pay principal on default contributions

    Args:
        ist: InvestorState
        d: Distribution date
        available: Cash available for this step
        default_balance: Outstanding default balance (from mAmount)
        is_cf_waterfall: Ignored - principal is never a CF distribution

    Returns:
        Amount paid
    """
    if available <= 0 or default_balance <= 0:
        return 0.0

    x = min(available, default_balance)
    if x > 0:
        # Principal return is never a CF distribution
        apply_distribution(ist, d, x, is_cf_waterfall=False)
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
    add_pref_rates: Dict[str, float] = None,
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
        add_pref_rates: Pref rates for additional/special capital by PropCode

    Returns:
        (remaining_cash, allocation_rows)
    """
    if add_pref_rates is None:
        add_pref_rates = {}
    alloc_rows = []

    # Accrue pref to this date for all investors
    for pc, stt in istates.items():
        rate = float(pref_rates.get(pc, 0.0))
        accrue_pref_to_date(stt, period_date, rate)
        add_rate = float(add_pref_rates.get(pc, 0.0))
        if add_rate > 0:
            accrue_add_pref_to_date(stt, period_date, add_rate)
    
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
            
            # CF waterfall distributions should be tracked for ROE calculation
            is_cf_wf = not is_capital_waterfall

            # Process based on vState
            if lead_state == "Pref":
                pref_rates[lead_pc] = lead_rate
                allocated = pay_pref(lead_stt, period_date, step_max, is_cf_wf)

            elif lead_state == "Initial":
                if is_capital_waterfall:
                    # Capital return - never a CF distribution
                    allocated = pay_initial_capital(lead_stt, period_date, step_max, is_cf_wf)

            elif lead_state == "Share":
                allocated = step_max
                if allocated > 0:
                    apply_distribution(lead_stt, period_date, allocated, is_cf_wf)

            elif lead_state == "IRR":
                target_irr = lead_rate
                allocated = irr_needed_distribution(lead_stt, period_date, target_irr, step_max)
                if allocated > 0:
                    apply_distribution(lead_stt, period_date, allocated, is_cf_wf)

            elif lead_state in ("Def&Int", "DefInt"):
                # Combined default interest and principal
                # Interest first, then principal
                int_paid = pay_default_interest(lead_stt, period_date, step_max, lead_m_amount, lead_rate, is_cf_wf)
                prin_avail = step_max - int_paid
                prin_paid = pay_default_principal(lead_stt, period_date, prin_avail, lead_m_amount - int_paid, is_cf_wf)
                allocated = int_paid + prin_paid

            elif lead_state == "Def_Int":
                # Default interest only
                allocated = pay_default_interest(lead_stt, period_date, step_max, lead_m_amount, lead_rate, is_cf_wf)

            elif lead_state == "Default":
                # Default principal only
                allocated = pay_default_principal(lead_stt, period_date, step_max, lead_m_amount, is_cf_wf)

            elif lead_state == "Add":
                # Additional/special capital steps — route by vtranstype
                lead_vtranstype = str(lead.get("vtranstype", "")).strip()
                if "pref" in lead_vtranstype.lower():
                    # Pref-type: pay from add-pref pool, record rate
                    add_pref_rates[lead_pc] = lead_rate
                    allocated = pay_add_pref(lead_stt, period_date, step_max, is_cf_wf)
                elif lead_vtranstype == "Operating Capital" and lead_m_amount > 0:
                    # Operating Capital with capped balance
                    cap = lead_m_amount
                    payable = min(step_max, max(0.0, cap))
                    allocated = pay_add_capital(lead_stt, period_date, payable, is_cf_wf)
                else:
                    # Capital return (Additional Capital, Prorata Special Capital, etc.)
                    if is_capital_waterfall:
                        allocated = pay_add_capital(lead_stt, period_date, step_max, is_cf_wf)
                    else:
                        allocated = 0.0  # CF_WF does not return capital

            elif lead_state == "Amt":
                # Fixed amount
                allocated = min(step_max, abs(lead_m_amount)) if lead_m_amount != 0 else 0.0
                if allocated > 0:
                    apply_distribution(lead_stt, period_date, allocated, is_cf_wf)
            
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
                # Tag distributions follow same CF/Cap classification as lead
                apply_distribution(tag_stt, period_date, allocated, not is_capital_waterfall)

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
    
    # NOTE: year-end compounding is handled by accrue_pref_to_date which
    # compounds at year boundaries when accruing across years.  For the
    # 12/31 period itself (no boundary crossing), pref_accrued_current_year
    # carries forward and is compounded when the next period accrues past
    # 12/31, effectively deferring the compound until the next distribution.

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
    capital_calls: Optional[List[dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, InvestorState]]:
    """
    Run complete waterfall for all periods

    Args:
        wf_steps: All waterfall step definitions
        vcode: Deal code
        wf_name: "CF_WF" or "Cap_WF"
        period_cash: DataFrame with columns [event_date, cash_available]
        initial_states: Optional pre-seeded investor states
        show_structure_when_no_cash: If True, create zero-allocation rows even when no cash
        capital_calls: Optional list of capital call dicts to apply at correct dates

    Returns:
        (allocations_df, investor_states_dict)
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
    # Pre-initialize pref rates from waterfall steps so the first period
    # accrues on the correct base (including compounded prior-year pref)
    pref_rates = pref_rates_from_waterfall_steps(wf_steps, vcode)
    add_pref_rates = add_pref_rates_from_waterfall_steps(wf_steps, vcode)

    # Build set of pending capital calls (apply when period_date >= call_date)
    pending_calls = []
    if capital_calls:
        for call in capital_calls:
            cd = call.get('call_date')
            if cd is not None:
                cd_date = cd.date() if hasattr(cd, 'date') else cd
                pending_calls.append({**call, '_date': cd_date, '_applied': False})

    # Determine if this is capital waterfall
    is_cap_wf = (wf_name == "Cap_WF")

    all_rows = []
    for _, r in period_cash.sort_values("event_date").iterrows():
        d = r["event_date"]
        cash = float(r["cash_available"])

        # Apply any capital calls whose date falls within or before this period
        for pc in pending_calls:
            if not pc['_applied'] and pc['_date'] <= d:
                inv_id = pc.get('investor_id')
                amount = abs(pc.get('amount', 0))
                if inv_id and amount and inv_id in istates:
                    stt = istates[inv_id]
                    stt.capital_outstanding += amount
                    stt.cashflows.append((pc['_date'], -amount))
                pc['_applied'] = True

        _rem, rows = run_waterfall_period(
            steps, istates, d, cash, pref_rates,
            is_capital_waterfall=is_cap_wf,
            add_pref_rates=add_pref_rates,
        )
        all_rows.extend(rows)
    
    alloc_df = pd.DataFrame(all_rows)
    
    # Investor summary with metrics
    inv_rows = []
    for pc, stt in istates.items():
        # Calculate unrealized NAV (initial + additional capital pools)
        unrealized = (stt.capital_outstanding + stt.pref_unpaid_compounded + stt.pref_accrued_current_year
                      + stt.add_capital_outstanding + stt.add_pref_unpaid_compounded + stt.add_pref_accrued_current_year)
        
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
    
    # Return allocations DataFrame and the investor states dict
    return alloc_df, istates


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


def add_pref_rates_from_waterfall_steps(wf_steps: pd.DataFrame, vcode: str) -> dict:
    """Extract pref rates from waterfall Add steps where vtranstype contains 'pref'"""
    s = wf_steps[wf_steps["vcode"].astype(str) == str(vcode)].copy()
    if s.empty:
        return {}
    add_pref = s[
        (s["vState"].astype(str).str.strip() == "Add") &
        (s.get("vtranstype", pd.Series(dtype=str)).fillna("").astype(str).str.lower().str.contains("pref"))
    ].copy()
    if add_pref.empty:
        return {}
    add_pref = add_pref.sort_values(["vmisc", "iOrder"])
    out = {}
    for pc, grp in add_pref.groupby("PropCode"):
        out[str(pc)] = float(grp["nPercent_dec"].iloc[0])
    return out


# Keywords in Typename that indicate additional/special capital contributions
_ADD_CAPITAL_TYPENAMES = {"operating capital", "additional capital", "special capital"}


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

            # Route to add-capital pool if Typename indicates additional/special capital
            typename = str(r.get("Typename", "")).strip().lower()
            is_add_capital = any(kw in typename for kw in _ADD_CAPITAL_TYPENAMES)
            if is_add_capital:
                stt.add_capital_outstanding += abs(cf)
            else:
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
# RECURSIVE UPSTREAM WATERFALL PROCESSING
# ============================================================

def get_upstream_waterfall_entities(wf_steps: pd.DataFrame) -> set:
    """
    Get set of entities that have their own waterfall definitions.
    These are PropCodes that also appear as vcodes in the waterfall table.
    """
    if wf_steps.empty:
        return set()
    vcodes = set(wf_steps["vcode"].astype(str).unique())
    return vcodes


def get_entity_investors(entity_id: str, relationships: pd.DataFrame) -> pd.DataFrame:
    """
    Get investors in an entity from relationships table.
    Returns DataFrame with InvestorID and OwnershipPct for the given InvestmentID.
    """
    if relationships is None or relationships.empty:
        return pd.DataFrame()

    investors = relationships[
        relationships["InvestmentID"].astype(str) == str(entity_id)
    ].copy()

    return investors


def run_upstream_waterfall_period(
    entity_id: str,
    cash_available: float,
    period_date: date,
    wf_steps: pd.DataFrame,
    relationships: pd.DataFrame,
    entity_states: Dict[str, InvestorState],
    upstream_entities: set,
    wf_type: str = "CF_WF",
    path: str = "",
    allocation_rows: List[dict] = None,
    max_depth: int = 10,
    current_depth: int = 0,
    source_vtranstype: str = "",
) -> Dict[str, float]:
    """
    Process upstream waterfall for a single entity in a single period.

    If entity has a waterfall definition, use it.
    Otherwise, distribute pari passu by ownership percentage.

    Uses source_vtranstype to select type-specific waterfalls when available.
    For example, if source_vtranstype contains "Promote" and the entity has a
    Promote_WF defined, that waterfall is used instead of the default wf_type.

    Args:
        entity_id: Entity receiving cash (PropCode that received from deal waterfall)
        cash_available: Cash to distribute at this level
        period_date: Distribution date
        wf_steps: All waterfall definitions
        relationships: Ownership relationships
        entity_states: InvestorState dict (modified in place)
        upstream_entities: Set of entities with waterfall definitions
        wf_type: "CF_WF" or "Cap_WF"
        path: Ownership path for tracking (e.g., "P0000088->PPIBEL->TGA24")
        allocation_rows: List to append allocation records
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        source_vtranstype: vtranstype from the parent allocation, used to route
            to type-specific waterfalls (e.g., "GP Promote" -> Promote_WF)

    Returns:
        Dict of terminal entity -> cash received
    """
    if allocation_rows is None:
        allocation_rows = []

    if current_depth >= max_depth:
        return {entity_id: cash_available}

    if cash_available <= 0.01:  # Less than 1 cent
        return {}

    current_path = f"{path}->{entity_id}" if path else entity_id
    terminal_cash = {}

    # Check if this entity has a waterfall definition
    if entity_id in upstream_entities:
        # Determine which waterfall type to use based on source_vtranstype.
        # If the incoming cash is tagged as "GP Promote" (or similar), check
        # whether this entity has a Promote_WF. If so, use it instead of
        # the default CF_WF/Cap_WF so that promote income is distributed
        # under different rules than regular investment income.
        effective_wf_type = wf_type
        if source_vtranstype and "Promote" in source_vtranstype:
            promote_steps = wf_steps[
                (wf_steps["vcode"].astype(str) == str(entity_id)) &
                (wf_steps["vmisc"] == "Promote_WF")
            ]
            if not promote_steps.empty:
                effective_wf_type = "Promote_WF"

        # Run this entity's waterfall
        steps = wf_steps[
            (wf_steps["vcode"].astype(str) == str(entity_id)) &
            (wf_steps["vmisc"] == effective_wf_type)
        ].copy()

        if not steps.empty:
            steps = steps.sort_values("iOrder")

            # Ensure entity states exist for recipients
            for pc in steps["PropCode"].unique():
                if str(pc) not in entity_states:
                    entity_states[str(pc)] = InvestorState(propcode=str(pc))

            # Run the waterfall period (simplified - use Share distribution)
            remaining = cash_available
            is_cf_wf = (effective_wf_type in ("CF_WF", "Promote_WF"))

            # Group by iOrder to handle paired steps
            for order in sorted(steps["iOrder"].unique()):
                order_steps = steps[steps["iOrder"] == order]

                # Find lead and tag steps at this order
                lead_steps = order_steps[order_steps["vState"].astype(str).str.strip() != "Tag"]
                tag_steps = order_steps[order_steps["vState"].astype(str).str.strip() == "Tag"]

                lead_distributions = {}

                # Process lead steps
                for _, step in lead_steps.iterrows():
                    pc = str(step["PropCode"])
                    state = str(step["vState"]).strip()
                    fx = float(step["FXRate"])
                    rate = float(step.get("nPercent_dec", step.get("nPercent", 0)) or 0)

                    if pc not in entity_states:
                        entity_states[pc] = InvestorState(propcode=pc)

                    stt = entity_states[pc]
                    allocated = 0.0
                    step_max = remaining * fx if fx > 0 else remaining

                    if state == "Amt":
                        # Fixed amount distribution (entity expenses, fees)
                        m_amount = float(step.get("mAmount", 0) or 0)
                        if m_amount > 0:
                            allocated = min(step_max, m_amount)
                        else:
                            allocated = step_max
                        if allocated > 0:
                            apply_distribution(stt, period_date, allocated, is_cf_wf)
                    elif state == "Pref":
                        allocated = pay_pref(stt, period_date, step_max, is_cf_wf)
                    elif state in ("Def_Int", "Def&Int"):
                        # Default preferred return / combined default interest+principal
                        # Uses same logic as Pref but at the default rate
                        # Balance comes from InvestorState (populated from accounting feed)
                        allocated = pay_pref(stt, period_date, step_max, is_cf_wf)
                    elif state == "Default":
                        # Return of defaulted capital contributions
                        if effective_wf_type == "Cap_WF":
                            allocated = pay_initial_capital(stt, period_date, step_max, False)
                        else:
                            # In CF waterfall, default capital can still be returned per some LLCs
                            allocated = pay_initial_capital(stt, period_date, step_max, False)
                    elif state == "Initial":
                        if effective_wf_type == "Cap_WF":
                            allocated = pay_initial_capital(stt, period_date, step_max, False)
                    elif state == "Share":
                        allocated = step_max
                        if allocated > 0:
                            apply_distribution(stt, period_date, allocated, is_cf_wf)
                    elif state == "IRR":
                        target_irr = rate
                        allocated = irr_needed_distribution(stt, period_date, target_irr, step_max)
                        if allocated > 0:
                            apply_distribution(stt, period_date, allocated, is_cf_wf)
                    elif state == "Catchup":
                        # Catchup: GP gets larger share until reaching target split
                        allocated = step_max * rate if rate > 0 else step_max
                        allocated = min(allocated, remaining)
                        if allocated > 0:
                            apply_distribution(stt, period_date, allocated, is_cf_wf)

                    lead_distributions[pc] = (allocated, fx)

                    # Record allocation
                    allocation_rows.append({
                        "event_date": period_date,
                        "Year": period_date.year,
                        "Level": current_depth + 1,
                        "Entity": entity_id,
                        "Path": current_path,
                        "iOrder": int(step["iOrder"]),
                        "PropCode": pc,
                        "vState": state,
                        "FXRate": fx,
                        "vtranstype": step.get("vtranstype", ""),
                        "Allocated": float(allocated),
                        "WaterfallType": effective_wf_type,
                    })

                    remaining -= allocated

                    # Recursively process this recipient
                    if allocated > 0.01:
                        step_vtranstype = str(step.get("vtranstype", ""))
                        sub_terminal = run_upstream_waterfall_period(
                            entity_id=pc,
                            cash_available=allocated,
                            period_date=period_date,
                            wf_steps=wf_steps,
                            relationships=relationships,
                            entity_states=entity_states,
                            upstream_entities=upstream_entities,
                            wf_type=wf_type,
                            path=current_path,
                            allocation_rows=allocation_rows,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            source_vtranstype=step_vtranstype,
                        )
                        for term_id, term_cash in sub_terminal.items():
                            terminal_cash[term_id] = terminal_cash.get(term_id, 0.0) + term_cash

                # Process tag steps
                for _, tag in tag_steps.iterrows():
                    tag_pc = str(tag["PropCode"])
                    tag_fx = float(tag["FXRate"])

                    if tag_pc not in entity_states:
                        entity_states[tag_pc] = InvestorState(propcode=tag_pc)

                    tag_stt = entity_states[tag_pc]

                    # Calculate tag share based on lead distributions
                    allocated = 0.0
                    for lead_pc, (lead_alloc, lead_fx) in lead_distributions.items():
                        if lead_alloc > 0 and lead_fx > 0:
                            tag_share = (lead_alloc / lead_fx) * tag_fx
                            tag_share = min(tag_share, remaining)
                            allocated = tag_share
                            break

                    if allocated > 0:
                        apply_distribution(tag_stt, period_date, allocated, is_cf_wf)

                    allocation_rows.append({
                        "event_date": period_date,
                        "Year": period_date.year,
                        "Level": current_depth + 1,
                        "Entity": entity_id,
                        "Path": current_path,
                        "iOrder": int(tag["iOrder"]),
                        "PropCode": tag_pc,
                        "vState": "Tag",
                        "FXRate": tag_fx,
                        "vtranstype": tag.get("vtranstype", ""),
                        "Allocated": float(allocated),
                        "WaterfallType": effective_wf_type,
                    })

                    remaining -= allocated

                    # Recursively process tag recipient
                    if allocated > 0.01:
                        tag_vtranstype = str(tag.get("vtranstype", ""))
                        sub_terminal = run_upstream_waterfall_period(
                            entity_id=tag_pc,
                            cash_available=allocated,
                            period_date=period_date,
                            wf_steps=wf_steps,
                            relationships=relationships,
                            entity_states=entity_states,
                            upstream_entities=upstream_entities,
                            wf_type=wf_type,
                            path=current_path,
                            allocation_rows=allocation_rows,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            source_vtranstype=tag_vtranstype,
                        )
                        for term_id, term_cash in sub_terminal.items():
                            terminal_cash[term_id] = terminal_cash.get(term_id, 0.0) + term_cash

                if remaining <= 0.01:
                    break

            return terminal_cash

    # No waterfall definition - check for ownership relationships (passthrough)
    investors = get_entity_investors(entity_id, relationships)

    if investors.empty:
        # Terminal entity - no further upstream, cash stops here
        if entity_id not in entity_states:
            entity_states[entity_id] = InvestorState(propcode=entity_id)

        stt = entity_states[entity_id]
        is_cf_wf = (wf_type == "CF_WF")
        apply_distribution(stt, period_date, cash_available, is_cf_wf)

        allocation_rows.append({
            "event_date": period_date,
            "Year": period_date.year,
            "Level": current_depth + 1,
            "Entity": entity_id,
            "Path": current_path,
            "iOrder": 0,
            "PropCode": entity_id,
            "vState": "Terminal",
            "FXRate": 1.0,
            "vtranstype": "Terminal Distribution",
            "Allocated": float(cash_available),
            "WaterfallType": wf_type,
        })

        return {entity_id: cash_available}

    # Passthrough entity - distribute by ownership percentage
    for _, inv in investors.iterrows():
        investor_id = str(inv["InvestorID"])
        ownership_pct = float(inv["OwnershipPct"])

        # Normalize ownership percentage if stored as whole number
        if ownership_pct > 1.0:
            ownership_pct = ownership_pct / 100.0

        investor_cash = cash_available * ownership_pct

        if investor_cash > 0.01:
            allocation_rows.append({
                "event_date": period_date,
                "Year": period_date.year,
                "Level": current_depth + 1,
                "Entity": entity_id,
                "Path": current_path,
                "iOrder": 0,
                "PropCode": investor_id,
                "vState": "Passthrough",
                "FXRate": ownership_pct,
                "vtranstype": "Pari Passu",
                "Allocated": float(investor_cash),
                "WaterfallType": wf_type,
            })

            # Recursively process investor - preserve source_vtranstype
            # so promote tagging survives through passthrough entities
            sub_terminal = run_upstream_waterfall_period(
                entity_id=investor_id,
                cash_available=investor_cash,
                period_date=period_date,
                wf_steps=wf_steps,
                relationships=relationships,
                entity_states=entity_states,
                upstream_entities=upstream_entities,
                wf_type=wf_type,
                path=current_path,
                allocation_rows=allocation_rows,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                source_vtranstype=source_vtranstype,
            )
            for term_id, term_cash in sub_terminal.items():
                terminal_cash[term_id] = terminal_cash.get(term_id, 0.0) + term_cash

    return terminal_cash


def run_recursive_upstream_waterfalls(
    deal_allocations: pd.DataFrame,
    wf_steps: pd.DataFrame,
    relationships: pd.DataFrame,
    wf_type: str = "CF_WF",
    target_beneficiary: str = None,
) -> Tuple[pd.DataFrame, Dict[str, InvestorState], Dict[str, float]]:
    """
    Run recursive upstream waterfalls starting from deal-level allocations.

    For each PropCode that received cash in the deal waterfall, traces upstream
    through any waterfall definitions or ownership relationships until reaching
    terminal beneficiaries.

    Args:
        deal_allocations: DataFrame from run_waterfall() with Allocated amounts
        wf_steps: All waterfall definitions
        relationships: Ownership relationships
        wf_type: "CF_WF" or "Cap_WF"
        target_beneficiary: Optional - if specified, only trace paths to this entity

    Returns:
        (allocations_df, entity_states, beneficiary_totals)
        - allocations_df: All allocation records at every level
        - entity_states: InvestorState for every entity touched
        - beneficiary_totals: Dict of terminal entity -> total cash received
    """
    if deal_allocations.empty:
        return pd.DataFrame(), {}, {}

    upstream_entities = get_upstream_waterfall_entities(wf_steps)
    entity_states: Dict[str, InvestorState] = {}
    allocation_rows: List[dict] = []
    beneficiary_totals: Dict[str, float] = {}

    # Group deal allocations by period and PropCode
    for period_date, period_group in deal_allocations.groupby("event_date"):
        for _, row in period_group.iterrows():
            propcode = str(row["PropCode"])
            allocated = float(row.get("Allocated", 0))

            if allocated <= 0.01:
                continue

            # Add Level 0 record (deal level)
            allocation_rows.append({
                "event_date": period_date,
                "Year": period_date.year if hasattr(period_date, 'year') else pd.to_datetime(period_date).year,
                "Level": 0,
                "Entity": str(row.get("vcode", "Deal")),
                "Path": str(row.get("vcode", "Deal")),
                "iOrder": int(row.get("iOrder", 0)),
                "PropCode": propcode,
                "vState": str(row.get("vState", "")),
                "FXRate": float(row.get("FXRate", 1.0)),
                "vtranstype": str(row.get("vtranstype", "")),
                "Allocated": allocated,
                "WaterfallType": wf_type,
            })

            # Process upstream - pass vtranstype for waterfall routing
            deal_vtranstype = str(row.get("vtranstype", ""))
            terminal = run_upstream_waterfall_period(
                entity_id=propcode,
                cash_available=allocated,
                period_date=period_date if isinstance(period_date, date) else pd.to_datetime(period_date).date(),
                wf_steps=wf_steps,
                relationships=relationships,
                entity_states=entity_states,
                upstream_entities=upstream_entities,
                wf_type=wf_type,
                path=str(row.get("vcode", "Deal")),
                allocation_rows=allocation_rows,
                max_depth=10,
                current_depth=0,
                source_vtranstype=deal_vtranstype,
            )

            # Aggregate beneficiary totals
            for term_id, term_cash in terminal.items():
                beneficiary_totals[term_id] = beneficiary_totals.get(term_id, 0.0) + term_cash

    # Filter to target beneficiary if specified
    if target_beneficiary and target_beneficiary in beneficiary_totals:
        beneficiary_totals = {target_beneficiary: beneficiary_totals[target_beneficiary]}

    return pd.DataFrame(allocation_rows), entity_states, beneficiary_totals


def calculate_beneficiary_summary(
    entity_states: Dict[str, InvestorState],
    beneficiary_totals: Dict[str, float],
    as_of_date: date = None,
) -> pd.DataFrame:
    """
    Calculate summary metrics for each terminal beneficiary.

    Args:
        entity_states: InvestorState dict from recursive processing
        beneficiary_totals: Total cash received by each beneficiary
        as_of_date: Date for metric calculations

    Returns:
        DataFrame with beneficiary summaries
    """
    if as_of_date is None:
        as_of_date = date.today()

    rows = []
    for entity_id, total_cash in sorted(beneficiary_totals.items(), key=lambda x: -x[1]):
        if entity_id in entity_states:
            stt = entity_states[entity_id]
            metrics = investor_metrics(stt, as_of_date=as_of_date, unrealized_nav=0)

            rows.append({
                "EntityID": entity_id,
                "TotalCashReceived": total_cash,
                "TotalContributions": metrics["TotalContributions"],
                "TotalDistributions": metrics["TotalDistributions"],
                "IRR": metrics["IRR"],
                "MOIC": metrics["MOIC"],
            })
        else:
            rows.append({
                "EntityID": entity_id,
                "TotalCashReceived": total_cash,
                "TotalContributions": 0,
                "TotalDistributions": total_cash,
                "IRR": None,
                "MOIC": None,
            })

    return pd.DataFrame(rows)


def calculate_entity_through_flow(
    allocations: pd.DataFrame,
    entity_id: str,
) -> dict:
    """
    Calculate total cash that flows THROUGH an entity (received and distributed).

    Useful for entities like OWPSC that receive cash but then distribute to investors.
    This shows the entity's total exposure before any further distribution.

    Args:
        allocations: DataFrame from run_recursive_upstream_waterfalls
        entity_id: Entity to analyze

    Returns:
        Dict with:
        - total_received: Cash received by this entity
        - by_source: Breakdown by source entity
        - by_vtranstype: Breakdown by transaction type
    """
    if allocations.empty:
        return {"total_received": 0, "by_source": {}, "by_vtranstype": {}}

    # Find all allocations TO this entity
    received = allocations[allocations["PropCode"].astype(str) == str(entity_id)].copy()

    if received.empty:
        return {"total_received": 0, "by_source": {}, "by_vtranstype": {}}

    total = received["Allocated"].sum()

    # By source entity
    by_source = received.groupby("Entity")["Allocated"].sum().to_dict()

    # By transaction type
    by_vtranstype = received.groupby("vtranstype")["Allocated"].sum().to_dict()

    # By path (for tracing)
    by_path = received.groupby("Path")["Allocated"].sum().to_dict()

    return {
        "total_received": total,
        "by_source": by_source,
        "by_vtranstype": by_vtranstype,
        "by_path": by_path,
    }


def calculate_ubo_revenue_streams(
    allocations: pd.DataFrame,
    ubo_id: str = "OWPSC",
    gp_entity: str = "PSC1",
) -> dict:
    """
    Calculate Ultimate Beneficial Owner's revenue streams.

    For Peaceable Street Capital (OWPSC), tracks:
    - Capital income (returns on invested capital)
    - GP promotes/catch-up payments
    - Fees (acquisition, asset management) - via vtranstype

    Args:
        allocations: DataFrame from run_recursive_upstream_waterfalls
        ubo_id: Ultimate beneficial owner entity ID
        gp_entity: GP entity that receives promotes (flows to UBO)

    Returns:
        Dict with revenue breakdown
    """
    if allocations.empty:
        return {}

    # Get all cash flowing to UBO
    ubo_flow = calculate_entity_through_flow(allocations, ubo_id)

    # Get GP entity flow (promotes)
    gp_flow = calculate_entity_through_flow(allocations, gp_entity)

    # Identify promotes from GP
    gp_allocations = allocations[allocations["PropCode"].astype(str) == str(gp_entity)]
    promote_states = ["Tag", "Catchup", "Promote"]
    promotes = gp_allocations[gp_allocations["vState"].isin(promote_states)]["Allocated"].sum()

    # Capital income = everything that isn't a fee or promote
    fee_types = ["ACQ_FEE", "AM_FEE", "Acquisition Fee", "Asset Management Fee"]
    fees = allocations[
        (allocations["PropCode"].astype(str) == str(ubo_id)) &
        (allocations["vtranstype"].isin(fee_types))
    ]["Allocated"].sum()

    return {
        "ubo_id": ubo_id,
        "total_through_flow": ubo_flow["total_received"],
        "gp_promotes": promotes,
        "fees": fees,
        "capital_income": ubo_flow["total_received"] - fees,
        "by_source": ubo_flow.get("by_source", {}),
        "by_path": ubo_flow.get("by_path", {}),
    }


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
