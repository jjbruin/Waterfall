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

from models import InvestorState, CapitalPool, PrefTier
from metrics import xirr, investor_metrics
from loaders import build_investmentid_to_vcode, normalize_accounting_feed
from config import typename_to_pool, resolve_pool_and_action


# ============================================================
# PREF ACCRUAL LOGIC
# ============================================================



def compound_if_year_end(istates: Dict[str, InvestorState], period_date: date):
    """
    Deferred year-end compounding with 45-day grace period.

    Distributions are typically made within 45 days after period end.
    So for 12/31 accrued pref, we look at distributions through ~Feb 14.

    This function is called AFTER distributions for each period. The logic:
    - During Jan 1 - Feb 14: pref_accrued_prior_year holds the year-end balance
    - Distributions during this time reduce pref_accrued_prior_year
    - After Feb 14 (45 days): remaining pref_accrued_prior_year compounds

    This defers compounding until after the grace period, allowing Q4
    distributions (made in Jan/early Feb) to reduce the year-end balance
    before it compounds.
    """
    grace_period_days = 45

    for s in istates.values():
        for pool in s.pools.values():
            for tier in pool.pref_tiers:
                # Initialize tracking attributes if needed
                if not hasattr(tier, 'pref_accrued_prior_year'):
                    tier.pref_accrued_prior_year = 0.0
                if not hasattr(tier, 'prior_year_for_accrued'):
                    tier.prior_year_for_accrued = 0

                # Check if there's prior year accrued pref that needs compounding
                if tier.pref_accrued_prior_year > 0 and tier.prior_year_for_accrued > 0:
                    prior_year_end = date(tier.prior_year_for_accrued, 12, 31)
                    days_since_year_end = (period_date - prior_year_end).days

                    # Compound once we're past the grace period
                    if days_since_year_end >= grace_period_days:
                        tier.pref_unpaid_compounded += tier.pref_accrued_prior_year
                        tier.pref_accrued_prior_year = 0.0
                        tier.prior_year_for_accrued = 0


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
# POOL-BASED PREF ACCRUAL & PAYMENT (GENERAL PURPOSE)
# ============================================================

def accrue_pool_pref(pool: CapitalPool, asof: date):
    """
    Accrue preferred return on a single pool to a specific date.

    Iterates all PrefTiers on the pool.  Each tier accrues on:
        base = pool.capital_outstanding + tier.pref_unpaid_compounded
    Act/365 day count.

    DEFERRED COMPOUNDING: Year-end compounding is handled by compound_if_year_end()
    with a 45-day grace period. This function accrues but does NOT compound at
    year-end. Instead, it tracks pref_accrued_prior_year for amounts that crossed
    a year boundary but haven't been compounded yet.
    """
    if pool.last_accrual_date is None:
        pool.last_accrual_date = asof
        return

    if asof <= pool.last_accrual_date:
        return

    grace_period_days = 45

    for tier in pool.pref_tiers:
        if tier.pref_rate <= 0:
            continue

        # Initialize tracking attributes if needed
        if not hasattr(tier, 'pref_accrued_prior_year'):
            tier.pref_accrued_prior_year = 0.0
        if not hasattr(tier, 'prior_year_for_accrued'):
            tier.prior_year_for_accrued = 0

        cur = pool.last_accrual_date
        while cur < asof:
            year_end = date(cur.year, 12, 31)
            next_stop = min(asof, year_end)
            days = (next_stop - cur).days
            if days > 0:
                base = max(0.0, pool.capital_outstanding + tier.pref_unpaid_compounded)
                tier.pref_accrued_current_year += base * tier.pref_rate * (days / 365.0)

            # At year-end boundary, move current year accrued to prior year bucket
            # but DON'T compound yet - wait for grace period
            if next_stop == year_end and next_stop < asof:
                # Check if we're past the grace period for this year-end
                days_past_year_end = (asof - year_end).days
                if days_past_year_end >= grace_period_days:
                    # Past grace period - compound now
                    if tier.pref_accrued_current_year > 0:
                        tier.pref_unpaid_compounded += tier.pref_accrued_current_year
                        tier.pref_accrued_current_year = 0.0
                else:
                    # Within grace period - move to prior year bucket, don't compound
                    if tier.pref_accrued_current_year > 0:
                        tier.pref_accrued_prior_year += tier.pref_accrued_current_year
                        tier.prior_year_for_accrued = cur.year
                        tier.pref_accrued_current_year = 0.0
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = next_stop
                break

    pool.last_accrual_date = asof


def accrue_all_pools(ist: InvestorState, asof: date):
    """Accrue pref on every pool for an investor."""
    for pool in ist.pools.values():
        accrue_pool_pref(pool, asof)


def _ensure_pool_tiers(ist: InvestorState, pref_rates, add_pref_rates):
    """
    Ensure every pool has its PrefTier(s) initialised from waterfall rates.

    Called once per period before accrual so that newly-created pools get
    their tier set up.  Handles multi-tier (Cocoplum) when pref_rates
    contains a list of rates for a PropCode.
    """
    pc = ist.propcode
    rate_val = pref_rates.get(pc, 0.0)
    pool = ist.get_pool("initial")
    if not pool.pref_tiers:
        if isinstance(rate_val, list):
            for r in rate_val:
                pool.pref_tiers.append(PrefTier(tier_name=f"pref_{r}", pref_rate=float(r)))
        else:
            pool.pref_tiers.append(PrefTier(tier_name="pref", pref_rate=float(rate_val)))

    # Additional pool pref tiers
    add_rate = float(add_pref_rates.get(pc, 0.0)) if add_pref_rates else 0.0
    if add_rate > 0:
        for pname in ("additional", "special", "cost_overrun"):
            if pname in ist.pools and not ist.pools[pname].pref_tiers:
                ist.pools[pname].pref_tiers.append(
                    PrefTier(tier_name="pref", pref_rate=add_rate)
                )


def pay_pool_pref(ist: InvestorState, pool_name: str, d: date,
                  available: float, is_cf_waterfall: bool = False,
                  tier_index: int = 0,
                  cocoplum_cross_reduce: bool = False) -> float:
    """
    Pay preferred return from a specific pool's tier.

    Payment priority:
    1. Compounded pref (from prior years)
    2. Prior year accrued (within grace period, not yet compounded)
    3. Current year accrued

    For Cocoplum dual-tier: if cocoplum_cross_reduce=True, paying
    tier 0 (5%) also reduces tier 1 (8.5%) balances dollar-for-dollar.

    Returns amount paid.
    """
    pool = ist.get_pool(pool_name)
    if not pool.pref_tiers or tier_index >= len(pool.pref_tiers):
        return 0.0

    tier = pool.pref_tiers[tier_index]
    pay = 0.0
    if available <= 0:
        return 0.0

    # Initialize tracking attribute if needed
    if not hasattr(tier, 'pref_accrued_prior_year'):
        tier.pref_accrued_prior_year = 0.0

    # 1. Pay compounded first (oldest pref)
    if tier.pref_unpaid_compounded > 0:
        x = min(available, tier.pref_unpaid_compounded)
        tier.pref_unpaid_compounded -= x
        available -= x
        pay += x

    # 2. Pay prior year accrued (within grace period)
    if available > 0 and tier.pref_accrued_prior_year > 0:
        x = min(available, tier.pref_accrued_prior_year)
        tier.pref_accrued_prior_year -= x
        available -= x
        pay += x

    # 3. Pay current year accrued
    if available > 0 and tier.pref_accrued_current_year > 0:
        x = min(available, tier.pref_accrued_current_year)
        tier.pref_accrued_current_year -= x
        available -= x
        pay += x

    if pay > 0:
        apply_distribution(ist, d, pay, is_cf_waterfall)
        if cocoplum_cross_reduce and len(pool.pref_tiers) > 1:
            _cross_reduce_tiers(pool, tier_index, pay)
    return pay


def _cross_reduce_tiers(pool: CapitalPool, paid_tier_index: int, amount: float):
    """Cocoplum rule: paying one tier reduces other tiers dollar-for-dollar."""
    remaining = amount
    for i, tier in enumerate(pool.pref_tiers):
        if i == paid_tier_index or remaining <= 0:
            continue
        # Initialize if needed
        if not hasattr(tier, 'pref_accrued_prior_year'):
            tier.pref_accrued_prior_year = 0.0
        # Reduce in priority order: compounded, prior year, current year
        if tier.pref_unpaid_compounded > 0:
            x = min(remaining, tier.pref_unpaid_compounded)
            tier.pref_unpaid_compounded -= x
            remaining -= x
        if remaining > 0 and tier.pref_accrued_prior_year > 0:
            x = min(remaining, tier.pref_accrued_prior_year)
            tier.pref_accrued_prior_year -= x
            remaining -= x
        if remaining > 0 and tier.pref_accrued_current_year > 0:
            x = min(remaining, tier.pref_accrued_current_year)
            tier.pref_accrued_current_year -= x
            remaining -= x


def pay_pool_capital(ist: InvestorState, pool_name: str, d: date,
                     available: float, is_cf_waterfall: bool = False) -> float:
    """
    Return capital from a specific pool.

    Handles cumulative cap for operating capital.
    Capital returns are never CF distributions.
    """
    pool = ist.get_pool(pool_name)
    if available <= 0 or pool.capital_outstanding <= 0:
        return 0.0

    payable = available

    # Enforce cumulative cap for operating capital
    if pool.cumulative_cap is not None:
        max_remaining = max(0.0, pool.cumulative_cap - pool.cumulative_returned)
        payable = min(payable, max_remaining)

    x = min(payable, pool.capital_outstanding)
    if x <= 0:
        return 0.0
    pool.capital_outstanding -= x
    pool.cumulative_returned += x
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

    # Ensure pref tiers are initialised and accrue all pools
    for pc, stt in istates.items():
        _ensure_pool_tiers(stt, pref_rates, add_pref_rates)
        accrue_all_pools(stt, period_date)
    
    remaining = float(cash_available)
    
    # Group steps by vAmtType to identify paired steps
    steps = steps.sort_values("iOrder")
    
    # Track which (iOrder, vAmtType) groups we've processed.
    # Both keys are needed because a single iOrder can contain
    # multiple vAmtType groups (e.g. P0000033 iOrder 2 has "Pref"
    # for PPI28 and "OP" for OPOREI — both need processing).
    processed_groups = set()

    for _, step in steps.iterrows():
        order = int(step["iOrder"])
        amt_type = str(step.get("vAmtType", "")).strip()

        # Skip if already processed (as part of a pair)
        if (order, amt_type) in processed_groups:
            continue
        pc = str(step["PropCode"])
        state = str(step["vState"]).strip()
        fx = float(step["FXRate"])
        rate = float(step["nPercent_dec"])
        m_amount = float(step.get("mAmount", 0.0) or 0.0)
        
        # Ensure investor state exists
        if pc not in istates:
            istates[pc] = InvestorState(propcode=pc)
        
        stt = istates[pc]
        
        # Find paired steps (same vAmtType AND same iOrder)
        if amt_type:
            paired_steps = steps[
                (steps["vAmtType"].astype(str).str.strip() == amt_type) &
                (steps["iOrder"] == order)
            ].copy()
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
            step_max = min(level_available * lead_fx, remaining) if lead_fx > 0 else remaining
            
            # CF waterfall distributions should be tracked for ROE calculation
            is_cf_wf = not is_capital_waterfall

            # Process based on vState
            if lead_state == "Pref":
                pref_rates[lead_pc] = lead_rate
                # Find which tier this step targets (match by rate)
                init_pool = lead_stt.get_pool("initial")
                tier_idx = 0
                multi_tier = len(init_pool.pref_tiers) > 1
                if multi_tier:
                    for ti, t in enumerate(init_pool.pref_tiers):
                        if abs(t.pref_rate - lead_rate) < 1e-6:
                            tier_idx = ti
                            break
                allocated = pay_pool_pref(
                    lead_stt, "initial", period_date, step_max, is_cf_wf,
                    tier_index=tier_idx,
                    cocoplum_cross_reduce=multi_tier,
                )

            elif lead_state == "Initial":
                if is_capital_waterfall:
                    allocated = pay_pool_capital(lead_stt, "initial", period_date, step_max, is_cf_wf)

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
                # Route via resolve_pool_and_action
                lead_vtranstype = str(lead.get("vtranstype", "")).strip()
                pool_name, action = resolve_pool_and_action(
                    lead_state, lead_vtranstype, is_capital_waterfall
                )
                if action == "pay_pref":
                    add_pref_rates[lead_pc] = lead_rate
                    # Ensure pref tier exists on the target pool
                    tgt_pool = lead_stt.get_pool(pool_name)
                    if not tgt_pool.pref_tiers:
                        tgt_pool.pref_tiers.append(
                            PrefTier(tier_name="pref", pref_rate=lead_rate)
                        )
                    allocated = pay_pool_pref(lead_stt, pool_name, period_date, step_max, is_cf_wf)
                elif action == "pay_capital":
                    allocated = pay_pool_capital(lead_stt, pool_name, period_date, step_max, is_cf_wf)
                elif action == "pay_capital_capped":
                    allocated = pay_pool_capital(lead_stt, pool_name, period_date, step_max, is_cf_wf)
                else:
                    allocated = 0.0  # "skip" or unknown

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

            if remaining <= 1e-9:
                remaining = 0.0
                break
        
        # Mark this (iOrder, vAmtType) group as processed
        processed_groups.add((order, amt_type))
        
        if remaining <= 1e-9:
            remaining = 0.0
            break
    
    # NOTE: year-end compounding is handled by accrue_pool_pref which
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
                    # Route to correct pool via typename
                    pool_name = typename_to_pool(pc.get('typename', ''))
                    pool = stt.get_pool(pool_name)
                    pool.capital_outstanding += amount
                    if pool_name == "operating":
                        if pool.cumulative_cap is None:
                            pool.cumulative_cap = 0.0
                        pool.cumulative_cap += amount
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
        # Calculate unrealized NAV across all pools and tiers
        unrealized = stt.total_capital_outstanding + stt.total_pref_balance
        
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
    """Extract pref rates from waterfall Pref steps.

    Returns dict of PropCode -> rate (float) for single-rate deals,
    or PropCode -> [rate1, rate2, ...] for multi-tier deals (e.g. Cocoplum).
    """
    s = wf_steps[wf_steps["vcode"].astype(str) == str(vcode)].copy()
    if s.empty:
        return {}
    pref = s[s["vState"].astype(str).str.strip() == "Pref"].copy()
    if pref.empty:
        return {}
    pref = pref.sort_values(["vmisc", "iOrder"])
    out = {}
    for pc, grp in pref.groupby("PropCode"):
        # Filter out 0.0 / NaN rates (data quality: some Cap_WF steps have NaN nPercent)
        rates = sorted(r for r in grp["nPercent_dec"].unique().tolist() if r and r > 0)
        if not rates:
            # All rates were zero/NaN — fall back to 0.0 so pool gets a tier
            out[str(pc)] = 0.0
        elif len(rates) == 1:
            out[str(pc)] = float(rates[0])
        else:
            # Multi-tier (e.g. Cocoplum 5% + 8.5%)
            out[str(pc)] = [float(r) for r in rates]
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
    2. Track capital balances per named pool
    3. DO NOT modify pref balances (forward model handles that)

    Contributions are routed to the correct CapitalPool via
    typename_to_pool() (config.py).  Operating capital contributions
    also set the pool's cumulative_cap.

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

        # CONTRIBUTIONS: Route to correct pool and record cashflow
        if is_contribution:
            cf = amt if amt < 0 else -abs(amt)  # Ensure negative
            stt.cashflows.append((d, cf))

            # Route to named pool via Typename
            pool_name = typename_to_pool(str(r.get("Typename", "")))
            pool = stt.get_pool(pool_name)
            pool.capital_outstanding += abs(cf)

            # Operating capital: track cumulative cap from actual contributions
            if pool_name == "operating":
                if pool.cumulative_cap is None:
                    pool.cumulative_cap = 0.0
                pool.cumulative_cap += abs(cf)

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

    # CRITICAL FIX: Set last_accrual_date to the LATEST historical date
    # This prevents the forward model from re-accruing pref that was already
    # paid historically. The forward model will only accrue from this point.
    # Also seed unpaid pref balance by calculating (expected accrual - actual paid).
    if not acct.empty:
        latest_date = acct["EffectiveDate"].max()

        # Calculate unpaid pref for each investor
        for pc, stt in states.items():
            if stt.last_accrual_date is None:
                continue

            # Get pref rate from waterfall steps
            pref_rate = 0.0
            pref_steps = wf_steps[
                (wf_steps["vcode"].astype(str) == str(target_vcode)) &
                (wf_steps["PropCode"].astype(str) == pc) &
                (wf_steps["vState"].astype(str).str.strip() == "Pref")
            ]
            if not pref_steps.empty:
                rate_col = "nPercent_dec" if "nPercent_dec" in pref_steps.columns else "nPercent"
                if rate_col in pref_steps.columns:
                    pref_rate = float(pref_steps[rate_col].iloc[0])

            # Get this investor's transactions
            inv_acct = acct[acct["InvestorID"] == pc].copy()
            inv_acct = inv_acct.sort_values("EffectiveDate")

            if not inv_acct.empty and pref_rate > 0:
                # Calculate expected pref accrual from capital history
                # with year-end compounding (pref accrues on capital + unpaid compounded pref)
                current_capital = 0.0
                pref_accrued_current_year = 0.0
                pref_unpaid_compounded = 0.0
                prev_date = None

                # Get pref distributions for reducing accrued amounts
                pref_dists = inv_acct[inv_acct["TypeID"] == 1019.0][["EffectiveDate", "Amt"]].copy()
                pref_dists["Amt"] = pref_dists["Amt"].abs()

                def accrue_to_date(from_date, to_date, capital, compounded, accrued_prior_year=0.0):
                    """
                    Accrue pref with deferred year-end compounding (45-day grace period).

                    Returns (accrued_cy, compounded, accrued_prior_year)

                    At year-end boundary:
                    - If > 45 days past year-end: compound immediately
                    - If <= 45 days past: move to prior_year bucket (compound later)
                    """
                    grace_period_days = 45

                    if from_date is None or to_date <= from_date or capital <= 0:
                        return 0.0, compounded, accrued_prior_year

                    accrued_cy = 0.0
                    cur = from_date
                    while cur < to_date:
                        year_end = date(cur.year, 12, 31)
                        next_stop = min(to_date, year_end)
                        days = (next_stop - cur).days
                        if days > 0:
                            base = max(0.0, capital + compounded)
                            accrued_cy += base * pref_rate * (days / 365.0)

                        # At year-end boundary
                        if next_stop == year_end and next_stop < to_date:
                            days_past_year_end = (to_date - year_end).days
                            if days_past_year_end >= grace_period_days:
                                # Past grace period - compound now
                                compounded += accrued_cy + accrued_prior_year
                                accrued_cy = 0.0
                                accrued_prior_year = 0.0
                            else:
                                # Within grace period - move to prior year bucket
                                accrued_prior_year += accrued_cy
                                accrued_cy = 0.0
                            cur = date(cur.year + 1, 1, 1)
                        else:
                            cur = next_stop
                            break
                    return accrued_cy, compounded, accrued_prior_year

                def apply_pref_payment(amount, accrued_cy, compounded, accrued_prior_year=0.0):
                    """Apply pref payment - reduces compounded first, then prior year, then current year"""
                    remaining = amount
                    if compounded > 0 and remaining > 0:
                        pay = min(remaining, compounded)
                        compounded -= pay
                        remaining -= pay
                    if accrued_prior_year > 0 and remaining > 0:
                        pay = min(remaining, accrued_prior_year)
                        accrued_prior_year -= pay
                        remaining -= pay
                    if accrued_cy > 0 and remaining > 0:
                        pay = min(remaining, accrued_cy)
                        accrued_cy -= pay
                        remaining -= pay
                    return accrued_cy, compounded, accrued_prior_year

                pref_accrued_prior_year = 0.0

                for _, tr in inv_acct.iterrows():
                    tr_date = tr["EffectiveDate"]
                    tr_amt = float(tr["Amt"])

                    # Accrue pref from previous date to this transaction date
                    if prev_date is not None:
                        accrued, pref_unpaid_compounded, pref_accrued_prior_year = accrue_to_date(
                            prev_date, tr_date, current_capital, pref_unpaid_compounded, pref_accrued_prior_year
                        )
                        pref_accrued_current_year += accrued

                    # Handle pref distribution (reduces accrued pref)
                    if tr["TypeID"] == 1019.0:
                        pref_accrued_current_year, pref_unpaid_compounded, pref_accrued_prior_year = apply_pref_payment(
                            abs(tr_amt), pref_accrued_current_year, pref_unpaid_compounded, pref_accrued_prior_year
                        )

                    # Update capital balance
                    if tr["is_contribution"]:
                        current_capital += abs(tr_amt)
                    elif tr["is_capital"] and tr["is_distribution"]:
                        current_capital = max(0.0, current_capital - abs(tr_amt))

                    prev_date = tr_date

                # Accrue from last transaction to latest_date
                if prev_date is not None:
                    final_accrued, final_compounded, final_prior_year = accrue_to_date(
                        prev_date, latest_date, current_capital, pref_unpaid_compounded, pref_accrued_prior_year
                    )
                    pref_accrued_current_year = final_accrued
                    pref_unpaid_compounded = final_compounded
                    pref_accrued_prior_year = final_prior_year

                # Total unpaid pref
                unpaid_pref = pref_accrued_current_year + pref_unpaid_compounded + pref_accrued_prior_year

                # Seed unpaid pref into the initial pool's tier
                pool = stt.get_pool("initial")
                if not pool.pref_tiers:
                    # Create the pref tier if it doesn't exist
                    pool.pref_tiers.append(PrefTier(tier_name="pref", pref_rate=pref_rate))

                tier = pool.pref_tiers[0]
                # Set the three buckets based on calculated values
                tier.pref_unpaid_compounded = pref_unpaid_compounded
                tier.pref_accrued_prior_year = pref_accrued_prior_year
                tier.pref_accrued_current_year = pref_accrued_current_year

                # Track which year the prior_year accrued is for
                if pref_accrued_prior_year > 0 and latest_date:
                    # If we're in the grace period (Jan-Feb), prior year is last year
                    if latest_date.month <= 2:
                        tier.prior_year_for_accrued = latest_date.year - 1
                    else:
                        tier.prior_year_for_accrued = 0

            # Update last_accrual_date for all pools
            for pool in stt.pools.values():
                pool.last_accrual_date = latest_date
            stt.last_accrual_date = latest_date

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
                        # Ensure initial pool has a pref tier for upstream entities
                        pool = stt.get_pool("initial")
                        if not pool.pref_tiers:
                            pool.pref_tiers.append(PrefTier(tier_name="pref", pref_rate=rate))
                        allocated = pay_pool_pref(stt, "initial", period_date, step_max, is_cf_wf)
                    elif state in ("Def_Int", "Def&Int"):
                        # Default preferred return / combined default interest+principal
                        pool = stt.get_pool("initial")
                        if not pool.pref_tiers:
                            pool.pref_tiers.append(PrefTier(tier_name="pref", pref_rate=rate))
                        allocated = pay_pool_pref(stt, "initial", period_date, step_max, is_cf_wf)
                    elif state == "Default":
                        # Return of defaulted capital contributions
                        if effective_wf_type == "Cap_WF":
                            allocated = pay_pool_capital(stt, "initial", period_date, step_max, False)
                        else:
                            # In CF waterfall, default capital can still be returned per some LLCs
                            allocated = pay_pool_capital(stt, "initial", period_date, step_max, False)
                    elif state == "Initial":
                        if effective_wf_type == "Cap_WF":
                            allocated = pay_pool_capital(stt, "initial", period_date, step_max, False)
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
