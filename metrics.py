"""
metrics.py
Investment performance metrics: XIRR, ROE, MOIC
"""

from __future__ import annotations

from datetime import date
from typing import List, Tuple, Optional, TYPE_CHECKING
from scipy.optimize import brentq

if TYPE_CHECKING:
    from models import InvestorState


def xnpv(rate: float, cfs: List[Tuple[date, float]]) -> float:
    """
    Net present value with irregular cashflow dates

    Args:
        rate: Annual discount rate (as decimal, e.g., 0.15 for 15%)
        cfs: List of (date, amount) tuples

    Returns:
        Net present value
    """
    if not cfs or rate <= -1.0:
        return float('inf')

    cfs = sorted(cfs, key=lambda t: t[0])
    t0 = cfs[0][0]

    npv = 0.0
    for d, amount in cfs:
        years = (d - t0).days / 365.0  # Use 365 to match Excel XIRR convention
        npv += amount / ((1 + rate) ** years)

    return npv


def xirr(cfs: List[Tuple[date, float]]) -> Optional[float]:
    """
    Internal Rate of Return with irregular cashflow dates

    Args:
        cfs: List of (date, amount) tuples
             Negative amounts = investments
             Positive amounts = returns

    Returns:
        Annual IRR as decimal (e.g., 0.15 for 15%)
        None if unable to calculate
    """
    if not cfs or len(cfs) < 2:
        return None

    amounts = [a for _, a in cfs]

    # Must have both negative (investments) and positive (returns)
    if min(amounts) >= 0 or max(amounts) <= 0:
        return None

    try:
        # Find root between -99% and 1000% annual return
        irr = brentq(lambda r: xnpv(r, cfs), -0.99, 10.0, maxiter=100)
        return float(irr)
    except (ValueError, RuntimeError):
        # No root found in range
        return None


def calculate_roe(
    capital_events: List[Tuple[date, float]],
    cf_distributions: List[Tuple[date, float]],
    start_date: date,
    end_date: date
) -> float:
    """
    Annualized Return on Equity: (CF Distributions / Weighted Average Capital) annualized

    ROE measures operating cash return on invested capital:
    - Numerator: Only CF waterfall distributions (operating income)
    - Excludes return of capital (Cap_WF) and balance sheet cash at sale
    - Denominator: Time-weighted average capital outstanding
    - Capital balance is reduced by capital returns (Cap_WF), not CF distributions
    - Days with zero capital balance contribute zero to weighted average

    Args:
        capital_events: [(date, amount)] where amount < 0 is contribution, > 0 is capital return
                        Used for weighted average capital calculation
        cf_distributions: [(date, amount)] CF waterfall distributions only (amount > 0)
                          Operating income for ROE numerator
        start_date: Period start (typically first funding date)
        end_date: Period end (typically sale date)

    Returns:
        Annualized ROE as decimal (e.g., 0.12 for 12% per year)
    """
    # Calculate total contributions (invested capital)
    total_contributions = sum(
        -amt for d, amt in capital_events
        if amt < 0 and start_date <= d <= end_date
    )

    if total_contributions <= 0:
        return 0.0

    # Build capital balance timeline from capital events
    # (contributions and capital returns, NOT CF distributions)
    events = []

    for d, amt in capital_events:
        if d < start_date or d > end_date:
            continue
        # Negative = contribution (increases balance)
        # Positive = return of capital (decreases balance)
        events.append((d, -amt))  # Flip sign for balance tracking

    if not events:
        return 0.0

    events = sorted(events, key=lambda x: x[0])

    # Calculate weighted average capital
    total_weighted_capital = 0.0
    current_balance = 0.0
    prev_date = start_date

    for evt_date, change in events:
        # Add weighted capital for period before this event
        days = (evt_date - prev_date).days
        total_weighted_capital += current_balance * days

        # Update balance (floor at 0 - can't have negative capital outstanding)
        current_balance = max(0, current_balance + change)
        prev_date = evt_date

    # Add weighted capital for final period (to end_date/sale date)
    # If capital is fully returned, this contributes 0 (balance is 0)
    days = (end_date - prev_date).days
    total_weighted_capital += current_balance * days

    # Calculate weighted average
    total_days = (end_date - start_date).days
    if total_days == 0:
        return 0.0

    weighted_avg_capital = total_weighted_capital / total_days

    if weighted_avg_capital <= 0:
        return 0.0

    # Calculate years for annualization
    years = total_days / 365.0

    if years <= 0:
        return 0.0

    # Sum CF waterfall distributions (operating income only)
    total_cf_distributions = sum(
        amt for d, amt in cf_distributions
        if amt > 0 and start_date <= d <= end_date
    )

    # Annualized ROE = (CF distributions / weighted avg capital) / years
    # Note: Only CF waterfall distributions count, not return of capital or gain on sale
    return (total_cf_distributions / weighted_avg_capital) / years


def calculate_moic(
    contributions: List[Tuple[date, float]],
    distributions: List[Tuple[date, float]],
    unrealized_value: float = 0.0
) -> float:
    """
    Multiple on Invested Capital

    MOIC = (Total Distributions + Unrealized Value) / Total Contributions

    Args:
        contributions: [(date, amount)] where amount < 0
        distributions: [(date, amount)] where amount > 0 (includes capital returns)
        unrealized_value: Current NAV of remaining investment

    Returns:
        MOIC as multiple (e.g., 1.5 for 1.5x)
    """
    total_invested = abs(sum(amt for _, amt in contributions if amt < 0))

    if total_invested == 0:
        return 0.0

    total_distributed = sum(amt for _, amt in distributions if amt > 0)

    return (total_distributed + unrealized_value) / total_invested


def investor_metrics(
    ist: InvestorState,
    as_of_date: date,
    inception_date: Optional[date] = None,
    unrealized_nav: float = 0.0
) -> dict:
    """
    Calculate all three metrics: IRR, ROE, MOIC

    Args:
        ist: InvestorState with cashflows and cf_distributions populated
        as_of_date: Calculation date
        inception_date: Start date for ROE calc (defaults to first contribution)
        unrealized_nav: Current value of unrealized investment

    Returns:
        Dictionary with IRR, ROE, MOIC and supporting data
    """
    if not ist.cashflows:
        return {
            'IRR': None,
            'ROE': 0.0,
            'MOIC': 0.0,
            'TotalContributions': 0.0,
            'TotalDistributions': 0.0,
            'CFDistributions': 0.0,
            'CapitalOutstanding': ist.capital_outstanding
        }

    # Separate contributions and all distributions
    contributions = [(d, a) for d, a in ist.cashflows if a < 0]
    distributions = [(d, a) for d, a in ist.cashflows if a > 0]

    # CF waterfall distributions only (for ROE)
    cf_distributions = ist.cf_distributions if hasattr(ist, 'cf_distributions') else []

    # Capital events (all cashflows - used for weighted average capital)
    capital_events = ist.cashflows.copy()

    # For unrealized IRR, add terminal value
    cfs_with_terminal = ist.cashflows.copy()
    if unrealized_nav > 0:
        cfs_with_terminal.append((as_of_date, unrealized_nav))

    # Calculate IRR
    irr = xirr(cfs_with_terminal)

    # Calculate ROE using CF distributions only (not capital returns)
    if inception_date is None and contributions:
        inception_date = min(d for d, _ in contributions)

    if inception_date is None:
        roe = 0.0
    else:
        roe = calculate_roe(capital_events, cf_distributions, inception_date, as_of_date)

    # Calculate MOIC (uses all distributions including capital returns)
    moic = calculate_moic(contributions, distributions, unrealized_nav)

    total_cf_dist = sum(a for _, a in cf_distributions) if cf_distributions else 0.0

    return {
        'IRR': irr,
        'ROE': roe,
        'MOIC': moic,
        'TotalContributions': abs(sum(a for _, a in contributions)),
        'TotalDistributions': sum(a for _, a in distributions),
        'CFDistributions': total_cf_dist,
        'CapitalOutstanding': ist.capital_outstanding,
        'UnrealizedNAV': unrealized_nav
    }
