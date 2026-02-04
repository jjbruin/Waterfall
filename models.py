"""
models.py
Data structures for waterfall model
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

from config import INDEX_BASE_RATES


# ============================================================
# CAPITAL POOL DATA STRUCTURES
# ============================================================

@dataclass
class PrefTier:
    """One tier of preferred return on a capital pool.

    A pool can have multiple tiers (e.g. Cocoplum has 5% and 8.5%
    tiers on the same initial capital). Each tier accrues independently.
    """
    tier_name: str             # "pref", "pref_5pct", "pref_8.5pct"
    pref_rate: float = 0.0     # 0.0 = no pref (e.g. operating capital)
    pref_unpaid_compounded: float = 0.0
    pref_accrued_current_year: float = 0.0


@dataclass
class CapitalPool:
    """A single capital class for one investor.

    Pool names: "initial", "additional", "special", "operating", "cost_overrun"
    """
    pool_name: str
    capital_outstanding: float = 0.0
    last_accrual_date: Optional[date] = None
    pref_tiers: List[PrefTier] = field(default_factory=list)
    cumulative_cap: Optional[float] = None   # for operating capital max return
    cumulative_returned: float = 0.0         # for operating capital cumulative tracking


@dataclass
class InvestorState:
    """
    State tracking for a single investor/partner

    Tracks:
    - Capital account balances across named pools
    - Preferred return accruals (compounded and current year) per tier
    - Cashflow history for XIRR calculation
    - CF waterfall distributions (for ROE calculation)

    Backward-compatible: code that reads/writes capital_outstanding,
    pref_unpaid_compounded, etc. still works via properties that
    delegate to the "initial" or "additional" pool.
    """
    propcode: str
    pools: Dict[str, CapitalPool] = field(default_factory=dict)
    cashflows: List[Tuple[date, float]] = field(default_factory=list)
    # Cashflows: negative = contribution, positive = distribution
    cf_distributions: List[Tuple[date, float]] = field(default_factory=list)
    # CF waterfall distributions only (operating income for ROE calculation)

    # ------------------------------------------------------------------
    # Pool helpers
    # ------------------------------------------------------------------

    def get_pool(self, name: str) -> CapitalPool:
        """Get or create a capital pool by name."""
        if name not in self.pools:
            self.pools[name] = CapitalPool(pool_name=name)
        return self.pools[name]

    # ------------------------------------------------------------------
    # Backward-compat properties — initial pool
    # ------------------------------------------------------------------

    @property
    def capital_outstanding(self) -> float:
        """Capital outstanding in the initial pool."""
        if "initial" not in self.pools:
            return 0.0
        return self.pools["initial"].capital_outstanding

    @capital_outstanding.setter
    def capital_outstanding(self, value: float):
        self.get_pool("initial").capital_outstanding = value

    @property
    def pref_unpaid_compounded(self) -> float:
        """Unpaid compounded pref in the initial pool (all tiers)."""
        if "initial" not in self.pools:
            return 0.0
        return sum(t.pref_unpaid_compounded for t in self.pools["initial"].pref_tiers)

    @pref_unpaid_compounded.setter
    def pref_unpaid_compounded(self, value: float):
        pool = self.get_pool("initial")
        if pool.pref_tiers:
            pool.pref_tiers[0].pref_unpaid_compounded = value
        else:
            pool.pref_tiers.append(PrefTier(tier_name="pref", pref_unpaid_compounded=value))

    @property
    def pref_accrued_current_year(self) -> float:
        """Current-year accrued pref in the initial pool (all tiers)."""
        if "initial" not in self.pools:
            return 0.0
        return sum(t.pref_accrued_current_year for t in self.pools["initial"].pref_tiers)

    @pref_accrued_current_year.setter
    def pref_accrued_current_year(self, value: float):
        pool = self.get_pool("initial")
        if pool.pref_tiers:
            pool.pref_tiers[0].pref_accrued_current_year = value
        else:
            pool.pref_tiers.append(PrefTier(tier_name="pref", pref_accrued_current_year=value))

    # ------------------------------------------------------------------
    # Backward-compat properties — additional / non-initial pools
    # ------------------------------------------------------------------

    @property
    def add_capital_outstanding(self) -> float:
        """Capital outstanding across all non-initial pools."""
        return sum(p.capital_outstanding for n, p in self.pools.items() if n != "initial")

    @add_capital_outstanding.setter
    def add_capital_outstanding(self, value: float):
        self.get_pool("additional").capital_outstanding = value

    @property
    def add_pref_unpaid_compounded(self) -> float:
        """Unpaid compounded pref across all non-initial pools."""
        return sum(
            t.pref_unpaid_compounded
            for n, p in self.pools.items() if n != "initial"
            for t in p.pref_tiers
        )

    @add_pref_unpaid_compounded.setter
    def add_pref_unpaid_compounded(self, value: float):
        pool = self.get_pool("additional")
        if pool.pref_tiers:
            pool.pref_tiers[0].pref_unpaid_compounded = value
        else:
            pool.pref_tiers.append(PrefTier(tier_name="pref", pref_unpaid_compounded=value))

    @property
    def add_pref_accrued_current_year(self) -> float:
        """Current-year accrued pref across all non-initial pools."""
        return sum(
            t.pref_accrued_current_year
            for n, p in self.pools.items() if n != "initial"
            for t in p.pref_tiers
        )

    @add_pref_accrued_current_year.setter
    def add_pref_accrued_current_year(self, value: float):
        pool = self.get_pool("additional")
        if pool.pref_tiers:
            pool.pref_tiers[0].pref_accrued_current_year = value
        else:
            pool.pref_tiers.append(PrefTier(tier_name="pref", pref_accrued_current_year=value))

    # ------------------------------------------------------------------
    # Backward-compat property — last_accrual_date (shared across pools)
    # ------------------------------------------------------------------

    @property
    def last_accrual_date(self) -> Optional[date]:
        """Last accrual date (from initial pool; shared in legacy code)."""
        if "initial" not in self.pools:
            return None
        return self.pools["initial"].last_accrual_date

    @last_accrual_date.setter
    def last_accrual_date(self, value: Optional[date]):
        # Propagate to all existing pools (mirrors old shared-field behavior)
        self.get_pool("initial").last_accrual_date = value
        for p in self.pools.values():
            p.last_accrual_date = value

    # ------------------------------------------------------------------
    # Aggregate properties (new)
    # ------------------------------------------------------------------

    @property
    def total_capital_outstanding(self) -> float:
        """Total capital across ALL pools."""
        return sum(p.capital_outstanding for p in self.pools.values())

    @property
    def total_pref_balance(self) -> float:
        """Total pref owed across all pools and all tiers."""
        return sum(
            t.pref_unpaid_compounded + t.pref_accrued_current_year
            for p in self.pools.values()
            for t in p.pref_tiers
        )


@dataclass
class Loan:
    """
    Loan structure for debt service modeling

    Supports:
    - Fixed and variable rate loans
    - Interest-only and amortizing structures
    - Rate caps and floors
    """
    vcode: str
    loan_id: str
    orig_date: date
    maturity_date: date
    orig_amount: float
    loan_term_m: int
    amort_term_m: int
    io_months: int
    int_type: str  # Fixed / Variable
    index_name: str  # SOFR, LIBOR, WSJ
    fixed_rate: float  # nRate for fixed loans
    spread: float  # vSpread for variable loans
    floor: float  # nFloor for variable loans
    cap: float  # vIntRatereset used as cap on INDEX (pre-spread)

    @staticmethod
    def _as_decimal(x) -> float:
        """Convert percentage to decimal if needed"""
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return 0.0
        x = float(x)
        return x / 100.0 if x > 1.0 else x

    def is_variable(self) -> bool:
        """Check if loan is variable rate"""
        return (self.int_type or "").strip().lower().startswith("var")

    def rate_for_month(self) -> float:
        """
        Calculate effective annual rate for this loan

        Fixed:
          annual_rate = nRate (decimal)

        Variable:
          base index: SOFR/LIBOR -> 0.043, WSJ -> 0.075, else -> 0.043
          if cap exists (>0): effective_index = min(base, cap)
          else: effective_index = base
          annual_rate = max(floor + spread, effective_index + spread)
        """
        idx = (self.index_name or "").strip().upper()

        if not self.is_variable():
            return self._as_decimal(self.fixed_rate)

        base = INDEX_BASE_RATES.get(idx, 0.043)
        cap = self._as_decimal(self.cap)
        spread = self._as_decimal(self.spread)
        floor = self._as_decimal(self.floor)

        effective_index = min(base, cap) if cap > 0 else base
        annual_rate = max(floor + spread, effective_index + spread)
        return annual_rate


@dataclass
class DealInFund:
    """Represents a deal's participation in a fund"""
    fund_id: str
    vcode: str
    ppi_propcode: str  # Which PPI entity (e.g., PPI_SE, PPI_NE)
    ownership_pct: float  # What % of PPI proceeds go to this fund


@dataclass
class FundState:
    """Aggregated state for a fund across all its deals"""
    fund_id: str
    deals: List[DealInFund] = field(default_factory=list)
    cashflows: List[Tuple[date, float]] = field(default_factory=list)
    total_contributions: float = 0.0
    total_distributions: float = 0.0
    nav: float = 0.0  # Unrealized value
