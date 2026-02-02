"""
models.py
Data structures for waterfall model
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from config import INDEX_BASE_RATES


@dataclass
class InvestorState:
    """
    State tracking for a single investor/partner

    Tracks:
    - Capital account balance
    - Preferred return accruals (compounded and current year)
    - Cashflow history for XIRR calculation
    - CF waterfall distributions (for ROE calculation)
    """
    propcode: str
    capital_outstanding: float = 0.0  # positive = capital owed back to investor
    pref_unpaid_compounded: float = 0.0  # unpaid pref from prior years
    pref_accrued_current_year: float = 0.0  # pref accrued this year
    last_accrual_date: Optional[date] = None
    cashflows: List[Tuple[date, float]] = field(default_factory=list)
    # Cashflows: negative = contribution, positive = distribution
    cf_distributions: List[Tuple[date, float]] = field(default_factory=list)
    # CF waterfall distributions only (operating income for ROE calculation)


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
