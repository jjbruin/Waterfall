@dataclass
class Loan:
    vcode: str
    loan_id: str
    orig_date: date
    maturity_date: date
    orig_amount: float
    loan_term_m: int
    amort_term_m: int
    io_months: int
    int_type: str  # Fixed / Variable
    index_name: str
    fixed_rate: float  # nRate for fixed loans
    spread: float       # vSpread for variable loans
    floor: float        # nFloor for variable loans
    cap: float          # vIntRatereset used as cap on INDEX (pre-spread) if > 0

    @staticmethod
    def _as_decimal(x) -> float:
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return 0.0
        x = float(x)
        return x / 100.0 if x > 1.0 else x

    def is_variable(self) -> bool:
        return (self.int_type or "").strip().lower().startswith("var")

    def rate_for_month(self) -> float:
        """
        Fixed:
          annual_rate = nRate

        Variable (per your rules):
          base index:
            SOFR/LIBOR -> 0.043
            WSJ        -> 0.075
            else       -> 0.043
          if cap exists (>0): effective_index = min(base, cap)
          else:              effective_index = base
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

        # Your requested change:
        annual_rate = max(floor + spread, effective_index + spread)
        return annual_rate
