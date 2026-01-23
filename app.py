def amortize_monthly_schedule(loan: Loan, schedule_start: date, schedule_end: date) -> pd.DataFrame:
    """
    Month-end schedule.

    Fixed loans:
      - IO months: payment = interest
      - After IO: level-payment amortization (constant payment), unless maturity cuts it off (balloon possible)

    Variable loans (per your rules):
      - No amortization: interest-only for entire term
    """
    all_dates = month_ends_between(loan.orig_date, max(schedule_end, loan.maturity_date))
    if not all_dates:
        return pd.DataFrame()

    r_annual = loan.rate_for_month()
    r_m = r_annual / 12.0

    term_m = int(loan.loan_term_m) if loan.loan_term_m and loan.loan_term_m > 0 else len(all_dates)
    amort_m = int(loan.amort_term_m) if loan.amort_term_m and loan.amort_term_m > 0 else term_m
    io_m = int(loan.io_months) if loan.io_months and loan.io_months > 0 else 0

    # Variable: no amortization (interest-only for entire term)
    if loan.is_variable():
        io_m = term_m

    # remaining amort months after IO (used to set level payment ONCE)
    amort_after_io = max(1, amort_m - io_m)

    bal = float(loan.orig_amount)

    # We will set this at the first amort month and keep it constant thereafter (fixed loans only)
    level_payment: Optional[float] = None

    rows = []
    for i, dte in enumerate(all_dates, start=1):
        if dte > loan.maturity_date:
            break

        interest = bal * r_m

        if i <= io_m:
            # interest-only months (and all months for variable loans)
            payment = interest
            principal = 0.0
        else:
            # amortizing phase (fixed loans only)
            if level_payment is None:
                # compute once based on balance at amort start
                if r_m == 0:
                    level_payment = bal / amort_after_io
                else:
                    level_payment = bal * r_m / (1 - (1 + r_m) ** (-amort_after_io))

            payment = level_payment
            principal = max(0.0, payment - interest)

            # last-payment cleanup: don't pay more principal than remaining balance
            if principal > bal:
                principal = bal
                payment = interest + principal

        bal = max(0.0, bal - principal)

        rows.append({
            "vcode": loan.vcode,
            "LoanID": loan.loan_id,
            "event_date": dte,
            "rate": r_annual,
            "interest": float(interest),
            "principal": float(principal),
            "payment": float(payment),
            "ending_balance": float(bal),
        })

        if bal <= 0.0:
            break

    sched = pd.DataFrame(rows)
    if sched.empty:
        return sched

    sched = sched[
        (sched["event_date"] >= month_end(schedule_start)) &
        (sched["event_date"] <= month_end(schedule_end))
    ].copy()

    return sched
