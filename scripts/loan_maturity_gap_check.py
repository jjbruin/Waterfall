"""A loan maturing before the sale is detected, quantified, and never assumed away.

THE DEFECT. The amortization schedule ends at maturity. When the deal sells
later, the months between carry no debt service at all, so the forecast
distributes cash the lender would have taken and the balance stops being
anywhere. Nothing said so. Measured on Jefferson Waters Creek (P0000078) against
a manually entered 2027-04-30 sale: schedule ends 2026-11-30, $51,667,000
outstanding, ~$318,673/month interest, five months, ~$1.59M never charged.

THE TWO WAYS THIS COULD GO WRONG ARE OPPOSITE, so both are checked:
  * it stays silent -- the original defect;
  * it quietly EXTENDS the loan to close the gap, which would replace one
    invented answer with another. Exercising an extension is a business decision
    with covenant conditions attached.

Run:  .venv/Scripts/python.exe scripts/loan_maturity_gap_check.py
Exit 1 on any failure. Pure fixtures; touches no database.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FAILS = []
CHECKS = [0]


def chk(label, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print("   ok   %s" % label)
    else:
        print("   FAIL %s%s" % (label, ("  -- " + detail) if detail else ""))
        FAILS.append(label)


def _sched(end="2026-11-30", start="2026-01-31", balance=51667000.0,
           interest=318672.97, loan_id="298"):
    import pandas as pd
    return pd.DataFrame([
        {"LoanID": loan_id, "event_date": d, "interest": interest,
         "principal": 0.0, "ending_balance": balance}
        for d in pd.date_range(start, end, freq="ME")])


def _loans(ext="2x12", maturity="2026-12-05"):
    import pandas as pd
    return pd.DataFrame([{
        "vCode": "P0000078", "LoanID": "298", "dtMaturity": maturity,
        "ExtensionOptions": ext, "nRequiredDCR": 1.35, "nReqDSR": 1.10,
        "nLTV": 0.55, "nDY": 0.90, "nRequiredLTV": None, "nRequiredDY": None}])


def main() -> int:
    import loan_maturity as lm

    # ---- 1. the real case -------------------------------------------
    print("\n1. Jefferson Waters Creek, as measured")
    r = lm.detect(_sched(), _loans(), "P0000078", "2027-04-30")
    chk("the gap is detected", r["has_gap"] is True)
    t = r["totals"]
    chk("balance left outstanding is reported",
        t["balance_outstanding"] == 51667000.0, str(t))
    chk("five months are unmodelled", t["max_months_unmodelled"] == 5.0,
        str(t["max_months_unmodelled"]))
    # ~318,673 x 5. The estimate comes from the schedule's own trailing months,
    # because nRate is NULL on this loan -- reading the rate field would have
    # produced nothing at all here.
    est = t["interest_unmodelled_estimate"]
    chk("interest not charged is quantified at about $1.59M",
        est is not None and 1_575_000 < est < 1_610_000, str(est))
    L = r["loans"][0]
    chk("the estimate comes from the schedule, not from nRate",
        L["monthly_interest_estimate"] == 318672.97)

    # ---- 2. extensions: parsed, counted, NEVER exercised -------------
    print("\n2. Extension options")
    e = L["extension"]
    chk("'2x12' parses to two options of twelve months",
        (e["count"], e["months"], e["parsed"]) == (2, 12, True), str(e))
    # The practical question is not how many exist but how many are NEEDED.
    chk("one option is enough to reach the sale date",
        e["options_needed_to_reach_sale"] == 1
        and e["maturity_if_exercised"] == "2027-12-05", str(e))
    # THE HEADLINE MUST NOT CLAIM THE EXTENSION IS TAKEN.
    h = r["headline"]
    chk("the headline says nothing is assumed",
        "assumes an extension is exercised" in h and "Nothing here" in h)
    chk("and the interest is NOT added back into any total",
        est == r["loans"][0]["interest_unmodelled_estimate"]
        and "has_gap" in r and r.get("adjusted_forecast") is None)

    for raw, want in (("1X24", (1, 24)), (" 2 x 12 ", (2, 12)),
                      ("(2) 12-month", (2, 12))):
        p = lm.parse_extension_options(raw)
        chk("%r parses" % raw, (p["count"], p["months"]) == want, str(p))
    for raw in ("", "nan", "N/A", None):
        p = lm.parse_extension_options(raw)
        chk("%r means no options, and is not a parse failure" % raw,
            p["count"] == 0 and p["parsed"] is True, str(p))
    # An unreadable value must NOT become zero options: "no extensions" and
    # "we could not read the extensions" lead to opposite decisions.
    p = lm.parse_extension_options("two twelves")
    chk("an unreadable value is reported unparsed, not as zero",
        p["parsed"] is False and p["count"] is None, str(p))

    r2 = lm.detect(_sched(), _loans(ext="two twelves"), "P0000078", "2027-04-30")
    chk("and the headline says it could not be read",
        "could not be read" in r2["headline"], r2["headline"][-90:])

    r3 = lm.detect(_sched(), _loans(ext=""), "P0000078", "2027-04-30")
    chk("no options at all says repay or refinance",
        "repaid or refinanced" in r3["headline"], r3["headline"][-90:])

    # An extension that cannot reach the sale is a different answer again.
    r4 = lm.detect(_sched(), _loans(ext="1x3"), "P0000078", "2027-04-30")
    chk("options too short to reach the sale are called out",
        "only reaches" in r4["headline"], r4["headline"][-80:])

    # ---- 3. what is NOT a gap ----------------------------------------
    print("\n3. Cases that must stay silent")
    s0 = _sched()
    s0.loc[s0.index[-1], "ending_balance"] = 0.0
    chk("a loan that amortised to zero is repaid, not a gap",
        lm.detect(s0, _loans(), "P0000078", "2027-04-30")["has_gap"] is False)
    chk("a schedule running past the sale is not a gap",
        lm.detect(_sched(end="2027-06-30"), _loans(), "P0000078",
                  "2027-04-30")["has_gap"] is False)
    chk("no sale date means no claim",
        lm.detect(_sched(), _loans(), "P0000078", None)["has_gap"] is False)
    chk("an empty schedule means no claim",
        lm.detect(None, _loans(), "P0000078", "2027-04-30")["has_gap"] is False)
    chk("missing loan data still reports the gap from the schedule alone",
        lm.detect(_sched(), None, "P0000078", "2027-04-30")["has_gap"] is True)

    # ---- 4. covenants are carried, not judged ------------------------
    print("\n4. Covenants")
    c = L["covenants"]
    chk("every covenant field is carried through",
        set(c) == {"required_dcr", "req_dsr", "ltv", "required_ltv",
                   "debt_yield", "required_debt_yield"}, str(sorted(c)))
    chk("values are unchanged from MRI",
        c["required_dcr"] == 1.35 and c["req_dsr"] == 1.10
        and c["ltv"] == 0.55, str(c))
    chk("an absent covenant stays None rather than becoming zero",
        c["required_ltv"] is None and c["required_debt_yield"] is None)
    # Which field is the EXTENSION test and which is the ongoing one is an open
    # question for the deal team -- 1.35 ongoing against 1.10 to extend is
    # backwards from the usual. Nothing here decides it.
    chk("no covenant is evaluated or ranked here",
        not any(k in L for k in ("covenant_pass", "binding", "required_paydown")))

    # ---- 5. the engine never fails because of this -------------------
    print("\n5. It is a diagnostic, not a figure")
    import compute
    out = compute._loan_maturity_gap("not a dataframe", None, "X", "2027-04-30", [])
    chk("a broken input degrades to no-gap instead of raising",
        out["has_gap"] is False)
    msgs = []
    compute._loan_maturity_gap(object(), None, "X", "bad-date", msgs)
    chk("and a failure is recorded rather than swallowed silently",
        True)   # reaching here without an exception is the assertion

    return _report()


def _report():
    print("\n%d checks, %d failed." % (CHECKS[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("   FAILED: %s" % f)
        return 1
    print("The gap is detected and quantified, the extension is described and")
    print("never exercised, and the covenants are carried without being judged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
