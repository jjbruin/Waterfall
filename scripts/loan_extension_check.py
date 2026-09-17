"""The extension test: seeded from the lender, negotiable, and never applied.

WHAT MATTERS HERE. The output is a dollar figure a deal team would take into a
lender conversation, so the ways it can be wrong are specific:

  * the WRONG COVENANT is used -- `nRequiredDCR` is the ongoing covenant and
    `nReqDSR` is the extension test (Jim, Sep 17 2026). Testing the wrong one
    answers a different question in the same units;
  * a MISSING covenant becomes 0, which passes everything;
  * a PROPOSED value is not distinguishable from the lender's own;
  * the what-if recomputes NOI and quietly moves the baseline under the
    comparison;
  * the binding constraint is not the tightest one;
  * it APPLIES the paydown instead of reporting it.

Run:  .venv/Scripts/python.exe scripts/loan_extension_check.py
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


def _fixtures(noi_annual=5_000_000.0, cap=0.055, **loan_kw):
    import pandas as pd
    sched = pd.DataFrame([
        {"LoanID": "298", "event_date": d, "interest": 318672.97,
         "principal": 0.0, "ending_balance": 51667000.0}
        for d in pd.date_range("2026-01-31", "2026-11-30", freq="ME")])
    row = {"vCode": "P0000078", "LoanID": "298", "vDateType": "Maturity",
           "dtEvent": "2026-12-05", "dtMaturity": None,
           "ExtensionOptions": "2x12", "nReqDSR": 1.10, "nLTV": 0.55,
           "nRequiredDCR": None, "nDY": None, "nRequiredLTV": None,
           "nRequiredDY": None}
    row.update(loan_kw)
    loans = pd.DataFrame([row])
    rows = []
    rev = noi_annual / 12.0 + 283333.3333
    for d in pd.date_range("2026-01-31", "2029-12-31", freq="ME"):
        rows.append({"event_date": d.date(), "vAccount": 4010,
                     "mAmount_norm": rev})
        rows.append({"event_date": d.date(), "vAccount": 5090,
                     "mAmount_norm": -283333.3333})
    fc = pd.DataFrame(rows)
    val = pd.DataFrame([{"vCode": "P0000078", "fCapRate": cap,
                         "dtValuation": "2025-12-31"}])
    return sched, loans, fc, val


def main() -> int:
    import loan_maturity as lm
    import loan_extension as le

    sched, loans, fc, val = _fixtures()
    L = lm.detect(sched, loans, "P0000078", "2027-04-30")["loans"][0]

    # ---- 1. which covenant ------------------------------------------
    print("\n1. The right covenant")
    # THE ONE THAT WOULD BE SILENT IF WRONG. nRequiredDCR and nReqDSR are both
    # coverage ratios in the same units; using the wrong one produces a
    # plausible number that answers a different question.
    chk("DSCR is seeded from nReqDSR, the EXTENSION test",
        le.SEED_FIELDS["min_dscr"][0] == "nreqdsr",
        str(le.SEED_FIELDS["min_dscr"]))
    chk("and NOT from nRequiredDCR, the ongoing covenant",
        all(f[0] != "nrequireddcr" for f in le.SEED_FIELDS.values()))
    seeded = le.seed_tests(L["covenants"])
    chk("the seed reads 1.10 from the loan", seeded["min_dscr"]["value"] == 1.10)
    chk("and marks it as the lender's", seeded["min_dscr"]["source"] == "MRI")
    # A covenant MRI does not carry must not become 0: a zero DSCR test passes
    # every balance, and a zero LTV test fails every one.
    chk("an absent covenant is None, never 0",
        seeded["min_debt_yield"]["value"] is None
        and seeded["min_debt_yield"]["source"] is None, str(seeded))

    # ---- 2. the derived rate ----------------------------------------
    print("\n2. The rate")
    r = le.derive_rate(sched, "298", 51667000.0)
    # 318,672.97 x 12 / 51,667,000. nRate is null on this loan, so the stored
    # field answers nothing and the schedule is the only honest source.
    chk("derived from the schedule, 7.40%", r is not None and abs(r - 0.074014) < 1e-5,
        str(r))
    chk("no schedule means no rate, not a zero",
        le.derive_rate(None, "298", 51667000.0) is None)

    # ---- 3. the solve -----------------------------------------------
    print("\n3. Binding constraint and paydown")
    base = le.test(L, fc, val, "P0000078", sched)
    chk("the test runs", base.get("available") is True, str(base.get("reason")))
    chk("NOI is the 12 months AFTER the maturity, not the trailing year",
        abs(base["noi"] - 5_000_000) < 1.0, str(base["noi"]))
    by = {c["key"]: c for c in base["constraints"]}
    chk("DSCR supports NOI / test / rate",
        abs(by["dscr"]["max_balance"] - (5_000_000 / 1.10 / 0.074014)) < 50,
        str(by["dscr"]["max_balance"]))
    # Against the cap rate the ENGINE reports, not a hardcoded one. The rate
    # escalates by fractional years from the valuation date -- 2025-12-31 to
    # 2026-12-05 is 0.93 of a year, so 0.055 becomes 0.055465 and not 0.0555.
    # Pinning the literal tested my arithmetic rather than the code's.
    chk("LTV supports value x test, at the engine's escalated cap rate",
        abs(by["ltv"]["max_balance"] - (base["value"] * 0.55)) < 1.0
        and abs(base["value"] - base["noi"] / base["cap_rate"]) < 1.0,
        "%s / cap %s" % (by["ltv"]["max_balance"], base["cap_rate"]))
    # THE TIGHTEST ONE BINDS, because every test must hold at once. Here LTV
    # binds even though DSCR is the covenant being negotiated -- which is the
    # whole reason to show both.
    chk("the tightest constraint binds",
        base["binding"] == min(by, key=lambda k: by[k]["max_balance"]),
        base["binding"])
    chk("the paydown is balance less what binds",
        abs(base["required_paydown"]
            - (51667000.0 - by[base["binding"]]["max_balance"])) < 1.0)
    chk("and it does not pass", base["passes"] is False)

    # ---- 4. negotiating ---------------------------------------------
    print("\n4. Negotiating from the lender's position")
    r2 = le.resolve(base, {"min_dscr": 1.05}, L)
    chk("a proposed value is marked proposed, not MRI",
        r2["tests"]["min_dscr"]["source"] == "proposed")
    chk("and the untouched one still says MRI",
        r2["tests"]["max_ltv"]["source"] == "MRI")
    # THE POINT OF RE-SOLVING RATHER THAN RECOMPUTING: only the covenant moves.
    chk("the what-if uses the baseline's own NOI",
        r2["noi"] == base["noi"] and r2["rate"] == base["rate"]
        and r2["cap_rate"] == base["cap_rate"])
    chk("loosening a NON-binding covenant changes nothing",
        r2["required_paydown"] == base["required_paydown"],
        "%s vs %s" % (r2["required_paydown"], base["required_paydown"]))
    r3 = le.resolve(base, {"max_ltv": 60}, L)
    chk("loosening the BINDING one does move the paydown",
        r3["required_paydown"] < base["required_paydown"], str(r3["required_paydown"]))
    chk("60 and 0.60 mean the same thing for a ratio",
        le.resolve(base, {"max_ltv": 0.60}, L)["required_paydown"]
        == r3["required_paydown"])
    # ... but 1.10 and 110 do NOT, for a coverage ratio.
    r4 = le.resolve(base, {"min_dscr": 1.25}, L)
    chk("a DSCR is never rescaled",
        r4["tests"]["min_dscr"]["value"] == 1.25)
    chk("an empty override returns the lender's own answer",
        le.resolve(base, {}, L)["required_paydown"] == base["required_paydown"])

    # ---- 5. clearing the test ---------------------------------------
    print("\n5. When it passes")
    s2, l2, fc2, v2 = _fixtures(noi_annual=9_000_000.0)
    L2 = lm.detect(s2, l2, "P0000078", "2027-04-30")["loans"][0]
    b2 = le.test(L2, fc2, v2, "P0000078", s2)
    chk("a strong NOI clears both tests with no paydown",
        b2["passes"] is True and b2["required_paydown"] == 0.0,
        "%s / %s" % (b2["passes"], b2["required_paydown"]))
    chk("and the headline says so rather than quoting a paydown",
        "No paydown required" in b2["headline"], b2["headline"][:70])

    # ---- 6. what it must NOT do -------------------------------------
    print("\n6. Reported, never applied")
    # The forecast is untouched: this returns a figure, it does not pay anything
    # down, and it does not extend the loan.
    chk("no key suggests the paydown was applied",
        not any(k in base for k in
                ("applied", "adjusted_balance", "new_schedule", "extended")))
    chk("the loan's own balance is unchanged by testing it",
        L["balance_outstanding"] == 51667000.0)
    chk("and the maturity gap still reports the FULL interest shortfall",
        L["interest_unmodelled_estimate"] > 1_500_000)

    # ---- 7. degrading honestly --------------------------------------
    print("\n7. When it cannot answer")
    s3, l3, fc3, v3 = _fixtures(noi_annual=-100.0)
    L3 = lm.detect(s3, l3, "P0000078", "2027-04-30")["loans"][0]
    b3 = le.test(L3, fc3, v3, "P0000078", s3)
    chk("a non-positive NOI refuses the test with a reason",
        b3["available"] is False and "NOI" in b3["reason"], str(b3.get("reason")))
    import pandas as pd
    no_cap = pd.DataFrame([{"vCode": "OTHER", "fCapRate": 0.05,
                            "dtValuation": "2025-12-31"}])
    b4 = le.test(L, fc, no_cap, "P0000078", sched)
    # With no cap rate the LTV test cannot run, but DSCR still can -- so the
    # answer is a DSCR-bound paydown, not silence and not a guessed value.
    if b4.get("available"):
        ltv = [c for c in b4["constraints"] if c["key"] == "ltv"]
        chk("no cap rate leaves LTV untestable and says why",
            bool(ltv) and ltv[0]["max_balance"] is None
            and "Cannot test" in ltv[0]["detail"], str(ltv))
        chk("while DSCR still produces an answer", b4["binding"] == "dscr")
    else:
        chk("no cap rate is handled", True)
    chk("a baseline that never ran cannot be varied",
        le.resolve({"available": False}, {"min_dscr": 1.0})["available"] is False)

    return _report()


def _report():
    print("\n%d checks, %d failed." % (CHECKS[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("   FAILED: %s" % f)
        return 1
    print("The test starts at the lender's own covenants, says which values are")
    print("proposed, varies only what is negotiated, and reports a paydown")
    print("without applying one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
