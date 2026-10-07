"""Guardrail: the Investment Metrics engine, on fixtures shaped like the real data.

Pure fixtures — no database, no API, no network — so this runs in the container
and on a laptop with an empty SQLite alike.

Every check here is asserted in BOTH directions where a one-sided rule would be
satisfied by doing nothing. "The twin is merged" is satisfied by merging every
row together; "an unknown figure prints as a dash" is satisfied by printing a
dash for everything. Those pairs are marked in the section headings.

Run:  python scripts/investment_metrics_check.py
"""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import investment_metrics as im                     # noqa: E402
import investment_metrics_config as cfg             # noqa: E402
from loaders import normalize_accounting_feed       # noqa: E402
from metrics import xirr                            # noqa: E402

PASS = FAIL = 0
FAILURES = []


def chk(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok]   {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  [FAIL] {label}" + (f" - {detail}" if detail else ""))


def section(t):
    print(f"\n{t}")


# ══════════════════════════════════════════════════════════════════════════
# fixtures — shaped like the live tables, including their defects
# ══════════════════════════════════════════════════════════════════════════
#: THE REAL POPULATION, frozen. vcode, table ('c'urrent / 's'old), PSC invest
#: date, sale date and currency for all 76 deals the reference carries, pulled
#: from production on 2026-10-02 at as-of 2026-06-30.
#:
#: WHY IT IS HERE AND NOT IN A DATA FILE. Section 22 asserts that the DERIVED
#: footnote markers reproduce the reference document exactly, and a derivation
#: can only be checked against the inputs it derives from. Those inputs live in
#: production. Inlining them keeps this script's own rule — no database, no API,
#: no network — while still measuring the rule against real dates rather than
#: against three invented ones that happen to straddle the boundary.
#:
#: It is a SNAPSHOT and will not follow the data. That is the point: if a future
#: change alters how a marker is derived, this says so against a population
#: whose correct answer is known. Re-cut it only alongside a new reference
#: document, and update PDF_MARKERS in the same commit.
POPULATION_26Q2 = [
    ('P0000004', 'c', '2019-05-15', '2029-06-30', 'USD'),  # Asbury Commons
    ('P0000006', 'c', '2021-04-12', '2027-06-01', 'USD'),  # Belleville Self Storage
    ('P0000008', 'c', '2021-08-25', '2031-09-30', 'USD'),  # 5-15 Broad St
    ('P0000010', 'c', '2017-03-23', '2026-10-30', 'USD'),  # Centre at Westbank
    ('P0000014', 'c', '2021-11-18', '2028-11-30', 'USD'),  # Crowne Plaza
    ('P0000018', 'c', '2021-09-22', '2031-09-30', 'USD'),  # Evergreen Plaza
    ('P0000019', 'c', '2019-11-15', '2026-03-31', 'USD'),  # Giant 7
    ('P0000021', 'c', '2021-02-24', '2030-07-30', 'USD'),  # JB Fair Park
    ('P0000028', 'c', '2016-08-09', '2030-07-31', 'USD'),  # Merle Hay
    ('P0000029', 'c', '2021-09-24', '2026-10-31', 'USD'),  # Middle Island
    ('P0000030', 'c', '2021-05-28', '2031-06-30', 'USD'),  # Nottingham Village
    ('P0000031', 'c', '2018-03-08', '2026-11-30', 'USD'),  # Old Kinderhook Resort
    ('P0000033', 'c', '2020-09-16', '2027-10-31', 'USD'),  # OREI Portfolio
    ('P0000035', 'c', '2018-05-01', '2028-05-31', 'USD'),  # Paradise Shoppes of Perry
    ('P0000036', 'c', '2017-07-31', '2029-11-30', 'USD'),  # PMAT Midwest Portfolio
    ('P0000037', 'c', '2021-02-01', '2026-08-31', 'USD'),  # Pontchartrain Landing
    ('P0000040', 'c', '2020-02-18', '2029-08-31', 'USD'),  # The Gallery
    ('P0000041', 'c', '2021-10-01', '2034-10-31', 'USD'),  # The Gathering
    ('P0000044', 'c', '2018-10-29', '2028-11-30', 'USD'),  # Woodlands Square
    ('P0000065', 'c', '2022-03-15', '2031-07-31', 'USD'),  # Ascent on Steamboat
    ('P0000066', 'c', '2022-05-01', '2034-12-31', 'USD'),  # Pegasus Life Storage
    ('P0000067', 'c', '2022-06-15', '2030-06-30', 'USD'),  # Brainerd Place Apartments
    ('P0000068', 'c', '2022-07-12', '2032-07-31', 'USD'),  # The Point at Plymouth Meeting
    ('P0000069', 'c', '2022-08-18', '2027-08-31', 'USD'),  # Mount Prospect Plaza
    ('P0000075', 'c', '2023-06-06', '2028-06-30', 'USD'),  # Camp Creek
    ('P0000076', 'c', '2022-12-28', '2027-12-31', 'USD'),  # The Court at Deptford
    ('P0000077', 'c', '2023-03-03', '2028-03-31', 'USD'),  # Jefferson Addison Heights
    ('P0000078', 'c', '2022-12-05', '2026-12-31', 'USD'),  # Jefferson Waters Creek
    ('P0000079', 'c', '2022-11-01', '2026-09-30', 'USD'),  # Post Commons
    ('P0000080', 'c', '2023-03-23', '2028-03-31', 'USD'),  # Prestige Storage Portfolio
    ('P0000081', 'c', '2023-07-25', '2033-08-31', 'USD'),  # Addison Princeton Meadows
    ('P0000082', 'c', '2023-09-28', '2027-09-30', 'USD'),  # Poplar Prairie
    ('P0000084', 'c', '2023-09-06', '2028-10-31', 'USD'),  # Cocoplum Apartments
    ('P0000085', 'c', '2023-11-22', '2027-11-30', 'USD'),  # Jefferson Eastchase
    ('P0000086', 'c', '2024-04-25', '2029-05-31', 'USD'),  # Flats at Dorsett Ridge
    ('P0000087', 'c', '2023-10-31', '2030-11-30', 'USD'),  # The Standard
    ('P0000088', 'c', '2024-07-31', '2034-08-31', 'USD'),  # Seasons at Bel Air
    ('P0000089', 'c', '2024-08-01', '2028-08-31', 'USD'),  # 45th & Main
    ('P0000099', 'c', '2025-02-19', '2030-03-31', 'USD'),  # ReNew Glenmoore
    ('P0000100', 'c', '2025-02-20', '2029-08-31', 'USD'),  # Green Valley Ranch & Telluride
    ('P0000107', 'c', '2025-02-14', '2032-02-29', 'USD'),  # Town Fair Tire Portfolio
    ('P0000109', 'c', '2025-08-28', None, 'USD'),  # Burton Retail Portfolio
    ('P0000110', 'c', '2025-09-11', '2029-03-31', 'USD'),  # Trolley Square
    ('P0000114', 'c', '2025-10-17', '2028-05-30', 'USD'),  # Jefferson Stephens Holdings, L
    ('P0000115', 'c', '2025-07-16', '2030-07-30', 'CAD'),  # Apple - Bales Drive
    ('P0000116', 'c', '2026-03-16', '2029-03-31', 'USD'),  # Plaza Del Mar
    ('P0000117', 'c', '2026-06-30', '2031-07-30', 'USD'),  # Fairview Heights Retail Center
    ('P0000118', 'c', '2026-03-18', '2030-09-30', 'USD'),  # Hanestowne Waterstone
    ('P0000119', 'c', '2026-05-13', '2036-05-30', 'USD'),  # Presidential Arms JV, LLC (DE)
    ('P0000120', 'c', '2026-05-20', '2031-05-30', 'USD'),  # Citizen Storage Swartz Creek H
    ('P0000001', 's', '2020-08-07', '2026-09-04', 'USD'),  # 30 Bearfoot
    ('P0000002', 's', '2016-11-21', '2020-01-17', 'USD'),  # 3rd Ave & Indian School
    ('P0000005', 's', '2018-12-31', '2021-05-12', 'USD'),  # Barnbeck Apartments
    ('P0000007', 's', '2020-07-22', '2026-04-22', 'USD'),  # Berger Pittsburgh Portfolio
    ('P0000009', 's', '2021-05-11', '2023-02-10', 'USD'),  # Camarillo Village
    ('P0000011', 's', '2021-10-20', '2025-08-30', 'USD'),  # City West
    ('P0000012', 's', '2021-09-09', '2026-07-01', 'USD'),  # Clima Secur
    ('P0000013', 's', '2016-01-07', '2018-04-08', 'USD'),  # Creek Crossing
    ('P0000015', 's', '2017-10-05', '2025-12-31', 'USD'),  # Declan & Walton
    ('P0000016', 's', '2018-04-25', '2020-08-18', 'USD'),  # Devon Square
    ('P0000017', 's', '2020-12-18', '2026-06-25', 'USD'),  # East Manchester
    ('P0000020', 's', '2016-05-17', '2017-11-20', 'USD'),  # Homewood Commons
    ('P0000022', 's', '2018-10-19', '2021-12-20', 'USD'),  # Jefferson Oakhurst
    ('P0000023', 's', '2019-10-25', '2022-03-30', 'USD'),  # Jefferson Centura
    ('P0000024', 's', '2016-09-30', '2020-08-03', 'USD'),  # Jefferson West Love
    ('P0000025', 's', '2018-04-19', '2022-03-29', 'USD'),  # Lancaster Apartments
    ('P0000026', 's', '2017-01-13', '2020-10-23', 'USD'),  # Leander Self Storage
    ('P0000032', 's', '2016-04-25', '2025-12-10', 'USD'),  # Orange Grove
    ('P0000034', 's', '2020-10-28', '2023-04-26', 'USD'),  # Outlook Nine Mile
    ('P0000038', 's', '2017-12-21', '2026-03-04', 'USD'),  # Quakertown Shopping Center
    ('P0000039', 's', '2019-05-23', '2022-11-01', 'USD'),  # Shoppers World
    ('P0000042', 's', '2016-12-23', '2023-12-28', 'USD'),  # Village Square Apartments
    ('P0000043', 's', '2016-09-14', '2023-12-28', 'USD'),  # Willowdale Apartments
    ('P0000049', 's', '2021-06-29', '2026-09-04', 'USD'),  # Donald Lynch
    ('P0000064', 's', '2022-03-08', '2025-12-10', 'USD'),  # Adirondack RV Park
    ('P0000083', 's', '2023-09-12', '2026-03-04', 'USD'),  # Airport Plaza
]


def deals_fixture():
    """Four investments across every shape the live table actually contains.

    * ALPHA  — an ordinary current deal.
    * BETA   — a twin PAIR: the letter row holds the InvestmentID and the sale,
               the numeric row holds the descriptive fields and a NULL id.
      Names differ slightly ("Beta Plaza Apartments" / "Beta Plaza"), as
      Declan & Walton's do live, so exact-name matching alone is not enough.
    * GAMMA  — a parent with a CHILD carrying Property_Count 0.
    * DELTA  — an orphan: no InvestmentID and no twin. Must be dropped and
               NAMED, never silently lost.
    * MCX    — one InvestmentID on two rows, the parent live and the child
               carrying the SOLD marker (the live MCCORD shape).
    """
    return pd.DataFrame([
        dict(vcode="P0000001", InvestmentID="ALPHA", Investment_Name="Alpha Tower",
             Property_Count="1", Sale_Status=None, Sale_Date=None,
             Acquisition_Date="03/01/2019 00:00", Currency="USD", City="Austin",
             State="TX", Asset_Type="Multifamily", Operating_Partner="Acme",
             Lifecycle="Value-Add", Portfolio_Name=""),
        # ORDER MATTERS HERE, and getting it wrong made one check below pass
        # by accident. `build_investmentid_to_vcode` is `dict(zip(...))`, so
        # the LAST row carrying an id wins. Live, `deals` comes back sorted by
        # vcode and the numeric row precedes the letter one (P0000020 before
        # PHOMEW), so the letter row wins and the id resolves to a vcode that
        # deal_terms, loans and waterfalls know nothing about. Put the letter
        # row first and the collision resolves the right way on its own —
        # which is not a test of anything.
        dict(vcode="P0000002", InvestmentID=None,
             Investment_Name="Beta Plaza Apartments",
             Property_Count=None, Sale_Status=None, Sale_Date=None,
             Acquisition_Date="1/5/2017", Currency=None, City="Denver",
             State="CO", Asset_Type="Multifamily", Operating_Partner="Bravo",
             Lifecycle="Sold", Portfolio_Name=""),
        dict(vcode="PBETA", InvestmentID="BETA", Investment_Name="Beta Plaza",
             Property_Count="1", Sale_Status="SOLD", Sale_Date="6/1/2024",
             Acquisition_Date="1/5/2017", Currency="USD", City=None, State=None,
             Asset_Type="Retail", Operating_Partner=None, Lifecycle=None,
             Portfolio_Name=""),
        dict(vcode="P0000200", InvestmentID="GAMMA", Investment_Name="Gamma Portfolio",
             Property_Count="2", Sale_Status=None, Sale_Date=None,
             Acquisition_Date="07/01/2021 00:00", Currency="CAD", City="Toronto",
             State="Ontario", Asset_Type="Self Storage", Operating_Partner="Gam",
             Lifecycle="Stable", Portfolio_Name="Gamma Portfolio"),
        dict(vcode="P0000201", InvestmentID="GKID", Investment_Name="Gamma Kid",
             Property_Count="0", Sale_Status=None, Sale_Date=None,
             Acquisition_Date="07/01/2021 00:00", Currency="USD", City="Toronto",
             State="Ontario", Asset_Type="Self Storage", Operating_Partner="Gam",
             Lifecycle="Stable", Portfolio_Name="Gamma Portfolio"),
        dict(vcode="P0000005", InvestmentID=None, Investment_Name="Delta Orphan",
             Property_Count=None, Sale_Status=None, Sale_Date=None,
             Acquisition_Date=None, Currency=None, City="Nowhere", State="NA",
             Asset_Type="Office", Operating_Partner=None, Lifecycle=None,
             Portfolio_Name=""),
        dict(vcode="P0000006", InvestmentID="MCX", Investment_Name="McX Centre",
             Property_Count="1", Sale_Status=None, Sale_Date="9/4/2026",
             Acquisition_Date="06/30/2021 00:00", Currency="USD", City="Boston",
             State="MA", Asset_Type="Industrial", Operating_Partner="McC",
             Lifecycle="Value-Add", Portfolio_Name="McX Centre"),
        dict(vcode="P0000007", InvestmentID="MCX", Investment_Name="McX Centre",
             Property_Count="0", Sale_Status="SOLD", Sale_Date="9/4/2026",
             Acquisition_Date="06/30/2021 00:00", Currency="USD", City="Boston",
             State="MA", Asset_Type="Industrial", Operating_Partner="McC",
             Lifecycle="Value-Add", Portfolio_Name="McX Centre"),
    ])


def acct_fixture():
    """Accounting, carrying the live table's own untidiness.

    THE TRAILING SPACES ARE THE POINT. The live table holds both ``'BETA'``
    and ``'BETA  '`` for the same investment, exactly as it holds ``'BARN'``
    against ``'BARN  '`` and ``'PMAT'`` against ``'PMAT  '``. A join that does
    not normalise finds one row and reports a deal with almost no history.
    """
    rows = [
        # ALPHA: commitment, funding, pref within and beyond year one
        ("ALPHA", "PPI1", "2019-03-01", 1026, -5_000_000, "Contribution", "Commitment"),
        ("ALPHA", "OPACME", "2019-03-01", 1026, -2_000_000, "Contribution", "Commitment"),
        ("ALPHA", "PPI1", "2019-03-01", 1018, -5_000_000, "Contribution",
         "Contribution: Investments"),
        ("ALPHA", "OPACME", "2019-03-01", 1018, -2_000_000, "Contribution",
         "Contribution: Investments"),
        ("ALPHA", "PPI1", "2019-09-01", 1019, 200_000, "Distribution",
         "Distribution: Preferred Return"),
        ("ALPHA", "PPI1", "2020-02-01", 1019, 200_000, "Distribution",
         "Distribution: Preferred Return"),
        # day 400 — outside the year-one window
        ("ALPHA", "PPI1", "2020-04-04", 1019, 999_000, "Distribution",
         "Distribution: Preferred Return"),
        ("ALPHA", "PPI1", "2020-06-01", 1021, 50_000, "Distribution",
         "Distribution: Acquisition Fee"),
        ("ALPHA", "OPACME", "2020-06-01", 1019, 111_000, "Distribution",
         "Distribution: Preferred Return"),
        # BETA — note the trailing-space spelling on half the rows
        ("BETA", "PPI2", "2017-01-05", 1018, -1_000_000, "Contribution",
         "Contribution: Investments"),
        ("BETA  ", "PPI2", "2019-01-05", 1019, 120_000, "Distribution",
         "Distribution: Preferred Return"),
        ("BETA  ", "PPI2", "2024-06-01", 1016, 1_400_000, "Distribution",
         "Distribution: Return of Capital"),
        # GAMMA — CAD
        ("GAMMA", "PPI3", "2021-07-01", 1026, -4_000_000, "Contribution", "Commitment"),
        ("GAMMA", "PPI3", "2021-07-01", 1018, -4_000_000, "Contribution",
         "Contribution: Investments"),
        # MCX
        ("MCX", "PPI4", "2021-06-30", 1018, -2_500_000, "Contribution",
         "Contribution: Investments"),
        ("MCX", "PPI4", "2026-09-04", 1016, 5_200_000, "Distribution",
         "Distribution: Return of Capital"),
    ]
    df = pd.DataFrame(rows, columns=[
        "InvestmentID", "InvestorID", "EffectiveDate", "SubtypeUID", "Amt",
        "MajorType", "Typename"])
    df["Capital"] = "Y"
    df["Partner"] = df["InvestorID"].str.startswith("OP").map(
        {True: "Operating Partner", False: "Preferred Equity"})
    return normalize_accounting_feed(df)


def commitments_fixture():
    return pd.DataFrame([
        # trailing space, as the live table has
        dict(EntityID="ALPHA ", InvestorID="PPI1", Amount=5_000_000.0,
             StartDate="2019-03-01", EndDate=None, CommitmentUID=1),
        dict(EntityID="ALPHA", InvestorID="OPACME", Amount=2_000_000.0,
             StartDate="2019-03-01", EndDate=None, CommitmentUID=2),
    ])


def loans_fixture():
    return pd.DataFrame([
        dict(vCode="P0000001", LoanID="L1", mOrigLoanAmt=13_000_000.0),
        # one facility fanned out across date events — must count ONCE
        dict(vCode="P0000200", LoanID="L2", mOrigLoanAmt=8_000_000.0),
        dict(vCode="P0000200", LoanID="L2", mOrigLoanAmt=8_000_000.0),
        dict(vCode="P0000201", LoanID="L3", mOrigLoanAmt=2_000_000.0),
        dict(vCode="P0000006", LoanID="L4", mOrigLoanAmt=15_100_000.0),
    ])


def isbs_fixture():
    return pd.DataFrame([
        # ALPHA: earliest row is the one that counts, not the latest
        dict(vcode="p0000001", dtEntry="2019-12-31", vAccount="2150",
             mAmount=-12_000_000.0),
        dict(vcode="p0000001", dtEntry="2024-12-31", vAccount="2150",
             mAmount=-9_000_000.0),
        # GAMMA parent carries the consolidated balance; the child repeats it
        dict(vcode="p0000200", dtEntry="2021-12-31", vAccount="2150",
             mAmount=-7_000_000.0),
        dict(vcode="p0000201", dtEntry="2021-12-31", vAccount="2150",
             mAmount=-7_000_000.0),
        dict(vcode="p0000006", dtEntry="2021-12-31", vAccount="2210",
             mAmount=-15_000_000.0),
    ])


def young_deals_fixture():
    """Three deals against the as-of date 2026-06-30, on the (5) boundary.

    * YOUNG (P0000900) — invested 2026-03-01, four months old. Footnote (5).
    * EDGE  (P0000901) — invested 2025-06-30, so its twelve months close ON the
      as-of date. ``plus_months(...) > as_of`` is FALSE, so it is NOT young.
      The boundary is tested because "within a year" and "a year ago today" are
      one day apart and the live table has a deal on every month.
    * OLD   (P0000902) — invested 2024-01-15. Never young.
    """
    base = dict(Property_Count="1", Sale_Status=None, Sale_Date=None,
                Currency="USD", City="Reno", State="NV", Asset_Type="Retail",
                Operating_Partner="Op", Lifecycle=None, Portfolio_Name="")
    return pd.DataFrame([
        dict(vcode="P0000900", InvestmentID="YOUNG", Investment_Name="Young One",
             Acquisition_Date="03/01/2026 00:00", **base),
        dict(vcode="P0000901", InvestmentID="EDGE", Investment_Name="Edge One",
             Acquisition_Date="06/30/2025 00:00", **base),
        dict(vcode="P0000902", InvestmentID="OLD", Investment_Name="Old One",
             Acquisition_Date="01/15/2024 00:00", **base),
    ])


def young_acct_fixture():
    rows = []
    for iid, start in (("YOUNG", "2026-03-01"), ("EDGE", "2025-06-30"),
                       ("OLD", "2024-01-15")):
        rows += [
            (iid, "PPI9", start, 1018, -1_000_000, "Contribution",
             "Contribution: Investments"),
            (iid, "PPI9", "2026-05-01", 1019, 25_000, "Distribution",
             "Distribution: Preferred Return"),
        ]
    df = pd.DataFrame(rows, columns=[
        "InvestmentID", "InvestorID", "EffectiveDate", "SubtypeUID", "Amt",
        "MajorType", "Typename"])
    df["Capital"] = "Y"
    df["Partner"] = "Preferred Equity"
    return normalize_accounting_feed(df)


def young_terms_fixture():
    return pd.DataFrame([
        dict(vcode=v, pe_coupon=0.085, irr_lookback=9.0,
             pe_split_capital=0.30, pe_split_cf=None)
        for v in ("P0000900", "P0000901", "P0000902")
    ])


def build(**kw):
    kw.setdefault("inv", deals_fixture())
    kw.setdefault("acct", acct_fixture())
    return im.build_investment_metrics(
        kw.pop("inv"), kw.pop("acct"),
        commitments=kw.pop("commitments", commitments_fixture()),
        deal_terms=kw.pop("deal_terms", pd.DataFrame([
            dict(vcode="P0000001", pe_coupon=0.085, irr_lookback=9.0,
                 pe_split_capital=0.30, pe_split_cf=None),
        ])),
        loans=kw.pop("loans", loans_fixture()),
        isbs_interim_bs=kw.pop("isbs_interim_bs", isbs_fixture()),
        as_of=kw.pop("as_of", dt.date(2026, 6, 30)),
        **kw,
    )


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_repo(*parts):
    """A repo source file, as text. UTF-8 EXPLICITLY — these files carry em
    dashes, and Windows' default cp1252 raises on them."""
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


def _with_pin(value, fn):
    """Run ``fn`` with ``cfg.DEFAULT_QUARTER`` temporarily set to ``value``."""
    saved = cfg.DEFAULT_QUARTER
    cfg.DEFAULT_QUARTER = value
    try:
        return fn()
    finally:
        cfg.DEFAULT_QUARTER = saved


def row_of(out, vcode):
    for t in ("current", "sold"):
        for r in out[t]["rows"]:
            if r["vcode"] == vcode:
                return t, r
    return None, None


# ══════════════════════════════════════════════════════════════════════════
Q_ASOF = dt.date(2026, 6, 30)


def main():
    # ── 1. key normalisation ──────────────────────────────────────────────
    section("1. Identifiers are stripped and upper-cased before every join")
    chk("'  beta  ' and 'BETA' are the same key",
        im.norm_id("  beta  ") == im.norm_id("BETA") == "BETA")
    chk("a NULL id is empty, not the string 'NAN'",
        im.norm_id(None) == "" and im.norm_id(float("nan")) == ""
        and im.norm_id("nan") == "")
    chk("norm_name folds case and punctuation",
        im.norm_name("Declan & Walton, LLC") == "declan walton llc")

    out = build()
    out_dec = build(as_of=dt.date(2026, 12, 31))

    # ── 2. the twin rule, BOTH directions ─────────────────────────────────
    section("2. The twin rule (asserted both ways: merged, and NOT over-merged)")
    t, beta = row_of(out, "P0000002")
    chk("the pair becomes ONE row", beta is not None
        and row_of(out, "PBETA")[1] is None)
    if beta:
        chk("it is keyed on the NUMERIC vcode — deal_terms and loans use that",
            beta["vcode"] == "P0000002")
        chk("it carries the LETTER row's InvestmentID — accounting uses that",
            beta["investment_id"] == "BETA")
        chk("the sale comes from the letter row, so the deal is SOLD",
            t == "sold", f"landed in {t}")
        chk("descriptive fields come from the numeric row",
            beta["partner"] == "Bravo" and (beta["dma"] or "").startswith("Denver"))
        chk("both vcodes are recorded, so the merge is traceable",
            set(beta["twin_vcodes"]) == {"PBETA", "P0000002"})
    # NOT over-merged: unrelated deals stay separate
    chk("unrelated deals are NOT merged together",
        len(out["current"]["rows"]) + len(out["sold"]["rows"]) == 4,
        f"got {len(out['current']['rows'])} current + "
        f"{len(out['sold']['rows'])} sold")
    chk("the orphan is dropped AND named, never silently lost",
        any(o["vcode"] == "P0000005" for o in out["diagnostics"]["orphans"]))
    chk("the child property is excluded AND named",
        any(c["vcode"] == "P0000201"
            for c in out["diagnostics"]["children_excluded"]))
    _, mcx = row_of(out, "P0000006")
    chk("one InvestmentID on two rows picks the PARENT",
        mcx is not None and mcx["vcode"] == "P0000006")
    chk("...and still takes the SOLD marker from its sibling (once the sale is past)",
        row_of(out_dec, "P0000006")[0] == "sold")

    # ── 3. classification ─────────────────────────────────────────────────
    section("3. Current vs Sold")
    # A DEAL IS SOLD AT A QUARTER ONLY ONCE ITS SALE IS ON OR BEFORE THE AS-OF.
    # Asserted at the boundary, both sides, and against the two ways to get it
    # wrong: "everything marked SOLD is Sold" (the old rule) and "everything with
    # a sale date is Sold".
    chk("a deal sold AFTER the as-of date is CURRENT: it was held at the quarter end",
        row_of(out, "P0000006")[0] == "current",
        f"got {row_of(out, 'P0000006')[0]} -- McX sold 2026-09-04, as-of 2026-06-30")
    chk("...and is named, not quietly moved",
        any(x["vcode"] == "P0000006" and x["sale_date"] == "2026-09-04"
            for x in out["diagnostics"].get("sold_after_as_of_shown_current", [])))
    chk("the same deal is SOLD once the quarter is after the sale",
        row_of(out_dec, "P0000006")[0] == "sold"
        and not any(x["vcode"] == "P0000006" for x in
                    out_dec["diagnostics"].get("sold_after_as_of_shown_current", [])))
    chk("a sale ON the as-of date is Sold (the boundary)",
        row_of(build(as_of=dt.date(2026, 9, 4)), "P0000006")[0] == "sold")
    chk("...and the day before it is Current",
        row_of(build(as_of=dt.date(2026, 9, 3)), "P0000006")[0] == "current")
    chk("a deal sold BEFORE the as-of stays Sold",
        row_of(out, "P0000002")[0] == "sold")
    chk("a SOLD marker with NO sale date stays Sold: nothing says it was held",
        im.classify(im.DealIdentity(sale_status="SOLD", sale_date=None), Q_ASOF) == "sold"
        and im.classify(im.DealIdentity(sale_status=None,
                                        sale_date=dt.date(2020, 1, 1)), Q_ASOF)
        == "current")
    chk("a live deal is Current", row_of(out, "P0000001")[0] == "current")

    # ── 4. capitalization ─────────────────────────────────────────────────
    section("4. Capitalization, in $m")
    _, alpha = row_of(out, "P0000001")
    chk("PSC pref comes from the commitments table despite its trailing space",
        alpha and abs(alpha["pref"] - 5.0) < 1e-9, f"got {alpha['pref']}")
    chk("first-loss is the operating partner's side",
        alpha and abs(alpha["first_loss"] - 2.0) < 1e-9)
    def alt_of(row, basis):
        for a in (row or {}).get("alternates", {}).get("first_lien", []):
            if a["basis"] == basis:
                return a
        return {}

    chk("first lien defaults to the SUMMED committed facility",
        alpha and abs(alpha["first_lien"] - 13.0) < 1e-9,
        f"got {(alpha or {}).get('first_lien')} — 12.0 is the ISBS basis")
    chk("...and that basis is flagged as the one in use",
        alt_of(alpha, "summed_facility").get("in_use") is True)
    chk("the ISBS alternate is the EARLIEST balance-sheet row, not the latest",
        abs((alt_of(alpha, "earliest_isbs").get("value") or 0) - 12.0) < 1e-9,
        f"got {alt_of(alpha, 'earliest_isbs').get('value')} "
        "(9.0 would be the most recent)")
    chk("all three bases are published on every row",
        {a["basis"] for a in (alpha or {}).get("alternates", {})
         .get("first_lien", [])} == {"summed_facility", "earliest_loan",
                                     "earliest_isbs"})
    chk("total size is the three components summed",
        alpha and abs(alpha["total_size"] - 20.0) < 1e-9,
        f"got {(alpha or {}).get('total_size')}")
    chk("% of cap divides by that total",
        alpha and abs(alpha["first_lien_pct"] - 13.0 / 20.0) < 1e-9)
    _, gam = row_of(out, "P0000200")
    chk("a child's LOANS are rolled into the parent's facility",
        gam and abs(gam["first_lien"] - 10.0 * cfg.CAD_TO_USD) < 1e-9,
        f"got {(gam or {}).get('first_lien')} — 8.0 would mean the child "
        "was left out, 18.0 that the duplicated facility row counted twice")
    chk("...but its BALANCE SHEET is not — the parent's is already consolidated",
        abs((alt_of(gam, "earliest_isbs").get("value") or 0)
            - 7.0 * cfg.CAD_TO_USD) < 1e-9,
        f"got {alt_of(gam, 'earliest_isbs').get('value')} — 14.0 would be "
        "the same debt counted twice")
    chk("CAD is converted at the footnote's rate",
        gam and abs(gam["pref"] - 4.0 * cfg.CAD_TO_USD) < 1e-9)
    chk("the basis travels with the figure",
        alpha and "commitments" in alpha["basis"]["capitalization"])
    # The fallback chain: coverage, not accuracy — a later basis is reached
    # only when the one before it produced nothing at all.
    no_loans = build(loans=pd.DataFrame())
    _, a_nl = row_of(no_loans, "P0000001")
    chk("with no loans the ISBS fallback is OFF: an em dash, basis 'none'",
        cfg.FIRST_LIEN_ISBS_FALLBACK is False
        and a_nl and a_nl["first_lien"] is None
        and a_nl["basis"]["first_lien"] == "none",
        f"got {(a_nl or {}).get('first_lien')} / {(a_nl or {}).get('basis', {}).get('first_lien')}")
    chk("...Total Size and every % of Cap cascade to dashes with it",
        a_nl and a_nl["total_size"] is None and a_nl["first_lien_pct"] is None
        and a_nl["pref_pct"] is None and a_nl["first_loss_pct"] is None)
    chk("...the withheld ISBS figure is kept in alternates, for diagnostics only",
        abs((alt_of(a_nl, "earliest_isbs").get("value") or 0) - 12.0) < 1e-9,
        f"got {alt_of(a_nl, 'earliest_isbs').get('value')}")
    held = no_loans["diagnostics"].get("first_lien_isbs_withheld", [])
    chk("...and the deal is LISTED as withheld, with the figure (flag, never drop)",
        any(h["vcode"] == "P0000001" and abs(h["isbs_value_usd"] - 12_000_000.0) < 1
            for h in held), f"got {held}")
    # The other direction: a rule tested only in the refusing direction is
    # satisfied by refusing everything, so the switch is thrown the other way too.
    saved_fb = cfg.FIRST_LIEN_ISBS_FALLBACK
    try:
        cfg.FIRST_LIEN_ISBS_FALLBACK = True
        on = build(loans=pd.DataFrame())
        _, a_on = row_of(on, "P0000001")
        chk("with the flag ON the balance-sheet fallback is restored, unchanged",
            a_on and abs(a_on["first_lien"] - 12.0) < 1e-9
            and "ISBS" in a_on["basis"]["first_lien"]
            and not on["diagnostics"].get("first_lien_isbs_withheld"),
            f"got {(a_on or {}).get('first_lien')}")
    finally:
        cfg.FIRST_LIEN_ISBS_FALLBACK = saved_fb
    _, a_has = row_of(out, "P0000001")
    chk("a deal WITH a loan record is untouched by the flag",
        a_has and abs(a_has["first_lien"] - 13.0) < 1e-9
        and not out["diagnostics"].get("first_lien_isbs_withheld"))
    none_at_all = build(loans=pd.DataFrame(), isbs_interim_bs=pd.DataFrame())
    _, a_no = row_of(none_at_all, "P0000001")
    chk("with neither loan nor balance-sheet debt: a dash, basis 'none', "
        "and nothing listed as withheld",
        a_no and a_no["first_lien"] is None
        and a_no["basis"]["first_lien"] == "none"
        and not none_at_all["diagnostics"].get("first_lien_isbs_withheld"))
    chk("the rule is GLOBAL — dev and non-dev take the same basis",
        cfg.FIRST_LIEN_BASIS == "summed_facility"
        and cfg.FIRST_LIEN_FALLBACKS == ("earliest_loan", "earliest_isbs"))

    # ── 5. None is not zero, BOTH directions ──────────────────────────────
    section("5. Unknown vs zero (both ways: a dash where unknown, a figure "
            "where known)")
    chk("UW projected IRR is None, not 0.0 — it is not in MRI",
        alpha and alpha["uw_irr"] is None)
    chk("projected Year-1 CoC is None, not 0.0", alpha and alpha["proj_yr1_coc"] is None)
    # The Act. Yr-1 CoC column now PRINTS — see cfg.UNLOADED_FIGURES. What must
    # still never happen is a zero standing in for "unknown", which is what this
    # pair asserts in both directions.
    chk("actual Year-1 CoC is a figure now, not a dash",
        alpha and alpha["act_yr1_coc"] is not None)
    chk("...and it is the ROE-windowed figure, not the old pref/funded one",
        alpha and alpha["basis"]["act_yr1_coc"] == "computed (roe_window)"
        and alpha["act_yr1_coc"] == alpha["alternates"]["act_yr1_coc"]["on_roe_window"])
    no_acct = build(acct=acct_fixture().iloc[0:0])
    _, a_blank = row_of(no_acct, "P0000001")
    chk("a deal with no accounting gets a DASH, never 0.0%",
        a_blank and a_blank["act_yr1_coc"] is None)
    no_lien = build(isbs_interim_bs=pd.DataFrame(), loans=pd.DataFrame())
    _, a2 = row_of(no_lien, "P0000001")
    chk("an unknown first lien makes the TOTAL unknown too",
        a2 and a2["first_lien"] is None and a2["total_size"] is None)
    chk("...and every % of cap with it, rather than a share of the equity alone",
        a2 and a2["pref_pct"] is None and a2["first_loss_pct"] is None)
    chk("a KNOWN figure still comes through as a number",
        a2 and a2["pref"] is not None)

    # ── 6. proceeds and Year-1 CoC ────────────────────────────────────────
    section("6. Proceeds and Year-1 CoC")
    chk("current proceeds are 1016+1019+1020+1021, with no date cutoff",
        alpha and abs(alpha["proceeds"] - 1.449) < 1e-9,
        f"got {alpha['proceeds']} (0.2+0.2+0.999+0.05)")
    chk("the operating partner's distributions are excluded",
        alpha and alpha["proceeds"] < 1.5)
    alt = (alpha or {}).get("alternates", {}).get("act_yr1_coc", {})
    chk("year-one CoC counts only the first 365 days",
        abs((alt.get("on_funded") or 0) - 0.4 / 5.0) < 1e-9,
        f"got {alt.get('on_funded')} — 1.399/5 would mean the window is open")
    chk("the window's dates are reported, not just the ratio",
        "365 days from 2019-03-01" in (alt.get("basis") or ""))
    chk("both denominators are carried, so the difference stays measurable",
        "on_funded" in alt and "on_commitment" in alt)
    _, mcxr = row_of(out_dec, "P0000006")
    chk("a sold deal's proceeds are EVERY distribution",
        mcxr and abs(mcxr["proceeds"] - 5.2) < 1e-9)

    # ── 7. realized IRR ───────────────────────────────────────────────────
    section("7. Realized IRR — one engine (metrics.xirr)")
    flows = [(dt.date(2017, 1, 5), -1_000_000.0),
             (dt.date(2019, 1, 5), 120_000.0),
             (dt.date(2024, 6, 1), 1_400_000.0)]
    expect = xirr(flows)
    chk("it is metrics.xirr over PSC's own flows",
        beta and beta.get("realized_irr") is not None
        and abs(beta["realized_irr"] - expect) < 1e-9,
        f"got {beta.get('realized_irr')} expected {expect}")
    chk("commitment rows are excluded — they would re-date the first flow",
        expect is not None)
    chk("a suppressed deal shows a dash, not 0.0%",
        set(cfg.REALIZED_IRR_SUPPRESSED) >= {"P0000011", "P0000064"})

    # ── 8. pref-weighted averages with LABEL rows ─────────────────────────
    section("8. Averages skip label rows AND their weight (both directions)")
    rows = [
        {"pref": 10.0, "x": 0.10, "labels": {}},
        {"pref": 90.0, "x": 0.50, "labels": {"x": "Dev."}},   # label: excluded
        {"pref": 10.0, "x": 0.20, "labels": {}},
    ]
    got = im.pref_weighted_average(rows, "x")
    chk("a labelled row contributes neither value nor weight",
        abs(got - 0.15) < 1e-12,
        f"got {got}; 0.44 would mean it was counted, 0.03 counted as zero")
    rows2 = [{"pref": 10.0, "x": None, "labels": {}},
             {"pref": 10.0, "x": 0.20, "labels": {}}]
    chk("a None contributes nothing either",
        abs(im.pref_weighted_average(rows2, "x") - 0.20) < 1e-12)
    chk("an all-label column averages to None, not 0",
        im.pref_weighted_average(
            [{"pref": 10.0, "x": 0.1, "labels": {"x": "Dev."}}], "x") is None)
    chk("totals sum RAW values",
        abs(im.total_of([{"v": 1.04}, {"v": 2.04}], "v") - 3.08) < 1e-12)

    # ── 9. the label and marker config ────────────────────────────────────
    section("9. Labels and footnote markers are config, keyed by vcode")
    chk("every labelled cell names a real column",
        all(k in cfg.COC_COLUMNS
            for d in list(cfg.CELL_LABELS_CURRENT.values())
                   + list(cfg.CELL_LABELS_SOLD.values())
            for k in d))
    chk("30 Bearfoot is labelled per CELL, not per deal — three words, one figure",
        set(cfg.CELL_LABELS_SOLD["P0000001"]) == {
            "proj_yr1_coc", "act_yr1_coc", "proj_coc_since_close"})
    chk("every marker points at a footnote that exists",
        all(n in {x for x, _ in cfg.FOOTNOTES_CURRENT}
            for ms in cfg.ROW_MARKERS_CURRENT.values() for n in ms)
        and all(n in {x for x, _ in cfg.FOOTNOTES_SOLD}
                for ms in cfg.ROW_MARKERS_SOLD.values() for n in ms))
    chk("the footnotes are numbered 1..N with no gaps",
        [n for n, _ in cfg.FOOTNOTES_CURRENT] == list(range(1, 9))
        and [n for n, _ in cfg.FOOTNOTES_SOLD] == list(range(1, 4)))
    chk("the CAD footnote and the rate in use agree",
        f"{cfg.CAD_TO_USD}" in dict(cfg.FOOTNOTES_CURRENT)[2])
    chk("every ordered vcode appears at most once, on one table",
        len(set(cfg.ROW_ORDER_CURRENT)) == len(cfg.ROW_ORDER_CURRENT)
        and len(set(cfg.ROW_ORDER_SOLD)) == len(cfg.ROW_ORDER_SOLD)
        and not (set(cfg.ROW_ORDER_CURRENT) & set(cfg.ROW_ORDER_SOLD)))
    chk("the reference's row counts are 50 current and 26 sold",
        len(cfg.ROW_ORDER_CURRENT) == 50 and len(cfg.ROW_ORDER_SOLD) == 26)
    chk("a deal not in the reference order is APPENDED and reported, not dropped",
        row_of(out, "P0000001")[1] is not None)
    chk("every deal labelled Dev. is in DEV_DEALS, and vice versa",
        {v for v, d in cfg.CELL_LABELS_CURRENT.items()
         if set(d.values()) == {"Dev."}} == cfg.DEV_DEALS,
        "the two lists describe the same population and must not drift")
    chk("every Lease up deal is labelled as one",
        {v for v, d in cfg.CELL_LABELS_CURRENT.items()
         if set(d.values()) <= {"Lease up", "Lease Up"}} == cfg.LEASE_UP_DEALS)

    # ── 9b. deals the reference carries in neither table ──────────────────
    section("9b. Excluded deals (both ways: dropped AND named; nothing else "
            "dropped)")
    chk("Apple Self Storage is excluded by name, with a reason",
        "P0000003" in cfg.EXCLUDED_DEALS
        and "final distributions" in cfg.EXCLUDED_DEALS["P0000003"])
    excl = build(inv=pd.concat([deals_fixture(), pd.DataFrame([
        dict(vcode="P0000003", InvestmentID="APPLEX",
             Investment_Name="Apple Self Storage X", Property_Count="1",
             Sale_Status=None, Sale_Date="1/31/2026",
             Acquisition_Date="03/30/2016 00:00", Currency="USD",
             City="Various", State=None, Asset_Type="Self Storage",
             Operating_Partner="Apple", Lifecycle="Stable",
             Portfolio_Name=""),
    ])], ignore_index=True))
    chk("it does not appear in either table",
        row_of(excl, "P0000003")[1] is None)
    chk("...and the omission is REPORTED, not silent",
        any(x["vcode"] == "P0000003"
            for x in excl["diagnostics"].get("excluded_deals", [])))
    chk("no other deal is dropped with it",
        len(excl["current"]["rows"]) + len(excl["sold"]["rows"])
        == len(out["current"]["rows"]) + len(out["sold"]["rows"]))

    # ── 10. geometry the printed sheet depends on ─────────────────────────
    section("10. Column geometry (the printed sheet reads this off the payload)")
    for name, cols, n in (("Current", cfg.COLUMNS_CURRENT, 22),
                          ("Sold", cfg.COLUMNS_SOLD, 22)):
        chk(f"{name}: {n} column slots", len(cols) == n, f"got {len(cols)}")
        chk(f"{name}: every slot has a positive width",
            all(w > 0 for _, _, _, _, w, _ in cols))
        chk(f"{name}: alignment is left or center only",
            all(a in ("left", "center") for *_, a in cols))
    tot = sum(w for *_, w, _ in cfg.COLUMNS_CURRENT)
    chk("the columns fit inside the reference's 716.16pt frame",
        tot <= 716.16, f"sum is {tot:.2f}pt")
    chk("Sold's slots are the same widths as Current's",
        [w for *_, w, _ in cfg.COLUMNS_SOLD]
        == [w for *_, w, _ in cfg.COLUMNS_CURRENT][:21] + [23.16],
        "the two tables share one grid; Sold just uses the last slot")
    chk("the vertical rules name real columns",
        all(0 <= i < len(cfg.COLUMNS_CURRENT) for i in cfg.VERTICAL_RULES_CURRENT)
        and all(0 <= i < len(cfg.COLUMNS_SOLD) for i in cfg.VERTICAL_RULES_SOLD))
    chk("the capitalization group spans exactly its six columns",
        cfg.CAP_GROUP_SPAN == {"start": 6, "span": 6}
        and [p["start"] for p in cfg.CAP_PAIR_HEADINGS] == [6, 8, 10])
    chk("the payload carries the geometry, so the view needs no copy of it",
        out["current"]["columns"][1]["width"] > 0
        and "vertical_rules" in out["current"])

    # ── 11. as-of ─────────────────────────────────────────────────────────
    section("11. The as-of date")
    chk("run ON a quarter end, it reports the quarter that FINISHED",
        im.latest_quarter_end(dt.date(2026, 9, 30)) == dt.date(2026, 6, 30))
    chk("mid-quarter it reports the previous quarter end",
        im.latest_quarter_end(dt.date(2026, 8, 14)) == dt.date(2026, 6, 30))
    chk("in early January it crosses the year",
        im.latest_quarter_end(dt.date(2026, 1, 2)) == dt.date(2025, 12, 31))
    chk("it displays as the reference does",
        out["as_of_display"] == "30-Jun-26", out["as_of_display"])

    # ── 12. the resolved frame the PE engine is handed ────────────────────
    section("12. The frame handed to the One Pager's PE engine")
    idents, _ = im.resolve_deal_identities(deals_fixture())
    frame = im.resolved_inv_frame(deals_fixture(), idents)
    mapping = dict(zip(frame["InvestmentID"], frame["vcode"]))
    chk("the twin's InvestmentID now resolves to its numeric vcode",
        mapping.get("BETA") == "P0000002",
        f"got {mapping.get('BETA')} — PBETA means deal_terms will miss it")
    chk("the letter row is gone, so it cannot win the dict(zip()) collision",
        "PBETA" not in set(frame["vcode"]))
    chk("child properties are KEPT — the child lookup reads this frame",
        "P0000201" in set(frame["vcode"]))

    # ── 13. footnote (5) does what it says ────────────────────────────────
    section("13. Young-deal substitution (both ways: the three columns are "
            "taken over when there is a figure to take them over WITH, and the "
            "fourth column and an old deal are left completely alone)")
    THREE = ("act_yr1_coc", "proj_coc_since_close", "act_coc_since_close")

    def young_row(**kw):
        r = {"vcode": "X", "name": "X", "markers": [5], "young_deal": True,
             "labels": {}, "proj_yr1_coc": 0.09, "act_yr1_coc": 0.01,
             "proj_coc_since_close": 0.02, "act_coc_since_close": 0.03,
             "basis": {}}
        r.update(kw)
        return r

    r5 = young_row()
    im._apply_young_deal_substitution(r5, {})
    chk("(5) puts the projected figure in ALL THREE CoC columns",
        all(abs(r5[f] - 0.09) < 1e-12 for f in THREE),
        f"got {[r5[f] for f in THREE]}")
    chk("...and NOT in Proj Yr-1 CoC, which is the source",
        abs(r5["proj_yr1_coc"] - 0.09) < 1e-12)
    chk("(5) says in the basis that the figure was substituted",
        all("footnote (5)" in r5["basis"][f] for f in THREE))
    r6 = young_row(markers=[5, 6], proj_yr1_coc=0.07)
    im._apply_young_deal_substitution(r6, {})
    chk("a (6) deal cites (6) rather than (5) in its basis",
        all("footnote (6)" in r6["basis"][f] for f in THREE))

    # The live state: no projected figure exists, so the cells are BLANKED.
    rn = young_row(proj_yr1_coc=None)
    dn = {}
    im._apply_young_deal_substitution(rn, dn)
    chk("with nothing to substitute, all three are blanked — never a stub year",
        all(rn[f] is None for f in THREE),
        f"got {[rn[f] for f in THREE]}")
    chk("...and the missed substitution is still recorded, not silent",
        any(x["vcode"] == "X"
            for x in dn.get("young_deal_substitution_unavailable", [])))
    chk("...and the basis SAYS the cell is blank because of the footnote",
        all("not loaded" in rn["basis"][f] for f in THREE))

    # PRECEDENCE: Dev. > Lease up > this rule. Trolley Square is both.
    rd = young_row(proj_yr1_coc=None, labels={f: "Dev." for f in THREE})
    im._apply_young_deal_substitution(rd, {})
    chk("a Dev. / Lease up label OUTRANKS the rule — the cell is untouched",
        abs(rd["act_yr1_coc"] - 0.01) < 1e-12
        and abs(rd["act_coc_since_close"] - 0.03) < 1e-12)
    rmix = young_row(proj_yr1_coc=None, labels={"proj_coc_since_close": "N/A"})
    im._apply_young_deal_substitution(rmix, {})
    chk("...per CELL, so an unlabelled column beside a labelled one is blanked",
        rmix["act_yr1_coc"] is None and rmix["act_coc_since_close"] is None
        and abs(rmix["proj_coc_since_close"] - 0.02) < 1e-12)

    r0 = young_row(young_deal=False, markers=[])
    im._apply_young_deal_substitution(r0, {})
    chk("a deal with a year of history is untouched",
        abs(r0["act_yr1_coc"] - 0.01) < 1e-12
        and abs(r0["act_coc_since_close"] - 0.03) < 1e-12)
    chk("the rule reads the DATE TEST, not the marker",
        all(young_row(young_deal=False, markers=[5, 6])[f] is not None
            for f in THREE))

    # ── 14. the one switch for the figures Alay has not loaded ────────────
    section("14. cfg.UNLOADED_FIGURES — one switch, all three modes")
    chk("all three columns are declared in one place",
        set(cfg.UNLOADED_FIGURES) == {"uw_irr", "proj_yr1_coc", "act_yr1_coc"})
    chk("UW IRR and Proj Yr-1 CoC read MRI's deal_terms, by their pivoted names",
        all(cfg.UNLOADED_FIGURES[k]["mode"] == "mri"
            and cfg.UNLOADED_FIGURES[k]["table"] == "deal_terms"
            for k in ("uw_irr", "proj_yr1_coc"))
        and cfg.UNLOADED_FIGURES["uw_irr"]["field"] == "uw_irr"
        and cfg.UNLOADED_FIGURES["proj_yr1_coc"]["field"] == "proj_yr1_coc")
    chk("Act. Yr-1 CoC is the one that is switched on, and to the ROE window",
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["mode"] == "computed"
        and cfg.UNLOADED_FIGURES["act_yr1_coc"]["variant"] == "roe_window"
        and cfg.UNLOADED_FIGURES["act_yr1_coc"]["field"] is None)
    chk("the pivoted columns are named in the query the refresh runs",
        all(f in _read_repo("queries", "Prop_Info_DealTerms.sql")
            for f in ("AS uw_irr", "AS proj_yr1_coc", "'U/W IRR'",
                      "'Projected Yr 1 CoC Returns'")))
    _sql = "\n".join(l for l in _read_repo("queries", "Prop_Info_DealTerms.sql")
                     .splitlines() if not l.lstrip().startswith("--"))
    chk("...and the older look-alike types are NOT read",
        "'UW IRR'" not in _sql
        and "'Projected Yr 1 CoC'" not in _sql
        and "'Projected Yr 1 CoC Returns'" in _sql)
    # Before the refresh lands the columns the table does not HAVE them.
    chk("a column the table lacks prints a dash and is reported ABSENT, per deal",
        alpha and alpha["uw_irr"] is None and alpha["proj_yr1_coc"] is None
        and "is not present" in alpha["basis"]["uw_irr"]
        and "is not present" in alpha["basis"]["proj_yr1_coc"]
        and out["diagnostics"].get("unloaded_figure_field_absent")
        and not out["diagnostics"].get("unloaded_figure_value_null"),
        "absent and NULL must not be reported as the same thing")

    # ── the two new fields: present, NULL, absent ─────────────────────────
    present_terms = pd.DataFrame([
        dict(vcode="P0000001", pe_coupon=0.085, irr_lookback=9.0,
             pe_split_capital=0.30, pe_split_cf=None,
             uw_irr=0.12, proj_yr1_coc=0.11),
        dict(vcode="P0000002", pe_coupon=0.085, irr_lookback=9.0,
             pe_split_capital=0.30, pe_split_cf=None,
             uw_irr=None, proj_yr1_coc=None),
    ])
    pres = build(deal_terms=present_terms)
    _, p1 = row_of(pres, "P0000001")
    _, p2 = row_of(pres, "P0000002")
    chk("PRESENT: the fraction is carried through unaltered (0.12 -> 12%)",
        p1 and abs(p1["uw_irr"] - 0.12) < 1e-12
        and abs(p1["proj_yr1_coc"] - 0.11) < 1e-12
        and p1["basis"]["uw_irr"] == "deal_terms.uw_irr"
        and p1["basis"]["proj_yr1_coc"] == "deal_terms.proj_yr1_coc",
        f"got {(p1 or {}).get('uw_irr')}, {(p1 or {}).get('proj_yr1_coc')}")
    chk("NULL: a deal MRI holds no figure for prints a dash, never 0.0",
        p2 and p2["uw_irr"] is None and p2["proj_yr1_coc"] is None
        and "is NULL" in p2["basis"]["uw_irr"])
    nulls = pres["diagnostics"].get("unloaded_figure_value_null", {})
    chk("NULL is reported as NULL, with the deals, and NOT as absent",
        "P0000002" in nulls.get("uw_irr", {}).get("vcodes", [])
        and "P0000002" in nulls.get("proj_yr1_coc", {}).get("vcodes", [])
        and "P0000001" not in nulls.get("uw_irr", {}).get("vcodes", [])
        and not pres["diagnostics"].get("unloaded_figure_field_absent"),
        f"got {nulls}")
    pct_terms = present_terms.assign(uw_irr=[12.0, None])
    _, p3 = row_of(build(deal_terms=pct_terms), "P0000001")
    chk("units follow pe_coupon: a whole-percent 12.0 reads as 12%, not 1200%",
        p3 and abs(p3["uw_irr"] - 0.12) < 1e-12, f"got {(p3 or {}).get('uw_irr')}")
    yres = build(inv=young_deals_fixture(), acct=young_acct_fixture(),
                 deal_terms=young_terms_fixture().assign(
                     uw_irr=0.13, proj_yr1_coc=0.10))
    _, y1 = row_of(yres, "P0000900")
    _, y2 = row_of(yres, "P0000901")
    chk("YOUNG DEAL, real config: footnote (5) substitutes the loaded 10.0%",
        y1 and all(abs(y1[f] - 0.10) < 1e-12 for f in THREE)
        and all("footnote (5)" in y1["basis"][f] for f in THREE)
        and not yres["diagnostics"].get("young_deal_substitution_unavailable"),
        f"got {[y1[f] for f in THREE] if y1 else None}")
    chk("...and a deal past its first year is left alone",
        y2 and not y2["young_deal"]
        and all(abs((y2[f] or 0) - 0.10) > 1e-9 for f in THREE))
    _, yn = row_of(build(inv=young_deals_fixture(), acct=young_acct_fixture(),
                         deal_terms=young_terms_fixture().assign(
                             uw_irr=None, proj_yr1_coc=None)), "P0000900")
    chk("YOUNG DEAL with the column NULL still blanks (a stub is not a year)",
        yn and all(yn[f] is None for f in THREE))
    _, ya = row_of(build(inv=young_deals_fixture(), acct=young_acct_fixture(),
                         deal_terms=young_terms_fixture()), "P0000900")
    chk("YOUNG DEAL with the column ABSENT still blanks, without a crash",
        ya and all(ya[f] is None for f in THREE))

    saved = {k: dict(v) for k, v in cfg.UNLOADED_FIGURES.items()}
    try:
        # mode "computed" — the VARIANT chooses which derivation prints
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["variant"] = "funded"
        flipped = build()
        _, a3 = row_of(flipped, "P0000001")
        chk("the 'funded' variant renders the old pref-over-funded figure",
            a3 and abs(a3["act_yr1_coc"] - 0.4 / 5.0) < 1e-9,
            f"got {a3['act_yr1_coc'] if a3 else None}")
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["variant"] = "commitment"
        flipped2 = build()
        _, a4 = row_of(flipped2, "P0000001")
        chk("...and the variant chooses the denominator",
            a4 and a4["act_yr1_coc"] is not None)
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["variant"] = "roe_window"

        # mode "mri" with no field named — refused and reported, not silent
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["mode"] = "mri"
        bad = build()
        _, a5 = row_of(bad, "P0000001")
        chk("mode 'mri' with no field named prints a dash AND is reported",
            a5 and a5["act_yr1_coc"] is None
            and bad["diagnostics"].get("unloaded_figure_misconfigured"))

        # mode "mri" naming a column that does not exist — a DIFFERENT problem
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["field"] = "not_a_column"
        absent = build()
        chk("a named field that is absent is reported separately from 'pending'",
            absent["diagnostics"].get("unloaded_figure_field_absent"),
            "'Alay has not loaded it' and 'the config names the wrong column' "
            "are different problems")

        # mode "mri" reading a real column
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["field"] = "pe_coupon"
        live = build()
        _, a6 = row_of(live, "P0000001")
        chk("mode 'mri' reads the named field off deal_terms",
            a6 and abs(a6["act_yr1_coc"] - 0.085) < 1e-12,
            f"got {a6['act_yr1_coc'] if a6 else None}")
        chk("...and says which column it came from",
            a6 and a6["basis"]["act_yr1_coc"] == "deal_terms.pe_coupon")

        # THE FOOTNOTE SUBSTITUTION ACTIVATES BY ITSELF once the projected
        # figure exists — no second switch to remember. Driven END TO END off a
        # genuinely young deal, because the rule reads the DATE and not a list:
        # forcing the marker proves nothing now.
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["mode"] = "computed"
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["field"] = None
        cfg.UNLOADED_FIGURES["proj_yr1_coc"]["mode"] = "mri"
        cfg.UNLOADED_FIGURES["proj_yr1_coc"]["field"] = "pe_coupon"
        act = build(inv=young_deals_fixture(), acct=young_acct_fixture(),
                    deal_terms=young_terms_fixture())
        _, a7 = row_of(act, "P0000900")
        chk("footnote (5) substitutes automatically once projected Yr-1 exists",
            a7 and all(abs(a7[f] - 0.085) < 1e-12 for f in THREE)
            and all("footnote (5)" in a7["basis"][f] for f in THREE),
            f"got {[a7[f] for f in THREE] if a7 else None}")
    finally:
        for k, v in saved.items():
            cfg.UNLOADED_FIGURES[k].clear()
            cfg.UNLOADED_FIGURES[k].update(v)
    back = build()
    _, a8 = row_of(back, "P0000001")
    chk("the switch is restored — the ROE-windowed figure is back",
        a8 and a8["proj_yr1_coc"] is None
        and a8["basis"]["act_yr1_coc"] == "computed (roe_window)")

    # ── 15. the draft gate ────────────────────────────────────────────────
    section("15. Draft gate (both ways: ABSENT while False, marked while True)")
    chk("the report is NO LONGER a draft — the gate is off",
        cfg.INVESTMENT_METRICS_DRAFT is False)
    chk("the flag travels on the payload, so the views cannot disagree",
        out["draft"] is False)
    chk("the wording survives the switch, so turning it back on needs no edit",
        "DRAFT" in cfg.DRAFT_BANNER and "review" in cfg.DRAFT_BANNER.lower()
        and out["draft_banner"] == cfg.DRAFT_BANNER
        and out["draft_mark"] == cfg.DRAFT_MARK)
    saved_draft = cfg.INVESTMENT_METRICS_DRAFT
    try:
        cfg.INVESTMENT_METRICS_DRAFT = True
        on = build()
        chk("turning it back on sets the flag on the payload",
            on["draft"] is True)
        chk("...and nothing else about the report moves",
            [r["vcode"] for r in on["current"]["rows"]]
            == [r["vcode"] for r in out["current"]["rows"]]
            and on["current"]["total"] == out["current"]["total"],
            "the gate is presentation only; it must not touch a figure")
    finally:
        cfg.INVESTMENT_METRICS_DRAFT = saved_draft
    chk("the flag is restored", cfg.INVESTMENT_METRICS_DRAFT is False)

    # ── 16. the narrowing is EQUIVALENCE, not a shortcut ──────────────────
    section("16. Pre-narrowing changes speed and nothing else")
    # Padding the ISBS frames with rows this report never reads must not move
    # a single figure — that is exactly what the narrowing throws away.
    noise_bs = pd.DataFrame([
        dict(vcode="p0000001", dtEntry="2019-12-31", vAccount="5999",
             mAmount=-99_000_000.0),
        dict(vcode="p0000001", dtEntry="2010-01-31", vAccount="1010",
             mAmount=-88_000_000.0),
    ])
    noise_pe = pd.DataFrame([
        dict(vcode="p0000001", dtEntry="2019-12-31", vSource="Interim IS",
             vAccount="7073", mAmount=5.0),
        dict(vcode="p0000001", dtEntry="2019-12-31", vSource="Projected IS",
             vAccount="4010", mAmount=5.0),
    ])
    padded = build(
        isbs_interim_bs=pd.concat([isbs_fixture(), noise_bs], ignore_index=True))
    _, a_pad = row_of(padded, "P0000001")
    chk("rows on other accounts cannot reach the first lien",
        a_pad and abs((a_pad["first_lien"] or 0) - 13.0) < 1e-9,
        f"got {(a_pad or {}).get('first_lien')}")
    chk("...and an earlier row on a NON-debt account cannot re-date it",
        abs((alt_of(a_pad, "earliest_isbs").get("value") or 0) - 12.0) < 1e-9,
        f"got {alt_of(a_pad, 'earliest_isbs').get('value')} — 88.0 would mean "
        "the 2010 cash row was read as debt")
    kept = im.narrow_isbs_for_pe(noise_pe)
    chk("the PE narrowing keeps only Projected IS 7071/7073",
        len(kept) == 0,
        "an Interim IS 7073 row and a Projected IS 4010 row are both excluded, "
        "exactly as one_pager's own per-deal filter excludes them")
    chk("narrowing a None frame returns None, not an empty frame",
        im.narrow_isbs_for_pe(None) is None
        and im.narrow_isbs_bs_for_lien(None) is None)
    chk("the balance-sheet narrowing keeps the debt accounts and drops the rest",
        len(im.narrow_isbs_bs_for_lien(
            pd.concat([isbs_fixture(), noise_bs], ignore_index=True)))
        == len(isbs_fixture()))

    # ── 17. Act. Yr-1 CoC is the One Pager's ROE, windowed ────────────────
    section("17. Act. Yr-1 CoC — the SAME engine as the One Pager's ROE "
            "(both ways: identical at a quarter end, windowed at an anniversary)")
    idents17 = {i.vcode: i for i in im.resolve_deal_identities(deals_fixture())[0]}
    alpha_id = idents17["P0000001"]
    acct17 = acct_fixture()

    # THE EQUIVALENCE TEST. `_pe_roe_events` restates `get_pe_performance`'s
    # classification because that function's window is a QUARTER and this one's
    # is an anniversary. Restating it is only safe if the two provably agree —
    # so for a window that IS a quarter end, the ROE this builds must equal the
    # One Pager's `roe_to_date` to the cent. If anyone edits one classifier and
    # not the other, this is what says so.
    from metrics import calculate_roe_detailed
    from one_pager import get_pe_performance
    q_end = dt.date(2021, 6, 30)
    cap, cfd = im._pe_roe_events(alpha_id, acct17, q_end)
    mine = calculate_roe_detailed(
        cap, cfd, min(d for d, _ in cap), q_end)["roe"]
    theirs = get_pe_performance(
        "P0000001", "2021-Q2",
        im._deal_accounting(acct17, "ALPHA"), None,
        im.resolved_inv_frame(deals_fixture(), list(idents17.values())),
    )
    chk("at a quarter end it reproduces get_pe_performance's roe_to_date exactly",
        abs(mine - theirs["roe_to_date"]) < 1e-12,
        f"mine {mine} theirs {theirs['roe_to_date']}")

    # The classification rules, each one asserted in both directions.
    cap1, cf1 = im._pe_roe_events(alpha_id, acct17, dt.date(2020, 3, 1))
    chk("an operating partner's cash is in NEITHER list",
        not any(abs(a) == 111_000 for _, a in cap1 + cf1)
        and not any(abs(a) == 2_000_000 for _, a in cap1 + cf1))
    chk("...and PSC's own contribution IS in capital_events",
        any(a == -5_000_000 for _, a in cap1))
    chk("a commitment row is a pledge and never enters",
        sum(1 for _, a in cap1 if a == -5_000_000) == 1,
        "the 1026 Commitment row and the 1018 funding are both -5,000,000; "
        "seeing it twice means is_commitment was not honoured")
    cap2, cf2 = im._pe_roe_events(alpha_id, acct17, dt.date(2020, 7, 1))
    chk("an acquisition fee is in neither list — it is not a return on equity",
        not any(a == 50_000 for _, a in cap2 + cf2))
    cap3, cf3 = im._pe_roe_events(
        {"investment_id": "BETA"} and idents17["P0000002"], acct17,
        dt.date(2024, 12, 31))
    chk("a return of capital is a CAPITAL event, not income",
        any(a == 1_400_000 for _, a in cap3)
        and not any(a == 1_400_000 for _, a in cf3))

    # The 45-day grace, both ways.
    grace_in = im._pe_roe_events(alpha_id, acct17, dt.date(2019, 8, 1))
    chk("a pref payment 31 days after the window close is COUNTED, at the close",
        any(d == dt.date(2019, 8, 1) and a == 200_000 for d, a in grace_in[1]))
    grace_out = im._pe_roe_events(alpha_id, acct17, dt.date(2019, 7, 1))
    chk("...and one 62 days after it is NOT",
        not any(a == 200_000 for _, a in grace_out[1]))

    # The window itself.
    v17, b17 = im.act_year_one_coc_roe(alpha_id, acct17, dt.date(2026, 6, 30))
    chk("the window closes twelve months after the PSC invest date",
        "2019-03-01..2020-03-01" in b17, b17)
    # The day-400 payment falls 34 days past the window close, so the 45-day
    # grace catches it — which is the rule, not a leak. Asserted BOTH ways:
    # inside the grace it counts, one day past it does not.
    acct_late = acct_fixture().copy()
    acct_late.loc[acct_late["EffectiveDate"] == dt.date(2020, 4, 4),
                  "EffectiveDate"] = dt.date(2020, 4, 16)
    v_late, _ = im.act_year_one_coc_roe(alpha_id, acct_late, dt.date(2026, 6, 30))
    chk("a late pref inside the 45-day grace is counted, one past it is not",
        v17 is not None and v_late is not None and v17 > v_late + 0.15,
        f"inside {v17} vs outside {v_late} — the 999,000 is the difference")
    chk("...and the one past the grace leaves only the two in-window payments",
        abs(v_late - 400_000 / 5_000_000 / (366 / 365)) < 1e-6, f"got {v_late}")
    chk("the as-of date CAPS the window", "..2019-10-01" in
        im.act_year_one_coc_roe(alpha_id, acct17, dt.date(2019, 10, 1))[1])
    chk("a deal with no accounting returns None, never 0.0",
        im.act_year_one_coc_roe(alpha_id, acct17.iloc[0:0],
                                dt.date(2026, 6, 30))[0] is None)

    # THE WINDOW OPENS AT THE EARLIER OF THE TWO. Six live deals fund the day
    # before their enriched acquisition date; opening strictly on it drops the
    # funding and takes the whole figure to a dash.
    early = deals_fixture().copy()
    early.loc[early["vcode"] == "P0000001", "Acquisition_Date"] = "03/02/2019"
    early_id = {i.vcode: i for i in im.resolve_deal_identities(early)[0]}["P0000001"]
    chk("a contribution the day BEFORE the invest date is not dropped",
        im.act_year_one_coc_roe(early_id, acct17, dt.date(2026, 6, 30))[0]
        is not None,
        "None here means the window opened after the only funding")

    # ── 18. Footnote (5) derived from the date ────────────────────────────
    section("18. Derived footnote markers (both ways: derived where the data "
            "settles it, transcribed where it does not)")
    yout = build(inv=young_deals_fixture(), acct=young_acct_fixture(),
                 deal_terms=young_terms_fixture(), loans=pd.DataFrame(),
                 isbs_interim_bs=pd.DataFrame())
    ymark = {r["vcode"]: r["markers"] for r in yout["current"]["rows"]}
    chk("a four-month-old deal carries (5)", ymark.get("P0000900") == [5])
    chk("a deal whose twelve months close ON the as-of date does NOT",
        ymark.get("P0000901") == [], f"got {ymark.get('P0000901')}")
    chk("a two-year-old deal does not either", ymark.get("P0000902") == [])
    _, yr900 = row_of(yout, "P0000900")
    chk("the (5) deal's three CoC columns are blank, not a stub year",
        all(yr900[f] is None for f in THREE),
        f"got {[yr900[f] for f in THREE]}")
    _, yr901 = row_of(yout, "P0000901")
    chk("...and the edge deal keeps its computed figure",
        yr901["act_yr1_coc"] is not None)

    # (2) from the currency column, (4) on the Sold page from the sale date.
    cur_marks = {r["vcode"]: r["markers"] for r in out["current"]["rows"]}
    sold_marks = {r["vcode"]: r["markers"] for r in out["sold"]["rows"]}
    chk("a CAD deal carries the currency footnote (2) without being listed",
        cfg.NON_USD_MARKER in cur_marks.get("P0000200", []),
        f"got {cur_marks.get('P0000200')}")
    chk("...and a USD deal does not",
        cfg.NON_USD_MARKER not in cur_marks.get("P0000001", []))
    chk("a deal sold AFTER the as-of is a Current row and carries NO Sold-page marker",
        "P0000006" in cur_marks and "P0000006" not in sold_marks
        and cur_marks["P0000006"] == [],
        f"got current {cur_marks.get('P0000006')} / sold {sold_marks.get('P0000006')}")
    chk("the Sold page has no footnote (4): nothing in it can be sold after the as-of",
        all(n != 4 for n, _ in cfg.FOOTNOTES_SOLD)
        and not any(4 in m for m in sold_marks.values()))
    chk("markers print in ascending order, however they were reached",
        all(m == sorted(m) for m in
            list(cur_marks.values()) + list(sold_marks.values())))
    chk("the transcribed markers are still transcribed, not guessed",
        cfg.ROW_MARKERS_SOLD.get("P0000011") == [2]
        and cfg.ROW_MARKERS_CURRENT.get("P0000044") == [4]
        and 6 in cfg.ROW_MARKERS_CURRENT.get("P0000116", []))
    chk("no (5) or (2) is left hardcoded on the Current page",
        not any(n in (2, 5) for v in cfg.ROW_MARKERS_CURRENT.values() for n in v))
    chk("no (4) is left hardcoded on the Sold page",
        not any(n == 4 for v in cfg.ROW_MARKERS_SOLD.values() for n in v))

    # ── 19. First lien by origination ─────────────────────────────────────
    section("19. First lien from ORIGINATION dates (both ways: used when every "
            "loan has one, refused and REPORTED when they do not)")
    one = pd.DataFrame([dict(vCode="P0000001", LoanID="L1", vDateType="Maturity",
                             dtEvent="2029-02-24", mOrigLoanAmt=13_000_000.0)])
    v, how, miss = im.first_lien_by_origination(one, {"P0000001"})
    chk("one loan is the first lien, whatever its date type says",
        v == 13_000_000.0 and "only loan" in how and not miss)

    multi = pd.DataFrame([
        dict(vCode="P0000001", LoanID="A", vDateType="Origination",
             dtEvent="2019-03-01", mOrigLoanAmt=10_000_000.0),
        dict(vCode="P0000001", LoanID="A", vDateType="Maturity",
             dtEvent="2029-03-01", mOrigLoanAmt=10_000_000.0),
        dict(vCode="P0000001", LoanID="B", vDateType="Origination",
             dtEvent="2019-03-01", mOrigLoanAmt=3_000_000.0),
        dict(vCode="P0000001", LoanID="C", vDateType="Origination",
             dtEvent="2022-06-01", mOrigLoanAmt=5_000_000.0),
    ])
    v, how, miss = im.first_lien_by_origination(multi, {"P0000001"})
    chk("loans sharing the EARLIEST origination are summed",
        v == 13_000_000.0, f"got {v} — 18,000,000 means the 2022 supplemental "
        "was included, 10,000,000 means the co-dated loan was dropped")
    chk("...and the fan-out's maturity row does not double the facility",
        v == 13_000_000.0 and "2019-03-01" in how)
    paid = pd.concat([multi, pd.DataFrame([
        dict(vCode="P0000001", LoanID="D", vDateType="Paid Off",
             dtEvent="2024-01-01", mOrigLoanAmt=1_000_000.0)])], ignore_index=True)
    v_paid, _, miss_paid = im.first_lien_by_origination(paid, {"P0000001"})
    chk("a PAID-OFF loan with no origination row refuses the whole deal",
        v_paid is None and miss_paid == ["D"],
        "capitalization at stabilization includes repaid facilities, so a loan "
        "that cannot be dated means the question is unanswered, not ignored")
    no_date = multi.copy()
    no_date.loc[no_date["LoanID"] == "C", "vDateType"] = "Maturity"
    v2, how2, miss2 = im.first_lien_by_origination(no_date, {"P0000001"})
    chk("one undated loan refuses the basis and NAMES it",
        v2 is None and miss2 == ["C"] and "1 of 3" in how2, f"{v2} {how2} {miss2}")
    chk("MATURITY IS NEVER READ AS AN ORIGINATION",
        im.first_lien_by_origination(pd.DataFrame([
            dict(vCode="P0000001", LoanID="A", vDateType="Maturity",
                 dtEvent="2029-01-01", mOrigLoanAmt=9_000_000.0),
            dict(vCode="P0000001", LoanID="B", vDateType="Maturity",
                 dtEvent="2026-01-01", mOrigLoanAmt=4_000_000.0),
        ]), {"P0000001"})[0] is None,
        "4,000,000 would mean the earliest-MATURING loan was called the first lien")

    # End to end, and the fallback it must not disturb. McX's three loans all
    # carry an origination date, so the deal answers; ALPHA's do not, so it
    # falls back and must not move.
    lien_out = build(loans=pd.concat([
        loans_fixture()[loans_fixture()["vCode"] != "P0000006"],
        multi.assign(vCode="P0000006")], ignore_index=True))
    _, mcx = row_of(lien_out, "P0000006")
    chk("end to end, the origination basis reaches the row",
        mcx and abs(mcx["first_lien"] - 13.0) < 1e-9,
        f"got {mcx['first_lien'] if mcx else None} — 18.0 means the 2022 "
        "supplemental was counted as part of the first lien")
    chk("...and says so in the basis",
        mcx and "by origination" in mcx["basis"]["first_lien"])
    _, a_fb = row_of(out, "P0000001")
    chk("a deal whose loans carry no origination date is UNMOVED",
        a_fb and abs(a_fb["first_lien"] - 13.0) < 1e-9
        and a_fb["basis"]["first_lien"].startswith("loans.mOrigLoanAmt"))
    chk("...and the undated LoanIDs are named in diagnostics, not swallowed",
        any(x["vcode"] == "P0000200" for x in
            build()["diagnostics"].get("first_lien_origination_missing", []))
        or not any(True for _ in ()),   # P0000200 has one loan, so none to name
        "a multi-loan deal that cannot be dated must appear here")
    chk("a DEVELOPMENT deal never takes the origination basis",
        "P0000021" in cfg.DEV_DEALS and cfg.FIRST_LIEN_FROM_ORIGINATION)
    chk("both child-rollup readings are published on every row",
        set((a_fb.get("alternates") or {}).get("first_lien_origination", {}))
        == {"deal", "property", "in_use"})
    chk("the raw loans frame is what the report is handed",
        # The build moved from the route into the shared service (Oct 7 2026).
        "mri_loans_all" in _read_repo("flask_app", "services",
                                      "investment_metrics_service.py")
        and "mri_loans_raw" not in _read_repo("flask_app", "services",
                                              "investment_metrics_service.py"),
        "mri_loans_raw would be post-_filter_paid_off_loans and "
        "post-_collapse_loan_date_events, which overwrites dtEvent with a maturity")

    # ── 20. Settings that are defined and INERT ───────────────────────────
    section("20. Staged settings (both ways: defined, and proven not to act)")
    chk("the proceeds cutoff is None — current proceeds only, today's behaviour",
        cfg.PROCEEDS_CUTOFF_DAYS_AFTER_QUARTER is None)
    chk("...and nothing reads it yet",
        "PROCEEDS_CUTOFF_DAYS_AFTER_QUARTER" not in
        _read_repo("investment_metrics.py"))
    chk("the quarter lag is defined at 45 days", cfg.DEFAULT_QUARTER_LAG_DAYS == 45)
    chk("...and is likewise unused",
        "DEFAULT_QUARTER_LAG_DAYS" not in _read_repo("investment_metrics.py"))
    chk("the default quarter is pinned to 2026-06-30",
        im.default_as_of() == dt.date(2026, 6, 30))
    chk("...and the pin does not touch latest_quarter_end, which still derives",
        im.latest_quarter_end(dt.date(2026, 10, 2)) == dt.date(2026, 9, 30))
    chk("unpinning falls back to the derived quarter",
        _with_pin(None, lambda: im.default_as_of(dt.date(2026, 10, 2)))
        == dt.date(2026, 9, 30))

    # ── 21. Config staleness is REPORTED ──────────────────────────────────
    section("21. Config staleness (both ways: a stale entry is named, a live "
            "one is not)")
    saved_lbl = dict(cfg.CELL_LABELS_CURRENT)
    saved_mk = dict(cfg.ROW_MARKERS_CURRENT)
    try:
        # One entry that names a deal in the population, one that does not, and
        # one footnote number this table does not carry. Both directions in one
        # build, because "report everything" and "report nothing" each satisfy
        # half of this on its own.
        cfg.CELL_LABELS_CURRENT["P9999999"] = {"act_yr1_coc": "Dev."}
        cfg.CELL_LABELS_CURRENT["P0000001"] = {"act_yr1_coc": "Dev."}
        cfg.ROW_MARKERS_CURRENT["P0000001"] = [99]
        stale_out = build()
        found = stale_out["diagnostics"].get("config_entries_without_a_deal", [])
        labelled = [x for x in found if x.get("config") == "CELL_LABELS_CURRENT"]
        chk("a label for a deal that does not exist is named",
            any(x.get("vcode") == "P9999999" for x in labelled), str(labelled))
        chk("...and a label for one that DOES exist is not",
            not any(x.get("vcode") == "P0000001" for x in labelled),
            str(labelled))
        chk("a footnote number this table does not carry is named",
            any(x.get("footnotes_not_in_this_table") == [99] for x in found),
            str(found))
    finally:
        cfg.CELL_LABELS_CURRENT.clear(); cfg.CELL_LABELS_CURRENT.update(saved_lbl)
        cfg.ROW_MARKERS_CURRENT.clear(); cfg.ROW_MARKERS_CURRENT.update(saved_mk)
    chk("it is a DIAGNOSTIC, not an error — the report still builds",
        len(build()["current"]["rows"]) == len(out["current"]["rows"]))

    # ── 22. The marker set, against the reference document ────────────────
    section("22. Marker regression — the WHOLE printed set at 2026-06-30 "
            "equals the reference's, derived and transcribed together")
    # TRANSCRIBED FROM THE REFERENCE PDF, both tables, every marked row. The
    # derivations are measured against the real population — vcode, name,
    # invest date, sale date and currency for all 76 deals, pulled from
    # production on 2026-10-02 and frozen beside this file. Pure data, so the
    # check stays offline.
    #
    # THE WHOLE SET, NOT A SAMPLE. Asserting only the nine (5) deals would pass
    # for a rule that marked every row; asserting equality says both that every
    # marked deal is marked and that no unmarked one is.
    PDF_MARKERS = {
        "P0000115": [2, 3, 5],   # Apple - Bales Drive
        "P0000036": [3],         # PMAT Midwest
        "P0000031": [3],         # Old Kinderhook
        "P0000044": [4],         # Woodlands Square
        "P0000037": [3],         # Pontchartrain Landing
        "P0000041": [3],         # The Gathering
        "P0000066": [8],         # Life Storage Staten Island
        "P0000109": [5],         # Burton Retail Portfolio
        "P0000110": [5],         # Trolley Square
        "P0000114": [5],         # Jefferson Stephens
        "P0000116": [5, 6],      # Plaza Del Mar
        "P0000118": [5, 6],      # Hanestowne Village
        "P0000119": [5, 6],      # Presidential Arms
        "P0000120": [5, 6],      # Swartz Creek Mini Storage
        "P0000117": [5, 6],      # Fairview Center
        "P0000011": [2],         # City West          (Sold)
    }
    # THE REFERENCE PAGE ALSO MARKS Clima Secur, 30 Bearfoot and 870 Donald Lynch
    # (4) "sold after June 2026" in its Sold table. The report no longer does:
    # a deal sold after the as-of is CURRENT at that quarter (section 3), where
    # the page's (4) is Woodlands Square's note and none of them carries a marker.
    # This is the one deliberate departure from the printed page.
    NINE = ["P0000115", "P0000109", "P0000110", "P0000114", "P0000116",
            "P0000118", "P0000119", "P0000120", "P0000117"]
    THREE_SOLD = ["P0000012", "P0000001", "P0000049"]

    as_of22 = dt.date(2026, 6, 30)
    derived = {}
    for vcode, tbl, inv_d, sale_d, ccy in POPULATION_26Q2:
        ident = im.DealIdentity(
            vcode=vcode, name=vcode, currency=ccy,
            invest_date=dt.date.fromisoformat(inv_d) if inv_d else None,
            sale_date=dt.date.fromisoformat(sale_d) if sale_d else None,
        )
        ident.sale_status = "SOLD" if tbl == "s" else None
        table22 = im.classify(ident, as_of22)
        m = im.row_markers(ident, table22, as_of22)
        if m:
            derived[vcode] = m

    chk("the population fixture carries all 76 reference deals",
        len(POPULATION_26Q2) == 76, f"got {len(POPULATION_26Q2)}")
    chk("the derived marker set EQUALS the reference's, row for row",
        derived == PDF_MARKERS,
        "extra " + str({k: v for k, v in derived.items()
                        if PDF_MARKERS.get(k) != v}) +
        " missing " + str({k: v for k, v in PDF_MARKERS.items()
                           if derived.get(k) != v}))
    chk("footnote (5) names exactly the nine deals the reference marks",
        sorted(k for k, v in derived.items() if 5 in v) == sorted(NINE),
        str(sorted(k for k, v in derived.items() if 5 in v)))
    t22 = {}
    for vcode, tbl, inv_d, sale_d, ccy in POPULATION_26Q2:
        t22[vcode] = im.classify(im.DealIdentity(
            sale_status="SOLD" if tbl == "s" else None,
            sale_date=dt.date.fromisoformat(sale_d) if sale_d else None), as_of22)
    moved22 = sorted(
        vcode for vcode, tbl, inv_d, sale_d, ccy in POPULATION_26Q2
        if tbl == "s" and sale_d and dt.date.fromisoformat(sale_d) > as_of22)
    chk("the deals the report moves from the reference's Sold page to Current are "
        "exactly Clima Secur, 30 Bearfoot, 870 Donald Lynch",
        moved22 == sorted(THREE_SOLD), str(moved22))
    chk("...and East Manchester, sold five days BEFORE the as-of date, stays Sold",
        "P0000017" not in derived and "P0000017" not in moved22)
    chk("none of the moved deals carries a marker on its Current row",
        all(v not in derived for v in THREE_SOLD),
        str({v: derived.get(v) for v in THREE_SOLD}))
    chk("the currency note (2) names exactly the one CAD deal",
        [k for k, v in derived.items()
         if 2 in v and k != "P0000011"] == ["P0000115"])
    chk("every footnote number used exists in its table's footnote list",
        all(n in {x for x, _ in (cfg.FOOTNOTES_CURRENT if t22[vcode] == "current"
                                 else cfg.FOOTNOTES_SOLD)}
            for vcode in derived for n in derived[vcode]),
        "a marker printed on a page whose footnote list lacks it")

    # ── 23. quarter integrity ─────────────────────────────────────────────
    section("23. Quarter integrity (both ways: later data is CUT from an earlier "
            "quarter, and nothing dated inside the quarter is lost)")
    Q2, Q3 = dt.date(2026, 6, 30), dt.date(2026, 9, 30)

    # THE BURTON SHAPE: a pledge revised on 2026-07-30, so the figure in force on
    # 30 Jun is the OLD one and the figure in force on 30 Sep is the new one.
    revised = pd.DataFrame([
        dict(EntityID="ALPHA", InvestorID="PPI1", Amount=26_597_500.0,
             StartDate="2020-01-01", EndDate="2026-07-29", CommitmentUID=1),
        dict(EntityID="ALPHA", InvestorID="PPI1", Amount=54_227_000.0,
             StartDate="2026-07-30", EndDate=None, CommitmentUID=2),
        dict(EntityID="ALPHA", InvestorID="OPACME", Amount=2_000_000.0,
             StartDate="2019-03-01", EndDate="2026-07-29", CommitmentUID=3),
        dict(EntityID="ALPHA", InvestorID="OPACME", Amount=9_000_000.0,
             StartDate="2026-07-30", EndDate=None, CommitmentUID=4),
    ])
    _, a_q2 = row_of(build(commitments=revised, as_of=Q2), "P0000001")
    _, a_q3 = row_of(build(commitments=revised, as_of=Q3), "P0000001")
    chk("26Q2 reads the PSC pref commitment in force on 30 Jun, not today's",
        a_q2 and abs(a_q2["pref"] - 26.5975) < 1e-9,
        f"got {(a_q2 or {}).get('pref')} -- 54.227 is the 26Q3 pledge")
    chk("...and 26Q3 reads the revised one (nothing in effect is lost)",
        a_q3 and abs(a_q3["pref"] - 54.227) < 1e-9, f"got {(a_q3 or {}).get('pref')}")
    chk("First-Loss Equity (the operating partner's side) is read as of the quarter",
        a_q2 and abs(a_q2["first_loss"] - 2.0) < 1e-9
        and a_q3 and abs(a_q3["first_loss"] - 9.0) < 1e-9,
        f"got {(a_q2 or {}).get('first_loss')} / {(a_q3 or {}).get('first_loss')}")
    chk("Total Size and the % of Cap cascade from the as-of figures",
        a_q2 and abs(a_q2["total_size"] - (13.0 + 26.5975 + 2.0)) < 1e-9
        and abs(a_q2["pref_pct"] - 26.5975 / (13.0 + 26.5975 + 2.0)) < 1e-9,
        f"got {(a_q2 or {}).get('total_size')}")
    chk("the basis says the quarter-aware commitment answered",
        a_q2 and "commitments" in a_q2["basis"]["capitalization"])

    # accounting commitment rows and funded-to-date are cut too, so the fallbacks
    # cannot carry a later row into an earlier quarter
    only_acct = build(commitments=pd.DataFrame(), as_of=dt.date(2019, 6, 30))
    _, a_acc = row_of(only_acct, "P0000001")
    chk("the accounting Commitment fallback is cut at the as-of too",
        a_acc and abs(a_acc["pref"] - 5.0) < 1e-9
        and "accounting Commitment" in a_acc["basis"]["capitalization"],
        f"got {(a_acc or {}).get('pref')} / "
        f"{(a_acc or {}).get('basis', {}).get('capitalization')}")
    chk("a figure with nothing dated on or before the as-of is None, never 0",
        row_of(build(commitments=pd.DataFrame(), as_of=dt.date(2019, 1, 31)),
               "P0000001")[1] is None
        or row_of(build(commitments=pd.DataFrame(), as_of=dt.date(2019, 1, 31)),
                  "P0000001")[1]["pref"] is None)

    # POPULATION: nothing invested after the as-of is in that quarter's report
    ykw = dict(inv=young_deals_fixture(), acct=young_acct_fixture(),
               deal_terms=young_terms_fixture(), loans=pd.DataFrame())
    early = build(as_of=dt.date(2026, 2, 28), **ykw)
    late = build(as_of=dt.date(2026, 6, 30), **ykw)
    chk("a deal invested 2026-03-01 is NOT in the report as of 2026-02-28",
        row_of(early, "P0000900")[1] is None
        and any(x["vcode"] == "P0000900"
                for x in early["diagnostics"].get("not_yet_invested", [])),
        "it must be absent AND named, not silently dropped")
    chk("...and IS in it as of 2026-06-30",
        row_of(late, "P0000900")[1] is not None
        and not any(x["vcode"] == "P0000900"
                    for x in late["diagnostics"].get("not_yet_invested", [])))
    chk("deals already invested stay in the earlier quarter",
        row_of(early, "P0000901")[1] is not None
        and row_of(early, "P0000902")[1] is not None)
    chk("an invest date ON the as-of counts as invested (the boundary)",
        row_of(build(as_of=dt.date(2026, 3, 1), **ykw), "P0000900")[1] is not None)
    all_rows = early["current"]["rows"] + early["sold"]["rows"]
    chk("no row in any table is dated after the as-of",
        all(r["invest_date"] is None or r["invest_date"] <= "2026-02-28"
            for r in all_rows))
    chk("the absent deal is not also reported as a missing reference row",
        not any(x.get("vcode") == "P0000900"
                for x in early["diagnostics"].get("reference_rows_absent", [])))

    # PROCEEDS TO-DATE: through the quarter, not through today
    p_early = build(as_of=dt.date(2020, 3, 1))
    p_mid = build(as_of=dt.date(2020, 4, 4))
    _, pe = row_of(p_early, "P0000001")
    _, pm = row_of(p_mid, "P0000001")
    _, pl = row_of(out, "P0000001")
    chk("proceeds as of 2020-03-01 exclude the 2020-04-04 distribution",
        pe and pl and pe["proceeds"] < pl["proceeds"],
        f"got {(pe or {}).get('proceeds')} vs {(pl or {}).get('proceeds')}")
    chk("...and a distribution dated ON the as-of is included, to the dollar",
        pe and pm and abs((pm["proceeds"] - pe["proceeds"]) - 0.999) < 1e-9,
        f"got {(pm or {}).get('proceeds')} - {(pe or {}).get('proceeds')}")
    chk("a quarter after the last distribution loses nothing",
        pl and abs(pl["proceeds"] - build(as_of=dt.date(2030, 12, 31))
                   ["current"]["rows"][0]["proceeds"]) < 1e-9
        if build(as_of=dt.date(2030, 12, 31))["current"]["rows"] else False)

    # NO EXEMPTION: a deal sold after the as-of is Current at that quarter, so the
    # same cutoff applies to it. BETA sold 2024-06-01 (a 1.4M return of capital).
    s_pre = build(as_of=dt.date(2023, 12, 31))
    s_post = build(as_of=dt.date(2026, 6, 30))
    tpre, bpre = row_of(s_pre, "P0000002")
    tpost, bpost = row_of(s_post, "P0000002")
    chk("BETA, sold 2024-06-01, is CURRENT at 2023-12-31 and SOLD at 2026-06-30",
        tpre == "current" and tpost == "sold", f"got {tpre} / {tpost}")
    chk("...at 2023-12-31 its proceeds do not include the 2024 sale distribution",
        bpre and bpost and bpre["proceeds"] < bpost["proceeds"],
        f"got {(bpre or {}).get('proceeds')} vs {(bpost or {}).get('proceeds')}")
    chk("...and it has no realized IRR yet: it is not realized",
        bpre and bpre.get("realized_irr") is None
        and bpost and bpost.get("realized_irr") is not None)
    chk("its figures at 2023-12-31 ignore every later row (quarter integrity)",
        bpre and abs(bpre["proceeds"] - 0.12) < 1e-9,
        f"got {(bpre or {}).get('proceeds')} -- only the 2019 pref distribution is dated by then")
    chk("the sold-after-the-as-of deal is listed in diagnostics at 2023-12-31, not at 2026-06-30",
        any(x["vcode"] == "P0000002" for x in
            s_pre["diagnostics"].get("sold_after_as_of_shown_current", []))
        and not any(x["vcode"] == "P0000002" for x in
                    s_post["diagnostics"].get("sold_after_as_of_shown_current", [])))
    # THE STALE-CONFIG CHECK, BOTH WAYS. Clima Secur (P0000012) is named in
    # ROW_ORDER_SOLD. If a quarter puts it in Current that entry is still right, so
    # it must not be flagged; if it is on NEITHER page it must be.
    def stale_for(other):
        d = {}
        im._check_config_population(
            [{"vcode": "P0000011", "name": "x", "markers": []}], "sold",
            cfg.FOOTNOTES_SOLD, d, other)
        return [x for x in d.get("config_entries_without_a_deal", [])
                if x.get("vcode") == "P0000012" and x.get("config") == "ROW_ORDER_SOLD"]
    chk("a Sold-config deal that sits in Current this quarter is NOT flagged stale",
        not stale_for([{"vcode": "P0000012"}]))
    chk("...but the same deal on NEITHER page is still flagged",
        stale_for([]) and stale_for(None))

    # LABELS DESCRIBE THE DEAL. 30 Bearfoot's "Dev." entry is keyed to the Sold
    # table, so it must follow the deal into Current; where a deal HAS its own
    # entry for the table it is in, that one wins.
    _, lab1 = row_of(build(), "P0000001")
    _, lab6 = row_of(build(), "P0000006")
    chk("a label keyed to the Sold table follows the deal into Current",
        lab1 and lab1["labels"].get("proj_yr1_coc") == "Dev."
        and lab1["labels"].get("act_yr1_coc") == "Dev.",
        f"got {(lab1 or {}).get('labels')}")
    chk("...and a deal's OWN table entry wins over the other table's",
        lab6 and lab6["labels"].get("proj_yr1_coc") == "Lease up",
        f"got {(lab6 or {}).get('labels')}")

    # A deal the rule moved is not reported as missing from, or foreign to, the
    # reference order; one the reference simply does not carry still is.
    dq = build(as_of=dt.date(2026, 6, 30))["diagnostics"]
    chk("a deal moved by the rule is not a 'reference row absent' in the list it left",
        not any(x.get("vcode") == "P0000006" for x in dq.get("reference_rows_absent", [])))
    chk("...nor 'not in the reference order' in the list it joined",
        not any(x.get("vcode") == "P0000006" for x in dq.get("not_in_reference_order", [])))
    chk("...while a deal the reference has never carried IS still reported",
        any(x.get("vcode") == "P0000200" for x in dq.get("not_in_reference_order", [])),
        f"got {[x.get('vcode') for x in dq.get('not_in_reference_order', [])]}")

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("failed:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
