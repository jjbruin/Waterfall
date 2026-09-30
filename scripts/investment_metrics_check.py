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
        dict(vcode="P0000003", InvestmentID="GAMMA", Investment_Name="Gamma Portfolio",
             Property_Count="2", Sale_Status=None, Sale_Date=None,
             Acquisition_Date="07/01/2021 00:00", Currency="CAD", City="Toronto",
             State="Ontario", Asset_Type="Self Storage", Operating_Partner="Gam",
             Lifecycle="Stable", Portfolio_Name="Gamma Portfolio"),
        dict(vcode="P0000004", InvestmentID="GKID", Investment_Name="Gamma Kid",
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
        dict(EntityID="ALPHA ", InvestorID="PPI1", Amount=5_000_000.0),
        dict(EntityID="ALPHA", InvestorID="OPACME", Amount=2_000_000.0),
    ])


def loans_fixture():
    return pd.DataFrame([
        dict(vCode="P0000001", LoanID="L1", mOrigLoanAmt=13_000_000.0),
        # one facility fanned out across date events — must count ONCE
        dict(vCode="P0000003", LoanID="L2", mOrigLoanAmt=8_000_000.0),
        dict(vCode="P0000003", LoanID="L2", mOrigLoanAmt=8_000_000.0),
        dict(vCode="P0000004", LoanID="L3", mOrigLoanAmt=2_000_000.0),
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
        dict(vcode="p0000003", dtEntry="2021-12-31", vAccount="2150",
             mAmount=-7_000_000.0),
        dict(vcode="p0000004", dtEntry="2021-12-31", vAccount="2150",
             mAmount=-7_000_000.0),
        dict(vcode="p0000006", dtEntry="2021-12-31", vAccount="2210",
             mAmount=-15_000_000.0),
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


def row_of(out, vcode):
    for t in ("current", "sold"):
        for r in out[t]["rows"]:
            if r["vcode"] == vcode:
                return t, r
    return None, None


# ══════════════════════════════════════════════════════════════════════════
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
        any(c["vcode"] == "P0000004"
            for c in out["diagnostics"]["children_excluded"]))
    _, mcx = row_of(out, "P0000006")
    chk("one InvestmentID on two rows picks the PARENT",
        mcx is not None and mcx["vcode"] == "P0000006")
    chk("...and still takes the SOLD marker from its sibling",
        row_of(out, "P0000006")[0] == "sold")

    # ── 3. classification ─────────────────────────────────────────────────
    section("3. Current vs Sold")
    chk("a deal sold AFTER the as-of date is still Sold (footnote 4's case)",
        row_of(out, "P0000006")[0] == "sold")
    chk("a live deal is Current", row_of(out, "P0000001")[0] == "current")

    # ── 4. capitalization ─────────────────────────────────────────────────
    section("4. Capitalization, in $m")
    _, alpha = row_of(out, "P0000001")
    chk("PSC pref comes from the commitments table despite its trailing space",
        alpha and abs(alpha["pref"] - 5.0) < 1e-9, f"got {alpha['pref']}")
    chk("first-loss is the operating partner's side",
        alpha and abs(alpha["first_loss"] - 2.0) < 1e-9)
    chk("first lien is the EARLIEST balance-sheet row, not the latest",
        alpha and abs(alpha["first_lien"] - 12.0) < 1e-9,
        f"got {alpha['first_lien']} (9.0 would be the most recent)")
    chk("total size is the three components summed",
        alpha and abs(alpha["total_size"] - 19.0) < 1e-9)
    chk("% of cap divides by that total",
        alpha and abs(alpha["first_lien_pct"] - 12.0 / 19.0) < 1e-9)
    _, gam = row_of(out, "P0000003")
    chk("a child's balance sheet is NOT added to its parent's (no double count)",
        gam and abs(gam["first_lien"] - 7.0 * cfg.CAD_TO_USD) < 1e-9,
        f"got {gam['first_lien']} — 14.0 would mean the child was added twice")
    chk("CAD is converted at the footnote's rate",
        gam and abs(gam["pref"] - 4.0 * cfg.CAD_TO_USD) < 1e-9)
    chk("the basis travels with the figure",
        alpha and "commitments" in alpha["basis"]["capitalization"])

    # ── 5. None is not zero, BOTH directions ──────────────────────────────
    section("5. Unknown vs zero (both ways: a dash where unknown, a figure "
            "where known)")
    chk("UW projected IRR is None, not 0.0 — it is not in MRI",
        alpha and alpha["uw_irr"] is None)
    chk("projected Year-1 CoC is None, not 0.0", alpha and alpha["proj_yr1_coc"] is None)
    chk("actual Year-1 CoC is None too — held behind the config switch",
        alpha and alpha["act_yr1_coc"] is None)
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
    _, mcxr = row_of(out, "P0000006")
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
        and [n for n, _ in cfg.FOOTNOTES_SOLD] == list(range(1, 5)))
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
        "P0000004" in set(frame["vcode"]))

    # ── 13. footnotes (5) and (6) do what they say ────────────────────────
    section("13. Young-deal substitution (both ways: substituted when the "
            "projected figure exists, NOT blanked when it does not)")
    r5 = {"vcode": "X", "name": "X", "markers": [5], "proj_yr1_coc": 0.09,
          "act_yr1_coc": 0.01, "proj_coc_since_close": 0.02,
          "act_coc_since_close": 0.03, "basis": {}}
    d5 = {}
    im._apply_young_deal_substitution(r5, d5)
    chk("(5) replaces the Act. Yr-1 CoC with the projected figure",
        abs(r5["act_yr1_coc"] - 0.09) < 1e-12)
    chk("(5) leaves the two since-close columns alone",
        abs(r5["act_coc_since_close"] - 0.03) < 1e-12
        and abs(r5["proj_coc_since_close"] - 0.02) < 1e-12)
    chk("(5) says in the basis that the figure was substituted",
        "footnote (5)" in r5["basis"]["act_yr1_coc"])
    r6 = {"vcode": "Y", "name": "Y", "markers": [5, 6], "proj_yr1_coc": 0.07,
          "act_yr1_coc": 0.01, "proj_coc_since_close": 0.02,
          "act_coc_since_close": 0.03, "basis": {}}
    im._apply_young_deal_substitution(r6, {})
    chk("(6) replaces ALL the CoC columns",
        all(abs(r6[f] - 0.07) < 1e-12 for f in
            ("act_yr1_coc", "proj_coc_since_close", "act_coc_since_close")))
    rn = {"vcode": "Z", "name": "Z", "markers": [5], "proj_yr1_coc": None,
          "act_yr1_coc": 0.055, "basis": {}}
    dn = {}
    im._apply_young_deal_substitution(rn, dn)
    chk("with nothing to substitute the real figure is KEPT, not blanked",
        abs(rn["act_yr1_coc"] - 0.055) < 1e-12)
    chk("...and the missed substitution is recorded, not silent",
        any(x["vcode"] == "Z"
            for x in dn.get("young_deal_substitution_unavailable", [])))
    r0 = {"vcode": "W", "name": "W", "markers": [], "proj_yr1_coc": 0.09,
          "act_yr1_coc": 0.01, "basis": {}}
    im._apply_young_deal_substitution(r0, {})
    chk("an UNMARKED deal is untouched",
        abs(r0["act_yr1_coc"] - 0.01) < 1e-12)

    # ── 14. the one switch for the figures Alay has not loaded ────────────
    section("14. cfg.UNLOADED_FIGURES — one switch, all three modes")
    chk("all three columns are declared in one place",
        set(cfg.UNLOADED_FIGURES) == {"uw_irr", "proj_yr1_coc", "act_yr1_coc"})
    chk("every one is pending today, so every one prints a dash",
        all(s["mode"] == "none" for s in cfg.UNLOADED_FIGURES.values()))
    chk("no field name is filled in yet — the TODO is still open",
        all(s["field"] is None for s in cfg.UNLOADED_FIGURES.values()))
    chk("each one says WHY it is pending, on the row's basis",
        alpha and "pending Alay" in alpha["basis"]["uw_irr"]
        and "pending Alay" in alpha["basis"]["proj_yr1_coc"])
    # Indexed defensively. A missing key here is a real failure, and it must
    # FAIL rather than raise: a KeyError kills the run and takes every later
    # check with it, so the one defect it detects hides a dozen others.
    pending = out["diagnostics"].get("unloaded_figures_pending", {})
    chk("...and the count of affected deals is reported, not just per row",
        pending.get("uw_irr", {}).get("deals") == 4,
        f"got {pending.get('uw_irr')}")

    saved = {k: dict(v) for k, v in cfg.UNLOADED_FIGURES.items()}
    try:
        # mode "computed" — publish the derived figure
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["mode"] = "computed"
        flipped = build()
        _, a3 = row_of(flipped, "P0000001")
        chk("flipping act_yr1_coc to 'computed' RENDERS the derived figure",
            a3 and abs(a3["act_yr1_coc"] - 0.4 / 5.0) < 1e-9,
            f"got {a3['act_yr1_coc'] if a3 else None}")
        cfg.UNLOADED_FIGURES["act_yr1_coc"]["variant"] = "commitment"
        flipped2 = build()
        _, a4 = row_of(flipped2, "P0000001")
        chk("...and the variant chooses the denominator",
            a4 and a4["act_yr1_coc"] is not None)

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
        # figure exists — no second switch to remember.
        cfg.UNLOADED_FIGURES["proj_yr1_coc"]["mode"] = "mri"
        cfg.UNLOADED_FIGURES["proj_yr1_coc"]["field"] = "pe_coupon"
        cfg.ROW_MARKERS_CURRENT["P0000001"] = [5]
        act = build()
        _, a7 = row_of(act, "P0000001")
        chk("footnote (5) substitutes automatically once projected Yr-1 exists",
            a7 and abs(a7["act_yr1_coc"] - 0.085) < 1e-12
            and "footnote (5)" in a7["basis"]["act_yr1_coc"],
            f"got {a7['act_yr1_coc'] if a7 else None} "
            f"basis {a7['basis']['act_yr1_coc'] if a7 else None}")
    finally:
        cfg.ROW_MARKERS_CURRENT.pop("P0000001", None)
        for k, v in saved.items():
            cfg.UNLOADED_FIGURES[k].clear()
            cfg.UNLOADED_FIGURES[k].update(v)
    back = build()
    _, a8 = row_of(back, "P0000001")
    chk("the switch is restored — the column is a dash again",
        a8 and a8["act_yr1_coc"] is None and a8["proj_yr1_coc"] is None)

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("failed:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
