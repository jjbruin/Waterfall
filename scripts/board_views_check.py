"""Guardrail: the Board's Phase 1 views (pages 26, 27, 29-31) compose engines, never compute.

Run: .venv\\Scripts\\python scripts\\board_views_check.py

What it proves, on fixtures shaped like the engines' own payloads:

1. Page 26's capital totals ARE the PE exposure engine's totals, to the cent --
   the page only adds the engine's per-deal figures, and a sum that drifted from
   the engine's own would be a second engine.
2. Wholly owned vs joint venture, BOTH ways: a deal with no third-party capital
   is wholly owned; one with any is a JV; a deal with only an unfunded
   commitment is classed by the engine's split of it and adds nothing to the
   capital columns; a deal with neither is on its own line, never guessed.
3. What one engine has and the other lacks is COUNTED AND NAMED: a PE holding
   Investment Metrics does not carry, a deal without a Total Size (None, never
   0), a twin vcode folded into its deal.
4. Page 27 reads Investment Metrics' totals and the PE grand total unchanged,
   and its combined CoC is the engine's own pref-weighted average.
5. Investment summaries are the Investment Metrics payload, through the shared
   cache -- the module never calls the engine's build itself.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask_app.services import board_views_service as bv  # noqa: E402
import investment_metrics as engine  # noqa: E402

PASSED, FAILED = 0, []


def chk(label, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print("   ok  ", label)
    else:
        FAILED.append(label)
        print("   FAIL", label, detail)


G = ("PSC", "KOC", "TIAA", "Clarion", "F&F", "Declaration", "Ambassadors")


def split(**kw):
    return {g: kw.get(g.replace("&", "").replace("F", "FF") if g == "F&F" else g, 0.0) for g in G}


def holding(vcode, name, cost, **groups):
    return {"vcode": vcode, "deal_name": name, "holder": "H" + vcode[-2:], "cost": cost,
            "fx_missing": False, "cost_by_group": {g: groups.get(g, 0.0) for g in G}}


def im_row(vcode, name, total_size, pref, proceeds, coc, twins=None, **kw):
    return {"vcode": vcode, "name": name, "total_size": total_size, "pref": pref,
            "proceeds": proceeds, "act_coc_since_close": coc, "twin_vcodes": twins or [vcode],
            "labels": kw.get("labels", {})}


# PE exposure: WO (all PSC), JV, a twin holding, a holding IM lacks, an unfunded-only deal.
PE = {
    "as_of": "2025-12-31",
    "rows": [
        holding("P0000001", "Wholly", 10_000_000.0, PSC=10_000_000.0),
        holding("P0000002", "Venture", 50_000_000.0, PSC=5_000_000.0, TIAA=40_000_000.0, KOC=5_000_000.0),
        holding("P0000012", "Venture twin", 2_000_000.0, PSC=1_000_000.0, Declaration=1_000_000.0),
        holding("P0000003", "Not in IM", 4_000_000.0, PSC=4_000_000.0),
        # Float dust from the chain multiply must not make a wholly owned deal a JV.
        holding("P0000006", "Wholly dust", 3_000_000.0, PSC=3_000_000.0 - 1e-7, TIAA=1e-7),
    ],
    "future_funding": [
        {"vcode": "P0000004", "deal_name": "Unfunded only", "remaining_to_fund_usd": 20_000_000.0,
         "by_group": {g: (2_000_000.0 if g == "PSC" else 18_000_000.0 if g == "TIAA" else 0.0) for g in G}},
    ],
    "totals": {"cost": 69_000_000.0,
               "cost_by_group": {"PSC": 23_000_000.0, "KOC": 5_000_000.0, "TIAA": 40_000_000.0,
                                 "Clarion": 0.0, "F&F": 0.0, "Declaration": 1_000_000.0, "Ambassadors": 0.0},
               "future_funding": 20_000_000.0, "grand_cost": 89_000_000.0},
    "notes": [],
}
IM = {
    "as_of": "2025-12-31", "as_of_display": "31 Dec 25",
    "current": {"rows": [
        im_row("P0000001", "Wholly", 30.0, 10.0, 2.0, 0.08),
        im_row("P0000002", "Venture", 200.0, 52.0, 6.0, 0.06, twins=["P0000002", "P0000012"]),
        im_row("P0000004", "Unfunded only", 40.0, 20.0, 0.0, None, labels={"act_coc_since_close": "Dev."}),
        im_row("P0000005", "No exposure", None, 1.0, 0.1, 0.05),
        im_row("P0000006", "Wholly dust", 12.0, 3.0, 0.3, 0.07),
    ], "total": {"total_size": 282.0, "pref": 86.0, "proceeds": 8.4, "uw_irr": 0.15,
                 "act_coc_since_close": 0.065}},
    "sold": {"rows": [im_row("P0000009", "Sold one", 20.0, 5.0, 9.0, 0.10)],
             "total": {"pref": 5.0, "proceeds": 9.0, "realized_irr": 0.18, "act_coc_since_close": 0.10}},
    "grand_total": {"proceeds": 17.4},
}
PROPS = {"P0000001": 3, "P0000002": 1, "P0000012": 0, "P0000003": 2, "P0000004": 1, "P0000005": None,
         "P0000006": 1}

print("\n1. Page 26 totals are the PE exposure engine's")
cap = bv.compose_capitalization(PE, IM, PROPS)
lines = {x["key"]: x for x in cap["lines"]}
t = cap["total"]
chk("total net pref == the engine's funded cost, to the cent", abs(t["total"] - PE["totals"]["cost"]) < 0.01,
    (t["total"], PE["totals"]["cost"]))
chk("PSC == the engine's PSC column", abs(t["psc"] - PE["totals"]["cost_by_group"]["PSC"]) < 0.01, t["psc"])
third = sum(v for g, v in PE["totals"]["cost_by_group"].items() if g != "PSC")
chk("3rd party == the engine's other six columns", abs(t["third_party"] - third) < 0.01, t["third_party"])
chk("the reconciliation block says the same",
    abs(cap["reconciliation"]["page_total_net_pref"] - cap["reconciliation"]["pe_funded_cost"]) < 0.01)
chk("3rd-party sources are the engine's columns, PSC left out",
    [s["group"] for s in cap["third_party_sources"]] == list(bv.THIRD_PARTY)
    and cap["third_party_sources"][0]["amount"] == 40_000_000.0)
chk("...and their shares add to 100%", abs(sum(s["share"] or 0 for s in cap["third_party_sources"]) - 1) < 1e-9)
chk("the unfunded footnote is the engine's future funding", cap["unfunded"] == 20_000_000.0)

print("\n2. Wholly owned vs JV, both ways")
wo, jv = lines["wholly_owned"], lines["joint_venture"]
chk("all-PSC deals are wholly owned (dust in the split does not make a JV)",
    set(wo["deal_names"]) == {"Wholly", "Wholly dust", "Not in IM"}, wo["deal_names"])
chk("a deal with third-party capital is a JV", "Venture" in jv["deal_names"], jv["deal_names"])
chk("an unfunded-only deal is classed by the engine's split of its commitment (TIAA -> JV)",
    "Unfunded only" in jv["deal_names"])
uo = next(d for d in cap["deals"] if d["vcode"] == "P0000004")
chk("...and adds NOTHING to the capital columns (None, not 0)",
    uo["cost"] is None and uo["psc"] is None and uo["third_party"] is None, uo)
chk("...and is named in the notes", any("Unfunded only" in n and "unfunded commitment" in n for n in cap["notes"]))
chk("a deal with neither funded nor unfunded capital is on its own line",
    lines.get("no_exposure", {}).get("deal_names") == ["No exposure"], lines.get("no_exposure"))
chk("...whose capital cells are None, not 0",
    lines["no_exposure"]["psc"] is None and lines["no_exposure"]["total"] is None)

print("\n3. What one engine lacks is counted and named")
chk("the twin vcode's capital folds into its deal (Venture = 50M + 2M)",
    next(d for d in cap["deals"] if d["vcode"] == "P0000002")["cost"] == 52_000_000.0)
chk("...and the twin is not a deal of its own", not any(d["vcode"] == "P0000012" for d in cap["deals"]))
chk("a PE holding Investment Metrics lacks is counted", any(d["vcode"] == "P0000003" for d in cap["deals"]))
chk("...with no gross capitalization (None) and a note saying why",
    next(d for d in cap["deals"] if d["vcode"] == "P0000003")["gross_cap"] is None
    and any("Not in IM" in n and "not in Investment Metrics" in n for n in cap["notes"]))
chk("a deal without a Total Size is named on its line", lines["no_exposure"]["gross_cap_missing"] == ["No exposure"])
chk("...and a line whose every deal lacks one has gross cap None, not 0", lines["no_exposure"]["gross_cap"] is None)
chk("gross cap sums Investment Metrics' Total Size ($M -> $)",
    wo["gross_cap"] == 42_000_000.0 and jv["gross_cap"] == 240_000_000.0, (wo["gross_cap"], jv["gross_cap"]))
chk("deals = every IM current row + the holding IM lacks", t["deals"] == 6, t["deals"])
chk("properties: MRI's count per deal, twins summed; a missing count is named",
    wo["properties"] == 3 + 2 + 1 and jv["properties"] == 1 + 0 + 1
    and lines["no_exposure"]["properties_missing"] == ["No exposure"], (wo["properties"], jv["properties"]))

print("\n4. Page 27 reads the engines' totals unchanged")
perf = bv.compose_performance(PE, IM)
cur, ex = perf["rows"]
chk("current pref == the PE grand total (funded + unfunded)", cur["pref"] == PE["totals"]["grand_cost"])
chk("current proj IRR, proceeds, CoC == IM Current totals",
    cur["proj_irr"] == 0.15 and cur["proceeds"] == 8.4e6 and cur["coc"] == 0.065, cur)
chk("exited pref, realized IRR, proceeds, CoC == IM Sold totals",
    ex["pref"] == 5e6 and ex["realized_irr"] == 0.18 and ex["proceeds"] == 9e6 and ex["coc"] == 0.10, ex)
chk("no realized IRR on current, no projected IRR on exited (None, not 0)",
    cur["realized_irr"] is None and ex["proj_irr"] is None)
chk("total proceeds == IM's grand total", perf["total"]["proceeds"] == 17.4e6)
want = engine.pref_weighted_average(IM["current"]["rows"] + IM["sold"]["rows"], "act_coc_since_close")
chk("combined CoC == the engine's own pref-weighted average over both tables",
    perf["total"]["coc"] == want, (perf["total"]["coc"], want))
chk("...which skips a labelled (Dev.) row's weight", abs(want - (10 * .08 + 52 * .06 + 1 * .05 + 3 * .07 + 5 * .10)
                                                        / (10 + 52 + 1 + 3 + 5)) < 1e-12, want)

print("\n5. The views go through the shared engines")
src = open(os.path.join(ROOT, "flask_app", "services", "board_views_service.py"), encoding="utf-8").read()
chk("no direct call to build_investment_metrics (the shared cache is the way in)",
    "build_investment_metrics(" not in src)
chk("Investment Metrics is read through investment_metrics_service.get_report", "ims.get_report(" in src)
chk("PE exposure is read through its cached get_report", "pe_svc.get_report(" in src)
try:
    bv.build_view("debt", __import__("datetime").date(2025, 12, 31), data={})
    chk("a schedule with no view yet is refused", False)
except KeyError:
    chk("a schedule with no view yet is refused (KeyError -> 404 at the route)", True)
from flask_app.services import board_service  # noqa: E402
flagged = {s["key"] for s in board_service.SCHEDULES if s.get("view")}
chk("the catalog marks exactly the schedules that have a view", flagged == set(bv.VIEW_KEYS), flagged)

print("\n6. The deck layout (pp. 29-31) is the server's, and loses nothing")
import investment_metrics_config as imcfg  # noqa: E402
cfg_keys = {"current": [c[0] for c in imcfg.COLUMNS_CURRENT], "sold": [c[0] for c in imcfg.COLUMNS_SOLD]}
for t in ("current", "sold"):
    gone = [k for k in bv.DECK_COLUMNS[t] if k not in cfg_keys[t]]
    chk(f"every {t} deck column is an Investment Metrics column", not gone, gone)
    chk(f"...in Investment Metrics' own order ({t})",
        [k for k in cfg_keys[t] if k in bv.DECK_COLUMNS[t]] == list(bv.DECK_COLUMNS[t]))
chk("the deck drops DMA/Location and the Year-1 CoC columns, as the January deck does",
    not {"dma", "proj_yr1_coc", "act_yr1_coc"} & set(bv.DECK_COLUMNS["current"] + bv.DECK_COLUMNS["sold"]))


def im_payload(n_cur, n_sold, keys=None):
    def cols(t):
        return [{"key": k, "row1": "", "row2": "", "row3": k, "width": 20, "align": "center"}
                for k in (keys or cfg_keys)[t]]
    return {"as_of": "2025-12-31",
            "current": {"rows": [{"vcode": f"C{i}"} for i in range(n_cur)], "columns": cols("current")},
            "sold": {"rows": [{"vcode": f"S{i}"} for i in range(n_sold)], "columns": cols("sold")}}


lay = bv.deck_layout(im_payload(53, 19))
cur = [x for x in lay["slides"] if x["table"] == "current"]
sold = [x for x in lay["slides"] if x["table"] == "sold"]
chk("53 current deals -> two pages (30 + 23), as the deck's pp. 29-30",
    [(x["first"], x["last"]) for x in cur] == [(0, 30), (30, 53)], cur)
chk("19 exited deals -> one page", [(x["first"], x["last"]) for x in sold] == [(0, 19)])
chk("every row is on exactly one page, in the engine's order",
    sum(x["last"] - x["first"] for x in lay["slides"]) == 72
    and all(a["last"] == b["first"] for a, b in zip(cur, cur[1:])))
chk("the Total goes on each table's LAST page only",
    [x["is_last"] for x in cur] == [False, True] and [x["is_last"] for x in sold] == [True])
chk("titles: the first page, then Cont'd.",
    cur[0]["title"] == "Current Portfolio*" and cur[1]["title"].startswith("Current Portfolio, Cont")
    and sold[0]["title"] == "Exited Investments*")
chk("the footnote carries the schedule's ONE as-of date, written as the deck writes it",
    "closed through 12/31/25. Cash received through 12/31/25." in cur[0]["footnote"])
chk("exactly 30 rows -> one page, not an empty second one",
    len([x for x in bv.deck_layout(im_payload(30, 1))["slides"] if x["table"] == "current"]) == 1)
chk("no rows -> still one page, so the table and its Total are shown",
    len([x for x in bv.deck_layout(im_payload(0, 0))["slides"] if x["table"] == "sold"]) == 1)
renamed = {"current": [k if k != "uw_irr" else "uw_irr_v2" for k in cfg_keys["current"]], "sold": cfg_keys["sold"]}
lay2 = bv.deck_layout(im_payload(5, 5, renamed))
chk("a column the payload no longer carries is REPORTED, not silently dropped",
    lay2["missing_columns"] == ["current.uw_irr"], lay2["missing_columns"])

print("\n7. The screen turns pages without asking the server again")
deck_src = open(os.path.join(ROOT, "vue_app", "src", "components", "board", "BoardDeck.vue"), encoding="utf-8").read()
gets = [ln.strip() for ln in deck_src.splitlines() if "api.get(" in ln]
chk("the deck asks the server for one thing only: a schedule's view",
    len(gets) == 1 and "/schedules/${s.key}/view" in gets[0], gets)
chk("...and keeps it, keyed by meeting, schedule AND as-of date (a new date refetches)",
    "`${props.meeting.id}|${s.key}|${s.as_of}`" in deck_src and "CACHE.has(k)" in deck_src)
slide_src = open(os.path.join(ROOT, "vue_app", "src", "components", "board", "InvestmentSummarySlide.vue"),
                 encoding="utf-8").read()
chk("investment-summary cells are written by Investment Metrics' own formatter",
    "from '@/utils/investmentMetricsFormat'" in slide_src and "cellText(r, c)" in slide_src)
chk("...and its columns and pages come from the server's layout, not the screen",
    "view.deck.columns" in slide_src and "props.slide.first" in slide_src)

print("\n8. Page 23: exposure by asset class is the PE engine's, grouped, never guessed")
AT = {"P0000001": "Multifamily", "P0000002": "Retail - Non Groc.", "P0000012": "Retail - Non Groc.",
      "P0000003": "Retail - Grocery", "P0000006": "RV Park", "P0000004": "Retail"}
PE23 = {**PE, "future_funding": PE["future_funding"] + [
    {"vcode": "P0000007", "deal_name": "Multi-holder", "remaining_to_fund_usd": 3_000_000.0, "by_group": None}],
    "totals": {**PE["totals"], "grand_cost": 69_000_000.0 + 23_000_000.0,
               "grand_cost_by_group": {"PSC": 23_000_000.0 + 2_000_000.0}}}
ac = bv.compose_asset_class(PE23, AT)
by = {r["label"]: r for r in ac["rows"]}
chk("total == the engine's grand total (funded + unfunded), to the cent",
    abs(ac["total"]["total"] - PE23["totals"]["grand_cost"]) < 0.01, ac["total"])
chk("PSC == the engine's PSC grand total", abs(ac["total"]["psc"] - 25_000_000.0) < 0.01, ac["total"]["psc"])
chk("a deal's funded and unfunded land in ITS class (Wholly, MF: 10M funded)",
    by["Multifamily"]["total"] == 10_000_000.0)
chk("a twin vcode is classed by its own Asset_Type (Venture 50M + twin 2M -> Non-Grocery)",
    by["Non-Grocery Retail"]["total"] == 52_000_000.0)
chk("RV Park is Other, as the deck groups it", by["Other"]["total"] == 3_000_000.0)
chk("both MRI spellings of non-grocery retail are one class",
    bv.asset_class("Retail - Non Groc.") == bv.asset_class("Retail - Non-Grocery") == "Non-Grocery Retail")
chk("an Asset_Type the deck does not name is its OWN row, not absorbed (plain 'Retail')",
    "Retail" in by and by["Retail"]["total"] == 20_000_000.0 and not by["Retail"]["in_deck"])
chk("...and is named in the notes", any("Asset_Type Retail is not one of the deck's classes" in n for n in ac["notes"]))
chk("a deal with no Asset_Type is its own Unclassified row, not dropped",
    by.get(bv.UNCLASSIFIED, {}).get("total") == 3_000_000.0)
chk("an unsplit unfunded commitment is in the total, not in PSC, and named",
    bv.UNCLASSIFIED in by and by[bv.UNCLASSIFIED]["psc"] is None
    and any("Multi-holder" in n and "not in PSC" in n for n in ac["notes"]))
chk("every deck class is a row even when empty, in the deck's order",
    [r["label"] for r in ac["rows"]][:5] == list(bv.ASSET_CLASSES))
chk("an empty deck class shows a dash (None), not $0.0",
    by["Self Storage"]["total"] is None and by["Self Storage"]["total_share"] is None)
chk("shares are of the page's own total and add to 100%",
    abs(sum(r["total_share"] or 0 for r in ac["rows"]) - 1) < 1e-9
    and abs(sum(r["psc_share"] or 0 for r in ac["rows"]) - 1) < 1e-9)
chk("p.23 is on the catalog and in the deck's view keys",
    "exposure_asset_class" in bv.VIEW_KEYS
    and "exposure_asset_class" in {s_["key"] for s_ in board_service.SCHEDULES if s_.get("view")})

print("\n%d passed, %d failed" % (PASSED, len(FAILED)))
for f in FAILED:
    print("  -", f)
sys.exit(1 if FAILED else 0)
