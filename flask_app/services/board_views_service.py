"""The Board's Phase 1 schedule views: pages 26, 27 and 29-31 of the deck.

NOTHING HERE IS A CALCULATION. Every figure is read off an engine the app
already owns, at the schedule's own as-of date (ONE NUMBER, ONE ENGINE):

  | Page | Figure                                   | Engine                                       |
  |------|------------------------------------------|----------------------------------------------|
  | 26   | PSC / 3rd-party capital, by deal         | pe_exposure_service (funded Cost, split by   |
  |      |   and by investor; the unfunded footnote |   group_shares) -- accounting's tracker      |
  | 26   | Total gross capitalization, deals        | Investment Metrics, Current table (Total     |
  |      |                                          |   Size, the deals held at the as-of)         |
  | 26   | Properties                               | deals.Property_Count (MRI)                   |
  | 27   | Pref incl. unfunded (current)            | pe_exposure_service grand total              |
  | 27   | Pref, IRRs, proceeds, CoC                | Investment Metrics Total / Average rows      |
  | 27   | Combined CoC                             | investment_metrics.pref_weighted_average --  |
  |      |                                          |   the engine's own Total / Average rule      |
  | 29-31| The investment summaries                 | Investment Metrics, the payload the tab shows|
  | 28   | Occupancy, DSCR by asset class           | one_pager.get_property_performance per deal  |
  |      |                                          |   (YTD economic occupancy, YTD DSCR), rolled |
  |      |                                          |   up by the Snapshot's own rules: occupancy  |
  |      |                                          |   NOI-weighted, DSCR debt-weighted, dev out  |
  | 28   | Fixed / floating debt, maturities, rates | dashboard_service.get_loan_maturity_data     |
  |      | Rate caps, max interest                  | loan_caps.cap_terms                          |
  | 5    | 3rd-party AUM and its year-on-year growth| pe_exposure_service at the as-of AND a year  |
  |      |   by investor                            |   earlier (funded cost; unfunded by group)   |
  | 5    | Investment activity: new deals, new      | Investment Metrics rows (Current + Sold) by  |
  |      |   partners, exits                        |   PSC Invest. Date / sale date in the year   |
  | 9    | Pref invested by year, new + cumulative  | Investment Metrics' PSC pref per deal (Current|
  |      |                                          |   + Sold), in its PSC Invest. Date's year    |
  | 24   | The same, by operating partner           | as p. 23, grouped by deals.Operating_Partner |
  |      |                                          |   through PARTNER_NAMES                      |
  | 23   | Net pref incl. unfunded by asset class,  | pe_exposure_service: funded Cost + remaining |
  |      |   total and PSC                          |   to fund per deal, PSC by its split; the    |
  |      |                                          |   class is MRI's deals.Asset_Type, grouped   |

The only arithmetic is ADDING an engine's per-deal figures into the page's
rows, and every such total is checked against the engine's own total in
``reconciliation`` -- a sum here that disagreed with the engine would be a
second engine.

What one engine has and the other does not is SAID, deal by deal, in ``notes``:
a deal held at the as-of with no PE exposure row, or a PE exposure holding that
Investment Metrics does not carry, is counted where it can be and named. A
figure an engine cannot give is ``None``, never 0.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

#: Below this (in dollars), a third-party share is nothing: the split multiplies
#: shares through the ownership chain and leaves float dust.
_DUST = 0.5

#: The investor groups on page 26's 3rd-party table, in the deck's order.
THIRD_PARTY = ("TIAA", "KOC", "Clarion", "F&F", "Declaration", "Ambassadors")
GROUP_LABELS = {"PSC": "PSC", "TIAA": "TIAA", "KOC": "Knights of Columbus", "Clarion": "Clarion",
                "F&F": "Friends & Family", "Declaration": "Declaration", "Ambassadors": "Ambassadors"}

VIEW_KEYS = ("year_in_review", "pref_by_year", "exposure_asset_class", "exposure_partner", "capitalization", "performance", "debt",
             "investment_summaries")

# ------------------------------------------------------------------ the deck's layout
#
# HOW A SCHEDULE IS LAID OUT ON THE SLIDE is decided here, not in the screen, so
# there is one copy of it. None of it is a figure: titles, footnote wording, which
# of an engine's columns the deck prints, and how many rows fit one page.

#: The deck's slide titles (Jan 2026 deck, pp. 26-27).
SLIDE_TITLES = {"exposure_asset_class": "Exposure: Asset Class",
                "exposure_partner": "Exposure: Operating Partner (Current Deals)",
                "capitalization": "Current Portfolio Capitalization",
                "performance": "Performance: Portfolio Level",
                "debt": "Portfolio Metrics"}

#: The deck's asset classes (p. 23), in its order, from MRI's ``deals.Asset_Type``.
#: A GROUPING, not a figure. The Portfolio Snapshot has its own rollup
#: (``portfolio_snapshot_summary.ASSET_TYPE_ROLLUP``) because TIAA's report folds
#: every retail type into one bucket; the board deck splits grocery-anchored from
#: non-grocery and puts the small types in "Other". As there, an Asset_Type this
#: map does not name is its OWN row, named in the notes -- never absorbed into a
#: neighbour (plain "Retail", with no sub-type, is exactly that case today).
ASSET_CLASSES = ("Multifamily", "Non-Grocery Retail", "Grocery-Anchored Retail", "Self Storage", "Other")
ASSET_CLASS_OF = {
    "multifamily": "Multifamily",
    # MRI spells this two ways (Oct 8 2026: 23 deals "Retail - Non Groc.", and the two
    # reclassified that day, Merle Hay and 5-15 Broad, "Retail - Non-Grocery").
    "retail - non groc.": "Non-Grocery Retail", "retail - non groc": "Non-Grocery Retail",
    "retail - non-grocery": "Non-Grocery Retail",
    "retail - grocery": "Grocery-Anchored Retail",
    "self storage": "Self Storage", "self-storage": "Self Storage",
    "industrial": "Other", "rv park": "Other", "resort": "Other",
}
UNCLASSIFIED = "Unclassified (no Asset_Type)"

#: MRI's ``deals.Operating_Partner`` spellings -> the deck's short name (p. 24, and
#: the partner column of pp. 29-30). SPELLING VARIANTS ONLY: "JPI" and "JPI
#: Companies" are one partner. Where MRI and the January deck name DIFFERENT
#: partners for a deal -- Brainerd / Crowne Plaza ("Bertram and DiMarco",
#: "Bertram/Pyramid" vs the deck's "Bright Ravens"), JB Fair Park ("Dave West" vs
#: "L. Allen"), The Gallery ("Manhattan Five" vs "MFP") -- nothing is mapped: that
#: is a question about who the partner is, for asset management (open_items), not
#: a spelling. A name not listed passes through as MRI has it, and is named.
PARTNER_NAMES = {
    "abbell associates": "Abbell", "apple": "Apple", "apple self storage": "Apple",
    "ashcroft capital": "Ashcroft", "athena r.e.": "Athena", "berger communities": "Berger",
    "burton property group": "Burton", "capreit": "CAPREIT", "colony hills": "Colony Hills",
    "dave west": "D. West", "elan multifamily investments": "Elan", "evergreen devco, inc.": "Evergreen",
    "flag wharf": "Flag Wharf", "jpi": "JPI", "jpi companies": "JPI", "kempner properties": "Kempner",
    "lbx investments": "LBX", "mcb real estate": "MCB", "mccord development, inc.": "McCord",
    "orei": "OREI", "pegasus investment partners": "Pegasus", "pmat": "PMAT", "prestige": "Prestige",
    "pyramid": "Pyramid", "rainier companies": "Rainier", "rcg ventures": "RCG", "vastgood": "Vastgood",
    "vastgood properties llc": "Vastgood",
}
NO_PARTNER = "(no Operating_Partner in MRI)"


def partner_name(raw) -> str:
    """The deck's name for an MRI ``Operating_Partner``; an unlisted spelling passes through."""
    r = "" if raw is None or raw != raw else str(raw).strip()
    if not r:
        return NO_PARTNER
    return PARTNER_NAMES.get(r.lower(), r)

#: The board deck prints a NARROWER Investment Metrics table than the tab (pp.
#: 29-31): no DMA/Location and no Year-1 CoC columns, and the Sold page shows
#: only the actual CoC. Keys are Investment Metrics' own; every one must exist in
#: its column config (board_views_check), or a renamed column would vanish from
#: the deck without a word.
DECK_COLUMNS = {
    "current": ("name", "asset_class", "invest_date", "partner", "total_size",
                "first_lien", "first_lien_pct", "pref", "pref_pct", "first_loss",
                "first_loss_pct", "uw_irr", "proceeds", "proj_coc_since_close",
                "act_coc_since_close", "pref_coupon", "residual_cf_split", "irr_lookback"),
    "sold": ("name", "asset_class", "invest_date", "partner", "total_size",
             "first_lien", "first_lien_pct", "pref", "pref_pct", "first_loss",
             "first_loss_pct", "uw_irr", "realized_irr", "proceeds",
             "act_coc_since_close", "pref_coupon", "residual_cf_split", "irr_lookback"),
}

#: Deal rows on one investment-summary slide -- the deck's p. 29 carries 30.
ROWS_PER_SLIDE = 30

_SUMMARY_TITLES = {"current": ("Current Portfolio*", "Current Portfolio, Cont\u2019d.*"),
                   "sold": ("Exited Investments*", "Exited Investments, Cont\u2019d.*")}


def _short_date(iso: Optional[str]) -> str:
    """2025-12-31 -> 12/31/25, the deck's own spelling."""
    if not iso:
        return ""
    d = date.fromisoformat(str(iso)[:10])
    return f"{d.month}/{d.day}/{d.strftime('%y')}"


def deck_layout(im: dict) -> dict:
    """The investment-summary slides: the deck's columns, and the rows on each page.

    Pages split the engine's rows in the engine's order; the Total row (and the
    Sold table's Grand Total) goes on each table's LAST page. A column the deck
    names that the payload does not carry is reported, never silently dropped.
    """
    as_of = _short_date(im.get("as_of"))
    columns, missing, slides = {}, [], []
    for t in ("current", "sold"):
        by_key = {c["key"]: c for c in (im.get(t) or {}).get("columns") or []}
        columns[t] = [by_key[k] for k in DECK_COLUMNS[t] if k in by_key]
        missing += [f"{t}.{k}" for k in DECK_COLUMNS[t] if k not in by_key]
        n = len((im.get(t) or {}).get("rows") or [])
        starts = list(range(0, n, ROWS_PER_SLIDE)) or [0]
        for i, a in enumerate(starts):
            slides.append({
                "table": t, "first": a, "last": min(a + ROWS_PER_SLIDE, n),
                "is_last": i == len(starts) - 1,
                "title": _SUMMARY_TITLES[t][0 if i == 0 else 1],
                "footnote": (f"*Portfolio investments closed through {as_of}. Cash received through "
                             f"{as_of}. Pref Equity includes underwritten, capital call but does not "
                             f"include return of capital" if t == "current" else
                             f"*Exited investments through {as_of}. Gross Pref Equity includes "
                             f"underwritten, capital call but does not include return of capital."),
            })
    # The partner column prints the deck's short name (PARTNER_NAMES): display only,
    # the Investment Metrics payload itself is untouched.
    short = {}
    for t in ("current", "sold"):
        for r in (im.get(t) or {}).get("rows") or []:
            raw = r.get("partner")
            if raw and partner_name(raw) != str(raw).strip():
                short[str(raw)] = partner_name(raw)
    return {"columns": columns, "slides": slides, "missing_columns": missing, "partner_short": short}


def _add(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    return a if b is None else a + b


# ------------------------------------------------------------------ page 26

def compose_capitalization(pe: dict, im: dict, property_counts: Dict[str, Optional[int]]) -> dict:
    """Page 26 from the PE exposure report and the Investment Metrics payload.

    A DEAL is a row of Investment Metrics' Current table (twins merged as that
    engine merges them), plus any PE exposure holding at the as-of that table
    does not carry -- named in the notes. A deal is WHOLLY OWNED when it has
    funded capital and none of it is third-party; JOINT VENTURE when some is.

    A deal with NO FUNDED CAPITAL at the as-of but an unfunded commitment (a deal
    that has closed and not yet drawn -- Jefferson Stephens at 31 Dec 2025) is
    classed by the PE exposure engine's split of that commitment; its capital
    columns stay empty, since the page's capital is funded only, and the
    commitment is in the unfunded footnote. A deal with neither is shown on its
    own line rather than guessed into one.
    """
    notes: List[str] = list(pe.get("notes") or [])
    im_rows = (im.get("current") or {}).get("rows") or []
    deal_of: Dict[str, str] = {}
    deals: Dict[str, dict] = {}
    for r in im_rows:
        twins = r.get("twin_vcodes") or [r["vcode"]]
        for v in twins:
            deal_of[v] = r["vcode"]
        pc = [property_counts.get(v) for v in twins]
        deals[r["vcode"]] = {
            "vcode": r["vcode"], "name": r["name"], "in_investment_metrics": True,
            "gross_cap": r.get("total_size") * 1e6 if r.get("total_size") is not None else None,
            "properties": None if all(p is None for p in pc) else sum(p or 0 for p in pc),
            "cost": None, "psc": None, "third_party": None, "fx_missing": False,
            "by_group": {g: None for g in GROUP_LABELS},
        }

    for h in pe.get("rows") or []:
        key = deal_of.get(h["vcode"], h["vcode"])
        d = deals.get(key)
        if d is None:
            pc = property_counts.get(h["vcode"])
            d = deals[key] = {
                "vcode": key, "name": h["deal_name"], "in_investment_metrics": False,
                "gross_cap": None, "properties": pc, "cost": None, "psc": None,
                "third_party": None, "fx_missing": False, "by_group": {g: None for g in GROUP_LABELS},
            }
        if h.get("fx_missing") or h.get("cost") is None:
            d["fx_missing"] = d["fx_missing"] or bool(h.get("fx_missing"))
            continue
        d["cost"] = _add(d["cost"], h["cost"])
        for g in GROUP_LABELS:
            d["by_group"][g] = _add(d["by_group"][g], (h.get("cost_by_group") or {}).get(g))

    unfunded: Dict[str, dict] = {}
    for f in pe.get("future_funding") or []:
        unfunded[deal_of.get(f["vcode"], f["vcode"])] = f

    def classify(psc: float, third: float, positive: float) -> str:
        if abs(third) < _DUST and positive > _DUST:
            return "wholly_owned"
        return "joint_venture"

    for d in deals.values():
        d["unfunded"] = None
        f = unfunded.get(d["vcode"])
        split = (f or {}).get("by_group")
        if d["cost"] is not None:
            d["psc"] = d["by_group"]["PSC"] or 0.0
            d["third_party"] = sum(d["by_group"][g] or 0.0 for g in THIRD_PARTY)
            d["category"] = classify(d["psc"], d["third_party"], d["cost"])
            d["classed_by"] = "funded"
        elif f and split and (f.get("remaining_to_fund_usd") or 0) > _DUST:
            d["unfunded"] = f["remaining_to_fund_usd"]
            d["category"] = classify(split.get("PSC") or 0.0,
                                     sum(split.get(g) or 0.0 for g in THIRD_PARTY),
                                     f["remaining_to_fund_usd"])
            d["classed_by"] = "unfunded"
            notes.append(f"{d['name']} ({d['vcode']}) has no funded capital at the as-of; it is classed "
                         f"by its unfunded commitment (${f['remaining_to_fund_usd'] / 1e6:,.2f}M, in the "
                         f"unfunded footnote) and adds nothing to the capital columns.")
        else:
            d["category"] = "no_exposure"
            d["classed_by"] = None
        if not d["in_investment_metrics"]:
            notes.append(f"{d['name']} ({d['vcode']}) has PE exposure at the as-of but is not in "
                         f"Investment Metrics' Current table, so it has no gross capitalization here.")
        if d["fx_missing"]:
            notes.append(f"{d['name']} ({d['vcode']}): no USD/CAD rate at the as-of, so its capital "
                         f"is left out of the USD figures.")

    def line(cat: str, label: str) -> dict:
        ds = [d for d in deals.values() if d["category"] == cat]
        no_gc = [d for d in ds if d["gross_cap"] is None]
        no_pc = [d for d in ds if d["properties"] is None]
        return {
            "key": cat, "label": label, "deals": len(ds),
            "properties": sum(d["properties"] or 0 for d in ds),
            "properties_missing": [d["name"] for d in no_pc],
            "gross_cap": sum(d["gross_cap"] for d in ds if d["gross_cap"] is not None) if len(no_gc) < len(ds) else None,
            "gross_cap_missing": [d["name"] for d in no_gc],
            "psc": None if cat == "no_exposure" else sum(d["psc"] or 0.0 for d in ds),
            "third_party": None if cat == "no_exposure" else sum(d["third_party"] or 0.0 for d in ds),
            "total": None if cat == "no_exposure" else sum(d["cost"] or 0.0 for d in ds),
            "deal_names": sorted(d["name"] for d in ds),
        }

    lines = [line("wholly_owned", "Wholly Owned"), line("joint_venture", "JV Partnerships")]
    other = line("no_exposure", "Held, no PE exposure at the as-of")
    if other["deals"]:
        lines.append(other)
        notes.append("Held at the as-of with neither funded capital nor an unfunded commitment in PE "
                     "exposure (counted in deals, properties and gross capitalization, not in capital): "
                     + ", ".join(other["deal_names"]))
    total = {
        "key": "total", "label": "Total",
        "deals": sum(x["deals"] for x in lines),
        "properties": sum(x["properties"] for x in lines),
        "gross_cap": sum(x["gross_cap"] or 0.0 for x in lines),
        "psc": sum(x["psc"] or 0.0 for x in lines),
        "third_party": sum(x["third_party"] or 0.0 for x in lines),
        "total": sum(x["total"] or 0.0 for x in lines),
    }
    for x in lines:
        if x["gross_cap_missing"]:
            notes.append(f"{x['label']}: no Total Size in Investment Metrics for "
                         + ", ".join(x["gross_cap_missing"]) + " (left out of gross capitalization).")

    totals = pe.get("totals") or {}
    by_group = totals.get("cost_by_group") or {}
    third_total = sum(by_group.get(g) or 0.0 for g in THIRD_PARTY)
    sources = [{"group": g, "label": GROUP_LABELS[g], "amount": by_group.get(g),
                "share": (by_group.get(g) / third_total) if (third_total and by_group.get(g) is not None) else None}
               for g in THIRD_PARTY]

    reconciliation = {
        # Each page total against the engine's own -- they must agree.
        "pe_funded_cost": totals.get("cost"),
        "page_total_net_pref": total["total"],
        "pe_psc": by_group.get("PSC"),
        "page_psc": total["psc"],
        "pe_third_party": third_total,
        "page_third_party": total["third_party"],
        "im_current_total_size": ((im.get("current") or {}).get("total") or {}).get("total_size"),
        "page_gross_cap_millions": total["gross_cap"] / 1e6,
    }
    return {
        "as_of": pe.get("as_of"),
        "lines": lines, "total": total,
        "third_party_sources": sources,
        "third_party_total": third_total,
        "unfunded": totals.get("future_funding"),
        "unfunded_by_group": totals.get("future_by_group"),
        "deals": sorted(deals.values(), key=lambda d: d["name"].lower()),
        "reconciliation": reconciliation,
        "notes": notes,
        "fx": pe.get("fx"),
    }


# ------------------------------------------------------------------ page 27

def compose_performance(pe: dict, im: dict) -> dict:
    """Page 27 from the Investment Metrics totals and the PE exposure grand total."""
    import investment_metrics as engine

    cur = (im.get("current") or {})
    sold = (im.get("sold") or {})
    ct, st = cur.get("total") or {}, sold.get("total") or {}
    m = 1e6
    rows = [
        {"key": "current", "label": "Current Portfolio",
         "pref": (pe.get("totals") or {}).get("grand_cost"),
         "pref_basis": "PE exposure: funded cost plus unfunded commitments",
         "proj_irr": ct.get("uw_irr"), "realized_irr": None,
         "proceeds": None if ct.get("proceeds") is None else ct["proceeds"] * m,
         "coc": ct.get("act_coc_since_close")},
        {"key": "exited", "label": "Exited Investments",
         "pref": None if st.get("pref") is None else st["pref"] * m,
         "pref_basis": "Investment Metrics, Sold table: PSC pref equity",
         "proj_irr": None, "realized_irr": st.get("realized_irr"),
         "proceeds": None if st.get("proceeds") is None else st["proceeds"] * m,
         "coc": st.get("act_coc_since_close")},
    ]
    gt = im.get("grand_total") or {}
    all_rows = (cur.get("rows") or []) + (sold.get("rows") or [])
    total = {"key": "total", "label": "Total",
             "proceeds": None if gt.get("proceeds") is None else gt["proceeds"] * m,
             # The engine's own Total / Average rule, over both tables' rows.
             "coc": engine.pref_weighted_average(all_rows, "act_coc_since_close")}
    return {"as_of": im.get("as_of"), "rows": rows, "total": total,
            "im_as_of_display": im.get("as_of_display"),
            "notes": ["Proceeds and CoC are through the as-of date; the current pref includes "
                      "unfunded commitments at that date."]}


# ------------------------------------------------------------------ page 23

def asset_class(asset_type) -> str:
    """The deck's class for an MRI ``Asset_Type``; an unnamed type passes through as itself."""
    raw = "" if asset_type is None or asset_type != asset_type else str(asset_type).strip()
    if not raw:
        return UNCLASSIFIED
    return ASSET_CLASS_OF.get(raw.lower(), raw)


def compose_asset_class(pe: dict, asset_types: Dict[str, Optional[str]]) -> dict:
    """Page 23: net preferred equity INCLUDING unfunded commitments, by asset class.

    Per deal: the PE exposure engine's funded Cost plus its remaining to fund; PSC
    is the engine's PSC column of each. A deal whose unfunded commitment the engine
    could not split (several holders) is in the total and not in PSC, and is named;
    a holding with no USD rate is in neither, as in the engine's own totals.
    """
    notes: List[str] = []
    total: Dict[str, float] = {}
    psc: Dict[str, float] = {}
    deals: Dict[str, set] = {}
    raw_of: Dict[str, set] = {}

    def add(vcode, name, amount, psc_amount):
        cls = asset_class(asset_types.get(vcode))
        total[cls] = total.get(cls, 0.0) + amount
        if psc_amount is not None:
            psc[cls] = psc.get(cls, 0.0) + psc_amount
        deals.setdefault(cls, set()).add(name)
        raw_of.setdefault(cls, set()).add(str(asset_types.get(vcode) or ""))

    for r in pe.get("rows") or []:
        if r.get("fx_missing") or r.get("cost") is None:
            continue
        add(r["vcode"], r["deal_name"], r["cost"], (r.get("cost_by_group") or {}).get("PSC") or 0.0)
    for f in pe.get("future_funding") or []:
        amt = f.get("remaining_to_fund_usd")
        if amt is None:
            continue
        split = f.get("by_group")
        add(f["vcode"], f["deal_name"], amt, (split.get("PSC") or 0.0) if split else None)
        if not split:
            notes.append(f"{f['deal_name']}: its ${amt / 1e6:,.2f}M unfunded commitment has several "
                         f"holders and is not split, so it is in the total and not in PSC.")

    order = list(ASSET_CLASSES) + sorted(c for c in total if c not in ASSET_CLASSES)
    grand, grand_psc = sum(total.values()), sum(psc.values())
    rows = []
    for c in order:
        if c not in total and c not in ASSET_CLASSES:
            continue
        rows.append({"label": c, "total": total.get(c, 0.0) if c in total else None,
                     "total_share": (total[c] / grand) if (c in total and grand) else None,
                     "psc": psc.get(c) if c in total else None,
                     "psc_share": (psc[c] / grand_psc) if (c in psc and grand_psc) else None,
                     "deals": sorted(deals.get(c, ())), "in_deck": c in ASSET_CLASSES})
        if c not in ASSET_CLASSES:
            notes.append(f"Asset_Type {', '.join(sorted(raw_of[c])) or '(blank)'} is not one of the deck's "
                         f"classes, so it is its own row: {', '.join(sorted(deals[c]))}.")
    totals = pe.get("totals") or {}
    return {
        "as_of": pe.get("as_of"), "rows": rows,
        "total": {"total": grand, "psc": grand_psc},
        "reconciliation": {"pe_grand_cost": totals.get("grand_cost"), "page_total": grand,
                           "pe_grand_psc": (totals.get("grand_cost_by_group") or {}).get("PSC"),
                           "page_psc": grand_psc},
        "notes": notes,
    }


# ------------------------------------------------------------------ page 28

#: p. 28's rows: the p. 23 classes, with "Other" shown by its own MRI type (the
#: deck prints RV Park and Industrial separately here).
def metrics_class(asset_type) -> str:
    cls = asset_class(asset_type)
    if cls == "Other":
        return str(asset_type).strip()
    return cls


#: Fixed-rate maturity buckets, years from the as-of to maturity (deck p. 28).
MATURITY_BUCKETS = (("0-3", 3.0), ("4-6", 6.0), ("6+", None))
#: The deck's split of capped floating debt: index capped at or below 2.5%, or above.
CAP_SPLIT = 0.025


def compose_metrics_by_class(deal_rows: List[dict]) -> dict:
    """Occupancy and DSCR by class, and for the portfolio, by the SNAPSHOT'S rules.

    ``deal_rows``: {vcode, name, cls, is_dev, occ, noi, dscr, debt} per deal held.
    Occupancy is NOI-weighted (``portfolio_snapshot_operating._weighted``), DSCR
    debt-weighted (``portfolio_snapshot_loan._debt_weighted``) -- the functions
    themselves, so the two pages cannot weigh differently. Development deals are
    out, as on the Snapshot, and named.
    """
    from flask_app.services.portfolio_snapshot_loan import _debt_weighted
    from flask_app.services.portfolio_snapshot_operating import _weighted

    def shaped(rows):
        return ([{"econ_occ": {"ytd_actual": r["occ"]}, "noi_display": {"ytd_actual": r["noi"]}} for r in rows],
                [{"ytd_dscr": r["dscr"], "debt": r["debt"]} for r in rows])

    live = [r for r in deal_rows if not r["is_dev"]]
    order = [c for c in ("Multifamily", "Grocery-Anchored Retail", "Non-Grocery Retail", "Self Storage")]
    order += sorted({r["cls"] for r in live} - set(order))
    out = []
    for c in order:
        rows = [r for r in live if r["cls"] == c]
        if not rows:
            continue
        o_rows, d_rows = shaped(rows)
        out.append({"label": c, "occupancy": _weighted(o_rows, "ytd_actual", "ytd_actual"),
                    "dscr": _debt_weighted(d_rows, "ytd_dscr"), "deals": sorted(r["name"] for r in rows)})
    o_rows, d_rows = shaped(live)
    portfolio = {"label": "Portfolio", "occupancy": _weighted(o_rows, "ytd_actual", "ytd_actual"),
                 "dscr": _debt_weighted(d_rows, "ytd_dscr")}
    notes = []
    dev = sorted(r["name"] for r in deal_rows if r["is_dev"])
    if dev:
        notes.append("Development deals are left out of occupancy and DSCR, as on the Portfolio Snapshot: "
                     + ", ".join(dev))
    no_occ = sorted(r["name"] for r in live if r["occ"] is None or not r["noi"])
    if no_occ:
        notes.append("No YTD occupancy reading or NOI to weight it, so not in occupancy: " + ", ".join(no_occ))
    no_dscr = sorted(r["name"] for r in live if r["dscr"] is None or not r["debt"])
    if no_dscr:
        notes.append("No YTD DSCR or no debt to weight it, so not in DSCR: " + ", ".join(no_dscr))
    return {"rows": out, "portfolio": portfolio, "notes": notes}


def _add_years(d: date, n: int) -> date:
    """The same calendar day n years on; 29 Feb falls back to 28 Feb."""
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def compose_debt(loans: List[dict], as_of: date) -> dict:
    """Fixed debt by years to maturity and floating debt by cap, from per-loan rows.

    ``loans``: the Dashboard maturity engine's detail rows (amount = original
    facility, rate_type, rate, maturity), each with its deal, and -- for floating
    loans -- the loan's index, spread and ``loan_caps.cap_terms``. Amounts are the
    facility (``mOrigLoanAmt``), unfunded development debt included, and averages
    are weighted by it, as the Dashboard's maturity chart weights them.
    """
    notes: List[str] = []
    fixed = [l for l in loans if l["rate_type"] == "Fixed"]
    floating = [l for l in loans if l["rate_type"] != "Fixed"]
    total_debt = sum(l["amount"] for l in loans)

    def wavg(rows):
        w = sum(l["amount"] for l in rows if l.get("rate") is not None)
        return (sum(l["rate"] * l["amount"] for l in rows if l.get("rate") is not None) / w) if w else None

    # Years counted on the CALENDAR: a loan maturing exactly three years after the
    # as-of is in 0-3. (Days / 365.25 put 12/31/28 from 12/31/25 at 3.0007 -- a
    # leap day pushed it into 4-6.)
    buckets = []
    for label, upper in MATURITY_BUCKETS:
        lower = buckets and MATURITY_BUCKETS[len(buckets) - 1][1]
        rows = []
        for l in fixed:
            m = date.fromisoformat(l["maturity"])
            within_upper = upper is None or m <= _add_years(as_of, int(upper))
            above_lower = not lower or m > _add_years(as_of, int(lower))
            if within_upper and above_lower:
                rows.append(l)
        buckets.append({"label": label, "amount": sum(l["amount"] for l in rows), "avg_rate": wavg(rows),
                        "loans": len(rows)})
    matured = [l for l in fixed if date.fromisoformat(l["maturity"]) < as_of]
    if matured:
        notes.append("Fixed loans past maturity at the as-of, counted in 0-3: "
                     + ", ".join(f"{l['deal_name']} ({l['maturity']})" for l in matured))
    no_rate = [l for l in fixed if l.get("rate") is None]
    if no_rate:
        notes.append("Fixed loans with no rate in MRI, left out of the average rate: "
                     + ", ".join(l["deal_name"] for l in no_rate))

    groups = {"none": [], "low": [], "high": [], "unknown": []}
    for l in floating:
        cap = l.get("cap") or {}
        if cap.get("capped") is None:
            groups["unknown"].append(l)
        elif not cap.get("capped"):
            groups["none"].append(l)
        else:
            groups["low" if cap["strike"] <= CAP_SPLIT + 1e-9 else "high"].append(l)
    labels = {"none": "No Cap (incl. unfunded dev. debt)", "low": "Index Capped at <=2.5% + Spread",
              "high": "Index Capped at >2.5% + Spread", "unknown": "Cap terms not readable"}
    float_rows = []
    for k in ("none", "low", "high", "unknown"):
        rows = groups[k]
        if not rows and k == "unknown":
            continue
        float_rows.append({"key": k, "label": labels[k], "amount": sum(l["amount"] for l in rows),
                           "share": (sum(l["amount"] for l in rows) / total_debt) if total_debt else None,
                           "index": ", ".join(sorted({str(l.get("index") or "") for l in rows} - {""})),
                           "deals": [l["deal_name"] for l in rows]})
    float_total = sum(l["amount"] for l in floating)
    exposure = []
    for l in sorted(floating, key=lambda x: x["deal_name"].lower()):
        cap = l.get("cap") or {}
        mat = date.fromisoformat(l["maturity"])
        through = cap.get("expiry") or f"{mat.month}/{mat.strftime('%y')}"
        idx, spr = (l.get("index") or "Index"), l.get("spread")
        if cap.get("capped") and cap.get("max_rate") is not None:
            line = f"{cap['strike']:.2%} + {spr:.2%} = {cap['max_rate']:.2%}, through {through}"
        elif cap.get("capped") is False:
            line = (f"{idx} + {spr:.2%} uncapped, through {through}" if spr is not None
                    else f"{idx}, spread not in MRI, uncapped, through {through}")
        else:
            line = f"cap terms: \"{cap.get('text') or 'hedge recorded, no terms'}\""
        exposure.append({"deal": l["deal_name"], "index": l.get("index"), "spread": spr,
                         "capped": cap.get("capped"), "strike": cap.get("strike"), "max_rate": cap.get("max_rate"),
                         "expiry": cap.get("expiry"), "maturity": l["maturity"], "text": cap.get("text"),
                         "line": line})
        if cap.get("problem"):
            notes.append(f"{l['deal_name']}: {cap['problem']}" + (f" -- \"{cap['text']}\"" if cap.get("text") else ""))
    capped_or_fixed = sum(l["amount"] for l in fixed) + sum(l["amount"] for l in groups["low"] + groups["high"])
    return {
        "fixed": {"total": sum(l["amount"] for l in fixed), "avg_rate": wavg(fixed), "buckets": buckets},
        "floating": {"rows": float_rows, "total": float_total,
                     "share": (float_total / total_debt) if total_debt else None},
        "exposure": exposure,
        "total_debt": total_debt,
        "fixed_or_capped_share": (capped_or_fixed / total_debt) if total_debt else None,
        "notes": notes,
    }


# ------------------------------------------------------------------ page 5

def _year_label(as_of: date) -> str:
    return str(as_of.year) if (as_of.month, as_of.day) == (12, 31) else f"Year to {_short_date(as_of.isoformat())}"


def _m(v: float) -> str:
    return f"${v / 1e6:,.1f}M"


def compose_year_in_review(pe_now: dict, pe_prior: dict, im: dict, as_of: date, prior: date,
                           losses_now: Optional[Dict[tuple, float]] = None,
                           losses_prior: Optional[Dict[tuple, float]] = None,
                           names: Optional[Dict[str, str]] = None) -> dict:
    """Page 5's figures: 3rd-party AUM and its growth, and the year's investment activity.

    AUM is the PE exposure engine's FUNDED cost by investor group (the tracker's
    "Current AUM"), at the as-of and at the same date a year earlier; unfunded
    commitments are stated beside it, never added in. The year's activity is
    Investment Metrics' own rows -- every deal it carries, Current and Sold --
    whose PSC Invest. Date (new deals, new partners) or sale date (exits) falls in
    the twelve months to the as-of. The sentences are written here, once; the
    numbers in them are the engines'.

    Exits are listed by name, every one. A loss is NOT inferred from the row (its
    "proceeds to date" exclude returned capital, so proceeds below pref says
    nothing about a loss); the realized losses BOOKED in the year are their own
    line, from ``pe_exposure_service.noncash_by_holding`` -- accounting's write-offs
    -- at the as-of less a year earlier, by investment.
    """
    notes: List[str] = []
    now_g = (pe_now.get("totals") or {}).get("cost_by_group") or {}
    prev_g = (pe_prior.get("totals") or {}).get("cost_by_group") or {}
    unf = (pe_now.get("totals") or {}).get("future_by_group") or {}
    third_now = sum(now_g.get(g) or 0.0 for g in THIRD_PARTY)
    third_prev = sum(prev_g.get(g) or 0.0 for g in THIRD_PARTY)

    def growth(now, prev):
        if not prev:
            return ""
        d = now - prev
        return f", a YoY {'increase' if d >= 0 else 'decrease'} of {_m(abs(d))} ({d / prev:+.0%})"

    aum = [f"Total 3rd party AUM is {_m(third_now)}{growth(third_now, third_prev)}"]
    for g in sorted(THIRD_PARTY, key=lambda g: -(now_g.get(g) or 0.0)):
        v = now_g.get(g) or 0.0
        if v < 0.5:
            continue
        u = unf.get(g) or 0.0
        aum.append(f"{GROUP_LABELS[g]} AUM is {_m(v)}" + (f" (excluding {_m(u)} unfunded)" if u >= 50_000 else "")
                   + growth(v, prev_g.get(g) or 0.0))

    lo, hi = prior, as_of
    rows = [(t, r) for t in ("current", "sold") for r in (im.get(t) or {}).get("rows") or []]

    def d(x):
        try:
            return date.fromisoformat(str(x)[:10]) if x else None
        except ValueError:
            return None

    new = [r for _, r in rows if d(r.get("invest_date")) and lo < d(r.get("invest_date")) <= hi]
    new_pref = sum((r.get("pref") or 0.0) for r in new) * 1e6
    activity = []
    if new:
        activity.append(f"{_m(new_pref)} of Preferred Equity invested in {len(new)} new deal{'s' if len(new) != 1 else ''}")
    # A NEW partner is one whose first deal Investment Metrics carries closed in the year.
    first_by_partner: Dict[str, date] = {}
    for _, r in rows:
        nm, when = partner_name(r.get("partner")), d(r.get("invest_date"))
        if when and nm != NO_PARTNER and (nm not in first_by_partner or when < first_by_partner[nm]):
            first_by_partner[nm] = when
    new_partners = sorted(n for n, w in first_by_partner.items() if lo < w <= hi)
    if new_partners:
        funded = sum((r.get("pref") or 0.0) for r in new if partner_name(r.get("partner")) in new_partners) * 1e6
        who = " & ".join(new_partners) if len(new_partners) <= 2 else ", ".join(new_partners[:-1]) + " & " + new_partners[-1]
        activity.append(f"Added {len(new_partners)} new operating partner{'s' if len(new_partners) != 1 else ''}, "
                        f"{who}, funding {_m(funded)} preferred equity")
    exits = [r for t, r in rows if t == "sold" and d(r.get("sale_date")) and lo < d(r.get("sale_date")) <= hi]
    if exits:
        activity.append(f"Exited {len(exits)} investment{'s' if len(exits) != 1 else ''} ("
                        + ", ".join(r["name"] for r in sorted(exits, key=lambda r: r.get("sale_date") or "")) + ")")
    # Realized losses booked in the year: accounting's write-offs, by investment.
    year_loss: Dict[str, float] = {}
    for (inv_id, _holder), v in (losses_now or {}).items():
        year_loss[inv_id] = year_loss.get(inv_id, 0.0) + v
    for (inv_id, _holder), v in (losses_prior or {}).items():
        year_loss[inv_id] = year_loss.get(inv_id, 0.0) - v
    # The DEAL's own investment only: accounting books the same write-off again at
    # each fund up the chain (City West at PPICW, INVCW, TGACW...), so counting every
    # investment id would count one loss three or four times.
    deal_ids = set(names or {})
    booked = sorted(((k, -v) for k, v in year_loss.items() if v <= -0.5 and k in deal_ids),
                    key=lambda kv: -kv[1])
    if booked:
        activity.append("Realized losses booked in the year: "
                        + ", ".join(f"{(names or {}).get(k, k)} {_m(v)}" for k, v in booked))
    if not activity:
        notes.append("No deal closed or sold in the year in Investment Metrics.")
    missing = [r.get("name") for _, r in rows if not r.get("invest_date")]
    if missing:
        notes.append("No PSC invest date in Investment Metrics, so in no year: " + ", ".join(missing))
    return {
        "year": _year_label(as_of),
        "sections": [{"heading": "3rd Party AUM Growth", "bullets": aum},
                     {"heading": "Investment Activity", "bullets": activity}],
        "figures": {"third_party_now": third_now, "third_party_prior": third_prev,
                    "by_group_now": {g: now_g.get(g) for g in THIRD_PARTY},
                    "by_group_prior": {g: prev_g.get(g) for g in THIRD_PARTY},
                    "unfunded_by_group": {g: unf.get(g) for g in THIRD_PARTY},
                    "new_deals": [r.get("name") for r in new], "new_pref": new_pref,
                    "new_partners": new_partners, "exits": [r.get("name") for r in exits],
                    "losses_booked": {k: v for k, v in booked}},
        "prior_as_of": prior.isoformat(),
        "notes": notes,
    }


# ------------------------------------------------------------------ page 9

def compose_pref_by_year(im: dict) -> dict:
    """Page 9: preferred equity invested, new each year and cumulative.

    Every deal Investment Metrics carries -- Current and Sold -- at its PSC pref
    (funded or committed, return of capital not deducted: the figure its tables
    print) in the year of its PSC Invest. Date. The total IS the engine's Grand
    Total pref. A deal with no pref or no invest date is left out and named.
    """
    by_year: Dict[int, float] = {}
    left_out: List[str] = []
    for t in ("current", "sold"):
        for r in (im.get(t) or {}).get("rows") or []:
            if r.get("pref") is None or not r.get("invest_date"):
                left_out.append(r.get("name") or r.get("vcode"))
                continue
            y = int(str(r["invest_date"])[:4])
            by_year[y] = by_year.get(y, 0.0) + r["pref"] * 1e6
    years, cum = [], 0.0
    for y in sorted(by_year):
        years.append({"year": y, "new": by_year[y], "prior": cum, "cumulative": cum + by_year[y]})
        cum += by_year[y]
    gt = (im.get("grand_total") or {}).get("pref")
    notes = []
    if left_out:
        notes.append("No pref or no invest date in Investment Metrics, so not in any year: " + ", ".join(left_out))
    return {"years": years, "total": cum,
            "reconciliation": {"im_grand_total_pref": None if gt is None else gt * 1e6, "page_total": cum},
            "notes": notes}


# ------------------------------------------------------------------ page 24

def compose_partner_exposure(pe: dict, partners: Dict[str, Optional[str]],
                             property_counts: Dict[str, Optional[int]]) -> dict:
    """Page 24: p. 23's figures -- funded Cost + remaining to fund, total and PSC --
    by operating partner, with the deals and properties behind each.

    A deal is an MRI deal (Jim, Oct 7 2026: Brainerd I and II are one), so the deck's
    transaction count (Apple 7, Berger 8) is not this column. An unsplit unfunded
    commitment is in the total and not PSC, as on p. 23.
    """
    groups: Dict[str, dict] = {}
    notes: List[str] = []

    def g(vcode):
        raw = partners.get(vcode)
        name = partner_name(raw)
        x = groups.setdefault(name, {"total": 0.0, "psc": 0.0, "deals": set(), "vcodes": set(), "raw": set()})
        x["raw"].add("" if raw is None or raw != raw else str(raw).strip())
        x["vcodes"].add(vcode)
        return x

    for r in pe.get("rows") or []:
        if r.get("fx_missing") or r.get("cost") is None:
            continue
        x = g(r["vcode"])
        x["total"] += r["cost"]
        x["psc"] += (r.get("cost_by_group") or {}).get("PSC") or 0.0
        x["deals"].add(r["deal_name"])
    for f in pe.get("future_funding") or []:
        amt = f.get("remaining_to_fund_usd")
        if amt is None:
            continue
        x = g(f["vcode"])
        x["total"] += amt
        split = f.get("by_group")
        if split:
            x["psc"] += split.get("PSC") or 0.0
        else:
            notes.append(f"{f['deal_name']}: its ${amt / 1e6:,.2f}M unfunded commitment has several "
                         f"holders and is not split, so it is in the total and not in PSC.")
        x["deals"].add(f["deal_name"])

    grand = sum(x["total"] for x in groups.values())
    grand_psc = sum(x["psc"] for x in groups.values())
    rows = []
    for name in sorted(groups, key=lambda n: n.lower()):
        x = groups[name]
        pcs = [property_counts.get(v) for v in x["vcodes"]]
        rows.append({"label": name, "deals": len(x["deals"]),
                     "properties": None if all(p is None for p in pcs) else sum(p or 0 for p in pcs),
                     "total": x["total"], "total_share": x["total"] / grand if grand else None,
                     "psc": x["psc"], "psc_share": x["psc"] / grand_psc if grand_psc else None,
                     "deal_names": sorted(x["deals"]), "mri_names": sorted(x["raw"])})
        spellings = sorted(n for n in x["raw"] if n)
        if len(spellings) > 1:
            notes.append(f"{name}: MRI spells this partner {len(spellings)} ways ({'; '.join(spellings)}); "
                         f"shown as one.")
        if name not in PARTNER_NAMES.values():
            notes.append(f"Operating partner \"{name}\" is shown as MRI has it (no short name on the deck "
                         f"list): {', '.join(sorted(x['deals']))}.")
    totals = pe.get("totals") or {}
    return {"rows": rows,
            "total": {"total": grand, "psc": grand_psc, "deals": sum(r["deals"] for r in rows),
                      "properties": sum(r["properties"] or 0 for r in rows)},
            "reconciliation": {"pe_grand_cost": totals.get("grand_cost"), "page_total": grand,
                               "pe_grand_psc": (totals.get("grand_cost_by_group") or {}).get("PSC"),
                               "page_psc": grand_psc},
            "notes": notes}


# ------------------------------------------------------------------ the views

def _partners(inv) -> Dict[str, Optional[str]]:
    if inv is None or "vcode" not in inv.columns or "Operating_Partner" not in inv.columns:
        return {}
    return {str(v).strip().upper(): (None if t is None or t != t else str(t))
            for v, t in zip(inv["vcode"], inv["Operating_Partner"])}


def _asset_types(inv) -> Dict[str, Optional[str]]:
    if inv is None or "vcode" not in inv.columns or "Asset_Type" not in inv.columns:
        return {}
    return {str(v).strip().upper(): (None if t is None or t != t else str(t))
            for v, t in zip(inv["vcode"], inv["Asset_Type"])}


def _quarter_of(d: date) -> str:
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def _portfolio_metrics(as_of: date, data: dict, im: dict) -> dict:
    """Page 28 at ``as_of``: the deals Investment Metrics holds, through the engines named above."""
    import pandas as pd
    from config import is_dev_deal
    from consolidation import build_property_map
    from one_pager import _child_vcodes_for_parent
    from flask_app.services import dashboard_service, loan_caps
    from flask_app.services.portfolio_snapshot_debt import committed_facility, deal_loan_rows, resolve_debt
    from flask_app.services.portfolio_snapshot_freeze import _one_pager_provider

    inv = data["inv"]
    by_v = {str(v).strip().upper(): r for v, (_, r) in zip(inv["vcode"], inv.iterrows())}
    quarter = _quarter_of(as_of)
    provider = _one_pager_provider(data)
    loans_raw = data.get("mri_loans_raw")
    held = (im.get("current") or {}).get("rows") or []

    deal_rows, deal_of = [], {}
    prop_map = build_property_map(inv)
    for r in held:
        v = r["vcode"]
        for t in (r.get("twin_vcodes") or [v]):
            deal_of[t] = (v, r["name"])
            for child in prop_map.get(t, []):
                deal_of[str(child).strip().upper()] = (v, r["name"])
        row = by_v.get(v)
        strategy = ""
        if row is not None:
            strategy = str(row.get("Investment_Strategy") or "").strip() or str(row.get("Lifecycle") or "").strip()
        dev = is_dev_deal(strategy)
        payload = provider(v, quarter)
        perf = payload.get("property_performance") or {}
        committed = committed_facility(deal_loan_rows(loans_raw, v, _child_vcodes_for_parent(v, inv)))
        debt, _ = resolve_debt(payload.get("cap_stack"), dev, committed)
        deal_rows.append({
            "vcode": v, "name": r["name"], "is_dev": dev,
            "cls": metrics_class(row.get("Asset_Type") if row is not None else None),
            "occ": ((perf.get("economic_occ") or {}).get("ytd_actual")),
            "noi": ((perf.get("noi") or {}).get("ytd_actual")),
            "dscr": ((perf.get("dscr") or {}).get("ytd_actual")),
            "debt": debt,
        })
    metrics = compose_metrics_by_class(deal_rows)
    # One Pager occupancy is in percentage points (92.2); the page speaks decimals.
    for x in metrics["rows"] + [metrics["portfolio"]]:
        if x["occupancy"] is not None:
            x["occupancy"] = x["occupancy"] / 100.0

    population = pd.DataFrame({"vcode": [r["vcode"] for r in held]})
    maturity = dashboard_service.get_loan_maturity_data(loans_raw, population, inv)
    terms = {}
    if loans_raw is not None and "LoanID" in loans_raw.columns:
        for _, lr in loans_raw.iterrows():
            terms[str(lr.get("LoanID"))] = lr
    loans = []
    for d in maturity["detail"]:
        lr = terms.get(d["loan_id"])
        vc = str(lr.get("vCode")).strip().upper() if lr is not None else ""
        deal_v, deal_name = deal_of.get(vc, (vc, d["property"]))
        rate = d.get("rate")
        loan = {**d, "deal": deal_v, "deal_name": deal_name,
                "rate": (rate / 100.0 if rate and rate >= 1 else rate) or None}
        if d["rate_type"] != "Fixed" and lr is not None:
            mat = date.fromisoformat(d["maturity"]) if d.get("maturity") else None
            loan["index"] = None if pd.isna(lr.get("vIndex")) else str(lr.get("vIndex"))
            loan["spread"] = loan_caps._rate(lr.get("vSpread"))
            loan["cap"] = loan_caps.cap_terms(lr.get("vIntRatereset"), lr.get("vHedged"), lr.get("vHedgedStrat"),
                                              lr.get("vSpread"), f"{mat.month}/{mat.strftime('%y')}" if mat else None)
        loans.append(loan)
    debt = compose_debt(loans, as_of)
    notes = metrics["notes"] + debt["notes"] + [
        "Loans are as MRI holds them today: a loan repaid since the as-of is not shown, and one originated "
        "after it is.",
        "Loans with no rate type in MRI are counted as fixed, as the Dashboard's maturity chart counts them.",
    ]
    if quarter and as_of != _quarter_end(as_of):
        notes.append(f"Occupancy and DSCR are year-to-date at the end of {quarter}.")
    return {"metrics": metrics, "debt": debt, "notes": notes, "quarter": quarter}


def _quarter_end(d: date) -> date:
    m = ((d.month - 1) // 3 + 1) * 3
    nxt = date(d.year + (m == 12), 1 if m == 12 else m + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def _property_counts(inv) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    if inv is None or "vcode" not in inv.columns or "Property_Count" not in inv.columns:
        return out
    for v, pc in zip(inv["vcode"], inv["Property_Count"]):
        try:
            out[str(v).strip().upper()] = None if pc is None or pc != pc else int(pc)
        except (TypeError, ValueError):
            out[str(v).strip().upper()] = None
    return out


def build_view(key: str, as_of: date, data: Optional[dict] = None, engine=None) -> dict:
    """The view of one schedule at ``as_of``. ``KeyError`` for a schedule with no view yet."""
    from flask_app.services import investment_metrics_service as ims
    from flask_app.services import pe_exposure_service as pe_svc

    if key not in VIEW_KEYS:
        raise KeyError(key)
    if data is None:
        from flask_app.services.data_service import get_data
        data = get_data()
    when = _short_date(as_of.isoformat())
    if key == "exposure_partner":
        pe = pe_svc.get_report(as_of, data=data, engine=engine)
        return {"key": key, "slide_title": SLIDE_TITLES[key],
                **compose_partner_exposure(pe, _partners(data.get("inv")), _property_counts(data.get("inv"))),
                "footnotes": [f"*Preferred equity balance includes unfunded commitments. Portfolio data is "
                              f"updated through {when}."]}
    if key == "exposure_asset_class":
        # PE exposure only -- no Investment Metrics build for this page.
        pe = pe_svc.get_report(as_of, data=data, engine=engine)
        return {"key": key, "slide_title": SLIDE_TITLES[key],
                **compose_asset_class(pe, _asset_types(data.get("inv"))),
                "footnotes": [f"*Preferred equity balance includes unfunded commitments. Portfolio "
                              f"data is updated through {when}."]}
    im = ims.get_report(as_of, data=data)
    if key == "year_in_review":
        prior = _add_years(as_of, -1)
        from flask_app.db import get_engine
        db = engine or get_engine()
        names = {}
        inv = data.get("inv")
        if inv is not None and "InvestmentID" in inv.columns:
            for i, nm in zip(inv["InvestmentID"], inv["Investment_Name"]):
                if i is not None and i == i:
                    names.setdefault(pe_svc._norm(i), str(nm))
        out = compose_year_in_review(pe_svc.get_report(as_of, data=data, engine=engine),
                                     pe_svc.get_report(prior, data=data, engine=engine), im, as_of, prior,
                                     pe_svc.noncash_by_holding(db, as_of)["realized_loss"],
                                     pe_svc.noncash_by_holding(db, prior)["realized_loss"], names)
        return {"key": key, **out, "slide_title": f"{out['year']} In Review",
                "footnotes": [f"3rd party AUM is funded net preferred equity at {when}, against "
                              f"{_short_date(prior.isoformat())}; activity is the twelve months to {when}."]}
    if key == "pref_by_year":
        out = compose_pref_by_year(im)
        first = out["years"][0]["year"] if out["years"] else None
        return {"key": key, **out,
                "slide_title": (f"${out['total'] / 1e6:,.0f}mm Total Preferred Equity Invested"
                                + (f" Since {first}" if first else "")),
                "footnotes": [f"* Funded or Committed Pref Equity as of {when}. Each deal's pref is in the "
                              f"year it closed."]}
    if key == "debt":
        out = _portfolio_metrics(as_of, data, im)
        d = out["debt"]
        fl, cap = d["floating"]["share"], d["fixed_or_capped_share"]
        return {"key": key, "slide_title": SLIDE_TITLES[key], **out,
                "banner": (None if fl is None or cap is None else
                           f"Floating rate debt is {fl:.0%} of total debt; {cap:.0%} of all debt is fixed or capped"),
                "footnotes": [f"Occupancy and DSCR: year-to-date through {when}, development deals excluded. "
                              f"Debt at facility amount, unfunded development debt included; maturity "
                              f"buckets are years from {when} to maturity."]}
    if key == "investment_summaries":
        layout = deck_layout(im)
        notes = [f"The deck layout names a column Investment Metrics does not carry: {m}"
                 for m in layout["missing_columns"]]
        return {"key": key, "as_of": as_of.isoformat(), "investment_metrics": im,
                "deck": layout, "notes": notes}
    pe = pe_svc.get_report(as_of, data=data, engine=engine)
    if key == "performance":
        return {"key": key, "slide_title": SLIDE_TITLES[key], **compose_performance(pe, im),
                "footnotes": [f"*Preferred equity balance includes unfunded commitments through "
                              f"{when}. Proceeds and CoC return data is through {when}."]}
    out = {"key": key, "slide_title": SLIDE_TITLES[key],
           **compose_capitalization(pe, im, _property_counts(data.get("inv")))}
    unfunded = out.get("unfunded")
    out["footnotes"] = [
        f"*Portfolio data is updated through {when}",
        "**Current AUM is third party net preferred equity"
        + (f", excludes the ${unfunded / 1e6:,.1f}M unfunded" if unfunded else ""),
    ]
    return out
