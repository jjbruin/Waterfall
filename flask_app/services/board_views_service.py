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

VIEW_KEYS = ("capitalization", "performance", "investment_summaries")

# ------------------------------------------------------------------ the deck's layout
#
# HOW A SCHEDULE IS LAID OUT ON THE SLIDE is decided here, not in the screen, so
# there is one copy of it. None of it is a figure: titles, footnote wording, which
# of an engine's columns the deck prints, and how many rows fit one page.

#: The deck's slide titles (Jan 2026 deck, pp. 26-27).
SLIDE_TITLES = {"capitalization": "Current Portfolio Capitalization",
                "performance": "Performance: Portfolio Level"}

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
    return {"columns": columns, "slides": slides, "missing_columns": missing}


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


# ------------------------------------------------------------------ the views

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
    im = ims.get_report(as_of, data=data)
    when = _short_date(as_of.isoformat())
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
