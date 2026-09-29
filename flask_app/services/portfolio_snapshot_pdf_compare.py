"""Comparing a published PDF against what the app computes. COMPARISON ONLY.

THIS CANNOT FREEZE ANYTHING, AND THAT IS THE POINT. These helpers used to live
in ``portfolio_snapshot_freeze`` and fed a "Freeze as sent (from PDFs)" action
that wrote an overlay into the frozen store. That action is gone: the freeze is
for 26Q3 onward and always freezes from LIVE data, so there is no longer any
path by which a figure read off a PDF becomes the stored record of what was
sent.

What remains is the half that was always worth having -- reading a sent PDF and
asking where the app disagrees with it. That is a REPORT, not a write. Keeping
it here, in a module with no freeze imports, is what stops it growing a way
back into the store.

Used by the PDF-vs-app comparison checks (`scripts/build_26q2_overlay.py` and
the snapshot PDF variance checks). Nothing in `flask_app/api` imports it.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Callable, Optional

log = logging.getLogger(__name__)


def _set_path(obj, dotted: str, value) -> bool:
    """Set a dotted path, supporting ``a.b[2].c``. True when it landed."""
    cur = obj
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", dotted)
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if part.startswith("["):
            idx = int(part[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                return False
            if last:
                cur[idx] = value; return True
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return False
            if last:
                cur[part] = value; return True
            cur = cur[part]
    return False


def _apply_overlay(payload: dict, one_pagers: dict, overlay: dict) -> int:
    """Overwrite published cells, keeping the computed value beside each one.

    Records into ``payload['published_overrides']`` so drift stays measurable:
    a later reader can see what was published AND what the engine said at the
    moment of freezing.
    """
    recorded, unapplied, applied = [], [], 0
    for scope, cells in (overlay or {}).items():
        target = payload if scope == "__subtabs__" else one_pagers.get(scope)
        if target is None:
            continue
        for path, spec in (cells or {}).items():
            if not isinstance(spec, dict):
                spec = {"published": spec}
            before = _read_path(target, path)
            display = spec.get("display")

            if display is not None:
                # A PRINTED-UNITS CELL. The PDF prints something the stored
                # field cannot hold in the same units — the variance columns
                # print a PERCENT OF BUDGET while the field holds a DOLLAR
                # difference. Writing the percent into the dollar field would
                # be a lie about the field, and writing the dollars would not
                # reproduce the page.
                #
                # It also would not even show: the variance the One Pager
                # RENDERS is derived in the browser by `fmtVariance` from
                # ytd_actual and ytd_budget, so the server's stored `variance`
                # reaches no screen. Overlaying the path alone changes nothing.
                #
                # So the printed text is kept verbatim in `published_display`,
                # keyed by the same dotted path, and the view prefers it when
                # the report is frozen. The numeric field is left ALONE.
                target.setdefault("published_display", {})[path] = display
                applied += 1
                recorded.append({
                    "scope": scope, "path": path,
                    "published": spec.get("published"),
                    "display": display,
                    "units": spec.get("units") or "printed",
                    "computed_at_freeze": before,
                    "page": spec.get("page"), "source": spec.get("source"),
                })
                continue

            if _set_path(target, path, spec.get("published")):
                applied += 1
                recorded.append({
                    "scope": scope, "path": path,
                    "published": spec.get("published"),
                    "computed_at_freeze": before,
                    "page": spec.get("page"), "source": spec.get("source"),
                })
            else:
                # NOT SILENTLY DROPPED. `_set_path` refuses a path the live
                # payload does not already carry — which is right, since
                # inventing the key would put a published figure somewhere no
                # reader looks. But a printed cell that never landed means the
                # stored copy does NOT reproduce the page, and that has to be
                # visible rather than showing up only as a lower applied count.
                unapplied.append({
                    "scope": scope, "path": path,
                    "published": spec.get("published"),
                    "display": spec.get("display"),
                    "page": spec.get("page"),
                    "why": "no such field in the assembled report",
                })
    payload["published_overrides"] = recorded
    if unapplied:
        payload["published_unapplied"] = unapplied
    return applied


def _read_path(obj, dotted: str):
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", dotted):
        try:
            cur = cur[int(part[1:-1])] if part.startswith("[") else cur[part]
        except Exception:
            return None
    return cur


def _norm_title(s) -> str:
    """A deal name reduced to what survives PDF typesetting."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def resolve_roster(investor_code: str, quarter: str, titles: list) -> tuple:
    """(``{printed title: vcode}``, ``[titles that matched nothing]``).

    The overlay is keyed by the deal TITLE because that is all the sent PDF
    knows; only the app knows vcodes. Resolving here — against the same
    assembled report the freeze itself will store — keeps the preview and the
    freeze on one answer.

    A TITLE MATCHING SEVERAL DEALS IS LEFT UNRESOLVED, not resolved to the
    first. Picking one would attach a whole page of published figures to the
    wrong deal, which is the single worst thing this feature could do, and it
    would be invisible afterwards.
    """
    from flask_app.services import data_service
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    from flask_app.services.portfolio_snapshot_freeze import build_subtab

    data = data_service.get_data()
    resolved_deals = resolve_investor_deals(investor_code, quarter, data=data)
    fin = build_subtab("financial", investor_code, quarter, data, resolved_deals)

    by_name = {}
    for blk in (fin.get("groups") or {}).values():
        rows = (blk.get("deals") if isinstance(blk, dict) else blk) or []
        for r in rows:
            if r.get("name") and r.get("vcode"):
                by_name.setdefault(_norm_title(r["name"]), set()).add(r["vcode"])
    for r in (fin.get("ownership_flagged") or []):
        if r.get("name") and r.get("vcode"):
            by_name.setdefault(_norm_title(r["name"]), set()).add(r["vcode"])

    out, missing = {}, []
    for title in titles or []:
        key = _norm_title(title)
        hit = by_name.get(key)
        if not hit:
            # A printed title is often a prefix of the stored name, or the
            # other way round. Only an UNAMBIGUOUS partial is accepted.
            cands = {v for k, vs in by_name.items()
                     if key and (k.startswith(key) or key.startswith(k))
                     for v in vs}
            hit = cands
        if hit and len(hit) == 1:
            out[title] = next(iter(hit))
        else:
            missing.append(title)
    return out, missing


def _norm_row(s) -> str:
    """A Snapshot row label reduced for matching.

    Parenthetical suffixes are dropped — the page prints "Camarillo Village
    (Sold)", "Portfolio Totals (38)" and "Excluding development deals (28)",
    and the count in particular moves between quarters.
    """
    s = re.sub(r"\([^)]*\)", " ", str(s or ""))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _index_subtab(sub: str, blk: dict) -> dict:
    """{normalised row label: dotted path prefix} for one assembled subtab.

    Covers every row the sent page prints: deals, the ownership-flagged rows,
    each group's subtotal, Portfolio Totals and the Excluding-development row.
    A label matching two rows is dropped from the index rather than resolved to
    one — publishing a row of figures against the wrong property is the worst
    thing this can do and is invisible afterwards.
    """
    seen: dict = {}

    def add(label, path):
        k = _norm_row(label)
        if not k:
            return
        seen.setdefault(k, []).append(path)

    for gname, g in (blk.get("groups") or {}).items():
        rows = (g.get("deals") if isinstance(g, dict) else g) or []
        for i, r in enumerate(rows):
            if r.get("name"):
                add(r["name"], f"subtabs.{sub}.groups.{gname}.deals[{i}]")
        if isinstance(g, dict) and isinstance(g.get("subtotal"), dict):
            st = g["subtotal"]
            add(st.get("name") or st.get("label") or f"Total {gname}",
                f"subtabs.{sub}.groups.{gname}.subtotal")
    for i, r in enumerate(blk.get("ownership_flagged") or []):
        if r.get("name"):
            add(r["name"], f"subtabs.{sub}.ownership_flagged[{i}]")
    if isinstance(blk.get("total"), dict):
        add("Portfolio Totals", f"subtabs.{sub}.total")
    if isinstance(blk.get("total_excluding_dev"), dict):
        add("Excluding development deals",
            f"subtabs.{sub}.total_excluding_dev")
    return {k: v[0] for k, v in seen.items() if len(v) == 1}


#: Deal-row fields whose RENDERED cell is a `_display` twin, per subtab.
#:
#: THE RAW FIELD DOES NOT REACH THE SCREEN FOR THESE. SnapshotLoan.vue renders
#: `r.ltv_display`, SnapshotOperating.vue renders `r.noi_display.at_close`, and
#: SnapshotFinancial.vue renders `r.debt_display` — so writing only the raw
#: value leaves the frozen page showing the LIVE figure, which is the same
#: defect the One Pager variance had. Subtotal and total rows are the other way
#: round: they render the raw field, so they are NOT translated.
_DISPLAY_TWIN = {
    "financial": {"debt": "debt_display"},
    "operating": {
        "econ_occ": "econ_occ_display",
        "noi.at_close": "noi_display.at_close",
        "noi.uw_ye": "noi_display.uw_ye",
        "noi.projected_ye": "noi_display.projected_ye",
        "expected_growth": "expected_growth_display",
        "actual_growth": "actual_growth_display",
    },
    "loan": {
        "rate": "rate_display", "maturity": "maturity_display",
        "debt": "debt_display", "ytd_dscr": "ytd_dscr_display",
        "ltv": "ltv_display", "debt_yield": "debt_yield_display",
    },
}


def _snapshot_targets(sub: str, field: str, spec: dict, is_deal: bool) -> list:
    """The path suffix(es) one printed cell should be written to.

    A NUMBER GOES TO BOTH the raw field and its display twin: the twin is what
    renders, and the raw is what the subtotals are built from, so writing only
    one leaves the page and its totals disagreeing.

    A SENTINEL GOES TO THE TWIN ALONE. "—" and "Dev" are display strings; the
    twin legitimately holds them, and putting one in the raw numeric field
    would corrupt every sum that reads it.
    """
    twin = _DISPLAY_TWIN.get(sub, {}).get(field) if is_deal else None
    is_sentinel = spec.get("units") == "printed-sentinel"
    if twin and is_sentinel:
        return [twin]
    if twin:
        return [field, twin]
    return [] if is_sentinel and not twin else [field]


def resolve_snapshot_cells(payload: dict, snapshot: dict) -> tuple:
    """(``{dotted path: spec}``, ``[rows that matched nothing]``).

    The overlay is keyed by PRINTED ROW LABEL because that is all the sent page
    knows; the payload keys rows by group and index. Resolving against the
    assembled report — the very payload about to be stored — keeps the preview
    and the freeze on one answer.
    """
    cells, missing = {}, []
    subs = (payload or {}).get("subtabs") or {}
    for sub, rows in (snapshot or {}).items():
        blk = subs.get(sub)
        if not isinstance(blk, dict):
            missing.extend([f"{sub}/{lbl}" for lbl in (rows or {})])
            continue
        index = _index_subtab(sub, blk)
        for label, fields in (rows or {}).items():
            prefix = index.get(_norm_row(label))
            if not prefix:
                missing.append(f"{sub}/{label}")
                continue
            is_deal = ".deals[" in prefix or ".ownership_flagged[" in prefix
            for field, spec in (fields or {}).items():
                for target in _snapshot_targets(sub, field, spec, is_deal):
                    out = dict(spec)
                    if target.endswith("_display") or "_display." in target:
                        # The twin renders, so the printed text goes in as the
                        # VALUE here rather than through `published_display` —
                        # the Snapshot components do not read that map.
                        if spec.get("display") is not None:
                            out = {**spec, "published": spec["display"],
                                   "display": None}
                    cells[f"{prefix}.{target}"] = out
    return cells, missing


def dry_run_unapplied(assembled: dict, overlay: dict) -> list:
    """Which overlay cells would NOT land, without freezing anything.

    Applies the overlay to a DEEP COPY of the assembled report and returns what
    `_apply_overlay` could not place. The preview and the freeze therefore
    answer from the same code — a preview that predicted differently from the
    write it precedes would be worse than none.
    """
    import copy
    payload = copy.deepcopy(assembled or {})
    one_pagers = {k: copy.deepcopy(v) for k, v in
                  (payload.get("one_pagers") or {}).items()}
    # A One Pager scope needs a target to write into; absent ones are reported
    # by the roster resolution, not here.
    for scope in (overlay or {}):
        if scope != "__subtabs__":
            one_pagers.setdefault(scope, {})
    _apply_overlay(payload, one_pagers, overlay or {})
    return payload.get("published_unapplied") or []


#: Roughly how many cells the 26Q2 overlay is expected to change. Shown on the
#: preview so a big deviation is obvious. NOT enforced — the point is that a
#: reader who sees 900 where 114 was expected stops and asks why, which no
#: automatic threshold does as well.
EXPECTED_DIFFERENCES_26Q2 = 114

#: A column is flagged when most of its rows differ, or when the typical
#: overlay/live ratio is nowhere near 1. Both are signatures of a mechanical
#: fault rather than a genuine correction: a units error lands near 1e6 or
#: 1e-6, and a column shift makes almost every row differ at once.
_MOSTLY_DIFFER = 0.60
_RATIO_LO, _RATIO_HI = 0.5, 2.0


def _num_or_none(v):
    if isinstance(v, bool) or v is None or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return None if f != f else f


def get_one_pager_live(vcode: str, quarter: str):
    """One deal's LIVE One Pager, through the same getter the freeze uses.

    The preview compares against exactly what the freeze would store, so it
    must read it the same way — a second route to the One Pager would let the
    preview describe a payload the freeze never sees.
    """
    try:
        from flask_app.services.portfolio_snapshot_freeze import (
            _default_one_pager_getter)
        return _default_one_pager_getter()(vcode, quarter)
    except Exception:
        log.exception("live One Pager unavailable for %s %s", vcode, quarter)
        return None


def compare_overlay_to_live(targets: dict, overlay: dict) -> dict:
    """What the overlay would CHANGE, per page and per column.

    ``targets`` is ``{"__subtabs__": assembled, vcode: one_pager, ...}`` — the
    live payloads the overlay is about to be written over.

    THREE THINGS A READER CANNOT GET FROM A CELL COUNT. How many cells actually
    move (a freeze that changes nothing means the overlay never landed); which
    columns move TOGETHER (a whole column differing is a column shift, not
    thirty independent corrections); and the typical ratio (a units error sits
    at 1e6 or 1e-6 and every individual cell looks plausible).

    A SENTINEL WHOSE LIVE VALUE IS NOT BLANK is called out separately. The page
    printed "—" while the app holds a figure — East Manchester's loan rate is
    the known case — so the frozen row would show a number that was never sent
    unless the printed dash is stored over it.
    """
    per_page, per_col, differing, sentinels = {}, {}, [], []
    for scope, cells in (overlay or {}).items():
        target = targets.get(scope)
        if target is None:
            continue
        for path, spec in (cells or {}).items():
            page = spec.get("page")
            pg = per_page.setdefault(page, {"cells": 0, "differs": 0})
            pg["cells"] += 1
            col = path.rsplit(".", 1)[-1]
            cc = per_col.setdefault(f"{scope if scope == '__subtabs__' else 'one_pager'}.{col}",
                                    {"n": 0, "differ": 0, "ratios": []})
            cc["n"] += 1

            live = _read_path(target, path)
            pub = spec.get("published")
            disp = spec.get("display")
            is_sentinel = spec.get("units") == "printed-sentinel"

            if is_sentinel:
                if live not in (None, "", "—", "n/a", "N/A", "Dev"):
                    sentinels.append({"scope": scope, "path": path,
                                      "printed": disp, "live": live,
                                      "page": page})
                    pg["differs"] += 1
                    cc["differ"] += 1
                    differing.append({"scope": scope, "path": path,
                                      "published": disp, "live": live,
                                      "page": page, "kind": "sentinel"})
                continue

            want = pub if pub is not None else disp
            same = (want == live)
            ln, wn = _num_or_none(live), _num_or_none(want)
            if ln is not None and wn is not None:
                same = abs(wn - ln) <= max(0.005, abs(wn) * 1e-6)
                if ln:
                    cc["ratios"].append(wn / ln)
            if not same:
                pg["differs"] += 1
                cc["differ"] += 1
                differing.append({"scope": scope, "path": path,
                                  "published": want, "live": live,
                                  "page": page, "kind": "value"})

    warnings = []
    for col, c in per_col.items():
        rs = sorted(c["ratios"])
        med = rs[len(rs) // 2] if rs else None
        c["median_ratio"] = med
        c.pop("ratios", None)
        if c["n"] >= 4 and c["differ"] / c["n"] >= _MOSTLY_DIFFER:
            warnings.append({
                "column": col, "kind": "most-rows-differ",
                "detail": f"{c['differ']} of {c['n']} rows differ — a column "
                          f"shift looks exactly like this"})
        if med is not None and not (_RATIO_LO <= med <= _RATIO_HI):
            warnings.append({
                "column": col, "kind": "ratio-far-from-one",
                "detail": f"typical overlay/live ratio is {med:.4g}"
                          + (" — that is a units error (1e6)"
                             if med > 1e5 or (med and med < 1e-5) else "")})

    return {
        "differs_total": sum(p["differs"] for p in per_page.values()),
        "cells_total": sum(p["cells"] for p in per_page.values()),
        "expected_differences": EXPECTED_DIFFERENCES_26Q2,
        "by_page": {str(k): v for k, v in sorted(
            per_page.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "by_column": per_col,
        "differing_cells": differing,
        "sentinels_live_non_blank": sentinels,
        "warnings": warnings,
    }
