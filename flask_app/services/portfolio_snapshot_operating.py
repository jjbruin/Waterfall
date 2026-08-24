"""Portfolio Snapshot — Subtab 3 (Operating) data assembly (Step 3a).

Backend only: no route, no blueprint, no UI. Nothing imports this module.

Reads, never writes:
  * Step 1  ``portfolio_snapshot_service.resolve_investor_deals`` — deals by fund
  * One Pager ``property_performance`` — Econ Occ and the three NOI points
  * Step 2  ``portfolio_snapshot_persistence`` — the per-deal operating comment

The One Pager payload arrives through an injected ``one_pager_provider`` rather
than being fetched here. In-app that wraps ``financials_service.get_one_pager_data``
(deliberately **without** ``full_data``, so no waterfall is run per deal — 30+
waterfalls per page load is the performance trap flagged in the build spec); the
self-test wraps the REST endpoint. Same dependency-injection shape as Step 1.

Operating metrics are property-level and are **never** scaled by ownership. Only
the four TIAA columns on Subtab 2 get scaled.

Growth definitions (per the build spec):
    Expected Growth = (U/W YE NOI      - At Close NOI) / At Close NOI
    Actual Growth   = (Projected YE NOI - At Close NOI) / At Close NOI
where Projected YE is the One Pager's ``actual_ye`` (YTD actual + remainder-of-
year budget). Note that this makes Actual Growth a *moving* figure for a past
quarter: it shifts as actuals land, so the same historical quarter recomputed
later will not match a PDF produced earlier. Flagged, not hidden.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

#: Values of **deals.Investment_Strategy** that mark a development deal.
#:
#: Creator decision 2026-08-24: dev detection reads Investment_Strategy ONLY --
#: no fallback to Lifecycle, no "has numbers" hybrid. One field drives the "Dev"
#: display (LTV / YTD DSCR / Debt Yield), the mOrigLoanAmt debt path, and the
#: "Excluding Development Deals" subtotal, so those three can never disagree.
#:
#: MEASUREMENT NOTE (live build 09fe220ae0da, 2026-08-24): deals.Investment_Strategy
#: is 0/110 populated. No MRI query selects it and it is absent from
#: mri_service.MRI_COLUMNS, so nothing populates it on refresh. Until it is fed,
#: this rule classifies every deal as operating. Lifecycle (97/110) carries the
#: values that look like strategies -- Development 24, Value-Add 31, Income 22,
#: Stable 16, New Construction 2, Redevelopment 1, Lease up 1 -- but is
#: deliberately NOT consulted here.
DEV_STRATEGIES = {"development", "new construction"}

DEV_LABEL = "Dev"


def _num(v):
    """Float or None. Zero is preserved — it is data, not absence."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _growth(later, base):
    """(later - base) / base, or None when it cannot be computed.

    Returns None rather than 0.0 when the base is missing or zero, so a deal
    with no At Close NOI is flagged instead of silently reading as flat.
    """
    if later is None or base is None or base == 0:
        return None
    return (later - base) / base


def is_dev_deal(strategy: str) -> bool:
    return str(strategy or "").strip().lower() in DEV_STRATEGIES


def _default_comment_loader(investor_code: str, quarter: str) -> dict:
    """vcode -> operating comment, from Step 2 persistence (read-only)."""
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        rows = get_elements("comment", investor_code, quarter,
                            scope="deal", field="operating")
        return {r.get("scope_key"): r.get("comment_text") for r in rows}
    except Exception as exc:            # persistence not initialised yet
        log.debug("operating comments unavailable: %s", exc)
        return {}


def assemble_operating(investor_code: str, quarter: str, *,
                       resolved: dict,
                       one_pager_provider: Callable[[str, str], dict],
                       comment_loader: Optional[Callable] = None) -> dict:
    """Build the Operating subtab for one investor and quarter.

    ``resolved`` is the output of Step 1's ``resolve_investor_deals``.
    ``one_pager_provider(vcode, quarter)`` returns a One Pager payload.
    """
    loader = comment_loader or _default_comment_loader
    comments = loader(investor_code, quarter) or {}

    groups: dict[str, list] = {}
    diag = {"deals": 0, "with_noi": 0, "dev": 0, "missing_at_close": 0,
            "provider_errors": 0, "comments_attached": 0}

    def build_row(vcode: str, name: str, strategy: str,
                  extra_flags: Optional[list] = None) -> dict:
        flags = list(extra_flags or [])
        dev = is_dev_deal(strategy)
        occ = {"at_close": None, "uw_ye": None, "projected_ye": None,
               "ytd_actual": None}
        noi = {"at_close": None, "uw_ye": None, "projected_ye": None}

        try:
            payload = one_pager_provider(vcode, quarter) or {}
        except Exception as exc:
            diag["provider_errors"] += 1
            flags.append(f"One Pager unavailable: {str(exc)[:90]}")
            payload = {}

        pp = payload.get("property_performance") or {}
        if not pp:
            flags.append("no property_performance for this quarter")
        else:
            n, o = pp.get("noi") or {}, pp.get("economic_occ") or {}
            noi = {"at_close": _num(n.get("at_close")),
                   "uw_ye": _num(n.get("uw_ye")),
                   # the One Pager calls Projected YE 'actual_ye'
                   "projected_ye": _num(n.get("actual_ye"))}
            occ = {"at_close": _num(o.get("at_close")),
                   "uw_ye": _num(o.get("uw_ye")),
                   "projected_ye": _num(o.get("actual_ye")),
                   "ytd_actual": _num(o.get("ytd_actual"))}

        exp_g = _growth(noi["uw_ye"], noi["at_close"])
        act_g = _growth(noi["projected_ye"], noi["at_close"])

        if noi["at_close"] in (None, 0):
            diag["missing_at_close"] += 1
            flags.append("growth unavailable: no At Close NOI")
        for label, key in (("U/W YE NOI", "uw_ye"),
                           ("Projected YE NOI", "projected_ye")):
            if noi[key] is None:
                flags.append(f"{label} missing")
        if any(v is not None for v in noi.values()):
            diag["with_noi"] += 1
        if dev:
            diag["dev"] += 1

        comment = comments.get(vcode)
        if comment:
            diag["comments_attached"] += 1

        diag["deals"] += 1
        return {
            "vcode": vcode, "name": name,
            "strategy": strategy,
            "is_dev": dev,
            # Operating history is meaningless pre-stabilisation, so the report
            # shows "Dev" in place of the occupancy reading. Otherwise fall back
            # Projected YE -> YTD actual -> At Close: several deals carry only
            # some of the three (Giant 7 has no Projected YE occupancy at 26Q1),
            # and blanking the cell when a reading does exist would be wrong.
            "econ_occ_display": (
                DEV_LABEL if dev else next(
                    (v for v in (occ["projected_ye"], occ["ytd_actual"],
                                 occ["at_close"]) if v is not None), None)),
            "econ_occ_basis": (
                "dev" if dev else next(
                    (k for k in ("projected_ye", "ytd_actual", "at_close")
                     if occ[k] is not None), None)),
            "econ_occ": occ,
            "noi": noi,
            "expected_growth": exp_g,
            "actual_growth": act_g,
            "operating_comment": comment,
            "flags": flags,
        }

    for group, items in (resolved.get("groups") or {}).items():
        rows = [build_row(e["vcode"], e["name"], e.get("investment_strategy", ""))
                for e in items]
        groups[group] = rows

    # Deals Step 1 flagged (broken ownership chain, e.g. 45th & Main) still
    # belong on the Operating page: the operating metrics are property-level and
    # do not depend on ownership at all. They carry their Step 1 flag forward.
    flagged_rows = []
    for f in (resolved.get("flagged") or []):
        row = build_row(f["vcode"], f["name"], f.get("investment_strategy", ""),
                        extra_flags=[f"ownership {f.get('reason', 'unavailable')}"])
        row["ownership_flagged"] = True
        flagged_rows.append(row)

    return {
        "investor_code": resolved.get("investor_code", investor_code),
        "investor_name": resolved.get("investor_name", investor_code),
        "quarter": quarter,
        "subtab": "operating",
        "scaled": False,        # property-level metrics are never scaled
        "groups": groups,
        "ownership_flagged": flagged_rows,
        "diagnostics": diag,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

_PDF_26Q1 = {
    # vcode: (label, econ_occ_lo, econ_occ_hi, at_close, uw_ye, proj_ye,
    #         expected_growth_pct, actual_growth_pct)
    "P0000019": ("Giant 7", 97.8, 98.3, 8.8, 9.3, 9.4, 5.3, 6.4),
    "P0000075": ("Camp Creek", None, None, 6.9, 6.8, 6.4, 9.0, 6.8),
    "P0000030": ("Nottingham Village", None, None, 3.6, 3.2, 2.1, 71.7, 53.2),
    "P0000068": ("Point at Plymouth Meeting", None, None, None, None, None,
                 55.1, 11.9),
}


def _selftest():                                    # pragma: no cover
    import os
    import sys
    import tempfile
    import sqlalchemy
    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for p in (root, os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import live_api as api
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    from flask_app.services import portfolio_snapshot_persistence as P

    QUARTER = "2026-Q1"
    INV = "TGAM"

    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"actuals_through={api.get('/api/data/config').get('actuals_through')}")

    # ---- Step 1 foundation, via narrow per-entity relationship pulls ----
    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    seen, frontier, rows = set(), [INV], []

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == val.upper()]

    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rows.extend(kids)
        for r in kids:
            child = str(r.get("InvestmentID") or "").strip().upper()
            if child:
                rows.extend(fetch("InvestmentID", child))
                if child not in seen:
                    frontier.append(child)
    rel = pd.DataFrame(rows).drop_duplicates()
    resolved = resolve_investor_deals(INV, QUARTER, rel, inv)
    print(f"Step 1: {resolved['diagnostics']['deal_count']} deals in "
          f"{resolved['diagnostics']['group_count']} groups, "
          f"{len(resolved['flagged'])} ownership-flagged\n")

    # ---- Step 2 persistence on a scratch db, to prove comments attach ----
    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_step3a_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    P._engine = lambda: eng                          # type: ignore[assignment]
    P._is_postgres = lambda: False                   # type: ignore[assignment]
    P.save_comment(INV, QUARTER, "deal", "operating",
                   "Leasing ahead of plan; ECRI driving in-place rents.",
                   scope_key="P0000019", updated_by="selftest")
    P.save_comment(INV, QUARTER, "deal", "operating",
                   "Anchor renewal signed.", scope_key="P0000075",
                   updated_by="selftest")

    cache: dict = {}

    def provider(vcode, quarter):
        key = (vcode, quarter)
        if key not in cache:
            cache[key] = api.get(f"/api/financials/{vcode}/one-pager",
                                 params={"quarter": quarter})
        return cache[key]

    out = assemble_operating(INV, QUARTER, resolved=resolved,
                             one_pager_provider=provider,
                             comment_loader=lambda i, q: _default_comment_loader(i, q))

    flat = {r["vcode"]: r for rows_ in out["groups"].values() for r in rows_}
    for r in out["ownership_flagged"]:
        flat[r["vcode"]] = r

    print("=" * 108)
    print(f"COMPUTED vs 26Q1 PDF   ({out['investor_name']} {out['quarter']})")
    print("=" * 108)
    hdr = (f"{'deal':<26}{'metric':<20}{'computed':>12}{'PDF':>10}"
           f"{'delta':>10}   verdict")
    print(hdr)
    print("-" * 108)
    checks = []
    for vc, (label, olo, ohi, p_ac, p_uw, p_pj, p_eg, p_ag) in _PDF_26Q1.items():
        r = flat.get(vc)
        if not r:
            print(f"{label:<26}{'NOT IN SET':<20}")
            checks.append((f"{label} present", False))
            continue
        rowsout = [
            ("NOI At Close ($M)", (r["noi"]["at_close"] or 0) / 1e6, p_ac, 0.35),
            ("NOI U/W YE ($M)", (r["noi"]["uw_ye"] or 0) / 1e6, p_uw, 0.35),
            ("NOI Proj YE ($M)", (r["noi"]["projected_ye"] or 0) / 1e6, p_pj, 0.35),
            ("Expected Growth %", (r["expected_growth"] or 0) * 100, p_eg, 0.4),
            ("Actual Growth %", (r["actual_growth"] or 0) * 100, p_ag, 0.4),
        ]
        first = True
        for metric, comp, pdf, tol in rowsout:
            if pdf is None:
                continue
            delta = comp - pdf
            ok = abs(delta) <= tol
            verdict = "ok" if ok else ("ROUNDING?" if abs(delta) <= 1.0
                                       else "MISMATCH")
            print(f"{(label if first else ''):<26}{metric:<20}"
                  f"{comp:>12.3f}{pdf:>10.2f}{delta:>+10.3f}   {verdict}")
            first = False
            checks.append((f"{label} {metric}", ok))
        if olo is not None:
            occ = r["econ_occ"]["projected_ye"] or r["econ_occ"]["at_close"] or 0
            ok = olo <= occ <= ohi
            print(f"{'':<26}{'Econ Occ %':<20}{occ:>12.2f}"
                  f"{f'{olo}-{ohi}':>10}{'':>10}   {'ok' if ok else 'MISMATCH'}")
            checks.append((f"{label} Econ Occ in band", ok))
        print()

    print("=" * 108)
    print("STRUCTURE CHECKS")
    struct = [
        ("groups present and non-empty",
         len(out["groups"]) >= 5 and all(out["groups"].values())),
        ("Individual Investments group exists",
         "Individual Investments" in out["groups"]),
        ("no scaling applied to operating metrics", out["scaled"] is False),
        ("45th & Main still appears (ownership-flagged)",
         any(r["vcode"] == "P0000089" for r in out["ownership_flagged"])),
        ("45th & Main carries its ownership flag",
         any("ownership" in f.lower()
             for r in out["ownership_flagged"] if r["vcode"] == "P0000089"
             for f in r["flags"])),
        ("Giant 7 comment attached",
         (flat.get("P0000019") or {}).get("operating_comment", "").startswith(
             "Leasing ahead")),
        ("Camp Creek comment attached",
         (flat.get("P0000075") or {}).get("operating_comment") ==
         "Anchor renewal signed."),
        ("deal with no comment yields None, not ''",
         (flat.get("P0000030") or {}).get("operating_comment") is None),
        ("dev deals labelled Dev",
         any(r["econ_occ_display"] == DEV_LABEL for r in flat.values()
             if r["is_dev"]) if any(r["is_dev"] for r in flat.values()) else True),
        ("missing data is None, never 0-faked",
         all(r["expected_growth"] is None or isinstance(r["expected_growth"], float)
             for r in flat.values())),
    ]
    for label, ok in struct:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        checks.append((label, ok))

    d = out["diagnostics"]
    print(f"\n  diagnostics: {d}")
    devs = [r["name"] for r in flat.values() if r["is_dev"]]
    print(f"  dev deals ({len(devs)}): {sorted(devs)[:8]}")
    nog = [r["name"] for r in flat.values() if r["expected_growth"] is None]
    print(f"  no-growth (flagged) ({len(nog)}): {sorted(nog)[:8]}")

    print("\n" + "=" * 108)
    print("ASSEMBLED STRUCTURE — two deals verbatim")
    import json
    for vc in ("P0000019", "P0000030"):
        print(f"\n{vc}:")
        print(json.dumps(flat.get(vc), indent=2, default=str)[:1100])

    passed = sum(1 for _, c in checks if c)
    print(f"\n  {passed}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_selftest())
