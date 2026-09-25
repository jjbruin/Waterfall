"""Live per-deal value trace — the real numbers behind a calculated field.

WHAT THIS IS FOR. `lookup_field` says where a number comes from in the abstract:
"ROE is distributions over weighted-average capital over years". It cannot say
what those three numbers ARE for Burton Retail at 26Q2. This module answers that,
so an answer can show the arithmetic with the deal's own figures substituted in.

THE VALUE IS THE PAGE'S, NEVER A RECOMPUTE. Every figure returned here is read
out of the payload the SCREEN is rendered from — `get_one_pager_data` for the One
Pager, `portfolio_snapshot_freeze.build_subtab` for the Snapshot. Nothing in this
module computes a reported number. That is not a style preference: a trace that
quietly re-derives its own value will eventually disagree with the page, and a
breakdown that does not reconcile makes a CORRECT figure look wrong with no way
for the reader to tell which to believe — the `v505` statement-drilldown lesson,
and the reason that drilldown shares one row-selection function with the builder.

THE ARITHMETIC CHECK IS A CHECK, NOT A SOURCE. Each spec carries a `check`
recomputing the field from the components it just read. Its output is NEVER
returned as the value — it sets `reconciles`, so a breakdown that does not add up
says so instead of being presented as though it did.

NOTHING IS INVENTED. A component the builder does not publish comes back as
``{"available": false}`` with a reason. A field with no spec is reported as having
no backend breakdown rather than being given a plausible one.

ROE IS GATED ON AGREEMENT. `get_pe_performance` keeps only the scalar ROE — it
calls `calculate_roe`, not `calculate_roe_detailed`, and discards the components.
The only existing engine that publishes them is `reports_service
.build_roe_summary_row` (the ROE Summary report), which builds its event list on
slightly different rules — notably it has no 45-day look-forward window for late
pref distributions, which the One Pager does. So its breakdown is used ONLY when
its own ROE agrees with the One Pager's to within a tolerance. Where they differ,
the One Pager value is returned with the breakdown withheld and the disagreement
named. Showing a breakdown from a second engine that lands on a different number
would be the exact failure this module exists to avoid.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: A reconciliation tolerance, not a rounding rule. Ratios are held to a
#: relative tolerance because a 1e-9 float tail on 0.0812 is not a disagreement;
#: money is held to half a cent.
_REL_TOL = 1e-6
_ABS_TOL = 0.005


def _dig(payload: Any, path: tuple) -> Any:
    """Walk a key path, returning None rather than raising on any miss."""
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _num(value: Any) -> Optional[float]:
    """A float, or None for anything that is not a finite number.

    Strings are NOT coerced: the Snapshot writes display sentinels ("n/a",
    "Dev") into the same cells that otherwise carry ratios, and reading one of
    those as a number is how a suppressed cell would become a real figure.
    """
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _run_check(check: Optional[Callable], values: list) -> Optional[float]:
    """Run a spec's arithmetic check, never letting it break the answer.

    The check exists only to set `reconciles`. A trace whose figures are all
    correct must not be lost because a check divided by something unexpected, so
    a raise degrades to "not checked" (None) rather than propagating.
    """
    if not check:
        return None
    try:
        return check(values)
    except Exception:
        logger.exception("trace: reconciliation check raised")
        return None


def _agree(a: Optional[float], b: Optional[float]) -> Optional[bool]:
    """Do two figures agree? None when either is absent — never False."""
    if a is None or b is None:
        return None
    scale = max(abs(a), abs(b))
    return abs(a - b) <= max(_ABS_TOL, scale * _REL_TOL)


# ── what each traceable field is made of ─────────────────────────────────
#
# `value` and every `path` are READ POSITIONS in the page payload. `check`
# recomputes the field from the components for the reconciliation flag only.
# The component labels match the `inputs[].component` names in
# data_dictionary.json so each one can be paired with its own documented source.


def _debt_leg(cap: dict) -> Optional[float]:
    """The debt figure the printed columns actually add up to.

    A deal sold as of the quarter reports no debt, and `total_cap` is built from
    that zero — so checking `debt + pref + partner` against the published total
    would spuriously fail on every suppressed row. `sold_suppressed` is a
    published flag on the payload; this reads it rather than re-deriving when a
    deal counts as sold.
    """
    if not isinstance(cap, dict):
        return None
    if cap.get("sold_suppressed"):
        return 0.0
    return _num(cap.get("debt"))


_ONE_PAGER_SPECS: dict = {
    "one_pager.total_cap": {
        "value": ("cap_stack", "total_cap"),
        "components": [
            ("debt", ("cap_stack", "debt"), _debt_leg),
            ("pref_equity", ("cap_stack", "pref_equity"), None),
            ("partner_equity", ("cap_stack", "partner_equity"), None),
        ],
        "check": lambda c: (None if None in c else c[0] + c[1] + c[2]),
    },
    "one_pager.debt_pct": {
        "value": ("cap_stack", "debt_pct"),
        "components": [
            ("debt", ("cap_stack", "debt"), _debt_leg),
            ("total_cap", ("cap_stack", "total_cap"), None),
        ],
        "check": lambda c: (None if None in c or not c[1] else c[0] / c[1]),
    },
    "one_pager.pref_equity_pct": {
        "value": ("cap_stack", "pref_equity_pct"),
        "components": [
            ("pref_equity", ("cap_stack", "pref_equity"), None),
            ("total_cap", ("cap_stack", "total_cap"), None),
        ],
        "check": lambda c: (None if None in c or not c[1] else c[0] / c[1]),
    },
    "one_pager.partner_equity_pct": {
        "value": ("cap_stack", "partner_equity_pct"),
        "components": [
            ("partner_equity", ("cap_stack", "partner_equity"), None),
            ("total_cap", ("cap_stack", "total_cap"), None),
        ],
        "check": lambda c: (None if None in c or not c[1] else c[0] / c[1]),
    },
    # Published as a PERCENTAGE (x100), unlike the _pct fields above, which are
    # fractions. The check mirrors the builder rather than normalising either.
    "one_pager.pe_exposure_on_cap": {
        "value": ("cap_stack", "pe_exposure_on_cap"),
        "components": [
            ("debt", ("cap_stack", "debt"), _debt_leg),
            ("pref_equity", ("cap_stack", "pref_equity"), None),
            ("total_cap", ("cap_stack", "total_cap"), None),
        ],
        "check": lambda c: (None if None in c or not c[2]
                            else (c[0] + c[1]) / c[2] * 100),
    },
    "one_pager.pe_exposure_on_value": {
        "value": ("cap_stack", "pe_exposure_on_value"),
        "components": [
            ("debt", ("cap_stack", "debt"), _debt_leg),
            ("pref_equity", ("cap_stack", "pref_equity"), None),
            ("current_valuation", ("cap_stack", "current_valuation"), None),
        ],
        "check": lambda c: (None if None in c or not c[2]
                            else (c[0] + c[1]) / c[2] * 100),
    },
    # NOI here is actual_ye with a ytd_actual fallback — the same precedence
    # financials_service applies when it computes the field.
    "one_pager.pe_yield_on_exposure": {
        "value": ("cap_stack", "pe_yield_on_exposure"),
        "components": [
            ("NOI", ("property_performance", "noi", "actual_ye"),
             lambda p: _num(_dig(p, ("property_performance", "noi", "actual_ye")))
             if _num(_dig(p, ("property_performance", "noi", "actual_ye")))
             else _num(_dig(p, ("property_performance", "noi", "ytd_actual")))),
            ("debt", ("cap_stack", "debt"), None),
            ("pref_equity", ("cap_stack", "pref_equity"), None),
        ],
        "check": lambda c: (None if None in c or not (c[1] + c[2])
                            else c[0] / (c[1] + c[2])),
        # The builder uses cap['debt'] here, NOT the suppressed leg — so this
        # one deliberately does not go through _debt_leg.
    },
}

#: Bases the One Pager reports DSCR and NOI on, in the order the page prints.
_BASES = ("ytd_actual", "ytd_budget", "actual_ye", "uw_ye", "at_close")

#: (row key, [(component label, row key)]) for the Snapshot Loan subtab.
_SNAPSHOT_LOAN_SPECS: dict = {
    "snapshot_loan.ltv": {
        "value": "ltv",
        "components": [("debt", "debt"), ("valuation", "valuation")],
        # Published as a percentage.
        "check": lambda c: (None if None in c or not c[1] else c[0] / c[1] * 100),
    },
    "snapshot_loan.debt_yield": {
        "value": "debt_yield",
        "components": [("quarter NOI", "quarter_noi"), ("debt", "debt")],
        "check": lambda c: (None if None in c or not c[1]
                            else c[0] * 4 / c[1] * 100),
    },
    "snapshot_loan.ytd_dscr": {
        "value": "ytd_dscr",
        # The builder publishes the ratio and its NOI numerator; the debt
        # service it divided by is not on the row, so it is reported absent
        # rather than implied backwards out of the ratio.
        "components": [("ytd NOI", "ytd_noi"), ("debt service", None)],
        "check": None,
    },
}


def traceable_field_ids() -> list:
    """Every field this module can break down, for the tool's own reporting."""
    return (sorted(_ONE_PAGER_SPECS)
            + ["one_pager.dscr", "one_pager.roe_to_date",
               "one_pager.uw_roe_to_date"]
            + sorted(_SNAPSHOT_LOAN_SPECS))


# ── the page payloads ────────────────────────────────────────────────────

def _one_pager_payload(vcode: str, quarter: Optional[str]) -> tuple:
    """(payload, resolved_quarter) from the SAME call the One Pager route makes.

    `full_data` and `mri_loans_all` ARE passed. The page route
    (flask_app/api/financials.py) passes both; `_tool_get_one_pager` in
    assistant_service does not, and without `full_data` the PE enrichment
    (`_enrich_pe_from_deal_result`) never runs — so the balances would be the
    pre-enrichment ones and would not tie to the screen. A trace is worth
    nothing if it is tracing a different payload from the one on display.
    """
    from flask_app.services import data_service
    from flask_app.services.financials_service import get_one_pager_data
    from one_pager import get_available_quarters, most_recent_completed_quarter

    data = data_service.get_data()

    # The builder defaults the quarter internally but does not report which one
    # it chose, and the ROE report date has to match it. Resolved here with the
    # builder's own helpers so the two cannot pick different quarters.
    resolved_q = quarter
    if not resolved_q:
        available = (get_available_quarters(data["isbs_raw"])
                     if data.get("isbs_raw") is not None else [])
        if available:
            resolved_q = most_recent_completed_quarter(available) or available[0]

    payload = get_one_pager_data(
        vcode, resolved_q, data["inv"], data["isbs_raw"],
        data["mri_loans_raw"], data["mri_val"], data["wf"], data["acct"],
        occupancy_raw=data["occupancy_raw"],
        budget_econ_occ=data.get("budget_econ_occ"),
        deal_terms=data.get("deal_terms_raw"),
        at_close_noi=data.get("at_close_noi_raw"),
        event_dates=data.get("event_dates_raw"),
        full_data=data,
        relationships=data.get("relationships_raw"),
        mri_loans_all=data.get("mri_loans_all"),
        inspection=data.get("inspection_raw"),
    )
    return payload, resolved_q


def _snapshot_loan_row(vcode: str, investor_code: str, quarter: str) -> dict:
    """The Snapshot Loan row for one deal, via the page's own assembly point.

    `build_subtab` is documented as the only place an assembly is invoked, so it
    is what gets called — the subtab is built for the whole investor and the
    deal's row picked out of it, because that is the row the screen renders.
    """
    from flask_app.services import data_service
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    from flask_app.services.portfolio_snapshot_freeze import build_subtab

    data = data_service.get_data()
    resolved = resolve_investor_deals(investor_code, quarter, data=data)
    subtab = build_subtab("loan", investor_code, quarter, data, resolved)
    want = str(vcode).strip().upper()
    for row in (subtab.get("rows") or []):
        if str(row.get("vcode", "")).strip().upper() == want:
            return row
    return {}


# ── the ROE breakdown, gated ─────────────────────────────────────────────

def _roe_breakdown(vcode: str, quarter: str, page_value: Optional[float],
                   uw: bool = False) -> dict:
    """ROE components from the ROE Summary engine, used only if it agrees.

    Returns either a populated breakdown or a reason it is being withheld. See
    the module docstring: the One Pager keeps no components of its own, and the
    only engine that publishes them applies a different rule to late pref
    distributions.
    """
    from flask_app.services import data_service
    from flask_app.services.reports_service import build_roe_summary_row
    from one_pager import quarter_to_date_range

    try:
        _, report_date = quarter_to_date_range(quarter)
        data = data_service.get_data()
        row = build_roe_summary_row(
            vcode, vcode, data["acct"], data["inv"], report_date,
            wf_steps=data.get("wf"), isbs_raw=data.get("isbs_raw"))
    except Exception as exc:
        logger.exception("trace: ROE summary row failed for %s", vcode)
        return {"available": False,
                "reason": f"the ROE Summary engine could not price this deal: {exc}"}

    if not row:
        return {"available": False,
                "reason": ("the ROE Summary engine returned no row for this deal "
                           "and period, so there are no components to show")}

    engine_value = _num(row.get("U/W ITD ROE" if uw else "ITD ROE"))
    if _agree(engine_value, page_value) is not True:
        return {
            "available": False,
            "reason": ("the breakdown is withheld because the ROE Summary "
                       "engine computes this deal's ROE as "
                       f"{engine_value!r} where the One Pager shows "
                       f"{page_value!r}. The two apply different rules to "
                       "distributions arriving just after the quarter end, so "
                       "its components would not add up to the figure on the "
                       "page."),
            "one_pager_value": page_value,
            "roe_summary_value": engine_value,
        }

    if uw:
        return {
            "available": True,
            "source_engine": "reports_service.build_roe_summary_row (ROE Summary report)",
            "components": [
                {"component": "UW distributions (numerator)",
                 "value": _num(row.get("_uw_cf_total")), "available": True},
                {"component": "UW return of capital",
                 "value": _num(row.get("_uw_roc_total")), "available": True},
            ],
        }

    return {
        "available": True,
        "source_engine": "reports_service.build_roe_summary_row (ROE Summary report)",
        "components": [
            {"component": "Distributions (numerator)",
             "value": _num(row.get("CF Received")), "available": True},
            {"component": "Weighted-avg capital (denominator)",
             "value": _num(row.get("Wtd Avg Balance")), "available": True},
            {"component": "years",
             "value": _num(row.get("_years")), "available": True},
            {"component": "inception", "value": str(row.get("_inception")),
             "available": row.get("_inception") is not None},
            {"component": "total days", "value": _num(row.get("_total_days")),
             "available": True},
        ],
    }


# ── the dictionary's own source labels ───────────────────────────────────

def _sources_for(field_id: str) -> dict:
    """{component name: documented source} from data_dictionary.json.

    Every traced number is labelled with the source the dictionary records for
    that component. Nothing is invented: a component the dictionary does not
    carry gets no source line rather than a guessed one.
    """
    try:
        from flask_app.services import data_dictionary_service as D
        entry = D.lookup_field(field_id)
        if entry.get("error"):
            return {}
        out = {}
        for i in entry.get("inputs") or []:
            name = str(i.get("component") or "").strip()
            if name:
                out[name] = i.get("source")
        return out
    except Exception:
        logger.exception("trace: could not read dictionary sources for %s", field_id)
        return {}


def _dictionary_head(field_id: str) -> dict:
    """definition / formula / formula_latex, passed through verbatim."""
    try:
        from flask_app.services import data_dictionary_service as D
        entry = D.lookup_field(field_id)
        if entry.get("error"):
            return {}
        return {
            "definition": entry.get("definition"),
            "formula": entry.get("formula"),
            "formula_latex": entry.get("formula_latex"),
            "formula_is_arithmetic": entry.get("formula_is_arithmetic"),
        }
    except Exception:
        return {}


# ── the entry point ──────────────────────────────────────────────────────

def trace_field_value(deal_id: str, field_id: str, quarter: str = None,
                      investor_code: str = None) -> dict:
    """A field's computed value for one deal, plus the numbers behind it.

    Every figure is read from the payload the page renders. A component that is
    not published comes back unavailable with a reason; a field with no backend
    breakdown is reported as such rather than given one.
    """
    vcode = str(deal_id or "").strip()
    fid = str(field_id or "").strip()
    if not vcode:
        return {"error": "A deal id (vcode) is required to trace a value."}
    if not fid:
        return {"error": "A field_id is required."}

    head = _dictionary_head(fid)
    sources = _sources_for(fid)

    def _label(components: list) -> list:
        """Attach each component's documented source and drop empty values."""
        out = []
        for c in components:
            name = c.get("component")
            item = dict(c)
            if name in sources:
                item["source"] = sources[name]
            out.append(item)
        return out

    base = {"deal_id": vcode, "field_id": fid, **head}

    # ── Snapshot fields ──────────────────────────────────────────────
    if fid in _SNAPSHOT_LOAN_SPECS:
        if not investor_code:
            return {
                **base,
                "error": ("This is a Portfolio Snapshot field, and the Snapshot "
                          "is assembled per INVESTOR — the same deal appears "
                          "under several investors. Ask the user which investor "
                          "and pass investor_code. It is not carried in the page "
                          "context, so it cannot be inferred."),
            }
        if not quarter:
            return {**base, "error": "A quarter (e.g. 26Q2) is required for a Snapshot trace."}
        try:
            row = _snapshot_loan_row(vcode, investor_code, quarter)
        except Exception as exc:
            logger.exception("trace: snapshot loan failed for %s/%s", investor_code, vcode)
            return {**base, "error": f"Could not build the Snapshot Loan subtab: {exc}"}
        if not row:
            return {**base, "error": (f"'{vcode}' has no Loan row for investor "
                                      f"{investor_code} at {quarter}.")}

        spec = _SNAPSHOT_LOAN_SPECS[fid]
        value = _num(row.get(spec["value"]))
        comps, vals = [], []
        for name, key in spec["components"]:
            v = _num(row.get(key)) if key else None
            vals.append(v)
            comps.append({"component": name, "value": v,
                          "available": v is not None,
                          **({} if v is not None else
                             {"reason": ("not published on the Snapshot row — "
                                         "only the ratio and its numerator are")})})
        recomputed = _run_check(spec.get("check"), vals)
        return {
            **base,
            "quarter": quarter,
            "investor_code": investor_code,
            "tab": "snapshot_loan",
            "value": value,
            "value_available": value is not None,
            "source_payload": "portfolio_snapshot_freeze.build_subtab('loan') — the Snapshot's own assembly",
            "inputs": _label(comps),
            "reconciles": _agree(recomputed, value),
            "basis_note": row.get("debt_yield_basis") if fid.endswith("debt_yield") else None,
        }

    # ── One Pager fields ─────────────────────────────────────────────
    try:
        payload, resolved_q = _one_pager_payload(vcode, quarter)
    except Exception as exc:
        logger.exception("trace: One Pager payload failed for %s", vcode)
        return {**base, "error": f"Could not build the One Pager for '{vcode}': {exc}"}
    if not payload:
        return {**base, "error": f"No One Pager data for '{vcode}'."}

    base = {**base, "quarter": resolved_q, "tab": "one_pager",
            "source_payload": ("financials_service.get_one_pager_data — the same "
                               "call the One Pager route makes, full_data included")}

    # DSCR is reported on five bases, not as one number.
    if fid == "one_pager.dscr":
        rows = []
        for b in _BASES:
            ratio = _num(_dig(payload, ("property_performance", "dscr", b)))
            noi = _num(_dig(payload, ("property_performance", "noi", b)))
            rows.append({
                "basis": b, "dscr": ratio, "available": ratio is not None,
                "inputs": [
                    {"component": "NOI (numerator)", "value": noi,
                     "available": noi is not None,
                     "source": sources.get(b)},
                    {"component": "debt service (denominator)", "value": None,
                     "available": False,
                     "reason": ("the One Pager publishes the ratio and its NOI "
                                "numerator but not the debt service it divided "
                                "by, so it is not reported here rather than "
                                "being worked backwards out of the ratio"),
                     "source": sources.get(b)},
                ],
            })
        return {**base, "multi_basis": True, "bases": rows,
                "value": _num(_dig(payload, ("property_performance", "dscr", "ytd_actual"))),
                "note": ("DSCR is reported on five bases. The figure quoted "
                         "alone is ytd_actual.")}

    # ROE — value from the page, components gated on the two engines agreeing.
    if fid in ("one_pager.roe_to_date", "one_pager.uw_roe_to_date"):
        key = "roe_to_date" if fid.endswith(".roe_to_date") else "uw_roe_to_date"
        value = _num(_dig(payload, ("pe_performance", key)))
        breakdown = (_roe_breakdown(vcode, resolved_q, value,
                                    uw=key.startswith("uw"))
                     if resolved_q else
                     {"available": False, "reason": "no quarter could be resolved"})
        out = {**base, "value": value, "value_available": value is not None}
        # `.get(... ) or []` rather than indexing: a breakdown that says it is
        # available but carries no components is a bug, and it should degrade to
        # "no breakdown" rather than take the whole answer down with a KeyError.
        if breakdown.get("available") and (breakdown.get("components") or []):
            out["inputs"] = _label(breakdown.get("components") or [])
            out["breakdown_engine"] = breakdown.get("source_engine")
            out["reconciles"] = True
        else:
            out["inputs"] = []
            out["breakdown_available"] = False
            out["breakdown_unavailable_reason"] = breakdown.get("reason")
            out["reconciles"] = None
        return out

    spec = _ONE_PAGER_SPECS.get(fid)
    if spec:
        value = _num(_dig(payload, spec["value"]))
        comps, vals = [], []
        for name, path, getter in spec["components"]:
            # Three cases, in order: the suppressed-debt helper (which reads the
            # cap_stack block), any other getter (which reads the whole
            # payload), or a plain key path.
            if getter is _debt_leg:
                v = _debt_leg(payload.get("cap_stack") or {})
            elif getter is not None:
                try:
                    v = getter(payload)
                except Exception:
                    logger.exception("trace: component getter failed for %s/%s",
                                     fid, name)
                    v = None
            else:
                v = _num(_dig(payload, path))
            vals.append(v)
            comps.append({"component": name, "value": v,
                          "available": v is not None})
        recomputed = _run_check(spec.get("check"), vals)
        return {
            **base,
            "value": value,
            "value_available": value is not None,
            "inputs": _label(comps),
            "reconciles": _agree(recomputed, value),
        }

    # ── a field with no breakdown: report the value and say so ───────
    direct = _direct_value(payload, fid)
    return {
        **base,
        "value": direct.get("value"),
        "value_available": direct.get("value") is not None,
        "inputs": [],
        "breakdown_available": False,
        "breakdown_unavailable_reason": (
            "This field has no numeric breakdown in the trace tool — either it "
            "comes straight from a source rather than being calculated, or no "
            "component decomposition has been built for it. Its documented "
            "source is in the field dictionary."),
        "traceable_fields": traceable_field_ids(),
        **({"lookup_note": direct["note"]} if direct.get("note") else {}),
    }


def _direct_value(payload: dict, field_id: str) -> dict:
    """Best-effort read of a non-calculated field from the payload.

    Searches the three published sections for the bare field name. Returns no
    value rather than a wrong one: this never reaches outside the payload, and a
    name that is not a key in it simply comes back empty.
    """
    bare = field_id.rsplit(".", 1)[-1]
    for section in ("cap_stack", "pe_performance", "general", "property_performance"):
        block = payload.get(section)
        if isinstance(block, dict) and bare in block:
            v = block[bare]
            num = _num(v)
            return {"value": num if num is not None else v,
                    "note": f"read from the One Pager payload section '{section}'"}
    return {"value": None}
