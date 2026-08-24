"""Portfolio Snapshot API — the four subtab assemblies, dropdowns, and the
editable-element + approval endpoints.

Step 7 Layer 1. This is the FIRST module to import the six
``portfolio_snapshot_*`` service modules, which were inert until now.

Structure mirrors ``reports.py``: a module-level blueprint, a ``_get_data()``
wrapper over ``data_service.get_data()``, ``@login_required`` on every route,
``safe_json`` on anything holding numpy/pandas scalars.

One bundle endpoint (``GET /bundle``) builds all four subtabs in a single
request, so the shell fetches once per (investor, quarter) rather than four
times. The four per-subtab endpoints exist alongside it for debugging and for
the paired Excel work later; they share the same providers.

PERFORMANCE NOTE: the One Pager provider is deliberately called WITHOUT
``full_data``. Passing it runs a full deal-analysis waterfall per deal — 30+
waterfalls for one page load, the trap flagged in the build spec. The snapshot
needs ``cap_stack`` and ``property_performance``, neither of which depends on
that enrichment. Do not add ``full_data=data`` here without measuring.
"""

from flask import Blueprint, request, jsonify
import pandas as pd

from flask_app.auth.routes import login_required
from flask_app.serializers import safe_json
from flask_app.services import data_service

portfolio_snapshot_bp = Blueprint("portfolio_snapshot", __name__)


def _get_data():
    return data_service.get_data()


class _DataUnavailable(RuntimeError):
    """Raised when the shared data cache cannot be loaded."""


def _data_or_error():
    """(data, None) or (None, flask response).

    Without this a data-load failure escapes as an HTML 500 and the shell has
    nothing to display. Local dev hits it routinely — an empty SQLite raises
    "no such table: coa" — and in production a bad refresh would do the same.
    """
    try:
        return _get_data(), None
    except Exception as exc:
        return None, (jsonify({"error": f"data unavailable: {exc}"}), 503)


# ── providers ─────────────────────────────────────────────────────────────

def _make_one_pager_provider(data: dict):
    """(vcode, quarter) -> One Pager payload, memoised per request.

    Memoised because the four assemblies each read the same deals: without the
    cache a bundle call would hit the One Pager 4x per deal.
    """
    from flask_app.services.financials_service import get_one_pager_data

    cache: dict = {}

    def provider(vcode: str, quarter: str) -> dict:
        key = (vcode, quarter)
        if key not in cache:
            cache[key] = get_one_pager_data(
                vcode, quarter, data["inv"], data["isbs_raw"],
                data["mri_loans_raw"], data["mri_val"],
                data["wf"], data["acct"],
                occupancy_raw=data["occupancy_raw"],
                budget_econ_occ=data.get("budget_econ_occ"),
                deal_terms=data.get("deal_terms_raw"),
                at_close_noi=data.get("at_close_noi_raw"),
                event_dates=data.get("event_dates_raw"),
                relationships=data.get("relationships_raw"),
                mri_loans_all=data.get("mri_loans_all"),
                # full_data deliberately omitted — see module docstring
            )
        return cache[key]

    return provider


def _make_quarterly_noi_provider(data: dict):
    """(vcode, quarter) -> that quarter's periodic NOI, or None.

    The Loan subtab's Debt Yield is single-quarter NOI x 4. Reads the same
    ``get_performance_chart_data`` pipeline Property Financials uses, so the
    number cannot drift from the chart. A quarter missing any of its three
    months comes back None and the subtab flags it rather than annualising a
    stub.
    """
    from flask_app.services.financials_service import get_performance_chart_data

    cache: dict = {}

    def provider(vcode: str, quarter: str):
        key = (vcode, quarter)
        if key in cache:
            return cache[key]
        val = None
        try:
            year, qn = int(str(quarter).split("-Q")[0]), int(str(quarter).split("Q")[1])
            q_end = (pd.Timestamp(year=year, month=qn * 3, day=1)
                     + pd.offsets.MonthEnd(0))
            chart = get_performance_chart_data(
                data["isbs_raw"], data["occupancy_raw"], vcode,
                freq="Quarterly", periods=12, period_end=str(q_end.date()),
            ) or {}
            label = f"Q{qn} {year}"
            for lbl, actual in zip(chart.get("periods") or [],
                                   chart.get("actual_noi") or []):
                if lbl == label and actual is not None:
                    val = float(actual)
                    break
        except Exception:
            val = None
        cache[key] = val
        return val

    return provider


def _resolve(investor_code: str, quarter: str, data: dict) -> dict:
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    return resolve_investor_deals(
        investor_code, quarter,
        data.get("relationships_raw"), data["inv"],
    )


def _args():
    """(investor_code, quarter) from the query string, both required."""
    investor = (request.args.get("investor") or "").strip().upper()
    quarter = (request.args.get("quarter") or "").strip()
    return investor, quarter


def _missing(investor: str, quarter: str):
    if not investor:
        return jsonify({"error": "investor is required"}), 400
    if not quarter:
        return jsonify({"error": "quarter is required"}), 400
    return None


# ── dropdowns ─────────────────────────────────────────────────────────────

@portfolio_snapshot_bp.route("/investors", methods=["GET"])
@login_required
def investors():
    """Investor codes eligible for a snapshot, with display names.

    Reuses Review Tracking's authoritative upstream-investor filter (the SQL
    CTE in ``get_investor_list``), which already excludes OP%/PPI% entities,
    sold deals and child properties — the same population the Reports "By
    Partner" selector uses. Falls back to the relationships-derived list if
    that helper is unavailable.
    """
    from flask_app.services.portfolio_snapshot_service import get_investor_name

    out = []
    try:
        # get_investor_list() returns plain investor-ID strings
        from flask_app.services.review_service import get_investor_list
        for code in (get_investor_list() or []):
            code = str(code).strip().upper()
            if code:
                out.append({"code": code, "name": get_investor_name(code)})
    except Exception:
        out = []

    if not out:
        data, err = _data_or_error()
        if err:
            return err
        rel = data.get("relationships_raw")
        codes = set()
        if rel is not None and not getattr(rel, "empty", True):
            col = next((c for c in rel.columns if c.lower() == "investorid"), None)
            if col:
                for v in rel[col].dropna().astype(str).str.strip().str.upper():
                    if v and not v.startswith("OP") and not v.startswith("PPI"):
                        codes.add(v)
        out = [{"code": c, "name": get_investor_name(c)}
               for c in sorted(codes)]

    out.sort(key=lambda r: (r["name"] or r["code"]).lower())
    return jsonify({"investors": out})


@portfolio_snapshot_bp.route("/quarters", methods=["GET"])
@login_required
def quarters():
    """Reportable quarters, newest first.

    Derived from the calendar rather than from a deal's data: the snapshot is a
    portfolio document, so a quarter is reportable once it has ended. This also
    sidesteps the known ``get_available_quarters()`` bug (no vcode filter, so
    one deal's quarter leaks onto every deal's dropdown).
    """
    from flask_app.services.portfolio_snapshot_service import _quarter_end

    today = pd.Timestamp.today().normalize()
    out = []
    year = today.year
    for y in range(year - 3, year + 1):
        for q in (1, 2, 3, 4):
            label = f"{y}-Q{q}"
            try:
                if pd.Timestamp(_quarter_end(label)) <= today:
                    out.append(label)
            except Exception:
                continue
    out.reverse()
    return jsonify({"quarters": out, "default": out[0] if out else None})


# ── foundation ────────────────────────────────────────────────────────────

@portfolio_snapshot_bp.route("/deals", methods=["GET"])
@login_required
def deals():
    """Step 1 deal resolution for one investor + quarter (diagnostic)."""
    investor, quarter = _args()
    err = _missing(investor, quarter)
    if err:
        return err
    data, err = _data_or_error()
    if err:
        return err
    try:
        return jsonify(safe_json(_resolve(investor, quarter, data)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── subtabs ───────────────────────────────────────────────────────────────

def _build_subtab(name: str, investor: str, quarter: str, data: dict,
                  resolved: dict) -> dict:
    op = _make_one_pager_provider(data)

    if name == "summary":
        from flask_app.services.portfolio_snapshot_summary import assemble_summary
        return assemble_summary(investor, quarter, resolved=resolved,
                                one_pager_provider=op)
    if name == "financial":
        from flask_app.services.portfolio_snapshot_financial import assemble_financial
        return assemble_financial(investor, quarter, resolved=resolved,
                                  one_pager_provider=op)
    if name == "operating":
        from flask_app.services.portfolio_snapshot_operating import assemble_operating
        return assemble_operating(investor, quarter, resolved=resolved,
                                  one_pager_provider=op)
    if name == "loan":
        from flask_app.services.portfolio_snapshot_loan import assemble_loan
        return assemble_loan(
            investor, quarter, resolved=resolved, one_pager_provider=op,
            loans=data.get("mri_loans_raw"), valuations=data.get("mri_val"),
            inv=data["inv"],
            quarterly_noi_provider=_make_quarterly_noi_provider(data),
        )
    raise ValueError(f"unknown subtab {name!r}")


@portfolio_snapshot_bp.route("/<subtab>", methods=["GET"])
@login_required
def subtab(subtab):
    """One subtab: summary | financial | operating | loan."""
    if subtab not in ("summary", "financial", "operating", "loan"):
        return jsonify({"error": f"unknown subtab {subtab!r}"}), 404
    investor, quarter = _args()
    err = _missing(investor, quarter)
    if err:
        return err
    data, err = _data_or_error()
    if err:
        return err
    try:
        resolved = _resolve(investor, quarter, data)
        return jsonify(safe_json(
            _build_subtab(subtab, investor, quarter, data, resolved)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@portfolio_snapshot_bp.route("/bundle", methods=["GET"])
@login_required
def bundle():
    """All four subtabs + review status in one request.

    A per-subtab failure is reported in ``errors`` rather than failing the whole
    page: three good subtabs are more useful than none, and the shell renders
    the error in place of that tab body.
    """
    investor, quarter = _args()
    err = _missing(investor, quarter)
    if err:
        return err
    data, err = _data_or_error()
    if err:
        return err

    review = _review_payload(investor, quarter)

    # An APPROVED report serves its frozen payload, not a fresh computation:
    # live MRI data moves (45th & Main went 100% -> 90% on 2026-08-24) and an
    # approved report must not move with it. Anything not yet approved computes
    # live so work in progress reflects current data. `source` says which.
    #
    # NOTE this is the deliberate divergence from the One Pager, which defaults
    # to live and puts the frozen copy behind a manual toggle. See the module
    # docstring in portfolio_snapshot_freeze.
    try:
        from flask_app.services.portfolio_snapshot_freeze import load_report
        report = load_report(investor, quarter, status=review.get("status"))
    except Exception as exc:
        return jsonify({"error": f"report assembly failed: {exc}"}), 500

    out: dict = {
        "investor_code": investor, "quarter": quarter,
        "subtabs": report.get("subtabs") or {},
        "errors": report.get("errors") or {},
        "resolution": report.get("resolution") or {},
        "source": report.get("source"),
        "source_note": report.get("source_note"),
        "approved_by": report.get("approved_by"),
        "approved_at": report.get("approved_at"),
        "data_version": report.get("data_version"),
        "frozen_elements": (report.get("elements")
                            if report.get("source") == "frozen" else None),
        "review": review,
    }

    # Step 8 guardrails. Advisory only — a finding never blanks a metric or
    # fails the page, so a bad deal cannot hide the whole report. Skipped on a
    # frozen payload: the guardrails audit a live computation, and re-auditing
    # what was already approved would surface findings nobody can now act on.
    if out["source"] == "frozen":
        out["guardrails"] = {
            "skipped": "frozen payload — audited when it was approved",
            "findings": [], "counts": {"error": 0, "warn": 0, "info": 0},
            "ok": None}
    else:
        try:
            from flask_app.services.portfolio_snapshot_guardrails import run_guardrails
            resolved = _resolve(investor, quarter, data)
            out["guardrails"] = run_guardrails(
                resolved, out["subtabs"],
                pe_cap_comments=_pe_cap_comments(data, quarter),
            )
        except Exception as exc:
            out["guardrails"] = {"error": str(exc), "findings": [], "ok": None}

    return jsonify(safe_json(out))


def _pe_cap_comments(data: dict, quarter: str) -> dict:
    """vcode -> pe_cap_comment, for the pref-equity cross-check.

    Read-only reuse of the One Pager's own comment store. Failure is non-fatal:
    the cross-check reports that it could not run rather than passing silently.
    """
    try:
        from one_pager import get_one_pager_comments
    except Exception:
        return {}
    out: dict = {}
    inv = data.get("inv")
    if inv is None or getattr(inv, "empty", True):
        return out
    col = next((c for c in inv.columns if c.lower() == "vcode"), None)
    if not col:
        return out
    for vcode in inv[col].dropna().astype(str).str.strip():
        try:
            row = get_one_pager_comments(vcode, quarter) or {}
        except Exception:
            continue
        txt = row.get("pe_cap_comment")
        if txt:
            out[vcode] = txt
    return out


# ── editable elements ─────────────────────────────────────────────────────

def _current_user() -> dict:
    """The caller, from ``g.current_user`` as ``login_required`` sets it."""
    from flask import g
    return getattr(g, "current_user", None) or {}


@portfolio_snapshot_bp.route("/elements", methods=["GET"])
@login_required
def elements():
    """Everything editable for one page, plus its review status."""
    investor, quarter = _args()
    err = _missing(investor, quarter)
    if err:
        return err
    from flask_app.services import portfolio_snapshot_persistence as P
    try:
        page = P.load_page(investor, quarter)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    page["review"] = _review_payload(investor, quarter)
    return jsonify(safe_json(page))


@portfolio_snapshot_bp.route("/comment", methods=["PUT"])
@login_required
def put_comment():
    """Save a narrative (scope='report') or per-deal comment (scope='deal')."""
    from flask_app.services import portfolio_snapshot_persistence as P
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    username = _current_user().get("username") or ""
    try:
        row = P.save_comment(
            investor, quarter,
            scope=(body.get("scope") or "report"),
            field=(body.get("field") or ""),
            comment_text=body.get("text") or "",
            scope_key=(body.get("scope_key") or ""),
            updated_by=username,
        )
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json({"saved": row}))


@portfolio_snapshot_bp.route("/value", methods=["PUT"])
@login_required
def put_value():
    """Save a manual number (Financial subtab Net ROE / ITD, per deal)."""
    from flask_app.services import portfolio_snapshot_persistence as P
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    username = _current_user().get("username") or ""
    raw = body.get("value")
    value = None
    if raw not in (None, ""):
        try:
            value = float(str(raw).replace(",", "").replace("%", "").replace("$", ""))
        except (TypeError, ValueError):
            return jsonify({"error": f"value {raw!r} is not a number"}), 400
    try:
        row = P.save_value(
            investor, quarter,
            deal_vcode=(body.get("vcode") or ""),
            field=(body.get("field") or ""),
            value=value, updated_by=username,
        )
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json({"saved": row}))


@portfolio_snapshot_bp.route("/footnote", methods=["POST"])
@login_required
def post_footnote():
    """Append a footnote to an anchor; it takes the next number."""
    from flask_app.services import portfolio_snapshot_persistence as P
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    username = _current_user().get("username") or ""
    try:
        P.add_footnote(investor, quarter, anchor=(body.get("anchor") or ""),
                       footnote_text=(body.get("text") or ""),
                       updated_by=username)
        rows = P.get_elements("footnote", investor, quarter)
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json({"footnotes": rows}))


@portfolio_snapshot_bp.route("/footnote/<int:footnote_id>", methods=["DELETE"])
@login_required
def delete_footnote(footnote_id):
    """Delete a footnote; the rest re-sequence so numbering stays contiguous."""
    from flask_app.services import portfolio_snapshot_persistence as P
    investor, quarter = _args()
    err = _missing(investor, quarter)
    if err:
        return err
    try:
        rows = P.remove_footnote(investor, quarter, footnote_id)
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json({"footnotes": rows}))


# ── approval pipeline ─────────────────────────────────────────────────────

#: Which review role may act at each step, mirroring the One Pager pipeline.
_STEP_ROLE = {0: "asset_manager", 1: "head_am", 2: "president",
              3: "cco", 4: "ceo"}


def _user_review_roles(user_id) -> list:
    if user_id is None:
        return []
    try:
        from flask_app.services.review_service import get_user_review_roles
        return list(get_user_review_roles(user_id) or [])
    except Exception:
        return []


def _review_payload(investor: str, quarter: str) -> dict:
    """Review status shaped for the shell's inline status strip.

    Deliberately NOT ``ReviewPanel``'s 17-field contract: Step 2 has no notes
    table, so a threaded-notes panel would render permanently empty and its
    note-required Return would have nowhere to persist. This returns the status
    plus the three permission flags the strip needs, and nothing it cannot back
    with real data.
    """
    from flask_app.services import portfolio_snapshot_persistence as P
    try:
        doc = P.document_status(investor, quarter)
        editable = P.is_editable(investor, quarter)
    except Exception as exc:
        return {"error": str(exc)}

    user = _current_user()
    user_id = user.get("id")
    roles = _user_review_roles(user_id)
    step = doc.get("current_step") or 0
    status = doc.get("status")
    acting_role = _STEP_ROLE.get(step)

    is_admin = _current_user().get("role") == "admin"

    def may(role):
        return bool(role) and (role in roles or is_admin)

    doc.update({
        "editable": bool(editable),
        "user_review_roles": roles,
        "acting_role": acting_role,
        "can_submit": (status in ("draft", "returned")
                       and doc.get("element_count", 0) > 0
                       and may("asset_manager")),
        "can_approve": (status not in ("draft", "returned", "approved")
                        and 1 <= step <= 4 and may(acting_role)),
        "can_return": (status not in ("draft", "returned", "approved")
                       and may(acting_role)),
    })
    return doc


def _transition(action: str):
    from flask_app.services import portfolio_snapshot_persistence as P
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    user = _current_user()
    user_id, username = user.get("id"), user.get("username") or ""
    roles = _user_review_roles(user_id)
    try:
        if action == "submit":
            P.submit_for_review(investor, quarter, user_id, username, roles)
        elif action == "approve":
            P.approve(investor, quarter, user_id, username, roles)
        elif action == "return":
            P.reject(investor, quarter, user_id, username,
                     note_text=(body.get("note") or ""), roles=roles)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json({"review": _review_payload(investor, quarter)}))


@portfolio_snapshot_bp.route("/submit", methods=["POST"])
@login_required
def submit():
    return _transition("submit")


@portfolio_snapshot_bp.route("/approve", methods=["POST"])
@login_required
def approve():
    return _transition("approve")


@portfolio_snapshot_bp.route("/return", methods=["POST"])
@login_required
def return_to_draft():
    return _transition("return")
