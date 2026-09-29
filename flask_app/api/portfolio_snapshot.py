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
import json
import logging

import pandas as pd

from flask_app.auth.routes import login_required, roles_exactly
from flask_app.serializers import safe_json
from flask_app.services import data_service

logger = logging.getLogger(__name__)

portfolio_snapshot_bp = Blueprint("portfolio_snapshot", __name__)


#: Writes that stay legal on a frozen quarter, named one by one. Everything
#: else that writes is refused.
#:
#: DENY BY DEFAULT, and that direction is the point. An allowlist of endpoints
#: that may NOT write would have to be extended every time somebody adds an
#: editable field, and the day it is forgotten a sent quarter becomes editable
#: with nothing saying so. This way a new element endpoint is locked the moment
#: it exists and has to be named here to be let through.
_FROZEN_WRITE_ALLOWED = frozenset({
    "portfolio_snapshot.post_freeze_all_snapshots",    # the two batch buttons;
    "portfolio_snapshot.post_freeze_all_one_pagers",   # each refuses an
                                                       # already-frozen part on
                                                       # its own terms
    "portfolio_snapshot.post_refreeze",    # an admin correction
    "portfolio_snapshot.post_unfreeze",    # returning it to live
    "portfolio_snapshot.submit",           # the review chain is a separate
    "portfolio_snapshot.approve",          # authority and is not a typed
    "portfolio_snapshot.return_to_draft",  # field; freezing does not gate it
    "portfolio_snapshot.reopen",
})


def frozen_write_refusal(investor: str, quarter: str, fr: dict):
    """The 409 body for a write into a frozen quarter, or None.

    Shared with the One Pager comment path in ``financials.py``, which is on a
    DIFFERENT blueprint and so is never reached by the ``before_request``
    below. One refusal, one wording, both doors.
    """
    if not fr:
        return None
    who = (fr.get("frozen_by") or fr.get("approved_by") or "unknown")
    when = str(fr.get("frozen_at") or fr.get("approved_at") or "")[:10]
    parts = fr.get("frozen_parts") or []
    what = {"snapshot": "the Snapshot", "one_pagers": "the One Pagers"}
    named = " and ".join(what.get(p, p) for p in parts) or "this quarter"
    return {
        "error": f"{investor} {quarter} was frozen as sent by {who}"
                 + (f" on {when}" if when else "")
                 + f" ({named}). Frozen quarters are read-only. An admin can "
                   "Re-freeze or Unfreeze it if it genuinely has to change.",
        "frozen": True,
        "frozen_parts": parts,
    }


@portfolio_snapshot_bp.before_request
def _lock_frozen_quarters():
    """Refuse edits to a quarter that has been frozen.

    A frozen quarter is the record of what an investor was sent, so the typed
    fields on it — Net ROE, ITD, comments, footnotes — are no longer editable.

    ANY frozen part locks the page. The two halves are frozen independently but
    the editable content is shared: a comment reaches the Snapshot payload AND
    the One Pagers, so a lock that only fired when both were frozen would let
    an edit drift away from whichever half was already sent.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if (request.endpoint or "") in _FROZEN_WRITE_ALLOWED:
        return None
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or request.args.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or request.args.get("quarter") or "").strip()
    if not investor or not quarter:
        return None
    try:
        from flask_app.services.portfolio_snapshot_freeze import get_frozen
        fr = get_frozen(investor, quarter)
    except Exception:
        return None                       # never block a write on a lookup fault
    refusal = frozen_write_refusal(investor, quarter, fr)
    if refusal is None:
        return None
    return jsonify(refusal), 409


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


# ── request helpers ───────────────────────────────────────────────────────

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

def _investor_list(data=None) -> list:
    """Investor codes eligible for a snapshot, with display names.

    Reuses Review Tracking's authoritative upstream-investor filter (the SQL
    CTE in ``get_investor_list``), which already excludes OP%/PPI% entities,
    sold deals and child properties — the same population the Reports "By
    Partner" selector uses. Falls back to the relationships-derived list if
    that helper is unavailable.

    Extracted so the freeze-all batches iterate the SAME population the
    dropdown offers. Two lists would mean a batch that silently skips an
    investor the screen shows, or freezes one it does not.
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

    if not out and data is not None:
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
    return out


@portfolio_snapshot_bp.route("/investors", methods=["GET"])
@login_required
def investors():
    """Investor codes eligible for a snapshot, with display names."""
    out = _investor_list()
    if not out:
        data, err = _data_or_error()
        if err:
            return err
        out = _investor_list(data)
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
    """One subtab, for the per-subtab debug endpoint.

    Delegates to the freeze module, which owns the single assembly path shared
    with /bundle and the freeze itself. Two implementations would let a frozen
    payload differ from live even with unchanged data.
    """
    from flask_app.services.portfolio_snapshot_freeze import build_subtab
    return build_subtab(name, investor, quarter, data, resolved)


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

    # A FROZEN quarter serves its stored copy, not a fresh computation: live
    # MRI data moves (45th & Main went 100% -> 90% on 2026-08-24) and a report
    # that has been sent must not move with it. Only the current, unsent
    # quarter computes live. `source` says which.
    #
    # The One Pager now follows the SAME rule rather than defaulting to live —
    # see financials.one_pager. The two halves of one report diverging on which
    # copy they serve was the older behaviour and is gone.
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
        # What the Frozen / Live indicator renders, and what locks the inputs.
        "frozen": report.get("source") == "frozen",
        "read_only": bool(report.get("read_only")),
        "frozen_by": report.get("frozen_by"),
        "frozen_at": report.get("frozen_at"),
        "frozen_reason": report.get("frozen_reason"),
        "frozen_version": report.get("version"),
        "source_manifest": report.get("source_manifest"),
        "roster": report.get("roster"),
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
                pe_cap_comments=_pe_cap_comments(
                    quarter, _scope_vcodes(resolved)),
            )
        except Exception as exc:
            out["guardrails"] = {"error": str(exc), "findings": [], "ok": None}

    return jsonify(safe_json(out))


def _pe_cap_comments(quarter: str, vcodes) -> dict:
    """vcode -> pe_cap_comment, for the pref-equity cross-check.

    Scoped to the deals in this investor's report. Single batch query instead
    of one DB round-trip per deal (was 32 queries for TIAA).

    Read-only reuse of the One Pager's own comment store. Failure is non-fatal:
    the cross-check reports that it could not run rather than passing silently.
    Only non-empty comments are returned.
    """
    if not vcodes:
        return {}
    try:
        from flask_app.db import get_engine
        from sqlalchemy import text
        engine = get_engine()
        placeholders = ", ".join(f":v{i}" for i in range(len(vcodes)))
        params = {f"v{i}": str(vc) for i, vc in enumerate(vcodes)}
        params["q"] = str(quarter)
        with engine.connect() as conn:
            rows = conn.execute(text(
                f"SELECT vcode, pe_cap_comment FROM one_pager_comments "
                f"WHERE vcode IN ({placeholders}) AND reporting_period = :q"
            ), params).mappings().fetchall()
        return {r["vcode"]: r["pe_cap_comment"] for r in rows
                if r.get("pe_cap_comment")}
    except Exception:
        return {}


def _scope_vcodes(resolved: dict) -> list:
    """Every deal in the report, grouped plus ownership-flagged."""
    out = [e["vcode"] for items in (resolved.get("groups") or {}).values()
           for e in items]
    out += [f["vcode"] for f in (resolved.get("flagged") or [])]
    return out


# ── freeze as sent ────────────────────────────────────────────────────────
#
# DELIBERATELY SEPARATE FROM THE APPROVAL CHAIN. Freezing records that a
# quarter was SENT; approving records a decision somebody made. Rolling them
# together would have the button write an approval nobody gave, which is the
# confusion this split exists to prevent — so nothing here touches
# `portfolio_snapshot_documents` or its steps.

@portfolio_snapshot_bp.route("/freeze-status", methods=["GET"])
@login_required
def freeze_status():
    """Whether this investor+quarter is frozen, by whom, when, and from what."""
    investor, quarter = _args()
    err = _missing(investor, quarter)
    if err:
        return err
    from flask_app.services import portfolio_snapshot_freeze as FZ
    try:
        fr = FZ.get_frozen(investor, quarter)
        out = {
            "investor_code": investor, "quarter": quarter,
            "frozen": bool(fr),
            "history": FZ.frozen_history(investor, quarter),
        }
        if fr:
            out.update({
                "frozen_by": fr.get("frozen_by") or fr.get("approved_by"),
                "frozen_at": fr.get("frozen_at") or fr.get("approved_at"),
                "frozen_reason": fr.get("frozen_reason"),
                "version": fr.get("version"),
                "data_version": fr.get("data_version"),
                "source_manifest": fr.get("source_manifest"),
                "roster": fr.get("roster") or [],
                "one_pager_count": len(fr.get("one_pagers") or {}),
            })
        return jsonify(safe_json(out))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@portfolio_snapshot_bp.route("/quarter-status", methods=["GET"])
@login_required
def quarter_status():
    """How much of a quarter is frozen, per part, across every investor.

    THE QUESTION THE BUTTON NEEDS ANSWERED. "Frozen" is not a property of a
    quarter but of (investor, part), so a screen offering an all-investors
    freeze has to say which of the three states it is in — none, some, all —
    or the reader cannot tell a fresh quarter from one that is nearly done.

    Counts come from ONE read (``quarter_part_state``) and the population from
    ``_investor_list`` — the SAME list the batch iterates, so "12 of 127" is
    measured against the set the button will actually touch.

    ``one_pagers`` is only computed when asked for (``count_one_pagers=1``): it
    resolves every investor's deals, which is real work, and the confirmation
    dialog is the only caller that needs it. It is ``None``, never ``0``, when
    it cannot be computed — a zero here would read as "this quarter has no One
    Pagers", which is a different and alarming statement.
    """
    quarter = (request.args.get("quarter") or "").strip()
    if not quarter:
        return jsonify({"error": "quarter is required"}), 400
    from flask_app.services import portfolio_snapshot_freeze as FZ

    codes = [r["code"] for r in _investor_list()]
    if not codes:
        data, err = _data_or_error()
        if err:
            return err
        codes = [r["code"] for r in _investor_list(data)]
    total = len(codes)
    known = set(codes)

    state = FZ.quarter_part_state(quarter)
    parts = {}
    for part in FZ.PARTS:
        # Intersected with the CURRENT population on purpose: a frozen investor
        # no longer on the list must not make the count exceed the total, which
        # would render as "130 of 127" and look like a bug in the screen.
        frozen = sorted(set(state.get(part) or []) & known)
        parts[part] = {
            "frozen": len(frozen),
            "total": total,
            "remaining": max(0, total - len(frozen)),
            "investors": frozen,
            "state": ("all" if total and len(frozen) >= total
                      else "partly" if frozen else "none"),
        }

    from flask_app.services.freeze_gate import (
        freeze_enabled, FREEZE_DISABLED_MESSAGE)
    enabled = freeze_enabled()
    out = {
        "quarter": quarter,
        "investors": total,
        "parts": parts,
        "one_pagers_stored": state.get("one_pagers_stored"),
        "read_error": state.get("error"),
        # Investors counted above whose halves were INFERRED from a row written
        # before the per-part columns existed, rather than read off it. Carried
        # so the screen can qualify the count rather than present it as measured.
        "legacy_investors": sorted(
            set(state.get("legacy_investors") or []) & known),
        # Carried on the read the panel already makes, so the screen can say
        # WHY the button is unavailable instead of rendering a dead control.
        # The server refuses regardless; this only decides the wording.
        "freeze_enabled": enabled,
        "freeze_disabled_reason": None if enabled else FREEZE_DISABLED_MESSAGE,
    }

    if (request.args.get("count_one_pagers") or "").strip() in ("1", "true", "yes"):
        out["one_pagers"] = _count_one_pagers(codes, quarter)
    return jsonify(safe_json(out))


def _count_one_pagers(codes: list, quarter: str):
    """How many One Pagers an all-investors freeze would build.

    ``None`` when it cannot be worked out — never 0. One investor failing to
    resolve does not void the number, but it does make it a floor rather than a
    total, so the caller is told how many were counted.
    """
    data, err = _data_or_error()
    if err:
        return None
    pages = 0
    counted = 0
    for code in codes:
        try:
            pages += len(_scope_vcodes(_resolve(code, quarter, data)))
            counted += 1
        except Exception:                                     # noqa: BLE001
            logger.exception("one-pager count failed for %s %s", code, quarter)
    if not counted:
        return None
    return {"pages": pages, "investors_counted": counted,
            "investors": len(codes), "complete": counted == len(codes)}


def _freeze_disabled_response():
    """503 + the one sentence, or None when freezing is switched on.

    The CORE already refuses (``freeze_part`` raises ``FreezeDisabled``), so
    this is not the guarantee — it is the manners. Without it the refusal
    reaches the screen as a 500 with an exception string, which reads as a bug
    rather than a deliberate switch. 503 because the state is temporary; 403
    would say the caller lacks permission, which is a different and wrong
    statement about an admin.
    """
    from flask_app.services.freeze_gate import (
        freeze_enabled, FREEZE_DISABLED_MESSAGE)
    if freeze_enabled():
        return None
    return jsonify({"error": FREEZE_DISABLED_MESSAGE,
                    "freeze_disabled": True}), 503


def _freeze_all(part: str):
    """START a freeze of one PART of a quarter for every investor.

    RETURNS IMMEDIATELY WITH A JOB ID. It used to do the whole batch inside this
    request, which on Sep 29 2026 took the app down for ~35 minutes: one
    gunicorn worker, one request, everything else queued behind it. The work now
    runs on a background thread and the screen polls `/freeze-job/<id>`.

    The population, the skip-if-frozen rule and the per-investor isolation are
    unchanged — they moved to `freeze_batch._run`, which calls the same
    `freeze_part` core.
    """
    off = _freeze_disabled_response()
    if off:
        return off
    body = request.get_json(silent=True) or {}
    quarter = (body.get("quarter") or "").strip()
    if not quarter:
        return jsonify({"error": "quarter is required"}), 400

    # The SAME population the dropdown offers — see _investor_list.
    codes = [r["code"] for r in _investor_list()]
    if not codes:
        data, err = _data_or_error()
        if err:
            return err
        codes = [r["code"] for r in _investor_list(data)]
    if not codes:
        return jsonify({"error": "No investors found to freeze."}), 500
    only = {c.strip().upper() for c in (body.get("investors") or []) if c}
    if only:
        codes = [c for c in codes if c in only]

    from flask_app.services import freeze_batch as FB
    from flask import current_app
    job, err = FB.start_job(current_app._get_current_object(), quarter, part,
                            codes, _current_user().get("username") or "unknown")
    if err:
        # 409, not 500: the request is well-formed and the state is temporary.
        return jsonify({"error": err, "job_running": True}), 409
    return jsonify(safe_json(job)), 202


@portfolio_snapshot_bp.route("/freeze-job/<int:job_id>", methods=["GET"])
@login_required
def freeze_job(job_id):
    """Progress for one job: investors done / total, and the failures so far."""
    from flask_app.services import freeze_batch as FB
    job = FB.get_job(job_id)
    if not job:
        return jsonify({"error": f"no freeze job {job_id}"}), 404
    return jsonify(safe_json(job))


@portfolio_snapshot_bp.route("/freeze-job/active", methods=["GET"])
@login_required
def freeze_job_active():
    """The running job, or null. Lets a page reopened mid-run pick the bar up."""
    from flask_app.services import freeze_batch as FB
    return jsonify(safe_json({"job": FB.active_job()}))


@portfolio_snapshot_bp.route("/freeze-all/snapshots", methods=["POST"])
@login_required
@roles_exactly("admin")
def post_freeze_all_snapshots():
    """Freeze the SNAPSHOT half of a quarter for every investor."""
    from flask_app.services import portfolio_snapshot_freeze as FZ
    return _freeze_all(FZ.PART_SNAPSHOT)


@portfolio_snapshot_bp.route("/freeze-all/one-pagers", methods=["POST"])
@login_required
@roles_exactly("admin")
def post_freeze_all_one_pagers():
    """Freeze the ONE PAGER half of a quarter for every investor.

    Keyed per investor on purpose: the same deal and quarter can carry
    different published figures on two investors' reports, so there is no
    investor-independent One Pager to freeze.
    """
    from flask_app.services import portfolio_snapshot_freeze as FZ
    return _freeze_all(FZ.PART_ONE_PAGERS)



@portfolio_snapshot_bp.route("/refreeze", methods=["POST"])
@login_required
@roles_exactly("admin")
def post_refreeze():
    """Replace a frozen quarter. Admin only, reason required, history kept.

    GATED TOO: a re-freeze writes a new frozen copy, so it is a freeze. Unfreeze
    is not, and stays open — with freezing off, undoing one must remain possible
    or a mistake made before the switch could not be corrected.
    """
    off = _freeze_disabled_response()
    if off:
        return off
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    reason = (body.get("reason") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    from flask_app.services import portfolio_snapshot_freeze as FZ
    user = _current_user()
    try:
        receipt = FZ.refreeze(investor, quarter,
                              user.get("username") or "unknown", reason)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Re-freeze failed, the stored copy is "
                                 f"unchanged: {exc}"}), 500
    return jsonify(safe_json({**receipt, "frozen": True}))


@portfolio_snapshot_bp.route("/unfreeze", methods=["POST"])
@login_required
@roles_exactly("admin")
def post_unfreeze():
    """Return a quarter to live. Admin only, reason required, copy archived."""
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    reason = (body.get("reason") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    from flask_app.services import portfolio_snapshot_freeze as FZ
    user = _current_user()
    try:
        receipt = FZ.unfreeze(investor, quarter,
                              user.get("username") or "unknown", reason)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Unfreeze failed: {exc}"}), 500
    return jsonify(safe_json({**receipt, "frozen": False}))


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
    """Save a manual number, per deal.

    Financial subtab: Net ROE / ITD. Loan subtab: the typed LTV / YTD DSCR /
    Debt Yield cells. One endpoint because the row is keyed by field, not by
    subtab — see VALUE_FIELDS in portfolio_snapshot_persistence.
    """
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
        # Accept the cell back in the form it is DISPLAYED: "$5.87M", "4.4%",
        # "1,250". The number is stored in the unit its column shows — ITD in
        # millions, Net ROE in percentage points — so stripping the decoration
        # is all that is needed and nothing is rescaled. A trailing "M" is
        # stripped too, so a figure copied off the page (or out of the print
        # PDF) pastes straight back in instead of failing to parse.
        txt = str(raw).strip().replace(",", "").replace("%", "").replace("$", "")
        if txt[-1:] in ("M", "m"):
            txt = txt[:-1].strip()
        try:
            value = float(txt)
        except (TypeError, ValueError):
            return jsonify({"error": f"value {raw!r} is not a number"}), 400
    vcode = (body.get("vcode") or "").strip()
    field = (body.get("field") or "").strip()
    try:
        row = P.save_value(
            investor, quarter,
            deal_vcode=vcode, field=field,
            value=value, updated_by=username,
        )
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # The display twin and source string the assembly would have produced for
    # this cell, so the UI can patch the one row it changed instead of
    # refetching /bundle. Without these the shell refetched the whole payload
    # on every entry purely to read them back, rebuilding all four subtabs and
    # blanking the page behind "Building snapshot…".
    #
    # Which assembly owns the rule depends on the field, and the FIELD is the
    # only thing that decides — the request never says which subtab it came
    # from. A Loan ratio formats as "69.0%" / "1.9x" and reports a different
    # source (entered vs still pre-filled), so routing it through the Financial
    # helper would have handed back a bare "69.0" and called it Net ROE's
    # source string.
    from flask_app.services.portfolio_snapshot_loan import (
        MANUAL_RATIO_FIELDS, MANUAL_SOURCE_ENTERED, format_manual_ratio,
    )
    if field in MANUAL_RATIO_FIELDS:
        display = format_manual_ratio(field, value)
        source = MANUAL_SOURCE_ENTERED
    else:
        from flask_app.services.portfolio_snapshot_financial import (
            manual_display, manual_na_cells, MANUAL_SOURCE,
        )
        display = manual_display(field, value, manual_na_cells(vcode))
        source = MANUAL_SOURCE
    return jsonify(safe_json({
        "saved": row,
        "vcode": vcode,
        "field": field,
        "value": value,
        "display": display,
        "source": source,
    }))


def _footnote_payload(db_rows) -> dict:
    """What add/remove hand back: the SAME composed list ``/bundle`` sends.

    The shell splices this straight into ``subtabs.financial.footnotes``, so if
    this returned the raw persistence rows the page would lose the standing
    notes, the single numbering and every scope resolution until the next full
    rebuild — the markers on the headers and property names would then point at
    numbers that no longer exist. One composer, both paths.
    """
    from flask_app.services.portfolio_snapshot_financial import (
        compose_footnotes, footnote_marks, standing_removed,
    )
    composed = compose_footnotes(db_rows)
    return {"footnotes": composed,
            "footnote_marks": footnote_marks(composed),
            "standing_removed": standing_removed(db_rows)}


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
    return jsonify(safe_json(_footnote_payload(rows)))


@portfolio_snapshot_bp.route("/footnote/<int:footnote_id>", methods=["PUT"])
@login_required
def put_footnote(footnote_id):
    """Edit one analyst-entered footnote's text. Its number does not move."""
    from flask_app.services import portfolio_snapshot_persistence as P
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    username = _current_user().get("username") or ""
    try:
        rows = P.update_footnote(investor, quarter, footnote_id,
                                 footnote_text=(body.get("text") or ""),
                                 updated_by=username)
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json(_footnote_payload(rows)))


# ── standing footnotes: per-quarter edit / delete / restore ───────────────
#
# A standing footnote lives in STANDING_FOOTNOTES, not in the table, so it has
# no id to address. These three act on its KEY and record the change as a row
# under a reserved anchor, scoped to this investor and quarter. See
# STANDING_EDIT_PREFIX in portfolio_snapshot_financial for why.

def _standing_key_or_400(key: str):
    from flask_app.services.portfolio_snapshot_financial import STANDING_KEYS
    k = (key or "").strip()
    if k not in STANDING_KEYS:
        return None, (jsonify({"error": f"unknown standing footnote {k!r}",
                               "known": list(STANDING_KEYS)}), 404)
    return k, None


@portfolio_snapshot_bp.route("/footnote/standing/<key>", methods=["PUT"])
@login_required
def put_standing_footnote(key):
    """Reword one standing footnote on THIS quarter's page only."""
    from flask_app.services import portfolio_snapshot_persistence as P
    from flask_app.services.portfolio_snapshot_financial import (
        standing_edit_anchor, standing_delete_anchor,
    )
    k, bad = _standing_key_or_400(key)
    if bad:
        return bad
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    username = _current_user().get("username") or ""
    try:
        # Rewording a note that was deleted brings it back — the tombstone and
        # an edit cannot both stand, and the analyst just said what it should
        # say, which is not a request to keep it hidden.
        P.remove_footnotes_by_anchor(investor, quarter,
                                     [standing_delete_anchor(k)])
        rows = P.upsert_footnote_by_anchor(
            investor, quarter, anchor=standing_edit_anchor(k),
            footnote_text=(body.get("text") or ""), updated_by=username)
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json(_footnote_payload(rows)))


@portfolio_snapshot_bp.route("/footnote/standing/<key>", methods=["DELETE"])
@login_required
def delete_standing_footnote(key):
    """Take one standing footnote off THIS quarter's page. Reversible."""
    from flask_app.services import portfolio_snapshot_persistence as P
    from flask_app.services.portfolio_snapshot_financial import (
        standing_delete_anchor,
    )
    k, bad = _standing_key_or_400(key)
    if bad:
        return bad
    investor, quarter = _args()
    err = _missing(investor, quarter)
    if err:
        return err
    username = _current_user().get("username") or ""
    try:
        rows = P.upsert_footnote_by_anchor(
            investor, quarter, anchor=standing_delete_anchor(k),
            footnote_text="", updated_by=username)
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json(_footnote_payload(rows)))


@portfolio_snapshot_bp.route("/footnote/standing/<key>/restore",
                             methods=["POST"])
@login_required
def restore_standing_footnote(key):
    """Drop this quarter's edit and tombstone, restoring the default text."""
    from flask_app.services import portfolio_snapshot_persistence as P
    from flask_app.services.portfolio_snapshot_financial import (
        standing_edit_anchor, standing_delete_anchor,
    )
    k, bad = _standing_key_or_400(key)
    if bad:
        return bad
    body = request.get_json(silent=True) or {}
    investor = (body.get("investor") or "").strip().upper()
    quarter = (body.get("quarter") or "").strip()
    err = _missing(investor, quarter)
    if err:
        return err
    try:
        rows = P.remove_footnotes_by_anchor(
            investor, quarter,
            [standing_edit_anchor(k), standing_delete_anchor(k)])
    except P.NotEditable as exc:
        return jsonify({"error": str(exc), "locked": True}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(safe_json(_footnote_payload(rows)))


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
    return jsonify(safe_json(_footnote_payload(rows)))


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
        # Only an approved page can be reopened, and only by a REOPEN_ROLES
        # holder. Distinct from can_return, which applies mid-review.
        "can_reopen": (status == "approved"
                       and any(may(r) for r in P.REOPEN_ROLES)),
        "reopen_roles": list(P.REOPEN_ROLES),
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
        elif action == "reopen":
            # Unwinds a COMPLETED approval, which `return` cannot do — at the
            # approved step the pipeline's role is None. Separate action because
            # it carries different authority; see persistence.REOPEN_ROLES.
            P.reopen(investor, quarter, user_id, username,
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


@portfolio_snapshot_bp.route("/reopen", methods=["POST"])
@login_required
def reopen():
    """Reopen an APPROVED page so it can be corrected.

    Note required. The frozen payload is left in place — it stops being served
    the moment the status is no longer 'approved', and the next approval
    overwrites it.
    """
    return _transition("reopen")
