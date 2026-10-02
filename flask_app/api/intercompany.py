"""Intercompany endpoints — the Due to/from PSC Manager reconciliation.

    GET  /api/intercompany/periods                what can be picked
    GET  /api/intercompany/reconciliation         ?period=YYYYMM&tolerance=1
    GET  /api/intercompany/reconciliation/excel   the same, as a workbook
    GET  /api/intercompany/lines                  ?period&entity&side  (the GL behind a figure)
    GET  /api/intercompany/settings               every entity's stored settings
    PUT  /api/intercompany/settings/<entity_id>   alternate account, cash accounts, currency
    PUT  /api/intercompany/notes                  a comment on one entity for one period
    PUT  /api/intercompany/pay                    can-afford adjustment, amount, pay-from account
    GET  /api/intercompany/batches                generated journal entries, posted or not
    POST /api/intercompany/batches                preview (commit false) or generate the JE
    GET  /api/intercompany/batches/<id>/csv       the MRI GL upload file
    POST /api/intercompany/batches/<id>/void      a batch that will not be uploaded

Reads are open to any signed-in user and writes are `ACCOUNTING_ROLES`, as in
the rest of the accounting section. The drilldown is a GET on purpose, so it is
a read by its verb and needs no exemption in `accounting_access_check`.
"""
import logging

from flask import Blueprint, Response, g, jsonify, request

from flask_app.auth.routes import ACCOUNTING_ROLES, login_required, roles_exactly
from flask_app.db import get_engine
from flask_app.serializers import safe_json
from flask_app.services import intercompany_service as ic

logger = logging.getLogger(__name__)

intercompany_bp = Blueprint("intercompany", __name__, url_prefix="/api/intercompany")


def _user() -> str:
    return (getattr(g, "current_user", None) or {}).get("username", "unknown")


def _fail(name, e):
    if isinstance(e, ValueError):
        return jsonify({"error": str(e)}), 400
    logger.error(f"intercompany {name} failed: {e}", exc_info=True)
    return jsonify({"error": str(e)}), 500


def _tolerance():
    raw = request.args.get("tolerance")
    try:
        return float(raw) if raw not in (None, "") else ic.DEFAULT_TOLERANCE
    except ValueError:
        raise ValueError("Tolerance %r is not a number." % raw)


@intercompany_bp.route("/periods", methods=["GET"])
@login_required
def get_periods():
    try:
        return jsonify(safe_json(ic.periods(get_engine())))
    except Exception as e:
        return _fail("periods", e)


@intercompany_bp.route("/reconciliation", methods=["GET"])
@login_required
def get_reconciliation():
    try:
        return jsonify(safe_json(ic.reconcile(
            get_engine(), request.args.get("period"), _tolerance())))
    except Exception as e:
        return _fail("reconciliation", e)


@intercompany_bp.route("/reconciliation/excel", methods=["GET"])
@login_required
def get_reconciliation_excel():
    try:
        res = ic.reconcile(get_engine(), request.args.get("period"), _tolerance())
        if not res.get("available"):
            return jsonify({"error": res.get("reason")}), 404
        return Response(
            ic.to_excel(res),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition":
                     f"attachment; filename=Due_to_Manager_Recon_{res['period']}.xlsx"})
    except Exception as e:
        return _fail("excel", e)


@intercompany_bp.route("/lines", methods=["GET"])
@login_required
def get_lines():
    try:
        return jsonify(safe_json(ic.lines(
            get_engine(), request.args.get("period"), request.args.get("entity", ""),
            request.args.get("side", "entity"))))
    except Exception as e:
        return _fail("lines", e)


@intercompany_bp.route("/settings", methods=["GET"])
@login_required
def get_settings():
    try:
        return jsonify(safe_json({"settings": list(ic.get_settings(get_engine()).values())}))
    except Exception as e:
        return _fail("settings", e)


@intercompany_bp.route("/settings/<entity_id>", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_settings(entity_id):
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(safe_json(ic.save_settings(get_engine(), entity_id, body, _user())))
    except Exception as e:
        return _fail("save settings", e)


@intercompany_bp.route("/pay", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_pay():
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(safe_json(ic.save_pay(
            get_engine(), body.get("period"), body.get("entity_id"), body, _user())))
    except Exception as e:
        return _fail("save pay", e)


@intercompany_bp.route("/batches", methods=["GET"])
@login_required
def get_batches():
    try:
        return jsonify(safe_json({"batches": ic.batches(get_engine())}))
    except Exception as e:
        return _fail("batches", e)


@intercompany_bp.route("/batches", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def post_batch():
    """`commit: false` previews; `commit: true` stores the batch and returns
    the file. Both are accounting-only: the preview names amounts to pay."""
    try:
        b = request.get_json(silent=True) or {}
        out = ic.build_batch(get_engine(), b.get("period"), b.get("entities") or [],
                             b.get("je_period"), b.get("entrdate"), _user(),
                             commit=bool(b.get("commit")))
        return jsonify(safe_json(out)), (400 if b.get("commit") and out["errors"] else 200)
    except Exception as e:
        return _fail("batch", e)


@intercompany_bp.route("/batches/<batch_id>/csv", methods=["GET"])
@login_required
def get_batch_csv(batch_id):
    # A read, open like every other read in the section: the file only restates
    # a batch the accounting team already generated.
    try:
        b = ic.batch_csv(get_engine(), batch_id)
        if not b:
            return jsonify({"error": "No batch %s." % batch_id}), 404
        return Response(b["csv"], mimetype="text/csv", headers={
            "Content-Disposition": f"attachment; filename={b['batch_id']}.csv"})
    except Exception as e:
        return _fail("batch csv", e)


@intercompany_bp.route("/batches/<batch_id>/void", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def post_void(batch_id):
    try:
        return jsonify(safe_json(ic.void_batch(get_engine(), batch_id, _user())))
    except Exception as e:
        return _fail("void", e)


@intercompany_bp.route("/notes", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_note():
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(safe_json(ic.save_note(
            get_engine(), body.get("period"), body.get("entity_id"),
            body.get("comment"), _user())))
    except Exception as e:
        return _fail("save note", e)
