"""Expense coding endpoints -- accounting's side of expense reports (phase 3).

    GET  /api/expense-coding/lines                    every approved, unbatched line, pre-coded
    PUT  /api/expense-coding/lines/<lid>/<sid>        accounting's coding (sid 0 = the whole line)
    GET  /api/expense-coding/interco/<vcode>          the proposed owning entities for a deal
    GET  /api/expense-coding/settings                 entity currencies, recurring items
    PUT  /api/expense-coding/currency                 an entity's book currency
    POST /api/expense-coding/recurring                a standing reimbursement
    PUT  /api/expense-coding/recurring/<id>           change or retire one
    POST /api/expense-coding/batches                  preview (commit false) or generate
    GET  /api/expense-coding/batches                  every batch and whether the GL shows it
    GET  /api/expense-coding/batches/<id>/csv         the MRI GL upload
    POST /api/expense-coding/batches/<id>/void        release its reports

In the ACCOUNTING section. READS ARE ACCOUNTING_ROLES TOO, unlike the rest of
the section: the grid is every employee's spending, and an analyst opening
Accounting has no business reading their colleagues' expenses.
"""
import logging

from flask import Blueprint, Response, g, jsonify, request

from flask_app.auth.routes import ACCOUNTING_ROLES, login_required, roles_exactly
from flask_app.db import get_engine
from flask_app.serializers import safe_json
from flask_app.services import expense_coding as ec

logger = logging.getLogger(__name__)

expense_coding_bp = Blueprint("expense_coding", __name__, url_prefix="/api/expense-coding")


def _by() -> str:
    return (getattr(g, "current_user", None) or {}).get("username", "unknown")


def _run(name, fn, *a, **k):
    try:
        return jsonify(safe_json(fn(*a, **k)))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"expense coding {name} failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _body() -> dict:
    return request.get_json(silent=True) or {}


@expense_coding_bp.route("/lines", methods=["GET"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def get_lines():
    return _run("lines", lambda: {"rows": ec.coding_rows(get_engine())})


@expense_coding_bp.route("/lines/<int:line_id>/<int:split_id>", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_coding(line_id, split_id):
    return _run("save coding", ec.save_coding, get_engine(), line_id, split_id, _body(), _by())


@expense_coding_bp.route("/interco/<vcode>", methods=["GET"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def get_interco(vcode):
    return _run("interco", ec.propose_interco, get_engine(), vcode)


@expense_coding_bp.route("/settings", methods=["GET"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def get_settings():
    eng = get_engine()
    return _run("settings", lambda: {"currencies": ec.currencies(eng),
                                     "recurring": ec.recurring(eng)})


@expense_coding_bp.route("/currency", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_currency():
    b = _body()
    return _run("currency", ec.set_currency, get_engine(), b.get("entity_id"),
                b.get("currency"), _by())


@expense_coding_bp.route("/recurring", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def post_recurring():
    return _run("recurring", ec.save_recurring, get_engine(), _body(), _by())


@expense_coding_bp.route("/recurring/<int:item_id>", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_recurring(item_id):
    return _run("recurring", ec.save_recurring, get_engine(), _body(), _by(), item_id=item_id)


@expense_coding_bp.route("/batches", methods=["GET"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def get_batches():
    return _run("batches", lambda: {"batches": ec.batches(get_engine())})


@expense_coding_bp.route("/batches", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def post_batch():
    b = _body()
    try:
        out = ec.build_batch(get_engine(), b, _by(), commit=bool(b.get("commit")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"expense batch failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    return jsonify(safe_json(out)), (400 if b.get("commit") and out["errors"] else 200)


@expense_coding_bp.route("/batches/<batch_id>/csv", methods=["GET"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def get_batch_csv(batch_id):
    csv_text = ec.batch_csv(get_engine(), batch_id)
    if csv_text is None:
        return jsonify({"error": "No batch %s." % batch_id}), 404
    return Response(csv_text, mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=%s.csv" % batch_id})


@expense_coding_bp.route("/batches/<batch_id>/void", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def post_void(batch_id):
    return _run("void", ec.void_batch, get_engine(), batch_id, _by())
