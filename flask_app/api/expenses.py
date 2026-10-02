"""Expense report endpoints -- phase 1.

    GET    /api/expenses/me                          my name, my approver, where my report goes
    GET    /api/expenses/options                     categories, purposes, deals, mileage rates
    GET    /api/expenses/reports?scope=              mine | to_approve | all (all I may read)
    POST   /api/expenses/reports                     open a report for a period
    GET    /api/expenses/reports/<id>                one report, its lines, history, permissions
    PUT    /api/expenses/reports/<id>                period / title (owner, draft or returned)
    DELETE /api/expenses/reports/<id>                a draft never submitted
    POST   /api/expenses/reports/<id>/lines          add a line
    PUT    /api/expenses/reports/<id>/lines/<lid>    change a line
    DELETE /api/expenses/reports/<id>/lines/<lid>    remove a line
    POST   /api/expenses/reports/<id>/submit         to the approver
    POST   /api/expenses/reports/<id>/recall         back to draft, before a decision
    POST   /api/expenses/reports/<id>/decide         {action: approve | return, note}
    POST   /api/expenses/reports/<id>/copy-recurring  bring forward last report's recurring lines
    POST   /api/expenses/reports/<id>/accounting-return  accounting sends an approved report back
    POST   /api/expenses/reports/<id>/receipts       upload files (multipart, field "files")
    GET    /api/expenses/reports/<id>/receipts/<rid>/file   the image or PDF, to show
    POST   /api/expenses/reports/<id>/receipts/<rid>/extract  read it; propose its lines
    DELETE /api/expenses/reports/<id>/receipts/<rid>        remove a file
    GET    /api/expenses/employees                   everyone's name, approver and route
    PUT    /api/expenses/employees/<user_id>         name on reports, approver (admin)
    PUT    /api/expenses/mileage-rates               a rate and the date it takes effect

The whole prefix is the Expenses SECTION (`flask_app/auth/sections.py`). Inside
it, who may read or change a REPORT is decided per record by
`expense_service` -- the owner, whoever may decide it, and accounting once it is
approved. A report you may not read is a 404, never a 403, so an id reveals
nothing. Approvers are the admin's to set (Jim, Oct 2 2026: "the Admin will
ensure the employees have the right level of access"); the mileage rate is
accounting's.
"""
import logging

from flask import Blueprint, Response, g, jsonify, request

from flask_app.auth.routes import ACCOUNTING_ROLES, login_required, roles_exactly
from flask_app.db import get_engine
from flask_app.serializers import safe_json
from flask_app.services import expense_service as ex
from flask_app.services import expense_receipts as rc

logger = logging.getLogger(__name__)

expenses_bp = Blueprint("expenses", __name__, url_prefix="/api/expenses")

#: Who may read the whole employee list: the admin who sets it, and accounting,
#: who will need everyone's name on the journal entry.
EMPLOYEE_LIST_ROLES = ACCOUNTING_ROLES


def _actor() -> dict:
    return getattr(g, "current_user", None) or {}


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
        logger.error(f"expenses {name} failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _body() -> dict:
    return request.get_json(silent=True) or {}


@expenses_bp.route("/me", methods=["GET"])
@login_required
def get_me():
    def me():
        eng = get_engine()
        a = _actor()
        mine = next((e for e in ex.employees(eng) if e["user_id"] == a["id"]), None)
        return {"employee": mine, "route": ex.route_for(eng, a["id"])}
    return _run("me", me)


@expenses_bp.route("/options", methods=["GET"])
@login_required
def get_options():
    return _run("options", ex.options, get_engine())


@expenses_bp.route("/reports", methods=["GET"])
@login_required
def get_reports():
    scope = request.args.get("scope", "mine")
    if scope not in ("mine", "to_approve", "all"):
        return jsonify({"error": "scope is mine, to_approve or all"}), 400
    return _run("list", lambda: {"reports": ex.list_reports(get_engine(), _actor(), scope)})


@expenses_bp.route("/reports", methods=["POST"])
@login_required
def post_report():
    return _run("create", ex.create_report, get_engine(), _actor(), _body())


@expenses_bp.route("/reports/<int:report_id>", methods=["GET"])
@login_required
def get_report(report_id):
    return _run("get", ex.get_report, get_engine(), _actor(), report_id)


@expenses_bp.route("/reports/<int:report_id>", methods=["PUT"])
@login_required
def put_report(report_id):
    return _run("update", ex.update_report, get_engine(), _actor(), report_id, _body())


@expenses_bp.route("/reports/<int:report_id>", methods=["DELETE"])
@login_required
def delete_report(report_id):
    return _run("delete", ex.delete_report, get_engine(), _actor(), report_id)


@expenses_bp.route("/reports/<int:report_id>/lines", methods=["POST"])
@login_required
def post_line(report_id):
    return _run("add line", ex.save_line, get_engine(), _actor(), report_id, _body())


@expenses_bp.route("/reports/<int:report_id>/lines/<int:line_id>", methods=["PUT"])
@login_required
def put_line(report_id, line_id):
    return _run("save line", ex.save_line, get_engine(), _actor(), report_id, _body(),
                line_id=line_id)


@expenses_bp.route("/reports/<int:report_id>/lines/<int:line_id>", methods=["DELETE"])
@login_required
def delete_line(report_id, line_id):
    return _run("delete line", ex.delete_line, get_engine(), _actor(), report_id, line_id)


@expenses_bp.route("/reports/<int:report_id>/submit", methods=["POST"])
@login_required
def post_submit(report_id):
    return _run("submit", ex.submit, get_engine(), _actor(), report_id)


@expenses_bp.route("/reports/<int:report_id>/recall", methods=["POST"])
@login_required
def post_recall(report_id):
    return _run("recall", ex.recall, get_engine(), _actor(), report_id)


@expenses_bp.route("/reports/<int:report_id>/decide", methods=["POST"])
@login_required
def post_decide(report_id):
    b = _body()
    return _run("decide", ex.decide, get_engine(), _actor(), report_id,
                b.get("action"), b.get("note"))


@expenses_bp.route("/reports/<int:report_id>/receipts", methods=["POST"])
@login_required
def post_receipts(report_id):
    files = [(f.filename, f.read()) for f in request.files.getlist("files")]
    if not files:
        return jsonify({"error": "No files were sent."}), 400
    return _run("upload receipts", rc.upload, get_engine(), _actor(), report_id, files)


@expenses_bp.route("/reports/<int:report_id>/receipts/<int:receipt_id>/file", methods=["GET"])
@login_required
def get_receipt_file(report_id, receipt_id):
    # Readable by whoever may read the report -- the owner, the approver and,
    # once approved, accounting -- and by nobody else (404, like the report).
    try:
        data, ctype, name = rc.receipt_file(get_engine(), _actor(), report_id, receipt_id)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"expenses receipt file failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    safe = name.replace('"', "")
    return Response(data, mimetype=ctype, headers={
        "Content-Disposition": 'inline; filename="%s"' % safe,
        "Cache-Control": "private, no-store"})


@expenses_bp.route("/reports/<int:report_id>/receipts/<int:receipt_id>/extract",
                   methods=["POST"])
@login_required
def post_extract(report_id, receipt_id):
    return _run("read receipt", rc.extract, get_engine(), _actor(), report_id, receipt_id)


@expenses_bp.route("/reports/<int:report_id>/receipts/<int:receipt_id>", methods=["DELETE"])
@login_required
def delete_receipt(report_id, receipt_id):
    return _run("delete receipt", rc.delete_receipt, get_engine(), _actor(), report_id,
                receipt_id)


@expenses_bp.route("/reports/<int:report_id>/copy-recurring", methods=["POST"])
@login_required
def post_copy_recurring(report_id):
    return _run("copy recurring", ex.copy_recurring, get_engine(), _actor(), report_id)


@expenses_bp.route("/reports/<int:report_id>/accounting-return", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def post_accounting_return(report_id):
    return _run("accounting return", ex.accounting_return, get_engine(), _actor(), report_id,
                _body().get("note"))


@expenses_bp.route("/employees", methods=["GET"])
@login_required
@roles_exactly(*EMPLOYEE_LIST_ROLES)
def get_employees():
    return _run("employees", lambda: {"employees": ex.employees(get_engine())})


@expenses_bp.route("/employees/<int:user_id>", methods=["PUT"])
@login_required
@roles_exactly("admin")
def put_employee(user_id):
    return _run("save employee", ex.save_employee, get_engine(), user_id, _body(),
                _actor().get("username", "unknown"))


@expenses_bp.route("/mileage-rates", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_mileage_rate():
    b = _body()
    return _run("mileage rate", ex.set_mileage_rate, get_engine(), b.get("effective_date"),
                b.get("rate"), b.get("basis"), _actor().get("username", "unknown"))
