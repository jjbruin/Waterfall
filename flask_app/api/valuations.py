"""Valuation cycle API — annual property valuation review (Phase 1).

Endpoints (registered at /api/valuations):
    GET    /cycles                                — list cycles
    POST   /cycles                                — create + seed a cycle (admin)
    POST   /cycles/<id>/reseed                    — add records for new deals (admin)
    GET    /cycles/<id>/dashboard                 — status board rows
    GET    /records/<id>                          — full record detail
    PUT    /records/<id>                          — update assumptions / override
    POST   /records/<id>/action                   — sign_off | reopen | exclude
    POST   /records/<id>/documents                — upload evidence (multipart)
    GET    /records/<id>/documents/<doc_id>/view  — serve a stored document
    DELETE /records/<id>/documents/<doc_id>       — remove a document
    PUT    /records/<id>/comments                 — save a comment section
    POST   /records/<id>/argus                    — import the valuation Argus export
    GET    /records/<id>/budget-review            — Review Form p.1 comparison
    GET    /records/<id>/balance-sheet            — Review Form p.2 data
"""

import logging
from io import BytesIO

from flask import Blueprint, request, jsonify, g, send_file

from flask_app.auth.routes import login_required, role_required
from flask_app.db import get_engine
from flask_app.serializers import safe_json
from flask_app.services import data_service, valuation_service

logger = logging.getLogger(__name__)

valuations_bp = Blueprint("valuations", __name__)


def _username() -> str:
    return g.current_user.get("username", "unknown")


# ------------------------------------------------------------
# Cycles
# ------------------------------------------------------------

@valuations_bp.route("/cycles", methods=["GET"])
@login_required
def list_cycles():
    try:
        return jsonify({"cycles": valuation_service.list_cycles(get_engine())})
    except Exception as e:
        logger.error(f"list_cycles failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/cycles", methods=["POST"])
@login_required
@role_required("admin")
def create_cycle():
    body = request.get_json(silent=True) or {}
    year = body.get("year")
    if not year:
        return jsonify({"error": "year is required"}), 400
    try:
        result = valuation_service.create_cycle(
            get_engine(), int(year), _username(), data_service.get_data(),
            as_of_date=body.get("as_of_date"),
        )
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"create_cycle failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/cycles/<int:cycle_id>/reseed", methods=["POST"])
@login_required
@role_required("admin")
def reseed_cycle(cycle_id):
    try:
        result = valuation_service.seed_cycle_records(
            get_engine(), cycle_id, data_service.get_data(), _username())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"reseed_cycle failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/cycles/<int:cycle_id>/dashboard", methods=["GET"])
@login_required
def cycle_dashboard(cycle_id):
    try:
        result = valuation_service.get_cycle_dashboard(
            get_engine(), cycle_id, data_service.get_data())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"cycle_dashboard failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Records
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>", methods=["GET"])
@login_required
def get_record(record_id):
    try:
        result = valuation_service.get_record(
            get_engine(), record_id, data_service.get_data())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"get_record failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>", methods=["PUT"])
@login_required
@role_required("admin", "analyst")
def update_record(record_id):
    body = request.get_json(silent=True) or {}
    try:
        result = valuation_service.update_record(get_engine(), record_id, body, _username())
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"update_record failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/action", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def record_action(record_id):
    body = request.get_json(silent=True) or {}
    action = body.get("action", "")
    try:
        result = valuation_service.record_action(get_engine(), record_id, action, _username())
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"record_action failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Documents
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>/documents", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def upload_documents(record_id):
    if not request.files:
        return jsonify({"error": "No files provided"}), 400
    files_list = request.files.getlist("files")
    if not files_list:
        return jsonify({"error": "No files provided"}), 400
    doc_type = request.form.get("doc_type", "other")
    try:
        file_tuples = [(f.filename, f.read()) for f in files_list if f.filename]
        report = valuation_service.upload_documents(
            get_engine(), record_id, file_tuples, doc_type, _username())
        return jsonify({"status": "uploaded", **report})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"upload_documents failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/documents/<int:doc_id>/view", methods=["GET"])
@login_required
def view_document(record_id, doc_id):
    try:
        filename, file_bytes = valuation_service.get_document(get_engine(), record_id, doc_id)
        lower = (filename or "").lower()
        if lower.endswith(".pdf"):
            mimetype = "application/pdf"
        elif lower.endswith((".xlsx", ".xlsm", ".xls")):
            mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            mimetype = "application/octet-stream"
        return send_file(BytesIO(file_bytes), mimetype=mimetype, download_name=filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"view_document failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/documents/<int:doc_id>", methods=["DELETE"])
@login_required
@role_required("admin", "analyst")
def delete_document(record_id, doc_id):
    try:
        removed = valuation_service.delete_document(get_engine(), record_id, doc_id)
        if not removed:
            return jsonify({"error": "Document not found"}), 404
        return jsonify({"status": "deleted"})
    except Exception as e:
        logger.error(f"delete_document failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Comments
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>/comments", methods=["PUT"])
@login_required
@role_required("admin", "analyst")
def save_comment(record_id):
    body = request.get_json(silent=True) or {}
    try:
        result = valuation_service.save_comment(
            get_engine(), record_id, body.get("section", ""),
            body.get("text", ""), _username())
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"save_comment failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Argus + Review Form data
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>/argus", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def import_argus(record_id):
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file provided"}), 400
    try:
        result = valuation_service.import_argus(
            get_engine(), record_id, f.read(), f.filename, _username())
        status = result.get("status")
        if status == "duplicate":
            return jsonify(safe_json(result)), 409
        if status == "error":
            return jsonify(safe_json(result)), 400
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"import_argus failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/budget-review", methods=["GET"])
@login_required
def budget_review(record_id):
    try:
        result = valuation_service.get_budget_review(
            get_engine(), record_id, data_service.get_data())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"budget_review failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/balance-sheet", methods=["GET"])
@login_required
def balance_sheet(record_id):
    try:
        result = valuation_service.get_balance_sheet(
            get_engine(), record_id, data_service.get_data())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"balance_sheet failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Phase 2 — permissions
# ------------------------------------------------------------

def _review_roles() -> list:
    from flask_app.services.review_service import get_user_review_roles
    try:
        return get_user_review_roles(g.current_user["id"])
    except Exception:
        return []


@valuations_bp.route("/permissions", methods=["GET"])
@login_required
def permissions():
    roles = _review_roles()
    committee = [r for r in roles if r in valuation_service.COMMITTEE_ROLES]
    return jsonify({
        "review_roles": roles,
        "committee_roles": committee,
        "can_approve": bool(committee),
        "is_recorder": valuation_service.RECORDER_ROLE in roles,
        "app_role": g.current_user.get("role"),
    })


# ------------------------------------------------------------
# Phase 2 — Reviewer Q&A
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>/questions", methods=["POST"])
@login_required
def ask_question(record_id):
    body = request.get_json(silent=True) or {}
    roles = _review_roles()
    role_label = next((r for r in roles if r in valuation_service.COMMITTEE_ROLES
                       or r == valuation_service.RECORDER_ROLE), "")
    try:
        result = valuation_service.ask_question(
            get_engine(), record_id, body.get("text", ""), _username(), role_label)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"ask_question failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/questions/<int:question_id>/answer", methods=["PUT"])
@login_required
@role_required("admin", "analyst")
def answer_question(question_id):
    body = request.get_json(silent=True) or {}
    try:
        result = valuation_service.answer_question(
            get_engine(), question_id, body.get("text", ""), _username())
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"answer_question failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/questions/<int:question_id>/resolve", methods=["POST"])
@login_required
def resolve_question(question_id):
    try:
        result = valuation_service.resolve_question(get_engine(), question_id, _username())
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"resolve_question failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Phase 2 — Committee approvals + snapshot
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>/approve", methods=["POST"])
@login_required
def committee_approve(record_id):
    body = request.get_json(silent=True) or {}
    try:
        result = valuation_service.committee_approve(
            get_engine(), record_id, _review_roles(), _username(),
            data_service.get_data(), note=body.get("note", ""))
        return jsonify(safe_json(result))
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"committee_approve failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/return", methods=["POST"])
@login_required
def committee_return(record_id):
    body = request.get_json(silent=True) or {}
    try:
        result = valuation_service.committee_return(
            get_engine(), record_id, _review_roles(), _username(), body.get("note", ""))
        return jsonify(result)
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"committee_return failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/snapshot", methods=["GET"])
@login_required
def get_snapshot(record_id):
    try:
        result = valuation_service.get_snapshot(get_engine(), record_id)
        if result is None:
            return jsonify({"error": "No approved snapshot for this record"}), 404
        return jsonify(safe_json(result))
    except Exception as e:
        logger.error(f"get_snapshot failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Phase 2 — Committee summary + workbook
# ------------------------------------------------------------

@valuations_bp.route("/cycles/<int:cycle_id>/committee-summary", methods=["GET"])
@login_required
def committee_summary(cycle_id):
    try:
        result = valuation_service.get_committee_summary(
            get_engine(), cycle_id, data_service.get_data())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"committee_summary failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/cycles/<int:cycle_id>/committee-excel", methods=["GET"])
@login_required
def committee_excel(cycle_id):
    try:
        content = valuation_service.generate_committee_workbook(
            get_engine(), cycle_id, data_service.get_data())
        return send_file(
            BytesIO(content),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            download_name="valuation_committee_summary.xlsx",
            as_attachment=True,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"committee_excel failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/cycles/<int:cycle_id>/approve-all", methods=["POST"])
@login_required
def approve_all(cycle_id):
    """Batch approval: this member approves every signed-off record in the cycle."""
    roles = _review_roles()
    committee = [r for r in roles if r in valuation_service.COMMITTEE_ROLES]
    if not committee:
        return jsonify({"error": "Only Valuation Committee members can approve"}), 403
    try:
        engine = get_engine()
        data = data_service.get_data()
        dash = valuation_service.get_cycle_dashboard(engine, cycle_id, data)
        approved, completed = 0, 0
        for rec in dash["records"]:
            if rec["status"] != "signed_off":
                continue
            result = valuation_service.committee_approve(
                engine, rec["id"], committee, _username(), data,
                note="Batch approval — all unexceptional valuations")
            approved += 1
            if result["status"] == "approved":
                completed += 1
        return jsonify({"approved_by_member": approved, "fully_approved": completed})
    except Exception as e:
        logger.error(f"approve_all failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Phase 2 — AI appraisal summary
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>/ai-summary", methods=["GET"])
@login_required
def get_ai_summary(record_id):
    from flask_app.services import valuation_ai_service
    try:
        result = valuation_ai_service.get_ai_summary(get_engine(), record_id)
        if result is None:
            return jsonify({"exists": False})
        return jsonify({"exists": True, **safe_json(result)})
    except Exception as e:
        logger.error(f"get_ai_summary failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/ai-summary", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def generate_ai_summary(record_id):
    from flask_app.services import valuation_ai_service
    body = request.get_json(silent=True) or {}
    try:
        result = valuation_ai_service.generate_appraisal_summary(
            get_engine(), record_id, _username(), doc_id=body.get("doc_id"))
        return jsonify({"exists": True, **safe_json(result)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"generate_ai_summary failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Phase 3 — NAV engine, curation, packages, publish
# ------------------------------------------------------------

@valuations_bp.route("/records/<int:record_id>/nav", methods=["GET"])
@login_required
def get_nav(record_id):
    from flask_app.services import valuation_nav_service
    try:
        engine = get_engine()
        data = data_service.get_data()
        inputs = valuation_nav_service.get_nav_inputs(engine, record_id, data)
        result = valuation_nav_service.get_nav(engine, record_id)
        return jsonify(safe_json({"inputs": inputs, "result": result}))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"get_nav failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/bs-selections", methods=["PUT"])
@login_required
@role_required("admin", "analyst")
def save_bs_selections(record_id):
    from flask_app.services import valuation_nav_service
    body = request.get_json(silent=True) or {}
    try:
        result = valuation_nav_service.save_bs_selections(
            get_engine(), record_id, body.get("selections", {}), _username())
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"save_bs_selections failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/nav/compute", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def compute_nav(record_id):
    from flask_app.services import valuation_nav_service
    try:
        result = valuation_nav_service.compute_nav(
            get_engine(), record_id, data_service.get_data(), _username())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"compute_nav failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/step-refs", methods=["PUT"])
@login_required
@role_required("admin", "analyst")
def set_step_ref():
    from flask_app.services import valuation_nav_service
    body = request.get_json(silent=True) or {}
    vcode = body.get("vcode")
    iorder = body.get("iorder")
    if not vcode or iorder is None:
        return jsonify({"error": "vcode and iorder are required"}), 400
    try:
        result = valuation_nav_service.set_step_ref(
            get_engine(), str(vcode), int(iorder), body.get("agreement_ref", ""), _username())
        return jsonify(result)
    except Exception as e:
        logger.error(f"set_step_ref failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/nav-package", methods=["GET"])
@login_required
def nav_package(record_id):
    from flask_app.services import valuation_nav_service
    try:
        content = valuation_nav_service.generate_nav_package(
            get_engine(), record_id, data_service.get_data())
        return send_file(
            BytesIO(content),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            download_name=f"NAV_package_{record_id}.xlsx",
            as_attachment=True,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"nav_package failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/cycles/<int:cycle_id>/nav-packages", methods=["GET"])
@login_required
def cycle_nav_packages(cycle_id):
    from flask_app.services import valuation_nav_service
    try:
        content = valuation_nav_service.generate_cycle_packages_zip(
            get_engine(), cycle_id, data_service.get_data())
        return send_file(
            BytesIO(content), mimetype="application/zip",
            download_name="nav_audit_packages.zip", as_attachment=True,
        )
    except Exception as e:
        logger.error(f"cycle_nav_packages failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@valuations_bp.route("/records/<int:record_id>/publish", methods=["POST"])
@login_required
@role_required("admin")
def publish_record(record_id):
    from flask_app.services import valuation_nav_service
    try:
        result = valuation_nav_service.publish_record(
            get_engine(), record_id, data_service.get_data(), _username())
        return jsonify(safe_json(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"publish_record failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
