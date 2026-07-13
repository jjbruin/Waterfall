"""Surveillance API — portfolio monitoring endpoints."""

from flask import Blueprint, request, jsonify, g

from flask_app.auth.routes import login_required, role_required
from flask_app.services import surveillance_service
from flask_app.serializers import safe_json

surveillance_bp = Blueprint("surveillance", __name__)


@surveillance_bp.route("/", methods=["GET"])
@login_required
def get_surveillance_table():
    """Full surveillance table — one row per active deal."""
    rows = surveillance_service.get_surveillance_table()
    return safe_json(rows)


@surveillance_bp.route("/dashboard", methods=["GET"])
@login_required
def get_dashboard():
    """Surveillance KPIs and chart data."""
    data = surveillance_service.get_dashboard()
    return safe_json(data)


@surveillance_bp.route("/<vcode>", methods=["PATCH"])
@login_required
def update_surveillance(vcode):
    """Update editable surveillance fields for a deal."""
    fields = request.get_json(force=True)
    username = getattr(g, "username", None)
    result = surveillance_service.update_surveillance_property(vcode, fields, username)
    return jsonify(result)


# --- Insurance endpoints ---

@surveillance_bp.route("/insurance", methods=["GET"])
@login_required
def get_insurance():
    """All insurance records with days-to-expiration."""
    rows = surveillance_service.get_insurance_list()
    return safe_json(rows)


@surveillance_bp.route("/insurance", methods=["POST"])
@login_required
def upsert_insurance():
    """Create or update an insurance record."""
    data = request.get_json(force=True)
    vcode = data.pop("vcode", None)
    ins_type = data.pop("ins_type", None)
    if not vcode or not ins_type:
        return jsonify({"error": "vcode and ins_type required"}), 400
    result = surveillance_service.upsert_insurance(vcode, ins_type, data)
    return jsonify(result)


@surveillance_bp.route("/insurance/<int:ins_id>", methods=["DELETE"])
@login_required
def delete_insurance(ins_id):
    """Delete an insurance record."""
    result = surveillance_service.delete_insurance(ins_id)
    return jsonify(result)
