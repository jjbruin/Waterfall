"""Argus Enterprise API — import, projection management, COA mapping, forecast preview."""

from flask import Blueprint, request, jsonify, g

from flask_app.auth.routes import login_required
from flask_app.db import get_engine
from flask_app.services import argus_service, compute_service

argus_bp = Blueprint("argus", __name__)


@argus_bp.route("/<vcode>/import/cashflow", methods=["POST"])
@login_required
def import_cashflow(vcode):
    """Upload and parse an Argus Monthly Cash Flow Excel export."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename"}), 400

    import_label = request.form.get("import_label", "Argus Projection")
    import_type = request.form.get("import_type", "asset_management")
    username = g.current_user.get("username", "unknown")

    file_bytes = file.read()
    engine = get_engine()

    result = argus_service.import_argus_cashflow(
        engine, vcode, file_bytes, file.filename,
        import_label, import_type, username,
    )
    status_code = 200 if result["status"] == "success" else (409 if result["status"] == "duplicate" else 400)
    return jsonify(result), status_code


@argus_bp.route("/<vcode>/import/rent-roll", methods=["POST"])
@login_required
def import_rent_roll(vcode):
    """Upload and parse an Argus Rent Roll Summary Excel export."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    import_id = request.form.get("import_id")
    if not import_id:
        return jsonify({"error": "import_id required"}), 400

    username = g.current_user.get("username", "unknown")
    file_bytes = file.read()
    engine = get_engine()

    result = argus_service.import_argus_rent_roll(
        engine, vcode, file_bytes, file.filename,
        int(import_id), username,
    )
    return jsonify(result)


@argus_bp.route("/<vcode>/import/revenue-assumptions", methods=["POST"])
@login_required
def import_revenue_assumptions(vcode):
    """Upload and parse an Argus Revenue Assumptions Excel export."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    import_id = request.form.get("import_id")
    if not import_id:
        return jsonify({"error": "import_id required"}), 400

    username = g.current_user.get("username", "unknown")
    file_bytes = file.read()
    engine = get_engine()

    result = argus_service.import_argus_revenue_assumptions(
        engine, vcode, file_bytes, file.filename,
        int(import_id), username,
    )
    return jsonify(result)


@argus_bp.route("/<vcode>/projections", methods=["GET"])
@login_required
def list_projections(vcode):
    """List all Argus projection scenarios for a deal."""
    engine = get_engine()
    scenarios = argus_service.get_projection_scenarios(engine, vcode)
    return jsonify(scenarios)


@argus_bp.route("/<vcode>/projections/<int:import_id>/activate", methods=["PUT"])
@login_required
def activate_projection(vcode, import_id):
    """Set a projection as active, deactivate others. Invalidates compute cache."""
    engine = get_engine()
    argus_service.set_active_projection(engine, vcode, import_id)
    compute_service.clear_cache(vcode)
    return jsonify({"status": "ok"})


@argus_bp.route("/<vcode>/projections/<int:import_id>", methods=["DELETE"])
@login_required
def delete_projection(vcode, import_id):
    """Delete a projection and all related data."""
    engine = get_engine()
    argus_service.delete_projection(engine, vcode, import_id)
    compute_service.clear_cache(vcode)
    return jsonify({"status": "ok"})


@argus_bp.route("/<vcode>/projections/<int:import_id>/forecast", methods=["GET"])
@login_required
def preview_forecast(vcode, import_id):
    """Preview the forecast DataFrame for a specific projection as JSON."""
    from flask_app.config import config_by_name
    import os
    config_name = os.environ.get("FLASK_ENV", "development")
    cfg = config_by_name[config_name]
    pro_yr_base = int(request.args.get("pro_yr_base", cfg.PRO_YR_BASE_DEFAULT))

    engine = get_engine()
    df = argus_service.get_forecast_df_by_id(engine, vcode, import_id, pro_yr_base)
    if df is None:
        return jsonify({"error": "No cashflow data"}), 404

    # Convert to JSON-safe records
    records = df.copy()
    records["event_date"] = records["event_date"].astype(str)
    return jsonify(records.to_dict(orient="records"))


@argus_bp.route("/<vcode>/projections/<int:import_id>/tenants", methods=["GET"])
@login_required
def get_tenants(vcode, import_id):
    """Get tenant detail with rent steps for a specific import."""
    engine = get_engine()
    tenants = argus_service.get_argus_tenants(engine, vcode, import_id)
    return jsonify(tenants)


@argus_bp.route("/<vcode>/projections/<int:import_id>/mapping", methods=["GET"])
@login_required
def get_mapping(vcode, import_id):
    """Get COA mapping review — shows mapped and unmapped line items."""
    engine = get_engine()
    mapping = argus_service.get_coa_mapping(engine, import_id)
    return jsonify(mapping)


@argus_bp.route("/<vcode>/projections/<int:import_id>/mapping", methods=["PUT"])
@login_required
def update_mapping(vcode, import_id):
    """Override COA mappings for specific line items."""
    body = request.get_json(silent=True) or {}
    mappings = body.get("mappings", [])
    if not mappings:
        return jsonify({"error": "mappings array required"}), 400

    engine = get_engine()
    argus_service.update_coa_mapping(engine, import_id, mappings)
    compute_service.clear_cache(vcode)
    return jsonify({"status": "ok"})


@argus_bp.route("/<vcode>/projections/<int:import_id>/migrate", methods=["POST"])
@login_required
def migrate_projection(vcode, import_id):
    """Migrate a projection to the forecasts table for AM onboarding."""
    body = request.get_json(silent=True) or {}
    new_vcode = body.get("new_vcode")
    if not new_vcode:
        return jsonify({"error": "new_vcode required"}), 400

    from flask import current_app
    pro_yr_base = body.get("pro_yr_base", current_app.config.get("PRO_YR_BASE_DEFAULT", 2025))

    engine = get_engine()
    result = argus_service.migrate_projection_to_forecast(
        engine, vcode, new_vcode, import_id, pro_yr_base,
    )
    status_code = 200 if result["status"] == "success" else 400
    return jsonify(result), status_code
