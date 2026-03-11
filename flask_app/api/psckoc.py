"""PSCKOC API — upstream entity analysis for PSCKOC holding entity."""

from flask import Blueprint, request, jsonify, current_app, send_file
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service
from flask_app.services.psckoc_service import (
    find_psckoc_deals, run_psckoc_computation, get_cached_results,
    generate_psckoc_excel,
)
from flask_app.serializers import safe_json

psckoc_bp = Blueprint("psckoc", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


@psckoc_bp.route("/deals", methods=["GET"])
@login_required
def deals():
    """Discover deals linked to PSCKOC."""
    data = _get_data()
    deal_list = find_psckoc_deals(data["inv"], data["wf"], data["relationships_raw"])
    return jsonify({"deals": deal_list})


@psckoc_bp.route("/compute", methods=["POST"])
@login_required
def compute():
    """Run PSCKOC computation — deal analysis + upstream waterfalls."""
    data = _get_data()

    # Get deal vcodes from the deals list
    deal_list = find_psckoc_deals(data["inv"], data["wf"], data["relationships_raw"])
    deal_vcodes = [d["vcode"] for d in deal_list]

    if not deal_vcodes:
        return jsonify({"error": "No PSCKOC deals found"}), 400

    start_year = current_app.config.get("DEFAULT_START_YEAR", 2026)
    horizon_years = current_app.config.get("DEFAULT_HORIZON_YEARS", 10)
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]

    results = run_psckoc_computation(
        deal_vcodes=deal_vcodes,
        data=data,
        start_year=start_year,
        horizon_years=horizon_years,
        pro_yr_base=pro_yr_base,
    )

    if "error" in results and results.get("deals_computed", 0) == 0:
        return jsonify(results), 400

    return jsonify(safe_json(results))


@psckoc_bp.route("/results", methods=["GET"])
@login_required
def results():
    """Get cached PSCKOC results."""
    cached = get_cached_results()
    if not cached:
        return jsonify({"error": "No PSCKOC results cached. Run compute first."}), 404
    return jsonify(safe_json(cached))


@psckoc_bp.route("/excel", methods=["GET"])
@login_required
def excel():
    """Download PSCKOC Excel report."""
    cached = get_cached_results()
    if not cached:
        return jsonify({"error": "No PSCKOC results cached. Run compute first."}), 404

    xlsx_bytes = generate_psckoc_excel(cached)
    return send_file(
        io.BytesIO(xlsx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="psckoc_analysis.xlsx",
    )
