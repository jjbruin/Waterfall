"""Portfolio Analysis API — upstream entity analysis for active portfolios."""

from flask import Blueprint, request, jsonify, current_app, send_file
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service
from flask_app.services.portfolio_analysis_service import (
    find_portfolio_entities, find_entity_deals, get_entity_investors,
    compute_portfolio_actual, compute_portfolio_proposed,
    generate_portfolio_excel, get_deal_detail,
)
from flask_app.serializers import safe_json

portfolio_analysis_bp = Blueprint("portfolio_analysis", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


@portfolio_analysis_bp.route("/entities", methods=["GET"])
@login_required
def entities():
    """List portfolio entities available for analysis."""
    data = _get_data()
    entity_list = find_portfolio_entities(
        data["wf"], data["relationships_raw"], data["inv"]
    )
    return jsonify({"entities": entity_list})


@portfolio_analysis_bp.route("/entities/<entity_id>/deals", methods=["GET"])
@login_required
def entity_deals(entity_id):
    """List deals linked to a portfolio entity."""
    data = _get_data()
    deals = find_entity_deals(
        entity_id, data["inv"], data["wf"], data["relationships_raw"]
    )
    investors = get_entity_investors(entity_id, data["relationships_raw"])
    return jsonify({"deals": deals, "investors": investors})


@portfolio_analysis_bp.route("/entities/<entity_id>/compute", methods=["POST"])
@login_required
def compute(entity_id):
    """Compute portfolio analysis for an entity.

    Body: { mode: "actual"|"proposed", assumptions?: {...} }
    For proposed mode, assumptions: { am_fee_pct, hurdle_rate, promote_pct, annual_expenses }
    """
    data = _get_data()
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "actual")

    start_year = current_app.config.get("DEFAULT_START_YEAR", 2026)
    horizon_years = current_app.config.get("DEFAULT_HORIZON_YEARS", 10)
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]

    if mode == "proposed":
        assumptions = body.get("assumptions", {})
        if not assumptions:
            return jsonify({"error": "Assumptions required for proposed mode"}), 400
        results = compute_portfolio_proposed(
            entity_id=entity_id,
            data=data,
            assumptions=assumptions,
            start_year=start_year,
            horizon_years=horizon_years,
            pro_yr_base=pro_yr_base,
        )
    else:
        results = compute_portfolio_actual(
            entity_id=entity_id,
            data=data,
            start_year=start_year,
            horizon_years=horizon_years,
            pro_yr_base=pro_yr_base,
        )

    if "error" in results and results.get("deals_computed", 0) == 0:
        return jsonify(results), 400

    return jsonify(safe_json(results))


@portfolio_analysis_bp.route("/entities/<entity_id>/deals/<vcode>/detail", methods=["GET"])
@login_required
def deal_detail_view(entity_id, vcode):
    """Get detailed partner results for a specific deal (drill-down)."""
    data = _get_data()
    start_year = current_app.config.get("DEFAULT_START_YEAR", 2026)
    horizon_years = current_app.config.get("DEFAULT_HORIZON_YEARS", 10)
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]

    result = get_deal_detail(
        vcode=vcode,
        data=data,
        start_year=start_year,
        horizon_years=horizon_years,
        pro_yr_base=pro_yr_base,
    )
    if "error" in result:
        return jsonify(result), 400
    return jsonify(safe_json(result))


@portfolio_analysis_bp.route("/entities/<entity_id>/excel", methods=["POST"])
@login_required
def excel(entity_id):
    """Download Excel for portfolio analysis.

    Body: same as compute endpoint (mode + optional assumptions).
    """
    data = _get_data()
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "actual")

    start_year = current_app.config.get("DEFAULT_START_YEAR", 2026)
    horizon_years = current_app.config.get("DEFAULT_HORIZON_YEARS", 10)
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]

    if mode == "proposed":
        assumptions = body.get("assumptions", {})
        results = compute_portfolio_proposed(
            entity_id=entity_id,
            data=data,
            assumptions=assumptions,
            start_year=start_year,
            horizon_years=horizon_years,
            pro_yr_base=pro_yr_base,
        )
    else:
        results = compute_portfolio_actual(
            entity_id=entity_id,
            data=data,
            start_year=start_year,
            horizon_years=horizon_years,
            pro_yr_base=pro_yr_base,
        )

    if "error" in results and results.get("deals_computed", 0) == 0:
        return jsonify(results), 400

    xlsx_bytes = generate_portfolio_excel(results)
    entity_name = results.get("entity_name", entity_id).replace(" ", "_")
    return send_file(
        io.BytesIO(xlsx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"portfolio_analysis_{entity_name}_{mode}.xlsx",
    )
