"""Reports API — projected returns summary with population selectors and Excel export."""

from flask import Blueprint, request, jsonify, current_app, send_file
import pandas as pd
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service, compute_service
from flask_app.services.reports_service import (
    build_partner_returns, generate_returns_excel,
    build_deal_lookup, get_partner_deals, get_upstream_investor_deals,
)
from flask_app.serializers import safe_json

reports_bp = Blueprint("reports", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


@reports_bp.route("/deal-lookup", methods=["GET"])
@login_required
def deal_lookup():
    """Get eligible deals for report population selectors."""
    data = _get_data()
    lookup = build_deal_lookup(data["inv"], data["wf"])
    return jsonify({
        "eligible": lookup["eligible"],
    })


@reports_bp.route("/partners", methods=["GET"])
@login_required
def partners():
    """Get partners and their associated deals for By Partner selector."""
    data = _get_data()
    lookup = build_deal_lookup(data["inv"], data["wf"])
    partner_deals = get_partner_deals(
        lookup["wf_norm"], lookup["eligible_vcodes"], lookup["vcode_to_label"]
    )
    # Return as list of {partner, deals: [{vcode, label}]}
    result = []
    for partner, vcodes in partner_deals.items():
        result.append({
            "partner": partner,
            "deal_count": len(vcodes),
            "vcodes": vcodes,
        })
    return jsonify({"partners": result})


@reports_bp.route("/upstream-investors", methods=["GET"])
@login_required
def upstream_investors():
    """Get upstream investors and their deal exposure for By Upstream Investor selector."""
    data = _get_data()
    lookup = build_deal_lookup(data["inv"], data["wf"])
    investor_deals = get_upstream_investor_deals(
        data.get("relationships_raw"), data["inv"], lookup["eligible_vcodes"]
    )
    result = []
    for iid, info in investor_deals.items():
        result.append({
            "investor_id": iid,
            "name": info["name"],
            "display": info["display"],
            "deal_count": len(info["vcodes"]),
            "vcodes": info["vcodes"],
        })
    return jsonify({"investors": result})


@reports_bp.route("/projected-returns", methods=["POST"])
@login_required
def projected_returns():
    """Generate projected returns summary.

    Body: { vcodes: [list], start_year, horizon_years, pro_yr_base }
    """
    body = request.get_json(silent=True) or {}
    vcodes = body.get("vcodes", [])
    if not vcodes:
        return jsonify({"error": "vcodes list required"}), 400

    start_year = body.get("start_year", current_app.config["DEFAULT_START_YEAR"])
    horizon = body.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"])
    pro_yr_base = body.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"])

    data = _get_data()
    inv = data["inv"]
    all_rows = []
    errors = []

    for vcode in vcodes:
        deal_row = inv[inv["vcode"] == vcode]
        deal_name = deal_row.iloc[0].get("Investment_Name", vcode) if not deal_row.empty else vcode

        try:
            result = compute_service.get_cached_deal_result(vcode, start_year, horizon, pro_yr_base, data)
            rows = build_partner_returns(result, deal_name)
            all_rows.extend(rows)
        except Exception as e:
            errors.append({"vcode": vcode, "deal_name": deal_name, "error": str(e)})

    return jsonify({"rows": safe_json(all_rows), "errors": errors})


@reports_bp.route("/projected-returns/excel", methods=["POST"])
@login_required
def projected_returns_excel():
    """Download projected returns as Excel."""
    body = request.get_json(silent=True) or {}
    vcodes = body.get("vcodes", [])
    if not vcodes:
        return jsonify({"error": "vcodes list required"}), 400

    start_year = body.get("start_year", current_app.config["DEFAULT_START_YEAR"])
    horizon = body.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"])
    pro_yr_base = body.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"])

    data = _get_data()
    inv = data["inv"]
    all_rows = []

    for vcode in vcodes:
        deal_row = inv[inv["vcode"] == vcode]
        deal_name = deal_row.iloc[0].get("Investment_Name", vcode) if not deal_row.empty else vcode
        try:
            result = compute_service.get_cached_deal_result(vcode, start_year, horizon, pro_yr_base, data)
            rows = build_partner_returns(result, deal_name)
            all_rows.extend(rows)
        except Exception:
            continue

    df = pd.DataFrame(all_rows)
    excel_bytes = generate_returns_excel(df)

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="projected_returns.xlsx",
    )
