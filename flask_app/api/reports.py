"""Reports API — projected returns summary with population selectors and Excel export."""

from flask import Blueprint, request, jsonify, current_app, send_file
import pandas as pd
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service, compute_service
from flask_app.services.reports_service import (
    build_partner_returns, generate_returns_excel,
    build_deal_lookup, get_upstream_investor_deals,
    build_roe_summary_row, generate_roe_summary_excel,
)
from flask_app.serializers import safe_json

reports_bp = Blueprint("reports", __name__)


def _get_data():
    return data_service.get_data()


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
    """Get upstream investors and their associated deals for By Partner selector.

    Uses the same upstream investor list as Review Tracking — recursive
    ownership chain traversal excluding OP% and PPI% entities.
    """
    data = _get_data()
    lookup = build_deal_lookup(data["inv"], data["wf"])
    investor_deals = get_upstream_investor_deals(
        data.get("relationships_raw"), data["inv"], lookup["eligible_vcodes"]
    )
    # Filter to same set as Review Tracking (exclude OP% and PPI%)
    result = []
    for iid, info in investor_deals.items():
        if iid.startswith("OP") or iid.startswith("PPI"):
            continue
        result.append({
            "partner": iid,
            "display": info["display"],
            "deal_count": len(info["vcodes"]),
            "vcodes": info["vcodes"],
        })
    return jsonify({"partners": result})


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
    actuals_through = body.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"))

    data = _get_data()
    inv = data["inv"]
    all_rows = []
    errors = []

    for vcode in vcodes:
        deal_row = inv[inv["vcode"] == vcode]
        deal_name = deal_row.iloc[0].get("Investment_Name", vcode) if not deal_row.empty else vcode

        try:
            result = compute_service.get_cached_deal_result(
                vcode, start_year, horizon, pro_yr_base, data,
                actuals_through=actuals_through,
            )
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
    actuals_through = body.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"))

    data = _get_data()
    inv = data["inv"]
    all_rows = []

    for vcode in vcodes:
        deal_row = inv[inv["vcode"] == vcode]
        deal_name = deal_row.iloc[0].get("Investment_Name", vcode) if not deal_row.empty else vcode
        try:
            result = compute_service.get_cached_deal_result(
                vcode, start_year, horizon, pro_yr_base, data,
                actuals_through=actuals_through,
            )
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


# ---------------------------------------------------------------------------
# ROE Summary
# ---------------------------------------------------------------------------

@reports_bp.route("/roe-summary", methods=["POST"])
@login_required
def roe_summary():
    """Generate ROE summary by deal through a report date.

    Body: { vcodes: [list], report_date (optional, defaults to today) }
    """
    from datetime import date as dt_date

    body = request.get_json(silent=True) or {}
    vcodes = body.get("vcodes", [])
    if not vcodes:
        return jsonify({"error": "vcodes list required"}), 400

    report_date_str = body.get("report_date")
    report_date = pd.to_datetime(report_date_str).date() if report_date_str else dt_date.today()

    start_year = body.get("start_year", current_app.config["DEFAULT_START_YEAR"])
    horizon = body.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"])
    pro_yr_base = body.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"])
    actuals_through = body.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"))

    data = _get_data()
    inv = data["inv"]
    acct = data.get("acct")
    all_rows = []
    errors = []

    if acct is None or acct.empty:
        return jsonify({"error": "No accounting data available"}), 400

    # Load waterfall steps once for pref rate extraction
    from loaders import load_waterfalls
    wf_steps = load_waterfalls(data["wf"])

    for vcode in vcodes:
        deal_row = inv[inv["vcode"] == vcode]
        deal_name = deal_row.iloc[0].get("Investment_Name", vcode) if not deal_row.empty else vcode

        try:
            row = build_roe_summary_row(
                vcode, deal_name, acct, inv,
                report_date, wf_steps=wf_steps,
            )
            if row:
                all_rows.append(row)
        except Exception as e:
            errors.append({"vcode": vcode, "deal_name": deal_name, "error": str(e)})

    return jsonify({"rows": safe_json(all_rows), "errors": errors})


@reports_bp.route("/roe-summary/excel", methods=["POST"])
@login_required
def roe_summary_excel():
    """Download ROE summary as Excel."""
    from datetime import date as dt_date

    body = request.get_json(silent=True) or {}
    vcodes = body.get("vcodes", [])
    if not vcodes:
        return jsonify({"error": "vcodes list required"}), 400

    report_date_str = body.get("report_date")
    report_date = pd.to_datetime(report_date_str).date() if report_date_str else dt_date.today()

    data = _get_data()
    inv = data["inv"]
    acct = data.get("acct")
    all_rows = []

    if acct is None or acct.empty:
        return jsonify({"error": "No accounting data available"}), 400

    from loaders import load_waterfalls
    wf_steps = load_waterfalls(data["wf"])

    for vcode in vcodes:
        deal_row = inv[inv["vcode"] == vcode]
        deal_name = deal_row.iloc[0].get("Investment_Name", vcode) if not deal_row.empty else vcode
        try:
            row = build_roe_summary_row(
                vcode, deal_name, acct, inv,
                report_date, wf_steps=wf_steps,
            )
            if row:
                all_rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(all_rows)
    excel_bytes = generate_roe_summary_excel(df)

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="roe_summary.xlsx",
    )
