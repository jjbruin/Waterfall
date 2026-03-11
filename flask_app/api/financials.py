"""Property Financials API — performance chart, IS, BS, tenants, one pager."""

from flask import Blueprint, request, jsonify, current_app

from flask_app.auth.routes import login_required
from flask_app.services import data_service
from flask_app.services.financials_service import (
    get_performance_chart_data, get_income_statement,
    get_balance_sheet, get_tenant_roster,
    get_one_pager_data, get_one_pager_chart,
)
from flask_app.serializers import safe_json

financials_bp = Blueprint("financials", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


# ============================================================
# Performance Chart
# ============================================================

@financials_bp.route("/<vcode>/performance-chart", methods=["GET"])
@login_required
def performance_chart(vcode):
    """Get performance chart data (NOI + occupancy)."""
    freq = request.args.get("freq", "Quarterly")
    periods = request.args.get("periods", 12, type=int)
    period_end = request.args.get("period_end")
    data = _get_data()
    try:
        chart_data = get_performance_chart_data(
            data["isbs_raw"], data["occupancy_raw"], vcode,
            freq=freq, periods=periods, period_end=period_end,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(safe_json(chart_data))


# ============================================================
# Income Statement
# ============================================================

@financials_bp.route("/<vcode>/income-statement", methods=["GET"])
@login_required
def income_statement(vcode):
    """Get income statement comparison."""
    data = _get_data()

    # Optionally get fc_deal_modeled for Valuation source
    fc_deal_modeled = None
    if request.args.get("left_source") == "Valuation" or request.args.get("right_source") == "Valuation":
        from flask_app.services import compute_service
        try:
            start_year = request.args.get("start_year", current_app.config["DEFAULT_START_YEAR"], type=int)
            horizon = request.args.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"], type=int)
            pro_yr_base = request.args.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"], type=int)
            result = compute_service.get_cached_deal_result(
                vcode, start_year, horizon, pro_yr_base, data,
            )
            fc_deal_modeled = result.get("fc_deal_modeled")
        except Exception:
            pass

    try:
        is_data = get_income_statement(
            data["isbs_raw"], vcode,
            left_source=request.args.get("left_source", "Actual"),
            left_period=request.args.get("left_period", "TTM"),
            right_source=request.args.get("right_source", "Budget"),
            right_period=request.args.get("right_period", "YTD"),
            as_of_date=request.args.get("as_of_date"),
            year=request.args.get("year", type=int),
            fc_deal_modeled=fc_deal_modeled,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(safe_json(is_data))


# ============================================================
# Balance Sheet
# ============================================================

@financials_bp.route("/<vcode>/balance-sheet", methods=["GET"])
@login_required
def balance_sheet(vcode):
    """Get balance sheet comparison."""
    data = _get_data()
    try:
        bs_data = get_balance_sheet(
            data["isbs_raw"], vcode,
            period1=request.args.get("period1"),
            period2=request.args.get("period2"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(safe_json(bs_data))


# ============================================================
# Tenant Roster
# ============================================================

@financials_bp.route("/<vcode>/tenants", methods=["GET"])
@login_required
def tenants(vcode):
    """Get tenant roster and lease rollover."""
    data = _get_data()
    try:
        tenant_data = get_tenant_roster(data["tenants_raw"], vcode, inv=data["inv"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(safe_json(tenant_data))


# ============================================================
# One Pager
# ============================================================

@financials_bp.route("/<vcode>/one-pager", methods=["GET"])
@login_required
def one_pager(vcode):
    """Get one pager investor report data."""
    quarter = request.args.get("quarter")
    data = _get_data()
    try:
        result = get_one_pager_data(
            vcode, quarter, data["inv"], data["isbs_raw"],
            data["mri_loans_raw"], data["mri_val"],
            data["wf"], data["commitments_raw"], data["acct"],
            occupancy_raw=data["occupancy_raw"],
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(safe_json(result))


@financials_bp.route("/<vcode>/one-pager/chart", methods=["GET"])
@login_required
def one_pager_chart(vcode):
    """Get one pager quarterly NOI chart data."""
    data = _get_data()
    try:
        chart = get_one_pager_chart(vcode, data["isbs_raw"], data["occupancy_raw"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(safe_json(chart))


@financials_bp.route("/<vcode>/one-pager/comments", methods=["PUT"])
@login_required
def save_comments(vcode):
    """Save one pager comments."""
    body = request.get_json(silent=True) or {}
    quarter = body.get("quarter")
    if not quarter:
        return jsonify({"error": "quarter required"}), 400
    try:
        from one_pager import save_one_pager_comments
        save_one_pager_comments(
            vcode, quarter,
            econ_comments=body.get("econ_comments"),
            business_plan_comments=body.get("business_plan_comments"),
            accrued_pref_comment=body.get("accrued_pref_comment"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})
