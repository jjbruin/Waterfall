"""Property Financials API — performance chart, IS, BS, tenants, one pager."""

from flask import Blueprint, request, jsonify, current_app, send_file
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service
from flask_app.services.financials_service import (
    get_performance_chart_data, get_income_statement,
    get_balance_sheet, get_tenant_roster,
    get_one_pager_data, get_one_pager_chart,
)
from flask_app.services.reports_service import (
    build_deal_lookup, get_upstream_investor_deals,
)
from flask_app.serializers import safe_json

financials_bp = Blueprint("financials", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


def _get_deal_name(inv, vcode):
    """Get deal name for filename."""
    row = inv[inv["vcode"] == str(vcode)]
    if not row.empty:
        name = row.iloc[0].get("Investment_Name", vcode)
        return str(name).replace("/", "-").replace("\\", "-").strip() or vcode
    return vcode


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

    # Get fc_deal_modeled for Valuation source and date discovery
    fc_deal_modeled = None
    from flask_app.services import compute_service
    try:
        start_year = request.args.get("start_year", current_app.config["DEFAULT_START_YEAR"], type=int)
        horizon = request.args.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"], type=int)
        pro_yr_base = request.args.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"], type=int)
        actuals_through = request.args.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"))
        result = compute_service.get_cached_deal_result(
            vcode, start_year, horizon, pro_yr_base, data,
            actuals_through=actuals_through,
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
            left_as_of_date=request.args.get("left_as_of_date"),
            right_as_of_date=request.args.get("right_as_of_date"),
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
# Excel Downloads
# ============================================================

@financials_bp.route("/<vcode>/excel/performance-chart", methods=["GET"])
@login_required
def excel_performance_chart(vcode):
    """Download performance chart data as Excel."""
    data = _get_data()
    freq = request.args.get("freq", "Quarterly")
    periods = request.args.get("periods", 12, type=int)
    period_end = request.args.get("period_end")
    chart_data = get_performance_chart_data(
        data["isbs_raw"], data["occupancy_raw"], vcode,
        freq=freq, periods=periods, period_end=period_end,
    )
    deal_name = _get_deal_name(data["inv"], vcode)
    from flask_app.services.financials_service import generate_performance_chart_excel
    content = generate_performance_chart_excel(chart_data, deal_name)
    return send_file(
        io.BytesIO(content),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{deal_name}_performance.xlsx",
    )


@financials_bp.route("/<vcode>/excel/income-statement", methods=["GET"])
@login_required
def excel_income_statement(vcode):
    """Download income statement as Excel."""
    data = _get_data()
    fc_deal_modeled = None
    from flask_app.services import compute_service
    try:
        start_year = request.args.get("start_year", current_app.config["DEFAULT_START_YEAR"], type=int)
        horizon = request.args.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"], type=int)
        pro_yr_base = request.args.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"], type=int)
        actuals_through = request.args.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"))
        result = compute_service.get_cached_deal_result(
            vcode, start_year, horizon, pro_yr_base, data,
            actuals_through=actuals_through,
        )
        fc_deal_modeled = result.get("fc_deal_modeled")
    except Exception:
        pass
    is_data = get_income_statement(
        data["isbs_raw"], vcode,
        left_source=request.args.get("left_source", "Actual"),
        left_period=request.args.get("left_period", "TTM"),
        right_source=request.args.get("right_source", "Budget"),
        right_period=request.args.get("right_period", "YTD"),
        left_as_of_date=request.args.get("left_as_of_date"),
        right_as_of_date=request.args.get("right_as_of_date"),
        fc_deal_modeled=fc_deal_modeled,
    )
    deal_name = _get_deal_name(data["inv"], vcode)
    from flask_app.services.financials_service import generate_income_statement_excel
    content = generate_income_statement_excel(is_data, deal_name)
    return send_file(
        io.BytesIO(content),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{deal_name}_income_statement.xlsx",
    )


@financials_bp.route("/<vcode>/excel/balance-sheet", methods=["GET"])
@login_required
def excel_balance_sheet(vcode):
    """Download balance sheet as Excel."""
    data = _get_data()
    bs_data = get_balance_sheet(
        data["isbs_raw"], vcode,
        period1=request.args.get("period1"),
        period2=request.args.get("period2"),
    )
    deal_name = _get_deal_name(data["inv"], vcode)
    from flask_app.services.financials_service import generate_balance_sheet_excel
    content = generate_balance_sheet_excel(bs_data, deal_name)
    return send_file(
        io.BytesIO(content),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{deal_name}_balance_sheet.xlsx",
    )


@financials_bp.route("/<vcode>/excel/tenants", methods=["GET"])
@login_required
def excel_tenants(vcode):
    """Download tenant roster as Excel."""
    data = _get_data()
    tenant_data = get_tenant_roster(data["tenants_raw"], vcode, inv=data["inv"])
    deal_name = _get_deal_name(data["inv"], vcode)
    from flask_app.services.financials_service import generate_tenant_roster_excel
    content = generate_tenant_roster_excel(tenant_data, deal_name)
    return send_file(
        io.BytesIO(content),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{deal_name}_tenants.xlsx",
    )


@financials_bp.route("/<vcode>/excel/full", methods=["GET"])
@login_required
def excel_full_financials(vcode):
    """Download full property financials workbook (all sections)."""
    data = _get_data()
    freq = request.args.get("freq", "Quarterly")
    periods = request.args.get("periods", 12, type=int)
    chart_data = get_performance_chart_data(
        data["isbs_raw"], data["occupancy_raw"], vcode,
        freq=freq, periods=periods,
    )

    fc_deal_modeled = None
    from flask_app.services import compute_service
    try:
        start_year = request.args.get("start_year", current_app.config["DEFAULT_START_YEAR"], type=int)
        horizon = request.args.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"], type=int)
        pro_yr_base = request.args.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"], type=int)
        actuals_through = request.args.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"))
        result = compute_service.get_cached_deal_result(
            vcode, start_year, horizon, pro_yr_base, data,
            actuals_through=actuals_through,
        )
        fc_deal_modeled = result.get("fc_deal_modeled")
    except Exception:
        pass
    is_data = get_income_statement(
        data["isbs_raw"], vcode,
        left_source=request.args.get("left_source", "Actual"),
        left_period=request.args.get("left_period", "TTM"),
        right_source=request.args.get("right_source", "Budget"),
        right_period=request.args.get("right_period", "YTD"),
        left_as_of_date=request.args.get("left_as_of_date"),
        right_as_of_date=request.args.get("right_as_of_date"),
        fc_deal_modeled=fc_deal_modeled,
    )
    bs_data = get_balance_sheet(
        data["isbs_raw"], vcode,
        period1=request.args.get("period1"),
        period2=request.args.get("period2"),
    )
    tenant_data = get_tenant_roster(data["tenants_raw"], vcode, inv=data["inv"])

    deal_name = _get_deal_name(data["inv"], vcode)
    from flask_app.services.financials_service import generate_full_financials_excel
    content = generate_full_financials_excel(chart_data, is_data, bs_data, tenant_data, deal_name)
    return send_file(
        io.BytesIO(content),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{deal_name}_property_financials.xlsx",
    )


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

    # Block comment edits when document is in review/approved
    try:
        from flask_app.services.review_service import is_editable
        if not is_editable(vcode, quarter):
            return jsonify({"error": "Comments are locked while the document is in review"}), 403
    except Exception:
        pass  # If review tables don't exist yet, allow edits

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


# ============================================================
# Batch One Pager (by investor)
# ============================================================

@financials_bp.route("/one-pager/investors", methods=["GET"])
@login_required
def one_pager_investors():
    """List upstream investors for the batch one-pager selector."""
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


@financials_bp.route("/one-pager/batch", methods=["POST"])
@login_required
def one_pager_batch():
    """Get one-pager data + chart for multiple deals at once.

    Body: { vcodes: [...], quarter: "2025-Q4" (optional) }
    Returns: { pages: [{vcode, data, chart, error?}, ...] }
    """
    body = request.get_json(silent=True) or {}
    vcodes = body.get("vcodes", [])
    quarter = body.get("quarter")

    if not vcodes:
        return jsonify({"error": "vcodes list required"}), 400

    data = _get_data()
    pages = []
    for vcode in vcodes:
        page: dict = {"vcode": vcode}
        try:
            page["data"] = get_one_pager_data(
                vcode, quarter, data["inv"], data["isbs_raw"],
                data["mri_loans_raw"], data["mri_val"],
                data["wf"], data["commitments_raw"], data["acct"],
                occupancy_raw=data["occupancy_raw"],
            )
        except Exception as e:
            page["data"] = None
            page["error"] = str(e)
        try:
            page["chart"] = get_one_pager_chart(vcode, data["isbs_raw"], data["occupancy_raw"])
        except Exception:
            page["chart"] = None
        pages.append(page)

    return jsonify(safe_json({"pages": pages}))
