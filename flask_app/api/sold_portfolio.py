"""Sold Portfolio API — historical returns from accounting."""

from flask import Blueprint, request, jsonify, current_app, send_file
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service
from flask_app.services.sold_service import (
    compute_all_sold_returns, build_deal_detail,
    generate_sold_excel, generate_detail_excel,
    get_sold_deals, compute_all_net_returns, generate_net_returns_excel,
)
from flask_app.serializers import df_to_records, safe_json

sold_portfolio_bp = Blueprint("sold_portfolio", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


@sold_portfolio_bp.route("/summary", methods=["GET"])
@login_required
def summary():
    """Get sold portfolio returns summary."""
    data = _get_data()
    inv_sold = get_sold_deals(data["inv"])

    if inv_sold.empty:
        return jsonify({"rows": [], "count": 0, "deal_names": []})

    returns_df = compute_all_sold_returns(inv_sold, data["acct"], data["inv"])

    # Build deal names for drill-down selector (exclude portfolio total)
    deal_names = []
    if not returns_df.empty:
        non_total = returns_df[~returns_df["_is_deal_total"].fillna(False).astype(bool)]
        deal_names = [
            {"name": r["Investment Name"], "vcode": r.get("vcode", "")}
            for _, r in non_total.iterrows()
        ]

    return jsonify({
        "rows": safe_json(df_to_records(returns_df)),
        "count": len(returns_df),
        "deal_names": deal_names,
    })


@sold_portfolio_bp.route("/summary/excel", methods=["GET"])
@login_required
def summary_excel():
    """Download sold portfolio summary as Excel."""
    data = _get_data()
    inv_sold = get_sold_deals(data["inv"])

    if inv_sold.empty:
        return jsonify({"error": "No sold deals found"}), 404

    returns_df = compute_all_sold_returns(inv_sold, data["acct"], data["inv"])
    excel_bytes = generate_sold_excel(returns_df)

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="sold_portfolio_returns.xlsx",
    )


@sold_portfolio_bp.route("/detail/<vcode>", methods=["GET"])
@login_required
def deal_detail(vcode):
    """Get cashflow detail for a single sold deal with running capital balance."""
    data = _get_data()
    inv_sold = get_sold_deals(data["inv"])

    if inv_sold.empty:
        return jsonify({"rows": [], "summary": {}})

    result = build_deal_detail(vcode, inv_sold, data["acct"], data["inv"])
    return jsonify(safe_json(result))


@sold_portfolio_bp.route("/detail/<vcode>/excel", methods=["GET"])
@login_required
def deal_detail_excel(vcode):
    """Download deal activity detail as Excel."""
    data = _get_data()
    inv_sold = get_sold_deals(data["inv"])

    if inv_sold.empty:
        return jsonify({"error": "No sold deals found"}), 404

    result = build_deal_detail(vcode, inv_sold, data["acct"], data["inv"])

    # Get deal name for filename
    match = inv_sold[inv_sold["vcode"].astype(str).str.strip() == str(vcode)]
    deal_name = match.iloc[0].get("Investment_Name", vcode) if not match.empty else vcode
    safe_name = deal_name.replace(" ", "_").replace("/", "_")

    excel_bytes = generate_detail_excel(result["rows"], deal_name, result["summary"])

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"sold_activity_{safe_name}.xlsx",
    )


def _parse_assumptions(body):
    """Parse and validate net returns assumptions from request body."""
    required = ["ownership_pct", "am_fee_pct", "hurdle_rate", "promote_pct", "annual_expenses"]
    missing = [k for k in required if k not in body]
    if missing:
        return None, f"Missing required fields: {', '.join(missing)}"

    try:
        assumptions = {
            "ownership_pct": float(body["ownership_pct"]),
            "am_fee_pct": float(body["am_fee_pct"]),
            "hurdle_rate": float(body["hurdle_rate"]),
            "promote_pct": float(body["promote_pct"]),
            "annual_expenses": float(body["annual_expenses"]),
        }
    except (ValueError, TypeError):
        return None, "All assumption values must be numeric"

    for pct_field in ["ownership_pct", "am_fee_pct", "hurdle_rate", "promote_pct"]:
        if not 0 <= assumptions[pct_field] <= 1:
            return None, f"{pct_field} must be between 0 and 1"
    if assumptions["annual_expenses"] < 0:
        return None, "annual_expenses must be >= 0"

    return assumptions, None


@sold_portfolio_bp.route("/net-returns", methods=["POST"])
@login_required
def net_returns():
    """Compute net returns with fee/promote waterfall assumptions."""
    body = request.get_json(force=True)
    assumptions, err = _parse_assumptions(body)
    if err:
        return jsonify({"error": err}), 400

    data = _get_data()
    inv_sold = get_sold_deals(data["inv"])
    if inv_sold.empty:
        return jsonify({"error": "No sold deals found"}), 404

    result = compute_all_net_returns(inv_sold, data["acct"], data["inv"], assumptions)

    # Strip waterfall_detail from JSON response (too large)
    deal_results = []
    for dr in result["deal_results"]:
        deal_results.append({k: v for k, v in dr.items() if k != "waterfall_detail"})

    return jsonify(safe_json({
        "deal_results": deal_results,
        "portfolio": result["portfolio"],
        "assumptions": result["assumptions"],
    }))


@sold_portfolio_bp.route("/net-returns/excel", methods=["POST"])
@login_required
def net_returns_excel():
    """Download net returns Excel with per-deal waterfall detail."""
    body = request.get_json(force=True)
    assumptions, err = _parse_assumptions(body)
    if err:
        return jsonify({"error": err}), 400

    data = _get_data()
    inv_sold = get_sold_deals(data["inv"])
    if inv_sold.empty:
        return jsonify({"error": "No sold deals found"}), 404

    gross_df = compute_all_sold_returns(inv_sold, data["acct"], data["inv"])
    net_result = compute_all_net_returns(inv_sold, data["acct"], data["inv"], assumptions)
    excel_bytes = generate_net_returns_excel(gross_df, net_result)

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="sold_portfolio_net_returns.xlsx",
    )
