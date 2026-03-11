"""Deals API — compute, partner returns, XIRR, audits, debt service, forecast."""

from flask import Blueprint, request, jsonify, current_app, send_file
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service, compute_service
from flask_app.serializers import df_to_records, safe_json

deals_bp = Blueprint("deals", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


def _params():
    """Extract standard query params with defaults."""
    return (
        request.args.get("start_year", current_app.config["DEFAULT_START_YEAR"], type=int),
        request.args.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"], type=int),
        request.args.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"], type=int),
    )


def _get_result(vcode, force=False):
    """Get or compute deal result with standard params."""
    start_year, horizon, pro_yr_base = _params()
    data = _get_data()
    return compute_service.get_cached_deal_result(
        vcode, start_year, horizon, pro_yr_base, data, force=force,
    )


# ============================================================
# Core compute
# ============================================================

@deals_bp.route("/compute", methods=["POST"])
@login_required
def compute_deal():
    """Compute or retrieve cached deal analysis."""
    body = request.get_json(silent=True) or {}
    vcode = body.get("vcode")
    if not vcode:
        return jsonify({"error": "vcode required"}), 400

    start_year = body.get("start_year", current_app.config["DEFAULT_START_YEAR"])
    horizon = body.get("horizon_years", current_app.config["DEFAULT_HORIZON_YEARS"])
    pro_yr_base = body.get("pro_yr_base", current_app.config["PRO_YR_BASE_DEFAULT"])
    force = body.get("force", False)

    data = _get_data()
    try:
        result = compute_service.get_cached_deal_result(
            vcode, start_year, horizon, pro_yr_base, data, force=force,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "vcode": vcode,
        "partner_results": safe_json(result.get("partner_results", [])),
        "deal_summary": safe_json(result.get("deal_summary", {})),
        "debug_msgs": result.get("debug_msgs", []),
    })


# ============================================================
# Deal header / capitalization
# ============================================================

@deals_bp.route("/<vcode>/header", methods=["GET"])
@login_required
def deal_header(vcode):
    """Get deal metadata and capitalization data."""
    data = _get_data()
    inv = data["inv"]
    deal_row = inv[inv["vcode"] == vcode]
    if deal_row.empty:
        return jsonify({"error": f"Deal not found: {vcode}"}), 404

    row = deal_row.iloc[0]
    metadata = {
        "vcode": vcode,
        "Investment_Name": row.get("Investment_Name", vcode),
        "InvestmentID": row.get("InvestmentID", ""),
        "Asset_Type": row.get("Asset_Type", ""),
        "Lifecycle": row.get("Lifecycle", ""),
        "Total_Units": row.get("Total_Units", ""),
        "Total_SQF": row.get("Total_SQF", ""),
        "Acquisition_Date": str(row.get("Acquisition_Date", "")),
        "Sale_Date": str(row.get("Sale_Date", "")),
        "Sale_Status": row.get("Sale_Status", ""),
    }

    try:
        result = _get_result(vcode)
        cap_data = safe_json(result.get("cap_data", {}))
        consolidation = safe_json(result.get("consolidation_info", {}))
        sub_portfolio_msg = result.get("sub_portfolio_msg")
    except Exception:
        cap_data = {}
        consolidation = {}
        sub_portfolio_msg = None

    return jsonify(safe_json({
        "metadata": metadata,
        "cap_data": cap_data,
        "consolidation": consolidation,
        "sub_portfolio_msg": sub_portfolio_msg,
    }))


# ============================================================
# Partner returns
# ============================================================

@deals_bp.route("/<vcode>/partner-returns", methods=["GET"])
@login_required
def partner_returns(vcode):
    """Get partner returns for a computed deal."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "partner_results": safe_json(result.get("partner_results", [])),
        "deal_summary": safe_json(result.get("deal_summary", {})),
    })


# ============================================================
# Annual forecast
# ============================================================

@deals_bp.route("/<vcode>/forecast", methods=["GET"])
@login_required
def deal_forecast(vcode):
    """Get forecast/modeled deal data."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    fc_display = result.get("fc_deal_display")
    return jsonify({
        "forecast": df_to_records(fc_display) if fc_display is not None else [],
    })


@deals_bp.route("/<vcode>/annual-forecast", methods=["GET"])
@login_required
def annual_forecast(vcode):
    """Get annual aggregation table (pivoted: rows=line items, cols=years)."""
    start_year, horizon, _ = _params()
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    from reporting import annual_aggregation_table

    fc_display = result.get("fc_deal_display")
    if fc_display is None or fc_display.empty:
        return jsonify({"rows": [], "years": []})

    # Get proceeds_by_year from cap_events_df
    cap_events = result.get("cap_events_df")
    proceeds_by_year = None
    if cap_events is not None and not cap_events.empty and "Year" in cap_events.columns:
        proceeds_by_year = cap_events.groupby("Year")["amount"].sum()

    cf_alloc = result.get("cf_alloc")
    cap_alloc = result.get("cap_alloc")

    try:
        table = annual_aggregation_table(
            fc_display, start_year, horizon,
            proceeds_by_year=proceeds_by_year,
            cf_alloc=cf_alloc,
            cap_alloc=cap_alloc,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Transpose: rows=line items, cols=years
    # table has columns: Year, Revenues, Expenses, NOI, ...
    import pandas as pd
    if "Year" in table.columns:
        wide = table.set_index("Year").T
        years = [int(y) for y in wide.columns]
        rows = []
        for label, row_data in wide.iterrows():
            values = {}
            for y in years:
                val = row_data.get(y)
                if val is not None and not (isinstance(val, float) and (val != val)):
                    values[str(y)] = float(val)
                else:
                    values[str(y)] = None
            rows.append({"label": str(label), "values": values})
    else:
        years = []
        rows = []

    return jsonify(safe_json({"rows": rows, "years": years}))


# ============================================================
# Debt service
# ============================================================

@deals_bp.route("/<vcode>/debt-service", methods=["GET"])
@login_required
def debt_service(vcode):
    """Get debt service data (loans, schedules, sale proceeds)."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    loans = result.get("loans", [])
    loan_sched = result.get("loan_sched")
    sale_dbg = result.get("sale_dbg")

    loan_list = []
    for l in (loans or []):
        loan_list.append({
            "vcode": l.vcode, "loan_id": l.loan_id,
            "orig_amount": l.orig_amount,
            "orig_date": l.orig_date.isoformat() if l.orig_date else None,
            "maturity_date": l.maturity_date.isoformat() if l.maturity_date else None,
            "int_type": l.int_type, "fixed_rate": l.fixed_rate,
            "loan_term_m": l.loan_term_m, "amort_term_m": l.amort_term_m,
            "io_months": l.io_months,
        })

    return jsonify(safe_json({
        "loans": loan_list,
        "loan_schedule": df_to_records(loan_sched) if loan_sched is not None else [],
        "sale_proceeds": sale_dbg,
    }))


# ============================================================
# Cash management
# ============================================================

@deals_bp.route("/<vcode>/cash-schedule", methods=["GET"])
@login_required
def cash_schedule(vcode):
    """Get cash flow schedule and summary."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sched = result.get("cash_schedule")
    summary = result.get("cash_summary", {})

    return jsonify(safe_json({
        "schedule": df_to_records(sched) if sched is not None else [],
        "summary": summary,
        "beginning_cash": result.get("beginning_cash", 0),
    }))


# ============================================================
# Capital calls
# ============================================================

@deals_bp.route("/<vcode>/capital-calls", methods=["GET"])
@login_required
def capital_calls(vcode):
    """Get capital calls schedule."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    calls = result.get("capital_calls", [])
    return jsonify(safe_json({"capital_calls": calls}))


# ============================================================
# XIRR cashflows
# ============================================================

@deals_bp.route("/<vcode>/xirr-cashflows", methods=["GET"])
@login_required
def xirr_cashflows(vcode):
    """Get XIRR cashflows for a computed deal."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    cashflows = {}
    for pr in result.get("partner_results", []):
        details = pr.get("cashflow_details", [])
        cfs = []
        for row in details:
            d = row["Date"]
            cfs.append({
                "date": d.isoformat() if hasattr(d, 'isoformat') else str(d),
                "description": row.get("Description", ""),
                "amount": row["Amount"],
            })
        cashflows[pr["partner"]] = cfs

    return jsonify(safe_json({"cashflows": cashflows}))


# ============================================================
# Waterfall allocations
# ============================================================

@deals_bp.route("/<vcode>/waterfall-allocations", methods=["GET"])
@login_required
def waterfall_allocations(vcode):
    """Get CF and Cap waterfall allocation tables."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    cf_alloc = result.get("cf_alloc")
    cap_alloc = result.get("cap_alloc")

    return jsonify({
        "cf_alloc": df_to_records(cf_alloc) if cf_alloc is not None else [],
        "cap_alloc": df_to_records(cap_alloc) if cap_alloc is not None else [],
    })


# ============================================================
# ROE / MOIC audits
# ============================================================

@deals_bp.route("/<vcode>/roe-audit", methods=["GET"])
@login_required
def roe_audit(vcode):
    """Get ROE audit data for all partners + deal level."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    audit = compute_service.build_roe_audit(
        result["partner_results"], result["deal_summary"], result.get("sale_me"),
    )
    return jsonify(safe_json(audit))


@deals_bp.route("/<vcode>/moic-audit", methods=["GET"])
@login_required
def moic_audit(vcode):
    """Get MOIC audit data for all partners + deal level."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    audit = compute_service.build_moic_audit(
        result["partner_results"], result["deal_summary"], result.get("sale_me"),
    )
    return jsonify(safe_json(audit))


@deals_bp.route("/<vcode>/roe-audit/excel", methods=["GET"])
@login_required
def roe_audit_excel(vcode):
    """Download ROE audit Excel."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    excel_bytes = compute_service.generate_roe_audit_excel(
        result["partner_results"], result["deal_summary"], result.get("sale_me"),
    )
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"roe_audit_{vcode}.xlsx",
    )


@deals_bp.route("/<vcode>/moic-audit/excel", methods=["GET"])
@login_required
def moic_audit_excel(vcode):
    """Download MOIC audit Excel."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    excel_bytes = compute_service.generate_moic_audit_excel(
        result["partner_results"], result["deal_summary"], result.get("sale_me"),
    )
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"moic_audit_{vcode}.xlsx",
    )


# ============================================================
# Cache info
# ============================================================

@deals_bp.route("/cached", methods=["GET"])
@login_required
def cached_deals():
    """List currently cached deal vcodes."""
    return jsonify({"cached": compute_service.get_all_cached_vcodes()})
