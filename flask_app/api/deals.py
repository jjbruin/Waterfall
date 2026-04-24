"""Deals API — compute, partner returns, XIRR, audits, debt service, forecast, prospective loans."""

from flask import Blueprint, request, jsonify, current_app, send_file
import io

from flask_app.auth.routes import login_required
from flask_app.services import data_service, compute_service
from flask_app.serializers import df_to_records, safe_json
from database import (
    get_prospective_loans_for_deal, get_prospective_loan_by_id,
    save_prospective_loan, delete_prospective_loan as db_delete_prospective_loan,
    accept_prospective_loan as db_accept_prospective_loan,
    revert_prospective_loan as db_revert_prospective_loan,
)

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
        request.args.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"), type=str),
    )


def _get_result(vcode, force=False):
    """Get or compute deal result with standard params."""
    start_year, horizon, pro_yr_base, actuals_through = _params()
    data = _get_data()
    return compute_service.get_cached_deal_result(
        vcode, start_year, horizon, pro_yr_base, data, force=force,
        actuals_through=actuals_through,
    )


# ============================================================
# Diagnostics
# ============================================================

@deals_bp.route("/<vcode>/diag/forecast-dates", methods=["GET"])
@login_required
def diag_forecast_dates(vcode):
    """Diagnostic: show forecast event_date range and samples for a deal."""
    import pandas as pd
    data = _get_data()
    fc = data["fc"]
    fc_deal = fc[fc["vcode"].astype(str) == str(vcode)].copy()
    if fc_deal.empty:
        return jsonify({"error": f"No forecast rows for {vcode}"})
    dates = fc_deal["event_date"]
    date_strs = dates.astype(str).tolist()
    return jsonify({
        "vcode": vcode,
        "row_count": len(fc_deal),
        "event_date_dtype": str(dates.dtype),
        "min_date": str(min(dates)),
        "max_date": str(max(dates)),
        "first_5": date_strs[:5],
        "last_5": date_strs[-5:],
        "sorted_unique_first_10": sorted(set(date_strs))[:10],
    })


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
    actuals_through = body.get("actuals_through", current_app.config.get("ACTUALS_THROUGH"))
    force = body.get("force", False)

    data = _get_data()
    try:
        result = compute_service.get_cached_deal_result(
            vcode, start_year, horizon, pro_yr_base, data, force=force,
            actuals_through=actuals_through,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "vcode": vcode,
        "partner_results": safe_json(result.get("partner_results", [])),
        "deal_summary": safe_json(result.get("deal_summary", {})),
        "debug_msgs": result.get("debug_msgs", []),
        "refi_dbg": safe_json(result.get("refi_dbg")),
        "refi_capital_call_required": result.get("refi_capital_call_required", False),
        "refi_capital_call_amount": result.get("refi_capital_call_amount", 0),
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
    start_year, horizon, _, _ = _params()
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
            cash_schedule=result.get("cash_schedule"),
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
# Prospective Loans (Refinance / New Mortgage)
# ============================================================

@deals_bp.route("/<vcode>/prospective-loans", methods=["GET"])
@login_required
def list_prospective_loans(vcode):
    """List all prospective loans for a deal."""
    loans = get_prospective_loans_for_deal(vcode)
    return jsonify(safe_json({"loans": loans}))


@deals_bp.route("/<vcode>/prospective-loans", methods=["POST"])
@login_required
def create_prospective_loan(vcode):
    """Create a new prospective loan."""
    body = request.get_json(force=True)
    body["vcode"] = vcode
    if "status" not in body:
        body["status"] = "draft"
    username = request.headers.get("X-Username", "api")
    try:
        loan_id = save_prospective_loan(body, username)
    except Exception as e:
        return jsonify({"error": f"Failed to save loan: {e}"}), 500
    data_service.refresh_table("prospective_loans")
    return jsonify({"id": loan_id, "status": "created"}), 201


@deals_bp.route("/<vcode>/prospective-loans/<int:loan_id>", methods=["PUT"])
@login_required
def update_prospective_loan(vcode, loan_id):
    """Update an existing prospective loan."""
    body = request.get_json(force=True)
    body["id"] = loan_id
    body["vcode"] = vcode
    username = request.headers.get("X-Username", "api")
    save_prospective_loan(body, username)
    data_service.refresh_table("prospective_loans")
    compute_service.clear_cache(vcode)
    return jsonify({"status": "updated"})


@deals_bp.route("/<vcode>/prospective-loans/<int:loan_id>", methods=["DELETE"])
@login_required
def delete_prospective_loan(vcode, loan_id):
    """Delete a prospective loan."""
    existing = get_prospective_loan_by_id(loan_id)
    username = request.headers.get("X-Username", "api")
    ok = db_delete_prospective_loan(loan_id, username)
    if not ok:
        return jsonify({"error": "Not found"}), 404
    data_service.refresh_table("prospective_loans")
    if existing and existing.get("status") == "accepted":
        compute_service.clear_cache(vcode)
    return jsonify({"status": "deleted"})


@deals_bp.route("/<vcode>/prospective-loans/<int:loan_id>/accept", methods=["POST"])
@login_required
def accept_prospective_loan(vcode, loan_id):
    """Accept a prospective loan — triggers refi in next compute."""
    username = request.headers.get("X-Username", "api")
    ok = db_accept_prospective_loan(loan_id, username)
    if not ok:
        return jsonify({"error": "Not found"}), 404
    data_service.refresh_table("prospective_loans")
    compute_service.clear_cache(vcode)

    # Recompute to get refi analysis
    try:
        result = _get_result(vcode, force=True)
        refi_dbg = result.get("refi_dbg")
        return jsonify(safe_json({
            "status": "accepted",
            "refi_dbg": refi_dbg,
            "refi_capital_call_required": result.get("refi_capital_call_required", False),
            "refi_capital_call_amount": result.get("refi_capital_call_amount", 0),
        }))
    except Exception as e:
        return jsonify({"status": "accepted", "compute_error": str(e)})


@deals_bp.route("/<vcode>/prospective-loans/<int:loan_id>/revert", methods=["POST"])
@login_required
def revert_prospective_loan(vcode, loan_id):
    """Revert an accepted loan back to draft."""
    username = request.headers.get("X-Username", "api")
    ok = db_revert_prospective_loan(loan_id, username)
    if not ok:
        return jsonify({"error": "Not found"}), 404
    data_service.refresh_table("prospective_loans")
    compute_service.clear_cache(vcode)
    return jsonify({"status": "reverted"})


@deals_bp.route("/<vcode>/prospective-loans/<int:loan_id>/sizing", methods=["GET"])
@login_required
def sizing_analysis(vcode, loan_id):
    """Run sizing analysis for a prospective loan."""
    prospect = get_prospective_loan_by_id(loan_id)
    if not prospect:
        return jsonify({"error": "Not found"}), 404

    try:
        from planned_loans import size_prospective_loan
        from loans import amortize_monthly_schedule
        from loaders import load_mri_loans

        data = _get_data()

        # Get forecast for deal
        fc = data["fc"]
        fc_deal = fc[fc["vcode"].astype(str) == str(vcode)].copy()
        if fc_deal.empty:
            return jsonify({"error": "No forecast data for deal"}), 400

        mri_val = data["mri_val"]

        # Build original loan schedule for balance calculation
        loan_sched = None
        mri_loans_raw = data["mri_loans_raw"]
        if mri_loans_raw is not None and not mri_loans_raw.empty:
            from loans import build_loans_from_mri_loans
            ml = load_mri_loans(mri_loans_raw)
            ml = ml[ml["vCode"].astype(str) == str(vcode)].copy()
            loans = build_loans_from_mri_loans(ml)
            if loans:
                model_start = min(fc_deal["event_date"])
                model_end = max(fc_deal["event_date"])
                scheds = []
                for ln in loans:
                    s = amortize_monthly_schedule(ln, model_start, model_end)
                    if not s.empty:
                        scheds.append(s)
                if scheds:
                    import pandas as pd
                    loan_sched = pd.concat(scheds, ignore_index=True)

        est_amt, sizing = size_prospective_loan(
            fc_deal, mri_val, prospect, loan_sched, vcode,
        )
        return jsonify(safe_json(sizing))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Capital calls (from computed result)
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
# Capital calls CRUD (database-managed)
# ============================================================

@deals_bp.route("/<vcode>/raw-capital-calls", methods=["GET"])
@login_required
def raw_capital_calls(vcode):
    """Get raw capital calls from database for editing."""
    from database import get_db_connection
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT rowid as id, * FROM capital_calls WHERE Vcode = ? ORDER BY CallDate",
            (vcode,),
        ).fetchall()
        return jsonify(safe_json({"capital_calls": [dict(r) for r in rows]}))
    except Exception:
        return jsonify({"capital_calls": []})
    finally:
        conn.close()


@deals_bp.route("/<vcode>/raw-capital-calls", methods=["POST"])
@login_required
def create_capital_call(vcode):
    """Create a new capital call."""
    from database import get_db_connection
    body = request.get_json(force=True)
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO capital_calls (Vcode, PropCode, CallDate, Amount, CallType, FundingSource, Notes, Typename) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                vcode,
                body.get("PropCode", ""),
                body.get("CallDate", ""),
                float(body.get("Amount", 0)),
                body.get("CallType", ""),
                body.get("FundingSource", ""),
                body.get("Notes", ""),
                body.get("Typename", "Contribution: Investments"),
            ),
        )
        conn.commit()
        # Full data reload to ensure the new capital call is picked up —
        # refresh_table alone can miss it when capital_calls_raw was None.
        data_service.reload()
        compute_service.clear_cache(vcode)
        return jsonify({"status": "created"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@deals_bp.route("/<vcode>/raw-capital-calls/<int:call_id>", methods=["PUT"])
@login_required
def update_capital_call(vcode, call_id):
    """Update an existing capital call."""
    from database import get_db_connection
    body = request.get_json(force=True)
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE capital_calls SET PropCode=?, CallDate=?, Amount=?, CallType=?, FundingSource=?, Notes=?, Typename=? "
            "WHERE rowid=?",
            (
                body.get("PropCode", ""),
                body.get("CallDate", ""),
                float(body.get("Amount", 0)),
                body.get("CallType", ""),
                body.get("FundingSource", ""),
                body.get("Notes", ""),
                body.get("Typename", "Contribution: Investments"),
                call_id,
            ),
        )
        conn.commit()
        data_service.reload()
        compute_service.clear_cache(vcode)
        return jsonify({"status": "updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@deals_bp.route("/<vcode>/raw-capital-calls/<int:call_id>", methods=["DELETE"])
@login_required
def delete_capital_call(vcode, call_id):
    """Delete a capital call."""
    from database import get_db_connection
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM capital_calls WHERE rowid=?", (call_id,))
        conn.commit()
        data_service.reload()
        compute_service.clear_cache(vcode)
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


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

# ============================================================
# Per-section Excel downloads
# ============================================================

@deals_bp.route("/<vcode>/excel/partner-returns", methods=["GET"])
@login_required
def partner_returns_excel(vcode):
    """Download Partner Returns Excel."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    excel_bytes = compute_service.generate_partner_returns_excel(result)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=f"partner_returns_{vcode}.xlsx",
    )


@deals_bp.route("/<vcode>/excel/forecast", methods=["GET"])
@login_required
def forecast_excel(vcode):
    """Download Annual Forecast Excel."""
    start_year, horizon, _, _ = _params()
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    excel_bytes = compute_service.generate_forecast_excel(result, start_year, horizon)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=f"annual_forecast_{vcode}.xlsx",
    )


@deals_bp.route("/<vcode>/excel/debt-service", methods=["GET"])
@login_required
def debt_service_excel(vcode):
    """Download Debt Service Excel."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    excel_bytes = compute_service.generate_debt_service_excel(result)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=f"debt_service_{vcode}.xlsx",
    )


@deals_bp.route("/<vcode>/excel/cash-schedule", methods=["GET"])
@login_required
def cash_schedule_excel(vcode):
    """Download Cash Schedule Excel."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    excel_bytes = compute_service.generate_cash_schedule_excel(result)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=f"cash_schedule_{vcode}.xlsx",
    )


@deals_bp.route("/<vcode>/excel/capital-calls", methods=["GET"])
@login_required
def capital_calls_excel(vcode):
    """Download Capital Calls Excel."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    excel_bytes = compute_service.generate_capital_calls_excel(result)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=f"capital_calls_{vcode}.xlsx",
    )


@deals_bp.route("/<vcode>/excel/xirr-cashflows", methods=["GET"])
@login_required
def xirr_cashflows_excel(vcode):
    """Download XIRR Cash Flows Excel."""
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    excel_bytes = compute_service.generate_xirr_cashflows_excel(result)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=f"xirr_cashflows_{vcode}.xlsx",
    )


@deals_bp.route("/<vcode>/excel/full", methods=["GET"])
@login_required
def full_deal_excel(vcode):
    """Download comprehensive Deal Analysis Excel workbook."""
    start_year, horizon, _, _ = _params()
    try:
        result = _get_result(vcode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    excel_bytes = compute_service.generate_full_deal_excel(result, start_year, horizon)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=f"deal_analysis_{vcode}.xlsx",
    )


# ============================================================
# Cache info
# ============================================================

@deals_bp.route("/cached", methods=["GET"])
@login_required
def cached_deals():
    """List currently cached deal vcodes."""
    return jsonify({"cached": compute_service.get_all_cached_vcodes()})
