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
    get_deal_pe_investors, build_pref_balance_detail,
    generate_pref_balance_excel,
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
    isbs_raw = data.get("isbs_raw")
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
                isbs_raw=isbs_raw,
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
    isbs_raw = data.get("isbs_raw")
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
                isbs_raw=isbs_raw,
            )
            if row:
                all_rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(all_rows)
    excel_bytes = generate_roe_summary_excel(df, all_rows=all_rows)

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="roe_summary.xlsx",
    )


# ---------------------------------------------------------------------------
# Pref Balance Detail
# ---------------------------------------------------------------------------

@reports_bp.route("/pref-balance-detail/investors/<vcode>", methods=["GET"])
@login_required
def pref_balance_investors(vcode):
    """Get PE investors for a deal."""
    data = _get_data()
    acct = data.get("acct")
    if acct is None or acct.empty:
        return jsonify({"investors": []})
    investors = get_deal_pe_investors(vcode, acct, data["inv"])
    return jsonify({"investors": investors})


@reports_bp.route("/pref-balance-detail", methods=["POST"])
@login_required
def pref_balance_detail():
    """Generate pref balance detail for a deal + investor.

    Body: { vcode, investor_id, report_date }
    """
    from datetime import date as dt_date

    body = request.get_json(silent=True) or {}
    vcode = body.get("vcode")
    investor_id = body.get("investor_id")
    if not vcode or not investor_id:
        return jsonify({"error": "vcode and investor_id required"}), 400

    report_date_str = body.get("report_date")
    report_date = pd.to_datetime(report_date_str).date() if report_date_str else dt_date.today()

    data = _get_data()
    acct = data.get("acct")
    if acct is None or acct.empty:
        return jsonify({"error": "No accounting data available"}), 400

    from loaders import load_waterfalls
    wf_steps = load_waterfalls(data["wf"])

    try:
        result = build_pref_balance_detail(
            vcode, investor_id, report_date, acct, data["inv"],
            wf_steps=wf_steps,
        )
        return jsonify(safe_json(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reports_bp.route("/pref-balance-detail/excel", methods=["POST"])
@login_required
def pref_balance_detail_excel():
    """Download pref balance detail as Excel."""
    from datetime import date as dt_date

    body = request.get_json(silent=True) or {}
    vcode = body.get("vcode")
    investor_id = body.get("investor_id")
    if not vcode or not investor_id:
        return jsonify({"error": "vcode and investor_id required"}), 400

    report_date_str = body.get("report_date")
    report_date = pd.to_datetime(report_date_str).date() if report_date_str else dt_date.today()

    data = _get_data()
    acct = data.get("acct")
    if acct is None or acct.empty:
        return jsonify({"error": "No accounting data available"}), 400

    from loaders import load_waterfalls
    wf_steps = load_waterfalls(data["wf"])

    result = build_pref_balance_detail(
        vcode, investor_id, report_date, acct, data["inv"],
        wf_steps=wf_steps,
    )

    excel_bytes = generate_pref_balance_excel(result["header"], result["rows"])

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"pref_balance_{vcode}_{investor_id}.xlsx",
    )


# ── PSC Preferred Equity exposure (accounting's tracker) ─────────────────
# Open to everyone with the Reports section, as Jim asked (Oct 5 2026). Read
# only. The figures come from the engines that already own them -- see
# pe_exposure_service's module docstring for which engine answers which.

def _pe_as_of():
    from datetime import date as dt_date
    from flask_app.services import pe_exposure_service as pe
    raw = request.args.get("as_of")
    if raw == "live":
        return dt_date.today(), None
    if raw:
        try:
            return pd.to_datetime(raw).date(), None
        except Exception:
            return None, f"as_of {raw!r} is not a date"
    return dt_date.fromisoformat(pe.quarter_options()["default"]), None


@reports_bp.route("/pe-exposure/quarters", methods=["GET"])
@login_required
def pe_exposure_quarters():
    from flask_app.services import pe_exposure_service as pe
    return jsonify(pe.quarter_options())


@reports_bp.route("/pe-exposure", methods=["GET"])
@login_required
def pe_exposure():
    from flask_app.services import pe_exposure_service as pe
    as_of, err = _pe_as_of()
    if err:
        return jsonify({"error": err}), 400
    try:
        return jsonify(safe_json(pe.get_report(as_of, data=_get_data())))
    except Exception as e:
        current_app.logger.error("pe exposure failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@reports_bp.route("/pe-exposure/excel", methods=["GET"])
@login_required
def pe_exposure_excel():
    from flask_app.services import pe_exposure_service as pe
    as_of, err = _pe_as_of()
    if err:
        return jsonify({"error": err}), 400
    report = pe.get_report(as_of, data=_get_data())
    return send_file(
        io.BytesIO(pe.to_excel(report)),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"PSC_PE_Exposure_{report['as_of']}.xlsx",
    )


# ── Accounting's allocation overrides (Oct 5 2026) ────────────────────
# Reads are the report's: anyone who can open Reports sees what overrides the
# split, because the figures they read depend on it. WRITES are accounting's --
# the admin username, or an accounting role with the Accounting section
# (has_accounting_authority). The admin ROLE alone does not qualify.

def _accounting_actor():
    from flask import g
    from flask_app.auth.sections import has_accounting_authority
    u = getattr(g, "current_user", None) or {}
    return u if has_accounting_authority(u) else None


@reports_bp.route("/pe-exposure/overrides", methods=["GET"])
@login_required
def pe_overrides():
    from flask_app.services import allocation_override_service as aos
    return jsonify({"overrides": aos.list_overrides(),
                    "can_edit": _accounting_actor() is not None})


@reports_bp.route("/pe-exposure/overrides/owners", methods=["GET"])
@login_required
def pe_override_owners():
    """An entity's investors in MRI on a date, to start an override from."""
    from datetime import date as dt_date
    from flask_app.services import ownership_chain_service as oc
    ent = (request.args.get("entity") or "").strip().upper()
    try:
        on = dt_date.fromisoformat((request.args.get("as_of") or dt_date.today().isoformat())[:10])
    except ValueError:
        return jsonify({"error": "as_of must be a date"}), 400
    if not ent:
        return jsonify({"error": "Name the entity"}), 400
    src = oc._Source(as_of=on, with_balances=False)
    return jsonify({"entity": ent, "as_of": on.isoformat(),
                    "owners": [{"investor_id": o["entity_id"], "pct": o.get("pct"),
                                "committed": o.get("committed")}
                               for o in oc._owners_of(src, ent)]})


@reports_bp.route("/pe-exposure/overrides", methods=["POST"])
@login_required
def pe_override_create():
    from flask_app.services import allocation_override_service as aos
    actor = _accounting_actor()
    if actor is None:
        return jsonify({"error": "Forbidden", "message": "Allocation overrides are entered by "
                        "accounting (an accounting role with the Accounting section)."}), 403
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(aos.create(b.get("entity_id"), b.get("investment_id"), b.get("effective_date"),
                                  b.get("lines"), b.get("reason"), actor["username"])), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@reports_bp.route("/pe-exposure/overrides/<int:oid>", methods=["DELETE"])
@login_required
def pe_override_remove(oid):
    from flask_app.services import allocation_override_service as aos
    actor = _accounting_actor()
    if actor is None:
        return jsonify({"error": "Forbidden", "message": "Allocation overrides are removed by "
                        "accounting."}), 403
    try:
        return jsonify(aos.remove(oid, actor["username"]))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
