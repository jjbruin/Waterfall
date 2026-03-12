"""Dashboard API — KPIs, chart data, computed returns."""

from flask import Blueprint, request, jsonify, current_app, send_file, Response
import io
import json
import logging

log = logging.getLogger(__name__)

from flask_app.auth.routes import login_required
from flask_app.services import data_service, compute_service
from flask_app.services.dashboard_service import (
    get_portfolio_kpis, get_portfolio_caps, get_latest_occupancy,
    get_capital_structure, get_occupancy_by_type, get_asset_allocation,
    get_loan_maturity_data, compute_portfolio_noi,
)
from flask_app.serializers import safe_json

dashboard_bp = Blueprint("dashboard", __name__)

# Module-level cache for caps/occ (cleared on reload)
_caps_cache = {}
_occ_cache = {}


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


def _get_caps_and_occ(on_progress=None):
    """Get or compute caps and occupancy (cached)."""
    data = _get_data()
    inv_disp = data_service.get_inv_display(data["inv"])

    cache_key = len(inv_disp)
    if cache_key not in _caps_cache:
        _caps_cache[cache_key] = get_portfolio_caps(
            inv_disp, data["inv"], data["wf"], data["acct"],
            data["mri_val"], data["mri_loans_raw"],
            on_progress=on_progress,
        )
    if cache_key not in _occ_cache:
        _occ_cache[cache_key] = get_latest_occupancy(inv_disp, data["occupancy_raw"])

    return _caps_cache[cache_key], _occ_cache[cache_key], data, inv_disp


@dashboard_bp.route("/init-stream", methods=["GET"])
@login_required
def init_stream():
    """SSE endpoint: pre-compute caps/occ with progress updates.

    Streams events like: data: {"current":5,"total":84,"deal":"McCord Tower"}
    Final event:         data: {"done":true,"total":84}
    """
    data = _get_data()
    inv_disp = data_service.get_inv_display(data["inv"])
    cache_key = len(inv_disp)

    # If already cached, send done immediately
    if cache_key in _caps_cache and cache_key in _occ_cache:
        def cached():
            yield f"data: {json.dumps({'done': True, 'total': 0, 'cached': True})}\n\n"
        return Response(cached(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def generate():
        total = len(inv_disp)
        caps = []
        from compute import get_deal_capitalization, prepare_cap_lookups
        import pandas as pd

        # Pre-compute all lookups once (avoids 84x redundant copies/normalization)
        lookups = prepare_cap_lookups(
            data["acct"], data["inv"], data["mri_val"], data["mri_loans_raw"],
        )
        prop_map = lookups["prop_map"]

        for i, (_, row) in enumerate(inv_disp.iterrows()):
            vcode = str(row["vcode"])
            name = row.get("DealLabel", row.get("Investment_Name", vcode))
            msg = json.dumps({"current": i + 1, "total": total, "deal": name})
            yield f"data: {msg}\n\n"

            try:
                prop_vcodes = prop_map.get(vcode, []) or None
                cap = get_deal_capitalization(
                    data["acct"], data["inv"], data["wf"],
                    data["mri_val"], data["mri_loans_raw"],
                    deal_vcode=vcode,
                    property_vcodes=prop_vcodes,
                    lookups=lookups,
                )
                cap["vcode"] = vcode
                cap["name"] = name
                cap["asset_type"] = str(row.get("Asset_Type", "") or "")
                cap["total_units"] = float(
                    pd.to_numeric(row.get("Total_Units", 0), errors="coerce") or 0
                )
                caps.append(cap)
            except Exception:
                continue

        # Cache results
        _caps_cache[cache_key] = caps
        if cache_key not in _occ_cache:
            _occ_cache[cache_key] = get_latest_occupancy(inv_disp, data["occupancy_raw"])

        yield f"data: {json.dumps({'done': True, 'total': total})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@dashboard_bp.route("/kpis", methods=["GET"])
@login_required
def kpis():
    """Get portfolio KPI values."""
    log.info("GET /kpis — computing caps & occ...")
    caps, occ_map, _, _ = _get_caps_and_occ()
    log.info("GET /kpis — caps ready (%d deals), computing KPIs", len(caps))
    kpi_data = get_portfolio_kpis(caps, occ_map)
    return jsonify(safe_json(kpi_data))


@dashboard_bp.route("/capitalization", methods=["GET"])
@login_required
def capitalization():
    """Get capital structure data for stacked bar chart."""
    caps, _, _, _ = _get_caps_and_occ()
    structure = get_capital_structure(caps)
    return jsonify(safe_json(structure))


@dashboard_bp.route("/noi-trend", methods=["GET"])
@login_required
def noi_trend():
    """Get portfolio NOI trend data.

    Query params: freq (Monthly|Quarterly|Annually), periods (int).
    """
    freq = request.args.get("freq", "Quarterly")
    data = _get_data()
    inv_disp = data_service.get_inv_display(data["inv"])
    noi_data = compute_portfolio_noi(
        data["isbs_raw"], inv_disp, frequency=freq,
        occupancy_raw=data["occupancy_raw"],
    )
    return jsonify(safe_json(noi_data))


@dashboard_bp.route("/occupancy-by-type", methods=["GET"])
@login_required
def occupancy_by_type():
    """Get weighted-average occupancy by asset type."""
    caps, occ_map, _, _ = _get_caps_and_occ()
    occ_data = get_occupancy_by_type(caps, occ_map)

    # Compute portfolio average for reference line
    if occ_data:
        avg = sum(d["occupancy"] for d in occ_data) / len(occ_data)
    else:
        avg = 0
    return jsonify(safe_json({"data": occ_data, "portfolio_avg": avg}))


@dashboard_bp.route("/asset-allocation", methods=["GET"])
@login_required
def asset_allocation():
    """Get asset allocation by type for donut chart."""
    caps, _, _, _ = _get_caps_and_occ()
    allocation = get_asset_allocation(caps)
    return jsonify(safe_json({"allocation": allocation}))


@dashboard_bp.route("/loan-maturities", methods=["GET"])
@login_required
def loan_maturities():
    """Get loan maturity data for stacked bar chart."""
    _, _, data, inv_disp = _get_caps_and_occ()
    maturity_data = get_loan_maturity_data(data["mri_loans_raw"], inv_disp, data["inv"])
    return jsonify(safe_json(maturity_data))


@dashboard_bp.route("/computed-returns", methods=["POST"])
@login_required
def computed_returns():
    """Compute returns for all active deals. Button-gated in UI."""
    data = _get_data()
    inv_disp = data_service.get_inv_display(data["inv"])

    start_year = current_app.config["DEFAULT_START_YEAR"]
    horizon = current_app.config["DEFAULT_HORIZON_YEARS"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]

    # Filter to deals with waterfalls
    wf = data["wf"]
    if wf is not None and not wf.empty:
        wf_vcodes = set(wf["vcode"].astype(str).unique()) if "vcode" in wf.columns else set()
    else:
        wf_vcodes = set()
    eligible = inv_disp[inv_disp["vcode"].astype(str).isin(wf_vcodes)]

    results = []
    errors = []
    for _, row in eligible.iterrows():
        vcode = str(row["vcode"])
        name = row.get("Investment_Name", vcode)
        try:
            result = compute_service.get_cached_deal_result(
                vcode, start_year, horizon, pro_yr_base, data,
            )
            ds = result.get("deal_summary", {})
            results.append({
                "vcode": vcode,
                "name": name,
                "contributions": ds.get("total_contributions", 0),
                "cf_distributions": ds.get("total_cf_distributions", 0),
                "cap_distributions": ds.get("total_cap_distributions", 0),
                "irr": ds.get("deal_irr"),
                "roe": ds.get("deal_roe"),
                "moic": ds.get("deal_moic"),
            })
        except Exception as e:
            errors.append({"vcode": vcode, "name": name, "error": str(e)})

    return jsonify(safe_json({"returns": results, "errors": errors}))
