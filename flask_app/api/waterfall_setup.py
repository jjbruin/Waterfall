"""Waterfall Setup API — CRUD, validate, preview, copy, templates."""

from flask import Blueprint, request, jsonify, current_app, send_file
import pandas as pd
import io

from flask_app.auth.routes import login_required, role_required
from flask_app.services import data_service
from flask_app.services.waterfall_service import (
    validate_steps, save_steps, delete_steps,
    get_waterfall_steps, get_entities_with_waterfalls,
    get_entity_nav_data, get_entity_investors,
    preview_waterfall, copy_cf_to_cap, copy_waterfall_from_entity,
    prefill_new_waterfall, prefill_pari_passu_waterfall,
    build_save_df, VSTATE_OPTIONS,
)
from flask_app.serializers import safe_json

waterfall_setup_bp = Blueprint("waterfall_setup", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


# ── Entity Navigation ────────────────────────────────────────────────────

@waterfall_setup_bp.route("/entities", methods=["GET"])
@login_required
def entities():
    """List all entities with waterfalls + relationship entities."""
    data = _get_data()
    nav = get_entity_nav_data(data["wf"], data["inv"], data.get("relationships_raw"))
    return jsonify(nav)


@waterfall_setup_bp.route("/<vcode>/investors", methods=["GET"])
@login_required
def investors(vcode):
    """Get investors for an entity from relationships + accounting."""
    data = _get_data()
    inv_list = get_entity_investors(
        vcode, data.get("relationships_raw"), data.get("acct"), data["inv"]
    )
    return jsonify({"investors": inv_list})


# ── Steps CRUD ────────────────────────────────────────────────────────────

@waterfall_setup_bp.route("/<vcode>/steps", methods=["GET"])
@login_required
def steps(vcode):
    """Get waterfall steps (CF_WF + Cap_WF) for an entity."""
    data = _get_data()
    wf_data = get_waterfall_steps(data["wf"], vcode)
    return jsonify(wf_data)


@waterfall_setup_bp.route("/<vcode>/steps", methods=["PUT"])
@login_required
@role_required("admin", "analyst")
def save(vcode):
    """Save waterfall steps for an entity (admin/analyst only).

    Body: { wf_type: "CF_WF"|"Cap_WF", steps: [...], other_steps?: [...] }
    If wf_type is provided, saves only that type and preserves the other.
    If wf_type is not provided, steps should contain vmisc field for all rows.
    """
    body = request.get_json(silent=True) or {}
    steps_list = body.get("steps", [])
    wf_type = body.get("wf_type")
    other_steps = body.get("other_steps")

    if not steps_list:
        return jsonify({"error": "steps required"}), 400

    data = _get_data()

    if wf_type:
        # Save one type, preserve the other
        combined = build_save_df(
            vcode, wf_type, steps_list,
            other_type_steps=other_steps,
            full_wf=data["wf"],
        )
    else:
        # Legacy: steps already contain vmisc
        combined = pd.DataFrame(steps_list)

    result = save_steps(vcode, combined, data["wf"])
    if result["success"]:
        data_service.reload()
        return jsonify(result)
    else:
        return jsonify(result), 500


@waterfall_setup_bp.route("/<vcode>/steps", methods=["DELETE"])
@login_required
@role_required("admin")
def delete(vcode):
    """Delete waterfall steps for an entity (admin only)."""
    wf_type = request.args.get("wf_type")
    result = delete_steps(vcode, wf_type)
    if result["success"]:
        data_service.reload()
        return jsonify(result)
    else:
        return jsonify(result), 500


# ── Validate ──────────────────────────────────────────────────────────────

@waterfall_setup_bp.route("/validate", methods=["POST"])
@login_required
def validate():
    """Validate waterfall steps.

    Body: { steps: [...], wf_type: "CF_WF"|"Cap_WF" }
    """
    body = request.get_json(silent=True) or {}
    steps_list = body.get("steps", [])
    wf_type = body.get("wf_type", "CF_WF")

    df = pd.DataFrame(steps_list) if steps_list else pd.DataFrame()
    # Ensure numeric types for validation
    for col in ["FXRate", "nPercent", "mAmount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "iOrder" in df.columns:
        df["iOrder"] = pd.to_numeric(df["iOrder"], errors="coerce").fillna(0).astype(int)

    errors, warnings = validate_steps(df, wf_type)
    return jsonify({"errors": errors, "warnings": warnings})


# ── Preview ───────────────────────────────────────────────────────────────

@waterfall_setup_bp.route("/<vcode>/preview", methods=["POST"])
@login_required
def preview(vcode):
    """Preview waterfall with $100k test distribution.

    Body: { steps: [...], wf_type: "CF_WF"|"Cap_WF" }
    """
    body = request.get_json(silent=True) or {}
    steps_list = body.get("steps", [])
    wf_type = body.get("wf_type", "CF_WF")

    if not steps_list:
        return jsonify({"success": False, "error": "No steps to preview"}), 400

    result = preview_waterfall(vcode, wf_type, steps_list)
    return jsonify(safe_json(result))


# ── Copy CF_WF -> Cap_WF ─────────────────────────────────────────────────

@waterfall_setup_bp.route("/<vcode>/copy-cf-to-cap", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def copy_cf_to_cap_route(vcode):
    """Copy CF_WF steps to Cap_WF for an entity.

    Can accept steps in body (draft) or use saved steps from DB.
    Body (optional): { cf_steps: [...] }
    """
    body = request.get_json(silent=True) or {}
    cf_steps = body.get("cf_steps")

    if cf_steps:
        # Use provided draft steps
        return jsonify({"success": True, "cap_wf": cf_steps})
    else:
        # Use saved steps from DB
        data = _get_data()
        result = copy_cf_to_cap(data["wf"], vcode)
        return jsonify(result)


# ── Copy from another entity ─────────────────────────────────────────────

@waterfall_setup_bp.route("/copy-from/<source_vcode>", methods=["GET"])
@login_required
def copy_from(source_vcode):
    """Copy waterfall steps from another entity."""
    data = _get_data()
    result = copy_waterfall_from_entity(source_vcode, data["wf"])
    return jsonify(result)


# ── New Waterfall Templates ──────────────────────────────────────────────

@waterfall_setup_bp.route("/<vcode>/template/new", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def template_new(vcode):
    """Generate new waterfall template from relationships/accounting."""
    data = _get_data()
    result = prefill_new_waterfall(
        vcode, data.get("relationships_raw"), data.get("acct"), data["inv"]
    )
    return jsonify(result)


@waterfall_setup_bp.route("/<vcode>/template/pari-passu", methods=["POST"])
@login_required
@role_required("admin", "analyst")
def template_pari_passu(vcode):
    """Generate pari passu waterfall template."""
    data = _get_data()
    result = prefill_pari_passu_waterfall(
        vcode, data.get("relationships_raw"), data.get("acct"), data["inv"]
    )
    return jsonify(result)


# ── Reference Data ────────────────────────────────────────────────────────

@waterfall_setup_bp.route("/config", methods=["GET"])
@login_required
def config():
    """Return vState options and other configuration for the editor."""
    return jsonify({"vstate_options": VSTATE_OPTIONS})


# ── Export CSV ────────────────────────────────────────────────────────────

@waterfall_setup_bp.route("/<vcode>/export-csv", methods=["GET"])
@login_required
def export_csv(vcode):
    """Export waterfall steps as CSV."""
    data = _get_data()
    wf_data = get_waterfall_steps(data["wf"], vcode)

    rows = []
    for step in wf_data.get("cf_wf", []):
        step["vmisc"] = "CF_WF"
        rows.append(step)
    for step in wf_data.get("cap_wf", []):
        step["vmisc"] = "Cap_WF"
        rows.append(step)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"{vcode}_waterfall.csv")
