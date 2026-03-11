"""Ownership API — tree visualization, waterfall requirements, upstream analysis."""

from flask import Blueprint, request, jsonify, current_app

from flask_app.auth.routes import login_required
from flask_app.services import data_service
from flask_app.services.ownership_service import (
    get_ownership_tree, get_entity_tree_text, get_entity_investors,
    get_waterfall_requirements, run_upstream_analysis,
)
from flask_app.serializers import safe_json

ownership_bp = Blueprint("ownership", __name__)


def _get_data():
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


@ownership_bp.route("/tree", methods=["GET"])
@login_required
def tree():
    """Get full ownership tree."""
    data = _get_data()
    tree_data = get_ownership_tree(data["relationships_raw"])
    return jsonify(safe_json(tree_data))


@ownership_bp.route("/tree/<entity_id>", methods=["GET"])
@login_required
def entity_tree(entity_id):
    """Get ownership tree visualization for a specific entity."""
    data = _get_data()
    max_depth = request.args.get("max_depth", 20, type=int)
    result = get_entity_tree_text(data["relationships_raw"], entity_id, max_depth=max_depth)
    return jsonify(safe_json(result))


@ownership_bp.route("/<entity_id>/investors", methods=["GET"])
@login_required
def investors(entity_id):
    """Get direct investors for an entity."""
    data = _get_data()
    investor_list = get_entity_investors(data["relationships_raw"], entity_id)
    return jsonify({"investors": investor_list})


@ownership_bp.route("/requirements", methods=["GET"])
@login_required
def requirements():
    """Get entities that need waterfall definitions."""
    data = _get_data()
    reqs = get_waterfall_requirements(data["relationships_raw"], data["inv"])
    return jsonify({"requirements": reqs})


@ownership_bp.route("/upstream-analysis", methods=["POST"])
@login_required
def upstream_analysis():
    """Run upstream waterfall analysis for an entity.

    Body: { entity_id, distribution_amount }
    """
    body = request.get_json(force=True)
    entity_id = body.get("entity_id", "")
    distribution_amount = float(body.get("distribution_amount", 100000))

    if not entity_id:
        return jsonify({"error": "entity_id is required"}), 400

    data = _get_data()
    result = run_upstream_analysis(
        entity_id=entity_id,
        distribution_amount=distribution_amount,
        relationships_raw=data["relationships_raw"],
        wf=data["wf"],
        inv=data["inv"],
    )

    if "error" in result:
        return jsonify(result), 400

    return jsonify(safe_json(result))
