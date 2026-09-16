"""Ownership API — tree visualization, waterfall requirements, upstream analysis."""

from datetime import datetime

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
    return data_service.get_data()


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

    # The caller says whether this is operating cash or a capital event. The
    # two run different waterfalls and only one reduces capital outstanding, so
    # defaulting silently would model a sale as an operating distribution. The
    # literals match `vmisc` in the waterfalls table exactly.
    wf_type = str(body.get("wf_type", "CF_WF")).strip()
    if wf_type not in ("CF_WF", "Cap_WF"):
        return jsonify({"error": f"wf_type must be CF_WF or Cap_WF, not {wf_type!r}"}), 400

    # The date the distribution is assumed to happen. It decides how much pref
    # has accrued by then, so it is the user's to set; today is the default
    # because "what if we distributed now" is the question this screen answers.
    as_of = None
    raw_as_of = str(body.get("as_of", "") or "").strip()
    if raw_as_of:
        try:
            as_of = datetime.strptime(raw_as_of, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": f"as_of must be YYYY-MM-DD, not {raw_as_of!r}"}), 400

    if not entity_id:
        return jsonify({"error": "entity_id is required"}), 400

    data = _get_data()
    result = run_upstream_analysis(
        entity_id=entity_id,
        distribution_amount=distribution_amount,
        relationships_raw=data["relationships_raw"],
        wf=data["wf"],
        inv=data["inv"],
        wf_type=wf_type,
        # The accounting the engine seeds from. Without it the waterfall starts
        # from zero and ignores every accrued pref balance on the deal.
        acct=data.get("acct"),
        actuals_through=current_app.config.get("ACTUALS_THROUGH"),
        as_of=as_of,
    )

    if "error" in result:
        return jsonify(result), 400

    return jsonify(safe_json(result))


# ── Ownership chain from commitments (Investment Management) ────────────
#
# Separate from the endpoints above on purpose. Those walk `relationships`
# and its stored OwnershipPct; these derive ownership from committed dollars
# via ownership_chain_service, which is the source that proved correct when
# the two disagreed. Both are kept because the older tree still backs the
# waterfall-requirements analysis; see the chain service's docstring.

@ownership_bp.route("/chain/investments", methods=["GET"])
@login_required
def chain_investments():
    """The PE investment level — the left edge of the ownership tree."""
    from flask_app.services.ownership_chain_service import list_pe_investments
    try:
        return jsonify(safe_json({"investments": list_pe_investments()}))
    except Exception as e:
        current_app.logger.exception("chain_investments failed")
        return jsonify({"error": str(e)}), 500


@ownership_bp.route("/chain/<investment_id>", methods=["GET"])
@login_required
def chain(investment_id):
    """The full ownership chain above one PE investment, up to OWPSC."""
    from flask_app.services.ownership_chain_service import build_chain
    try:
        return jsonify(safe_json(build_chain(investment_id)))
    except Exception as e:
        current_app.logger.exception("chain failed for %s", investment_id)
        return jsonify({"error": str(e)}), 500
