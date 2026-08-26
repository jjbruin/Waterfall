"""Review workflow API — approval pipeline for One Pager reports."""

from flask import Blueprint, request, jsonify, g

from flask_app.auth.routes import login_required, role_required
from flask_app.services.review_service import (
    get_submission, submit_for_review, approve, return_to_draft,
    add_note, acknowledge_note, get_tracking_data, get_investor_list,
    get_user_review_roles, list_review_role_assignments,
    assign_review_role, remove_review_role,
    get_snapshot, REVIEW_STEPS, REVIEW_ROLE_NAMES,
)

reviews_bp = Blueprint("reviews", __name__)


def _enrich_submission(sub: dict) -> dict:
    """Add user permissions and action flags to a submission dict."""
    user_roles = get_user_review_roles(g.current_user["id"])
    sub["user_review_roles"] = user_roles
    step = sub["current_step"]
    step_role = sub.get("current_step_role")
    sub["can_submit"] = (
        sub["status"] in ("draft", "returned")
        and "asset_manager" in user_roles
    )
    sub["can_approve"] = (
        1 <= step <= 4
        and step_role in user_roles
    )
    sub["can_return"] = sub["can_approve"]
    sub["is_editable"] = sub["status"] != "approved" or sub["id"] is None
    sub["has_snapshot"] = get_snapshot(sub["vcode"], sub["quarter"]) is not None
    return sub


@reviews_bp.route("/<vcode>/<quarter>", methods=["GET"])
@login_required
def get_review_status(vcode, quarter):
    """Get submission status, notes, and user permissions for this document."""
    try:
        sub = get_submission(vcode, quarter)
        return jsonify(_enrich_submission(sub))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reviews_bp.route("/<vcode>/<quarter>/submit", methods=["POST"])
@login_required
def submit(vcode, quarter):
    """Submit for review (draft/returned -> pending_head_am)."""
    body = request.get_json(silent=True) or {}
    try:
        result = submit_for_review(
            vcode, quarter,
            g.current_user["id"], g.current_user["username"],
            note_text=body.get("note"),
        )
        return jsonify(_enrich_submission(result))
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@reviews_bp.route("/<vcode>/<quarter>/approve", methods=["POST"])
@login_required
def approve_review(vcode, quarter):
    """Approve at current step and advance."""
    body = request.get_json(silent=True) or {}
    try:
        result = approve(
            vcode, quarter,
            g.current_user["id"], g.current_user["username"],
            note_text=body.get("note"),
        )
        return jsonify(_enrich_submission(result))
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@reviews_bp.route("/<vcode>/<quarter>/return", methods=["POST"])
@login_required
def return_review(vcode, quarter):
    """Return to draft (note required)."""
    body = request.get_json(silent=True) or {}
    try:
        result = return_to_draft(
            vcode, quarter,
            g.current_user["id"], g.current_user["username"],
            note_text=body.get("note", ""),
        )
        return jsonify(_enrich_submission(result))
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@reviews_bp.route("/<vcode>/<quarter>/note", methods=["POST"])
@login_required
def post_note(vcode, quarter):
    """Add a discussion note."""
    body = request.get_json(silent=True) or {}
    try:
        result = add_note(
            vcode, quarter,
            g.current_user["id"], g.current_user["username"],
            note_text=body.get("note", ""),
        )
        return jsonify(_enrich_submission(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@reviews_bp.route("/<vcode>/<quarter>/notes/<int:note_id>/acknowledge", methods=["POST"])
@login_required
def acknowledge_review_note(vcode, quarter, note_id):
    """Toggle a review note as addressed/unaddressed (asset manager only)."""
    try:
        result = acknowledge_note(
            note_id,
            g.current_user["id"], g.current_user["username"],
        )
        return jsonify(_enrich_submission(result))
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@reviews_bp.route("/tracking", methods=["GET"])
@login_required
def tracking():
    """Get production tracking data with optional filters."""
    quarter = request.args.get("quarter")
    status = request.args.get("status")
    investor = request.args.get("investor")
    try:
        data = get_tracking_data(quarter_filter=quarter, status_filter=status,
                                 investor_filter=investor)
        return jsonify({"items": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reviews_bp.route("/investors", methods=["GET"])
@login_required
def investors():
    """Upstream investors for the tracking filter dropdown, as {code, name}.

    The CODE is what /tracking filters on — its recursive CTE binds
    ``TRIM(r."InvestorID") = :inv`` against the relationships feed, so only a
    real investor ID matches. The NAME is display only.

    That distinction is the whole point of returning pairs. TIAA reaches its 35
    deals as ``TGAM``; an option whose value were the literal string "TIAA"
    would match no relationship row and silently render an empty table. So the
    dropdown shows "TIAA" and still submits "TGAM".

    Shape and sort mirror /api/portfolio-snapshot/investors, which already does
    this — same helper, same ordering by display name — so the two dropdowns
    cannot disagree about what an investor is called.
    """
    from flask_app.services.portfolio_snapshot_service import get_investor_name

    try:
        out = [{"code": c, "name": get_investor_name(c)}
               for c in (get_investor_list() or [])]
        out.sort(key=lambda r: (r["name"] or r["code"]).lower())
        return jsonify({"investors": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Role management (admin only) ────────────────────────────

@reviews_bp.route("/roles", methods=["GET"])
@login_required
@role_required("admin")
def get_roles():
    """List all review role assignments."""
    try:
        assignments = list_review_role_assignments()
        return jsonify({
            "assignments": assignments,
            "available_roles": REVIEW_ROLE_NAMES,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reviews_bp.route("/roles", methods=["POST"])
@login_required
@role_required("admin")
def create_role():
    """Assign a review role to a user."""
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id")
    review_role = body.get("review_role")
    if not user_id or not review_role:
        return jsonify({"error": "user_id and review_role required"}), 400
    try:
        result = assign_review_role(user_id, review_role)
        if result is None:
            return jsonify({"error": "Assignment already exists or failed"}), 409
        return jsonify(result), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@reviews_bp.route("/roles/<int:role_id>", methods=["DELETE"])
@login_required
@role_required("admin")
def delete_role(role_id):
    """Remove a review role assignment."""
    ok = remove_review_role(role_id)
    if not ok:
        return jsonify({"error": "Assignment not found"}), 404
    return jsonify({"message": "Role assignment removed"})
