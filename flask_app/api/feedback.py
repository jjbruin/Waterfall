"""Feedback & request tracking API endpoints."""

import logging
from flask import Blueprint, request, jsonify, g

from flask_app.auth.routes import login_required, role_required
from flask_app.services.feedback_service import (
    create_request, get_request, list_requests,
    update_request_status, add_message, send_request_email,
    get_request_by_token, handle_inbound_email, export_all_requests,
    REQUEST_TYPES, REQUEST_STATUSES, PRIORITY_LEVELS,
)

log = logging.getLogger(__name__)

feedback_bp = Blueprint("feedback", __name__)


# ── User endpoints ──────────────────────────────────────────

@feedback_bp.route("", methods=["POST"])
@login_required
def submit_request():
    """Submit a new feedback request."""
    body = request.get_json(silent=True) or {}
    required = ["request_type", "title", "description"]
    missing = [f for f in required if not body.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        result = create_request(
            user_id=g.current_user["id"],
            username=g.current_user["username"],
            request_type=body["request_type"],
            title=body["title"],
            description=body["description"],
            priority=body.get("priority", "medium"),
            page_context=body.get("page_context"),
            deal_context=body.get("deal_context"),
        )
        return jsonify(result), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@feedback_bp.route("", methods=["GET"])
@login_required
def get_requests():
    """List requests. Non-admin users see only their own."""
    status = request.args.get("status")
    request_type = request.args.get("type")

    # Admin sees all; others see only their own
    user_id = None
    if g.current_user.get("role") != "admin":
        user_id = g.current_user["id"]

    items = list_requests(user_id=user_id, status=status,
                          request_type=request_type)
    return jsonify({
        "items": items,
        "types": REQUEST_TYPES,
        "statuses": REQUEST_STATUSES,
        "priorities": PRIORITY_LEVELS,
    })


@feedback_bp.route("/<int:request_id>", methods=["GET"])
@login_required
def get_single(request_id):
    """Get a single request with full message thread."""
    try:
        req = get_request(request_id)
        # Non-admin can only see their own
        if (g.current_user.get("role") != "admin"
                and req["user_id"] != g.current_user["id"]):
            return jsonify({"error": "Not authorized"}), 403
        return jsonify(req)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@feedback_bp.route("/<int:request_id>/messages", methods=["POST"])
@login_required
def post_message(request_id):
    """Add a message to a request thread (user reply)."""
    body = request.get_json(silent=True) or {}
    message = body.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message required"}), 400

    try:
        req = get_request(request_id)
        # Verify ownership or admin
        if (g.current_user.get("role") != "admin"
                and req["user_id"] != g.current_user["id"]):
            return jsonify({"error": "Not authorized"}), 403

        sender_type = "admin" if g.current_user.get("role") == "admin" else "user"
        result = add_message(
            request_id, sender_type,
            g.current_user["username"], message,
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ── Reply via token (from email link) ───────────────────────

@feedback_bp.route("/reply/<reply_token>", methods=["GET"])
@login_required
def get_by_token(reply_token):
    """Look up a request by reply token (from email link)."""
    try:
        req = get_request_by_token(reply_token)
        return jsonify(req)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ── Admin endpoints ─────────────────────────────────────────

@feedback_bp.route("/<int:request_id>/resolve", methods=["POST"])
@login_required
def resolve_own_request(request_id):
    """Allow the initiator to mark their own request as resolved."""
    try:
        req = get_request(request_id)
        if req["user_id"] != g.current_user["id"]:
            return jsonify({"error": "Not authorized"}), 403
        if req["status"] in ("resolved", "closed"):
            return jsonify({"error": "Already resolved"}), 400
        result = update_request_status(
            request_id, "resolved",
            admin_name=g.current_user["username"],
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@feedback_bp.route("/<int:request_id>/status", methods=["PUT"])
@login_required
@role_required("admin")
def change_status(request_id):
    """Update request status (admin only)."""
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    if not status:
        return jsonify({"error": "status required"}), 400

    try:
        result = update_request_status(
            request_id, status,
            admin_name=g.current_user["username"],
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@feedback_bp.route("/<int:request_id>/email", methods=["POST"])
@login_required
@role_required("admin")
def email_user(request_id):
    """Send an email to the request submitter (admin only)."""
    body = request.get_json(silent=True) or {}
    message = body.get("message", "").strip()
    if not message:
        return jsonify({"error": "message required"}), 400

    try:
        sent = send_request_email(
            request_id,
            admin_name=g.current_user["username"],
            message=message,
            subject=body.get("subject"),
        )
        return jsonify({"sent": sent, "request": get_request(request_id)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@feedback_bp.route("/export", methods=["GET"])
@login_required
@role_required("admin")
def export():
    """Export all requests with threads for design sessions."""
    data = export_all_requests()
    return jsonify({"requests": data, "count": len(data)})


# ── Inbound email webhook (SendGrid Inbound Parse) ─────────

@feedback_bp.route("/inbound-email", methods=["POST"])
def inbound_email():
    """Webhook for SendGrid Inbound Parse.

    SendGrid POSTs form data with: from, to, subject, text, html, etc.
    The reply token is extracted from the 'to' address:
      requests+<token>@domain.com
    """
    try:
        to_addr = request.form.get("to", "")
        from_addr = request.form.get("from", "")
        body_text = request.form.get("text", "")

        # Extract token from to address: requests+TOKEN@domain.com
        token = None
        if "+" in to_addr and "@" in to_addr:
            local_part = to_addr.split("@")[0]
            token = local_part.split("+", 1)[1]

        if not token:
            log.warning("Inbound email with no reply token: to=%s", to_addr)
            return jsonify({"error": "No reply token found"}), 400

        # Clean up quoted text from replies
        lines = body_text.split("\n")
        clean_lines = []
        for line in lines:
            if line.startswith(">") or line.startswith("On ") and "wrote:" in line:
                break
            clean_lines.append(line)
        clean_body = "\n".join(clean_lines).strip()

        if not clean_body:
            clean_body = body_text.strip()

        result = handle_inbound_email(token, from_addr, clean_body)
        return jsonify({"ok": True}), 200
    except ValueError as e:
        log.warning("Inbound email error: %s", e)
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        log.error("Inbound email processing error: %s", e)
        return jsonify({"error": "Internal error"}), 500
