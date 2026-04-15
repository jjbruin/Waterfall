"""Authentication routes: login, logout, me, user management, password reset."""

from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import Blueprint, request, jsonify, current_app, g

from flask_app.auth.models import (
    authenticate, get_user_by_id, ensure_default_admin,
    create_user, list_users, update_user_role, update_user_email,
    delete_user, change_password,
    get_user_by_email, create_reset_token, validate_reset_token, consume_reset_token,
)
from flask_app.auth.email_utils import send_password_reset_email, send_password_changed_email, send_welcome_email

auth_bp = Blueprint("auth", __name__)

# ── Valid roles (ordered by privilege level) ──────────────────────────
ROLES = ("viewer", "analyst", "admin")


def _create_token(user: dict) -> str:
    """Create a JWT for the given user."""
    exp = datetime.now(timezone.utc) + timedelta(
        hours=current_app.config["JWT_EXPIRATION_HOURS"]
    )
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "exp": exp,
    }
    return jwt.encode(payload, current_app.config["JWT_SECRET"], algorithm="HS256")


def login_required(f):
    """Decorator: require valid JWT in Authorization header or query param."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        # Fallback: token in query param (needed for EventSource/SSE)
        if not token:
            token = request.args.get("token")
        if not token:
            return jsonify({"error": "Missing token"}), 401
        try:
            payload = jwt.decode(
                token, current_app.config["JWT_SECRET"], algorithms=["HS256"]
            )
            g.current_user = {
                "id": int(payload["sub"]),
                "username": payload["username"],
                "role": payload["role"],
            }
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated


def role_required(*allowed_roles):
    """Decorator: require user to have one of the specified roles."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user_role = g.current_user.get("role", "viewer")
            if user_role not in allowed_roles:
                return jsonify({
                    "error": "Forbidden",
                    "message": f"Role '{user_role}' does not have access. Required: {', '.join(allowed_roles)}",
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ── Auth routes ──────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate and return JWT."""
    ensure_default_admin()
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    user = authenticate(username, password)
    if user is None:
        return jsonify({"error": "Invalid credentials"}), 401

    # Check if user must change password on first login
    if user.get("must_change_password"):
        return jsonify({
            "must_change_password": True,
            "user_id": user["id"],
            "username": user["username"],
        }), 200

    token = _create_token(user)
    return jsonify({"token": token, "user": {
        "id": user["id"], "username": user["username"], "role": user["role"],
    }})


@auth_bp.route("/force-change-password", methods=["POST"])
def force_change_password():
    """Change password for a user who must change on first login."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    current_pw = data.get("current_password", "")
    new_pw = data.get("new_password", "")

    if not username or not current_pw or not new_pw:
        return jsonify({"error": "username, current_password, and new_password required"}), 400
    if len(new_pw) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if new_pw == current_pw:
        return jsonify({"error": "New password must be different from current password"}), 400

    user = authenticate(username, current_pw)
    if user is None:
        return jsonify({"error": "Invalid credentials"}), 401

    ok = change_password(user["id"], new_pw, clear_must_change=True)
    if not ok:
        return jsonify({"error": "Failed to change password"}), 500

    # Send confirmation email
    full_user = get_user_by_id(user["id"])
    if full_user:
        from flask_app.auth.models import get_user_by_email
        from flask_app.db import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT email FROM users WHERE id = :id"), {"id": user["id"]}
            ).mappings().fetchone()
        if row and row.get("email"):
            send_password_changed_email(row["email"], user["username"])

    # Now issue a token
    updated_user = get_user_by_id(user["id"])
    token = _create_token(updated_user)
    return jsonify({"token": token, "user": {
        "id": updated_user["id"], "username": updated_user["username"],
        "role": updated_user["role"],
    }})


@auth_bp.route("/me", methods=["GET"])
@login_required
def me():
    """Return current user info."""
    return jsonify({"user": g.current_user})


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """Logout (client-side token discard; server acknowledges)."""
    return jsonify({"message": "Logged out"})


@auth_bp.route("/change-password", methods=["POST"])
@login_required
def change_own_password():
    """Change current user's password."""
    data = request.get_json(silent=True) or {}
    current_pw = data.get("current_password", "")
    new_pw = data.get("new_password", "")
    if not current_pw or not new_pw:
        return jsonify({"error": "current_password and new_password required"}), 400
    if len(new_pw) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    # Verify current password
    user = authenticate(g.current_user["username"], current_pw)
    if user is None:
        return jsonify({"error": "Current password is incorrect"}), 401

    ok = change_password(g.current_user["id"], new_pw)
    if not ok:
        return jsonify({"error": "Failed to change password"}), 500

    # Send confirmation email
    from flask_app.db import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT email FROM users WHERE id = :id"), {"id": g.current_user["id"]}
        ).mappings().fetchone()
    if row and row.get("email"):
        send_password_changed_email(row["email"], g.current_user["username"])

    return jsonify({"message": "Password changed"})


# ── Forgot / Reset password ──────────────────────────────────────────

@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Request a password reset link via email."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email address required"}), 400

    # Always return success to prevent email enumeration
    user = get_user_by_email(email)
    if user:
        token = create_reset_token(user["id"])
        send_password_reset_email(email, user["username"], token)

    return jsonify({"message": "If an account with that email exists, a reset link has been sent."})


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    """Reset password using a valid token."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "").strip()
    new_pw = data.get("new_password", "")

    if not token or not new_pw:
        return jsonify({"error": "token and new_password required"}), 400
    if len(new_pw) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    user = validate_reset_token(token)
    if user is None:
        return jsonify({"error": "Invalid or expired reset link. Please request a new one."}), 400

    ok = change_password(user["user_id"], new_pw)
    if not ok:
        return jsonify({"error": "Failed to reset password"}), 500

    consume_reset_token(token)

    # Send confirmation
    if user.get("email"):
        send_password_changed_email(user["email"], user["username"])

    return jsonify({"message": "Password has been reset. You can now log in with your new password."})


@auth_bp.route("/validate-reset-token", methods=["POST"])
def check_reset_token():
    """Check if a reset token is still valid (for frontend UX)."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"valid": False}), 400
    user = validate_reset_token(token)
    if user is None:
        return jsonify({"valid": False, "error": "Invalid or expired reset link"}), 400
    return jsonify({"valid": True, "username": user["username"]})


# ── User management (admin only) ────────────────────────────────────

@auth_bp.route("/users", methods=["GET"])
@login_required
@role_required("admin")
def get_users():
    """List all users (admin only)."""
    users = list_users()
    return jsonify({"users": users})


@auth_bp.route("/users", methods=["POST"])
@login_required
@role_required("admin")
def create_new_user():
    """Create a new user (admin only).

    Body: { username, password, role?, email?, must_change_password? }
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    role = data.get("role", "viewer")
    email = data.get("email", "").strip() or None
    must_change = data.get("must_change_password", False)

    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    if role not in ROLES:
        return jsonify({"error": f"Invalid role. Must be one of: {', '.join(ROLES)}"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    send_welcome = data.get("send_welcome_email", False)

    user = create_user(username, password, role, email=email, must_change_password=must_change)
    if user is None:
        return jsonify({"error": f"Username '{username}' already exists"}), 409

    email_sent = False
    if send_welcome and email:
        email_sent = send_welcome_email(email, username, password)

    msg = f"User '{username}' created"
    if send_welcome and email:
        msg += " — welcome email sent" if email_sent else " — welcome email failed (check SMTP config)"
    return jsonify({"user": user, "message": msg, "email_sent": email_sent}), 201


@auth_bp.route("/users/<int:user_id>/role", methods=["PUT"])
@login_required
@role_required("admin")
def update_role(user_id):
    """Update a user's role (admin only)."""
    if user_id == g.current_user["id"]:
        return jsonify({"error": "Cannot change your own role"}), 400

    data = request.get_json(silent=True) or {}
    role = data.get("role", "")
    if role not in ROLES:
        return jsonify({"error": f"Invalid role. Must be one of: {', '.join(ROLES)}"}), 400

    ok = update_user_role(user_id, role)
    if not ok:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"message": f"Role updated to '{role}'"})


@auth_bp.route("/users/<int:user_id>/send-welcome", methods=["POST"])
@login_required
@role_required("admin")
def send_welcome(user_id):
    """Send (or resend) welcome email to a user. Resets password and sets must_change_password."""
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Get user's email
    from flask_app.db import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT email FROM users WHERE id = :id"), {"id": user_id}
        ).mappings().fetchone()
    email = dict(row).get("email") if row else None
    if not email:
        return jsonify({"error": "User has no email address"}), 400

    # Reset to temporary password and require change
    temp_pw = "password"
    change_password(user_id, temp_pw, clear_must_change=False)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET must_change_password = TRUE WHERE id = :id"),
            {"id": user_id},
        )

    sent = send_welcome_email(email, user["username"], temp_pw)
    if sent:
        return jsonify({"message": f"Welcome email sent to {email}"})
    else:
        return jsonify({"error": "Password reset but email failed to send (check SMTP config)"}), 500


@auth_bp.route("/users/<int:user_id>/email", methods=["PUT"])
@login_required
@role_required("admin")
def update_email(user_id):
    """Update a user's email (admin only)."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400

    ok = update_user_email(user_id, email)
    if not ok:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"message": "Email updated"})


@auth_bp.route("/users/<int:user_id>", methods=["DELETE"])
@login_required
@role_required("admin")
def remove_user(user_id):
    """Delete a user (admin only)."""
    if user_id == g.current_user["id"]:
        return jsonify({"error": "Cannot delete yourself"}), 400

    ok = delete_user(user_id)
    if not ok:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"message": "User deleted"})


@auth_bp.route("/roles", methods=["GET"])
@login_required
def get_roles():
    """Get available roles and their descriptions."""
    return jsonify({"roles": [
        {"name": "viewer", "description": "Read-only access to all views and reports"},
        {"name": "analyst", "description": "Run computations, edit waterfalls, generate reports"},
        {"name": "admin", "description": "Full access: manage users, import data, configure system"},
    ]})
