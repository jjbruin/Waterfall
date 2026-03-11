"""Authentication routes: login, logout, me, user management."""

from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import Blueprint, request, jsonify, current_app, g

from flask_app.auth.models import (
    authenticate, get_user_by_id, ensure_default_admin,
    create_user, list_users, update_user_role, delete_user, change_password,
)

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
    """Decorator: require user to have one of the specified roles.

    Usage:
        @login_required
        @role_required("admin")
        def admin_only_route(): ...

        @login_required
        @role_required("admin", "analyst")
        def analyst_or_admin(): ...
    """
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
    token = _create_token(user)
    return jsonify({"token": token, "user": user})


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
    if len(new_pw) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    # Verify current password
    user = authenticate(g.current_user["username"], current_pw)
    if user is None:
        return jsonify({"error": "Current password is incorrect"}), 401

    ok = change_password(g.current_user["id"], new_pw)
    if not ok:
        return jsonify({"error": "Failed to change password"}), 500
    return jsonify({"message": "Password changed"})


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

    Body: { username, password, role? }
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    role = data.get("role", "viewer")

    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    if role not in ROLES:
        return jsonify({"error": f"Invalid role. Must be one of: {', '.join(ROLES)}"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    user = create_user(username, password, role)
    if user is None:
        return jsonify({"error": f"Username '{username}' already exists"}), 409
    return jsonify({"user": user, "message": f"User '{username}' created"}), 201


@auth_bp.route("/users/<int:user_id>/role", methods=["PUT"])
@login_required
@role_required("admin")
def update_role(user_id):
    """Update a user's role (admin only).

    Body: { role }
    """
    # Prevent self-demotion
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
