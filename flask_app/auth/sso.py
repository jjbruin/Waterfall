"""SSO authentication via OAuth2/OIDC (Azure AD or Okta).

Disabled by default. Enable by setting SSO_CLIENT_ID in environment.

Flow:
1. GET /auth/sso/login → redirects to identity provider
2. Provider authenticates user → redirects to /auth/sso/callback
3. Callback validates token, creates/finds user, issues JWT
4. Redirects to Vue app with token in URL fragment

Configuration (environment / .env):
    SSO_PROVIDER=azure     # "azure" or "okta"
    SSO_CLIENT_ID=...
    SSO_CLIENT_SECRET=...
    SSO_TENANT_ID=...      # Azure AD tenant ID
    SSO_ISSUER=...         # Okta issuer URL (e.g. https://yourorg.okta.com)
    SSO_DEFAULT_ROLE=viewer  # Role assigned to new SSO users
"""

import os
from flask import Blueprint, redirect, request, current_app, url_for
from authlib.integrations.flask_client import OAuth

from flask_app.auth.models import get_user_by_id, create_user
from flask_app.auth.routes import _create_token

sso_bp = Blueprint("sso", __name__)
oauth = OAuth()


def is_sso_enabled() -> bool:
    """Check if SSO is configured."""
    return bool(os.environ.get("SSO_CLIENT_ID"))


def init_sso(app):
    """Register SSO provider with the Flask app. No-op if SSO is not configured."""
    if not is_sso_enabled():
        return

    oauth.init_app(app)

    provider = os.environ.get("SSO_PROVIDER", "azure").lower()

    if provider == "azure":
        tenant_id = os.environ.get("SSO_TENANT_ID", "common")
        oauth.register(
            name="sso",
            client_id=os.environ["SSO_CLIENT_ID"],
            client_secret=os.environ.get("SSO_CLIENT_SECRET", ""),
            server_metadata_url=f"https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    elif provider == "okta":
        issuer = os.environ.get("SSO_ISSUER", "")
        if not issuer:
            raise ValueError("SSO_ISSUER required for Okta provider")
        oauth.register(
            name="sso",
            client_id=os.environ["SSO_CLIENT_ID"],
            client_secret=os.environ.get("SSO_CLIENT_SECRET", ""),
            server_metadata_url=f"{issuer.rstrip('/')}/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    else:
        raise ValueError(f"Unknown SSO_PROVIDER: {provider}. Use 'azure' or 'okta'.")


def _find_or_create_user(email: str, name: str) -> dict:
    """Find existing user by email/username or create a new one.

    SSO users are matched by username = email (lowercase).
    New users get the default role from SSO_DEFAULT_ROLE.
    """
    from sqlalchemy import text
    from flask_app.db import get_engine

    username = email.lower()
    engine = get_engine()

    # Look up by username
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, username, role FROM users WHERE username = :u"),
            {"u": username},
        ).mappings().fetchone()

    if row:
        return {"id": row["id"], "username": row["username"], "role": row["role"]}

    # Create new SSO user (no password — SSO-only)
    default_role = os.environ.get("SSO_DEFAULT_ROLE", "viewer")
    # Use a random non-guessable password since SSO users don't use password auth
    import secrets
    result = create_user(username, secrets.token_hex(32), role=default_role)
    if result is None:
        # Race condition — another request created the user
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, username, role FROM users WHERE username = :u"),
                {"u": username},
            ).mappings().fetchone()
        return {"id": row["id"], "username": row["username"], "role": row["role"]}

    # Get the created user's ID
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, username, role FROM users WHERE username = :u"),
            {"u": username},
        ).mappings().fetchone()
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


# ── SSO Routes ──────────────────────────────────────────────────────

@sso_bp.route("/login", methods=["GET"])
def sso_login():
    """Redirect to identity provider for authentication."""
    if not is_sso_enabled():
        return {"error": "SSO not configured"}, 404

    redirect_uri = url_for("sso.sso_callback", _external=True)
    return oauth.sso.authorize_redirect(redirect_uri)


@sso_bp.route("/callback", methods=["GET"])
def sso_callback():
    """Handle OAuth2 callback from identity provider."""
    if not is_sso_enabled():
        return {"error": "SSO not configured"}, 404

    try:
        token = oauth.sso.authorize_access_token()
        userinfo = token.get("userinfo") or oauth.sso.userinfo()

        email = userinfo.get("email") or userinfo.get("preferred_username", "")
        name = userinfo.get("name", email)

        if not email:
            return {"error": "No email in SSO response"}, 400

        # Find or create local user
        user = _find_or_create_user(email, name)

        # Issue our JWT
        jwt_token = _create_token(user)

        # Redirect to Vue app with token
        frontend_url = os.environ.get("SSO_REDIRECT_URL", "/")
        return redirect(f"{frontend_url}#token={jwt_token}")

    except Exception as e:
        current_app.logger.error(f"SSO callback error: {e}")
        frontend_url = os.environ.get("SSO_REDIRECT_URL", "/")
        return redirect(f"{frontend_url}#sso_error=authentication_failed")


@sso_bp.route("/config", methods=["GET"])
def sso_config():
    """Return SSO configuration for the frontend (public endpoint)."""
    return {
        "enabled": is_sso_enabled(),
        "provider": os.environ.get("SSO_PROVIDER", "azure") if is_sso_enabled() else None,
    }
