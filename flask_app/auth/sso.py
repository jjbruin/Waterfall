"""SSO authentication via OAuth2/OIDC (Azure AD or Okta).

Disabled by default. Enable by setting SSO_CLIENT_ID in environment.

Flow:
1. GET /auth/sso/login → redirects to identity provider
2. Provider authenticates user → redirects to /auth/sso/callback
3. Callback validates token, matches the EXISTING account by email, issues JWT
   (no account is created -- see ``_match_user``)
4. Redirects to Vue app with token in URL fragment, or ``#sso_error=<reason>``

Configuration (environment / .env):
    SSO_PROVIDER=azure     # "azure" or "okta"
    SSO_CLIENT_ID=...
    SSO_CLIENT_SECRET=...
    SSO_TENANT_ID=...      # Azure AD tenant ID
    SSO_ISSUER=...         # Okta issuer URL (e.g. https://yourorg.okta.com)
    SSO_REDIRECT_URL=/     # Where the callback sends the browser
"""

import os
from flask import Blueprint, redirect, request, current_app, url_for
from authlib.integrations.flask_client import OAuth

from flask_app.auth.routes import _create_token, login_required

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


# The username that holds every section and alone manages access. A Microsoft
# sign-in never opens it: its email is Jim's, so matching on email alone would
# hand the superuser to anyone signing in as him.
SUPERUSER = "admin"


def _match_user(email: str) -> tuple[dict | None, str | None]:
    """The existing account whose EMAIL is this Microsoft sign-in's.

    Returns ``(user, None)`` on exactly one match, else ``(None, reason)``.

    Matched on the ``email`` column, case- and space-insensitive -- usernames
    are short names (``jbruin``), never the email, so matching on username
    would miss every existing account. No account is CREATED: one made here
    would be a second, sectionless account for a person who already has one,
    and who may sign in is decided in User Management, not by Entra. Two
    accounts sharing an email is refused rather than guessed between.
    """
    from sqlalchemy import text
    from flask_app.db import get_engine

    key = (email or "").strip().lower()
    if not key:
        return None, "no_email"
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT id, username, role FROM users "
                 "WHERE lower(trim(email)) = :e AND username <> :su ORDER BY id"),
            {"e": key, "su": SUPERUSER},
        ).mappings().fetchall()
    if not rows:
        return None, "no_account"
    if len(rows) > 1:
        current_app.logger.warning(
            "SSO: %d accounts share email %s (%s) -- refused",
            len(rows), key, ", ".join(r["username"] for r in rows))
        return None, "ambiguous"
    r = rows[0]
    return {"id": r["id"], "username": r["username"], "role": r["role"]}, None


#: Where the browser goes after the callback when SSO_REDIRECT_URL is unset or unusable.
DEFAULT_FRONTEND = "/login"


def _frontend_url() -> str:
    """``SSO_REDIRECT_URL``, but only if a browser can actually go there.

    Allowed: an app path ("/login") or an http(s) address. Anything else is
    refused, logged, and replaced by ``/login``. WHY (Oct 8 2026): the value was
    set from Git Bash, whose path conversion silently turned ``/login`` into
    ``C:/Program Files/Git/login``; every Microsoft sign-in since ``v580`` was
    redirected there, and Chrome refused it (ERR_UNSAFE_REDIRECT) -- nobody could
    sign in with Microsoft, and nothing in the app said why.
    """
    raw = (os.environ.get("SSO_REDIRECT_URL") or "").strip()
    if not raw:
        return DEFAULT_FRONTEND
    low = raw.lower()
    if (raw.startswith("/") and not raw.startswith("//")) or low.startswith("https://") or low.startswith("http://"):
        return raw
    current_app.logger.error("SSO_REDIRECT_URL %r is not an app path or a web address; using %s",
                             raw, DEFAULT_FRONTEND)
    return DEFAULT_FRONTEND


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

        frontend_url = _frontend_url()
        user, reason = _match_user(email)
        if user is None:
            current_app.logger.info("SSO: sign-in for %r refused (%s)", email, reason)
            return redirect(f"{frontend_url}#sso_error={reason}")

        # Issue our JWT
        jwt_token = _create_token(user)

        # Redirect to Vue app with token
        return redirect(f"{frontend_url}#token={jwt_token}")

    except Exception as e:
        current_app.logger.error(f"SSO callback error: {e}")
        frontend_url = _frontend_url()
        return redirect(f"{frontend_url}#sso_error=authentication_failed")


@sso_bp.route("/sharepoint", methods=["GET"])
@login_required
def sharepoint_config():
    """The Entra app the browser's SharePoint picker signs in with.

    The picker runs entirely in the browser (MSAL, delegated Graph read
    scopes): no Microsoft token reaches this server and no secret is needed,
    so it can be on while password sign-in is still the only sign-in.
    ``SHAREPOINT_CLIENT_ID``/``SHAREPOINT_TENANT_ID`` switch it on, falling back
    to the SSO pair -- it is the same app registration. Signed-in users only.
    """
    client_id = os.environ.get("SHAREPOINT_CLIENT_ID") or os.environ.get("SSO_CLIENT_ID")
    tenant_id = os.environ.get("SHAREPOINT_TENANT_ID") or os.environ.get("SSO_TENANT_ID")
    if not (client_id and tenant_id):
        return {"enabled": False}
    return {"enabled": True, "client_id": client_id, "tenant_id": tenant_id}


@sso_bp.route("/config", methods=["GET"])
def sso_config():
    """Return SSO configuration for the frontend (public endpoint)."""
    return {
        "enabled": is_sso_enabled(),
        "provider": os.environ.get("SSO_PROVIDER", "azure") if is_sso_enabled() else None,
    }
