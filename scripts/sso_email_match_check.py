"""Guardrail: a Microsoft sign-in opens the EXISTING account whose email it is (Oct 6 2026).

Usernames are short names (``jbruin``), never the email, and the original SSO
code matched on username = email -- so every existing person would have got a
second, sectionless viewer account on their first Microsoft sign-in. The
``admin`` username carries Jim's email, so matching on email alone would hand
the superuser to Jim's Microsoft identity.

Drives the real /auth/sso/callback through the Flask test client, the identity
provider's token stubbed. Asserted in BOTH directions:
  1. Admits: the account with that email is opened -- case and spaces ignored --
     and the JWT names it.
  2. Admits the right one: an email shared with ``admin`` opens the other
     account, never ``admin``.
  3. Refuses: no account with the email -> #sso_error=no_account, and NO account
     is created. An email only ``admin`` carries -> no_account. Two non-admin
     accounts sharing an email -> ambiguous. No email -> no_email.
  4. The screen agrees: LoginView names every refusal reason the server sends.

Usage: python scripts/sso_email_match_check.py [--inject=username|admin|create]
  username -- match on username = email (the original code)
  admin    -- the superuser not excluded
  create   -- an account created when none matches
"""
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="sso_email_match_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")
    os.environ["SSO_CLIENT_ID"] = "check-client"
    os.environ["SSO_TENANT_ID"] = "check-tenant"
    os.environ["SSO_PROVIDER"] = "azure"
    os.environ["SSO_REDIRECT_URL"] = "/login"

    import jwt
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth import sso as SSO
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine

    if INJECT == "username":
        def _by_username(email):
            with get_engine().connect() as conn:
                r = conn.execute(text("SELECT id, username, role FROM users WHERE username = :u"),
                                 {"u": (email or "").strip().lower()}).mappings().fetchone()
            return (dict(r), None) if r else (None, "no_account")
        SSO._match_user = _by_username
    if INJECT == "admin":
        SSO.SUPERUSER = "\x00nobody"
    if INJECT == "create":
        _orig = SSO._match_user

        def _creating(email):
            user, reason = _orig(email)
            if reason == "no_account":
                create_user(email.strip().lower(), "x" * 40, role="viewer", email=email)
                return _orig(email)
            return user, reason
        SSO._match_user = _creating

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()

    people = [  # (username, role, email) -- the production shape
        ("admin", "admin", "jbruin@peaceablestreet.com"),
        ("jbruin", "analyst", "jbruin@peaceablestreet.com"),
        ("jday", "analyst", "jday@peaceablestreet.com"),
        ("jstewart", "cfo", "JStewart@PeaceableStreet.com "),
        ("onlyadmin_owner", "analyst", "someone@peaceablestreet.com"),
        ("twin_a", "analyst", "twin@peaceablestreet.com"),
        ("twin_b", "analyst", "twin@peaceablestreet.com"),
    ]
    with app.app_context():
        for u, role, em in people:
            if u == "admin":
                # create_app may already have seeded the admin username
                if not any(x["username"] == "admin" for x in list_users()):
                    create_user(u, "pw-" + u, role=role, email=em)
                else:
                    with get_engine().begin() as c:
                        c.execute(text("UPDATE users SET email = :e WHERE username = 'admin'"), {"e": em})
            else:
                create_user(u, "pw-" + u, role=role, email=em)
        n_before = len(list_users())

    def sign_in(email):
        # never empty: an empty userinfo makes the callback fetch it over the network
        info = {"email": email} if email is not None else {"name": "No Email"}
        SSO.oauth.sso.authorize_access_token = lambda **kw: {"userinfo": info}
        r = client.get("/auth/sso/callback")
        loc = unquote(r.headers.get("Location", ""))
        m = re.search(r"#token=([^&]+)", loc)
        if m:
            p = jwt.decode(m.group(1), app.config["JWT_SECRET"], algorithms=["HS256"])
            return p["username"], None
        e = re.search(r"#sso_error=([^&]+)", loc)
        return None, (e.group(1) if e else "no redirect: %s %s" % (r.status_code, loc))

    print("1. Admits the account with that email")
    u, err = sign_in("jday@peaceablestreet.com")
    chk("jday's email opens jday", u == "jday", (u, err))
    u, err = sign_in("  JDAY@PeaceableStreet.COM ")
    chk("case and spaces ignored", u == "jday", (u, err))
    u, err = sign_in("jstewart@peaceablestreet.com")
    chk("a stored email with odd case/space still matches", u == "jstewart", (u, err))

    print("2. Never the superuser")
    u, err = sign_in("jbruin@peaceablestreet.com")
    chk("Jim's email opens jbruin, not admin", u == "jbruin", (u, err))
    with app.app_context(), get_engine().begin() as c:
        c.execute(text("UPDATE users SET email = 'adminonly@peaceablestreet.com' WHERE username = 'admin'"))
    u, err = sign_in("adminonly@peaceablestreet.com")
    chk("an email only admin carries is refused", u is None and err == "no_account", (u, err))

    print("3. Refuses, and creates nothing")
    u, err = sign_in("stranger@peaceablestreet.com")
    chk("unknown email refused as no_account", u is None and err == "no_account", (u, err))
    with app.app_context():
        n_after = len(list_users())
    chk("no account created by any sign-in", n_after == n_before, (n_before, n_after))
    u, err = sign_in("twin@peaceablestreet.com")
    chk("two accounts sharing an email refused as ambiguous",
        u is None and err == "ambiguous", (u, err))
    u, err = sign_in(None)
    chk("no email refused", u is None and err == "no_email", (u, err))

    print("4. The screen names every refusal")
    vue = (ROOT / "vue_app/src/views/LoginView.vue").read_text(encoding="utf-8")
    for reason in ("no_account", "ambiguous"):
        chk("LoginView handles %s" % reason, "'%s'" % reason in vue)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
