"""Guardrail: the Microsoft sign-in callback only ever sends the browser somewhere it can go.

Run: .venv\\Scripts\\python scripts\\sso_redirect_check.py

Oct 8 2026: SSO_REDIRECT_URL was set from Git Bash, which rewrote "/login" to
"C:/Program Files/Git/login". From v580 on, every Microsoft sign-in was redirected
there and Chrome refused it (ERR_UNSAFE_REDIRECT). Checked both ways: a good value
is used as set; a bad one is replaced by /login -- never passed to the browser.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask  # noqa: E402

from flask_app.auth import sso  # noqa: E402

PASSED, FAILED = 0, []


def chk(label, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print("   ok  ", label)
    else:
        FAILED.append(label)
        print("   FAIL", label, detail)


app = Flask(__name__)
CASES = [
    # (value, expected, why)
    ("/login", "/login", "the production value"),
    ("/", "/", "the app root"),
    ("https://app.example.com/login", "https://app.example.com/login", "an absolute web address"),
    ("", "/login", "unset"),
    ("C:/Program Files/Git/login", "/login", "the Git Bash rewrite that broke sign-in"),
    ("c:/Program%20Files/Git/login", "/login", "...as the browser saw it"),
    ("//evil.example.com", "/login", "a protocol-relative address (an open redirect)"),
    ("javascript:alert(1)", "/login", "a script URL"),
    ("file:///etc/passwd", "/login", "a file URL"),
]
with app.app_context():
    for value, want, why in CASES:
        os.environ["SSO_REDIRECT_URL"] = value
        got = sso._frontend_url()
        chk(f"{why}: {value!r} -> {want!r}", got == want, got)
os.environ.pop("SSO_REDIRECT_URL", None)

src = open(os.path.join(ROOT, "flask_app", "auth", "sso.py"), encoding="utf-8").read()
chk("the callback reads the target only through the guard (no raw SSO_REDIRECT_URL read left)",
    src.count('os.environ.get("SSO_REDIRECT_URL"') == 1 and src.count("frontend_url = _frontend_url()") == 2)

print("\n%d passed, %d failed" % (PASSED, len(FAILED)))
for f in FAILED:
    print("  -", f)
sys.exit(1 if FAILED else 0)
