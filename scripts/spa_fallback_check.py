"""Guardrail: the SPA fallback answers a missing FILE with 404, an app route with the page.

Oct 7 2026: Jim added the expense app to his iPhone home screen and was not offered the
PSC Expenses icon. The server answered every unknown path with index.html and a 200, so
/apple-touch-icon-precomposed.png -- a name iOS asks for -- came back "found", as HTML.
A file that is not there must be a 404; an app route (no file extension, or a dot that is
not a file type) must still get the page, or deep links break.

Both directions are asserted: a rule tested only for 404s is satisfied by refusing every
route, and one tested only for routes by serving HTML for everything.

Usage: python scripts/spa_fallback_check.py [--inject=all|none|unwired]
       python scripts/spa_fallback_check.py --live      # also probe production
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
LIVE = "--live" in sys.argv
APP = "https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io"
_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def main():
    import flask_app
    rule = flask_app.is_missing_file_request
    if INJECT == "all":        # everything is a missing file: deep links 404
        rule = lambda p: True
    if INJECT == "none":       # nothing is: the old behaviour, HTML for an icon
        rule = lambda p: False
    src = (ROOT / "flask_app/__init__.py").read_text(encoding="utf-8")
    if INJECT == "unwired":
        src = src.replace("if is_missing_file_request(path):", "if False:")

    print("1. A missing file is a 404")
    for p in ("apple-touch-icon-precomposed.png", "apple-touch-icon-180x180.png", "favicon.ico",
              "assets/ExpensesView-OLD123.js", "assets/index-OLD.css", "manifest.webmanifest",
              "robots.txt", "Apple-Touch-Icon.PNG"):
        chk(f"/{p} -> 404", rule(p) is True)

    print("2. An app route still gets the page")
    for p in ("", "expenses", "login", "deals/P0000044", "valuations/2026.06",
              "reports/pref-balance", "users/j.bruin"):
        chk(f"/{p} -> the app", rule(p) is False)

    print("3. serve_spa asks the rule, after real files and before the page")
    body = src[src.find("def serve_spa(path):"):src.find("else:\n        # Development mode")]
    f, m, i = (body.find("os.path.isfile(file_path)"), body.find("if is_missing_file_request(path):"),
               body.find('send_from_directory(static_dir, "index.html")'))
    chk("wired into serve_spa", m > 0, m)
    chk("...after serving a file that exists, before falling back to index.html",
        0 < f < m < i, (f, m, i))
    chk("...and answers 404", re.search(r"if is_missing_file_request\(path\):\s*return [^\n]*404", body)
        is not None)

    print("4. The icon is shipped under both names iOS uses")
    pub = ROOT / "vue_app/public"
    a, b = pub / "apple-touch-icon.png", pub / "apple-touch-icon-precomposed.png"
    chk("apple-touch-icon.png and -precomposed.png both exist and are the same image",
        a.exists() and b.exists() and a.read_bytes() == b.read_bytes())

    if LIVE:
        import urllib.request
        import urllib.error
        print("5. Production")

        def get(p):
            try:
                r = urllib.request.urlopen(urllib.request.Request(APP + p, headers={
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)"}), timeout=30)
                return r.status, r.headers.get("Content-Type", "")
            except urllib.error.HTTPError as e:
                return e.code, e.headers.get("Content-Type", "")
        for p in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"):
            st, ct = get(p)
            chk(f"{p} is a PNG", st == 200 and ct.startswith("image/png"), (st, ct))
        st, ct = get("/apple-touch-icon-180x180.png")
        chk("/apple-touch-icon-180x180.png (not shipped) is a 404, not HTML", st == 404, (st, ct))
        st, ct = get("/expenses")
        chk("/expenses is the app", st == 200 and "text/html" in ct, (st, ct))

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
