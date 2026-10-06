"""Guardrail: the "From SharePoint" picker (Oct 6 2026).

The picker runs in the browser: MSAL signs the user in to the Waterfall XIRR
Entra app with delegated Graph scopes, the files they tick are downloaded as
browser File objects, and the host screen hands them to the SAME function its
own file input uses. What must stay true:

  1. The server says whether the picker is on, to signed-in users only:
     401 without a token; off when no Entra app is configured; on, with the
     client and tenant ids, when SHAREPOINT_* or (falling back) SSO_* are set.
  2. The scopes are READ-ONLY. A picker that asked for write access would let
     a bug in this app change SharePoint.
  3. No Microsoft token goes to our server: the service talks to our API only
     to read its configuration.
  4. The popup's return page is in the BUILD and runs MSAL's redirect bridge --
     MSAL 5 requires it, and without the page production 404s there and no
     token is ever granted. The redirect URI the code uses is that page.
  5. ONE PATH PER IMPORTER: on every host screen the picker emits into the very
     function the screen's own file input feeds, so a SharePoint file is
     imported exactly as a dragged-in one -- never through a second importer.

Usage: python scripts/sharepoint_picker_check.py [--inject=scope|token|entry|fork|open|popup]
  popup -- the sign-in popup opened from token(), after awaits (blocked by browsers)
  scope -- a write scope requested
  token -- the service posts to our API
  entry -- the return page dropped from the build
  fork  -- a host's picker wired to a different function than its input
  open  -- the config route without login_required
"""
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VUE = ROOT / "vue_app"
sys.path.insert(0, str(ROOT))

INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def read(rel):
    return (VUE / rel).read_text(encoding="utf-8")


# Each host: (file, the function its own <input type=file> handler routes to,
#             the handler the input calls, or None when the input sets a ref)
# For ref-setting hosts the shared "path" is the ref the import button reads.
HOSTS = [
    ("src/views/ExpensesView.vue", "uploadFileList", "uploadFiles"),
    ("src/components/layout/AppSidebar.vue", "matchUploadFiles", "handleFileSelect"),
    ("src/views/LeaseReviewView.vue", "uploadDocumentFiles", "onDocumentUpload"),
    ("src/views/ValuationsView.vue", "uploadDocumentFiles", "onDocumentUpload"),
    # Valuations > Budget Review: the partner budget AND the Argus cash flow (the only
    # place Argus is loaded since Sep 28 2026). Missed in the first pass -- Jim, Oct 6
    # 2026: "the uploads in this section do not have the sharepoint load buttons".
    ("src/components/common/LineMappingPanel.vue", "parseFile", "onFile"),
]
REF_HOSTS = [  # the picker assigns the same ref the native input assigns
    ("src/views/TreasuryView.vue", ["activityFile", "statementFile", "stmtFiles"]),
    ("src/components/common/ArgusImport.vue", ["cashflowFile", "rentRollFile", "revenueFile"]),
]


def static_checks():
    svc = read("src/services/sharepoint.ts")
    if INJECT == "scope":
        svc = svc.replace("'Sites.Read.All']", "'Sites.Read.All', 'Files.ReadWrite.All']")
    if INJECT == "token":
        svc += "\nexport async function leak(t: string) { await api.post('/api/x', { t }) }\n"

    print("2. Read-only scopes")
    m = re.search(r"const SCOPES\s*=\s*\[([^\]]*)\]", svc)
    scopes = re.findall(r"'([^']+)'", m.group(1)) if m else []
    chk("SCOPES found", bool(scopes))
    bad = [s for s in scopes if not re.fullmatch(r"[A-Za-z]+\.Read(\.All)?", s)]
    chk("every scope is a .Read scope", scopes and not bad, bad)

    print("3. No Microsoft token to our server")
    calls = re.findall(r"\bapi\.(\w+)\(\s*'([^']+)'", svc)
    chk("the service calls our API only to GET its config",
        calls == [("get", "/auth/sso/sharepoint")], calls)

    print("2b. The sign-in window opens ONLY from the click")
    # Jim, Oct 6 2026, first real test: "The Microsoft sign-in window was blocked."
    # The popup was requested after several awaits, outside the click's activation.
    if INJECT == "popup":
        svc = svc.replace("  throw new SignInRequired()\n}",
                          "  const r = await app.acquireTokenPopup({ scopes: SCOPES })\n"
                          "  return r.accessToken\n}")
    tok = svc.split("async function token()", 1)[1].split("\n}\n", 1)[0]
    chk("token() never opens the popup (it throws SignInRequired instead)",
        "acquireTokenPopup" not in tok and "throw new SignInRequired()" in tok)
    sign = svc.split("export function signIn()", 1)[1].split("\n}\n", 1)[0]
    chk("signIn() is NOT async, so acquireTokenPopup runs inside the click",
        "export async function signIn" not in svc and "acquireTokenPopup" in sign
        and "await" not in sign)
    pick = read("src/components/common/SharePointPicker.vue")
    handler = pick.split("function doSignIn()", 1)[1].split("\n}\n", 1)[0]
    first = [ln.strip() for ln in handler.splitlines()[1:] if ln.strip()]
    chk("the picker's click handler calls sp.signIn() before awaiting anything",
        "await" not in handler.split("sp.signIn()")[0] and any("sp.signIn()" in ln for ln in first[:2]),
        first[:3])
    chk("the dialog offers the button rather than opening a window itself",
        '@click="doSignIn"' in pick and "Sign in with Microsoft" in pick
        and "sp.ready" in pick)

    print("4. The popup return page is built and bridges")
    vite = read("vite.config.ts")
    if INJECT == "entry":
        vite = vite.replace("msal-redirect.html", "")
    chk("vite builds msal-redirect.html", "./msal-redirect.html" in vite)
    page = read("msal-redirect.html")
    chk("the page loads src/msalRedirect.ts", 'src="/src/msalRedirect.ts"' in page)
    bridge = read("src/msalRedirect.ts")
    chk("it runs broadcastResponseToMainFrame",
        "broadcastResponseToMainFrame()" in bridge
        and "@azure/msal-browser/redirect-bridge" in bridge)
    chk("the code's redirect URI is that page",
        "redirectUri: `${window.location.origin}/msal-redirect.html`" in svc)

    print("5. One path per importer")
    for rel, path_fn, input_fn in HOSTS:
        src = read(rel)
        if INJECT == "fork" and rel.endswith("ExpensesView.vue"):
            src = src.replace('@picked="uploadFileList"', '@picked="importFromSharePoint"')
        pickers = re.findall(r"<SharePointPicker\b[^>]*@picked=\"([^\"]+)\"", src, re.S)
        chk("%s: has a picker" % Path(rel).stem, bool(pickers))
        chk("%s: the picker emits into %s" % (Path(rel).stem, path_fn),
            pickers and all(p == path_fn or re.search(r"\b%s\(" % path_fn, p) for p in pickers),
            pickers)
        body = re.search(r"function %s\([^)]*\)[^{]*\{(.*?)\n\}" % re.escape(input_fn), src, re.S)
        chk("%s: the file input's %s calls %s too" % (Path(rel).stem, input_fn, path_fn),
            bool(body) and re.search(r"\b%s\(" % path_fn, body.group(1)) is not None)
    for rel, refs in REF_HOSTS:
        src = read(rel)
        pickers = re.findall(r"<SharePointPicker\b[^>]*@picked=\"([^\"]+)\"", src, re.S)
        chk("%s: one picker per file input" % Path(rel).stem, len(pickers) == len(refs), pickers)
        for ref in refs:
            by_input = re.search(r"type=\"file\"[^>]*@change=\"[^\"]*\b%s\s*=" % ref, src, re.S)
            by_picker = any(re.search(r"\b%s\s*=" % ref, p) for p in pickers)
            chk("%s: input and picker both set %s" % (Path(rel).stem, ref),
                bool(by_input) and by_picker)


def server_checks():
    tmp = tempfile.mkdtemp(prefix="sharepoint_picker_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")
    for k in ("SSO_CLIENT_ID", "SSO_TENANT_ID", "SHAREPOINT_CLIENT_ID", "SHAREPOINT_TENANT_ID"):
        os.environ.pop(k, None)

    import jwt
    from datetime import datetime, timedelta, timezone
    from flask_app import create_app
    from flask_app.auth import sso as SSO

    app = create_app()
    app.config["DATABASE_URL"] = None
    if INJECT == "open":  # the route with its login_required peeled off
        app.view_functions["sso.sharepoint_config"] = SSO.sharepoint_config.__wrapped__
    client = app.test_client()
    H = {"Authorization": "Bearer " + jwt.encode(
        {"sub": "1", "username": "admin", "role": "admin",
         "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        app.config["JWT_SECRET"], algorithm="HS256")}

    print("1. The config route")
    r = client.get("/auth/sso/sharepoint")
    chk("no token: 401", r.status_code == 401, r.status_code)
    r = client.get("/auth/sso/sharepoint", headers=H)
    chk("nothing configured: off", r.status_code == 200 and r.get_json() == {"enabled": False},
        (r.status_code, r.get_json()))
    os.environ["SSO_CLIENT_ID"], os.environ["SSO_TENANT_ID"] = "sso-c", "sso-t"
    j = client.get("/auth/sso/sharepoint", headers=H).get_json()
    chk("SSO pair set: on, with those ids",
        j == {"enabled": True, "client_id": "sso-c", "tenant_id": "sso-t"}, j)
    os.environ["SHAREPOINT_CLIENT_ID"], os.environ["SHAREPOINT_TENANT_ID"] = "sp-c", "sp-t"
    j = client.get("/auth/sso/sharepoint", headers=H).get_json()
    chk("SHAREPOINT pair wins over SSO",
        j == {"enabled": True, "client_id": "sp-c", "tenant_id": "sp-t"}, j)
    os.environ.pop("SSO_TENANT_ID")
    os.environ.pop("SHAREPOINT_TENANT_ID")
    j = client.get("/auth/sso/sharepoint", headers=H).get_json()
    chk("a client id with no tenant: off", j == {"enabled": False}, j)


def main():
    server_checks()
    static_checks()
    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
