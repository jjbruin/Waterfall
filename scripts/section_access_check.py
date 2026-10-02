"""Guardrail: section access by username, and THE RULE that a new section is
added to the User Management table.

Jim, Oct 1 2026: access is granted per sidebar SECTION, by username, with a
checkbox per section in User Management defaulting to checked; users without
Accounting cannot see gl_accounts, gl_detail or ia_transactions; the USERNAME
"admin" always has every section while the admin ROLE has only what is
ticked; and "create a rule that adds future sections to the user management
table whenever we create new sections in the app."

THE RULE IS PART 1, AND IT IS STRUCTURAL. The User Management columns, the
sidebar gate, the router guard and the API gate all read ONE registry,
``flask_app/auth/sections.py``. So "adding a section to the table" means
"adding it to the registry", and this script fails until that is done:

  * every section header in AppSidebar.vue must be a registry label, gated on
    ``auth.hasSection('<its key>')``, and every registry entry must appear;
  * every screen linked inside a section block must belong to that section;
  * every Vue route must belong to a section or be listed as open;
  * every /api route in the RUNNING app must be assigned or listed as open --
    enumerated from url_map, never grepped, so a new blueprint is caught the
    day it is registered;
  * every assistant tool must say which section its data belongs to.

PART 2 drives the real endpoints against a scratch SQLite database, in BOTH
directions: a rule checked only in the refusing direction is satisfied by
locking everyone out.

    .venv/Scripts/python.exe scripts/section_access_check.py            # all
    .venv/Scripts/python.exe scripts/section_access_check.py --static   # part 1
"""
import io
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % detail) if detail and not cond else ""))


def skip(label, why):
    print("   skip %s   (%s)" % (label, why))


# ── Part 1: every section, screen and endpoint is assigned ───────────

def _sidebar_blocks(src):
    """[(key, block_text)] split at each hasSection('key') marker."""
    marks = list(re.finditer(r"hasSection\('([a-z_]+)'\)", src))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(src)
        out.append((m.group(1), src[m.start():end]))
    return out


def _sidebar_labels(template):
    labels = []
    for m in re.finditer(
            r'class="nav-section-header".*?>\s*(?:<span>)?\s*([^<\n]+?)\s*(?:</span>|<span|\n)',
            template, re.S):
        labels.append(m.group(1).strip())
    for m in re.finditer(r'class="nav-section-link"[^>]*>\s*([^<]+?)\s*</router-link>',
                         template, re.S):
        labels.append(m.group(1).strip())
    return labels


def static_checks(app=None):
    from flask_app.auth import sections as S

    print("1a. The registry")
    keys = [s["key"] for s in S.SECTIONS]
    chk("section keys are unique", len(keys) == len(set(keys)))
    chk("the superuser is the USERNAME 'admin'", S.SUPERUSER == "admin")
    for t in ("gl_accounts", "gl_detail", "ia_transactions",
              "tr_accounts", "tr_periods", "tr_statements", "wp_fs_map",
              "wp_packages", "wp_tracker_cell", "ic_entity_settings",
              "ic_recon_notes"):
        chk("%s is restricted to Accounting" % t,
            S.table_section(t) == "accounting")
    for t in ("deals", "accounting", "isbs_interim_is", "lease_tenants"):
        chk("%s is NOT restricted" % t, S.table_section(t) is None)
    chk("Asset Management and New Business are linked",
        S.linked(["asset_management"]) == {"asset_management", "new_business"})
    chk("no other section is linked", S.linked(["accounting"]) == {"accounting"})

    print("\n1b. The sidebar matches the registry, section for section")
    side = (ROOT / "vue_app/src/components/layout/AppSidebar.vue")
    if not side.exists():
        skip("sidebar checks", "no vue_app/ in this image")
    else:
        src = side.read_text(encoding="utf-8")
        template = src[src.index("<template>"):]
        labels = _sidebar_labels(template.split('class="sidebar-extras"')[0])
        reg_labels = [s["label"] for s in S.SECTIONS]
        for lab in labels:
            chk("sidebar section '%s' is in the registry" % lab,
                lab in reg_labels,
                "add it to SECTIONS in flask_app/auth/sections.py")
        for lab in reg_labels:
            chk("registry section '%s' is in the sidebar" % lab, lab in labels)
        for k in keys:
            chk("sidebar gates '%s' on auth.hasSection" % k,
                ("hasSection('%s')" % k) in template)
        for key, block in _sidebar_blocks(template):
            for to in re.findall(r'to="(/[^"]*)"', block):
                owner = S.section_for_route(to)
                chk("%s, linked in the %s block, belongs to %s" % (to, key, key),
                    owner == key or owner == "",
                    "registry says %r" % owner)

    print("\n1c. Every Vue route belongs to a section or is open")
    router = ROOT / "vue_app/src/router/index.ts"
    if not router.exists():
        skip("router checks", "no vue_app/ in this image")
    else:
        for path in re.findall(r"path:\s*'([^']+)'",
                               router.read_text(encoding="utf-8")):
            chk("route %s is assigned" % path,
                S.section_for_route(path) is not None,
                "add it to a section's routes or OPEN_ROUTES")

    print("\n1d. Every assistant tool says which section its data is")
    from flask_app.services.assistant_service import TOOLS, TOOL_SECTIONS
    names = {t["name"] for t in TOOLS}
    for n in sorted(names):
        chk("tool %s is mapped" % n, n in TOOL_SECTIONS)
    for n in sorted(set(TOOL_SECTIONS) - names):
        chk("mapped tool %s exists" % n, False)
    for n, secs in TOOL_SECTIONS.items():
        chk("tool %s names only real sections" % n,
            all(s in keys for s in secs))

    print("\n1e. Every /api route in the running app is assigned or open")
    if app is None:
        from flask_app import create_app
        app = create_app()
    unassigned = sorted({str(r) for r in app.url_map.iter_rules()
                         if str(r).startswith("/api/")
                         and S.sections_for_api(str(r)) is None})
    chk("no /api route is unassigned (%d rules checked)"
        % sum(1 for r in app.url_map.iter_rules() if str(r).startswith("/api/")),
        not unassigned, ", ".join(unassigned[:8]))
    for prefix, secs in S.API_SECTIONS:
        chk("API prefix %s names only real sections" % prefix,
            all(s in keys for s in secs))
    return app


# ── Part 2: behaviour ────────────────────────────────────────────────

def behaviour_checks():
    tmp = tempfile.mkdtemp(prefix="section_access_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")

    import jwt
    from flask_app import create_app
    from flask_app.auth import sections as S
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    from sqlalchemy import text

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()

    with app.app_context():
        eng = get_engine()
        with eng.begin() as c:
            for t in ("gl_detail", "gl_accounts", "ia_transactions", "deals_x",
                      "tr_x_check", "wp_x_check", "ic_x_check"):
                c.execute(text('CREATE TABLE IF NOT EXISTS "%s" (a TEXT)' % t))
                c.execute(text('INSERT INTO "%s" VALUES (\'row\')' % t))
        create_user("admin", "pw-admin", role="admin")
        create_user("boss", "pw-boss", role="admin")
        create_user("ana", "pw-ana", role="analyst")
        ids = {u["username"]: u["id"] for u in list_users()}

    def tok(name, role):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[name]), "username": name, "role": role,
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    H = {"admin": tok("admin", "admin"), "boss": tok("boss", "admin"),
         "ana": tok("ana", "analyst")}

    def me(name):
        return client.get("/auth/me", headers=H[name]).get_json()["user"]["sections"]

    def put(target, body, who="admin"):
        return client.put("/auth/users/%d/sections" % ids[target],
                          json={"sections": body}, headers=H[who])

    def tables(name):
        r = client.get("/api/data/tables", headers=H[name])
        return r.status_code, {t["name"] for t in (r.get_json() or {}).get("tables", [])}

    def export_names(name):
        r = client.get("/api/data/export", headers=H[name])
        if r.status_code != 200:
            return r.status_code, set()
        return 200, set(zipfile.ZipFile(io.BytesIO(r.data)).namelist())

    all_keys = list(S.SECTION_KEYS)

    # A 403 from a path that matches NO route would prove only that the gate
    # runs before routing -- the before_request hook fires for a 404 too. Every
    # path refused below is first proved to be a real endpoint.
    adapter = app.url_map.bind("localhost")

    def real(path, method="GET"):
        try:
            adapter.match(path, method=method)
            return True
        except Exception:
            return False

    print("\n2a. Default: every section, for every user")
    chk("an analyst with no rows has every section", me("ana") == all_keys)
    chk("an admin-role user with no rows has every section", me("boss") == all_keys)
    st, names = tables("ana")
    chk("Data Explorer lists gl_detail to a user with Accounting",
        st == 200 and "gl_detail" in names, st)

    print("\n2b. Unticking Accounting blocks the section AND the three tables")
    r = put("ana", {"accounting": False})
    chk("the admin user can untick a section", r.status_code == 200, r.status_code)
    chk("me() no longer lists accounting", "accounting" not in me("ana"))
    chk("me() keeps the other sections", len(me("ana")) == len(all_keys) - 1)
    for path in ("/api/workpapers/cycles", "/api/treasury/accounts",
                 "/api/gl-ia-query/gl/options", "/api/intercompany/periods"):
        chk("%s refused (403)" % path, real(path)
            and client.get(path, headers=H["ana"]).status_code == 403)
    st, names = tables("ana")
    for t in ("gl_detail", "gl_accounts", "ia_transactions",
              "tr_x_check", "wp_x_check", "ic_x_check"):
        chk("Data Explorer does not list %s" % t, st == 200 and t not in names)
        chk("rows of %s refused" % t,
            client.get("/api/data/tables/%s/rows" % t,
                       headers=H["ana"]).status_code == 403)
    chk("an unrestricted table is still listed", "deals_x" in names)
    chk("an unrestricted table's rows still served",
        client.get("/api/data/tables/deals_x/rows",
                   headers=H["ana"]).status_code == 200)
    st, zn = export_names("ana")
    chk("the export omits gl_detail", st == 200
        and "gl_detail_db_export.csv" not in zn, st)
    for fam in ("tr_x_check", "wp_x_check", "ic_x_check"):
        chk("the export omits %s (treasury/workpaper/intercompany family)" % fam,
            "%s_db_export.csv" % fam not in zn)
    chk("the export still carries other tables",
        "deals_x_db_export.csv" in zn)
    for q in ("MRI_GL_Detail", "MRI_GL_Accounts", "MRI_IA_Transactions"):
        for verb, path in (("get", "download"), ("post", "run")):
            r = getattr(client, verb)("/api/data/mri/queries/%s/%s" % (q, path),
                                      headers=H["ana"])
            chk("MRI query %s %s refused BEFORE it runs" % (q, path),
                r.status_code == 403, r.status_code)
    with app.test_request_context("/"):
        from flask import g
        from flask_app.services.assistant_service import execute_tool
        g.current_user = {"id": ids["ana"], "username": "ana", "role": "analyst"}
        out = execute_tool("query_database", {"sql": "SELECT * FROM gl_detail"})
        chk("assistant SQL refuses gl_detail", "visible only" in out, out[:80])
        out = execute_tool("query_database",
                           {"sql": 'select a from (select * from "IA_TRANSACTIONS") x'})
        chk("assistant SQL refuses a quoted, nested, upper-case reference",
            "visible only" in out, out[:80])
        for fam in ("tr_x_check", "WP_X_CHECK", '"ic_x_check"'):
            out = execute_tool("query_database",
                               {"sql": "SELECT * FROM deals_x JOIN %s ON 1=1" % fam})
            chk("assistant SQL refuses a join to %s" % fam, "visible only" in out, out[:80])
        out = execute_tool("query_database", {"sql": "SELECT * FROM deals_x"})
        chk("assistant SQL still answers an unrestricted table",
            "visible only" not in out and "row" in out, out[:80])

    print("\n2c. Re-ticking restores it (and stores nothing)")
    put("ana", {"accounting": True})
    chk("accounting is back", me("ana") == all_keys)
    st, names = tables("ana")
    chk("gl_detail is listed again", "gl_detail" in names)
    chk("the treasury family is listed again", "tr_x_check" in names)
    chk("the workpapers API is no longer refused for section",
        client.get("/api/workpapers/cycles", headers=H["ana"]).status_code != 403)
    with app.app_context():
        with get_engine().connect() as c:
            n = c.execute(text("SELECT COUNT(*) FROM user_section_access "
                               "WHERE user_id = :u"), {"u": ids["ana"]}).scalar()
    chk("a re-ticked box is stored as NO row, so 'no row = allowed' holds", n == 0, n)

    print("\n2d. Other sections gate their own APIs")
    put("ana", {"reports": False, "asset_management": False,
                "new_business": False, "dashboard": False,
                "investment_management": False, "data_management": False})
    for path, sec in (("/api/reports/partners", "reports"),
                      ("/api/dashboard/kpis", "dashboard"),
                      ("/api/ownership/tree", "investment_management"),
                      ("/api/valuations/cycles", "asset_management"),
                      ("/api/prospects/1", "new_business"),
                      ("/api/data/tables", "data_management")):
        chk("%s refused without %s" % (path, sec), real(path)
            and client.get(path, headers=H["ana"]).status_code == 403)
    chk("/api/deals refused with neither Asset Management nor New Business",
        client.post("/api/deals/compute", json={},
                    headers=H["ana"]).status_code == 403)
    put("ana", {"new_business": True})
    chk("/api/deals reachable again once New Business is ticked (shared API)",
        client.post("/api/deals/compute", json={},
                    headers=H["ana"]).status_code != 403)

    print("\n2d'. Asset Management and New Business are granted together")
    chk("ticking New Business ticked Asset Management too",
        {"asset_management", "new_business"} <= set(me("ana")))
    put("ana", {"asset_management": False})
    chk("unticking Asset Management unticked New Business too",
        not ({"asset_management", "new_business"} & set(me("ana"))))
    r = put("ana", {"asset_management": True, "new_business": False})
    chk("a request that splits the pair is refused", r.status_code == 400)
    chk("...and changed nothing",
        not ({"asset_management", "new_business"} & set(me("ana"))))
    put("ana", {"asset_management": True})
    chk("re-ticking Asset Management restores both",
        {"asset_management", "new_business"} <= set(me("ana")))
    for path in ("/auth/me", "/auth/sections", "/api/data/version",
                 "/api/feedback"):
        chk("%s stays open to a user with almost nothing" % path,
            client.get(path, headers=H["ana"]).status_code != 403)
    with app.test_request_context("/"):
        from flask import g
        from flask_app.services.assistant_service import execute_tool
        g.current_user = {"id": ids["ana"], "username": "ana", "role": "analyst"}
        chk("assistant tool get_sold_returns refused without Reports",
            "does not have access" in execute_tool("get_sold_returns", {}))
        chk("assistant tool get_one_pager (Asset Management) available once New "
            "Business is ticked -- the two are granted together",
            "does not have access" not in execute_tool("get_one_pager", {"vcode": "X"}))

    print("\n2e. The admin ROLE gets only what is ticked; the admin USERNAME all")
    put("boss", {"data_management": False}, who="admin")
    chk("admin-role user unticked for Data Management is refused Data Explorer",
        client.get("/api/data/tables", headers=H["boss"]).status_code == 403)
    put("boss", {"data_management": True}, who="admin")
    chk("...and admitted once ticked",
        client.get("/api/data/tables", headers=H["boss"]).status_code == 200)
    r = put("admin", {"accounting": False})
    chk("unticking the admin USERNAME is refused", r.status_code == 400)
    with app.app_context():
        with get_engine().begin() as c:
            c.execute(text("INSERT INTO user_section_access "
                           "(user_id, section, allowed) VALUES (:u, 'accounting', :f)"),
                      {"u": ids["admin"], "f": False})
    chk("the admin USERNAME has every section even with a stored untick",
        me("admin") == all_keys)
    st, names = tables("admin")
    chk("...and sees gl_detail", "gl_detail" in names)
    r = put("boss", {"accounting": False}, who="ana")
    chk("an analyst cannot change section access", r.status_code == 403)
    r = put("ana", {"accounting": False}, who="boss")
    chk("an admin-ROLE user cannot assign section access (admin USERNAME only)",
        r.status_code == 403, r.status_code)
    chk("...and the refusal changed nothing", "accounting" in me("ana"))
    r = put("boss", {"reports": False}, who="boss")
    chk("an admin-ROLE user cannot change their own access either",
        r.status_code == 403, r.status_code)
    sec_admin = client.get("/auth/sections", headers=H["admin"]).get_json()
    sec_boss = client.get("/auth/sections", headers=H["boss"]).get_json()
    chk("the screen is told the admin user may assign", sec_admin.get("can_assign") is True)
    chk("the screen is told an admin-role user may not", sec_boss.get("can_assign") is False)
    chk("the screen is told which sections are linked",
        ["asset_management", "new_business"] in sec_admin.get("linked", []))
    r = put("ana", {"no_such_section": False})
    chk("an unknown section key is refused", r.status_code == 400)

    print("\n2f. A section added tomorrow is ticked for everyone already")
    saved = S.SECTIONS, S.SECTION_KEYS
    try:
        S.SECTIONS = saved[0] + ({"key": "future_section", "label": "Future",
                                  "routes": ("/future",)},)
        S.SECTION_KEYS = tuple(s["key"] for s in S.SECTIONS)
        chk("existing user has the new section with no backfill",
            "future_section" in me("boss"))
        cols = [s["key"] for s in
                client.get("/auth/sections", headers=H["boss"]).get_json()["sections"]]
        chk("the User Management column list carries it", "future_section" in cols)
    finally:
        S.SECTIONS, S.SECTION_KEYS = saved

    print("\n2g. No token still means 401, not 403")
    chk("an unauthenticated section call is 401",
        client.get("/api/workpapers/cycles").status_code == 401)
    return app


def main():
    static_only = "--static" in sys.argv
    app = None
    if not static_only:
        app = behaviour_checks()
    static_checks(app)
    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
