"""Guardrail: the Board section's access model (Board plan, Phase 0, Oct 5 2026).

The Phase 0 acceptance test, as the plan states it: "a guardrail calls every Board
route as every kind of user: no grant means no access, the admin role alone gets
nothing, and compensation tables never surface in Data Explorer, the export or the
assistant."

Every Board route is ENUMERATED FROM THE RUNNING APP, not listed by hand, so a
route added later is called as every kind of user the day it exists.

Asserted in BOTH directions -- a rule checked only by refusals is satisfied by
locking everyone out:
  1. No token: 401. A user without the grant, the admin ROLE included: 403 on
     every route.
  2. A reader reads and changes nothing; an editor edits schedules and narrative
     but cannot create a meeting or touch access; the builder creates meetings;
     only the admin USERNAME manages access and reads the log.
  3. Permissions need the section: removing Board, or passing its end date, voids
     every permission even though the rows remain. Implied permissions follow.
  4. Only the admin username grants Board, and every grant is logged.
  5. comp_* tables, the access log and the permission grants never appear in Data
     Explorer, the export or the assistant -- to the admin username, or to a
     salary-planning holder.
  6. The meeting record refuses what cannot be true (unknown schedule, bad date,
     a meeting no longer in draft) and warns on what is odd (an as-of date after
     the meeting).
  7. The screen agrees with the server: the Vue opt-in list is the registry's, and
     the sidebar link is gated on the section.

Usage: python scripts/board_access_check.py [--inject=optin|role|section]
  optin   -- Board ticked by default (the registry flag dropped)
  role    -- the admin ROLE treated as holding every permission
  section -- permissions not tied to the Board section
"""
import io
import os
import re
import sys
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="board_access_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")

    import jwt
    from flask import g
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth import sections as S, permissions as P
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine

    if INJECT == "optin":
        S.SECTIONS = tuple({k: v for k, v in s.items() if k != "opt_in"} for s in S.SECTIONS)
    if INJECT == "role":
        _orig_eff = P.effective
        P.effective = lambda user, engine=None: (set(P.PERMISSION_KEYS)
                                                 if (user or {}).get("role") == "admin"
                                                 else _orig_eff(user, engine))
    if INJECT == "section":
        P.denied_sections = lambda uid, engine=None: set()

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()
    names = ["admin", "boss", "ana", "reader", "editor", "builder", "cfo", "advisor", "gone"]
    with app.app_context():
        eng = get_engine()
        for n in names:
            create_user(n, "pw-" + n, role="admin" if n in ("admin", "boss") else
                        ("cfo" if n == "cfo" else "analyst"))
        ids = {u["username"]: u["id"] for u in list_users()}
        with eng.begin() as c:
            c.execute(text("CREATE TABLE IF NOT EXISTS comp_payroll_plan (position TEXT, salary REAL)"))
            c.execute(text("INSERT INTO comp_payroll_plan VALUES ('CFO', 1)"))

    def tok(n):
        role = "admin" if n in ("admin", "boss") else ("cfo" if n == "cfo" else "analyst")
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[n]), "username": n, "role": role,
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    H = {n: tok(n) for n in names}

    def grant(n, board=True, perms=None, until=None):
        body = {"board": board}
        if until:
            body["board_until"] = until
        if perms:
            body["permissions"] = {p: True for p in perms}
        return client.put("/api/board/access/%d" % ids[n], json=body, headers=H["admin"])

    print("\n4. Only the admin USERNAME grants Board")
    r = client.put("/auth/users/%d/sections" % ids["reader"], json={"sections": {"board": True}},
                   headers=H["boss"])
    chk("an admin-ROLE user cannot tick Board in User Management (403)", r.status_code == 403, r.status_code)
    r = client.put("/api/board/access/%d" % ids["reader"], json={"board": True}, headers=H["boss"])
    chk("...nor through Board > Access", r.status_code in (403,), r.status_code)
    users = {u["username"]: u for u in client.get("/auth/users", headers=H["admin"]).get_json()["users"]}
    chk("User Management shows Board UNTICKED for a new user (denied_sections)",
        "board" in users["ana"]["denied_sections"], users["ana"]["denied_sections"])
    secs = {s["key"]: s for s in client.get("/auth/sections", headers=H["admin"]).get_json()["sections"]}
    chk("...and the column says it is opt-in", secs["board"].get("opt_in") is True)
    r = client.put("/auth/users/%d/sections" % ids["reader"], json={"sections": {"board": True}},
                   headers=H["admin"])
    chk("the admin username grants Board in User Management", r.status_code == 200, r.get_json())
    chk("editor/builder/cfo/advisor/gone granted through Board > Access", all(
        grant(n, perms=p, until=u).status_code == 200 for n, p, u in [
            ("editor", ["board_edit"], None), ("builder", ["board_build"], None),
            ("cfo", ["comp_edit"], None),
            ("advisor", None, (date.today() + timedelta(days=30)).isoformat()),
            ("gone", ["board_edit", "comp_view"], None),
            # The admin ROLE, granted Board and nothing else: a reader. If the role
            # conferred anything, it would show here, past the section gate.
            ("boss", None, None)]))
    r = grant("ana", until=(date.today() - timedelta(days=1)).isoformat())
    chk("an end date in the past is refused (it would grant nothing)", r.status_code == 400, r.status_code)
    with app.app_context():
        log = client.get("/api/board/audit", headers=H["admin"]).get_json()["entries"]
    acts = {(e["action"], e["target"]) for e in log}
    chk("every grant is in the access log (section and permission)",
        ("section_granted", "reader") in acts and ("permission_granted", "cfo") in acts, sorted(acts)[:6])
    chk("...recorded as done by the admin username", all(e["actor"] == "admin" for e in log))

    # Expire the advisor's grant in the past, the way time would.
    with app.app_context():
        with get_engine().begin() as c:
            c.execute(text("UPDATE user_section_access SET expires_at = :d WHERE user_id = :u "
                           "AND section = 'board'"), {"d": date.today() - timedelta(days=1),
                                                      "u": ids["advisor"]})
    # Remove Board from 'gone', who keeps permission ROWS.
    grant("gone", board=False)

    print("\n1-2. Every Board route, as every kind of user")
    with app.app_context():
        m = client.post("/api/board/meetings", json={"title": "Q1 2026 Board", "meeting_date": "2026-01-29",
                                                     "default_as_of": "2025-12-31"}, headers=H["builder"])
    chk("the builder creates a meeting (201)", m.status_code == 201, m.get_json())
    mid = (m.get_json() or {}).get("id", 1)
    # A real attachment for the routes that address one (a 1x1 PNG).
    import base64
    from flask_app.services import board_package_service as pkg
    PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    with app.app_context():
        aid = pkg.add_attachment(mid, "strategy", "seed.png", PNG, "image/png", "", "admin")["id"]
    routes = []
    for rule in app.url_map.iter_rules():
        if not str(rule).startswith("/api/board"):
            continue
        for meth in sorted(rule.methods - {"HEAD", "OPTIONS"}):
            # A narrative's key, or a schedule's for a schedule view.
            path = (str(rule).replace("<int:mid>", str(mid)).replace("<int:user_id>", str(ids["ana"]))
                    .replace("<key>", "performance" if str(rule).endswith("/view") else "strategy")
                    .replace("<page_key>", "capitalization").replace("<int:aid>", str(aid))
                    .replace("<int:n>", "1"))
            routes.append((meth, path, str(rule)))
    # The schedule views read every engine; this check is about WHO may call
    # them, so the figures are stubbed (board_views_check proves the figures).
    from flask_app.services import board_views_service as views
    views.build_view = lambda key, as_of, **k: {"key": key, "as_of": as_of.isoformat(), "stub": True}
    chk("the schedule view route is among those enumerated",
        any(str(r).endswith("/schedules/<key>/view") for r in app.url_map.iter_rules()))
    st = client.get("/api/board/meetings", headers=H["ana"]).status_code
    chk("a user never granted Board is refused (403)", st == 403, st)
    chk("there are Board routes to call (enumerated: %d)" % len(routes), len(routes) >= 10)

    BODY = {"/api/board/meetings": {"title": "x", "meeting_date": "2026-04-30", "default_as_of": "2026-03-31"},
            "/api/board/meetings/<int:mid>": {"title": "Q1 2026 Board"},
            "/api/board/meetings/<int:mid>/schedules": {"schedules": [{"key": "performance", "as_of": "2025-12-31"}]},
            "/api/board/meetings/<int:mid>/narratives/<key>": {"body": "Strategy text"},
            "/api/board/access/<int:user_id>": {"permissions": {}},
            "/api/board/meetings/<int:mid>/pages/<page_key>/notes": {"footnotes": ["A note"], "disclosure": "D"},
            "/api/board/meetings/<int:mid>/attachments/<int:aid>": {"caption": "A caption"}}
    UPLOAD = "/api/board/meetings/<int:mid>/narratives/<key>/attachments"

    def call(who, meth, path, rule):
        import io as _io
        h = H[who] if who else {}
        if meth == "GET":
            return client.get(path, headers=h).status_code
        if meth == "DELETE":
            # Each caller deletes an attachment of its own, so one success does not
            # turn the next caller's answer into a 404.
            with app.app_context():
                fresh = pkg.add_attachment(mid, "strategy", "d.png", PNG, "image/png", "", "admin")["id"]
            return client.delete(path.rsplit("/", 1)[0] + "/%d" % fresh, headers=h).status_code
        if rule == UPLOAD:
            return client.post(path, data={"file": (_io.BytesIO(PNG), "x.png"), "caption": "c"},
                               content_type="multipart/form-data", headers=h).status_code
        fn = client.post if meth == "POST" else client.put
        return fn(path, json=BODY.get(rule, {}), headers=h).status_code

    WRITE_NEEDS = {"POST /api/board/meetings": "board_build",
                   "PUT /api/board/meetings/<int:mid>": "board_build",
                   "PUT /api/board/meetings/<int:mid>/schedules": "board_edit",
                   "PUT /api/board/meetings/<int:mid>/narratives/<key>": "board_edit",
                   "PUT /api/board/meetings/<int:mid>/pages/<page_key>/notes": "board_edit",
                   "POST " + UPLOAD: "board_edit",
                   "PUT /api/board/meetings/<int:mid>/attachments/<int:aid>": "board_edit",
                   "DELETE /api/board/meetings/<int:mid>/attachments/<int:aid>": "board_edit"}
    SUPER = {"/api/board/access", "/api/board/access/<int:user_id>", "/api/board/audit"}
    HOLDS = {"reader": set(), "boss": set(), "editor": {"board_edit"}, "builder": {"board_build", "board_edit"},
             "cfo": {"comp_edit", "comp_view"}}
    for meth, path, rule in routes:
        k = "%s %s" % (meth, rule)
        chk("%s: no token -> 401" % k, call(None, meth, path, rule) == 401)
        for who in ("ana", "advisor", "gone"):
            st = call(who, meth, path, rule)
            chk("%s: %s (no live grant) -> 403" % (k, who), st == 403, st)
        for who, holds in HOLDS.items():
            st = call(who, meth, path, rule)
            if rule in SUPER:
                ok = st == 403
            elif k in WRITE_NEEDS:
                ok = (st in (200, 201)) if WRITE_NEEDS[k] in holds else st == 403
            else:
                ok = st == 200
            chk("%s: %s -> %s" % (k, who, st), ok)
        st = call("admin", meth, path, rule)
        chk("%s: admin username -> %s" % (k, st), st in (200, 201))

    print("\n3. Permissions need the section; implications follow")
    with app.app_context():
        ROLE = {"admin": "admin", "boss": "admin", "cfo": "cfo"}
        eff = lambda n: P.effective({"id": ids[n], "username": n, "role": ROLE.get(n, "analyst")})
        chk("comp_edit implies comp_view", eff("cfo") == {"comp_edit", "comp_view"}, eff("cfo"))
        chk("the builder is an editor", eff("builder") == {"board_build", "board_edit"}, eff("builder"))
        chk("Board removed: permissions void although the rows remain",
            eff("gone") == set() and P.stored(ids["gone"]) == {"board_edit", "comp_view"},
            (eff("gone"), P.stored(ids["gone"])))
        chk("an expired Board grant voids the section", "board" in S.denied_sections(ids["advisor"]))
        chk("the admin ROLE alone holds no permission", eff("boss") == set(), eff("boss"))
        chk("the admin USERNAME holds every permission", eff("admin") == set(P.PERMISSION_KEYS))

    print("\n5. Compensation, the log and the grants never surface")
    hidden = {"comp_payroll_plan", "access_audit", "user_permissions"}
    for who in ("admin", "cfo"):
        tabs = {t.get("name") for t in client.get("/api/data/tables", headers=H[who]).get_json()["tables"]}
        chk("Data Explorer lists none of them to %s" % who, not (hidden & tabs), hidden & tabs)
        for t in hidden:
            st = client.get("/api/data/tables/%s/rows" % t, headers=H[who]).status_code
            chk("...%s rows refused to %s (403)" % (t, who), st == 403, st)
        r = client.get("/api/data/export", headers=H[who])
        names_ = set(zipfile.ZipFile(io.BytesIO(r.data)).namelist()) if r.status_code == 200 else set()
        chk("the export carries none of them for %s" % who, r.status_code == 200
            and not any(n.lower().startswith(tuple(hidden)) for n in names_), r.status_code)
    from flask_app.services.assistant_service import _tool_query_database
    for who in ("admin", "cfo"):
        with app.test_request_context("/api/assistant"):
            g.current_user = {"id": ids[who], "username": who, "role": "admin" if who == "admin" else "cfo"}
            out = _tool_query_database({"sql": "SELECT * FROM comp_payroll_plan"})
            out2 = _tool_query_database({"sql": "SELECT * FROM access_audit"})
        chk("the assistant refuses comp_payroll_plan to %s" % who, "never shown" in out and "CFO" not in out, out[:90])
        chk("...and the access log", "never shown" in out2, out2[:90])
    tabs = {t.get("name") for t in client.get("/api/data/tables", headers=H["ana"]).get_json()["tables"]}
    chk("board_* tables are hidden from a user without Board", not any(t.startswith("board_") for t in tabs))
    tabs = {t.get("name") for t in client.get("/api/data/tables", headers=H["reader"]).get_json()["tables"]}
    chk("...and visible to a Board reader (the rule admits, too)", "board_meetings" in tabs)
    pull = (ROOT / "scripts/pull_production_db.py").read_text(encoding="utf-8")
    chk("the production pull skips comp_* unless --include-compensation",
        'SKIP_PREFIXES = ("comp_",)' in pull and "include_compensation" in pull)
    chk("...and the grants and the log", '"user_permissions", "access_audit"' in pull)

    print("\n6. The meeting record")
    put = lambda body: client.put("/api/board/meetings/%d/schedules" % mid, json=body, headers=H["editor"])
    r = put({"schedules": [{"key": "no_such", "as_of": "2025-12-31"}]})
    chk("an unknown schedule is refused (400)", r.status_code == 400)
    r = put({"schedules": [{"key": "performance", "as_of": "31/12/2025"}]})
    chk("a date that does not parse is refused, not guessed (400)", r.status_code == 400)
    r = put({"schedules": [{"key": "projected_sales", "as_of": "2026-03-31"}]})
    chk("an as-of AFTER the meeting is saved WITH a warning", r.status_code == 200 and r.get_json()["warnings"],
        r.get_json())
    m = client.get("/api/board/meetings/%d" % mid, headers=H["reader"]).get_json()
    by = {s["key"]: s for s in m["schedules"]}
    chk("each schedule keeps its own as-of date", by["performance"]["as_of"] == "2025-12-31"
        and by["projected_sales"]["as_of"] == "2026-03-31" and by["debt"]["as_of"] == "2025-12-31")
    chk("the narrative a reader sees is the one the editor saved",
        next(n for n in m["narratives"] if n["key"] == "strategy")["body"] == "Strategy text")
    with app.app_context():
        with get_engine().begin() as c:
            c.execute(text("UPDATE board_meetings SET status = 'frozen' WHERE id = :i"), {"i": mid})
    r = put({"schedules": [{"key": "performance", "as_of": "2025-09-30"}]})
    chk("a meeting no longer in draft refuses edits (409)", r.status_code == 409, r.status_code)

    print("\n7. The screen agrees with the server")
    auth_path = ROOT / "vue_app/src/stores/auth.ts"
    if not auth_path.exists():
        # The runtime image ships no vue_app/ source (Dockerfile copies only the
        # built dist). SKIPPED, said so -- skip is not pass; run it locally.
        print("   skip the Vue source is not in this tree (the container image ships none)")
    else:
        auth_ts = auth_path.read_text(encoding="utf-8")
        vue_optin = set(re.findall(r"'([a-z_]+)'", re.search(r"OPT_IN_SECTIONS = \[([^\]]*)\]", auth_ts).group(1)))
        chk("the Vue opt-in list is the registry's", vue_optin == set(S.opt_in_keys()), (vue_optin, S.opt_in_keys()))
        side = (ROOT / "vue_app/src/components/layout/AppSidebar.vue").read_text(encoding="utf-8")
        chk("the sidebar's Board link is gated on the section", "auth.hasSection('board')" in side)

    print("\n8. The end-date column arrives on production's existing table, race-safe")
    from sqlalchemy import create_engine, inspect as sa_inspect
    old = create_engine("sqlite:///" + os.path.join(tmp, "old.db"))
    with old.begin() as c:   # the table as production holds it today
        c.execute(text("CREATE TABLE user_section_access (user_id INTEGER NOT NULL, section TEXT NOT NULL, "
                       "allowed BOOLEAN NOT NULL, updated_by TEXT, updated_at TIMESTAMP, "
                       "PRIMARY KEY (user_id, section))"))
        c.execute(text("INSERT INTO user_section_access VALUES (7, 'accounting', 0, 'admin', NULL)"))
    S._ensure_table(old)
    cols = {c["name"] for c in sa_inspect(old).get_columns("user_section_access")}
    chk("an existing table gains expires_at, its rows kept", "expires_at" in cols
        and "accounting" in S.denied_sections(7, old), cols)
    # Another worker added it between our look and our ALTER: the ALTER fails,
    # and that must not surface as an error.
    S._TABLE_READY.discard(id(old))
    import sqlalchemy
    real = sqlalchemy.inspect
    calls = {"n": 0}

    def stale(engine):
        insp = real(engine)
        calls["n"] += 1
        if calls["n"] == 1:
            got = insp.get_columns
            insp.get_columns = lambda t: [c for c in got(t) if c["name"] != "expires_at"]
        return insp
    sqlalchemy.inspect = stale
    try:
        S._ensure_table(old)
        chk("a worker that loses the race to add the column does not raise", calls["n"] >= 1)
    except Exception as e:
        chk("a worker that loses the race to add the column does not raise", False, e)
    finally:
        sqlalchemy.inspect = real

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
