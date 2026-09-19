"""Guardrail: who may edit the accounting section, and who may only read it.

Jim, Sep 17 2026: "Only the accountants, accounting manager, and cfo should be
able to edit anything in the accounting section of the app generally. Me as
admin, can edit only so I can help them get something fixed while we are
building and testing the model."

THE ANALYST IS THE POINT. `role_required` compares LEVELS, and analyst,
accountant, accounting_manager and cfo are all level 1, so every level-based
gate in the accounting section admitted analysts no matter which of those roles
it named. Jim's own day-to-day login is an analyst one and he wants it
read-only here, so the gate had to become membership (`roles_exactly`). This
checks the thing that was wrong before rather than the thing that was easy to
assert: an analyst gets 403 on EVERY write in the section.

IT ENUMERATES THE ROUTES FROM THE APP, it does not list them. A new endpoint
added to either accounting blueprint is covered the day it is written, which is
the only way a rule like this survives. An unguarded write shows up as a failure
naming the route.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed, _failed = [], []

#: Blueprints that make up the accounting section.
ACCOUNTING_PREFIXES = ("/api/workpapers", "/api/treasury")

#: Anything that changes something. GET is a read and stays open.
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

#: Reads that are open to everyone signed in, by design -- accounting has to be
#: visible to the people who depend on it. POSTs appear here only where the verb
#: is a quirk of shape, not of intent: a batch print sends a list of entities in
#: a body because the list is too long for a query string, and it changes
#: nothing.
OPEN_POSTS = {
    "/api/workpapers/statements/batch",
    # Same reason: the coded lines are POSTed because they are a list, and the
    # endpoint only totals them and says whether they balance. It stores
    # nothing. The two endpoints that produce the actual FILES are gated.
    "/api/treasury/upload/preview",
    # Same again: the statement drilldown POSTs because it sends the line's list
    # of accounts in a body. It SELECTs GL rows and returns them; it writes
    # nothing. Reads in this section are open, and this is a read of the same
    # figures the statement above it already shows.
    "/api/workpapers/statements/drilldown",
}

#: Writes that are NARROWER than the section rule, with the roles that may do
#: them. The CFO sets the PLAN of the close -- when it opens, when each thing
#: is due, and the order entities are worked in -- and the team works inside
#: it. Jim, Sep 17 2026, in three passes: "starting a close cycle should belong
#: to the CFO anyone on the accounting team can sync entities", then "deadlines
#: should be CFO only too", then "order number should be CFO only too".
NARROWER = {
    ("POST", "/api/workpapers/cycles"): ("admin", "cfo"),
    # "deadlines should be CFO only too" (Jim, Sep 17 2026). Both of them: the
    # tracker's target date, which is the one an accountant sees, and the
    # per-step due date, which currently has no caller but is the same decision.
    ("PUT", "/api/workpapers/packages/<int:package_id>/schedule/target"):
        ("admin", "cfo"),
    ("PUT", "/api/workpapers/cycles/<int:cycle_id>/steps"): ("admin", "cfo"),
    # "order number should be CFO only too" (Jim, Sep 17 2026) -- and the two
    # buttons that rewrite the same column with it, or the rule is defeated by
    # clicking a different one.
    ("PUT", "/api/workpapers/packages/<int:package_id>/schedule/order"):
        ("admin", "cfo"),
    ("POST", "/api/workpapers/cycles/<int:cycle_id>/schedule/renumber"):
        ("admin", "cfo"),
    ("POST", "/api/workpapers/cycles/<int:cycle_id>/schedule/carry-forward"):
        ("admin", "cfo"),
}


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % detail) if detail and not cond else ""))


def _token(app, role):
    import jwt
    return jwt.encode({"sub": "1", "username": "check-%s" % role, "role": role,
                       "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                      app.config["JWT_SECRET"], algorithm="HS256")


def _sample_path(rule):
    """A callable URL for a rule with converters in it."""
    path = str(rule)
    for arg in rule.arguments:
        for conv, val in (("int:", "1"), ("", "X")):
            path = path.replace("<%s%s>" % (conv, arg), val)
    return path


def main():
    from flask_app import create_app
    from flask_app.auth.routes import (ACCOUNTING_ROLES, CLOSE_PLAN_ROLES,
                                       ROLE_LEVELS)

    app = create_app()
    client = app.test_client()

    print("1. The roles themselves")
    chk("analyst is NOT an accounting role", "analyst" not in ACCOUNTING_ROLES)
    chk("viewer is NOT an accounting role", "viewer" not in ACCOUNTING_ROLES)
    for r in ("admin", "cfo", "accounting_manager", "accountant"):
        chk("%s IS an accounting role" % r, r in ACCOUNTING_ROLES)
    # The reason membership was needed in the first place. If this ever stops
    # being true, the level gate would have worked and this file can be revisited.
    chk("analyst sits at the same level as the accounting roles, which is why "
        "a level gate cannot express this",
        ROLE_LEVELS["analyst"] == ROLE_LEVELS["accountant"] == ROLE_LEVELS["cfo"])

    # Collect every write in the section, from the app itself.
    writes = []
    for rule in app.url_map.iter_rules():
        p = str(rule)
        if not p.startswith(ACCOUNTING_PREFIXES):
            continue
        for m in (rule.methods or set()) & WRITE_METHODS:
            if p in OPEN_POSTS:
                continue
            writes.append((m, p, _sample_path(rule)))
    writes.sort()

    print("\n2. Every write in the accounting section refuses an analyst")
    print("   (%d writes found across %s)"
          % (len(writes), ", ".join(ACCOUNTING_PREFIXES)))
    chk("the section actually has writes to check", len(writes) >= 15,
        str(len(writes)))

    analyst = {"Authorization": "Bearer %s" % _token(app, "analyst")}
    viewer = {"Authorization": "Bearer %s" % _token(app, "viewer")}
    leaks = []
    for method, pattern, path in writes:
        r = client.open(path, method=method, headers=analyst, json={})
        # 403 is the answer. Anything else -- including a 400 for a bad body --
        # means the request got PAST the gate and was refused for some other
        # reason, which is not the same thing and would pass silently.
        if r.status_code != 403:
            leaks.append("%s %s -> %s" % (method, pattern, r.status_code))
    chk("an analyst is refused on all %d" % len(writes), not leaks,
        "; ".join(leaks[:4]))

    leaks_v = []
    for method, pattern, path in writes:
        r = client.open(path, method=method, headers=viewer, json={})
        if r.status_code != 403:
            leaks_v.append("%s %s -> %s" % (method, pattern, r.status_code))
    chk("and so is a viewer", not leaks_v, "; ".join(leaks_v[:4]))

    print("\n3. The accounting roles are NOT refused")
    # Not asserting success -- an empty body legitimately gives a 400. The
    # assertion is only that the GATE let them through, which is what this file
    # is about. A gate that refuses the accountants is the failure v480 shipped
    # on the screen, in the other direction.
    for role in ("accountant", "accounting_manager", "cfo", "admin"):
        h = {"Authorization": "Bearer %s" % _token(app, role)}
        blocked = []
        for method, pattern, path in writes:
            # A route in NARROWER is allowed to refuse this role -- that is the
            # point of it being there. It gets its own checks below.
            allowed_here = NARROWER.get((method, pattern))
            if allowed_here and role not in allowed_here:
                continue
            if client.open(path, method=method, headers=h,
                           json={}).status_code == 403:
                blocked.append("%s %s" % (method, pattern))
        chk("%s is not blocked by the gate anywhere" % role, not blocked,
            "; ".join(blocked[:4]))

    print("\n3b. The CFO sets the frame; the team works inside it")
    # Checked in BOTH directions. A rule that only ever refuses is satisfied by
    # refusing everyone, and a rule that only ever admits is satisfied by a gate
    # that does nothing.
    def _post_cycle(role, body=None):
        return client.post("/api/workpapers/cycles", json=(body or {}),
                           headers={"Authorization": "Bearer %s"
                                                     % _token(app, role)}
                           ).status_code
    for role in ("cfo", "admin"):
        chk("%s may start a close cycle" % role, _post_cycle(role) != 403)
    for role in ("accountant", "accounting_manager"):
        chk("%s may NOT start a close cycle" % role, _post_cycle(role) == 403)
    def _call(method, path, role, **kw):
        return client.open(path, method=method, json=kw.get("json", {}),
                           headers={"Authorization": "Bearer %s"
                                                     % _token(app, role)}
                           ).status_code

    # The DEADLINE: the tracker's target date is the one an accountant sees, so
    # it is the one that matters. Both directions again.
    TGT = "/api/workpapers/packages/1/schedule/target"
    for role in ("cfo", "admin"):
        chk("%s may set a target date" % role, _call("PUT", TGT, role) != 403)
    for role in ("accountant", "accounting_manager"):
        chk("%s may NOT set a target date" % role, _call("PUT", TGT, role) == 403)
    # The ORDER NUMBER, and every route that writes the same column.
    for label, method, path in (
            ("the order cell", "PUT",
             "/api/workpapers/packages/1/schedule/order"),
            ("renumber", "POST",
             "/api/workpapers/cycles/1/schedule/renumber"),
            ("carry forward", "POST",
             "/api/workpapers/cycles/1/schedule/carry-forward")):
        chk("cfo may use %s" % label, _call(method, path, "cfo") != 403)
        chk("accountant may NOT use %s" % label,
            _call(method, path, "accountant") == 403)

    STEP_DUE = "/api/workpapers/cycles/1/steps"
    chk("the per-step deadline follows the same rule",
        _call("PUT", STEP_DUE, "cfo") != 403
        and _call("PUT", STEP_DUE, "accountant") == 403)

    # ...but the same people must still be able to do the ordinary preparation,
    # or the split has just moved the lockout rather than removed it. THIS IS
    # THE HALF THAT CATCHES OVER-TIGHTENING: signing off against a deadline is
    # not the same act as setting one, and the two live in adjacent cells of
    # the same table.
    for role in ("accountant", "accounting_manager"):
        chk("%s may still sync entities into a cycle" % role,
            _call("POST", "/api/workpapers/cycles/1/sync", role) != 403)
        chk("%s may still sign off against a deadline" % role,
            _call("PUT", "/api/workpapers/packages/1/schedule/signoff",
                  role) != 403)
        chk("%s may still name a preparer" % role,
            _call("PUT", "/api/workpapers/packages/1/schedule/preparer",
                  role) != 403)
        chk("%s may still set a property" % role,
            _call("PUT", "/api/workpapers/packages/1/schedule/property",
                  role) != 403)
        chk("%s may still fill the Property column in bulk" % role,
            _call("POST", "/api/workpapers/cycles/1/schedule/properties",
                  role) != 403)

    print("\n4. Reads stay open")
    reads = [(str(r), _sample_path(r)) for r in app.url_map.iter_rules()
             if str(r).startswith(ACCOUNTING_PREFIXES) and "GET" in (r.methods or set())]
    refused = [p for p, path in reads
               if client.get(path, headers=analyst).status_code == 403]
    chk("an analyst can still READ all %d accounting endpoints" % len(reads),
        not refused, "; ".join(refused[:4]))
    chk("an unsigned request is still refused",
        client.get("/api/treasury/accounts").status_code == 401)

    print("\n5. The screen agrees with the server")
    # THE CONTAINER SHIPS THE BUILT BUNDLE, NOT THE SOURCE, so these checks
    # cannot run on production and must SKIP rather than crash -- a guardrail
    # that dies where the real data lives is a guardrail that never runs there,
    # and sections 1-4 above are exactly the ones worth running against it.
    _src = Path(__file__).resolve().parent.parent / "vue_app" / "src"
    if not (_src / "stores" / "auth.ts").exists():
        print("   (Vue source not present -- screen checks skipped; they run "
              "wherever the repo is checked out)")
        return _report()
    # The screen deciding differently from the API is how the CFO was locked
    # out of his own section in v480 -- the API would have taken his writes and
    # the buttons were simply not rendered. So the Vue list is compared to the
    # Python one by name, not trusted to match.
    store = (Path(__file__).resolve().parent.parent / "vue_app" / "src" /
             "stores" / "auth.ts").read_text(encoding="utf-8")
    line = next((l for l in store.splitlines()
                 if "const ACCOUNTING_ROLES" in l), "")
    vue_roles = {w.strip().strip("'\"") for w in
                 line.split("[", 1)[-1].split("]")[0].split(",") if w.strip()}
    chk("the Vue store defines the same roles as the server",
        vue_roles == set(ACCOUNTING_ROLES),
        "vue=%s server=%s" % (sorted(vue_roles), sorted(ACCOUNTING_ROLES)))

    cyc_line = next((l for l in store.splitlines()
                     if "const CLOSE_PLAN_ROLES" in l), "")
    vue_cyc = {w.strip().strip("'\"") for w in
               cyc_line.split("[", 1)[-1].split("]")[0].split(",") if w.strip()}
    chk("and the same close-plan roles", vue_cyc == set(CLOSE_PLAN_ROLES),
        "vue=%s server=%s" % (sorted(vue_cyc), sorted(CLOSE_PLAN_ROLES)))

    for view in ("WorkpapersView.vue", "TreasuryView.vue"):
        src = (Path(__file__).resolve().parent.parent / "vue_app" / "src" /
               "views" / view).read_text(encoding="utf-8")
        chk("%s reads the shared gate rather than its own list" % view,
            "auth.canEditAccounting" in src
            and "['admin', 'cfo'].includes" not in src)

    wp = (Path(__file__).resolve().parent.parent / "vue_app" / "src" /
          "views" / "WorkpapersView.vue").read_text(encoding="utf-8")
    chk("the New close cycle button is on the narrower gate",
        'v-if="canSetPlan" class="btn primary"' in wp)
    chk("and so is the form it opens, not just the button",
        'v-if="showNewCycle && canSetPlan"' in wp)
    chk("Sync entities is still on the team's gate",
        'v-if="canManageClose"' in wp and "@click=\"sync\"" in wp)
    chk("the order input is on the CFO's gate",
        'v-if="canSetPlan" class="ord-in"' in wp)
    chk("Renumber and Carry forward are with it",
        'v-if="canSetPlan"' in wp and "@click=\"renumber\"" in wp)
    chk("but Fill properties is not -- it writes the Property column, "
        "which is the team's",
        'v-if="canManageClose" class="btn" @click="fillProperties"' in wp)
    chk("and the dead setDue() is gone", "setDue" not in wp)
    chk("the target-date input is on the CFO's gate",
        'v-if="canSetPlan" class="date-in"' in wp)
    chk("and the sign-off cells beside it are not",
        'v-if="canSetPlan"' not in wp.split("class=\"sign\"")[1][:900])

    return _report()


def _report():
    print("\n%d checks, %d failed." % (len(_passed) + len(_failed), len(_failed)))
    if _failed:
        for f in _failed:
            print("  FAILED: %s" % f)
        return 1
    print("Every write in the accounting section is closed to analysts and\n"
          "viewers and open to the four accounting roles, enumerated from the\n"
          "app rather than from a list kept by hand. The PLAN of the close --\n"
          "when it opens, when things are due, what order they are worked in --\n"
          "is the CFO's alone; syncing, naming a preparer, setting a property\n"
          "and signing off stay with the team. Both directions, every time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
