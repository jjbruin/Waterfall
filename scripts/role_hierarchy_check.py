"""Guardrail: the accounting roles reach what they must, and nothing more.

WHY THIS EXISTS. ``role_required`` used to match the role string exactly while
the comment above ROLES claimed the roles were "ordered by privilege level".
Nothing depended on the difference until a fourth role existed -- and then it
decided everything, because 104 endpoints name only "admin" and "analyst". A
role added to ROLES alone produces a login that is refused by almost the whole
application.

So this asserts the two halves that can silently drift apart:

  1. The three accounting roles reach every analyst-gated endpoint, and are
     still refused by the admin-only ones.
  2. Adding them changed NOTHING for viewer/analyst/admin.

Run:  .venv/Scripts/python.exe scripts/role_hierarchy_check.py
Fails against the commit before the hierarchy landed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask_app.auth.routes import ROLES, ROLE_LEVELS, role_level, WP_ROLE_FOR_LOGIN

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def allowed(user_role, *decorator_roles):
    """Re-implements role_required's decision, which is the thing under test."""
    known = [role_level(r) for r in decorator_roles if r in ROLE_LEVELS]
    required = min(known) if known else None
    return user_role in decorator_roles or (
        required is not None and role_level(user_role) >= required)


ACCOUNTING = ("accountant", "accounting_manager", "cfo")

# ── 1. The new roles exist and sit at analyst level ──────────────────────
for r in ACCOUNTING:
    check(r in ROLES, f"{r} missing from ROLES")
    check(ROLE_LEVELS.get(r) == ROLE_LEVELS["analyst"],
          f"{r} is not at analyst level (got {ROLE_LEVELS.get(r)})")

# ── 2. They reach analyst-gated endpoints ────────────────────────────────
# These are the two decorator shapes actually used in the codebase.
for r in ACCOUNTING:
    check(allowed(r, "admin", "analyst"),
          f"{r} cannot reach an analyst-gated endpoint -- it is locked out of "
          f"the app, which is the exact bug this guardrail exists for")

# ── 3. They are still refused admin-only endpoints ───────────────────────
# User management, MRI refresh and CSV import stay with admin. (Jim's call.)
for r in ACCOUNTING:
    check(not allowed(r, "admin"),
          f"{r} can reach an admin-only endpoint -- it should not manage users "
          f"or import data")

# ── 4. Nothing changed for the three original roles ──────────────────────
# Every one of these held before the hierarchy existed, under exact matching.
check(allowed("admin", "admin"),               "admin lost admin-only access")
check(allowed("admin", "admin", "analyst"),    "admin lost analyst-gated access")
check(allowed("analyst", "admin", "analyst"),  "analyst lost analyst-gated access")
check(not allowed("analyst", "admin"),         "analyst GAINED admin-only access")
check(not allowed("viewer", "admin"),          "viewer GAINED admin-only access")
check(not allowed("viewer", "admin", "analyst"), "viewer GAINED analyst-gated access")

# An unknown role must not fall through to anything.
check(not allowed("", "admin", "analyst"),     "empty role GAINED access")
check(not allowed("nonsense", "admin", "analyst"), "unknown role GAINED access")

# ── 5. The close-cycle linkage is wired to the chain's own names ─────────
# workpaper_service's middle role is "manager", not "accounting_manager"; a
# typo here means the CFO holds a login called cfo and cannot sign anything.
from flask_app.services.workpaper_service import STEP_TEMPLATE
chain_roles = {s["owner"] for s in STEP_TEMPLATE}
for login, wp in WP_ROLE_FOR_LOGIN.items():
    check(login in ROLES, f"WP_ROLE_FOR_LOGIN names {login}, which is not a role")
    check(wp in chain_roles,
          f"WP_ROLE_FOR_LOGIN maps {login} -> {wp}, which no workpaper step owns "
          f"(the chain owns {sorted(chain_roles)})")
for wp in chain_roles:
    check(wp in WP_ROLE_FOR_LOGIN.values(),
          f"workpaper role {wp} has no login role that can act as it")

# ── 6. A typo in a decorator must fail CLOSED ────────────────────────────
# role_level() returns 0 for an unknown name. If that fed the minimum, a
# decorator typo would drop the bar to 0 and admit everyone -- strictly worse
# than the exact matching it replaced, which admitted nobody.
for r in ("viewer", "analyst", "cfo", "admin"):
    check(not allowed(r, "Admin"),
          f"{r} got past a decorator with a typo'd role name -- the hierarchy "
          f"fails OPEN, which is worse than the exact matching it replaced")
    check(not allowed(r, "nonexistent_role"),
          f"{r} got past a decorator naming an unknown role")
# A typo'd name alongside a real one must still enforce the real one.
check(not allowed("viewer", "Admin", "analyst"),
      "viewer got past a decorator whose valid half required analyst")
check(allowed("analyst", "Admin", "analyst"),
      "analyst was refused a decorator that names analyst")

if FAIL:
    print("FAIL")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print(f"OK - {len(ROLES)} roles; {', '.join(ACCOUNTING)} reach analyst-gated "
      f"endpoints, are refused admin-only ones, and map onto the close chain")
