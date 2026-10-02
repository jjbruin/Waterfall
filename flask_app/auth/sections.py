"""Section access by USERNAME -- which sidebar sections each user may open.

Jim, Oct 1 2026: "I would like to control access to various sections and
tables by username. The first level of access can be defined by section as
listed in the sidebar. Anyone without access to the accounting section should
also be blocked from viewing the gl_accounts, gl_detail, and ia_transactions.
... checkboxes defaulting to checked ... The username "admin" will always have
access to all sections however, users with "admin" as a role will only have
access to the sections as checked ... Create a rule that adds future sections
to the user management table whenever we create new sections in the app."

THIS IS A SECOND AXIS, NOT A REPLACEMENT FOR ROLES. A role says what a user may
DO (edit, approve, import); a section says where they may GO. Both must pass.
An admin-role user who is unchecked for Accounting cannot open it, and an
analyst who is checked for Accounting still cannot edit it (`roles_exactly`).

THIS FILE IS THE ONE PLACE A SECTION IS DEFINED. The User Management table
draws one checkbox column per entry in ``SECTIONS``, the sidebar hides what the
user lacks, the router refuses the screens, and ``enforce_section_access``
refuses the API -- all from the registry below. ADDING A SECTION TO THE APP
MEANS ADDING IT HERE, and `scripts/section_access_check.py` fails until you
do: it reads every section header out of AppSidebar.vue, every route out of
the Vue router and every endpoint out of the running Flask app, and demands
each one is assigned.

A NEW SECTION IS CHECKED FOR EVERYONE BY CONSTRUCTION. Only an UNCHECKED box
is stored (`user_section_access.allowed = FALSE`); no row means allowed. So a
section added tomorrow appears in the table already ticked for every existing
user, and so does every section for a user created tomorrow, without a
backfill that somebody has to remember to run.

ENFORCED ON THE SERVER. The screen hiding a link is a convenience; the API
refusing the request is the control. Every /api path must be either assigned
to sections here or listed as open, and the guardrail enumerates the app's
real url_map to prove it -- a check that greps for a marker is blind to the
endpoint that never had one (the six ungated accounting writes of v489).
"""

from __future__ import annotations

import logging
import re

from flask import g, jsonify, request
from sqlalchemy import text

logger = logging.getLogger(__name__)

#: The one username that always has every section, whatever the table says.
#: A ROLE of admin does not -- Jim's instruction, and the reason this is a
#: username and not a role check.
SUPERUSER = "admin"

#: The sidebar's sections, in sidebar order. ``label`` must match the header
#: text in AppSidebar.vue exactly (the guardrail compares them). ``routes``
#: are the Vue paths the section owns, matched as prefixes, so
#: /portfolio-snapshot/print belongs with /portfolio-snapshot.
SECTIONS = (
    {"key": "dashboard", "label": "Dashboard",
     "routes": ("/dashboard",)},
    {"key": "asset_management", "label": "Asset Management",
     "routes": ("/deal-analysis", "/property-financials", "/surveillance",
                "/valuations", "/one-pager", "/portfolio-snapshot",
                "/review-tracking", "/investment-metrics", "/waterfall-setup")},
    {"key": "accounting", "label": "Accounting",
     "routes": ("/workpapers", "/treasury", "/intercompany", "/gl-ia-query")},
    {"key": "new_business", "label": "New Business",
     "routes": ("/pipeline", "/prospect-analysis", "/lease-review",
                "/lease-risk-analysis", "/lease-abstract")},
    {"key": "investment_management", "label": "Investment Management",
     "routes": ("/ownership",)},
    {"key": "reports", "label": "Reports",
     "routes": ("/reports", "/sold-portfolio", "/portfolio-analysis",
                "/psckoc")},
    {"key": "data_management", "label": "Data Management",
     "routes": ("/data-explorer",)},
    # Employee expense reports (Oct 2 2026). Ticked by default like every
    # section; inside it, who sees which REPORT is per record, in
    # expense_service -- not a change to section access.
    {"key": "expenses", "label": "Expenses",
     "routes": ("/expenses",)},
)
SECTION_KEYS = tuple(s["key"] for s in SECTIONS)

#: Vue paths that belong to no section. /settings is the user's own account
#: (change password) and must stay reachable for someone with nothing else --
#: it sits under Data Management in the sidebar, but locking a user out of
#: their own password because they cannot see the MRI tools would be absurd.
#: User management on that page is still admin-role only.
OPEN_ROUTES = ("/", "/login", "/forgot-password", "/reset-password",
               "/settings")

#: API prefix -> the sections that may call it (ANY ONE suffices). Longest
#: prefix wins, so the /api/data split below works. Where a screen in one
#: section calls another section's API, the prefix names both -- /api/deals is
#: Deal Analysis's engine AND New Business's parcel-sale editor, /api/argus is
#: both sections' Argus import. Blocking it on one section would break the
#: other's screen.
API_SECTIONS = (
    ("/api/dashboard", ("dashboard",)),
    ("/api/deals", ("asset_management", "new_business")),
    ("/api/financials", ("asset_management",)),
    ("/api/surveillance", ("asset_management",)),
    ("/api/valuations", ("asset_management",)),
    ("/api/portfolio-snapshot", ("asset_management",)),
    ("/api/reviews", ("asset_management",)),
    ("/api/waterfall-setup", ("asset_management",)),
    ("/api/investment-metrics", ("asset_management",)),
    ("/api/argus", ("asset_management", "new_business")),
    ("/api/workpapers", ("accounting",)),
    ("/api/treasury", ("accounting",)),
    ("/api/intercompany", ("accounting",)),
    ("/api/gl-ia-query", ("accounting",)),
    ("/api/prospects", ("new_business",)),
    ("/api/lease-review", ("new_business",)),
    ("/api/ownership", ("investment_management",)),
    ("/api/reports", ("reports",)),
    ("/api/sold-portfolio", ("reports",)),
    ("/api/portfolio-analysis", ("reports",)),
    ("/api/psckoc", ("reports",)),
    ("/api/expenses", ("expenses",)),
    # Data Management's own tools. The rest of /api/data is shared plumbing.
    ("/api/data/tables", ("data_management",)),
    ("/api/data/export", ("data_management",)),
    ("/api/data/import", ("data_management",)),
    ("/api/data/upload-import", ("data_management",)),
    ("/api/data/list-csvs", ("data_management",)),
    ("/api/data/table-definitions", ("data_management",)),
    ("/api/data/mri", ("data_management",)),
)

#: API prefixes every signed-in user may call. Shared plumbing every screen
#: needs (the deal list, the config, reload, the version check), the user's own
#: account, feedback, and the assistant -- whose TOOLS are gated one by one in
#: assistant_service, since one chat can touch several sections.
OPEN_API_PREFIXES = (
    "/auth",
    "/api/data/version",
    "/api/data/deals",
    "/api/data/config",
    "/api/data/reload",
    "/api/data/sources",
    "/api/feedback",
    "/api/assistant",
)

#: Tables whose rows only a section's users may see, wherever a path exposes
#: raw table contents: Data Explorer, the database export, the MRI query
#: download/run, and the assistant's SQL tool. The accounting screens read
#: them too, and need nothing extra -- they are already behind the section.
RESTRICTED_TABLES = {
    "gl_accounts": "accounting",
    "gl_detail": "accounting",
    "ia_transactions": "accounting",
}

#: Whole table FAMILIES, by name prefix (Jim, Oct 1 2026: "add the treasury,
#: workpaper, and intercompany tables to the list"). A prefix rather than a
#: list, so a ``tr_``/``wp_``/``ic_`` table added next month is covered the day
#: it is created. Measured: every table in those families is accounting's
#: (tr_* six, wp_* eleven, ic_* four) and no other table carries the prefixes.
RESTRICTED_TABLE_PREFIXES = {
    "tr_": "accounting",   # treasury
    "wp_": "accounting",   # workpapers
    "ic_": "accounting",   # intercompany
    # Expense reports: every employee's spending and, from phase 2, their
    # receipts. Raw-table paths would otherwise show all of them to anyone with
    # Data Management, past the per-report rule the Expenses screens enforce.
    "er_": "accounting",
}

#: Sections granted TOGETHER. Jim, Oct 1 2026: "people with access to asset
#: management can also access new business and visa versa for now." Ticking
#: or unticking one writes all of the group, so the table never shows a state
#: that is not applied. To separate them later, delete the group -- each box
#: is already stored on its own.
LINKED_SECTIONS = (
    ("asset_management", "new_business"),
)


# ── Lookup ───────────────────────────────────────────────────────────

def _prefix_match(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def sections_for_api(path: str):
    """The sections that may call ``path``; ``()`` if open, ``None`` if
    unassigned. Longest matching prefix wins."""
    best, best_len = None, -1
    for prefix, secs in API_SECTIONS:
        if _prefix_match(path, prefix) and len(prefix) > best_len:
            best, best_len = secs, len(prefix)
    for prefix in OPEN_API_PREFIXES:
        if _prefix_match(path, prefix) and len(prefix) > best_len:
            best, best_len = (), len(prefix)
    return best


def section_for_route(path: str):
    """The section key owning a Vue path, ``""`` if open, ``None`` if
    unassigned."""
    if path in OPEN_ROUTES:
        return ""
    for s in SECTIONS:
        if any(_prefix_match(path, r) for r in s["routes"]):
            return s["key"]
    return None


# ── Storage ──────────────────────────────────────────────────────────

_TABLE_READY = set()


def _ensure_table(engine):
    # Once per process per engine -- this runs on every gated request, and
    # DDL on a read path is exactly the v530 defect.
    key = id(engine)
    if key in _TABLE_READY:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_section_access (
                user_id INTEGER NOT NULL,
                section TEXT NOT NULL,
                allowed BOOLEAN NOT NULL,
                updated_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, section)
            )
        """))
    _TABLE_READY.add(key)


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def denied_sections(user_id: int) -> set:
    """The sections this user is UNCHECKED for. Empty means everything."""
    engine = _engine()
    _ensure_table(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT section FROM user_section_access "
            "WHERE user_id = :u AND allowed = :f"),
            {"u": int(user_id), "f": False}).fetchall()
    # A stored key for a section since removed from the registry is ignored.
    return {r[0] for r in rows if r[0] in SECTION_KEYS}


def allowed_sections(user: dict) -> list:
    """Section keys ``user`` may open, in sidebar order."""
    if not user:
        return []
    if user.get("username") == SUPERUSER:
        return list(SECTION_KEYS)
    denied = denied_sections(user["id"])
    return [k for k in SECTION_KEYS if k not in denied]


def all_users_access() -> dict:
    """{user_id: [denied section keys]} for the User Management table."""
    engine = _engine()
    _ensure_table(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT user_id, section FROM user_section_access "
            "WHERE allowed = :f"), {"f": False}).fetchall()
    out = {}
    for uid, sec in rows:
        if sec in SECTION_KEYS:
            out.setdefault(int(uid), []).append(sec)
    return out


def set_user_sections(user_id: int, sections: dict, updated_by: str) -> list:
    """Apply {section_key: bool}. Unknown keys raise. Returns the denied list.

    Checked is stored as the ABSENCE of a row, so re-checking deletes rather
    than writing TRUE -- that is what keeps "no row means allowed" true for
    every section, including ones added after this call.
    """
    unknown = [k for k in sections if k not in SECTION_KEYS]
    if unknown:
        raise ValueError("Unknown section(s): %s" % ", ".join(unknown))
    # A linked section moves with the one that was clicked.
    sections = dict(sections)
    for key, allowed in list(sections.items()):
        for other in linked([key]) - {key}:
            if other in sections and bool(sections[other]) != bool(allowed):
                raise ValueError("%s and %s are granted together"
                                 % (label_for(key), label_for(other)))
            sections[other] = allowed
    engine = _engine()
    _ensure_table(engine)
    with engine.begin() as conn:
        for key, allowed in sections.items():
            conn.execute(text(
                "DELETE FROM user_section_access "
                "WHERE user_id = :u AND section = :s"),
                {"u": int(user_id), "s": key})
            if not allowed:
                conn.execute(text(
                    "INSERT INTO user_section_access "
                    "(user_id, section, allowed, updated_by) "
                    "VALUES (:u, :s, :f, :by)"),
                    {"u": int(user_id), "s": key, "f": False,
                     "by": updated_by})
    return sorted(denied_sections(user_id))


def forget_user(user_id: int):
    """Drop a deleted user's rows, so a recycled id does not inherit them."""
    engine = _engine()
    _ensure_table(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM user_section_access WHERE user_id = :u"),
            {"u": int(user_id)})


# ── Questions asked by the gated paths ───────────────────────────────

def current_allowed() -> set:
    """Allowed section keys for the request's signed-in user, cached on g."""
    cached = getattr(g, "_allowed_sections", None)
    if cached is not None:
        return cached
    user = getattr(g, "current_user", None)
    allowed = set(allowed_sections(user)) if user else set()
    g._allowed_sections = allowed
    return allowed


def table_section(table_name: str):
    """The section a table is restricted to, or None if anyone may see it."""
    name = str(table_name).lower()
    if name in RESTRICTED_TABLES:
        return RESTRICTED_TABLES[name]
    for prefix, sec in RESTRICTED_TABLE_PREFIXES.items():
        if name.startswith(prefix):
            return sec
    return None


def can_see_table(table_name: str) -> bool:
    need = table_section(table_name)
    return need is None or need in current_allowed()


def any_hidden() -> bool:
    """Whether the current user is barred from any restricted table at all."""
    allowed = current_allowed()
    secs = set(RESTRICTED_TABLES.values()) | set(RESTRICTED_TABLE_PREFIXES.values())
    return any(s not in allowed for s in secs)


def sql_names_hidden_table(sql: str):
    """The first hidden table a SQL string names, or None.

    Every identifier in the text is tested, quoted or not, anywhere --
    deliberately broad: a CTE, a subquery or a join reaches the rows just the
    same, and refusing a query that merely mentions the name in a comment is
    the right side to err on. Testing identifiers (not a list of names) is
    what lets a prefix family be matched.
    """
    if not any_hidden():
        return None
    for ident in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', sql):
        if not can_see_table(ident):
            return ident.lower()
    return None


def linked(keys) -> set:
    """``keys`` plus every section linked to any of them."""
    out = set(keys)
    for group in LINKED_SECTIONS:
        if out & set(group):
            out |= set(group)
    return out


def label_for(key: str) -> str:
    for s in SECTIONS:
        if s["key"] == key:
            return s["label"]
    return key


def enforce_section_access():
    """``before_request`` hook: refuse an /api call to a section the user lacks.

    No token or a bad one is passed through, so the route's own
    ``login_required`` answers 401 exactly as before. An /api path assigned to
    nothing is passed through too -- the guardrail is what stops that from
    existing, and failing closed here would turn a forgotten mapping into a
    production outage for every user.
    """
    path = request.path
    if not path.startswith("/api/"):
        return None
    secs = sections_for_api(path)
    if not secs:
        return None
    from flask_app.auth.routes import decode_request_token
    user = decode_request_token()
    if user is None:
        return None
    g.current_user = user
    allowed = current_allowed()
    if any(s in allowed for s in secs):
        return None
    names = " or ".join(label_for(s) for s in secs)
    return jsonify({
        "error": "Forbidden",
        "message": "You do not have access to the %s section. "
                   "An administrator can grant it in Settings > User "
                   "Management." % names,
        "section": list(secs),
    }), 403
