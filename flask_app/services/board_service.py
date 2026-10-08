"""Board section, Phase 0: meetings, the schedules each carries, narrative blocks.

Board plan, Oct 5 2026 (Claude Docs "Board Package -- Development Plan"). Phase 0
is the foundation the schedule phases build on. It computes NO figures: every
schedule will be drawn from an engine the app already owns (PE exposure,
Investment Metrics, Sold Portfolio, Deal Analysis...), and a schedule's only state
here is WHICH DATE it is drawn at. The January deck mixed dates -- positions at
12/31/25, cash received through 9/30/25 -- so every schedule carries its own
as-of date, defaulted from the meeting's and changeable one by one.

Who may do what is decided by the caller (api/board.py) from
``flask_app.auth.permissions``; this module only refuses what cannot be true:
an unknown schedule or narrative key, a date that does not parse, an edit to a
meeting that is no longer a draft. An as-of date AFTER the meeting is saved with
a warning -- odd, not impossible (a projection page may be dated forward).
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import text

#: The January package's schedules, from the plan's inventory. ``status`` is the
#: plan's own reading of where each one comes from; the phase that builds it
#: replaces the placeholder with the engine's view.
SCHEDULES = (
    {"key": "highlights", "pages": "4", "phase": 2,
     "title": "Since-inception highlights",
     "source": "Sold Portfolio (realized IRR), committed pref, deals, One Pager capitalization",
     "status": "View to build"},
    {"key": "mix_pies", "pages": "4", "phase": 2,
     "title": "Mix by asset type, deal type and region",
     "source": "deals (Asset_Type, Lifecycle, State) weighted by PE exposure",
     "status": "Needs data"},
    {"key": "year_in_review", "pages": "5", "phase": 2,
     "title": "Year in review: AUM by investor, raises, activity, exits",
     "source": "PE exposure at two dates, deals",
     "status": "View to build"},
    {"key": "pref_by_year", "pages": "9", "phase": 2,
     "title": "Pref equity invested by year, new and cumulative",
     "source": "Committed pref and accounting contributions by close year",
     "status": "View to build"},
    {"key": "sponsors", "pages": "10", "phase": 2,
     "title": "Deal sourcing: sponsors and repeat business",
     "source": "deals.Operating_Partner (needs a canonical sponsor list)",
     "status": "Needs data"},
    {"key": "originations", "pages": "11-12", "phase": 3,
     "title": "Originations funnel, channels, pass reasons, pipeline mix",
     "source": "Originations spreadsheet, imported before each meeting",
     "status": "Needs data"},
    {"key": "platform_projection", "pages": "17-19", "phase": 5,
     "title": "5-year platform projection",
     "source": "Platform-model engine (new) and the payroll plan",
     "status": "New engine"},
    {"key": "projected_sales", "pages": "20-21", "phase": 4,
     "title": "5-year projected sales returns",
     "source": "Deal Analysis per deal; PSC share from PE exposure",
     "status": "View to build"},
    {"key": "exposure_asset_class", "pages": "23", "phase": 1,
     "title": "Exposure by asset class",
     "source": "PE exposure (funded + unfunded, total and PSC) by MRI Asset_Type",
     "status": "View built", "view": True},
    {"key": "exposure_partner", "pages": "24", "phase": 1,
     "title": "Exposure by operating partner",
     "source": "PE exposure + Operating_Partner (same cleanup as page 10)",
     "status": "Needs data"},
    {"key": "portfolio_map", "pages": "25", "phase": 1,
     "title": "Portfolio map",
     "source": "deals addresses, geocoded",
     "status": "View to build"},
    {"key": "capitalization", "pages": "26", "phase": 1,
     "title": "Capitalization and third-party capital by investor",
     "source": "PE exposure (funded capital by investor) + Investment Metrics (deals held, Total Size)",
     "status": "View built", "view": True},
    {"key": "performance", "pages": "27", "phase": 1,
     "title": "Portfolio performance",
     "source": "Investment Metrics totals + PE exposure (pref incl. unfunded)",
     "status": "View built", "view": True},
    {"key": "debt", "pages": "28", "phase": 1,
     "title": "Debt, occupancy and DSCR by asset class",
     "source": "loans, Market Rates (SOFR), One Pager occupancy and DSCR",
     "status": "View to build"},
    {"key": "investment_summaries", "pages": "29-31", "phase": 1,
     "title": "Investment summaries, current and exited",
     "source": "Investment Metrics",
     "status": "View built", "view": True},
)
SCHEDULE_KEYS = tuple(s["key"] for s in SCHEDULES)

#: The pages that are edited text, one block each.
NARRATIVES = (
    {"key": "year_in_review", "pages": "5", "title": "Year in review"},
    {"key": "strategy", "pages": "6", "title": "Strategy"},
    {"key": "headwinds", "pages": "7", "title": "Headwinds"},
    {"key": "lessons_learned", "pages": "13", "title": "Lessons learned"},
    {"key": "outlook", "pages": "14", "title": "Outlook"},
    {"key": "marketing", "pages": "15", "title": "Marketing"},
    {"key": "org_chart", "pages": "33", "title": "Organization chart (names and titles only)"},
)
NARRATIVE_KEYS = tuple(n["key"] for n in NARRATIVES)

#: Draft is the only state Phase 0 creates. Phase 6 adds approval and freezing;
#: anything but a draft is refused for edit already, so that arrives safely.
EDITABLE = ("draft",)

_READY = set()


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def ensure(engine=None):
    engine = engine or _engine()
    if id(engine) in _READY:
        return engine
    pk = ("id SERIAL PRIMARY KEY" if engine.dialect.name == "postgresql"
          else "id INTEGER PRIMARY KEY AUTOINCREMENT")
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS board_meetings (
                {pk},
                title TEXT NOT NULL,
                meeting_date DATE NOT NULL,
                default_as_of DATE NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS board_meeting_schedules (
                meeting_id INTEGER NOT NULL,
                schedule_key TEXT NOT NULL,
                as_of DATE NOT NULL,
                included BOOLEAN NOT NULL,
                sort_order INTEGER NOT NULL,
                updated_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (meeting_id, schedule_key)
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS board_narratives (
                meeting_id INTEGER NOT NULL,
                block_key TEXT NOT NULL,
                body TEXT,
                updated_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (meeting_id, block_key)
            )"""))
    _READY.add(id(engine))
    return engine


def parse_date(v, field: str) -> date:
    """A date, or ValueError naming the field -- never a guess."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v or "").strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("%s must be a date (YYYY-MM-DD), not %r" % (field, v))


def _d(v):
    return str(v)[:10] if v is not None else None


def list_meetings(engine=None) -> list[dict]:
    engine = ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, title, meeting_date, default_as_of, status, created_by, updated_by, "
            "updated_at FROM board_meetings ORDER BY meeting_date DESC, id DESC")).mappings().fetchall()
    return [{**dict(r), "meeting_date": _d(r["meeting_date"]),
             "default_as_of": _d(r["default_as_of"]), "updated_at": str(r["updated_at"])}
            for r in rows]


def create_meeting(title: str, meeting_date, default_as_of, actor: str, engine=None) -> dict:
    title = (title or "").strip()
    if not title:
        raise ValueError("A meeting needs a title")
    md = parse_date(meeting_date, "Meeting date")
    ao = parse_date(default_as_of, "As-of date")
    engine = ensure(engine)
    with engine.begin() as conn:
        mid = conn.execute(text(
            "INSERT INTO board_meetings (title, meeting_date, default_as_of, created_by, updated_by) "
            "VALUES (:t, :m, :a, :by, :by) RETURNING id"),
            {"t": title, "m": md, "a": ao, "by": actor}).scalar()
        # Every schedule, in the deck's order, at the meeting's default date.
        for i, s in enumerate(SCHEDULES):
            conn.execute(text(
                "INSERT INTO board_meeting_schedules "
                "(meeting_id, schedule_key, as_of, included, sort_order, updated_by) "
                "VALUES (:m, :k, :a, :inc, :o, :by)"),
                {"m": mid, "k": s["key"], "a": ao, "inc": True, "o": i, "by": actor})
    from flask_app.auth import audit
    audit.log(actor, "board_meeting_created", detail={"meeting_id": mid, "title": title},
              engine=engine)
    return get_meeting(mid, engine)


def get_meeting(meeting_id: int, engine=None) -> dict | None:
    engine = ensure(engine)
    with engine.connect() as conn:
        m = conn.execute(text(
            "SELECT id, title, meeting_date, default_as_of, status, created_by, created_at, "
            "updated_by, updated_at FROM board_meetings WHERE id = :i"),
            {"i": int(meeting_id)}).mappings().fetchone()
        if m is None:
            return None
        sched = {r["schedule_key"]: dict(r) for r in conn.execute(text(
            "SELECT schedule_key, as_of, included, sort_order, updated_by FROM "
            "board_meeting_schedules WHERE meeting_id = :i"), {"i": int(meeting_id)}).mappings()}
        narr = {r["block_key"]: dict(r) for r in conn.execute(text(
            "SELECT block_key, body, updated_by, updated_at FROM board_narratives "
            "WHERE meeting_id = :i"), {"i": int(meeting_id)}).mappings()}
    md = _d(m["meeting_date"])
    schedules = []
    for s in SCHEDULES:
        row = sched.get(s["key"])
        # A schedule added to the catalog after the meeting was created is shown,
        # at the meeting's default date, and SAID to be new rather than silently
        # stored on the spot.
        as_of = _d(row["as_of"]) if row else _d(m["default_as_of"])
        schedules.append({**s, "as_of": as_of,
                          "included": bool(row["included"]) if row else False,
                          "sort_order": row["sort_order"] if row else 999,
                          "not_yet_in_meeting": row is None,
                          "as_of_after_meeting": bool(as_of and md and as_of > md),
                          "updated_by": row["updated_by"] if row else None})
    schedules.sort(key=lambda x: (x["sort_order"], SCHEDULE_KEYS.index(x["key"])))
    narratives = [{**n, "body": (narr.get(n["key"]) or {}).get("body") or "",
                   "updated_by": (narr.get(n["key"]) or {}).get("updated_by"),
                   "updated_at": str((narr.get(n["key"]) or {}).get("updated_at") or "") or None}
                  for n in NARRATIVES]
    return {"id": m["id"], "title": m["title"], "meeting_date": md,
            "default_as_of": _d(m["default_as_of"]), "status": m["status"],
            "editable": m["status"] in EDITABLE,
            "created_by": m["created_by"], "updated_by": m["updated_by"],
            "schedules": schedules, "narratives": narratives}


def _require_draft(meeting_id: int, engine) -> dict:
    m = get_meeting(meeting_id, engine)
    if m is None:
        raise LookupError("Meeting %s not found" % meeting_id)
    if not m["editable"]:
        raise PermissionError("This meeting is %s and can no longer be edited" % m["status"])
    return m


def update_meeting(meeting_id: int, fields: dict, actor: str, engine=None) -> dict:
    """Title, meeting date, default as-of. The default does NOT move schedules
    already dated -- each schedule's date is its own once set."""
    engine = ensure(engine)
    _require_draft(meeting_id, engine)
    sets, params = [], {"i": int(meeting_id), "by": actor}
    if "title" in fields:
        t = (fields["title"] or "").strip()
        if not t:
            raise ValueError("A meeting needs a title")
        sets.append("title = :t"); params["t"] = t
    if "meeting_date" in fields:
        sets.append("meeting_date = :m"); params["m"] = parse_date(fields["meeting_date"], "Meeting date")
    if "default_as_of" in fields:
        sets.append("default_as_of = :a"); params["a"] = parse_date(fields["default_as_of"], "As-of date")
    if not sets:
        raise ValueError("Nothing to change")
    with engine.begin() as conn:
        conn.execute(text("UPDATE board_meetings SET %s, updated_by = :by, "
                          "updated_at = CURRENT_TIMESTAMP WHERE id = :i" % ", ".join(sets)), params)
    from flask_app.auth import audit
    audit.log(actor, "board_meeting_updated",
              detail={"meeting_id": int(meeting_id), "fields": sorted(fields)}, engine=engine)
    return get_meeting(meeting_id, engine)


def set_schedules(meeting_id: int, items: list, actor: str, engine=None) -> dict:
    """[{key, as_of?, included?, sort_order?}] -- only what is sent changes.
    All-or-nothing: every item is checked before anything is written."""
    engine = ensure(engine)
    m = _require_draft(meeting_id, engine)
    current = {s["key"]: s for s in m["schedules"]}
    clean = []
    for it in items or []:
        k = (it or {}).get("key")
        if k not in SCHEDULE_KEYS:
            raise ValueError("Unknown schedule %r" % k)
        cur = current[k]
        clean.append({
            "k": k,
            "a": parse_date(it["as_of"], "As-of date for %s" % k) if "as_of" in it else parse_date(cur["as_of"], "as_of"),
            "inc": bool(it["included"]) if "included" in it else bool(cur["included"]),
            "o": int(it["sort_order"]) if "sort_order" in it else (cur["sort_order"] if cur["sort_order"] != 999 else SCHEDULE_KEYS.index(k)),
        })
    if not clean:
        raise ValueError("No schedules sent")
    with engine.begin() as conn:
        for c in clean:
            conn.execute(text("DELETE FROM board_meeting_schedules WHERE meeting_id = :m "
                              "AND schedule_key = :k"), {"m": int(meeting_id), "k": c["k"]})
            conn.execute(text(
                "INSERT INTO board_meeting_schedules "
                "(meeting_id, schedule_key, as_of, included, sort_order, updated_by) "
                "VALUES (:m, :k, :a, :inc, :o, :by)"), {"m": int(meeting_id), "by": actor, **c})
    from flask_app.auth import audit
    audit.log(actor, "board_schedules_updated",
              detail={"meeting_id": int(meeting_id), "schedules": [c["k"] for c in clean]},
              engine=engine)
    out = get_meeting(meeting_id, engine)
    out["warnings"] = ["%s is dated %s, after the meeting (%s)" % (s["title"], s["as_of"], out["meeting_date"])
                       for s in out["schedules"]
                       if s["as_of_after_meeting"] and s["key"] in {c["k"] for c in clean}]
    return out


def save_narrative(meeting_id: int, block_key: str, body: str, actor: str, engine=None) -> dict:
    if block_key not in NARRATIVE_KEYS:
        raise ValueError("Unknown narrative block %r" % block_key)
    engine = ensure(engine)
    _require_draft(meeting_id, engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM board_narratives WHERE meeting_id = :m AND block_key = :k"),
                     {"m": int(meeting_id), "k": block_key})
        conn.execute(text("INSERT INTO board_narratives (meeting_id, block_key, body, updated_by) "
                          "VALUES (:m, :k, :b, :by)"),
                     {"m": int(meeting_id), "k": block_key, "b": body or "", "by": actor})
    from flask_app.auth import audit
    audit.log(actor, "board_narrative_saved",
              detail={"meeting_id": int(meeting_id), "block": block_key}, engine=engine)
    return get_meeting(meeting_id, engine)
