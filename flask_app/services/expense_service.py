"""Employee expense reports -- phase 1: people, reports, lines, approval.

Design, the accounting files it was measured against, and the later phases:
`.claude/memory/expense_reporting.md`.

THE FLOW. An employee opens a report for a period, enters lines, and submits
it. It goes to whoever must approve it, who approves it or returns it with a
note. An approved report is locked and is accounting's (phase 3 codes it).

    draft --submit--> submitted --approve--> approved
      ^                  |  \
      +---- recall ------+   +--return (note required)--> returned --submit--> ...

WHO APPROVES (Jim, Oct 2 2026):
  * an employee's report goes to the APPROVER the admin named for them;
  * an approver's OWN report goes to the CFO -- anyone holding the ``cfo``
    role -- and the CFO's own goes to the CEO or President;
  * the CEO or the President may approve ANY report, for when its approver is
    out. They are the ``ceo`` / ``president`` REVIEW roles already in
    ``review_roles`` (the One Pager chain), not new login roles.
  Nobody decides their own report, whatever roles they hold. The route is
  COMPUTED at submit and stored on the report, so who may decide it does not
  move if the setup changes while it is waiting; the setup screen shows the
  same computation, so it never disagrees with what submit will do.

WHO SEES WHAT. This is a rule about RECORDS inside the Expenses section, not a
change to section access, which stays the Admin's (Jim, Oct 2 2026):
  * a draft is its owner's alone;
  * once submitted, whoever may decide it can read it;
  * from approval on, the accounting roles can read it.
Asking for a report you may not read answers "not found", the same as a report
that does not exist, so an id cannot be used to learn whose reports exist.

MILEAGE IS COMPUTED, NEVER TYPED. Accounting's Sep 24 upload carries Fred's 85
miles as 61.61, 61.62 and 61.63 on different lines, and Elaine's at $0.76 while
everyone else used $0.725. miles x the rate in force on the line's date,
rounded once, and the rate is recorded on the line.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import text

from flask_app.auth.routes import ACCOUNTING_ROLES

logger = logging.getLogger(__name__)

#: The template's "Purpose" list (`Expense Reimbursement - Example.xlsx`,
#: Account Code tab), exactly as accounting wrote it.
PURPOSES = ("Business Development", "Company Related", "Meetings & Conferences",
            "Property Visit - Existing", "Property Visit - Due Diligence", "Other")

#: The deal choice for general business. Accounting: "I added operations if
#: its general business".
OPERATIONS = {"code": "OPERATIONS", "kind": "operations", "name": "Operations"}

#: Accounting's own source for the category list: "should be able to pull
#: from the GL Tables for all MR5* codes". TYPE 'I' only -- the M rows are
#: roll-up headers ("Other Expenses"), not accounts.
CATEGORY_PREFIX = "MR5"

#: The CEO and President approve any report; review roles, not login roles.
BACKUP_REVIEW_ROLES = ("ceo", "president")

#: Statuses at which the report is the employee's to change.
EDITABLE = ("draft", "returned")
#: From here on, accounting may read it.
ACCOUNTING_VISIBLE = ("approved",)

MONEY_TOLERANCE = 0.005

_DDL_DONE: set = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pk(engine) -> str:
    return ("SERIAL PRIMARY KEY" if engine.dialect.name == "postgresql"
            else "INTEGER PRIMARY KEY AUTOINCREMENT")


def ensure_tables(engine) -> None:
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _DDL_DONE:
        return
    pk = _pk(engine)
    ddl = [
        # Who an employee is on a report, and who approves them. Kept apart from
        # `users` so the account and its access are untouched by this feature.
        """CREATE TABLE IF NOT EXISTS er_employees (
            user_id          INTEGER PRIMARY KEY,
            full_name        TEXT,
            approver_user_id INTEGER,
            updated_by       TEXT,
            updated_at       TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS er_reports (
            id               {pk},
            user_id          INTEGER NOT NULL,
            period_start     TEXT NOT NULL,
            period_end       TEXT NOT NULL,
            title            TEXT,
            status           TEXT NOT NULL DEFAULT 'draft',
            route_kind       TEXT,
            approver_user_id INTEGER,
            submitted_at     TEXT,
            decided_by       INTEGER,
            decided_basis    TEXT,
            decided_at       TEXT,
            created_at       TEXT,
            updated_at       TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS er_lines (
            id                {pk},
            report_id         INTEGER NOT NULL,
            sort_order        INTEGER,
            line_date         TEXT,
            line_date_end     TEXT,
            category_account  TEXT,
            purpose           TEXT,
            deal_code         TEXT,
            deal_kind         TEXT,
            deal_name         TEXT,
            vendor            TEXT,
            comment           TEXT,
            amount            DOUBLE PRECISION,
            miles             DOUBLE PRECISION,
            mileage_rate      DOUBLE PRECISION,
            receipt           TEXT,
            no_receipt_reason TEXT,
            updated_at        TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS er_line_splits (
            id        {pk},
            line_id   INTEGER NOT NULL,
            deal_code TEXT NOT NULL,
            deal_kind TEXT,
            deal_name TEXT,
            amount    DOUBLE PRECISION NOT NULL)""",
        f"""CREATE TABLE IF NOT EXISTS er_events (
            id         {pk},
            report_id  INTEGER NOT NULL,
            action     TEXT NOT NULL,
            actor_id   INTEGER,
            actor_name TEXT,
            basis      TEXT,
            note       TEXT,
            at         TEXT)""",
        """CREATE TABLE IF NOT EXISTS er_mileage_rates (
            effective_date TEXT PRIMARY KEY,
            rate           DOUBLE PRECISION NOT NULL,
            basis          TEXT,
            set_by         TEXT,
            set_at         TEXT)""",
    ]
    with engine.begin() as c:
        for s in ddl:
            c.execute(text(s))
    _DDL_DONE.add(key)


# ------------------------------------------------------------------ helpers

def _date(v, label: str, required: bool = True) -> Optional[str]:
    if v in (None, ""):
        if required:
            raise ValueError("%s is required." % label)
        return None
    if isinstance(v, (date, datetime)):
        return (v.date() if isinstance(v, datetime) else v).isoformat()
    try:
        return date.fromisoformat(str(v).strip()[:10]).isoformat()
    except ValueError:
        raise ValueError("%s %r is not a date (YYYY-MM-DD)." % (label, v))


def _money(v, label: str) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return round(float(str(v).replace(",", "").replace("$", "").strip()), 2)
    except (TypeError, ValueError):
        raise ValueError("%s %r is not an amount." % (label, v))


def _users(engine) -> Dict[int, dict]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT id, username, role FROM users")).mappings().all()
    return {int(r["id"]): dict(r) for r in rows}


def _employees(engine) -> Dict[int, dict]:
    ensure_tables(engine)
    with engine.connect() as c:
        rows = c.execute(text("SELECT * FROM er_employees")).mappings().all()
    return {int(r["user_id"]): dict(r) for r in rows}


def _review_role_holders(engine) -> Dict[int, set]:
    """user id -> the backup review roles (ceo / president) they hold."""
    from sqlalchemy import inspect
    if not inspect(engine).has_table("review_roles"):
        return {}
    with engine.connect() as c:
        rows = c.execute(text("SELECT user_id, review_role FROM review_roles")).fetchall()
    out: Dict[int, set] = {}
    for uid, role in rows:
        if role in BACKUP_REVIEW_ROLES:
            out.setdefault(int(uid), set()).add(role)
    return out


def _name(users, emps, uid) -> str:
    if uid is None:
        return ""
    e = emps.get(int(uid)) or {}
    return (e.get("full_name") or "").strip() or (users.get(int(uid)) or {}).get("username", "#%s" % uid)


# ------------------------------------------------------------------ routing

def route_for(engine, user_id: int, users=None, emps=None) -> dict:
    """Where this employee's report goes if submitted now.

    {kind: approver | cfo | ceo_president, approver_user_id, label, error}.
    ``error`` set means submit will be refused, and says why.
    """
    users = users if users is not None else _users(engine)
    emps = emps if emps is not None else _employees(engine)
    uid = int(user_id)
    me = users.get(uid) or {}
    is_approver = any(int(e.get("approver_user_id") or 0) == uid
                      for k, e in emps.items() if k != uid)
    if is_approver:
        if me.get("role") == "cfo":
            return {"kind": "ceo_president", "approver_user_id": None,
                    "label": "CEO or President (the CFO's own report)", "error": None}
        cfos = [k for k, u in users.items() if u.get("role") == "cfo" and k != uid]
        if not cfos:
            return {"kind": "cfo", "approver_user_id": None, "label": "CFO",
                    "error": "This employee approves others, so their report goes to "
                             "the CFO, and no user holds the cfo role."}
        return {"kind": "cfo", "approver_user_id": None,
                "label": "CFO (an approver's own report)", "error": None}
    appr = (emps.get(uid) or {}).get("approver_user_id")
    if not appr:
        return {"kind": "approver", "approver_user_id": None, "label": "",
                "error": "No approver is set for this employee. The admin sets it "
                         "under Expenses > Employees & approvers."}
    appr = int(appr)
    if appr == uid:
        return {"kind": "approver", "approver_user_id": None, "label": "",
                "error": "An employee cannot approve their own report."}
    if appr not in users:
        return {"kind": "approver", "approver_user_id": None, "label": "",
                "error": "The approver set for this employee no longer has an account."}
    return {"kind": "approver", "approver_user_id": appr,
            "label": _name(users, emps, appr), "error": None}


def decide_basis(engine, actor: dict, report: dict, holders=None) -> Optional[str]:
    """How ``actor`` may decide this submitted report, or None if they may not.

    'approver' | 'cfo' | 'ceo' | 'president'. Never the report's own employee.
    """
    if int(actor["id"]) == int(report["user_id"]):
        return None
    kind = report.get("route_kind")
    if kind == "approver" and report.get("approver_user_id") and \
            int(report["approver_user_id"]) == int(actor["id"]):
        return "approver"
    if kind == "cfo" and actor.get("role") == "cfo":
        return "cfo"
    holders = holders if holders is not None else _review_role_holders(engine)
    held = holders.get(int(actor["id"])) or set()
    for r in BACKUP_REVIEW_ROLES:
        if r in held:
            return r
    return None


def can_view(engine, actor: dict, report: dict, holders=None) -> bool:
    if int(actor["id"]) == int(report["user_id"]):
        return True
    if report["status"] == "draft":
        return False
    if report["status"] in ACCOUNTING_VISIBLE and actor.get("role") in ACCOUNTING_ROLES:
        return True
    # Whoever may decide it may read it, before and after the decision.
    return decide_basis(engine, actor, report, holders) is not None


# ------------------------------------------------------------------ setup

def options(engine) -> dict:
    """What the form offers: categories, purposes, deals, the mileage rates."""
    ensure_tables(engine)
    from sqlalchemy import inspect
    insp = inspect(engine)
    cats: List[dict] = []
    if insp.has_table("gl_accounts"):
        with engine.connect() as c:
            rows = c.execute(text(
                'SELECT "ACCTNUM", "ACCTNAME" FROM gl_accounts WHERE "ACCTNUM" LIKE :p '
                'AND "TYPE" = :t ORDER BY "ACCTNUM"'),
                {"p": CATEGORY_PREFIX + "%", "t": "I"}).fetchall()
        cats = [{"account": str(a).strip(), "name": str(n or "").strip()} for a, n in rows]

    deals = [dict(OPERATIONS)]
    if insp.has_table("deals"):
        with engine.connect() as c:
            rows = c.execute(text(
                'SELECT vcode, "Investment_Name", "Portfolio_Name", "Lifecycle" FROM deals')).fetchall()
        owned = []
        for vc, name, parent, life in rows:
            # A child property is reached through its parent deal; a sold deal
            # takes no new expenses.
            if (parent or "").strip() or (life or "").strip().lower() == "sold":
                continue
            owned.append({"code": str(vc), "kind": "deal",
                          "name": str(name or vc).strip()})
        # The deals table carries some deals under two codes with one name
        # (Adirondack RV Park is PADIRON and P0000064). Two identical entries in
        # a dropdown cannot be told apart, so a repeated name shows its code.
        seen: Dict[str, int] = {}
        for d in owned:
            seen[d["name"].lower()] = seen.get(d["name"].lower(), 0) + 1
        for d in owned:
            if seen[d["name"].lower()] > 1:
                d["name"] = "%s (%s)" % (d["name"], d["code"])
        deals += sorted(owned, key=lambda d: d["name"].lower())
    if insp.has_table("prospect_deals"):
        with engine.connect() as c:
            rows = c.execute(text(
                "SELECT vcode, deal_name, stage, onboarded_vcode FROM prospect_deals")).fetchall()
        pipe = [{"code": str(vc), "kind": "prospect",
                 "name": "%s (pipeline, %s)" % (str(n or vc).strip(), vc)}
                for vc, n, st, onb in rows
                if (st or "") not in ("closed", "passed") and not onb]
        deals += sorted(pipe, key=lambda d: d["name"].lower())
    return {"categories": cats, "purposes": list(PURPOSES), "deals": deals,
            "mileage_rates": mileage_rates(engine)}


def employees(engine) -> List[dict]:
    """Every user, with their name on a report, their approver and their route."""
    users, emps = _users(engine), _employees(engine)
    out = []
    for uid, u in sorted(users.items(), key=lambda kv: kv[1]["username"].lower()):
        e = emps.get(uid) or {}
        r = route_for(engine, uid, users, emps)
        out.append({"user_id": uid, "username": u["username"], "role": u["role"],
                    "full_name": e.get("full_name") or "",
                    "approver_user_id": e.get("approver_user_id"),
                    "approver_name": _name(users, emps, e.get("approver_user_id")),
                    "route": r, "updated_by": e.get("updated_by"),
                    "updated_at": e.get("updated_at")})
    return out


def save_employee(engine, user_id: int, body: dict, by: str) -> dict:
    users = _users(engine)
    uid = int(user_id)
    if uid not in users:
        raise ValueError("No user %s." % user_id)
    full_name = str(body.get("full_name") or "").strip()
    appr = body.get("approver_user_id")
    appr = int(appr) if appr not in (None, "") else None
    if appr is not None:
        if appr == uid:
            raise ValueError("An employee cannot be their own approver.")
        if appr not in users:
            raise ValueError("No user %s to approve." % appr)
    with engine.begin() as c:
        c.execute(text("DELETE FROM er_employees WHERE user_id = :u"), {"u": uid})
        c.execute(text(
            "INSERT INTO er_employees (user_id, full_name, approver_user_id, "
            "updated_by, updated_at) VALUES (:u, :n, :a, :by, :at)"),
            {"u": uid, "n": full_name or None, "a": appr, "by": by, "at": _now()})
    return next(e for e in employees(engine) if e["user_id"] == uid)


def mileage_rates(engine) -> List[dict]:
    ensure_tables(engine)
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT * FROM er_mileage_rates ORDER BY effective_date DESC")).mappings().all()
    return [dict(r) for r in rows]


def set_mileage_rate(engine, effective_date, rate, basis, by: str) -> List[dict]:
    d = _date(effective_date, "Effective date")
    if rate in (None, ""):
        raise ValueError("A rate is required.")
    # A rate carries a fraction of a cent (0.725), so it is NOT rounded like money.
    try:
        r = round(float(str(rate).replace("$", "").strip()), 4)
    except ValueError:
        raise ValueError("Rate %r is not a number." % (rate,))
    if not (0 < r < 5):
        raise ValueError("A mileage rate of %s per mile is not plausible." % r)
    with engine.begin() as c:
        c.execute(text("DELETE FROM er_mileage_rates WHERE effective_date = :d"), {"d": d})
        c.execute(text(
            "INSERT INTO er_mileage_rates (effective_date, rate, basis, set_by, set_at) "
            "VALUES (:d, :r, :b, :by, :at)"),
            {"d": d, "r": r, "b": (basis or "").strip() or None, "by": by, "at": _now()})
    return mileage_rates(engine)


def rate_on(engine, on: str) -> Optional[dict]:
    """The mileage rate in force on a date: the latest effective on or before it."""
    for r in mileage_rates(engine):
        if r["effective_date"] <= on:
            return r
    return None


# ------------------------------------------------------------------ reports

def _report_row(engine, report_id) -> Optional[dict]:
    ensure_tables(engine)
    with engine.connect() as c:
        r = c.execute(text("SELECT * FROM er_reports WHERE id = :i"),
                      {"i": int(report_id)}).mappings().first()
    return dict(r) if r else None


def _visible_report(engine, actor, report_id) -> dict:
    r = _report_row(engine, report_id)
    if not r or not can_view(engine, actor, r):
        raise LookupError("No expense report %s." % report_id)
    return r


def _owned_editable(engine, actor, report_id) -> dict:
    r = _visible_report(engine, actor, report_id)
    if int(r["user_id"]) != int(actor["id"]):
        raise PermissionError("Only the employee who owns this report can change it.")
    if r["status"] not in EDITABLE:
        raise PermissionError("This report is %s and can no longer be changed%s." % (
            r["status"], " -- recall it first" if r["status"] == "submitted" else ""))
    return r


def _event(c, report_id, action, actor, basis=None, note=None):
    c.execute(text(
        "INSERT INTO er_events (report_id, action, actor_id, actor_name, basis, note, at) "
        "VALUES (:r, :a, :i, :n, :b, :note, :at)"),
        {"r": int(report_id), "a": action, "i": actor.get("id"),
         "n": actor.get("username"), "b": basis, "note": note, "at": _now()})


def create_report(engine, actor, body: dict) -> dict:
    ensure_tables(engine)
    start = _date(body.get("period_start"), "Period start")
    end = _date(body.get("period_end"), "Period end")
    if end < start:
        raise ValueError("The period ends (%s) before it starts (%s)." % (end, start))
    now = _now()
    with engine.begin() as c:
        rid = c.execute(text(
            "INSERT INTO er_reports (user_id, period_start, period_end, title, status, "
            "created_at, updated_at) VALUES (:u, :s, :e, :t, 'draft', :at, :at) RETURNING id"),
            {"u": int(actor["id"]), "s": start, "e": end,
             "t": (body.get("title") or "").strip() or None, "at": now}).scalar()
        _event(c, rid, "created", actor)
    return get_report(engine, actor, rid)


def update_report(engine, actor, report_id, body: dict) -> dict:
    r = _owned_editable(engine, actor, report_id)
    start = _date(body.get("period_start", r["period_start"]), "Period start")
    end = _date(body.get("period_end", r["period_end"]), "Period end")
    if end < start:
        raise ValueError("The period ends (%s) before it starts (%s)." % (end, start))
    title = body.get("title", r["title"])
    with engine.begin() as c:
        c.execute(text("UPDATE er_reports SET period_start = :s, period_end = :e, "
                       "title = :t, updated_at = :at WHERE id = :i"),
                  {"s": start, "e": end, "t": (title or "").strip() or None,
                   "at": _now(), "i": r["id"]})
    return get_report(engine, actor, r["id"])


def delete_report(engine, actor, report_id) -> dict:
    """A DRAFT may be deleted. A report that was ever submitted has a history
    somebody else has acted on, so it is kept."""
    r = _owned_editable(engine, actor, report_id)
    if r["status"] != "draft" or r.get("submitted_at"):
        raise PermissionError("Only a draft that was never submitted can be deleted.")
    with engine.begin() as c:
        ids = [x[0] for x in c.execute(text("SELECT id FROM er_lines WHERE report_id = :r"),
                                       {"r": r["id"]}).fetchall()]
        for lid in ids:
            c.execute(text("DELETE FROM er_line_splits WHERE line_id = :l"), {"l": lid})
        c.execute(text("DELETE FROM er_lines WHERE report_id = :r"), {"r": r["id"]})
        c.execute(text("DELETE FROM er_events WHERE report_id = :r"), {"r": r["id"]})
        c.execute(text("DELETE FROM er_reports WHERE id = :r"), {"r": r["id"]})
    return {"deleted": r["id"]}


def _lines(engine, report_id) -> List[dict]:
    with engine.connect() as c:
        lines = [dict(x) for x in c.execute(text(
            "SELECT * FROM er_lines WHERE report_id = :r ORDER BY sort_order, id"),
            {"r": int(report_id)}).mappings().all()]
        ids = [x["id"] for x in lines]
        splits: Dict[int, list] = {}
        if ids:
            from sqlalchemy import bindparam
            for s in c.execute(text(
                    "SELECT * FROM er_line_splits WHERE line_id IN :ids ORDER BY id").bindparams(
                    bindparam("ids", expanding=True)), {"ids": ids}).mappings().all():
                splits.setdefault(s["line_id"], []).append(dict(s))
    for x in lines:
        x["splits"] = splits.get(x["id"], [])
    return lines


def line_problems(line: dict, categories: set, deal_codes: set) -> tuple:
    """(errors, warnings) for one line. Errors block submit; a draft may hold them."""
    errs, warns = [], []
    if not line.get("line_date"):
        errs.append("has no date")
    if line.get("line_date_end") and line.get("line_date") and \
            line["line_date_end"] < line["line_date"]:
        errs.append("ends before it starts")
    if not line.get("category_account"):
        errs.append("has no accounting category")
    elif categories and line["category_account"] not in categories:
        errs.append("names category %s, which is not an MR5 expense account"
                    % line["category_account"])
    if not line.get("purpose"):
        errs.append("has no purpose")
    elif line["purpose"] not in PURPOSES:
        errs.append("names purpose %r, which is not on the list" % line["purpose"])
    if not (line.get("comment") or "").strip():
        errs.append("has no comment describing the expense")
    amt = line.get("amount")
    if amt is None or abs(amt) < MONEY_TOLERANCE:
        errs.append("has no amount")
    elif amt < 0:
        warns.append("is negative (a refund or credit?)")
    splits = line.get("splits") or []
    if splits:
        if len(splits) < 2:
            errs.append("is split to one deal -- choose the deal instead")
        tot = round(sum(s["amount"] for s in splits), 2)
        if amt is not None and abs(tot - amt) >= MONEY_TOLERANCE:
            errs.append("is split %s across deals but totals %s" % (
                "{:,.2f}".format(tot), "{:,.2f}".format(amt)))
        for s in splits:
            if deal_codes and s["deal_code"] not in deal_codes:
                errs.append("is split to %s, which is not on the deal list" % s["deal_code"])
    elif not line.get("deal_code"):
        errs.append("has no deal (choose Operations for general business)")
    elif deal_codes and line["deal_code"] not in deal_codes:
        errs.append("names deal %s, which is not on the deal list" % line["deal_code"])
    if line.get("receipt") not in ("Y", "N"):
        errs.append("does not say whether a receipt was submitted")
    elif line["receipt"] == "N" and not (line.get("no_receipt_reason") or "").strip():
        errs.append("has no receipt and no reason why")
    return errs, warns


def _option_sets(engine):
    o = options(engine)
    return ({x["account"] for x in o["categories"]}, {d["code"] for d in o["deals"]},
            {d["code"]: d for d in o["deals"]}, {x["account"]: x["name"] for x in o["categories"]})


def save_line(engine, actor, report_id, body: dict, line_id=None) -> dict:
    r = _owned_editable(engine, actor, report_id)
    cats, codes, deal_by_code, _ = _option_sets(engine)
    d = _date(body.get("line_date"), "Date", required=False)
    d_end = _date(body.get("line_date_end"), "End date", required=False)
    miles = body.get("miles")
    miles = round(float(miles), 2) if miles not in (None, "") else None
    rate = None
    if miles is not None:
        # A mileage line: the amount is computed, never typed.
        if miles <= 0:
            raise ValueError("Miles must be more than zero.")
        if not d:
            raise ValueError("A mileage line needs its date, to find the rate in force.")
        rr = rate_on(engine, d)
        if not rr:
            raise ValueError("No mileage rate is in force on %s. Accounting sets it under "
                             "Expenses > Employees & approvers." % d)
        rate = float(rr["rate"])
        amount = round(miles * rate, 2)
    else:
        amount = _money(body.get("amount"), "Amount")

    def deal_fields(code):
        if not code:
            return None, None, None
        dd = deal_by_code.get(code) or {}
        return code, dd.get("kind"), dd.get("name") or code

    splits = []
    for s in body.get("splits") or []:
        code = (s.get("deal_code") or "").strip()
        a = _money(s.get("amount"), "Split amount")
        if not code and a is None:
            continue
        if not code:
            raise ValueError("A split row has an amount and no deal.")
        if a is None:
            raise ValueError("The split to %s has no amount." % code)
        c_, k_, n_ = deal_fields(code)
        splits.append({"deal_code": c_, "deal_kind": k_, "deal_name": n_, "amount": a})
    deal_code, deal_kind, deal_name = (None, None, None) if splits else \
        deal_fields((body.get("deal_code") or "").strip() or None)

    receipt = (body.get("receipt") or "").strip().upper() or None
    vals = {"report": r["id"], "d": d, "de": d_end,
            "cat": (body.get("category_account") or "").strip() or None,
            "pur": (body.get("purpose") or "").strip() or None,
            "dc": deal_code, "dk": deal_kind, "dn": deal_name,
            "v": (body.get("vendor") or "").strip() or None,
            "cm": (body.get("comment") or "").strip() or None,
            "amt": amount, "mi": miles, "rate": rate, "rc": receipt,
            "why": (body.get("no_receipt_reason") or "").strip() or None, "at": _now()}
    with engine.begin() as c:
        if line_id is None:
            n = c.execute(text("SELECT COALESCE(MAX(sort_order), 0) FROM er_lines "
                               "WHERE report_id = :r"), {"r": r["id"]}).scalar() or 0
            vals["so"] = int(n) + 1
            line_id = c.execute(text(
                "INSERT INTO er_lines (report_id, sort_order, line_date, line_date_end, "
                "category_account, purpose, deal_code, deal_kind, deal_name, vendor, comment, "
                "amount, miles, mileage_rate, receipt, no_receipt_reason, updated_at) VALUES "
                "(:report, :so, :d, :de, :cat, :pur, :dc, :dk, :dn, :v, :cm, :amt, :mi, :rate, "
                ":rc, :why, :at) RETURNING id"), vals).scalar()
        else:
            got = c.execute(text("UPDATE er_lines SET line_date = :d, line_date_end = :de, "
                                 "category_account = :cat, purpose = :pur, deal_code = :dc, "
                                 "deal_kind = :dk, deal_name = :dn, vendor = :v, comment = :cm, "
                                 "amount = :amt, miles = :mi, mileage_rate = :rate, receipt = :rc, "
                                 "no_receipt_reason = :why, updated_at = :at "
                                 "WHERE id = :id AND report_id = :report"),
                            {**vals, "id": int(line_id)})
            if got.rowcount != 1:
                raise LookupError("No line %s on report %s." % (line_id, r["id"]))
        c.execute(text("DELETE FROM er_line_splits WHERE line_id = :l"), {"l": int(line_id)})
        for s in splits:
            c.execute(text("INSERT INTO er_line_splits (line_id, deal_code, deal_kind, "
                           "deal_name, amount) VALUES (:l, :c, :k, :n, :a)"),
                      {"l": int(line_id), "c": s["deal_code"], "k": s["deal_kind"],
                       "n": s["deal_name"], "a": s["amount"]})
        c.execute(text("UPDATE er_reports SET updated_at = :at WHERE id = :i"),
                  {"at": _now(), "i": r["id"]})
    return get_report(engine, actor, r["id"])


def delete_line(engine, actor, report_id, line_id) -> dict:
    r = _owned_editable(engine, actor, report_id)
    with engine.begin() as c:
        got = c.execute(text("DELETE FROM er_lines WHERE id = :l AND report_id = :r"),
                        {"l": int(line_id), "r": r["id"]})
        if got.rowcount != 1:
            raise LookupError("No line %s on report %s." % (line_id, r["id"]))
        c.execute(text("DELETE FROM er_line_splits WHERE line_id = :l"), {"l": int(line_id)})
    return get_report(engine, actor, r["id"])


def _check(engine, report: dict, lines: List[dict]) -> dict:
    cats, codes, _, _ = _option_sets(engine)
    errors, warnings, by_line = [], [], {}
    if not lines:
        errors.append("The report has no lines.")
    for i, ln in enumerate(lines, start=1):
        e, w = line_problems(ln, cats, codes)
        by_line[ln["id"]] = {"errors": e, "warnings": w}
        errors += ["Line %d %s." % (i, x) for x in e]
        warnings += ["Line %d %s." % (i, x) for x in w]
        if ln.get("line_date") and ln["line_date"] > report["period_end"]:
            warnings.append("Line %d is dated %s, after the period ends." % (i, ln["line_date"]))
    return {"errors": errors, "warnings": warnings, "by_line": by_line}


def get_report(engine, actor, report_id) -> dict:
    r = _visible_report(engine, actor, report_id)
    users, emps = _users(engine), _employees(engine)
    holders = _review_role_holders(engine)
    lines = _lines(engine, r["id"])
    _, _, _, cat_names = _option_sets(engine)
    for ln in lines:
        ln["category_name"] = cat_names.get(ln.get("category_account"), "")
    with engine.connect() as c:
        events = [dict(x) for x in c.execute(text(
            "SELECT * FROM er_events WHERE report_id = :r ORDER BY id"),
            {"r": r["id"]}).mappings().all()]
    mine = int(r["user_id"]) == int(actor["id"])
    basis = decide_basis(engine, actor, r, holders) if r["status"] == "submitted" else None
    route = route_for(engine, r["user_id"], users, emps) if r["status"] in EDITABLE else None
    return {**_summary(r, users, emps, lines), "lines": lines, "events": events,
            "check": _check(engine, r, lines),
            "route": route,
            "permissions": {"edit": mine and r["status"] in EDITABLE,
                            "delete": mine and r["status"] == "draft" and not r.get("submitted_at"),
                            "submit": mine and r["status"] in EDITABLE,
                            "recall": mine and r["status"] == "submitted",
                            "decide": basis is not None, "decide_as": basis}}


def _summary(r, users, emps, lines=None) -> dict:
    total = round(sum((x.get("amount") or 0) for x in lines), 2) if lines is not None else None
    waiting = None
    if r["status"] == "submitted":
        waiting = {"approver": _name(users, emps, r.get("approver_user_id")),
                   "cfo": "CFO", "ceo_president": "CEO or President"}.get(r.get("route_kind"))
    return {"id": r["id"], "user_id": r["user_id"],
            "employee": _name(users, emps, r["user_id"]),
            "period_start": r["period_start"], "period_end": r["period_end"],
            "title": r.get("title") or "", "status": r["status"],
            "route_kind": r.get("route_kind"),
            "approver": _name(users, emps, r.get("approver_user_id")),
            "waiting_on": waiting, "submitted_at": r.get("submitted_at"),
            "decided_by": _name(users, emps, r.get("decided_by")),
            "decided_basis": r.get("decided_basis"), "decided_at": r.get("decided_at"),
            "total": total, "updated_at": r.get("updated_at")}


def list_reports(engine, actor, scope: str = "mine") -> List[dict]:
    """``mine`` | ``to_approve`` | ``all`` (everything this user may read)."""
    ensure_tables(engine)
    users, emps = _users(engine), _employees(engine)
    holders = _review_role_holders(engine)
    with engine.connect() as c:
        rows = [dict(x) for x in c.execute(text(
            "SELECT r.*, (SELECT COALESCE(SUM(amount), 0) FROM er_lines l "
            "WHERE l.report_id = r.id) AS line_total, (SELECT COUNT(*) FROM er_lines l "
            "WHERE l.report_id = r.id) AS line_count FROM er_reports r "
            "ORDER BY r.period_end DESC, r.id DESC")).mappings().all()]
    out = []
    for r in rows:
        mine = int(r["user_id"]) == int(actor["id"])
        if scope == "mine" and not mine:
            continue
        if scope == "to_approve" and not (r["status"] == "submitted" and
                                          decide_basis(engine, actor, r, holders)):
            continue
        if not can_view(engine, actor, r, holders):
            continue
        s = _summary(r, users, emps)
        s["total"] = round(float(r["line_total"] or 0), 2)
        s["line_count"] = int(r["line_count"] or 0)
        s["mine"] = mine
        out.append(s)
    return out


def submit(engine, actor, report_id) -> dict:
    r = _owned_editable(engine, actor, report_id)
    lines = _lines(engine, r["id"])
    chk = _check(engine, r, lines)
    if chk["errors"]:
        raise ValueError("The report cannot be submitted yet: " + " ".join(chk["errors"]))
    route = route_for(engine, actor["id"])
    if route["error"]:
        raise ValueError(route["error"])
    now = _now()
    with engine.begin() as c:
        c.execute(text("UPDATE er_reports SET status = 'submitted', route_kind = :k, "
                       "approver_user_id = :a, submitted_at = :at, decided_by = NULL, "
                       "decided_basis = NULL, decided_at = NULL, updated_at = :at "
                       "WHERE id = :i"),
                  {"k": route["kind"], "a": route["approver_user_id"], "at": now, "i": r["id"]})
        _event(c, r["id"], "submitted", actor, basis=route["label"])
    return get_report(engine, actor, r["id"])


def recall(engine, actor, report_id) -> dict:
    """The employee takes back a report nobody has decided yet."""
    r = _visible_report(engine, actor, report_id)
    if int(r["user_id"]) != int(actor["id"]):
        raise PermissionError("Only the employee who submitted it can recall it.")
    if r["status"] != "submitted":
        raise PermissionError("Only a submitted report that has not been decided can be recalled.")
    with engine.begin() as c:
        c.execute(text("UPDATE er_reports SET status = 'draft', updated_at = :at WHERE id = :i"),
                  {"at": _now(), "i": r["id"]})
        _event(c, r["id"], "recalled", actor)
    return get_report(engine, actor, r["id"])


def decide(engine, actor, report_id, action: str, note: Optional[str]) -> dict:
    r = _visible_report(engine, actor, report_id)
    if r["status"] != "submitted":
        raise PermissionError("This report is %s, not waiting for a decision." % r["status"])
    basis = decide_basis(engine, actor, r)
    if not basis:
        raise PermissionError("You are not an approver for this report.")
    note = (note or "").strip() or None
    if action not in ("approve", "return"):
        raise ValueError("The action is approve or return.")
    if action == "return" and not note:
        raise ValueError("Say why the report is being returned.")
    users, emps = _users(engine), _employees(engine)
    # An approval by someone other than the one the report was sent to says so.
    named = {"approver": _name(users, emps, r.get("approver_user_id")),
             "cfo": "the CFO", "ceo_president": "the CEO or President"}.get(r["route_kind"], "")
    recorded = {"approver": "approver", "cfo": "CFO", "ceo": "CEO",
                "president": "President"}[basis]
    in_place = (basis in BACKUP_REVIEW_ROLES and r["route_kind"] != "ceo_president")
    basis_text = recorded + (" in place of %s" % named if in_place else "")
    now = _now()
    with engine.begin() as c:
        if action == "approve":
            c.execute(text("UPDATE er_reports SET status = 'approved', decided_by = :b, "
                           "decided_basis = :bt, decided_at = :at, updated_at = :at "
                           "WHERE id = :i AND status = 'submitted'"),
                      {"b": int(actor["id"]), "bt": basis_text, "at": now, "i": r["id"]})
        else:
            c.execute(text("UPDATE er_reports SET status = 'returned', decided_by = :b, "
                           "decided_basis = :bt, decided_at = :at, updated_at = :at "
                           "WHERE id = :i AND status = 'submitted'"),
                      {"b": int(actor["id"]), "bt": basis_text, "at": now, "i": r["id"]})
        _event(c, r["id"], "approved" if action == "approve" else "returned", actor,
               basis=basis_text, note=note)
    return get_report(engine, actor, r["id"])
