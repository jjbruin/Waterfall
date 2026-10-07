"""Email for the expense workflow (the CFO's ask, decided by Jim, Oct 7 2026).

    submitted          -> whoever may decide it: the approver; the CFO for an approver's
                          own report; the CEO and President for the CFO's own
    approved           -> accounting (to code it) and the employee
    returned           -> the employee, with the reason (by the approver, or by accounting)
    coding submitted   -> the reviewers, CFO and accounting manager
    coding returned    -> whoever submitted the coding, with the reason

ACCOUNTING IS THE ROLES, not a list: every user whose role is cfo, accounting_manager or
accountant AND who holds accounting authority (`sections.has_accounting_authority`) --
so the recipients follow User Management and the same gate that decides who may read an
approved report. Nobody is emailed about their own report or their own submission.

SENT AFTER THE ACTION, IN THE BACKGROUND. The action is the record; an email is a
courtesy about it. ACS can take up to 20 seconds per message, and an approval must not
wait on, or fail because of, a mail server. Every send -- delivered or not -- is written
to the report's history, so "did the approver get an email?" has an answer on screen.
A recipient with no email address on file is recorded as such, never skipped silently.
"""
from __future__ import annotations

import html
import logging
import threading
from typing import Dict, List, Optional

from sqlalchemy import text

from flask_app.services import expense_service as ex

logger = logging.getLogger(__name__)

ACCOUNTING_NOTIFY_ROLES = ("cfo", "accounting_manager", "accountant")
REVIEWER_ROLES = ("cfo", "accounting_manager")

#: Tests set this to send inline, so a check can read the history it wrote.
SYNC = False


def _users_with_email(engine) -> Dict[int, dict]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT id, username, role, email FROM users")).mappings().all()
    return {int(r["id"]): dict(r) for r in rows}


def _person(users, emps, uid) -> dict:
    u = users.get(int(uid)) or {}
    return {"id": int(uid), "name": ex._name(users, emps, uid),
            "email": (u.get("email") or "").strip() or None, "username": u.get("username")}


def _with_authority(engine, users, roles) -> List[int]:
    from flask_app.auth.sections import has_accounting_authority
    return [uid for uid, u in users.items()
            if u.get("role") in roles and has_accounting_authority(u, engine)]


def deciders(engine, report: dict, users=None, emps=None) -> List[dict]:
    """Who may decide this submitted report -- the people its submission is sent to."""
    users = users if users is not None else _users_with_email(engine)
    emps = emps if emps is not None else ex._employees(engine)
    kind = report.get("route_kind")
    if kind == "approver" and report.get("approver_user_id"):
        ids = [int(report["approver_user_id"])]
    elif kind == "cfo":
        ids = [k for k, u in users.items() if u.get("role") == "cfo"]
    else:                                   # the CFO's own report: CEO and President
        ids = sorted(ex._review_role_holders(engine))
    return [_person(users, emps, i) for i in ids if i != int(report["user_id"]) and i in users]


def accounting(engine, exclude_id=None, users=None, emps=None) -> List[dict]:
    users = users if users is not None else _users_with_email(engine)
    emps = emps if emps is not None else ex._employees(engine)
    return [_person(users, emps, i) for i in _with_authority(engine, users, ACCOUNTING_NOTIFY_ROLES)
            if exclude_id is None or i != int(exclude_id)]


def reviewers(engine, exclude_username=None, users=None, emps=None) -> List[dict]:
    users = users if users is not None else _users_with_email(engine)
    emps = emps if emps is not None else ex._employees(engine)
    return [_person(users, emps, i) for i in _with_authority(engine, users, REVIEWER_ROLES)
            if (users[i].get("username") or "") != (exclude_username or "")]


# ------------------------------------------------------------------ the messages

def _page(lead: str, rows: List[tuple], link: str, button: str, note: Optional[str] = None) -> str:
    esc = html.escape
    table = "".join(
        f'<tr><td style="color:#56606b;padding:2px 14px 2px 0">{esc(k)}</td>'
        f'<td style="padding:2px 0"><strong>{esc(str(v))}</strong></td></tr>' for k, v in rows)
    quote = (f'<p style="margin:12px 0;padding:10px 14px;background:#fff4e5;border-left:4px solid '
             f'#c77700;border-radius:4px">{esc(note)}</p>' if note else "")
    return (f'<div style="font-family:Segoe UI,Calibri,Arial,sans-serif;font-size:14.5px;'
            f'line-height:1.5;color:#1d2733;max-width:620px">'
            f'<p style="margin:6px 0">{esc(lead)}</p>'
            f'<table style="border-collapse:collapse;margin:8px 0">{table}</table>{quote}'
            f'<p style="margin:16px 0"><a href="{esc(link)}" style="display:inline-block;'
            f'background:#1f4e79;color:#fff;padding:9px 16px;border-radius:6px;'
            f'text-decoration:none;font-weight:600">{esc(button)}</a></p>'
            f'<p style="margin:6px 0;color:#56606b;font-size:12.5px">Waterfall XIRR · Expenses. '
            f'This message was sent automatically.</p></div>')


def _report_facts(engine, report: dict, users, emps) -> List[tuple]:
    with engine.connect() as c:
        n, total = c.execute(text("SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM er_lines "
                                  "WHERE report_id = :r"), {"r": int(report["id"])}).first()
    period = f"{report.get('period_start') or ''} to {report.get('period_end') or ''}"
    return [("Employee", ex._name(users, emps, report["user_id"])),
            ("Report", (report.get("title") or "").strip() or f"#{report['id']}"),
            ("Period", period), ("Lines", int(n or 0)), ("Total", f"${float(total or 0):,.2f}")]


# ------------------------------------------------------------------ sending

def _record(engine, report_id: int, person: dict, why: str, result: dict) -> None:
    who = f"{person['name']}{' <' + person['email'] + '>' if person.get('email') else ''}, {why}"
    with engine.begin() as c:
        ex._event(c, report_id, "emailed" if result.get("ok") else "email not sent",
                  {"id": None, "username": "the app"}, basis=who,
                  note=None if result.get("ok") else (result.get("error") or "not sent"))


def _nobody(engine, report_ids: List[int], who: str, why: str) -> None:
    """An action that should have notified someone and found NOBODY says so in the
    history -- an email that was never attempted must not read like one that went."""
    with engine.begin() as c:
        for rid in report_ids:
            ex._event(c, rid, "email not sent", {"id": None, "username": "the app"},
                      basis=who, note=why)


def _send_all(app, report_ids: List[int], messages: List[dict]) -> None:
    from flask_app.auth.email_utils import send_email_result
    from flask_app.db import get_engine
    with app.app_context():
        engine = get_engine()
        for m in messages:
            p = m["person"]
            if not p.get("email"):
                res = {"ok": False, "error": "no email address on file for this user"}
            else:
                try:
                    res = send_email_result(p["email"], m["subject"], m["html"])
                except Exception as e:                      # never let one stop the rest
                    res = {"ok": False, "error": str(e)}
            for rid in report_ids:
                try:
                    _record(engine, rid, p, m["why"], res)
                except Exception as e:
                    logger.warning("Could not record email for report %s: %s", rid, e)
            logger.info("Expense email %s -> %s: %s", m["subject"], p.get("email"),
                        "sent" if res.get("ok") else res.get("error"))


def _dispatch(report_ids: List[int], messages: List[dict]) -> None:
    if not messages:
        return
    from flask import current_app
    app = current_app._get_current_object()
    if SYNC:
        _send_all(app, report_ids, messages)
        return
    threading.Thread(target=_send_all, args=(app, report_ids, messages), daemon=True).start()


def report_event(engine, report_id: int, event: str, base_url: str,
                 actor: Optional[dict] = None, note: Optional[str] = None) -> int:
    """Queue the emails one report action calls for. Returns how many were queued.

    event: submitted | approved | returned.
    """
    report = ex._report_row(engine, report_id)
    if not report:
        return 0
    users, emps = _users_with_email(engine), ex._employees(engine)
    facts = _report_facts(engine, report, users, emps)
    who = ex._name(users, emps, report["user_id"])
    link = base_url.rstrip("/") + f"/expenses?report={int(report_id)}"
    actor_name = ex._name(users, emps, actor["id"]) if actor and actor.get("id") else "Accounting"
    msgs: List[dict] = []
    if event == "submitted":
        if not deciders(engine, report, users, emps):
            _nobody(engine, [int(report_id)], "no approver to email",
                    "nobody who may decide this report has an account")
        for p in deciders(engine, report, users, emps):
            msgs.append({"person": p, "why": "to approve",
                         "subject": f"Expense report to approve: {who}",
                         "html": _page(f"{who} submitted an expense report for your approval.",
                                       facts, link, "Open the report")})
    elif event == "approved":
        coding = base_url.rstrip("/") + "/expense-coding"
        if not accounting(engine, exclude_id=report["user_id"], users=users, emps=emps):
            _nobody(engine, [int(report_id)], "no one in accounting to email",
                    "no user holds the cfo, accounting_manager or accountant role with "
                    "Accounting access")
        for p in accounting(engine, exclude_id=report["user_id"], users=users, emps=emps):
            msgs.append({"person": p, "why": "to code",
                         "subject": f"Expense report approved, ready to code: {who}",
                         "html": _page(f"{actor_name} approved {who}'s expense report. It is "
                                       f"ready to code.", facts, coding, "Open Expense Coding")})
        emp = _person(users, emps, report["user_id"])
        msgs.append({"person": emp, "why": "report approved",
                     "subject": "Your expense report was approved",
                     "html": _page(f"{actor_name} approved your expense report. Accounting "
                                   f"will reimburse it through payroll.", facts, link,
                                   "Open the report")})
    elif event == "returned":
        emp = _person(users, emps, report["user_id"])
        msgs.append({"person": emp, "why": "report returned",
                     "subject": "Your expense report was returned",
                     "html": _page(f"{actor_name} returned your expense report. Make the change "
                                   f"and submit it again.", facts, link, "Open the report",
                                   note=f"Why: {note}" if note else None)})
    _dispatch([int(report_id)], msgs)
    return len(msgs)


def coding_event(engine, report_ids: List[int], event: str, base_url: str,
                 actor: dict, note: Optional[str] = None, submitter: Optional[str] = None) -> int:
    """event: coding_submitted (to the reviewers) | coding_returned (to the submitter)."""
    if not report_ids:
        return 0
    users, emps = _users_with_email(engine), ex._employees(engine)
    actor_name = ex._name(users, emps, actor["id"]) if actor.get("id") else actor.get("username")
    names = []
    for rid in report_ids:
        r = ex._report_row(engine, rid)
        if r:
            names.append(f"{ex._name(users, emps, r['user_id'])} (#{rid})")
    rows = [("Reports", ", ".join(names)), ("By", actor_name)]
    link = base_url.rstrip("/") + "/expense-coding"
    msgs: List[dict] = []
    if event == "coding_submitted":
        if not reviewers(engine, exclude_username=actor.get("username"), users=users, emps=emps):
            _nobody(engine, [int(x) for x in report_ids], "no reviewer to email",
                    "no other user holds the cfo or accounting_manager role with Accounting access")
        for p in reviewers(engine, exclude_username=actor.get("username"), users=users, emps=emps):
            msgs.append({"person": p, "why": "to review the coding",
                         "subject": f"Expense coding to review: {len(names)} report(s)",
                         "html": _page(f"{actor_name} submitted expense coding for your review.",
                                       rows, link, "Open Expense Coding")})
    elif event == "coding_returned" and submitter:
        uid = next((k for k, u in users.items() if u.get("username") == submitter), None)
        if uid is not None:
            msgs.append({"person": _person(users, emps, uid), "why": "coding returned",
                         "subject": "Expense coding returned for changes",
                         "html": _page(f"{actor_name} returned expense coding to you.", rows, link,
                                       "Open Expense Coding", note=f"Why: {note}" if note else None)})
    _dispatch([int(x) for x in report_ids], msgs)
    return len(msgs)
