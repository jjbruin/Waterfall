"""Expense reports -- phase 3: accounting's coding, and the MRI upload.

Accounting's side, from their description and three files (design:
`.claude/memory/expense_reporting.md`). Every approved line is PRE-CODED and
accounting corrects what needs it; nothing they decide is overwritten, and
what the app proposed stays beside it.

THE BOOKING FOLLOWS THE DEAL (measured on the Sep 24 upload, confirmed by Jim):
  operations -> PSC Manager, the line's expense category
  pipeline   -> PSC Manager, Deal Cost Receivable (MR11000012)
  owned deal -> PSC Manager, Due to/from Intercompany (MR15000001) with
                RLTDENTITY = the entity that owns the expense; AND at that
                entity, the expense (debit) against Due To/From PSC Manager
                (MR15000002, credit). One line per owning entity.

WHICH ENTITY OWNS THE EXPENSE IS READ, NOT NAMED. The deal's ownership chain is
`ownership_chain_service.build_chain` -- commitments, each share derived from
committed DOLLARS (the stored percentages were shown wrong) -- and the walk up
each branch stops at the first owner that KEEPS AN INTERCOMPANY ACCOUNT WITH
PSC MANAGER: it carries MR15000002 in the GL, or PSC Manager carries it as a
segment of MR15000001. That one rule reproduces both of accounting's examples,
which a rule about names could not: Gallery's PPI25 "is a simple pass through"
(it keeps no such account) so the cost goes to PSCKOC 70% / PSC3 30%; Apple's
PPI2 does keep one, so Apple's cost stops at PPI2. Branches that never reach
such an entity (an operating partner's share) are not PSC's to bear, so the
shares are re-based over the branches that do. RLTDENTITY at the entity is the
entity directly below it on the path. It is a PROPOSAL: accounting may set the
entities and percentages by hand.

AMOUNTS ARE SPLIT BY `treasury_upload.allocate`, the largest-remainder split the
investor allocation uses, so a split line sums to its total to the cent.

THE FILE IS `treasury_upload.build_gl_csv`, which reproduces files MRI accepted
byte for byte, and `validate_gl` refuses an entry out of balance IN TOTAL OR BY
ENTITY -- the Sep 24 upload is out by 0.30 on PPI2, and would be refused.

ONE BATCH PER PAYROLL DATE (Jim, Oct 2 2026). The credit is one line at PSC
Manager to MR20000001, reimbursed through TriNet payroll. A batched report is
locked to it; voiding the batch releases its reports. Recurring reimbursements
accounting adds by hand today ("FK - Benefits") are standing items here,
included in each batch unless unticked.

A NON-USD ENTITY BOOKS IN ITS OWN CURRENCY. PPI2's lines in the Sep 24 file
are Canadian dollars with the US amount in the description. The app holds no
FX rate, so the batch takes one per such entity and says which it used.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from typing import Dict, List, Optional

from sqlalchemy import bindparam, text

from flask_app.services import expense_service as ex
from flask_app.services.intercompany_service import (
    BASES, ENTITY_ACCOUNT, MANAGER_ACCOUNT, MANAGER_ENTITY)

logger = logging.getLogger(__name__)

DEAL_COST_ACCOUNT = "MR11000012"     # Deal Cost Receivable
PAYROLL_ACCOUNT = "MR20000001"       # reimbursed through TriNet payroll
BASIS = "B"
BOOKINGS = ("expense", "deal_cost", "interco")
DEFAULT_BOOKING = {"operations": "expense", "pipeline": "deal_cost", "deal": "interco"}

_DONE: set = set()


def ensure_tables(engine) -> None:
    ex.ensure_tables(engine)
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _DONE:
        return
    pk = "SERIAL PRIMARY KEY" if engine.dialect.name == "postgresql" else \
        "INTEGER PRIMARY KEY AUTOINCREMENT"
    with engine.begin() as c:
        # Accounting's decisions for one line, or one split of a line
        # (split_id 0 = the whole line). Only what they CHANGED is stored; the
        # proposal is recomputed, so a corrected chart or ownership record
        # reaches every line nobody has decided by hand.
        c.execute(text("""CREATE TABLE IF NOT EXISTS er_coding (
            line_id         INTEGER NOT NULL,
            split_id        INTEGER NOT NULL DEFAULT 0,
            description     TEXT,
            expense_account TEXT,
            booking         TEXT,
            interco_json    TEXT,
            updated_by      TEXT,
            updated_at      TEXT,
            PRIMARY KEY (line_id, split_id))"""))
        c.execute(text("""CREATE TABLE IF NOT EXISTS er_entity_currency (
            entity_id TEXT PRIMARY KEY,
            currency  TEXT NOT NULL,
            set_by    TEXT,
            set_at    TEXT)"""))
        c.execute(text(f"""CREATE TABLE IF NOT EXISTS er_recurring (
            id          {pk},
            user_id     INTEGER NOT NULL,
            description TEXT NOT NULL,
            account     TEXT NOT NULL,
            amount      DOUBLE PRECISION NOT NULL,
            active      BOOLEAN NOT NULL DEFAULT TRUE,
            updated_by  TEXT,
            updated_at  TEXT)"""))
        c.execute(text("""CREATE TABLE IF NOT EXISTS er_batches (
            batch_id     TEXT PRIMARY KEY,
            payroll_date TEXT NOT NULL,
            period       TEXT NOT NULL,
            report_ids   TEXT NOT NULL,
            recurring    TEXT,
            credit_description TEXT NOT NULL,
            fx_json      TEXT,
            csv          TEXT NOT NULL,
            total        DOUBLE PRECISION,
            created_by   TEXT,
            created_at   TEXT,
            voided_by    TEXT,
            voided_at    TEXT)"""))
    from sqlalchemy import inspect
    have = {c["name"] for c in inspect(engine).get_columns("er_reports")}
    # batch_id, and the coding review (Oct 7 2026) -- see `review_state`.
    for col in ("batch_id", "coding_state", "coding_submitted_by", "coding_submitted_at",
                "coding_reviewed_by", "coding_reviewed_at"):
        if col not in have:
            # TWO REQUESTS RACE HERE: the coding screen loads its lines and its settings at
            # once, both find the column missing, and the second ALTER fails "duplicate
            # column" -- which returned a 500 and an empty grid (found Oct 7 2026). A column
            # that appeared in between is the outcome wanted; anything else is raised.
            try:
                with engine.begin() as c:
                    c.execute(text("ALTER TABLE er_reports ADD COLUMN %s TEXT" % col))
            except Exception:
                if col not in {x["name"] for x in inspect(engine).get_columns("er_reports")}:
                    raise
    _DONE.add(key)


# ------------------------------------------------------------------ the coding review
#
# The CFO, Oct 7 2026: "Do we have the ability to have Ken/Nghia submit the expense coding
# for final review?" Jim: yes, the CFO approves, and only reviewed reports can be batched.
#
#   (not submitted) --submit (accounting)--> submitted --approve (reviewer)--> reviewed
#                          ^                     |  return, with a reason       |
#                          +---------------------+                              |
#                          +--- the coding changed after review: back to submitted
#
# A report returned to the employee and approved AGAIN starts over: the state counts only
# if it was submitted after the report's latest approval (`decided_at`), so nothing in the
# approval code has to remember to clear it.
REVIEWER_ROLES = ("cfo", "accounting_manager")


def review_state(rep: dict) -> str:
    """'' (not submitted) | 'submitted' | 'reviewed', for an approved report."""
    st = (rep.get("coding_state") or "").strip()
    if st and (rep.get("coding_submitted_at") or "") < (rep.get("decided_at") or ""):
        return ""                                    # submitted before a re-approval
    return st if st in ("submitted", "reviewed") else ""


def can_review(user: dict, engine=None) -> bool:
    from flask_app.auth.sections import has_accounting_authority
    return bool(user) and user.get("role") in REVIEWER_ROLES and \
        has_accounting_authority(user, engine)


def submit_coding(engine, report_ids, actor: dict) -> dict:
    """Accounting submits these reports' coding for review. Refused while any line has a
    problem -- a reviewer should not be asked to approve coding that cannot be batched."""
    ensure_tables(engine)
    ids = sorted({int(x) for x in (report_ids or [])})
    if not ids:
        raise ValueError("Choose the reports to submit.")
    reports = {int(r["id"]): r for r in _approved_reports(engine, ids)}
    missing = [i for i in ids if i not in reports]
    if missing:
        raise ValueError("Report(s) %s are not approved and unbatched." % ", ".join(map(str, missing)))
    already = [i for i in ids if review_state(reports[i])]
    if already:
        raise ValueError("Report(s) %s are already %s." % (", ".join(map(str, already)),
                         review_state(reports[already[0]])))
    bad = sorted({r["report_id"] for r in coding_rows(engine, ids) if r["problems"]})
    if bad:
        raise ValueError("Report(s) %s still have coding problems to fix first."
                         % ", ".join(map(str, bad)))
    now = ex._now()
    with engine.begin() as c:
        for i in ids:
            c.execute(text("UPDATE er_reports SET coding_state = 'submitted', "
                           "coding_submitted_by = :by, coding_submitted_at = :at, "
                           "coding_reviewed_by = NULL, coding_reviewed_at = NULL WHERE id = :i"),
                      {"by": actor.get("username"), "at": now, "i": i})
            ex._event(c, i, "coding submitted for review", actor)
    return {"submitted": ids}


def review_coding(engine, report_ids, action: str, note: Optional[str], actor: dict) -> dict:
    """A reviewer approves submitted coding, or returns it with a reason.

    Returns the submitters, so the caller can tell each whose coding came back."""
    ensure_tables(engine)
    if not can_review(actor, engine):
        raise PermissionError("Only the CFO or the accounting manager reviews expense coding.")
    if action not in ("approve", "return"):
        raise ValueError("The action is approve or return.")
    note = (note or "").strip() or None
    if action == "return" and not note:
        raise ValueError("Say why the coding is being returned.")
    ids = sorted({int(x) for x in (report_ids or [])})
    reports = {int(r["id"]): r for r in _approved_reports(engine, ids)}
    not_waiting = [i for i in ids if i not in reports or review_state(reports[i]) != "submitted"]
    if not ids or not_waiting:
        raise ValueError("Report(s) %s are not waiting for review."
                         % ", ".join(map(str, not_waiting or ids)))
    now = ex._now()
    submitters: Dict[str, List[int]] = {}
    with engine.begin() as c:
        for i in ids:
            submitters.setdefault(reports[i].get("coding_submitted_by") or "", []).append(i)
            if action == "approve":
                c.execute(text("UPDATE er_reports SET coding_state = 'reviewed', "
                               "coding_reviewed_by = :by, coding_reviewed_at = :at WHERE id = :i"),
                          {"by": actor.get("username"), "at": now, "i": i})
                ex._event(c, i, "coding reviewed", actor)
            else:
                c.execute(text("UPDATE er_reports SET coding_state = NULL, "
                               "coding_reviewed_by = NULL, coding_reviewed_at = NULL WHERE id = :i"),
                          {"i": i})
                ex._event(c, i, "coding returned", actor, note=note)
    return {"action": action, "reports": ids, "submitters": submitters}


# ------------------------------------------------------------------ who owns it

def interco_population(engine) -> set:
    """Entities that keep an intercompany account with PSC Manager, from the GL."""
    from sqlalchemy import inspect
    if not inspect(engine).has_table("gl_detail"):
        return set()
    q = lambda n: '"%s"' % n
    stmt = text(
        f"SELECT DISTINCT UPPER(TRIM({q('ENTITYID')})) FROM gl_detail WHERE {q('ACCTNUM')} = :e "
        f"AND {q('BASIS')} IN :b UNION SELECT DISTINCT UPPER(TRIM({q('RLTDENTITY')})) FROM "
        f"gl_detail WHERE {q('ENTITYID')} = :m AND {q('ACCTNUM')} = :ma AND {q('BASIS')} IN :b"
    ).bindparams(bindparam("b", expanding=True))
    with engine.connect() as c:
        rows = c.execute(stmt, {"e": ENTITY_ACCOUNT, "m": MANAGER_ENTITY,
                                "ma": MANAGER_ACCOUNT, "b": list(BASES)}).fetchall()
    return {r[0] for r in rows if r[0] and r[0] != MANAGER_ENTITY}


def _deal_investment_id(engine, vcode: str) -> Optional[str]:
    with engine.connect() as c:
        r = c.execute(text('SELECT "InvestmentID" FROM deals WHERE vcode = :v'),
                      {"v": vcode}).first()
    return (str(r[0]).strip().upper() if r and r[0] else None)


def propose_interco(engine, vcode: str, population=None, chain=None) -> dict:
    """{allocations: [{entity, pct, rltd, path}], basis, error} for an owned deal."""
    population = population if population is not None else interco_population(engine)
    iid = _deal_investment_id(engine, vcode)
    if not iid:
        return {"allocations": [], "error": "The deal %s has no InvestmentID, so its "
                "ownership cannot be walked. Set the entities by hand." % vcode}
    if not population:
        return {"allocations": [], "error": "No entity in the GL keeps an intercompany "
                "account with PSC Manager, so none can be proposed. Set them by hand."}
    if chain is None:
        from flask_app.services.ownership_chain_service import build_chain
        chain = build_chain(iid, engine=engine)
    found: Dict[str, dict] = {}

    def walk(nodes, share_above, path):
        for n in nodes or []:
            pct = n.get("pct")
            if pct is None:
                continue
            share = share_above * pct / 100.0
            eid = str(n.get("entity_id") or "").upper()
            here = path + [eid]
            if eid in population:
                f = found.setdefault(eid, {"entity": eid, "weight": 0.0,
                                           "rltd": str(n.get("into_entity_id") or "").upper(),
                                           "path": " <- ".join([iid] + here)})
                f["weight"] += share
                continue          # stop at the first such entity on this branch
            walk(n.get("owners"), share, here)

    walk((chain.get("root") or {}).get("owners"), 1.0, [])
    total = sum(f["weight"] for f in found.values())
    if not found or total <= 0:
        return {"allocations": [], "error": "No owner of %s up the chain keeps an "
                "intercompany account with PSC Manager. Set the entities by hand." % iid}
    allocs = sorted(({"entity": f["entity"], "pct": round(100.0 * f["weight"] / total, 6),
                      "rltd": f["rltd"], "path": f["path"]} for f in found.values()),
                    key=lambda a: -a["pct"])
    basis = ("Commitment-dollar shares of %s's owners, walked up to the first entity "
             "that keeps an intercompany account with PSC Manager" % iid)
    if abs(total - 1.0) > 1e-6:
        basis += ("; %.2f%% of the deal is held outside those entities and the shares "
                  "are re-based over the rest" % (100.0 * (1 - total)))
    return {"allocations": allocs, "basis": basis, "error": None}


# ------------------------------------------------------------------ the grid

def _approved_reports(engine, report_ids=None) -> List[dict]:
    with engine.connect() as c:
        rows = [dict(r) for r in c.execute(text(
            "SELECT * FROM er_reports WHERE status = 'approved' AND batch_id IS NULL "
            "ORDER BY period_end, id")).mappings().all()]
    if report_ids is not None:
        want = {int(x) for x in report_ids}
        rows = [r for r in rows if int(r["id"]) in want]
    return rows


def _codings(engine, line_ids) -> Dict[tuple, dict]:
    if not line_ids:
        return {}
    with engine.connect() as c:
        rows = c.execute(text("SELECT * FROM er_coding WHERE line_id IN :ids").bindparams(
            bindparam("ids", expanding=True)), {"ids": list(line_ids)}).mappings().all()
    out = {}
    for r in rows:
        d = dict(r)
        d["interco"] = json.loads(d.pop("interco_json") or "null")
        out[(int(r["line_id"]), int(r["split_id"]))] = d
    return out


def initials(full_name: str) -> str:
    """Fred Kurz -> FK. MRI's 80 characters are better spent on the expense."""
    return "".join(w[0] for w in str(full_name or "").split() if w[:1].isalnum()).upper()


def _desc(employee, deal_kind, deal_name, comment, interco):
    """"ER FK Pontchartrain Landing Site Visit Dinner" -- MRI's rule (Jim, Oct 2
    2026): 80 characters, no punctuation. Intercompany is prefixed "IC". The
    comment shortens first, then the deal; the prefix and initials never do."""
    from flask_app.services.treasury_upload import mri_description
    from flask_app.services.treasury_upload import mri_text
    prefix = "IC ER" if interco else "ER"
    deal = deal_name if deal_kind != "operations" else ""
    # A comment often repeats the deal ("Pontchartrain Site Visit" on Pontchartrain
    # Landing); those words are already there. Only words of four letters or more,
    # so "at" or "to" in a comment is never taken for the deal's.
    seen = {w.lower() for w in mri_text(deal).split() if len(w) >= 4}
    comment = " ".join(w for w in mri_text(comment).split() if w.lower() not in seen)
    return mri_description(prefix, initials(employee), deal, comment, keep=2)


def coding_rows(engine, report_ids=None) -> List[dict]:
    """One row per approved line, or per split of a line, proposed and as decided."""
    ensure_tables(engine)
    users, emps = ex._users(engine), ex._employees(engine)
    cats = {c["account"]: c["name"] for c in ex.options(engine)["categories"]}
    population = None
    chains: Dict[str, dict] = {}
    rows = []
    reports = _approved_reports(engine, report_ids)
    all_lines = {r["id"]: ex._lines(engine, r["id"]) for r in reports}
    # Possible duplicates travel to accounting too -- they are the last to see
    # a line before it is paid.
    dupes = {}
    for rep in reports:
        dupes.update(ex.possible_duplicates(engine, rep, all_lines[rep["id"]]))
    decided = _codings(engine, [ln["id"] for ls in all_lines.values() for ln in ls])
    for rep in reports:
        employee = ex._name(users, emps, rep["user_id"])
        named = bool(((emps.get(int(rep["user_id"])) or {}).get("full_name") or "").strip())
        for ln in all_lines[rep["id"]]:
            parts = ([(s["id"], s["deal_code"], s["deal_kind"], s["deal_name"], s["amount"])
                      for s in ln["splits"]] or
                     [(0, ln["deal_code"], ln["deal_kind"], ln["deal_name"], ln["amount"])])
            for split_id, dcode, dkind, dname, amount in parts:
                booking_p = DEFAULT_BOOKING.get(dkind or "", "expense")
                d = decided.get((ln["id"], split_id)) or {}
                booking = d.get("booking") or booking_p
                interco_p = None
                if booking == "interco" and dkind == "deal" and dcode:
                    if population is None:
                        population = interco_population(engine)
                    if dcode not in chains:
                        chains[dcode] = propose_interco(engine, dcode, population)
                    interco_p = chains[dcode]
                interco = d.get("interco") or ((interco_p or {}).get("allocations"))
                exp_acct = d.get("expense_account") or ln.get("category_account")
                desc_p = _desc(employee, dkind, dname, ln.get("comment"), booking == "interco")
                problems = []
                if not named:
                    problems.append("the employee has no name on reports (Expenses > "
                                    "Employees & approvers)")
                if booking == "interco":
                    if not interco:
                        problems.append((interco_p or {}).get("error") or
                                        "no entity is set for the intercompany booking")
                    elif abs(sum(float(a["pct"]) for a in interco) - 100.0) > 0.0001:
                        problems.append("the entity shares total %.4f%%, not 100%%"
                                        % sum(float(a["pct"]) for a in interco))
                if booking in ("expense", "interco") and not exp_acct:
                    problems.append("no expense account")
                rows.append({
                    "report_id": rep["id"], "line_id": ln["id"], "split_id": split_id,
                    "employee": employee, "line_date": ln.get("line_date"),
                    "line_date_end": ln.get("line_date_end"),
                    "deal_kind": dkind, "deal_code": dcode, "deal_name": dname,
                    "comment": ln.get("comment"), "vendor": ln.get("vendor"),
                    "amount": amount, "receipt_id": ln.get("receipt_id"),
                    "receipt_page": ln.get("receipt_page"),
                    "employee_account": ln.get("category_account"),
                    "employee_category": cats.get(ln.get("category_account"), ""),
                    "booking": booking, "booking_proposed": booking_p,
                    "expense_account": exp_acct,
                    "expense_account_changed": bool(d.get("expense_account")),
                    "description": d.get("description") or desc_p,
                    "description_proposed": desc_p,
                    "interco": interco, "interco_proposed": interco_p,
                    "interco_changed": bool(d.get("interco")),
                    "decided_by": d.get("updated_by"), "decided_at": d.get("updated_at"),
                    "review_state": review_state(rep),
                    "coding_submitted_by": rep.get("coding_submitted_by"),
                    "coding_submitted_at": rep.get("coding_submitted_at"),
                    "coding_reviewed_by": rep.get("coding_reviewed_by"),
                    "coding_reviewed_at": rep.get("coding_reviewed_at"),
                    "problems": problems,
                    "warnings": ["may be a duplicate: " + w for w in dupes.get(ln["id"], [])],
                })
    return rows


def _clean_desc(v) -> Optional[str]:
    """Accounting's typed description, held to MRI's rule as it is saved."""
    from flask_app.services.treasury_upload import mri_description
    return mri_description(v) or None


def save_coding(engine, line_id: int, split_id: int, body: dict, by: str) -> dict:
    ensure_tables(engine)
    with engine.connect() as c:
        ok = c.execute(text(
            "SELECT r.status, r.batch_id, r.id, r.coding_state, r.coding_submitted_at, "
            "r.decided_at FROM er_lines l JOIN er_reports r ON r.id = l.report_id "
            "WHERE l.id = :l"), {"l": int(line_id)}).first()
    if not ok:
        raise LookupError("No line %s." % line_id)
    if ok[0] != "approved" or ok[1]:
        raise PermissionError("Only an approved report not yet in a batch can be coded.")
    # Coding changed AFTER review goes back for review: a reviewer approved what they saw.
    reviewed = review_state({"coding_state": ok[3], "coding_submitted_at": ok[4],
                             "decided_at": ok[5]}) == "reviewed"
    booking = body.get("booking") or None
    if booking and booking not in BOOKINGS:
        raise ValueError("Booking is one of %s." % ", ".join(BOOKINGS))
    interco = body.get("interco")
    if interco:
        clean = []
        for a in interco:
            ent = str(a.get("entity") or "").strip().upper()
            try:
                pct = float(a.get("pct"))
            except (TypeError, ValueError):
                raise ValueError("Each entity needs a percentage.")
            if not ent or pct <= 0:
                raise ValueError("Each entity needs a name and a positive share.")
            clean.append({"entity": ent, "pct": pct,
                          "rltd": str(a.get("rltd") or "").strip().upper()})
        if abs(sum(a["pct"] for a in clean) - 100.0) > 0.0001:
            raise ValueError("The entity shares must total 100%%; they total %.4f%%."
                             % sum(a["pct"] for a in clean))
        interco = clean
    with engine.begin() as c:
        c.execute(text("DELETE FROM er_coding WHERE line_id = :l AND split_id = :s"),
                  {"l": int(line_id), "s": int(split_id)})
        c.execute(text(
            "INSERT INTO er_coding (line_id, split_id, description, expense_account, booking, "
            "interco_json, updated_by, updated_at) VALUES (:l, :s, :d, :a, :b, :i, :by, :at)"),
            {"l": int(line_id), "s": int(split_id),
             "d": _clean_desc(body.get("description")),
             "a": (body.get("expense_account") or "").strip() or None, "b": booking,
             "i": json.dumps(interco) if interco else None, "by": by, "at": ex._now()})
        if reviewed:
            c.execute(text("UPDATE er_reports SET coding_state = 'submitted', "
                           "coding_reviewed_by = NULL, coding_reviewed_at = NULL WHERE id = :i"),
                      {"i": int(ok[2])})
            ex._event(c, int(ok[2]), "coding changed after review -- back for review",
                      {"id": None, "username": by})
    return {"saved": True, "back_for_review": reviewed}


# ------------------------------------------------------------------ settings

def currencies(engine) -> Dict[str, str]:
    ensure_tables(engine)
    with engine.connect() as c:
        return {r[0]: r[1] for r in c.execute(text(
            "SELECT entity_id, currency FROM er_entity_currency")).fetchall()}


def set_currency(engine, entity_id: str, currency: str, by: str) -> Dict[str, str]:
    ensure_tables(engine)
    e = (entity_id or "").strip().upper()
    cur = (currency or "").strip().upper()
    if not e or len(cur) != 3:
        raise ValueError("An entity and a three-letter currency are required.")
    with engine.begin() as c:
        c.execute(text("DELETE FROM er_entity_currency WHERE entity_id = :e"), {"e": e})
        if cur != "USD":           # USD is the default and is not stored
            c.execute(text("INSERT INTO er_entity_currency (entity_id, currency, set_by, "
                           "set_at) VALUES (:e, :c, :by, :at)"),
                      {"e": e, "c": cur, "by": by, "at": ex._now()})
    return currencies(engine)


def recurring(engine) -> List[dict]:
    ensure_tables(engine)
    users, emps = ex._users(engine), ex._employees(engine)
    with engine.connect() as c:
        rows = [dict(r) for r in c.execute(text(
            "SELECT * FROM er_recurring ORDER BY id")).mappings().all()]
    for r in rows:
        r["employee"] = ex._name(users, emps, r["user_id"])
        r["active"] = bool(r["active"])
    return rows


def save_recurring(engine, body: dict, by: str, item_id=None) -> List[dict]:
    ensure_tables(engine)
    uid = body.get("user_id")
    if uid in (None, "") or int(uid) not in ex._users(engine):
        raise ValueError("Choose the employee the reimbursement is for.")
    desc = (body.get("description") or "").strip()
    acct = (body.get("account") or "").strip().upper()
    amt = ex._money(body.get("amount"), "Amount")
    if not desc or not acct or not amt:
        raise ValueError("A description, an account and an amount are required.")
    vals = {"u": int(uid), "d": desc, "a": acct, "m": amt,
            "act": bool(body.get("active", True)), "by": by, "at": ex._now()}
    with engine.begin() as c:
        if item_id is None:
            c.execute(text("INSERT INTO er_recurring (user_id, description, account, amount, "
                           "active, updated_by, updated_at) VALUES (:u, :d, :a, :m, :act, :by, "
                           ":at)"), vals)
        else:
            got = c.execute(text("UPDATE er_recurring SET user_id = :u, description = :d, "
                                 "account = :a, amount = :m, active = :act, updated_by = :by, "
                                 "updated_at = :at WHERE id = :i"), {**vals, "i": int(item_id)})
            if got.rowcount != 1:
                raise LookupError("No recurring item %s." % item_id)
    return recurring(engine)


# ------------------------------------------------------------------ the batch

def _us(iso: str) -> str:
    d = date.fromisoformat(iso)
    return "%d/%d/%d" % (d.month, d.day, d.year)


def build_batch(engine, body: dict, by: str, commit: bool = False) -> dict:
    """The journal entry for one payroll date: preview, or store it and return the file.

    body: report_ids, payroll_date (YYYY-MM-DD), period (YYYYMM, default the
    payroll date's), credit_suffix, fx {entity: rate}, recurring_ids.
    """
    from flask_app.services import treasury_upload as tu
    ensure_tables(engine)
    pdate = ex._date(body.get("payroll_date"), "Payroll date")
    period = str(body.get("period") or pdate[:4] + pdate[5:7]).strip()
    report_ids = [int(x) for x in (body.get("report_ids") or [])]
    errors, warnings = [], []
    reports = _approved_reports(engine, report_ids)
    missing = sorted(set(report_ids) - {int(r["id"]) for r in reports})
    if missing:
        errors.append("Report(s) %s are not approved and unbatched."
                      % ", ".join(map(str, missing)))
    # ONLY REVIEWED CODING IS BATCHED (Jim, Oct 7 2026).
    for r in reports:
        st = review_state(r)
        if st != "reviewed":
            errors.append("Report %s's coding %s." % (r["id"], "is waiting for review"
                          if st == "submitted" else "has not been submitted for review"))
    rows = coding_rows(engine, report_ids) if reports else []
    rec_ids = {int(x) for x in (body.get("recurring_ids") or [])}
    recs = [r for r in recurring(engine) if r["id"] in rec_ids and r["active"]]
    # THE MONTHLY CELL PHONE REIMBURSEMENT (expense_phone): every month owed through the
    # payroll month, once. On unless the batch says phone=False.
    from flask_app.services import expense_phone
    phone = (expense_phone.due(engine, pdate[:7]) if body.get("phone", True)
             else {"items": [], "problems": []})
    warnings += phone["problems"]
    if not rows and not recs and not phone["items"]:
        errors.append("There is nothing to batch.")
    for r in rows:
        for p in r["problems"]:
            errors.append("%s, %s %s: %s" % (r["employee"], r["line_date"] or "",
                                             (r["comment"] or "")[:40], p))
    fx = {str(k).upper(): v for k, v in (body.get("fx") or {}).items()}
    cur = currencies(engine)

    def gl(entity, acct, amount, desc, rltd=""):
        return {"entityid": entity, "acctnum": acct, "amount": round(amount, 2),
                "descrpn": desc, "rltdentity": rltd, "period": period, "basis": BASIS,
                "entrdate": pdate}

    manager, entity_side, used_fx = [], [], {}
    for r in rows:
        if r["problems"]:
            continue
        r["description"] = tu.mri_description(r["description"])
        amt = float(r["amount"] or 0)
        if r["booking"] == "expense":
            manager.append(gl(MANAGER_ENTITY, r["expense_account"], amt, r["description"]))
        elif r["booking"] == "deal_cost":
            manager.append(gl(MANAGER_ENTITY, DEAL_COST_ACCOUNT, amt, r["description"]))
        else:
            split = tu.allocate(amt, {a["entity"]: a["pct"] for a in r["interco"]})
            rltd = {a["entity"]: a.get("rltd") or "" for a in r["interco"]}
            for part in split["rows"]:
                ent, usd = part["investorid"], part["amount"]
                manager.append(gl(MANAGER_ENTITY, MANAGER_ACCOUNT, usd, r["description"], ent))
                c = cur.get(ent, "USD")
                local, desc = usd, r["description"]
                if c != "USD":
                    try:
                        rate = float(fx.get(ent))
                    except (TypeError, ValueError):
                        errors.append("%s books in %s: enter its USD-to-%s rate for this "
                                      "batch." % (ent, c, c))
                        continue
                    used_fx[ent] = {"currency": c, "rate": rate}
                    local = round(usd * rate, 2)
                    # "USD 712 98": the amount without a decimal point, which MRI
                    # counts as punctuation; its room is reserved before trimming.
                    desc = tu.mri_description(desc, suffix="USD %s" % "{:.2f}".format(usd))
                entity_side.append(gl(ent, r["expense_account"], local, desc, rltd[ent]))
                entity_side.append(gl(ent, ENTITY_ACCOUNT, -local, desc))
    for rc in recs:
        manager.append(gl(MANAGER_ENTITY, rc["account"], float(rc["amount"]),
                          tu.mri_description("ER", initials(rc["employee"]), rc["description"],
                                             keep=2)))
    emp_names = {int(e["user_id"]): e.get("full_name") or e.get("username") for e in ex.employees(engine)}
    for it in phone["items"]:
        manager.append(gl(MANAGER_ENTITY, expense_phone.PHONE_ACCOUNT, it["amount"],
                          tu.mri_description("ER", initials(emp_names.get(it["user_id"]) or it["employee"]),
                                             "Cell Phone Reimbursement",
                                             expense_phone.month_label(it["month"]), keep=2)))
    credit_desc = tu.mri_description("ER Trinet Payroll", period,
                                     body.get("credit_suffix") or "", keep=2)
    total = round(sum(ln["amount"] for ln in manager), 2)
    lines = manager + ([gl(MANAGER_ENTITY, PAYROLL_ACCOUNT, -total, credit_desc)]
                       if manager else []) + entity_side
    v = tu.validate_gl(lines) if lines else {"errors": [], "warnings": []}
    errors += v["errors"]
    warnings += [w for w in v.get("warnings", []) if "more than one entity" not in w]
    out = {"lines": lines, "errors": errors, "warnings": warnings, "total": total,
           "period": period, "payroll_date": pdate, "fx_used": used_fx,
           "credit_description": credit_desc, "reports": [r["id"] for r in reports],
           "recurring": [r["id"] for r in recs],
           "phone": phone["items"],
           "phone_total": round(sum(it["amount"] for it in phone["items"]), 2)}
    if not commit:
        return out
    if errors:
        return out
    csv_text = tu.build_gl_csv(lines)
    batch_id = "ER-%s-%s" % (pdate.replace("-", ""), uuid.uuid4().hex[:6].upper())
    with engine.begin() as c:
        # Claimed by a conditional UPDATE, so a report batched by someone else a
        # moment ago is not batched twice.
        for r in reports:
            got = c.execute(text("UPDATE er_reports SET batch_id = :b, status = 'batched', "
                                 "updated_at = :at WHERE id = :i AND status = 'approved' "
                                 "AND batch_id IS NULL"),
                            {"b": batch_id, "at": ex._now(), "i": r["id"]})
            if got.rowcount != 1:
                raise ValueError("Report %s was batched by someone else just now." % r["id"])
            ex._event(c, r["id"], "batched", {"username": by}, basis=batch_id)
        c.execute(text(
            "INSERT INTO er_batches (batch_id, payroll_date, period, report_ids, recurring, "
            "credit_description, fx_json, csv, total, created_by, created_at) VALUES "
            "(:b, :pd, :p, :r, :rc, :cd, :fx, :csv, :t, :by, :at)"),
            {"b": batch_id, "pd": pdate, "p": period, "r": json.dumps(out["reports"]),
             "rc": json.dumps(out["recurring"]), "cd": credit_desc, "fx": json.dumps(used_fx),
             "csv": csv_text, "t": total, "by": by, "at": ex._now()})
        # The months this batch pays, in the same transaction: a month another batch
        # paid a moment ago fails the ledger's key and nothing is written.
        try:
            expense_phone.record_paid(c, phone["items"], batch_id)
        except Exception:
            raise ValueError("A cell phone reimbursement in this batch was just paid by another "
                             "batch. Preview again.")
    out.update({"batch_id": batch_id, "csv": csv_text})
    return out


def batches(engine) -> List[dict]:
    """Every batch, each saying whether the GL shows its payroll credit yet."""
    ensure_tables(engine)
    with engine.connect() as c:
        rows = [dict(r) for r in c.execute(text(
            "SELECT batch_id, payroll_date, period, report_ids, recurring, credit_description, "
            "fx_json, total, created_by, created_at, voided_by, voided_at FROM er_batches "
            "ORDER BY created_at DESC")).mappings().all()]
    posted = set()
    from sqlalchemy import inspect
    live = [r for r in rows if not r["voided_at"]]
    if live and inspect(engine).has_table("gl_detail"):
        stmt = text('SELECT "DESCRPN", "AMT" FROM gl_detail WHERE "ENTITYID" = :m AND '
                    '"ACCTNUM" = :a AND "PERIOD" IN :ps').bindparams(
            bindparam("ps", expanding=True))
        with engine.connect() as c:
            for d, a in c.execute(stmt, {"m": MANAGER_ENTITY, "a": PAYROLL_ACCOUNT,
                                         "ps": sorted({r["period"] for r in live})}).fetchall():
                posted.add(((d or "").strip(), round(float(a or 0), 2)))
    for r in rows:
        r["report_ids"] = json.loads(r["report_ids"] or "[]")
        r["fx"] = json.loads(r.pop("fx_json") or "{}")
        r["status"] = ("voided" if r["voided_at"] else
                       "posted" if (r["credit_description"], round(-float(r["total"] or 0), 2))
                       in posted else "generated, not yet in the GL")
    return rows


def batch_csv(engine, batch_id: str) -> Optional[str]:
    ensure_tables(engine)
    with engine.connect() as c:
        r = c.execute(text("SELECT csv FROM er_batches WHERE batch_id = :b"),
                      {"b": batch_id}).first()
    return r[0] if r else None


def void_batch(engine, batch_id: str, by: str) -> dict:
    """A batch that will not be uploaded. Its reports return to approved, coded as before."""
    ensure_tables(engine)
    with engine.begin() as c:
        got = c.execute(text("UPDATE er_batches SET voided_by = :by, voided_at = :at "
                             "WHERE batch_id = :b AND voided_at IS NULL"),
                        {"by": by, "at": ex._now(), "b": batch_id})
        if got.rowcount != 1:
            raise LookupError("No open batch %s." % batch_id)
        ids = [x[0] for x in c.execute(text("SELECT id FROM er_reports WHERE batch_id = :b"),
                                       {"b": batch_id}).fetchall()]
        c.execute(text("UPDATE er_reports SET batch_id = NULL, status = 'approved', "
                       "updated_at = :at WHERE batch_id = :b"), {"at": ex._now(), "b": batch_id})
        for i in ids:
            ex._event(c, i, "unbatched", {"username": by}, basis="%s voided" % batch_id)
        # Its cell phone months are owed again, and the next batch pays them.
        from flask_app.services import expense_phone
        expense_phone.ensure_tables(engine)
        released = expense_phone.release(c, batch_id)
    return {"voided": batch_id, "reports": ids, "phone_months_released": released}
