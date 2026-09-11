#!/usr/bin/env python
"""Guardrail: an admin may cast a committee vote for a seat they do not hold, and the
database says so.

The point of this feature is NOT that an admin can approve. It is that when they do, the
record distinguishes "three role-holders agreed" from "one person voted three seats".
An override that is not labelled is worse than no override, because later it reads as
routine. These checks defend the labelling as much as the permission.

  * An override must be ASKED FOR — being admin is not enough, so it is never a side
    effect of who you are.
  * A non-admin cannot use it at all.
  * A NOTE IS MANDATORY. An exception with no stated reason is the failure mode.
  * Only committee roles can be named.
  * A role the user actually HOLDS is cast normally, never downgraded to an override
    even when it is also named.
  * The approval row carries `cast_on_behalf` / `cast_by_admin`, the note carries a
    human-readable marker, and the response reports which seats were voted by whom.

Run:  python scripts/valuation_admin_override_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services import valuation_service as vs  # noqa: E402

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def fresh(status="signed_off"):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = create_engine(f"sqlite:///{path}")
    vs.ensure_valuation_tables(eng)
    with eng.begin() as c:
        c.execute(text("INSERT INTO valuation_cycles (year, as_of_date, status) "
                       "VALUES (2025, '2025-12-31', 'open')"))
        cyc = c.execute(text("SELECT id FROM valuation_cycles WHERE year=2025")).fetchone()[0]
        c.execute(text("INSERT INTO valuation_records (cycle_id, vcode, classification, "
                       "status) VALUES (:c, 'P0000107', 'cost', :s)"),
                  {"c": cyc, "s": status})
        rid = c.execute(text("SELECT id FROM valuation_records")).fetchone()[0]
    return eng, rid, path


def rows(eng, rid):
    with eng.connect() as c:
        return c.execute(text(
            "SELECT member_role, username, COALESCE(cast_on_behalf,0), "
            "COALESCE(cast_by_admin,0), note FROM valuation_approvals "
            "WHERE record_id=:r AND active=1 ORDER BY member_role"), {"r": rid}).fetchall()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"COMMITTEE_ROLES = {vs.COMMITTEE_ROLES}\n")

    print("1. The override must be asked for, and only an admin may ask")
    eng, rid, p = fresh()
    try:
        vs.committee_approve(eng, rid, [], "jim", {}, note="x", is_admin=True)
        chk("an admin with no roles and no on_behalf_of is still refused", False)
    except PermissionError:
        chk("an admin with no roles and no on_behalf_of is still refused", True)
    try:
        vs.committee_approve(eng, rid, [], "charlene", {}, note="x",
                             is_admin=False, on_behalf_of=["ceo"])
        chk("a NON-admin naming on_behalf_of is refused", False)
    except PermissionError:
        chk("a NON-admin naming on_behalf_of is refused", True)

    print("\n2. A note is mandatory for an override")
    for label, note in (("empty note", ""), ("whitespace note", "   "), ("no note", None)):
        try:
            vs.committee_approve(eng, rid, [], "jim", {}, note=note or "",
                                 is_admin=True, on_behalf_of=["ceo"])
            chk(f"an override with {label} is refused", False)
        except ValueError:
            chk(f"an override with {label} is refused", True)

    print("\n3. Only committee roles may be named")
    try:
        vs.committee_approve(eng, rid, [], "jim", {}, note="fixing an error",
                             is_admin=True, on_behalf_of=["cfo"])
        chk("naming a non-committee role is refused", False)
    except ValueError:
        chk("naming a non-committee role is refused", True)

    print("\n4. The override works, and the DATABASE records who voted what")
    res = vs.committee_approve(eng, rid, [], "jim", {},
                               note="Correcting a mis-entered 2025 valuation.",
                               is_admin=True,
                               on_behalf_of=list(vs.COMMITTEE_ROLES))
    chk("the record reaches approved", res["status"] == "approved", f"got {res}")
    r = rows(eng, rid)
    chk("one row per seat", len(r) == len(vs.COMMITTEE_ROLES), f"got {len(r)}")
    chk("every row is flagged cast_on_behalf", all(x[2] == 1 for x in r))
    chk("every row is flagged cast_by_admin", all(x[3] == 1 for x in r))
    chk("every row names the human who actually voted",
        all(x[1] == "jim" for x in r))
    chk("the note carries a readable marker too",
        all("CAST ON BEHALF OF" in (x[4] or "") for x in r), f"{[x[4] for x in r][:1]}")
    chk("the reason survives into every row",
        all("mis-entered" in (x[4] or "") for x in r))
    chk("the response reports the overridden seats",
        sorted(res["cast_on_behalf_roles"]) == sorted(vs.COMMITTEE_ROLES), f"got {res}")
    chk("the response names who cast them", res["cast_on_behalf_by"] == ["jim"])
    chk("has_admin_override is true", res["has_admin_override"] is True)

    print("\n5. A role you HOLD is cast normally, not downgraded to an override")
    eng2, rid2, p2 = fresh()
    res = vs.committee_approve(eng2, rid2, ["cio"], "jim", {},
                               note="Holding cio; covering the vacant seats.",
                               is_admin=True, on_behalf_of=["president", "ceo", "cio"])
    r2 = {x[0]: x for x in rows(eng2, rid2)}
    chk("the held role (cio) is NOT flagged as on-behalf", r2["cio"][2] == 0,
        f"got {r2['cio']}")
    chk("the unheld roles ARE flagged", r2["president"][2] == 1 and r2["ceo"][2] == 1)
    chk("the response lists only the seats actually voted for someone else",
        sorted(res["cast_on_behalf_roles"]) == ["ceo", "president"],
        f"got {res['cast_on_behalf_roles']}")

    print("\n6. A normal approval is unchanged — no flags, no marker")
    eng3, rid3, p3 = fresh()
    vs.committee_approve(eng3, rid3, ["cio"], "admin", {}, note="routine")
    r3 = rows(eng3, rid3)
    chk("a role-holder's approval is not flagged",
        len(r3) == 1 and r3[0][2] == 0 and r3[0][3] == 0, f"got {r3}")
    chk("and its note is untouched", r3[0][4] == "routine", f"got {r3[0][4]!r}")

    print("\n7. Sign-off is still required first — the override does not skip the analyst")
    eng4, rid4, p4 = fresh(status="open")
    try:
        vs.committee_approve(eng4, rid4, [], "jim", {}, note="skip the analyst?",
                             is_admin=True, on_behalf_of=list(vs.COMMITTEE_ROLES))
        chk("an unsigned record cannot be approved even by an admin override", False)
    except ValueError:
        chk("an unsigned record cannot be approved even by an admin override", True)

    for f in (p, p2, p3, p4):
        try:
            os.unlink(f)
        except Exception:
            pass

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
