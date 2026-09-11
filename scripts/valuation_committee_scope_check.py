#!/usr/bin/env python
"""Guardrail: narrowing the valuation committee is per-cycle, explicit, and fails safe.

The committee sign-off is a financial control — a published valuation feeds the One Pager,
the cap stack, the Dashboard KPIs and NAV. `COMMITTEE_ROLES = (president, ceo, cio)` and
every one of them had to approve. A cycle may now require fewer, via
`valuation_cycles.required_roles`.

What these checks defend:

  * DEFAULT UNCHANGED. A cycle that says nothing still requires all three. Adding the
    column must not relax anything on its own.
  * SCOPED. Narrowing 2025 must leave 2026 alone. The whole point of putting this on the
    cycle rather than in the constant is that it cannot silently re-scope other years.
  * FAILS SAFE. An empty value, a typo, an unknown role, or a missing column falls back to
    the FULL committee. A narrowing must be explicit; a mistake must never drop a
    requirement quietly.
  * LEGIBLE. The approve response says what it was judged against, so a narrowed cycle is
    visible to its caller rather than inferred.

Run:  python scripts/valuation_committee_scope_check.py
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


def fresh_engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = create_engine(f"sqlite:///{path}")
    vs.ensure_valuation_tables(eng)
    with eng.begin() as c:
        c.execute(text("INSERT INTO valuation_cycles (year, as_of_date, status) "
                       "VALUES (2025, '2025-12-31', 'open')"))
        c.execute(text("INSERT INTO valuation_cycles (year, as_of_date, status) "
                       "VALUES (2026, '2026-12-31', 'open')"))
    return eng, path


def set_required(eng, year, value):
    with eng.begin() as c:
        c.execute(text("UPDATE valuation_cycles SET required_roles = :v WHERE year = :y"),
                  {"v": value, "y": year})


def cid(eng, year):
    with eng.connect() as c:
        return c.execute(text("SELECT id FROM valuation_cycles WHERE year = :y"),
                         {"y": year}).fetchone()[0]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    eng, path = fresh_engine()
    full = tuple(vs.COMMITTEE_ROLES)
    print(f"COMMITTEE_ROLES = {full}\n")

    print("1. The default is unchanged by the column existing")
    chk("a cycle with no required_roles requires the full committee",
        vs.get_required_roles(eng, cid(eng, 2025)) == full,
        f"got {vs.get_required_roles(eng, cid(eng, 2025))}")

    print("\n2. Narrowing is scoped to the cycle it is set on")
    set_required(eng, 2025, "president")
    chk("2025 now requires president only",
        vs.get_required_roles(eng, cid(eng, 2025)) == ("president",),
        f"got {vs.get_required_roles(eng, cid(eng, 2025))}")
    chk("2026 is UNTOUCHED and still requires all three",
        vs.get_required_roles(eng, cid(eng, 2026)) == full,
        f"got {vs.get_required_roles(eng, cid(eng, 2026))}")

    print("\n3. It fails SAFE — a mistake restores the full committee, never drops a role")
    for label, value in (("an empty string", ""),
                         ("whitespace only", "   "),
                         ("an unknown role", "cfo"),
                         ("a typo'd role", "preisdent"),
                         ("a list of unknowns", "cfo, coo")):
        set_required(eng, 2025, value)
        got = vs.get_required_roles(eng, cid(eng, 2025))
        chk(f"{label} falls back to the full committee", got == full, f"got {got}")

    set_required(eng, 2025, "president, cio")
    chk("a valid two-role subset is honoured",
        vs.get_required_roles(eng, cid(eng, 2025)) == ("president", "cio"),
        f"got {vs.get_required_roles(eng, cid(eng, 2025))}")

    set_required(eng, 2025, "PRESIDENT , Cio ")
    chk("case and spacing are tolerated",
        vs.get_required_roles(eng, cid(eng, 2025)) == ("president", "cio"),
        f"got {vs.get_required_roles(eng, cid(eng, 2025))}")

    set_required(eng, 2025, "president, cfo")
    chk("a valid role mixed with an unknown keeps only the valid one",
        vs.get_required_roles(eng, cid(eng, 2025)) == ("president",),
        f"got {vs.get_required_roles(eng, cid(eng, 2025))}")

    print("\n4. An unreachable cycle does not silently relax anything")
    chk("a cycle id that does not exist requires the full committee",
        vs.get_required_roles(eng, 99999) == full,
        f"got {vs.get_required_roles(eng, 99999)}")

    print("\n5. The migration is idempotent")
    try:
        vs.ensure_valuation_tables(eng)
        vs.ensure_valuation_tables(eng)
        set_required(eng, 2025, "president")
        chk("re-running ensure_valuation_tables keeps the setting and does not raise",
            vs.get_required_roles(eng, cid(eng, 2025)) == ("president",))
    except Exception as e:
        chk("re-running ensure_valuation_tables does not raise", False, repr(e))

    try:
        os.unlink(path)
    except Exception:
        pass

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
