#!/usr/bin/env python
"""Set (or restore) which committee roles must approve a valuation cycle.

    # see the current state, change nothing
    $env:DATABASE_URL = 'postgresql://USER:PASS@psql-waterfall-dev.postgres.database.azure.com:5432/waterfall_xirr?sslmode=require'
    .venv\\Scripts\\python.exe scripts\\set_valuation_committee_scope.py

    # 2025 requires PRESIDENT only (drops the ceo and cio requirement)
    .venv\\Scripts\\python.exe scripts\\set_valuation_committee_scope.py --year 2025 --roles president --commit

    # put 2025 back to the full committee
    .venv\\Scripts\\python.exe scripts\\set_valuation_committee_scope.py --year 2025 --restore --commit

WHAT THIS CHANGES, stated plainly because it is a financial control and not a setting.
A published valuation feeds the One Pager, the cap stack, the Dashboard KPIs and NAV, and
the committee sign-off is what makes it authoritative. Narrowing the requirement means
fewer people must agree before a figure becomes the published record.

It is scoped to one cycle, stored in the data rather than in code, reversible with
--restore, and visible afterwards: the approve response reports `required_roles` and
`committee_narrowed`, so a reader can see what a record was judged against rather than
assuming the full committee.

WORTH CHECKING FIRST. If the blockage is that nobody HOLDS president/ceo/cio, the fix is
assigning the roles (Settings -> Review Roles), not removing the requirement. This script
prints who holds what so that question is answered before the change is made.

AND ONE THING NOT TO DO: do not instead give one person all three roles. That leaves an
audit trail showing three approvals where one human acted — strictly worse than an
explicit, recorded, reversible narrowing, which is what this is.

Dry by default. --commit writes.
"""
from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services.valuation_service import COMMITTEE_ROLES  # noqa: E402


def get_engine():
    url = os.environ.get("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return create_engine(url), f"LIVE PostgreSQL ({url.split('@')[-1].split('?')[0]})"
    db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "waterfall.db")
    return create_engine(f"sqlite:///{db}"), f"LOCAL sqlite ({db})"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="cycle year to change")
    ap.add_argument("--roles", help="comma-separated subset of " + ",".join(COMMITTEE_ROLES))
    ap.add_argument("--restore", action="store_true", help="restore the full committee")
    ap.add_argument("--commit", action="store_true", help="actually write")
    a = ap.parse_args()

    eng, source = get_engine()
    print(f"source: {source}\n")

    with eng.connect() as c:
        print("CYCLES")
        try:
            rows = c.execute(text(
                "SELECT id, year, as_of_date, status, required_roles FROM valuation_cycles "
                "ORDER BY year DESC")).fetchall()
        except Exception:
            rows = c.execute(text(
                "SELECT id, year, as_of_date, status FROM valuation_cycles "
                "ORDER BY year DESC")).fetchall()
            rows = [tuple(r) + (None,) for r in rows]
        for r in rows:
            req = r[4] or f"(default: {', '.join(COMMITTEE_ROLES)})"
            flag = "  <- NARROWED" if r[4] else ""
            print(f"  cycle {r[0]}  {r[1]}  as-of {r[2]}  {r[3]:<8}  requires: {req}{flag}")

        print("\nWHO HOLDS A COMMITTEE ROLE  (if this is empty, assign roles instead)")
        try:
            held = c.execute(text(
                "SELECT u.username, r.review_role FROM review_roles r "
                "JOIN users u ON u.id = r.user_id ORDER BY r.review_role")).fetchall()
            if not held:
                print("  (nobody holds any review role)")
            for u, role in held:
                mark = "  [committee]" if role in COMMITTEE_ROLES else ""
                print(f"  {u:<22} {role}{mark}")
            missing = [r for r in COMMITTEE_ROLES
                       if r not in {x[1] for x in held}]
            if missing:
                print(f"  NOT HELD BY ANYONE: {', '.join(missing)}")
        except Exception as e:
            print(f"  (could not read review_roles: {type(e).__name__})")

    if a.year is None:
        print("\nRead-only. Pass --year with --roles or --restore to change something.")
        return 0

    if a.restore:
        new = None
        desc = f"the full committee ({', '.join(COMMITTEE_ROLES)})"
    else:
        if not a.roles:
            raise SystemExit("Pass --roles, or --restore.")
        want = [r.strip().lower() for r in a.roles.split(",") if r.strip()]
        bad = [r for r in want if r not in COMMITTEE_ROLES]
        if bad:
            raise SystemExit(f"Not committee roles: {bad}. Valid: {list(COMMITTEE_ROLES)}")
        if not want:
            raise SystemExit("Refusing to set an empty requirement.")
        new = ",".join(want)
        desc = ", ".join(want)

    print(f"\n{'WRITING' if a.commit else 'DRY RUN — would set'}: cycle year {a.year} "
          f"requires {desc}")
    if not a.commit:
        print("Add --commit to apply.")
        return 0

    with eng.begin() as c:
        n = c.execute(text("UPDATE valuation_cycles SET required_roles = :v WHERE year = :y"),
                      {"v": new, "y": a.year}).rowcount
    if not n:
        raise SystemExit(f"No cycle for year {a.year} — nothing changed.")
    print(f"Done — {n} cycle updated. Reversible with --restore.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
