"""Guardrail: capital-call CRUD works on BOTH SQLite and PostgreSQL.

The four `/api/deals/<vcode>/raw-capital-calls` endpoints used to talk to the
database through `database.get_db_connection()` with sqlite3-shaped calls —
raw SQL strings, `?` placeholders, tuple params and `rowid`.  Whenever a
SQLAlchemy engine is wired in (i.e. always on Azure/PostgreSQL, see
`flask_app/__init__`), `Connection.execute("...", (tuple,))` raises before it
ever reaches the database, so:

  * POST returned 500 and the Vue form silently stayed open — "nothing
    happened when I hit save";
  * GET swallowed the exception and returned `{"capital_calls": []}`, so the
    editable list always looked empty;
  * PUT/DELETE additionally keyed on `rowid`, which PostgreSQL does not have.

This check exercises the real view functions through a Flask test request
against a real database.  Run with no arguments for SQLite; pass a PostgreSQL
URL to run the same battery there:

    python scripts/capital_call_crud_check.py
    python scripts/capital_call_crud_check.py postgresql+psycopg2://...
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, inspect, text  # noqa: E402

VCODE = "P0000004"          # Asbury Commons
PROPCODES = ("PPI22", "OPFLAG")
CALL_DATE = "2026-09-30"
AMOUNT = 130000.0

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def seed_csv_shaped_table(engine):
    """Recreate capital_calls exactly the way the CSV import does.

    `import_csv_dataframe` -> `_import_dataframe` uses
    ``to_sql(if_exists="replace")``, which drops the table and rebuilds it from
    the CSV's own headers.  That preserves mixed case (quoted on PostgreSQL),
    leaves the table without any key, and omits `Typename` — the SQLite-only
    `run_migrations` never adds it on the PostgreSQL path.
    """
    df = pd.DataFrame(
        [
            {
                "Vcode": "P0000088", "PropCode": "PPIBEL", "CallDate": "6/30/2026",
                "Amount": 4812512.0, "CallType": "Scheduled",
                "FundingSource": "Preferred", "Notes": "existing feed row",
            }
        ]
    )
    df.to_sql("capital_calls", engine, if_exists="replace", index=False)


def auth_headers(app):
    """A real JWT, so the genuine @login_required decorator is exercised."""
    import jwt
    token = jwt.encode(
        {"sub": "1", "username": "check", "role": "admin"},
        app.config["JWT_SECRET"], algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def build_app(database_url, sqlite_path):
    from flask import Flask
    import flask_app.db as fdb
    import database as legacy

    fdb.reset_engine()
    app = Flask(__name__)
    app.config["DB_PATH"] = sqlite_path
    app.config["DATABASE_URL"] = database_url
    app.config["TESTING"] = True
    app.config["JWT_SECRET"] = "capcall-check-secret"

    # Stub out the two side effects the endpoints trigger after a write; they
    # reload the whole portfolio and are not what is under test here.
    import flask_app.api.deals as deals_api
    calls = {"reload": 0, "clear": 0}
    deals_api.data_service.reload = lambda *a, **k: calls.__setitem__("reload", calls["reload"] + 1)
    deals_api.compute_service.clear_cache = lambda *a, **k: calls.__setitem__("clear", calls["clear"] + 1)


    with app.app_context():
        legacy.set_engine(fdb.get_engine())
    return app, deals_api, calls


def run(database_url=None):
    label = "PostgreSQL" if database_url else "SQLite"
    print(f"\n=== {label} ===")

    tmpdir = tempfile.mkdtemp()
    sqlite_path = os.path.join(tmpdir, "capcall_check.db")
    if not database_url:
        create_engine(f"sqlite:///{sqlite_path}").connect().close()

    app, deals_api, side_effects = build_app(database_url, sqlite_path)

    with app.app_context():
        from flask_app.db import get_engine, is_postgres
        engine = get_engine()
        hdrs = auth_headers(app)
        seed_csv_shaped_table(engine)

        cols_before = {c["name"] for c in inspect(engine).get_columns("capital_calls")}
        check("table starts CSV-shaped: no Typename, no id",
              "Typename" not in cols_before and "id" not in cols_before,
              f"columns={sorted(cols_before)}")

        # --- POST both rows ---------------------------------------------
        created_ids = []
        for pc in PROPCODES:
            with app.test_request_context(
                f"/api/deals/{VCODE}/raw-capital-calls",
                method="POST",
                headers=hdrs,
                json={"PropCode": pc, "CallDate": CALL_DATE, "Amount": AMOUNT,
                      "Notes": "guardrail", "Typename": "Contribution: Investments"},
            ):
                resp, status = deals_api.create_capital_call(VCODE)
            check(f"POST {pc} returns 201", status == 201, str(resp.get_json()))

        cols_after = {c["name"] for c in inspect(engine).get_columns("capital_calls")}
        check("Typename healed onto the CSV-shaped table", "Typename" in cols_after)
        if is_postgres():
            check("surrogate id added (PostgreSQL has no rowid)", "id" in cols_after,
                  f"columns={sorted(cols_after)}")

        # --- GET ---------------------------------------------------------
        with app.test_request_context(f"/api/deals/{VCODE}/raw-capital-calls", headers=hdrs):
            resp = deals_api.raw_capital_calls(VCODE)
        payload = resp.get_json() if not isinstance(resp, tuple) else resp[0].get_json()
        rows = payload.get("capital_calls", [])
        check("GET returns both new rows", len(rows) == 2, f"got {len(rows)}")
        check("GET does not report an error", "error" not in payload,
              str(payload.get("error", "")))
        if len(rows) == 2:
            check("GET row keys are canonical-cased",
                  all(k in rows[0] for k in ("id", "PropCode", "CallDate", "Amount", "Typename")),
                  f"keys={sorted(rows[0].keys())}")
            check("GET amounts round-trip",
                  all(float(r["Amount"]) == AMOUNT for r in rows),
                  str([r["Amount"] for r in rows]))
            check("GET propcodes round-trip",
                  {r["PropCode"] for r in rows} == set(PROPCODES),
                  str([r["PropCode"] for r in rows]))
            check("GET ids are usable (non-null)",
                  all(r["id"] is not None for r in rows),
                  str([r["id"] for r in rows]))
            created_ids = [r["id"] for r in rows]

        # the pre-existing feed row for another deal must be untouched
        with engine.connect() as conn:
            other = conn.execute(text(
                'SELECT COUNT(*) FROM capital_calls WHERE "Vcode" = :v'), {"v": "P0000088"}
            ).scalar()
        check("other deals' rows untouched", other == 1, f"count={other}")

        # --- PUT ---------------------------------------------------------
        if created_ids:
            target = created_ids[0]
            with app.test_request_context(
                f"/api/deals/{VCODE}/raw-capital-calls/{target}",
                method="PUT",
                headers=hdrs,
                json={"PropCode": PROPCODES[0], "CallDate": CALL_DATE, "Amount": 99.0,
                      "Notes": "edited", "Typename": "Contribution: Investments"},
            ):
                out = deals_api.update_capital_call(VCODE, target)
            resp, status = out if isinstance(out, tuple) else (out, 200)
            check("PUT returns 200", status == 200, str(resp.get_json()))
            with engine.connect() as conn:
                amt = conn.execute(text(
                    'SELECT "Amount" FROM capital_calls WHERE "Notes" = :n'), {"n": "edited"}
                ).scalar()
            check("PUT actually wrote the new amount", amt == 99.0, f"amount={amt}")

            # a foreign id must not be editable through another deal's route
            with app.test_request_context(
                f"/api/deals/P0000088/raw-capital-calls/{target}",
                method="PUT", headers=hdrs, json={"Amount": 1.0},
            ):
                out = deals_api.update_capital_call("P0000088", target)
            _, status = out if isinstance(out, tuple) else (out, 200)
            check("PUT is scoped to the deal in the URL", status == 404, f"status={status}")

        # --- DELETE ------------------------------------------------------
        if created_ids:
            for cid in created_ids:
                with app.test_request_context(
                    f"/api/deals/{VCODE}/raw-capital-calls/{cid}", method="DELETE", headers=hdrs
                ):
                    out = deals_api.delete_capital_call(VCODE, cid)
                _, status = out if isinstance(out, tuple) else (out, 200)
                check(f"DELETE id={cid} returns 200", status == 200)
            with engine.connect() as conn:
                left = conn.execute(text(
                    'SELECT COUNT(*) FROM capital_calls WHERE "Vcode" = :v'), {"v": VCODE}
                ).scalar()
            check("DELETE removed both rows", left == 0, f"remaining={left}")

            with app.test_request_context(
                f"/api/deals/{VCODE}/raw-capital-calls/{created_ids[0]}", method="DELETE", headers=hdrs
            ):
                out = deals_api.delete_capital_call(VCODE, created_ids[0])
            _, status = out if isinstance(out, tuple) else (out, 200)
            check("DELETE of a missing row reports 404", status == 404, f"status={status}")

        check("writes invalidated the caches", side_effects["reload"] > 0 and side_effects["clear"] > 0,
              str(side_effects))

        if database_url:
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS capital_calls"))


def run_pg_branch():
    """Exercise the PostgreSQL code path without a PostgreSQL server.

    `is_postgres()` is forced true, so the endpoints emit exactly the SQL they
    send to Azure: double-quoted mixed-case identifiers, named binds, and the
    surrogate `"id"` key instead of `rowid`.  The table is pre-created with a
    real `id` column so that SQL genuinely executes end to end here; the
    `SERIAL` DDL that only PostgreSQL accepts is asserted as text.
    """
    import tempfile as _tf
    from sqlalchemy import event

    print("\n=== PostgreSQL branch (is_postgres forced) ===")
    tmpdir = _tf.mkdtemp()
    sqlite_path = os.path.join(tmpdir, "capcall_pgshape.db")
    create_engine(f"sqlite:///{sqlite_path}").connect().close()

    app, deals_api, _ = build_app(None, sqlite_path)
    deals_api.is_postgres = lambda: True

    statements = []
    with app.app_context():
        from flask_app.db import get_engine
        engine = get_engine()
        hdrs = auth_headers(app)

        with engine.begin() as conn:
            conn.execute(text(
                'CREATE TABLE capital_calls ('
                '"id" INTEGER PRIMARY KEY AUTOINCREMENT, '
                '"Vcode" TEXT, "PropCode" TEXT, "CallDate" TEXT, "Amount" REAL, '
                '"CallType" TEXT, "FundingSource" TEXT, "Notes" TEXT)'
            ))

        # `before_execute` fires BEFORE the dialect rewrites binds into the
        # DBAPI's paramstyle, so this is the SQL as the endpoint wrote it —
        # named binds, not sqlite3's qmarks.
        @event.listens_for(engine, "before_execute")
        def _collect(conn, clauseelement, multiparams, params, execution_options):
            statements.append(str(clauseelement))

        for pc in PROPCODES:
            with app.test_request_context(
                f"/api/deals/{VCODE}/raw-capital-calls", method="POST", headers=hdrs,
                json={"PropCode": pc, "CallDate": CALL_DATE, "Amount": AMOUNT,
                      "Notes": "pgshape", "Typename": "Contribution: Investments"},
            ):
                resp, status = deals_api.create_capital_call(VCODE)
            check(f"PG-shape POST {pc} succeeds", status == 201, str(resp.get_json()))

        with app.test_request_context(f"/api/deals/{VCODE}/raw-capital-calls", headers=hdrs):
            resp = deals_api.raw_capital_calls(VCODE)
        payload = resp.get_json() if not isinstance(resp, tuple) else resp[0].get_json()
        rows = payload.get("capital_calls", [])
        check("PG-shape GET returns both rows", len(rows) == 2, str(payload.get("error", "")))

        ids = [r["id"] for r in rows]
        if ids:
            with app.test_request_context(
                f"/api/deals/{VCODE}/raw-capital-calls/{ids[0]}", method="PUT", headers=hdrs,
                json={"PropCode": PROPCODES[0], "CallDate": CALL_DATE, "Amount": 42.0,
                      "Notes": "pgshape-edit", "Typename": "Contribution: Investments"},
            ):
                out = deals_api.update_capital_call(VCODE, ids[0])
            resp, status = out if isinstance(out, tuple) else (out, 200)
            check("PG-shape PUT succeeds", status == 200, str(resp.get_json()))

            with app.test_request_context(
                f"/api/deals/{VCODE}/raw-capital-calls/{ids[1]}", method="DELETE", headers=hdrs
            ):
                out = deals_api.delete_capital_call(VCODE, ids[1])
            _, status = out if isinstance(out, tuple) else (out, 200)
            check("PG-shape DELETE succeeds", status == 200)

    crud = [q for q in statements if q.strip().split()[0].upper()
            in ("SELECT", "INSERT", "UPDATE", "DELETE") and "capital_calls" in q]
    check("PG branch emitted CRUD statements", len(crud) >= 4, f"{len(crud)} statements")

    check("no rowid anywhere in the PG branch",
          not any("rowid" in q.lower() for q in crud))
    check('row key is the quoted surrogate "id"',
          all('"id"' in q for q in crud if q.strip().upper().startswith(("UPDATE", "DELETE"))))
    check("named binds, never the sqlite3-shaped ? placeholders",
          not any("?" in q for q in crud) and any(":" in q for q in crud),
          next((q[:80] for q in crud if "?" in q), ""))

    import re
    unquoted = []
    for q in crud:
        # every mixed-case column must be quoted, or PostgreSQL folds it to
        # lowercase and errors against the to_sql-created table
        for col in ("Vcode", "PropCode", "CallDate", "Amount", "CallType",
                    "FundingSource", "Notes", "Typename"):
            for m in re.finditer(col, q):
                before = q[m.start() - 1] if m.start() else " "
                after = q[m.end()] if m.end() < len(q) else " "
                if before == ":":
                    continue  # a bind name that happens to match the column
                if before != '"' or after != '"':
                    unquoted.append((col, q[:90]))
    check("every mixed-case identifier is double-quoted", not unquoted, str(unquoted[:2]))

    ddl = [q for q in statements if "ADD COLUMN" in q.upper()]
    check("Typename is added with a quoted identifier",
          any('"Typename"' in q for q in ddl), str(ddl))

    # the SERIAL DDL is PostgreSQL-only, so assert the text the branch builds
    src = open("flask_app/api/deals.py", encoding="utf-8").read()
    check("surrogate key DDL is quoted id SERIAL", '"id" SERIAL' in src)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else None
    run(None)
    run_pg_branch()
    if url:
        run(url)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
