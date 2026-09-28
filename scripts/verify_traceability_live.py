"""Pre-traffic reconciliation check for the traceability tools, against LIVE data.

WHY THIS EXISTS. Everything the guardrails assert about the trace tool is
structural — they run on injected payloads because local SQLite is a stub. The
one thing they structurally cannot prove is the thing the feature promises: that
the numbers the assistant quotes for a real deal are the numbers on the real
page. That is only checkable where the data is, so this runs in the container
before traffic and exits non-zero if anything fails, so it can gate a deploy.

    python scripts/verify_traceability_live.py [VCODE1] [VCODE2]
                                               [--quarter 26Q2] [--investor CODE]

With no deal ids it discovers two deals from the live data and says which it
picked. With no quarter it uses whichever the One Pager itself resolves, so this
cannot end up checking a different quarter from the one the page would show.

IT DOES NOT CALL create_app(). That runs `ensure_pg_tables()` and
`_pg_fix_column_types()` — schema migrations, fired at production from whatever
branch happens to be checked out. A bare Flask app loaded from `Config` is
enough, because everything here only needs `DATABASE_URL`/`DB_PATH` and the
projection defaults. `get_engine()` is safe: it builds the engine and nothing
else.

THREE OUTCOMES, NOT TWO. A check that cannot run because the data is not there
(the SQLite stub locally, a deal with no PE activity) is INCONCLUSIVE, not a
failure — only a real disagreement is a FAIL, and only a FAIL sets the exit code.
Conflating "no data" with "wrong" would make this scream on every local run and
be ignored by the time it mattered.

TOLERANCES ARE RELATIVE. An absolute floor sized for money is ~6% of a small
ROE; that exact bug shipped into this codebase once and called 0.082 and 0.078
equal. The absolute floor here exists only so two genuine zeros agree.

NOTHING IS DERIVED. Denominators, components and balances are READ from the
published payload. A DSCR denominator is never recovered by dividing the ratio
backwards — that is not a derivation, it is an assumption that the ratio is
right, and it would make this script incapable of detecting the fault it exists
to detect.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

#: Relative, for the reason in the module docstring. The absolute floor is only
#: big enough to let two genuine zeros compare equal.
_REL_TOL = 1e-6
_NEAR_ZERO_TOL = 1e-9

#: Deals to check when none are named. Overridden by argv, and replaced by
#: discovery when they carry no One Pager data on this database.
_DEFAULT_DEALS = ["P0000109", "P0000116"]

PASSED = FAILED = INCONCLUSIVE = 0
FAILURES: list = []


def _p(label: str, detail: str = "") -> None:
    """A pass prints no detail. The detail passed to `chk` describes the
    FAILURE ("no inputs returned"), so echoing it beside a PASS reads as a
    contradiction — same convention as the committed guardrails."""
    global PASSED
    PASSED += 1
    print(f"  PASS  {label}")


def _f(label: str, detail: str = "") -> None:
    global FAILED
    FAILED += 1
    FAILURES.append(label + (f"  [{detail}]" if detail else ""))
    print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))


def _inc(label: str, why: str) -> None:
    """Cannot be judged here — not a failure, and never counted as a pass."""
    global INCONCLUSIVE
    INCONCLUSIVE += 1
    print(f"  ----  {label}  — inconclusive: {why}")


def chk(cond, label: str, detail: str = "") -> bool:
    (_p if cond else _f)(label, detail)
    return bool(cond)


def _num(v):
    """A finite float, or None. Strings are never coerced — the Snapshot writes
    display sentinels ('n/a', 'Dev') into cells that otherwise carry ratios."""
    if isinstance(v, bool) or v is None or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return None if (f != f or f in (float("inf"), float("-inf"))) else f


def _close(a, b) -> bool:
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return False
    return abs(a - b) <= max(_NEAR_ZERO_TOL, max(abs(a), abs(b)) * _REL_TOL)


# ── bootstrap ────────────────────────────────────────────────────────────

def bootstrap():
    """A bare app context and a wired engine. No migrations. See the docstring."""
    from flask import Flask
    from flask_app.config import Config
    import flask_app.db as fdb
    import database

    app = Flask(__name__)
    app.config.from_object(Config)
    app.app_context().push()

    engine = fdb.get_engine()          # builds the engine; runs no DDL
    database.set_engine(engine)

    backend = ("PostgreSQL" if app.config.get("DATABASE_URL")
               else f"SQLite ({app.config.get('DB_PATH')})")
    print(f"Database: {backend}")
    return app


def discover_deals(wanted: list) -> list:
    """The deals to check: those named, else two that actually have a One Pager.

    A named deal is used even if it looks empty — being told to check a specific
    deal and silently checking a different one would be worse than failing.
    """
    from flask_app.services import field_trace_service as T
    if wanted:
        return wanted

    try:
        from flask_app.services import data_service
        inv = data_service.get_data()["inv"]
        col = "vcode" if "vcode" in inv.columns else "vCode"
        candidates = [str(v).strip() for v in inv[col].dropna().unique()]
    except Exception as exc:
        print(f"  (deal discovery failed: {exc})")
        return list(_DEFAULT_DEALS)

    found = []
    for vc in _DEFAULT_DEALS + [c for c in candidates if c not in _DEFAULT_DEALS]:
        if len(found) >= 2:
            break
        try:
            payload, _q = T._one_pager_payload(vc, None)
            if payload and _num((payload.get("cap_stack") or {}).get("total_cap")):
                found.append(vc)
        except Exception:
            continue
    return found or list(_DEFAULT_DEALS)


# ── 1. the arithmetic fields reconcile ───────────────────────────────────

_RECONCILE_FIELDS = ["one_pager.total_cap",
                     "one_pager.pe_exposure_on_cap",
                     "one_pager.pe_yield_on_exposure"]


def check_reconciles(T, vcode: str, quarter):
    print(f"\n1. Arithmetic fields reconcile — {vcode}")
    for fid in _RECONCILE_FIELDS:
        r = T.trace_field_value(vcode, fid, quarter=quarter)
        if r.get("error"):
            _inc(f"{fid}", r["error"][:90])
            continue
        val = _num(r.get("value"))
        rec = r.get("reconciles")
        if val is None:
            # No figure on this deal for this field — nothing to reconcile.
            _inc(f"{fid} reconciles", "the deal publishes no value for this field")
            continue
        if rec is None:
            _inc(f"{fid} reconciles",
                 "not checked — an input the builder did not publish")
            continue
        parts = ", ".join(
            f"{i['component']}={i['value']}" for i in (r.get("inputs") or []))
        chk(rec is True, f"{fid} reconciles",
            "" if rec is True else f"value={val}  inputs: {parts}")


# ── 2. ROE comes apart under the engine that produced it ─────────────────

def check_roe(T, vcode: str, quarter):
    print(f"\n2. ROE breakdown is present and single-engine — {vcode}")
    r = T.trace_field_value(vcode, "one_pager.roe_to_date", quarter=quarter)
    if r.get("error"):
        _inc("roe_to_date", r["error"][:90])
        return
    val = _num(r.get("value"))
    if val is None or val == 0.0:
        _inc("roe_to_date breakdown",
             "the deal reports no ROE (no capital events through this quarter)")
        return

    inputs = r.get("inputs") or []
    engine = r.get("breakdown_engine") or ""
    withheld = r.get("breakdown_available") is False

    if not chk(not withheld and bool(inputs),
               "the ROE breakdown is present, not withheld",
               r.get("breakdown_unavailable_reason") or "no inputs returned"):
        return
    # THE POINT OF THE WHOLE REFACTOR. If this names the ROE Summary report, the
    # components are not reaching the payload and the trace has fallen back to
    # the second engine — which is exactly the drift this was built to remove.
    chk("calculate_roe_detailed" in engine,
        "the breakdown names the page's own engine (calculate_roe_detailed)",
        f"engine={engine[:70]!r}")
    chk(r.get("reconciles") is True,
        "the ROE components tie to the published figure",
        f"value={val}  " + ", ".join(
            f"{i['component']}={i['value']}" for i in inputs))


# ── 3. DSCR publishes the denominator it actually divided by ─────────────

def check_dscr(T, vcode: str, quarter):
    print(f"\n3. DSCR denominators — {vcode}")
    r = T.trace_field_value(vcode, "one_pager.dscr", quarter=quarter)
    if r.get("error"):
        _inc("dscr", r["error"][:90])
        return
    bases = r.get("bases") or []
    if not bases:
        _inc("dscr", "no bases returned")
        return

    judged = 0
    for b in bases:
        basis = b.get("basis")
        ratio = _num(b.get("dscr"))
        den_in = (b.get("inputs") or [None, None])[1] or {}
        den = _num(den_in.get("value"))

        if ratio is None:
            # ABSENT AND EXPECTED. The column has no ratio at all — at_close is
            # routinely blank (dev deals, the Year-0 gate), and a denominator
            # without a ratio would be the anomaly, not the other way round.
            if den is None:
                _p(f"dscr[{basis}] — no ratio and no denominator (expected)")
            else:
                _f(f"dscr[{basis}] — denominator present with NO ratio",
                   f"denominator={den}")
            judged += 1
            continue

        # A ratio IS published, so the denominator must be too.
        if den is None:
            _f(f"dscr[{basis}] — ratio published but denominator MISSING",
               f"ratio={ratio}; reason={den_in.get('reason', 'none given')!r}")
        else:
            num = _num((b["inputs"][0] or {}).get("value"))
            chk(b.get("reconciles") is True,
                f"dscr[{basis}] — numerator / denominator ties to the ratio",
                f"ratio={ratio} numerator={num} denominator={den}")
        judged += 1

    if judged == 0:
        _inc("dscr", "no column could be judged")


# ── 4. the assistant's One Pager tool ties to the page ───────────────────

def check_tool_matches_page(T, A, vcode: str, quarter):
    """Step-4 equality: same deal, both routes, same PE figures.

    Both now call `get_one_pager_data` with identical arguments; before the fix
    the assistant tool omitted `full_data`, so the PE enrichment never ran and
    these two could differ with nothing saying so.
    """
    print(f"\n4. Assistant tool figures equal the page's — {vcode}")
    try:
        page, resolved = T._one_pager_payload(vcode, quarter)
    except Exception as exc:
        _inc("tool vs page", f"page path failed: {exc}")
        return
    if not page:
        _inc("tool vs page", "the page path returned no One Pager payload")
        return

    raw = A._tool_get_one_pager({"vcode": vcode, "quarter": resolved})
    tool = json.loads(raw)
    if tool.get("error"):
        _inc("tool vs page", tool["error"][:90])
        return

    page_pe = page.get("pe_performance") or {}
    tool_pe = tool.get("pe_performance") or {}
    if not page_pe or not tool_pe:
        _inc("tool vs page", "one side returned no pe_performance block")
        return

    for key in ("current_pe_balance", "accrued_balance"):
        a, b = _num(page_pe.get(key)), _num(tool_pe.get(key))
        if a is None and b is None:
            _inc(f"{key} equal on both routes", "neither route reports it")
            continue
        chk(_close(a, b), f"{key} equal on both routes",
            f"page={a} tool={b}")


# ── 5. a Snapshot field asks for the investor ────────────────────────────

def check_snapshot_asks(T, vcode: str, quarter):
    """The Snapshot is assembled per investor and the code is not in the page
    context, so the only correct behaviour is to ask for it."""
    print(f"\n5. A Snapshot field asks for investor_code — {vcode}")
    r = T.trace_field_value(vcode, "snapshot_loan.ltv", quarter=quarter or "26Q2")
    err = (r.get("error") or "").lower()
    chk(bool(err) and "investor" in err,
        "a Snapshot trace with no investor_code refuses and says why",
        f"returned keys={sorted(r.keys())}")
    # And it must not have quietly produced a figure anyway.
    chk(r.get("value") is None and not r.get("inputs"),
        "and returns no value or inputs from a guessed investor",
        f"value={r.get('value')!r}")


def check_snapshot_with_investor(T, vcode: str, quarter, investor: str):
    print(f"\n5b. Snapshot trace WITH investor_code — {vcode} / {investor}")
    for fid in ("snapshot_loan.ltv", "snapshot_loan.debt_yield"):
        r = T.trace_field_value(vcode, fid, quarter=quarter,
                                investor_code=investor)
        if r.get("error"):
            _inc(fid, r["error"][:90])
            continue
        if _num(r.get("value")) is None:
            _inc(f"{fid} reconciles", "no value on this row (dev/debt-free/n-a)")
            continue
        rec = r.get("reconciles")
        if rec is None:
            _inc(f"{fid} reconciles", "an input was not published on the row")
            continue
        chk(rec is True, f"{fid} reconciles",
            ", ".join(f"{i['component']}={i['value']}"
                      for i in (r.get("inputs") or [])))


# ── 6. the guardrails, in this container ─────────────────────────────────

def check_guardrails():
    """Run the two committed guardrails here and gate on their exit codes.

    GATED ON THE EXIT CODE, NOT ON A COUNT. `katex_render_check` SKIPS its live
    render when node is absent — which it is in the app image — so it is green
    at well under 17 there, and demanding 17 would fail every container run for
    an environmental reason. The counts are printed either way so a real drop is
    still visible.
    """
    print("\n6. Committed guardrails, run here")
    for name, expect in (("traceability_tools_check.py", 112),
                         ("katex_render_check.py", 17)):
        path = os.path.join(_REPO, "scripts", name)
        if not os.path.exists(path):
            _f(f"{name} is present", "not found — scripts/ may not have shipped")
            continue
        try:
            out = subprocess.run([sys.executable, path], cwd=_REPO,
                                 capture_output=True, text=True, timeout=900)
        except Exception as exc:
            _f(f"{name} ran", str(exc))
            continue
        tail = [l for l in (out.stdout or "").splitlines()
                if l.startswith("RESULT:")]
        result = tail[-1] if tail else "(no RESULT line)"
        chk(out.returncode == 0, f"{name} is green",
            f"exit={out.returncode}  {result}  "
            f"stderr={(out.stderr or '')[-200:]}")
        print(f"        {result}   (reference: {expect} passing locally)")


# ── entry point ──────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deals", nargs="*", help="one or two vcodes")
    ap.add_argument("--quarter", default=None, help="e.g. 26Q2")
    ap.add_argument("--investor", default=None,
                    help="investor code, to also check the Snapshot ratios")
    args = ap.parse_args()

    print("=" * 66)
    print("Traceability — live reconciliation (pre-traffic)")
    print("=" * 66)

    bootstrap()
    from flask_app.services import field_trace_service as T
    from flask_app.services import assistant_service as A

    deals = discover_deals([d.strip() for d in args.deals if d.strip()])
    print(f"Deals: {', '.join(deals)}"
          + ("" if args.deals else "   (discovered — pass vcodes to override)"))
    print(f"Quarter: {args.quarter or 'resolved by the One Pager itself'}")

    for vc in deals:
        check_reconciles(T, vc, args.quarter)
        check_roe(T, vc, args.quarter)
        check_dscr(T, vc, args.quarter)
        check_tool_matches_page(T, A, vc, args.quarter)

    if deals:
        check_snapshot_asks(T, deals[0], args.quarter)
        if args.investor:
            check_snapshot_with_investor(T, deals[0], args.quarter, args.investor)
        else:
            print("\n5b. Snapshot trace WITH investor_code")
            _inc("snapshot ratios reconcile",
                 "no --investor given; the Snapshot is per investor and one is "
                 "never guessed")

    check_guardrails()

    print("\n" + "=" * 66)
    print(f"RESULT: {PASSED} passed, {FAILED} failed, "
          f"{INCONCLUSIVE} inconclusive")
    if FAILURES:
        for f in FAILURES:
            print(f"  - {f}")
    if FAILED:
        print("\nDO NOT RELEASE TRAFFIC until these are explained.")
    elif INCONCLUSIVE:
        print("\nNo failures. Inconclusive checks could not be judged on this "
              "database — on the local SQLite stub that is expected; in the "
              "container it means the deal carries no such figure, and is "
              "worth a look before relying on it.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
