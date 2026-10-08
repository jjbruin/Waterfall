"""Guardrail -- retire the Giant 7 stand-in the moment Giant 7 is sold.

PROJECTED_YE_NOI_FALLBACK (portfolio_snapshot_loan) is a vcode list, kept on
purpose (Charlene, Oct 8 2026): Giant 7 (P0000019) is the only deal still owned,
about to be sold, and without recent financials, so its Loan ratios come from
Projected YE NOI. Once it is SOLD it must be handled like every other sold deal:
the entry deleted, and P0000019 added to KEEP_DESPITE_SOLD (portfolio_snapshot_
service) so it stays on the page with Debt n/a and dashed ratios, like East
Manchester. Without that it silently DISAPPEARS from the Snapshot, because a
sold deal off the kept list is dropped.

FAILS when, in the deals data:
  1. a vcode still in PROJECTED_YE_NOI_FALLBACK is Sale_Status SOLD
     -> delete it from the list;
  2. Giant 7 is SOLD but not in KEEP_DESPITE_SOLD -> add it;
  3. Giant 7 is SOLD with a Sale_Date before 2026-07-01. It was still held,
     under PSA, in the sent 26Q2 report; MRI carried a stale 3/31/2026, and
     marking it SOLD on that date would pull it out of 26Q2 retroactively.
     Reject what cannot be true.

Data: WF_TOKEN set -> live /api/data/deals/all; else DB_PATH or ./waterfall.db.
No data -> SKIP (printed; skip is not pass). The rule itself is exercised on
synthetic rows every run, in both directions, so a broken rule cannot pass.

The lists are read from the source with `ast`, not imported, so the pre-commit
hook stays fast.

    .venv/Scripts/python.exe scripts/projected_ye_fallback_retirement_check.py
"""
from __future__ import annotations

import ast
import os
import sqlite3
import sys
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIANT7 = "P0000019"
EARLIEST_SALE = date(2026, 7, 1)


def _literal_set(path: str, name: str) -> set:
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        target = (node.target if isinstance(node, ast.AnnAssign)
                  else node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == name:
            v = node.value
            if isinstance(v, ast.Call):            # frozenset({...})
                v = v.args[0] if v.args else ast.Set(elts=[])
            return {str(e.value).strip().upper() for e in getattr(v, "elts", [])
                    if isinstance(e, ast.Constant)}
    raise SystemExit(f"FAIL {name} not found in {path} -- the check cannot run")


def _parse_date(v):
    if not v:
        return None
    s = str(v).strip().split(" ")[0].split("T")[0]
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def evaluate(fallback: set, kept: set, deals: dict) -> list:
    """deals: vcode -> (sale_status, sale_date). Returns failure messages."""
    out = []
    for vc in sorted(fallback):
        status, _ = deals.get(vc, ("", None))
        if str(status or "").upper() == "SOLD":
            out.append(f"{vc} is SOLD but still in PROJECTED_YE_NOI_FALLBACK -- "
                       "delete it from the list (portfolio_snapshot_loan.py)")
    status, sale_date = deals.get(GIANT7, ("", None))
    if str(status or "").upper() == "SOLD":
        if GIANT7 not in kept:
            out.append(f"{GIANT7} Giant 7 is SOLD but not in KEEP_DESPITE_SOLD -- "
                       "it will vanish from the Snapshot; add it "
                       "(portfolio_snapshot_service.py)")
        d = _parse_date(sale_date)
        if d is None or d < EARLIEST_SALE:
            out.append(f"{GIANT7} Giant 7 is SOLD with Sale_Date {sale_date!r} -- "
                       f"before {EARLIEST_SALE}, when it was still held under "
                       "PSA in the sent 26Q2 report; correct the date first")
    return out


def _selftest() -> list:
    f, k, none = {GIANT7}, set(), {}
    cases = [
        ("held, listed -> pass", f, k, {GIANT7: ("", "3/31/2026")}, False),
        ("sold, still listed -> fail", f, k, {GIANT7: ("SOLD", "11/15/2026")}, True),
        ("sold, retired, kept -> pass", set(), {GIANT7}, {GIANT7: ("SOLD", "11/15/2026")}, False),
        ("sold, retired, NOT kept -> fail", set(), set(), {GIANT7: ("SOLD", "11/15/2026")}, True),
        ("sold on the stale 3/31 date -> fail", set(), {GIANT7}, {GIANT7: ("SOLD", "3/31/2026")}, True),
        ("no deal data -> pass", f, k, none, False),
    ]
    bad = []
    for label, fb, kp, dl, should_fail in cases:
        if bool(evaluate(fb, kp, dl)) != should_fail:
            bad.append(label)
    return bad


def _load_deals(vcodes: set):
    tok = os.environ.get("WF_TOKEN")
    if tok:
        import json
        import urllib.request
        base = os.environ.get("WF_BASE", "https://app-waterfall-dev-v2.icyplant-"
                              "026fb2db.eastus.azurecontainerapps.io")
        req = urllib.request.Request(f"{base}/api/data/deals/all",
                                     headers={"Authorization": "Bearer " + tok})
        rows = json.load(urllib.request.urlopen(req, timeout=60)).get("deals") or []
        src = "live"
    else:
        db = os.environ.get("DB_PATH") or os.path.join(ROOT, "waterfall.db")
        if not os.path.exists(db):
            # A linked worktree has no database of its own; use the main
            # checkout's, found through git's shared directory.
            try:
                import subprocess
                common = subprocess.run(
                    ["git", "rev-parse", "--path-format=absolute",
                     "--git-common-dir"], cwd=ROOT, capture_output=True,
                    text=True, timeout=10).stdout.strip()
                db = os.path.join(os.path.dirname(common), "waterfall.db")
            except Exception:
                pass
        if not os.path.exists(db):
            return None, "no data"
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            'SELECT vcode, "Sale_Status", "Sale_Date" FROM deals')]
        src = db
    out = {}
    for r in rows:
        vc = str(r.get("vcode") or "").strip().upper()
        if vc in vcodes:
            out[vc] = (r.get("Sale_Status"), r.get("Sale_Date"))
    return out, src


def main() -> int:
    bad = _selftest()
    if bad:
        for b in bad:
            print(f"FAIL self-test: {b}")
        return 1
    fallback = _literal_set(os.path.join(ROOT, "flask_app", "services",
                                         "portfolio_snapshot_loan.py"),
                            "PROJECTED_YE_NOI_FALLBACK")
    kept = _literal_set(os.path.join(ROOT, "flask_app", "services",
                                     "portfolio_snapshot_service.py"),
                        "KEEP_DESPITE_SOLD")
    deals, src = _load_deals(fallback | {GIANT7})
    if deals is None:
        print("SKIP projected_ye_fallback_retirement_check: no deals data "
              "(set DB_PATH or WF_TOKEN) -- skip is not pass")
        return 0
    fails = evaluate(fallback, kept, deals)
    for m in fails:
        print("FAIL " + m)
    if not fails:
        g = deals.get(GIANT7, ("", None))
        print(f"PASS projected_ye_fallback_retirement_check ({src}): Giant 7 "
              f"Sale_Status={g[0]!r}, Sale_Date={g[1]!r}; fallback={sorted(fallback)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
