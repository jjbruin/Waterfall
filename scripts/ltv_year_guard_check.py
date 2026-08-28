"""Guardrail for the LTV valuation year-end guard.

Three things to prove, because the guard fires on nothing in today's data:

  A. REGRESSION. Recompute every deal's LTV on the real committed
     ``_latest_valuation`` -- "before" out of a worktree pinned at origin/main,
     "after" out of the working tree -- against the LIVE valuations frame.
     Nothing may move, and the loan-page diagnostics must be identical.

  B. THE GUARD WORKS. Inject a SYNTHETIC in-memory 2026-06-30 row for one
     clean deal and show the old code picks it while the new code does not.
     Nothing is written to any table.

  C. REAL YEAR-ENDS STILL QUALIFY. Same synthetic trick with a 2026-12-31 row
     reported at 2027-Q1, proving the boundary moves with the report rather
     than pinning to 2025.

Usage (WF_TOKEN must be set):
    python scripts/ltv_year_guard_check.py fetch   <cache.json>
    python scripts/ltv_year_guard_check.py capture <cache.json> <root> <out.json>
    python scripts/ltv_year_guard_check.py report  <cache.json> <before> <after>
    python scripts/ltv_year_guard_check.py synth   <cache.json> <root> <label>
"""
import json
import os
import sys

QUARTER = "2026-Q1"
SYNTH_DEAL = "P0000068"          # Plymouth — clean, real 2025-12-31 row
DATA_BLOCKED = ["P0000107", "P0000109"]


def _load(root, cache_path):
    sys.path.insert(0, os.path.abspath(root))
    import pandas as pd
    from flask_app.services.portfolio_snapshot_loan import _latest_valuation
    import flask_app.services.portfolio_snapshot_loan as mod
    assert os.path.abspath(mod.__file__).startswith(os.path.abspath(root)), \
        f"wrong module: {mod.__file__}"
    with open(cache_path, encoding="utf-8") as fh:
        c = json.load(fh)
    return pd, _latest_valuation, mod, c


def _fetch(cache_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import live_api as api
    ti = api.token_info()
    print(f"LIVE {ti['username']} {ti['hours_left']}h  "
          f"build={api.get('/api/data/version').get('version')}")
    vals = api.get("/api/data/tables/valuations/rows",
                   params={"page": 1, "page_size": 500})
    assert vals.get("total_pages") == 1, "valuations paged — dedupe needed"
    inv = api.get("/api/data/deals/all").get("deals") or []
    loan = api.get("/api/portfolio-snapshot/loan",
                   params={"investor": "TGAM", "quarter": QUARTER})
    print(f"  valuations={vals.get('total')} deals={len(inv)} "
          f"loan_deals={(loan.get('diagnostics') or {}).get('deals')}")
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump({"valuations": vals.get("rows") or [], "deals": inv,
                   "loan_live": loan}, fh, default=str)
    print(f"  cached -> {cache_path}")
    return 0


def _capture(cache_path, root, out_path):
    pd, latest, mod, c = _load(root, cache_path)
    guarded = "quarter" in latest.__code__.co_varnames
    print(f"  root={root}\n  module={mod.__file__}\n  guard present: {guarded}")
    vals = pd.DataFrame(c["valuations"])
    inv = pd.DataFrame(c["deals"])
    vc_col = next(x for x in inv.columns if x.lower() == "vcode")
    out = {}
    for vcode in sorted({str(v).strip().upper() for v in inv[vc_col]
                         if str(v).strip()}):
        # call the same way the app does — positionally, quarter last
        r = latest(vals, vcode, QUARTER) if guarded else latest(vals, vcode)
        out[vcode] = {"value": r["value"], "as_of": str(r["as_of"])}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"  captured {len(out)} deals -> {out_path}")
    return 0


def _report(cache_path, before_path, after_path):
    with open(cache_path, encoding="utf-8") as fh:
        c = json.load(fh)
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)
    inv = {str(d.get("vcode", "")).upper(): d.get("Investment_Name")
           for d in c["deals"]}
    checks = []

    def chk(label, cond):
        checks.append(bool(cond))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    moved = [k for k in before if before[k] != after.get(k)]
    print("=" * 96)
    print(f"A. REGRESSION — real _latest_valuation over {len(before)} deals, "
          f"live valuations frame, quarter={QUARTER}")
    print("=" * 96)
    for k in moved:
        print(f"  {k} {str(inv.get(k))[:30]:<32}{before[k]} -> {after[k]}")
    print(f"  deals whose selected valuation changed: {len(moved)}")
    chk("no deal's selected valuation moved", not moved)

    with_val = [k for k in after if after[k]["value"]]
    chk(f"deals carrying a valuation unchanged at {len(with_val)}",
        len(with_val) == len([k for k in before if before[k]["value"]]))

    for vc in DATA_BLOCKED:
        chk(f"{vc} unchanged (still data-blocked): "
            f"{before.get(vc, {}).get('value')} -> {after.get(vc, {}).get('value')}",
            before.get(vc) == after.get(vc))

    d = (c["loan_live"].get("diagnostics") or {})
    print(f"\n  live loan-page diagnostics (pre-change render): "
          f"ltv_ok={d.get('ltv_ok')} ltv_no_valuation={d.get('ltv_no_valuation')} "
          f"ltv_dev={d.get('ltv_dev')} ltv_dev_exception={d.get('ltv_dev_exception')}")
    chk("baseline diagnostics are the expected 20/4/9/1",
        (d.get("ltv_ok"), d.get("ltv_no_valuation"), d.get("ltv_dev"),
         d.get("ltv_dev_exception")) == (20, 4, 9, 1))

    print(f"\n  {sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


def _synth(cache_path, root, label):
    """Synthetic-row proofs. In-memory only — nothing is written anywhere."""
    pd, latest, mod, c = _load(root, cache_path)
    guarded = "quarter" in latest.__code__.co_varnames
    vals = pd.DataFrame(c["valuations"])
    print(f"  [{label}] guard present: {guarded}")

    def add(vcode, dt, value):
        row = {col: None for col in vals.columns}
        row.update({"vCode": vcode, "vPropertyName": "SYNTHETIC",
                    "dtValuation": dt, "vMethod": "DCF",
                    "mIncomeCapConcludedValue": value})
        return pd.concat([vals, pd.DataFrame([row])], ignore_index=True)

    def call(frame, vcode, quarter):
        return latest(frame, vcode, quarter) if guarded else latest(frame, vcode)

    base = call(vals, SYNTH_DEAL, QUARTER)
    print(f"\n  baseline, real data only -> {base['as_of']}  "
          f"{base['value']:,.0f}")

    # Two spellings on purpose:
    #   MRI    "2026-06-30T00:00:00"  — what the existing rows look like
    #   PUBLISH "2026-06-30"          — what valuation_nav_service actually
    #                                   writes (as_of.strftime("%Y-%m-%d"))
    # The old code can only parse the first; the second lands as NaT and the
    # row disappears, which is the separate parse bug this branch also fixes.
    print("\n  B. MID-YEAR ROW MUST BE IGNORED "
          f"(deal {SYNTH_DEAL}, report {QUARTER})")
    for spelling, dt in (("MRI      ", "2026-06-30T00:00:00"),
                         ("PUBLISHED", "2026-06-30")):
        r = call(add(SYNTH_DEAL, dt, 999_000_000), SYNTH_DEAL, QUARTER)
        seen = str(r["as_of"]) != "2025-12-31"
        if guarded:
            note = ("PASS — row seen and correctly excluded"
                    if not seen else "FAIL — mid-year row was selected")
        else:
            note = ("row selected — this is the bug the guard fixes" if seen
                    else "row INVISIBLE (parsed to NaT) — the parse bug")
        print(f"     + synthetic {spelling} {dt:<21} -> {r['as_of']}  "
              f"{r['value']:>13,.0f}   {note}")

    print("\n  C. A REAL YEAR-END MUST STILL QUALIFY "
          f"(deal {SYNTH_DEAL}, synthetic 2026-12-31 present)")
    for spelling, dt in (("MRI      ", "2026-12-31T00:00:00"),
                         ("PUBLISHED", "2026-12-31")):
        f2 = add(SYNTH_DEAL, dt, 111_000_000)
        r27 = call(f2, SYNTH_DEAL, "2027-Q1")
        r26 = call(f2, SYNTH_DEAL, QUARTER)
        ok27 = str(r27["as_of"]) == "2026-12-31"
        ok26 = str(r26["as_of"]) == "2025-12-31"
        print(f"     {spelling} {dt:<21} @2027-Q1 -> {r27['as_of']}  "
              f"{'PASS — selected' if ok27 else 'not selected'}")
        print(f"     {'':<9} {'':<21} @{QUARTER}  -> {r26['as_of']}  "
              f"{'PASS — correctly excluded' if ok26 else 'LEAKED'}")

    if guarded:
        print("\n  boundary table (valuation_year_end):")
        for q in ("2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4", "2027-Q1",
                  "garbage", None):
            print(f"     {str(q):<10} -> {mod.valuation_year_end(q)}")
    print("\n  (synthetic rows were in-memory DataFrames only — "
          "no table was written)")
    return 0


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "fetch":
        return _fetch(argv[2])
    if cmd == "capture":
        return _capture(argv[2], argv[3], argv[4])
    if cmd == "report":
        return _report(argv[2], argv[3], argv[4])
    if cmd == "synth":
        return _synth(argv[2], argv[3], argv[4])
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
