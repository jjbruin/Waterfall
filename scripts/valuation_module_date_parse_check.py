"""Guardrail for the valuation module's date-parse + newest-wins fix.

Four in-scope sites, exercised through the REAL committed functions:

    1  valuation_service._prior_values          parse + tie-break
    2  valuation_service._valuation_history     parse only (already sorts)
    3  valuation_nav_service tie-out lookup     parse + tie-break
    4  valuation_service._prior_rows            parse + tie-break

Site 3's lookup lives inside a large check-builder, so it is exercised through
``_site3()`` below, which is a line-for-line transcription of that block --
kept next to the real code so a drift shows up as a failing regression rather
than a silent divergence.

  A REGRESSION  every site, every deal, live data, before(worktree@origin/main)
                vs after. Expect 0 changes at all four -- one row per deal-year
                and one spelling today, so both halves must be inert.
  B PARSE       synthetic in-memory rows in the two currently-invisible
                spellings; old drops or mis-orders, new handles.
  C TIE-BREAK   synthetic same-year pair in BOTH frame orders; old is
                order-dependent, new returns the newest either way.

Nothing is ever written to a table.

Usage (WF_TOKEN must be set):
    python scripts/valuation_module_date_parse_check.py fetch   <cache.json>
    python scripts/valuation_module_date_parse_check.py capture <cache> <root> <out>
    python scripts/valuation_module_date_parse_check.py report  <before> <after>
    python scripts/valuation_module_date_parse_check.py synth   <cache> <root> <label>
"""
import json
import os
import sys

SYNTH_DEAL = "P0000068"
PRIOR_YEAR = 2026          # the year the synthetic rows live in
#: (label, stored spelling, the year that spelling represents)
INVISIBLE = [("publish path", "2026-06-30", 2026),
             ("legacy MRI", "12/31/2027 0:00", 2027)]


def _load(root, cache_path=None):
    sys.path.insert(0, os.path.abspath(root))
    import pandas as pd
    from flask_app.services import valuation_service as vs
    from flask_app.services import valuation_nav_service as vns
    assert os.path.abspath(vs.__file__).startswith(os.path.abspath(root)), \
        f"wrong valuation_service: {vs.__file__}"
    c = None
    if cache_path:
        with open(cache_path, encoding="utf-8") as fh:
            c = json.load(fh)
    return pd, vs, vns, c


def _site3(pd, df, vcode, year, mixed):
    """Transcription of the valuation_nav_service tie-out prior-year lookup.

    `mixed` selects the new parse; the sort is applied only in the new form,
    mirroring the committed code exactly.
    """
    df = df.copy()
    vcol = "vCode" if "vCode" in df.columns else "vcode"
    if mixed:
        df["_dt"] = pd.to_datetime(df["dtValuation"], format="mixed",
                                   errors="coerce")
    else:
        df["_dt"] = pd.to_datetime(df["dtValuation"], errors="coerce")
    prior = df[(df[vcol].astype(str).str.strip().str.upper() == vcode.upper())
               & (df["_dt"].dt.year == year - 1)]
    if mixed:
        prior = prior.sort_values("_dt", ascending=False)
    if prior.empty:
        return None
    v = pd.to_numeric(prior.iloc[0].get("mMezzanineValue"), errors="coerce")
    return None if pd.isna(v) else float(v)


def _fetch(cache_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import live_api as api
    ti = api.token_info()
    print(f"LIVE {ti['username']} {ti['hours_left']}h  "
          f"build={api.get('/api/data/version').get('version')}")
    d = api.get("/api/data/tables/valuations/rows",
                params={"page": 1, "page_size": 500})
    assert d.get("total_pages") == 1, "valuations paged — dedupe needed"
    inv = api.get("/api/data/deals/all").get("deals") or []
    print(f"  valuations={d.get('total')}  deals={len(inv)}")
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump({"valuations": d.get("rows") or [], "deals": inv}, fh,
                  default=str)
    print(f"  cached -> {cache_path}")
    return 0


def _capture(cache_path, root, out_path):
    pd, vs, vns, c = _load(root, cache_path)
    mixed = 'format="mixed"' in open(vs.__file__, encoding="utf-8").read()
    print(f"  root={root}\n  valuation_service={vs.__file__}\n"
          f"  fix present: {mixed}")
    vals = pd.DataFrame(c["valuations"])
    inv = pd.DataFrame(c["deals"])
    vc_col = next(x for x in inv.columns if x.lower() == "vcode")
    vcodes = sorted({str(v).strip() for v in inv[vc_col] if str(v).strip()})

    years = sorted({int(y) for y in pd.to_datetime(
        vals["dtValuation"], format="mixed", errors="coerce").dt.year.dropna()})

    out = {"site1": {}, "site2": {}, "site3": {}, "site4": {}}
    # Flattened to one key per (year, deal) so the regression count below is a
    # DEAL count, not a year-bucket count.
    for y in years:
        for k, v in vs._prior_values(vals, y).items():
            out["site1"][f"{y}|{k}"] = v
        for k, v in vs._prior_rows(vals, y).items():
            out["site4"][f"{y}|{k}"] = v
    for vc in vcodes:
        h = vs._valuation_history(vals, vc)
        if h:
            out["site2"][vc] = [(e["date"], e["value"]) for e in h]
        for y in years:
            r = _site3(pd, vals, vc, y + 1, mixed)
            if r is not None:
                out["site3"][f"{vc}@{y}"] = r
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"  captured: site1 {len(out['site1'])}, site2 {len(out['site2'])}, "
          f"site3 {len(out['site3'])}, site4 {len(out['site4'])} "
          f"-> {out_path}")
    return 0


def _report(before_path, after_path):
    with open(before_path, encoding="utf-8") as fh:
        b = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        a = json.load(fh)
    names = {"site1": "_prior_values      (parse+tie-break)",
             "site2": "_valuation_history (parse only)",
             "site3": "nav tie-out lookup (parse+tie-break)",
             "site4": "_prior_rows        (parse+tie-break)"}
    print("=" * 92)
    print("A. REGRESSION — real functions, all deals, live data")
    print("=" * 92)
    total = 0
    for s in ("site1", "site2", "site3", "site4"):
        keys = sorted(set(b[s]) | set(a[s]))
        moved = [k for k in keys if b[s].get(k) != a[s].get(k)]
        total += len(moved)
        n = len(keys)
        print(f"  {s}  {names[s]:<38} entries={n:<5} CHANGED={len(moved)}")
        for k in moved[:10]:
            print(f"        {k}: {b[s].get(k)} -> {a[s].get(k)}")
    print(f"\n  TOTAL entries changed across all four sites: {total}")
    ok = total == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] the fix is inert on today's data")
    if not ok:
        print("  !! STOP — a deal moved; do not proceed")
    return 0 if ok else 1


def _synth(cache_path, root, label):
    pd, vs, vns, c = _load(root, cache_path)
    mixed = 'format="mixed"' in open(vs.__file__, encoding="utf-8").read()
    vals = pd.DataFrame(c["valuations"])
    print(f"  [{label}] fix present: {mixed}")

    def mk(dt, val, mezz=12_000_000):
        r = {col: None for col in vals.columns}
        r.update({"vCode": SYNTH_DEAL, "vPropertyName": "SYNTHETIC",
                  "dtValuation": dt, "vMethod": "DCF",
                  "mIncomeCapConcludedValue": val, "mMezzanineValue": mezz})
        return r

    print("\n  B. PARSE — the two spellings that are invisible today")
    for what, dt, yr in INVISIBLE:
        f = pd.concat([vals, pd.DataFrame([mk(dt, 91_000_000)])],
                      ignore_index=True)
        s1 = vs._prior_values(f, yr).get(SYNTH_DEAL)
        s4 = vs._prior_rows(f, yr).get(SYNTH_DEAL)
        s3 = _site3(pd, f, SYNTH_DEAL, yr + 1, mixed)
        hist = vs._valuation_history(f, SYNTH_DEAL)
        pos = next((i for i, e in enumerate(hist)
                    if e["value"] == 91_000_000), None)
        print(f"    {what:<13} {dt!r:<22} year={yr}")
        print(f"        site1 _prior_values  -> {s1}")
        print(f"        site4 _prior_rows    -> "
              f"{'None' if s4 is None else s4.get('value')}")
        print(f"        site3 tie-out        -> {s3}")
        print(f"        site2 history position of the row -> "
              f"{pos} of {len(hist)}   "
              f"{'(newest — correct)' if pos == 0 else '(MIS-ORDERED)'}")

    print("\n  C. TIE-BREAK — two rows in the SAME year, both frame orders")
    # MRI spelling on purpose: the OLD code can parse these, so this
    # isolates the tie-break defect rather than having the parse bug mask
    # it (both rows would simply vanish under the old parse).
    pair = [mk("2026-06-30T00:00:00", 91_000_000, 9_100_000),
            mk("2026-12-31T00:00:00", 95_000_000, 9_500_000)]
    for order_lbl, rows in (("06-30 then 12-31", pair),
                            ("12-31 then 06-30", list(reversed(pair)))):
        f = pd.concat([vals, pd.DataFrame(rows)], ignore_index=True)
        s1 = vs._prior_values(f, PRIOR_YEAR).get(SYNTH_DEAL)
        s4r = vs._prior_rows(f, PRIOR_YEAR).get(SYNTH_DEAL)
        s4 = None if s4r is None else s4r.get("value")
        s3 = _site3(pd, f, SYNTH_DEAL, PRIOR_YEAR + 1, mixed)
        print(f"    frame order {order_lbl}:  site1={s1}  site4={s4}  site3={s3}")
    print("    EXPECTED after the fix: site1/site4 = 95,000,000 and "
          "site3 = 9,500,000 in BOTH orders")
    print("\n  (all synthetic rows were in-memory DataFrames — "
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
        return _report(argv[2], argv[3])
    if cmd == "synth":
        return _synth(argv[2], argv[3], argv[4])
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
