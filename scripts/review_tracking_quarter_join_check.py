"""Guardrail: Review Tracking's quarter filter must not hide prior-quarter deals.

Runs the REAL committed ``review_service.get_tracking_data()`` on each side —
"before" from a worktree pinned at the pre-fix commit, "after" from the working
tree — against a LOCAL SQLite mirror of the four live tables it reads
(deals, review_submissions, review_notes, relationships).

Why a real database rather than stubs: the bug is in SQL, so only SQL can
prove it fixed. The mirror is built from the live rows, and the before-side
result is tied out against the LIVE endpoint so a harness that does not
reproduce production is reported rather than trusted.

The defect: the join is unqualified by quarter and the quarter test is applied
afterwards, so a deal whose ONLY submission is an earlier quarter matches
neither ``rs.quarter = :qf`` nor ``rs.quarter IS NULL`` and vanishes instead of
reading Draft. Green Valley Ranch (P0000100, submitted 2026-Q1 only) is the
sole deal in the live table able to show it.

Usage (WF_TOKEN needed for fetch/tieout):
    python scripts/review_tracking_quarter_join_check.py fetch   <cache.json>
    python scripts/review_tracking_quarter_join_check.py capture <cache.json> <root> <out.json>
    python scripts/review_tracking_quarter_join_check.py tieout  <cache.json> <before.json>
    python scripts/review_tracking_quarter_join_check.py report  <cache.json> <before.json> <after.json>
"""
import json
import os
import sqlite3
import sys

INVESTOR = "TGAM"
QUARTERS = [None, "2026-Q1", "2026-Q2", "2026-Q3", "2025-Q3"]
GVR = "P0000100"          # Green Valley Ranch & Telluride — the target
MULTI = "P0000001"        # 30 Bearfoot — two submissions, the duplicate-row case

TABLES = {
    "review_submissions": ["id", "vcode", "quarter", "status", "current_step",
                           "returned_to_step", "submitted_by", "created_at",
                           "updated_at"],
    "review_notes": ["id", "vcode", "quarter", "action", "review_role",
                     "note_text", "user_id", "username", "addressed",
                     "addressed_at", "addressed_by", "created_at"],
    "relationships": ["InvestorID", "InvestmentID", "Name", "OwnershipPct",
                      "StartDate", "EndDate"],
}
DEAL_COLS = ["vcode", "InvestmentID", "Investment_Name", "Sale_Status",
             "Sale_Date", "Lifecycle", "Portfolio_Name"]


def _api():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import live_api as api
    return api


def _rows(api, table):
    """Single page per request where possible; verified against `total`."""
    out, page = [], 1
    while True:
        d = api.get(f"/api/data/tables/{table}/rows",
                    params={"page": page, "page_size": 500})
        total = d.get("total") or 0
        r = d.get("rows") or []
        out.extend(r)
        if len(r) < 500:
            break
        page += 1
        if page > 60:
            break
    if len(out) != total:
        raise RuntimeError(f"{table}: got {len(out)} of {total}")
    return out


def _fetch(cache_path):
    api = _api()
    print("build:", api.get("/api/data/version").get("version"))
    data = {"deals": api.get("/api/data/deals/all").get("deals") or []}
    for t in TABLES:
        data[t] = _rows(api, t)
        print(f"  {t}: {len(data[t])}")
    print(f"  deals: {len(data['deals'])}")

    # The live answers this harness must reproduce, per quarter.
    live = {}
    for q in QUARTERS:
        p = {"investor": INVESTOR}
        if q:
            p["quarter"] = q
        items = api.get("/api/reviews/tracking", params=p).get("items") or []
        live[str(q)] = [{"vcode": i["vcode"], "quarter": i.get("quarter"),
                         "status": i.get("status"),
                         "current_step": i.get("current_step")} for i in items]
        print(f"  live tracking {str(q):<9} -> {len(items)} items")
    data["live"] = live
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=str)
    print(f"cached -> {cache_path}")
    return 0


def _build_db(data, db_path):
    """A real SQLite mirror — the bug is in SQL, so the test must be SQL."""
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cols = ", ".join(f'"{c}" TEXT' for c in DEAL_COLS)
    cur.execute(f"CREATE TABLE deals ({cols})")
    cur.executemany(
        f'INSERT INTO deals VALUES ({",".join("?" * len(DEAL_COLS))})',
        [tuple(None if d.get(c) is None else str(d.get(c)) for c in DEAL_COLS)
         for d in data["deals"]])

    for t, tcols in TABLES.items():
        tdef = ", ".join('"{}" TEXT'.format(c) for c in tcols)
        cur.execute(f"CREATE TABLE {t} ({tdef})")
        cur.executemany(
            f'INSERT INTO {t} VALUES ({",".join("?" * len(tcols))})',
            [tuple(None if r.get(c) is None else str(r.get(c)) for c in tcols)
             for r in data[t]])
    con.commit()
    con.close()


def _apply_variant(data, variant):
    """Which dataset to test against.

    ``live``  — exactly what the live tables hold now.
    ``asof``  — the same rows with Green Valley Ranch's 2026-Q2 submission
                removed, reproducing the state observed earlier on 2026-08-31
                before someone submitted it for Q2 at 18:47 UTC. That
                submission masked the symptom on live data but did not fix the
                defect, so the missing-deal case is still exercised here rather
                than quietly going unproven.
    """
    if variant == "live":
        return data
    if variant != "asof":
        raise SystemExit(f"unknown variant {variant!r}")
    d = dict(data)
    d["review_submissions"] = [
        r for r in data["review_submissions"]
        if not (str(r.get("vcode")).upper() == GVR
                and r.get("quarter") == "2026-Q2")]
    dropped = len(data["review_submissions"]) - len(d["review_submissions"])
    print(f"  variant=asof: dropped {dropped} submission row(s) for {GVR} 2026-Q2")
    return d


def _capture(cache_path, root, out_path, variant="live"):
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    data = _apply_variant(data, variant)

    db_path = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                           f"mirror_{os.path.basename(root)}_{variant}.sqlite")
    _build_db(data, db_path)

    import sqlalchemy as sa
    engine = sa.create_engine(f"sqlite:///{db_path}")

    import flask_app.services.review_service as rsvc
    assert os.path.abspath(rsvc.__file__).startswith(root), (
        f"imported the wrong review_service: {rsvc.__file__}")
    print(f"  root={root}\n  review_service={rsvc.__file__}")

    # The mirror IS the database for this run. _ensure_tables would try to
    # create what already exists, so both are pointed at the mirror.
    rsvc.get_engine = lambda: engine
    rsvc._ensure_tables = lambda: None

    out = {}
    for q in QUARTERS:
        try:
            items = rsvc.get_tracking_data(quarter_filter=q,
                                           investor_filter=INVESTOR)
            out[str(q)] = [{"vcode": i["vcode"], "quarter": i.get("quarter"),
                            "status": i.get("status"),
                            "current_step": i.get("current_step")}
                           for i in items]
        except Exception as exc:
            out[str(q)] = {"error": f"{type(exc).__name__}: {exc}"}
    # Recorded so the report rebuilds the submission map from the SAME dataset
    # this ran against. Reading it from the raw cache instead made the report
    # believe Green Valley Ranch had a 2026-Q2 submission that the `asof`
    # variant had removed.
    out["_variant"] = variant
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    for q, v in out.items():
        if q == "_variant":
            continue
        print(f"  {q:<9} -> "
              f"{v['error'] if isinstance(v, dict) else str(len(v)) + ' items'}")
    return 0


def _tieout(cache_path, before_path):
    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    live = data["live"]
    print("Tie-out: local BEFORE vs LIVE, per quarter")
    bad = 0
    for q in map(str, QUARTERS):
        lv, bv = live.get(q) or [], before.get(q) or []
        if isinstance(bv, dict):
            print(f"  {q:<9} local ERRORED: {bv}")
            bad += 1
            continue
        lset = {(r["vcode"], r["status"], str(r["current_step"])) for r in lv}
        bset = {(r["vcode"], r["status"], str(r["current_step"])) for r in bv}
        ok = lset == bset
        bad += not ok
        print(f"  {q:<9} live {len(lv):>3}  local {len(bv):>3}   "
              f"{'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            for x in sorted(lset - bset)[:6]:
                print(f"       only live : {x}")
            for x in sorted(bset - lset)[:6]:
                print(f"       only local: {x}")
    print(f"\n  {'FAITHFUL — the mirror reproduces production' if not bad else 'NOT FAITHFUL'}")
    return 0 if not bad else 1


def _report(cache_path, before_path, after_path):
    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)

    variant = after.get("_variant", "live")
    print(f"dataset variant: {variant}")
    data = _apply_variant(data, variant)

    names = {str(d.get("vcode")).upper(): d.get("Investment_Name")
             for d in data["deals"]}
    subs = {}
    for r in data["review_submissions"]:
        subs.setdefault(str(r.get("vcode")).upper(), []).append(
            (r.get("quarter"), r.get("status")))

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    def idx(side, q):
        v = side.get(str(q))
        return {} if isinstance(v, dict) else {r["vcode"]: r for r in v}

    print("=" * 96)
    print("DEAL COUNTS PER QUARTER  (TGAM)")
    print("=" * 96)
    print(f"  {'quarter':<10}{'before':>8}{'after':>8}{'delta':>8}"
          f"   {'GVR before':>22}{'GVR after':>22}")
    for q in QUARTERS:
        b, a = idx(before, q), idx(after, q)
        gb = b.get(GVR)
        ga = a.get(GVR)
        fb = f"{gb['status']}/{gb['quarter']}" if gb else "MISSING"
        fa = f"{ga['status']}/{ga['quarter']}" if ga else "MISSING"
        print(f"  {str(q):<10}{len(b):>8}{len(a):>8}{len(a) - len(b):>+8}"
              f"   {fb:>22}{fa:>22}")

    b2, a2 = idx(before, "2026-Q2"), idx(after, "2026-Q2")
    b1, a1 = idx(before, "2026-Q1"), idx(after, "2026-Q1")

    print("\n" + "=" * 96)
    print("CHECKS")
    print("=" * 96)
    chk("no quarter errored on either side",
        not any(isinstance(v, dict) for v in list(before.values()) + list(after.values())))

    # --- the headline, on whichever dataset this run was given ---------
    # The missing-deal case only exists while Green Valley Ranch's ONLY
    # submission is 2026-Q1. Someone submitted it for Q2 mid-investigation, so
    # the `live` dataset can no longer show it and the `asof` dataset must.
    # Detected rather than assumed, so neither run can pass vacuously.
    missing_before = GVR not in b2
    print(f"\n  dataset: Green Valley Ranch was "
          f"{'MISSING' if missing_before else 'PRESENT'} in 26Q2 before "
          f"-> running the {'MISSING-DEAL' if missing_before else 'NO-REGRESSION'}"
          f" assertions\n")
    if missing_before:
        chk(f"26Q2 gains exactly one deal ({len(b2)} -> {len(a2)})",
            len(a2) == len(b2) + 1)
        chk(f"...and it is Green Valley Ranch ({GVR})",
            set(a2) - set(b2) == {GVR})
        chk("...now reading Draft, stamped with the requested quarter",
            (a2.get(GVR) or {}).get("status") == "draft"
            and (a2.get(GVR) or {}).get("quarter") == "2026-Q2"
            and str((a2.get(GVR) or {}).get("current_step")) == "0")
    else:
        chk(f"26Q2 membership unchanged ({len(b2)} -> {len(a2)})",
            set(a2) == set(b2))
        chk("Green Valley Ranch keeps its real 2026-Q2 submission",
            (a2.get(GVR) or {}).get("status")
            == (b2.get(GVR) or {}).get("status") != "draft")

    # --- its real Q1 submission is not lost, either way ---
    chk("Green Valley Ranch's real 2026-Q1 submission still shows in the "
        "26Q1 view",
        (a1.get(GVR) or {}).get("status") == "pending_head_am"
        and (a1.get(GVR) or {}).get("quarter") == "2026-Q1")
    chk("...unchanged from before", b1.get(GVR) == a1.get(GVR))

    # --- real Q2 submissions keep their status ---
    real_q2 = [v for v in a2 if any(qq == "2026-Q2" for qq, _ in subs.get(v, []))]
    moved = [v for v in real_q2 if b2.get(v) != a2.get(v)]
    chk(f"every deal with a real 2026-Q2 submission keeps its exact status "
        f"({len(real_q2)} deals, {len(moved)} moved)", not moved)
    for v in moved[:8]:
        print(f"           {v} {names.get(v)}: {b2.get(v)} -> {a2.get(v)}")
    chk("...and none of them was reset to Draft",
        all((a2.get(v) or {}).get("status") != "draft" or
            any(s == "draft" for _, s in subs.get(v, []))
            for v in real_q2))

    # --- no deal LOST ---
    for q in QUARTERS:
        b, a = idx(before, q), idx(after, q)
        lost = set(b) - set(a)
        chk(f"{str(q):<9} no deal lost ({len(lost)})", not lost)
        if lost:
            print(f"           {sorted(lost)}")

    # --- duplicate rows ---
    print()
    for q in QUARTERS:
        v = after.get(str(q))
        if isinstance(v, dict):
            continue
        vcs = [r["vcode"] for r in v]
        dupes = len(vcs) - len(set(vcs))
        chk(f"{str(q):<9} no duplicate deal rows after ({dupes})", dupes == 0)
    vb = before.get("None")
    dupes_before = (len([r["vcode"] for r in vb])
                    - len({r["vcode"] for r in vb})) if not isinstance(vb, dict) else -1
    print(f"           (unfiltered duplicates before: {dupes_before}; "
          f"{MULTI} submissions = {subs.get(MULTI)})")

    # --- sold-deal differences untouched ---
    for vc, label in (("PCITWES", "City West"), ("P0000017", "East Manchester")):
        same = all(idx(before, q).get(vc) == idx(after, q).get(vc)
                   for q in QUARTERS)
        chk(f"{label} ({vc}) unchanged in every quarter", same)

    # --- nothing else moved ---
    others = []
    for q in QUARTERS:
        b, a = idx(before, q), idx(after, q)
        for vc in set(b) & set(a):
            if vc != GVR and b[vc] != a[vc]:
                others.append((q, vc))
    chk(f"no other deal's status changed in any quarter ({len(others)})",
        not others)
    for q, vc in others[:10]:
        print(f"           {q} {vc} {names.get(vc)}: "
              f"{idx(before, q)[vc]} -> {idx(after, q)[vc]}")

    passed = sum(1 for _, c in checks if c)
    print(f"\n  {passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "fetch":
        return _fetch(argv[2])
    if cmd == "capture":
        return _capture(argv[2], argv[3], argv[4],
                        argv[5] if len(argv) > 5 else "live")
    if cmd == "tieout":
        return _tieout(argv[2], argv[3])
    if cmd == "report":
        return _report(argv[2], argv[3], argv[4])
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
