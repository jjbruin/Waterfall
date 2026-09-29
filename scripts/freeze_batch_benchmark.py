"""Measure a full quarter's One Pager freeze: per-investor build vs build-once.

WHAT IS REAL HERE AND WHAT IS MODELLED, because the difference matters.

REAL: the population shape and the code path. 130 investors over 84 distinct
deals with a skewed holding distribution (TIAA ~30 One Pagers, KOC ~15, a long
tail of small investors) — the shape recorded in the deploy history. Every
freeze goes through the SHIPPING `freeze_part`, the real `FreezeBatch`, the real
job runner and a real SQLite store, so the write volume, the JSON serialisation
and the per-investor commits are all genuine.

MODELLED: the cost of building one One Pager. Building a real one needs the
whole application DataFrame set, which is production data and is not copied
here. It is replaced by a stub that burns a fixed, configurable amount of CPU
and returns a payload of realistic size. `--deal-ms` sets it; the default 2300ms
is the figure measured on production on Sep 29 2026 (`GET /one-pager` for
P0000001 26Q2 returned in 2.31s).

So the RATIO is measured and the WALL CLOCK is an extrapolation at a measured
per-deal cost. Both are reported separately and neither is presented as the
other.

Usage
    .venv/Scripts/python.exe scripts/freeze_batch_benchmark.py
    .venv/Scripts/python.exe scripts/freeze_batch_benchmark.py --deal-ms 50
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import tempfile
import time
import tracemalloc

os.environ["FREEZE_ENABLED"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sqlalchemy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--investors", type=int, default=130)
ap.add_argument("--deals", type=int, default=84)
ap.add_argument("--deal-ms", type=float, default=2300.0,
                help="modelled cost of building ONE One Pager, in ms "
                     "(2300 = the figure measured on production)")
ap.add_argument("--work-ms", type=float, default=2.0,
                help="CPU actually burnt per build during the run; the wall "
                     "clock is then scaled to --deal-ms")
args = ap.parse_args()

rnd = random.Random(20260929)
DEALS = [f"P{i:07d}" for i in range(1, args.deals + 1)]
# A SKEWED distribution, not a uniform one: the saving depends entirely on how
# much investors overlap, and a uniform population would flatter the result.
HOLDINGS = {}
for i in range(args.investors):
    if i == 0:
        n = 30                       # TIAA-sized
    elif i == 1:
        n = 15                       # KOC-sized
    elif i < 12:
        n = rnd.randint(8, 20)
    else:
        n = rnd.randint(1, 8)        # the long tail
    HOLDINGS[f"INV{i:03d}"] = rnd.sample(DEALS, min(n, len(DEALS)))

CODES = list(HOLDINGS)
PAIRS = sum(len(v) for v in HOLDINGS.values())
DISTINCT = len({d for v in HOLDINGS.values() for d in v})

#: A payload of roughly the size a real One Pager serialises to. Measured on
#: production: GET /one-pager returned 5,622 bytes for P0000001.
_FILLER = "x" * 4800


def _burn(ms: float):
    end = time.perf_counter() + ms / 1000.0
    x = 0
    while time.perf_counter() < end:
        x += 1
    return x


BUILDS = {"n": 0}


def make_builder():
    def _build(vcode, quarter):
        BUILDS["n"] += 1
        _burn(args.work_ms)
        return {"vcode": vcode, "quarter": quarter, "filler": _FILLER,
                "cap_stack": {"debt": 12.0, "pref": 9_100_000},
                "property_performance": {"noi": 5.0, "dscr": 1.34}}
    return _build


def run(dedup: bool, label: str) -> dict:
    tmpdir = tempfile.mkdtemp(prefix="freeze_bench_")
    eng = sqlalchemy.create_engine(
        f"sqlite:///{os.path.join(tmpdir, 't.db')}",
        connect_args={"check_same_thread": False})

    from flask_app.services import portfolio_snapshot_freeze as F
    from flask_app.services import portfolio_snapshot_persistence as P
    from flask_app.services import freeze_batch as FB
    from flask_app.services import data_service as DS

    F._engine = lambda: eng
    F._is_postgres = lambda: False
    F._data_version = lambda: "build=bench"
    FB._engine = lambda: eng
    FB._is_postgres = lambda: False
    FB._schema_ready.clear()
    DS.get_data = lambda *a, **kw: {}
    P.load_page = lambda i, q: {"comments": [], "footnotes": [], "values": []}
    F.assemble_full_report = lambda inv, q, **kw: {
        "subtabs": {"financial": {"groups": {"G": {"deals": [
            {"vcode": v, "name": v} for v in HOLDINGS[inv]]}}}},
        "errors": {}, "resolution": {"investor_name": inv}}
    # `_default_one_pager_getter` is a FACTORY: it returns the (vcode, quarter)
    # getter. Assigning the getter itself here made every build call it with no
    # arguments, which the freeze reported faithfully as "One Pager could not be
    # built" — a useful reminder that a benchmark can fail loudly and still
    # print a number.
    _builder = make_builder()
    F._default_one_pager_getter = lambda: _builder

    batch = FB.FreezeBatch("2026-Q3")
    if not dedup:
        # THE BEFORE CASE, reproduced exactly: no cache, so a deal is rebuilt for
        # every investor holding it — which is what a fresh provider pair per
        # report amounted to.
        def _no_cache(vcode, quarter):
            if batch._full_op_builder is None:
                batch._full_op_builder = F._default_one_pager_getter()
            return batch._full_op_builder(vcode, quarter)
        getter = _no_cache

    else:
        getter = batch.one_pager_getter

    BUILDS["n"] = 0
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    frozen = 0
    for code in CODES:
        try:
            F.freeze_part(code, "2026-Q3", "one_pagers", "bench",
                          assembler=batch.assembler, one_pager_getter=getter)
            frozen += 1
        except Exception as exc:                                  # noqa: BLE001
            print("  !! failed", code, exc)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {"label": label, "builds": BUILDS["n"], "frozen": frozen,
            "seconds": elapsed, "peak_mb": peak / 1024 / 1024,
            "cache_entries": len(batch._full_op_cache)}


print(f"Population: {len(CODES)} investors, {DISTINCT} distinct deals, "
      f"{PAIRS} (investor, deal) holdings")
print(f"Modelled cost of building one One Pager: {args.deal_ms:.0f}ms "
      f"(burnt {args.work_ms}ms per build, wall clock scaled below)\n")

before = run(dedup=False, label="BEFORE — rebuilt per investor")
after = run(dedup=True, label="AFTER  — built once per deal")

print(f"{'':34s} {'builds':>8s} {'peak MB':>9s} {'measured s':>11s}")
for r in (before, after):
    print(f"{r['label']:34s} {r['builds']:8d} {r['peak_mb']:9.1f} "
          f"{r['seconds']:11.2f}")

# The wall clock that matters is dominated by the per-deal build, so scale the
# build count by the measured production cost. Stated as an extrapolation.
b_s = before["builds"] * args.deal_ms / 1000.0
a_s = after["builds"] * args.deal_ms / 1000.0
print(f"\nExtrapolated at {args.deal_ms:.0f}ms per One Pager build:")
print(f"  BEFORE  {before['builds']:5d} builds -> {b_s/60:6.1f} min")
print(f"  AFTER   {after['builds']:5d} builds -> {a_s/60:6.1f} min")
print(f"  SAVING  {before['builds'] - after['builds']:5d} builds "
      f"({100 * (1 - after['builds'] / max(1, before['builds'])):.1f}%), "
      f"{(b_s - a_s)/60:.1f} min")
print(f"\nCache holds {after['cache_entries']} deal payloads; peak tracemalloc "
      f"went {before['peak_mb']:.1f} MB -> {after['peak_mb']:.1f} MB "
      f"({after['peak_mb'] - before['peak_mb']:+.1f} MB)")
print(f"Both runs froze {before['frozen']} and {after['frozen']} investors.")
