"""Guardrail: a development deal prints N/A for P.E. Yield on Exposure; a
non-development deal with negative NOI still prints its negative yield.

THE DEFECT. Commit bcf19f2 (Sep 30 2026) stopped treating a negative NOI as
"cannot compute", so the One Pager started printing -0.38% (Jefferson Addison
Heights) and -0.98% (Jefferson Eastchase) for yield on exposure. Both are
development deals in lease-up. The Portfolio Snapshot reads "Dev" for every
ratio of a development deal and the sent 26Q2 report printed N/A for these two,
so the two tabs disagreed about the same deal. The fix gates the yield on the
app's one development test (`one_pager._is_dev_deal`).

WHAT IS ASSERTED, IN BOTH DIRECTIONS:
  * a development deal gets None (rendered N/A) whatever its NOI, positive or
    negative;
  * a NON-development deal with negative NOI still gets its negative yield, so
    the fix cannot be satisfied by blanking every negative;
  * a non-development deal with positive NOI still gets its yield;
  * a deal with no NOI, or no exposure, still gets None (no fake 0.0%);
  * the One Pager actually routes the figure through the rule (source check);
  * on real data, every development deal at the quarter prints no yield and the
    non-development deals that printed one before still print it.

Run with --inject to restore the two ways this can go wrong. Every check in the
named section must then fail.

  --inject=nogate      the development gate is dropped  -> Addison Heights and
                       Eastchase print negative yields again
  --inject=blankneg    negatives are blanked for everyone (the pre-bcf19f2
                       rule)                            -> the non-dev negative
                       check fails

The data section reads the local database READ-ONLY (SQLite, DB_PATH or
./waterfall.db). If neither exists it prints SKIPPED and the exit code is still
success for the rule sections only: a skipped section is not a pass.

Usage
    python scripts/onepager_pe_yield_dev_check.py
    python scripts/onepager_pe_yield_dev_check.py --inject=nogate
"""
from __future__ import annotations

import logging
import os
import re
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from flask_app.services import financials_service as FS  # noqa: E402

_PASS = _FAIL = 0
_SKIPPED = []


def chk(label: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [ok]   {label}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))


def _inject(mode: str) -> None:
    if mode == "nogate":
        print("=== INJECTION: the development gate is dropped ===")
        print("    Sections A (dev cases) and C MUST fail.\n")
        orig = FS.compute_pe_yield_on_exposure
        FS.compute_pe_yield_on_exposure = lambda pp, cs, is_dev: orig(pp, cs, False)
    elif mode == "blankneg":
        print("=== INJECTION: negatives are blanked for everyone ===")
        print("    The non-development negative check MUST fail.\n")
        orig = FS.compute_pe_yield_on_exposure

        def blank_negative(pp, cs, is_dev):
            v = orig(pp, cs, is_dev)
            return None if (v is not None and v < 0) else v
        FS.compute_pe_yield_on_exposure = blank_negative


def _pp(noi_ye=None, ytd=None):
    n = {}
    if noi_ye is not None:
        n["actual_ye"] = noi_ye
    if ytd is not None:
        n["ytd_actual"] = ytd
    return {"noi": n}


def _cs(debt=60_000_000.0, pref=20_000_000.0):
    return {"debt": debt, "pref_equity": pref}


def section_rule() -> None:
    f = lambda *a: FS.compute_pe_yield_on_exposure(*a)   # noqa: E731  (resolved late so --inject applies)
    print("\nA. the rule, each case, both directions")

    chk("development deal, NEGATIVE NOI -> no yield (Addison Heights shape)",
        f(_pp(-235_114.0), _cs(37_053_983.1, 24_800_999.99), True) is None)
    chk("development deal, POSITIVE NOI -> no yield either (the gate is on the "
        "classification, not on the sign)",
        f(_pp(900_000.0), _cs(), True) is None)

    neg = f(_pp(-500_000.0), _cs(), False)
    chk("NON-development deal, negative NOI -> still prints a negative yield",
        neg is not None and neg < 0, f"got {neg!r}")
    chk("... and it is the exact ratio, -500,000 / 80,000,000",
        neg is not None and abs(neg - (-500_000.0 / 80_000_000.0)) < 1e-12)

    pos = f(_pp(2_000_000.0), _cs(), False)
    chk("NON-development deal, positive NOI -> prints its yield",
        pos is not None and abs(pos - 0.025) < 1e-12, f"got {pos!r}")

    chk("NON-development deal, NOI never assigned (0) -> None, not a fake 0.0%",
        f(_pp(0), _cs(), False) is None)
    chk("NON-development deal, no NOI keys at all -> None",
        f({"noi": {}}, _cs(), False) is None)
    chk("NON-development deal, actual_ye 0 falls back to ytd_actual",
        abs((f(_pp(0, 400_000.0), _cs(), False) or 0) - 0.005) < 1e-12)
    chk("no exposure (debt + pref = 0) -> None, not a division by zero",
        f(_pp(1_000_000.0), _cs(0.0, 0.0), False) is None)
    chk("missing property performance -> None",
        f({}, _cs(), False) is None and f(None, _cs(), False) is None)
    chk("missing cap stack -> None",
        f(_pp(1_000_000.0), {}, False) is None and f(_pp(1_000_000.0), None, False) is None)


def section_wiring() -> None:
    print("\nB. the One Pager routes the figure through the rule")
    src = open(os.path.join(ROOT, "flask_app", "services", "financials_service.py"),
               encoding="utf-8").read()
    body = src[src.index("def get_one_pager_data("):]
    chk("get_one_pager_data calls compute_pe_yield_on_exposure",
        "compute_pe_yield_on_exposure(" in body)
    chk("... passing the app's one development test (_is_dev_deal)",
        re.search(r"compute_pe_yield_on_exposure\(\s*prop_perf,\s*cap_stack,\s*_is_dev_deal\(", body) is not None)
    chk("the old inline division is gone from get_one_pager_data",
        "noi_ye / senior_plus_pe" not in body)


def _load_app():
    db = os.environ.get("DB_PATH") or os.path.join(ROOT, "waterfall.db")
    if not os.path.exists(db):
        return None
    import pandas as pd
    from flask import Flask
    app = Flask(__name__)
    app.config.update(DB_PATH=os.path.abspath(db), PRO_YR_BASE_DEFAULT=2025,
                      DEFAULT_START_YEAR=2026, DEFAULT_HORIZON_YEARS=10,
                      ACTUALS_THROUGH="2026-07-31")
    return app, pd


def section_data(quarter: str = "2026-Q2") -> None:
    print(f"\nC. the real deals at {quarter} (read-only)")
    loaded = _load_app()
    if loaded is None:
        _SKIPPED.append("C (no local database)")
        print("  SKIPPED: no waterfall.db / DB_PATH. A skipped section is not a pass.")
        return
    app, pd = loaded
    with app.app_context():
        from flask_app.services import data_service as ds
        from config import is_dev_deal
        d = ds.get_data()
        # Local SQLite stores one East Manchester date with a 'T'; the loader's
        # bare to_datetime then fails. Normalise in memory only.
        m = d["mri_loans_raw"]
        m["dtEvent"] = pd.to_datetime(m["dtEvent"], format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S")
        inv = d["inv"]

        def one(vc):
            return FS.get_one_pager_data(
                vc, quarter, d["inv"], d["isbs_raw"], d["mri_loans_raw"], d["mri_val"],
                d["wf"], d["acct"], occupancy_raw=d["occupancy_raw"],
                budget_econ_occ=d.get("budget_econ_occ"), deal_terms=d.get("deal_terms_raw"),
                at_close_noi=d.get("at_close_noi_raw"), commitments_raw=d.get("commitments_raw"),
                event_dates=d.get("event_dates_raw"), full_data=d,
                relationships=d.get("relationships_raw"), mri_loans_all=d.get("mri_loans_all"),
                inspection=d.get("inspection_raw"))["cap_stack"].get("pe_yield_on_exposure")

        devs = []
        for _, r in inv.iterrows():
            strat = r.get("Investment_Strategy") or r.get("Lifecycle")
            sold = str(r.get("Sale_Status", "")).upper() == "SOLD"
            if is_dev_deal(strat) and not sold and int(pd.to_numeric(r.get("Property_Count"), errors="coerce") or 0) >= 1:
                devs.append((str(r["vcode"]), str(r["Investment_Name"])))
        ys = {vc: one(vc) for vc, _ in devs}
        chk(f"there are development deals to test ({len(devs)})", len(devs) >= 5)
        bad = {vc: y for vc, y in ys.items() if y is not None}
        chk("EVERY development deal prints no yield", not bad, f"printing: {bad}")
        for vc, nm in devs:
            if vc in ("P0000077", "P0000085"):
                chk(f"{nm} ({vc}) prints N/A", ys[vc] is None, f"got {ys[vc]!r}")
        # non-development deals that carry a yield today must keep it
        keep = {"P0000028": "Merle Hay", "P0000119": "Presidential Arms", "P0000018": "Evergreen Plaza"}
        for vc, nm in keep.items():
            y = one(vc)
            chk(f"{nm} ({vc}), non-development, still prints a yield", y is not None and y > 0, f"got {y!r}")


def main() -> int:
    for a in sys.argv[1:]:
        if a.startswith("--inject"):
            _inject(a.split("=", 1)[1] if "=" in a else "nogate")
    section_rule()
    section_wiring()
    if "--no-data" not in sys.argv:
        section_data()
    print(f"\n{_PASS} passed, {_FAIL} failed" + (f", skipped: {', '.join(_SKIPPED)}" if _SKIPPED else ""))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
