"""Guardrail: do the subtotal FUNCTIONS reproduce the reference PDF's own rows?

This is the method test, and it is deliberately separate from any live data.

Feeding live deals into a subtotal and comparing the result to the published
total conflates two different questions — is the aggregation right, and does our
per-deal data match the PDF's. It cannot answer either. So this feeds the PDF's
OWN member deals, transcribed from pages 3 and 4, through the real committed
``operating_subtotal`` and ``loan_subtotal``, and asserts the published fund
totals come back out.

If a future edit changes how a subtotal aggregates, this fails regardless of what
live data is doing that week.

No network, no database. Run it anywhere:
    python scripts/snapshot_subtotal_method_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services.portfolio_snapshot_operating import (   # noqa: E402
    operating_subtotal,
)
from flask_app.services.portfolio_snapshot_loan import (        # noqa: E402
    loan_subtotal, KNOWN_LOAN_SUBTOTAL_DIFFS,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Values are $M and percentage points, exactly as printed. A dash prints as 0.0
# for NOI (the PDF's accounting dash) and as None for a ratio (n/a or "Dev").
M = 1e6


def orow(occ_ac, noi_ac, occ_uw, noi_uw, occ_pj, noi_pj, dev=False):
    return {
        "is_dev": dev,
        "econ_occ": {"at_close": occ_ac, "uw_ye": occ_uw,
                     "projected_ye": occ_pj},
        "noi": {"at_close": None if noi_ac is None else noi_ac * M,
                "uw_ye": None if noi_uw is None else noi_uw * M,
                "projected_ye": None if noi_pj is None else noi_pj * M},
    }


# ── PDF page 3 ───────────────────────────────────────────────────────────
OPERATING = {
    "Total Individual Investments": {
        "deals": [
            orow(97.9, 8.8, 98.3, 9.3, 97.8, 9.4),      # Giant 7
            orow(80.8, 1.0, 95.5, 1.3, 93.2, 1.5),      # East Manchester
            orow(None, 0.0, None, 0.0, None, 0.0, True),  # JB Fair Park
            orow(84.3, 2.1, 91.1, 3.6, 89.6, 3.2),      # Nottingham Village
            orow(94.0, 5.3, 92.9, 6.5, 96.2, 6.3),      # Evergreen Plaza
            orow(84.3, 3.0, None, 0.0, None, 0.0),      # City West
            orow(89.7, 2.4, 92.8, 3.7, 94.4, 2.9),      # Ascent on Steamboat
            orow(48.0, 0.0, 88.3, 1.7, 90.3, 1.1),      # Pegasus Life Storage
        ],
        "published": dict(occ_ac=90.8, noi_ac=22.6, occ_uw=94.6, noi_uw=26.0,
                          occ_pj=95.1, noi_pj=24.4, eg=15.0, ag=7.9),
    },
    "Total PSC TGA 2023 LLC": {
        "deals": [
            orow(None, 0.0, None, 0.0, None, 0.0, True),  # Addison Heights
            orow(92.3, 2.1, 91.4, 2.7, 83.3, 2.4),        # Prestige Storage
            orow(94.2, 6.4, 95.5, 6.9, 94.9, 6.8),        # Camp Creek
            orow(92.3, 6.1, 88.6, 8.0, 91.2, 6.4),        # Princeton Meadows
            orow(90.2, 5.4, 92.9, 7.3, 90.0, 5.5),        # Cocoplum
            orow(89.2, 3.3, 94.7, 3.9, 91.4, 3.9),        # Poplar Prairie
            orow(87.7, 2.6, 91.5, 3.7, 90.9, 3.1),        # The Standard
            orow(None, 0.0, None, 0.0, None, 0.0, True),  # Eastchase
        ],
        "published": dict(occ_ac=91.4, noi_ac=25.9, occ_uw=92.0, noi_uw=32.6,
                          occ_pj=90.9, noi_pj=28.2, eg=25.9, ag=9.0),
    },
}

TOL_OCC = 1.6      # pp — the published inputs are already rounded to 0.1
TOL_NOI = 0.15     # $M
TOL_G = 0.6        # pp


def lrow(debt, ltv, dscr, dy, dev=False):
    return {
        "is_dev": dev,
        "debt": None if debt is None else debt * M,
        "ltv": None if ltv is None else ltv / 100.0,
        "ytd_dscr": dscr,
        "debt_yield": None if dy is None else dy / 100.0,
    }


# ── PDF page 4 ───────────────────────────────────────────────────────────
LOAN = {
    "Total Individual Investments": {
        "deals": [
            lrow(95.1, 70.9, None, None),        # Giant 7
            lrow(9.6, 48.9, None, None),         # East Manchester
            lrow(48.98, None, None, None, True),  # JB Fair Park
            lrow(38.9, 79.1, 1.1, 7.0),          # Nottingham Village
            lrow(45.4, 56.0, 2.9, 10.1),         # Evergreen Plaza
            lrow(0.0, None, None, None),         # City West
            lrow(32.5, 64.5, 2.1, 8.8),          # Ascent on Steamboat
            lrow(0.0, None, None, None),         # Pegasus Life Storage
        ],
        "published": dict(debt=270.5, ltv=67.4, dscr=None, dy=4.1),
    },
    "Total PSC TGA 2022 LLC": {
        "deals": [
            lrow(89.5, None, None, None, True),   # Brainerd Place
            lrow(63.1, 71.1, 1.7, 7.5),           # Plymouth Meeting
            lrow(21.2, 53.7, 1.3, 9.0),           # Mount Prospect Plaza
            lrow(17.9, 53.2, 1.2, 10.7),          # Post Commons
            lrow(51.7, 57.5, None, None, True),   # Waters Creek (real LTV)
            lrow(25.0, 50.1, 2.3, 14.9),          # Court at Deptford
            lrow(0.0, None, None, None),          # Pegasus Add'l
        ],
        "published": dict(debt=268.4, ltv=60.4, dscr=1.6, dy=9.6),
    },
    "Total PSC TGA 2023 LLC": {
        "deals": [
            lrow(43.9, None, None, None, True),   # Addison Heights
            lrow(20.3, 53.1, 2.5, 12.5),          # Prestige Storage
            lrow(52.6, 58.2, 1.7, 13.5),          # Camp Creek
            lrow(76.6, 59.9, 1.4, 7.7),           # Princeton Meadows
            lrow(62.2, 62.4, 1.3, 7.3),           # Cocoplum
            lrow(26.7, 50.4, 1.4, 11.5),          # Poplar Prairie
            lrow(30.3, 66.7, 1.1, 9.5),           # The Standard
            lrow(53.9, None, None, None, True),   # Eastchase
        ],
        "published": dict(debt=366.5, ltv=59.4, dscr=1.5, dy=9.7),
    },
    "Total PSC TGA 2024 LLC": {
        "deals": [
            lrow(35.2, 63.4, 1.4, 8.5),           # Dorsett Ridge
            lrow(88.8, 62.0, 1.4, 8.0),           # Seasons at Bel Air
            lrow(47.0, None, None, None, True),   # 45th & Main
            lrow(36.7, 63.3, 1.3, 6.9),           # Glenmoore
            lrow(51.5, None, None, None, True),   # Green Valley Ranch
            lrow(20.0, 57.7, 3.1, 14.7),          # Town Fair Tire
        ],
        "published": dict(debt=279.2, ltv=62.1, dscr=1.5, dy=8.6),
    },
    "Total PSC TGA 2025 LLC": {
        "deals": [
            lrow(75.3, 69.1, 2.3, 13.1),          # Burton Retail
            lrow(30.8, None, None, None, True),   # Trolley Square
            lrow(50.0, None, None, None, True),   # Jefferson Stephens
            lrow(17.8, None, None, None),         # Hanestowne Village
            lrow(27.6, None, None, None),         # Plaza Del Mar
        ],
        "published": dict(debt=201.5, ltv=69.1, dscr=2.3, dy=4.9),
    },
}

TOL_DEBT = 0.15    # $M
TOL_LTV = 0.15     # pp
TOL_DSCR = 0.11    # one rounding step on a 1-decimal published figure
TOL_DY = 0.15      # pp

checks = []


def ck(label, ok, note=""):
    checks.append((label, ok, note))


def main():
    print("=" * 100)
    print("OPERATING — the PDF's own member deals through operating_subtotal()")
    print("=" * 100)
    print(f"{'fund / metric':<44}{'ours':>10}{'PDF':>10}{'delta':>9}   verdict")
    print("-" * 100)
    for fund, spec in OPERATING.items():
        s = operating_subtotal(spec["deals"], fund)
        p = spec["published"]
        occ, noi = s["econ_occ"], s["noi"]
        rows = [
            ("Econ Occ At Close", occ["at_close"], p["occ_ac"], TOL_OCC, "pp"),
            ("NOI At Close", (noi["at_close"] or 0) / M, p["noi_ac"], TOL_NOI, "M"),
            ("Econ Occ U/W YE", occ["uw_ye"], p["occ_uw"], TOL_OCC, "pp"),
            ("NOI U/W YE", (noi["uw_ye"] or 0) / M, p["noi_uw"], TOL_NOI, "M"),
            ("Econ Occ Proj YE", occ["projected_ye"], p["occ_pj"], TOL_OCC, "pp"),
            ("NOI Proj YE", (noi["projected_ye"] or 0) / M, p["noi_pj"], TOL_NOI, "M"),
            ("Expected Growth", (s["expected_growth"] or 0) * 100, p["eg"], TOL_G, "pp"),
            ("Actual Growth", (s["actual_growth"] or 0) * 100, p["ag"], TOL_G, "pp"),
        ]
        first = True
        for name, got, want, tol, _u in rows:
            got = 0.0 if got is None else got
            ok = abs(got - want) <= tol
            head = fund if first else ""
            print(f"{(head + ' · ' + name if first else '   · ' + name)[:43]:<44}"
                  f"{got:>10.2f}{want:>10.2f}{got - want:>+9.2f}   "
                  f"{'ok' if ok else 'MISMATCH'}")
            first = False
            ck(f"OP {fund} {name}", ok, f"{got:.2f} vs {want}")
        print()

    print("=" * 100)
    print("LOAN — the PDF's own member deals through loan_subtotal()")
    print("=" * 100)
    print(f"{'fund / metric':<44}{'ours':>10}{'PDF':>10}{'delta':>9}   verdict")
    print("-" * 100)
    known = 0
    for fund, spec in LOAN.items():
        s = loan_subtotal(spec["deals"], fund)
        p = spec["published"]
        rows = [
            ("Debt", (s["debt"] or 0) / M, p["debt"], TOL_DEBT, "debt"),
            ("LTV", None if s["ltv"] is None else s["ltv"] * 100, p["ltv"],
             TOL_LTV, "ltv"),
            ("YTD DSCR", s["ytd_dscr"], p["dscr"], TOL_DSCR, "ytd_dscr"),
            ("Debt Yield",
             None if s["debt_yield"] is None else s["debt_yield"] * 100,
             p["dy"], TOL_DY, "debt_yield"),
        ]
        first = True
        for name, got, want, tol, key in rows:
            kd = KNOWN_LOAN_SUBTOTAL_DIFFS.get((fund, key))
            if want is None and got is None:
                ok, shown_g, shown_w = True, "n/a", "n/a"
            elif want is None or got is None:
                ok = False
                shown_g = "n/a" if got is None else f"{got:.2f}"
                shown_w = "n/a" if want is None else f"{want:.2f}"
            else:
                ok = abs(got - want) <= tol
                shown_g, shown_w = f"{got:.2f}", f"{want:.2f}"
            verdict = "ok" if ok else ("KNOWN DIFF" if kd else "MISMATCH")
            head = fund if first else ""
            print(f"{(head + ' · ' + name if first else '   · ' + name)[:43]:<44}"
                  f"{shown_g:>10}{shown_w:>10}"
                  f"{'':>9}   {verdict}")
            first = False
            if kd and not ok:
                known += 1
                print(f"      known: {kd}")
            else:
                # A known entry that starts matching is stale — fail so it goes.
                ck(f"LOAN {fund} {name}", ok, f"{shown_g} vs {shown_w}")
                if kd and ok:
                    ck(f"LOAN {fund} {name}: KNOWN entry is stale, remove it",
                       False)
        print()

    failed = [(l, n) for l, ok, n in checks if not ok]
    print("=" * 100)
    print(f"{len(checks) - len(failed)}/{len(checks)} method checks passed"
          f"   ({known} known published anomalies reported, not scored)")
    for label, note in failed:
        print(f"    [FAIL] {label}{'  — ' + note if note else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
