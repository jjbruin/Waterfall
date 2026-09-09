"""Guardrail: Current Pref Equity Balance is AS OF THE REPORT QUARTER.

THE DEFECT this was written for. ``_enrich_pe_from_deal_result`` overwrote
``current_pe_balance`` from ``seed_states``, and built that deal result with the
GLOBAL ``actuals_through`` (2026-07-31 on live) instead of the report quarter.
Every other figure on the page is filtered to the quarter end, so a
contribution landing between the two dates appeared in the balance and nowhere
else, and the balance stopped tying to the Funded to Date printed above it.

The symptom was QUARTER-INVARIANCE, which is what makes it unmistakable. On
live, Burton Retail Portfolio (P0000109) printed the SAME balance in every
quarter while its funded figure correctly stayed at the quarter-end position:

    quarter    funded_to_date    current_pe_balance
    2025-Q4        26,597,500            54,227,500
    2026-Q1        26,597,500            54,227,500
    2026-Q2        26,597,500            54,227,500

The 54,227,500 includes a contribution dated 2026-07-01 — so the 25Q4 report
was showing capital that would not be contributed for another seven months.

IT WAS NEVER READING THE COMMITMENT, though it looked exactly as if it were:
that 7/1 draw completed the tranche, so 26,597,500 + 27,630,000 happens to
equal ``committed_pe`` to the cent. JB Fair Park (P0000021) had the same defect
without the coincidence — +1,462,095 from a 2026-07-30 contribution, against a
54.2M-style commitment it is nowhere near. Its next draw (2026-08-19,
1,286,682) was NOT in the figure, which is what pins the cause to the 7/31
boundary rather than to "as of today".

WHAT IS ASSERTED, and why it is done this way. The One Pager cannot be
assembled locally — the local SQLite carries only app-managed tables, no MRI
data — so this does not try. It checks the two things that actually define the
fix, and leaves the third to live:

  1. THE CUTOFF RULE, as pure logic: quarter end, capped at the global actuals
     boundary, and unchanged when no quarter is given.
  2. THE WIRING, by instrumenting ``get_cached_deal_result`` to capture the
     cutoff it is handed and returning a seed built from BURTON'S REAL
     ACCOUNTING ROWS (below). So the balance this produces is the balance the
     real seeding would produce at that cutoff, and the quarter-invariance is
     reproduced or gone.
  3. ``--live`` re-reads the deployed page for the before/after against
     production. It FAILS until this ships, by design.

Read-only. ``--live`` needs WF_TOKEN; the default mode needs nothing.

Usage
    python scripts/onepager_pe_balance_quarter_check.py
    set WF_TOKEN=<jwt> && python scripts/onepager_pe_balance_quarter_check.py --live
"""
import argparse
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: The real non-OP capital rows behind the two affected deals, read off live on
#: 2026-09-09 (pref-balance-detail, PPIBRP / PPI32). Dates are the whole point,
#: so they are transcribed rather than refetched: the test must keep meaning the
#: same thing when live moves on.
#: vcode -> (name, [(EffectiveDate, signed amount as accounting stores it)])
REAL_CAPITAL = {
    "P0000109": ("Burton Retail Portfolio", [
        (date(2025, 8, 28), -26_597_500.00),
        (date(2026, 7, 1), -27_630_000.00),      # ONE DAY after 26Q2 quarter end
    ]),
    "P0000021": ("JB Fair Park", [
        (date(2021, 2, 24), -7_150_000.00),
        (date(2026, 7, 30), -1_462_094.72),      # inside the 7/31 boundary
        (date(2026, 8, 19), -1_286_682.07),      # OUTSIDE it — must never appear
    ]),
}

#: What each deal's balance MUST be once the cutoff is the quarter end.
EXPECTED = {
    ("P0000109", "2025-Q4"): 26_597_500.00,
    ("P0000109", "2026-Q1"): 26_597_500.00,
    ("P0000109", "2026-Q2"): 26_597_500.00,
    ("P0000021", "2026-Q1"): 7_150_000.00,
    ("P0000021", "2026-Q2"): 7_150_000.00,
}

GLOBAL_BOUNDARY = "2026-07-31"


class FakeState:
    """Only what the enrichment reads off a seed state."""

    def __init__(self, outstanding):
        self.total_capital_outstanding = outstanding
        self.pref_unpaid_compounded = 0.0
        self.pref_accrued_current_year = 0.0
        self.add_pref_unpaid_compounded = 0.0
        self.add_pref_accrued_current_year = 0.0


def seeded_balance(vcode, cutoff):
    """Capital outstanding the real seeding would produce at ``cutoff``.

    Contributions are stored negative, so the outstanding balance is the
    negated sum of the rows on or before the cutoff — the same sign convention
    ``capital_after`` applies.
    """
    rows = REAL_CAPITAL[vcode][1]
    return -sum(a for d, a in rows if d <= cutoff)


def run_local():
    from flask_app import create_app
    from flask_app.services import compute_service
    from flask_app.services import financials_service as FS
    from one_pager import quarter_to_date_range

    app = create_app()
    results = []

    def chk(name, ok):
        results.append((name, bool(ok)))

    # ── 1. the cutoff rule, as pure logic ────────────────────────────────
    captured = {}

    def fake_get_cached(vcode, sy, hy, pyb, data, **kw):
        cutoff = kw.get("actuals_through")
        captured[vcode] = cutoff
        c = cutoff
        if hasattr(c, "date"):
            c = c.date()
        if isinstance(c, str):
            from datetime import datetime as _dt
            c = _dt.fromisoformat(c[:10]).date()
        return {"partner_results": [],
                "seed_states": {"PPI": FakeState(seeded_balance(vcode, c))}}

    orig = compute_service.get_cached_deal_result
    compute_service.get_cached_deal_result = fake_get_cached
    # The enrichment also calls the pref-detail accrual, which needs real data.
    # Stubbed to None so it takes the seed_states fallback — accrued is not what
    # this guardrail is about, and it was already quarter-scoped.
    orig_accr = FS._compute_accrued_from_pref_detail
    FS._compute_accrued_from_pref_detail = lambda *a, **k: None

    print("CUTOFF AND BALANCE, per quarter (instrumented wiring)")
    print(f"  {'deal':<26}{'quarter':<10}{'cutoff passed':<16}"
          f"{'balance':>16}{'expected':>16}")
    try:
        with app.app_context():
            app.config["ACTUALS_THROUGH"] = GLOBAL_BOUNDARY
            for (vcode, q), want in EXPECTED.items():
                pe = {"committed_pe": 1.0, "accrued_balance": 0.0,
                      "current_pe_balance": None, "funded_to_date": 0.0,
                      "return_of_capital": 0.0}
                FS._enrich_pe_from_deal_result(pe, vcode, {}, q)
                got = pe["current_pe_balance"]
                cut = captured.get(vcode)
                name = REAL_CAPITAL[vcode][0]
                ok = got is not None and abs(got - want) < 1
                print(f"  {name[:26]:<26}{q:<10}{str(cut)[:10]:<16}"
                      f"{(got or 0):>16,.0f}{want:>16,.0f}"
                      f"{'' if ok else '   << FAILS'}")
                chk(f"{name} {q} balance is the quarter-end position", ok)
                # The cutoff must never exceed the quarter end...
                _, q_end = quarter_to_date_range(q)
                c = cut.date() if hasattr(cut, "date") else cut
                if isinstance(c, str):
                    from datetime import datetime as _dt
                    c = _dt.fromisoformat(c[:10]).date()
                chk(f"{name} {q} cutoff <= quarter end", c <= q_end)
                # ...nor the global boundary, past which the engine is on
                # forecast rather than accounting.
                chk(f"{name} {q} cutoff <= global actuals boundary",
                    c <= date(2026, 7, 31))

            # QUARTER-INVARIANCE IS THE SIGNATURE OF THE BUG. Burton's balance
            # must now MOVE with the quarter... or rather, must equal the
            # quarter-end funded position in each, which for Burton is the same
            # 26,597,500 in all three (its only other draw is post-boundary).
            # So the sharper test is that the 7/1 draw is absent from ALL of
            # them, and that JB Fair Park's 7/30 draw is absent too.
            burton = []
            for q in ("2025-Q4", "2026-Q1", "2026-Q2"):
                pe = {"committed_pe": 1.0, "accrued_balance": 0.0,
                      "current_pe_balance": None}
                FS._enrich_pe_from_deal_result(pe, "P0000109", {}, q)
                burton.append(pe["current_pe_balance"])
            chk("Burton no longer carries the 2026-07-01 draw in any quarter",
                all(abs(b - 26_597_500) < 1 for b in burton))

            # A deal whose capital is fully returned must stay at 0, not go
            # negative: `return_of_capital` on the page also absorbs realized
            # gain, so `funded - ROC` would print -1,539,662 for East
            # Manchester where the engine reports 0. This is why the fix moves
            # the DATE and does not swap in that formula.
            pe = {"committed_pe": 1.0, "accrued_balance": 0.0,
                  "current_pe_balance": None}
            compute_service.get_cached_deal_result = (
                lambda v, sy, hy, pyb, d, **kw: {
                    "partner_results": [], "seed_states": {"PPI": FakeState(0.0)}})
            FS._enrich_pe_from_deal_result(pe, "P0000017", {}, "2026-Q2")
            chk("a fully-returned deal stays at 0, never negative",
                pe["current_pe_balance"] == 0.0)

            # No quarter -> unchanged behaviour, the global boundary.
            captured.clear()
            compute_service.get_cached_deal_result = fake_get_cached
            pe = {"committed_pe": 1.0, "accrued_balance": 0.0,
                  "current_pe_balance": None}
            FS._enrich_pe_from_deal_result(pe, "P0000109", {}, None)
            c = captured.get("P0000109")
            chk("no quarter given -> falls back to the global boundary",
                str(c)[:10] == GLOBAL_BOUNDARY)
    finally:
        compute_service.get_cached_deal_result = orig
        FS._compute_accrued_from_pref_detail = orig_accr

    return results


def run_live():
    """The deployed page. FAILS until the fix ships — that is the point."""
    import live_api as api
    results = []
    print("\nLIVE (deployed) — balance vs funded_to_date")
    print(f"  {'deal':<26}{'quarter':<10}{'funded':>16}{'balance':>16}{'gap':>14}")
    for (vcode, q), _want in EXPECTED.items():
        pe = api.get(f"/api/financials/{vcode}/one-pager",
                     params={"quarter": q})["pe_performance"]
        f = pe.get("funded_to_date") or 0
        b = pe.get("current_pe_balance") or 0
        ok = abs(b - f) < 1
        print(f"  {REAL_CAPITAL[vcode][0][:26]:<26}{q:<10}{f:>16,.0f}"
              f"{b:>16,.0f}{b - f:>14,.0f}{'' if ok else '   << FAILS'}")
        results.append((f"LIVE {REAL_CAPITAL[vcode][0]} {q} ties to funded", ok))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also check the deployed page (needs WF_TOKEN)")
    args = ap.parse_args()

    results = run_local()
    if args.live:
        results += run_live()

    print(f"\n{'-' * 72}")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    bad = sum(1 for _, ok in results if not ok)
    print(f"\n{'PASS' if not bad else 'FAIL'} — {len(results) - bad}/"
          f"{len(results)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
