"""Guardrail: Economic Occupancy scale on the Portfolio Snapshot Operating subtab.

THE BUG. ``SnapshotOperating.vue`` rendered Econ Occ with ``disp(v, 'pct')``,
whose ``fmtPct`` multiplies a decimal ratio by 100. But the value is copied
verbatim out of the One Pager ``property_performance.economic_occ``, where every
branch of ``one_pager.get_property_performance`` has ALREADY scaled it to
percentage points (0-100). So a real 92.23% rendered as "9223.0%". The fix adds
an explicit ``fmtPctPts`` / ``'pctpts'`` kind and uses it on that one cell.

WHAT THIS PROVES. Nothing about the fix can be checked from the DOM without a
browser, so this replicates BOTH formatters in Python -- byte-for-byte the
arithmetic in ``format.ts`` -- and runs them over the real live backend payload
for every deal, every investor and every quarter requested. It then asserts:

  * every OLD rendering of a numeric reading is out of range (that is the bug),
  * every NEW rendering is inside 0-100 (that is the fix),
  * the backend numbers are untouched -- this is a display-only change, so the
    raw ``econ_occ`` dict must be identical before and after; the script reads
    live, which is unaffected by the local edit, and prints the raw values so
    that identity is inspectable rather than asserted,
  * ``is_dev`` deals still render the literal "Dev" and no-reading deals still
    render an em dash -- the two cases that looked CORRECT while the bug was
    live, and which a careless fix could turn into "0.0%".

Read-only against live: GET only, via ``scripts/live_api.py`` (needs WF_TOKEN).
Frozen (approved) payloads are covered too -- ``/bundle`` reports ``source``, and
because the fix is at the render boundary the stored percentage-point values in
an already-frozen payload render correctly without a backfill.

Usage
    set WF_TOKEN=<jwt>
    python scripts/snapshot_econ_occ_scale_check.py                 # TIAA, all quarters
    python scripts/snapshot_econ_occ_scale_check.py --all-investors # every investor
    python scripts/snapshot_econ_occ_scale_check.py --investor TGAM --quarter 2026-Q1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_api as api                                   # noqa: E402

# DASH must be byte-identical to format.ts's em dash for the comparisons below
# to mean anything, so the console has to be able to print it. cp1252 cannot.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DASH = "—"
SNAP = "/api/portfolio-snapshot"


# ── the two formatters, replicated from vue_app/src/components/snapshot/format.ts

def disp_old(v, dp=1):
    """BEFORE: disp(v, 'pct') -> fmtPct -> (v * 100).toFixed(dp) + '%'."""
    if isinstance(v, str):
        return v
    if v is None:
        return DASH
    return f"{v * 100:.{dp}f}%"


def disp_new(v, dp=1):
    """AFTER: disp(v, 'pctpts') -> fmtPctPts -> v.toFixed(dp) + '%'."""
    if isinstance(v, str):
        return v
    if v is None:
        return DASH
    return f"{v:.{dp}f}%"


def rendered_pct(text):
    """The number a rendered cell shows, or None for 'Dev' / em dash."""
    if not text.endswith("%"):
        return None
    try:
        return float(text[:-1])
    except ValueError:
        return None


def rows_of(payload):
    """Every deal row on an operating payload, groups + ownership-flagged."""
    out = []
    for group, rows in (payload.get("groups") or {}).items():
        for r in (rows or []):
            out.append((group, r))
    for r in (payload.get("ownership_flagged") or []):
        out.append(("Ownership % unavailable", r))
    return out


def check_page(investor, quarter, verbose):
    """One investor + quarter. Returns (checks, n_numeric, n_dev, n_dash)."""
    payload = api.get(f"{SNAP}/operating",
                      params={"investor": investor, "quarter": quarter})
    if payload.get("error"):
        print(f"  !! {investor} {quarter}: {payload['error']}")
        return [(f"{investor} {quarter} assembled", False)], 0, 0, 0

    rows = rows_of(payload)
    checks = []
    n_num = n_dev = n_dash = 0

    if verbose:
        print(f"\n  {'deal':<34}{'raw econ_occ':>13}{'basis':>14}"
              f"{'BEFORE':>12}{'AFTER':>10}")
        print("  " + "-" * 83)

    for _group, r in rows:
        v = r.get("econ_occ_display")
        before, after = disp_old(v), disp_new(v)
        basis = r.get("econ_occ_basis") or "-"
        raw = "Dev" if isinstance(v, str) else (DASH if v is None else f"{v:.4f}")

        if verbose:
            print(f"  {(r.get('name') or r.get('vcode'))[:33]:<34}{raw:>13}"
                  f"{basis:>14}{before:>12}{after:>10}")

        if isinstance(v, str):
            # Dev (and any future backend literal) must pass through untouched.
            n_dev += 1
            checks.append((f"{r['vcode']} literal {v!r} passes through",
                           before == after == v))
            continue
        if v is None:
            n_dash += 1
            checks.append((f"{r['vcode']} no reading renders em dash",
                           after == DASH))
            continue

        n_num += 1
        old_n, new_n = rendered_pct(before), rendered_pct(after)

        # The bug: a reading above 1 point was inflated x100 out of the 0-100
        # band. Asserted, not assumed -- if a deal were somehow already a ratio,
        # this fails and says so rather than being quietly "fixed". Guarded on
        # v > 1 because a genuine reading of 0 (or of 0.4 points) renders inside
        # the band either way; that is not a counter-example, just a value the
        # bug could not push out of range.
        if v > 1:
            checks.append((f"{r['vcode']} BEFORE was out of band ({before})",
                           old_n is not None and old_n > 100))
        checks.append((f"{r['vcode']} AFTER in 0-100 ({after})",
                       new_n is not None and 0.0 <= new_n <= 100.0))
        # Display-only: AFTER is the backend number itself, to 1dp -- no
        # arithmetic. Compared against the raw value rather than against BEFORE,
        # because BEFORE's own rounding is amplified 100x and would need a
        # 5-point tolerance to pass, which would prove nothing.
        checks.append((f"{r['vcode']} AFTER == raw econ_occ ({v:.4f})",
                       new_n is not None and abs(new_n - v) <= 0.05))
        checks.append((f"{r['vcode']} BEFORE == raw econ_occ x100",
                       old_n is not None and abs(old_n - v * 100) <= 0.05))
        # Occupancy below ~50% is not a scale bug but is worth a human look.
        if new_n is not None and new_n < 50.0:
            print(f"     ~ {r.get('name')}: {after} is low for an occupancy "
                  f"reading (basis {basis}) -- data question, not scale")

    return checks, n_num, n_dev, n_dash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM",
                    help="one code, or a comma-separated list")
    ap.add_argument("--quarter", default=None,
                    help="one quarter; default = every reportable quarter")
    ap.add_argument("--all-investors", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-deal table")
    args = ap.parse_args()

    ti = api.token_info()
    ver = api.get("/api/data/version").get("version")
    cfg = api.get("/api/data/config").get("actuals_through")
    print(f"LIVE  token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={ver}  actuals_through={cfg}")

    quarters = ([args.quarter] if args.quarter
                else (api.get(f"{SNAP}/quarters").get("quarters") or []))
    if args.all_investors:
        investors = [i["code"] for i in
                     (api.get(f"{SNAP}/investors").get("investors") or [])]
    else:
        investors = [c.strip().upper()
                     for c in args.investor.split(",") if c.strip()]
    print(f"scope: {len(investors)} investor(s) x {len(quarters)} quarter(s)")

    checks, tot = [], {"num": 0, "dev": 0, "dash": 0, "pages": 0}
    for inv in investors:
        for q in quarters:
            # Verbose only for the first quarter of each investor -- the full
            # per-deal table x 16 quarters is unreadable.
            verbose = (not args.quiet) and q == quarters[0]
            print(f"\n{'=' * 88}\n{inv}  {q}")
            c, n, d, x = check_page(inv, q, verbose)
            checks += c
            tot["num"] += n
            tot["dev"] += d
            tot["dash"] += x
            tot["pages"] += 1
            print(f"  {n} numeric, {d} Dev, {x} no-reading  "
                  f"({sum(1 for _, ok in c if ok)}/{len(c)} checks passed)")

    # Frozen payloads: same render path, so the stored percentage-point values
    # come out right without a backfill. Reported so it is visible, not assumed.
    print(f"\n{'=' * 88}\nFROZEN (approved) PAYLOADS")
    frozen = 0

    # portfolio_snapshot_frozen is created lazily by freeze()._ensure_table, so
    # its absence from live introspection means nothing has EVER been approved.
    # Checked first because /bundle assembles all four subtabs, and probing it
    # per investor x quarter is the most expensive thing this script could do.
    try:
        live_tables = {(t if isinstance(t, str) else t.get("name"))
                       for t in (api.get("/api/data/tables").get("tables") or [])}
    except Exception:
        live_tables = None
    if live_tables is not None and "portfolio_snapshot_frozen" not in live_tables:
        print("  portfolio_snapshot_frozen does not exist on this database -- "
              "no report has ever been approved, so there is nothing frozen to\n"
              "  re-render. The fix is unit-preserving (the stored numbers stay "
              "percentage points), so a future freeze renders correctly too.")
        checks.append(("frozen-payload state determined", True))
        frozen = -1             # -1 = determined by introspection, not probed
        investors = []          # skip the /bundle probe entirely

    for inv in investors:
        for q in quarters:
            try:
                b = api.get(f"{SNAP}/bundle",
                            params={"investor": inv, "quarter": q})
            except Exception as exc:
                print(f"  {inv} {q}: bundle unavailable ({str(exc)[:60]})")
                continue
            if str(b.get("source") or "").lower() != "frozen":
                continue
            frozen += 1
            op = ((b.get("subtabs") or {}).get("operating")) or {}
            vals = [r.get("econ_occ_display") for _g, r in rows_of(op)]
            nums = [v for v in vals if isinstance(v, (int, float))]
            ok = all(0.0 <= v <= 100.0 for v in nums)
            print(f"  {inv} {q}: frozen, {len(nums)} numeric readings, "
                  f"range {min(nums):.2f}-{max(nums):.2f}" if nums
                  else f"  {inv} {q}: frozen, no numeric readings")
            checks.append((f"{inv} {q} frozen readings render in 0-100", ok))
    if frozen == 0:
        print("  none approved yet in scope -- nothing frozen to re-render")

    failed = [label for label, ok in checks if not ok]
    print(f"\n{'=' * 88}")
    print(f"{tot['pages']} page(s): {tot['num']} numeric readings, "
          f"{tot['dev']} Dev, {tot['dash']} no-reading")
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    for label in failed[:25]:
        print(f"    [FAIL] {label}")
    if len(failed) > 25:
        print(f"    ... and {len(failed) - 25} more")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
