"""Guardrail: development-deal display on the Portfolio Snapshot Operating subtab.

THE BUG. A development classification only ever suppressed ONE column. Econ Occ
rendered the literal "Dev" while the three NOI columns and both growth columns
formatted whatever the data held — so a dev row showed a withheld occupancy
beside real numbers, and where At Close NOI was near zero the growth ratio was
noise:

    Green Valley Ranch (P0000100)   At Close -59,333  ->  Expected Growth -2761.9%
    Pegasus Life Storage (P0000066) At Close 624,689  ->  Expected Growth +169.0%

The PDF (26Q1, page 3) prints "n/a" in EVERY metric cell of a dev row, keeping
only the comment. The fix suppresses all four metric columns to NA_LABEL via
``*_display`` fields, with per-column TEMPORARY exceptions for Jefferson Waters
Creek (NOI shown) and Pegasus Life Storage (Econ Occ + NOI shown).

WHAT THIS PROVES. It runs the REAL committed ``assemble_operating`` -- not a
replica -- against real live data, by injecting the live REST endpoints as its
two dependencies:

    resolved            <- GET /api/portfolio-snapshot/deals   (Step 1 output)
    one_pager_provider  <- GET /api/financials/<vcode>/one-pager

so the exact function the app calls is exercised over the exact data the app
sees. Every rendered cell is then checked against ``_PDF_26Q1_OPERATING``,
transcribed from page 3, and every raw field is checked to confirm nothing moved.

Read-only: GET only, via ``scripts/live_api.py`` (needs WF_TOKEN). Nothing is
written and no local DB is touched.

Usage
    set WF_TOKEN=<jwt>
    python scripts/snapshot_dev_display_check.py                    # TIAA 26Q1 vs PDF
    python scripts/snapshot_dev_display_check.py --quarter 2026-Q2  # rule only, no PDF
    python scripts/snapshot_dev_display_check.py --investor PSC1
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_api as api                                          # noqa: E402
from flask_app.services.portfolio_snapshot_operating import (   # noqa: E402
    NA_LABEL, DEV_SUPPRESSED_COLUMNS, DEV_DISPLAY_EXCEPTIONS,
    assemble_operating, display_values,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DASH = "—"

#: The PDF prints "no figure here" two different ways in the SAME dev row --
#: JB Fair Park reads `n/a | - | n/a | - | n/a | n/a`, with a bare accounting
#: dash in the first two NOI columns and the literal "n/a" in the third. That is
#: Excel formatting noise (a zero under an accounting format renders as "-",
#: an overwritten cell renders as text), not a distinction the report is making,
#: so on a suppressed row either output satisfies the expectation.
#:
#: Kept SEPARATE from None, which still means strictly "a dash, not n/a" and is
#: used where the difference is real -- an exempted or non-dev row that shows a
#: figure must not silently fall to n/a.
BLANK = "<blank>"

#: Differences that are NOT the display rule, recorded so a clean run means
#: clean rather than "clean except the bit we stopped looking at".
KNOWN_DATA_DIFFS = {
    ("P0000066", "noi.at_close"):
        "PDF source carried no At Close NOI for Pegasus; live has $624,689. "
        "Data vintage, not the display rule -- and it is why the PDF's growth "
        "columns read n/a from missing data while ours read n/a by rule.",
}

# ── the PDF, page 3 (Operating), transcribed cell by cell ────────────────
#
# vcode -> (name, econ_occ, noi_at_close, noi_uw_ye, noi_projected_ye,
#           expected_growth, actual_growth)
#
# "n/a" is the PDF's own literal. BLANK accepts n/a-or-dash (see above). None
# means strictly a dash / a figure rounding to zero, and rejects n/a. Numbers
# are $M as printed, growth as a decimal ratio. Econ Occ is the single column
# this subtab renders (the PDF prints three; the backend picks Projected YE ->
# YTD -> At Close, so the PDF's Projected YE column is the comparison).
_PDF_26Q1_OPERATING = {
    # ---- development deals: nothing shown in any metric column ----
    "P0000021": ("JB Fair Park",              "n/a", BLANK, BLANK, "n/a", "n/a", "n/a"),
    "P0000067": ("Brainerd Place Apartments", "n/a", BLANK, "n/a", "n/a", "n/a", "n/a"),
    "P0000077": ("Jefferson Addison Heights", "n/a", BLANK, BLANK, "n/a", "n/a", "n/a"),
    "P0000085": ("Jefferson Eastchase",       "n/a", BLANK, BLANK, "n/a", "n/a", "n/a"),
    "P0000089": ("45th & Main",               "n/a", BLANK, BLANK, "n/a", "n/a", "n/a"),
    "P0000100": ("Green Valley Ranch",        "n/a", BLANK, BLANK, "n/a", "n/a", "n/a"),
    "P0000114": ("Jefferson Stephens",        "n/a", BLANK, BLANK, "n/a", "n/a", "n/a"),
    "P0000110": ("Trolley Square",            "n/a", BLANK, BLANK, "n/a", "n/a", "n/a"),
    # ---- the two TEMPORARY exceptions ----
    # Waters Creek's At Close stays None, not BLANK: NOI is exempted for it, so
    # an n/a in that cell would be the exemption failing and must fail here.
    "P0000078": ("Jefferson Waters Creek",    "n/a", None, 2.9, 2.3, "n/a", "n/a"),
    "P0000066": ("Pegasus Life Storage",      90.3,  None, 1.7, 1.1, "n/a", "n/a"),
    # ---- non-dev controls: must be untouched by the dev rule ----
    "P0000019": ("Giant 7",                   97.8,  8.8, 9.3, 9.4, 0.053, 0.064),
    "P0000075": ("Camp Creek",                94.9,  6.4, 6.9, 6.8, 0.090, 0.068),
    "P0000030": ("Nottingham Village",        89.6,  2.1, 3.6, 3.2, 0.717, 0.532),
    "P0000068": ("The Point at Plymouth Meeting", 91.6, 4.2, 6.6, 4.7, 0.551, 0.119),
}

#: Occupancy and growth move as actuals land (Projected YE is YTD actual plus
#: remainder-of-year budget), and the PDF was produced from an earlier vintage.
#: These tolerances test the DISPLAY RULE, not data agreement -- a wide miss on
#: a control deal still fails, but a couple of points of drift does not.
TOL_OCC_PTS = 2.5
TOL_NOI_M = 0.35
TOL_GROWTH_PTS = 25.0

COLUMNS = ("econ_occ", "noi.at_close", "noi.uw_ye", "noi.projected_ye",
           "expected_growth", "actual_growth")


# ── formatters, replicated from vue_app/src/components/snapshot/format.ts ──

def render(v, kind):
    """What SnapshotOperating.vue prints for one *_display value."""
    if isinstance(v, str):
        return v                                    # "n/a" passes through
    if v is None:
        return DASH
    if kind == "pctpts":
        return f"{v:.1f}%"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "m":
        import re
        return re.sub(r"^-(0\.0*)$", r"\1", f"{v / 1e6:.1f}")
    return str(v)


def cells(row):
    """The six metric cells of a row, as the UI renders them."""
    nd = row.get("noi_display") or {}
    return {
        "econ_occ": render(row.get("econ_occ_display"), "pctpts"),
        "noi.at_close": render(nd.get("at_close"), "m"),
        "noi.uw_ye": render(nd.get("uw_ye"), "m"),
        "noi.projected_ye": render(nd.get("projected_ye"), "m"),
        "expected_growth": render(row.get("expected_growth_display"), "pct"),
        "actual_growth": render(row.get("actual_growth_display"), "pct"),
    }


def raw_of(row, col):
    """The unsuppressed backend value behind one cell."""
    if col.startswith("noi."):
        return (row.get("noi") or {}).get(col.split(".", 1)[1])
    if col == "econ_occ":
        o = row.get("econ_occ") or {}
        return next((o[k] for k in ("projected_ye", "ytd_actual", "at_close")
                     if o.get(k) is not None), None)
    return row.get(col)


def matches_pdf(col, expected, row):
    """(ok, note) for one cell against the PDF."""
    rendered = cells(row)[col]
    if expected == "n/a":
        return rendered == NA_LABEL, ""
    if expected == BLANK:
        # n/a, an em dash, or a figure rounding to zero all read as "nothing
        # shown" -- which is all the PDF's dash-vs-n/a inconsistency conveys.
        if rendered in (NA_LABEL, DASH):
            return True, ""
        try:
            return abs(float(rendered)) < 0.05, "real figure where PDF is blank"
        except ValueError:
            return False, "unexpected literal"
    if expected is None:
        # PDF prints a bare "-": accept the em dash, or a figure that rounds to
        # zero at the PDF's own precision. Reject a real number and reject n/a.
        if rendered == NA_LABEL:
            return False, "suppressed where PDF shows a dash"
        if rendered == DASH:
            return True, ""
        try:
            return abs(float(rendered)) < 0.05, "non-zero where PDF shows a dash"
        except ValueError:
            return False, "unexpected literal"
    # a real number in the PDF
    if rendered in (NA_LABEL, DASH):
        return False, f"{rendered} where PDF shows {expected}"
    txt = rendered.rstrip("%")
    try:
        got = float(txt)
    except ValueError:
        return False, "unparseable"
    if col == "econ_occ":
        return abs(got - expected) <= TOL_OCC_PTS, f"PDF {expected}"
    if col.startswith("noi."):
        return abs(got - expected) <= TOL_NOI_M, f"PDF {expected}"
    return abs(got - expected * 100) <= TOL_GROWTH_PTS, f"PDF {expected * 100:.1f}"


def build(investor, quarter):
    """Run the REAL assemble_operating over live data."""
    resolved = api.get("/api/portfolio-snapshot/deals",
                       params={"investor": investor, "quarter": quarter})
    if resolved.get("error"):
        raise SystemExit(f"Step 1 failed: {resolved['error']}")

    cache = {}

    def one_pager_provider(vcode, q):
        key = (vcode, q)
        if key not in cache:
            cache[key] = api.get(f"/api/financials/{vcode}/one-pager",
                                 params={"quarter": q})
        return cache[key]

    out = assemble_operating(investor, quarter, resolved=resolved,
                             one_pager_provider=one_pager_provider,
                             comment_loader=lambda i, q: {})
    flat = {r["vcode"]: r for rows in out["groups"].values() for r in rows}
    for r in out["ownership_flagged"]:
        flat[r["vcode"]] = r
    return out, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q1")
    args = ap.parse_args()

    ti = api.token_info()
    print(f"LIVE  token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"actuals_through={api.get('/api/data/config').get('actuals_through')}")
    print(f"running the real assemble_operating for {args.investor} "
          f"{args.quarter}\n")

    out, flat = build(args.investor, args.quarter)
    checks = []
    # The transcription is TIAA's 26Q1 page 3. Any other investor or quarter
    # gets the rule checks only -- applying this table to PSC1 would just report
    # that TIAA's deals are missing from PSC1's page, which is not a finding.
    pdf_scope = (args.investor.upper() == "TGAM" and args.quarter == "2026-Q1")
    pdf = _PDF_26Q1_OPERATING if pdf_scope else {}
    if not pdf:
        print(f"note: no PDF transcription for {args.investor} {args.quarter} "
              f"— checking the display rule only\n")

    # ---- 1. the rule, on every dev row ----
    print("=" * 112)
    print("RULE — every metric column of a development row reads n/a "
          "(exceptions excluded)")
    print("=" * 112)
    hdr = (f"{'deal':<32}{'Econ Occ':>10}{'NOI AC':>9}{'NOI UW':>9}"
           f"{'NOI PJ':>9}{'ExpG':>10}{'ActG':>10}  cmt  exception")
    print(hdr)
    print("-" * 112)
    devs = [r for r in flat.values() if r["is_dev"]]
    for r in sorted(devs, key=lambda x: x["name"]):
        c = cells(r)
        exc = ",".join(r["dev_display_exception"]) or "-"
        print(f"{r['name'][:31]:<32}{c['econ_occ']:>10}{c['noi.at_close']:>9}"
              f"{c['noi.uw_ye']:>9}{c['noi.projected_ye']:>9}"
              f"{c['expected_growth']:>10}{c['actual_growth']:>10}"
              f"{'  ok ' if 'operating_comment' in r else ' MISS'}  {exc}")
        for col in r["dev_suppressed_columns"]:
            for v in display_values(r, col):
                checks.append((f"{r['vcode']} {col} suppressed", v == NA_LABEL))
        for col in r["dev_display_exception"]:
            for v in display_values(r, col):
                checks.append((f"{r['vcode']} {col} exempted (real value)",
                               v != NA_LABEL))
        checks.append((f"{r['vcode']} keeps comment column",
                       "operating_comment" in r))
    print(f"\n  {len(devs)} development deal(s), "
          f"{sum(1 for r in devs if r['dev_display_exception'])} with exceptions")

    # ---- 2. display-only: no raw value may carry the literal ----
    bad = [(r["vcode"], col) for r in flat.values() for col in COLUMNS
           if isinstance(raw_of(r, col), str)]
    checks.append(("no raw metric carries the n/a literal", not bad))
    print(f"  raw fields still numeric/None on all {len(flat)} rows: "
          f"{'yes' if not bad else 'NO — ' + str(bad[:4])}")

    # ---- 3. non-dev rows untouched by THIS rule ----
    #
    # Widened 2026-08-25: the dev rule is no longer the only thing that can
    # suppress a cell. A deal owned for less than one quarter also reads n/a in
    # every metric column (INSUFFICIENT_HISTORY_MONTHS), so a non-dev row
    # carrying that flag is correct rather than a leak — Plaza Del Mar and
    # Hanestowne Waterstone, both bought inside 26Q1, are the two at this
    # quarter. The check still has teeth: any OTHER non-dev row showing n/a
    # fails, and an insufficient-history row must show n/a in EVERY column, so
    # the flag cannot be used to excuse a single stray cell.
    nd = [r for r in flat.values()
          if not r["is_dev"] and not r.get("insufficient_history")]
    leaked = [r["vcode"] for r in nd for col in COLUMNS
              if cells(r)[col] == NA_LABEL]
    checks.append(("no non-dev, non-new row shows n/a", not leaked))
    print(f"  {len(nd)} non-dev row(s) subject to this rule, none suppressed: "
          f"{'yes' if not leaked else 'NO — ' + str(sorted(set(leaked)))}")

    insuf = [r for r in flat.values() if r.get("insufficient_history")]
    partial = [r["vcode"] for r in insuf
               if not all(cells(r)[col] == NA_LABEL for col in COLUMNS)]
    checks.append(("insufficient-history rows are n/a in EVERY column",
                   not partial))
    print(f"  {len(insuf)} insufficient-history row(s) "
          f"({', '.join(sorted(r['name'] for r in insuf)) or 'none'}), "
          f"all fully suppressed: "
          f"{'yes' if not partial else 'NO — ' + str(sorted(set(partial)))}")

    # ---- 4. cell-by-cell vs the PDF ----
    if pdf:
        print("\n" + "=" * 112)
        print("vs 26Q1 PDF, page 3 — every metric cell")
        print("=" * 112)
        print(f"{'deal':<32}{'column':<18}{'rendered':>12}{'PDF':>12}"
              f"   verdict")
        print("-" * 112)
        for vc, (label, *want) in _PDF_26Q1_OPERATING.items():
            r = flat.get(vc)
            if not r:
                print(f"{label:<32}{'NOT IN SET':<18}")
                checks.append((f"{label} present", False))
                continue
            c = cells(r)
            for col, expected in zip(COLUMNS, want):
                ok, note = matches_pdf(col, expected, r)
                shown = ("n/a" if expected == "n/a"
                         else "-" if expected in (None, BLANK)
                         else str(expected))
                known = KNOWN_DATA_DIFFS.get((vc, col))
                verdict = ("ok" if ok else
                           "KNOWN DIFF" if known else "MISMATCH")
                print(f"{label[:31]:<32}{col:<18}{c[col]:>12}{shown:>12}"
                      f"   {verdict}"
                      f"{('  ' + note) if (note and not ok) else ''}")
                # A known data difference is reported, not scored -- but if it
                # ever starts matching, the note is stale and should be deleted.
                if not (known and not ok):
                    checks.append((f"{label} {col} vs PDF", ok))
                if known and ok:
                    checks.append(
                        (f"{label} {col}: KNOWN_DATA_DIFFS entry is now stale "
                         f"and should be removed", False))
            print()
        if KNOWN_DATA_DIFFS:
            print("KNOWN DATA DIFFERENCES (reported, not scored)")
            for (vc, col), why in KNOWN_DATA_DIFFS.items():
                nm = (flat.get(vc) or {}).get("name", vc)
                print(f"  {nm} {col}: {why}")
            print()

    # ---- 5. the exception table is still only the two known deals ----
    checks.append(("exception table unchanged (2 deals, per-column)",
                   set(DEV_DISPLAY_EXCEPTIONS) == {"P0000078", "P0000066"}))
    checks.append(("all four metric columns are in the suppression set",
                   set(DEV_SUPPRESSED_COLUMNS) ==
                   {"econ_occ", "noi", "expected_growth", "actual_growth"}))

    print("=" * 112)
    print(f"  diagnostics: {out['diagnostics']}")
    failed = [lbl for lbl, ok in checks if not ok]
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    for lbl in failed[:30]:
        print(f"    [FAIL] {lbl}")
    if len(failed) > 30:
        print(f"    ... and {len(failed) - 30} more")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
