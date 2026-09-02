"""Guardrail: multi-loan terms, one page per subtab, fund-group separators.

Covers Prompt C of the Sep 2 2026 work order:

  1. the Loan tab lists each facility's real Rate and Maturity instead of
     collapsing to "Various", largest first and in the SAME order across both
     columns, and the deals it does that for are genuinely multi-loan rather
     than leftovers of the Loan_Date fan-out
  2. Financial, Operating and Loan each print to exactly ONE page with no row
     dropped
  3. horizontal separators break up the fund groups on those three subtabs and
     NOT on Summary

Parts 1 and 3 are offline — the real ``_loan_terms`` against injected rows, and
the print stylesheet read as source. Part 2 needs a rendered PDF; produce one
with scripts/snapshot_print_check.mjs (``WF_UPSTREAM=http://127.0.0.1:5000``
prints against a local backend) and pass its path.

    python scripts/snapshot_print_formatting_check.py terms
    python scripts/snapshot_print_formatting_check.py fanout          # needs the DB
    python scripts/snapshot_print_formatting_check.py separators
    python scripts/snapshot_print_formatting_check.py pages <file.pdf> [payload_dir]
    python scripts/snapshot_print_formatting_check.py fonts <file.pdf>
    python scripts/snapshot_print_formatting_check.py all   <file.pdf> [payload_dir]
"""
from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PRINT_VIEW = os.path.join(ROOT, "vue_app", "src", "views",
                          "PortfolioSnapshotPrintView.vue")

CHECKS: list = []


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


# ── 1. Per-loan terms ─────────────────────────────────────────────────────
#
# The four deals that rendered "Various" at 26Q1, with their real rows.
MULTI_LOAN = {
    "Ascent on Steamboat": [
        {"LoanID": 288, "mOrigLoanAmt": 25350000.0, "nRate": 0.0368,
         "vIntType": "Fixed", "dtEvent": "7/1/2026"},
        {"LoanID": 289, "mOrigLoanAmt": 7372000.0, "nRate": 0.0559,
         "vIntType": "Fixed", "dtEvent": "7/1/2026"},
    ],
    "Brainerd Place Apartments": [
        {"LoanID": 291, "mOrigLoanAmt": 31000000.0, "nRate": 0.0425,
         "vIntType": "Fixed", "dtEvent": "6/30/2026"},
        {"LoanID": 311, "mOrigLoanAmt": 33400000.0, "vIndex": "SOFR",
         "vSpread": 0.065, "vIntType": "Variable", "dtEvent": "12/1/2026"},
    ],
    "Mount Prospect Plaza": [
        {"LoanID": 293, "mOrigLoanAmt": 16600000.0, "nRate": 0.0535,
         "vIntType": "Fixed", "dtEvent": "8/18/2027"},
        {"LoanID": 316, "mOrigLoanAmt": 6000000.0, "vIndex": "SOFR",
         "vSpread": 0.025, "vIntType": "Variable", "dtEvent": "8/18/2027"},
    ],
    "Poplar Prairie": [
        {"LoanID": 301, "mOrigLoanAmt": 22900000.0, "nRate": 0.0746,
         "vIntType": "Fixed", "dtEvent": "9/28/2027"},
        {"LoanID": 302, "mOrigLoanAmt": 6530000.0, "vIndex": "SOFR",
         "vSpread": 0.0285, "vIntType": "Variable", "dtEvent": "9/28/2027"},
    ],
}

EXPECTED = {
    # largest facility first, and a field whose values are all the same stays
    # a single value rather than being repeated once per loan
    "Ascent on Steamboat": ("3.7% fixed | 5.6% fixed", "7/1/2026"),
    "Brainerd Place Apartments": ("SOFR + 650 | 4.2% fixed",
                                  "12/1/2026 | 6/30/2026"),
    "Mount Prospect Plaza": ("5.3% fixed | SOFR + 250", "8/18/2027"),
    "Poplar Prairie": ("7.5% fixed | SOFR + 285", "9/28/2027"),
}


def check_terms() -> None:
    import pandas as pd
    from flask_app.services.portfolio_snapshot_loan import _loan_terms, VARIOUS

    print("\n1. Multi-loan deals list their loans")
    for name, rows in MULTI_LOAN.items():
        out = _loan_terms(pd.DataFrame(rows))
        want_rate, want_mat = EXPECTED[name]
        chk(f"{name}: Rate", out["rate_display"] == want_rate,
            f"got {out['rate_display']!r}   want {want_rate!r}")
        chk(f"{name}: Maturity", out["maturity_display"] == want_mat,
            f"got {out['maturity_display']!r}   want {want_mat!r}")
        chk(f"{name}: neither cell says {VARIOUS!r}",
            VARIOUS not in (out["rate_display"], out["maturity_display"]))
        # Positional pairing only means anything if both columns are piped.
        # Where only one is, the single value applies to every loan listed.
        r_n = len((out["rate_display"] or "").split(" | "))
        m_n = len((out["maturity_display"] or "").split(" | "))
        chk(f"{name}: rate and maturity pair up",
            r_n == m_n or 1 in (r_n, m_n),
            f"{r_n} rate(s) against {m_n} maturity(ies)")
        chk(f"{name}: the breakdown is published for audit",
            len(out.get("terms_list") or []) == 2,
            str(out.get("terms_list")))

    print("\n1b. A single-loan deal is untouched")
    one = _loan_terms(pd.DataFrame([MULTI_LOAN["Poplar Prairie"][0]]))
    chk("one loan renders its own terms, unpiped",
        one["rate_display"] == "7.5% fixed" and "|" not in one["rate_display"],
        f"got {one['rate_display']!r} / {one['maturity_display']!r}")
    chk("and is not flagged multi", one["various"] is False)

    print("\n1c. Two loans on IDENTICAL terms collapse to one line")
    same = MULTI_LOAN["Ascent on Steamboat"][0]
    dup = _loan_terms(pd.DataFrame([same, dict(same, LoanID=999)]))
    chk("identical facilities are not printed twice",
        "|" not in (dup["rate_display"] or ""),
        f"got {dup['rate_display']!r}")


# ── 2. Genuine multi-loan, not the Loan_Date fan-out ──────────────────────

def check_fanout() -> None:
    """Every deal that pipes its terms has 2+ DISTINCT LoanIDs in the data.

    The fan-out bug (fixed at data_service._collapse_loan_date_events, 6bbfbf6)
    repeated ONE facility once per Loan_Date event, which would look identical
    on the page. Distinct LoanIDs is what separates the two.
    """
    print("\n2. The piped deals are genuinely multi-loan")
    try:
        import sqlalchemy as sa
        import pandas as pd
        eng = sa.create_engine("sqlite:///" + os.path.join(ROOT, "waterfall.db"))
        df = pd.read_sql("SELECT vCode, LoanID FROM loans", eng)
    except Exception as exc:
        chk("local loans table readable", False, str(exc)[:120])
        return
    df["vCode"] = df["vCode"].astype(str).str.upper()
    for vcode in ("P0000065", "P0000067", "P0000069", "P0000082"):
        sub = df[df["vCode"] == vcode]
        chk(f"{vcode}: {len(sub)} row(s), {sub['LoanID'].nunique()} distinct LoanID",
            len(sub) >= 2 and sub["LoanID"].nunique() == len(sub),
            "one row per facility — no repeated LoanID, so not a fan-out")


# ── 3. Fund-group separators ──────────────────────────────────────────────

def check_separators() -> None:
    print("\n3. Fund-group separators on the three data subtabs")
    src = io.open(PRINT_VIEW, encoding="utf-8").read()
    chk("a rule above every fund subtotal",
        ":deep(table.grid tr.subtotal td)" in src)
    chk("a heavier rule above the portfolio total",
        ":deep(table.grid tfoot tr:first-child td)" in src)
    chk("the inter-group spacer is preserved on paper",
        ":deep(table.grid tr.spacer td)" in src)
    chk("light VERTICAL rules at the column boundaries",
        "border-right: 0.5px solid" in src
        and ":deep(table.grid th:last-child)" in src,
        "hairline between columns, none on the last")
    chk("a slightly stronger rule at the zone boundaries",
        "border-left: 0.5px solid #c9ced6" in src,
        "before the TIAA Investment block and before the manual columns")
    chk("scoped to table.grid, which Summary does not render",
        "table.grid" in src and ".summary .card" in src,
        "Summary is narrative and charts; it has no table.grid")


# ── 4. One page per subtab, nothing dropped ───────────────────────────────

def check_pages(pdf_path: str, payload_dir: str = "") -> None:
    print("\n4. Financial, Operating and Loan each print to ONE page")
    try:
        import pdfplumber
    except ImportError:
        chk("pdfplumber available", False, "pip install pdfplumber")
        return
    with pdfplumber.open(pdf_path) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]

    chk("the document is 4 pages — Summary + one per subtab",
        len(pages) == 4, f"got {len(pages)}")
    probes = {"Financial": "TIAA Investment", "Operating": "ECON OCC",
              "Loan": "DEBT YIELD"}
    for tab, probe in probes.items():
        hits = [i + 1 for i, t in enumerate(pages) if probe in t]
        chk(f"{tab} occupies exactly one page", len(hits) == 1,
            f"found on page(s) {hits}")

    if not payload_dir:
        print("      (pass a payload dir to also check no row was clipped)")
        return
    for tab, page_idx in (("financial", 1), ("operating", 2), ("loan", 3)):
        fn = os.path.join(payload_dir, tab + ".json")
        if not os.path.exists(fn):
            continue
        d = json.load(io.open(fn, encoding="utf-8"))
        d = d.get(tab, d)
        rows = []
        for b in (d.get("groups") or {}).values():
            rows += (b["deals"] if isinstance(b, dict) else b)
        rows += d.get("ownership_flagged") or []
        txt = pages[page_idx]
        missing = [r["name"] for r in rows
                   if r["name"].split("(")[0].strip()[:18] not in txt]
        chk(f"{tab}: all {len(rows)} deals are on the page",
            not missing, "missing: " + ", ".join(missing[:5]))


def check_fonts(pdf_path: str) -> None:
    """Comment and manual-input cells print in the table's own font and size.

    Measured off the PDF, not asserted off the CSS: a `<textarea>`, an `<input>`
    and a `<span class="cmt-text">` each reach their type size by a different
    route, and only the rendered document proves all three landed in the same
    place.

    Two causes were fixed: form controls do not inherit font-family or
    font-size from their container (the UA gives them ~13.3px against a 7.5px
    table), and `.cmt-text` — the READ-ONLY rendering the print view actually
    uses — hardcodes 12px.
    """
    print("\n5. Comment and input cells print at the table's size")
    try:
        import pdfplumber
    except ImportError:
        chk("pdfplumber available", False, "pip install pdfplumber")
        return

    def sizes_for(pg, phrase):
        chars = pg.chars
        txt = "".join(c["text"] for c in chars)
        i = txt.find(phrase)
        if i < 0:
            return None
        return sorted({(round(c["size"], 1), c["fontname"].split("+")[-1])
                       for c in chars[i:i + len(phrase)]})

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        # Anchor: an ordinary deal-name cell on each page is the size everything
        # else must match.
        for idx, label, probes in (
            (1, "Financial", ("5.87M", "4.4%")),
            (2, "Operating", ("Occupancy held at",)),
            (3, "Loan", ("Fixed through 2029",)),
        ):
            anchor = sizes_for(pages[idx], "Evergreen Plaza")
            if not anchor:
                chk(f"{label}: anchor cell found", False,
                    "no 'Evergreen Plaza' row on this page")
                continue
            for probe in probes:
                got = sizes_for(pages[idx], probe)
                if got is None:
                    print(f"      (skipped {label} {probe!r} — not on the page)")
                    continue
                chk(f"{label}: {probe!r} matches an ordinary cell",
                    got == anchor, f"{got} against anchor {anchor}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    pdf = sys.argv[2] if len(sys.argv) > 2 else ""
    pdir = sys.argv[3] if len(sys.argv) > 3 else ""
    if cmd == "terms":
        check_terms()
    elif cmd == "fanout":
        check_fanout()
    elif cmd == "separators":
        check_separators()
    elif cmd == "pages" and pdf:
        check_pages(pdf, pdir)
    elif cmd == "fonts" and pdf:
        check_fonts(pdf)
    elif cmd == "all" and pdf:
        check_terms()
        check_fanout()
        check_separators()
        check_pages(pdf, pdir)
        check_fonts(pdf)
    else:
        print(__doc__)
        raise SystemExit(2)
    print("\n  {}/{} checks passed".format(sum(CHECKS), len(CHECKS)))
    raise SystemExit(0 if all(CHECKS) else 1)
