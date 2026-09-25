"""Build the 26Q2 published overlay from the sent PDFs. OFFLINE — no live call.

WHY THIS REPLACES THE TOKEN-DRIVEN SEEDER. `seed_26q2_freeze.py` needed a live
JWT in the shell to read the app, and `WF_TOKEN` has never reached a tool shell.
So the work is split at the only seam that actually needs splitting:

    THIS SCRIPT   the PDFs -> a JSON overlay. Offline, deterministic, no auth.
    THE APP       resolves deal titles to vcodes and freezes, under the
                  admin's own logged-in session (POST .../freeze-as-sent-overlay).

EVERY PRINTED CELL IS EMITTED, not only the ones that differ from live. The old
script compared against live to decide what to overwrite, which is exactly what
forced it to hold a token. Overwriting a cell that already agrees is a no-op on
the figure and still records provenance, so nothing is lost by emitting all of
them — and the freeze receipt then proves the whole page came from the PDF
rather than partly from the engine.

KEYED BY DEAL TITLE, NOT VCODE. The PDF knows titles; only the app knows vcodes.
Resolving here would mean either a live call (the thing we are removing) or a
hardcoded map that rots. The app resolves at freeze time and reports any title
it cannot place, rather than silently dropping a page.

TWO KINDS OF CELL:
  published  a number in the field's own units — written into the field.
  display    text the field cannot hold in the same units. The money-row
             variances print a PERCENT OF BUDGET while the field stores a
             DOLLAR difference, and the percent on the page is derived in the
             BROWSER, so the stored number reaches no screen. The printed text
             is kept verbatim and the number is left alone. Charlene, Sep 25
             2026: the PDF is the baseline.

    python scripts/build_26q2_overlay.py [--out overlay_26q2.json]

The output is gitignored: it is derived from documents that are not in the repo,
and it carries investor figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUARTER = "2026-Q2"

#: The sent documents. Confirmed as the versions received; the KOC file is
#: byte-identical to the one named vFinal.
SOURCES = {
    "TGAM": {
        "pdf": r"C:\Users\cbui\Downloads\TIAA_26Q2_Portfolio_Report_Revised.pdf",
        "snapshot_pages": [6, 7, 8],
        "one_pager_pages": [10, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 46,
                            49, 53, 56, 59, 62, 65, 68, 72, 76, 80, 83, 86, 90,
                            93, 98, 102, 107],
    },
    "KOCINV": {
        "pdf": r"C:\Users\cbui\Downloads\KOC_26Q2_Report.pdf",
        "snapshot_pages": [],          # KOC's portfolio tables are out of scope
        "one_pager_pages": [8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44,
                            47, 50],
    },
}

#: PDF label -> the dotted path inside a One Pager payload.
OP_PATHS = {
    "Debt": "cap_stack.debt",
    "Pref. Equity": "cap_stack.pref_equity",
    "Partner Equity": "cap_stack.partner_equity",
    "Total Cap": "cap_stack.total_cap",
    "Purchase Price": "cap_stack.purchase_price",
    "Loan Terms": "cap_stack.loan_terms_str",
    "Committed Pref Equity": "pe_performance.committed_pe",
    "Remaining to Fund": "pe_performance.remaining_to_fund",
    "Funded to Date": "pe_performance.funded_to_date",
    "Return of Capital": "pe_performance.return_of_capital",
    "Current Pref Equity Balance": "pe_performance.current_pe_balance",
    "Accrued Balance": "pe_performance.accrued_balance",
}
PP_COLS = ["at_close", "actual_ye", "uw_ye", "ytd_actual", "ytd_budget"]
PP_KEYS = {"Economic Occ.": "economic_occ", "Revenue": "revenue",
           "Expenses": "expenses", "NOI": "noi"}

#: Percent labels whose FIELD STORES A FRACTION. The One Pager prints
#: "8.5%" and `fmtPct` renders `cap_stack.pe_coupon` = 0.085, so the printed
#: percent is divided by 100 — the field can hold it exactly, unlike the
#: variance columns, so these are ordinary published values and NOT display
#: text. The page prints each of them TWICE, once in Capitalization and once in
#: PE Performance, and both places render a different field.
OP_PCT_PATHS = {
    "P.E. Coupon": "cap_stack.pe_coupon",
    "P.E. Participation": "cap_stack.pe_participation",
    "Coupon": "pe_performance.coupon",
    "Participation": "pe_performance.participation",
}

#: Nothing is left unmapped now; kept so a future label added to the extraction
#: without a path is REPORTED rather than silently dropped.
UNMAPPED: list = []


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def money(s):
    """"$2.10M" / "-$0.20M" / "33.5" -> a float.

    THE SIGN CAN PRECEDE THE DOLLAR. The page prints a negative ITD as
    "-$0.20M", and a regex expecting "$" first rejected it — so the row parsed
    one cell short and the LABEL absorbed the missing value ("Brainerd Place
    Apartments 64.4"). Every deal with a negative distribution was affected.
    """
    t = str(s).strip()
    m = re.match(r"^(-)?\$?(-)?([\d,]+(?:\.\d+)?)\s*([MK]?)$", t)
    if not m:
        return None
    v = float(m.group(3).replace(",", ""))
    if m.group(1) or m.group(2):
        v = -v
    return v * (1e6 if m.group(4) == "M" else 1e3 if m.group(4) == "K" else 1)


def pct(s):
    m = re.match(r"^(-?[\d.]+)\s*%$", str(s).strip())
    return float(m.group(1)) if m else None


def one_pager_pages(pdf_path: str, pages: list) -> dict:
    """{deal title: {label: printed string}} for each One Pager page."""
    import pdfplumber
    LABELS = list(OP_PATHS) + list(OP_PCT_PATHS) + UNMAPPED
    ROWS = list(PP_KEYS)
    out = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pno in pages:
            t = pdf.pages[pno - 1].extract_text() or ""
            title = (t.splitlines() or [""])[0].strip()
            cells = {"__page__": pno}
            for lbl in LABELS:
                if lbl == "Loan Terms":
                    continue
                # A bare "Coupon:" must not match "P.E. Coupon:" — the two
                # print the same figure but render DIFFERENT fields.
                pre = r"(?<!P\.E\. )" if lbl in ("Coupon", "Participation") else ""
                m = re.search(rf"{pre}{re.escape(lbl)}:\s*(\$?-?[\d,.]+[MK%]?)", t)
                if m:
                    cells[lbl] = m.group(1)
            for row in ROWS:
                m = re.search(rf"^{re.escape(row)}\s+(.+)$", t, re.M)
                if m:
                    cells[row] = m.group(1).split()
            m = re.search(r"Loan Terms:\s*(.+)$", t, re.M)
            if m:
                cells["Loan Terms"] = re.split(
                    r"\s{2,}|\s(?=Partner Equity:|Pref\. Equity:|Total Cap:|Debt:)",
                    m.group(1).strip())[0].strip()
            out[title] = cells
    return out


def cells_for_page(cells: dict, pdf_name: str) -> tuple:
    """({dotted path: spec}, [unmapped labels seen]) for one One Pager page."""
    page = cells.get("__page__")
    out, seen_unmapped = {}, []

    def put(path, **kw):
        out[path] = {"page": page, "source": pdf_name, **kw}

    for lbl, path in OP_PATHS.items():
        if lbl not in cells:
            continue
        printed = cells[lbl]
        if lbl == "Loan Terms":
            # Text field, text on the page — stored as printed, no conversion.
            put(path, published=str(printed), printed=str(printed))
            continue
        v = money(printed)
        if v is not None:
            put(path, published=v, printed=str(printed))

    for lbl, path in OP_PCT_PATHS.items():
        if lbl not in cells:
            continue
        v = pct(cells[lbl])
        if v is None:
            seen_unmapped.append({"label": lbl, "printed": cells[lbl],
                                  "page": page, "why": "not a percent"})
            continue
        put(path, published=v / 100.0, printed=str(cells[lbl]))

    for lbl in UNMAPPED:
        if lbl in cells:
            seen_unmapped.append({"label": lbl, "printed": cells[lbl],
                                  "page": page, "why": "no vetted field"})

    for row, key in PP_KEYS.items():
        toks = cells.get(row)
        if not toks:
            continue
        for i, col in enumerate(PP_COLS + ["variance"]):
            if i >= len(toks):
                break
            tok = toks[i]
            path = f"property_performance.{key}.{col}"
            if col == "variance" and key != "economic_occ":
                # PRINTED UNITS — see the module docstring. Kept as text.
                if pct(tok) is None:
                    continue
                put(path, published=None, display=str(tok),
                    units="percent_of_budget", printed=str(tok))
                continue
            v = pct(tok) if key == "economic_occ" else money(tok)
            if v is not None:
                put(path, published=v, printed=str(tok))
    return out, seen_unmapped


# ── the Snapshot pages ────────────────────────────────────────────────────
#
# Three tables, one per page, each a different shape. The column order is read
# off the printed header and the row label is everything before the first cell,
# so a deal whose name contains a number ("Giant 7") still parses.
#
# UNITS ARE PER FIELD, because the app does NOT store what the page prints:
# `ltv` and `debt_yield` are FRACTIONS on the Loan subtab (debt / valuation)
# with the x100 in the display layer, `pct_of_pref` likewise, and DSCR prints
# "2.0x". Converting per field is the whole reason this is a table and not a
# loop over `money()`.
#
# A PRINTED SENTINEL IS NOT A VALUE. "—", "n/a" and "Dev" are how the page says
# a cell does not apply, and the app has its own logic for that (`pdf_na_cells`,
# the Dev suppression). Writing the text into a numeric field would corrupt
# every total built from it, so sentinels are SKIPPED and listed instead.
_SENTINELS = {"—", "-", "–", "n/a", "N/A", "na", "Dev", "dev", ""}


def _num(tok):
    """A leading number, sign-before-or-after-$ tolerant. See `money`."""
    m = re.match(r"^(-)?\$?(-)?([\d,]+(?:\.\d+)?)", str(tok).strip())
    if not m:
        return None
    v = float(m.group(3).replace(",", ""))
    return -v if (m.group(1) or m.group(2)) else v


def _pct_frac(tok):
    """"66.5%" -> 0.665. The app stores these as fractions."""
    v = pct(tok)
    return None if v is None else v / 100.0


def _pct_points(tok):
    """"92.1%" -> 92.1. Occupancy is stored in percentage points."""
    return pct(tok)


def _dscr(tok):
    m = re.match(r"^(-?[\d.]+)\s*x$", str(tok).strip(), re.I)
    return float(m.group(1)) if m else None


def _millions(tok):
    """The Snapshot tables are headed "$ millions" — 33.5 means 33,500,000."""
    v = _num(tok)
    return None if v is None else v * 1e6


#: page -> (subtab, [(field, parser)]) in printed column order.
SNAPSHOT_COLS = {
    6: ("financial", [
        ("debt", _millions), ("total_pref", _millions), ("ptr_equity", _millions),
        ("total_cap", _millions), ("pct_of_pref", _pct_frac),
        ("invested", _millions), ("unfunded", _millions),
        ("total_commitment", _millions), ("itd", _millions),
        ("net_roe", _pct_frac),
    ]),
    7: ("operating", [
        ("econ_occ", _pct_points), ("noi.at_close", _millions),
        ("noi.uw_ye", _millions), ("noi.projected_ye", _millions),
        ("expected_growth", _pct_frac), ("actual_growth", _pct_frac),
    ]),
    8: ("loan", [
        ("rate", None), ("maturity", None), ("debt", _millions),
        ("ytd_dscr", _dscr), ("ltv", _pct_frac), ("debt_yield", _pct_frac),
    ]),
}

#: Rows that are headers, not data.
_HEADER_ROWS = {"individual investments", "proprietary & confidential"}


#: Words that belong to a RATE, not to a comment. "5.6% fixed" and
#: "SOFR + 250" are cell content on the Loan page.
_RATE_WORDS = {"fixed", "floating", "sofr", "+", "libor", "prime", "var",
               "variable"}


def _cellish(t: str) -> bool:
    return (_num(t) is not None or pct(t) is not None
            or str(t).strip() in _SENTINELS
            or bool(_DATE.match(t)) or str(t).lower() in _RATE_WORDS)


def _split_row(line: str, ncols: int) -> tuple:
    """(label, [cells], comment) for one printed table row.

    TAKEN FROM THE RIGHT, NOT THE LEFT, AND THIS IS THE WHOLE POINT. Scanning
    left to right for "the first token that looks like a cell" breaks on every
    row label containing a number: "Giant 7" became label "Giant" with 7 read
    as Debt, and "Total PSC TGA 2022 LLC" became "Total PSC TGA" — which
    collided four ways on one page, so three of the four fund subtotals
    overwrote each other and were lost. Both bugs were invisible in a spot
    check of a row whose name has no digits.

    So the COLUMN COUNT is the anchor: the table has a known width, the comment
    is prose at the end, and the label is whatever is left at the front.
    """
    toks = line.split()
    if not toks:
        return None, [], ""

    # Where does the free-text comment start? The first word-like token that
    # appears AFTER at least one cell — so a label's own words are not mistaken
    # for a comment, and a rate's "fixed" is not either.
    seen_cell = False
    cut = len(toks)
    for i, t in enumerate(toks):
        if _cellish(t):
            seen_cell = True
            continue
        if seen_cell and re.match(r"^[A-Za-z]", t):
            cut = i
            break
    region = toks[:cut]
    comment = " ".join(toks[cut:])

    cells = [t for t in region if _cellish(t)]
    if len(cells) < 2:
        return None, [], ""
    # The last `ncols` cell-like tokens are the data; everything before the
    # first of them is the label.
    take = cells[-ncols:] if ncols and len(cells) > ncols else cells
    first = region.index(take[0]) if take else len(region)
    # `index` finds the first occurrence, which can be too early when a label
    # token equals a cell token; walk back from the end instead.
    first = len(region) - len(take)
    label = " ".join(region[:first]).strip()
    if not label:
        return None, [], ""
    return label, take, comment


_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


def _split_loan_row(label: str, rest: list) -> list:
    """Loan-page cells, anchored on the MATURITY DATE.

    THE RATE IS NOT ONE TOKEN. It prints as "5.6% fixed" (two) or
    "SOFR + 250" (three), so walking the columns left to right shifts
    everything after it — `debt_yield` was being handed LTV's number, which is
    a real figure landing in the wrong field and is invisible once stored.
    Found by checking the cell counts rather than trusting them.

    The date is the one unambiguous landmark: rate is everything before it,
    and the four numerics follow it. A subtotal row prints no rate or maturity
    at all, so with no date the row starts at `debt`.
    """
    dates = [i for i, t in enumerate(rest) if _DATE.match(t)]
    if dates:
        # FIRST date to LAST date, because a two-loan row prints BOTH:
        # Brainerd is "SOFR + 650 | 4.2% fixed   12/1/2026 | 7/1/2027", and
        # anchoring on the first date alone left the pipe and the second date
        # to be read as Debt.
        f, l = dates[0], dates[-1]
        return [" ".join(rest[:f]), " ".join(rest[f:l + 1])] + rest[l + 1:]

    # NO DATE MEANS ONE OF TWO DIFFERENT ROWS, and guessing wrong shifts every
    # column. A sold deal prints a dash in ALL SIX columns — East Manchester is
    # "— — — — — —", no date anywhere — while a subtotal prints no rate or
    # maturity at all and starts at `debt`. The first token tells them apart: a
    # sentinel means the row is dashed across, a number means it starts at debt.
    if rest and str(rest[0]).strip() in _SENTINELS:
        return rest
    return ["", ""] + rest              # subtotal/total: no rate, no maturity


def snapshot_rows(pdf_path: str, page: int, known: list = None) -> dict:
    """{row label: [printed cells]} for one Snapshot page.

    ``known`` is the canonical label list, taken from page 6. The Loan page's
    rate is variable width ("5.6% fixed", "SOFR + 650 | 4.2% fixed"), so there
    is no left-hand anchor that survives it — "Giant 7" lost its 7 and "Mount
    Prospect Plaza 5.3%" swallowed its rate. The same deals appear on all three
    pages, and page 6 is a fixed 10-wide grid that parses cleanly, so its
    labels are the evidence for where the label ends here.
    """
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page - 1].extract_text() or ""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.lower() in _HEADER_ROWS or s.isdigit():
            continue
        # The Loan page's rate is multi-token, so its numeric tail is 4 wide
        # (debt, dscr, ltv, debt yield) and rate/maturity are recovered from
        # the date anchor inside what is left.
        ncols = {6: 10, 7: 6, 8: 6}.get(page, 0)
        label = cells = None
        if known:
            # Longest canonical label that prefixes this line wins.
            hit = max((k for k in known if s.startswith(k)), key=len, default=None)
            if hit:
                label = hit
                # STOP AT THE COMMENT. Taking every cell-like token would pull
                # figures out of the prose — "sale now expected 11/2026" is a
                # date, "phase I & II" has numbers — which widened the row past
                # the table and made the shape guard drop it. Nine of the forty
                # operating rows were being lost that way. The table columns
                # come first; the comment begins at the first word-like token
                # after at least one cell.
                rest = s[len(hit):].split()
                cells, seen = [], False
                for t in rest:
                    if _cellish(t) or t == "|":
                        cells.append(t)
                        seen = True
                        continue
                    if seen:
                        break            # the comment starts here
                    break
        if label is None:
            label, cells, _ = _split_row(s, 0 if page == 8 else ncols)
        if not label or not cells or len(cells) < 2:
            continue
        if page == 8:
            cells = _split_loan_row(label, cells)
        out[label] = cells
    return out


def snapshot_cells(pdf_path: str, pages: list, pdf_name: str) -> tuple:
    """({subtab: {row label: {field: spec}}}, [skipped sentinel cells])."""
    by_subtab, skipped = {}, []
    # Page 6 first: its labels are canonical for the other two.
    canonical = sorted(snapshot_rows(pdf_path, 6), key=len, reverse=True)         if 6 in (pages or []) else []
    for pno in pages or []:
        spec = SNAPSHOT_COLS.get(pno)
        if not spec:
            continue
        subtab, cols = spec
        rows = snapshot_rows(pdf_path, pno, known=canonical)
        dest = by_subtab.setdefault(subtab, {})
        for label, cells in rows.items():
            cmap = {}
            for i, (field, parser) in enumerate(cols):
                if i >= len(cells):
                    break
                tok = cells[i]
                if not str(tok).strip():
                    continue          # an empty column is not a printed cell
                if str(tok).strip() in _SENTINELS:
                    # A SENTINEL IS STILL WHAT THE PAGE PRINTED. East Manchester
                    # prints "—" for rate and maturity while the app holds
                    # 3.65% / 1/11/2031, so skipping the cell leaves the frozen
                    # page showing a live figure that was never sent — the exact
                    # failure freezing exists to prevent.
                    #
                    # Kept as DISPLAY text, never as a number: writing "—" into
                    # `debt` would corrupt every subtotal built from it. The
                    # numeric field is left alone and the printed dash is what
                    # renders. Where the live value is blank too this is a
                    # harmless no-op, which is why all of them are stored rather
                    # than only the ones that differ — deciding that needs live
                    # data, and getting it wrong silently publishes the wrong
                    # cell.
                    cmap[field] = {"published": None, "display": str(tok).strip(),
                                   "units": "printed-sentinel", "page": pno,
                                   "source": pdf_name, "printed": str(tok)}
                    skipped.append({"page": pno, "row": label, "field": field,
                                    "printed": tok,
                                    "why": "printed as not-applicable — kept as "
                                           "display text"})
                    continue
                if parser is None:
                    # RATE and MATURITY print as text ("5.6% fixed",
                    # "SOFR + 250", "7/1/2031"). The Loan subtab composes its
                    # own terms string, so these are kept as printed text
                    # rather than forced into a number.
                    cmap[field] = {"published": None, "display": str(tok),
                                   "units": "printed", "page": pno,
                                   "source": pdf_name, "printed": str(tok)}
                    continue
                v = parser(tok)
                if v is None:
                    skipped.append({"page": pno, "row": label, "field": field,
                                    "printed": tok, "why": "did not parse"})
                    continue
                cmap[field] = {"published": v, "page": pno, "source": pdf_name,
                               "printed": str(tok)}
            # A ROW THAT IS NOT THE TABLE'S WIDTH IS NOT MAPPED POSITIONALLY.
            # Page 6 prints "Excluding Development Deals = $325.9 $45.79M 5.5%"
            # — three figures, and they are NOT debt/pref/equity. Assigning
            # them by position would publish $325.9M as Debt. Skipped and
            # listed instead; pages 6 and 7 are fixed grids, page 8 pads its
            # missing rate/maturity itself.
            if pno in (6, 7) and len(cells) != len(cols):
                skipped.append({"page": pno, "row": label, "field": "(row)",
                                "printed": " ".join(map(str, cells)),
                                "why": f"row has {len(cells)} cells, the table "
                                       f"is {len(cols)} wide — not mapped"})
                continue
            if cmap:
                dest[label] = cmap
    return by_subtab, skipped


def build(only: str = None) -> dict:
    doc = {"quarter": QUARTER, "generated_by": os.environ.get("USERNAME") or "?",
           "investors": {}}
    for investor, spec in SOURCES.items():
        if only and investor != only.upper():
            continue
        if not os.path.exists(spec["pdf"]):
            print(f"!! {investor}: {spec['pdf']} not found — skipped")
            continue
        digest = sha256(spec["pdf"])
        pages = one_pager_pages(spec["pdf"], spec["one_pager_pages"])
        pdf_name = os.path.basename(spec["pdf"])

        reports, unmapped, display_cells, n_cells = {}, [], 0, 0
        for title, cells in pages.items():
            cmap, seen = cells_for_page(cells, pdf_name)
            reports[title] = cmap
            unmapped.extend([dict(u, deal=title) for u in seen])
            n_cells += len(cmap)
            display_cells += sum(1 for c in cmap.values() if c.get("display"))

        snap, snap_skipped = snapshot_cells(
            spec["pdf"], spec["snapshot_pages"], pdf_name)
        snap_cells = sum(len(c) for rows in snap.values() for c in rows.values())

        doc["investors"][investor] = {
            "snapshot": snap,
            "snapshot_skipped": snap_skipped,
            "source": {"file": pdf_name, "sha256": digest,
                       "one_pager_pages": spec["one_pager_pages"],
                       "snapshot_pages": spec["snapshot_pages"]},
            # PRINTED ORDER — the batch print serves this, so the order is data.
            "roster_titles": list(pages),
            "reports": reports,
            "unmapped_labels": unmapped,
        }
        print(f"{investor}: {len(pages)} One Pagers, {n_cells} cells "
              f"({display_cells} printed-units), {len(unmapped)} unmapped "
              f"label(s), sha256 {digest[:12]}…")
        for sub, rows in sorted(snap.items()):
            pg = next((p for p, (sb, _) in SNAPSHOT_COLS.items() if sb == sub), "?")
            print(f"    snapshot p{pg} {sub:<10} {len(rows):>3} rows, "
                  f"{sum(len(c) for c in rows.values()):>4} cells")
        if snap_skipped:
            print(f"    snapshot sentinel cells kept as display text: {len(snap_skipped)}")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="overlay_26q2.json")
    ap.add_argument("--investor", default=None)
    args = ap.parse_args()
    doc = build(args.investor)
    if not doc["investors"]:
        print("nothing built — no source PDF was readable")
        return 1
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}  "
          f"({sum(len(v['reports']) for v in doc['investors'].values())} reports)")
    print("This file is gitignored: derived from documents not in the repo, "
          "and it carries investor figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
