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

#: Labels the PDF prints that map to no vetted path. NOT guessed into one — a
#: wrong path writes a real figure into the wrong field. Emitted as
#: `unmapped_labels` so the admin screen can show that they were seen and
#: deliberately not applied.
UNMAPPED = ["P.E. Coupon", "P.E. Participation"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def money(s):
    m = re.match(r"^\$?(-?[\d,]+(?:\.\d+)?)\s*([MK]?)$", str(s).strip())
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * (1e6 if m.group(2) == "M" else 1e3 if m.group(2) == "K" else 1)


def pct(s):
    m = re.match(r"^(-?[\d.]+)\s*%$", str(s).strip())
    return float(m.group(1)) if m else None


def one_pager_pages(pdf_path: str, pages: list) -> dict:
    """{deal title: {label: printed string}} for each One Pager page."""
    import pdfplumber
    LABELS = list(OP_PATHS) + UNMAPPED
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
                m = re.search(rf"{re.escape(lbl)}:\s*(\$?-?[\d,.]+[MK%]?)", t)
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

    for lbl in UNMAPPED:
        if lbl in cells:
            seen_unmapped.append({"label": lbl, "printed": cells[lbl],
                                  "page": page})

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

        doc["investors"][investor] = {
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
