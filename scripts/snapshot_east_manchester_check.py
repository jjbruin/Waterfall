"""Guardrail: a sold deal we still measure keeps a typeable Net ROE.

East Manchester (P0000017) is kept on the report after its 6/25/2026 sale
because its ROE and ITD distributions are tracked — that is the reason the row
is there. ``SOLD_NA_CELLS`` used to blank ``net_roe`` on any kept-despite-sold
row, which made the cell read-only and left no way to enter the figure the row
exists to show.

The rule now blanks only ``debt``, which the sale genuinely invalidates (the
loan left with the asset). City West keeps its n/a Net ROE through its own
static ``PDF_NA_CELLS`` entry — it was foreclosed, which is a different thing
from a sale, and it really is out of the ROE numbers.

    python scripts/snapshot_east_manchester_check.py

Runs the real ``assemble_financial`` at 26Q2. The local snapshot carries
Sale_Date 12/01/2030 for East Manchester where live has 6/25/2026, so the sold
gate never fires here; the check injects the live date, which is the only way
to render the row as the report author sees it.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKS: list = []


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


def main() -> int:
    from flask_app import create_app
    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        from flask_app.services.portfolio_snapshot_freeze import build_subtab
        from flask_app.services.portfolio_snapshot_service import (
            resolve_investor_deals, KEEP_DESPITE_SOLD,
        )
        from flask_app.services.portfolio_snapshot_financial import (
            SOLD_NA_CELLS, PDF_NA_CELLS, STANDING_FOOTNOTES,
        )

        d = dict(data_service.get_data())
        inv = d["inv"].copy()
        inv.loc[inv["vcode"].astype(str).str.upper() == "P0000017",
                "Sale_Date"] = "6/25/2026"
        d["inv"] = inv
        resolved = resolve_investor_deals("TGAM", "2026-Q2",
                                          d.get("relationships_raw"), inv)
        fin = build_subtab("financial", "TGAM", "2026-Q2", d, resolved)
        rows = {r["vcode"].upper(): r
                for b in (fin.get("groups") or {}).values() for r in b["deals"]}
        rows.update({r["vcode"].upper(): r
                     for r in (fin.get("ownership_flagged") or [])})

        em = rows.get("P0000017")
        cw = rows.get("PCITWES")

        print("\n1. The rule blanks only what the sale invalidates")
        chk("SOLD_NA_CELLS is debt only", set(SOLD_NA_CELLS) == {"debt"},
            f"{sorted(SOLD_NA_CELLS)} — net_roe was removed Sep 2 2026")
        chk("no per-deal entry was added for East Manchester",
            "P0000017" not in PDF_NA_CELLS,
            "the split is City West's own PDF_NA_CELLS entry, not a new hardcode")

        print("\n2. East Manchester — on the page, measurable")
        chk("is on the 26Q2 page at all", em is not None)
        if em is None:
            return 1
        chk("kept despite sold, with the (Sold) label",
            em.get("kept_despite_sold") is True and em.get("sold_label") == "(Sold)")
        chk("Net ROE is a manual cell, not n/a",
            em.get("net_roe_display") == "pending entry",
            f"{em.get('net_roe_display')!r} — was 'n/a', which rendered read-only")
        chk("and nothing marks it n/a but Debt",
            list(em.get("pdf_na_cells") or []) == ["debt"],
            str(em.get("pdf_na_cells")))
        chk("ITD Distributions is a manual cell too",
            "itd" not in (em.get("pdf_na_cells") or []),
            "shows 'pending entry' until typed, then the figure")
        chk("Debt stays n/a", em.get("debt_display") == "n/a",
            f"raw debt {em.get('debt'):,.0f} preserved underneath")
        chk("Debt stays OUT of the column total",
            em.get("debt_summable") is None)

        print("\n3. East Manchester — the original capital stack shows")
        for field, want in (("total_pref", 3_600_000.0),
                            ("ptr_equity", 2_400_000.0),
                            ("invested", 2_723_400.0),
                            ("total_commitment", 2_723_400.0)):
            got = em.get(field)
            chk(f"{field} = {want:,.0f}",
                got is not None and abs(got - want) < 1.0, f"got {got}")
        chk("% of Pref is a real figure",
            em.get("pct_of_pref") and 0.7 < em["pct_of_pref"] < 0.8,
            f"{em.get('pct_of_pref')}")
        chk("Total Cap foots to what the row PRINTS (pref + ptr, debt is n/a)",
            abs((em.get("total_cap") or 0) - 6_000_000.0) < 1.0,
            f"{em.get('total_cap'):,.0f} = 0 + 3,600,000 + 2,400,000")

        print("\n4. City West — untouched")
        chk("is still on the page", cw is not None)
        if cw is not None:
            chk("Net ROE still n/a", cw.get("net_roe_display") == "n/a",
                "through its own PDF_NA_CELLS entry, not the sale rule")
            chk("Debt still n/a", cw.get("debt_display") == "n/a")
            chk("both cells still marked",
                sorted(cw.get("pdf_na_cells") or []) == ["debt", "net_roe"],
                str(cw.get("pdf_na_cells")))
            chk("still kept despite sold", cw.get("kept_despite_sold") is True)
        chk("both deals are still in KEEP_DESPITE_SOLD",
            {"PCITWES", "P0000017"} <= set(KEEP_DESPITE_SOLD))

        print("\n5. The ROE-exclusion footnote — City West only")
        # Decided Sep 2 2026: East Manchester came out. A page cannot show a
        # Net ROE for a deal and also footnote that the deal is excluded from
        # ROE. City West stays — it was foreclosed, not sold, so there is
        # genuinely no ROE to report.
        from flask_app.services.portfolio_snapshot_financial import (
            compose_footnotes, footnote_marks,
        )
        marks = footnote_marks(compose_footnotes([]))
        note = next((f for f in STANDING_FOOTNOTES
                     if "excluded from ROE" in f["text"]), None)
        print(f"      text:    {note['text'] if note else '(missing)'}")
        print(f"      anchors: {list(note.get('anchors') or ()) if note else []}")
        anchors = [str(x) for x in (note.get("anchors") or ())] if note else []
        chk("East Manchester is NOT anchored to it",
            "deal:P0000017" not in anchors,
            "its Net ROE is typeable and its ITD shows, so the note would "
            "contradict its own row")
        chk("and its name is not in the text either",
            note is not None and "East Manchester" not in note["text"],
            "removing only the anchor would leave a note naming a deal it no "
            "longer marks")
        chk("City West is still anchored — foreclosed, genuinely no ROE",
            "deal:PCITWES" in anchors)
        chk("no marker is placed on East Manchester's property name",
            not (marks["property"] or {}).get("P0000017"),
            str(marks["property"]))
        chk("City West still carries its marker",
            bool((marks["property"] or {}).get("PCITWES")))

    passed = sum(CHECKS)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
