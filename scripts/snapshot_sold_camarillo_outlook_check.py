"""Guardrail: every sold deal the page CLAIMS to carry is actually on it.

THE DEFECT. Footnote 8 on the live 26Q2 Financial subtab — analyst-entered,
anchored to the Total Commitment column — reads:

    "Invested and Total Commitment on this page include City West, East
     Manchester, Camarillo Village Square and Outlook Nine Mile which have
     been sold."

It named four deals. The page carried TWO. Camarillo Village (PCAMARI, sold
2/10/2023) and Outlook Nine Mile (POUTLOO, sold 4/26/2023) were dropped by
``is_sold_as_of`` and were in no row, no subtotal and no total. The page
asserted a population it did not have, in a footnote hanging off the very
column the missing dollars belonged in.

Measured live 2026-09-15 at 26Q2, before the fix:
    Individual Investments invested   68,229,340.07   (2 sold deals present)
    Portfolio Totals       invested  422,570,136.78
    every subtotal footed EXACTLY to the sum of its visible rows, delta 0.00

That last line is the proof there was no double count to worry about: had the
two deals' dollars been reaching the totals by some other route — a manually
entered subtotal, say — the subtotals would have exceeded their rows. They did
not, so the footnote was simply false as to those two, and adding the rows is
purely additive.

THE FIX. Both vcodes join ``KEEP_DESPITE_SOLD``. Every downstream rule already
keys on the derived ``kept_despite_sold`` flag rather than on a vcode, so they
inherit the treatment whole.

ROE FOLLOWS EAST MANCHESTER, NOT CITY WEST (corrected Sep 16 2026). It followed
City West for one day. City West was FORECLOSED — no realised return, and the
reference PDF excludes it by name — while these two are ordinary sales whose
realised return is the reason the rows are kept at all. Their ``PDF_NA_CELLS``
entries are gone, so ``net_roe`` and ``itd`` are enterable, and the
ROE-exclusion footnote is back to City West alone. Leaving them in that
footnote while their ROE cell prompts for entry is the East Manchester
contradiction of Sep 2 2026, which ``PDF_NA_CELLS`` has now caused twice.

THE VCODE TRAP THIS SCRIPT ALSO PINS. Each of these deals has TWO roster rows:
a deal-level one (PCAMARI / POUTLOO / PCITWES) carrying InvestmentID,
Sale_Status and Sale_Date, and a property-level one (P0000009 / P0000034 /
P0000011) carrying NONE of them. ``KEEP_DESPITE_SOLD`` is only ever consulted
inside the ``is_sold_as_of`` branch, so keying it on a property-level vcode —
which is never sold, because it has no Sale_Status — matches nothing and raises
nothing. The deals just stay missing. ``scripts/noi_changed_quarters.py``
carries exactly that property-level map.

    WF_TOKEN=... .venv/Scripts/python.exe \
        scripts/snapshot_sold_camarillo_outlook_check.py

Read-only. Live GETs only; asserts against the live page, not a fixture.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import live_api as L  # noqa: E402

INVESTOR = "TGAM"          # TIAA
QUARTER = "2026-Q2"

#: vcode -> (name, the quarter its stack must be read at). The last held
#: quarter is always the one BEFORE the quarter containing the sale.
SOLD = {
    "PCITWES":  ("City West",         "2025-Q2"),
    "P0000017": ("East Manchester",   "2026-Q1"),
    "PCAMARI":  ("Camarillo Village", "2022-Q4"),
    "POUTLOO":  ("Outlook Nine Mile", "2023-Q1"),
}

#: The sold deals whose Net ROE is suppressed, i.e. that belong in the
#: ROE-exclusion footnote. ONLY the foreclosure.
#:
#: Camarillo Village and Outlook Nine Mile were here from Sep 15 2026 and came
#: out again Sep 16. Being kept on the page after a sale is not what suppresses
#: a ROE — having no realised return is, and only City West qualifies. The two
#: of them are ordinary sales on the East Manchester footing, which is also
#: deliberately not here and never was.
ROE_EXCLUDED = {"PCITWES"}

#: Property-level twins that must never be used as keys. See the module note.
DECOYS = {"P0000009": "Camarillo Village", "P0000034": "Outlook Nine Mile",
          "P0000011": "City West"}

CHECKS: list = []


def chk(label: str, ok, note: str = "") -> bool:
    CHECKS.append((bool(ok), label, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {note}" if note else ""))
    return bool(ok)


def flatten(page: dict) -> dict:
    return {r["vcode"]: (gk, r)
            for gk, g in page["groups"].items() for r in g["deals"]}


def main() -> int:
    print(f"live {INVESTOR} @ {QUARTER}\n")
    page = L.get("/api/portfolio-snapshot/financial",
                 {"investor": INVESTOR, "quarter": QUARTER})
    flat = flatten(page)

    print("1. Every sold deal the footnote names is ON THE PAGE")
    note8 = next((f for f in page["footnotes"]
                  if "which have been sold" in (f.get("text") or "")), None)
    chk("the 'includes sold deals' footnote is still there", bool(note8))
    for vc, (name, _) in SOLD.items():
        on = vc in flat
        named = bool(note8) and name.split(" Square")[0] in note8["text"]
        chk(f"{name} ({vc}) is on the page", on)
        if named:
            chk(f"{name} is named by the footnote AND present", named and on,
                "a footnote naming a deal the page does not carry is false")

    print("\n2. They carry the SOLD treatment, identically")
    for vc, (name, want_q) in SOLD.items():
        if vc not in flat:
            continue
        gk, r = flat[vc]
        chk(f"{name}: kept_despite_sold", r.get("kept_despite_sold") is True)
        chk(f"{name}: labelled '(Sold)'", r.get("sold_label") == "(Sold)")
        chk(f"{name}: stack read at {want_q}", r.get("stack_quarter") == want_q,
            f"got {r.get('stack_quarter')!r}")
        chk(f"{name}: Debt is n/a", "debt" in set(r.get("pdf_na_cells") or ()))
        chk(f"{name}: its debt is OUT of the debt subtotal",
            not (r.get("debt_summable") or 0))

    print("\n3. The rebase reproduces the historical quarter EXACTLY")
    # The whole point of last_held_quarter: the row must show the stack the
    # deal really had, not a post-sale netting to zero.
    for vc, (name, q) in SOLD.items():
        if vc not in flat:
            continue
        hist = flatten(L.get("/api/portfolio-snapshot/financial",
                             {"investor": INVESTOR, "quarter": q}))
        h = (hist.get(vc) or (None, {}))[1]
        for f in ("total_pref", "ptr_equity", "invested", "total_commitment"):
            a, b = flat[vc][1].get(f), h.get(f)
            chk(f"{name}: {f} matches its own {q} page",
                a is not None and b is not None and abs(a - b) < 0.01,
                f"{a} vs {b}")
        chk(f"{name}: equity did NOT net to zero",
            (flat[vc][1].get("total_pref") or 0)
            + (flat[vc][1].get("ptr_equity") or 0) > 0,
            "a sold row netted to zero loses the capital its ROE is a return on")

    print("\n4. Subtotals and Portfolio Totals foot to the rows")
    for gk, g in page["groups"].items():
        for f in ("total_pref", "ptr_equity", "invested", "total_commitment"):
            s = sum(r.get(f) or 0 for r in g["deals"])
            st = (g.get("subtotal") or {}).get(f)
            if st is None:
                continue
            chk(f"{gk}: {f} subtotal == sum(rows)", abs(s - st) < 0.01,
                f"{s:,.2f} vs {st:,.2f}")
    for f in ("total_pref", "ptr_equity", "invested", "total_commitment"):
        s = sum(r.get(f) or 0 for _, r in flat.values())
        t = (page.get("total") or {}).get(f)
        chk(f"Portfolio Totals: {f} == sum(all rows)",
            t is not None and abs(s - t) < 0.01, f"{s:,.2f} vs {t:,.2f}")

    print("\n5. The ROE-exclusion footnote matches the suppressed cells")
    roe = next((f for f in page["footnotes"]
                if "excluded from ROE" in (f.get("text") or "")), None)
    chk("there is exactly one ROE-exclusion footnote",
        sum(1 for f in page["footnotes"]
            if "excluded from ROE" in (f.get("text") or "")) == 1)
    marks = (page.get("footnote_marks") or {}).get("property") or {}
    for vc, (name, _) in SOLD.items():
        if vc not in flat:
            continue
        suppressed = "net_roe" in set(flat[vc][1].get("pdf_na_cells") or ())
        marked = bool(roe) and marks.get(vc) == [roe["number"]]
        chk(f"{name}: Net ROE suppressed == named by the footnote",
            suppressed == marked and suppressed == (vc in ROE_EXCLUDED),
            f"suppressed={suppressed} marked={marked}")

    print("\n6. The property-level decoy vcodes are NOT what got used")
    for vc, name in DECOYS.items():
        chk(f"{vc} ({name}, property-level twin) is not a page row",
            vc not in flat,
            "keying KEEP_DESPITE_SOLD on this would match nothing, silently")

    bad = [c for c in CHECKS if not c[0]]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    for _, label, note in bad:
        print(f"  FAILED: {label}" + (f"  — {note}" if note else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except L.TokenExpired as exc:
        print(f"token rejected: {exc}")
        sys.exit(2)
