"""Guardrail: the "(Sold)" label keeps a real space in front of it.

THE DEFECT. The markup read `<span class="sold"> {{ r.sold_label }}</span>` and
that leading space never reached the browser. Vue's template compiler runs in
its default `condense` whitespace mode, which strips whitespace at the start of
an element's children — so the published Portfolio Snapshot printed
"East Manchester(Sold)", and ran the footnote marker straight into the label on
City West. Measured in the real printed PDF, the gap before "(Sold)" was 0.01pt.

WHY THIS CHECKS THE BUILD AND NOT ONLY THE SOURCE. A source-only check is
exactly the check that missed this: the source *did* contain a space. The
stripping happens in the compiler, so the compiled render function is the only
place the truth is visible. If `vue_app/dist` is present this asserts the
character survived into it; without a build it checks the source and says so.

The separator must be `&nbsp;` (U+00A0) rather than a plain space — a plain one
gets stripped again — and rather than a CSS margin, which would draw a gap
without putting a character in the investor PDF's text.

Usage
    cd vue_app && npx vite build        # optional but recommended
    python scripts/snapshot_sold_label_spacing_check.py
"""
import glob
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "vue_app" / "src" / "components" / "snapshot"

#: Every component that renders the label, and how many `.sold` sites it has.
#:
#: THIS USED TO BE SnapshotFinancial ALONE, and that is exactly why the defect
#: it now catches survived: Financial was the only subtab that emitted
#: `sold_label` at all, so a check scoped to Financial passed 6/6 while City
#: West, East Manchester, Camarillo Village and Outlook Nine Mile rendered on
#: Operating and Loan as ordinary rows with nothing marking them sold. A
#: spacing check over one file cannot see a label that is missing from another.
#:
#: Financial has two sites — the deal rows and the ownership-unresolved rows.
#: Operating and Loan have one each; their unresolved rows render through the
#: same `blk.rows` loop, so one site covers both.
#:
#: Summary is deliberately absent: it does not list deals individually, so it
#: has no name cell to label. Add it here the moment that changes.
COMPONENTS = {
    "SnapshotFinancial.vue": 2,
    "SnapshotOperating.vue": 1,
    "SnapshotLoan.vue": 1,
}
DIST = ROOT / "vue_app" / "dist" / "assets"

_checks = []


def chk(label, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if detail:
        print(f"         {detail}")


print("Portfolio Snapshot: '(Sold)' label spacing")
print()

print("source")
for fname, expected in COMPONENTS.items():
    path = SNAP / fname
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Match the TEMPLATE only, with comments removed. Several blocks quote the
    # old `<span class="sold"> {{ ... }}` markup while explaining why it was
    # wrong, and a naive search over the whole file counts those as live sites
    # and fails. print_page_rule_check had the same bug against a comment
    # mentioning @page.
    tmpl = text.split("\n<style", 1)[0]
    tmpl = re.sub(r"<!--.*?-->", "", tmpl, flags=re.S)
    tmpl = re.sub(r"/\*.*?\*/", "", tmpl, flags=re.S)

    sites = re.findall(r'class="sold">(.{0,8}?)\{\{', tmpl)
    chk(f"{fname}: every .sold span was found", len(sites) == expected,
        f"found {len(sites)}, expected {expected}")
    chk(f"{fname}: each one starts with &nbsp;, not a plain space",
        sites and all(s == "&nbsp;" for s in sites), f"prefixes: {sites!r}")
    chk(f"{fname}: no .sold span still relies on a plain leading space",
        ' class="sold"> {{' not in tmpl)

    # The OTHER half of the separator rule, and the one a character alone does
    # not cover. Vue strips whitespace at the START of an element's children —
    # which is why the &nbsp; is needed — but whitespace BETWEEN elements
    # CONDENSES TO ONE SPACE. So a `.sold` span placed on its own line after
    # the name renders TWO separators, the condensed space plus the character.
    # The span must be jammed against `{{ r.name }}` with nothing between.
    chk(f"{fname}: the span is jammed against the name (no double space)",
        not re.search(r"\}\}\s*\n\s*<span[^>]*class=\"sold\"", tmpl),
        "a newline before the span condenses to a space, doubling the gap")

    # A margin would be the other way to draw the gap, and would put no
    # character in the PDF's text layer. If someone adds one later, say so.
    sold_rule = re.search(r"\.sold \{([^}]*)\}", text)
    chk(f"{fname}: the .sold rule adds no margin (the character does the work)",
        sold_rule is not None and "margin" not in sold_rule.group(1),
        (sold_rule.group(1).strip() if sold_rule else "rule not found"))

# The footnote marker is a SUPERSCRIPT reference and sits tight against the
# name — "City West⁽²⁾ (Sold)", not "City West (2) (Sold)". That is how the
# reference document sets it, and 7bc8d5f deliberately left it alone while
# fixing `.sold`. Asserted so a later "fix" to the spacing has to argue with
# this line rather than silently change the published convention.
fin = (SNAP / "SnapshotFinancial.vue").read_text(encoding="utf-8")
fin_tmpl = re.sub(r"<!--.*?-->", "", fin.split("\n<style", 1)[0], flags=re.S)
chk("the footnote marker stays tight against the name (superscript convention)",
    not re.search(r"\}\}\s+<span[^>]*class=\"fnmark\"", fin_tmpl),
    "reference document sets a superscript reference against the word")

print()
print("build")
chunks = [p for p in glob.glob(str(DIST / "*.js"))
          if "sold_label" in Path(p).read_text(encoding="utf-8", errors="replace")]
if not chunks:
    print("  [SKIP] no build at vue_app/dist — run: cd vue_app && npx vite build")
    print("         source checks alone CANNOT catch the compiler stripping the")
    print("         space; this is the half that matters.")
else:
    for c in chunks:
        body = Path(c).read_text(encoding="utf-8", errors="replace")
        found = re.findall(r'sold_label\?\(\w+\(\),\w+\("span",\w+,"(.)"\+', body)
        bare = re.findall(r'sold_label\?\(\w+\(\),\w+\("span",\w+,\w\(', body)
        name = Path(c).name
        chk(f"{name}: every compiled .sold span prepends a character",
            len(found) > 0 and len(bare) == 0,
            f"with separator: {len(found)}, without: {len(bare)}")
        chk(f"{name}: that character is U+00A0 (non-breaking space)",
            all(ch == " " for ch in found),
            "codepoints: " + ", ".join("U+%04X" % ord(ch) for ch in found))

print()
ok = sum(_checks)
print(f"{ok}/{len(_checks)} checks passed")
sys.exit(0 if ok == len(_checks) else 1)
