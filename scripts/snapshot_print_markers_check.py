"""Guardrail: the print document carries no working annotations, and keeps the
footnote markers.

The print route (`/portfolio-snapshot/print`) suppresses the analyst-facing
markers with CSS in ``PortfolioSnapshotPrintView.vue`` rather than by branching
the subtab components, which means the suppression list and the markers it has
to cover live in different files and can silently drift apart. This reads both
and fails when they disagree:

  * every inline marker a subtab renders next to a deal's name or figure
    (``.tag`` and its variants, ``.warn-dot`` "!", ``.star`` "*") must be in the
    print view's hide list;
  * a literal "?" marker must not exist anywhere — the work order calls for its
    removal, and this fails if one is ever introduced;
  * ``.fnmark``, the footnote marker on a property name, must NOT be hidden: it
    is part of the published document, not an annotation, and it sits in the
    same cell as the markers that are.

Static: reads the .vue sources, renders nothing, needs no browser, no server
and no database. It is a wiring check — it proves the rules exist and cover
what they must, NOT what a PDF looks like. Re-render the PDF for that.

    python scripts/snapshot_print_markers_check.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "vue_app", "src", "components", "snapshot")
PRINT_VIEW = os.path.join(ROOT, "vue_app", "src", "views",
                          "PortfolioSnapshotPrintView.vue")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: Subtabs that appear on the printed document.
SUBTABS = ("SnapshotFinancial.vue", "SnapshotOperating.vue",
           "SnapshotLoan.vue", "SnapshotSummary.vue")

#: Base classes that mark a working annotation rather than document content.
#: A variant (`tag new`, `tag alt`) inherits its base class, so covering the
#: base covers the variant — which is what the print CSS relies on.
ANNOTATION_BASES = ("tag", "warn-dot", "star")

#: Classes that look like annotations but ARE document content.
MUST_SURVIVE = ("fnmark",)

CHECKS: list = []


def chk(label, ok):
    CHECKS.append((label, bool(ok)))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    return bool(ok)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def print_block(src: str) -> str:
    """The @media print block of the print view, comments stripped.

    Stripping matters: a comment sits between the previous rule's ``}`` and the
    next selector, so a class NAMED in prose ("NOT hidden: .fnmark") would
    otherwise be read as part of that selector and counted as suppressed.
    """
    i = src.find("@media print")
    if i < 0:
        return ""
    return re.sub(r"/\*.*?\*/", " ", src[i:], flags=re.S)


def main() -> int:
    printed = read(PRINT_VIEW)
    block = print_block(printed)
    if not block:
        print("no @media print block found in PortfolioSnapshotPrintView.vue")
        return 2

    # Every class named anywhere in the SELECTOR LIST of a rule whose body
    # hides. A rule like `:deep(.tag), :deep(.warn-dot), :deep(.star) { ... }`
    # hides all three, so the whole selector list has to be read, not just the
    # first name in it — reading only the first is how a marker gets reported
    # as unsuppressed when it is in fact covered.
    hidden = set()
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", block):
        selector, body = m.group(1), m.group(2)
        if not re.search(r"display:\s*none|visibility:\s*hidden", body):
            continue
        hidden.update(re.findall(r"\.([A-Za-z0-9_-]+)", selector))
    print(f"  print view hides: {', '.join(sorted(hidden))}\n")

    print("  ANNOTATION CLASSES RENDERED BY EACH SUBTAB")
    found: dict = {}
    for name in SUBTABS:
        path = os.path.join(SNAP, name)
        if not os.path.exists(path):
            continue
        src = read(path)
        tmpl = src[src.find("<template>"):src.find("</template>")]
        for base in ANNOTATION_BASES:
            if re.search(r'class="[^"]*\b' + re.escape(base) + r'\b', tmpl):
                found.setdefault(base, []).append(name)
    for base, files in sorted(found.items()):
        print(f"    .{base:<10} {', '.join(f[8:-4] for f in files)}")

    print()
    for base in sorted(found):
        chk(f".{base} — rendered on a row, suppressed in print",
            base in hidden)

    # A "?" marker: the work order removes it. It does not exist today; this
    # fails if one is introduced without also being suppressed.
    q_hits = []
    for name in SUBTABS:
        path = os.path.join(SNAP, name)
        if not os.path.exists(path):
            continue
        src = read(path)
        tmpl = src[src.find("<template>"):src.find("</template>")]
        for m in re.finditer(r">\s*\?\s*<", tmpl):
            q_hits.append((name, m.start()))
    chk('no literal "?" marker is rendered on any subtab', not q_hits)
    for n, pos in q_hits:
        print(f"        found in {n} at offset {pos}")

    for cls in MUST_SURVIVE:
        chk(f".{cls} survives print — it is document content, not an "
            f"annotation", cls not in hidden)

    # The footnote marker must actually be rendered somewhere, or "it survives
    # print" is vacuously true.
    fin = read(os.path.join(SNAP, "SnapshotFinancial.vue"))
    chk("the property footnote marker is rendered on the property name",
        'class="fnmark"' in fin and "dealMark(r.vcode)" in fin)
    chk("column footnote markers are rendered on the column headers",
        fin.count("colMark(") >= 10)
    chk("no hardcoded standing-footnote list remains in the component "
        "(it is declared once, in the backend)",
        not re.search(r"const\s+STANDING_FOOTNOTES", fin))
    chk("the removed footnote's wording is gone from the component",
        "depressed ROEs" not in fin)

    passed = sum(1 for _, c in CHECKS if c)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    for label, ok in CHECKS:
        if not ok:
            print(f"    FAILED: {label}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
