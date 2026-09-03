"""Guardrail: exactly ONE ``@page`` rule exists in the whole front end.

``@page`` cannot be scoped. Vue's scoped CSS appends a ``[data-v-hash]``
attribute to selectors, and an at-rule that styles the page box has no selector
to append to — so a ``@page`` written inside any component's ``<style scoped>``
escapes into the global sheet and applies to every route. Routes are
lazy-loaded and Vite leaves a route's stylesheet in the document after you
navigate away, so with more than one ``@page`` in the build the winner is
decided by whichever route the user happened to visit last.

That is how ``830934d`` (Aug 24 2026, "Fix: Portfolio Snapshot formatting and
print styles") silently changed the One Pager's printed output. Its
``@page { margin: 0.5in }`` beat the One Pager's ``margin: 0`` whenever the
Snapshot had been visited first — growing the side margins and, because a
non-zero page margin is what gives Chrome room to draw its own header and
footer, putting the "Waterfall XIRR" title back on the page. Nothing in
OnePagerView.vue had been touched since July.

    python scripts/print_page_rule_check.py

Checks the SOURCE always, and the BUILD too when ``vue_app/dist`` is present
(the build is the real proof — it is what the browser loads).
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "vue_app", "src")
DIST = os.path.join(ROOT, "vue_app", "dist", "assets")

#: The one file allowed to declare the page box.
OWNER = os.path.join("vue_app", "src", "App.vue")

#: Every view that prints, and the container whose padding IS its margin now
#: that the page box is global and zero. A view that loses both would print
#: edge-to-edge, so each is asserted rather than assumed.
PRINT_VIEWS = {
    "OnePagerView": "one-pager-page",
    "PortfolioSnapshotView": "snapshot",
    "PortfolioSnapshotPrintView": "print-page",
    "LeaseAbstractView": "lease-abstract-page",
}

CHECKS: list = []


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


def _strip_comments(css: str) -> str:
    """Drop /* ... */ so a comment ABOUT @page is not counted as one."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _page_rules(text: str):
    """Every real ``@page`` at-rule in ``text``, as written."""
    return re.findall(r"@page\s*\{[^}]*\}|@page\s*\{", _strip_comments(text))


def main() -> int:
    print("\n1. The source declares the page box exactly once")
    found = {}
    for path in glob.glob(os.path.join(SRC, "**", "*.vue"), recursive=True):
        rules = _page_rules(open(path, encoding="utf-8").read())
        if rules:
            found[os.path.relpath(path, ROOT)] = rules

    chk("only one file declares @page", len(found) == 1,
        "declared in: " + ", ".join(sorted(found)) if len(found) != 1
        else "")
    chk("and it is App.vue, the only always-loaded stylesheet",
        list(found) == [OWNER],
        "found in " + str(list(found)) + " — a component's <style scoped> "
        "cannot contain @page; move it here and use container padding")
    if found:
        rule = " ".join(next(iter(found.values()))[0].split())
        chk("it zeroes the margin, which is what suppresses the browser's "
            "own header/footer", "margin:0" in rule.replace(" ", ""), rule)
        chk("and pins the page size", "letter" in rule, rule)

    print("\n2. Every print view supplies its own margin as padding")
    # Without this, removing a view's @page would print it edge-to-edge.
    for view, container in PRINT_VIEWS.items():
        hits = glob.glob(os.path.join(SRC, "**", view + ".vue"), recursive=True)
        if not hits:
            chk(view + " exists", False)
            continue
        css = open(hits[0], encoding="utf-8").read()
        block = css.split("@media print", 1)[-1]
        pat = re.compile(r"\." + re.escape(container)
                         + r"\s*\{[^}]*padding\s*:\s*([^;}]+)", re.S)
        m = pat.search(block)
        chk("{:<26} .{} sets its own padding".format(view, container),
            m is not None and "0" != m.group(1).strip(),
            (m.group(1).strip() if m else "no padding found — this view would "
             "print edge-to-edge now that the page box is zero"))

    print("\n3. The BUILD agrees — this is what the browser loads")
    css_files = sorted(glob.glob(os.path.join(DIST, "*.css")))
    if not css_files:
        print("      (no build present — run `npm run build` in vue_app to "
              "check the emitted CSS as well)")
    else:
        built = {}
        for path in css_files:
            rules = _page_rules(open(path, encoding="utf-8").read())
            if rules:
                built[os.path.basename(path)] = rules
        chk("exactly one @page survives into the bundle", len(built) == 1,
            "emitted by: " + ", ".join(sorted(built)))
        chk("in the always-loaded index chunk, not a lazy route chunk",
            list(built) and list(built)[0].startswith("index-"),
            str(list(built)) + " — a route chunk's rule applies only after "
            "that route is visited, which is what made this order-dependent")
        if built:
            rule = next(iter(built.values()))[0].replace(" ", "")
            chk("and it is the zero-margin page box",
                "margin:0" in rule, rule)

    passed = sum(CHECKS)
    print("\n  {}/{} checks passed".format(passed, len(CHECKS)))
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
