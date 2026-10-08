"""Guardrail: the printed One Pager is the on-screen One Pager, scaled.

THE DEFECT THIS GUARDS. The print CSS used to restyle the sheet for paper:
12.5-13px text against 10-11px on screen, in a 7.5in column that is 20%
narrower than the 960px screen sheet, inside a fixed-height `overflow: hidden`
page. Text wrapped onto more lines than the author saw while typing, and the
comment and Business Plan blocks -- the only parts that could give -- were cut
off mid-sentence with no marker. On top of that a <textarea> prints its BOX,
not its text, so anything scrolled out of view on screen was never on paper.

THE RULE. Print lays the sheet out at a width W and zooms it by 720/W onto the
printable column, so the layout is the screen's, uniformly scaled. The Print
button's fitSheetsForPrint() picks W per deal (the narrowest, i.e. largest
type, that still fits one page) and spends what is left on the chart and
section spacing; with no fit run (a plain Ctrl+P) the CSS fallbacks are the
screen sheet at 0.75, which fits every deal:

  1. `.op-sheet` under @media print has `width: W` and `zoom: Z` with
     W == the screen `.one-pager-page` max-width, and W * Z == 7.5in (720px).
  2. No per-element font-size inside @media print -- type size comes from the
     zoom alone, so nothing re-wraps differently from its neighbours.
  3. No fixed page height and no `overflow: hidden` on the sheet or the
     Business Plan block under print -- nothing may be clipped to fit.
  4. Every <textarea> is `print-hide` and has a `print-only` twin rendering
     the same v-model.
  5. The fit's widths run narrowest-first, stop at the 8pt floor (740px) and
     end at the screen sheet; PRINT_COLUMN_PX is the 720px column.

Static, by design: rendered evidence comes from scripts/onepager_print_sweep.mjs
+ scripts/onepager_overflow_report.py. This one runs in the pre-commit hook.

NON-VACUOUS: `--file <path>` checks any copy of the view; run it against the
pre-fix file (git show 7194275:vue_app/src/views/OnePagerView.vue) and it must
FAIL. Both directions are asserted by `--self-test`.

Usage
  .venv/Scripts/python.exe scripts/onepager_print_matches_screen_check.py
  .venv/Scripts/python.exe scripts/onepager_print_matches_screen_check.py --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEW = os.path.join(ROOT, "vue_app", "src", "views", "OnePagerView.vue")
PRINT_COLUMN_PX = 720.0  # 8.5in - 2 x 0.5in padding, at 96px/in
PRE_FIX_COMMIT = "7194275"

# Plain-text twin -> the screen class whose font-size it must reproduce.
TWIN_FONT_SOURCE = {
    ".comment-print": ".comment-input",
    ".comment-print.small": ".comment-input.small",
    ".bp-print-text": ".comment-input",
}


def _style(src: str) -> str:
    m = re.search(r"<style[^>]*>(.*?)</style>", src, re.S)
    return m.group(1) if m else ""


def _block_after(css: str, at: int) -> str:
    """The brace-balanced body starting at the first '{' at or after `at`."""
    i = css.index("{", at)
    depth, j = 0, i
    while j < len(css):
        if css[j] == "{":
            depth += 1
        elif css[j] == "}":
            depth -= 1
            if depth == 0:
                return css[i + 1:j]
        j += 1
    raise ValueError("unbalanced braces")


def _rules(css: str) -> list[tuple[str, str]]:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(s.strip(), b) for s, b in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


def _decl(body: str, prop: str) -> str | None:
    """A declaration's value; for `var(--x, fallback)` the fallback, which is
    what prints when fitSheetsForPrint() has not run (a plain Ctrl+P)."""
    m = re.search(rf"(?:^|;)\s*{re.escape(prop)}\s*:\s*([^;]+)", body)
    if not m:
        return None
    v = m.group(1).replace("!important", "").strip()
    fb = re.fullmatch(r"var\(\s*--[\w-]+\s*,\s*([^)]+)\)", v)
    return fb.group(1).strip() if fb else v


def _px(v: str | None) -> float | None:
    if v is None:
        return None
    m = re.fullmatch(r"([\d.]+)px", v)
    return float(m.group(1)) if m else None


def _selectors(sel: str) -> list[str]:
    return [s.strip() for s in sel.split(",")]


def _screen_value(screen_rules, selector: str, prop: str) -> str | None:
    for sel, body in screen_rules:
        if selector in _selectors(sel):
            v = _decl(body, prop)
            if v is not None:
                return v
    return None


def check(src: str) -> list[str]:
    fails: list[str] = []
    css = re.sub(r"/\*.*?\*/", "", _style(src), flags=re.S)
    m = re.search(r"@media\s+print", css)
    if not m:
        return ["no @media print block"]
    print_body = _block_after(css, m.start())
    screen_css = css[:m.start()] + css[m.start() + len(print_body):]
    screen_rules = _rules(re.sub(r"@media[^{]*\{", "", screen_css))
    print_rules = _rules(print_body)

    # 1. width x zoom == the printable column, at the screen width
    screen_w = _px(_screen_value(screen_rules, ".one-pager-page", "max-width"))
    sheet = [b for s, b in print_rules if ".op-sheet" in _selectors(s)]
    w = next((_px(_decl(b, "width")) for b in sheet if _decl(b, "width")), None)
    z = next((_decl(b, "zoom") for b in sheet if _decl(b, "zoom")), None)
    if w is None or z is None:
        fails.append(".op-sheet under print has no width/zoom pair "
                     f"(width={w}, zoom={z}) -- print is not the screen scaled")
    else:
        if screen_w is not None and w != screen_w:
            fails.append(f"print .op-sheet width {w}px != screen sheet {screen_w}px")
        if abs(w * float(z) - PRINT_COLUMN_PX) > 0.5:
            fails.append(f"{w}px x zoom {z} = {w * float(z):.1f}px, not the "
                         f"{PRINT_COLUMN_PX:.0f}px printable column")

    # 2. no print-only font sizes, except twins that match their textarea
    for sel, body in print_rules:
        fs = _decl(body, "font-size")
        if fs is None:
            continue
        for s in _selectors(sel):
            src_sel = TWIN_FONT_SOURCE.get(s)
            if src_sel is None:
                fails.append(f"print font-size on `{s}` ({fs}) -- print must "
                             "keep the screen's type size")
                continue
            want = _screen_value(screen_rules, src_sel, "font-size")
            if fs != want:
                fails.append(f"`{s}` prints at {fs} but `{src_sel}` is {want} "
                             "on screen")

    # 3. nothing clipped to fit a fixed page
    for sel, body in print_rules:
        sels = _selectors(sel)
        if any(x in sels for x in (".op-sheet", ".bp-section", ".bp-print-text")):
            if (_decl(body, "overflow") or "") == "hidden":
                fails.append(f"`{sel}` is overflow:hidden under print -- clips text")
        h = _decl(body, "height") or ""
        if "100vh" in h:
            fails.append(f"`{sel}` has a fixed page height ({h}) under print")

    # 5. the per-deal fit keeps width x zoom on the column, and never goes
    #    below the 8pt floor (740px -> 11px x 720/740 x 0.75 = 8.0pt)
    script = src.split("</script>", 1)[0]
    col = re.search(r"PRINT_COLUMN_PX\s*=\s*(\d+)", script)
    widths = re.search(r"PRINT_WIDTHS\s*=\s*\[([^\]]+)\]", script)
    if col and int(col.group(1)) != int(PRINT_COLUMN_PX):
        fails.append(f"PRINT_COLUMN_PX = {col.group(1)}, not {PRINT_COLUMN_PX:.0f}")
    if widths:
        ws = [int(x) for x in re.findall(r"\d+", widths.group(1))]
        if min(ws) < 740:
            fails.append(f"PRINT_WIDTHS reaches {min(ws)}px -- larger than the "
                         "8pt floor allows; text would outgrow the page")
        if screen_w is not None and max(ws) != screen_w:
            fails.append(f"PRINT_WIDTHS ends at {max(ws)}px, not the screen "
                         f"sheet {screen_w}px that every deal fits at")
        if ws != sorted(ws):
            fails.append("PRINT_WIDTHS must run narrowest (largest type) first")

    # 4. every textarea has a print twin
    tpl = src.split("<style", 1)[0]
    for ta in re.finditer(r"<textarea\b[^>]*>", tpl):
        tag = ta.group(0)
        model = re.search(r'v-model="([^"]+)"', tag)
        cls = re.search(r'class="([^"]*)"', tag)
        name = model.group(1) if model else "?"
        if not cls or "print-hide" not in cls.group(1).split():
            fails.append(f"<textarea v-model={name}> is not print-hide -- it "
                         "prints its box, not its text")
        twin = re.search(
            r'class="[^"]*\bprint-only\b[^"]*"[^>]*>\{\{\s*' + re.escape(name)
            + r"\s*\}\}", tpl[ta.end():ta.end() + 600])
        if not twin:
            fails.append(f"<textarea v-model={name}> has no print-only twin")
    return fails


def _self_test() -> int:
    ok = True
    now = check(open(VIEW, encoding="utf-8").read())
    print(f"current file : {'PASS' if not now else 'FAIL'} ({len(now)} failures)")
    ok &= not now
    try:
        old = subprocess.run(
            ["git", "show", f"{PRE_FIX_COMMIT}:vue_app/src/views/OnePagerView.vue"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
    except subprocess.CalledProcessError:
        print(f"cannot read {PRE_FIX_COMMIT}; self-test incomplete")
        return 1
    bad = check(old)
    print(f"pre-fix file : {'FAIL as required' if bad else 'PASSED -- check is vacuous'}"
          f" ({len(bad)} failures)")
    for f in bad[:8]:
        print(f"    {f}")
    ok &= bool(bad)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=VIEW)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    fails = check(open(a.file, encoding="utf-8").read())
    for f in fails:
        print(f"FAIL  {f}")
    print("onepager print matches screen: " + ("OK" if not fails else f"{len(fails)} failure(s)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
