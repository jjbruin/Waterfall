"""Guardrail: the expense app on an iPhone or iPad (Oct 7 2026).

Jim: "If someone chooses to open the expense app from an iphone or ipad, I'd like them
to be able to load receipts from their photos." What made that hard, and must stay
fixed:

  1. iOS offers Photo Library and the camera from the MIME types in `accept`
     (`image/*`), so the receipt picker lists them beside the extensions.
  2. A touch device gets "Take a photo" (camera, `capture=environment`) and does NOT get
     "Upload a folder" -- iOS and Android cannot pick folders. A desktop keeps the
     folder button and gets no camera button.
  3. iOS names nearly every photo "image.jpg"; such a name becomes "Photo <date time>"
     before upload. A real name (IMG_4417.HEIC, a PDF) is kept -- asserted both ways.
  4. THE CASCADE BUG: the narrow-window rule that stacks the form above the receipt sat
     BEFORE `.line-form.with-receipt` in the sheet, so it never applied -- on a phone the
     form and the receipt were 420px + 360px inside a 360px dialog. The stacking rule
     must come AFTER it.
  5. On a phone the sidebar starts folded and overlays when opened; the lines table
     scrolls inside itself below 900px (an upright iPad overflowed the page by 370px).

Usage: python scripts/expense_mobile_check.py [--inject=nophotos|folder|order|rename|noicon|stickyname]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def main():
    view = (ROOT / "vue_app/src/views/ExpensesView.vue").read_text(encoding="utf-8")
    side = (ROOT / "vue_app/src/components/layout/AppSidebar.vue").read_text(encoding="utf-8")
    if INJECT == "nophotos":
        view = view.replace("+ ',image/*,application/pdf'", "")
    if INJECT == "folder":
        view = view.replace('<label v-if="!touch" class="add-opt"', '<label class="add-opt"')
    if INJECT == "order":
        # the stacking rule moved back to where it was: before the rule it must override
        rule = "  .line-form.with-receipt { grid-template-columns: 1fr; }\n"
        view = view.replace(rule, "", 1)
        view = view.replace("<style scoped>", "<style scoped>\n@media (max-width: 900px) {\n"
                            + rule + "}", 1)
    if INJECT == "rename":
        view = view.replace("const files = picked.map(friendlyName)", "const files = picked")

    print("1. Photos are offered")
    pick = re.search(r"const RECEIPT_PICK = (.+)", view)
    chk("the receipt picker lists image/* and application/pdf",
        pick and "image/*" in pick.group(1) and "application/pdf" in pick.group(1),
        pick and pick.group(1))
    chk("...and the multi-file input uses it", ':accept="RECEIPT_PICK"' in view)

    print("2. Camera on touch, folders on desktop")
    cam = re.search(r'<label v-if="touch"[^>]*>Take a photo\s*<input type="file" accept="image/\*" '
                    r'capture="environment"', view)
    chk("a touch device gets 'Take a photo' (camera)", cam is not None)
    # the card's label: the last <label before the "Upload a folder" name
    at = view.find(">Upload a folder<")
    lab = view.rfind("<label", 0, at) if at > 0 else -1
    folder = re.match(r'<label([^>]*)>', view[lab:]) if lab >= 0 else None
    chk("'Upload a folder' is not offered on a touch device",
        folder and 'v-if="!touch"' in folder.group(1), folder and folder.group(1))
    chk("touch is the pointer, not the width (an iPad is wide)",
        "matchMedia?.('(pointer: coarse)')" in view)

    print("3. Generic iOS names are replaced, real names kept")
    m = re.search(r"const GENERIC_NAME = /(.+)/i", view)
    rx = re.compile(m.group(1), re.I) if m else None
    chk("the rule exists", rx is not None)
    if rx:
        for name in ("image.jpg", "image.jpeg", "IMAGE.PNG", "image (2).jpg", "image.heic"):
            chk(f"'{name}' is renamed", rx.match(name) is not None)
        for name in ("IMG_4417.HEIC", "Harbor Street Grill.pdf", "imagery.jpg", "receipt.jpg"):
            chk(f"'{name}' is kept", rx.match(name) is None)
    chk("every upload path goes through it",
        "const files = picked.map(friendlyName)" in view)

    print("4. The form stacks above the receipt below 900px -- AFTER the side-by-side rule")
    base = view.find(".line-form.with-receipt { display: grid; grid-template-columns: minmax(420px")
    stack = view.find(".line-form.with-receipt { grid-template-columns: 1fr; }")
    chk("both rules present, the stacking rule ONCE (a dead copy above hid this bug)",
        base > 0 and stack > 0
        and view.count(".line-form.with-receipt { grid-template-columns: 1fr; }") == 1,
        (base, stack))
    chk("the stacking rule comes later in the sheet, so it wins", stack > base > 0, (base, stack))
    before = view[:stack]
    chk("...inside a max-width: 900px block", before.rfind("@media (max-width: 900px)") > base)

    print("5. Phone layout")
    chk("the lines table scrolls inside itself below 900px",
        re.search(r"@media \(max-width: 900px\) \{[^}]*\{[^}]*\}\s*/\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*"
                  r"\.data-table \{ display: block; overflow-x: auto;", view) is not None)
    chk("the sidebar starts folded on a phone", "const PHONE = '(max-width: 600px)'" in side
        and "if (isPhone() && !collapsed.value) toggleCollapsed()" in side)
    chk("...keeps the page at the rail's width when opened (it overlays)",
        "isPhone() || collapsed.value ? '40px' : '240px'" in side)
    chk("...and folds away again after picking a screen",
        "watch(() => route.fullPath" in side)

    print("6. 'PSC Expenses' on the home screen (Add to Home Screen, Oct 7 2026)")
    index = (ROOT / "vue_app/index.html").read_text(encoding="utf-8")
    if INJECT == "noicon":
        index = index.replace('<link rel="apple-touch-icon" href="/apple-touch-icon.png" />', "")
    if INJECT == "stickyname":
        view = view.replace("onUnmounted(() => homeScreenName(null))", "")
    chk("index.html declares the touch icon",
        '<link rel="apple-touch-icon" href="/apple-touch-icon.png" />' in index)
    icon = ROOT / "vue_app/public/apple-touch-icon.png"
    chk("...and it exists, 180x180 PNG (what iOS asks for)",
        icon.exists() and icon.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        and int.from_bytes(icon.read_bytes()[16:20], "big") == 180
        and int.from_bytes(icon.read_bytes()[20:24], "big") == 180)
    chk("the Expenses page names it 'PSC Expenses'",
        "const HOME_NAME = 'PSC Expenses'" in view and "homeScreenName(HOME_NAME)" in view
        and "apple-mobile-web-app-title" in view)
    chk("...and takes the name down on leaving, so no other screen is called Expenses",
        "onUnmounted(() => homeScreenName(null))" in view)
    chk("NOT full-screen mode: it would break Sign in with Microsoft's popup",
        "apple-mobile-web-app-capable" not in index + view)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
