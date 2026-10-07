"""Guardrail: the three ways to add to an expense report, and the receipt read inside
the expense pop-up (Oct 7 2026).

Jim: "Employees are likely to see the '+ Add an expense' button and click that first
before knowing to upload a file or folder. ... Put all three option in the same level
visually. Within the add an expense option, allow the employee to load an invoice and
run the extraction process for that invoice within that pop up. Make the process
foolproof."

What must stay true:

  1. ONE bar, three equal cards: "Add an expense", the multi-file upload, and the folder
     (desktop) or camera (touch). All three are `.add-opt` inside `.add-options`, and the
     upload buttons are NOT also left in the Receipts block below (two places to look is
     what hid them).
  2. The pop-up uploads and reads through the SAME two endpoints as the upload cards --
     `/receipts` then `/receipts/<id>/extract`. No second reader.
  3. Foolproof: while the pop-up is uploading or reading, Save, Cancel, Close and the
     backdrop do nothing (closing mid-read orphaned the half-made line); one file at a
     time; the employee's typing wins over the receipt (the receipt fills blanks), asserted
     in BOTH directions; an expense already on the report takes the receipt and the
     reader's extra line is deleted, so nothing is claimed twice; a file that cannot be
     read is still attached; Cancel takes back what the pop-up uploaded (after asking), and
     Save makes it the employee's so a later Cancel cannot; a file already on the report
     is found by the id the server returns (a re-picked iPhone photo has a new name); and
     a typed amount kept over the receipt's says so.

Usage: python scripts/expense_add_options_check.py
         [--inject=split|reader|close|merge|mergeall|twice|unread|keep|savekeep|dupname|quiet|dupserver]
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def fn_body(src, name):
    """The text of `async function name(` / `function name(` up to the next top-level one."""
    m = re.search(r"\n(async )?function %s\(" % re.escape(name), src)
    if not m:
        return ""
    nxt = re.search(r"\n(async )?function \w+\(|\n// ---- ", src[m.end():])
    return src[m.start(): m.end() + (nxt.start() if nxt else len(src))]


def main():
    view = (ROOT / "vue_app/src/views/ExpensesView.vue").read_text(encoding="utf-8")
    if INJECT == "split":      # the upload button left behind in the Receipts block too
        view = view.replace('<strong>Receipts on this report</strong>',
                            '<strong>Receipts on this report</strong><label class="btn-primary '
                            'file-btn">Upload files<input type="file" multiple hidden '
                            '@change="uploadFiles" /></label>')
    if INJECT == "reader":     # a second reader in the pop-up
        view = view.replace("/receipts/${rid}/extract`", "/receipts/${rid}/read-invoice`")
    if INJECT == "close":      # the backdrop closes mid-read
        view = view.replace('class="modal-backdrop" @click.self="closeEditing"',
                            'class="modal-backdrop" @click.self="editing = null"')
    if INJECT == "merge":      # the receipt overwrites what the employee typed
        view = view.replace("if (changed || readBlank) out[k] = mine", "if (readBlank) out[k] = mine")
    if INJECT == "mergeall":   # the employee's blanks wipe what the receipt read
        view = view.replace("if (changed || readBlank) out[k] = mine", "out[k] = mine")
    if INJECT == "twice":      # the reader's line kept beside the existing one
        view = view.replace("report.value = (await api.delete(`/api/expenses/reports/${rep}/lines/${ln.id}`)).data",
                            "void ln")
    if INJECT == "unread":     # an unreadable file left unattached
        view = view.replace("if (!added.length) {\n      // Unreadable or no receipt in it: still the "
                            "receipt for this expense.\n      e.receiptChoice = rid",
                            "if (!added.length) {\n      // Unreadable or no receipt in it: still the "
                            "receipt for this expense.\n      void rid")

    srv = (ROOT / "flask_app/services/expense_receipts.py").read_text(encoding="utf-8")
    if INJECT == "keep":       # Cancel leaves the uploaded receipt and line behind
        view = view.replace("if (made && report.value) {", "if (false && made && report.value) {")
    if INJECT == "savekeep":   # a saved expense still counts as "made", so Cancel later deletes it
        view = view.replace("    resetInvoice()          // saved:", "    // saved:")
    if INJECT == "dupname":    # the duplicate looked up by name again
        view = view.replace("find((r: any) => r.id === res.receipt_id)",
                            "find((r: any) => r.filename === file.name)")
    if INJECT == "quiet":      # the typed amount silently wins
        view = view.replace("const diff = amountNote(typed.amount, added[0].amount)", "const diff = ''")
    if INJECT == "dupserver":  # the server stops saying which receipt it is
        srv = srv.replace('"result": "duplicate", "receipt_id": here[0],', '"result": "duplicate",')

    tpl = view[view.find("<template>"):view.find("<style scoped>")]

    print("1. Three equal options, in one place")
    bar = re.search(r'<div v-if="report\.permissions\.edit" class="add-bar">(.*?)<table class="data-table">',
                    tpl, re.S)
    chk("the add bar sits above the lines table", bar is not None)
    body = bar.group(1) if bar else ""
    opts = re.search(r'<div class="add-options">(.*?)\n        </div>', body, re.S)
    o = opts.group(1) if opts else ""
    names = re.findall(r'<span class="add-name">([^<]+)</span>', o)
    chk("'Add an expense' is one of the cards", "Add an expense" in names, names)
    chk("the multi-file upload is one of the cards",
        "{{ touch ? 'Photos or files' : 'Upload receipts' }}" in names, names)
    chk("the folder (desktop) and the camera (touch) are cards",
        "Upload a folder" in names and "Take a photo" in names, names)
    cards = re.findall(r'<(button|label)[^>]*class="add-opt"', o)
    chk("every card is the same element class (no primary/secondary ranking)",
        len(cards) == 4 and len(re.findall(r'class="add-opt"', o)) == 4, len(cards))
    chk("...and 3 columns on desktop, 1 on a phone",
        ".add-options { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));" in view
        and re.search(r"@media \(max-width: 600px\) \{[^@]*\.add-options \{ grid-template-columns: 1fr; \}",
                      view) is not None)
    rec = tpl[tpl.find('<div class="receipts">'):tpl.find("a line, read-only")]
    chk("no upload control is left in the Receipts block (one place to look)",
        rec and 'type="file"' not in rec and "SharePointPicker" not in rec)
    chk("the old '+ Add an expense' button below the table is gone",
        "+ Add an expense</button>" not in tpl)

    print("2. The pop-up reads through the same endpoints")
    up = fn_body(view, "uploadInvoice")
    chk("uploadInvoice exists and is wired to the pop-up's file inputs",
        up and tpl.count('@change="uploadInvoice"') == 2)
    chk("it stores via POST /receipts (the upload cards' endpoint)",
        "api.post(`/api/expenses/reports/${rep}/receipts`, fd)" in up)
    chk("it reads via POST /receipts/<id>/extract (the 'Read N waiting' endpoint)",
        "api.post(`/api/expenses/reports/${rep}/receipts/${rid}/extract`)" in up)
    chk("...the same extract endpoint readReceipts uses",
        "/receipts/${todo[i]}/extract`" in fn_body(view, "readReceipts"))
    chk("photos are renamed the same way (friendlyName)", "friendlyName(picked, 0)" in up)

    print("3. Foolproof")
    modal = tpl[tpl.find("line form: a pop-up"):tpl.find("what stops a submit")]
    chk("one file at a time in the pop-up",
        modal.count('@change="uploadInvoice"') == 2
        and not re.search(r'<input[^>]*multiple[^>]*uploadInvoice', modal))
    chk("the backdrop goes through closeEditing", '@click.self="closeEditing"' in modal)
    chk("closeEditing refuses while busy", "if (invoiceBusy.value) return" in fn_body(view, "closeEditing"))
    chk("Save and Cancel are disabled while busy",
        ':disabled="invoiceBusy" @click="saveLine"' in modal
        and ':disabled="invoiceBusy" @click="closeEditing">Cancel' in modal)
    chk("...and Close", ':disabled="invoiceBusy" @click="closeEditing">✕ Close' in modal)
    chk("nothing in the pop-up closes it except closeEditing / saveLine",
        "editing = null" not in modal, re.findall(r".{30}editing = null", modal))
    chk("a new upload is refused while one is running", "invoiceBusy.value) return" in up)
    chk("an expense already on the report deletes the reader's extra line(s)",
        "for (const ln of added) {\n      report.value = (await api.delete("
        "`/api/expenses/reports/${rep}/lines/${ln.id}`)).data" in up)
    chk("an unreadable file is still attached",
        "if (!added.length) {\n      // Unreadable or no receipt in it: still the receipt for "
        "this expense.\n      e.receiptChoice = rid" in up)
    chk("every way into the pop-up clears the last upload's message",
        all("resetInvoice()" in fn_body(view, f) for f in ("newLine", "editLine", "lineForReceipt",
                                                            "closeEditing")))

    close = fn_body(view, "closeEditing")
    chk("Cancel asks, then removes the line and the receipt this pop-up added",
        "if (made && report.value) {" in close and "window.confirm(" in close
        and "/lines/${made.lineId}`" in close and "/receipts/${made.receiptId}`" in close)
    chk("...the receipt is recorded as made as soon as it is stored",
        "invoice.value.made = { receiptId: rid, lineId: null, file: file.name }" in up)
    chk("...and the line once the reader makes one",
        "invoice.value.made = made && { ...made, lineId: added[0].id }" in up)
    chk("Save makes it the employee's: a later Cancel cannot remove it",
        "resetInvoice()          // saved:" in fn_body(view, "saveLine"))
    chk("a file already on the report is found by the receipt id the server returns",
        "find((r: any) => r.id === res.receipt_id)" in up)
    chk("...and the server returns it",
        re.search(r'"result": "duplicate", "receipt_id": here\[0\]', srv) is not None)
    chk("a typed amount kept over the receipt's is said, on a new expense and an existing one",
        "const diff = amountNote(typed.amount, added[0].amount)" in up
        and "amountNote(e.amount, x.amount)" in up and up.count("${diff}") == 2)

    print("4. The employee's typing wins; the receipt fills blanks (both directions)")
    merge = fn_body(view, "mergeTyped")
    m = re.search(r"\{\n(.*?)\n\s*return out", merge, re.S)
    node = shutil.which("node")
    if not merge or not node:
        chk("mergeTyped runs", False, "missing function" if not merge else "node not on PATH")
    else:
        # Run the real function under node, with the real blankLine().
        blank = re.search(r"const blankLine = \(\) => \((\{.*?\})\)\n", view, re.S).group(1)
        blank = re.sub(r" as \{[^}]*\}\[\]", "", blank)      # TypeScript casts off
        blank = re.sub(r" as [\w |]+", "", blank)
        js = re.sub(r"\(typed: any, read: any\)", "(typed, read)", merge)
        js = js.replace("const blank: any", "const blank").replace("const out: any", "const out")
        prog = ("const blankLine = () => (%s);\n%s\n"
                "const typed = {...blankLine(), purpose:'Site visit', amount:'45.00'};\n"
                "const read = {...blankLine(), id:7, amount:52.1, vendor:'Harbor Grill', "
                "line_date:'2026-09-30', deal_code:'', purpose:''};\n"
                "console.log(JSON.stringify(mergeTyped(typed, read)));\n") % (blank, js)
        out = subprocess.run([node, "-e", prog], capture_output=True, text=True)
        try:
            import json
            got = json.loads(out.stdout.strip())
        except Exception:
            got = {}
        chk("mergeTyped runs", bool(got), out.stderr.strip()[:200])
        if got:
            chk("a typed amount is kept over the receipt's", got.get("amount") == "45.00", got.get("amount"))
            chk("a typed purpose is kept", got.get("purpose") == "Site visit", got.get("purpose"))
            chk("the receipt's vendor fills a blank one", got.get("vendor") == "Harbor Grill", got.get("vendor"))
            chk("the receipt's date fills a blank one", got.get("line_date") == "2026-09-30",
                got.get("line_date"))
            chk("the line is the reader's (its id), so Save updates it, not a second one",
                got.get("id") == 7, got.get("id"))
            chk("an untouched default deal survives a blank read (not left empty)",
                got.get("deal_code") == "OPERATIONS", got.get("deal_code"))

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
