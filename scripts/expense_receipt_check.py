"""Guardrail: expense receipts, phase 2 -- upload, read, show beside the line.

Jim, Oct 2 2026: the employee sees "the image of the uploaded invoice related
to the line they are completing", to "correct amounts that may have been hand
written and not picked up by the extractor".

MAKES NO API CALLS. The model is replaced by a recorder, because what matters
here is what is SENT (an upright, bounded image; a PDF as a document; the
fifteen categories in the prompt) and what is DONE with the answer (a line per
receipt, on its page, the reader's figures kept beside the employee's). The
stub answers with a THINKING block first, as the real model does by default,
so an indexed `content[0].text` read fails here.
"""
import base64
import io
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_passed, _failed = [], []
SENT = []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, blocks, stop):
        self.content, self.stop_reason = blocks, stop


REPLIES = []      # (text, stop_reason) consumed one per call


def install_stub():
    class _Stream:
        def __init__(self, m):
            self._m = m

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return self._m

    class _Messages:
        def stream(self, **kw):
            SENT.append(kw)
            body, stop = REPLIES.pop(0) if REPLIES else ('{"receipts": []}', "end_turn")
            return _Stream(_Msg([_Block("thinking", thinking="..."),
                                 _Block("text", text=body)], stop))

    class _Client:
        def __init__(self, **kw):
            self.messages = _Messages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = _Client
    sys.modules["anthropic"] = mod


def make_jpeg(w=3000, h=4000, rotated=False, color=(240, 240, 240)):
    from PIL import Image
    img = Image.new("RGB", (w, h), color)
    out = io.BytesIO()
    if rotated:
        exif = Image.Exif()
        exif[0x0112] = 6          # "rotate 90 CW to view" -- how a phone saves a portrait
        img.save(out, "JPEG", exif=exif)
    else:
        img.save(out, "JPEG")
    return out.getvalue()


def make_pdf(pages=2):
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    doc = pymupdf.open()
    for i in range(pages):
        pg = doc.new_page()
        pg.insert_text((72, 72), "RECEIPT %d  TOTAL 12.34" % (i + 1))
    data = doc.tobytes()
    doc.close()
    return data


def main():
    install_stub()
    os.environ["ANTHROPIC_API_KEY"] = "stub-not-a-real-key"
    tmp = tempfile.mkdtemp(prefix="expense_receipt_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")

    import jwt
    import pandas as pd
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    from flask_app.services import data_service
    from flask_app.services.lease_review_service import EXTRACTION_MODEL
    from database import PROTECTED_TABLES

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()
    data_service.get_data = lambda *a, **k: {"inv": pd.DataFrame([
        {"vcode": "P0000001", "Investment_Name": "Apple Self Storage", "Portfolio_Name": "",
         "Sale_Status": "", "Sale_Date": None, "Lifecycle": "Stable"}])}

    people = {"admin": "admin", "emp": "analyst", "mgr": "analyst", "other": "analyst",
              "acct": "accountant"}
    with app.app_context():
        eng = get_engine()
        for u, role in people.items():
            create_user(u, "pw-" + u, role=role)
        ids = {u["username"]: u["id"] for u in list_users()}
        with eng.begin() as c:
            c.execute(text('CREATE TABLE IF NOT EXISTS gl_accounts ("ACCTNUM" TEXT, "ACCTNAME" TEXT, "TYPE" TEXT)'))
            for a, n in (("MR53000004", "Other Expense: Meals & Entertainment"),
                         ("MR53000011", "Other Expense: Travel")):
                c.execute(text("INSERT INTO gl_accounts VALUES (:a, :n, 'I')"), {"a": a, "n": n})

    def H(name):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[name]), "username": name, "role": people[name],
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    def call(method, path, who, body=None):
        r = client.open("/api/expenses" + path, method=method, json=body, headers=H(who))
        return r.status_code, (r.get_json(silent=True) or {})

    def upload(rid, who, files):
        data = {"files": [(io.BytesIO(b), n) for n, b in files]}
        r = client.post("/api/expenses/reports/%d/receipts" % rid, data=data,
                        headers=H(who), content_type="multipart/form-data")
        return r.status_code, (r.get_json(silent=True) or {})

    def fetch(rid, rcid, who):
        r = client.get("/api/expenses/reports/%d/receipts/%d/file" % (rid, rcid), headers=H(who))
        return r.status_code, r.headers.get("Content-Type", ""), r.data

    call("PUT", "/employees/%d" % ids["emp"], "admin", {"full_name": "Erin", "approver_user_id": ids["mgr"]})
    call("PUT", "/employees/%d" % ids["other"], "admin", {"approver_user_id": ids["mgr"]})
    rid = call("POST", "/reports", "emp", {"period_start": "2026-09-01", "period_end": "2026-09-30"})[1]["id"]
    other_rid = call("POST", "/reports", "other", {"period_start": "2026-09-01", "period_end": "2026-09-30"})[1]["id"]

    JPG = make_jpeg(rotated=True)
    PDF = make_pdf(2)

    print("1. Upload: stored, skipped, refused, duplicates named")
    chk("er_receipts is protected", "er_receipts" in PROTECTED_TABLES)
    st, b = upload(rid, "emp", [("dinner.jpg", JPG), ("two.pdf", PDF), (".DS_Store", b"x"),
                                ("notes.txt", b"hello"), ("empty.png", b"")])
    res = {x["file"]: x for x in b.get("results", [])}
    chk("a JPG and a PDF are stored", res["dinner.jpg"]["result"] == "stored"
        and res["two.pdf"]["result"] == "stored", res)
    chk("a folder's .DS_Store is skipped, not refused", res[".DS_Store"]["result"] == "skipped")
    chk("a text file is refused, saying why",
        res["notes.txt"]["result"] == "refused" and "not a receipt format" in res["notes.txt"]["why"])
    chk("an empty file is refused", res["empty.png"]["result"] == "refused")
    rc = {x["filename"]: x for x in b["receipts"]}
    chk("the PDF's page count is recorded", rc["two.pdf"]["page_count"] == 2)
    chk("the same file twice on one report is not stored again",
        upload(rid, "emp", [("again.jpg", JPG)])[1]["results"][0]["result"] == "duplicate")
    st, b = upload(other_rid, "other", [("shared-dinner.jpg", JPG)])
    chk("the same file on ANOTHER employee's report is stored and flagged",
        b["results"][0]["result"] == "stored" and "another expense report" in b["results"][0]["why"])
    chk("another employee cannot upload to this report",
        upload(rid, "other", [("x.jpg", make_jpeg(10, 10))])[0] == 404)

    print("\n2. HEIC, an iPhone's default, is shown as a JPEG")
    try:
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        buf = io.BytesIO()
        Image.new("RGB", (400, 300), (200, 10, 10)).save(buf, "HEIF")
        st, b = upload(rid, "emp", [("IMG_0001.HEIC", buf.getvalue())])
        heic = b["results"][0]
        chk("a HEIC photo is stored", heic["result"] == "stored", heic)
        s2, ct, data = fetch(rid, heic["receipt_id"], "emp")
        chk("...and served as a JPEG a browser can show", s2 == 200 and ct.startswith("image/jpeg")
            and data[:2] == b"\xff\xd8", ct)
        call("DELETE", "/reports/%d/receipts/%d" % (rid, heic["receipt_id"]), "emp")
    except ImportError:
        print("   skip HEIC   (pillow-heif not installed here)")

    print("\n3. The file is the report's: owner, then approver, then accounting")
    jid, pid = rc["dinner.jpg"]["id"], rc["two.pdf"]["id"]
    s2, ct, data = fetch(rid, jid, "emp")
    chk("the owner sees the image as uploaded", s2 == 200 and ct.startswith("image/jpeg") and data == JPG)
    for who in ("other", "mgr", "acct"):
        chk("%s gets 404 on a draft's receipt" % who, fetch(rid, jid, who)[0] == 404)
    chk("a receipt id from another report is 404 through this one",
        fetch(rid, upload(other_rid, "other", [("o.jpg", make_jpeg(20, 20))])[1]["results"][0]["receipt_id"], "emp")[0] == 404)

    print("\n4. Reading: what is sent")
    REPLIES.append((json.dumps({"receipts": [{
        "page": 1, "date": "2026-09-10", "vendor": "Arnaud's", "total": 228.03,
        "printed_total": 190.03, "tip": 38.00, "handwritten_amount": True,
        "amount_note": "Tip and total are handwritten.", "description": "Dinner, 3 guests",
        "category": "Other Expense: Meals & Entertainment"}]}), "end_turn"))
    st, rep = call("POST", "/reports/%d/receipts/%d/extract" % (rid, jid), "emp")
    sent = SENT[-1]
    blocks = sent["messages"][0]["content"]
    chk("the lease engine's model reads it", sent["model"] == EXTRACTION_MODEL, sent["model"])
    chk("an image goes as an image block before the prompt",
        blocks[0]["type"] == "image" and blocks[-1]["type"] == "text")
    raw = blocks[0]["source"]["data"]
    chk("base64 carries no newlines", "\n" not in raw)
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(raw)))
    chk("the image is bounded to 2000 px on its long side", max(im.size) <= 2000, im.size)
    chk("a phone photo is sent UPRIGHT (EXIF rotation applied): 3000x4000 tagged rotate-90 "
        "arrives landscape", im.size[0] > im.size[1], im.size)
    chk("the prompt lists accounting's categories",
        "Other Expense: Meals & Entertainment" in blocks[-1]["text"])

    print("\n5. Reading: what is done with the answer")
    lines = [x for x in rep.get("lines", []) if x["receipt_id"] == jid]
    ln = lines[0] if lines else {}
    chk("one line per receipt, carrying the receipt", len(lines) == 1)
    chk("the amount is the HANDWRITTEN total, tip included", ln.get("amount") == 228.03, ln.get("amount"))
    chk("vendor and date come across", ln.get("vendor") == "Arnaud's" and ln.get("line_date") == "2026-09-10")
    chk("the suggested category maps to its account", ln.get("category_account") == "MR53000004")
    chk("the reader's figures are kept beside the line",
        (ln.get("extracted") or {}).get("printed_total") == 190.03
        and (ln.get("extracted") or {}).get("handwritten_amount") is True)
    chk("the receipt is marked read", [r.get("status") for r in rep.get("receipts", []) if r["id"] == jid] == ["read"], st)

    REPLIES.append((json.dumps({"receipts": [
        {"page": 1, "date": "2026-09-11", "vendor": "Marriott", "total": 278.13, "category": "Other Expense: Travel"},
        {"page": 2, "date": "2026-09-12", "vendor": "Chick-fil-A", "total": 21.95, "category": "Not A Category"}]}),
        "end_turn"))
    st, rep = call("POST", "/reports/%d/receipts/%d/extract" % (rid, pid), "emp")
    chk("a PDF goes as a document block", SENT[-1]["messages"][0]["content"][0]["type"] == "document")
    pl = sorted([x for x in rep.get("lines", []) if x["receipt_id"] == pid], key=lambda x: x["receipt_page"])
    chk("two receipts in one PDF make two lines, each on its page",
        [x["receipt_page"] for x in pl] == [1, 2], [x.get("receipt_page") for x in pl])
    chk("a category the reader invented is left blank, not forced",
        pl[1]["category_account"] is None if len(pl) > 1 else False)

    print("\n6. Correcting a line keeps what was read")
    st, rep = call("PUT", "/reports/%d/lines/%d" % (rid, ln["id"]), "emp", {
        "line_date": "2026-09-10", "category_account": "MR53000004", "purpose": "Property Visit - Existing",
        "deal_code": "P0000001", "vendor": "Arnaud's", "comment": "Pontchartrain - Dinner",
        "amount": "238.03", "receipt_id": jid, "receipt_page": 1})
    ln2 = [x for x in rep["lines"] if x["id"] == ln["id"]][0]
    chk("the employee's correction is the line's amount", ln2["amount"] == 238.03)
    chk("...and the reader's 228.03 is still beside it", ln2["extracted"]["total"] == 228.03)
    chk("an attached receipt satisfies the receipt rule", not rep["check"]["by_line"][str(ln["id"])]["errors"]
        if str(ln["id"]) in rep["check"]["by_line"] else not rep["check"]["by_line"][ln["id"]]["errors"])
    st, b = call("PUT", "/reports/%d/lines/%d" % (rid, ln["id"]), "emp", {**{
        "line_date": "2026-09-10", "category_account": "MR53000004", "purpose": "Other",
        "deal_code": "OPERATIONS", "comment": "x", "amount": "1"}, "receipt_id": pid, "receipt_page": 5})
    chk("a page the receipt does not have is refused", st == 400 and "page" in b.get("error", ""))
    other_rc = upload(other_rid, "other", [("p.jpg", make_jpeg(30, 30))])[1]["results"][0]["receipt_id"]
    st, b = call("PUT", "/reports/%d/lines/%d" % (rid, ln["id"]), "emp", {
        "line_date": "2026-09-10", "category_account": "MR53000004", "purpose": "Other",
        "deal_code": "OPERATIONS", "comment": "x", "amount": "1", "receipt_id": other_rc})
    chk("another report's receipt cannot be attached", st == 400)

    print("\n7. A Y with nothing attached is not a receipt")
    st, rep = call("POST", "/reports/%d/lines" % rid, "emp", {
        "line_date": "2026-09-12", "category_account": "MR53000011", "purpose": "Other",
        "deal_code": "OPERATIONS", "comment": "Parking", "amount": "12", "receipt": "Y"})
    chk("is refused at submit, saying to attach one or say why",
        any("no receipt attached" in e for e in rep["check"]["errors"]))
    call("DELETE", "/reports/%d/lines/%d" % (rid, rep["lines"][-1]["id"]), "emp")

    print("\n8. Re-reading, failures and refusals")
    chk("a file with lines from it cannot be re-read",
        call("POST", "/reports/%d/receipts/%d/extract" % (rid, pid), "emp")[0] == 403)
    p3 = upload(rid, "emp", [("bad.pdf", make_pdf(1))])[1]["results"][0]["receipt_id"]
    n_before = len(SENT)
    REPLIES.extend([("I could not find anything.", "end_turn"), ("Still nothing.", "end_turn")])
    st, rep = call("POST", "/reports/%d/receipts/%d/extract" % (rid, p3), "emp")
    r3 = [r for r in rep["receipts"] if r["id"] == p3][0]
    chk("no answer is retried once", len(SENT) - n_before == 2, len(SENT) - n_before)
    chk("...and a PDF's retry goes as page images",
        SENT[-1]["messages"][0]["content"][0]["type"] == "image")
    chk("a failed reading says so and why, and adds no line",
        r3["status"] == "error" and r3["error"] and not [x for x in rep["lines"] if x["receipt_id"] == p3], r3)
    p4 = upload(rid, "emp", [("odd.jpg", make_jpeg(50, 60))])[1]["results"][0]["receipt_id"]
    REPLIES.append(("", "refusal"))
    st, rep = call("POST", "/reports/%d/receipts/%d/extract" % (rid, p4), "emp")
    r4 = [r for r in rep["receipts"] if r["id"] == p4][0]
    chk("a refusal is recorded as such, not as a parse failure",
        r4["status"] == "error" and "declined" in (r4["error"] or ""), r4.get("error"))
    REPLIES.append((json.dumps({"receipts": [], "notes": "This is a photo of a cat."}), "end_turn"))
    p5 = upload(rid, "emp", [("cat.jpg", make_jpeg(70, 80))])[1]["results"][0]["receipt_id"]
    st, rep = call("POST", "/reports/%d/receipts/%d/extract" % (rid, p5), "emp")
    r5 = [r for r in rep["receipts"] if r["id"] == p5][0]
    chk("a file with no receipt says so, with the reader's reason",
        r5["status"] == "no_receipt" and "cat" in (r5["error"] or ""))

    print("\n9. Removing a receipt unlinks its lines, which then need one")
    st, rep = call("DELETE", "/reports/%d/receipts/%d" % (rid, pid), "emp")
    chk("the file is gone", pid not in [r["id"] for r in rep["receipts"]])
    orphan = [x for x in rep["lines"] if x["vendor"] == "Marriott"][0]
    chk("its line keeps its figures and loses the link",
        orphan["amount"] == 278.13 and orphan["receipt_id"] is None)
    chk("...and must now be given a receipt or a reason",
        any("no receipt attached" in e for e in rep["check"]["by_line"][str(orphan["id"])]["errors"]))

    print("\n10. After submit and approval, the approver and accounting see the receipt")
    for x in rep["lines"]:
        call("DELETE", "/reports/%d/lines/%d" % (rid, x["id"]), "emp")
    for rr in rep["receipts"]:
        if rr["id"] != jid:
            call("DELETE", "/reports/%d/receipts/%d" % (rid, rr["id"]), "emp")
    st, rep = call("POST", "/reports/%d/lines" % rid, "emp", {
        "line_date": "2026-09-10", "category_account": "MR53000004", "purpose": "Property Visit - Existing",
        "deal_code": "P0000001", "comment": "Dinner", "amount": "228.03", "receipt_id": jid})
    st, rep = call("POST", "/reports/%d/submit" % rid, "emp")
    chk("a report whose line carries its receipt submits", st == 200, rep.get("error"))
    chk("the approver sees the image", fetch(rid, jid, "mgr")[0] == 200)
    chk("accounting does not yet", fetch(rid, jid, "acct")[0] == 404)
    chk("another employee never does", fetch(rid, jid, "other")[0] == 404)
    chk("the owner cannot remove a receipt from a submitted report",
        call("DELETE", "/reports/%d/receipts/%d" % (rid, jid), "emp")[0] == 403)
    call("POST", "/reports/%d/decide" % rid, "mgr", {"action": "approve"})
    chk("accounting sees it once approved", fetch(rid, jid, "acct")[0] == 200)

    print("\n11. A shared receipt is flagged on the second report's line")
    st, orep = call("GET", "/reports/%d" % other_rid, "other")
    shared = [r for r in orep["receipts"] if r["filename"] == "shared-dinner.jpg"][0]
    st, orep = call("POST", "/reports/%d/lines" % other_rid, "other", {
        "line_date": "2026-09-10", "category_account": "MR53000004", "purpose": "Other",
        "deal_code": "OPERATIONS", "comment": "Dinner", "amount": "228.03", "receipt_id": shared["id"]})
    chk("the line warns its receipt is also on another report",
        any("also on another expense report" in w for w in orep["check"]["warnings"]), orep["check"]["warnings"])

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
