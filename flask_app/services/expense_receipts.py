"""Expense receipts -- phase 2: upload, read, and show beside the line.

Jim, Oct 2 2026: "I would like the employee to see the image of the uploaded
invoice related to the line they are completing. This will help them correct
amounts that may have been hand written and not picked up by the extractor."

THE FLOW. Files are uploaded to a report (pick files, or pick a folder -- the
browser sends every file in it, exactly as Lease Review's document upload).
Each is stored as it arrived and then READ, one file per request, so twenty
receipts are twenty short calls the screen counts through rather than one call
that might outlive the connection. Reading a file proposes one LINE per receipt
it holds, carrying the receipt and the page it is on. The employee completes
the line with the receipt's image beside the form.

WHAT WAS READ IS KEPT BESIDE WHAT THE EMPLOYEE SETTLED. `extracted_json` on the
line holds the reader's figures; the line's own fields are the employee's. So a
corrected amount still shows what the reader saw ("read as 23.50"), and a
handwritten tip the reader flagged is pointed at, not silently trusted.

THE READER IS THE LEASE ENGINE'S ROUTE, NOT A SECOND ONE. The same model
(`EXTRACTION_MODEL`), the same scan handling (a PDF as a document block; page
images when that yields nothing -- `_render_pdf_pages`, measured on GNC's
lease), the same request-size check, and the same two traps handled the same
way: thinking is on by default so the first block is a THINKING block, never
`content[0].text`; and a refusal is HTTP 200 with no text, so `stop_reason` is
read first. What differs is the prompt, because a receipt is not a lease.

A DUPLICATE IS NAMED, NOT HIDDEN. The same file twice on one report is not
stored again. The same file on ANOTHER report is stored and flagged, because a
shared dinner is one receipt and accounting needs to see that two people
claimed it.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from flask_app.services import expense_service as ex

logger = logging.getLogger(__name__)

#: What a receipt may arrive as. Browser-displayable formats are shown as they
#: are; HEIC (an iPhone's default), TIFF and BMP are converted to a JPEG for
#: viewing and keep their original beside it.
TYPES = {
    ".pdf": "application/pdf", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    ".heic": "image/heic", ".heif": "image/heif",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".bmp": "image/bmp",
}
NEEDS_VIEW = {"image/heic", "image/heif", "image/tiff", "image/bmp"}
#: What a folder upload carries that is not a receipt. Skipped, and said so.
IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

MAX_FILE_BYTES = 25 * 1024 * 1024
#: A receipt is read at up to 2000 px on the long side: a handwritten tip is
#: the smallest thing on it that matters.
MODEL_LONG_SIDE = 2000
MAX_OUTPUT_TOKENS = 16_000

RECEIPT_PROMPT = """You are reading an employee's receipt or invoice for an expense report.
The file may hold ONE receipt or SEVERAL (for example a scan of several receipts, or a
multi-page PDF with one receipt per page). Return one entry per distinct receipt or
invoice. A single statement or invoice that lists several charges is ONE entry.

For each, read:
- page: the page number it is on (1 for a single image)
- date: the transaction date, YYYY-MM-DD
- date_end: only if the document covers a period (a monthly bill, a statement), the
  period's last day, YYYY-MM-DD; otherwise null
- vendor: the merchant or supplier name as printed
- total: the amount actually PAID, as a number. If a tip or a total has been WRITTEN BY
  HAND, the paid amount includes it: report the handwritten total. If the handwriting is
  not legible, do NOT guess digits -- give null and say so in amount_note.
- printed_total: the total as PRINTED before any handwriting, as a number, or null
- tip: the tip, as a number, or null
- tax: the tax, as a number, or null
- currency: three-letter code, USD unless the receipt says otherwise
- card_last4: the last four digits of the card, if shown, else null
- handwritten_amount: true if any amount that affects the total is handwritten
- amount_note: one short sentence for the employee about anything uncertain in the
  amount (illegible digits, a total that does not add up, two totals), else null
- description: a few words saying what was bought (e.g. "Dinner, 4 guests",
  "Airfare PHL-MEM round trip", "Hotel 2 nights")
- category: the ONE best match from this list, exactly as written, or null:
{categories}

Respond with JSON only, in this shape:
{{"receipts": [{{"page": 1, "date": "2026-09-10", "date_end": null, "vendor": "...",
  "total": 0.00, "printed_total": null, "tip": null, "tax": null, "currency": "USD",
  "card_last4": null, "handwritten_amount": false, "amount_note": null,
  "description": "...", "category": "..."}}],
 "notes": "anything about the file as a whole, or null"}}
If the file is not a receipt or invoice at all, return {{"receipts": [], "notes": "why"}}.
"""

_COLS_DONE: set = set()


def ensure_tables(engine) -> None:
    ex.ensure_tables(engine)


def ensure_receipt_tables(engine) -> None:
    """Called by `expense_service.ensure_tables`, after er_lines exists."""
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _COLS_DONE:
        return
    pg = engine.dialect.name == "postgresql"
    blob = "BYTEA" if pg else "BLOB"
    pk = "SERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with engine.begin() as c:
        c.execute(text(f"""CREATE TABLE IF NOT EXISTS er_receipts (
            id              {pk},
            report_id       INTEGER NOT NULL,
            user_id         INTEGER NOT NULL,
            filename        TEXT NOT NULL,
            content_type    TEXT NOT NULL,
            size_bytes      INTEGER,
            sha256          TEXT NOT NULL,
            file_data       {blob} NOT NULL,
            view_data       {blob},
            view_type       TEXT,
            page_count      INTEGER,
            status          TEXT NOT NULL DEFAULT 'pending',
            extraction_json TEXT,
            extraction_source TEXT,
            error           TEXT,
            duplicate_of    INTEGER,
            uploaded_at     TEXT,
            extracted_at    TEXT)"""))
    # The line's link to its receipt, and what the reader read for it. Added by
    # migration as well as being part of the line, so a table created before
    # phase 2 gains them (the v507 lesson: CREATE IF NOT EXISTS adds nothing to
    # a table that already exists).
    from sqlalchemy import inspect
    have = {c["name"] for c in inspect(engine).get_columns("er_lines")}
    with engine.begin() as c:
        for col, typ in (("receipt_id", "INTEGER"), ("receipt_page", "INTEGER"),
                         ("extracted_json", "TEXT")):
            if col not in have:
                c.execute(text("ALTER TABLE er_lines ADD COLUMN %s %s" % (col, typ)))
    _COLS_DONE.add(key)


# ------------------------------------------------------------------ files

def _ext(name: str) -> str:
    return os.path.splitext(name or "")[1].lower()


def _open_image(data: bytes):
    """A Pillow image, upright (phone photos carry their rotation in EXIF)."""
    from PIL import Image, ImageOps
    if _sniff_heic(data):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise ValueError("this is an iPhone HEIC photo and HEIC support is not "
                             "installed on the server")
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _sniff_heic(data: bytes) -> bool:
    return data[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1", b"ftypmsf1")


def _jpeg(img, long_side: int, quality: int = 85) -> bytes:
    w, h = img.size
    scale = min(1.0, float(long_side) / max(w, h, 1))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=quality)
    return out.getvalue()


def _pdf_pages(data: bytes) -> Optional[int]:
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        doc = pymupdf.open(stream=bytes(data), filetype="pdf")
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


def prepare(filename: str, data: bytes) -> dict:
    """What to store for one file, or raise ValueError saying why it cannot be."""
    ext = _ext(filename)
    if ext not in TYPES:
        raise ValueError("%s is not a receipt format (PDF, JPG, PNG, HEIC, GIF, "
                         "WEBP, TIFF or BMP)" % (ext or "a file with no extension"))
    if not data:
        raise ValueError("the file is empty")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("the file is %.1f MB, over the %d MB limit"
                         % (len(data) / 1e6, MAX_FILE_BYTES // (1024 * 1024)))
    ctype = TYPES[ext]
    out = {"content_type": ctype, "view_data": None, "view_type": None, "page_count": 1}
    if ctype == "application/pdf":
        n = _pdf_pages(data)
        if n is None:
            raise ValueError("the PDF could not be opened")
        out["page_count"] = n
    else:
        try:
            img = _open_image(data)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError("the image could not be opened (%s)" % e)
        if ctype in NEEDS_VIEW:
            out["view_data"] = _jpeg(img, 2400, 88)
            out["view_type"] = "image/jpeg"
    return out


def _receipt_rows(engine, report_id) -> List[dict]:
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT id, report_id, user_id, filename, content_type, size_bytes, sha256, "
            "view_type, page_count, status, extraction_json, extraction_source, error, "
            "duplicate_of, uploaded_at, extracted_at FROM er_receipts WHERE report_id = :r "
            "ORDER BY id"), {"r": int(report_id)}).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d["extraction"] = json.loads(d.pop("extraction_json") or "null")
        out.append(d)
    return out


def receipts_for(engine, actor, report_id) -> List[dict]:
    ensure_tables(engine)
    r = ex._visible_report(engine, actor, report_id)
    return _receipt_rows(engine, r["id"])


def upload(engine, actor, report_id, files: List[Tuple[str, bytes]]) -> dict:
    """Store each file. One result per file: stored, duplicate, skipped or refused."""
    ensure_tables(engine)
    r = ex._owned_editable(engine, actor, report_id)
    results = []
    for name, data in files:
        base = os.path.basename((name or "").replace("\\", "/"))
        if base.lower() in IGNORED_NAMES or base.startswith("._"):
            results.append({"file": base, "result": "skipped", "why": "not a receipt file"})
            continue
        try:
            prep = prepare(base, data)
        except ValueError as e:
            results.append({"file": base, "result": "refused", "why": str(e)})
            continue
        sha = hashlib.sha256(data).hexdigest()
        with engine.connect() as c:
            same = c.execute(text("SELECT id, report_id FROM er_receipts WHERE sha256 = :s "
                                  "ORDER BY id"), {"s": sha}).fetchall()
        if any(int(x[1]) == int(r["id"]) for x in same):
            results.append({"file": base, "result": "duplicate",
                            "why": "this file is already on this report"})
            continue
        dup = int(same[0][1]) if same else None
        with engine.begin() as c:
            rid = c.execute(text(
                "INSERT INTO er_receipts (report_id, user_id, filename, content_type, "
                "size_bytes, sha256, file_data, view_data, view_type, page_count, status, "
                "duplicate_of, uploaded_at) VALUES (:rep, :u, :f, :ct, :sz, :sha, :fd, :vd, "
                ":vt, :pc, 'pending', :dup, :at) RETURNING id"),
                {"rep": r["id"], "u": int(actor["id"]), "f": base, "ct": prep["content_type"],
                 "sz": len(data), "sha": sha, "fd": data, "vd": prep["view_data"],
                 "vt": prep["view_type"], "pc": prep["page_count"], "dup": dup,
                 "at": ex._now()}).scalar()
        res = {"file": base, "result": "stored", "receipt_id": rid}
        if dup:
            res["why"] = ("the same file is already on another expense report -- it is "
                          "kept, and flagged for your approver and accounting")
        results.append(res)
    return {"results": results, "receipts": _receipt_rows(engine, r["id"])}


def receipt_file(engine, actor, report_id, receipt_id) -> Tuple[bytes, str, str]:
    """(bytes, content type, filename) to SHOW: the viewable copy where one exists."""
    ensure_tables(engine)
    r = ex._visible_report(engine, actor, report_id)
    with engine.connect() as c:
        row = c.execute(text(
            "SELECT filename, content_type, file_data, view_data, view_type FROM er_receipts "
            "WHERE id = :i AND report_id = :r"), {"i": int(receipt_id), "r": r["id"]}).first()
    if not row:
        raise LookupError("No receipt %s on report %s." % (receipt_id, r["id"]))
    if row[3] is not None:
        return bytes(row[3]), row[4], row[0]
    return bytes(row[2]), row[1], row[0]


def delete_receipt(engine, actor, report_id, receipt_id) -> dict:
    """Remove a file. A line that pointed at it keeps its figures and loses the
    link, so it then needs a receipt or a reason before the report can go."""
    ensure_tables(engine)
    r = ex._owned_editable(engine, actor, report_id)
    with engine.begin() as c:
        got = c.execute(text("DELETE FROM er_receipts WHERE id = :i AND report_id = :r"),
                        {"i": int(receipt_id), "r": r["id"]})
        if got.rowcount != 1:
            raise LookupError("No receipt %s on report %s." % (receipt_id, r["id"]))
        c.execute(text("UPDATE er_lines SET receipt_id = NULL, receipt_page = NULL, "
                       "receipt = NULL WHERE report_id = :r AND receipt_id = :i"),
                  {"r": r["id"], "i": int(receipt_id)})
    return ex.get_report(engine, actor, r["id"])


# ------------------------------------------------------------------ reading

def _model_content(data: bytes, ctype: str, prompt: str) -> Tuple[List[dict], str]:
    from flask_app.services.lease_review_service import _pdf_fits, _render_pdf_pages
    if ctype == "application/pdf":
        if _pdf_fits(data, prompt):
            return ([{"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.b64encode(data).decode("ascii")}},
                {"type": "text", "text": prompt}], "pdf")
        images, why = _render_pdf_pages(data)
        if not images:
            raise ValueError(why)
        return images + [{"type": "text", "text": prompt}], "images"
    jpg = _jpeg(_open_image(data), MODEL_LONG_SIDE, 88)
    return ([{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(jpg).decode("ascii")}},
             {"type": "text", "text": prompt}], "image")


def read_receipt_file(data: bytes, ctype: str, category_names: List[str],
                      api_key: Optional[str] = None) -> dict:
    """Ask the model what receipts the file holds. Never raises for a bad reading:
    the failure and its reason come back in the result."""
    import anthropic
    from flask_app.services.lease_review_service import EXTRACTION_MODEL, _render_pdf_pages
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set, so receipts cannot be read")
    prompt = RECEIPT_PROMPT.format(categories="\n".join("  " + n for n in category_names))
    content, route = _model_content(data, ctype, prompt)
    client = anthropic.Anthropic(api_key=key)

    def ask(cont):
        with client.messages.stream(model=EXTRACTION_MODEL, max_tokens=MAX_OUTPUT_TOKENS,
                                    messages=[{"role": "user", "content": cont}]) as s:
            return s.get_final_message()

    def body(msg):
        # NOT content[0]: the first block is a THINKING block on this model.
        return "".join(b.text for b in msg.content if b.type == "text")

    msg = ask(content)
    if msg.stop_reason != "refusal" and not re.search(r"\{[\s\S]*\}", body(msg)):
        # One retry. A PDF that yields nothing goes again as page images -- the
        # lease engine's measured remedy for a scan the PDF route will not read.
        if route == "pdf":
            images, _ = _render_pdf_pages(data)
            if images:
                content, route = images + [content[-1]], "images"
        msg = ask(content)
    if msg.stop_reason == "refusal":
        return {"_failed": True, "_source": route,
                "_reason": "the reader declined to read this file"}
    txt = body(msg)
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        try:
            out = json.loads(m.group())
            if isinstance(out.get("receipts"), list):
                out["_source"] = route
                return out
        except json.JSONDecodeError:
            pass
    return {"_failed": True, "_source": route,
            "_reason": ("the reading ran out of room" if msg.stop_reason == "max_tokens"
                        else "the reader returned no readable answer")}


def _num(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return round(float(str(v).replace(",", "").replace("$", "")), 2)
    except (TypeError, ValueError):
        return None


def _iso(v) -> Optional[str]:
    try:
        return ex._date(v, "date", required=False)
    except ValueError:
        return None


def extract(engine, actor, report_id, receipt_id, api_key=None) -> dict:
    """Read one stored file and propose a line for each receipt in it.

    Refused once lines point at the file: a re-read would either duplicate them
    or overwrite what the employee settled. Remove those lines to read it again.
    """
    ensure_tables(engine)
    r = ex._owned_editable(engine, actor, report_id)
    with engine.connect() as c:
        row = c.execute(text("SELECT content_type, file_data FROM er_receipts "
                             "WHERE id = :i AND report_id = :r"),
                        {"i": int(receipt_id), "r": r["id"]}).first()
        linked = c.execute(text("SELECT COUNT(*) FROM er_lines WHERE receipt_id = :i"),
                           {"i": int(receipt_id)}).scalar()
    if not row:
        raise LookupError("No receipt %s on report %s." % (receipt_id, r["id"]))
    if linked:
        raise PermissionError("%d line(s) already come from this receipt. Remove them to "
                              "read it again." % linked)
    opts = ex.options(engine)
    cat_by_name = {c["name"].lower(): c["account"] for c in opts["categories"]}
    try:
        got = read_receipt_file(bytes(row[1]), row[0], [c["name"] for c in opts["categories"]],
                                api_key=api_key)
    except ValueError as e:
        got = {"_failed": True, "_source": None, "_reason": str(e)}
    now = ex._now()
    if got.get("_failed"):
        # Committed BEFORE the report is read back: built inside the transaction,
        # the reply showed the receipt still unread while the row said otherwise.
        with engine.begin() as c:
            c.execute(text("UPDATE er_receipts SET status = 'error', error = :e, "
                           "extraction_source = :s, extracted_at = :at WHERE id = :i"),
                      {"e": got["_reason"], "s": got.get("_source"), "at": now,
                       "i": int(receipt_id)})
        return ex.get_report(engine, actor, r["id"])
    with engine.begin() as c:
        found = got.get("receipts") or []
        c.execute(text("UPDATE er_receipts SET status = :st, error = :e, extraction_json = :j, "
                       "extraction_source = :s, extracted_at = :at WHERE id = :i"),
                  {"st": "read" if found else "no_receipt",
                   "e": None if found else (got.get("notes") or "no receipt was found in the file"),
                   "j": json.dumps(got), "s": got.get("_source"), "at": now,
                   "i": int(receipt_id)})
        n = c.execute(text("SELECT COALESCE(MAX(sort_order), 0) FROM er_lines "
                           "WHERE report_id = :r"), {"r": r["id"]}).scalar() or 0
        for k, rc in enumerate(found, start=1):
            read = {
                "date": _iso(rc.get("date")), "date_end": _iso(rc.get("date_end")),
                "vendor": (rc.get("vendor") or "").strip() or None,
                "total": _num(rc.get("total")), "printed_total": _num(rc.get("printed_total")),
                "tip": _num(rc.get("tip")), "tax": _num(rc.get("tax")),
                "currency": (rc.get("currency") or "USD").upper()[:3],
                "card_last4": rc.get("card_last4"),
                "handwritten_amount": bool(rc.get("handwritten_amount")),
                "amount_note": rc.get("amount_note"),
                "description": rc.get("description"), "category": rc.get("category"),
                "page": int(rc.get("page") or 1),
            }
            c.execute(text(
                "INSERT INTO er_lines (report_id, sort_order, line_date, line_date_end, "
                "category_account, vendor, comment, amount, receipt, receipt_id, receipt_page, "
                "extracted_json, updated_at) VALUES (:rep, :so, :d, :de, :cat, :v, :cm, :amt, "
                "'Y', :rid, :pg, :x, :at)"),
                {"rep": r["id"], "so": int(n) + k, "d": read["date"], "de": read["date_end"],
                 "cat": cat_by_name.get((read["category"] or "").lower()),
                 "v": read["vendor"], "cm": read["description"], "amt": read["total"],
                 "rid": int(receipt_id), "pg": read["page"], "x": json.dumps(read), "at": now})
    return ex.get_report(engine, actor, r["id"])
