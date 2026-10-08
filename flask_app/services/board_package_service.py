"""The board package around the figures: each page's footnotes and disclosure,
and the attachments placed in a narrative section (Oct 8 2026).

FOOTNOTES ARE THE PAGE'S OWN DEFAULTS UNTIL SOMEONE EDITS THEM. A page's
default footnotes come with its view (the engines know what their figures
mean); a saved list REPLACES them for that page of that meeting, and "reset"
deletes the row so the defaults return -- including any later change to them.
``footnotes`` NULL therefore means "use the defaults" and ``[]`` means "none".

ATTACHMENTS are images, or PDFs rendered to one image per page at upload, so a
schedule from an outside source drops into a narrative section as pages. The
original file is kept; what the deck shows is the rendered page.

Every write is refused once the meeting is no longer a draft, and logged.
"""
from __future__ import annotations

import io
import json
from typing import Optional

from sqlalchemy import text

from flask_app.services import board_service as board

#: A page key: a schedule ("capitalization"), one page of a multi-page schedule
#: ("investment_summaries:1"), a narrative section ("n:strategy"), "cover", "contents".
MAX_KEY = 80
MAX_FOOTNOTES = 12
MAX_FOOTNOTE_CHARS = 600
MAX_DISCLOSURE_CHARS = 2000

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 30
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}
PDF_TYPE = "application/pdf"
#: PDF pages are rendered at this resolution: sharp on a projector, light enough to store.
RENDER_DPI = 150

_READY: set = set()


def ensure(engine=None):
    engine = board.ensure(engine)
    if id(engine) in _READY:
        return engine
    pg = engine.dialect.name == "postgresql"
    pk = "id SERIAL PRIMARY KEY" if pg else "id INTEGER PRIMARY KEY AUTOINCREMENT"
    blob = "BYTEA" if pg else "BLOB"
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS board_page_notes (
                meeting_id INTEGER NOT NULL,
                page_key TEXT NOT NULL,
                footnotes TEXT,
                disclosure TEXT,
                updated_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (meeting_id, page_key)
            )"""))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS board_attachments (
                {pk},
                meeting_id INTEGER NOT NULL,
                block_key TEXT NOT NULL,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                caption TEXT,
                sort_order INTEGER NOT NULL DEFAULT 0,
                page_count INTEGER NOT NULL,
                data {blob} NOT NULL,
                uploaded_by TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS board_attachment_pages (
                attachment_id INTEGER NOT NULL,
                page_no INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                data {blob} NOT NULL,
                PRIMARY KEY (attachment_id, page_no)
            )"""))
    _READY.add(id(engine))
    return engine


def _log(actor: str, action: str, detail: dict, engine) -> None:
    from flask_app.auth import audit
    audit.log(actor, action, detail=detail, engine=engine)


def _check_key(page_key: str) -> str:
    k = (page_key or "").strip()
    if not k or len(k) > MAX_KEY:
        raise ValueError("A page key is required (at most %d characters)" % MAX_KEY)
    return k


# ------------------------------------------------------------------ footnotes and disclosures

def page_notes(meeting_id: int, engine=None) -> dict:
    """{page_key: {footnotes: list|None, disclosure: str|None, updated_by, updated_at}}."""
    engine = ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT page_key, footnotes, disclosure, updated_by, updated_at "
                                 "FROM board_page_notes WHERE meeting_id = :m"),
                            {"m": int(meeting_id)}).mappings().fetchall()
    return {r["page_key"]: {"footnotes": None if r["footnotes"] is None else json.loads(r["footnotes"]),
                            "disclosure": r["disclosure"], "updated_by": r["updated_by"],
                            "updated_at": str(r["updated_at"]) if r["updated_at"] else None}
            for r in rows}


def save_page_notes(meeting_id: int, page_key: str, footnotes, disclosure, actor: str, engine=None) -> dict:
    """Save a page's footnotes (a list REPLACES the defaults; None keeps the defaults)
    and its disclosure. Blank lines are dropped; an over-long entry is refused."""
    engine = ensure(engine)
    board._require_draft(meeting_id, engine)
    key = _check_key(page_key)
    notes = None
    if footnotes is not None:
        if not isinstance(footnotes, list):
            raise ValueError("Footnotes must be a list of lines")
        notes = [str(f).strip() for f in footnotes if str(f or "").strip()]
        if len(notes) > MAX_FOOTNOTES:
            raise ValueError("At most %d footnotes on a page" % MAX_FOOTNOTES)
        long = [i + 1 for i, f in enumerate(notes) if len(f) > MAX_FOOTNOTE_CHARS]
        if long:
            raise ValueError("Footnote %s is longer than %d characters" % (long[0], MAX_FOOTNOTE_CHARS))
    disc = (disclosure or "").strip() or None
    if disc and len(disc) > MAX_DISCLOSURE_CHARS:
        raise ValueError("The disclosure is longer than %d characters" % MAX_DISCLOSURE_CHARS)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM board_page_notes WHERE meeting_id = :m AND page_key = :k"),
                     {"m": int(meeting_id), "k": key})
        if notes is not None or disc is not None:
            conn.execute(text("INSERT INTO board_page_notes (meeting_id, page_key, footnotes, disclosure, "
                              "updated_by) VALUES (:m, :k, :f, :d, :by)"),
                         {"m": int(meeting_id), "k": key, "f": None if notes is None else json.dumps(notes),
                          "d": disc, "by": actor})
    _log(actor, "board_page_notes_saved",
         {"meeting_id": int(meeting_id), "page": key,
          "footnotes": "default" if notes is None else len(notes), "disclosure": bool(disc)}, engine)
    return page_notes(meeting_id, engine).get(key) or {"footnotes": None, "disclosure": None}


def reset_page_notes(meeting_id: int, page_key: str, actor: str, engine=None) -> None:
    """Back to the page's default footnotes and no disclosure."""
    engine = ensure(engine)
    board._require_draft(meeting_id, engine)
    key = _check_key(page_key)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM board_page_notes WHERE meeting_id = :m AND page_key = :k"),
                     {"m": int(meeting_id), "k": key})
    _log(actor, "board_page_notes_reset", {"meeting_id": int(meeting_id), "page": key}, engine)


# ------------------------------------------------------------------ attachments

def _sniff(data: bytes, declared: str) -> Optional[str]:
    """The content type by the file's own bytes; the declared type is not trusted."""
    if data[:5] == b"%PDF-":
        return PDF_TYPE
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _render(data: bytes, ctype: str) -> list:
    """[(content_type, width, height, bytes)] -- one per page shown. A file that
    starts like an image or a PDF but cannot be read is REFUSED with the reason,
    never a server error."""
    try:
        return _render_pages(data, ctype)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("That file could not be read as %s (%s)"
                         % ("a PDF" if ctype == PDF_TYPE else "an image", type(e).__name__))


def _render_pages(data: bytes, ctype: str) -> list:
    if ctype == PDF_TYPE:
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        if doc.page_count > MAX_PDF_PAGES:
            raise ValueError("That PDF has %d pages; at most %d can be attached -- split it"
                             % (doc.page_count, MAX_PDF_PAGES))
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=RENDER_DPI)
            pages.append(("image/png", pix.width, pix.height, pix.tobytes("png")))
        if not pages:
            raise ValueError("That PDF has no pages")
        return pages
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im.load()                       # a header alone is not an image: read the pixels
        w, h = im.size
    return [(ctype, w, h, data)]


def attachments(meeting_id: int, engine=None) -> dict:
    """{block_key: [{id, filename, caption, sort_order, page_count, pages: [{n, width, height}]}]}"""
    engine = ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, block_key, filename, content_type, caption, sort_order, page_count, "
                                 "uploaded_by, uploaded_at FROM board_attachments WHERE meeting_id = :m "
                                 "ORDER BY block_key, sort_order, id"), {"m": int(meeting_id)}).mappings().fetchall()
        dims = {}
        if rows:
            for p in conn.execute(text(
                    "SELECT p.attachment_id, p.page_no, p.width, p.height FROM board_attachment_pages p "
                    "JOIN board_attachments a ON a.id = p.attachment_id WHERE a.meeting_id = :m "
                    "ORDER BY p.attachment_id, p.page_no"), {"m": int(meeting_id)}).mappings():
                dims.setdefault(p["attachment_id"], []).append(
                    {"n": p["page_no"], "width": p["width"], "height": p["height"]})
    out: dict = {}
    for r in rows:
        out.setdefault(r["block_key"], []).append({
            "id": r["id"], "filename": r["filename"], "content_type": r["content_type"],
            "caption": r["caption"] or "", "sort_order": r["sort_order"], "page_count": r["page_count"],
            "pages": dims.get(r["id"], []), "uploaded_by": r["uploaded_by"],
            "uploaded_at": str(r["uploaded_at"]) if r["uploaded_at"] else None})
    return out


def add_attachment(meeting_id: int, block_key: str, filename: str, data: bytes, declared_type: str,
                   caption: str, actor: str, engine=None) -> dict:
    if block_key not in board.NARRATIVE_KEYS:
        raise ValueError("Unknown narrative section %r" % block_key)
    engine = ensure(engine)
    board._require_draft(meeting_id, engine)
    if not data:
        raise ValueError("The file is empty")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("That file is %.1f MB; the limit is %d MB" % (len(data) / 1e6, MAX_FILE_BYTES // (1024 * 1024)))
    ctype = _sniff(data, declared_type)
    if ctype is None:
        raise ValueError("Attach an image (PNG, JPEG, GIF, WebP) or a PDF; %r is neither" % (filename or "the file"))
    pages = _render(data, ctype)
    with engine.begin() as conn:
        order = conn.execute(text("SELECT COALESCE(MAX(sort_order), 0) + 1 FROM board_attachments "
                                  "WHERE meeting_id = :m AND block_key = :k"),
                             {"m": int(meeting_id), "k": block_key}).scalar()
        aid = conn.execute(text(
            "INSERT INTO board_attachments (meeting_id, block_key, filename, content_type, caption, sort_order, "
            "page_count, data, uploaded_by) VALUES (:m, :k, :f, :t, :c, :o, :n, :d, :by) RETURNING id"),
            {"m": int(meeting_id), "k": block_key, "f": (filename or "attachment")[:200], "t": ctype,
             "c": (caption or "").strip()[:300] or None, "o": int(order), "n": len(pages), "d": data,
             "by": actor}).scalar()
        for i, (pt, w, h, b) in enumerate(pages, start=1):
            conn.execute(text("INSERT INTO board_attachment_pages (attachment_id, page_no, content_type, width, "
                              "height, data) VALUES (:a, :n, :t, :w, :h, :d)"),
                         {"a": aid, "n": i, "t": pt, "w": int(w), "h": int(h), "d": b})
    _log(actor, "board_attachment_added", {"meeting_id": int(meeting_id), "block": block_key, "attachment": aid,
                                           "filename": filename, "pages": len(pages)}, engine)
    return next(a for a in attachments(meeting_id, engine).get(block_key, []) if a["id"] == aid)


def _attachment_row(meeting_id: int, attachment_id: int, conn):
    return conn.execute(text("SELECT id, block_key FROM board_attachments WHERE id = :a AND meeting_id = :m"),
                        {"a": int(attachment_id), "m": int(meeting_id)}).mappings().fetchone()


def update_attachment(meeting_id: int, attachment_id: int, fields: dict, actor: str, engine=None) -> dict:
    """Caption and / or position: {"caption": str} | {"move": -1 | 1}."""
    engine = ensure(engine)
    board._require_draft(meeting_id, engine)
    with engine.begin() as conn:
        row = _attachment_row(meeting_id, attachment_id, conn)
        if row is None:
            raise LookupError("Attachment %s not found" % attachment_id)
        if "caption" in fields:
            conn.execute(text("UPDATE board_attachments SET caption = :c WHERE id = :a"),
                         {"c": (fields.get("caption") or "").strip()[:300] or None, "a": int(attachment_id)})
        if fields.get("move") in (-1, 1):
            ids = [r[0] for r in conn.execute(text(
                "SELECT id FROM board_attachments WHERE meeting_id = :m AND block_key = :k ORDER BY sort_order, id"),
                {"m": int(meeting_id), "k": row["block_key"]})]
            i = ids.index(int(attachment_id))
            j = i + int(fields["move"])
            if 0 <= j < len(ids):
                ids[i], ids[j] = ids[j], ids[i]
                for pos, x in enumerate(ids, start=1):
                    conn.execute(text("UPDATE board_attachments SET sort_order = :o WHERE id = :a"), {"o": pos, "a": x})
    _log(actor, "board_attachment_updated", {"meeting_id": int(meeting_id), "attachment": int(attachment_id),
                                             "fields": sorted(fields)}, engine)
    return attachments(meeting_id, engine)


def delete_attachment(meeting_id: int, attachment_id: int, actor: str, engine=None) -> None:
    engine = ensure(engine)
    board._require_draft(meeting_id, engine)
    with engine.begin() as conn:
        row = _attachment_row(meeting_id, attachment_id, conn)
        if row is None:
            raise LookupError("Attachment %s not found" % attachment_id)
        conn.execute(text("DELETE FROM board_attachment_pages WHERE attachment_id = :a"), {"a": int(attachment_id)})
        conn.execute(text("DELETE FROM board_attachments WHERE id = :a"), {"a": int(attachment_id)})
    _log(actor, "board_attachment_deleted", {"meeting_id": int(meeting_id), "attachment": int(attachment_id)}, engine)


def attachment_page(meeting_id: int, attachment_id: int, page_no: int, engine=None):
    """(content_type, bytes) of one shown page, or None."""
    engine = ensure(engine)
    with engine.connect() as conn:
        r = conn.execute(text(
            "SELECT p.content_type, p.data FROM board_attachment_pages p JOIN board_attachments a "
            "ON a.id = p.attachment_id WHERE a.meeting_id = :m AND p.attachment_id = :a AND p.page_no = :n"),
            {"m": int(meeting_id), "a": int(attachment_id), "n": int(page_no)}).fetchone()
    return None if r is None else (r[0], bytes(r[1]))
