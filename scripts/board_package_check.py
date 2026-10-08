"""Guardrail: the board package's footnotes, disclosures and attachments.

Run: .venv\\Scripts\\python scripts\\board_package_check.py

On a throwaway SQLite database:
1. Footnotes: NULL means "the page's defaults", a list REPLACES them ([] = none),
   reset returns to the defaults; blank lines dropped; limits refused with the reason.
2. Attachments: the file's own bytes decide what it is (a text file named .png is
   refused); an image is one page with its size; a PDF is one rendered page per
   PDF page; too many pages refused; order moves; delete removes its pages too.
3. Nothing is written once the meeting is no longer a draft; every write is logged.
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text  # noqa: E402

from flask_app.services import board_package_service as pkg  # noqa: E402
from flask_app.services import board_service as board  # noqa: E402

PASSED, FAILED = 0, []


def chk(label, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print("   ok  ", label)
    else:
        FAILED.append(label)
        print("   FAIL", label, detail)


def refused(fn, needle):
    """True when ``fn`` is REFUSED with a message naming ``needle``. Any other
    exception -- a crash -- is a failure, not a refusal."""
    try:
        fn()
    except (ValueError, PermissionError, LookupError) as e:
        return needle.lower() in str(e).lower()
    except Exception:
        return False
    return False


def run(fn):
    """The result, or the exception a crash raised -- so a crash fails the check
    beside it instead of stopping the guardrail."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return e


tmp = tempfile.mkdtemp()
engine = create_engine("sqlite:///" + os.path.join(tmp, "pkg.db"))
mid = board.create_meeting("Q4 Board", "2026-01-14", "2025-12-31", "builder", engine=engine)["id"]

print("\n1. Footnotes and disclosures")
chk("a page nobody edited has no row: its defaults apply", pkg.page_notes(mid, engine) == {})
r = pkg.save_page_notes(mid, "capitalization", ["First", "  ", "Second"], "Gross returns.", "ed", engine)
chk("a saved list replaces the defaults, blank lines dropped", r["footnotes"] == ["First", "Second"], r)
chk("...with the disclosure", r["disclosure"] == "Gross returns.")
r = run(lambda: pkg.save_page_notes(mid, "capitalization", [], None, "ed", engine))
chk("an EMPTY list means no footnotes (not the defaults)", isinstance(r, dict) and r["footnotes"] == [], r)
r = pkg.save_page_notes(mid, "performance", None, "Only a disclosure", "ed", engine)
chk("null footnotes keep the defaults while a disclosure is saved",
    r["footnotes"] is None and r["disclosure"] == "Only a disclosure")
pkg.reset_page_notes(mid, "capitalization", "ed", engine)
chk("reset returns the page to its defaults (the row is gone)", "capitalization" not in pkg.page_notes(mid, engine))
chk("too many footnotes are refused, with the limit",
    refused(lambda: pkg.save_page_notes(mid, "x", ["a"] * (pkg.MAX_FOOTNOTES + 1), None, "ed", engine), "at most"))
chk("an over-long footnote is refused, naming which",
    refused(lambda: pkg.save_page_notes(mid, "x", ["ok", "z" * (pkg.MAX_FOOTNOTE_CHARS + 1)], None, "ed", engine),
            "footnote 2"))
chk("a page key is required", refused(lambda: pkg.save_page_notes(mid, " ", ["a"], None, "ed", engine), "page key"))

print("\n2. Attachments")
import base64  # noqa: E402
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
a = pkg.add_attachment(mid, "strategy", "chart.png", PNG, "image/png", "A chart", "ed", engine)
chk("an image is one page, with its size", a["page_count"] == 1 and a["pages"] == [{"n": 1, "width": 1, "height": 1}], a)
chk("its page is served back as the image", pkg.attachment_page(mid, a["id"], 1, engine) == ("image/png", PNG))
chk("a text file NAMED .png and declared image/png is refused by its bytes",
    refused(lambda: pkg.add_attachment(mid, "strategy", "x.png", b"not an image", "image/png", "", "ed", engine),
            "is neither"))
CORRUPT = PNG[:16] + b"\x00" * 40
chk("a file that STARTS like a PNG but is not one is refused with the reason, not a server error",
    refused(lambda: pkg.add_attachment(mid, "strategy", "bad.png", CORRUPT, "image/png", "", "ed", engine),
            "could not be read"))
chk("...and a broken PDF likewise",
    refused(lambda: pkg.add_attachment(mid, "strategy", "bad.pdf", b"%PDF-1.7 garbage", "application/pdf", "", "ed",
                                       engine), "could not be read"))
chk("an unknown section is refused", refused(lambda: pkg.add_attachment(mid, "nope", "c.png", PNG, "", "", "ed",
                                                                        engine), "unknown narrative"))
import pymupdf  # noqa: E402
doc = pymupdf.open()
for i in range(3):
    doc.new_page(width=612, height=792).insert_text((72, 72), "Schedule page %d" % (i + 1))
pdf = doc.tobytes()
b = pkg.add_attachment(mid, "strategy", "schedule.pdf", pdf, "application/pdf", "", "ed", engine)
chk("a PDF is one rendered page per PDF page", b["page_count"] == 3 and len(b["pages"]) == 3, b)
chk("...rendered as PNG at the set resolution (612pt at %d dpi)" % pkg.RENDER_DPI,
    b["pages"][0]["width"] == round(612 * pkg.RENDER_DPI / 72)
    and pkg.attachment_page(mid, b["id"], 2, engine)[0] == "image/png", b["pages"][0])
big = pymupdf.open()
for i in range(pkg.MAX_PDF_PAGES + 1):
    big.new_page()
chk("a PDF over the page limit is refused, saying split it",
    refused(lambda: pkg.add_attachment(mid, "strategy", "big.pdf", big.tobytes(), "", "", "ed", engine), "split"))
order = [x["id"] for x in pkg.attachments(mid, engine)["strategy"]]
chk("attachments list in the order added", order == [a["id"], b["id"]], order)
pkg.update_attachment(mid, b["id"], {"move": -1, "caption": "Schedule"}, "ed", engine)
now = pkg.attachments(mid, engine)["strategy"]
chk("move up swaps it with the one before; the caption is saved",
    [x["id"] for x in now] == [b["id"], a["id"]] and now[0]["caption"] == "Schedule")
pkg.update_attachment(mid, b["id"], {"move": -1}, "ed", engine)
chk("moving the first up changes nothing", [x["id"] for x in pkg.attachments(mid, engine)["strategy"]] == [b["id"], a["id"]])
pkg.delete_attachment(mid, b["id"], "ed", engine)
with engine.connect() as c:
    left = c.execute(text("SELECT COUNT(*) FROM board_attachment_pages WHERE attachment_id = :a"), {"a": b["id"]}).scalar()
chk("delete removes the attachment AND its pages", left == 0 and len(pkg.attachments(mid, engine)["strategy"]) == 1)
chk("the meeting payload carries attachments and page notes",
    board.get_meeting(mid, engine)["narratives"][1]["attachments"][0]["id"] == a["id"]
    and "performance" in board.get_meeting(mid, engine)["page_notes"])
chk("another meeting's attachment cannot be reached through this one",
    pkg.attachment_page(mid + 1, a["id"], 1, engine) is None
    and refused(lambda: pkg.delete_attachment(mid + 1, a["id"], "ed", engine), "not found"))

print("\n3. Drafts only, and logged")
with engine.begin() as c:
    c.execute(text("UPDATE board_meetings SET status = 'approved' WHERE id = :m"), {"m": mid})
chk("footnotes cannot be changed once the meeting is not a draft",
    refused(lambda: pkg.save_page_notes(mid, "capitalization", ["x"], None, "ed", engine), "no longer be edited"))
chk("...nor attachments added or deleted",
    refused(lambda: pkg.add_attachment(mid, "strategy", "c.png", PNG, "", "", "ed", engine), "no longer be edited")
    and refused(lambda: pkg.delete_attachment(mid, a["id"], "ed", engine), "no longer be edited"))
with engine.connect() as c:
    acts = {r[0] for r in c.execute(text("SELECT action FROM access_audit"))}
chk("every kind of write is in the access log",
    {"board_page_notes_saved", "board_page_notes_reset", "board_attachment_added", "board_attachment_updated",
     "board_attachment_deleted"} <= acts, acts)
import database  # noqa: E402
chk("the package tables are PROTECTED (app-written, the only copy)",
    {"board_page_notes", "board_attachments", "board_attachment_pages", "board_meetings"} <= database.PROTECTED_TABLES)

print("\n%d passed, %d failed" % (PASSED, len(FAILED)))
for f in FAILED:
    print("  -", f)
sys.exit(1 if FAILED else 0)
