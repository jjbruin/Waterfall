"""Print setup for the statement tabs, taken from the real package.

THE SOURCE IS A REAL DELIVERED PACKAGE, not a preference:
``PPI Eastchase (TX) LLC - WP - 06.30.2026.xlsx``, 37 tabs. Jim, Sep 16 2026:
"Tabs, Cover, Index, Balance Sheet, SOI, IS, Member's Capital, and Cash Flow all
are the final financial statement products and have print ranges." Every number
below was read out of that workbook rather than chosen here.

WHAT WAS THERE, exactly:

  tab                print area   scale  margins L/R/T/B     centred   footer
  Cover              A1:D15         87   .65/.65/.75/.75     vertical    -
  Index              A1:D15        fit   .65/.65/2.25/.75    horizontal  -
  Balance Sheet      A1:C25         99   1/1/2/.75           horizontal  1
  SOI                A1:J11         61   1/1/2/.75           horizontal  2
  IS                 A1:C22         93   1/1/2/.75           horizontal  3
  Members' Capital   A1:I12         78   1/1/2/.75           horizontal  4
  Cash Flow          A1:C23        fit   1/1/2/.75           horizontal  5

All portrait, all ``fitToPage``, header and footer inset 0.30. Body type is
Arial 11, headings and totals Arial 11 bold; the Cover title is Arial 24 bold.

THE ONE THING NOT COPIED LITERALLY IS THE PRINT RANGE. Those addresses are fixed
because that workbook's data is fixed -- one entity, one period, a known number
of members and investments. Ours is generated for 58 entities whose statements
are not the same height, so a hardcoded ``A1:C25`` would cut the bottom off a
longer balance sheet and print blank rows under a shorter one. The range is
therefore computed from what was actually written, and the COLUMN span is taken
from the reference. That preserves what the fixed address was for -- the sheet
prints its statement and nothing to the right of it -- without inheriting a row
count that was only ever true for Eastchase.

AND THE SCALE IS NOT COPIED EITHER, for the same reason and one more: in OOXML
``fitToPage`` and ``scale`` are contradictory, and when ``fitToPage`` is set
Excel uses ``fitToWidth``/``fitToHeight`` and ignores the percentage. The 87, 99,
61, 93 and 78 in that workbook are leftovers from before someone ticked "fit to
page"; they are not what Excel obeys when it prints. One page wide by one page
tall is what it obeys, which is also what the footer page numbers 1 to 5 assert.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)

#: Body and heading type, from the reference workbook.
BODY_FONT = "Arial"
BODY_SIZE = 11
COVER_TITLE_SIZE = 24

#: Accounting formats as the reference carries them. `[Black]` forces negatives
#: black rather than red -- a financial statement shows a negative in
#: parentheses, not in colour -- and the third clause renders zero as a dash.
FMT_WHOLE = '[Black]#,###_);[Black]\\(#,###\\);[Black]"-"'
FMT_WHOLE_DOLLAR = '[Black]"$"* #,###_);[Black]"$"* \\(#,###\\);[Black]"$"* "-"'
FMT_ACCOUNTING = '_(* #,##0_);_(* \\(#,##0\\);_(* "-"_);_(@_)'
FMT_ACCOUNTING_DOLLAR = '_("$"* #,##0_);_("$"* \\(#,##0\\);_("$"* "-"_);_(@_)'
FMT_PCT = "0.00%"

#: Per-tab setup. `cols` is the column span the print range covers, taken from
#: the reference's own print area; `widths` are its column widths.
#: `title` is the statement's formal name as the reference's header carries it,
#: and `dated` decides which of the two period phrases follows it.
SHEETS = {
    "Cover": {
        "cols": 4, "page": None, "title": None, "dated": None,
        "margins": (0.65, 0.65, 0.75, 0.75), "centre": "vertical",
        "widths": {"A": 74.4, "B": 10.0, "C": 11.6, "D": 8.4},
        "header": False, "body_from": 1,
    },
    "Index": {
        "cols": 4, "page": None, "title": None, "dated": None,
        "margins": (0.65, 0.65, 2.25, 0.75), "centre": "horizontal",
        "widths": {"A": 42.0, "B": 18.3, "C": 23.0, "D": 5.6},
        "header": True, "header_title": None, "body_from": 1,
    },
    "Balance Sheet": {
        "cols": 3, "page": 1, "dated": "as_of",
        "title": "Statement of Assets, Liabilities and Members' Capital",
        "margins": (1.0, 1.0, 2.0, 0.75), "centre": "horizontal",
        "widths": {"A": 4.3, "B": 61.6, "C": 17.7},
        "header": True, "body_from": 4,
    },
    "SOI": {
        "cols": 10, "page": 2, "dated": "as_of",
        "title": "Schedule of Investment",
        "margins": (1.0, 1.0, 2.0, 0.75), "centre": "horizontal",
        "widths": {"A": 2.7, "D": 43.7, "E": 20.0, "F": 19.7, "G": 1.3,
                   "H": 19.7, "I": 1.3, "J": 21.3},
        "header": True, "body_from": 4,
    },
    "Income Statement": {
        "cols": 3, "page": 3, "dated": "period",
        "title": "Statement of Operations",
        "margins": (1.0, 1.0, 2.0, 0.75), "centre": "horizontal",
        "widths": {"A": 2.4, "B": 68.0, "C": 17.7},
        "header": True, "body_from": 4,
    },
    "Members Capital": {
        "cols": 9, "page": 4, "dated": "period",
        "title": "Statement of Changes in Members' Capital",
        "margins": (1.0, 1.0, 2.0, 0.75), "centre": "horizontal",
        "widths": {"A": 42.6, "B": 3.6, "C": 14.3, "D": 1.0, "E": 14.3,
                   "F": 1.0, "G": 13.6, "H": 1.0, "I": 13.6},
        "header": True, "body_from": 4,
    },
    "Cash Flow": {
        "cols": 3, "page": 5, "dated": "period",
        "title": "Statement of Cash Flows",
        "margins": (1.0, 1.0, 2.0, 0.75), "centre": "horizontal",
        "widths": {"A": 61.0, "B": 4.7, "C": 13.6},
        "header": True, "body_from": 4,
    },
}

_MONTHS_WORD = {
    1: "one month", 2: "two months", 3: "three months", 4: "four months",
    5: "five months", 6: "six months", 7: "seven months", 8: "eight months",
    9: "nine months", 10: "ten months", 11: "eleven months", 12: "year",
}


def _long_date(period_end) -> str:
    """``June 30, 2026``. Returns the input unchanged if it will not parse."""
    from datetime import date, datetime
    d = period_end
    if isinstance(d, str):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                d = datetime.strptime(d[:len(fmt) + 2], fmt)
                break
            except ValueError:
                continue
    if not isinstance(d, (date, datetime)):
        return str(period_end)
    return "%s %d, %d" % (d.strftime("%B"), d.day, d.year)


def period_phrase(period_end, dated: str) -> str:
    """``As of June 30, 2026`` or ``For the six months ended June 30, 2026``.

    Which one is a property of the STATEMENT, not a formatting choice: a balance
    sheet and a schedule of investments are as of an instant, while operations,
    members' capital and cash flows cover a span. Getting it the wrong way round
    would put a false assertion in 11-point type at the top of an investor
    document.
    """
    from datetime import date, datetime
    long = _long_date(period_end)
    if dated == "as_of":
        return "As of %s" % long
    d = period_end
    if isinstance(d, str):
        try:
            d = datetime.fromisoformat(d[:10])
        except ValueError:
            d = None
    month = d.month if isinstance(d, (date, datetime)) else None
    if month == 12:
        return "For the year ended %s" % long
    word = _MONTHS_WORD.get(month)
    if not word:
        # Unknown period length: say the date and NOT a span we cannot support.
        return "For the period ended %s" % long
    return "For the %s ended %s" % (word, long)


def entity_subtitle(entity_name: str) -> Optional[str]:
    """``(A Limited Liability Company)``, but only when it IS one.

    The reference carries that line because every entity in this population is
    an LLC. It is still derived rather than assumed: printing a legal form on a
    financial statement is an assertion about the entity, and inventing it for
    something that turns out to be a partnership would be wrong in the one place
    it would never be questioned.
    """
    n = (entity_name or "").strip().rstrip(".")
    if re.search(r"\bL\.?L\.?C\.?$", n, re.I):
        return "(A Limited Liability Company)"
    if re.search(r"\bL\.?P\.?$", n, re.I):
        return "(A Limited Partnership)"
    return None


def header_text(entity_name: str, statement_title: Optional[str],
                period_end, dated: Optional[str]) -> str:
    """The reference's five-line centred header block.

        PPI Eastchase (TX) LLC
        (A Limited Liability Company)
        <blank>
        Statement of Operations (Unaudited)
        <blank>
        For the six months ended June 30, 2026
    """
    lines = [entity_name or ""]
    sub = entity_subtitle(entity_name)
    if sub:
        lines.append(sub)
    if statement_title:
        lines += ["", "%s (Unaudited)" % statement_title]
    if dated:
        lines += ["", period_phrase(period_end, dated)]
    elif statement_title is None:
        # Index: the reference's header names the document, not a statement.
        lines += ["", "Financial Statements (Unaudited)", "",
                  period_phrase(period_end, "period")]
    return "\n".join(lines)


def apply(sheet, key: str, entity_name: str = "", period_end=None,
          last_row: Optional[int] = None) -> None:
    """Put one generated sheet on the reference's print setup.

    ``key`` is a name in :data:`SHEETS`; anything else is left alone rather than
    given a made-up layout.
    """
    spec = SHEETS.get(key)
    if not spec:
        return
    ps, pm, po = sheet.page_setup, sheet.page_margins, sheet.print_options

    ps.orientation = "portrait"
    # One page wide by one page tall, which is what the reference's footer
    # numbering asserts and what Excel actually obeys -- see the module note on
    # why its 87/99/61/93/78 percentages are not copied.
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    ps.fitToWidth = 1
    ps.fitToHeight = 1

    pm.left, pm.right, pm.top, pm.bottom = spec["margins"]
    pm.header = pm.footer = 0.30
    po.horizontalCentered = spec["centre"] == "horizontal"
    po.verticalCentered = spec["centre"] == "vertical"

    # COLUMN WIDTHS ARE DELIBERATELY NOT SET HERE ANY MORE.
    #
    # They were, from the reference workbook, and it made the statements
    # illegible (Jim, Sep 17 2026: "Column widths and spacing are making the
    # reports illegible"). Two compounding mistakes:
    #
    #   1. THE LAYOUTS ARE NOT THE SAME. Eastchase uses column A as a narrow
    #      indent and B for the line label; the generated statements put the
    #      label in A. So `A = 4.3` — an indent — was applied to the label
    #      column, and `B = 61.6` — a label column — to the amounts. The Income
    #      Statement got A = 2.4.
    #   2. IT OVERWROTE A BETTER ANSWER. `workpaper_excel._table` already sizes
    #      every column to the widest value in it, and `apply()` runs after the
    #      sheets are built, so the measured widths were replaced by transcribed
    #      ones that did not correspond to the same columns.
    #
    # Everything else here IS layout-independent — margins, centring, the page
    # header, the footer, orientation, fit-to-page — and stays. Widths are the
    # one part of that workbook that cannot be copied across without copying its
    # column layout too, which is a different piece of work.

    # THE RANGE IS COMPUTED, THE COLUMN SPAN IS THE REFERENCE'S. A fixed A1:C25
    # would cut a longer statement off and print blank rows under a shorter one;
    # 58 entities do not share a row count.
    #
    # AND IT STARTS BELOW THE WORKPAPER ROWS. Each generated statement carries
    # its name in A1 and a provenance note in A2 -- where the figures came from,
    # which accounts were dormant. Both earn their place in a WORKPAPER and
    # neither belongs on a statement sent to an investor, which is why the
    # reference puts the titling in the page header instead. So they stay in the
    # sheet and fall outside the print range: read on screen, absent on paper.
    start = int(spec.get("body_from", 1))
    end_row = last_row or sheet.max_row or 1
    if end_row < start:
        end_row = start
    col = get_column_letter(spec["cols"])
    sheet.print_area = "A%d:%s%d" % (start, col, end_row)

    if spec.get("header"):
        sheet.oddHeader.center.text = header_text(
            entity_name,
            spec.get("header_title", spec.get("title")),
            period_end, spec.get("dated"))
        sheet.oddHeader.center.size = BODY_SIZE
        sheet.oddHeader.center.font = BODY_FONT
    if spec.get("page"):
        # A LITERAL NUMBER, not `&P`. Each statement is its own sheet and prints
        # as one page, so `&P` would print "1" on all five. The reference has
        # the same literals for the same reason.
        sheet.oddFooter.right.text = str(spec["page"])
        sheet.oddFooter.right.size = BODY_SIZE
        sheet.oddFooter.right.font = BODY_FONT


def style_body(sheet, last_row: Optional[int] = None,
               max_col: int = 12) -> None:
    """Arial 11 throughout, keeping bold where the builder set it.

    Applied after the sheet is written so the type matches the reference without
    the builders each having to know about fonts.
    """
    end = last_row or sheet.max_row or 1
    for row in sheet.iter_rows(min_row=1, max_row=end, max_col=max_col):
        for c in row:
            f = c.font
            if f and (f.name != BODY_FONT or f.sz != BODY_SIZE):
                c.font = Font(name=BODY_FONT, sz=BODY_SIZE,
                              b=bool(f.b), i=bool(f.i), u=f.u,
                              color=f.color)


def cover_title(cell) -> None:
    """The Cover's Arial 24 bold title line."""
    cell.font = Font(name=BODY_FONT, sz=COVER_TITLE_SIZE, b=True)
    cell.alignment = Alignment(horizontal="center", vertical="center",
                               wrap_text=True)
