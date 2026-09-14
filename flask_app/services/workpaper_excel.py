"""Build the downloadable workpaper package.

One .xlsx per entity per period, in the order the example package's own Index
lists things: cover, index, the statements, then the supporting tabs, then the
accountant's exhibits.

EXHIBITS ARE PLACED, NOT ATTACHED. The point of the app holding them is that
the download arrives assembled -- an auditor opens one file, not a workbook
plus a folder. What that means per file type:

    .xlsx/.xls  every sheet is copied in as values, one tab per sheet
    .csv        parsed to a tab
    image       embedded on its own tab
    .pdf/other  listed on the Exhibits tab with its name, size and uploader,
                and shipped alongside -- openpyxl cannot inline a PDF, and
                pretending otherwise would lose the evidence

Every tab says where its numbers came from. A workpaper whose provenance is
"the app produced it" is not reviewable; one that names the table, the period
and the basis is.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Any, Dict, List

from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from flask_app.services import workpaper_data as wd
from flask_app.services import workpaper_service as ws

logger = logging.getLogger(__name__)

TITLE = Font(bold=True, size=14)
H1 = Font(bold=True, size=12)
HDR = Font(bold=True, color="FFFFFF")
HDR_FILL = PatternFill("solid", fgColor="1F3864")
SUB = Font(italic=True, size=9, color="666666")
MONEY = '#,##0.00;(#,##0.00)'
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif"}
SHEET_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def _safe_title(name: str, used: set) -> str:
    """Excel: 31 chars, no []:*?/\\ , and unique."""
    clean = "".join(ch for ch in str(name) if ch not in "[]:*?/\\")[:31] or "Sheet"
    base, i = clean, 2
    while clean.lower() in used:
        suffix = f" ({i})"
        clean = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(clean.lower())
    return clean


def _provenance(sheet, row: int, text_: str) -> int:
    sheet.cell(row=row, column=1, value=text_).font = SUB
    return row + 2


def _table(sheet, row: int, columns: List[str], rows: List[dict],
           money_cols: List[str] = None) -> int:
    money_cols = money_cols or []
    for j, c in enumerate(columns, start=1):
        cell = sheet.cell(row=row, column=j, value=c)
        cell.font = HDR
        cell.fill = HDR_FILL
        cell.border = BOX
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    row += 1
    for r in rows:
        for j, c in enumerate(columns, start=1):
            v = r.get(c)
            cell = sheet.cell(row=row, column=j, value=v)
            cell.border = BOX
            if c in money_cols and isinstance(v, (int, float)):
                cell.number_format = MONEY
        row += 1
    for j, c in enumerate(columns, start=1):
        width = max(len(str(c)) + 2,
                    *(len(str(r.get(c, ""))) + 2 for r in rows[:200])) if rows else len(str(c)) + 2
        sheet.column_dimensions[get_column_letter(j)].width = min(42, max(10, width))
    return row + 1


def build_package(package_id: int, engine=None) -> bytes:
    """Assemble the workbook for one package."""
    from flask_app.db import get_engine
    engine = engine or get_engine()

    detail = ws.package_detail(package_id, engine)
    pkg = detail["package"]
    entity = pkg["entityid"]
    period_end = pkg["period_end"]
    exhibits = ws.package_exhibits(package_id, engine)

    wb = Workbook()
    used: set = set()
    wb.remove(wb.active)

    _cover(wb, used, pkg)
    _index(wb, used, pkg, detail, exhibits)
    _statements(wb, used, entity, period_end, engine)
    _trial_balance(wb, used, entity, period_end, engine)
    _gl_detail(wb, used, entity, period_end, engine)
    _account_summary(wb, used, entity, period_end, engine)
    _capital_activity(wb, used, entity, period_end, engine)
    _exhibits(wb, used, exhibits)
    _signoffs(wb, used, detail)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _cover(wb, used, pkg):
    s = wb.create_sheet(_safe_title("Cover", used))
    s.column_dimensions["A"].width = 4
    s.column_dimensions["B"].width = 70
    s["B3"] = pkg.get("entity_name") or pkg["entityid"]
    s["B3"].font = Font(bold=True, size=18)
    s["B4"] = "(A Limited Liability Company)"
    s["B6"] = "Accounting Workpaper Package (Unaudited)"
    s["B6"].font = H1
    s["B8"] = f"As of and for the period ended {pkg['period_end']}"
    s["B10"] = f"Entity ID: {pkg['entityid']}"
    s["B11"] = f"Close cycle: {pkg['period_label']}"
    s["B12"] = f"Status: {pkg.get('state_label', pkg['state'])}"
    s["B14"] = f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} by Waterfall XIRR"
    s["B14"].font = SUB
    s["B15"] = ("Figures are produced from the MRI general ledger and investor "
                "activity tables held in the app. Supporting exhibits are the "
                "preparer's own and are reproduced as supplied.")
    s["B15"].font = SUB
    s["B15"].alignment = Alignment(wrap_text=True)


def _index(wb, used, pkg, detail, exhibits):
    s = wb.create_sheet(_safe_title("Index", used))
    s["A1"] = "Table of Contents"
    s["A1"].font = TITLE
    row = 3
    for name in ("Financial Statements", "Trial Balance", "GL Detail",
                 "Account Summary", "Investor Detail", "Investment Detail",
                 "IA Rollforward", "Commitments"):
        s.cell(row=row, column=1, value=name)
        row += 1
    row += 1
    s.cell(row=row, column=1, value="Exhibits").font = H1
    row += 1
    if exhibits:
        for e in exhibits:
            label = next((x["label"] for x in ws.EXHIBIT_SLOTS if x["key"] == e["slot_key"]),
                         e["slot_key"])
            s.cell(row=row, column=1, value=f"   {label} — {e['filename']}")
            row += 1
    else:
        c = s.cell(row=row, column=1, value="   (none attached)")
        c.font = SUB
        row += 1
    row += 1
    s.cell(row=row, column=1, value="Close checklist").font = H1
    row += 1
    for st in detail["steps"]:
        mark = "x" if st["done"] else " "
        due = f"  due {st['due_date']}" if st.get("due_date") else ""
        who = f"  {st['completed_by']} {st['completed_at']}" if st["done"] else ""
        s.cell(row=row, column=1, value=f"   [{mark}] {st['label']}{due}{who}")
        row += 1
    s.column_dimensions["A"].width = 90


def _statements(wb, used, entity, period_end, engine):
    fs = wd.financial_statements(entity, period_end, engine=engine)
    s = wb.create_sheet(_safe_title("Financial Statements", used))
    s["A1"] = "Financial Statements"
    s["A1"].font = TITLE
    row = _provenance(s, 2,
                      f"Trial balance through {fs['periods']['ytd_last']} folded through the "
                      f"account-to-FS-line mapping (wp_fs_map).")
    if not fs["statements"]:
        s.cell(row=row, column=1,
               value="No account mapping yet — map accounts to statement lines "
                     "to populate this tab.").font = SUB
        row += 2
    for stmt, lines in fs["statements"].items():
        s.cell(row=row, column=1, value=stmt).font = H1
        row += 1
        row = _table(s, row, ["fs_line", "ytd_change", "ytd_ending"],
                     [{"fs_line": l["fs_line"], "ytd_change": l["ytd_change"],
                       "ytd_ending": l["ytd_ending"]} for l in lines],
                     money_cols=["ytd_change", "ytd_ending"])
    if fs["unmapped"]:
        # Named, not hidden. An unmapped account is the statement's error bar.
        s.cell(row=row, column=1,
               value=f"UNMAPPED ACCOUNTS — {len(fs['unmapped'])} accounts, "
                     f"{fs['unmapped_total']:,.2f} not in any statement line above").font = H1
        row += 1
        row = _table(s, row, ["acctnum", "acctname", "ytd_ending"],
                     fs["unmapped"], money_cols=["ytd_ending"])


def _trial_balance(wb, used, entity, period_end, engine):
    tb = wd.trial_balance(entity, period_end, engine=engine)
    s = wb.create_sheet(_safe_title("Trial Balance", used))
    s["A1"] = "Trial Balance"
    s["A1"].font = TITLE
    p = tb["periods"]
    row = _provenance(s, 2,
                      f"gl_detail, entity {entity}, basis {'/'.join(tb.get('bases', []))}. "
                      f"YTD {p['ytd_first']}-{p['ytd_last']}, QTD {p['qtd_first']}-{p['qtd_last']}. "
                      f"Beginning = balance-forward rows at {p['ytd_first']}; change = period entries.")
    _table(s, row, ["acctnum", "acctname", "fs_statement", "fs_line",
                    "ytd_beginning", "ytd_change", "ytd_ending", "qtd_change"],
           tb["rows"], money_cols=["ytd_beginning", "ytd_change", "ytd_ending", "qtd_change"])


def _gl_detail(wb, used, entity, period_end, engine):
    rows = wd.gl_detail(entity, period_end, engine=engine)
    s = wb.create_sheet(_safe_title("GL Detail", used))
    s["A1"] = "GL Detail"
    s["A1"].font = TITLE
    row = _provenance(s, 2, f"gl_detail (JOURNAL + GHIS), entity {entity}, year to {period_end}.")
    cols = ["PERIOD", "ENTRDATE", "ACCTNUM", "ACCTNAME", "BASIS", "BALFOR",
            "ITEM", "REF", "DESCRPN", "RLTDENTITY", "RLTDENTITY_NAME", "AMT"]
    _table(s, row, cols, rows, money_cols=["AMT"])


def _account_summary(wb, used, entity, period_end, engine):
    rows = wd.account_summary(entity, period_end, engine=engine)
    s = wb.create_sheet(_safe_title("Account Summary", used))
    s["A1"] = "Account Summary"
    s["A1"].font = TITLE
    row = _provenance(s, 2,
                      "Period activity grouped by account — the shape the workbook's "
                      "Cash / Intercompany / Accruals / Expenses pivots each take.")
    _table(s, row, ["ACCTNUM", "ACCTNAME", "entries", "amount"], rows,
           money_cols=["amount"])


def _capital_activity(wb, used, entity, period_end, engine):
    inv = wd.investor_detail(entity, period_end, engine=engine)
    s = wb.create_sheet(_safe_title("Investor Detail", used))
    s["A1"] = "Investor Detail — who invested into this entity"
    s["A1"].font = TITLE
    row = _provenance(s, 2, f"ia_transactions where InvestmentID = {entity}, "
                            f"transactions through {period_end}.")
    _table(s, row, ["InvestorID", "InvestorName", "TransactionDate", "EffectiveDate",
                    "MajorType", "Typename", "Amount", "Period"], inv, money_cols=["Amount"])

    ivd = wd.investment_detail(entity, period_end, engine=engine)
    s2 = wb.create_sheet(_safe_title("Investment Detail", used))
    s2["A1"] = "Investment Detail — what this entity invested into"
    s2["A1"].font = TITLE
    row = _provenance(s2, 2, f"ia_transactions where InvestorID = {entity}, "
                             f"transactions through {period_end}.")
    _table(s2, row, ["InvestmentID", "InvestmentName", "TransactionDate", "EffectiveDate",
                     "MajorType", "Typename", "Amount", "Period"], ivd, money_cols=["Amount"])

    rf = wd.ia_rollforward(entity, period_end, engine=engine)
    s3 = wb.create_sheet(_safe_title("IA Rollforward", used))
    s3["A1"] = "IA Rollforward"
    s3["A1"].font = TITLE
    row = _provenance(s3, 2,
                      "ia_transactions, all major types INCLUDING non-cash ('Other'). "
                      "The app's accounting_feed filters those out, which is why this "
                      "tab is built from ia_transactions instead.")
    _table(s3, row, rf["columns"], rf["rows"],
           money_cols=["Beginning Balance", "Current Period", "Total"])

    com = wd.commitment_rollforward(entity, engine=engine)
    s4 = wb.create_sheet(_safe_title("Commitments", used))
    s4["A1"] = "Commitment Rollforward"
    s4["A1"].font = TITLE
    row = _provenance(s4, 2, "commitments (IA_Commitment, open commitments only).")
    if com:
        _table(s4, row, list(com[0].keys()), com)
    else:
        s4.cell(row=row, column=1, value="No open commitments for this entity.").font = SUB


def _exhibits(wb, used, exhibits):
    s = wb.create_sheet(_safe_title("Exhibits", used))
    s["A1"] = "Supporting Exhibits"
    s["A1"].font = TITLE
    row = _provenance(s, 2, "Uploaded by the preparer and reproduced as supplied.")
    manifest = []
    for e in exhibits:
        label = next((x["label"] for x in ws.EXHIBIT_SLOTS if x["key"] == e["slot_key"]),
                     e["slot_key"])
        placed = "listed only"
        try:
            ct = (e.get("content_type") or "").lower()
            name = (e.get("filename") or "").lower()
            if ct in SHEET_TYPES or name.endswith((".xlsx", ".xlsm", ".xls")):
                placed = _place_workbook(wb, used, e, label)
            elif ct == "text/csv" or name.endswith((".csv", ".tsv")):
                placed = _place_csv(wb, used, e, label)
            elif ct in IMAGE_TYPES or name.endswith((".png", ".jpg", ".jpeg", ".gif")):
                placed = _place_image(wb, used, e, label)
        except Exception as ex:
            # One unreadable upload must not cost the whole package.
            logger.warning(f"exhibit {e['id']} could not be placed", exc_info=True)
            placed = f"could not be placed ({str(ex)[:60]})"
        manifest.append({
            "Slot": label, "File": e["filename"],
            "Size (KB)": round((e.get("size_bytes") or 0) / 1024, 1),
            "Uploaded by": e.get("uploaded_by"), "Uploaded": e.get("uploaded_at"),
            "Caption": e.get("caption"), "In this workbook": placed,
        })
    if manifest:
        _table(s, row, list(manifest[0].keys()), manifest)
    else:
        s.cell(row=row, column=1,
               value="No exhibits attached. Upload supporting material against "
                     "the package in the app.").font = SUB


def _place_workbook(wb, used, e, label) -> str:
    src = load_workbook(io.BytesIO(e["content"]), data_only=True)
    names = []
    for sheet in src.worksheets:
        if sheet.sheet_state != "visible":
            continue
        t = _safe_title(f"{label[:18]}-{sheet.title}", used)
        dest = wb.create_sheet(t)
        for r, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            for c, v in enumerate(row, start=1):
                if v is not None:
                    dest.cell(row=r, column=c, value=v)
        names.append(t)
    return "tabs: " + ", ".join(names) if names else "no visible sheets"


def _place_csv(wb, used, e, label) -> str:
    t = _safe_title(label, used)
    dest = wb.create_sheet(t)
    txt = e["content"].decode("utf-8", errors="replace")
    for r, row in enumerate(csv.reader(io.StringIO(txt)), start=1):
        for c, v in enumerate(row, start=1):
            dest.cell(row=r, column=c, value=v)
    return f"tab: {t}"


def _place_image(wb, used, e, label) -> str:
    t = _safe_title(label, used)
    dest = wb.create_sheet(t)
    dest["A1"] = label
    dest["A1"].font = H1
    dest["A2"] = e["filename"]
    dest["A2"].font = SUB
    img = XLImage(io.BytesIO(e["content"]))
    dest.add_image(img, "A4")
    return f"tab: {t}"


def _signoffs(wb, used, detail):
    s = wb.create_sheet(_safe_title("Sign-off", used))
    s["A1"] = "Preparation and Approval"
    s["A1"].font = TITLE
    pkg = detail["package"]
    row = 3
    for k, v in (("Entity", pkg["entityid"]), ("Entity name", pkg.get("entity_name")),
                 ("Period", pkg["period_end"]), ("Cycle", pkg["period_label"]),
                 ("Status", pkg.get("state_label")), ("Preparer", pkg.get("preparer")),
                 ("Reviewer", pkg.get("reviewer"))):
        s.cell(row=row, column=1, value=k).font = Font(bold=True)
        s.cell(row=row, column=2, value=v)
        row += 1
    row += 1
    s.cell(row=row, column=1, value="Activity").font = H1
    row += 1
    events = [{"When": e["created_at"], "Action": e["action"], "By": e["actor"],
               "From": e["from_state"], "To": e["to_state"], "Note": e["note"]}
              for e in detail["events"]]
    if events:
        _table(s, row, list(events[0].keys()), events)
    else:
        s.cell(row=row, column=1, value="No activity recorded yet.").font = SUB
