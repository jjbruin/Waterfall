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

from flask_app.services import statement_service as ss
from flask_app.services import workpaper_data as wd
from flask_app.services import workpaper_service as ws
from flask_app.services import workpaper_print as wpp

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
    _financial_statements(wb, used, entity, period_end, engine)
    _schedule_of_investments(wb, used, entity, period_end, engine)
    _members_capital(wb, used, entity, period_end, engine)
    _cash_flow(wb, used, entity, period_end, engine)
    _trial_balance(wb, used, entity, period_end, engine)
    _gl_detail(wb, used, entity, period_end, engine)
    _account_summary(wb, used, entity, period_end, engine)
    _capital_activity(wb, used, entity, period_end, engine)
    _exhibits(wb, used, exhibits)
    _signoffs(wb, used, detail)

    # THE STATEMENT TABS ARE THE DELIVERED PRODUCT, so they carry the print
    # setup of a real delivered package rather than whatever Excel defaults to:
    # `PPI Eastchase (TX) LLC - WP - 06.30.2026.xlsx`. Applied here, once, after
    # the sheets exist -- the builders stay about figures and know nothing about
    # margins. Every other tab is a workpaper and is deliberately left alone.
    ent_name = pkg.get("entity_name") or entity
    for title in wpp.SHEETS:
        if title in wb.sheetnames:
            sh = wb[title]
            wpp.style_body(sh)
            wpp.apply(sh, title, ent_name, period_end)

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
    for name in ("Balance Sheet", "Income Statement", "SOI", "Members Capital",
                 "Cash Flow", "Trial Balance", "GL Detail", "Account Summary",
                 "Investor Detail", "Investment Detail", "IA Rollforward",
                 "Commitments"):
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


def _financial_statements(wb, used, entity, period_end, engine):
    """Balance Sheet and Income Statement, from the statement engine.

    The same call any entity's standalone statements come from -- the package
    is one caller, not the owner, so a figure here cannot differ from the one
    an auditor is shown elsewhere.
    """
    st = ss.build(entity, period_end, "both", engine=engine)

    def sheet(title, block, value_label):
        sh = wb.create_sheet(_safe_title(title, used))
        sh["A1"] = title
        sh["A1"].font = TITLE
        r = _provenance(sh, 2,
                        f"{entity} — period ended {period_end}. Built from gl_detail "
                        f"through the account mapping; accounts are placed by GACC.TYPE. "
                        f"Amounts are presented positive; the GL figure is beside each line.")
        if not block or not block["sections"]:
            sh.cell(row=r, column=1,
                    value="No mapped accounts — map accounts to statement lines "
                          "to populate this statement.").font = SUB
            return sh, r + 2
        dormant_total = 0
        for sec in block["sections"]:
            sh.cell(row=r, column=1, value=sec["section"]).font = H1
            r += 1
            # A line with no balance AND no movement is left off the printed
            # statement -- a page of 0.00 rows reads as a trial balance. The
            # count is stated below so nothing is silently absent, and a zero
            # line that HAD movement still prints.
            shown = [l for l in sec["lines"] if not l.get("dormant")]
            dormant_total += len(sec["lines"]) - len(shown)
            r = _table(sh, r, ["fs_line", "amount", "gl_amount"],
                       [{"fs_line": l["fs_line"], "amount": l["amount"],
                         "gl_amount": l["gl_amount"]} for l in shown],
                       money_cols=["amount", "gl_amount"])
            c = sh.cell(row=r - 1, column=1, value=f"Total {sec['section']}")
            c.font = Font(bold=True)
            t = sh.cell(row=r - 1, column=2, value=sec["total"])
            t.font = Font(bold=True)
            t.number_format = MONEY
            r += 1
        if dormant_total:
            sh.cell(row=r, column=1,
                    value=f"{dormant_total} line(s) with no balance and no movement "
                          f"are not shown; they are on the Trial Balance tab."
                    ).font = SUB
            r += 2
        return sh, r

    bs = st.get("balance_sheet")
    sh, r = sheet("Balance Sheet", bs, "closing")
    if bs:
        # The tie-out, printed. In GL signs a complete balance sheet nets to
        # zero; whatever is left is what is unmapped or misclassified.
        c = sh.cell(row=r, column=1,
                    value="In balance" if bs["balanced"]
                          else f"OUT OF BALANCE by {bs['out_of_balance']:,.2f} "
                               f"— see unmapped accounts below")
        c.font = Font(bold=True, color="2C7A3D" if bs["balanced"] else "B3261E")
        r += 2
        _exceptions(sh, r, st)

    inc = st.get("income_statement")
    sh2, r2 = sheet("Income Statement", inc, "ytd")
    if inc:
        c = sh2.cell(row=r2, column=1, value="Net income (loss)")
        c.font = Font(bold=True)
        v = sh2.cell(row=r2, column=2, value=inc["net_income"])
        v.font = Font(bold=True)
        v.number_format = MONEY


def _exceptions(sh, row, st):
    """Everything the statements could not place. Named, with totals."""
    for label, rows, cols in (
        (f"UNMAPPED — {len(st['unmapped'])} accounts, {st['unmapped_total']:,.2f} "
         f"not on any statement line",
         st["unmapped"], ["acctnum", "acctname", "statement", "closing"]),
        (f"UNTYPED — {len(st['untyped'])} accounts with no usable GACC.TYPE",
         st["untyped"], ["acctnum", "acctname", "type", "closing"]),
        (f"CONFLICTS — {len(st['conflicts'])} accounts whose mapped section "
         f"disagrees with GACC.TYPE",
         st["conflicts"], ["acctnum", "acctname", "mapped_section",
                           "gacc_type", "statement_by_type", "closing"]),
    ):
        if not rows:
            continue
        sh.cell(row=row, column=1, value=label).font = Font(bold=True, color="B3261E")
        row += 1
        row = _table(sh, row, cols, rows, money_cols=["closing"]) + 1
    return row


def _schedule_of_investments(wb, used, entity, period_end, engine):
    """Schedule of Investments — cost, fair value, % of members' capital."""
    soi = ss.build_schedule_of_investments(entity, period_end, engine=engine)
    sh = wb.create_sheet(_safe_title("SOI", used))
    sh["A1"] = "Schedule of Investments"
    sh["A1"].font = TITLE
    r = _provenance(sh, 2,
                    f"{entity} — cost and fair value from the GL's investment accounts "
                    f"(Purchase + Return of Capital, plus Unrealized). Name and "
                    f"membership interest from relationships (MRI_IA_Relationship). "
                    f"Percentage is fair value over members' capital.")
    lines = list(soi.get("lines") or [])
    if soi.get("unallocated"):
        lines.append({**soi["unallocated"], "name": "Unallocated — GL rows carry no related entity"})
    if not lines:
        sh.cell(row=r, column=1,
                value=soi.get("note", "No investment balances for this entity")).font = SUB
        return

    # Both figures, side by side. Membership interest is derived from
    # committed amounts; the relationships percentage sits beside it because
    # accounting is mid-update on that table and the two currently disagree on
    # EASTCH (derived 66.67%, relationships 100%). Showing one and hiding the
    # other would make a live data question invisible on a signed document.
    rows = [{
        "Name of Investment": l.get("name") or l.get("related_entity") or "—",
        "Membership Interest": (l["ownership_pct"] / 100.0
                                if l.get("ownership_pct") is not None else None),
        "Per relationships": (l["ownership_pct_relationships"] / 100.0
                              if l.get("ownership_pct_relationships") is not None else None),
        "Committed": l.get("committed_amount"),
        "Cost": l["cost"],
        "Fair Value": l["fair_value"],
        "Fair Value as % of Members' Capital": l.get("pct_of_members_capital"),
    } for l in lines]
    rows.append({"Name of Investment": "Total", "Membership Interest": None,
                 "Per relationships": None, "Committed": None,
                 "Cost": soi["total_cost"], "Fair Value": soi["total_fair_value"],
                 "Fair Value as % of Members' Capital": None})
    r = _table(sh, r, list(rows[0].keys()), rows,
               money_cols=["Committed", "Cost", "Fair Value"])

    disagree = [l for l in lines if l.get("ownership_disagrees")]
    if disagree:
        sh.cell(row=r, column=1,
                value="MEMBERSHIP INTEREST DISAGREES with relationships on "
                      + ", ".join(str(l.get("name") or l.get("related_entity"))
                                  for l in disagree)
                      + " — the figure shown is derived from committed amounts."
                ).font = Font(bold=True, color="8A5A00")
        r += 2

    ok = soi["ties"]
    c = sh.cell(row=r, column=1,
                value="Ties to the GL investment accounts" if ok
                      else f"DOES NOT TIE to the GL by {soi['difference']:,.2f}")
    c.font = Font(bold=True, color="2C7A3D" if ok else "B3261E")
    r += 2
    if soi.get("unassigned_accounts"):
        # An investment account whose role the engine does not recognise is
        # named here rather than folded into a total nobody would question.
        sh.cell(row=r, column=1,
                value="UNRECOGNISED INVESTMENT ACCOUNTS — not included in cost or "
                      "fair value: " + ", ".join(soi["unassigned_accounts"])
                ).font = Font(bold=True, color="B3261E")


def _members_capital(wb, used, entity, period_end, engine):
    """Statement of Changes in Members' Capital, per member.

    Columns are members, rows are movements — the layout the example package
    uses. The opening column comes from the subledger, not from last quarter's
    workbook, and the statement prints its own reconciliation to the GL.
    """
    mc = ss.build_members_capital(entity, period_end, engine=engine)
    sh = wb.create_sheet(_safe_title("Members Capital", used))
    sh["A1"] = "Statement of Changes in Members' Capital"
    sh["A1"].font = TITLE
    r = _provenance(sh, 2,
                    f"{entity} — per member from ia_transactions, all major types "
                    f"including non-cash. Opening is every transaction before "
                    f"{mc['periods']['year']}-01-01, taken from the subledger itself.")
    if not mc.get("members"):
        sh.cell(row=r, column=1, value=mc.get("note", "No investor activity")).font = SUB
        return

    members = mc["members"]
    cols = ["Movement"] + [m["InvestorName"] or m["InvestorID"] for m in members] + ["Total"]
    rows = []
    for row in mc["rows"]:
        d = {"Movement": row["label"]}
        for m in members:
            d[m["InvestorName"] or m["InvestorID"]] = row["by_member"].get(m["InvestorID"], 0.0)
        d["Total"] = row["total"]
        rows.append(d)
    r = _table(sh, r, cols, rows, money_cols=cols[1:])

    # Subledger against control account. The GL holds ONE equity balance for
    # the entity; this statement splits it by member. If the two disagree,
    # one of them is wrong, and the statement says so rather than presenting
    # a total nobody checked.
    if mc.get("gl_equity") is not None:
        ok = mc["ties"]
        sh.cell(row=r, column=1, value="Subledger total").font = Font(bold=True)
        sh.cell(row=r, column=2, value=mc["subledger_total"]).number_format = MONEY
        sh.cell(row=r + 1, column=1, value="GL members' capital").font = Font(bold=True)
        sh.cell(row=r + 1, column=2, value=mc["gl_equity"]).number_format = MONEY
        c = sh.cell(row=r + 2, column=1,
                    value="Ties to the GL" if ok
                          else f"DOES NOT TIE to the GL by {mc['difference']:,.2f}")
        c.font = Font(bold=True, color="2C7A3D" if ok else "B3261E")
    else:
        sh.cell(row=r, column=1,
                value="No GL equity section mapped — the subledger cannot be "
                      "reconciled to the control account.").font = SUB


def _cash_flow(wb, used, entity, period_end, engine):
    """Statement of Cash Flows, indirect, with the identity it rests on."""
    cf = ss.build_cash_flow(entity, period_end, engine=engine)
    sh = wb.create_sheet(_safe_title("Cash Flow", used))
    sh["A1"] = "Statement of Cash Flows"
    sh["A1"].font = TITLE
    r = _provenance(sh, 2,
                    f"{entity} — every period's entries balance, so the change in cash "
                    f"is the negative of the change in every other account. Each line "
                    f"is an account's movement, classified; the total is checked "
                    f"against the cash accounts below.")
    if not cf.get("sections"):
        sh.cell(row=r, column=1, value=cf.get("note", "No GL rows")).font = SUB
        return
    for sec in cf["sections"]:
        sh.cell(row=r, column=1, value=sec["section"]).font = H1
        r += 1
        r = _table(sh, r, ["fs_line", "amount"],
                   [{"fs_line": l["fs_line"], "amount": l["amount"]} for l in sec["lines"]],
                   money_cols=["amount"])
        c = sh.cell(row=r - 1, column=1, value=f"Net cash from {sec['category'].lower()}")
        c.font = Font(bold=True)
        t = sh.cell(row=r - 1, column=2, value=sec["total"])
        t.font = Font(bold=True)
        t.number_format = MONEY
        r += 1

    for label, val in (("Net change in cash (computed)", cf["net_change_computed"]),
                       ("Net change in cash (per the cash accounts)", cf["net_change_actual"])):
        sh.cell(row=r, column=1, value=label).font = Font(bold=True)
        sh.cell(row=r, column=2, value=val).number_format = MONEY
        r += 1
    ok = cf["ties"]
    c = sh.cell(row=r, column=1,
                value="Ties" if ok else f"DOES NOT TIE by {cf['difference']:,.2f}")
    c.font = Font(bold=True, color="2C7A3D" if ok else "B3261E")
    r += 2

    if cf["defaulted_accounts"]:
        sh.cell(row=r, column=1,
                value=f"CLASSIFIED BY DEFAULT — {len(cf['defaulted_accounts'])} accounts "
                      f"carry no explicit operating/investing/financing category"
                ).font = Font(bold=True, color="8A5A00")
        r += 1
        r = _table(sh, r, ["acctnum", "acctname", "section", "category", "ytd"],
                   cf["defaulted_accounts"], money_cols=["ytd"]) + 1
    if cf["unclassified"]:
        sh.cell(row=r, column=1,
                value=f"UNCLASSIFIED — {len(cf['unclassified'])} accounts with no usable "
                      f"GACC.TYPE, excluded from the statement"
                ).font = Font(bold=True, color="B3261E")
        r += 1
        _table(sh, r, ["acctnum", "acctname", "ytd"], cf["unclassified"], money_cols=["ytd"])


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
