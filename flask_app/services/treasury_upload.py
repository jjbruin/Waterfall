"""The MRI upload files: a GL journal entry and an IA transaction list.

THE OUTPUT OF THE ACCOUNTANT'S WORK. The reconciliation says what the bank did;
this says what the ledger should say about it. Nothing here posts to MRI -- it
produces the two files a person uploads, which is the same boundary the rest of
the accounting section keeps.

THE CONTRACT WAS READ OFF JIM'S REAL AUGUST AMB6 FILES, not invented:

    GL upload, all lines          49 rows    sums to  0.00   (a balanced JE)
    GL cash lines MR10005000      24 rows      -560,022.54   = the bank's own
                                                               net movement
    GL distributions MR31000001   13 rows        12,580.47
    IA upload                     13 rows        12,580.47   the same thirteen,
                                                               one per investor

So the three files tie to each other and to the bank. One coded distribution
produces BOTH a GL line and an IA row, and the GL line's description names the
investor whose ID the IA row carries.

THE IA FILE IS WRITTEN INTO A COPY OF MRI'S OWN TEMPLATE, never built from
scratch. Column L of `Transaction Values` is "Number of Shares" and its header
cell is EMPTY in MRI's template -- pandas reads it as `Unnamed: 11`, and a
workbook reconstructed from those column names would carry a header MRI does not
expect. Copying the template also preserves the Guide sheet, so the file that
reaches the accountant still explains itself.

WHAT IS REFUSED RATHER THAN REPAIRED. An unbalanced journal entry, a zero
amount, a period that is not YYYYMM, a transaction type MRI's own Guide does not
allow, and an IA total that does not tie to the GL lines it is meant to mirror.
A file that uploads and is wrong costs more than a file that does not upload.
"""
import csv
import io
import logging
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: The GL template's columns, in its own order. From
#: `GL Uploads X.7 Template.csv`.
GL_COLUMNS = ("EntityID", "AcctNum", "Amount", "Descrpn", "ADDLDESC",
              "RLTDENTITY", "JobCode", "Period", "Basis", "ENTRDATE")

#: MRI's own sheet name, and the row its data starts on.
IA_SHEET = "Transaction Values"
IA_FIRST_ROW = 2

#: Column positions in `Transaction Values`, 1-based. Position matters more
#: than name here: column 12 ("Number of Shares") has a BLANK header in MRI's
#: template, so it can only be addressed by index.
IA_COLUMNS = {
    "transaction_type": 1, "sub_type": 2, "amount": 3, "investmentid": 4,
    "investorid": 5, "transaction_date": 6, "effective_date": 7,
    "journal_period": 8, "payable": 9, "payment_status": 10,
    "share_class": 11, "number_of_shares": 12, "price_per_share": 13,
    "exchange_period": 14, "notes": 15,
}

#: The Guide sheet's *Validation* column is NARROWER than its *Available
#: Values* column: the latter lists Commitment and CapitalCall, the former says
#: "Can only be Contribution, Distribution, Income Allocation". The validation
#: is what MRI enforces, so it is what is enforced here.
IA_TYPES = ("Contribution", "Distribution", "Income Allocation")

#: Balanced to the cent. A journal entry that does not foot is not a journal
#: entry, and MRI will reject the whole file rather than the offending line.
BALANCE_TOLERANCE = 0.005

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "mri_templates"
IA_TEMPLATE = TEMPLATE_DIR / "IA Upload Template.xlsx"


# ---------------------------------------------------------------- helpers

def _num(v) -> Optional[float]:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _iso(v) -> Optional[str]:
    """Any of the shapes these files carry, as YYYY-MM-DD, or None."""
    if v is None or str(v).strip() == "":
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    t = str(v).strip()
    for f in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d %H:%M:%S",
              "%m/%d/%Y %H:%M", "%d-%b-%Y"):
        try:
            return datetime.strptime(t, f).date().isoformat()
        except ValueError:
            continue
    return None


def _us_date(iso: str) -> str:
    """The GL file's own date shape: 8/18/2026, no leading zeros."""
    d = date.fromisoformat(iso)
    return "%d/%d/%d" % (d.month, d.day, d.year)


def _period_ok(p) -> bool:
    t = str(p).strip()
    if t.endswith(".0"):          # a period read back through pandas
        t = t[:-2]
    if len(t) != 6 or not t.isdigit():
        return False
    return 1 <= int(t[4:]) <= 12


def _period(p) -> str:
    t = str(p).strip()
    return t[:-2] if t.endswith(".0") else t


# ------------------------------------------------------------ the GL side

def validate_gl(lines: Iterable[dict]) -> dict:
    """Check a journal entry before it is written.

    Errors BLOCK -- they are the things MRI would reject, or worse, accept
    while meaning something other than what the accountant intended. Warnings
    do not block: an empty description is untidy, not wrong.
    """
    lines = list(lines)
    errors, warnings = [], []
    if not lines:
        return {"errors": ["There are no lines to upload."], "warnings": [],
                "total": 0.0, "balanced": False, "line_count": 0}

    total = 0.0
    periods, entities = set(), set()
    for i, ln in enumerate(lines, start=1):
        amt = _num(ln.get("amount"))
        if amt is None:
            errors.append("Line %d has no amount." % i)
        elif abs(amt) < 0.005:
            # The IA Guide says "Must not be 0"; a zero GL line is equally
            # meaningless and usually a coding slip rather than an intent.
            errors.append("Line %d has a zero amount." % i)
        else:
            total += amt

        if not str(ln.get("acctnum") or "").strip():
            errors.append("Line %d has no account number." % i)
        if not str(ln.get("entityid") or "").strip():
            errors.append("Line %d has no entity." % i)
        else:
            entities.add(str(ln["entityid"]).strip().upper())
        if not _period_ok(ln.get("period")):
            errors.append("Line %d has period %r, which is not YYYYMM."
                          % (i, ln.get("period")))
        else:
            periods.add(_period(ln.get("period")))
        if _iso(ln.get("entrdate")) is None:
            errors.append("Line %d has no readable entry date (%r)."
                          % (i, ln.get("entrdate")))
        if not str(ln.get("descrpn") or "").strip():
            warnings.append("Line %d has no description." % i)
        basis = str(ln.get("basis") or "B").strip().upper()
        if basis != "B":
            warnings.append("Line %d is basis %r, not B." % (i, basis))

    total = round(total, 2)
    balanced = abs(total) < BALANCE_TOLERANCE
    if not balanced:
        # Named as debits-minus-credits, which is how an accountant will look
        # for it, and with the figure to look for.
        errors.append("The entry does not balance: it is out by %s. A journal "
                      "entry must sum to zero." % ("{:,.2f}".format(total)))
    if len(periods) > 1:
        errors.append("The lines span more than one period (%s). One upload is "
                      "one period." % ", ".join(sorted(periods)))
    if len(entities) > 1:
        warnings.append("The lines span more than one entity (%s)."
                        % ", ".join(sorted(entities)))

    return {"errors": errors, "warnings": warnings, "total": total,
            "balanced": balanced, "line_count": len(lines),
            "period": (sorted(periods)[0] if len(periods) == 1 else None),
            "entityid": (sorted(entities)[0] if len(entities) == 1 else None)}


def build_gl_csv(lines: Iterable[dict]) -> str:
    """The GL upload, in the template's own column order and date format.

    Refuses rather than writing something MRI will reject or mis-post; the
    caller shows `validate_gl`'s errors.
    """
    lines = list(lines)
    v = validate_gl(lines)
    if v["errors"]:
        raise ValueError("; ".join(v["errors"][:4]))

    buf = io.StringIO()
    # MRI reads this with a plain CSV parser; \r\n matches the template.
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(GL_COLUMNS)
    for ln in lines:
        w.writerow([
            str(ln.get("entityid") or "").strip().upper(),
            str(ln.get("acctnum") or "").strip(),
            _fmt_amount(_num(ln.get("amount"))),
            str(ln.get("descrpn") or "").strip(),
            str(ln.get("addldesc") or "").strip(),
            str(ln.get("rltdentity") or "").strip(),
            str(ln.get("jobcode") or "").strip(),
            _period(ln.get("period")),
            str(ln.get("basis") or "B").strip().upper(),
            _us_date(_iso(ln.get("entrdate"))),
        ])
    return buf.getvalue()


def _fmt_amount(v: float) -> str:
    """Two decimals unless the value genuinely carries more.

    The August file writes `13313.8`, not `13313.80`, so trailing zeros are
    trimmed the same way rather than imposed -- the point is a file that looks
    like the ones already accepted.
    """
    t = ("%.2f" % round(float(v), 2)).rstrip("0").rstrip(".")
    return t if t not in ("", "-") else "0"


# ------------------------------------------------------------ the IA side

def validate_ia(rows: Iterable[dict], gl_lines=None,
                ia_account: str = "") -> dict:
    """Check the investor transactions, and that they tie to the GL.

    THE TIE IS THE POINT. The IA rows mirror the GL lines hitting the
    distribution account -- in August, thirteen rows summing to 12,580.47 on
    both sides. Two files that disagree post two different truths, and MRI
    accepts each on its own.
    """
    rows = list(rows)
    errors, warnings = [], []
    if not rows:
        return {"errors": ["There are no investor transactions to upload."],
                "warnings": [], "total": 0.0, "row_count": 0, "ties": None}

    total = 0.0
    for i, r in enumerate(rows, start=1):
        amt = _num(r.get("amount"))
        if amt is None or abs(amt) < 0.005:
            errors.append("Row %d has a zero or missing amount; MRI requires "
                          "a non-zero amount." % i)
        else:
            total += amt
        t = str(r.get("transaction_type") or "").strip()
        if t not in IA_TYPES:
            errors.append("Row %d has transaction type %r. MRI's guide allows "
                          "only: %s." % (i, t, ", ".join(IA_TYPES)))
        if not str(r.get("sub_type") or "").strip():
            errors.append("Row %d has no sub-type." % i)
        if not str(r.get("investmentid") or "").strip():
            errors.append("Row %d has no InvestmentID." % i)
        if not str(r.get("investorid") or "").strip():
            errors.append("Row %d has no InvestorID." % i)
        if _iso(r.get("transaction_date")) is None:
            errors.append("Row %d has no readable transaction date." % i)
        if r.get("journal_period") and not _period_ok(r["journal_period"]):
            errors.append("Row %d has journal period %r, which is not YYYYMM."
                          % (i, r["journal_period"]))
        pay = str(r.get("payable") or "N").strip().upper()
        if pay not in ("Y", "N"):
            errors.append("Row %d has Payable %r; it must be Y or N." % (i, pay))

    total = round(total, 2)
    out = {"errors": errors, "warnings": warnings, "total": total,
           "row_count": len(rows), "ties": None}

    if gl_lines is not None and ia_account:
        gl_total = round(sum(
            (_num(l.get("amount")) or 0.0) for l in gl_lines
            if str(l.get("acctnum") or "").strip().upper()
            == ia_account.strip().upper()), 2)
        out["gl_total"] = gl_total
        out["difference"] = round(total - gl_total, 2)
        out["ties"] = abs(out["difference"]) < BALANCE_TOLERANCE
        if not out["ties"]:
            errors.append(
                "The investor transactions total %s but the %s lines on the "
                "journal entry total %s, a difference of %s. They must agree."
                % ("{:,.2f}".format(total), ia_account,
                   "{:,.2f}".format(gl_total),
                   "{:,.2f}".format(out["difference"])))
    return out


def build_ia_xlsx(rows: Iterable[dict], gl_lines=None,
                  ia_account: str = "") -> bytes:
    """The IA upload, written into a COPY of MRI's own template.

    See the module note: column 12's header is blank in the template and a
    rebuilt workbook would not reproduce it. Copying also keeps the Guide sheet
    in the file, so it still explains its own rules to whoever opens it.
    """
    import openpyxl

    rows = list(rows)
    v = validate_ia(rows, gl_lines=gl_lines, ia_account=ia_account)
    if v["errors"]:
        raise ValueError("; ".join(v["errors"][:4]))
    if not IA_TEMPLATE.exists():
        raise FileNotFoundError(
            "MRI's IA template is missing from %s. It is vendored into the "
            "repo deliberately -- the file cannot be rebuilt faithfully from "
            "column names alone." % TEMPLATE_DIR)

    tmp = io.BytesIO()
    with open(IA_TEMPLATE, "rb") as fh:
        shutil.copyfileobj(fh, tmp)
    tmp.seek(0)
    wb = openpyxl.load_workbook(tmp)
    ws = wb[IA_SHEET]

    c = IA_COLUMNS
    for n, r in enumerate(rows):
        row = IA_FIRST_ROW + n
        ws.cell(row=row, column=c["transaction_type"],
                value=str(r.get("transaction_type") or "").strip())
        ws.cell(row=row, column=c["sub_type"],
                value=str(r.get("sub_type") or "").strip())
        ws.cell(row=row, column=c["amount"],
                value=round(float(_num(r.get("amount"))), 2))
        ws.cell(row=row, column=c["investmentid"],
                value=str(r.get("investmentid") or "").strip().upper())
        ws.cell(row=row, column=c["investorid"],
                value=str(r.get("investorid") or "").strip().upper())
        ws.cell(row=row, column=c["transaction_date"],
                value=_iso(r.get("transaction_date")))
        # MRI defaults Effective Date to the transaction date; written
        # explicitly so the file says what it means rather than relying on it.
        ws.cell(row=row, column=c["effective_date"],
                value=_iso(r.get("effective_date"))
                or _iso(r.get("transaction_date")))
        if r.get("journal_period"):
            ws.cell(row=row, column=c["journal_period"],
                    value=_period(r["journal_period"]))
        if r.get("payable"):
            ws.cell(row=row, column=c["payable"],
                    value=str(r["payable"]).strip().upper())
        for key in ("payment_status", "share_class", "number_of_shares",
                    "price_per_share", "exchange_period", "notes"):
            if r.get(key) not in (None, ""):
                ws.cell(row=row, column=c[key], value=r[key])

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# ------------------------------------------- reading one back, for checking

def read_gl_csv(text: str) -> list:
    """Parse a GL upload back into lines. Used to check a file against itself."""
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        if not any((v or "").strip() for v in r.values()):
            continue
        out.append({
            "entityid": r.get("EntityID"), "acctnum": r.get("AcctNum"),
            "amount": _num(r.get("Amount")), "descrpn": r.get("Descrpn"),
            "addldesc": r.get("ADDLDESC"), "rltdentity": r.get("RLTDENTITY"),
            "jobcode": r.get("JobCode"), "period": r.get("Period"),
            "basis": r.get("Basis"), "entrdate": r.get("ENTRDATE"),
        })
    return out


def summarise(gl_lines, ia_rows=None, cash_account: str = "",
              ia_account: str = "") -> dict:
    """What the accountant should see before downloading either file.

    Reports the three figures that made the August files verifiable: the entry
    balances, the cash lines equal the bank's movement, and the investor
    transactions tie to their GL account.
    """
    gl_lines = list(gl_lines)
    v = validate_gl(gl_lines)
    out = {"line_count": len(gl_lines), "total": v["total"],
           "balanced": v["balanced"], "errors": list(v["errors"]),
           "warnings": list(v["warnings"]),
           "period": v.get("period"), "entityid": v.get("entityid")}
    if cash_account:
        out["cash_total"] = round(sum(
            (_num(l.get("amount")) or 0.0) for l in gl_lines
            if str(l.get("acctnum") or "").strip().upper()
            == cash_account.strip().upper()), 2)
        out["cash_line_count"] = sum(
            1 for l in gl_lines
            if str(l.get("acctnum") or "").strip().upper()
            == cash_account.strip().upper())
    if ia_rows is not None:
        iv = validate_ia(ia_rows, gl_lines=gl_lines, ia_account=ia_account)
        out["ia_row_count"] = iv["row_count"]
        out["ia_total"] = iv["total"]
        out["ia_ties"] = iv["ties"]
        out["ia_difference"] = iv.get("difference")
        out["errors"] += iv["errors"]
        out["warnings"] += iv["warnings"]
    return out
