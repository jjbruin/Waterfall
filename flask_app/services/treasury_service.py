"""Bank accounts, activity, and the three-way tie behind a reconciliation.

WHAT THIS IS FOR. The accountants monitor each cash account, reconcile it to the
GL, and post the resulting journal entries to MRI. This module holds the bank
side: the accounts, the activity imported from PNC, the statement balances, and
the arithmetic that says whether a period ties.

THE TIE, PROVEN ON REAL DATA before any of this was written. August 2026, AMB6,
account 8514245765:

    beginning balance (PNC statement)     571,750.04
    net movement (PNC activity export)   -560,022.54
    computed ending                        11,727.50
    ending balance (PNC statement)         11,727.50   <- ties
    MRI's September reconciliation opens   11,727.50   <- and carries forward

and the same -560,022.54 is the net of the GL's own cash account for the month.
Twenty-four bank transactions against twenty-four GL cash lines, every amount
matching one for one.

THE OPENING BALANCE COMES FROM OUR OWN PRIOR CLOSE, not from the statement.
Jim, Sep 17 2026: "Carry the prior period's computed ending forward." The PNC
activity export carries no running balance -- only transactions -- so a period
needs an opening figure from somewhere. Taking it from the previous period's
COMPUTED ending means a reconciliation does not depend on a statement PDF having
been filed, and it chains: if one month is wrong, every later month says so
instead of quietly re-basing on a fresh statement figure and hiding the break.
A statement balance, when present, is compared against the carried-forward
figure rather than replacing it.

CASH ACCOUNTS ARE PER BANK ACCOUNT. MR10005000 is the generic cash account for
an entity with a single bank account and is the default; an entity with several
uses MR10006000, MR10007000, MR10008000 or MR10008100 (Jim's list). Stored on
the bank account, because which GL account a given bank account posts to is a
fact about that account and not something to infer from a balance.

WHAT THIS MODULE WILL NOT DO. It does not decide the offsetting entry. Every
bank transaction has one mechanical cash line -- amount, date and description
all come from the bank -- and a contra account that is the accountant's coding
decision: capital calls, distributions, investments, intercompany, fees,
income. Proposing the cash side is help; inventing the offset would be putting
words in an accountant's mouth on a signed document.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from flask_app.db import get_engine

logger = logging.getLogger(__name__)

#: The GL cash accounts a bank account may post to. MR10005000 is the generic
#: one for an entity with a single bank account (Jim, Sep 17 2026); the rest are
#: for entities holding more than one.
CASH_ACCOUNTS = ("MR10005000", "MR10006000", "MR10007000", "MR10008000",
                 "MR10008100")
DEFAULT_CASH_ACCOUNT = "MR10005000"

_DDL = [
    # One row per bank account. `gl_cash_account` is stored rather than
    # inferred -- see the module note.
    """
    CREATE TABLE IF NOT EXISTS tr_accounts (
        id               {pk},
        account_number   TEXT NOT NULL,
        bank_id          TEXT,
        account_name     TEXT,
        entityid         TEXT,
        gl_cash_account  TEXT,
        active           INTEGER DEFAULT 1,
        updated_by       TEXT,
        updated_at       TEXT
    )
    """,
    # Imported activity. `source_file` and `imported_at` are kept so a figure
    # can always be traced to the file it came from, which is the first thing
    # asked when a reconciliation is queried.
    """
    CREATE TABLE IF NOT EXISTS tr_activity (
        id               {pk},
        account_number   TEXT NOT NULL,
        as_of_date       TEXT,
        bai_control      TEXT,
        transaction_type TEXT,
        amount           DOUBLE PRECISION,
        direction        TEXT,
        signed_amount    DOUBLE PRECISION,
        reference        TEXT,
        description      TEXT,
        currency         TEXT,
        source_file      TEXT,
        imported_at      TEXT,
        row_hash         TEXT
    )
    """,
    # Statement balances, when a statement has been filed. NOT the source of
    # the opening balance -- a comparison for it.
    """
    CREATE TABLE IF NOT EXISTS tr_statements (
        id                {pk},
        account_number    TEXT NOT NULL,
        period_start      TEXT,
        period_end        TEXT,
        beginning_balance DOUBLE PRECISION,
        ending_balance    DOUBLE PRECISION,
        credits_total     DOUBLE PRECISION,
        debits_total      DOUBLE PRECISION,
        source_file       TEXT,
        imported_at       TEXT
    )
    """,
    # Pairings an accountant made by hand. Honoured before any automatic
    # match and never re-decided: they looked at both sides, the matcher only
    # looked at the amount.
    """
    CREATE TABLE IF NOT EXISTS tr_matches (
        id             {pk},
        account_number TEXT NOT NULL,
        period         TEXT NOT NULL,
        bank_id        INTEGER NOT NULL,
        gl_item        TEXT,
        matched_by     TEXT,
        matched_at     TEXT
    )
    """,
    # A closed period's computed ending, which becomes the next period's
    # opening. Written when a period is reconciled.
    """
    CREATE TABLE IF NOT EXISTS tr_periods (
        id                {pk},
        account_number    TEXT NOT NULL,
        period            TEXT NOT NULL,
        opening_balance   DOUBLE PRECISION,
        computed_ending   DOUBLE PRECISION,
        status            TEXT DEFAULT 'open',
        closed_by         TEXT,
        closed_at         TEXT,
        note              TEXT
    )
    """,
]

_DDL_DONE = set()


def ensure_tables(engine=None) -> None:
    engine = engine or get_engine()
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _DDL_DONE:
        return
    is_pg = engine.dialect.name == "postgresql"
    pk = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with engine.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl.format(pk=pk)))
    _DDL_DONE.add(key)


# ------------------------------------------------------------------ helpers

def _clean(v) -> str:
    """PNC's CSV guards text fields with a leading apostrophe.

    `'031000053` and `'00000000000` arrive that way so Excel will not eat the
    leading zeros. Left in place, an account number never matches itself.
    """
    s = "" if v is None else str(v)
    return s.strip().lstrip("'").strip()


def _num(v) -> Optional[float]:
    try:
        f = float(str(v).replace(",", "").strip())
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _iso(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (datetime, date)):
        return (v.date() if isinstance(v, datetime) else v).isoformat()
    s = str(v).strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def period_of(iso_date: str) -> Optional[str]:
    """``2026-08-14`` -> ``202608``, the period MRI's upload template uses."""
    if not iso_date:
        return None
    return iso_date[:4] + iso_date[5:7]


def prior_period(period: str) -> Optional[str]:
    if not period or len(period) != 6 or not period.isdigit():
        return None
    y, m = int(period[:4]), int(period[4:])
    return "%04d%02d" % ((y - 1, 12) if m == 1 else (y, m - 1))


# ------------------------------------------------------------------ parsing

#: The PNC activity export's own header, as exported from PINACLE.
PNC_ACTIVITY_COLUMNS = (
    "AsOfDate", "BankId", "AccountNumber", "AccountName", "BaiControl",
    "Currency", "Transaction", "Amount", "Credit/Debit", "Reference",
    "Description")


def parse_activity(df: pd.DataFrame, source_file: str = "") -> dict:
    """Normalise a PNC activity export into rows this module stores.

    Returns the rows plus what could NOT be read, because a transaction dropped
    silently is a reconciliation that ties for the wrong reason.
    """
    if df is None:
        return {"rows": [], "skipped": [], "accounts": [],
                "error": "Nothing could be read from this file."}
    # THE COLUMNS ARE CHECKED BEFORE THE ROW COUNT, deliberately. A file that
    # is not an activity export at all often parses into an empty frame, and
    # short-circuiting on `df.empty` first reported it as "0 transactions
    # imported" -- a refusal dressed up as a successful no-op. An export with
    # the right columns and no rows is a different thing and is still fine.
    cols = {str(c).strip().lower(): c for c in df.columns}
    need = ("asofdate", "accountnumber", "amount", "credit/debit")
    missing = [n for n in need if n not in cols]
    if missing:
        return {"rows": [], "skipped": [],
                "error": "Not a PNC activity export: missing %s. Found: %s"
                         % (", ".join(missing),
                            ", ".join(str(c) for c in list(df.columns)[:10])
                            or "no columns at all")}
    if df.empty:
        return {"rows": [], "skipped": [], "accounts": []}

    rows, skipped = [], []
    for i, r in df.iterrows():
        acct = _clean(r.get(cols["accountnumber"]))
        amt = _num(r.get(cols["amount"]))
        when = _iso(r.get(cols["asofdate"]))
        direction = _clean(r.get(cols["credit/debit"])).lower()
        if not acct or amt is None or not when:
            # A wholly blank trailing row is not a problem; anything else is.
            if any(str(x).strip() for x in r.values if x is not None):
                skipped.append({"row": int(i) + 2, "reason":
                                "missing account, amount or date"})
            continue
        if direction not in ("credit", "debit"):
            skipped.append({"row": int(i) + 2,
                            "reason": "direction %r is neither credit nor debit"
                                      % direction})
            continue
        signed = amt if direction == "credit" else -amt
        desc = _clean(r.get(cols.get("description"), ""))
        rows.append({
            "account_number": acct,
            "as_of_date": when,
            "bai_control": _clean(r.get(cols.get("baicontrol"), "")),
            "transaction_type": _clean(r.get(cols.get("transaction"), "")),
            "amount": amt,
            "direction": direction,
            "signed_amount": signed,
            "reference": _clean(r.get(cols.get("reference"), "")),
            "description": desc,
            "currency": _clean(r.get(cols.get("currency"), "")) or "USD",
            "account_name": _clean(r.get(cols.get("accountname"), "")),
            "bank_id": _clean(r.get(cols.get("bankid"), "")),
            "source_file": source_file,
            # Identity for de-duplication: re-importing the same export must
            # not double the month. The description is included because two
            # wires on one day for one amount are two real transactions.
            "row_hash": "|".join([acct, when, "%.2f" % signed,
                                  _clean(r.get(cols.get("reference"), "")),
                                  desc[:120]]),
        })
    accounts = sorted({r["account_number"] for r in rows})
    return {"rows": rows, "skipped": skipped, "accounts": accounts}


#: One money figure as PNC prints it. THE INTEGER PART IS OPTIONAL: a
#: zero-balance account prints `.00`, not `0.00`, and 46 of the 64 June 2026
#: statements were refused because of it -- including ones carrying real
#: amounts, since a single `.00` anywhere in the row broke the whole line.
#: A TRAILING MINUS is PNC's overdraft notation and must be read, or a negative
#: balance parses as positive and the account reconciles to the wrong sign.
_MONEY = r"(-?[\d,]*\.\d{2}-?)"
_BAL_LINE = re.compile(r"%s\s+%s\s+%s\s+%s" % (_MONEY, _MONEY, _MONEY, _MONEY))
#: The summary's own header, so the four figures are read from the row BELOW it
#: rather than from the first four money figures anywhere in the document.
_BAL_HEADER = re.compile(r"other\s+debits\s+balance", re.I)


def _money(v) -> Optional[float]:
    """A statement figure, including PNC's trailing-minus overdraft notation.

    Kept separate from `_num` deliberately: the ACTIVITY export carries its
    sign in a Credit/Debit column and never a trailing minus, so teaching the
    shared parser to strip one would widen it for no reason and quietly accept
    a malformed amount there.
    """
    if v is None:
        return None
    t = str(v).strip()
    neg = t.endswith("-")
    if neg:
        t = t[:-1].strip()
    n = _num(t)
    if n is None:
        return None
    return -abs(n) if neg else n
_PERIOD = re.compile(r"period\s+(\d{2}/\d{2}/\d{4})\s+to\s+(\d{2}/\d{2}/\d{4})",
                     re.I)
_ACCT = re.compile(r"Account\s+Number:\s*([X\-\d]+)", re.I)


def parse_statement_text(txt: str, source_file: str = "") -> dict:
    """Beginning and ending balance out of a PNC statement PDF's text.

    The August AMB6 statement carries them on one line under "Balance Summary":

        Beginning   Deposits and   Checks and    Ending
        balance     other credits  other debits  balance
        571,750.04  1,780,496.92   2,340,519.46  11,727.50

    Returns what it found and says when it found nothing -- a statement whose
    balances could not be read must not come back as zeros, which would tie
    against an empty month.
    """
    out = {"beginning_balance": None, "ending_balance": None,
           "credits_total": None, "debits_total": None,
           "period_start": None, "period_end": None,
           "account_suffix": None, "source_file": source_file}
    if not txt:
        out["error"] = ("No text in this PDF. A scanned statement cannot be "
                        "read without OCR; enter the balances by hand.")
        return out
    m = _PERIOD.search(txt)
    if m:
        out["period_start"] = _iso(m.group(1))
        out["period_end"] = _iso(m.group(2))
    a = _ACCT.search(txt)
    if a:
        out["account_suffix"] = a.group(1).strip()
    # ANCHORED TO THE SUMMARY HEADER when there is one. Loosening the number
    # pattern to accept `.00` makes more lines matchable, so "the first four
    # money figures in the document" stopped being a safe rule -- a total block
    # further up could win. Falls back to the whole text for any layout that
    # does not carry the header.
    h = _BAL_HEADER.search(txt)
    b = _BAL_LINE.search(txt, h.end()) if h else _BAL_LINE.search(txt)
    if b:
        out["beginning_balance"] = _money(b.group(1))
        out["credits_total"] = _money(b.group(2))
        out["debits_total"] = _money(b.group(3))
        out["ending_balance"] = _money(b.group(4))
        # The statement's own four figures have to agree before any of them is
        # trusted; a mis-parse usually shows up here first.
        beg, cr, db, end = (out["beginning_balance"], out["credits_total"],
                            out["debits_total"], out["ending_balance"])
        out["internally_consistent"] = abs((beg + cr - db) - end) < 0.01
    else:
        out["error"] = ("Could not find the balance summary line in this "
                        "statement.")
    return out


# ------------------------------------------------------------------ storing

def import_activity(rows, engine=None) -> dict:
    """Store parsed activity, skipping rows already held.

    Re-importing the same export is a normal thing to do -- an accountant
    re-pulls a month after a correction -- so identity is by ``row_hash`` and a
    second import adds only what is new.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    if not rows:
        return {"inserted": 0, "already_held": 0, "accounts": []}
    now = datetime.utcnow().isoformat(timespec="seconds")
    inserted = dup = 0
    with engine.begin() as conn:
        have = {r[0] for r in conn.execute(text(
            "SELECT row_hash FROM tr_activity")).fetchall()}
        for r in rows:
            if r["row_hash"] in have:
                dup += 1
                continue
            conn.execute(text(
                "INSERT INTO tr_activity (account_number, as_of_date, "
                " bai_control, transaction_type, amount, direction, "
                " signed_amount, reference, description, currency, "
                " source_file, imported_at, row_hash) VALUES "
                "(:a,:d,:b,:t,:m,:dir,:s,:r,:de,:c,:f,:i,:h)"),
                {"a": r["account_number"], "d": r["as_of_date"],
                 "b": r["bai_control"], "t": r["transaction_type"],
                 "m": r["amount"], "dir": r["direction"],
                 "s": r["signed_amount"], "r": r["reference"],
                 "de": r["description"], "c": r["currency"],
                 "f": r["source_file"], "i": now, "h": r["row_hash"]})
            have.add(r["row_hash"])
            inserted += 1
        known = {r[0] for r in conn.execute(text(
            "SELECT account_number FROM tr_accounts")).fetchall()}
        # An account seen in a file is registered even before anybody maps it to
        # an entity. An unmapped account is a question to answer on screen, not
        # a row to leave out.
        for acct in sorted({r["account_number"] for r in rows}):
            if acct in known:
                continue
            one = next(r for r in rows if r["account_number"] == acct)
            conn.execute(text(
                "INSERT INTO tr_accounts (account_number, bank_id, "
                " account_name, entityid, gl_cash_account, active, updated_at) "
                "VALUES (:a,:b,:n,NULL,:g,1,:t)"),
                {"a": acct, "b": one.get("bank_id"),
                 "n": one.get("account_name"),
                 "g": DEFAULT_CASH_ACCOUNT, "t": now})
    return {"inserted": inserted, "already_held": dup,
            "accounts": sorted({r["account_number"] for r in rows})}


def import_statement(parsed: dict, account_number: str, engine=None) -> dict:
    """Store a statement's balances. Refused when they could not be read."""
    engine = engine or get_engine()
    ensure_tables(engine)
    if parsed.get("error") or parsed.get("ending_balance") is None:
        return {"error": parsed.get("error")
                or "No ending balance could be read from this statement."}
    if parsed.get("internally_consistent") is False:
        return {"error": "The statement's own figures do not agree "
                         "(beginning + credits - debits does not equal ending), "
                         "so they were not stored."}
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO tr_statements (account_number, period_start, "
            " period_end, beginning_balance, ending_balance, credits_total, "
            " debits_total, source_file, imported_at) "
            "VALUES (:a,:ps,:pe,:b,:e,:c,:d,:f,:i)"),
            {"a": str(account_number).strip(),
             "ps": parsed.get("period_start"), "pe": parsed.get("period_end"),
             "b": parsed.get("beginning_balance"),
             "e": parsed.get("ending_balance"),
             "c": parsed.get("credits_total"), "d": parsed.get("debits_total"),
             "f": parsed.get("source_file"), "i": now})
    return {"ok": True, "period_end": parsed.get("period_end"),
            "ending_balance": parsed.get("ending_balance")}



def create_account(account_number: str, entityid: str = "",
                   gl_cash_account: str = "", account_name: str = "",
                   user: str = "", engine=None) -> dict:
    """Register a bank account by hand.

    An account normally registers itself the first time its activity is
    imported. That is not enough when PNC will only serve 90 days of activity
    and an account has been quiet longer than that: its June statement shows a
    real balance -- PPI Life Storage NY holds 119,701.35 -- but there is no
    transaction anywhere to introduce it.

    THE FULL ACCOUNT NUMBER IS REQUIRED AND IS NOT INFERRED. The statement
    prints a mask (`XX-XXXX-7891`) and several of these accounts sit in obvious
    number ranges, so guessing the hidden digits would usually work and would
    occasionally be wrong -- and a wrong account number silently splits one
    account into two the moment real activity arrives under the true number.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    acct = _clean(account_number)
    if not acct:
        return {"error": "An account number is required."}
    if not acct.isdigit():
        return {"error": "%r is not an account number: digits only, exactly as "
                         "PNC exports it." % account_number}
    if gl_cash_account and gl_cash_account not in CASH_ACCOUNTS:
        return {"error": "%s is not one of the cash accounts: %s"
                         % (gl_cash_account, ", ".join(CASH_ACCOUNTS))}
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as conn:
        seen = conn.execute(text(
            "SELECT account_number FROM tr_accounts WHERE account_number = :a"),
            {"a": acct}).fetchone()
        if seen:
            return {"error": "%s is already registered." % acct,
                    "account_number": acct}
        conn.execute(text(
            "INSERT INTO tr_accounts (account_number, bank_id, account_name, "
            " entityid, gl_cash_account, active, updated_by, updated_at) "
            "VALUES (:a, NULL, :n, :e, :g, 1, :u, :t)"),
            {"a": acct, "n": (account_name or "").strip() or None,
             "e": (str(entityid).strip().upper() or None),
             "g": gl_cash_account or DEFAULT_CASH_ACCOUNT,
             "u": user, "t": now})
    return {"ok": True, "account_number": acct,
            "gl_cash_account": gl_cash_account or DEFAULT_CASH_ACCOUNT}

def set_account(account_number, entityid=None, gl_cash_account=None,
                active=None, user: str = "", engine=None) -> dict:
    """Map a bank account to an entity and to its GL cash account."""
    engine = engine or get_engine()
    ensure_tables(engine)
    if gl_cash_account and gl_cash_account not in CASH_ACCOUNTS:
        return {"error": "%s is not one of the cash accounts: %s"
                         % (gl_cash_account, ", ".join(CASH_ACCOUNTS))}
    sets, params = [], {"a": str(account_number).strip()}
    if entityid is not None:
        sets.append("entityid = :e")
        params["e"] = (str(entityid).strip().upper() or None)
    if gl_cash_account is not None:
        sets.append("gl_cash_account = :g")
        params["g"] = gl_cash_account
    if active is not None:
        sets.append("active = :ac")
        params["ac"] = 1 if active else 0
    if not sets:
        return {"error": "Nothing to change."}
    params["u"] = user
    params["t"] = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE tr_accounts SET %s, updated_by = :u, updated_at = :t "
            " WHERE account_number = :a" % ", ".join(sets)), params)
    return {"ok": True}


# -------------------------------------------------------- the three-way tie

def opening_balance(account_number: str, period: str, engine=None) -> dict:
    """The period's opening figure, carried from the prior period's close.

    Jim, Sep 17 2026: "Carry the prior period's computed ending forward."

    It chains deliberately. A month that has never been closed has NO opening
    figure and this says so rather than starting from zero: a reconciliation
    opening at zero would report the entire balance as a difference, and one
    that silently re-based on a fresh statement figure would hide a break in
    the chain instead of showing it.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    prev = prior_period(period)
    if not prev:
        return {"opening": None, "source": None,
                "reason": "%r is not a period like 202608." % period}
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT computed_ending, status FROM tr_periods "
            " WHERE account_number = :a AND period = :p"),
            {"a": account_number, "p": prev}).fetchone()
        if row and row[0] is not None:
            return {"opening": float(row[0]),
                    "source": "carried from %s" % prev,
                    "prior_status": row[1]}
        st = conn.execute(text(
            "SELECT beginning_balance FROM tr_statements "
            " WHERE account_number = :a AND period_end LIKE :pe "
            " ORDER BY id DESC"),
            {"a": account_number,
             "pe": "%s-%s%%" % (period[:4], period[4:])}).fetchone()
    if st and st[0] is not None:
        # Named as a different provenance on purpose: this is the statement's
        # word for the opening, not our own close.
        return {"opening": float(st[0]),
                "source": "this period's statement (no prior period closed)"}
    return {"opening": None, "source": None,
            "reason": ("No prior period has been closed for this account and no "
                       "statement has been filed for %s, so there is no opening "
                       "balance to carry forward." % period)}


def reconcile(account_number: str, period: str, gl_net=None,
              engine=None) -> dict:
    """Opening plus bank movement, against the statement and against the ledger.

    Three figures from three places that have to agree. Each comparison is
    reported SEPARATELY: rolling them into one "difference" would hide which
    leg disagrees, and which leg it is happens to be the only thing the number
    is for.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    acct = str(account_number).strip()
    like = "%s-%s%%" % (period[:4], period[4:])

    with engine.connect() as conn:
        act = pd.read_sql(text(
            "SELECT * FROM tr_activity WHERE account_number = :a "
            "  AND as_of_date LIKE :p ORDER BY as_of_date"),
            conn, params={"a": acct, "p": like})
        st = conn.execute(text(
            "SELECT beginning_balance, ending_balance, source_file "
            "  FROM tr_statements WHERE account_number = :a "
            "   AND period_end LIKE :p ORDER BY id DESC"),
            {"a": acct, "p": like}).fetchone()
        meta = conn.execute(text(
            "SELECT entityid, gl_cash_account, account_name "
            "  FROM tr_accounts WHERE account_number = :a"),
            {"a": acct}).fetchone()

    op = opening_balance(acct, period, engine)
    movement = float(act["signed_amount"].sum()) if not act.empty else 0.0
    opening = op.get("opening")
    computed = None if opening is None else opening + movement

    out = {
        "account_number": acct,
        "entityid": meta[0] if meta else None,
        "gl_cash_account": (meta[1] if meta else None) or DEFAULT_CASH_ACCOUNT,
        "account_name": meta[2] if meta else None,
        "period": period,
        "opening_balance": opening,
        "opening_source": op.get("source"),
        "opening_reason": op.get("reason"),
        "transaction_count": int(len(act)),
        "credits": (float(act.loc[act["direction"] == "credit", "amount"].sum())
                    if not act.empty else 0.0),
        "debits": (float(act.loc[act["direction"] == "debit", "amount"].sum())
                   if not act.empty else 0.0),
        "bank_movement": movement,
        "computed_ending": computed,
        "statement_ending": (float(st[1]) if st and st[1] is not None else None),
        "statement_file": (st[2] if st else None),
        "gl_net": (None if gl_net is None else float(gl_net)),
    }

    if computed is not None and out["statement_ending"] is not None:
        d = computed - out["statement_ending"]
        out["statement_difference"] = d
        out["ties_to_statement"] = abs(d) < 0.01
    else:
        out["statement_difference"] = None
        out["ties_to_statement"] = None

    if gl_net is not None:
        d = movement - float(gl_net)
        out["gl_difference"] = d
        out["ties_to_gl"] = abs(d) < 0.01
    else:
        out["gl_difference"] = None
        out["ties_to_gl"] = None

    out["headline"] = _tie_headline(out)
    return out


def _tie_headline(r: dict) -> str:
    if r["opening_balance"] is None:
        return r.get("opening_reason") or "No opening balance to carry forward."
    parts = ["%s transactions move %s, from %s to %s."
             % (r["transaction_count"], _m(r["bank_movement"]),
                _m(r["opening_balance"]), _m(r["computed_ending"]))]
    if r["ties_to_statement"] is True:
        parts.append("That ties to the statement.")
    elif r["ties_to_statement"] is False:
        parts.append("The statement says %s, a difference of %s."
                     % (_m(r["statement_ending"]),
                        _m(r["statement_difference"])))
    else:
        parts.append("No statement has been filed for this period.")
    if r["ties_to_gl"] is True:
        parts.append("The ledger's cash account moves by the same amount.")
    elif r["ties_to_gl"] is False:
        parts.append("The ledger moves %s, a difference of %s."
                     % (_m(r["gl_net"]), _m(r["gl_difference"])))
    return " ".join(parts)


def close_period(account_number: str, period: str, user: str = "",
                 engine=None) -> dict:
    """Record the computed ending so the next period can open from it."""
    engine = engine or get_engine()
    r = reconcile(account_number, period, engine=engine)
    if r["computed_ending"] is None:
        return {"error": r.get("opening_reason") or "Nothing to close."}
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT id FROM tr_periods WHERE account_number = :a "
            "  AND period = :p"), {"a": account_number, "p": period}).fetchone()
        if row:
            conn.execute(text(
                "UPDATE tr_periods SET opening_balance = :o, "
                " computed_ending = :c, status = 'closed', closed_by = :u, "
                " closed_at = :t WHERE id = :i"),
                {"o": r["opening_balance"], "c": r["computed_ending"],
                 "u": user, "t": now, "i": row[0]})
        else:
            conn.execute(text(
                "INSERT INTO tr_periods (account_number, period, "
                " opening_balance, computed_ending, status, closed_by, "
                " closed_at) VALUES (:a,:p,:o,:c,'closed',:u,:t)"),
                {"a": account_number, "p": period, "o": r["opening_balance"],
                 "c": r["computed_ending"], "u": user, "t": now})
    return {"ok": True, "period": period,
            "computed_ending": r["computed_ending"]}


def seed_opening(account_number: str, period: str, amount, user: str = "",
                 engine=None) -> dict:
    """Set the opening for the FIRST period, which has nothing before it.

    Stored as the prior period's close, because that is what it is: the chain
    has to start somewhere and the start should look like every other link.
    Marked ``seeded`` so it is never mistaken for a period that was reconciled.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    amt = _num(amount)
    if amt is None:
        return {"error": "%r is not an amount." % amount}
    prev = prior_period(period)
    if not prev:
        return {"error": "%r is not a period like 202608." % period}
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT id FROM tr_periods WHERE account_number = :a "
            "  AND period = :p"), {"a": account_number, "p": prev}).fetchone()
        if row:
            conn.execute(text(
                "UPDATE tr_periods SET computed_ending = :c, "
                " status = 'seeded', closed_by = :u, closed_at = :t, "
                " note = 'opening balance entered by hand' WHERE id = :i"),
                {"c": amt, "u": user, "t": now, "i": row[0]})
        else:
            conn.execute(text(
                "INSERT INTO tr_periods (account_number, period, "
                " computed_ending, status, closed_by, closed_at, note) VALUES "
                "(:a,:p,:c,'seeded',:u,:t,'opening balance entered by hand')"),
                {"a": account_number, "p": prev, "c": amt, "u": user,
                 "t": now})
    return {"ok": True, "seeded_period": prev, "amount": amt}



# ------------------------------------ starting the chain from a statement

def match_account_by_suffix(suffix: str, engine=None):
    """Which known account a statement's masked number refers to.

    PNC prints only the last four digits (``XX-XXXX-5765``), so that is all
    there is to route on. Measured across the 50 accounts in the September
    90-day export, all fifty last-four groups are unique -- but that is a fact
    about today's accounts, not a guarantee, so an AMBIGUOUS suffix is refused
    rather than resolved by picking one. A statement filed against the wrong
    account would corrupt a reconciliation silently.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    rx = _mask_pattern(suffix)
    if rx is None:
        return {"error": "No account digits could be read from the statement."}
    with engine.connect() as conn:
        rows = [str(r[0]).strip() for r in conn.execute(text(
            "SELECT account_number FROM tr_accounts")).fetchall()]
    hits = [a for a in rows if rx.match(a)]
    if not hits:
        return {"error": "No imported account matches %s. Import that "
                         "account's activity first, or add the account."
                         % suffix, "suffix": suffix}
    if len(hits) > 1:
        return {"error": "More than one account matches %s (%s), so this "
                         "statement cannot be routed automatically. File it "
                         "against the account by hand."
                         % (suffix, ", ".join(sorted(hits))), "suffix": suffix}
    return {"account_number": hits[0], "suffix": suffix}


def _mask_pattern(suffix):
    r"""Turn PNC's masked number into a pattern for the WHOLE account number.

    THE MASK IS NOT ALWAYS A TAIL. `XX-XXXX-5765` hides the front, but
    `790-XXXXX55` hides the MIDDLE -- and reading "the last four visible
    digits" off that gives `790` + `55` = `79055` -> `9055`, an account that
    does not exist. Five of Jim's six June statements with real balances were
    reported as unknown accounts for exactly that reason; every one of them was
    already registered.

    So the mask is read as what it is: each run of X is that many unknown
    digits, each printed digit is itself, and the whole thing must match the
    whole account number. `790-XXXXX55` becomes `^790\d{5}55$`, which picks
    7900021255 and nothing else.
    """
    import re as _re
    t = str(suffix or "")
    if not any(c.isdigit() for c in t):
        return None
    parts = []
    for run in _re.findall(r"[Xx]+|\d+|[^Xx\d]+", t):
        if run[0] in "Xx":
            parts.append(r"\d{%d}" % len(run))
        elif run[0].isdigit():
            parts.append(_re.escape(run))
        # Separators (dashes, spaces) are formatting and carry no digits.
    return _re.compile("^" + "".join(parts) + "$")


def seed_from_statement(account_number: str, period: str, user: str = "",
                        engine=None) -> dict:
    """Open the chain from the PRIOR period's filed statement.

    Jim, Sep 17 2026, asking whether to type fifty opening balances or read
    them off the June statements. The statement is the better source: it is an
    external authority, it carries its own arithmetic check, and the figure
    stays traceable to a named file instead of to somebody's typing.

    THIS IS NOT THE SAME AS RE-BASING. `opening_balance()` deliberately never
    reads a statement -- it carries the prior period's computed ending, so a
    break in the chain shows up as a difference instead of being papered over.
    Starting the chain is the one case where a statement IS the right source,
    because there is nothing behind it to contradict. So this refuses the
    moment a real close exists, and the distinction is the whole point of the
    function.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    acct = str(account_number).strip()
    prev = prior_period(period)
    if not prev:
        return {"error": "%r is not a period like 202607." % period}

    with engine.connect() as conn:
        closed = conn.execute(text(
            "SELECT period, status FROM tr_periods "
            " WHERE account_number = :a AND computed_ending IS NOT NULL "
            "   AND status = 'closed' ORDER BY period"),
            {"a": acct}).fetchall()
        st = conn.execute(text(
            "SELECT ending_balance, period_end, source_file FROM tr_statements "
            " WHERE account_number = :a AND period_end LIKE :p "
            " ORDER BY id DESC"),
            {"a": acct, "p": "%s-%s%%" % (prev[:4], prev[4:])}).fetchone()

    if closed:
        return {"error": "This account already has a reconciled period (%s), "
                         "so its opening is carried forward rather than seeded. "
                         "Seeding now would hide any break in the chain."
                         % closed[0][0]}
    if not st or st[0] is None:
        return {"error": "No statement is on file for %s. Import that month's "
                         "statement PDF first, then seed from it." % prev}

    res = seed_opening(acct, period, float(st[0]), user, engine)
    if res.get("error"):
        return res
    # The note is the audit trail: it names the file the figure came from, so
    # the starting point can be checked years later without asking anybody.
    note = ("opening from the %s statement ending %s (%s)"
            % (prev, st[1], (st[2] or "statement")[:120]))
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE tr_periods SET note = :n WHERE account_number = :a "
            "  AND period = :p"), {"n": note, "a": acct, "p": prev})
    return {**res, "source": note, "statement_ending": float(st[0]),
            "period_end": st[1]}

def accounts(engine=None):
    """Every known bank account, with the latest period it has been closed at."""
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT account_number, bank_id, account_name, entityid, "
            "       gl_cash_account, active FROM tr_accounts "
            " ORDER BY account_number")).fetchall()
        last = {}
        for a, p, c in conn.execute(text(
                "SELECT account_number, period, computed_ending "
                "  FROM tr_periods WHERE computed_ending IS NOT NULL "
                " ORDER BY period")).fetchall():
            last[a] = (p, c)
        # Activity rolled up by month, so the position below can be carried
        # forward from the last close without reading every transaction.
        by_month = {}
        for a, ym, net, n, through in conn.execute(text(
                "SELECT account_number, SUBSTR(as_of_date, 1, 7) AS ym, "
                "       SUM(signed_amount), COUNT(*), MAX(as_of_date) "
                "  FROM tr_activity WHERE as_of_date IS NOT NULL "
                " GROUP BY account_number, SUBSTR(as_of_date, 1, 7)")).fetchall():
            by_month.setdefault(a, []).append((ym, float(net or 0.0), int(n),
                                               through))

    out = []
    for r in rows:
        acct = r[0]
        period, balance = last.get(acct, (None, None))
        row = {"account_number": acct, "bank_id": r[1], "account_name": r[2],
               "entityid": r[3],
               "gl_cash_account": r[4] or DEFAULT_CASH_ACCOUNT,
               "active": bool(r[5]),
               "last_period": period,
               "last_balance": (float(balance) if balance is not None else None)}
        row.update(_position(period, balance, by_month.get(acct, [])))
        out.append(row)
    return out


def _position(last_period, last_balance, months) -> dict:
    """What we can honestly say the account holds right now.

    CURRENT LEDGER is carried the same way the opening balance is: the last
    closed ending plus every imported transaction dated after it. It is a
    computed position from what we hold, NOT a reading of PNC's ledger balance,
    and it says which date it runs through so nobody reads a stale import as
    today's cash.

    CURRENT AVAILABLE IS NOT DERIVABLE AND IS RETURNED AS None. Available is
    ledger less holds, float and pending debits -- facts that exist only at the
    bank and never appear in an activity export. Showing the ledger figure in
    an available column would be inventing the one number a treasurer acts on.
    It arrives when the PNC connection does.
    """
    ym = ("%s-%s" % (last_period[:4], last_period[4:])) if last_period else None
    after = [m for m in months if ym is None or m[0] > ym]
    net = sum(m[1] for m in after)
    txns = sum(m[2] for m in after)
    through = max([m[3] for m in after if m[3]], default=None)
    pos = {
        "activity_since_close": (net if after else 0.0),
        "transactions_since_close": txns,
        "activity_through": through,
        "unclosed_months": sorted(m[0] for m in after),
        "current_available": None,
        "available_reason": ("Available balance is held only at the bank -- it "
                             "is the ledger less holds, float and pending "
                             "debits, none of which appear in an activity "
                             "export. It arrives with the PNC connection."),
    }
    if last_balance is None:
        pos["current_ledger"] = None
        pos["ledger_reason"] = (
            "No period has been closed for this account yet, so there is "
            "nothing to carry forward. Seed the opening balance for its first "
            "month and reconcile it." if not months else
            "Activity has been imported but no period has been closed, so "
            "there is no balance to carry it forward from.")
    else:
        pos["current_ledger"] = float(last_balance) + net
        pos["ledger_reason"] = (
            "Closed %s at %s%s." % (
                last_period, _m(last_balance),
                (", plus %d transaction%s through %s netting %s"
                 % (txns, "" if txns == 1 else "s", through, _m(net)))
                if txns else ", with no activity imported since"))
    return pos


def _m(v) -> str:
    try:
        return "{:,.2f}".format(float(v))
    except (TypeError, ValueError):
        return "-"


# ------------------------------------------------------------------ matching

#: How far apart a bank date and a GL entry date may be and still be the same
#: transaction. A wire posts the day it is sent; a cheque clears days later.
#: Beyond this the pairing is still OFFERED but flagged, never silently made.
NEAR_DAYS = 5


def _gl_cash_lines(entityid: str, gl_account: str, period: str,
                   engine) -> pd.DataFrame:
    """The ledger's own cash-account entries for the period.

    BALFOR 'B' rows are balance-forward carriers, not entries -- including them
    would put a year's opening balance into a list of transactions to match.
    """
    try:
        with engine.connect() as conn:
            gl = pd.read_sql(text(
                'SELECT "ENTRDATE", "ACCTNUM", "ITEM", "REF", "DESCRPN", '
                '       "AMT", "BALFOR", "PERIOD", "ENTITYID" '
                '  FROM gl_detail '
                ' WHERE UPPER(TRIM("ENTITYID")) = :e '
                '   AND UPPER(TRIM("ACCTNUM")) = :a '
                '   AND TRIM("PERIOD") = :p'),
                conn, params={"e": (entityid or "").strip().upper(),
                              "a": (gl_account or "").strip().upper(),
                              "p": str(period).strip()})
    except Exception as e:
        logger.warning("gl cash lines unavailable: %s", e)
        return pd.DataFrame()
    if gl.empty:
        return gl
    gl = gl[gl["BALFOR"].astype(str).str.strip().str.upper() != "B"].copy()
    gl["_amt"] = pd.to_numeric(gl["AMT"], errors="coerce").fillna(0.0)
    gl["_date"] = gl["ENTRDATE"].map(_iso)
    return gl


def _days_apart(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    try:
        return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)
    except ValueError:
        return None


def match(account_number: str, period: str, engine=None) -> dict:
    """Pair bank transactions with the ledger's cash entries for a period.

    PAIRED, NOT SET-COMPARED. August 2026 carries the amount 285.92 seven
    times; comparing sets of amounts would call all seven matched the moment
    one was. Items are consumed as they are used, so seven bank lines need
    seven ledger lines.

    WITHIN AN AMOUNT, THE NEAREST DATE WINS, and a pairing further apart than
    `NEAR_DAYS` is still offered but flagged -- a cheque takes days to clear and
    that is normal, while a month apart usually means the wrong pair.

    WHAT IS LEFT OVER IS THE RECONCILIATION. Ledger entries with no bank line
    are deposits in transit or outstanding payments; bank lines with no ledger
    entry are activity not yet recorded. Those two lists ARE the reconciling
    items -- the September screenshot's 510,000.00 is four ledger deposits the
    bank had not yet seen -- so they are returned in full rather than counted.

    Nothing here writes a journal entry or decides an offset.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    acct = str(account_number).strip()
    like = "%s-%s%%" % (period[:4], period[4:])

    with engine.connect() as conn:
        bank = pd.read_sql(text(
            "SELECT id, as_of_date, amount, direction, signed_amount, "
            "       reference, description, transaction_type "
            "  FROM tr_activity WHERE account_number = :a "
            "   AND as_of_date LIKE :p ORDER BY as_of_date, id"),
            conn, params={"a": acct, "p": like})
        meta = conn.execute(text(
            "SELECT entityid, gl_cash_account FROM tr_accounts "
            " WHERE account_number = :a"), {"a": acct}).fetchone()
        manual = {r[0]: r[1] for r in conn.execute(text(
            "SELECT bank_id, gl_item FROM tr_matches "
            " WHERE account_number = :a AND period = :p"),
            {"a": acct, "p": period}).fetchall()}

    entityid = meta[0] if meta else None
    gl_account = (meta[1] if meta else None) or DEFAULT_CASH_ACCOUNT
    out = {"account_number": acct, "period": period, "entityid": entityid,
           "gl_cash_account": gl_account, "matched": [], "bank_only": [],
           "gl_only": [], "ambiguous": 0, "manual_count": len(manual)}

    if not entityid:
        out["error"] = ("This bank account is not mapped to an entity, so "
                        "there is no ledger to match against.")
        # The bank side is still known, and showing it is what makes the
        # refusal actionable: the accountant can see what came through the
        # account while being told which mapping is missing. Same reason the
        # no-GL-feed branch below carries it.
        out["bank_only"] = [_bank_row(r) for _, r in bank.iterrows()]
        return out

    gl = _gl_cash_lines(entityid, gl_account, period, engine)
    if gl.empty:
        out["error"] = ("No %s entries for %s in period %s. Either the GL feed "
                        "has not been refreshed or this account posts elsewhere."
                        % (gl_account, entityid, period))
        out["bank_only"] = [_bank_row(r) for _, r in bank.iterrows()]
        return out

    # Manual pairings are honoured first and never re-decided.
    used_gl, matched = set(), []
    gl_by_item = {str(r["ITEM"]).strip(): r for _, r in gl.iterrows()}
    for bid, item in manual.items():
        row = bank[bank["id"] == bid]
        g = gl_by_item.get(str(item).strip())
        if row.empty or g is None:
            continue
        used_gl.add(str(item).strip())
        matched.append(_pair(row.iloc[0], g, manual=True))
    done_bank = {m["bank_id"] for m in matched}

    # Then automatic pairing, amount group by amount group.
    by_amount = {}
    for _, g in gl.iterrows():
        if str(g["ITEM"]).strip() in used_gl:
            continue
        by_amount.setdefault(round(float(g["_amt"]), 2), []).append(g)

    ambiguous = 0
    for _, b in bank.iterrows():
        if b["id"] in done_bank:
            continue
        amt = round(float(b["signed_amount"]), 2)
        cands = by_amount.get(amt) or []
        if not cands:
            continue
        if len(cands) > 1:
            ambiguous += 1
        # Nearest entry date wins; an undated candidate sorts last rather than
        # being treated as a perfect match.
        cands.sort(key=lambda g: (_days_apart(b["as_of_date"], g["_date"])
                                  if _days_apart(b["as_of_date"], g["_date"])
                                  is not None else 9999))
        g = cands.pop(0)
        matched.append(_pair(b, g, manual=False))
        used_gl.add(str(g["ITEM"]).strip())

    paired_bank = {m["bank_id"] for m in matched}
    out["matched"] = matched
    out["ambiguous"] = ambiguous
    out["bank_only"] = [_bank_row(r) for _, r in bank.iterrows()
                        if r["id"] not in paired_bank]
    out["gl_only"] = [_gl_row(g) for _, g in gl.iterrows()
                      if str(g["ITEM"]).strip() not in used_gl]

    out["bank_total"] = float(bank["signed_amount"].sum()) if not bank.empty else 0.0
    out["gl_total"] = float(gl["_amt"].sum())
    out["difference"] = out["bank_total"] - out["gl_total"]
    out["ties"] = abs(out["difference"]) < 0.01
    # Named the way an accountant names them, because that is what they are.
    out["deposits_in_transit"] = sum(r["amount"] for r in out["gl_only"]
                                     if r["amount"] > 0)
    out["outstanding_payments"] = sum(r["amount"] for r in out["gl_only"]
                                      if r["amount"] < 0)
    out["unrecorded_on_the_ledger"] = sum(r["signed_amount"]
                                          for r in out["bank_only"])
    out["headline"] = _match_headline(out)
    return out


def _pair(b, g, manual: bool) -> dict:
    d = _days_apart(b["as_of_date"], g["_date"])
    return {
        "bank_id": int(b["id"]),
        "bank_date": b["as_of_date"],
        "bank_amount": float(b["signed_amount"]),
        "bank_description": (b["description"] or "")[:160],
        "gl_item": str(g["ITEM"]).strip(),
        "gl_date": g["_date"],
        "gl_amount": float(g["_amt"]),
        "gl_description": str(g["DESCRPN"] or "")[:160],
        "gl_ref": str(g["REF"] or "").strip(),
        "days_apart": d,
        # Offered, not hidden: a pair this far apart is usually the wrong pair.
        "far_apart": (d is not None and d > NEAR_DAYS),
        "manual": manual,
    }


def _bank_row(r) -> dict:
    return {"bank_id": int(r["id"]), "date": r["as_of_date"],
            "amount": float(r["amount"]), "direction": r["direction"],
            "signed_amount": float(r["signed_amount"]),
            "reference": r["reference"],
            "transaction_type": r["transaction_type"],
            "description": (r["description"] or "")[:200]}


def _gl_row(g) -> dict:
    return {"gl_item": str(g["ITEM"]).strip(), "date": g["_date"],
            "amount": float(g["_amt"]), "ref": str(g["REF"] or "").strip(),
            "description": str(g["DESCRPN"] or "")[:200],
            # An accountant reads these as two different things.
            "kind": "deposit in transit" if float(g["_amt"]) > 0
                    else "outstanding payment"}


def _match_headline(o: dict) -> str:
    parts = ["%d of %d bank items matched to the ledger."
             % (len(o["matched"]), len(o["matched"]) + len(o["bank_only"]))]
    if o["ties"]:
        parts.append("The totals agree.")
    else:
        parts.append("The totals differ by %s." % _m(o["difference"]))
    if o["gl_only"]:
        parts.append("%d ledger entr%s not on the bank: %s of deposits in "
                     "transit and %s of outstanding payments."
                     % (len(o["gl_only"]),
                        "y is" if len(o["gl_only"]) == 1 else "ies are",
                        _m(o["deposits_in_transit"]),
                        _m(abs(o["outstanding_payments"]))))
    if o["bank_only"]:
        parts.append("%d bank item%s not on the ledger, netting %s."
                     % (len(o["bank_only"]), "" if len(o["bank_only"]) == 1
                        else "s", _m(o["unrecorded_on_the_ledger"])))
    if o["ambiguous"]:
        parts.append("%d had more than one candidate at the same amount and "
                     "were paired on the nearest date." % o["ambiguous"])
    return " ".join(parts)


def set_match(account_number: str, period: str, bank_id: int, gl_item,
              user: str = "", engine=None) -> dict:
    """Pin a pairing by hand, or remove one when `gl_item` is empty.

    A manual pairing is honoured before any automatic one and is never
    re-decided, because the accountant looked at both sides and the matcher
    only looked at the amount.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    now = datetime.utcnow().isoformat(timespec="seconds")
    item = (str(gl_item).strip() if gl_item is not None else "")
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM tr_matches WHERE account_number = :a AND period = :p "
            "  AND bank_id = :b"),
            {"a": account_number, "p": period, "b": int(bank_id)})
        if item:
            conn.execute(text(
                "INSERT INTO tr_matches (account_number, period, bank_id, "
                " gl_item, matched_by, matched_at) VALUES (:a,:p,:b,:i,:u,:t)"),
                {"a": account_number, "p": period, "b": int(bank_id),
                 "i": item, "u": user, "t": now})
    return {"ok": True, "bank_id": int(bank_id), "gl_item": item or None}
