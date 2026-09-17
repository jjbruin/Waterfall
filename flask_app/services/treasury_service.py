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
    if df is None or df.empty:
        return {"rows": [], "skipped": [], "accounts": []}
    cols = {str(c).strip().lower(): c for c in df.columns}
    need = ("asofdate", "accountnumber", "amount", "credit/debit")
    missing = [n for n in need if n not in cols]
    if missing:
        return {"rows": [], "skipped": [],
                "error": "Not a PNC activity export: missing %s. Found: %s"
                         % (", ".join(missing), ", ".join(list(df.columns)[:10]))}

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


_BAL_LINE = re.compile(
    r"([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})")
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
    b = _BAL_LINE.search(txt)
    if b:
        out["beginning_balance"] = _num(b.group(1))
        out["credits_total"] = _num(b.group(2))
        out["debits_total"] = _num(b.group(3))
        out["ending_balance"] = _num(b.group(4))
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
    return [{"account_number": r[0], "bank_id": r[1], "account_name": r[2],
             "entityid": r[3],
             "gl_cash_account": r[4] or DEFAULT_CASH_ACCOUNT,
             "active": bool(r[5]),
             "last_period": last.get(r[0], (None, None))[0],
             "last_balance": last.get(r[0], (None, None))[1]}
            for r in rows]


def _m(v) -> str:
    try:
        return "{:,.2f}".format(float(v))
    except (TypeError, ValueError):
        return "-"
