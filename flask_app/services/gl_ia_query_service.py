"""GL and IA detail queries with the CFO's filters, against our own tables.

The CFO's workbook `GL & IA Queries with Filters - 09182026.xlsx` carries the two
Spreadsheet Server queries he runs against MRI, and what he wants to be able to vary:

    GL Query   multiple entities / change period / select account(s) / export
    IA Query   multiple investment IDs / multiple investor IDs / select date /
               MajorType(s) / SubType(s) / export

WE DO NOT RE-RUN HIS SQL. `queries/MRI_GL_Detail.sql` is already that query --
"the Spreadsheet Server NEW JOURNAL query one-for-one, with the &SPARM smart
parameters removed" -- and `queries/MRI_IA_Transactions.sql` is the IA one. Both are
imported into `gl_detail` and `ia_transactions`. So the ask is not a second copy of
the SQL; it is to put the parameters BACK, against the copy we already hold (Jim,
Sep 19 2026: "since we are already pulling these tables into our database we can
have the query hit our tables"). Pasting his SQL in again would be a second engine
for the same numbers -- see CLAUDE.md, ONE NUMBER, ONE ENGINE.

Consequences of reading our copy, both of which the screen states rather than hides:
  * the data is as fresh as the last MRI refresh, not live. The freshness is
    reported with every result.
  * `MRI_GL_Detail.sql` is bounded at `PERIOD >= '202401'`, so periods before that
    are not in our copy and the tool says so rather than returning an empty grid
    that reads as "no activity".

NO USER INPUT IS EVER CONCATENATED INTO SQL. Every filter is a bound parameter and
every IN list is an expanding bindparam. The column names that can be sorted or
selected are matched against a fixed allow-list, never passed through.
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import bindparam, text

logger = logging.getLogger(__name__)

# The row cap exists so one unfiltered query cannot take the 2GB container down with
# it. It is a REPORTED cap: the result says it was reached, so a truncated grid can
# never be mistaken for the whole answer.
MAX_ROWS = 5000
EXPORT_MAX_ROWS = 100_000

# The period floor `MRI_GL_Detail.sql` imports from. Stated, not guessed -- a query
# for 2023 returns nothing, and the reason is the import bound, not the ledger.
GL_PERIOD_FLOOR = '202401'

GL_COLUMNS = [
    ('ENTITYID', 'Entity'), ('PERIOD', 'Period'), ('ENTRDATE', 'Entry Date'),
    ('ACCTNUM', 'Account'), ('ACCTNAME', 'Account Name'),
    ('BASIS', 'Basis'), ('BALFOR', 'Bal/Fwd'),
    ('ITEM', 'Item'), ('REF', 'Ref'), ('DESCRPN', 'Description'),
    ('SEGMENTID', 'Segment'), ('RLTDENTITY', 'Related Entity'),
    ('RLTDENTITY_NAME', 'Related Entity Name'), ('AMT', 'Amount'),
]

IA_COLUMNS = [
    ('InvestmentID', 'Investment ID'), ('InvestmentName', 'Investment Name'),
    ('InvestorID', 'Investor ID'), ('InvestorName', 'Investor Name'),
    ('TransactionDate', 'Transaction Date'), ('EffectiveDate', 'Effective Date'),
    ('MajorType', 'Major Type'), ('Typename', 'Sub Type'), ('Amount', 'Amount'),
]

# Which date column the IA filter applies to. The CFO's query filters on
# contributiondate / distributiondate, which import as TransactionDate.
IA_DATE_FIELDS = {'TransactionDate', 'EffectiveDate'}


def _quote(engine, name: str) -> str:
    """Quote an identifier for this dialect. Only ever called with names from the
    module's own allow-lists, never with anything a caller supplied."""
    return f'"{name}"' if engine.dialect.name == 'postgresql' else f'[{name}]'


def _has_table(engine, table: str) -> bool:
    from sqlalchemy import inspect
    try:
        return table in inspect(engine).get_table_names()
    except Exception:
        return False


def _clean_list(values) -> List[str]:
    """A list of non-empty strings, de-duplicated, order preserved."""
    out, seen = [], set()
    for v in values or []:
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _as_date(value) -> Optional[date]:
    if value in (None, ''):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(str(value)).date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# What is available to pick
# ---------------------------------------------------------------------------

def gl_filter_options(engine) -> Dict[str, Any]:
    """Entities, periods, accounts and bases actually present in `gl_detail`.

    Read from the DATA, not from a list somebody maintains: a picker offering an
    entity with no rows sends the analyst looking for a bug that is not there.
    """
    if not _has_table(engine, 'gl_detail'):
        return {'available': False,
                'reason': 'The GL detail table has not been imported yet. '
                          'Run "MRI_GL_Detail" from Data Management > MRI Data.',
                'entities': [], 'periods': [], 'accounts': [], 'bases': []}

    q = _quote(engine, 'ENTITYID')
    with engine.connect() as c:
        ents = [r[0] for r in c.execute(text(
            f'SELECT DISTINCT {q} FROM gl_detail WHERE {q} IS NOT NULL '
            f'ORDER BY {q}')).fetchall()]
        pq = _quote(engine, 'PERIOD')
        pers = [str(r[0]) for r in c.execute(text(
            f'SELECT DISTINCT {pq} FROM gl_detail WHERE {pq} IS NOT NULL '
            f'ORDER BY {pq}')).fetchall()]
        aq, nq = _quote(engine, 'ACCTNUM'), _quote(engine, 'ACCTNAME')
        accts = [{'account': str(r[0]), 'name': r[1]} for r in c.execute(text(
            f'SELECT DISTINCT {aq}, {nq} FROM gl_detail WHERE {aq} IS NOT NULL '
            f'ORDER BY {aq}')).fetchall()]
        bq = _quote(engine, 'BASIS')
        bases = [str(r[0]) for r in c.execute(text(
            f'SELECT DISTINCT {bq} FROM gl_detail WHERE {bq} IS NOT NULL '
            f'ORDER BY {bq}')).fetchall()]

        names = {}
        if _has_table(engine, 'entities'):
            eq, nmq = _quote(engine, 'ENTITYID'), _quote(engine, 'NAME')
            names = {str(r[0]).strip(): r[1] for r in c.execute(text(
                f'SELECT {eq}, {nmq} FROM entities')).fetchall()}

    return {
        'available': True,
        'entities': [{'id': e, 'name': names.get(e)} for e in ents],
        'periods': pers,
        'accounts': accts,
        'bases': bases,
        'period_floor': GL_PERIOD_FLOOR,
    }


def ia_filter_options(engine) -> Dict[str, Any]:
    """Investments, investors, major types and sub types present in `ia_transactions`."""
    if not _has_table(engine, 'ia_transactions'):
        return {'available': False,
                'reason': 'The IA transactions table has not been imported yet. '
                          'Run "MRI_IA_Transactions" from Data Management > MRI Data.',
                'investments': [], 'investors': [],
                'major_types': [], 'sub_types': []}

    def _pairs(c, idcol, namecol):
        i, n = _quote(engine, idcol), _quote(engine, namecol)
        return [{'id': str(r[0]).strip(), 'name': r[1]} for r in c.execute(text(
            f'SELECT DISTINCT {i}, {n} FROM ia_transactions '
            f'WHERE {i} IS NOT NULL ORDER BY {i}')).fetchall()]

    with engine.connect() as c:
        investments = _pairs(c, 'InvestmentID', 'InvestmentName')
        investors = _pairs(c, 'InvestorID', 'InvestorName')
        mq = _quote(engine, 'MajorType')
        majors = [str(r[0]) for r in c.execute(text(
            f'SELECT DISTINCT {mq} FROM ia_transactions WHERE {mq} IS NOT NULL '
            f'ORDER BY {mq}')).fetchall()]
        # Sub types are returned WITH their major type. "Return of Capital" under
        # Distribution is not the same line as one under Contribution, and a flat
        # list would let the two be picked as though they were.
        tq = _quote(engine, 'Typename')
        subs = [{'major_type': r[0], 'sub_type': r[1]} for r in c.execute(text(
            f'SELECT DISTINCT {mq}, {tq} FROM ia_transactions '
            f'WHERE {tq} IS NOT NULL ORDER BY {mq}, {tq}')).fetchall()]

    return {'available': True, 'investments': investments, 'investors': investors,
            'major_types': majors, 'sub_types': subs,
            'date_fields': sorted(IA_DATE_FIELDS)}


# ---------------------------------------------------------------------------
# The queries
# ---------------------------------------------------------------------------

def _freshness(engine, table: str) -> Optional[str]:
    """When MRI was last refreshed, so the screen can say how fresh this is.

    `mri_refresh_status` holds ONE row for the whole refresh job, not a per-table
    timestamp, so this is "when the last full refresh finished" and not "when this
    table changed". Close enough to be worth showing and not the same claim, so the
    screen words it as the refresh, not the table. Returns None rather than a
    guess when no refresh has completed.
    """
    if not _has_table(engine, 'mri_refresh_status'):
        return None
    try:
        with engine.connect() as c:
            return c.execute(text(
                "SELECT finished_at FROM mri_refresh_status "
                "WHERE id = 1 AND state = 'complete'")).scalar()
    except Exception:
        logger.info("mri_refresh_status unreadable", exc_info=True)
        return None


def run_gl_query(engine, entities=None, period_from=None, period_to=None,
                 accounts=None, bases=None, limit: int = MAX_ROWS) -> Dict[str, Any]:
    """GL detail with the CFO's filters. Every filter is optional and ANDed."""
    if not _has_table(engine, 'gl_detail'):
        return {'available': False, 'rows': [], 'columns': GL_COLUMNS,
                'reason': 'The GL detail table has not been imported yet.'}

    entities = _clean_list(entities)
    accounts = _clean_list(accounts)
    bases = _clean_list(bases)
    where, params = [], {}

    if entities:
        where.append(f'{_quote(engine, "ENTITYID")} IN :entities')
        params['entities'] = entities
    if accounts:
        where.append(f'{_quote(engine, "ACCTNUM")} IN :accounts')
        params['accounts'] = accounts
    if bases:
        where.append(f'{_quote(engine, "BASIS")} IN :bases')
        params['bases'] = bases
    # PERIOD is 'YYYYMM' text and compares correctly as text, which is why it is
    # left as text rather than cast -- casting would break a period like '202413'
    # that MRI uses for adjustment entries.
    if period_from:
        where.append(f'{_quote(engine, "PERIOD")} >= :pfrom')
        params['pfrom'] = str(period_from).strip()
    if period_to:
        where.append(f'{_quote(engine, "PERIOD")} <= :pto')
        params['pto'] = str(period_to).strip()

    cols = ', '.join(_quote(engine, c) for c, _ in GL_COLUMNS)
    sql = f'SELECT {cols} FROM gl_detail'
    if where:
        sql += ' WHERE ' + ' AND '.join(where)
    sql += (f' ORDER BY {_quote(engine, "ENTITYID")}, {_quote(engine, "PERIOD")}, '
            f'{_quote(engine, "ACCTNUM")}, {_quote(engine, "ENTRDATE")}')

    return _execute(engine, sql, params, GL_COLUMNS, limit,
                    numeric=('AMT',), table='gl_detail',
                    notes=_gl_notes(period_from))


def _gl_notes(period_from) -> List[str]:
    notes = []
    pf = str(period_from or '').strip()
    if pf and pf < GL_PERIOD_FLOOR:
        notes.append(
            f'Our copy of the GL starts at period {GL_PERIOD_FLOOR}, so nothing '
            f'before that is here. An empty result for {pf} means the period was '
            f'not imported, not that the ledger is empty.')
    return notes


def run_ia_query(engine, investments=None, investors=None, date_from=None,
                 date_to=None, date_field: str = 'TransactionDate',
                 major_types=None, sub_types=None,
                 limit: int = MAX_ROWS) -> Dict[str, Any]:
    """IA detail with the CFO's filters.

    `date_field` is checked against a fixed set before it reaches the SQL.

    NOTE ON THE DATE BOUND. The CFO's query reads `contributiondate < &SPARM03` --
    strictly before. This takes a FROM and a TO and the TO is INCLUSIVE, because
    "to 6/30" excluding 6/30 surprises people. A transaction dated exactly on the
    end date is therefore IN here and OUT of his spreadsheet; set the end date one
    day earlier to reproduce his figure exactly. Stated on screen.
    """
    if not _has_table(engine, 'ia_transactions'):
        return {'available': False, 'rows': [], 'columns': IA_COLUMNS,
                'reason': 'The IA transactions table has not been imported yet.'}

    if date_field not in IA_DATE_FIELDS:
        raise ValueError(f'Unknown date field {date_field!r}. '
                         f'Expected one of {sorted(IA_DATE_FIELDS)}.')

    investments = _clean_list(investments)
    investors = _clean_list(investors)
    major_types = _clean_list(major_types)
    sub_types = _clean_list(sub_types)
    where, params = [], {}

    if investments:
        where.append(f'{_quote(engine, "InvestmentID")} IN :investments')
        params['investments'] = investments
    if investors:
        where.append(f'{_quote(engine, "InvestorID")} IN :investors')
        params['investors'] = investors
    if major_types:
        where.append(f'{_quote(engine, "MajorType")} IN :majors')
        params['majors'] = major_types
    if sub_types:
        where.append(f'{_quote(engine, "Typename")} IN :subs')
        params['subs'] = sub_types

    dcol = _quote(engine, date_field)
    df, dt = _as_date(date_from), _as_date(date_to)
    if df:
        where.append(f'{dcol} >= :dfrom')
        params['dfrom'] = df.isoformat()
    if dt:
        where.append(f'{dcol} <= :dto')
        params['dto'] = dt.isoformat()

    cols = ', '.join(_quote(engine, c) for c, _ in IA_COLUMNS)
    sql = f'SELECT {cols} FROM ia_transactions'
    if where:
        sql += ' WHERE ' + ' AND '.join(where)
    sql += (f' ORDER BY {_quote(engine, "InvestmentID")}, '
            f'{_quote(engine, "InvestorID")}, {dcol}')

    notes = []
    if dt:
        notes.append(
            f'The end date is inclusive here. The CFO\'s spreadsheet query uses '
            f'"before" ({date_field} < date), so a transaction dated exactly '
            f'{dt.isoformat()} is included here and excluded there.')
    return _execute(engine, sql, params, IA_COLUMNS, limit,
                    numeric=('Amount',), table='ia_transactions', notes=notes)


def _execute(engine, sql: str, params: Dict, columns, limit: int,
             numeric=(), table: str = '', notes=None) -> Dict[str, Any]:
    """Run it, cap the rows, and always say whether the cap was hit.

    The total is computed over the WHOLE match, not the capped page, so the figure
    on screen is the answer to the question asked even when the grid is truncated.
    A truncated grid whose total silently described only the visible rows would be
    the worst of both.
    """
    stmt = text(sql)
    for key, val in params.items():
        if isinstance(val, list):
            stmt = stmt.bindparams(bindparam(key, expanding=True))

    with engine.connect() as c:
        df = pd.read_sql(stmt, c, params=params)

    total_rows = len(df)
    totals = {}
    for col in numeric:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            totals[col] = float(vals.sum()) if len(vals) else 0.0

    truncated = total_rows > limit
    if truncated:
        df = df.head(limit)

    out_notes = list(notes or [])
    if truncated:
        out_notes.append(
            f'{total_rows:,} rows matched and the first {limit:,} are shown. The '
            f'total below is for ALL {total_rows:,}, not just the rows displayed. '
            f'Narrow the filters, or export to get every row.')

    return {
        'available': True,
        'columns': [{'key': k, 'label': lbl} for k, lbl in columns],
        'rows': df.where(pd.notna(df), None).to_dict(orient='records'),
        'row_count': int(total_rows),
        'shown': int(len(df)),
        'truncated': bool(truncated),
        'totals': totals,
        'data_as_of': _freshness(engine, table),
        'notes': out_notes,
    }


def to_excel(result: Dict[str, Any], sheet_name: str, criteria: List[str]) -> bytes:
    """The result as a workbook, with the filters that produced it on the sheet.

    An exported grid with no record of what was asked for cannot be checked or
    repeated, and these get mailed around.
    """
    import openpyxl
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]

    ws.cell(1, 1, sheet_name).font = Font(bold=True, size=13)
    r = 2
    for line in criteria:
        ws.cell(r, 1, line)
        r += 1
    r += 1

    cols = result.get('columns') or []
    head = r
    for i, col in enumerate(cols, start=1):
        cell = ws.cell(head, i, col['label'])
        cell.font = Font(bold=True)
    r = head + 1
    for row in result.get('rows') or []:
        for i, col in enumerate(cols, start=1):
            ws.cell(r, i, row.get(col['key']))
        r += 1

    for i, col in enumerate(cols, start=1):
        width = max(len(str(col['label'])), 10)
        for row in (result.get('rows') or [])[:200]:
            width = max(width, len(str(row.get(col['key']) or '')))
        ws.column_dimensions[get_column_letter(i)].width = min(width + 2, 45)
    ws.freeze_panes = ws.cell(head + 1, 1)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
