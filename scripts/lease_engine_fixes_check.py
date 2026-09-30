"""Guardrail: three findings from the v538 acceptance comparison (Sep 30 2026).

  1. A settlement made before the documents were read correctly is FLAGGED when
     the documents now say otherwise (Mattress Firm: settled 2027-09-30, the
     documents 2035-03-21). The analyst's value still stands.
  2. Little Petals' stray rows: a re-read ADDED rent steps rather than replacing
     that document's, and the original lease's "months 65-124" were counted from
     its own estimated commencement rather than the commencement letter's.
  3. Patton Computers' six 30-day options: "6 x term not stated" -> "6 x 30-Day
     Rolling", and one row on the exhibit, not six.

Each is asserted in BOTH directions -- a flag that always fires, or a replace
that deletes everything, would satisfy the one-sided check.

Run:  .venv\\Scripts\\python.exe scripts\\lease_engine_fixes_check.py
"""
import calendar
import os
import sys
from datetime import date

sys.path.insert(0, os.getcwd())

from sqlalchemy import create_engine, text  # noqa: E402

from flask_app.services.lease_review_service import _write_document_rent_steps  # noqa: E402
from flask_app.services.lease_terms import (  # noqa: E402
    reanchor_original_steps, resolve_rent_steps)
from flask_app.services.lease_timeline import build_timeline, summarise_options  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % (detail,) if detail and not cond else ''))


def section(t):
    print('\n--- ' + t)


BASE = {'lease_commencement': '2015-04-01', 'lease_expiration': '2035-03-21'}
STEPS = [{'effective_date': '2015-04-01', 'annual_rent': 30000}]
ASOF = date(2026, 6, 30)

# ---------------------------------------------------------------- 1
section('1. a settled date the documents now contradict is flagged')
tl = build_timeline(BASE, STEPS, 1000, ASOF, settled_expiration='2027-09-30')
codes = [f['code'] for f in tl['flags']]
chk('settled expiry differing from the documents is flagged',
    'settled_expiration_differs' in codes, codes)
msg = next((f['message'] for f in tl['flags'] if f['code'] == 'settled_expiration_differs'), '')
chk('the flag names both dates', '2027-09-30' in msg and '2035-03-21' in msg, msg)
chk("the analyst's settled expiry still stands", tl['rows'][0]['end'] == '2027-09-30',
    tl['rows'][0]['end'])
tl = build_timeline(BASE, STEPS, 1000, ASOF, settled_expiration='2035-03-21')
chk('a settlement that AGREES with the documents raises nothing',
    not any(f['code'].startswith('settled_') for f in tl['flags']))
tl = build_timeline(BASE, STEPS, 1000, ASOF, settled_start='2015-06-01')
chk('settled start differing is flagged too',
    'settled_start_differs' in [f['code'] for f in tl['flags']])
tl = build_timeline({'lease_commencement': '2015-04-01'}, STEPS, 1000, ASOF,
                    settled_expiration='2027-09-30')
chk('no document date -> nothing to contradict, no flag',
    'settled_expiration_differs' not in [f['code'] for f in tl['flags']])
tl = build_timeline(BASE, STEPS, 1000, ASOF)
chk('no settlement -> no flag', not any(f['code'].startswith('settled_') for f in tl['flags']))

# ---------------------------------------------------------------- 2a
section("2a. a re-read replaces the document's rent steps; it does not add")
eng = create_engine('sqlite://')
with eng.begin() as c:
    c.execute(text("""CREATE TABLE lease_rent_steps (id INTEGER PRIMARY KEY, tenant_id INTEGER,
        effective_date TEXT, period_start_month INTEGER, period_end_month INTEGER,
        monthly_rent REAL, annual_rent REAL, rent_per_sf REAL, source_doc TEXT,
        source_doc_id INTEGER, term_start TEXT, is_additional INTEGER)"""))
    c.execute(text("INSERT INTO lease_rent_steps (tenant_id, annual_rent, source_doc) "
                   "VALUES (1, 99999, NULL)"))
read1 = {'rent_commencement': '2025-10-15', 'rent_steps': [
    {'period': 'Months 1-4', 'annual_rent': 0},
    {'period': 'Months 5-64', 'annual_rent': 36000},
    {'effective_date': None, 'annual_rent': 12000},
    {'effective_date': None, 'annual_rent': 12000}]}
read2 = {'rent_commencement': '2025-10-15', 'rent_steps': [
    {'effective_date': '2026-03-01', 'annual_rent': 36000},
    {'period': 'Months 65-124', 'annual_rent': 39600}]}


def rows(c, doc=None):
    q = "SELECT effective_date, period_start_month, annual_rent, source_doc FROM lease_rent_steps"
    return c.execute(text(q + (" WHERE source_doc = :d" if doc else "")),
                     {'d': doc} if doc else {}).fetchall()


with eng.begin() as c:
    _write_document_rent_steps(c, text, 1, 11, 'Lease.pdf', read1)
    _write_document_rent_steps(c, text, 1, 12, 'Commencement Letter.pdf',
                               {'rent_steps': [{'effective_date': '2026-03-01', 'annual_rent': 36000}]})
with eng.begin() as c:
    n1 = len(rows(c, 'Lease.pdf'))
    chk('an identical undated step stated twice in ONE reading is kept once', n1 == 3, n1)
    _write_document_rent_steps(c, text, 1, 11, 'Lease.pdf', read1)
    chk('reading the same document again does not duplicate its steps',
        len(rows(c, 'Lease.pdf')) == 3, len(rows(c, 'Lease.pdf')))
    _write_document_rent_steps(c, text, 1, 11, 'Lease.pdf', read2)
    got = sorted(((r[0], r[1], r[2]) for r in rows(c, 'Lease.pdf')), key=str)
    chk("a re-read that states the schedule differently REPLACES the document's steps",
        got == sorted([('2026-03-01', None, 36000.0), (None, 65, 39600.0)], key=str), got)
    chk("another document's step on the SAME date is kept",
        len(rows(c, 'Commencement Letter.pdf')) == 1)
    chk('a row with no source document is never touched',
        any(r[3] is None and r[2] == 99999 for r in rows(c)))

# ---------------------------------------------------------------- 2b
section("2b. an original lease's months count from the commencement letter")
types = {11: 'Original Lease', 12: 'Commencement Letter'}
lp = [{'period_start_month': 1, 'period_end_month': 4, 'annual_rent': 0,
       'source_doc_id': 11, 'term_start': '2025-10-15'},
      {'period_start_month': 5, 'period_end_month': 64, 'annual_rent': 36000,
       'source_doc_id': 11, 'term_start': '2025-10-15'},
      {'period_start_month': 65, 'period_end_month': 124, 'annual_rent': 39600,
       'source_doc_id': 11, 'term_start': '2025-10-15'}]
out = reanchor_original_steps(lp, types, '2025-11-01', '2026-03-01')
res, _ = resolve_rent_steps(out, '2025-10-15', 1000)
got = {s['period_start_month']: s['effective_date'] for s in res}
chk('month 5 (first paying month) lands on the actual rent commencement 2026-03-01',
    got.get(5) == '2026-03-01', got)
chk("months 65-124 begin 2031-03-01, new business's option date", got.get(65) == '2031-03-01', got)
chk('month 1 is the actual lease commencement', got.get(1) == '2025-11-01', got)
out = reanchor_original_steps(
    [dict(s) for s in lp if s['period_start_month'] != 1] and
    [dict(s, period_start_month=1, period_end_month=60) if s['period_start_month'] == 5 else s
     for s in lp if s['period_start_month'] != 1], types, '2025-11-01', '2026-03-01')
first = next(s for s in out if s['annual_rent'] == 36000)
chk('when the schedule does NOT confirm lease commencement, rent commencement is month 1',
    first['term_start'] == '2026-03-01', first['term_start'])
amend = [dict(lp[1], source_doc_id=13)]
out = reanchor_original_steps(amend, {13: 'Amendment'}, '2025-11-01', '2026-03-01')
chk("an amendment's steps keep their own term", out[0]['term_start'] == '2025-10-15')
chk('no actual dates -> steps unchanged', reanchor_original_steps(lp, types, None, None) == lp)

# ---------------------------------------------------------------- 3
section("3. Patton's 30-day rolling options")
opts = []
for i in range(6):
    m = 2 + i
    opts.append({'option_number': i + 1, 'option_start': '2026-%02d-01' % m,
                 'option_end': '2026-%02d-%02d' % (m, calendar.monthrange(2026, m)[1]),
                 'term_years': None, 'rent_terms': 'Thirty (30) day option'})
chk('six 30-day options summarise as "6 x 30-Day Rolling"',
    summarise_options(opts) == '6 x 30-Day Rolling', summarise_options(opts))
undated = [{'term_years': None, 'rent_terms': 'Thirty (30) day renewal'} for _ in range(6)]
chk('the day count is read from the wording when there are no dates',
    summarise_options(undated) == '6 x 30-Day Rolling', summarise_options(undated))
chk('five-year options are unchanged ("2 x 5 Years", not rolling)',
    summarise_options([{'term_years': 5}, {'term_years': 5}]) == '2 x 5 Years')
chk('a single short option is not "rolling"',
    summarise_options(opts[:1]) == '1 x 30-Day', summarise_options(opts[:1]))
chk('an option with nothing stated still says so',
    summarise_options([{'term_years': None}]) == '1 x term not stated')
terms = {'lease_commencement': '2020-01-01', 'lease_expiration': '2026-01-31',
         '_remaining_options': opts, '_options_summary': summarise_options(opts)}
tl = build_timeline(terms, [{'effective_date': '2020-01-01', 'annual_rent': 12000}], 1000, ASOF)
orows = [r for r in tl['rows'] if r['kind'] == 'option']
chk('the six print as ONE option row', len(orows) == 1, [r['label'] for r in orows])
chk('labelled "Option 1 (30-day rolling, 6x)", spanning all six',
    orows and orows[0]['label'] == 'Option 1 (30-day rolling, 6x)'
    and orows[0]['start'] == '2026-02-01' and orows[0]['end'] == '2026-07-31',
    orows and (orows[0]['label'], orows[0]['start'], orows[0]['end']))
chk('the underlying options are all still there', len(tl['options']) == 6)
five = {'lease_commencement': '2015-01-01', 'lease_expiration': '2025-12-31',
        '_remaining_options': [{'option_start': '2026-01-01', 'option_end': '2030-12-31',
                                'term_years': 5},
                               {'option_start': '2031-01-01', 'option_end': '2035-12-31',
                                'term_years': 5}]}
tl = build_timeline(five, [{'effective_date': '2015-01-01', 'annual_rent': 12000}], 1000, ASOF)
chk('five-year options are NOT collapsed',
    [r['label'] for r in tl['rows'] if r['kind'] == 'option'] == ['Option 1', 'Option 2'],
    [r['label'] for r in tl['rows'] if r['kind'] == 'option'])

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
sys.exit(1 if BAD else 0)
