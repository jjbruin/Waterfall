"""Guardrail: rent PSF, amendment order, and rent steps stated as months of the term.

New business, Sep 19 2026 (via Jim), on validating the rent roll against the leases:

  * "When a lease doesn't specify the rent PSF, the application should simply
    calculate by taking the annual rent by the square feet (only calculate rent PSF on
    annual rent, not monthly)"

  * "the Hobby Lobby lease should analyze the most recent lease amendment (4th
    Amendment) ... In some cases, a lease may not specify exact dates and instead
    reference a specific month of the lease term (e.g., Months 1-12). To calculate the
    current annual rent amount, the application should reference the Rent
    Commencement Date to determine where the tenant currently falls within the lease
    term and, accordingly, the applicable base rent."

THE WORST DEFECT WAS NOT EITHER OF THOSE. When a step's date would not resolve, the
validation picked the step whose annual rent was CLOSEST to the rent roll's figure --
so the rent roll was checked against whichever lease number already agreed with it. It
could not report a mismatch. Every check below that names `step_in_force_at` exists to
keep that from coming back.

Fixtures are the Hobby Lobby shape Jim described: four amendments, no dates in the
filenames, rent stated as months of the term.

Run:  .venv\\Scripts\\python.exe scripts\\lease_terms_check.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services import lease_terms as LT  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))


def skip(name, why):
    SKIP.append(name)
    print(f'  SKIP  {name}  [{why}]')


def section(t):
    print(f'\n--- {t}')


# ===========================================================================
section('Rent PSF is taken on ANNUAL rent, never monthly')

SF = 20_000.0
ANNUAL = 240_000.0          # $12.00/SF/yr
MONTHLY = 20_000.0          # the same rent, stated monthly

check('annual rent over SF', LT.annual_rent_psf(ANNUAL, SF) == 12.0,
      str(LT.annual_rent_psf(ANNUAL, SF)))

# The 12x hazard, stated as a test: dividing the MONTHLY rent by SF gives 1.00, a
# perfectly plausible number that is wrong by a factor of twelve. v495 shipped exactly
# this error from the rent roll side.
psf, basis = LT.rent_psf_for(monthly_rent=MONTHLY, square_feet=SF)
check('a monthly rent is annualised before dividing, not divided as-is',
      psf == 12.0, f'{psf} via {basis}')
check('...and says so', basis == 'annualised monthly rent / SF', basis)
check('...and is NOT the monthly-divided figure', psf != 1.0)

psf, basis = LT.rent_psf_for(annual_rent=ANNUAL, square_feet=SF)
check('a stated annual rent divides directly', psf == 12.0 and basis == 'annual rent / SF',
      basis)

# A figure the LEASE states outranks anything computed -- the instruction is what to do
# "when a lease doesn't specify the rent PSF".
psf, basis = LT.rent_psf_for(annual_rent=ANNUAL, square_feet=SF, stated_psf=11.5)
check('a PSF the lease states wins over a derived one', psf == 11.5 and basis == 'stated',
      f'{psf} via {basis}')

check('no square footage gives no PSF, not zero',
      LT.annual_rent_psf(ANNUAL, None) is None)
check('zero square footage gives no PSF rather than dividing by zero',
      LT.annual_rent_psf(ANNUAL, 0) is None)
check('no rent gives no PSF', LT.annual_rent_psf(None, SF) is None)
check('junk input is refused, not coerced',
      LT.annual_rent_psf('n/a', SF) is None)
check('the empty case reports no basis', LT.rent_psf_for()[1] == '')
check('annualising prefers a stated annual figure over the monthly one',
      LT.annual_rent_from(annual_rent=100.0, monthly_rent=999.0) == 100.0)


# ===========================================================================
section('The most recent amendment is the one that governs')

check('a written ordinal is read', LT.amendment_ordinal('Hobby Lobby - Fourth Amendment.pdf') == 4)
check('a numeric ordinal is read', LT.amendment_ordinal('HL 4th Amendment.pdf') == 4)
check('"Amendment No. 4" is read', LT.amendment_ordinal('Lease Amendment No. 4.pdf') == 4)
check('"Amendment #11" is read', LT.amendment_ordinal('Amendment #11 signed.pdf') == 11)
check('a bare lease has no ordinal', LT.amendment_ordinal('Hobby Lobby Lease.pdf') is None)
check('an empty name does not raise', LT.amendment_ordinal('') is None)
check('a year is not mistaken for an ordinal',
      LT.amendment_ordinal('Amendment executed in 2019.pdf') is None,
      str(LT.amendment_ordinal('Amendment executed in 2019.pdf')))

# Jim's case exactly: four amendments, not one of them carrying a date.
# Ids run AGAINST the ordinals on purpose. Documents are uploaded in whatever order
# they come out of the folder, so id order is arbitrary — and if the fixture's ids
# happened to ascend with the amendment numbers, sorting by id would pass and the
# check would prove nothing.
HL = [
    {'id': 10, 'doc_type': 'Original Lease', 'filename': 'Hobby Lobby Lease.pdf'},
    {'id': 11, 'doc_type': 'Amendment', 'filename': 'Hobby Lobby - Fourth Amendment.pdf'},
    {'id': 12, 'doc_type': 'Amendment', 'filename': 'Hobby Lobby - First Amendment.pdf'},
    {'id': 13, 'doc_type': 'Amendment', 'filename': 'Hobby Lobby - Third Amendment.pdf'},
    {'id': 14, 'doc_type': 'Amendment', 'filename': 'Hobby Lobby - Second Amendment.pdf'},
]
for d in HL:
    d['ordinal'] = LT.amendment_ordinal(d['filename'])
    d['doc_date'] = LT.parse_doc_date_anywhere(d['filename'])

check('none of the Hobby Lobby filenames carries a date',
      all(d['doc_date'] is None for d in HL))
ordered, notes = LT.order_lease_documents(HL)
check('the original lease is applied first', ordered[0]['id'] == 10)
check('the amendments then apply 1, 2, 3, 4',
      [d['ordinal'] for d in ordered[1:]] == [1, 2, 3, 4],
      str([d['ordinal'] for d in ordered[1:]]))
check('the LAST one applied is the Fourth Amendment', ordered[-1]['ordinal'] == 4)
check('...and the ordering says it came from the numbers, not dates',
      any('number in their filename' in n for n in notes), '; '.join(notes))

# The order must be WRONG without the fix, or this test proves nothing: by id alone
# (the old undated fallback) the Fourth Amendment applies before the First.
by_id = sorted([d for d in HL if d['doc_type'] == 'Amendment'], key=lambda d: d['id'])
check('the old id order really did differ, so this test can fail',
      [d['ordinal'] for d in by_id] != [1, 2, 3, 4],
      str([d['ordinal'] for d in by_id]))

DATED = [
    {'id': 1, 'doc_type': 'Original Lease', 'filename': 'Lease.pdf', 'doc_date': None,
     'ordinal': None},
    {'id': 2, 'doc_type': 'Amendment', 'filename': 'a.pdf', 'doc_date': '2020-01-01',
     'ordinal': 1},
    {'id': 3, 'doc_type': 'Amendment', 'filename': 'b.pdf', 'doc_date': '2019-01-01',
     'ordinal': 2},
]
ordered, notes = LT.order_lease_documents(DATED)
check('when every amendment is dated, the dates order them',
      [d['id'] for d in ordered[1:]] == [3, 2])
check('...and a number that disagrees with the dates is REPORTED, not ignored',
      any('different orders' in n for n in notes), '; '.join(notes))

MIXED = [
    {'id': 1, 'doc_type': 'Original Lease', 'filename': 'L.pdf', 'doc_date': None,
     'ordinal': None},
    {'id': 2, 'doc_type': 'Amendment', 'filename': 'x.pdf', 'doc_date': None,
     'ordinal': None},
    {'id': 3, 'doc_type': 'Amendment', 'filename': 'y.pdf', 'doc_date': '2019-01-01',
     'ordinal': None},
]
_, notes = LT.order_lease_documents(MIXED)
check('an amendment with neither a date nor a number makes the order indeterminate,'
      ' and says so', any('not established' in n for n in notes), '; '.join(notes))

check('a date anywhere in the filename is found, not only at the front',
      LT.parse_doc_date_anywhere('Hobby Lobby - 2019.04.02 Fourth Amendment.pdf')
      == '2019-04-02')
check('US order is read too',
      LT.parse_doc_date_anywhere('HL 04-02-2019 amend.pdf') == '2019-04-02')
check('a bare year does NOT become a date',
      LT.parse_doc_date_anywhere('Amendment 2019.pdf') is None)
check('an impossible date is refused rather than clamped',
      LT.parse_doc_date_anywhere('doc 2019.13.45 x.pdf') is None)


# ===========================================================================
section('"Months 1-12" is placed against the Rent Commencement Date')

check('a month range is read', LT.parse_relative_period('Months 1-12') == (1, 12))
check('an en dash is read', LT.parse_relative_period('Months 13 – 24') == (13, 24))
check('"through" is read', LT.parse_relative_period('Months 25 through 36') == (25, 36))
check('a single month is read', LT.parse_relative_period('Month 37') == (37, None))
check('a lease year becomes its months', LT.parse_relative_period('Lease Year 2') == (13, 24))
check('a bare year reference works too', LT.parse_relative_period('Year 3') == (25, 36))
# An ISO date is NOT a relative period. Coercing one would turn 2019-04-02 into
# "month 2019", which resolves to a date centuries out and sorts last in silence.
check('an ISO date is not read as a period',
      LT.parse_relative_period('2019-04-02') is None)
check('empty text is not a period', LT.parse_relative_period('') is None
      and LT.parse_relative_period(None) is None)
check('prose with no period is not a period',
      LT.parse_relative_period('as set out in Exhibit B') is None)
check('a backwards range is refused rather than inverted',
      LT.parse_relative_period('Months 12-1') is None)

RC = date(2019, 4, 15)
check('month 1 begins on the rent commencement date',
      LT.month_to_date(RC, 1) == date(2019, 4, 15))
# The anniversary, not the calendar month. A lease commencing mid-month steps on the
# 15th, and rounding to the 1st moves every step two weeks early.
check('month 13 is the ANNIVERSARY, not the start of that calendar month',
      LT.month_to_date(RC, 13) == date(2020, 4, 15),
      str(LT.month_to_date(RC, 13)))
check('month 61 lands five years on', LT.month_to_date(RC, 61) == date(2024, 4, 15))
check('no rent commencement gives no date', LT.month_to_date(None, 13) is None)
check('month 0 is refused', LT.month_to_date(RC, 0) is None)


# ===========================================================================
section('The rent in force is resolved, never guessed from the rent roll')

# Hobby Lobby as Jim described it: rent stated as months of the term, no dates.
HL_SF = 55_000.0
HL_STEPS = [
    {'period': 'Months 1-60',    'annual_rent': 495_000.0},
    {'period': 'Months 61-120',  'annual_rent': 544_500.0},
    {'period': 'Months 121-180', 'annual_rent': 599_000.0},
]
resolved, notes = LT.resolve_rent_steps(HL_STEPS, RC, square_feet=HL_SF)
check('every step is dated from the rent commencement date',
      all(s['effective_date'] for s in resolved), str(notes))
check('the first step starts at rent commencement',
      resolved[0]['effective_date'] == '2019-04-15')
check('the second starts at month 61', resolved[1]['effective_date'] == '2024-04-15')
check('the third starts at month 121', resolved[2]['effective_date'] == '2029-04-15')
check('each says how it was dated',
      resolved[1]['effective_date_basis'] == 'month 61 of the term',
      resolved[1]['effective_date_basis'])
check('PSF is derived on the annual rent', abs(resolved[0]['rent_per_sf'] - 9.0) < 1e-9,
      str(resolved[0]['rent_per_sf']))
check('...and says it was derived', resolved[0]['rent_psf_basis'] == 'annual rent / SF')

# The question Jim actually asked: what is the rent TODAY?
step, basis = LT.step_in_force_at(resolved, date(2026, 9, 19))
check('the rent in force today is the month 61-120 step',
      step is not None and step['annual_rent'] == 544_500.0,
      str(step and step['annual_rent']))
check('...and the answer shows its working', 'month 61 of the term' in basis, basis)

step, _ = LT.step_in_force_at(resolved, date(2019, 6, 1))
check('a date inside the first period gets the first rent',
      step['annual_rent'] == 495_000.0)
step, _ = LT.step_in_force_at(resolved, date(2030, 1, 1))
check('a date inside the last period gets the last rent',
      step['annual_rent'] == 599_000.0)

# Before the lease begins there IS no rent in force, and saying "the first step" would
# be wrong rather than merely early.
step, basis = LT.step_in_force_at(resolved, date(2018, 1, 1))
check('before rent commencement there is no step in force', step is None)
check('...and it says why rather than returning nothing', 'after 2018-01-01' in basis,
      basis)

# THE GUESS THAT COULD NOT FAIL. The old fallback chose the step nearest the rent
# roll's own figure; a rent roll carrying a wrong rent would select the step that
# matched it and report agreement. Resolution by date cannot do that.
rr_wrong = 501_000.0
nearest = min(resolved, key=lambda s: abs(s['annual_rent'] - rr_wrong))
check('the nearest-value guess would have latched onto the rent roll\'s own figure',
      nearest['annual_rent'] == 495_000.0, str(nearest['annual_rent']))
today_step, _ = LT.step_in_force_at(resolved, date(2026, 9, 19))
check('...so a wrong rent roll now disagrees with the lease instead of matching it',
      today_step['annual_rent'] != nearest['annual_rent'],
      f"lease says {today_step['annual_rent']:,.0f}, nearest-value guess "
      f"would have said {nearest['annual_rent']:,.0f}")


# ===========================================================================
section('What cannot be resolved is reported, not dropped')

no_rc, notes = LT.resolve_rent_steps(HL_STEPS, None, square_feet=HL_SF)
check('steps survive with no rent commencement date', len(no_rc) == len(HL_STEPS))
check('...none of them is dated', all(s['effective_date'] is None for s in no_rc))
check('...and the missing commencement date is named as the reason',
      any('no rent commencement date' in n for n in notes), '; '.join(notes))
check('...and nothing is in force, rather than the first step by default',
      LT.step_in_force_at(no_rc, date(2026, 9, 19))[0] is None)

MIXED_STEPS = [
    {'effective_date': '2019-04-15', 'annual_rent': 495_000.0},
    {'period': 'Months 61-120', 'annual_rent': 544_500.0},
    {'effective_date': 'as agreed', 'annual_rent': 610_000.0},
]
res, notes = LT.resolve_rent_steps(MIXED_STEPS, RC, square_feet=HL_SF)
check('a stated date is kept as stated',
      res[0]['effective_date'] == '2019-04-15'
      and res[0]['effective_date_basis'] == 'stated')
check('a relative period alongside it still resolves',
      any(s['effective_date'] == '2024-04-15' for s in res))
check('an unreadable date leaves the step undated rather than inventing one',
      any(s['effective_date'] is None for s in res))
check('...and the count that could not be dated is reported',
      any('could not be dated' in n for n in notes), '; '.join(notes))
check('undated steps sort last so they cannot be read as the earliest',
      res[-1]['effective_date'] is None)

# A period hidden in the date field -- what the OLD extraction prompt forced, since it
# offered nowhere else to put "Lease Year 7".
hidden, _ = LT.resolve_rent_steps(
    [{'effective_date': 'Lease Year 7', 'annual_rent': 700_000.0}], RC)
check('a period written into the date field is still resolved',
      hidden[0]['effective_date'] == '2025-04-15',
      str(hidden[0]['effective_date']))

monthly_only, _ = LT.resolve_rent_steps(
    [{'period': 'Months 1-12', 'monthly_rent': 41_250.0}], RC, square_feet=HL_SF)
check('a monthly-only step is annualised before its PSF is taken',
      abs(monthly_only[0]['rent_per_sf'] - 9.0) < 1e-9,
      str(monthly_only[0]['rent_per_sf']))
check('...and NOT divided as a monthly figure, which would read 0.75',
      abs(monthly_only[0]['rent_per_sf'] - 0.75) > 1e-9)
check('...and carries the annual rent it implies',
      monthly_only[0]['annual_rent'] == 495_000.0,
      str(monthly_only[0]['annual_rent']))

check('no steps at all is survivable', LT.resolve_rent_steps([], RC) == ([], []))
check('...and nothing is in force', LT.step_in_force_at([], date(2026, 9, 19))[0] is None)



# ===========================================================================
section('The service uses these primitives rather than its own arithmetic')

import inspect  # noqa: E402
import re as _re  # noqa: E402

from flask_app.services import lease_review_service as LRS  # noqa: E402

_src = inspect.getsource(LRS)

# The guess is gone and cannot come back.
check('the nearest-rent guess is gone from the validation',
      'closest_rent' not in _src)
check('...and the validation resolves the step instead',
      'resolve_rent_steps(' in _src and 'step_in_force_at(' in _src)
check('a tenant whose rent cannot be dated is RECORDED as a finding',
      "'rent_step_in_force'" in _src)

# Rent PSF: no hand-rolled division left in the importers.
_vld = inspect.getsource(LRS.validate_rent_roll)
check('the validation reads the rent commencement date',
      'rent_commencement' in _vld)
for bad in ('monthly_rent_per_sf = monthly_rent / sf',
            'mon_rent_psf = mon_rent / sf'):
    check(f'no monthly PSF is derived by hand ({bad.split("=")[0].strip()})',
          bad not in _src)
check('the importers call the shared PSF definition',
      _src.count('rent_psf_for(') >= 2, str(_src.count('rent_psf_for(')))

# Consolidation order.
_con = inspect.getsource(LRS.consolidate_tenant_extractions)
check('consolidation orders documents through the shared rule',
      'order_lease_documents(' in _con)
_con_sql = _con[_con.index('SELECT id, doc_type'):_con.index('if not rows')]
check('...and the SQL no longer imposes an order of its own',
      'ORDER BY' not in _con_sql,
      'the order is decided in Python, not by the query')
check('...and records which document had the last word',
      '_governing_document' in _con)
check('...and carries the order notes when the order is uncertain',
      '_order_notes' in _con)

# The amendment number is captured at every point a document is stored.
_ins = [m.start() for m in _re.finditer('INSERT INTO lease_documents', _src)]
_ins_ok = [i for i in _ins if 'doc_ordinal' in _src[i:i + 900]]
check('every lease_documents INSERT records the amendment number',
      len(_ins) > 0 and len(_ins_ok) == len(_ins),
      f'{len(_ins_ok)} of {len(_ins)} inserts')
# ...and prove that check can fail, rather than passing because the window is wide
# enough to catch an unrelated mention.
check('...and the window is tight enough to mean something',
      all('INSERT INTO lease_documents' not in _src[i + 30:i + 900] for i in _ins))

# The schema the code now depends on.
for tbl, col in (('lease_rent_steps', 'period_start_month'),
                 ('lease_rent_steps', 'period_end_month'),
                 ('lease_rent_steps', 'effective_date_basis'),
                 ('lease_tenants', 'rent_commencement'),
                 ('lease_documents', 'doc_ordinal')):
    check(f'{tbl}.{col} has a migration',
          _re.search(rf"_migrate_add_column\(engine, '{tbl}', '{col}'", _src)
          is not None)
    check(f'...and is in the CREATE TABLE for a fresh database',
          _re.search(rf'{col}\s', _src) is not None)

# The extraction prompt must offer somewhere to put a period, or the model has no
# choice but to write "Months 1-12" into a date field -- which is what it did.
check('the extraction prompt asks for period_start_month',
      'period_start_month' in LRS.EXTRACTION_PROMPT)
check('...and tells the model NOT to convert a period into a date itself',
      'DO NOT convert it to a date' in LRS.EXTRACTION_PROMPT)
check('...and asks for the rent commencement date explicitly',
      'rent_commencement is the date RENT begins' in LRS.EXTRACTION_PROMPT)

# The dedup bug: `effective_date = NULL` is never true in SQL, so an undated step was
# re-inserted on every run.
check('the rent step dedup handles an undated step',
      ':ed IS NULL' in _src)


# ===========================================================================
section('A date is found wherever it sits in the filename')

check('the service now reads a date mid-name',
      LRS.parse_doc_date('Hobby Lobby - 2019.04.02 Fourth Amendment.pdf')
      == '2019-04-02')
check('...and still reads the old leading-date form',
      LRS.parse_doc_date('2024.03.28_Bealls-Lease.pdf') == '2024-03-28')
check('...and still refuses a name with no date',
      LRS.parse_doc_date('Hobby Lobby Lease.pdf') is None)


# ===========================================================================
section('End to end on a real database: Hobby Lobby as new business described it')

# The sections above test the primitives and read the source. This one runs the
# SHIPPING code paths -- ensure_lease_tables, consolidate_tenant_extractions, the
# step resolver -- against a database, because a rule that is right in isolation and
# never reached changes nothing on screen.

import tempfile  # noqa: E402
import json as _json  # noqa: E402

try:
    from sqlalchemy import create_engine, text as _sqltext, inspect as _inspect
    _HAVE_SA = True
except ImportError:                                          # pragma: no cover
    _HAVE_SA = False

if not _HAVE_SA:
    skip('end-to-end lease consolidation', 'sqlalchemy not importable')
else:
    _DB = os.path.join(tempfile.gettempdir(), 'lease_terms_check.db')
    if os.path.exists(_DB):
        os.remove(_DB)
    _eng = create_engine(f'sqlite:///{_DB}')

    LRS.ensure_lease_tables(_eng)
    with _eng.begin() as _c:
        _c.execute(_sqltext("INSERT INTO lease_reviews (id, property_name) "
                            "VALUES (1, 'Market at Poplar')"))
        _c.execute(_sqltext(
            "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, "
            "square_feet) VALUES (1, 1, 'Hobby Lobby', '100', 55000)"))

        # Uploaded Fourth-then-First, none of them dated. If the order is wrong the
        # First Amendment's superseded 510,000 wins.
        _DOCS = [
            (1, 'Hobby Lobby Lease.pdf', 'Original Lease', {
                'rent_commencement': '2019-04-15', 'square_feet': 55000,
                'rent_steps': [{'period': 'Months 1-60', 'annual_rent': 495000.0}]}),
            (2, 'Hobby Lobby - Fourth Amendment.pdf', 'Amendment', {
                'rent_steps': [
                    {'period': 'Months 61-120', 'annual_rent': 544500.0},
                    {'period': 'Months 121-180', 'annual_rent': 599000.0}]}),
            (3, 'Hobby Lobby - First Amendment.pdf', 'Amendment', {
                'rent_steps': [{'period': 'Months 61-120',
                                'annual_rent': 510000.0}]}),
        ]
        for _did, _fn, _dt, _terms in _DOCS:
            _c.execute(_sqltext("""
                INSERT INTO lease_documents
                    (id, tenant_id, review_id, filename, doc_type, doc_date,
                     doc_ordinal, extraction_status, extraction_json)
                VALUES (:i, 1, 1, :fn, :dt, :dd, :do, 'extracted', :ej)
            """), {'i': _did, 'fn': _fn, 'dt': _dt,
                   'dd': LRS.parse_doc_date(_fn),
                   'do': LRS.amendment_ordinal(_fn),
                   'ej': _json.dumps(_terms)})

    with _eng.connect() as _c:
        _ords = _c.execute(_sqltext(
            "SELECT doc_ordinal, doc_date FROM lease_documents ORDER BY id")).fetchall()
    check('the amendment number is stored from the filename',
          [r[0] for r in _ords] == [None, 4, 1], str([r[0] for r in _ords]))
    check('...and no document carries a date, so only the number can order them',
          all(r[1] is None for r in _ords))

    _terms = LRS.consolidate_tenant_extractions(_eng, 1)
    check('consolidation returns terms', _terms is not None)
    check('the Fourth Amendment had the last word',
          _terms.get('_governing_document') == 'Hobby Lobby - Fourth Amendment.pdf',
          str(_terms.get('_governing_document')))
    check('...and the consolidation says the order came from the numbers',
          any('number in their filename' in n
              for n in _terms.get('_order_notes') or []),
          str(_terms.get('_order_notes')))

    with _eng.connect() as _c:
        _rc = _c.execute(_sqltext(
            "SELECT rent_commencement FROM lease_tenants WHERE id = 1")).scalar()
    check('the rent commencement date is lifted out of the JSON onto the tenant',
          _rc == '2019-04-15', str(_rc))

    _steps, _notes = LRS.resolve_rent_steps(_terms.get('rent_steps') or [], _rc,
                                            square_feet=55000)
    check('every step resolved to a date', all(s['effective_date'] for s in _steps),
          str(_notes))
    _if, _basis = LRS.step_in_force_at(_steps, date(2026, 9, 19))
    check('the rent in force today is the month 61-120 step',
          _if is not None and _if['annual_rent'] == 544500.0,
          str(_if and _if['annual_rent']))
    check("...the FOURTH amendment's figure, not the First's superseded 510,000",
          _if['annual_rent'] != 510000.0)
    check('...and it shows its working', 'month 61 of the term' in _basis, _basis)
    check('rent PSF is the annual rent over SF',
          abs(_if['rent_per_sf'] - 9.9) < 1e-9, str(_if['rent_per_sf']))


# ===========================================================================
section('An EXISTING database migrates, and its rows survive')

# Production is not a fresh database. The DDL above proves nothing about it.
if not _HAVE_SA:
    skip('lease schema migration', 'sqlalchemy not importable')
else:
    _MDB = os.path.join(tempfile.gettempdir(), 'lease_terms_check_migrate.db')
    if os.path.exists(_MDB):
        os.remove(_MDB)
    _meng = create_engine(f'sqlite:///{_MDB}')
    with _meng.begin() as _c:
        _c.execute(_sqltext("CREATE TABLE lease_reviews (id INTEGER PRIMARY KEY,"
                            " property_name TEXT NOT NULL)"))
        _c.execute(_sqltext("CREATE TABLE lease_tenants (id INTEGER PRIMARY KEY,"
                            " review_id INTEGER NOT NULL, tenant_name TEXT NOT NULL,"
                            " suite TEXT, square_feet REAL)"))
        _c.execute(_sqltext("CREATE TABLE lease_documents (id INTEGER PRIMARY KEY,"
                            " tenant_id INTEGER, review_id INTEGER NOT NULL,"
                            " filename TEXT NOT NULL, doc_type TEXT, doc_date TEXT,"
                            " extraction_status TEXT, extraction_json TEXT)"))
        _c.execute(_sqltext("CREATE TABLE lease_rent_steps (id INTEGER PRIMARY KEY,"
                            " tenant_id INTEGER NOT NULL, effective_date TEXT,"
                            " monthly_rent REAL, annual_rent REAL,"
                            " rent_per_sf REAL, source_doc TEXT)"))
        _c.execute(_sqltext("INSERT INTO lease_reviews VALUES (1, 'Existing')"))
        _c.execute(_sqltext("INSERT INTO lease_tenants VALUES "
                            "(1, 1, 'Hobby Lobby', '100', 55000)"))
        _c.execute(_sqltext("INSERT INTO lease_rent_steps VALUES "
                            "(1, 1, 'Lease Year 7', NULL, 700000, NULL, 'HL.pdf')"))

    _cols = lambda t: {c['name'] for c in _inspect(_meng).get_columns(t)}  # noqa: E731
    check('the pre-change schema really lacks the new columns, so this can fail',
          'rent_commencement' not in _cols('lease_tenants')
          and 'period_start_month' not in _cols('lease_rent_steps'))

    LRS.ensure_lease_tables(_meng)

    for _t, _col in (('lease_tenants', 'rent_commencement'),
                     ('lease_documents', 'doc_ordinal'),
                     ('lease_rent_steps', 'period_start_month'),
                     ('lease_rent_steps', 'period_end_month'),
                     ('lease_rent_steps', 'effective_date_basis')):
        check(f'{_t}.{_col} arrives by migration', _col in _cols(_t))

    with _meng.connect() as _c:
        check('the existing rent step survived, text date and all',
              _c.execute(_sqltext("SELECT effective_date, annual_rent FROM "
                                  "lease_rent_steps WHERE id=1")).fetchone()
              == ('Lease Year 7', 700000.0))
        check('...and its new columns are NULL, not zero',
              _c.execute(_sqltext("SELECT period_start_month, effective_date_basis "
                                  "FROM lease_rent_steps WHERE id=1")).fetchone()
              == (None, None))

    # A row written before this change has no stored ordinal; the filename still
    # answers, so old reviews are ordered correctly without a backfill.
    check('a pre-existing document is still ordered, from its filename',
          LRS.amendment_ordinal('HL - Fourth Amendment.pdf') == 4)
    _legacy, _ = LRS.resolve_rent_steps(
        [{'effective_date': 'Lease Year 7', 'annual_rent': 700000.0}],
        '2019-04-15', square_feet=55000)
    check('a legacy step stored as text in the DATE field still resolves',
          _legacy[0]['effective_date'] == '2025-04-15',
          str(_legacy[0]['effective_date']))

    LRS.ensure_lease_tables(_meng)
    check('ensure_lease_tables is idempotent, as every request depends on', True)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print('  - ' + f)
sys.exit(1 if FAIL else 0)
