"""Guardrail: the governing lease terms -- step 2 of the rent-roll plan.

New business's rent-roll specification (Sep 29 2026) measured against their Market
at Poplar exhibit found the app's START right on 12 of 27 tenants, its EXPIRATION
wrong wherever an option had been exercised, and 115 option rows standing for 39.
Every fixture below is built from what the tenant's documents actually say on
production (read Sep 29 2026), so each check is the real case:

  MATTRESS FIRM  the 2024 3rd Amendment extends to 2035-03-21; an UNDATED
                 "Commencement Date (Exhibit)" sorted last and reset it to 2027.
  OUTBACK        the 2026 option letter exercises to 2031-12-09; an undated
                 third-party abstract reset it to 2026 and the start to 2016.
  BOOYA'S        original commencement 2008-03-15; Addendum #4 and the 5th
                 Amendment state their RENEWAL terms' starts (2019, 2024), which
                 overwrote it.
  MUDDY PAWS     the original lease came back with its own option "exercised".
  HOBBY LOBBY    the 3rd Amendment restates the option set: 1-3 exercised, 4-5 new.

No API calls. Run:  .venv\\Scripts\\python.exe scripts\\lease_governing_terms_check.py
"""
import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

from sqlalchemy import create_engine, text  # noqa: E402
from flask_app.services import lease_review_service as S  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % (detail,) if detail and not cond else ''))


def section(t):
    print('\n--- ' + t)


DB = os.path.join(tempfile.gettempdir(), 'lease_governing.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

TENANTS = {
    1: ('Mattress Firm', [
        ('2016.02.11-Mattress Firm - Lease Agmt.pdf', 'Original Lease', '2016-02-11',
         {'square_feet': 4200}),
        ('2017.12.18-Mattress Firm - Commencement Agmt.pdf', 'Commencement Letter',
         '2017-12-18', {'lease_commencement': '2017-06-26', 'lease_expiration': '2027-09-30'}),
        ('2024.08.05-Mattress Firm - 3rd Amend.pdf', 'Amendment', '2024-08-05',
         {'lease_expiration': '2035-03-21', 'renewal_options': [
             {'option_number': 1, 'term_years': 5, 'option_start': '2035-03-22',
              'option_end': '2040-03-21', 'exercised': False},
             {'option_number': 2, 'term_years': 5, 'option_start': '2040-03-22',
              'option_end': '2045-03-21', 'exercised': False}]}),
        ('Mattress Firm-Commencement Date (Exhibit).pdf', 'Commencement Letter', None,
         {'lease_expiration': '2027-09-30', 'lease_commencement': '2017-06-26'}),
    ]),
    2: ('Outback Steakhouse', [
        ('1991.09.00_Outback-Lease.pdf', 'Original Lease', None,
         {'lease_commencement': '1992-01-08', 'lease_expiration': '2002-01-31'}),
        ('2005.02.01_Outback-Pylon Sign Agreement.pdf', 'Other', '2005-02-01',
         {'lease_commencement': '2005-03-01', 'lease_expiration': '2006-12-09'}),
        ('2016.07.28_Outback-4th Amendment.pdf', 'Amendment', '2016-07-28',
         {'lease_expiration': '2026-12-09', 'renewal_options': [
             {'option_number': 1, 'term_years': 5, 'option_start': '2026-12-10',
              'option_end': '2031-12-09', 'exercised': False},
             {'option_number': 2, 'term_years': 5, 'option_start': '2031-12-10',
              'option_end': '2036-12-09', 'exercised': False}]}),
        ('2026.04.28_Outback-Option Letter.pdf', 'Option Letter', '2026-04-28',
         {'renewal_options': [{'option_number': 1, 'term_years': 5,
                               'option_start': '2026-12-10', 'option_end': '2031-12-09',
                               'exercised': True}]}),
        ('Outback_Abstracts_Misc.pdf', 'Other', None,
         {'lease_commencement': '2016-12-10', 'lease_expiration': '2026-12-09'}),
    ]),
    3: ("BooYa's", [
        ('2007.12.17-Booyas-Lease Agmt.pdf', 'Original Lease', '2007-12-17',
         {'lease_commencement': '2008-03-15', 'lease_expiration': '2014-03-31'}),
        ('2019.02.27-Booyas-Addendum #4.pdf', 'Amendment', '2019-02-27',
         {'lease_commencement': '2019-04-01', 'lease_expiration': '2024-03-30'}),
        ('2024.02.23-Booyas-5th Amend.pdf', 'Amendment', '2024-02-23',
         {'lease_commencement': '2024-04-01', 'lease_expiration': '2029-03-31',
          'renewal_options': [
              {'option_number': 1, 'term_years': 5, 'option_start': '2029-04-01',
               'option_end': '2034-03-31', 'exercised': False},
              {'option_number': 2, 'term_years': 5, 'option_start': '2034-04-01',
               'option_end': '2039-03-31', 'exercised': False}]}),
    ]),
    4: ('Muddy Paws', [
        ('2020.06.10-Muddy Paws-Lease Agmt.pdf', 'Original Lease', '2020-06-10',
         {'lease_expiration': '2025-05-31', 'renewal_options': [
             {'option_number': 1, 'term_years': 3, 'option_start': '2025-06-01',
              'option_end': '2028-05-31', 'exercised': True}]}),
    ]),
    5: ('Hobby Lobby', [
        ('2002.04.15-HOBBY LOBBY-Lease Agmt.pdf', 'Original Lease', '2002-04-15',
         {'renewal_options': [{'option_number': 1, 'term_years': 5},
                              {'option_number': 2, 'term_years': 5}]}),
        ('2020.10.31-HOBBY LOBBY-3rd Amend.pdf', 'Amendment', '2020-10-31',
         {'lease_expiration': '2024-07-31', 'renewal_options': [
             {'option_number': 3, 'term_years': 3.5, 'option_start': '2021-02-01',
              'option_end': '2024-07-31', 'exercised': True},
             {'option_number': 4, 'term_years': 3, 'option_start': '2024-08-01',
              'option_end': '2027-07-31', 'exercised': False},
             {'option_number': 5, 'term_years': 5, 'option_start': '2027-08-01',
              'option_end': '2032-07-31', 'exercised': False}]}),
        ('2024.03.06-HOBBY LOBBY-Option Exercise.pdf', 'Option Letter', '2024-03-06',
         {'renewal_options': [{'option_number': 4, 'option_start': '2024-08-01',
                               'option_end': '2027-07-31', 'exercised': True}]}),
    ]),
}

with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name) VALUES (1, 'Poplar')"))
    did = 0
    for tid, (name, docs) in TENANTS.items():
        c.execute(text("INSERT INTO lease_tenants (id, review_id, tenant_name, tenant_status)"
                       " VALUES (:i, 1, :n, 'active')"), {'i': tid, 'n': name})
        for fn, dt, dd, terms in docs:
            did += 1
            c.execute(text(
                "INSERT INTO lease_documents (id, tenant_id, review_id, filename, doc_type,"
                " doc_date, extraction_status, extraction_json) VALUES"
                " (:i,:t,1,:f,:dt,:dd,'extracted',:j)"),
                {'i': did, 't': tid, 'f': fn, 'dt': dt, 'dd': dd, 'j': json.dumps(terms)})

C = {tid: S.consolidate_tenant_extractions(eng, tid) for tid in TENANTS}

section('Mattress Firm: an undated exhibit does not overwrite a dated amendment')
chk('expiration is the 3rd Amendment\'s 2035-03-21 (was reset to 2027)',
    C[1]['lease_expiration'] == '2035-03-21', C[1]['lease_expiration'])
chk('...the commencement is the commencement agreement\'s', C[1].get('lease_commencement')
    == '2017-06-26', C[1].get('lease_commencement'))
chk('two options remain: "2 x 5 Years"', C[1]['_options_summary'] == '2 x 5 Years',
    C[1]['_options_summary'])

section('Outback: the option letter governs; the undated abstract only fills gaps')
chk('expiration is the exercised option\'s 2031-12-09 (was 2026)',
    C[2]['lease_expiration'] == '2031-12-09', C[2]['lease_expiration'])
chk('the lease START stays the original lease\'s 1992-01-08 (was 2016)',
    C[2]['lease_commencement'] == '1992-01-08', C[2]['lease_commencement'])
chk('the exercised option is not counted as remaining: "1 x 5 Years"',
    C[2]['_options_summary'] == '1 x 5 Years', C[2]['_options_summary'])
chk('a later document\'s commencement is kept as the current term\'s start, not lost',
    C[2].get('current_term_commencement') == '2005-03-01', C[2].get('current_term_commencement'))

section("BooYa's: a renewal term's start is not the lease start")
chk('lease START is the original 2008-03-15 (was 2024)',
    C[3]['lease_commencement'] == '2008-03-15', C[3]['lease_commencement'])
chk('...the current term runs from 2024-04-01', C[3].get('current_term_commencement')
    == '2024-04-01', C[3].get('current_term_commencement'))
chk('expiration 2029-03-31, 2 options remaining', C[3]['lease_expiration'] == '2029-03-31'
    and C[3]['_options_summary'] == '2 x 5 Years', (C[3]['lease_expiration'], C[3]['_options_summary']))

section('Muddy Paws: an original lease cannot record its own exercise')
chk('the option is NOT treated as exercised, so the expiry stays 2025-05-31',
    C[4]['lease_expiration'] == '2025-05-31', C[4]['lease_expiration'])
chk('...and it remains available: "1 x 3 Years"', C[4]['_options_summary'] == '1 x 3 Years',
    C[4]['_options_summary'])

section('Hobby Lobby: a restated option set, then an exercise')
chk('the exercise notice carries the expiration to option 4\'s 2027-07-31',
    C[5]['lease_expiration'] == '2027-07-31', C[5]['lease_expiration'])
chk('...and records why', 'exercised option' in (C[5].get('_expiration_basis') or ''))
chk('only option 5 remains: "1 x 5 Years" (the original lease\'s undated 1-2 are gone)',
    C[5]['_options_summary'] == '1 x 5 Years', C[5]['_options_summary'])

section('Unit: _merge_renewal_options')
cur = [{'option_number': 1, 'term_years': 5, 'option_start': '2014-04-01',
        'option_end': '2019-03-31', 'exercised': False}]
undated = [{'option_number': 1, 'term_years': 5, 'option_start': None, 'option_end': None,
            'exercised': False}]
out = S._merge_renewal_options(cur, undated)
chk('an undated restatement does not strip the dates the lease gave',
    out[0]['option_start'] == '2014-04-01', out)
out = S._merge_renewal_options(cur, [{'option_number': 9, 'option_end': '2019-03-31',
                                      'exercised': True}])
chk('an exercise matches by end date when the number disagrees',
    len(out) == 1 and out[0]['exercised'] is True, out)

section('Option rows are replaced per document')
with eng.begin() as c:
    S._write_document_option_rows(c, text, 1, 'doc.pdf', {'renewal_options': [
        {'option_number': 1}, {'option_number': 2}]})
    S._write_document_option_rows(c, text, 1, 'doc.pdf', {'renewal_options': [
        {'option_number': 1}], 'termination_options': [{'option_number': 1}]})
    n = c.execute(text("SELECT option_type, COUNT(*) FROM lease_options WHERE tenant_id=1"
                       " AND source_doc='doc.pdf' GROUP BY option_type")).fetchall()
chk('a re-read replaces the document\'s option rows (1 renewal + 1 termination, not 3)',
    dict(n) == {'renewal': 1, 'termination': 1}, n)

section('Classifier: an addendum is an amendment')
chk('"Addendum #4" is typed Amendment',
    S.classify_document('2019.02.27-Booyas-Addendum #4-Poplar.pdf') == 'Amendment',
    S.classify_document('2019.02.27-Booyas-Addendum #4-Poplar.pdf'))

section('The prompt asks for option rent as figures, and forbids estimating it')
P = S.EXTRACTION_PROMPT
chk('rent_schedule and rent_basis are in the schema',
    '"rent_schedule"' in P and '"rent_basis"' in P)
chk('...fair market rent is never estimated, a percentage never computed',
    'never estimate a market rent' in P and 'do not compute them' in P)

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
sys.exit(1 if BAD else 0)
