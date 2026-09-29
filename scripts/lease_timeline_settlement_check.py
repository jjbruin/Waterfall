"""Guardrail: the analyst settles a tenant's timeline -- step 4 of the rent-roll plan.

New business's exhibit is "a result of the lease review and analyst validation",
so what it prints must be the ANALYST's conclusion where they reached one:

  * START and EXPIRATION are settled through the existing field resolutions
    (lease_start / lease_end), which the timeline now honours;
  * the FUTURE RENT SCHEDULE and the OPTIONS are settled whole, per tenant,
    with a required reason and a cited document of that tenant's;
  * overlaps are refused, gaps allowed but reported (spec §24);
  * a re-read that changes what the documents give is FLAGGED on the settled
    timeline, never silently kept -- the pattern the abstracts use.

Run:  .venv\\Scripts\\python.exe scripts\\lease_timeline_settlement_check.py
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
import database  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % (detail,) if detail and not cond else ''))


def section(t):
    print('\n--- ' + t)


def refused(fn, *a):
    try:
        fn(*a)
        return False
    except ValueError:
        return True


DB = os.path.join(tempfile.gettempdir(), 'lease_settle.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)
TERMS = {'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31',
         'renewal_options': [{'option_number': 1, 'term_years': 5,
                              'option_start': '2031-01-01', 'option_end': '2035-12-31'}]}
with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name, rent_roll_date)"
                   " VALUES (1, 'Poplar', '2026-09-01')"))
    for tid, name in ((10, 'A Perfect Bloom'), (20, 'Someone Else')):
        c.execute(text("INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
                       " square_feet, annual_rent, tenant_status, is_vacant, extraction_json)"
                       " VALUES (:i, 1, :n, 'A1', 2300, 55200, 'active', 0, :j)"),
                  {'i': tid, 'n': name, 'j': json.dumps(TERMS)})
    c.execute(text("INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
                   " doc_type, doc_date) VALUES (5, 10, 1, 'Lease.pdf', 'Original Lease',"
                   " '2019-12-01'), (6, 20, 1, 'Other.pdf', 'Original Lease', '2019-12-01')"))
    for d, psf in (('2020-01-01', 22), ('2025-01-01', 24), ('2027-01-01', 25)):
        c.execute(text("INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
                       " rent_per_sf, source_doc, source_doc_id) VALUES (10, :d, :a, :p,"
                       " 'Lease.pdf', 5)"), {'d': d, 'a': psf * 2300, 'p': psf})
S.consolidate_tenant_extractions(eng, 10)

section('before anything is settled')
tl = S.get_tenant_timeline(eng, 1, 10)
chk('the derived timeline: one future step, one option, from the documents',
    [r['start'] for r in tl['future']] == ['2027-01-01'] and len(tl['options']) == 1
    and tl['schedule_basis'] == 'lease schedule', (tl['future'], tl['options']))

section('settling the future rent schedule')
SCHED = [{'start': '2027-04-01', 'end': '2028-03-31', 'annual_rent': 58200},
         {'start': '2028-04-01', 'end': '2030-12-31', 'annual_rent': 61200}]
chk('refused without a reason', refused(S.save_timeline_settlement, eng, 1, 10, 'schedule',
                                        SCHED, '  ', 5, 'kh'))
chk('refused without a cited document', refused(S.save_timeline_settlement, eng, 1, 10,
                                                'schedule', SCHED, 'why', None, 'kh'))
chk("refused citing another tenant's document",
    refused(S.save_timeline_settlement, eng, 1, 10, 'schedule', SCHED, 'why', 6, 'kh'))
chk('refused when periods overlap', refused(
    S.save_timeline_settlement, eng, 1, 10, 'schedule',
    [{'start': '2027-04-01', 'end': '2028-06-30', 'annual_rent': 1},
     {'start': '2028-04-01', 'end': '2030-12-31', 'annual_rent': 2}], 'why', 5, 'kh'))
chk('refused with a row missing its rent', refused(
    S.save_timeline_settlement, eng, 1, 10, 'schedule',
    [{'start': '2027-04-01', 'end': '2028-03-31'}], 'why', 5, 'kh'))
res = S.save_timeline_settlement(eng, 1, 10, 'schedule',
                                 [SCHED[0], {'start': '2028-06-01', 'end': '2030-12-31',
                                             'annual_rent': 61200}],
                                 'First Amendment re-dates the steps to April', 5, 'kh')
chk('a gap is accepted but reported', res['warnings'] and 'gap' in res['warnings'][0],
    res['warnings'])
S.save_timeline_settlement(eng, 1, 10, 'schedule', SCHED,
                           'First Amendment re-dates the steps to April', 5, 'kh')
tl = S.get_tenant_timeline(eng, 1, 10)
chk('the timeline shows the settled schedule, and says so',
    [r['start'] for r in tl['future']] == ['2027-04-01', '2028-04-01']
    and tl['schedule_basis'] == 'settled by the analyst', tl['future'])
chk('...the rent-roll rows follow it',
    [(r['label'], r['start']) for r in tl['rows'] if r['kind'] == 'step'] ==
    [('Rent Step Dates', '2027-04-01'), (None, '2028-04-01')])
chk('...and carries who settled it and why',
    tl['settlements']['schedule']['reason'].startswith('First Amendment')
    and tl['settlements']['schedule']['settled_by'] == 'kh')
chk('no re-read flag while the documents are unchanged',
    not any(f['code'] == 'reread_since_settled' for f in tl['flags']), tl['flags'])

section('settling the options')
S.save_timeline_settlement(eng, 1, 10, 'options', [
    {'term_years': 5, 'periods': [
        {'start': '2031-01-01', 'end': '2031-12-31', 'annual_rent': 70000},
        {'start': '2032-01-01', 'end': '2035-12-31', 'annual_rent': 72000}]},
    {'start': '2036-01-01', 'end': '2040-12-31', 'annual_rent': None,
     'rent_basis': 'fmv'}], 'Rider 3 states option rent', 5, 'kh')
tl = S.get_tenant_timeline(eng, 1, 10)
chk('two settled options, the first with its own step',
    [(r['label'], r['start']) for r in tl['rows'] if r['kind'].startswith('option')] ==
    [('Option 1', '2031-01-01'), ('Rent Step Dates', '2032-01-01'), ('Option 2', '2036-01-01')],
    [(r['label'], r['start']) for r in tl['rows']])
chk('the Option(s) column is recomputed from the settled options: "2 x 5 Years"',
    tl['rows'][0]['options'] == '2 x 5 Years', tl['rows'][0]['options'])
chk('an FMV option may carry no rent', [r for r in tl['rows'] if r['label'] == 'Option 2'][0]
    ['annual_rent'] is None)

section('start and expiration, through the field resolutions')
S.resolve_field(eng, 10, 'lease_end', '2031-03-31', resolved_by='kh',
                reason='Commencement letter moves the term', source_doc_id=5)
S.resolve_field(eng, 10, 'lease_start', '2026-04-01', resolved_by='kh',
                reason='Current lease term start per IC convention', source_doc_id=5)
tl = S.get_tenant_timeline(eng, 1, 10)
chk('the settled expiration governs the tenant row', tl['rows'][0]['end'] == '2031-03-31'
    and tl['expiration_basis'] == 'settled by the analyst', tl['rows'][0])
chk('the settled start governs the tenant row', tl['rows'][0]['start'] == '2026-04-01'
    and tl['start_basis'] == 'settled by the analyst', tl['rows'][0])

section('a re-read that changes the documents is flagged on the settlement')
with eng.begin() as c:
    c.execute(text("INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
                   " rent_per_sf, source_doc, source_doc_id) VALUES (10, '2029-01-01',"
                   " 62100, 27, 'Lease.pdf', 5)"))
tl = S.get_tenant_timeline(eng, 1, 10)
chk("the settled schedule stays -- it is the analyst's -- but is flagged",
    [r['start'] for r in tl['future']] == ['2027-04-01', '2028-04-01']
    and any(f['code'] == 'reread_since_settled' and 'schedule' in f['message']
            for f in tl['flags']), tl['flags'])
S.save_timeline_settlement(eng, 1, 10, 'schedule', SCHED, 'Re-checked after re-read', 5, 'kh')
tl = S.get_tenant_timeline(eng, 1, 10)
chk('settling again clears the flag',
    not any(f['code'] == 'reread_since_settled' and 'schedule' in f['message']
            for f in tl['flags']), tl['flags'])

section('clearing, scoping, protection')
S.clear_timeline_settlement(eng, 1, 10, 'schedule')
tl = S.get_tenant_timeline(eng, 1, 10)
chk('clearing returns the derived schedule', tl['schedule_basis'] == 'lease schedule')
S.clear_timeline_settlement(eng, 2, 10, 'options')     # wrong review: no effect
chk("a clear naming another review does not touch this tenant's settlement",
    'options' in S.get_timeline_settlements(eng, 10))
chk('the settlements table is protected from CSV replacement',
    'lease_timeline_settlements' in database.PROTECTED_TABLES)
rr = S.get_rent_roll_timeline(eng, 1)
chk('the whole-property rent roll carries the settled tenant', any(
    t['tenant_id'] == 10 and t['options_basis'] == 'settled by the analyst'
    for t in rr['tenants']))

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
sys.exit(1 if BAD else 0)
