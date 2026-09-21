"""Guardrail: the amendment governs, and a document that was never read says so.

Analyst feedback on the Sep 20 2026 re-run, seven tenants. It resolved into SIX
defects, five of them corpus-wide rather than particular to those leases:

  A  Thirteen documents had never reached the model (8 `text_extracted`, 4
     `error`, 1 `pending`), including Ciao Baby's 4th Amendment and both of Hobby
     Lobby's option notices -- the exact documents reported as "not recognized".
     `error` was excluded from the retry, so a document that failed once could
     never be read again, and nothing on any screen said so.

  B  An original lease's month-of-term schedule was anchored to whatever rent
     commencement a LATER amendment set, re-dating it on top of the amendment's
     own rent. 22 tenants. Benjamin Moore reported $38,038 against an amendment
     saying $50,052 -- the rent roll was right and the app was wrong -- and Kohls
     projected its 2017 schedule to 2062.

  C  168 of 809 steps had neither a date nor a period, so they could never be in
     force and lost silently to the original lease. Chapultepec's amendment raised
     the rent to $53,331.96 and the app went on reporting $51,999.96.

  D  An amendment that ADDS space adds rent. Marco's Pizza's first amendment takes
     another 160 SF for another $242 a month; read as a replacement it reported
     $2,904 a year against a rent roll of $65,558 -- while the square footage in
     the same amendment was combined correctly, so one document disagreed with
     itself.

  E  The merge dropped the base lease's undated steps whenever an amendment
     supplied any. 53 tenants held a blob far shorter than their step table
     (Kohls: 1 against 11), which is what the abstract reads.

  F  82 tied effective dates across 31 tenants, where `max` returned whichever row
     the list happened to hold first. BooYa's has four steps on 2024-04-01 --
     $52,800 from the 2008 lease and $103,596 from the 5th amendment.

EVERY RULE IS ASSERTED IN BOTH DIRECTIONS. "The amendment wins" is satisfied by
always taking the newest number; "additional rent is added" is satisfied by adding
everything. So each has a case that must move and a case that must not.

Run:  .venv\\Scripts\\python.exe scripts\\lease_amendment_governs_check.py
"""
import json
import os
import re
import sys
import tempfile
import logging

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

from sqlalchemy import create_engine, text  # noqa: E402
from flask_app.services import lease_review_service as S  # noqa: E402
from flask_app.services.lease_terms import (  # noqa: E402
    resolve_rent_steps, step_in_force_at, additional_in_force,
)

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


def section(t):
    print('\n--- ' + t)


DB = os.path.join(tempfile.gettempdir(), 'lease_gov.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

RR = '2026-09-01'
with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name, rent_roll_date)"
                   " VALUES (4, 'Governs Centre', :d)"), {'d': RR})

    # --- B: Benjamin Moore. The amendment extends and states $50,052; the
    # ORIGINAL lease's months 3-14 must stay on the ORIGINAL term.
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
        " square_feet, annual_rent, monthly_rent, rent_per_sf, lease_end,"
        " is_vacant, tenant_status, rent_commencement,"
        " original_rent_commencement, extraction_json, extraction_status)"
        " VALUES (1,4,'Benjamin Moore','820',2002,50052,4171,25,'2031-03-31',0,"
        " 'active','2026-04-01','2021-06-01',:j,'extracted')"),
        {'j': json.dumps({'square_feet': 2002})})
    for did, fn, dt, dd in ((11, 'BM/Lease.pdf', 'Original Lease', '2021-05-01'),
                            (12, 'BM/1st Amend.pdf', 'Amendment', '2026-03-01')):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
            " doc_type, doc_date, extraction_status) VALUES"
            " (:i,1,4,:f,:t,:d,'extracted')"),
            {'i': did, 'f': fn, 't': dt, 'd': dd})
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, period_start_month,"
        " period_end_month, annual_rent, monthly_rent, source_doc, source_doc_id,"
        " term_start, effective_date, effective_date_basis)"
        " VALUES (1,3,14,38037.96,3169.83,'BM/Lease.pdf',11,NULL,"
        " '2026-06-01','month 3 of the term')"))
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
        " monthly_rent, source_doc, source_doc_id, term_start)"
        " VALUES (1,'2026-04-01',50052.0,4171.0,'BM/1st Amend.pdf',12,"
        " '2026-04-01')"))

    # --- C: Chapultepec. The amendment's step has NO date and NO period.
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
        " square_feet, annual_rent, monthly_rent, rent_per_sf, lease_end,"
        " is_vacant, tenant_status, rent_commencement,"
        " original_rent_commencement, extraction_json, extraction_status)"
        " VALUES (2,4,'Chapultepec','640',2600,53331.96,4444.33,20.51,"
        " '2027-03-31',0,'active','2017-04-01','2017-04-01',:j,'extracted')"),
        {'j': json.dumps({'square_feet': 2600})})
    for did, fn, dt, dd in ((21, 'CH/Lease.pdf', 'Original Lease', '2017-02-28'),
                            (22, 'CH/1st Amend.pdf', 'Amendment', '2022-03-30')):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
            " doc_type, doc_date, extraction_status) VALUES"
            " (:i,2,4,:f,:t,:d,'extracted')"),
            {'i': did, 'f': fn, 't': dt, 'd': dd})
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, period_start_month,"
        " period_end_month, annual_rent, monthly_rent, source_doc, source_doc_id)"
        " VALUES (2,1,60,51999.96,4333.33,'CH/Lease.pdf',21)"))
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, annual_rent, monthly_rent,"
        " source_doc, source_doc_id) VALUES (2,53331.96,4444.33,"
        " 'CH/1st Amend.pdf',22)"))

    # --- D: Marco's. The amendment ADDS 160 SF for $242 a month.
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
        " square_feet, annual_rent, monthly_rent, rent_per_sf, lease_end,"
        " is_vacant, tenant_status, rent_commencement,"
        " original_rent_commencement, extraction_json, extraction_status)"
        " VALUES (3,4,'Marcos Pizza','700',3772,71365.8,5947.15,18.92,"
        " '2027-02-28',0,'active','2017-03-01','2017-03-01',:j,'extracted')"),
        {'j': json.dumps({'square_feet': 3772})})
    for did, fn, dt, dd in ((31, 'MP/Lease.pdf', 'Original Lease', '2016-10-17'),
                            (32, 'MP/1st Amend.pdf', 'Amendment', '2023-07-18')):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
            " doc_type, doc_date, extraction_status) VALUES"
            " (:i,3,4,:f,:t,:d,'extracted')"),
            {'i': did, 'f': fn, 't': dt, 'd': dd})
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
        " monthly_rent, source_doc, source_doc_id, is_additional)"
        " VALUES (3,'2022-05-01',68461.8,5705.15,'MP/Lease.pdf',31,0)"))
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
        " monthly_rent, source_doc, source_doc_id, is_additional)"
        " VALUES (3,'2023-07-17',2904.0,242.0,'MP/1st Amend.pdf',32,1)"))

    # --- F: BooYa's. Three steps on ONE date, from three documents.
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
        " square_feet, annual_rent, monthly_rent, rent_per_sf, lease_end,"
        " is_vacant, tenant_status, rent_commencement,"
        " original_rent_commencement, extraction_json, extraction_status)"
        " VALUES (4,4,'BooYas','954',4777,103596,8633,21.69,'2029-03-31',0,"
        " 'active','2024-04-01','2008-03-15',:j,'extracted')"),
        {'j': json.dumps({'square_feet': 4777})})
    for did, fn, dt, dd, amt in (
            (41, 'BY/Lease.pdf', 'Original Lease', '2007-12-17', 52800.0),
            (42, 'BY/Addendum 4.pdf', 'Other', '2019-02-27', 97448.64),
            (43, 'BY/5th Amend.pdf', 'Amendment', '2024-02-23', 103596.0)):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
            " doc_type, doc_date, extraction_status) VALUES"
            " (:i,4,4,:f,:t,:d,'extracted')"),
            {'i': did, 'f': fn, 't': dt, 'd': dd})
        c.execute(text(
            "INSERT INTO lease_rent_steps (tenant_id, effective_date,"
            " annual_rent, monthly_rent, source_doc, source_doc_id)"
            " VALUES (4,'2024-04-01',:a,:m,:f,:i)"),
            {'a': amt, 'm': amt / 12.0, 'f': fn, 'i': did})

    # --- A: documents that were never read, one of each state.
    c.execute(text(
        "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
        " doc_type, doc_date, extraction_status) VALUES"
        " (51,2,4,'CH/4th Amend.pdf','Amendment','2023-03-07','text_extracted')"))
    c.execute(text(
        "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
        " doc_type, doc_date, extraction_status) VALUES"
        " (52,1,4,'BM/2025_COI.pdf','COI','2025-01-01','error')"))

res = S.validate_rent_roll(eng, 4)
got = {}
with eng.connect() as c:
    for r in c.execute(text(
            "SELECT t.tenant_name, v.field_name, v.seller_value, v.lease_value,"
            " v.status, v.notes FROM lease_validation v"
            " JOIN lease_tenants t ON t.id = v.tenant_id")).fetchall():
        got[(r[0], r[1])] = (r[2], r[3], r[4], r[5])


def lease_annual(name):
    v = got.get((name, 'annual_rent'))
    return float(v[1]) if v and v[1] not in (None, '') else None


section('B. The original schedule stays on the ORIGINAL term')
chk('the amendment governs the rent in force',
    lease_annual('Benjamin Moore') == 50052.0, str(lease_annual('Benjamin Moore')))
chk('...so the rent roll agrees with the lease',
    (got.get(('Benjamin Moore', 'annual_rent')) or ('', '', ''))[2] == 'match',
    str((got.get(('Benjamin Moore', 'annual_rent')) or [None] * 3)[2]))
# BOTH DIRECTIONS: anchoring everything to the ORIGINAL date would be just as
# wrong. The amendment's own months must count from the amendment's own term.
_steps = [
    {'period_start_month': 1, 'period_end_month': 12, 'annual_rent': 10.0,
     'term_start': None},
    {'period_start_month': 1, 'period_end_month': 12, 'annual_rent': 99.0,
     'term_start': '2026-04-01'},
]
_r, _ = resolve_rent_steps(_steps, '2021-06-01')
chk('a step with its OWN term start is dated from that',
    _r[1]['effective_date'] == '2026-04-01', str(_r[1]['effective_date']))
chk('...and one without it from the original',
    _r[0]['effective_date'] == '2021-06-01', str(_r[0]['effective_date']))
chk('...and the basis names which term it counted from',
    'term beginning 2026-04-01' in (_r[1].get('effective_date_basis') or ''),
    str(_r[1].get('effective_date_basis')))
# THE STORED DATE IS THE WHOLE PROBLEM. Consolidation writes the resolved date
# back onto the step, so every row already in the table carries one computed from
# the OLD anchor -- Benjamin Moore's months 3-14 sits in the database as
# 2026-06-01. Read back as stated, it beats the amendment for ever and the fix
# above changes nothing at all. The fixture above stores exactly that.
_stale = [{'effective_date': '2026-06-01',
           'effective_date_basis': 'month 3 of the term',
           'period_start_month': 3, 'annual_rent': 1.0, 'term_start': None}]
chk('a date the app DERIVED is re-derived, not trusted',
    resolve_rent_steps(_stale, '2021-06-01')[0][0]['effective_date']
    == '2021-08-01',
    str(resolve_rent_steps(_stale, '2021-06-01')[0][0]['effective_date']))
# BOTH DIRECTIONS: a date the DOCUMENT stated must survive, period or no period.
_real = [{'effective_date': '2026-06-01', 'effective_date_basis': 'stated',
          'period_start_month': 3, 'annual_rent': 1.0, 'term_start': None}]
chk("...but a date the document STATED is kept",
    resolve_rent_steps(_real, '2021-06-01')[0][0]['effective_date']
    == '2026-06-01',
    str(resolve_rent_steps(_real, '2021-06-01')[0][0]['effective_date']))


section('C. An amendment that states no date still applies')
chk('the amendment rent is in force, not the original',
    lease_annual('Chapultepec') == 53331.96, str(lease_annual('Chapultepec')))
chk('...dated from the document that states it',
    'from the document that states it' in (
        (got.get(('Chapultepec', 'annual_rent')) or [None] * 4)[3] or ''),
    str((got.get(('Chapultepec', 'annual_rent')) or [None] * 4)[3])[:80])
# BOTH DIRECTIONS: a step with nothing at all to date it must still be undatable,
# or every stray figure becomes "in force" on the day its document was signed.
_u, _ = resolve_rent_steps([{'annual_rent': 1.0}], None)
chk('a step with no date, no period and no document stays undated',
    _u[0]['effective_date'] is None, str(_u[0]['effective_date']))


section('D. An additional charge is ADDED, not substituted')
chk('the base rent and the added space are combined',
    lease_annual('Marcos Pizza') == 71365.8, str(lease_annual('Marcos Pizza')))
chk('...and the note says an addition is included',
    'additional charge' in (
        (got.get(('Marcos Pizza', 'annual_rent')) or [None] * 4)[3] or ''),
    str((got.get(('Marcos Pizza', 'annual_rent')) or [None] * 4)[3])[:90])
# BOTH DIRECTIONS: adding everything would be the same defect facing the other
# way -- a step NOT marked additional must REPLACE.
_m = [{'effective_date': '2022-05-01', 'annual_rent': 100.0},
      {'effective_date': '2023-01-01', 'annual_rent': 200.0}]
_b, _ = step_in_force_at(_m, RR)
_a, _t = additional_in_force(_m, RR)
chk('an ordinary later step replaces rather than adds',
    _b['annual_rent'] == 200.0 and _t == 0.0, '%s / %s' % (_b['annual_rent'], _t))
chk('an additional charge is not a candidate for the base rent',
    step_in_force_at([{'effective_date': '2023-01-01', 'annual_rent': 5.0,
                       'is_additional': True}], RR)[0] is None)


section('F. Steps sharing a date: the later DOCUMENT governs')
chk('the 5th amendment wins over the 2008 lease',
    lease_annual('BooYas') == 103596.0, str(lease_annual('BooYas')))
chk('...and the basis names the document',
    '5th Amend' in ((got.get(('BooYas', 'annual_rent')) or [None] * 4)[3] or ''),
    str((got.get(('BooYas', 'annual_rent')) or [None] * 4)[3])[:90])
# A step we cannot attribute must not outrank one we can.
_tie = [{'effective_date': '2024-04-01', 'annual_rent': 1.0},
        {'effective_date': '2024-04-01', 'annual_rent': 2.0,
         'doc_date': '2024-02-23', 'source_doc_id': 43}]
chk('an unattributed step does not beat an attributed one',
    step_in_force_at(_tie, RR)[0]['annual_rent'] == 2.0)


section('A. A document that was never read SAYS so')
unread = S.unread_documents(eng, 4)
names = [d['filename'] for d in unread]
chk('a text_extracted document is listed', 'CH/4th Amend.pdf'.split('/')[-1]
    in names, str(names))
chk('an ERRORED document is listed too', '2025_COI.pdf' in names, str(names))
chk('the term-bearing one is first', unread[0]['term_bearing'] is True,
    str(unread[:1]))
chk('...and a COI is marked as not term-bearing',
    any(d['term_bearing'] is False for d in unread))
# BOTH DIRECTIONS: listing everything would be useless.
chk('an extracted document is NOT listed',
    'Lease.pdf' not in [n for n in names], str(names))
# `error` was excluded from the retry, so a document that failed once could never
# be read again by any run.
import inspect  # noqa: E402
# COMMENTS ARE STRIPPED FIRST. The initial version of this check grepped the
# function source for "'error'" and passed with the retry REMOVED, because the
# comment above the query explains the rule and contains the word in quotes. A
# check satisfied by its own documentation is worth nothing.
_src = inspect.getsource(S.extract_all_documents)
_code = chr(10).join(ln for ln in _src.splitlines()
                     if not ln.strip().startswith(('#', '--')))
_sel = re.search(r'extraction_status IN \(([^)]*)\)', _code)
chk('the retry selects a status set at all', _sel is not None)
_statuses = (_sel.group(1) if _sel else '')
chk('an errored document is retried by the next run', "'error'" in _statuses,
    _statuses.strip())
chk('...alongside pending and text_extracted',
    "'pending'" in _statuses and "'text_extracted'" in _statuses,
    _statuses.strip())


section('E. The merge keeps the base lease\'s undated steps')
base = {'rent_steps': [
    {'period_start_month': 1, 'period_end_month': 12, 'annual_rent': 100},
    {'period_start_month': 13, 'period_end_month': 24, 'annual_rent': 110}]}
amend = {'rent_steps': [{'effective_date': '2026-04-01', 'annual_rent': 200}]}
merged = S._merge_extraction_terms(base, amend)
chk('an amendment does not wipe the base schedule',
    len(merged['rent_steps']) == 3, str(len(merged['rent_steps'])))
chk('...and its own step is there', any(
    s.get('annual_rent') == 200 for s in merged['rent_steps']))
# BOTH DIRECTIONS: keeping everything would leave superseded rent in the
# schedule. An amendment restating the SAME period replaces it.
re_stated = S._merge_extraction_terms(base, {'rent_steps': [
    {'period_start_month': 1, 'period_end_month': 12, 'annual_rent': 555}]})
chk('an amendment restating the same months SUPERSEDES',
    len(re_stated['rent_steps']) == 2
    and any(s.get('annual_rent') == 555 for s in re_stated['rent_steps'])
    and not any(s.get('annual_rent') == 100 for s in re_stated['rent_steps']),
    str([s.get('annual_rent') for s in re_stated['rent_steps']]))
# The same period stated only as TEXT must collide too -- the months are not
# parsed until `resolve_rent_steps`, so without reading the wording here the
# First Amendment's "Months 61-120" and the Fourth's are two different keys and
# the superseded figure comes back. `lease_terms_check` passed on that case only
# because the old merge dropped every undated base step.
_text_period = S._merge_extraction_terms(
    {'rent_steps': [{'period': 'Months 61-120', 'annual_rent': 510000.0}]},
    {'rent_steps': [{'period': 'Months 61-120', 'annual_rent': 544500.0}]})
chk('a period stated only in words still supersedes',
    len(_text_period['rent_steps']) == 1
    and _text_period['rent_steps'][0]['annual_rent'] == 544500.0,
    str([x.get('annual_rent') for x in _text_period['rent_steps']]))
chk('...and a DIFFERENT period is kept alongside it',
    len(S._merge_extraction_terms(
        {'rent_steps': [{'period': 'Months 1-60', 'annual_rent': 1.0}]},
        {'rent_steps': [{'period': 'Months 61-120', 'annual_rent': 2.0}]}
    )['rent_steps']) == 2)
chk('...and one restating the same DATE still supersedes', len(
    S._merge_extraction_terms(
        {'rent_steps': [{'effective_date': '2026-04-01', 'annual_rent': 1}]},
        {'rent_steps': [{'effective_date': '2026-04-01', 'annual_rent': 2}]}
    )['rent_steps']) == 1)


section('The backfill is what makes any of it live on data we already hold')
# EVERY FIX ABOVE READS source_doc_id / term_start, and only a NEW extraction
# writes them -- so on the 809 steps already stored nothing would have changed and
# the analyst would have seen no difference at all.
DB2 = os.path.join(tempfile.gettempdir(), 'lease_gov2.db')
if os.path.exists(DB2):
    os.remove(DB2)
eng2 = create_engine('sqlite:///%s' % DB2)
S.ensure_lease_tables(eng2)
with eng2.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name)"
                   " VALUES (9, 'Backfill Centre')"))
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
        " square_feet, is_vacant, tenant_status, extraction_status)"
        " VALUES (1,9,'Backfill Me','S1',1000,0,'active','extracted')"))
    c.execute(text(
        "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
        " doc_type, doc_date, extraction_status, extraction_json) VALUES"
        " (1,1,9,'B/Lease.pdf','Original Lease','2019-01-01','extracted',:j)"),
        {'j': json.dumps({'rent_commencement': '2019-06-01'})})
    c.execute(text(
        "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
        " doc_type, doc_date, extraction_status, extraction_json) VALUES"
        " (2,1,9,'B/Amend.pdf','Amendment','2026-01-01','extracted',:j)"),
        {'j': json.dumps({'rent_commencement': '2026-04-01'})})
    # Steps as they sit today: a source FILENAME, and nothing else.
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, period_start_month,"
        " annual_rent, source_doc) VALUES (1,3,100.0,'B/Lease.pdf')"))
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
        " source_doc, term_start) VALUES (1,'2026-04-01',200.0,'B/Amend.pdf',"
        " '2099-01-01')"))

S.ensure_lease_tables(eng2)   # idempotent; this is what runs the backfill
with eng2.connect() as c:
    rows = c.execute(text(
        "SELECT source_doc, source_doc_id, term_start FROM lease_rent_steps"
        " ORDER BY id")).fetchall()
    orc = c.execute(text(
        "SELECT original_rent_commencement FROM lease_tenants"
        " WHERE id = 1")).scalar()
chk('the document behind each step is matched by its filename',
    rows[0][1] == 1 and rows[1][1] == 2, str([(r[0], r[1]) for r in rows]))
chk("...and the step takes that document's own term start",
    rows[0][2] == '2019-06-01', str(rows[0][2]))
chk('a value already stored is NOT overwritten',
    rows[1][2] == '2099-01-01', str(rows[1][2]))
chk("the tenant's ORIGINAL commencement is the earliest a document states",
    orc == '2019-06-01', str(orc))
# BOTH DIRECTIONS: taking the latest is the defect this whole build is about.
chk('...not the latest', orc != '2026-04-01')


print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
