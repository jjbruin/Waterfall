"""Guardrail: a finding can be settled, the change is reported, and fixed
recoveries are checked against the rent roll.

Jim, Sep 20 2026, four things on the validation screen:

  1. "format all rents on the validation screen as whole dollars with commas and
     no decimals. For $/SF items format as dollars with cents."
  2. "how does the analyst clear the mismatches on this page?"  -- there was no
     control at all; the only thing near a finding was a per-TENANT approve/flag
     on a later step, recording no value, no reason and no document.
  3. "If the analyst determines that the applicable rent is different from the
     rent roll, we need a clear report showing the changes with the reasons for
     the change citing the lease document that was used."
  4. "If the seller provides missing documentation, then we should be able to
     load that document right here and update lease side accordingly."

and a question, having read the AT&T Mobility 4th Amendment on Market at Poplar:

  5. "one of the lease amendments was stating a fixed CAM charge for the lease.
     Is this situation part of the lease review and validation to the rent roll?"

It was not. The extraction captured `cam_structure = "fixed"` -- the WORD -- and
nothing captured the AMOUNT, so the figure the rent roll could be checked against
did not exist anywhere in the app. That amendment states $2.16 / $2.38 / $2.62 per
SF by year and caps it in both directions ("in excess of or below"), while the rent
roll carries $4.632/SF of total recoveries.

EVERY NARROWING IS ASSERTED IN BOTH DIRECTIONS. "A reason is required" is
satisfied by refusing everything; "the finding is settled" is satisfied by
accepting anything. So each rule is checked by a call that must pass and a call
that must fail.

Run:  .venv\\Scripts\\python.exe scripts\\lease_validation_resolve_check.py
"""
import json
import os
import sys
import tempfile
import logging

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

from sqlalchemy import create_engine, text  # noqa: E402
from flask_app.services import lease_review_service as S  # noqa: E402
from flask_app.services.lease_terms import (  # noqa: E402
    cam_fixed_in_force, annual_recovery_psf,
)

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


def section(t):
    print('\n--- ' + t)


# ---------------------------------------------------------------------------
# The AT&T Mobility schedule, transcribed from the amendment itself
# ---------------------------------------------------------------------------
CAM_SCHEDULE = [
    {'period': '2025', 'year_start': 2025, 'year_end': 2025,
     'per_sf': 2.16, 'annual': 8640.0, 'monthly': 720.0},
    {'period': '2026 to 2030', 'year_start': 2026, 'year_end': 2030,
     'per_sf': 2.38, 'annual': 9520.0, 'monthly': 793.33},
    {'period': '2031 to 2035', 'year_start': 2031, 'year_end': 2035,
     'per_sf': 2.62, 'annual': 10480.0, 'monthly': 873.33},
]


section('The fixed recovery in force is resolved like a rent step')
row, basis = cam_fixed_in_force(CAM_SCHEDULE, '2026-09-01')
chk('the 2026-2030 row governs a 2026 rent roll date',
    row is not None and row['per_sf'] == 2.38, str(row))
chk('...and says so in a sentence, not a flag',
    '2026 to 2030' in basis and '2026-09-01' in basis, basis)
chk('a date before the schedule begins is DECLINED, not clamped',
    cam_fixed_in_force(CAM_SCHEDULE, '2024-06-30')[0] is None)
chk('...and says why', 'begins 2025' in cam_fixed_in_force(CAM_SCHEDULE, '2024-06-30')[1])
chk('a date past the end is declined too',
    cam_fixed_in_force(CAM_SCHEDULE, '2040-01-01')[0] is None)
chk('the stated per-SF figure is used as stated',
    annual_recovery_psf(row, 4000) == 2.38)
# The 12x hazard v495 shipped once from this importer: a MONTHLY figure divided
# as-is gives a plausible number a twelfth of the right one.
chk('a MONTHLY-only row is ANNUALISED before dividing',
    abs(annual_recovery_psf({'monthly': 793.33}, 4000) - 2.38) < 0.001,
    str(annual_recovery_psf({'monthly': 793.33}, 4000)))
chk('...and an ANNUAL-only row divides directly',
    abs(annual_recovery_psf({'annual': 9520.0}, 4000) - 2.38) < 0.001)
chk('no square feet means no derived figure, never a zero',
    annual_recovery_psf({'annual': 9520.0}, None) is None)


section('An amendment carries the schedule THROUGH consolidation')
# cam_fixed is a LIST, so a whitelist that knows only scalars and objects drops it
# silently -- and in the wrong direction, since the schedule is stated BY the
# amendment.
base = {'square_feet': 4000, 'cam_structure': 'pro rata', 'annual_rent': 100000}
amend = {'cam_structure': 'fixed', 'cam_fixed': CAM_SCHEDULE}
merged = S._merge_extraction_terms(base, amend)
chk('the amendment\'s fixed schedule survives the merge',
    len(merged.get('cam_fixed') or []) == 3, str(merged.get('cam_fixed'))[:60])
chk('...and its structure with it', merged.get('cam_structure') == 'fixed')
chk('an amendment silent on recoveries leaves the schedule standing',
    len(S._merge_extraction_terms(merged, {'annual_rent': 120000})
        .get('cam_fixed') or []) == 3)


# ---------------------------------------------------------------------------
DB = os.path.join(tempfile.gettempdir(), 'lease_resolve.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)
S.ensure_resolution_table(eng)

# Tenant 1: the AT&T shape -- fixed CAM, but tax and insurance pass through
# SEPARATELY, so the rent roll's single recoveries column is not the same
# quantity and a difference is a question rather than a mismatch.
T1 = {
    'square_feet': 4000, 'lease_expiration': '2035-12-31',
    'rent_commencement': '2020-01-01',
    'cam_structure': 'fixed', 'cam_fixed': CAM_SCHEDULE,
    'tax_pass_through': True, 'insurance_pass_through': True,
    '_documents_applied': ['Leases/ATT/Original Lease.pdf',
                           'Leases/ATT/4th Amendment.pdf'],
    '_governing_document': 'Leases/ATT/4th Amendment.pdf',
}
# Tenants 3 and 4: the schedule stated in LEASE YEARS, which is how the Poplar
# lease states it ("for the first five (5) Lease Years"). Tenant 3 has a rent
# commencement date, so it can be placed on the calendar; tenant 4 has none, so it
# cannot -- and must SAY so rather than fall silent.
CAM_LEASE_YEARS = [
    {'period': 'Lease Years 1-5', 'lease_year_start': 1, 'lease_year_end': 5,
     'per_sf': 1.96},
    {'period': 'Lease Years 6-10', 'lease_year_start': 6, 'lease_year_end': 10,
     'per_sf': 2.156},
    {'period': 'Lease Years 11-15', 'lease_year_start': 11, 'lease_year_end': 15,
     'per_sf': 2.3716},
]

# Tenant 2: a gross-style lease where the fixed amount IS the whole recovery, so
# the two sides ARE comparable and a difference is a real mismatch.
T2 = dict(T1)
T2 = {**T1, 'tax_pass_through': False, 'insurance_pass_through': False,
      '_documents_applied': ['Leases/B/Original Lease.pdf'],
      '_governing_document': 'Leases/B/Original Lease.pdf'}

with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name, rent_roll_date) "
                   "VALUES (7, 'Resolve Centre', '2026-09-01')"))
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, square_feet,"
        " annual_rent, monthly_rent, rent_per_sf, lease_end, is_vacant,"
        " tenant_status, rent_commencement, annual_recoveries_per_sf,"
        " extraction_json, extraction_status) VALUES "
        "(1,7,'ATT Mobility','900-01',4000,120000,10000,30,'2035-12-31',0,'active',"
        " '2020-01-01',4.632,:j,'extracted')"), {'j': json.dumps(T1)})
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, square_feet,"
        " annual_rent, monthly_rent, rent_per_sf, lease_end, is_vacant,"
        " tenant_status, rent_commencement, annual_recoveries_per_sf,"
        " extraction_json, extraction_status) VALUES "
        "(2,7,'Gross Tenant','B1',4000,120000,10000,30,'2035-12-31',0,'active',"
        " '2020-01-01',4.632,:j,'extracted')"), {'j': json.dumps(T2)})
    T3 = {**T1, 'cam_fixed': CAM_LEASE_YEARS,
          'tax_pass_through': False, 'insurance_pass_through': False,
          'rent_commencement': '2022-04-15'}
    T4 = {**T3}
    T4.pop('rent_commencement')
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, square_feet,"
        " annual_rent, monthly_rent, rent_per_sf, lease_end, is_vacant,"
        " tenant_status, rent_commencement, annual_recoveries_per_sf,"
        " extraction_json, extraction_status) VALUES "
        "(3,7,'Lease Year Tenant','C1',12000,300000,25000,25,'2037-04-14',0,'active',"
        " '2022-04-15',1.96,:j,'extracted')"), {'j': json.dumps(T3)})
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, square_feet,"
        " annual_rent, monthly_rent, rent_per_sf, lease_end, is_vacant,"
        " tenant_status, rent_commencement, annual_recoveries_per_sf,"
        " extraction_json, extraction_status) VALUES "
        "(4,7,'Undated Tenant','D1',12000,300000,25000,25,'2037-04-14',0,'active',"
        " NULL,1.96,:j,'extracted')"), {'j': json.dumps(T4)})
    for did, tid, fn, dt in ((21, 1, 'Leases/ATT/Original Lease.pdf', 'Original Lease'),
                             (22, 1, 'Leases/ATT/4th Amendment.pdf', 'Amendment'),
                             (23, 2, 'Leases/B/Original Lease.pdf', 'Original Lease')):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
            " doc_type, extraction_status, file_data) VALUES "
            "(:i,:t,7,:f,:d,'extracted',:b)"),
            {'i': did, 't': tid, 'f': fn, 'd': dt, 'b': b'%PDF-x'})
    # A rent step that resolves, so the rent comparison runs and the recovery
    # comparison is not the only row on the page.
    for tid in (1, 2):
        c.execute(text(
            "INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
            " monthly_rent, rent_per_sf) VALUES (:t,'2026-01-01',100000,8333.33,25)"),
            {'t': tid})


section('The rent roll IS checked against a fixed recovery now')
S.validate_rent_roll(eng, 7)
with eng.connect() as c:
    rec = {r[0]: (r[1], r[2], r[3], r[4]) for r in c.execute(text(
        "SELECT v.tenant_id, v.seller_value, v.lease_value, v.status, v.notes "
        "FROM lease_validation v WHERE v.field_name='annual_recoveries_per_sf'"
    )).fetchall()}
chk('a finding is raised for the fixed-CAM tenant', 1 in rec, str(sorted(rec)))
if 1 in rec:
    chk('...carrying the rent roll figure', rec[1][0] == '4.632', str(rec[1][0]))
    chk('...and the LEASE figure for the year in force', rec[1][1] == '2.38',
        str(rec[1][1]))
    # The honest call: our rent roll column sums CAM, insurance and tax, and this
    # lease fixes only the operating-expense share. Calling that a mismatch would
    # be comparing two different quantities and would be believed.
    chk('...reported as a QUESTION when tax/insurance pass through separately',
        rec[1][2] == 'review', str(rec[1][2]))
    chk('...saying exactly why the two sides are not the same quantity',
        'not the same quantity' in (rec[1][3] or ''), (rec[1][3] or '')[:80])
if 2 in rec:
    # BOTH DIRECTIONS: if every recovery row were reported as 'review', the check
    # above would pass while the feature said nothing. Where the lease passes
    # neither through, the figures ARE comparable and a difference is a mismatch.
    chk('a lease passing NEITHER through is compared for real',
        rec[2][2] == 'mismatch', str(rec[2][2]))
    chk('...and says the fixed amount is the whole recovery',
        'whole recovery' in (rec[2][3] or ''), (rec[2][3] or '')[:80])
chk('every tenant with a fixed schedule raises one, and no other tenant does',
    sorted(rec) == [1, 2, 3, 4], str(sorted(rec)))


section('A schedule stated in LEASE YEARS is placed against rent commencement')
# Lease year 1 begins ON rent commencement (2022-04-15), so at the rent roll date
# 2026-09-01 the tenant is inside Lease Years 1-5 at $1.96.
if 3 in rec:
    chk('the lease-year schedule resolves', rec[3][1] == '1.96', str(rec[3][1]))
    chk('...and says which period, and from when',
        'Lease Years 1-5' in (rec[3][3] or '')
        and '2022-04-15' in (rec[3][3] or ''), (rec[3][3] or '')[:90])
    chk('...and is compared for real, not parked as a question',
        rec[3][2] == 'match', str(rec[3][2]))
# THE CASE THE FIX EXISTS FOR: no rent commencement date, so the schedule cannot be
# placed on the calendar. It must report that -- not fall silent, and above all not
# approximate it from the calendar year, which would give $1.96 by luck here and a
# wrong figure on any lease that did not commence in April.
if 4 in rec:
    chk('a lease-year schedule with NO rent commencement is reported',
        rec[4][2] == 'review', str(rec[4][2]))
    chk('...naming what is missing',
        'no rent commencement date' in (rec[4][3] or ''), (rec[4][3] or '')[:110])
    chk('...and offering NO lease figure rather than a guess',
        rec[4][1] is None, str(rec[4][1]))

# The seam, checked on both sides: lease year 6 begins on the FIFTH anniversary.
_r5, _b5 = cam_fixed_in_force(CAM_LEASE_YEARS, '2027-04-14', '2022-04-15')
_r6, _b6 = cam_fixed_in_force(CAM_LEASE_YEARS, '2027-04-15', '2022-04-15')
chk('the day before the fifth anniversary is still Lease Years 1-5',
    _r5 and _r5['per_sf'] == 1.96, str(_r5))
chk('...and the anniversary itself steps to Lease Years 6-10',
    _r6 and _r6['per_sf'] == 2.156, str(_r6))
chk('a date before lease year 1 is declined',
    cam_fixed_in_force(CAM_LEASE_YEARS, '2022-01-01', '2022-04-15')[0] is None)
# The same row stated only as text, which is what an extraction often returns.
_txt = [{'period': 'Lease Years 6-10', 'per_sf': 2.156}]
chk('a period given only as TEXT resolves the same way',
    cam_fixed_in_force(_txt, '2027-06-01', '2022-04-15')[0] is not None)
# AND THE STRUCTURED FIELDS MUST WORK ON THEIR OWN. Every fixture above carries a
# period label that happens to parse, so deleting the lease_year_start branch
# entirely still passed -- the text fallback covered for it. This row's label is
# the way the real Poplar lease words it, which no parser reads, so only the
# structured fields can date it.
_struct = [{'period': 'first five (5) Lease Years',
            'lease_year_start': 1, 'lease_year_end': 5, 'per_sf': 1.96},
           {'period': 'thereafter', 'lease_year_start': 6, 'per_sf': 2.156}]
_sr, _sb = cam_fixed_in_force(_struct, '2024-01-01', '2022-04-15')
chk('a schedule whose WORDING no parser reads still dates from its fields',
    _sr is not None and _sr['per_sf'] == 1.96, str(_sr))
chk('...and an open-ended final row runs on',
    (cam_fixed_in_force(_struct, '2035-01-01', '2022-04-15')[0] or {})
    .get('per_sf') == 2.156)
chk('...but without a rent commencement date it is REPORTED, not guessed',
    cam_fixed_in_force(_struct, '2024-01-01')[0] is None
    and 'no rent commencement date' in cam_fixed_in_force(_struct, '2024-01-01')[1])
# ONE ENGINE: the period parser the rent steps use, not a second one.
from flask_app.services.lease_terms import parse_relative_period  # noqa: E402
chk('"Lease Years 1-5" parses as months 1-60', parse_relative_period('Lease Years 1-5') == (1, 60))
chk('...and "Lease Years 6 through 10" as 61-120',
    parse_relative_period('Lease Years 6 through 10') == (61, 120))
chk('a backwards range is refused, not silently reordered',
    parse_relative_period('Lease Years 5-1') is None)
chk('a single lease year still resolves', parse_relative_period('Lease Year 1') == (1, 12))
chk('an ISO date is NOT read as a period',
    parse_relative_period('2026-01-01') is None)


section('Settling a finding: the API')
os.environ.setdefault('DATABASE_URL', '')
from flask_app import create_app  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
import jwt  # noqa: E402
import flask_app.db as DB_  # noqa: E402
import flask_app.api.lease_review as LR_  # noqa: E402

app = create_app()
DB_.get_engine = lambda: eng
LR_.get_engine = lambda: eng
tok = jwt.encode({'sub': '1', 'username': 'chk', 'role': 'admin',
                  'exp': datetime.now(timezone.utc) + timedelta(minutes=10)},
                 app.config['JWT_SECRET'], algorithm='HS256')
H = {'Authorization': 'Bearer %s' % tok}
cl = app.test_client()

rules = {str(r.rule) for r in app.url_map.iter_rules()}
chk('the resolve endpoint exists',
    '/api/lease-review/reviews/<int:review_id>/validation/resolve' in rules)
chk('the change report exists',
    '/api/lease-review/reviews/<int:review_id>/rent-roll-changes' in rules)
chk('...and downloads',
    '/api/lease-review/reviews/<int:review_id>/rent-roll-changes/excel' in rules)

BODY = {'tenant_id': 1, 'field': 'annual_rent', 'value': '100000',
        'prior_value': '120000', 'source_doc_id': 22}

r = cl.put('/api/lease-review/reviews/7/validation/resolve', headers=H,
           json={**BODY, 'reason': ''})
chk('a resolution with NO REASON is refused', r.status_code == 400,
    str(r.status_code))
chk('...and says why', 'reason is required' in (r.get_json() or {}).get('error', ''))

r = cl.put('/api/lease-review/reviews/7/validation/resolve', headers=H,
           json={**BODY, 'reason': 'x', 'source_doc_id': 23})
chk('a document belonging to ANOTHER tenant cannot be cited',
    r.status_code == 400, str(r.status_code))
chk('...because an unchecked citation reads as evidence',
    'does not belong' in (r.get_json() or {}).get('error', ''))

r = cl.put('/api/lease-review/reviews/7/validation/resolve', headers=H,
           json={**BODY, 'tenant_id': 999, 'reason': 'x'})
chk('a tenant outside this review is refused', r.status_code == 400)

REASON = ('4th Amendment fixes base rent at 100,000 from 2026-01-01; the rent '
          'roll carries the pre-amendment figure.')
r = cl.put('/api/lease-review/reviews/7/validation/resolve', headers=H,
           json={**BODY, 'reason': REASON})
chk('a complete resolution is accepted', r.status_code == 200,
    str(r.status_code) + str(r.get_json())[:80])


section('A settled finding stops asking')
v = cl.get('/api/lease-review/reviews/7/validation', headers=H).get_json() or []
ar = [x for x in v if x['field'] == 'annual_rent' and x['tenant_id'] == 1]
chk('the validation row carries its resolution',
    bool(ar) and ar[0].get('resolution') is not None, str(ar[:1])[:120])
# `or {}` because a defect that empties the resolution must FAIL these checks,
# not raise AttributeError and take the rest of the run down with it -- the v490
# shape, where a crashing section hid the sections after it.
_res = (ar[0].get('resolution') or {}) if ar else {}
if ar:
    chk('...with the reason', _res.get('reason') == REASON)
    chk('...and the document it cites',
        _res.get('source_doc') == '4th Amendment.pdf',
        str(_res.get('source_doc')))
    chk('...and the screen is told which field settles this row',
        ar[0].get('resolvable_field') == 'annual_rent')
other = [x for x in v if x['field'] == 'annual_rent' and x['tenant_id'] == 2]
chk('another tenant\'s identical finding is NOT marked settled',
    bool(other) and other[0].get('resolution') is None, str(other[:1])[:100])

# A validation RE-RUN deletes and rebuilds every row, so a decision stored on one
# would vanish. This is the check that it lives somewhere else.
S.validate_rent_roll(eng, 7)
v2 = cl.get('/api/lease-review/reviews/7/validation', headers=H).get_json() or []
ar2 = [x for x in v2 if x['field'] == 'annual_rent' and x['tenant_id'] == 1]
chk('the reading SURVIVES a re-validation',
    bool(ar2) and ar2[0].get('resolution') is not None)

# lease_expiration is stored as lease_end, and rent_step_in_force is not a value
# at all -- both map, or the screen offers a control that cannot save.
r = cl.put('/api/lease-review/reviews/7/validation/resolve', headers=H,
           json={'tenant_id': 1, 'field': 'lease_expiration', 'value': '2035-12-31',
                 'reason': 'Confirmed against the 4th Amendment.',
                 'prior_value': '2035-12-31', 'source_doc_id': 22})
chk('lease_expiration resolves (it is stored as lease_end)', r.status_code == 200,
    str(r.get_json())[:90])
r = cl.put('/api/lease-review/reviews/7/validation/resolve', headers=H,
           json={'tenant_id': 2, 'field': 'rent_step_in_force', 'value': '100000',
                 'reason': 'Lease year 7 rent applies.', 'prior_value': '120000'})
chk('rent_step_in_force resolves (it is a decision about the annual rent)',
    r.status_code == 200, str(r.get_json())[:90])


section('The change report separates changes from confirmations')
rep = cl.get('/api/lease-review/reviews/7/rent-roll-changes',
             headers=H).get_json() or {}
chk('the rent change is reported', rep.get('change_count') == 2,
    str(rep.get('change_count')))
chk('...and the confirmation is NOT counted as a change',
    rep.get('confirmed_count') == 1, str(rep.get('confirmed_count')))
ch = [c for c in rep.get('changes', [])
      if c['tenant_id'] == 1 and c['field'] == 'annual_rent']
if ch:
    c0 = ch[0]
    chk('the change names the rent roll figure it replaced',
        c0.get('prior_value') == '120000', str(c0.get('prior_value')))
    chk('...the figure that applies', c0.get('value') == '100000')
    chk('...the difference', c0.get('difference') == -20000.0,
        str(c0.get('difference')))
    chk('...the reason', c0.get('reason') == REASON)
    chk('...the document cited', c0.get('source_doc') == '4th Amendment.pdf')
    chk('...and who decided it', c0.get('by') == 'chk')
else:
    chk('the change is in the report', False)

x = cl.get('/api/lease-review/reviews/7/rent-roll-changes/excel', headers=H)
chk('the report downloads as a workbook', x.status_code == 200, str(x.status_code))
chk('...and is a real xlsx', x.data[:2] == b'PK', str(x.data[:4]))


section('Undoing puts the finding back')
r = cl.delete('/api/lease-review/reviews/7/validation/resolve'
              '?tenant_id=1&field=annual_rent', headers=H)
chk('the resolution clears', r.status_code == 200, str(r.status_code))
v3 = cl.get('/api/lease-review/reviews/7/validation', headers=H).get_json() or []
ar3 = [x for x in v3 if x['field'] == 'annual_rent' and x['tenant_id'] == 1]
chk('...and the row is outstanding again',
    bool(ar3) and ar3[0].get('resolution') is None)
rep2 = cl.get('/api/lease-review/reviews/7/rent-roll-changes',
              headers=H).get_json() or {}
chk('...and it leaves the change report', rep2.get('change_count') == 1,
    str(rep2.get('change_count')))


section('The seller\'s document is loaded against THIS tenant')
chk('the per-tenant upload endpoint exists',
    '/api/lease-review/reviews/<int:review_id>/tenants/<int:tenant_id>/documents'
    in rules)
r = cl.post('/api/lease-review/reviews/7/tenants/999/documents', headers=H,
            data={}, content_type='multipart/form-data')
chk('a tenant outside the review is refused', r.status_code == 400,
    str(r.status_code))
# ONE ENGINE: the upload forces the tenant rather than matching on filename, and
# the extraction is the same function scoped by a parameter. A second, lighter
# path here would read the same PDF into different terms.
import inspect  # noqa: E402
chk('the uploader takes an explicit tenant',
    'force_tenant_id' in inspect.signature(S.upload_documents_to_review).parameters)
chk('...and the extraction is SCOPED, not reimplemented',
    'tenant_id' in inspect.signature(S.extract_all_documents).parameters)
src = inspect.getsource(LR_.upload_tenant_document)
chk('the upload runs the same extraction', 'extract_all_documents(' in src)
chk('...then consolidates the tenant', 'consolidate_tenant_extractions(' in src)
chk('...then re-validates, so the screen cannot look unchanged',
    'validate_rent_roll(' in src)


section('The screen formats by FIELD, and offers the control')
V = os.path.join('vue_app', 'src', 'views', 'LeaseReviewView.vue')
if os.path.exists(V):
    v = open(V, encoding='utf-8').read()
    chk('rents render as whole dollars with commas',
        "Math.round(n).toLocaleString('en-US')" in v)
    # The sign belongs outside the dollar sign -- the change report is full of
    # negative differences and "$-12,014" is not how a figure is written.
    chk('$/SF renders with cents', "Math.abs(n).toFixed(2)" in v)
    chk('a negative renders as -$X, not $-X',
        "(n < 0 ? '-$' : '$')" in v)
    chk('the format follows the field, not the column', 'function fmtByField(' in v)
    for f in ('annual_rent', 'monthly_rent'):
        chk('%s is a money field' % f, "'%s'," % f in v)
    chk('rent_per_sf is a $/SF field', "PSF_FIELDS = new Set(['rent_per_sf'" in v)
    # The full comparison table is where the raw values were rendered.
    chk('the comparison table uses it',
        'fmtByField(v.field, v.seller_value)' in v
        and 'fmtByField(v.field, v.lease_value)' in v)
    chk('there is a Settle control', 'openSettle(v)' in v and 'saveSettle' in v)
    chk('...that cannot save without a reason',
        "!settleReason.trim()" in v)
    chk('...and cites a document', 'settleDocId' in v)
    chk('a settled row shows its reading and can be undone',
        'clearSettle(v)' in v)
    chk('the change report is on the page', 'changes-box' in v
        and 'Changes against the rent roll' in v)
    chk('...and downloads', 'changesUrl()' in v)
    chk('the seller\'s document can be loaded from the finding',
        'uploadSettleDoc' in v)
else:
    print('  SKIP  Vue source not present')


print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
