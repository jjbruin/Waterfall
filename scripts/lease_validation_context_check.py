"""Guardrail: the validation screen shows enough to resolve a finding.

Jim, Sep 20 2026: "we don't have enough information on the validation page to
resolve the issues... format the annual rent in dollars with commas but no
decimals and add the sf and $/sf of the lease per the rent roll... add links to
the scanned leases, options, and amendments in order of how they are applied...
I find it hard to believe that none of the rents rendered from the leases which
is what this page says right now."

HE WAS RIGHT TO DISBELIEVE IT. The leases had rendered: Market at Poplar holds
148 rent steps. `lease_reviews.rent_roll_date` was NULL, and no rent can be
placed "in force at an unknown date", so `step_in_force_at` returned nothing for
every tenant -- and `current_step` gates EVERY rent comparison. Measured: with a
date set, 22 of 23 tenants resolve and the comparisons are real (ATC Fitness
235,176 vs 235,176; Benjamin Moore 50,052 vs 38,038).

The field was settable only in the payload that CREATED a review and had no UI
at all, so a review that arrived without one could never be given one.

AND THE MESSAGE BLAMED THE WRONG THING -- "could not be determined from the
lease" -- which is what sent a reader hunting for missing lease data that was not
missing.

Run:  .venv\\Scripts\\python.exe scripts\\lease_validation_context_check.py
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

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


def section(t):
    print('\n--- ' + t)


DB = os.path.join(tempfile.gettempdir(), 'lease_ctx.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

TERMS = {
    'square_feet': 2500, 'lease_expiration': '2030-06-30',
    'rent_commencement': '2020-07-01',
    '_documents_applied': ['Leases/T/Original Lease.pdf',
                           'Leases/T/First Amendment.pdf'],
    '_governing_document': 'Leases/T/First Amendment.pdf',
}
with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name) "
                   "VALUES (9, 'Ctx Centre')"))
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, "
        " square_feet, annual_rent, rent_per_sf, lease_end, is_vacant, "
        " tenant_status, rent_commencement, extraction_json) VALUES "
        "(1,9,'Ctx Tenant','S1',2400,120000,50,'2030-06-30',0,'active',"
        " '2020-07-01',:j)"), {'j': json.dumps(TERMS)})
    # A row read as not-a-tenant must not appear in the context either.
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, "
        " square_feet, is_vacant, tenant_status) VALUES "
        "(2,9,'Grand Total for Report','',0,0,'no_lease')"))
    for did, fn, dt in ((10, 'Leases/T/Original Lease.pdf', 'Original Lease'),
                        (11, 'Leases/T/First Amendment.pdf', 'Amendment'),
                        (12, 'Leases/T/2025_COI.pdf', 'COI')):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename, "
            " doc_type, extraction_status, file_data) VALUES "
            "(:i,1,9,:f,:t,'extracted',:b)"),
            {'i': did, 'f': fn, 't': dt, 'b': b'%PDF-x'})
    # steps that CAN be dated, so the only thing missing is the review's date
    for ed, ar in (('2020-07-01', 100000), ('2025-07-01', 120000)):
        c.execute(text(
            "INSERT INTO lease_rent_steps (tenant_id, effective_date, "
            " annual_rent, monthly_rent, rent_per_sf) VALUES (1,:e,:a,:m,:p)"),
            {'e': ed, 'a': ar, 'm': ar / 12.0, 'p': ar / 2500.0})


section('With NO rent roll date, the reason names the REVIEW, not the lease')
S.validate_rent_roll(eng, 9)
with eng.connect() as c:
    rows = c.execute(text(
        "SELECT field_name, status, notes FROM lease_validation")).fetchall()
notes = ' '.join((r[2] or '') for r in rows)
chk('a finding is raised', len(rows) >= 1, str(len(rows)))
chk('...and it says the REVIEW has no rent roll date',
    'no rent roll date' in notes.lower(), notes[:110])
# The message that sent a reader hunting for lease data that was not missing.
# THIS ASSERTION PASSED VACUOUSLY at first -- it tested a phrase the message
# never contained, while the real prefix ("could not be determined from the
# lease") was still there and still wrong. Assert the phrase that IS used.
chk('...and does NOT blame the lease for it',
    'from the lease' not in notes, notes[:110])
chk('...and says the steps are present', 'steps are present' in notes.lower())


section('With the date set, the rents resolve')
with eng.begin() as c:
    c.execute(text("UPDATE lease_reviews SET rent_roll_date = '2026-07-20' "
                   "WHERE id = 9"))
S.validate_rent_roll(eng, 9)
with eng.connect() as c:
    got = c.execute(text(
        "SELECT field_name, seller_value, lease_value, status "
        "FROM lease_validation")).fetchall()
fields = {g[0] for g in got}
chk('the annual rent is now compared', 'annual_rent' in fields, str(sorted(fields)))
ar = [g for g in got if g[0] == 'annual_rent']
chk('...with BOTH sides present',
    bool(ar) and ar[0][1] is not None and ar[0][2] is not None, str(ar[:1]))
# BOTH DIRECTIONS: emitting the finding and no comparison would satisfy the
# first section on its own.
chk('...and the "no rent roll date" finding is gone',
    'rent_step_in_force' not in fields, str(sorted(fields)))


section('The context the screen needs')
os.environ.setdefault('DATABASE_URL', '')
from flask_app import create_app  # noqa: E402
app = create_app()
rules = {str(r.rule) for r in app.url_map.iter_rules()}
chk('the context endpoint exists',
    '/api/lease-review/reviews/<int:review_id>/validation-context' in rules)
chk('the rent roll date can be SET from the app',
    '/api/lease-review/reviews/<int:review_id>/rent-roll-date' in rules)

from datetime import datetime, timedelta, timezone  # noqa: E402
import jwt  # noqa: E402
import flask_app.db as DB_  # noqa: E402
import flask_app.api.lease_review as LR_  # noqa: E402
DB_.get_engine = lambda: eng
LR_.get_engine = lambda: eng
_tok = jwt.encode({'sub': '1', 'username': 'chk', 'role': 'admin',
                   'exp': datetime.now(timezone.utc) + timedelta(minutes=10)},
                  app.config['JWT_SECRET'], algorithm='HS256')
_r = app.test_client().get(
    '/api/lease-review/reviews/9/validation-context',
    headers={'Authorization': 'Bearer %s' % _tok})
chk('the context endpoint answers', _r.status_code == 200, str(_r.status_code))
ctx = _r.get_json() or {}
t = (ctx.get('tenants') or {}).get('1')
chk('the tenant is in the context', bool(t), str(list((ctx.get("tenants") or {}).keys())))
if t:
    chk('the RENT ROLL square feet are carried',
        t['rent_roll']['square_feet'] == 2400, str(t['rent_roll']))
    chk('...and the LEASE square feet, which differ',
        t['lease']['square_feet'] == 2500, str(t['lease']))
    names = [d['filename'].rsplit('/', 1)[-1] for d in t['documents']]
    chk('the documents come back IN APPLIED ORDER',
        names[:2] == ['Original Lease.pdf', 'First Amendment.pdf'], str(names))
    # A COI is a real document; showing it as excluded beats hiding it.
    coi = [d for d in t['documents'] if 'COI' in d['filename']]
    chk('...with the un-applied COI listed and marked, not hidden',
        bool(coi) and coi[0].get('applied') is False, str(coi))
    chk('...each linkable', all(d.get('id') for d in t['documents']))
chk('a row read as not-a-tenant is NOT in the context',
    '2' not in (ctx.get('tenants') or {}), str(list((ctx.get('tenants') or {}).keys())))


section('The screen formats it the way Jim asked')
V = os.path.join('vue_app', 'src', 'views', 'LeaseReviewView.vue')
if os.path.exists(V):
    v = open(V, encoding='utf-8').read()
    # Whole dollars, commas, no decimals -- and from v516 the sign sits outside
    # the dollar sign, so the assertion follows the rule rather than one spelling
    # of it.
    chk('money renders with commas and NO decimals',
        "Math.round(Math.abs(n)).toLocaleString('en-US')" in v)
    chk('$/SF is computed from rent over SF', 'function psf(' in v
        and '(r / f).toFixed(2)' in v)
    for col in ('RR SF', 'RR $/SF', 'Lease SF', 'Lease $/SF'):
        chk('the "%s" column is on the table' % col, col in v)
    chk('the documents column links them in order',
        'docUrl(d.id)' in v and 'di + 1' in v)
    chk('the rent roll date is settable on the page',
        'saveRentRollDate' in v and 'rent-roll-date' in v)
    chk('...and a missing one explains what it blocks',
        'no lease rent can be placed in force' in v)
    chk('setting it re-runs validation, so the screen cannot look unchanged',
        'await runValidation()' in v)
else:
    print('  SKIP  Vue source not present')


section('The upload ASKS for the date, so this cannot recur')
# Jim, Sep 20 2026: "To avoid this going forward we should request that date from
# the analyst during the upload process." Asked at the moment the analyst has the
# rent roll in front of them -- not enforced, because a date typed wrong is worse
# than one supplied a moment later, and the validation step now says when it is
# missing and lets it be set there.
_api = open(os.path.join('flask_app', 'api', 'lease_review.py'),
            encoding='utf-8').read()
_commit = _api[_api.index('def commit_rent_roll'):]
_commit = _commit[:_commit.index('@lease_review_bp', 10)]
chk('the commit accepts a rent roll date',
    "request.form.get('rent_roll_date')" in _commit)
chk('...parsed or refused, never guessed',
    'is not a date' in _commit and 'to_datetime' in _commit)
chk('...and stored on the review', 'SET rent_roll_date' in _commit)
chk('...and reported back, so the screen can confirm it landed',
    "'rent_roll_date': rrd or None" in _commit)

if os.path.exists(V):
    chk('the upload panel asks for it',
        'scanRentRollDate' in v and 'map-rrd' in v)
    chk('...sends it with the commit',
        "formData.append('rent_roll_date'" in v)
    chk('...says what a blank one costs',
        'every rent comparison on the' in v)
    # Requested, not blocking: the Import button must not depend on it.
    chk('...but does NOT block the import on it',
        'committing || unansweredPeriods.length > 0' in v)


print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
