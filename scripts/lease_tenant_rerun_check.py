"""Guardrail: adding a document to a scanned tenant re-reads that tenant's whole
set, and everything derived from it follows.

Jim, Sep 20 2026: "When we add a lease document to a previously scanned tenant's
record, we should rerun the extraction just on that tenant's set of leases. Ask
the user if any other files will be loaded before running, if yes, prompt the next
upload, if no, load the files and rerun that tenant's extract updating the
abstract, downstream lease risk analyses, and all validation records etc."

Three things this pins, each of which was wrong or absent before:

  THE WHOLE SET, NOT THE NEW FILE. The upload read only the document that had just
  arrived, so a tenant's terms ended up assembled from a mixture of prompt
  versions -- and the prompt moves; cam_fixed and escalation_pct arrived this
  week. Two tenants could then differ for no reason visible to anyone.

  THE ABSTRACT WAS FROZEN THE MOMENT ANYONE SAVED IT. `get_tenant_abstract`
  assembles from data only when NO section is stored, so an abstract touched once
  never saw another document again -- add an amendment extending the term and it
  went on stating the old expiry, silently. AN ANALYST'S WORDS ARE STILL NEVER
  OVERWRITTEN: a section a person wrote is marked and the newly assembled text is
  carried beside it.

  NOTHING RUNS UNTIL THE ANALYST SAYS THERE ARE NO MORE FILES. Uploading on choose
  started a re-read per drop: three files arriving one at a time meant three runs
  over the same tenant, each paying for the whole set again.

NO API CALLS. `extract_all_documents` is stubbed, because what is being checked is
which documents get reset, what runs afterwards and in what order -- not the
model's reading, which `lease_scan_extraction_check` covers.

Run:  .venv\\Scripts\\python.exe scripts\\lease_tenant_rerun_check.py
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


DB = os.path.join(tempfile.gettempdir(), 'lease_rerun.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

TERMS_OLD = {'square_feet': 2000, 'lease_expiration': '2027-12-31',
             'rent_commencement': '2020-01-01'}
TERMS_NEW = {'square_feet': 2000, 'lease_expiration': '2032-12-31',
             'rent_commencement': '2020-01-01'}

with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name, rent_roll_date)"
                   " VALUES (5, 'Rerun Centre', '2026-09-01')"))
    for tid, name in ((1, 'Re-read Me'), (2, 'Leave Me Alone')):
        c.execute(text(
            "INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
            " square_feet, annual_rent, rent_per_sf, lease_end, is_vacant,"
            " tenant_status, extraction_json, extraction_status) VALUES "
            "(:i,5,:n,'S1',2000,50000,25,'2027-12-31',0,'active',:j,'extracted')"),
            {'i': tid, 'n': name, 'j': json.dumps(TERMS_OLD)})
    # Tenant 1: an original lease already read, a COI (never term-bearing), and a
    # new amendment that has just arrived.
    for did, tid, fn, dt, st in (
            (11, 1, 'L/Re-read Me/Original Lease.pdf', 'Original Lease', 'extracted'),
            (12, 1, 'L/Re-read Me/2025_COI.pdf', 'COI', 'extracted'),
            (13, 1, 'L/Re-read Me/First Amendment.pdf', 'Amendment', 'pending'),
            (21, 2, 'L/Leave Me Alone/Original Lease.pdf', 'Original Lease',
             'extracted')):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename,"
            " doc_type, extraction_status, extraction_json) VALUES"
            " (:i,:t,5,:f,:d,:s,:j)"),
            {'i': did, 't': tid, 'f': fn, 'd': dt, 's': st,
             'j': json.dumps(TERMS_OLD)})
    c.execute(text(
        "INSERT INTO lease_rent_steps (tenant_id, effective_date, annual_rent,"
        " monthly_rent, rent_per_sf) VALUES (1,'2026-01-01',50000,4166.67,25)"))

# The abstract as it stands: one section this code wrote, one a person wrote.
S.save_abstract_sections(eng, 1, [
    {'section_key': 'term', 'section_title': 'Term',
     'content': 'Expires 2027-12-31', 'sort_order': 6},
    {'section_key': 'square_feet', 'section_title': 'Square Feet',
     'content': 'Analyst note: measured on site, 2,000 SF.', 'sort_order': 4},
], username='kh')
with eng.begin() as c:
    c.execute(text("UPDATE lease_abstract_sections SET updated_by = 'extraction'"
                   " WHERE tenant_id = 1 AND section_key = 'term'"))


# --- the stub: records what it was asked to do, calls nothing --------------
CALLS = {'extract': [], 'assemble': 0}
_real_extract = S.extract_all_documents
_real_assemble = S._assemble_abstract_from_data


def _stub_extract(engine, review_id, api_key=None, progress_callback=None,
                  tenant_id=None):
    with engine.connect() as conn:
        pend = [r[0] for r in conn.execute(text(
            "SELECT id FROM lease_documents WHERE review_id = :r"
            " AND extraction_status = 'pending'"
            " AND (:t IS NULL OR tenant_id = :t)"),
            {'r': review_id, 't': tenant_id}).fetchall()]
    CALLS['extract'].append({'review_id': review_id, 'tenant_id': tenant_id,
                             'pending': sorted(pend)})
    # Stand in for the model: every pending document now reads the NEW terms.
    with engine.begin() as conn:
        for did in pend:
            conn.execute(text(
                "UPDATE lease_documents SET extraction_status = 'extracted',"
                " extraction_json = :j WHERE id = :i"),
                {'j': json.dumps(TERMS_NEW), 'i': did})


def _stub_assemble(engine, tenant_id, review_id):
    CALLS['assemble'] += 1
    with engine.connect() as conn:
        ej = conn.execute(text(
            "SELECT extraction_json FROM lease_tenants WHERE id = :t"),
            {'t': tenant_id}).scalar()
    e = json.loads(ej) if isinstance(ej, str) else (ej or {})
    return {'term': {'content': 'Expires %s' % e.get('lease_expiration'),
                     'lease_ref': 'Sec 2'},
            'square_feet': {'content': '%s SF' % e.get('square_feet'),
                            'lease_ref': 'Sec 1'}}


S.extract_all_documents = _stub_extract
S._assemble_abstract_from_data = _stub_assemble


section('It re-reads the tenant\'s WHOLE set, not just what arrived')
report = S.rerun_tenant_extraction(eng, 5, 1)
chk('extraction ran once', len(CALLS['extract']) == 1, str(len(CALLS['extract'])))
chk('...scoped to this tenant', CALLS['extract'][0]['tenant_id'] == 1)
# 11 was already 'extracted' and must be read again; 12 is a COI and must not be;
# 21 belongs to another tenant. Reading only the new file would show [13] alone.
chk('...and every TERM-BEARING document of that tenant was reset',
    CALLS['extract'][0]['pending'] == [11, 13],
    str(CALLS['extract'][0]['pending']))
chk('a COI is left alone, as everywhere else',
    12 not in CALLS['extract'][0]['pending'])
with eng.connect() as c:
    other = c.execute(text(
        "SELECT extraction_status FROM lease_documents WHERE id = 21")).scalar()
chk('another tenant\'s documents are untouched', other == 'extracted', str(other))
chk('the report says what was read', report['documents_read'] == 2
    and report['documents_skipped'] == 1, str(report))


section('Everything derived from the leases follows')
with eng.connect() as c:
    ej = c.execute(text(
        "SELECT extraction_json FROM lease_tenants WHERE id = 1")).scalar()
terms = json.loads(ej) if isinstance(ej, str) else (ej or {})
chk('the tenant\'s consolidated terms are rebuilt',
    terms.get('lease_expiration') == '2032-12-31',
    str(terms.get('lease_expiration')))
with eng.connect() as c:
    vrows = c.execute(text(
        "SELECT COUNT(*) FROM lease_validation v JOIN lease_tenants t"
        " ON t.id = v.tenant_id WHERE t.review_id = 5")).scalar()
chk('the validation records are rebuilt', vrows > 0, str(vrows))
# The risk analysis needs no step of its own, and this is the check that says so
# rather than leaving it to hope: the histogram is computed from the resolved
# tenants at READ time, so the new expiry is there with nothing refreshing it.
# THE RISK ANALYSIS DOES NOT TAKE THE LEASE'S DATE BY ITSELF, AND MUST NOT.
# `lease_tenants.lease_end` is the RENT ROLL's figure and the extraction's
# `lease_expiration` is the LEASE's; the expiration histogram reads the former
# with analyst resolutions applied on top. Writing the lease's date into that
# column on every re-read would make the rent roll agree with the leases by
# construction, and no expiry mismatch could ever be reported again -- the exact
# failure v503 removed from the rent comparison ("a validation that always passes
# is worse than none"). So: the re-read surfaces the disagreement, and the
# analyst's reading is what moves the analysis.
with eng.connect() as c:
    rr_end = c.execute(text(
        "SELECT lease_end FROM lease_tenants WHERE id = 1")).scalar()
chk("the rent roll's own expiry is preserved, so the comparison survives",
    str(rr_end)[:10] == '2027-12-31', str(rr_end))
with eng.connect() as c:
    exp = c.execute(text(
        "SELECT seller_value, lease_value, status FROM lease_validation v"
        " WHERE v.tenant_id = 1 AND v.field_name = 'lease_expiration'")).fetchone()
chk('the re-read reports the expiry disagreement',
    exp is not None and exp[2] == 'mismatch', str(exp))
chk("...with the lease's new date on the lease side",
    exp is not None and str(exp[1])[:10] == '2032-12-31', str(exp and exp[1]))

hist_before = {str(y.get('year')): y for y in
               (S.get_resolved_expiration_histogram(eng, 5).get('yearly_data') or [])}
# 4,000 = both fixture tenants, which share the rent roll's 2027 expiry; only
# tenant 1's lease says 2032. Asserting 2,000 here was wrong about the fixture,
# not about the code.
chk('the analysis still shows the rent roll year until somebody reads it',
    (hist_before.get('2027') or {}).get('expiring_sf') == 4000,
    str((hist_before.get('2027') or {}).get('expiring_sf')))
chk('...and nothing in the lease year yet',
    (hist_before.get('2032') or {}).get('expiring_sf') in (0, None),
    str((hist_before.get('2032') or {}).get('expiring_sf')))

# Now the analyst settles it, which is the act that changes the analysis.
S.resolve_field(eng, 1, 'lease_end', '2032-12-31', resolved_by='kh',
                reason='First Amendment extends the term to 2032-12-31.',
                prior_value='2027-12-31')
hist_after = {str(y.get('year')): y for y in
              (S.get_resolved_expiration_histogram(eng, 5).get('yearly_data') or [])}
chk('settling the finding moves the expiration analysis',
    (hist_after.get('2032') or {}).get('expiring_sf') == 2000,
    str((hist_after.get('2032') or {}).get('expiring_sf')))
chk('...and takes THAT tenant off the old year, leaving the other one there',
    (hist_after.get('2027') or {}).get('expiring_sf') == 2000,
    str((hist_after.get('2027') or {}).get('expiring_sf')))


section('The abstract updates, but an analyst\'s words are never overwritten')
with eng.connect() as c:
    rows = {r[0]: (r[1], r[2], r[3]) for r in c.execute(text(
        "SELECT section_key, content, updated_by, proposed_content"
        " FROM lease_abstract_sections WHERE tenant_id = 1")).fetchall()}
chk('a section this code wrote is REFRESHED',
    rows['term'][0] == 'Expires 2032-12-31', str(rows['term'][0]))
chk('...and it is reported', 'term' in report['abstract_refreshed'],
    str(report['abstract_refreshed']))
# BOTH DIRECTIONS: refreshing everything would satisfy the check above and destroy
# the analyst's own sentence, which is the outcome this rule exists to prevent.
chk('a section a PERSON wrote is left exactly as it was',
    rows['square_feet'][0] == 'Analyst note: measured on site, 2,000 SF.',
    str(rows['square_feet'][0]))
chk('...but marked, so it cannot sit there quietly out of date',
    'square_feet' in report['abstract_flagged'], str(report['abstract_flagged']))
chk('...with what the leases now say kept beside it',
    rows['square_feet'][2] == '2000 SF', str(rows['square_feet'][2]))

abs_read = S.get_tenant_abstract(eng, 5, 1)
by_key = {s['section_key']: s for s in abs_read['sections']}
chk('the screen is told which section is stale', by_key['square_feet'].get('stale') is True)
chk('...and which is not', by_key['term'].get('stale') is False)
chk('...and is handed the proposed text',
    by_key['square_feet'].get('proposed_content') == '2000 SF')

# Saving is what settles it, whichever text the analyst kept.
S.save_abstract_sections(eng, 1, [
    {'section_key': 'square_feet', 'section_title': 'Square Feet',
     'content': '2,000 SF per the amendment.', 'sort_order': 4}], username='kh')
after = S.get_tenant_abstract(eng, 5, 1)
prem = {s['section_key']: s for s in after['sections']}['square_feet']
chk('saving clears the marker', prem.get('stale') is False, str(prem.get('stale')))
chk('...and the proposal with it', not prem.get('proposed_content'))


section('A tenant outside the review is refused')
try:
    S.rerun_tenant_extraction(eng, 5, 999)
    chk('an unknown tenant is refused', False)
except ValueError:
    chk('an unknown tenant is refused', True)

S.extract_all_documents = _real_extract
S._assemble_abstract_from_data = _real_assemble


section('The endpoints, and the screen that drives them')
os.environ.setdefault('DATABASE_URL', '')
from flask_app import create_app  # noqa: E402
app = create_app()
rules = {str(r.rule) for r in app.url_map.iter_rules()}
chk('a tenant can be re-read on its own',
    '/api/lease-review/reviews/<int:review_id>/tenants/<int:tenant_id>/reextract'
    in rules)
import inspect  # noqa: E402
import flask_app.api.lease_review as LR_  # noqa: E402
chk('the upload runs the tenant-wide re-read, not a single-document one',
    'rerun_tenant_extraction(' in inspect.getsource(LR_.upload_tenant_document))

V = os.path.join('vue_app', 'src', 'views', 'LeaseReviewView.vue')
if os.path.exists(V):
    v = open(V, encoding='utf-8').read()
    chk('files are STAGED rather than uploaded on choose',
        'stageSettleDocs' in v and 'stagedFiles' in v)
    # The question Jim asked for, and the two answers.
    chk('the analyst is asked whether more files are coming',
        'Any other files for this tenant?' in v)
    chk('...yes prompts the next upload', 'Yes - add more' in v)
    chk('...no loads them and re-reads the tenant',
        'loadStagedAndRerun' in v and 'and re-read this tenant' in v)
    # BOTH DIRECTIONS: staging is pointless if choosing a file still fires the run.
    chk('choosing a file does NOT start anything',
        'stageSettleDocs' in v and 'uploadSettleDoc' not in v)
    chk('a tenant can be re-read without adding a document',
        'rerunTenant' in v and '/reextract' in v)
else:
    print('  SKIP  Vue source not present')

A = os.path.join('vue_app', 'src', 'views', 'LeaseAbstractView.vue')
if os.path.exists(A):
    a = open(A, encoding='utf-8').read()
    chk('the abstract shows a stale section', 's.stale' in a
        and 'The leases have changed since this was written' in a)
    chk('...and offers the updated text rather than applying it',
        'takeProposed' in a)
else:
    print('  SKIP  Abstract view not present')


print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
