"""Guardrail: co-tenancy and exclusive-use rows, and the analysts' review of them.

New business, Sep 29 2026, after correcting Market at Poplar by hand. Three things
this pins, each of which was wrong or absent before:

  A RE-READ REPLACES A DOCUMENT'S ROWS, IT DOES NOT ADD TO THEM. Exclusives were
  deduplicated on the model's wording of `restricted_use`, which moves between
  runs, so every re-read added copies: 291 rows for 35 tenant/document pairs, one
  lease contributing 25. Co-tenancy had the opposite fault -- skipped whenever the
  document already had a row -- so a changed answer could never land. Rows with no
  source_doc (the seller's spreadsheet) and other documents' rows are untouched.

  THE EXPORT SAYS WHO HOLDS A RESTRICTION. The Exclusive Use sheet carried only the
  restriction text, so a lease's exhibit disclosing OTHER tenants' exclusives read
  as that tenant's own -- the Firehouse Subs error the analysts corrected by hand.

  THE ANALYSTS' READING IS KEPT, AND SURVIVES A RE-READ. Per tenant and section,
  apart from the rows, never deleted by reset; a re-read after sign-off is MARKED.

No API calls. Run:  .venv\\Scripts\\python.exe scripts\\lease_clause_rows_check.py
"""
import io
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


DB = os.path.join(tempfile.gettempdir(), 'lease_clause_rows.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

LEASE = 'L/Firehouse/12.02.16-Firehouse Subs-Lease Agmt.pdf'
AMEND = 'L/Firehouse/2020-Firehouse-1st Amend.pdf'

READ_1 = {
    'cotenancy': {'has_clause': True, 'named_cotenants': ['Kroger'],
                  'trigger_threshold': 'Kroger closes', 'termination_right': True},
    'exclusive_use': [
        {'restricted_use': 'Sandwich restaurant', 'clause_role': 'holder',
         'restriction_text': 'Landlord shall not lease to a sandwich shop.'},
        {'restricted_use': "Sale of prepared pizza (CiCi's exclusive)",
         'clause_role': 'subject', 'source_section': 'Exhibit D',
         'restriction_text': "CiCi's Pizza: prepared pizza."},
        {'restricted_use': "Sale of prepared pizza (CiCi's exclusive)",
         'clause_role': 'subject', 'restriction_text': 'repeated in the same read'},
        {'restricted_use': 'none', 'restriction_text': 'none'},
        {'restricted_use': 'Prohibited uses', 'clause_role': 'holder',
         'radius_feet': '200', 'restriction_text': 'No bowling within 200 feet.'},
    ],
}
# The SAME lease read again: the model words everything differently.
READ_2 = {
    'cotenancy': {'has_clause': True, 'named_cotenants': ['Kroger', 'Target'],
                  'trigger_threshold': 'Kroger or Target closes'},
    'exclusive_use': [
        {'restricted_use': 'Restaurant with a sandwich menu', 'clause_role': 'holder',
         'restriction_text': 'No competing sandwich restaurant.'},
        {'restricted_use': "Prepared pizza (CICI'S PIZZA exclusive)",
         'clause_role': 'subject', 'restriction_text': "Exhibit D: CiCi's."},
    ],
}

with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name, rent_roll_date)"
                   " VALUES (3, 'Market at Poplar (fixture)', '2026-09-01')"))
    c.execute(text("UPDATE lease_reviews SET total_gla = 2600, total_annual_rent = 60000, total_tenants = 2 WHERE id = 3"))
    c.execute(text("INSERT INTO lease_reviews (id, property_name, rent_roll_date)"
                   " VALUES (4, 'Another Review', '2026-09-01')"))
    for tid, rid, name in ((166, 3, 'Firehouse Subs'), (159, 3, 'GNC'),
                           (900, 4, 'Somebody Else')):
        c.execute(text(
            "INSERT INTO lease_tenants (id, review_id, tenant_name, suite,"
            " square_feet, annual_rent, is_vacant, tenant_status) VALUES"
            " (:i,:r,:n,'S1',1300,30000,0,'active')"), {'i': tid, 'r': rid, 'n': name})
    for did, fn, j in ((1, LEASE, READ_2), (2, AMEND, {'exclusive_use': [
            {'restricted_use': 'Catering', 'clause_role': 'holder'}]})):
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename, doc_type,"
            " extraction_status, extraction_json) VALUES (:i,166,3,:f,'Original Lease',"
            "'extracted',:j)"), {'i': did, 'f': fn, 'j': json.dumps(j)})
    # A row from the seller's spreadsheet: no source_doc. Must never be touched.
    c.execute(text("INSERT INTO lease_exclusive_use (tenant_id, restriction_text)"
                   " VALUES (166, 'Seller schedule: sandwich exclusive')"))


def rows(tid=166, doc=None):
    with eng.connect() as c:
        q = "SELECT restricted_use, clause_role, radius_feet, source_doc FROM lease_exclusive_use WHERE tenant_id = :t"
        p = {'t': tid}
        if doc is not None:
            q += " AND source_doc = :d"
            p['d'] = doc
        return c.execute(text(q), p).fetchall()


def cot(tid=166):
    with eng.connect() as c:
        out = []
        for cid, trig, sd in c.execute(text(
                "SELECT id, trigger_threshold, source_doc FROM lease_cotenancy"
                " WHERE tenant_id = :t"), {'t': tid}).fetchall():
            refs = [r[0] for r in c.execute(text(
                "SELECT referenced_tenant_name FROM lease_cotenancy_refs"
                " WHERE cotenancy_id = :c"), {'c': cid})]
            out.append((trig, sd, sorted(refs)))
        return out


def write(doc, terms):
    with eng.begin() as c:
        return S._write_document_clause_rows(c, text, 166, 3, doc, terms)


section('a document read once')
w = write(LEASE, READ_1)
r = rows(doc=LEASE)
chk('three distinct restrictions written (negative skipped, in-read repeat once)',
    len(r) == 3 and w['exclusive_use'] == 3, r)
chk('holder vs subject kept', sorted(x[1] for x in r) == ['holder', 'holder', 'subject'])
chk('radius stored as a number', any(x[2] == 200.0 for x in r))
chk('co-tenancy written with its named co-tenant',
    cot() == [('Kroger closes', LEASE, ['Kroger'])], cot())

section('the same document read again, worded differently')
write(AMEND, {'exclusive_use': [{'restricted_use': 'Catering', 'clause_role': 'holder'}]})
w = write(LEASE, READ_2)
r = rows(doc=LEASE)
chk('the re-read REPLACES that document\'s rows (2, not 5)',
    len(r) == 2 and sorted(x[0] for x in r) == sorted(
        ["Prepared pizza (CICI'S PIZZA exclusive)", 'Restaurant with a sandwich menu']), r)
chk('another document\'s rows are untouched', len(rows(doc=AMEND)) == 1)
with eng.connect() as c:
    seller = c.execute(text("SELECT COUNT(*) FROM lease_exclusive_use WHERE tenant_id=166"
                            " AND source_doc IS NULL")).scalar()
chk('the seller-spreadsheet row (no source_doc) is untouched', seller == 1)
chk('co-tenancy REPLACED with the new answer (the old code kept the first)',
    cot() == [('Kroger or Target closes', LEASE, ['Kroger', 'Target'])], cot())
write(LEASE, {'cotenancy': {'has_clause': False}, 'exclusive_use': []})
chk('a re-read finding no clause leaves no co-tenancy row, and no orphan refs',
    cot() == [] and len(rows(doc=LEASE)) == 0)
with eng.connect() as c:
    orphans = c.execute(text("SELECT COUNT(*) FROM lease_cotenancy_refs WHERE cotenancy_id"
                             " NOT IN (SELECT id FROM lease_cotenancy)")).scalar()
chk('no orphaned co-tenancy refs', orphans == 0, orphans)

section('rebuilding from stored extractions clears old duplicates')
with eng.begin() as c:
    for _ in range(3):   # the state production is in: copies from earlier runs
        c.execute(text("INSERT INTO lease_exclusive_use (tenant_id, review_id,"
                       " restricted_use, clause_role, source_doc) VALUES"
                       " (166, 3, 'Sandwich shop, older wording', 'holder', :d)"), {'d': LEASE})
before = len(rows(doc=LEASE))
res = S.rebuild_clause_rows(eng, 3)
after = rows(doc=LEASE)
chk('duplicates gone: the lease now carries exactly its stored reading (2)',
    before == 3 and len(after) == 2, (before, after))
chk('rebuild reports what it did', res['documents'] == 2 and res['exclusive_use'] == 3, res)
res2 = S.rebuild_clause_rows(eng, 3)
chk('rebuild is idempotent', len(rows()) == len(rows()) and res2['exclusive_use'] == 3
    and len(rows(doc=LEASE)) == 2)
chk('rebuild leaves the seller row alone', len([x for x in rows() if x[3] is None]) == 1)

section('the analysts\' review')
S.save_clause_review(eng, 3, 166, 'exclusive_use', 'confirmed',
                     "Exhibit D is a disclosure; Firehouse holds sandwich only.", 'kh')
S.save_clause_review(eng, 3, 159, 'exclusive_use', 'confirmed',
                     'Vitamins/supplements exclusive, from the full lease chain.', 'kh')
S.save_clause_review(eng, 3, 159, 'cotenancy', 'confirmed', 'No co-tenancy provision.', 'kh')
for args, why in (((3, 166, 'exclusive_use', 'flagged', ''), 'a flag with no note'),
                  ((3, 166, 'radius', 'confirmed', 'x'), 'an unknown section'),
                  ((3, 166, 'exclusive_use', 'done', 'x'), 'an unknown status'),
                  ((3, 900, 'exclusive_use', 'confirmed', 'x'), "another review's tenant")):
    try:
        S.save_clause_review(eng, *args, 'kh')
        chk('refused: ' + why, False)
    except ValueError:
        chk('refused: ' + why, True)
rv = S.get_clause_reviews(eng, 3)
chk('reviews read back per tenant and section',
    rv[166]['exclusive_use']['status'] == 'confirmed' and rv[159]['cotenancy']['notes']
    == 'No co-tenancy provision.' and 900 not in rv)
n = S.mark_clause_reviews_reread(eng, 166)
rv = S.get_clause_reviews(eng, 3)
chk('a re-read marks a signed-off review', n == 1 and rv[166]['exclusive_use']['reread_at'])
chk('...and only that tenant', not rv[159]['exclusive_use'].get('reread_at'))
S.save_clause_review(eng, 3, 166, 'exclusive_use', 'confirmed', 'Re-checked.', 'kh')
chk('saving the review clears the mark',
    not S.get_clause_reviews(eng, 3)[166]['exclusive_use'].get('reread_at'))
S.reset_extraction_data(eng, 3)
chk('reset-extraction wipes the rows but KEEPS the reviews',
    len(rows()) == 0 and S.get_clause_reviews(eng, 3)[159]['cotenancy']['status'] == 'confirmed')

section('rerun marks the review; risk analysis carries it')
calls = []
_real = S.extract_all_documents
S.extract_all_documents = lambda *a, **k: calls.append(k.get('tenant_id'))
try:
    rep = S.rerun_tenant_extraction(eng, 3, 159)
finally:
    S.extract_all_documents = _real
chk('rerun_tenant_extraction marks the tenant\'s signed-off reviews',
    rep.get('clause_reviews_marked') == 2 and calls == [159], rep)
S.rebuild_clause_rows(eng, 3)
ra = S.get_risk_analysis_data(eng, 3)
chk('risk analysis returns the reviews keyed by tenant id',
    ra['clause_reviews'].get('159', {}).get('cotenancy', {}).get('status') == 'confirmed')
chk('exclusive rows carry their tenant id and radius',
    all(e.get('tenant_id') == 166 for e in ra['exclusive_use'])
    and 'radius_feet' in ra['exclusive_use'][0])

section('the workbook says who holds what')
import openpyxl  # noqa: E402
with eng.begin() as c:
    c.execute(text("UPDATE lease_documents SET extraction_json = :j WHERE id = 1"),
              {'j': json.dumps(READ_1)})
S.rebuild_clause_rows(eng, 3)
wb = openpyxl.load_workbook(io.BytesIO(S.generate_lease_review_excel(eng, 3)))
ws = wb['Exclusive Use']
head = [c.value for c in ws[3]]
chk('Exclusive Use sheet carries role, radius, carve-outs, source, review, notes',
    head == ['Tenant', 'Suite', 'Holds / Bound by', 'Restricted Use', 'Radius (ft)',
             'Carve-outs', 'Restriction Text', 'Source', 'Review', 'Notes / Flags'], head)
body = [[c.value for c in r] for r in ws.iter_rows(min_row=4) if r[0].value]
pizza = [r for r in body if r[3] and 'pizza' in str(r[3]).lower()]
chk("the disclosed CiCi's exclusive reads 'Bound by', not as Firehouse's own",
    pizza and pizza[0][2] == 'Bound by', pizza)
chk('the radius reaches the sheet', any(r[4] == 200.0 for r in body))
gnc = [r for r in body if r[0] == 'GNC']
chk('a reviewed tenant with no rows is still on the sheet, with its note',
    gnc and gnc[0][3] == '(no exclusive-use rows)' and 'Vitamins' in (gnc[0][9] or ''), gnc)
fh = [r for r in body if r[0] == 'Firehouse Subs' and r[2] == 'Holds']
chk('the review status reaches every row of the tenant, re-read marked',
    fh and fh[0][8] == 'confirmed', fh[:1])
ct = [c.value for c in wb['Co-Tenancy Detail'][4]]
chk('Co-Tenancy sheet carries Source, Review and Notes / Flags',
    ct[-3:] == ['Source', 'Review', 'Notes / Flags'], ct[-3:])

section('a failed reading is recorded as a failure, with why')
with eng.begin() as c:
    c.execute(text(
        "INSERT INTO lease_documents (id, tenant_id, review_id, filename, doc_type,"
        " extraction_status, extracted_text) VALUES (77,159,3,'1996.10.01-GNC-Lease.pdf',"
        "'Original Lease','text_extracted',:t)"), {'t': 'LEASE TEXT ' * 100})
_real_api = S.extract_lease_terms_via_api
S.extract_lease_terms_via_api = lambda *a, **k: {
    '_parse_error': True, '_failure_reason': 'the model returned no readable answer'}
try:
    S.extract_all_documents(eng, 3, api_key='k', tenant_id=159)
finally:
    S.extract_lease_terms_via_api = _real_api
with eng.connect() as c:
    st, why = c.execute(text("SELECT extraction_status, extraction_error FROM"
                             " lease_documents WHERE id = 77")).fetchone()
chk("status 'error' with the reason kept (it used to stay 'text_extracted', silent)",
    st == 'error' and 'no readable answer' in (why or ''), (st, why))
u = [d for d in S.unread_documents(eng, 3) if d['id'] == 77]
chk('the unread-documents list carries the reason', u and 'no readable' in (u[0]['error'] or ''))
S.extract_lease_terms_via_api = lambda *a, **k: {'square_feet': 1300}
try:
    with eng.begin() as c:
        c.execute(text("UPDATE lease_documents SET extraction_status = 'text_extracted' WHERE id = 77"))
    S.extract_all_documents(eng, 3, api_key='k', tenant_id=159)
finally:
    S.extract_lease_terms_via_api = _real_api
with eng.connect() as c:
    st, why = c.execute(text("SELECT extraction_status, extraction_error FROM"
                             " lease_documents WHERE id = 77")).fetchone()
chk('a later successful reading clears the old reason', st == 'extracted' and why is None,
    (st, why))

section('the New Business downside list reads the settled roster')
# scenario_service.get_risk_candidates read lease_tenants raw: a settled expiry
# never reached the downside scenario, and a vacated tenant was still offered.
from flask_app.services import scenario_service  # noqa: E402
with eng.begin() as c:
    c.execute(text("CREATE TABLE IF NOT EXISTS prospect_properties (id INTEGER PRIMARY KEY,"
                   " prospect_id INTEGER)"))
    c.execute(text("INSERT INTO prospect_properties (id, prospect_id) VALUES (1, 10)"))
    c.execute(text("UPDATE lease_reviews SET prospect_property_id = 1 WHERE id = 3"))
    c.execute(text("UPDATE lease_tenants SET is_material = 1, lease_end = '2027-12-31'"
                   " WHERE id = 166"))
    c.execute(text(
        "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, square_feet,"
        " annual_rent, is_vacant, tenant_status, is_material, lease_end) VALUES"
        " (170, 3, 'CiCi''s Pizza', 'S9', 2000, 40000, 0, 'vacated', 1, '2025-01-31')"))
S.ensure_resolution_table(eng)
with eng.begin() as c:
    c.execute(text("INSERT INTO lease_field_resolutions (tenant_id, field_name,"
                   " resolved_value, reason) VALUES (166, 'lease_end', '2031-12-31',"
                   " 'First Amendment extends the term')"))
cands = {x['tenant_id']: x for x in scenario_service.get_risk_candidates(eng, 10)}
chk("the analyst's settled expiry reaches the downside candidate (2031, not the rent roll's 2027)",
    cands.get(166, {}).get('lease_end') == '2031-12-31', cands.get(166))
chk('a tenant read as vacated is not offered as a downside candidate', 170 not in cands,
    sorted(cands))

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
sys.exit(1 if BAD else 0)
