"""Guardrail: a document is typed by its FILE NAME, and term-bearing ones are kept.

Jim, Sep 20 2026, after the measurement: "do both changes."

TWO CHANGES, AND EITHER ONE ALONE DOES DAMAGE.

  1. `classify_document` matched the whole stored PATH. Every production document
     sits under `Tenant Leases/...`, so the `lease` pattern matched the FOLDER and
     short-circuited: 409 of 530 typed `Original Lease`, only 77 with "lease" in
     the file name.

  2. Extraction was gated on `('Original Lease', 'Amendment')`. Fixing (1) alone
     pushes 328 documents out of that gate, and MEASURED on production that
     strips the rent commencement date from 16 of the 38 tenants that have one --
     15 of those sources being Commencement Letters, the document whose whole
     purpose is to state that date. So the gate widened to `is_term_bearing`,
     which excludes only what cannot carry terms.

So the checks below run in BOTH directions: the noise is excluded AND the
term-bearing documents are still admitted. A check written only for "COIs are
gone" is satisfied by dropping everything, which is the failure that would cost
16 tenants a date.

Run:  .venv\\Scripts\\python.exe scripts\\lease_doc_type_check.py
"""
import os
import sys
import json
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


section('The type comes from the file name, not the folder')
# Every one of these sits under a folder containing the word "lease". Before the
# fix all six came back `Original Lease`.
CASES = [
    ('Tenant Leases/HotWorx/COI/Hotworx_COI EXAMPLE.pdf', 'COI'),
    ('Tenant Leases/Sal/2023.12.13_SalonCentric-Option Letter.pdf',
     'Option Letter'),
    ('Tenant Leases/F/2022.05.24_Francis Hair Lounge-Move-In.pdf',
     'Move-In Notice'),
    ('Tenant Leases/T/2023.05.25_Tasty Sichuan-Landlord Consent.pdf',
     'Consent Letter'),
    ('Tenant Leases/W/Fourth Amendment.pdf', 'Amendment'),
    ('Tenant Leases/W/2021.03.01_Windsor-Commencement Letter.pdf',
     'Commencement Letter'),
]
for fn, want in CASES:
    got = S.classify_document(fn)
    chk('%-34s -> %s' % (fn.rsplit('/', 1)[-1][:34], want), got == want, got)

# A real lease still reads as one -- the fix must not over-correct.
chk('a genuine lease is still Original Lease',
    S.classify_document('Tenant Leases/W/Windsor Square Lease.pdf')
    == 'Original Lease')
# Windows separators reach this from a folder scan.
chk('a backslash path is split too',
    S.classify_document(r'Tenant Leases\\HotWorx\\COI\\x_COI.pdf') == 'COI')
# The fallback that would undo the whole fix.
chk('a name with no keyword does NOT fall back to the folder',
    S.classify_document('Tenant Leases/W/1987 Easement Agreement.pdf')
    == 'Other')


section('Term-bearing: what is kept, and what is not')
chk('a COI is not term-bearing', not S.is_term_bearing('COI'))
for t in ('Original Lease', 'Amendment', 'Commencement Letter', 'Option Letter',
          'Move-In Notice', 'Opening Notice', 'Estoppel', 'SNDA', 'Other'):
    chk('...%s IS kept' % t, S.is_term_bearing(t))
# Never drop what nobody anticipated: visibly wrong beats invisibly absent.
chk('an unknown type is kept', S.is_term_bearing('Something New'))
chk('a missing type is kept', S.is_term_bearing(None))
chk('only COI is excluded', S.NON_TERM_TYPES == frozenset({'COI'}),
    str(S.NON_TERM_TYPES))


section('Against a real database: the backfill and the consolidation')
DB = os.path.join(tempfile.gettempdir(), 'lease_dt.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name) "
                   "VALUES (1, 'Check')"))
    c.execute(text("INSERT INTO lease_tenants (id, review_id, tenant_name) "
                   "VALUES (7, 1, 'Check Tenant')"))
    # Stored the way the OLD classifier would have: everything Original Lease.
    seed = [
        (1, 'Tenant Leases/C/Check Tenant Lease.pdf',
         {'square_feet': 1000, 'rent_commencement': '2020-01-01'}),
        (2, 'Tenant Leases/C/First Amendment.pdf',
         {'square_feet': 1200}),
        (3, 'Tenant Leases/C/2021.03.01_Check-Commencement Letter.pdf',
         {'rent_commencement': '2021-03-01'}),
        (4, 'Tenant Leases/C/2025.01.01_COI-Check.pdf',
         {'rent_commencement': '2099-12-31', 'square_feet': 5}),
    ]
    for did, fn, terms in seed:
        c.execute(text(
            "INSERT INTO lease_documents (id, tenant_id, review_id, filename, "
            " doc_type, extraction_status, extraction_json) VALUES "
            "(:i,7,1,:f,'Original Lease','extracted',:j)"),
            {'i': did, 'f': fn, 'j': json.dumps(terms)})

with eng.connect() as c:
    before = dict(c.execute(text(
        "SELECT id, doc_type FROM lease_documents")).fetchall())
chk('the fixture starts mistyped, as production was',
    set(before.values()) == {'Original Lease'}, str(before))

res = S._reclassify_documents(eng)
chk('the backfill runs', res.get('checked') == 4, str(res))
chk('...and corrects exactly the three that were wrong',
    res.get('changed') == 3, str(res))
with eng.connect() as c:
    after = dict(c.execute(text(
        "SELECT id, doc_type FROM lease_documents")).fetchall())
chk('...the lease is still a lease', after[1] == 'Original Lease', after[1])
chk('...the amendment is an Amendment', after[2] == 'Amendment', after[2])
chk('...the commencement letter is one',
    after[3] == 'Commencement Letter', after[3])
chk('...and the COI is a COI', after[4] == 'COI', after[4])

again = S._reclassify_documents(eng)
chk('running it twice changes nothing', again.get('changed') == 0, str(again))

con = S.consolidate_tenant_extractions(eng, 7)
chk('the tenant still consolidates', isinstance(con, dict), str(type(con)))
applied = [os.path.basename(p) for p in (con or {}).get('_documents_applied', [])]
chk('the COI is NOT layered in', not any('COI' in a for a in applied),
    str(applied))
# BOTH DIRECTIONS. Dropping everything would satisfy the check above.
chk('...but the commencement letter IS',
    any('Commencement' in a for a in applied), str(applied))
chk('...and so are the lease and the amendment',
    any('Lease' in a for a in applied) and any('Amendment' in a
                                               for a in applied), str(applied))
chk('the commencement letter\'s date wins, not the COI\'s',
    (con or {}).get('rent_commencement') == '2021-03-01',
    str((con or {}).get('rent_commencement')))
chk('...and the amendment\'s square feet survive',
    (con or {}).get('square_feet') == 1200,
    str((con or {}).get('square_feet')))

with eng.connect() as c:
    rc = c.execute(text(
        "SELECT rent_commencement FROM lease_tenants WHERE id=7")).scalar()
chk('the tenant carries the right commencement date', rc == '2021-03-01', str(rc))

section('A later amendment is not overwritten by an older document')
# THE REGRESSION THIS FIX CAUSED, AND THE DIFF CAUGHT. order_lease_documents
# returned `originals + amendments + others`, so every non-amendment applied
# AFTER every amendment. It was invisible while the classifier typed nearly
# everything `Original Lease` and `others` was almost empty; correcting the
# classifier put 147 documents in there and three tenants' terms moved the wrong
# way at once -- Style Studio's expiry 2031 -> 2026, Green Zone's 2026 -> 2025,
# and Appliances 4 Less's suite from N625 to a misread "G".
from flask_app.services.lease_terms import order_lease_documents  # noqa: E402

STYLE = [
    {'id': 1, 'doc_type': 'Original Lease', 'doc_date': '2021-03-31'},
    {'id': 2, 'doc_type': 'Other', 'doc_date': '2021-04-07'},
    {'id': 3, 'doc_type': 'Other', 'doc_date': '2021-04-02'},
    {'id': 4, 'doc_type': 'Amendment', 'doc_date': '2026-02-17', 'ordinal': 1},
    {'id': 5, 'doc_type': 'COI', 'doc_date': '2025-08-02'},
]
ordered, _ = order_lease_documents(STYLE)
chk('the 2026 amendment has the last word, not a 2021 notice',
    ordered[-1]['id'] == 4, str([d['id'] for d in ordered]))
chk('...the base lease is still first', ordered[0]['id'] == 1)
chk('...and the 2021 documents are in date order between them',
    [d['id'] for d in ordered[1:3]] == [3, 2],
    str([d['id'] for d in ordered]))

# BOTH DIRECTIONS: sorting everything by date must not break the case this
# ordering was built for, where no document carries a date at all.
UND = [{'id': 9, 'doc_type': 'Original Lease'}] + [
    {'id': i, 'doc_type': 'Amendment', 'ordinal': k}
    for i, k in ((41, 4), (11, 1), (31, 3), (21, 2))]
ordered2, _ = order_lease_documents(UND)
chk('undated numbered amendments still apply 1,2,3,4',
    [d.get('ordinal') for d in ordered2[1:]] == [1, 2, 3, 4],
    str([d.get('ordinal') for d in ordered2]))
chk('...with the Fourth governing', ordered2[-1].get('ordinal') == 4)

# An undated document cannot claim to supersede a dated amendment.
MIX = [{'id': 1, 'doc_type': 'Original Lease', 'doc_date': '2020-01-01'},
       {'id': 2, 'doc_type': 'Amendment', 'doc_date': '2024-01-01', 'ordinal': 1},
       {'id': 3, 'doc_type': 'Other'}]
ordered3, _ = order_lease_documents(MIX)
chk('an undated document sorts after a dated amendment, not before it',
    [d['id'] for d in ordered3] == [1, 2, 3], str([d['id'] for d in ordered3]))

# On the same day the amendment wins: it is the document that changes the deal.
SAME = [{'id': 1, 'doc_type': 'Original Lease', 'doc_date': '2020-01-01'},
        {'id': 2, 'doc_type': 'Other', 'doc_date': '2024-01-01'},
        {'id': 3, 'doc_type': 'Amendment', 'doc_date': '2024-01-01', 'ordinal': 1}]
ordered4, _ = order_lease_documents(SAME)
chk('on an equal date the amendment is applied last',
    ordered4[-1]['id'] == 3, str([d['id'] for d in ordered4]))


print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
