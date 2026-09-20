"""Guardrail: a validation screen with rows in the database does not render blank.

Jim, Sep 20 2026: "Should we relocate the lease validation screen which exists in
the lease risk section to the lease validation area in the lease review? Right
now that page is blank."

NO -- the screen was complete and its data existed. THREE THINGS STACKED UP:

  1. Five rows in Market at Poplar carry the literal string 'NaN' as `lease_end`
     -- the building banner, two subtotal rows and two vacant suites, the same
     debris `v501` taught the roster to hide. `pd.to_datetime('NaN')` returns NaT
     WITHOUT raising, so the try/except never fires, `.year` is nan, and BOTH
     range comparisons are False because NaN never compares. The row reached
     `yearly[nan]` -> KeyError: nan -> HTTP 500 {"error":"nan"}.

  2. The validation fetch shared a `Promise.all` with that endpoint, and was the
     LAST assignment in the block, so the rejection meant it was never assigned.
     23 rows sat in the database while the page rendered empty.

  3. The catch logged "(expected for new reviews)", so the failure read as
     normal.

Windsor Square (175 rows, all four endpoints 200) is why this looked like a
missing screen rather than a broken one -- it worked there the whole time.

Run:  .venv\\Scripts\\python.exe scripts\\lease_validation_blank_check.py
"""
import os
import re
import sys
import tempfile
import logging

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from flask_app.services import lease_review_service as S  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


def section(t):
    print('\n--- ' + t)


section('The NaN that walked past the range guard')
# The mechanism, pinned so nobody "simplifies" the isna() check away.
y = pd.to_datetime('NaN').year
chk("pd.to_datetime('NaN') does NOT raise; its year is nan", pd.isna(y), str(y))
chk('...and BOTH range comparisons are False, so a range guard cannot catch it',
    (y < 2026) is False and (y > 2036) is False)


section('The histogram survives it, against a real database')
DB = os.path.join(tempfile.gettempdir(), 'lease_val.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

YEAR = pd.Timestamp.now().year
with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name) "
                   "VALUES (3, 'Check Centre')"))
    rows = [
        # a real tenant, active, expiring inside the window
        (1, 'Real Tenant', '%d-06-30' % (YEAR + 2), 0, 'active'),
        # the production debris: read as not-a-tenant, lease_end the STRING 'NaN'
        (2, 'Check @ Centre', 'NaN', 0, 'no_lease'),
        (3, 'Sub-total for Building: 925', 'NaN', 0, 'no_lease'),
        (4, 'Grand Total for Report', 'NaN', 0, 'no_lease'),
        # an ACTIVE tenant with an unparseable date -- the status filter alone
        # would not save this one, which is why the isna() guard is also needed
        (5, 'Active But Undated', 'NaN', 0, 'active'),
    ]
    for tid, nm, le, vac, st in rows:
        c.execute(text(
            "INSERT INTO lease_tenants (id, review_id, tenant_name, suite, "
            " square_feet, lease_end, annual_rent, rent_per_sf, is_vacant, "
            " tenant_status) VALUES (:i,3,:n,'S1',1000,:le,50000,50,:v,:s)"),
            {'i': tid, 'n': nm, 'le': le, 'v': vac, 's': st})

try:
    hist = S.get_expiration_histogram(eng, 3)
    raised = None
except Exception as ex:
    hist, raised = None, ex
chk('the histogram does not raise on the NaN rows', raised is None, str(raised))
chk('...and returns something usable', isinstance(hist, dict), str(type(hist)))

if isinstance(hist, dict):
    # `yearly_data` exposes counts, not names -- checked against the real
    # payload rather than assumed, after a first version of this asserted on a
    # `tenants` key the output does not carry.
    counted = sum(r.get('tenant_count') or 0
                  for r in (hist.get('yearly_data') or []))
    rent = sum(r.get('expiring_rent') or 0
               for r in (hist.get('yearly_data') or []))
    chk('the real tenant is counted', counted == 1, 'count=%s' % counted)
    chk('...with its rent', rent == 50000, str(rent))
    # BOTH DIRECTIONS: swallowing the exception and returning an empty histogram
    # would satisfy "the debris is excluded" on its own. Four rows went in and
    # exactly one is expected out.
    chk('...and the other FOUR rows are excluded, not merely survived',
        counted == 1 and len(rows) == 5, 'in=%d out=%d' % (len(rows), counted))

# The status filter must be the one every other consumer uses.
_src = open(os.path.join('flask_app', 'services', 'lease_review_service.py'),
            encoding='utf-8').read()
_fn = _src[_src.index('def get_expiration_histogram'):]
_fn = _fn[:_fn.index('\ndef ', 10)]
chk('the histogram filters on tenant_status, as the roster does',
    "COALESCE(tenant_status, 'active') = 'active'" in _fn)
chk('...and still guards the unparseable date', 'pd.isna(exp_year)' in _fn)


section('One failing panel cannot blank the others')
V = os.path.join('vue_app', 'src', 'views', 'LeaseReviewView.vue')
if os.path.exists(V):
    v = open(V, encoding='utf-8').read()
    # THE DEFECT: Promise.all rejects on the first failure and validation was
    # assigned last.
    chk('the four secondary panels use allSettled, not all',
        'Promise.allSettled([' in v and 'Promise.all([' not in v)
    chk('validation is assigned from its own settled result',
        "valRes.status === 'fulfilled'" in v)
    chk('...and so are scenarios and expirations',
        "scenRes.status === 'fulfilled'" in v
        and "expRes.status === 'fulfilled'" in v)
    # Shaping the cotenancy payload must not take scenarios/validation with it.
    chk('scenarios and validation are assigned OUTSIDE the cotenancy try',
        v.index("validation.value = valRes.status")
        > v.index('cotenancy.value = null'))
    # A blank panel has to say why.
    chk('a failed panel is reported on screen, not only to the console',
        'panelErrors' in v and 'panel-error' in v)
    # The dismissal itself, not the words -- they survive in the comment that
    # explains why it was removed.
    chk('...and the console.warn that dismissed it is gone',
        "console.warn('Secondary data load error" not in v)
else:
    print('  SKIP  Vue source not present')

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
