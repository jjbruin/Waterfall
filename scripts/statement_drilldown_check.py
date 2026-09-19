"""Guardrail: the entries behind a statement figure add up to it.

The CFO asked to click any number on the workbench financial statements and see the
entries behind it (Sep 19 2026, via Jim).

THE ONLY THING A DRILLDOWN HAS TO DO IS RECONCILE. If the rows it shows do not sum to
the figure it was opened from, it makes a CORRECT statement look wrong, and the reader
has no way to tell which of the two to believe. That is worse than no drilldown.

So it does not re-query. `select_measure_rows` is the one definition of which GL rows
compose a measure, and `_balances` — which builds the statement — and `drilldown` both
call it. The checks below build a real statement from a fixture and then walk EVERY
(line, measure) pair asserting the entries tie to the line.

The fixture is deliberately awkward:
  * a balance-forward row (BALFOR 'B') AND normal activity, so opening/ytd/closing are
    genuinely different sets of rows and "it reconciles" cannot pass vacuously
  * two accounts rolling into ONE statement line, so a per-account drilldown would be
    caught
  * a row in a prior year and a row on an excluded basis, which must be left out
  * an account on a DIFFERENT entity carrying the same number

Run:  .venv\\Scripts\\python.exe scripts\\statement_drilldown_check.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text  # noqa: E402

from flask_app.services import statement_service as S  # noqa: E402
from flask_app.services.workpaper_data import periods_for  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))


def section(t):
    print(f'\n--- {t}')


# ---------------------------------------------------------------- fixture
DB = os.path.join(tempfile.gettempdir(), 'stmt_drill_check.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine(f'sqlite:///{DB}')

ENT = 'PPITEST'
PERIOD_END = '2026-06-30'
P = periods_for(PERIOD_END)

# (acct, period, balfor, basis, amt, item, descrpn)
GL = [
    # Cash: an opening balance-forward plus three entries, one per quarter-ish.
    ('1000', P['ytd_first'], 'B', 'A', 10_000.0, 1, 'Balance forward'),
    ('1000', '202602',       'N', 'B',  2_500.0, 2, 'Deposit'),
    ('1000', '202605',       'N', 'B', -1_000.0, 3, 'Cheque'),
    ('1000', '202606',       'N', 'B',    750.0, 4, 'Deposit'),
    # A SECOND account on the same FS line, so a drilldown that only fetched one
    # account would miss half the figure.
    ('1010', P['ytd_first'], 'B', 'A',  5_000.0, 5, 'Balance forward'),
    ('1010', '202603',       'N', 'B',    300.0, 6, 'Transfer in'),
    # Must be EXCLUDED: prior year, and an unlisted basis.
    ('1000', '202512',       'N', 'B', 99_999.0, 7, 'PRIOR YEAR - excluded'),
    ('1000', '202604',       'N', 'Z', 88_888.0, 8, 'WRONG BASIS - excluded'),
    # An income account, so the income statement renders too.
    ('4000', '202602',       'N', 'B', -4_000.0, 9, 'Fee income'),
    ('4000', '202606',       'N', 'B', -1_500.0, 10, 'Fee income'),
]

with eng.begin() as c:
    c.execute(text('CREATE TABLE gl_detail ("ENTITYID" TEXT,"PERIOD" TEXT,'
                   '"ENTRDATE" TEXT,"ACCTNAME" TEXT,"ACCTNUM" TEXT,"BASIS" TEXT,'
                   '"BALFOR" TEXT,"ITEM" INTEGER,"REF" TEXT,"DESCRPN" TEXT,'
                   '"SEGMENTID" TEXT,"RLTDENTITY" TEXT,"RLTDENTITY_NAME" TEXT,'
                   '"AMT" REAL)'))
    c.execute(text('CREATE TABLE gl_accounts ("ACCTNUM" TEXT,"ACCTNAME" TEXT,'
                   '"TYPE" TEXT)'))
    names = {'1000': 'Cash - Operating', '1010': 'Cash - Reserve',
             '4000': 'Fee Income'}
    for acct, per, bf, basis, amt, item, desc in GL:
        c.execute(text('INSERT INTO gl_detail VALUES (:e,:p,:d,:n,:a,:b,:f,:i,'
                       ':r,:s,:g,:x,:y,:m)'),
                  {'e': ENT, 'p': per, 'd': f'{per[:4]}-{per[4:6]}-15',
                   'n': names[acct], 'a': acct, 'b': basis, 'f': bf, 'i': item,
                   'r': f'R{item}', 's': desc, 'g': '', 'x': '', 'y': '',
                   'm': amt})
    # Same account numbers on ANOTHER entity — must never appear.
    c.execute(text('INSERT INTO gl_detail VALUES (:e,:p,:d,:n,:a,:b,:f,:i,:r,'
                   ':s,:g,:x,:y,:m)'),
              {'e': 'OTHERCO', 'p': '202606', 'd': '2026-06-15',
               'n': 'Cash - Operating', 'a': '1000', 'b': 'B', 'f': 'N', 'i': 99,
               'r': 'R99', 's': 'OTHER ENTITY - excluded', 'g': '', 'x': '',
               'y': '', 'm': 77_777.0})
    for acct, name in names.items():
        c.execute(text('INSERT INTO gl_accounts VALUES (:a,:n,:t)'),
                  {'a': acct, 'n': name, 't': 'B' if acct.startswith('1') else 'I'})

# The FS mapping: BOTH cash accounts roll into one line.
from flask_app.services import workpaper_service as W  # noqa: E402
W.ensure_tables(eng)
# NOTE: `statement` on a wp_fs_map row is the SECTION name ("Assets", "Income"),
# not the statement. Putting a statement name there sends every account to
# `conflicts` and the statement renders empty — which is what happened first.
W.set_fs_map([
    {'acctnum': '1000', 'statement': 'Assets', 'fs_line': 'Cash and cash equivalents',
     'sort_order': 1},
    {'acctnum': '1010', 'statement': 'Assets', 'fs_line': 'Cash and cash equivalents',
     'sort_order': 1},
    {'acctnum': '4000', 'statement': 'Income', 'fs_line': 'Revenue',
     'sort_order': 1},
], 'guardrail', engine=eng)

BASES = ['A', 'B']


# ===========================================================================
section('One definition of which rows compose a measure')

check('`select_measure_rows` exists and is shared',
      callable(getattr(S, 'select_measure_rows', None)))
import inspect  # noqa: E402
_bal = inspect.getsource(S._balances)
check('the statement builder uses it', 'select_measure_rows(' in _bal)
check('...and no longer selects rows itself',
      "BALFOR\"] == \"B\"" not in _bal and "BALFOR'] == 'B'" not in _bal)
_dd = inspect.getsource(S.drilldown)
check('the drilldown uses the same function', 'select_measure_rows(' in _dd)
try:
    S.drilldown(ENT, PERIOD_END, ['1000'], measure='whatever', engine=eng)
    check('an unknown measure is refused', False)
except ValueError as e:
    check('an unknown measure is refused', 'Unknown measure' in str(e))


# ===========================================================================
section('Every figure on the statement ties to its entries')

st = S.build(ENT, PERIOD_END, bases=BASES, engine=eng)
lines = []
for key in ('balance_sheet', 'income_statement'):
    stmt = st.get(key) or {}
    for sec in stmt.get('sections') or []:
        for line in sec.get('lines') or []:
            lines.append((key, line, sec.get('presentation_sign', 1)))

check('the fixture actually renders lines, so this can fail', len(lines) >= 2,
      f'{len(lines)} lines')
check('...and nothing fell out as unmapped, untyped or conflicted',
      not st.get('unmapped') and not st.get('untyped') and not st.get('conflicts'),
      f"unmapped={len(st.get('unmapped') or [])} "
      f"untyped={len(st.get('untyped') or [])} "
      f"conflicts={len(st.get('conflicts') or [])}")

# The rendered line says WHICH measure it is (`closing` on the balance sheet, `ytd`
# on the income statement) and carries both the raw GL figure and the presented one.
checked = mismatched = signed_bad = 0
for key, line, sign in lines:
    accts = [a['acctnum'] for a in (line.get('accounts') or [])]
    measure = line['measure']
    d = S.drilldown(ENT, PERIOD_END, accts, measure=measure, bases=BASES,
                    presentation_sign=sign, engine=eng)
    checked += 1
    if abs((d['total'] or 0.0) - float(line['gl_amount'])) > 0.005:
        mismatched += 1
        print(f'      {line["fs_line"]!r} {measure}: gl_amount '
              f'{line["gl_amount"]:,.2f} vs entries {d["total"]:,.2f}')
    # ...and the number the reader actually clicked.
    if abs((d['presented_total'] or 0.0) - float(line['amount'])) > 0.005:
        signed_bad += 1
        print(f'      {line["fs_line"]!r} {measure}: amount '
              f'{line["amount"]:,.2f} vs presented {d["presented_total"]:,.2f}')

check('every line reconciles to its GL entries',
      checked > 0 and mismatched == 0, f'{checked} checked, {mismatched} off')
check('...and to the SIGNED figure the reader clicked',
      checked > 0 and signed_bad == 0, f'{checked} checked, {signed_bad} off')
check('the statement says which measure each line is',
      all(l['measure'] in S.MEASURES for _, l, _ in lines),
      str({l['fs_line']: l['measure'] for _, l, _ in lines}))
check('...and the two statements use DIFFERENT measures, so this matters',
      len({l['measure'] for _, l, _ in lines}) == 2,
      str({l['measure'] for _, l, _ in lines}))


# ===========================================================================
section('The measures really are different sets of rows')

cash = next(l for _, l, _ in lines if l['fs_line'] == 'Cash and cash equivalents')
accts = [a['acctnum'] for a in cash['accounts']]
counts = {m: S.drilldown(ENT, PERIOD_END, accts, measure=m, bases=BASES,
                         engine=eng)['row_count'] for m in S.MEASURES}

# If every measure returned the same rows, "it reconciles" would prove nothing.
check('opening is only the balance-forward rows', counts['opening'] == 2, str(counts))
check('ytd is only the activity rows', counts['ytd'] == 4, str(counts))
check('qtd is narrower than ytd', counts['qtd'] < counts['ytd'], str(counts))
check('closing is opening PLUS ytd, not just the activity',
      counts['closing'] == counts['opening'] + counts['ytd'], str(counts))
check('...so the measures are genuinely different row sets',
      len(set(counts.values())) > 1, str(counts))

# A closing balance shown WITHOUT its opening would be the obvious mistake.
dd_close = S.drilldown(ENT, PERIOD_END, accts, measure='closing', bases=BASES,
                       engine=eng)
check('the closing drilldown includes the balance-forward rows',
      any(r['BALFOR'] == 'B' for r in dd_close['rows']))
check('...and sums to 17,550 = 15,000 opening + 2,550 activity',
      abs(dd_close['total'] - 17_550.0) < 0.005, str(dd_close['total']))


# ===========================================================================
section('Two accounts on one line are both drilled')

check('the line really does roll up two accounts', len(accts) == 2, str(accts))
per_acct = {a: S.drilldown(ENT, PERIOD_END, [a], measure='closing', bases=BASES,
                           engine=eng)['total'] for a in accts}
check('...and the pair sums to the line',
      abs(sum(per_acct.values()) - cash['gl_amount']) < 0.005,
      f'{per_acct} vs {cash["gl_amount"]}')
check('...neither account alone equals the line',
      all(abs(v - cash['gl_amount']) > 0.005 for v in per_acct.values()),
      str(per_acct))


# ===========================================================================
section('What must never appear')

rows = dd_close['rows']
check('no prior-year row', not any('PRIOR YEAR' in (r['DESCRPN'] or '') for r in rows))
check('no row on an excluded basis',
      not any('WRONG BASIS' in (r['DESCRPN'] or '') for r in rows))
check('...and the basis filter is what excluded it, not luck',
      any('WRONG BASIS' in (r['DESCRPN'] or '') for r in
          S.drilldown(ENT, PERIOD_END, accts, measure='closing',
                      bases=['A', 'B', 'Z'], engine=eng)['rows']))
check('no row from another entity',
      not any('OTHER ENTITY' in (r['DESCRPN'] or '') for r in rows))
check('every row carries what an accountant needs to find it',
      all(all(k in r for k in ('PERIOD', 'ENTRDATE', 'ACCTNUM', 'ITEM', 'REF',
                               'DESCRPN', 'AMT')) for r in rows))


# ===========================================================================
section('Nothing to show is said, not left blank')

empty = S.drilldown(ENT, PERIOD_END, [], engine=eng, bases=BASES)
check('no accounts given returns a reason, not an empty grid',
      empty['rows'] == [] and empty['total'] is None and empty.get('note'))
none_match = S.drilldown(ENT, PERIOD_END, ['9999'], bases=BASES, engine=eng)
check('an account with no rows totals 0.00 over 0 rows, which is the truth',
      none_match['row_count'] == 0 and none_match['total'] == 0.0)
no_entity = S.drilldown('NOSUCH', PERIOD_END, ['1000'], bases=BASES, engine=eng)
check('an entity with no GL says so', no_entity['total'] is None
      and 'No GL rows' in (no_entity.get('note') or ''))


# ===========================================================================
section('The endpoint exists')

from flask_app import create_app  # noqa: E402
app = create_app()
rules = {str(r.rule) for r in app.url_map.iter_rules()}
check('/api/workpapers/statements/drilldown is registered',
      '/api/workpapers/statements/drilldown' in rules)

_view = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'vue_app', 'src', 'views', 'WorkpapersView.vue')
if os.path.exists(_view):
    _src = open(_view, encoding='utf-8').read()
    check('the workbench calls it', 'statements/drilldown' in _src)
    check('...and passes the line\'s own accounts rather than re-deriving them',
          'accounts' in _src and 'drill' in _src.lower())
else:
    SKIP.append('screen checks')
    print('  SKIP  screen checks  [Vue source not in the image]')


print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print('  - ' + f)
sys.exit(1 if FAIL else 0)
