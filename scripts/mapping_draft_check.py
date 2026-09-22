"""Guardrail: a line mapping in progress survives leaving the screen.

Asset management reported losing mapping work: "How do you save mapping adjustments? I
don't see a save button, and when the page refreshed my mapping work was gone."

Both halves of that were true. The mapping lived only in browser memory between parse
and commit, and there was no endpoint that read one back -- so a refresh threw away
twenty minutes of judgement, and re-opening the screen after a SUCCESSFUL import showed
an empty page, which reads exactly the same as losing it. The one button on the screen
was labelled "Apply mapping to the Valuation column", which is why it was not found
when looking for a save button.

What these checks pin:
  * the parsed file is stored with the mapping, so resuming does not mean hunting down
    the spreadsheet again;
  * saving twice updates one row rather than accumulating drafts;
  * a committed mapping stays READABLE rather than being cleared on success;
  * sources and records do not bleed into each other;
  * the screen actually loads a draft, saves as the analyst works, and says so.

Run:  .venv\\Scripts\\python.exe scripts\\mapping_draft_check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy import text  # noqa: E402

from flask_app.services import line_mapping_service as L  # noqa: E402
from flask_app.services import valuation_service as V  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))


def skip(name, why):
    SKIP.append(name)
    print(f'  SKIP  {name}  [{why}]')


def section(t):
    print(f'\n--- {t}')


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A mapping the size of the one that was lost: 65 partner lines, 20 assigned.
PARSED = {
    'filename': 'partner budget.xlsx',
    'periods': ['2026-01-01', '2026-02-01'],
    'lines': [{'row': i, 'label': f'Line {i} - 40{i:02d}', 'total': 1000 + i}
              for i in range(65)],
}
MAPPING = {str(i): {'category': 'Rental Income', 'account': '4010', 'flip': False}
           for i in range(20)}


def _fixture():
    eng = sa.create_engine('sqlite:///:memory:')
    V.ensure_valuation_tables(eng)
    with eng.begin() as c:
        cyc = c.execute(text(
            "INSERT INTO valuation_cycles (year, as_of_date) "
            "VALUES (2026, '2026-12-31') RETURNING id")).scalar()
        r1 = c.execute(text(
            "INSERT INTO valuation_records (cycle_id, vcode) "
            "VALUES (:c, 'P0000010') RETURNING id"), {'c': cyc}).scalar()
        r2 = c.execute(text(
            "INSERT INTO valuation_records (cycle_id, vcode) "
            "VALUES (:c, 'P0000028') RETURNING id"), {'c': cyc}).scalar()
    return eng, r1, r2


section('The table exists on both dialects')

eng, rid, rid2 = _fixture()
with eng.connect() as c:
    tables = set(sa.inspect(c).get_table_names())
check('valuation_mapping_drafts is created by ensure_valuation_tables',
      'valuation_mapping_drafts' in tables)
V.ensure_valuation_tables(eng)      # idempotent
check('creating it twice is harmless', True)


section('Work survives leaving the screen')

check('nothing stored to begin with', L.get_draft(eng, rid, 'budget') is None)

L.save_draft(eng, rid, 'budget', 'partner budget.xlsx', PARSED, MAPPING, 'jday')
d = L.get_draft(eng, rid, 'budget')
check('a saved mapping comes back', d is not None)
check('...with the mapping intact',
      d['mapping']['0'] == MAPPING['0'], str(d['mapping'].get('0')))
# Without the parsed file, resuming means finding the spreadsheet again, which is most
# of the friction the draft removes.
check('...and the parsed FILE alongside it, so resuming needs no re-upload',
      len(d['parsed']['lines']) == 65 and d['parsed']['lines'][7]['label'] == 'Line 7 - 4007',
      f"{d['line_count']} lines")
check('...and the filename, so the screen can say what it resumed',
      d['filename'] == 'partner budget.xlsx')
check('...counted for the banner', (d['line_count'], d['mapped_count']) == (65, 20),
      f"{d['line_count']}/{d['mapped_count']}")
check('...and attributed', d['updated_by'] == 'jday')
check('a draft that has not been applied says so', d['status'] == 'draft')

m2 = dict(MAPPING)
m2['20'] = {'category': 'CAM', 'account': '4090', 'flip': False}
L.save_draft(eng, rid, 'budget', 'partner budget.xlsx', PARSED, m2, 'jday')
with eng.connect() as c:
    rows = c.execute(text("SELECT COUNT(*) FROM valuation_mapping_drafts")).scalar()
check('saving again updates in place rather than stacking drafts', rows == 1, str(rows))
check('...and the newer mapping wins',
      L.get_draft(eng, rid, 'budget')['mapped_count'] == 21)


section('A successful import does not empty the screen')

L.mark_draft_committed(eng, rid, 'budget')
d = L.get_draft(eng, rid, 'budget')
check('an applied mapping is still readable', d is not None and d['mapped_count'] == 21)
check('...and is marked as applied', d['status'] == 'committed', str(d['status']))
check('...with when', bool(d['committed_at']))
# The refusing direction alone is satisfied by never clearing anything.
L.save_draft(eng, rid, 'budget', 'partner budget.xlsx', PARSED, m2, 'jday')
check('editing after an import puts it back to a draft',
      L.get_draft(eng, rid, 'budget')['status'] == 'draft')


section('Drafts do not bleed across sources or records')

L.save_draft(eng, rid, 'argus', 'appraiser.xlsx', PARSED, {'0': {'category': 'CAM'}},
             'jday')
check('the two sources on one record are separate',
      L.get_draft(eng, rid, 'budget')['mapped_count'] == 21
      and L.get_draft(eng, rid, 'argus')['mapped_count'] == 1)
check('another record sees nothing', L.get_draft(eng, rid2, 'budget') is None)

L.discard_draft(eng, rid, 'budget')
check('starting over removes it', L.get_draft(eng, rid, 'budget') is None)
check('...and leaves the other source alone',
      L.get_draft(eng, rid, 'argus') is not None)

try:
    L.save_draft(eng, rid, 'nonsense', 'f.xlsx', PARSED, MAPPING, 'x')
    check('an unknown source is refused', False)
except ValueError as e:
    check('an unknown source is refused', 'Unknown source' in str(e))


section('The endpoints exist and are gated')

try:
    from flask_app import create_app
    app = create_app()
    # Flask registers one Rule per view function, so several rules share this path.
    # Keying them by path keeps only the last and reports two of the three missing.
    methods = set()
    for r in app.url_map.iter_rules():
        if str(r).endswith('/mapping/draft'):
            methods |= (r.methods - {'HEAD', 'OPTIONS'})
    check('GET, PUT and DELETE all present on /mapping/draft',
          {'GET', 'PUT', 'DELETE'} <= methods, str(sorted(methods)))
except Exception as e:
    skip('GET, PUT and DELETE all present on /mapping/draft', f'app would not build: {e}')


section('The screen uses it')

VIEW = os.path.join(ROOT, 'vue_app', 'src', 'components', 'common', 'LineMappingPanel.vue')
try:
    with open(VIEW, encoding='utf-8') as fh:
        vue = fh.read()
except OSError:
    skip('the panel loads a draft on open', 'Vue source not present')
    skip('the panel saves as the analyst works', 'Vue source not present')
    skip('the save button says it saves', 'Vue source not present')
else:
    check('the panel loads a draft on open',
          'loadDraft' in vue and 'onMounted' in vue)
    # Saving on a button is what failed; every edit already runs the check, so the save
    # rides along with it.
    check('the panel saves as the analyst works, not on a button',
          'scheduleDraftSave' in vue and 'scheduleDraftSave()' in vue.split('async function runCheck')[1][:200],
          'autosave not wired into runCheck')
    check('an upload is stored immediately, before any mapping is done',
          'await saveDraft()' in vue)
    check('the save button says it saves',
          "'Save and apply to the Valuation column'" in vue
          and "'Save and import into the Budget column'" in vue)
    check('the analyst is told their work is stored',
          'Mapping saved' in vue and 'Picked up where you left off' in vue)
    check('a failed save says so rather than pretending',
          'Could not save your mapping' in vue)
    check('there is a way to start over', 'discardDraft' in vue)

section('Reading a stated account number is not guessing')

import io as _io  # noqa: E402
import datetime as _dt  # noqa: E402
import openpyxl as _xl  # noqa: E402

from flask_app.services import budget_import_service as _B  # noqa: E402


def _jack_workbook(with_acct_col=True):
    """Jack's layout: Account Name in A, Account Number in B, months across."""
    wb = _xl.Workbook()
    ws = wb.active
    if with_acct_col:
        ws.append(['Account', 'Account', None, None, None, None])
        ws.append(['Name', 'Number'] + [_dt.datetime(2026, m, 1) for m in range(1, 5)])
    else:
        ws.append(['Account', None, None, None, None])
        ws.append(['Name'] + [_dt.datetime(2026, m, 1) for m in range(1, 5)])
    rows = [('Base Rent - 4010', 4010, 370871, 372268, 372546, 374607),
            ('CAM Reimb - 4090', 4090, 43934, 43934, 67927, 43934),
            ('Management Fees - 5040', 5040, 21875, 21875, 21875, 21875),
            ('Some Unnumbered Line', None, 100, 100, 100, 100),
            ('Total Revenue', None, 540179, 541576, 628127, 543915)]
    for r in rows:
        ws.append(list(r) if with_acct_col else [r[0]] + list(r[2:]))
    buf = _io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


_p = _B.parse_budget_workbook(_jack_workbook(), 'partner budget.xlsx')
_by = {l['label']: l for l in _p['lines']}
check('the account-number column is found by its header',
      _p['account_column'] == 1, str(_p['account_column']))
check('an account stated in its own column is read',
      _by['Base Rent - 4010']['stated_account'] == '4010')
check('an account stated on the end of the label is read',
      _B._account_from([], None, 'CAM Reimb - 4090') == '4090')
check('a line with no account number gets none, rather than a guess',
      _by['Some Unnumbered Line']['stated_account'] is None)
check('the count of stated accounts is reported', _p['stated_account_count'] == 3,
      str(_p['stated_account_count']))

# No account COLUMN, but the numbers are still on the labels -- which is the other
# half of Jack's file, and reading them is still reading.
_p2 = _B.parse_budget_workbook(_jack_workbook(with_acct_col=False), 'labels only.xlsx')
check('with no account column, the number on the label is still read',
      [l['stated_account'] for l in _p2['lines']][:3] == ['4010', '4090', '5040'],
      str([l['stated_account'] for l in _p2['lines']]))

# Nothing stated anywhere: nothing pre-filled. This is the case the "never guess" rule
# governs, and it must still hold.
_ws = _xl.Workbook().active
_wb3 = _xl.Workbook()
_ws3 = _wb3.active
_ws3.append(['Account', None, None, None, None])
_ws3.append(['Name'] + [_dt.datetime(2026, m, 1) for m in range(1, 5)])
for _r in [('Base Rent', 1, 2, 3, 4), ('CAM Reimbursement', 5, 6, 7, 8)]:
    _ws3.append(list(_r))
_b3 = _io.BytesIO()
_wb3.save(_b3)
_p3 = _B.parse_budget_workbook(_b3.getvalue(), 'nothing stated.xlsx')
check('a sheet stating no account numbers anywhere pre-fills nothing',
      all(l['stated_account'] is None for l in _p3['lines'])
      and _p3['stated_account_count'] == 0,
      str([l['stated_account'] for l in _p3['lines']]))
check('"Account Name" is not mistaken for an account-number column',
      not _B._ACCT_HEADER_RE.search('Account Name'))
check('a year or a suite number is not read as an account',
      _B._account_from([], None, 'Suite 101') is None
      and _B._account_from([], None, 'Rent 2026') is None)
# A month amount and an account number are both 3-6 digits. A January figure of
# 370,871 read as account 370871 is the kind of thing that maps silently and wrongly.
check('a month column is never taken for the account column, whatever its header says',
      _B._find_account_column(
          [['Account', 'Account'], ['Name', 'Number'], ['Base Rent', 370871]],
          1, 0, {1}) is None)


section('The chart of accounts in statement order')

_coa = _B.chart_of_accounts()
_titles = [x['title'] for x in _coa['sections']]
check('sections run revenue, opex, debt service, below the line, capex',
      _titles == ['Revenue', 'Operating expenses', 'Debt service',
                  'Other below the line', 'Capital expenditure'], str(_titles))
check('NOI is struck after operating expenses',
      next(x['subtotal_after'] for x in _coa['sections']
           if x['title'] == 'Operating expenses') == 'Net operating income (NOI)')
check('every category carries accounts',
      all(c['category'] and c['accounts']
          for x in _coa['sections'] for c in x['categories']))
check('the whole chart is there', _coa['account_count'] >= 70, str(_coa['account_count']))
check('capex is included even though it is not an income statement line',
      any(x['key'] == 'CAPEX' for x in _coa['sections']))


section('The partnership line is proposed, never injected')

_proposed = L.proposed_lines(None, 1, 'argus',
                             {'periods': ['2026-01-31'] * 12, 'lines': []})
check('an Argus import is offered the partnership line', len(_proposed) == 1)
check('...at the house default of $20,000 to 5130',
      _proposed[0]['account'] == '5130' and _proposed[0]['amount'] == 20000.0,
      str(_proposed[0]['amount']))
check('...priced over the months the file covers, not always a full year',
      L.proposed_lines(None, 1, 'argus',
                       {'periods': ['x'] * 6, 'lines': []})[0]['amount'] == 10000.0)
check('a file that already carries 5130 is not offered it again',
      L.proposed_lines(None, 1, 'argus',
                       {'periods': ['x'] * 12,
                        'lines': [{'label': 'Partnership - 5130'}]}) == [])
check('a budget is not offered it at all',
      L.proposed_lines(None, 1, 'budget', {'periods': [], 'lines': []}) == [])


section('The screen shows where each pre-fill came from')

try:
    with open(VIEW, encoding='utf-8') as fh:
        vue2 = fh.read()
except OSError:
    skip('provenance is shown per line', 'Vue source not present')
else:
    check('a stated account is labelled as read from the file',
          'from_file' in vue2 and 'from the file' in vue2)
    check('a prior mapping is labelled as such',
          'from_history' in vue2 and 'as mapped before' in vue2)
    check('a keyword match is still labelled a guess', 'keyword guess' in vue2)
    # The whole chart is offered UNCONDITIONALLY now. It used to sit behind a
    # `showFullCoa` tick box because the category narrowed the account list; with the
    # category derived FROM the account (Jack, Sep 22 2026) there is nothing left to
    # narrow it, so a tick box would leave most accounts unreachable.
    check('the whole chart is offered, not hidden behind a toggle',
          'Whole chart of accounts' in vue2 and 'showFullCoa' not in vue2)
    check('the account brings its own category', 'owning?.category' in vue2)
    # BOTH DIRECTIONS: the category must be shown and must NOT be selectable. A check
    # for "no dropdown" alone is satisfied by removing the column altogether, which
    # would hide where the figure lands.
    check('the category is displayed, read-only',
          'categoryOf(' in vue2 and 'setCategory(' not in vue2)
    check('unnumbered rows can be hidden, with a count',
          'onlyNumbered' in vue2 and 'unnumberedCount' in vue2)
    check('the chart of accounts opens beside the work', 'toggleCoa' in vue2)
    check('the partnership line is a tick box, off by default',
          'acceptedProposals' in vue2 and 'toggleProposal' in vue2)


print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print('  - ' + f)
sys.exit(1 if FAIL else 0)
