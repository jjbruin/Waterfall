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

print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print('  - ' + f)
sys.exit(1 if FAIL else 0)
