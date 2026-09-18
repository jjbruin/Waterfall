"""Guardrail: rent roll column mapping.

Every check here pins a defect that was live in the Market at Poplar import and that
fails SILENTLY -- a wrong figure in a roster reads exactly like a right one:

  * only the FIRST recovery column imported (CAM, 34% of the recovery; Insurance and
    Tax dropped, and Property Tax alone is larger than CAM);
  * an unqualified "Base Rent" read as annual, so every rent landed 12x low;
  * a subtotal row and a building banner imported as tenants, adding 459,444 phantom
    SF against a real 228,122;
  * pd.NaT reading as a real date, which is what let the banner row through;
  * "Non-Recoverable Utilities" classified as a recovery because it contains "recover".

The stacked-PDF checks pin the parse defects the real file found: a line split across
a 0.24pt rounding boundary, a tenant's charges dropped at a page break, and the
building totals block read as tenant charges.

Fixtures are built in memory. The two real Market at Poplar files live in OneDrive and
cannot ship, so the checks that need them SKIP with a reason rather than fail -- but
when they are present they assert the import ties to the file's OWN stated totals,
which is the only check that proves the whole chain rather than its parts.

Run:  .venv\\Scripts\\python.exe scripts\\rent_roll_mapping_check.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from flask_app.services import rent_roll_mapping as M  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))


def skip(name, why):
    SKIP.append(name)
    print(f'  SKIP  {name}  [{why}]')


def section(t):
    print(f'\n--- {t}')


# ---------------------------------------------------------------------------
# A columnar fixture shaped like the Market at Poplar export: a building banner,
# tenants, then the file's own subtotal and grand-total rows.
# ---------------------------------------------------------------------------

def _columnar_bytes():
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Rent Roll'
    ws.append(['Rent Roll'])
    ws.append([])
    ws.append(['Tenant', 'Floor #', 'Unit #', 'Unit Type', 'Lease Start', 'Lease Expiry',
               'Usable Area', 'Rentable Area', 'Base Rent', 'CAM', 'Insurance', 'Tax',
               'Total Charges', 'Security Deposit'])
    ws.append([])
    # Building banner: a name, and no figure or date anywhere.
    ws.append(['Market @ Poplar', None, None, None, 'Building ID: 925'])
    ws.append(['Alpha Retail', 1, '100-01', 'Retail',
               pd.Timestamp('2020-01-01'), pd.Timestamp('2030-12-31'),
               10000, 10000, 20000, 1000, 200, 800, 22000, 5000])
    ws.append(['Beta Shop', 1, '100-02', 'Retail',
               pd.Timestamp('2021-06-01'), pd.Timestamp('2028-05-31'),
               2000, 2000, 5000, 250, 50, 200, 5500, 1000])
    # A vacancy: no lease dates, but it reports its area -- must NOT read as a banner.
    ws.append(['Vacant', 1, '100-03', 'Retail', None, None, 1500, 1500,
               0, None, None, None, 0, None])
    ws.append(['Sub-total for Building: 925   Market @ Poplar', None, None, None,
               None, None, 13500, 13500, 25000, 1250, 250, 1000, 27500, 6000])
    ws.append(['Grand Total for Report', None, None, None, None, None,
               13500, 13500, 25000, 1250, 250, 1000, 27500, 6000])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


section('Proposals — what the header can and cannot tell us')

check('a bare "Base Rent" states no period, so none is proposed',
      M._propose_basis('Base Rent') is None)
check('a bare "CAM" states no period either', M._propose_basis('CAM') is None)
check('"Minimum Monthly Rent" states monthly and is read',
      M._propose_basis('Minimum Monthly Rent') == M.BASIS_MONTHLY)
check('"Annual Base Rent" states annual and is read',
      M._propose_basis('Annual Base Rent') == M.BASIS_ANNUAL)
check('"Rent per SF" with no period is still unanswered',
      M._propose_basis('Rent per SF') is None)
check('"Annual Rate/SF" is an annual per-SF rate',
      M._propose_basis('Annual Rate/SF') == M.BASIS_ANNUAL_PSF)

check('"Base Rent" proposes base rent, not ignore',
      M._propose_role('Base Rent', 'number') == M.ROLE_BASE_RENT)
for lbl in ('CAM', 'Insurance', 'Tax', 'CAM Recoveries', 'Property Tax Recoveries',
            'Insurance Recoveries', 'Expense Reimbursement'):
    check(f'"{lbl}" proposes recovery',
          M._propose_role(lbl, 'number') == M.ROLE_RECOVERY)
check('"Non-Recoverable Utilities" is NOT a recovery',
      M._propose_role('Non-Recoverable Utilities', 'number') != M.ROLE_RECOVERY,
      M._propose_role('Non-Recoverable Utilities', 'number'))
check('"Total Charges" is not a charge to import',
      M._propose_role('Total Charges', 'number') == M.ROLE_IGNORE)
check('"Market Rent" is not base rent',
      M._propose_role('Market Rent', 'number') != M.ROLE_BASE_RENT)
check('"Tax Parcel" is not a recovery',
      M._propose_role('Tax Parcel', 'text') != M.ROLE_RECOVERY)


section('Total and banner rows')

for nm in ('Total', 'Totals', 'Sub-total for Building: 925   Market @ Poplar',
           'Grand Total for Report', '** Total Charges', 'Subtotal', 'Total Area',
           'Totals for Building: Market @ Poplar'):
    check(f'"{nm}" reads as a total row', M.is_total_row(nm))
# The refusing direction alone is satisfied by dropping every row, and a rent roll
# full of real retail tenants is exactly where that bites.
for nm in ('Total Wine & More', 'Totally Nails', 'Patton Computers',
           'Total Body Fitness', 'Totem Lake Cleaners'):
    check(f'"{nm}" is a tenant, not a total', not M.is_total_row(nm))

na_row = pd.Series({'sf': None, 'rent': None, 'start': pd.NaT, 'end': pd.NaT})
check('pd.NaT does not count as a date (this is what let the banner through)',
      M._is_banner_row(na_row, ['sf', 'rent'], ['start', 'end']))
real_row = pd.Series({'sf': 1500, 'rent': None, 'start': pd.NaT, 'end': pd.NaT})
check('a vacancy reporting its area is not a banner',
      not M._is_banner_row(real_row, ['sf', 'rent'], ['start', 'end']))
dated = pd.Series({'sf': None, 'rent': None,
                   'start': pd.Timestamp('2020-01-01'), 'end': pd.NaT})
check('a row carrying a real date is not a banner',
      not M._is_banner_row(dated, ['sf', 'rent'], ['start', 'end']))


section('Unit conversion')

check('monthly amount -> annual $/SF', M._to_annual_psf(1200, M.BASIS_MONTHLY, 1000) == 14.4)
check('annual amount -> annual $/SF', M._to_annual_psf(12000, M.BASIS_ANNUAL, 1000) == 12.0)
check('annual per-SF passes through', M._to_annual_psf(12.0, M.BASIS_ANNUAL_PSF, 1000) == 12.0)
check('monthly per-SF x12', M._to_annual_psf(1.0, M.BASIS_MONTHLY_PSF, 1000) == 12.0)
check('a per-SF basis with no area returns None, never 0',
      M._to_annual_amount(12.0, M.BASIS_ANNUAL_PSF, 0) is None)
check('a monthly amount needs no area', M._to_annual_amount(1000, M.BASIS_MONTHLY, 0) == 12000)


section('Columnar scan + apply (in-memory fixture)')

blob = _columnar_bytes()
scan = M.scan(blob, 'fixture.xlsx')
cols = {c['label']: c for c in scan['columns']}

check('layout detected as columnar', scan['layout'] == 'columnar')
check('banner and both total rows excluded, 3 tenants left',
      scan['row_count'] == 3, f"row_count={scan['row_count']}")
check('3 rows reported as excluded, with reasons',
      len(scan['excluded_rows']) == 3, str([e['name'] for e in scan['excluded_rows']]))

recs = [k for k, v in scan['mapping']['roles'].items() if v == M.ROLE_RECOVERY]
check('ALL THREE recovery columns proposed, not just the first',
      sorted(recs) == ['CAM', 'Insurance', 'Tax'], str(sorted(recs)))
check('only one column proposed for square feet',
      sum(1 for v in scan['mapping']['roles'].values() if v == M.ROLE_SF) == 1)
check('Usable Area left unassigned in favour of Rentable Area',
      scan['mapping']['roles']['Rentable Area'] == M.ROLE_SF
      and scan['mapping']['roles']['Usable Area'] == M.ROLE_IGNORE)
check('Total Charges not proposed as a charge (it would double count)',
      scan['mapping']['roles']['Total Charges'] == M.ROLE_IGNORE)
check('the four periodic columns are reported unanswered',
      sorted(scan['unanswered']) == ['Base Rent', 'CAM', 'Insurance', 'Tax'],
      str(sorted(scan['unanswered'])))
check('a security deposit is not asked for a period',
      not cols['Security Deposit']['needs_basis'])

# --- refusing direction ---
mapping = {'roles': dict(scan['mapping']['roles']), 'bases': {}}
try:
    M.apply_mapping(blob, 'fixture.xlsx', mapping)
    check('an import with no period set is REFUSED', False)
except ValueError as e:
    check('an import with no period set is REFUSED', 'monthly or annual' in str(e).lower(),
          str(e)[:60])

bad = {'roles': dict(scan['mapping']['roles']), 'bases': dict.fromkeys(
    ['Base Rent', 'CAM', 'Insurance', 'Tax'], M.BASIS_MONTHLY)}
bad['roles']['Leased Area'] = M.ROLE_SF if 'Leased Area' in bad['roles'] else None
bad['roles'].pop(None, None)
bad2 = {'roles': {k: (M.ROLE_TENANT if k in ('Tenant', 'Unit #') else v)
                  for k, v in scan['mapping']['roles'].items()},
        'bases': bad['bases']}
try:
    M.apply_mapping(blob, 'fixture.xlsx', bad2)
    check('two columns mapped to tenant name is REFUSED', False)
except ValueError as e:
    check('two columns mapped to tenant name is REFUSED', 'more than one' in str(e))

# --- accepting direction: the rule must not be satisfied by refusing everything ---
good = {'roles': dict(scan['mapping']['roles']),
        'bases': dict.fromkeys(['Base Rent', 'CAM', 'Insurance', 'Tax'], M.BASIS_MONTHLY)}
df, rep = M.apply_mapping(blob, 'fixture.xlsx', good)

check('a complete mapping imports', len(df) == 3, f'{len(df)} rows')
check('the banner and both totals are excluded from the import',
      len(rep['excluded']) == 3, str(rep['excluded']))
check('total SF is the 3 real rows, not the doubled subtotals',
      df['square_feet'].sum() == 13500, str(df['square_feet'].sum()))

alpha = df[df.tenant_name == 'Alpha Retail'].iloc[0]
check('monthly base rent annualised: 20,000/mo -> 240,000/yr',
      alpha['annual_rent'] == 240000, str(alpha['annual_rent']))
check('rent per SF is 24.00, not 2.00',
      alpha['rent_per_sf_year'] == 24.0, str(alpha['rent_per_sf_year']))
check('recoveries SUM all three columns: (1000+200+800)*12/10000 = 2.40',
      abs(alpha['annual_recoveries_per_sf'] - 2.40) < 1e-9,
      str(alpha['annual_recoveries_per_sf']))
check('CAM alone would have given 1.20 — the old behaviour is gone',
      abs(alpha['annual_recoveries_per_sf'] - 1.20) > 1e-9)
check('all three recovery columns named in the report',
      sorted(rep['recovery_columns']) == ['CAM', 'Insurance', 'Tax'])
check('the deposit is taken as a balance, not annualised',
      alpha['security_deposit'] == 5000, str(alpha['security_deposit']))
check('the vacancy survives the import', bool(df['is_vacant'].any()))

annual = {'roles': dict(scan['mapping']['roles']),
          'bases': dict.fromkeys(['Base Rent', 'CAM', 'Insurance', 'Tax'], M.BASIS_ANNUAL)}
df_a, _ = M.apply_mapping(blob, 'fixture.xlsx', annual)
check('the period choice actually changes the answer (12x)',
      df_a[df_a.tenant_name == 'Alpha Retail'].iloc[0]['annual_rent'] == 20000)


section('Stacked layout helpers')

check('floor is stripped from a combined Floor/Unit cell',
      M._split_floor_unit('1 264 01') == '264 01')
check('a unit that is only digits is left alone',
      M._split_floor_unit('950-00') == '950-00')
check('a single-token cell keeps its value',
      M._split_floor_unit('PARCEL1A') == 'PARCEL1A')
check('page furniture is recognised',
      all(M._PAGE_FURNITURE_RE.search(t) for t in
          ['building: page 2', 'market @ poplar asof: 31-may-26',
           'master rent roll', 'byfloor/unit 12-jun-2026']))
check('a charge line is not mistaken for page furniture',
      not M._PAGE_FURNITURE_RE.search('cam recoveries $1,385.64 $1.39'))


section('Roster layout — a clipped name must still be readable')

VIEW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'vue_app', 'src', 'views', 'LeaseReviewView.vue')
try:
    with open(VIEW, encoding='utf-8') as fh:
        vue = fh.read()
except OSError:
    # The container image ships the built bundle, not the Vue source.
    skip('every tenant-name cell carries the full name on hover', 'Vue source not present')
    skip('the tenant name is clipped, not wrapped', 'Vue source not present')
else:
    import re as _re
    cells = _re.findall(r'<td class="tenant-name"[^>]*>', vue)
    check('tenant-name cells found in the roster', len(cells) >= 4, f'{len(cells)} cells')
    # Clipping hides the tail. A cell without a title shows a truncated name with no
    # way to read the rest -- worse than the wrapping it replaced.
    check('every tenant-name cell carries the full name on hover',
          all(':title=' in c for c in cells),
          str([c for c in cells if ':title=' not in c]))
    check('every tenant-name cell wraps its text in the clipping span',
          vue.count('class="tname"') == len(cells),
          f'{vue.count(chr(34)+"tname"+chr(34))} spans for {len(cells)} cells')
    check('the tenant name is clipped, not wrapped',
          'text-overflow: ellipsis' in vue and 'max-width: 190px' in vue)
    check('the clip is on the inner block, not the td (display:block on a td '
          'drops it out of the row)',
          _re.search(r'\.tenant-name \.tname \{[^}]*display: block', vue) is not None)


section('Against the real Market at Poplar files (skipped when absent)')

BASE = os.path.join(
    os.path.expanduser('~'), 'OneDrive - peaceablestreet.com',
    'Documents - Peaceable Street Capital', 'New Business',
    'Market at Poplar (Hendon) - Collierville, TN',
    'DD Uploaded by Hendon - Market at Poplar', '1 Rent Rolls')
XLSX = os.path.join(BASE, 'Market Rent Roll Susan Boswell.xlsx')
PDF = os.path.join(BASE, 'Rent Roll June 2026.pdf')


def _read(path):
    """Return the bytes, or a reason it could not be read.

    "Not present" and "open in Excel" are different situations and a skip line that
    conflates them sends the next person looking for a missing file.
    """
    if not os.path.exists(path):
        return None, 'file not present'
    try:
        with open(path, 'rb') as fh:
            return fh.read(), None
    except PermissionError:
        return None, 'file locked (open in Excel / syncing)'
    except OSError as e:
        return None, f'unreadable: {e.__class__.__name__}'


raw, why = _read(XLSX)
if raw is None:
    skip('columnar export ties to its own subtotal row', why)
else:
    sc = M.scan(raw, 'rr.xlsx')
    mp = sc['mapping']
    for k in ('Base Rent', 'CAM', 'Insurance', 'Tax'):
        mp['bases'][k] = M.BASIS_MONTHLY
    d, r = M.apply_mapping(raw, 'rr.xlsx', mp)
    rent = d['annual_rent'].sum()
    rec = (d['annual_recoveries_per_sf'] * d['square_feet']).sum()
    # The file's own "Sub-total for Building" row, x12 for the monthly basis.
    check('base rent ties to the file subtotal (259,323.99/mo x 12)',
          abs(rent - 259323.99 * 12) < 0.5, f'{rent:,.2f}')
    check('recoveries tie to CAM+Insurance+Tax (65,558.19/mo x 12)',
          abs(rec - 65558.19 * 12) < 0.5, f'{rec:,.2f}')
    check('SF ties to the file subtotal (229,722)',
          abs(d['square_feet'].sum() - 229722) < 1, f"{d['square_feet'].sum():,.0f}")
    check('35 tenant rows, phantom rows excluded', len(d) == 35, str(len(d)))

raw, why = _read(PDF)
if raw is None:
    skip('stacked PDF ties to its own building totals', why)
else:
    check('the stacked layout is detected', M._looks_stacked(raw))
    sc = M.scan(raw, 'rr.pdf')
    mp = sc['mapping']
    for e in sc['charges']:
        mp['bases'].setdefault(e['key'], None)
        if not mp['bases'][e['key']]:
            mp['bases'][e['key']] = M.BASIS_MONTHLY
    d, r = M.apply_mapping(raw, 'rr.pdf', mp)
    rent = d['annual_rent'].sum()
    rec = (d['annual_recoveries_per_sf'] * d['square_feet']).sum()
    check('base rent ties to the building total (253,794.11/mo x 12)',
          abs(rent - 253794.11 * 12) < 0.5, f'{rent:,.2f}')
    check('recoveries tie to the building total (65,215.85/mo x 12)',
          abs(rec - 65215.85 * 12) < 0.5, f'{rec:,.2f}')
    check('every tenant block reconciles to its own "* Tenant Total *"',
          not r['tie_out'], str(r['tie_out'][:3]))
    check('nothing is left unclassified as a stray identity line',
          sc['skipped']['other'] == 0, str(sc['skipped']))
    check('"Minimum Monthly Rent" states its own period, so it is not asked',
          'Minimum Monthly Rent' not in sc['unanswered'])
    check('35 tenant blocks, matching the columnar export', len(d) == 35, str(len(d)))
    check('the suite carries no floor prefix',
          all(not s.strip().startswith('1 ') for s in d['suite']),
          str([s for s in d['suite'] if s.strip().startswith('1 ')][:3]))

print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print('  - ' + f)
sys.exit(1 if FAIL else 0)
