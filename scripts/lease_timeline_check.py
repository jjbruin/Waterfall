"""Guardrail: the lease timeline -- step 3 of the rent-roll plan.

Every case is one of new business's OWN worked examples from their rent-roll
specification (Sep 29 2026), so the check is their definition, not ours:

  §5/§6   the rent in force on the as-of date; history analysed, not displayed
  §7/§8   future steps, each ending the day before the next; the last on expiry
  §12     an option with its own rent steps
  §20     fair market option rent: no figure invented, flagged
  §28/29  Start is the lease start, not the date the current rent began
  §24     the checks -- expired-but-active, psf, sequence, undated steps

Plus `governing_steps`: a later schedule replaces earlier steps within its span,
and a single later rent with earlier steps after it is FLAGGED, not guessed.

Pure: no database. Run:  .venv\\Scripts\\python.exe scripts\\lease_timeline_check.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

from flask_app.services.lease_terms import governing_steps  # noqa: E402
from flask_app.services.lease_timeline import build_timeline  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % (detail,) if detail and not cond else ''))


def section(t):
    print('\n--- ' + t)


def step(d, psf=None, annual=None, doc=1, docdate='2019-12-01', src='Lease.pdf', sf=1000):
    return {'effective_date': d, 'annual_rent': annual if annual is not None else psf * sf,
            'rent_per_sf': psf, 'source_doc_id': doc, 'doc_date': docdate,
            'source_doc': src, 'effective_date_basis': 'stated'}


def codes(tl):
    return {f['code'] for f in tl['flags']}


section('§5/§6: the rent in force on the as-of date; history is not displayed')
steps = [step('2020-01-01', 20), step('2023-01-01', 21), step('2025-01-01', 22),
         step('2027-01-01', 23), step('2029-01-01', 24)]
terms = {'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31',
         '_remaining_options': [], '_options_summary': 'None'}
tl = build_timeline(terms, steps, 1000, '2026-09-01')
chk('current rent = $22.00 PSF', tl['current']['psf'] == 22.0, tl['current'])
chk('the 2020 and 2023 periods are not displayed; 2027 and 2029 are',
    [r['start'] for r in tl['rows'][1:]] == ['2027-01-01', '2029-01-01'],
    [r['start'] for r in tl['rows']])
chk('...the first future row is labelled "Rent Step Dates", the next blank',
    tl['rows'][1]['label'] == 'Rent Step Dates' and tl['rows'][2]['label'] is None)

section('§7/§8: continuous periods, the last ending on the expiration')
chk('each period ends the day before the next begins',
    [p['end'] for p in tl['term']] == ['2022-12-31', '2024-12-31', '2026-12-31',
                                        '2028-12-31', '2030-12-31'], [p['end'] for p in tl['term']])
chk('the final period ends on the expiration', tl['term'][-1]['end'] == '2030-12-31')
chk('no flags on a clean lease', not tl['flags'], tl['flags'])

section('§28/§29: Start is the lease start; amendments read, history hidden')
steps = [step('2020-01-01', 20, doc=1, docdate='2019-12-01', src='Lease.pdf'),
         step('2022-01-01', 22, doc=2, docdate='2021-12-01', src='Amendment 1.pdf'),
         step('2025-01-01', 24, doc=3, docdate='2024-12-01', src='Amendment 2.pdf'),
         step('2027-01-01', 26, doc=4, docdate='2026-06-01', src='Amendment 3.pdf')]
terms = {'lease_commencement': '2020-01-01', 'lease_expiration': '2032-12-31',
         '_remaining_options': [], '_options_summary': 'None'}
tl = build_timeline(terms, steps, 1000, '2026-09-01')
chk('Start is 2020-01-01, not the date the current rent began',
    tl['rows'][0]['start'] == '2020-01-01', tl['rows'][0])
chk('expiration 2032, current $24, one future step at $26',
    tl['rows'][0]['end'] == '2032-12-31' and tl['current']['psf'] == 24.0
    and [(r['start'], r['psf']) for r in tl['rows'][1:]] == [('2027-01-01', 26.0)],
    tl['rows'])

section('§12: an option with its own rent steps; §13: the option count')
opt = {'option_number': 1, 'term_years': 5, 'option_start': '2031-01-01',
       'option_end': '2035-12-31', 'rent_basis': 'fixed', 'rent_schedule': [
           {'period': 'Year 1', 'start': '2031-01-01', 'rent_psf': 30.00},
           {'period': 'Year 2', 'start': '2032-01-01', 'rent_psf': 30.60},
           {'period': 'Year 3', 'start': '2033-01-01', 'rent_psf': 31.21},
           {'period': 'Year 4', 'start': '2034-01-01', 'rent_psf': 31.84},
           {'period': 'Year 5', 'start': '2035-01-01', 'rent_psf': 32.47}]}
terms = {'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31',
         '_remaining_options': [opt], '_options_summary': '1 x 5 Years'}
tl = build_timeline(terms, [step('2020-01-01', 25)], 1000, '2026-09-01')
orows = [r for r in tl['rows'] if r['kind'] in ('option', 'option_step')]
chk('Option 1 row, then its rent steps, one per option year',
    [r['label'] for r in orows] == ['Option 1', 'Rent Step Dates', None, None, None]
    and [r['psf'] for r in orows] == [30.0, 30.6, 31.21, 31.84, 32.47],
    [(r['label'], r['psf']) for r in orows])
chk('the option year periods are continuous and end with the option',
    orows[0]['end'] == '2031-12-31' and orows[-1]['end'] == '2035-12-31',
    [(r['start'], r['end']) for r in orows])
chk('the primary row summarises the options', tl['rows'][0]['options'] == '1 x 5 Years')

section('§20: fair market option rent -- no figure, a flag')
terms['_remaining_options'] = [dict(opt, rent_basis='fmv', rent_schedule=[])]
tl = build_timeline(terms, [step('2020-01-01', 25)], 1000, '2026-09-01')
orow = [r for r in tl['rows'] if r['kind'] == 'option'][0]
chk('the option shows no rent', orow['annual_rent'] is None and orow['psf'] is None, orow)
chk('...and is flagged for review', 'option_rent_fmv' in codes(tl), codes(tl))

section('A stated percentage is compounded, and says so (§25: no assumed increases)')
terms['_remaining_options'] = [dict(opt, rent_schedule=[
    {'period': 'Option year 1', 'start': '2031-01-01', 'escalation_pct': 10}])]
tl = build_timeline(terms, [step('2020-01-01', 25)], 1000, '2026-09-01')
orow = [r for r in tl['rows'] if r['kind'] == 'option'][0]
chk('10% on the last contractual rent (25,000) = 27,500, marked derived',
    orow['annual_rent'] == 27500.0 and orow['derived'] is True, orow)

section('§24: the checks')
terms = {'lease_commencement': '2020-06-10', 'lease_expiration': '2025-05-31',
         '_remaining_options': [], '_options_summary': 'None'}
tl = build_timeline(terms, [step('2020-06-10', 20)], 1000, '2026-09-01')
chk('an active tenant whose documents ended the lease before the as-of date is flagged'
    ' (Muddy Paws)', 'expired_but_active' in codes(tl), codes(tl))
bad = step('2020-01-01', 20)
bad['annual_rent'] = 25000
tl = build_timeline({'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31'},
                    [bad], 1000, '2026-09-01')
chk('annual rent / SF disagreeing with a stated PSF is flagged', 'psf_mismatch' in codes(tl))
tl = build_timeline({'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31',
                     '_remaining_options': [{'option_start': '2032-01-01', 'term_years': 5}]},
                    [step('2020-01-01', 20)], 1000, '2026-09-01')
chk('an option not starting the day after expiry is flagged', 'option_sequence' in codes(tl))
chk('...its end is derived from its term, and says so',
    'option_dates_derived' in codes(tl) and tl['options'][0]['end'] == '2036-12-31',
    tl['options'][0])
undated = dict(step('2020-01-01', 20), effective_date=None)
tl = build_timeline({'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31'},
                    [step('2020-01-01', 20), undated], 1000, '2026-09-01')
chk('a step that cannot be dated is reported, not placed', 'undated_steps' in codes(tl))
tl = build_timeline({'lease_expiration': '2030-12-31', 'rent_commencement': '2020-03-01'},
                    [step('2020-03-01', 20)], 1000, '2026-09-01')
chk('no lease commencement -> rent commencement shown AND flagged',
    tl['start'] == '2020-03-01' and 'start_from_rent_commencement' in codes(tl))
tl = build_timeline({'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31'},
                    [], 1000, '2026-09-01', rent_roll_annual_rent=30000)
chk('no dated schedule -> the rent roll\'s figure, flagged, never silently',
    tl['current']['annual_rent'] == 30000 and 'current_rent_from_rent_roll' in codes(tl))
tl = build_timeline({'lease_commencement': '2020-01-01', 'lease_expiration': '2030-12-31'},
                    [step('2020-01-01', 20)], 1000, '2026-09-01', settled_annual_rent=21000)
chk("an analyst's settled rent governs, and the difference from the lease is flagged",
    tl['current']['annual_rent'] == 21000 and 'settled_differs' in codes(tl))

section('governing_steps: a later schedule replaces earlier steps within its span')
g, notes = governing_steps([
    step('2020-01-01', 20, doc=1, docdate='2019-12-01'),
    step('2025-01-01', 22, doc=1, docdate='2019-12-01'),
    step('2027-01-01', 23, doc=1, docdate='2019-12-01'),
    step('2024-06-01', 21.5, doc=2, docdate='2024-05-01'),
    step('2026-06-01', 22.5, doc=2, docdate='2024-05-01')])
chk("the amendment's 2024-2026 schedule replaces the lease's 2025 step; 2020 and 2027 stay",
    [(s['effective_date'], s['rent_per_sf']) for s in g] ==
    [('2020-01-01', 20), ('2024-06-01', 21.5), ('2026-06-01', 22.5), ('2027-01-01', 23)],
    [(s['effective_date'], s['rent_per_sf']) for s in g])
chk('...no ambiguity flag: the amendment stated a schedule', not notes, notes)
g, notes = governing_steps([
    step('2020-01-01', 20, doc=1, docdate='2019-12-01'),
    step('2027-01-01', 23, doc=1, docdate='2019-12-01'),
    step('2024-01-01', 25, doc=2, docdate='2023-12-01', src='Amendment.pdf')])
chk('a single later rent with an earlier step after it keeps the step AND flags it',
    len(g) == 3 and notes and 'confirm' in notes[0], notes)
g, notes = governing_steps([
    step('2020-01-01', 20, doc=1, docdate='2019-12-01'),
    step('2020-01-01', 20, doc=2, docdate='2020-02-01', src='Commencement Letter.pdf'),
    step('2025-01-01', 22, doc=1, docdate='2019-12-01')])
chk("a commencement letter restating year-one rent does NOT wipe the lease's schedule",
    [s['effective_date'] for s in g] == ['2020-01-01', '2025-01-01'], [s['effective_date'] for s in g])

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
sys.exit(1 if BAD else 0)
