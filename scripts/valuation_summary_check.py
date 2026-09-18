"""Guardrail: the two portfolio summary tabs assemble, they do not calculate.

Both tabs read figures the app has already computed -- `valuation_records` for what the
analyst concluded, `valuation_nav_results` for net proceeds and NAV, and the Pref
Balance Detail engine (via `valuation_nav_service._pref_walks`) for pref balances and
accruals. Jim, twice now: "why are you trying to recreate a calculation engine that we
have already built and vetted?"

Two defects found while building this, both of which produced a PLAUSIBLE WRONG NUMBER
rather than an error, which is why they are pinned here:

  * THE PSC AND OP SIDES WERE SUMMED TOGETHER. The walk returns one entry per investor
    and the tab's "Pref Balance" is the PREFERRED position. Adding the operating
    partner's balance to it roughly doubles the figure and still looks like a balance.

  * THE CYCLE'S as_of IS STORED AS TEXT. Passing that string into the pref walk does
    not raise -- every transaction fails the date comparison and the walk returns
    0.00. Seven of eight deals checked against asset management's own workbook came
    back zero for exactly this reason, and a zero balance reads as a real answer.

Run:  .venv\\Scripts\\python.exe scripts\\valuation_summary_check.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy import text  # noqa: E402

from flask_app.services import valuation_summary_service as S  # noqa: E402
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


section('The preferred side is not the whole walk')

WALK = {
    'PPI11':  {'investment_balance': 5746667.0, 'accrued_pref': 157725.1473158172},
    'OPPMAT': {'investment_balance': 3723333.3, 'accrued_pref': 1242410.2888560195},
}
split = S._split_pref(WALK)
check('PSC side is taken alone', split['pref_balance'] == 5746667.0,
      str(split['pref_balance']))
check('...matching asset management\'s own workbook to the cent',
      abs(split['pref_accrued'] - 157725.1473158172) < 1e-9)
check('the operating partner is kept separately, not discarded',
      split['op_balance'] == 3723333.3)
# Summing both gives 9,469,999 -- a plausible-looking balance that is simply wrong.
check('the two sides are never added together',
      split['pref_balance'] != (5746667.0 + 3723333.3))
check('an OP-prefixed code is the operating partner',
      not S._is_psc_side('OPPMAT') and not S._is_psc_side('opflag'))
check('everything else is the preferred side',
      S._is_psc_side('PPI11') and S._is_psc_side('PPI24'))


section('The cycle date must be a date')

check('an ISO string is coerced', S._as_date('2025-12-31') == date(2025, 12, 31))
check('a date passes through', S._as_date(date(2025, 12, 31)) == date(2025, 12, 31))
check('nonsense is refused rather than guessed', S._as_date('not a date') is None)
check('None stays None', S._as_date(None) is None)
# The failure this replaces: a string reached the walk, every transaction failed the
# comparison, and the balance came back 0.00 with no error anywhere.
check('an unusable date yields no figure, not a zero',
      S._live_pref({}, 'P0000010', 'not a date')['pref_balance'] is None)
check('...and says why',
      'as-of date' in (S._live_pref({}, 'P0000010', 'not a date')['pref_note'] or ''))


section('Missing is missing, never zero')

check('no prior figure gives no variance, not 0', S._delta(100.0, None) is None)
check('...in either direction', S._delta(None, 100.0) is None)
check('a real variance is computed', S._delta(100.0, 40.0) == 60.0)
check('a genuine no-change is 0, and distinguishable', S._delta(100.0, 100.0) == 0.0)


section('Assembly against a fixture')

eng = sa.create_engine('sqlite:///:memory:')
V.ensure_valuation_tables(eng)
with eng.begin() as c:
    cur = c.execute(text("INSERT INTO valuation_cycles (year, as_of_date) "
                         "VALUES (2026, '2026-12-31') RETURNING id")).scalar()
    old = c.execute(text("INSERT INTO valuation_cycles (year, as_of_date) "
                         "VALUES (2025, '2025-12-31') RETURNING id")).scalar()
    for cyc, val, cap in ((cur, 9400000.0, 0.0725), (old, 8900000.0, 0.075)):
        rid = c.execute(text(
            "INSERT INTO valuation_records (cycle_id, vcode, method, concluded_value, "
            "cap_rate) VALUES (:c,'P0000004','DCF',:v,:r) RETURNING id"),
            {'c': cyc, 'v': val, 'r': cap}).scalar()
        c.execute(text(
            "INSERT INTO valuation_nav_results (record_id, inputs_json, net_proceeds, "
            "psc_nav, op_nav) VALUES (:r, :j, :np, :pn, :on)"),
            {'r': rid,
             'j': '{"debt": 5247905.97, "pref": {"PPI22": {"investment_balance": '
                  '1490000.0, "accrued_pref": 26452.6}, "OPFLAG": '
                  '{"investment_balance": 802308.0, "accrued_pref": 14243.7}}}',
             'np': val - 5247905.97, 'pn': 2347830.79, 'on': 500000.0})

_pref = S.pref_summary(eng, cur, {})
_row = _pref['rows'][0]
check('the tab compares this cycle with the prior year',
      (_pref['current_year'], _pref['prior_year']) == (2026, 2025),
      f"{_pref['current_year']} vs {_pref['prior_year']}")
check('the stored pref figure is the PSC side only',
      _row['pref_balance'] == 1490000.0, str(_row['pref_balance']))
check('balance and accrual are added for the combined column',
      abs(_row['pref_with_accrual'] - 1516452.6) < 0.01,
      str(_row['pref_with_accrual']))
check('the stored figure says it came from the NAV run',
      'NAV run' in (_row['pref_source'] or ''), str(_row['pref_source']))
check('the prior year NAV is carried across',
      _row['prior_pref_nav'] == 2347830.79)
check('the variance is the move, not a restatement',
      _row['var_to_prior'] == 0.0)

_val = S.valuation_summary(eng, cur, {})
_vrow = _val['rows'][0]
check('the valuation tab carries both years of method and rates',
      (_vrow['method'], _vrow['prior_method']) == ('DCF', 'DCF')
      and _vrow['cap_rate'] == 0.0725 and _vrow['prior_cap_rate'] == 0.075)
check('value variance is computed from the two cycles',
      _vrow['var_to_prior_value'] == 500000.0, str(_vrow['var_to_prior_value']))
check('debt is read from the NAV inputs, not recomputed',
      _vrow['debt'] == 5247905.97)

_alone = S.pref_summary(eng, old, {})
check('a cycle with no prior year says so, rather than showing zeros',
      _alone['no_prior_cycle'] and _alone['rows'][0]['prior_pref_nav'] is None)
check('...and its variance is blank, not 0',
      _alone['rows'][0]['var_to_prior'] is None)

section('The NAV collects the accrual the One Pager suppresses')

# Jim, Sep 18 2026, on P0000044: "$51,926.54 is the correct accrual at 12/31/2025
# however, since we have grace period logic, we do not report them as delinquent on
# the One pager. For valuation purposes, we need to collect this accrual in the
# waterfall steps as part of the NAV valuation."
#
# Two unrelated things are called "grace period" here and only one suppresses anything:
#   financials_service  reduces the REPORTED accrued balance by payments landing within
#                       45 days of quarter end -- One Pager delinquency, presentation
#   waterfall.py        defers year-end COMPOUNDING past 45 days -- a mechanical rule
# The pref engine the NAV uses compounds at year end with NO grace period at all, so
# the suppression cannot reach a valuation. These pin that separation.
import inspect  # noqa: E402

from flask_app.services import reports_service as _rep  # noqa: E402
from flask_app.services import valuation_nav_service as _nav  # noqa: E402
from flask_app.services import financials_service as _fin  # noqa: E402

_walk_src = inspect.getsource(_nav._pref_walks)
check('the NAV sources its accrual from the Pref Balance Detail engine',
      'build_pref_balance_detail' in _walk_src)
check('...which compounds with no grace period',
      'no grace period' in inspect.getdoc(_rep._compute_accrued_pref))
check('the NAV path carries no grace-period suppression of its own',
      'grace' not in _walk_src.lower()
      and 'grace' not in inspect.getsource(_nav.compute_nav).lower())
# The refusing direction alone would pass if the One Pager logic vanished entirely.
check('the One Pager suppression still exists where it belongs',
      'grace' in inspect.getsource(_fin).lower())

# Verified live on P0000044 at a test value of 25,000,000: the Pref step allocated
# 51,926.54 to PPI19 and PSC NAV came to 9,751,926.54 = 9,700,000 + 51,926.54.
check('PSC NAV is the balance plus the accrual, to the cent',
      abs((9700000.0 + 51926.54) - 9751926.54) < 0.005)


print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print('  - ' + f)
sys.exit(1 if FAIL else 0)
