"""Guardrail: one number, one engine.

Jim, Sep 18 2026: "We should not have conflicting calculation results. It will cause
doubt in the accuracy of the entire work. Make sure the vetted calculation engines are
used consistently and we do not have separate calculation engines for the same number.
The only differences in results should come from changes in time frames or projections
that we are running through the engines."

WHAT WENT WRONG. "Accrued pref" had two implementations in ONE FILE:

  * `reports_service._compute_accrued_pref`      -> ROE Summary, Committee Summary
  * `reports_service.build_pref_balance_detail`  -> Pref Balance Detail, One Pager,
                                                    Ownership, NAV, valuation tabs

Same ledger, same rate, different answer on 34 of the 68 deals both could price, with
the ROE path $633,807.54 LOW in aggregate at 2025-12-31. The cause was a lost day at
every year end: it accrued `cur -> 31 Dec`, compounded, then resumed at `1 Jan`, so
31 Dec -> 1 Jan was never accrued. One day per year end, always short, worse the older
the deal. It never looked wrong, because a slightly low accrual is still a plausible
accrual -- which is exactly why a second implementation is dangerous even when it is
nearly right.

The vetted engine reproduces the figures Jim gave from the workbook: P0000044
51,926.54 and P0000031 37,394.57. The other one gave 26,489.03 for P0000031.

This check asserts the SECOND ENGINE CANNOT COME BACK, and that every consumer of the
number reaches the same answer. It pins behaviour, not text: the fixtures drive the
real functions.

Run:  .venv\\Scripts\\python.exe scripts\\one_engine_per_number_check.py
"""

import inspect
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from flask_app.services import reports_service as R  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))


def skip(name, why):
    SKIP.append(name)
    print(f'  SKIP  {name}  [{why}]')


def section(t):
    print(f'\n--- {t}')


# ---------------------------------------------------------------- fixtures
VCODE = 'PTEST1'
IID = 'ITEST1'
INVESTOR = 'PPITEST'
RATE = 0.08
START = date(2016, 6, 30)
ASOF = date(2025, 12, 31)
CAPITAL = 10_000_000.0

ACCT = pd.DataFrame([{
    'InvestmentID': IID, 'InvestorID': INVESTOR, 'EffectiveDate': START,
    'MajorType': 'Contribution', 'TypeName': 'Capital Contribution',
    'Amt': -CAPITAL, 'TypeID': 1001,
}])
INV = pd.DataFrame([{'vcode': VCODE, 'InvestmentID': IID,
                     'Investment_Name': 'Test Deal'}])
WF = pd.DataFrame([{'vcode': VCODE, 'vState': 'Pref', 'PropCode': INVESTOR,
                    'nPercent_dec': RATE, 'iOrder': 1}])


section('The second engine is gone and cannot come back')

check('`_compute_accrued_pref` no longer exists',
      not hasattr(R, '_compute_accrued_pref'))
_src = inspect.getsource(R)
check('...and nothing in reports_service still calls it',
      '_compute_accrued_pref(' not in _src)
check('there is ONE named way to ask for a deal\'s accrued pref',
      hasattr(R, 'deal_accrued_pref') and callable(R.deal_accrued_pref))
check('...and it delegates to the vetted walk rather than re-deriving',
      'build_pref_balance_detail(' in inspect.getsource(R.deal_accrued_pref))
check('...and it does no day-count arithmetic of its own',
      not any(t in inspect.getsource(R.deal_accrued_pref)
              for t in ('365', '366', 'timedelta', '.days')))

_roe_src = inspect.getsource(R.build_roe_summary_row)
check('the ROE Summary takes its accrued pref from that one function',
      'deal_accrued_pref(' in _roe_src)
check('...and holds no pref loop of its own',
      'pref_compounded' not in _roe_src and 'pref_cy' not in _roe_src)


section('The One Pager adapts to the shared engine, it does not repeat it')

from flask_app.services import financials_service as F  # noqa: E402

_op_src = inspect.getsource(F._compute_accrued_from_pref_detail)
check('the One Pager calls the shared function',
      'deal_accrued_pref' in _op_src)
check('...and no longer sums the walk itself',
      'build_pref_balance_detail(' not in _op_src)
check('...and still returns None rather than 0 when it cannot answer',
      F._compute_accrued_from_pref_detail('NOPE', {}, '', {}) is None)


section('A day is never lost at the year end')

# The defect, reduced to arithmetic. A single contribution held to a later date must
# accrue EVERY day in between -- no gap at 31 Dec -> 1 Jan.
res = R.build_pref_balance_detail(VCODE, INVESTOR, ASOF, ACCT, INV, wf_steps=WF)
head = res.get('header') or {}
accrued = abs(float(head.get('accrued_pref') or 0.0))
check('the fixture prices at all', accrued > 0, f'{accrued:,.2f}')


def _closed_form(start, end, capital, rate):
    """Every day accrued, compounding each 31 Dec. No gap."""
    def diy(y):
        return 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    compounded, cy, cur = 0.0, 0.0, start
    while cur < end:
        ye = date(cur.year, 12, 31)
        stop = min(end, ye)
        d = (stop - cur).days
        if d > 0:
            cy += (capital + compounded) * rate * (d / diy(cur.year))
        if stop == ye and stop < end:
            cy += (capital + compounded) * rate * (1 / diy(cur.year))
            compounded += cy
            cy = 0.0
            cur = date(cur.year + 1, 1, 1)
        else:
            break
    return compounded + cy


def _lost_day(start, end, capital, rate):
    """The old loop: restart at 1 Jan, never accruing 31 Dec -> 1 Jan."""
    def diy(y):
        return 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    compounded, cy, cur = 0.0, 0.0, start
    while cur < end:
        ye = date(cur.year, 12, 31)
        stop = min(end, ye)
        d = (stop - cur).days
        if d > 0:
            cy += (capital + compounded) * rate * (d / diy(cur.year))
        if stop == ye and stop < end:
            compounded += cy
            cy = 0.0
            cur = date(cur.year + 1, 1, 1)
        else:
            break
    return compounded + cy


whole = _closed_form(START, ASOF, CAPITAL, RATE)
lossy = _lost_day(START, ASOF, CAPITAL, RATE)
gap = whole - lossy
check('the two conventions really are different, so this test can fail',
      gap > 1000.0, f'gap {gap:,.2f} over 9 year ends')
# The model reconstructs the convention; it is not the specification, and it differs
# from the engine by a rounding-scale amount at the first and last day. So the test is
# WHICH CONVENTION the engine sits with, by a wide margin -- not equality with a model.
check('the engine sits with the whole-day convention',
      abs(accrued - whole) < gap / 10,
      f'engine {accrued:,.2f} vs whole {whole:,.2f} (gap between conventions {gap:,.2f})')
check('...and NOT with the lost-day one',
      abs(accrued - lossy) > gap / 2,
      f'engine {accrued:,.2f} vs lossy {lossy:,.2f}')


section('Every consumer reaches the same answer')

# The ROE Summary row and a direct call must be the same number, not merely close.
row = R.build_roe_summary_row(VCODE, 'Test Deal', ACCT, INV, ASOF, wf_steps=WF)
direct = R.deal_accrued_pref(VCODE, ASOF, ACCT, INV, wf_steps=WF)
check('the ROE Summary row prices the fixture', row is not None)
if row is not None:
    check('ROE Summary == the shared engine, to the cent',
          abs(float(row['Accrued Pref']) - float(direct)) < 0.005,
          f"{row['Accrued Pref']:,.2f} vs {direct:,.2f}")
    check('...and == the walk it is built on',
          abs(float(row['Accrued Pref']) - accrued) < 0.005)

# Only the CUT-OFF may move the answer -- that is the one difference Jim allows.
earlier = R.deal_accrued_pref(VCODE, date(2020, 12, 31), ACCT, INV, wf_steps=WF)
check('a different as-of date gives a different figure', earlier < direct,
      f'{earlier:,.2f} at 2020-12-31 vs {direct:,.2f} at 2025-12-31')
check('...and the same as-of date gives the identical figure twice',
      R.deal_accrued_pref(VCODE, ASOF, ACCT, INV, wf_steps=WF) == direct)


section('Missing is missing, never zero')

check('a deal with no PE investors returns None, not 0.0',
      R.deal_accrued_pref('NOSUCH', ASOF, ACCT, INV, wf_steps=WF) is None)

# ONE investor listed under TWO casings must be walked once. `build_pref_balance_detail`
# filters the ledger case-INSENSITIVELY, so each casing would return the WHOLE ledger
# and the deal's pref would double. The ledger is unchanged here -- one contribution --
# and only the investor list carries the duplicate, which is the real shape of the bug.
_real_list = R.get_deal_pe_investors
R.get_deal_pe_investors = lambda vc, a, i: [
    {'investor_id': INVESTOR}, {'investor_id': INVESTOR.lower()}]
try:
    both = R.deal_accrued_pref(VCODE, ASOF, ACCT, INV, wf_steps=WF)
finally:
    R.get_deal_pe_investors = _real_list
check('one investor under two casings is counted once',
      both is not None and abs(both - direct) < 0.005,
      f'{both:,.2f} vs {direct:,.2f}')
check('...and the real investor list is restored',
      R.get_deal_pe_investors is _real_list)

# A second, genuine contribution SHOULD raise the answer -- otherwise the guard above
# could be satisfied by a function that ignores extra rows entirely.
ACCT3 = pd.concat([ACCT, ACCT.assign(EffectiveDate=date(2018, 6, 30))],
                  ignore_index=True)
more = R.deal_accrued_pref(VCODE, ASOF, ACCT3, INV, wf_steps=WF)
check('a genuinely second contribution does raise the accrual',
      more is not None and more > direct, f'{more:,.2f} vs {direct:,.2f}')


section('Net proceeds comes from the NAV engine or not at all')

from flask_app.services import valuation_service as V  # noqa: E402

_cs = inspect.getsource(V.get_committee_summary)
check('the value-less-debt estimate is gone',
      'cur_value - cur_debt' not in _cs)
check('...and the column is still sourced from the NAV run',
      'nav_by_vcode.get(vcode, {}).get("net_proceeds")' in _cs)
check('...and says whether a NAV exists, so a blank is explicable',
      '"has_nav"' in _cs)


print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print('  - ' + f)
sys.exit(1 if FAIL else 0)
