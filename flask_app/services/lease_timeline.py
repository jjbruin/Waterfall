"""The lease timeline: governing terms -> continuous periods, as of a date.

Step 3 of the rent-roll plan (`.claude/memory/rent_roll_exhibit.md`). New
business's specification (Sep 29 2026), §31: the app is "a lease interpretation
and timeline-building tool": read everything, determine the governing terms,
build the complete timeline, show the rent in force on the as-of date and every
change after it, then the remaining options -- and flag uncertainty rather than
guess.

ONE ENGINE. Every screen and the IC exhibit read this. It computes nothing the
app already answers elsewhere: step dates come from `resolve_rent_steps`, the
governing schedule from `governing_steps`, additional rent from
`additional_in_force`, and rent PSF from `rent_psf_for` -- the same primitives
the rent-roll validation uses, so the two cannot disagree about what a lease
says.

Pure: no database. `build_timeline` takes what the service loads and returns
periods, the exhibit's rows, and the checks/flags of spec §21, §23 and §24.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from dateutil.relativedelta import relativedelta

from flask_app.services.lease_terms import (
    _as_date, additional_in_force, annual_rent_from, governing_steps, rent_psf_for,
)

#: Annual rent / SF against a stated PSF: a difference beyond this is flagged.
PSF_TOLERANCE = 0.02


def _iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def _annual(step: Dict[str, Any]) -> Optional[float]:
    return annual_rent_from(annual_rent=step.get('annual_rent'),
                            monthly_rent=step.get('monthly_rent'))


def _flag(flags: List[Dict[str, Any]], code: str, message: str, **kw) -> None:
    flags.append({'code': code, 'message': message, **kw})


def _option_periods(opt: Dict[str, Any], start: date, end: Optional[date],
                    prior_annual: Optional[float], sf: Optional[float],
                    flags: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """An option's rent periods. Stated amounts as stated; a stated percentage is
    compounded from the rent before it and SAYS it was derived; fair market or
    unstated rent is left blank and flagged (spec §12, §20, §25)."""
    basis = (opt.get('rent_basis') or '').lower() or 'not_stated'
    sched = [e for e in (opt.get('rent_schedule') or []) if isinstance(e, dict)]
    periods: List[Dict[str, Any]] = []
    if not sched:
        _flag(flags, 'option_rent_missing' if basis in ('not_stated', '') else
              'option_rent_fmv' if 'fmv' in basis else 'option_rent_unquantified',
              f"{label}: rent is {basis.replace('_', ' ')} -- no figure to show.",
              option=label)
        return [{'start': _iso(start), 'end': _iso(end), 'annual_rent': None,
                 'psf': None, 'basis': basis, 'derived': False}]
    cur = start
    running = prior_annual
    for i, e in enumerate(sched):
        s = _as_date(e.get('start')) or cur
        nxt = _as_date(sched[i + 1].get('start')) if i + 1 < len(sched) else None
        if nxt is None and i + 1 < len(sched):
            nxt = s + relativedelta(years=1)
        stop = _as_date(e.get('end')) or ((nxt - timedelta(days=1)) if nxt else end)
        amt = annual_rent_from(annual_rent=e.get('annual_rent'),
                               monthly_rent=e.get('monthly_rent'))
        if amt is None and e.get('rent_psf') and sf:
            amt = float(e['rent_psf']) * float(sf)
        derived = False
        if amt is None and e.get('escalation_pct') is not None and running:
            amt = round(running * (1 + float(e['escalation_pct']) / 100.0), 2)
            derived = True
        if amt is None:
            _flag(flags, 'option_rent_missing',
                  f"{label}: no figure for {e.get('period') or _iso(s)}.", option=label)
        psf, _ = rent_psf_for(annual_rent=amt, square_feet=sf,
                              stated_psf=e.get('rent_psf') if not derived else None)
        periods.append({'start': _iso(s), 'end': _iso(stop), 'annual_rent': amt,
                        'psf': psf, 'basis': basis, 'derived': derived,
                        'period': e.get('period')})
        running = amt or running
        cur = (stop + timedelta(days=1)) if stop else cur
    if end and periods and periods[-1]['end'] != _iso(end):
        _flag(flags, 'option_period_mismatch',
              f"{label}: its rent periods end {periods[-1]['end']}, the option "
              f"{_iso(end)}.", option=label)
    return periods


_DAY_TERM = re.compile(r'(\d+)\s*\)?\s*-?\s*days?\b', re.I)


def _opt_dates(o: Dict[str, Any]):
    return (_as_date(o.get('start') or o.get('option_start')),
            _as_date(o.get('end') or o.get('option_end')))


def option_term(o: Dict[str, Any]):
    """An option's term as (label, days): the stated `term_years`; else a stated
    day count in its own wording ("Thirty (30) day option" -- Patton Computers,
    whose term_years is empty); else measured from its dates. Never assumed."""
    ty = o.get('term_years')
    try:
        ty = float(ty)
        if ty >= 1:
            return '%g Years' % ty, round(ty * 365)
        if ty > 0:
            return '%g Months' % round(ty * 12), round(ty * 365)
    except (TypeError, ValueError):
        pass
    words = ' '.join(str(x) for x in [o.get('rent_terms'), o.get('period')]
                     + [e.get('period') for e in (o.get('rent_schedule') or [])
                        if isinstance(e, dict)] if x)
    m = _DAY_TERM.search(words)
    if m:
        return '%d-Day' % int(m.group(1)), int(m.group(1))
    st, en = _opt_dates(o)
    if st and en:
        days = (en - st).days + 1
        if days < 45:
            return '%d-Day' % days, days
        d = relativedelta(en + timedelta(days=1), st)
        yrs = d.years + d.months / 12.0
        return (('%g Years' % yrs) if yrs >= 1 else ('%g Months' % round(yrs * 12))), days
    return 'term not stated', None


def summarise_options(options: List[Dict[str, Any]]) -> str:
    """ "2 x 5 Years" / "6 x 30-Day Rolling" / "None" -- spec §13. Rolling when two
    or more options of the same term under a year follow one another. THE one
    definition: the consolidation's `_options_summary` calls this too."""
    counts: Dict[str, int] = {}
    days_of: Dict[str, Any] = {}
    for o in options:
        label, days = option_term(o)
        counts[label] = counts.get(label, 0) + 1
        days_of[label] = days
    parts = []
    for k, n in counts.items():
        rolling = n >= 2 and days_of.get(k) is not None and days_of[k] < 365
        parts.append('%d x %s%s' % (n, k, ' Rolling' if rolling else ''))
    return ', '.join(parts) or 'None'


def build_timeline(terms: Dict[str, Any], steps: List[Dict[str, Any]],
                   square_feet: Optional[float], as_of: Any,
                   settled_annual_rent: Optional[float] = None,
                   rent_roll_annual_rent: Optional[float] = None,
                   tenant_active: bool = True,
                   settled_start: Any = None, settled_expiration: Any = None,
                   settled_schedule: Optional[List[Dict[str, Any]]] = None,
                   settled_options: Optional[List[Dict[str, Any]]] = None
                   ) -> Dict[str, Any]:
    """The complete timeline and the rent-roll rows for one tenant.

    `steps` are RESOLVED steps (`resolve_rent_steps` output). `settled_annual_rent`
    is an analyst's settled figure, which outranks the lease-derived current rent
    because it is the analyst's conclusion from the same documents;
    `rent_roll_annual_rent` is the last resort, used only when the lease states no
    schedule at all, and is flagged when used.
    """
    flags: List[Dict[str, Any]] = []
    asof = _as_date(as_of)
    # AN ANALYST'S SETTLED VALUE OUTRANKS THE DERIVED ONE (step 4): it is their
    # conclusion from the same documents, recorded with a reason and a citation.
    doc_start, doc_exp = terms.get('lease_commencement'), terms.get('lease_expiration')
    for what, settled, doc in (('start', settled_start, doc_start),
                               ('expiration', settled_expiration, doc_exp)):
        # A SETTLEMENT MADE BEFORE THE DOCUMENTS WERE READ CORRECTLY OUTRANKS THEM
        # SILENTLY. Mattress Firm's expiry was settled 2027-09-30 when the app still
        # read its documents wrongly; they now give 2035-03-21 (Sep 30 2026). The
        # analyst's value stands -- it is theirs -- but the difference is shown.
        if settled and _as_date(doc) and _as_date(settled) != _as_date(doc):
            _flag(flags, 'settled_%s_differs' % what,
                  f"The analyst settled the {what} as {_iso(_as_date(settled))}; the "
                  f"governing documents now give {_iso(_as_date(doc))} -- re-check the "
                  f"settlement.")
    if settled_start or settled_expiration:
        terms = dict(terms)
        if settled_start:
            terms['lease_commencement'] = settled_start
        if settled_expiration:
            terms['lease_expiration'] = settled_expiration
            terms['_expiration_basis'] = 'settled by the analyst'
    start = _as_date(terms.get('lease_commencement'))
    start_basis = 'settled by the analyst' if settled_start else 'lease commencement'
    if start is None and _as_date(terms.get('rent_commencement')):
        start = _as_date(terms.get('rent_commencement'))
        start_basis = 'rent commencement (no lease commencement stated)'
        _flag(flags, 'start_from_rent_commencement',
              'No lease commencement is stated; the start shown is rent commencement.')
    if start is None:
        _flag(flags, 'start_missing', 'No lease commencement or rent commencement date.')
    expiration = _as_date(terms.get('lease_expiration'))
    if expiration is None:
        _flag(flags, 'expiration_missing', 'No lease expiration date.')
    if asof and expiration and expiration < asof and tenant_active:
        _flag(flags, 'expired_but_active',
              f"The documents end the lease {_iso(expiration)}, before the as-of date "
              f"{_iso(asof)}, yet the tenant is active -- an extension or option "
              f"exercise may be missing from the uploaded documents.")

    gov, notes = governing_steps(steps)
    for n in notes:
        _flag(flags, 'schedule_ambiguous', n)
    base = [s for s in gov if not s.get('is_additional') and _as_date(s.get('effective_date'))]
    undated = [s for s in gov if not s.get('is_additional')
               and not _as_date(s.get('effective_date'))]
    if undated:
        _flag(flags, 'undated_steps',
              f"{len(undated)} rent step(s) could not be dated and are not in the timeline.")
    beyond = [s for s in base if expiration and _as_date(s['effective_date']) > expiration]
    if beyond:
        _flag(flags, 'step_after_expiration',
              f"{len(beyond)} rent step(s) begin after the expiration {_iso(expiration)} "
              f"and are not part of the term.")
    base = [s for s in base if s not in beyond]

    # ---- the contractual term, continuous by construction
    term: List[Dict[str, Any]] = []
    for i, s in enumerate(base):
        s_start = _as_date(s['effective_date'])
        nxt = _as_date(base[i + 1]['effective_date']) if i + 1 < len(base) else None
        s_end = (nxt - timedelta(days=1)) if nxt else expiration
        amt = _annual(s)
        extra_steps, extra = additional_in_force(steps, s_start)
        if extra:
            amt = (amt or 0) + extra
        stated = s.get('rent_per_sf') if not extra else None
        psf, _ = rent_psf_for(annual_rent=amt, square_feet=square_feet, stated_psf=stated)
        if (stated and amt and square_feet
                and abs(float(amt) / float(square_feet) - float(stated))
                > PSF_TOLERANCE * float(stated)):
            _flag(flags, 'psf_mismatch',
                  f"Step from {_iso(s_start)}: annual rent / SF is "
                  f"{float(amt) / float(square_feet):.2f}, the lease states {float(stated):.2f}.")
        term.append({'start': _iso(s_start), 'end': _iso(s_end), 'annual_rent': amt,
                     'psf': psf, 'source': (s.get('source_doc') or '').rsplit('/', 1)[-1],
                     'basis': s.get('effective_date_basis') or 'stated'})
    if term and expiration and term[-1]['end'] != _iso(expiration):
        _flag(flags, 'final_period_mismatch',
              f"The last rent period ends {term[-1]['end']}, not the expiration "
              f"{_iso(expiration)}.")

    # ---- the rent in force on the as-of date, and what follows
    current = None
    if asof:
        for p in term:
            if _as_date(p['start']) <= asof and (not p['end'] or asof <= _as_date(p['end'])):
                current = dict(p)
    current_basis = 'lease schedule'
    if current is None and term and asof and asof < _as_date(term[0]['start']):
        current = dict(term[0])
        current_basis = 'first scheduled rent (rent has not commenced by the as-of date)'
        _flag(flags, 'rent_not_commenced',
              f"Rent begins {term[0]['start']}, after the as-of date.")
    if settled_annual_rent is not None:
        psf, _ = rent_psf_for(annual_rent=settled_annual_rent, square_feet=square_feet)
        if current and current.get('annual_rent') and abs(
                float(current['annual_rent']) - float(settled_annual_rent)) > 1:
            _flag(flags, 'settled_differs',
                  f"The analyst settled {settled_annual_rent:,.0f}; the lease schedule "
                  f"gives {float(current['annual_rent']):,.0f} on the as-of date.")
        current = dict(current or {}, annual_rent=float(settled_annual_rent), psf=psf)
        current_basis = 'settled by the analyst'
    elif current is None and rent_roll_annual_rent is not None:
        psf, _ = rent_psf_for(annual_rent=rent_roll_annual_rent, square_feet=square_feet)
        current = {'annual_rent': float(rent_roll_annual_rent), 'psf': psf}
        current_basis = 'rent roll (the lease states no dated schedule)'
        _flag(flags, 'current_rent_from_rent_roll',
              'No dated rent schedule in the lease; the current rent shown is the rent roll\'s.')
    future = [p for p in term if asof and _as_date(p['start']) > asof]
    schedule_basis = 'lease schedule'
    if settled_schedule is not None:
        future = [dict(p, source='settled') for p in settled_schedule
                  if not asof or not p.get('end') or _as_date(p['end']) >= asof]
        schedule_basis = 'settled by the analyst'

    # ---- remaining options, in sequence after the term
    opts = [o for o in (terms.get('_remaining_options') or []) if isinstance(o, dict)]
    options_basis = 'governing documents'
    if settled_options is not None:
        opts = []
        options_basis = 'settled by the analyst' 
    opts.sort(key=lambda o: (_as_date(o.get('option_start')) or date.max,
                             o.get('option_number') or 0))
    options = []
    prev_end = expiration
    prior = (future[-1]['annual_rent'] if future else
             (current or {}).get('annual_rent'))
    for i, o in enumerate(opts, start=1):
        label = f'Option {i}'
        o_start = _as_date(o.get('option_start'))
        o_end = _as_date(o.get('option_end'))
        if o_start is None and prev_end:
            o_start = prev_end + timedelta(days=1)
            _flag(flags, 'option_dates_derived',
                  f"{label}: start date derived as the day after the prior term.", option=label)
        if o_end is None and o_start and o.get('term_years'):
            try:
                o_end = (o_start + relativedelta(months=round(float(o['term_years']) * 12))
                         - timedelta(days=1))
                _flag(flags, 'option_dates_derived',
                      f"{label}: end date derived from its {o['term_years']}-year term.",
                      option=label)
            except (TypeError, ValueError):
                pass
        if prev_end and o_start and o_start != prev_end + timedelta(days=1):
            _flag(flags, 'option_sequence',
                  f"{label} starts {_iso(o_start)}, not the day after "
                  f"{_iso(prev_end)}.", option=label)
        periods = _option_periods(o, o_start, o_end, prior, square_feet, flags, label) \
            if o_start else []
        options.append({'label': label, 'start': _iso(o_start), 'end': _iso(o_end),
                        'term_years': o.get('term_years'),
                        'rent_terms': o.get('rent_terms'),
                        'rent_basis': o.get('rent_basis') or 'not_stated',
                        'periods': periods})
        prev_end = o_end or prev_end
        prior = next((p['annual_rent'] for p in reversed(periods) if p['annual_rent']),
                     prior)
    options_summary = terms.get('_options_summary') or 'None'
    if settled_options is not None:
        options = []
        for i, o in enumerate(settled_options, start=1):
            periods = [dict(p) for p in (o.get('periods') or [])] or [
                {'start': o.get('start'), 'end': o.get('end'),
                 'annual_rent': o.get('annual_rent'), 'psf': o.get('psf')}]
            options.append({'label': f'Option {i}', 'start': o.get('start'),
                            'end': o.get('end'), 'term_years': o.get('term_years'),
                            'rent_basis': o.get('rent_basis') or 'settled',
                            'periods': periods})
        options_summary = summarise_options(settled_options)

    # ---- the rent-roll rows (spec §26-27): tenant, steps, then each option
    rows = [{'kind': 'tenant', 'label': None, 'sf': square_feet,
             'start': _iso(start), 'end': _iso(expiration),
             'annual_rent': (current or {}).get('annual_rent'),
             'psf': (current or {}).get('psf'),
             'options': options_summary}]
    for i, p in enumerate(future):
        rows.append({'kind': 'step', 'label': 'Rent Step Dates' if i == 0 else None,
                     'start': p['start'], 'end': p['end'],
                     'annual_rent': p['annual_rent'], 'psf': p['psf']})
    # CONSECUTIVE SHORT OPTIONS AT ONE RENT PRINT AS ONE ROW, as new business's
    # exhibit shows Patton's six 30-day options: "Option 1 (30-day rolling, 6x)".
    display = []
    i = 0
    while i < len(options):
        o = options[i]
        lab, days = option_term(o)
        run = [o]
        if days is not None and days < 365 and len(o['periods']) <= 1:
            while (i + len(run) < len(options)
                   and option_term(options[i + len(run)])[0] == lab
                   and len(options[i + len(run)]['periods']) <= 1
                   and (options[i + len(run)]['periods'] or [{}])[0].get('annual_rent')
                   == (o['periods'] or [{}])[0].get('annual_rent')):
                run.append(options[i + len(run)])
        if len(run) >= 2:
            p0 = (o['periods'] or [{}])[0]
            display.append({'label': '%s (%s rolling, %dx)' % (o['label'], lab.lower(), len(run)),
                            'start': o['start'], 'end': run[-1]['end'],
                            'periods': [dict(p0, start=o['start'], end=run[-1]['end'])]})
        else:
            display.append(o)
        i += len(run)
    for o in display:
        for j, p in enumerate(o['periods'] or [{'start': o['start'], 'end': o['end'],
                                                'annual_rent': None, 'psf': None}]):
            rows.append({'kind': 'option' if j == 0 else 'option_step',
                         'label': o['label'] if j == 0 else ('Rent Step Dates' if j == 1 else None),
                         'start': p['start'], 'end': p['end'],
                         'annual_rent': p['annual_rent'], 'psf': p['psf'],
                         'derived': p.get('derived', False)})
    return {
        'as_of': _iso(asof), 'start': _iso(start), 'start_basis': start_basis,
        'expiration': _iso(expiration),
        'expiration_basis': terms.get('_expiration_basis') or 'governing documents',
        'square_feet': square_feet, 'current': current, 'current_basis': current_basis,
        'term': term, 'future': future, 'options': options,
        'options_summary': options_summary,
        'schedule_basis': schedule_basis, 'options_basis': options_basis,
        'rows': rows, 'flags': flags,
    }
