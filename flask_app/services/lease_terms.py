"""Lease term primitives: rent PSF, amendment order, and relative rent periods.

Pure functions, no database and no I/O, so every consumer reaches the same answer and
the arithmetic can be tested against a lease rather than against itself. See CLAUDE.md,
**ONE NUMBER, ONE ENGINE**.

New business, Sep 19 2026 (via Jim), on validating the rent roll against the leases:

  1. "When a lease doesn't specify the rent PSF, the application should simply
     calculate by taking the annual rent by the square feet (only calculate rent PSF
     on annual rent, not monthly)"

  2. "the Hobby Lobby lease should analyze the most recent lease amendment (4th
     Amendment) and follow the corresponding dates for rental increases during
     step-up periods and rental options. In some cases, a lease may not specify exact
     dates and instead reference a specific month of the lease term (e.g., Months
     1-12). To calculate the current annual rent amount, the application should
     reference the Rent Commencement Date to determine where the tenant currently
     falls within the lease term and, accordingly, the applicable base rent."

Both describe the same underlying gap: the app could not say WHICH rent is in force
today, so it guessed -- and the guess it made was the worst available one. See
`resolve_rent_steps`.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Rent PSF
# ---------------------------------------------------------------------------

def annual_rent_psf(annual_rent: Optional[float],
                    square_feet: Optional[float]) -> Optional[float]:
    """Rent PSF = ANNUAL rent over square feet. The only way this app derives one.

    New business: "only calculate rent PSF on annual rent, not monthly". A monthly
    figure divided by square feet is a perfectly plausible-looking number roughly a
    twelfth of the right one, and nothing downstream can tell the two apart -- the
    rent roll import already shipped a 12x error of exactly this shape (v495).

    Returns None when it cannot be computed. Never 0.0: a deal with no square footage
    has no rent PSF, and that is not the same as a rent PSF of zero.
    """
    try:
        ar = float(annual_rent) if annual_rent is not None else None
        sf = float(square_feet) if square_feet is not None else None
    except (TypeError, ValueError):
        return None
    if not ar or not sf or sf <= 0:
        return None
    return ar / sf


def annual_rent_from(annual_rent: Optional[float] = None,
                     monthly_rent: Optional[float] = None) -> Optional[float]:
    """The annual rent, taking a stated annual figure over an annualised monthly one.

    Annualising is not the same as computing a monthly PSF: the PSF is still taken on
    an annual rent, which is what was asked for.
    """
    try:
        ar = float(annual_rent) if annual_rent is not None else None
    except (TypeError, ValueError):
        ar = None
    if ar:
        return ar
    try:
        mr = float(monthly_rent) if monthly_rent is not None else None
    except (TypeError, ValueError):
        mr = None
    return mr * 12 if mr else None


def rent_psf_for(annual_rent: Optional[float] = None,
                 monthly_rent: Optional[float] = None,
                 square_feet: Optional[float] = None,
                 stated_psf: Optional[float] = None) -> Tuple[Optional[float], str]:
    """(rent PSF, how it was arrived at). A STATED figure always wins over a derived one.

    The basis travels with the number so a reader can tell the lease's own figure from
    ours. Bases: 'stated', 'annual rent / SF', 'annualised monthly rent / SF', or ''.
    """
    try:
        sp = float(stated_psf) if stated_psf is not None else None
    except (TypeError, ValueError):
        sp = None
    if sp:
        return sp, 'stated'

    try:
        ar_direct = float(annual_rent) if annual_rent is not None else None
    except (TypeError, ValueError):
        ar_direct = None

    ar = annual_rent_from(annual_rent=annual_rent, monthly_rent=monthly_rent)
    psf = annual_rent_psf(ar, square_feet)
    if psf is None:
        return None, ''
    return psf, ('annual rent / SF' if ar_direct
                 else 'annualised monthly rent / SF')


# ---------------------------------------------------------------------------
# Amendment order
# ---------------------------------------------------------------------------

_ORDINAL_WORDS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14,
    'fifteenth': 15,
}

# "Fourth Amendment", "4th Amendment", "Amendment No. 4", "Amendment #4", "4 Amend"
_ORD_BEFORE = re.compile(
    r'(?i)\b(' + '|'.join(_ORDINAL_WORDS) + r'|\d{1,2}(?:st|nd|rd|th)?)\s*'
    r'(?:\([^)]*\)\s*)?amend')
_ORD_AFTER = re.compile(r'(?i)\bamend\w*\s*(?:no\.?|number|#)?\s*(\d{1,2})\b')


def amendment_ordinal(filename: str) -> Optional[int]:
    """Which amendment this is, read from the filename. None when it does not say.

    The ordinal was ALREADY being matched by `DOC_TYPE_PATTERNS` and thrown away, so a
    folder of "First/Second/Third/Fourth Amendment" carried the sequence in plain text
    while the consolidation ordered by a date none of those filenames contains.
    """
    if not filename:
        return None
    m = _ORD_BEFORE.search(filename)
    if m:
        tok = m.group(1).lower()
        if tok in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[tok]
        digits = re.sub(r'\D', '', tok)
        if digits:
            return int(digits)
    m = _ORD_AFTER.search(filename)
    if m:
        return int(m.group(1))
    return None


# A date ANYWHERE in the name, not only at the front. Four-digit year required in one
# of the three positions -- a bare "2019" is a year, not a date, and inventing
# 2019-01-01 from it would put an amendment in a month it was never signed.
_DATE_PATTERNS = [
    (re.compile(r'(\d{4})[.\-_/](\d{1,2})[.\-_/](\d{1,2})'), (1, 2, 3)),
    (re.compile(r'(\d{1,2})[.\-_/](\d{1,2})[.\-_/](\d{4})'), (3, 1, 2)),
]


def parse_doc_date_anywhere(filename: str) -> Optional[str]:
    """A YYYY-MM-DD date found anywhere in the filename, or None.

    The previous rule anchored at the START of the name, so `Hobby Lobby -
    2019.04.02 Fourth Amendment.pdf` yielded nothing and the document sorted as
    undated. Refuses anything that is not a real calendar date.
    """
    if not filename:
        return None
    for pat, (yi, mi, di) in _DATE_PATTERNS:
        for m in pat.finditer(filename):
            try:
                y, mo, d = int(m.group(yi)), int(m.group(mi)), int(m.group(di))
                return date(y, mo, d).isoformat()
            except ValueError:
                continue    # e.g. 2019.13.45 -- not a date, keep looking
    return None


def order_lease_documents(docs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[str]]:
    """Original lease first, then amendments in the order they were made.

    Returns (ordered docs, notes). The NOTES are the point: when the order cannot be
    established the caller must be able to say so, because consolidation layers later
    documents over earlier ones and a wrong order silently reinstates superseded rent.

    Each doc needs `doc_type`, and may carry `doc_date`, `ordinal` and `id`.

    The rule, in order of preference:
      * every amendment dated  -> order by date (and say so if a stated ordinal
        disagrees, which means one of the two is wrong)
      * every amendment numbered -> order by number
      * otherwise -> best effort by (date, number, id) AND a note that it is partial
    """
    notes: List[str] = []
    originals = [d for d in docs if (d.get('doc_type') or '') == 'Original Lease']
    amendments = [d for d in docs if (d.get('doc_type') or '') == 'Amendment']
    others = [d for d in docs
              if (d.get('doc_type') or '') not in ('Original Lease', 'Amendment')]

    if len(originals) > 1:
        notes.append(f"{len(originals)} documents are classified as the original "
                     f"lease; only one can be the base.")

    dated = [d for d in amendments if d.get('doc_date')]
    numbered = [d for d in amendments if d.get('ordinal') is not None]

    if amendments and len(dated) == len(amendments):
        ordered_am = sorted(amendments,
                            key=lambda d: (d['doc_date'], d.get('ordinal') or 0,
                                           d.get('id') or 0))
        if len(numbered) == len(amendments):
            by_num = sorted(amendments, key=lambda d: d['ordinal'])
            if [d.get('id') for d in by_num] != [d.get('id') for d in ordered_am]:
                notes.append("The amendment dates and the amendment numbers give "
                             "different orders; the dates were used.")
    elif amendments and len(numbered) == len(amendments):
        ordered_am = sorted(amendments, key=lambda d: d['ordinal'])
        notes.append("Amendments ordered by the number in their filename; none "
                     "carries a date.")
    else:
        ordered_am = sorted(
            amendments,
            key=lambda d: (d.get('doc_date') or '', d.get('ordinal') or 0,
                           d.get('id') or 0))
        if amendments:
            missing = len(amendments) - max(len(dated), len(numbered))
            notes.append(
                f"{missing} of {len(amendments)} amendments carry neither a date nor "
                f"a number, so the order they were applied in is not established.")

    # EVERYTHING AFTER THE BASE IS LAYERED IN DATE ORDER, amendments and other
    # documents together. This used to return `originals + ordered_am + others`,
    # which applied every non-amendment AFTER every amendment — so a 2021
    # Acceptance of Premises overwrote the expiration set by a 2026 First
    # Amendment. It went unnoticed while the classifier typed almost everything
    # `Original Lease` and `others` was nearly empty; correcting the classifier
    # filled `others` with 147 documents and the regression surfaced at once, on
    # Style Studio (expiry 2031 -> 2026), Green Zone (2026 -> 2025) and
    # Appliances 4 Less (suite N625 -> a misread "G").
    #
    # UNDATED DOCUMENTS SORT LAST, amendments among them by their number, which
    # is what keeps the case this ordering was built for: a folder of "First /
    # Second / Third / Fourth Amendment.pdf" carrying no dates at all still
    # applies 1, 2, 3, 4 with the Fourth governing.
    #
    # On an equal date the amendment wins, since a non-amendment carries no
    # ordinal and sorts ahead of one that does.
    UNDATED = '9999-99-99'
    seq = {id(d): i for i, d in enumerate(ordered_am)}

    def _key(d):
        is_am = (d.get('doc_type') or '') == 'Amendment'
        return ((d.get('doc_date') or UNDATED),
                seq.get(id(d), -1) + 1 if is_am else 0,
                d.get('id') or 0)

    return originals + sorted(ordered_am + others, key=_key), notes


# ---------------------------------------------------------------------------
# Relative rent periods
# ---------------------------------------------------------------------------

# "Months 1-12", "Month 13 - 24", "Months 1 through 12", "Lease Year 2", "Year 3"
_MONTHS_RANGE = re.compile(
    r'(?i)\bmonths?\s*(\d{1,3})\s*(?:-|–|—|to|through|thru)\s*(\d{1,3})')
_MONTH_SINGLE = re.compile(r'(?i)\bmonths?\s*(\d{1,3})\b')
_LEASE_YEAR = re.compile(r'(?i)\b(?:lease\s*)?year\s*(\d{1,2})\b')


def parse_relative_period(textval: Any) -> Optional[Tuple[int, Optional[int]]]:
    """Read "Months 1-12" / "Lease Year 2" as (start_month, end_month).

    Months are 1-based from rent commencement: month 1 is the month rent starts.
    A lease year N is months 12(N-1)+1 .. 12N.

    Returns None for anything that does not state a period, INCLUDING a real date --
    an ISO date is not a relative period and must not be coerced into one.
    """
    if textval is None:
        return None
    s = str(textval).strip()
    if not s:
        return None
    if re.match(r'^\d{4}-\d{2}-\d{2}', s):
        return None

    m = _MONTHS_RANGE.search(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a >= 1 and b >= a:
            return a, b
        return None
    m = _LEASE_YEAR.search(s)
    if m:
        y = int(m.group(1))
        if y >= 1:
            return 12 * (y - 1) + 1, 12 * y
        return None
    m = _MONTH_SINGLE.search(s)
    if m:
        a = int(m.group(1))
        if a >= 1:
            return a, None
    return None


def _as_date(value: Any) -> Optional[date]:
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


def month_to_date(rent_commencement: Any, month_number: int) -> Optional[date]:
    """The first day of month N of the term, counting rent commencement as month 1.

    Month 1 begins ON the rent commencement date, so month N begins
    `rent_commencement + (N-1) months`. A lease that commences 2019-04-15 has month 13
    beginning 2020-04-15, not 2020-04-01 -- the anniversary, not the calendar month.
    """
    rc = _as_date(rent_commencement)
    if rc is None or not month_number or month_number < 1:
        return None
    return rc + relativedelta(months=month_number - 1)


def resolve_rent_steps(steps: List[Dict[str, Any]],
                       rent_commencement: Any,
                       square_feet: Optional[float] = None
                       ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Give every rent step a real date where the lease makes one knowable.

    A step may state a date, or a period of the term ("Months 1-12"), or -- when the
    extraction put the period in the date field, which is what the old prompt forced
    it to do -- a date field holding text like "Lease Year 7".

    Each step comes back with:
        effective_date        resolved ISO date, or None when unknowable
        effective_date_basis  'stated' | 'month N of the term' | ''
        period_start_month / period_end_month
        rent_psf / rent_psf_basis   (annual rent over SF -- never monthly)
        annual_rent           annualised from monthly when only monthly was given

    Steps that cannot be dated are RETURNED, not dropped, with the reason in `notes`.
    A step nobody can place is a finding about the lease; silently discarding it makes
    the schedule look complete.
    """
    out: List[Dict[str, Any]] = []
    notes: List[str] = []
    rc = _as_date(rent_commencement)
    needed_rc = False

    for step in steps or []:
        s = dict(step)
        stated = _as_date(s.get('effective_date'))
        start_m = s.get('period_start_month')
        end_m = s.get('period_end_month')

        # A period may arrive in its own fields, or hidden in the date field.
        if start_m is None:
            rel = parse_relative_period(s.get('period') or s.get('effective_date'))
            if rel:
                start_m, end_m = rel[0], (end_m if end_m is not None else rel[1])

        if stated:
            s['effective_date'] = stated.isoformat()
            s['effective_date_basis'] = 'stated'
        elif start_m:
            needed_rc = True
            d = month_to_date(rc, int(start_m))
            if d:
                s['effective_date'] = d.isoformat()
                s['effective_date_basis'] = f'month {int(start_m)} of the term'
            else:
                s['effective_date'] = None
                s['effective_date_basis'] = ''
        else:
            s['effective_date'] = None
            s['effective_date_basis'] = ''

        s['period_start_month'] = int(start_m) if start_m else None
        s['period_end_month'] = int(end_m) if end_m else None

        ar = annual_rent_from(annual_rent=s.get('annual_rent'),
                              monthly_rent=s.get('monthly_rent'))
        if ar and not s.get('annual_rent'):
            s['annual_rent'] = ar
        psf, basis = rent_psf_for(annual_rent=s.get('annual_rent'),
                                  monthly_rent=s.get('monthly_rent'),
                                  square_feet=square_feet,
                                  stated_psf=s.get('rent_per_sf'))
        s['rent_per_sf'] = psf
        s['rent_psf_basis'] = basis
        out.append(s)

    if needed_rc and rc is None:
        notes.append("Some rent steps are stated as months of the term, and the lease "
                     "has no rent commencement date recorded, so they cannot be "
                     "placed on the calendar.")
    undated = [s for s in out if not s.get('effective_date')]
    if undated:
        notes.append(f"{len(undated)} of {len(out)} rent steps could not be dated.")

    out.sort(key=lambda s: (s.get('effective_date') is None,
                            s.get('effective_date') or '',
                            s.get('period_start_month') or 0))
    return out, notes


def step_in_force_at(steps: List[Dict[str, Any]], as_of: Any
                     ) -> Tuple[Optional[Dict[str, Any]], str]:
    """The rent step in force on a date: the latest one starting on or before it.

    Returns (step, basis). Basis is '' with no step, otherwise a sentence naming how
    it was reached, because the answer is only usable if the reader can see why.

    THIS REPLACES A GUESS THAT COULD NOT FAIL. When dates did not resolve, the
    validation picked the step whose annual rent was CLOSEST to the rent roll's --
    which is the rent roll validating itself. It agreed by construction, so a rent
    roll that disagreed with every step in the lease still reported a match.
    """
    d = _as_date(as_of)
    if not steps or d is None:
        return None, ''
    dated = [s for s in steps if _as_date(s.get('effective_date'))]
    if not dated:
        return None, ''
    eligible = [s for s in dated if _as_date(s['effective_date']) <= d]
    if not eligible:
        first = min(dated, key=lambda s: _as_date(s['effective_date']))
        return None, (f"The earliest rent step begins "
                      f"{first['effective_date']}, after {d.isoformat()}.")
    best = max(eligible, key=lambda s: _as_date(s['effective_date']))
    basis = best.get('effective_date_basis') or 'stated'
    return best, (f"Step effective {best['effective_date']} ({basis}), "
                  f"in force at {d.isoformat()}.")


# ---------------------------------------------------------------------------
# Fixed recoveries (CAM / operating expenses)
# ---------------------------------------------------------------------------
#
# Jim, Sep 20 2026, reading the AT&T Mobility 4th Amendment on Market at Poplar:
# "one of the lease amendments was stating a fixed CAM charge for the lease. Is this
# situation part of the lease review and validation to the rent roll?"
#
# It was not. The extraction captured `cam_structure = 'fixed'` -- the WORD -- and
# nothing captured the AMOUNT, so the one figure the rent roll could be checked
# against did not exist anywhere in the app. That amendment states a schedule:
#
#     2025          $2.16/SF   $8,640.00/yr   $720.00/mo
#     2026 to 2030  $2.38/SF   $9,520.00/yr   $793.33/mo
#     2031 to 2035  $2.62/SF  $10,480.00/yr   $873.33/mo
#
# and adds "In no event shall Tenant be required to pay any amount in excess of or
# below the fixed Tenant's Proportionate Share of Operating Expenses set forth above."
# A capped, stated, checkable number -- exactly the kind of term a rent roll gets
# wrong quietly.


def cam_fixed_in_force(entries: List[Dict[str, Any]],
                       as_of: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """The fixed recovery schedule row that applies on a date.

    Rows carry a year range (`year_start` / `year_end`); a row with no `year_end`
    runs open-ended from its start. Returns (row, basis) and, like every other
    resolver here, the basis is a sentence rather than a flag -- a recovery figure
    the reader cannot trace is not usable as evidence against a rent roll.
    """
    d = _as_date(as_of)
    if not entries or d is None:
        return None, ''
    year = d.year
    eligible = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        ys, ye = e.get('year_start'), e.get('year_end')
        try:
            ys = int(ys) if ys is not None else None
            ye = int(ye) if ye is not None else None
        except (TypeError, ValueError):
            continue
        if ys is None:
            continue
        if year >= ys and (ye is None or year <= ye):
            eligible.append((ys, e))
    if not eligible:
        starts = [int(e['year_start']) for e in entries
                  if isinstance(e, dict) and e.get('year_start') is not None]
        if starts and year < min(starts):
            return None, (f"The fixed recovery schedule begins {min(starts)}, "
                          f"after {d.isoformat()}.")
        return None, (f"The fixed recovery schedule does not cover {year}.")
    # The latest range that has begun, so overlapping rows resolve the way rent
    # steps do rather than by document order.
    ys, best = max(eligible, key=lambda t: t[0])
    label = best.get('period') or (
        str(ys) if best.get('year_end') in (None, ys) else
        '%s to %s' % (ys, best.get('year_end')))
    return best, ('Fixed recovery stated for %s, in force at %s.'
                  % (label, d.isoformat()))


def annual_recovery_psf(entry: Optional[Dict[str, Any]],
                        square_feet: Any) -> Optional[float]:
    """A fixed recovery row as an ANNUAL per-SF figure.

    Same rule as `annual_rent_psf`: a MONTHLY amount is annualised before dividing,
    never divided as-is. Prefers the stated per-SF figure, then the annual amount,
    then twelve times the monthly -- so a lease that states all three is read as it
    is written, and one that states only a monthly charge still produces the figure
    the rent roll can be checked against.
    """
    if not isinstance(entry, dict):
        return None
    psf = entry.get('per_sf')
    if psf not in (None, ''):
        try:
            return float(psf)
        except (TypeError, ValueError):
            pass
    try:
        sf = float(square_feet) if square_feet not in (None, '') else None
    except (TypeError, ValueError):
        sf = None
    if not sf:
        return None
    for key, mult in (('annual', 1.0), ('monthly', 12.0)):
        val = entry.get(key)
        if val in (None, ''):
            continue
        try:
            return float(val) * mult / sf
        except (TypeError, ValueError):
            continue
    return None
