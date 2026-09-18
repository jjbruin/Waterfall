"""Rent roll mapping — scan an uploaded file, let the analyst classify it, then load.

Keyword matching alone cannot answer two questions a rent roll does not state, and
Market at Poplar got both wrong:

  * **Which charge columns are recoveries.** The file prints CAM, Insurance and Tax as
    three separate columns. ``_find_col`` returns the FIRST match and stops, so CAM
    imported and the other two were dropped -- 34% of the recovery. Property Tax alone
    is larger than CAM.
  * **Whether a charge is monthly or annual.** The header says "Base Rent", with no
    qualifier. It was read as annual, so every figure landed 12x low ($1.14/SF/yr
    against a real $13.64).

Neither is recoverable from the header text, so neither is guessed here. This module
PROPOSES a role and a period for every column, and the analyst confirms before anything
is written. ``scan()`` reads nothing into the database; ``apply_mapping()`` does the
load once the analyst has answered.

Two layouts are handled, because the same property arrives in both:

  * ``columnar`` -- one row per tenant, charges across the columns (the Excel export).
  * ``stacked``  -- one header line per tenant followed by one line per charge, which is
    how the MRI "Master Rent Roll" PDF prints. Charges are ROWS there, so the analyst
    classifies charge LABELS rather than column headers; the mapping is otherwise
    identical, which is why both layouts share this module rather than getting a parser
    each.

Both layouts carry the file's own arithmetic -- a "Total Charges" column, a
"* Tenant Total *" line, a building subtotal row -- and ``tie_out`` reports it. A
recovery column left unclassified shows up there as a difference instead of passing
silently.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# --- Roles an analyst can assign to a column (columnar) or charge label (stacked) ---
ROLE_IGNORE = 'ignore'
ROLE_TENANT = 'tenant_name'
ROLE_SUITE = 'suite'
ROLE_SF = 'square_feet'
ROLE_LEASE_TYPE = 'lease_type'
ROLE_LEASE_START = 'lease_start'
ROLE_LEASE_END = 'lease_end'
ROLE_DEPOSIT = 'security_deposit'
ROLE_BASE_RENT = 'base_rent'
ROLE_RECOVERY = 'recovery'
ROLE_MISC = 'misc'

IDENTITY_ROLES = [
    ROLE_TENANT, ROLE_SUITE, ROLE_SF, ROLE_LEASE_TYPE,
    ROLE_LEASE_START, ROLE_LEASE_END,
]
# Charges that recur, so the analyst must say over what period they are stated.
# A security deposit is a one-off balance and takes no period.
PERIODIC_ROLES = [ROLE_BASE_RENT, ROLE_RECOVERY, ROLE_MISC]
MONEY_ROLES = PERIODIC_ROLES + [ROLE_DEPOSIT]
ALL_ROLES = [ROLE_IGNORE] + IDENTITY_ROLES + MONEY_ROLES

# Roles that may be assigned to at most one column. Recoveries are deliberately NOT
# here -- three separate recovery columns is the normal case and the bug this fixes.
SINGLE_ROLES = set(IDENTITY_ROLES) | {ROLE_DEPOSIT}

# --- How to read a money figure. The period is the half the header cannot tell us. ---
BASIS_MONTHLY = 'monthly_amount'
BASIS_ANNUAL = 'annual_amount'
BASIS_MONTHLY_PSF = 'monthly_per_sf'
BASIS_ANNUAL_PSF = 'annual_per_sf'
ALL_BASES = [BASIS_MONTHLY, BASIS_ANNUAL, BASIS_MONTHLY_PSF, BASIS_ANNUAL_PSF]

BASIS_LABELS = {
    BASIS_MONTHLY: 'Monthly amount ($/month)',
    BASIS_ANNUAL: 'Annual amount ($/year)',
    BASIS_MONTHLY_PSF: 'Monthly rate ($/SF/month)',
    BASIS_ANNUAL_PSF: 'Annual rate ($/SF/year)',
}

ROLE_LABELS = {
    ROLE_IGNORE: 'Ignore',
    ROLE_TENANT: 'Tenant name',
    ROLE_SUITE: 'Suite / unit',
    ROLE_SF: 'Square feet',
    ROLE_LEASE_TYPE: 'Lease type',
    ROLE_LEASE_START: 'Lease start',
    ROLE_LEASE_END: 'Lease end',
    ROLE_DEPOSIT: 'Security deposit',
    ROLE_BASE_RENT: 'Base rent',
    ROLE_RECOVERY: 'Recovery',
    ROLE_MISC: 'Other charge',
}


# ---------------------------------------------------------------------------
# Row / label vocabulary
# ---------------------------------------------------------------------------

# A row whose name matches any of these is a subtotal, not a tenant. The old check
# was an exact-match tuple plus a short substring list, so "Sub-total for Building:
# 925  Market @ Poplar" and "Grand Total for Report" both imported as tenants --
# 229,722 SF each, against a real 228,122 leased.
# These openings are unambiguous -- nothing is named "Sub-total ...".
_TOTAL_PREFIX_RE = re.compile(
    r'(?i)^\s*\**\s*('
    r'sub\s*-?\s*total|grand\s*total|report\s*total|'
    r'building\s*total|property\s*total|summary'
    r')\b'
)
# A bare "Total" is only a total when nothing but a summary word follows it.
# Matching "total" as a prefix would throw out Total Wine & More, a real retail
# tenant; \b keeps Totally Nails as well.
_TOTAL_BARE_RE = re.compile(
    r'(?i)^\s*\**\s*totals?\b\s*'
    r'(charges?|amounts?|area|units?|rent|sf|gla|for\b.*|of\b.*|:.*)?'
    r'\s*\**\s*$'
)
# Building/section banner rows carry a name and nothing else -- no area, no charges.
_BANNER_HINT_RE = re.compile(r'(?i)^\s*(building|property|floor|section)\s*[:#]')

_RECOVERY_PATTERNS = [
    r'recover', r'reimburs', r'\brec\b', r'\brec\.',
    r'\bcam\b', r'common\s*area',
    r'\binsurance\b', r'\bins\b',
    r'\btaxe?s?\b', r'real\s*estate\s*tax', r'\bre\s*tax\b', r'property\s*tax',
    r'\bnnn\b', r'\bopex\b', r'operating\s*expense',
    r'expense\s*stop', r'pass[\s\-]*thr', r'escalation',
    r'admin(istrative)?\s*fee',
]
_RECOVERY_EXCLUDE = [
    'non-recoverable', 'nonrecoverable', 'non recoverable', 'not recoverable',
    'misc', 'base rent', 'annual rent', 'market rent', 'expected rent',
    'deposit', 'balance', 'sales tax', 'tax id', 'tax parcel', 'tax year',
    'date', 'expir',
]

_BASE_RENT_PATTERNS = [
    r'base\s*rent', r'minimum\s*(monthly\s*)?rent', r'\bmin\s*rent',
    r'scheduled\s*base', r'potential\s*base', r'contract\s*rent',
    r'^\s*rent\s*$', r'current\s*rent', r'actual\s*rent',
]
_BASE_RENT_EXCLUDE = ['market', 'expected', 'increase', 'recover', 'psf', 'per sf']

_MISC_PATTERNS = [
    r'\bmisc', r'other\s*charge', r'non[\s\-]*recoverable', r'utilit',
    r'percentage\s*rent', r'storage', r'sign(age)?\s*rent', r'parking',
]

_DATE_TOKEN_RE = re.compile(
    r'^\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}$|^\d{4}-\d{2}-\d{2}|^\d{1,2}/\d{1,2}/\d{2,4}$'
)
_MONEY_TOKEN_RE = re.compile(r'^\(?\$?-?[\d,]+(\.\d+)?\)?$')


def _norm(s: Any) -> str:
    return str(s or '').replace('\n', ' ').replace('\r', ' ').lower().strip()


def _matches(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def is_total_row(name: Any) -> bool:
    """True when a tenant-name cell is really a subtotal / grand total line."""
    s = str(name or '').strip()
    if not s:
        return False
    return bool(_TOTAL_PREFIX_RE.search(s) or _TOTAL_BARE_RE.search(s))


def _is_na(val: Any) -> bool:
    """True for None, NaN and NaT alike.

    ``isinstance(val, float) and pd.isna(val)`` misses pd.NaT, which is not a float
    but does carry a strftime attribute that raises when called.
    """
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _has_value(v: Any) -> bool:
    if _is_na(v):
        return False
    return str(v).strip() not in ('', 'nan', 'NaT', 'None')


def _is_banner_row(row: pd.Series, numeric_cols: List[Any],
                   date_cols: List[Any]) -> bool:
    """A section banner names a building, not a tenant: no figure, no date anywhere.

    "Market @ Poplar" sits above the tenants with its building id and city dropped
    into whatever columns happened to be free, so a blank-cell test is not enough --
    but it carries no area, no charge and no lease date, and a real tenant row
    always carries at least one. A vacant unit still reports its area and vacancy
    date, so this does not catch one.
    """
    if any(_has_value(row.get(c)) for c in numeric_cols):
        return False
    for c in date_cols:
        v = row.get(c)
        # _has_value FIRST: pd.NaT carries a strftime attribute, so an empty date
        # cell reads as a real date if the type is tested before the value.
        if not _has_value(v):
            continue
        if hasattr(v, 'strftime') or _DATE_TOKEN_RE.match(str(v).strip()):
            return False
    return True


def _safe_float(val: Any) -> float:
    if _is_na(val):
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    neg = s.startswith('(') and s.endswith(')')
    s = s.strip('()').replace(',', '').replace('$', '').strip()
    if s.endswith('-'):          # trailing-minus notation
        neg, s = True, s[:-1]
    try:
        v = float(s)
    except (TypeError, ValueError):
        return 0.0
    return -v if neg else v


def _safe_date(val: Any) -> Optional[str]:
    if _is_na(val):
        return None
    if hasattr(val, 'strftime'):
        return val.strftime('%Y-%m-%d')
    s = str(val).strip()
    if not s or s.lower() in ('none', 'nan', 'nat', 'tbd', '-', '--'):
        return None
    try:
        return pd.to_datetime(s).strftime('%Y-%m-%d')
    except Exception:
        return s


# ---------------------------------------------------------------------------
# Proposals — a starting point for the analyst, never the final answer
# ---------------------------------------------------------------------------

def _propose_basis(label: str) -> Optional[str]:
    """Read the period off the header, or return None when the header does not say.

    None is the honest answer for a bare "Base Rent" or "CAM" and is what drives the
    analyst prompt. Guessing here is exactly the failure this module exists to stop.
    """
    t = _norm(label)
    per_sf = bool(re.search(r'per\s*(sf|s\.f\.|area|sq)|/\s*sf|\bpsf\b|\brate\b', t))
    monthly = bool(re.search(r'month|\bmo\b|/\s*mo\b', t))
    annual = bool(re.search(r'annual|yearly|per\s*year|/\s*yr|\byr\b|\bpa\b', t))
    if monthly and annual:
        return None
    if per_sf:
        if monthly:
            return BASIS_MONTHLY_PSF
        if annual:
            return BASIS_ANNUAL_PSF
        return None
    if monthly:
        return BASIS_MONTHLY
    if annual:
        return BASIS_ANNUAL
    return None


def _propose_role(label: str, kind: str) -> str:
    t = _norm(label)
    if not t:
        return ROLE_IGNORE
    if kind == 'date':
        if re.search(r'expir|end|termin|maturity|\bto\b', t):
            return ROLE_LEASE_END
        if re.search(r'start|commence|begin|\bfrom\b', t):
            return ROLE_LEASE_START
        return ROLE_IGNORE
    if re.search(r'tenant|lessee|\bdba\b|occupant', t) and 'group' not in t:
        return ROLE_TENANT
    if re.search(r'suite|unit|space', t) and 'type' not in t:
        return ROLE_SUITE
    if re.search(r'lease\s*type|unit\s*type|\btype\b|status', t):
        return ROLE_LEASE_TYPE
    if re.search(r'area|sq\s*ft|sqft|square|\bsf\b|\bgla\b|footage', t):
        if re.search(r'override|usable', t):
            return ROLE_IGNORE          # rentable/leased is the underwriting basis
        return ROLE_SF
    if re.search(r'deposit', t):
        return ROLE_DEPOSIT
    if _matches(t, _BASE_RENT_PATTERNS) and not any(x in t for x in _BASE_RENT_EXCLUDE):
        return ROLE_BASE_RENT
    if _matches(t, _RECOVERY_PATTERNS) and not any(x in t for x in _RECOVERY_EXCLUDE):
        return ROLE_RECOVERY
    if _matches(t, _MISC_PATTERNS):
        return ROLE_MISC
    return ROLE_IGNORE


def _column_kind(values: List[Any]) -> str:
    """Classify a column from its values, so an empty header cannot mislabel it."""
    seen = [v for v in values if v is not None and str(v).strip() not in ('', 'nan')]
    if not seen:
        return 'empty'
    dates = sum(1 for v in seen if hasattr(v, 'strftime')
                or _DATE_TOKEN_RE.match(str(v).strip()))
    if dates >= max(1, int(0.6 * len(seen))):
        return 'date'
    nums = sum(1 for v in seen if _MONEY_TOKEN_RE.match(str(v).strip()))
    if nums >= max(1, int(0.6 * len(seen))):
        return 'number'
    return 'text'


# ---------------------------------------------------------------------------
# Columnar scan (Excel / CSV / tabular PDF)
# ---------------------------------------------------------------------------

def _scan_columnar(df_raw: pd.DataFrame) -> Dict[str, Any]:
    # Which column names the tenant? Needed to spot subtotal rows before sampling.
    name_col = None
    for c in df_raw.columns:
        if _propose_role(c, _column_kind(list(df_raw[c].head(30)))) == ROLE_TENANT:
            name_col = c
            break
    if name_col is None:
        for c in df_raw.columns:
            if _column_kind(list(df_raw[c].head(30))) == 'text':
                name_col = c
                break

    kinds = {c: _column_kind(list(df_raw[c].head(40))) for c in df_raw.columns}
    numeric_cols = [c for c, k in kinds.items() if k == 'number']
    date_cols = [c for c, k in kinds.items() if k == 'date']

    total_idx, banner_idx = [], []
    if name_col is not None:
        for i, v in df_raw[name_col].items():
            nm = str(v or '').strip()
            if not nm:
                continue
            if is_total_row(nm):
                total_idx.append(i)
            elif (_BANNER_HINT_RE.search(nm)
                  or _is_banner_row(df_raw.loc[i], numeric_cols, date_cols)):
                banner_idx.append(i)

    data_idx = [i for i in df_raw.index
                if i not in set(total_idx) and i not in set(banner_idx)]
    body = df_raw.loc[data_idx]

    columns = []
    claimed = set()
    for pos, c in enumerate(df_raw.columns):
        vals = list(body[c].head(40))
        kind = _column_kind(vals)
        if kind == 'empty':
            continue
        samples = [str(v).strip() for v in vals
                   if v is not None and str(v).strip() not in ('', 'nan')][:4]
        role = _propose_role(c, kind)
        # Tenant name, area and the lease dates can each come from one column only.
        # This file offers Usable, Rentable AND Leased Area; proposing all three
        # would hand the analyst a mapping that cannot validate. First wins, the
        # rest are left unassigned for them to switch.
        if role in SINGLE_ROLES:
            if role in claimed:
                role = ROLE_IGNORE
            else:
                claimed.add(role)
        entry = {
            'key': str(c),
            'label': str(c),
            'index': pos,
            'kind': kind,
            'samples': samples,
            'proposed_role': role,
            'proposed_basis': _propose_basis(c) if role in PERIODIC_ROLES else None,
            'needs_basis': role in PERIODIC_ROLES,
        }
        columns.append(entry)

    return {
        'layout': 'columnar',
        'columns': columns,
        'row_count': len(body),
        'excluded_rows': [
            {'index': int(i), 'name': str(df_raw.at[i, name_col]),
             'reason': 'subtotal / total row'}
            for i in total_idx
        ] + [
            {'index': int(i), 'name': str(df_raw.at[i, name_col]),
             'reason': 'building / section banner'}
            for i in banner_idx
        ],
    }


# ---------------------------------------------------------------------------
# Stacked scan (MRI "Master Rent Roll" PDF — charges are rows, not columns)
# ---------------------------------------------------------------------------

_STACK_HEADER_WORDS = ['floor/unit', 'tenant', 'lease', 'area', 'charge', 'amount']

# The running header/footer repeated at the top of every page.
_PAGE_FURNITURE_RE = re.compile(
    r'(?i)^\s*(building\s*:|master\s+rent\s+roll|byfloor/unit)|as\s*of\s*:|\bpage\s+\d+\s*$'
)


# Words on one printed line can differ by a fraction of a point. Rounding to the
# nearest integer splits a line straddling x.5; the row pitch here is ~11pt, so a
# 3pt tolerance groups a line without ever merging two.
_LINE_TOL = 3.0


def _line_words(page) -> List[Tuple[float, List[dict]]]:
    groups: List[Tuple[float, List[dict]]] = []
    for w in sorted(page.extract_words(), key=lambda x: (x['top'], x['x0'])):
        if groups and abs(w['top'] - groups[-1][0]) <= _LINE_TOL:
            groups[-1][1].append(w)
        else:
            groups.append((w['top'], [w]))
    return [(t, sorted(ws, key=lambda x: x['x0'])) for t, ws in groups]


def _stack_bands(words: List[dict]) -> Optional[Dict[str, float]]:
    """Derive the column bands from the page's own header line.

    Hardcoding x positions would break on the next property; the header row states
    where its columns are, so read it from there.
    """
    pos = {}
    for w in words:
        pos.setdefault(w['text'].lower().strip('/:'), w)
    need = ['tenant', 'lease', 'area', 'charge', 'amount']
    if not all(k in pos for k in need):
        return None
    leases = [w for w in words if w['text'].lower() == 'lease']
    if len(leases) < 2:
        return None
    rates = [w for w in words if w['text'].lower() == 'rate']
    amounts = [w for w in words if w['text'].lower() == 'amount']
    return {
        'tenant': pos['tenant']['x0'] - 4,
        'date': leases[0]['x0'] - 4,
        'area': pos['area']['x0'] - 4,
        'charge': pos['charge']['x0'] - 12,
        'amount_end': (amounts[0]['x1'] + rates[0]['x0']) / 2 if rates else 1e9,
        'rate_end': (rates[0]['x1'] + 24) if rates else 1e9,
    }


def _split_floor_unit(text: str) -> str:
    """Drop the leading floor number from a combined "Floor/Unit" cell.

    The stacked layout prints floor and unit under one header, so the raw cell is
    "1 264 01". The suite is a match key when merging against an existing roster,
    and the columnar export of the same property gives "264 01" from its own
    Unit # column -- the two must agree or every tenant looks new.
    """
    parts = text.split()
    if len(parts) > 1 and parts[0].isdigit() and len(parts[0]) <= 2:
        return ' '.join(parts[1:])
    return text.strip()


def _read_stacked_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """Walk the stacked layout into tenant blocks plus their charge lines."""
    import pdfplumber

    tenants: List[Dict[str, Any]] = []
    charge_labels: Dict[str, Dict[str, Any]] = {}
    building_totals: List[Dict[str, Any]] = []
    skipped = {'increase': 0, 'vacant': 0, 'other': 0}
    bands = None
    in_totals = False
    current: Optional[Dict[str, Any]] = None

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            for _top, words in _line_words(page):
                text = ' '.join(w['text'] for w in words).strip()
                low = text.lower()

                if any(k in low for k in _STACK_HEADER_WORDS[:2]) and 'floor/unit' in low:
                    b = _stack_bands(words)
                    if b:
                        bands = b
                    continue
                if bands is None:
                    continue

                # Page furniture. A tenant's charge lines routinely continue across a
                # page break, so this must NOT clear `current` -- treating the running
                # header as an unrecognised identity line dropped every charge for the
                # two tenants whose block straddled a break.
                if _PAGE_FURNITURE_RE.search(low):
                    continue

                if low.startswith('increase:'):
                    skipped['increase'] += 1        # a future step-up, not today's rent
                    continue
                if 'totals for building' in low:
                    in_totals = True
                # Everything from the building totals banner onward is the summary
                # block. Its per-charge breakdown lines look exactly like a tenant's
                # charge lines, so capture has to STOP here rather than filter -- the
                # 227,475 area line would otherwise register as an unnamed charge.
                if in_totals:
                    money = [w for w in words if _MONEY_TOKEN_RE.match(w['text'])]
                    label = ' '.join(w['text'] for w in words
                                     if not _MONEY_TOKEN_RE.match(w['text']))
                    building_totals.append({
                        'label': re.sub(r'[\*\$]', '', label).strip(),
                        'values': [_safe_float(w['text']) for w in money],
                    })
                    current = None
                    continue

                leftmost = words[0]['x0']
                is_identity = leftmost < bands['date']

                if is_identity:
                    dates = [w for w in words
                             if _DATE_TOKEN_RE.match(w['text'])
                             and bands['date'] <= w['x0'] < bands['area']]
                    is_vac = ('*** vacant ***' in low
                              or 'expected rent' in low
                              or bool(re.search(r'\bsince\b', low)))
                    if is_vac and len(dates) < 2:
                        unit = ' '.join(w['text'] for w in words
                                        if w['x0'] < bands['tenant'])
                        area = [w for w in words if bands['area'] <= w['x0']
                                < bands['charge'] and _MONEY_TOKEN_RE.match(w['text'])]
                        skipped['vacant'] += 1
                        tenants.append({
                            'suite': _split_floor_unit(unit), 'tenant_name': 'Vacant',
                            'lease_start': None, 'lease_end': None,
                            'square_feet': _safe_float(area[0]['text']) if area else 0.0,
                            'security_deposit': 0.0, 'charges': {},
                            'stated_total': None,
                        })
                        current = None
                        continue
                    if len(dates) < 2:
                        skipped['other'] += 1   # page header/footer
                        current = None
                        continue
                    unit = ' '.join(w['text'] for w in words
                                    if w['x0'] < bands['tenant'])
                    name = ' '.join(w['text'] for w in words
                                    if bands['tenant'] <= w['x0'] < bands['date'])
                    area = [w for w in words
                            if bands['area'] <= w['x0'] < bands['charge']]
                    deposit = [w for w in words if w['x1'] > bands['rate_end']]
                    current = {
                        'suite': _split_floor_unit(unit),
                        'tenant_name': name.strip(),
                        'lease_start': _safe_date(dates[0]['text']),
                        'lease_end': _safe_date(dates[1]['text']),
                        'square_feet': _safe_float(area[0]['text']) if area else 0.0,
                        'security_deposit': (_safe_float(deposit[-1]['text'])
                                             if deposit else 0.0),
                        'charges': {},
                        'stated_total': None,
                    }
                    tenants.append(current)
                    continue

                # A continuation line: a charge, a tenant total, or an override note.
                if current is None:
                    continue
                if low.startswith('override area leased'):
                    n = re.search(r'([\d,]+)\s*$', text)
                    if n:
                        current['square_feet'] = _safe_float(n.group(1))
                    continue
                money = [w for w in words if _MONEY_TOKEN_RE.match(w['text'])
                         and w['x0'] >= bands['charge']]
                label = ' '.join(w['text'] for w in words
                                 if not (_MONEY_TOKEN_RE.match(w['text'])
                                         and w['x0'] >= bands['charge'])).strip()
                if not money:
                    continue
                amount = _safe_float(money[0]['text'])
                rate = _safe_float(money[1]['text']) if len(money) > 1 else None
                clean = re.sub(r'^\*+\s*|\s*\*+$', '', label).strip()
                if 'tenant total' in clean.lower():
                    current['stated_total'] = amount
                    continue
                key = clean
                current['charges'][key] = current['charges'].get(key, 0.0) + amount
                slot = charge_labels.setdefault(key, {
                    'key': key, 'label': key, 'count': 0,
                    'samples': [], 'rate_samples': [],
                })
                slot['count'] += 1
                if len(slot['samples']) < 4:
                    slot['samples'].append(f'{amount:,.2f}')
                    if rate is not None:
                        slot['rate_samples'].append(f'{rate:,.2f}')

    return {
        'tenants': tenants,
        'charge_labels': charge_labels,
        'building_totals': building_totals,
        'skipped': skipped,
    }


def _scan_stacked(file_bytes: bytes) -> Dict[str, Any]:
    read = _read_stacked_pdf(file_bytes)
    charges = []
    for key, slot in read['charge_labels'].items():
        role = _propose_role(key, 'number')
        charges.append({
            'key': key,
            'label': key,
            'kind': 'number',
            'count': slot['count'],
            'samples': slot['samples'],
            'rate_samples': slot['rate_samples'],
            'proposed_role': role if role in MONEY_ROLES else ROLE_MISC,
            'proposed_basis': _propose_basis(key),
            'needs_basis': True,
        })
    charges.sort(key=lambda c: -c['count'])
    return {
        'layout': 'stacked',
        'charges': charges,
        'row_count': len(read['tenants']),
        'excluded_rows': [],
        'skipped': read['skipped'],
        'building_totals': read['building_totals'],
    }


def _looks_stacked(file_bytes: bytes) -> bool:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = (pdf.pages[0].extract_text() or '').lower()
    except Exception:
        return False
    # A charge-per-line layout repeats a charge label under each tenant and prints a
    # per-tenant total; a tabular PDF does neither.
    return 'tenant total' in text or bool(
        re.search(r'floor/unit.*charge.*amount', text))


# ---------------------------------------------------------------------------
# Public: scan
# ---------------------------------------------------------------------------

def scan(file_bytes: bytes, filename: str = '') -> Dict[str, Any]:
    """Describe an uploaded rent roll so the analyst can classify it. Writes nothing."""
    lower = (filename or '').lower()
    if lower.endswith('.pdf') and _looks_stacked(file_bytes):
        out = _scan_stacked(file_bytes)
    else:
        from .lease_review_service import read_rent_roll_table
        df_raw = read_rent_roll_table(file_bytes, filename)
        out = _scan_columnar(df_raw)

    entries = out.get('columns') or out.get('charges') or []
    out['filename'] = filename
    out['role_options'] = [{'value': r, 'label': ROLE_LABELS[r]} for r in ALL_ROLES]
    out['basis_options'] = [{'value': b, 'label': BASIS_LABELS[b]} for b in ALL_BASES]
    out['mapping'] = {
        'roles': {e['key']: e['proposed_role'] for e in entries},
        'bases': {e['key']: e['proposed_basis'] for e in entries if e['needs_basis']},
    }
    out['unanswered'] = [
        e['key'] for e in entries
        if e['proposed_role'] in PERIODIC_ROLES and not e['proposed_basis']
    ]
    out['warnings'] = _scan_warnings(out, entries)
    return out


def _scan_warnings(out: Dict[str, Any], entries: List[Dict[str, Any]]) -> List[str]:
    warns = []
    roles = [e['proposed_role'] for e in entries]
    if ROLE_BASE_RENT not in roles:
        warns.append('No column proposed as base rent — pick one before importing.')
    recs = [e['label'] for e in entries if e['proposed_role'] == ROLE_RECOVERY]
    if len(recs) > 1:
        warns.append(f'{len(recs)} recovery columns proposed: {", ".join(recs)}. '
                     'All of them will be summed.')
    if out['unanswered']:
        warns.append(
            'These columns do not say whether they are monthly or annual, so the '
            'period must be set by hand: ' + ', '.join(out['unanswered']))
    if out.get('excluded_rows'):
        warns.append(f'{len(out["excluded_rows"])} subtotal/banner rows excluded.')
    return warns


# ---------------------------------------------------------------------------
# Public: apply
# ---------------------------------------------------------------------------

def _to_annual_psf(amount: float, basis: str, sf: float) -> Optional[float]:
    """Convert a figure on the analyst's stated basis to annual dollars per SF."""
    if basis == BASIS_ANNUAL_PSF:
        return amount
    if basis == BASIS_MONTHLY_PSF:
        return amount * 12.0
    if sf and sf > 0:
        if basis == BASIS_ANNUAL:
            return amount / sf
        if basis == BASIS_MONTHLY:
            return amount * 12.0 / sf
    return None


def _to_annual_amount(amount: float, basis: str, sf: float) -> Optional[float]:
    if basis == BASIS_ANNUAL:
        return amount
    if basis == BASIS_MONTHLY:
        return amount * 12.0
    if sf and sf > 0:
        if basis == BASIS_ANNUAL_PSF:
            return amount * sf
        if basis == BASIS_MONTHLY_PSF:
            return amount * sf * 12.0
    return None


def _validate_mapping(mapping: Dict[str, Any], keys: List[str]) -> None:
    roles = mapping.get('roles') or {}
    bases = mapping.get('bases') or {}
    assigned = [k for k, r in roles.items() if r and r != ROLE_IGNORE]
    unknown = [k for k in assigned if k not in keys]
    if unknown:
        raise ValueError(f'Mapping names columns that are not in the file: {unknown}')
    for role in SINGLE_ROLES:
        hits = [k for k, r in roles.items() if r == role]
        if len(hits) > 1:
            raise ValueError(
                f'{ROLE_LABELS[role]} is mapped to more than one column: {hits}')
    if not any(r == ROLE_TENANT for r in roles.values()):
        raise ValueError('No column is mapped to the tenant name.')
    missing_basis = [k for k, r in roles.items()
                     if r in PERIODIC_ROLES and not bases.get(k)]
    if missing_basis:
        raise ValueError(
            'Monthly or annual has not been set for: ' + ', '.join(missing_basis))
    bad = [k for k, b in bases.items() if b and b not in ALL_BASES]
    if bad:
        raise ValueError(f'Unknown period for: {bad}')


def apply_mapping(file_bytes: bytes, filename: str,
                  mapping: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a rent roll using the analyst's confirmed mapping.

    Returns the tenant frame (the schema ``import_rent_roll_to_review`` expects) and a
    report naming every decision applied, so the import can be audited afterwards
    rather than only at the moment it was clicked.
    """
    lower = (filename or '').lower()
    if lower.endswith('.pdf') and _looks_stacked(file_bytes):
        return _apply_stacked(file_bytes, mapping)
    from .lease_review_service import read_rent_roll_table
    return _apply_columnar(read_rent_roll_table(file_bytes, filename), mapping)


def _blank_row() -> Dict[str, Any]:
    return {
        'tenant_name': '', 'suite': '', 'lease_type': 'Retail', 'square_feet': 0.0,
        'lease_start': None, 'lease_end': None, 'term_months': 0,
        'monthly_rent': 0.0, 'rent_per_sf_month': 0.0,
        'annual_rent': 0.0, 'rent_per_sf_year': 0.0,
        'annual_recoveries_per_sf': 0.0, 'annual_misc_per_sf': 0.0,
        'security_deposit': 0.0, 'is_vacant': False,
    }


def _finish_row(row: Dict[str, Any], rent_annual: Optional[float],
                rec_annual: Optional[float], misc_annual: Optional[float],
                report: Dict[str, Any]) -> None:
    """Fill the derived rent fields from annual dollars, or leave them at zero.

    ``None`` here means "the basis was per-SF and the file gave no area", so the gross
    figure genuinely is not known. It stays 0 rather than being invented, and the row
    is counted in ``report['no_area']``.
    """
    sf = row['square_feet']
    if rent_annual is None:
        report['no_area'] += 1
    else:
        row['annual_rent'] = rent_annual
        row['monthly_rent'] = rent_annual / 12.0
        if sf > 0:
            row['rent_per_sf_year'] = rent_annual / sf
            row['rent_per_sf_month'] = rent_annual / sf / 12.0
    if rec_annual is not None and sf > 0:
        row['annual_recoveries_per_sf'] = rec_annual / sf
    if misc_annual is not None and sf > 0:
        row['annual_misc_per_sf'] = misc_annual / sf


def _apply_columnar(df_raw: pd.DataFrame,
                    mapping: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    keys = [str(c) for c in df_raw.columns]
    _validate_mapping(mapping, keys)
    roles: Dict[str, str] = mapping['roles']
    bases: Dict[str, str] = mapping.get('bases') or {}

    def cols_for(role: str) -> List[str]:
        return [k for k, r in roles.items() if r == role]

    name_col = cols_for(ROLE_TENANT)[0]
    one = lambda role: (cols_for(role) or [None])[0]  # noqa: E731
    suite_col, sf_col = one(ROLE_SUITE), one(ROLE_SF)
    type_col, dep_col = one(ROLE_LEASE_TYPE), one(ROLE_DEPOSIT)
    start_col, end_col = one(ROLE_LEASE_START), one(ROLE_LEASE_END)
    rent_cols, rec_cols, misc_cols = (cols_for(ROLE_BASE_RENT),
                                      cols_for(ROLE_RECOVERY), cols_for(ROLE_MISC))

    kinds = {c: _column_kind(list(df_raw[c].head(40))) for c in df_raw.columns}
    numeric_cols = [c for c, k in kinds.items() if k == 'number']
    date_cols = [c for c, k in kinds.items() if k == 'date']

    report = {
        'layout': 'columnar', 'excluded': [], 'no_area': 0,
        'base_rent_columns': rent_cols, 'recovery_columns': rec_cols,
        'misc_columns': misc_cols,
        'bases': {k: bases.get(k) for k in rent_cols + rec_cols + misc_cols},
    }

    out = []
    for i, r in df_raw.iterrows():
        raw_name = r.get(name_col)
        name = str(raw_name or '').strip()
        if not name or name.lower() in ('nan', 'none'):
            continue
        if (is_total_row(name) or _BANNER_HINT_RE.search(name)
                or _is_banner_row(r, numeric_cols, date_cols)):
            report['excluded'].append(name)
            continue
        try:                              # a purely numeric "name" is summary data
            float(name.replace(',', ''))
            report['excluded'].append(name)
            continue
        except ValueError:
            pass

        row = _blank_row()
        row['tenant_name'] = name
        row['is_vacant'] = 'VACANT' in name.upper()
        row['suite'] = str(r.get(suite_col) or '').strip() if suite_col else ''
        if type_col:
            row['lease_type'] = str(r.get(type_col) or '').strip() or 'Retail'
        row['square_feet'] = _safe_float(r.get(sf_col)) if sf_col else 0.0
        row['lease_start'] = _safe_date(r.get(start_col)) if start_col else None
        row['lease_end'] = _safe_date(r.get(end_col)) if end_col else None
        row['security_deposit'] = _safe_float(r.get(dep_col)) if dep_col else 0.0
        sf = row['square_feet']

        def total(cols):
            if not cols:
                return 0.0
            vals = [_to_annual_amount(_safe_float(r.get(c)), bases[c], sf)
                    for c in cols]
            return None if any(v is None for v in vals) else sum(vals)

        _finish_row(row, total(rent_cols), total(rec_cols), total(misc_cols), report)
        out.append(row)

    if not out:
        raise ValueError('No tenant rows found once subtotals were excluded.')
    return pd.DataFrame(out), report


def _apply_stacked(file_bytes: bytes,
                   mapping: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    read = _read_stacked_pdf(file_bytes)
    keys = list(read['charge_labels'].keys())
    roles: Dict[str, str] = mapping['roles']
    bases: Dict[str, str] = mapping.get('bases') or {}

    unknown = [k for k, r in roles.items()
               if r and r != ROLE_IGNORE and k not in keys]
    if unknown:
        raise ValueError(f'Mapping names charges that are not in the file: {unknown}')
    missing = [k for k, r in roles.items()
               if r in PERIODIC_ROLES and not bases.get(k)]
    if missing:
        raise ValueError('Monthly or annual has not been set for: ' + ', '.join(missing))

    report = {
        'layout': 'stacked', 'excluded': [], 'no_area': 0,
        'base_rent_charges': [k for k, r in roles.items() if r == ROLE_BASE_RENT],
        'recovery_charges': [k for k, r in roles.items() if r == ROLE_RECOVERY],
        'misc_charges': [k for k, r in roles.items() if r == ROLE_MISC],
        'skipped': read['skipped'],
        'tie_out': [],
    }

    out = []
    for t in read['tenants']:
        row = _blank_row()
        row.update({
            'tenant_name': t['tenant_name'], 'suite': t['suite'],
            'square_feet': t['square_feet'], 'lease_start': t['lease_start'],
            'lease_end': t['lease_end'], 'security_deposit': t['security_deposit'],
            'is_vacant': 'VACANT' in (t['tenant_name'] or '').upper(),
        })
        sf = row['square_feet']

        def total(role):
            cols = [k for k, r in roles.items() if r == role and k in t['charges']]
            if not cols:
                return 0.0
            vals = [_to_annual_amount(t['charges'][k], bases[k], sf) for k in cols]
            return None if any(v is None for v in vals) else sum(vals)

        _finish_row(row, total(ROLE_BASE_RENT), total(ROLE_RECOVERY),
                    total(ROLE_MISC), report)

        # The file states its own per-tenant total; report any disagreement.
        if t['stated_total'] is not None:
            summed = sum(t['charges'].values())
            if abs(summed - t['stated_total']) > 0.02:
                report['tie_out'].append({
                    'tenant': t['tenant_name'],
                    'stated': t['stated_total'], 'summed': round(summed, 2),
                })
        out.append(row)

    if not out:
        raise ValueError('No tenant blocks found in the PDF.')
    return pd.DataFrame(out), report
