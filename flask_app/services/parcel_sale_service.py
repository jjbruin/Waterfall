"""
parcel_sale_service.py
Interim parcel sales — sales of part of a property before the final disposition.

A deal can carry several. Each records the projected sale, what the proceeds
pay down, what is held back in the CapEx reserve, how the remainder is
distributed, and the revenue and expenses that leave with the parcel.

Phase 01 scope: persistence and validation only. Nothing here feeds the
forecast yet — the engine integration is phases 02-05 of the build plan.
Validation lives here rather than in the route so the compute path can reuse
it before acting on a sale.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Columns that round-trip as JSON blobs
_JSON_COLUMNS = ('debt_application', 'distribution_fixed',
                 'lost_revenue', 'lost_expense')

DISTRIBUTION_MODES = ('pro_rata', 'fixed', 'waterfall')

_SELECT = """
    SELECT id, vcode, label, sale_date, sale_price,
           cost_of_sale_value, cost_of_sale_type, debt_application,
           capex_reserve_hold, distribution_mode, distribution_fixed,
           lost_revenue, lost_expense, notes, sort_order,
           created_at, updated_at, updated_by
    FROM parcel_sales
"""


def _row_to_dict(row) -> Dict[str, Any]:
    d = {
        'id': row[0], 'vcode': row[1], 'label': row[2],
        'sale_date': row[3], 'sale_price': row[4],
        'cost_of_sale_value': row[5], 'cost_of_sale_type': row[6] or 'pct',
        'debt_application': row[7],
        'capex_reserve_hold': row[8], 'distribution_mode': row[9] or 'waterfall',
        'distribution_fixed': row[10],
        'lost_revenue': row[11], 'lost_expense': row[12],
        'notes': row[13], 'sort_order': row[14],
        'created_at': str(row[15]) if row[15] else None,
        'updated_at': str(row[16]) if row[16] else None,
        'updated_by': row[17],
    }
    for col in _JSON_COLUMNS:
        raw = d.get(col)
        if not raw:
            d[col] = [] if col == 'debt_application' else {}
            continue
        try:
            d[col] = json.loads(raw)
        except (TypeError, ValueError):
            # Never let one malformed blob take down the whole list
            logger.warning("parcel_sales.%s for id=%s is not valid JSON", col, d['id'])
            d[col] = [] if col == 'debt_application' else {}
    d['economics'] = compute_economics(d)
    return d


def _dump(value) -> Optional[str]:
    if value in (None, '', [], {}):
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _num(value, default=0.0) -> float:
    """Coerce a form value to a float without raising on '' or None."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Economics — the waterfall of a single parcel sale
# ---------------------------------------------------------------------------

def compute_economics(sale: Dict[str, Any]) -> Dict[str, float]:
    """Derive the money flow for one parcel sale.

    Order matches the build plan: price, less cost of sale, less debt
    paydown, less the reserve hold, and whatever remains is distributed.
    Returned alongside every sale so the UI and the engine agree on the
    arithmetic instead of each computing its own.
    """
    price = _num(sale.get('sale_price'))
    cost_val = _num(sale.get('cost_of_sale_value'))
    cost_type = (sale.get('cost_of_sale_type') or 'pct').lower()

    cost = price * (cost_val / 100.0) if cost_type == 'pct' else cost_val
    net = price - cost

    debt = sum(_num(a.get('amount')) for a in (sale.get('debt_application') or [])
               if isinstance(a, dict))
    reserve = _num(sale.get('capex_reserve_hold'))
    remainder = net - debt - reserve

    return {
        'gross_price': price,
        'cost_of_sale': cost,
        'net_proceeds': net,
        'debt_paydown': debt,
        'reserve_hold': reserve,
        'remainder': remainder,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(sale: Dict[str, Any],
             known_loan_ids: Optional[List[str]] = None,
             final_sale_date=None,
             horizon_end=None) -> Dict[str, List[str]]:
    """Check one parcel sale. Returns {'errors': [...], 'warnings': [...]}.

    Errors block a save; warnings do not. The distinction matters because an
    analyst part-way through entering a sale should not be stopped by a
    remainder that does not balance yet, but must be stopped from saving a
    paydown that exceeds the proceeds.
    """
    errors: List[str] = []
    warnings: List[str] = []
    econ = compute_economics(sale)

    # --- date ---
    raw_date = sale.get('sale_date')
    sale_ts = None
    if not raw_date:
        errors.append("Sale date is required.")
    else:
        sale_ts = pd.to_datetime(raw_date, errors='coerce')
        if pd.isna(sale_ts):
            errors.append(f"Sale date '{raw_date}' is not a valid date.")
        else:
            if final_sale_date is not None:
                final_ts = pd.to_datetime(final_sale_date, errors='coerce')
                if not pd.isna(final_ts) and sale_ts >= final_ts:
                    errors.append(
                        f"Parcel sale date ({sale_ts.date()}) must fall before the "
                        f"deal's final sale date ({final_ts.date()})."
                    )
            if horizon_end is not None:
                hz = pd.to_datetime(horizon_end, errors='coerce')
                if not pd.isna(hz) and sale_ts > hz:
                    warnings.append(
                        f"Sale date ({sale_ts.date()}) is beyond the forecast "
                        f"horizon ({hz.date()}), so it will not affect the projection."
                    )

    # --- price and cost of sale ---
    if econ['gross_price'] <= 0:
        errors.append("Sale price must be greater than zero.")
    cost_type = (sale.get('cost_of_sale_type') or 'pct').lower()
    if cost_type not in ('pct', 'fixed'):
        errors.append(f"Cost of sale type must be 'pct' or 'fixed', got '{cost_type}'.")
    if _num(sale.get('cost_of_sale_value')) < 0:
        errors.append("Cost of sale cannot be negative.")
    if cost_type == 'pct' and _num(sale.get('cost_of_sale_value')) > 100:
        errors.append("Cost of sale percentage cannot exceed 100%.")
    if econ['net_proceeds'] < 0:
        errors.append(
            f"Cost of sale (${econ['cost_of_sale']:,.0f}) exceeds the sale price "
            f"(${econ['gross_price']:,.0f})."
        )

    # --- debt paydown ---
    apps = sale.get('debt_application') or []
    if not isinstance(apps, list):
        errors.append("Debt application must be a list of {loan_id, amount}.")
        apps = []
    seen_loans = set()
    for app in apps:
        if not isinstance(app, dict):
            errors.append("Each debt application entry must be an object.")
            continue
        lid = str(app.get('loan_id') or '').strip()
        amt = _num(app.get('amount'))
        if not lid:
            errors.append("A debt application entry is missing its loan.")
            continue
        if lid in seen_loans:
            errors.append(f"Loan {lid} appears more than once in the paydown allocation.")
        seen_loans.add(lid)
        if amt < 0:
            errors.append(f"Paydown for loan {lid} cannot be negative.")
        if known_loan_ids is not None and lid not in known_loan_ids:
            errors.append(
                f"Loan {lid} is not a loan on this deal. Known loans: "
                + (", ".join(known_loan_ids) if known_loan_ids else "none")
            )

    if _num(sale.get('capex_reserve_hold')) < 0:
        errors.append("CapEx reserve hold cannot be negative.")

    # --- the proceeds must reconcile ---
    if econ['net_proceeds'] > 0 and econ['remainder'] < -1.0:
        errors.append(
            f"Debt paydown (${econ['debt_paydown']:,.0f}) plus reserve hold "
            f"(${econ['reserve_hold']:,.0f}) exceeds net proceeds of "
            f"${econ['net_proceeds']:,.0f} by ${abs(econ['remainder']):,.0f}."
        )

    # --- distribution ---
    mode = (sale.get('distribution_mode') or 'waterfall').lower()
    if mode not in DISTRIBUTION_MODES:
        errors.append(
            f"Distribution mode must be one of {', '.join(DISTRIBUTION_MODES)}, "
            f"got '{mode}'."
        )
    if mode == 'fixed':
        fixed = sale.get('distribution_fixed') or {}
        if not isinstance(fixed, dict):
            errors.append("Fixed distribution must be a map of partner to amount.")
        elif not fixed:
            errors.append("Fixed distribution selected but no partner amounts entered.")
        else:
            total = sum(_num(v) for v in fixed.values())
            if any(_num(v) < 0 for v in fixed.values()):
                errors.append("Fixed distribution amounts cannot be negative.")
            if abs(total - econ['remainder']) > 1.0:
                errors.append(
                    f"Fixed distribution totals ${total:,.0f} but the remainder to "
                    f"distribute is ${econ['remainder']:,.0f}."
                )

    # --- lost revenue / expense ---
    for key, label in (('lost_revenue', 'revenue'), ('lost_expense', 'expense')):
        block = sale.get(key) or {}
        if not isinstance(block, dict):
            errors.append(f"Lost {label} must be an object.")
            continue
        for acct, amt in (block.get('accounts') or {}).items():
            if _num(amt) < 0:
                errors.append(f"Lost {label} for account {acct} cannot be negative.")

    lost_rev = sale.get('lost_revenue') or {}
    if isinstance(lost_rev, dict):
        rev_total = sum(_num(v) for v in (lost_rev.get('accounts') or {}).values())
        exp_block = sale.get('lost_expense') or {}
        exp_total = sum(_num(v) for v in (exp_block.get('accounts') or {}).values()) \
            if isinstance(exp_block, dict) else 0.0
        if rev_total > 0 and exp_total == 0:
            warnings.append(
                "Revenue is being removed but no expenses are. Selling a parcel "
                "usually removes operating costs too, so NOI after the sale date "
                "may be overstated."
            )

    return {'errors': errors, 'warnings': warnings}


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def list_parcel_sales(engine, vcode: str) -> List[Dict[str, Any]]:
    """All parcel sales for a deal, in sale-date order."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(_SELECT + " WHERE vcode = :v ORDER BY sort_order, sale_date, id"),
            {'v': vcode},
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_parcel_sale(engine, sale_id: int) -> Optional[Dict[str, Any]]:
    with engine.connect() as conn:
        row = conn.execute(text(_SELECT + " WHERE id = :i"), {'i': sale_id}).fetchone()
    return _row_to_dict(row) if row else None


def create_parcel_sale(engine, vcode: str, data: Dict[str, Any],
                       user: Optional[str] = None) -> Dict[str, Any]:
    params = {
        'v': vcode,
        'lb': data.get('label') or 'Parcel sale',
        'sd': data.get('sale_date'),
        'sp': _num(data.get('sale_price')),
        'cv': _num(data.get('cost_of_sale_value')),
        'ct': (data.get('cost_of_sale_type') or 'pct').lower(),
        'da': _dump(data.get('debt_application')),
        'rh': _num(data.get('capex_reserve_hold')),
        'dm': (data.get('distribution_mode') or 'waterfall').lower(),
        'df': _dump(data.get('distribution_fixed')),
        'lr': _dump(data.get('lost_revenue')),
        'le': _dump(data.get('lost_expense')),
        'nt': data.get('notes'),
        'so': int(data.get('sort_order') or 0),
        'ub': user,
    }
    with engine.begin() as conn:
        new_id = conn.execute(text("""
            INSERT INTO parcel_sales
                (vcode, label, sale_date, sale_price, cost_of_sale_value,
                 cost_of_sale_type, debt_application, capex_reserve_hold,
                 distribution_mode, distribution_fixed, lost_revenue,
                 lost_expense, notes, sort_order, updated_by)
            VALUES (:v, :lb, :sd, :sp, :cv, :ct, :da, :rh, :dm, :df, :lr,
                    :le, :nt, :so, :ub)
            RETURNING id
        """), params).fetchone()[0]
    logger.info("Created parcel sale %s for %s", new_id, vcode)
    return get_parcel_sale(engine, new_id)


def update_parcel_sale(engine, sale_id: int, data: Dict[str, Any],
                       user: Optional[str] = None) -> Optional[Dict[str, Any]]:
    existing = get_parcel_sale(engine, sale_id)
    if not existing:
        return None
    merged = {**existing, **{k: v for k, v in data.items() if k != 'id'}}
    params = {
        'i': sale_id,
        'lb': merged.get('label') or 'Parcel sale',
        'sd': merged.get('sale_date'),
        'sp': _num(merged.get('sale_price')),
        'cv': _num(merged.get('cost_of_sale_value')),
        'ct': (merged.get('cost_of_sale_type') or 'pct').lower(),
        'da': _dump(merged.get('debt_application')),
        'rh': _num(merged.get('capex_reserve_hold')),
        'dm': (merged.get('distribution_mode') or 'waterfall').lower(),
        'df': _dump(merged.get('distribution_fixed')),
        'lr': _dump(merged.get('lost_revenue')),
        'le': _dump(merged.get('lost_expense')),
        'nt': merged.get('notes'),
        'so': int(merged.get('sort_order') or 0),
        'ub': user,
    }
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE parcel_sales SET
                label = :lb, sale_date = :sd, sale_price = :sp,
                cost_of_sale_value = :cv, cost_of_sale_type = :ct,
                debt_application = :da, capex_reserve_hold = :rh,
                distribution_mode = :dm, distribution_fixed = :df,
                lost_revenue = :lr, lost_expense = :le, notes = :nt,
                sort_order = :so, updated_by = :ub,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :i
        """), params)
    logger.info("Updated parcel sale %s", sale_id)
    return get_parcel_sale(engine, sale_id)


def delete_parcel_sale(engine, sale_id: int) -> bool:
    with engine.begin() as conn:
        n = conn.execute(text("DELETE FROM parcel_sales WHERE id = :i"),
                         {'i': sale_id}).rowcount
    if n:
        logger.info("Deleted parcel sale %s", sale_id)
    return bool(n)


# ---------------------------------------------------------------------------
# Context for the editor
# ---------------------------------------------------------------------------

def get_deal_loans(mri_loans_raw, vcode: str) -> List[Dict[str, Any]]:
    """Loans available to receive a paydown, for the allocation picker.

    `loan_id` is the raw string form, matching how `loans.py` builds its ids
    (`str(r["LoanID"])`), so an allocation saved here joins to the amortisation
    schedule in phase 03 without re-keying. `label` is the tidied version for
    display only.

    Balances are deliberately absent: a loan's balance on a future parcel sale
    date comes from the amortisation schedule, which is phase 03 work.
    """
    if mri_loans_raw is None or mri_loans_raw.empty:
        return []
    df = mri_loans_raw.copy()
    cols = {c.lower(): c for c in df.columns}
    vc = cols.get('vcode')
    lid = cols.get('loanid')
    if not vc or not lid:
        return []
    df = df[df[vc].astype(str).str.strip().str.upper() == str(vcode).strip().upper()]
    if df.empty:
        return []

    c_prop = cols.get('vpropertyname')
    c_amt = cols.get('morigloanamt')
    c_mat = cols.get('dtmaturity')
    c_rate = cols.get('nrate')
    c_type = cols.get('vinttype')

    def _val(row, col):
        if not col or col not in row.index:
            return None
        v = row[col]
        return None if pd.isna(v) else v

    out = []
    seen = set()
    for _, r in df.iterrows():
        raw_id = str(r[lid]).strip()
        if not raw_id or raw_id.lower() == 'nan' or raw_id in seen:
            continue
        seen.add(raw_id)

        # "249.0" -> "249" for display, while the stored id stays raw
        display_id = raw_id
        try:
            f = float(raw_id)
            if f.is_integer():
                display_id = str(int(f))
        except (TypeError, ValueError):
            pass

        amt = _val(r, c_amt)
        prop = _val(r, c_prop)
        mat = _val(r, c_mat)
        rate = _val(r, c_rate)

        bits = [f"Loan {display_id}"]
        if prop:
            bits.append(str(prop).strip())
        if amt:
            bits.append(f"${float(amt):,.0f}")
        if mat:
            bits.append(f"matures {str(mat)[:10]}")

        out.append({
            'loan_id': raw_id,
            'label': " \u00b7 ".join(bits),
            'property_name': str(prop).strip() if prop else None,
            'orig_amount': float(amt) if amt else None,
            'rate': float(rate) if rate is not None else None,
            'int_type': str(_val(r, c_type)).strip() if _val(r, c_type) else None,
            'maturity': str(mat)[:10] if mat else None,
        })

    # Largest loan first — the likely paydown target
    out.sort(key=lambda x: (x['orig_amount'] or 0), reverse=True)
    return out
