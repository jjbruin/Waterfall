"""
scenario_service.py
Scenario analysis for New Business deals.

A scenario is a named binding of three things, saved per prospect deal:
a cash flow source (which Argus import per property, or the default
cascade), assumption overrides, and income-adjustment events. The Prospect
Deal Analysis page selects one from a dropdown and runs the full waterfall
through it.
"""

import json
import logging
from datetime import date as _date
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_JSON_COLS = ('argus_import_ids', 'assumption_overrides', 'adjustments')

_SELECT = """
    SELECT id, prospect_id, name, description, is_base, argus_import_ids,
           assumption_overrides, adjustments, sort_order,
           created_at, updated_at, updated_by
    FROM prospect_scenarios
"""


def _row(r) -> Dict[str, Any]:
    d = {
        'id': r[0], 'prospect_id': r[1], 'name': r[2], 'description': r[3],
        'is_base': bool(r[4]), 'argus_import_ids': r[5],
        'assumption_overrides': r[6], 'adjustments': r[7],
        'sort_order': r[8],
        'created_at': str(r[9]) if r[9] else None,
        'updated_at': str(r[10]) if r[10] else None,
        'updated_by': r[11],
    }
    for c in _JSON_COLS:
        raw = d.get(c)
        default: Any = [] if c == 'adjustments' else {}
        if not raw:
            d[c] = default
            continue
        try:
            d[c] = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("prospect_scenarios.%s for id=%s is not valid JSON", c, d['id'])
            d[c] = default
    return d


def _dump(v) -> Optional[str]:
    if v in (None, '', [], {}):
        return None
    return v if isinstance(v, str) else json.dumps(v)


def list_scenarios(engine, prospect_id: int) -> List[Dict[str, Any]]:
    with engine.connect() as c:
        rows = c.execute(text(_SELECT + " WHERE prospect_id = :p ORDER BY sort_order, id"),
                         {'p': prospect_id}).fetchall()
    return [_row(r) for r in rows]


def get_scenario(engine, scenario_id: int) -> Optional[Dict[str, Any]]:
    with engine.connect() as c:
        r = c.execute(text(_SELECT + " WHERE id = :i"), {'i': scenario_id}).fetchone()
    return _row(r) if r else None


def create_scenario(engine, prospect_id: int, data: Dict[str, Any],
                    user: Optional[str] = None) -> Dict[str, Any]:
    with engine.begin() as c:
        new_id = c.execute(text("""
            INSERT INTO prospect_scenarios
                (prospect_id, name, description, is_base, argus_import_ids,
                 assumption_overrides, adjustments, sort_order, updated_by)
            VALUES (:p, :n, :d, :b, :ai, :ao, :aj, :so, :ub)
            RETURNING id
        """), {
            'p': prospect_id,
            'n': data.get('name') or 'Scenario',
            'd': data.get('description'),
            'b': bool(data.get('is_base')),
            'ai': _dump(data.get('argus_import_ids')),
            'ao': _dump(data.get('assumption_overrides')),
            'aj': _dump(data.get('adjustments')),
            'so': int(data.get('sort_order') or 0),
            'ub': user,
        }).fetchone()[0]
    logger.info("Created scenario %s for prospect %s", new_id, prospect_id)
    return get_scenario(engine, new_id)


def update_scenario(engine, scenario_id: int, data: Dict[str, Any],
                    user: Optional[str] = None) -> Optional[Dict[str, Any]]:
    existing = get_scenario(engine, scenario_id)
    if not existing:
        return None
    merged = {**existing, **{k: v for k, v in data.items() if k != 'id'}}
    with engine.begin() as c:
        c.execute(text("""
            UPDATE prospect_scenarios SET
                name = :n, description = :d, is_base = :b,
                argus_import_ids = :ai, assumption_overrides = :ao,
                adjustments = :aj, sort_order = :so, updated_by = :ub,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :i
        """), {
            'i': scenario_id,
            'n': merged.get('name') or 'Scenario',
            'd': merged.get('description'),
            'b': bool(merged.get('is_base')),
            'ai': _dump(merged.get('argus_import_ids')),
            'ao': _dump(merged.get('assumption_overrides')),
            'aj': _dump(merged.get('adjustments')),
            'so': int(merged.get('sort_order') or 0),
            'ub': user,
        })
    return get_scenario(engine, scenario_id)


def delete_scenario(engine, scenario_id: int) -> bool:
    with engine.begin() as c:
        n = c.execute(text("DELETE FROM prospect_scenarios WHERE id = :i"),
                      {'i': scenario_id}).rowcount
    return bool(n)


# ---------------------------------------------------------------------------
# Downside seeding from the linked lease review
# ---------------------------------------------------------------------------

def get_risk_candidates(engine, prospect_id: int) -> List[Dict[str, Any]]:
    """Tenants worth a downside scenario, from the deal's lease reviews.

    Lease reviews link back to prospect properties via
    lease_reviews.prospect_property_id. A tenant
    is a candidate when it carries a termination option, a cotenancy clause,
    or is simply material -- the analyst decides which become adjustments.
    Suggested dates: the earliest termination date where one exists,
    otherwise the lease end. Rent comes from the reviewed roster, so the
    seeded adjustment reflects due-diligence numbers, not the seller's.
    """
    with engine.connect() as c:
        review_ids = [r[0] for r in c.execute(text("""
            SELECT DISTINCT lr.id
            FROM lease_reviews lr
            JOIN prospect_properties pp ON pp.id = lr.prospect_property_id
            WHERE pp.prospect_id = :p
        """), {'p': prospect_id})]
        if not review_ids:
            return []

        _today = _date.today().isoformat()
        out: List[Dict[str, Any]] = []
        for rid in review_ids:
            rows = c.execute(text("""
                SELECT t.id, t.tenant_name, t.suite, t.square_feet,
                       t.annual_rent, t.lease_end, t.is_material,
                       t.has_cotenancy, t.has_exclusive_use, t.is_vacant
                FROM lease_tenants t
                WHERE t.review_id = :r
                ORDER BY COALESCE(t.annual_rent, 0) DESC
            """), {'r': rid}).fetchall()

            term_dates = {}
            for tr in c.execute(text("""
                SELECT tenant_id, MIN(COALESCE(option_start, notice_deadline))
                FROM lease_options
                WHERE option_type = 'termination'
                  AND tenant_id IN (SELECT id FROM lease_tenants WHERE review_id = :r)
                GROUP BY tenant_id
            """), {'r': rid}):
                term_dates[tr[0]] = tr[1]

            cot_refs = {}
            for cr in c.execute(text("""
                SELECT cr.referenced_tenant_name, COUNT(DISTINCT c.tenant_id)
                FROM lease_cotenancy_refs cr
                JOIN lease_cotenancy c ON c.id = cr.cotenancy_id
                WHERE c.review_id = :r
                GROUP BY cr.referenced_tenant_name
            """), {'r': rid}):
                cot_refs[str(cr[0] or '').strip().lower()] = cr[1]

            for t in rows:
                (tid, name, suite, sf, rent, lease_end,
                 material, cotenancy, exclusive, vacant) = t
                if vacant or not (rent or 0):
                    continue
                dependents = cot_refs.get(str(name or '').strip().lower(), 0)
                term = term_dates.get(tid)
                if not (term or cotenancy or dependents or material):
                    continue
                reasons = []
                if term:
                    reasons.append(f"termination right ({str(term)[:10]})")
                if dependents:
                    reasons.append(f"named co-tenant for {dependents} lease(s)")
                if cotenancy:
                    reasons.append("has cotenancy clause")
                if material and not reasons:
                    reasons.append("material lease")
                out.append({
                    'review_id': rid, 'tenant_id': tid,
                    'tenant_name': name, 'suite': suite,
                    'square_feet': sf, 'annual_rent': rent,
                    'lease_end': str(lease_end)[:10] if lease_end else None,
                    'termination_date': str(term)[:10] if term else None,
                    'cotenancy_dependents': dependents,
                    'reasons': reasons,
                    # A termination window already in the past is history,
                    # not a scenario -- suggest the lease end instead
                    'suggested_start': (str(term)[:10]
                                        if term and str(term)[:10] >= _today
                                        else (str(lease_end)[:10] if lease_end else None)),
                })
        return out
