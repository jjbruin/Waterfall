"""
ppi_stack_service.py
The PPI ownership stack for a New Business deal.

Peaceable raises outside capital for part of each preferred equity
investment. The PE vehicle's position can be shared pro-rata across one or
more INVESTOR RELATIONSHIPS, each with its own waterfall between Peaceable
(PSC) and its investor(s). This service owns the declaration of that stack:

  prospect_entities  role='pe_vehicle'        the deal-level PE participant
  prospect_entities  role='ppi_relationship'  one row per relationship,
                                              parent_entity_id -> vehicle,
                                              ownership_pct = pro-rata slice,
                                              terms_json = the term sheet
  prospect_investors                          participants per relationship
                                              (PSC co-invest + investors)

terms_json shape (all percentages as plain numbers, 0.95 = 0.95%):
  {
    "am_fee_pct": 0.95,          # annual, on funded capital net of ROC
    "fee_periods_per_year": 4,   # quarterly by default
    "pref_rate_pct": null,       # optional CF pref to the investor(s)
    "min_irr_pct": 9.0,          # investor IRR gate (net of fees) for promote
    "promote_pct": 20.0,         # PSC's share after the gate
    "promote_shared_pct": 0.0,   # portion of the promote shared w/ investors
    "existing_entity": false     # true = linked to an existing AM entity
  }

The waterfall steps themselves are generated in phase 2 and stored in the
shared `waterfalls` table keyed by the relationship's entity id.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

VEHICLE_ROLE = 'pe_vehicle'
RELATIONSHIP_ROLE = 'ppi_relationship'

_TERM_DEFAULTS = {
    'am_fee_pct': 0.0,
    'fee_periods_per_year': 4,
    'pref_rate_pct': None,
    'min_irr_pct': None,
    'promote_pct': 0.0,
    'promote_shared_pct': 0.0,
    'existing_entity': False,
}


def _load_terms(blob) -> Dict[str, Any]:
    try:
        data = json.loads(blob) if isinstance(blob, str) else (blob or {})
    except (TypeError, ValueError):
        data = {}
    out = dict(_TERM_DEFAULTS)
    out.update({k: v for k, v in (data or {}).items() if k in _TERM_DEFAULTS})
    return out


def placeholder_entity_id(prospect_id: int, n: int) -> str:
    """Relationship id used until MRI assigns the real one at closing.

    The NR prefix marks it as a New-business Relationship placeholder; the
    onboarding wizard re-keys these the way it re-keys the deal's N-vcode.
    """
    return f"NR{prospect_id:05d}-{n}"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_stack(engine, prospect_id: int) -> Dict[str, Any]:
    """The declared stack: vehicle + relationships with their participants."""
    with engine.connect() as c:
        ents = [dict(r) for r in c.execute(text(
            "SELECT id, entity_name, entity_type, planned_entity_id, "
            "parent_entity_id, ownership_pct, role, notes, terms_json "
            "FROM prospect_entities WHERE prospect_id = :p ORDER BY id"),
            {'p': prospect_id}).mappings()]
        inv_rows = [dict(r) for r in c.execute(text(
            "SELECT pi.id, pi.entity_id, pi.investor_name, "
            "pi.planned_investor_id, pi.ownership_pct, pi.commitment, "
            "pi.investor_type, pi.notes "
            "FROM prospect_investors pi "
            "JOIN prospect_entities pe ON pe.id = pi.entity_id "
            "WHERE pe.prospect_id = :p ORDER BY pi.id"),
            {'p': prospect_id}).mappings()]

    by_entity: Dict[int, List[dict]] = {}
    for r in inv_rows:
        by_entity.setdefault(r['entity_id'], []).append(r)

    vehicle = next((e for e in ents if e.get('role') == VEHICLE_ROLE), None)
    relationships = []
    for e in ents:
        if e.get('role') != RELATIONSHIP_ROLE:
            continue
        relationships.append({
            'id': e['id'],
            'name': e.get('entity_name'),
            'entity_id': e.get('planned_entity_id'),
            'slice_pct': float(e.get('ownership_pct') or 0),
            'terms': _load_terms(e.get('terms_json')),
            'participants': [{
                'id': p['id'],
                'name': p.get('investor_name'),
                'investor_id': p.get('planned_investor_id'),
                'share_pct': float(p.get('ownership_pct') or 0),
                'commitment': p.get('commitment'),
                'type': p.get('investor_type') or 'lp',
            } for p in by_entity.get(e['id'], [])],
        })

    return {
        'vehicle': ({
            'id': vehicle['id'],
            'name': vehicle.get('entity_name'),
            'entity_id': vehicle.get('planned_entity_id'),
        } if vehicle else None),
        'relationships': relationships,
    }


def list_existing_entities(engine) -> List[Dict[str, Any]]:
    """AM entities a relationship can link to: anything that already has a
    waterfall or appears in the ownership relationships."""
    out, seen = [], set()
    with engine.connect() as c:
        for (eid,) in c.execute(text(
                'SELECT DISTINCT vcode FROM waterfalls ORDER BY vcode')):
            e = str(eid or '').strip()
            if e and e not in seen and not e.upper().startswith(('N', 'P0')):
                seen.add(e)
                out.append({'entity_id': e, 'source': 'waterfall'})
        try:
            for (eid,) in c.execute(text(
                    'SELECT DISTINCT "InvestorID" FROM relationships')):
                e = str(eid or '').strip()
                if e and e not in seen:
                    seen.add(e)
                    out.append({'entity_id': e, 'source': 'relationships'})
        except Exception as e:
            logger.debug("relationships listing unavailable: %s", e)
    out.sort(key=lambda x: x['entity_id'])
    return out


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_stack(stack: Dict[str, Any]) -> Dict[str, List[str]]:
    """Errors block a save; warnings do not."""
    errors: List[str] = []
    warnings: List[str] = []
    rels = stack.get('relationships') or []
    if not rels:
        return {'errors': errors, 'warnings': warnings}

    slice_total = sum(float(r.get('slice_pct') or 0) for r in rels)
    if abs(slice_total - 100.0) > 0.5:
        (errors if slice_total > 100.5 else warnings).append(
            f"Relationship slices total {slice_total:g}% of the PE vehicle "
            f"(expected 100%)."
        )

    names = [str(r.get('name') or '').strip() for r in rels]
    if len(set(n for n in names if n)) < len([n for n in names if n]):
        errors.append("Two relationships share the same name.")

    for r in rels:
        label = r.get('name') or r.get('entity_id') or 'relationship'
        parts = r.get('participants') or []
        terms = r.get('terms') or {}

        if float(r.get('slice_pct') or 0) <= 0:
            errors.append(f"{label}: pro-rata slice % is required.")
        if not parts:
            errors.append(f"{label}: no participants declared.")
            continue

        share_total = sum(float(p.get('share_pct') or 0) for p in parts)
        if abs(share_total - 100.0) > 0.5:
            errors.append(
                f"{label}: participant shares total {share_total:g}% "
                f"(must be 100%)."
            )
        psc = [p for p in parts if (p.get('type') or '').lower() == 'psc']
        if not psc:
            warnings.append(
                f"{label}: no PSC co-invest participant — fees and promote "
                f"have no recipient until one is added."
            )
        elif len(psc) > 1:
            errors.append(f"{label}: more than one PSC participant.")

        fee = float(terms.get('am_fee_pct') or 0)
        if fee < 0 or fee > 5:
            errors.append(f"{label}: AM fee {fee:g}% is outside 0–5%.")
        per = terms.get('fee_periods_per_year')
        if per not in (None, 1, 2, 4, 12):
            errors.append(f"{label}: fee periods/year must be 1, 2, 4 or 12.")

        promote = float(terms.get('promote_pct') or 0)
        min_irr = terms.get('min_irr_pct')
        if promote > 0 and not min_irr:
            errors.append(
                f"{label}: a promote needs the investor minimum IRR that "
                f"gates it."
            )
        if promote < 0 or promote > 100:
            errors.append(f"{label}: promote {promote:g}% is outside 0–100%.")
        shared = float(terms.get('promote_shared_pct') or 0)
        if shared < 0 or shared > 100:
            errors.append(
                f"{label}: promote shared with investors must be 0–100% "
                f"of the promote."
            )
        if min_irr is not None and (float(min_irr) <= 0 or float(min_irr) > 30):
            warnings.append(
                f"{label}: minimum IRR {min_irr}% looks unusual — confirm."
            )

    return {'errors': errors, 'warnings': warnings}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_stack(engine, prospect_id: int, stack: Dict[str, Any],
               username: str = '') -> Dict[str, Any]:
    """Replace the declared stack for a deal (vehicle + relationships +
    participants). Validation errors block the save. Returns the saved stack
    with its validation result.
    """
    check = validate_stack(stack)
    if check['errors']:
        return {'ok': False, 'validation': check}

    vehicle = stack.get('vehicle') or {}
    rels = stack.get('relationships') or []

    with engine.begin() as c:
        existing = {r[1]: r[0] for r in c.execute(text(
            "SELECT id, role FROM prospect_entities "
            "WHERE prospect_id = :p AND role = :r"),
            {'p': prospect_id, 'r': VEHICLE_ROLE})}
        # -- vehicle (upsert one row) --
        vid = existing.get(VEHICLE_ROLE)
        if vehicle:
            if vid:
                c.execute(text(
                    "UPDATE prospect_entities SET entity_name = :n, "
                    "planned_entity_id = :pid, updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = :i"),
                    {'n': vehicle.get('name'), 'pid': vehicle.get('entity_id'),
                     'i': vid})
            else:
                vid = c.execute(text(
                    "INSERT INTO prospect_entities (prospect_id, entity_name, "
                    "entity_type, planned_entity_id, role) "
                    "VALUES (:p, :n, 'pe_vehicle', :pid, :r) RETURNING id"),
                    {'p': prospect_id, 'n': vehicle.get('name'),
                     'pid': vehicle.get('entity_id'),
                     'r': VEHICLE_ROLE}).scalar()

        # -- relationships: upsert by id, delete the removed --
        kept_ids = []
        old_ids = [r[0] for r in c.execute(text(
            "SELECT id FROM prospect_entities "
            "WHERE prospect_id = :p AND role = :r"),
            {'p': prospect_id, 'r': RELATIONSHIP_ROLE})]

        for n, rel in enumerate(rels, 1):
            entity_id = (rel.get('entity_id') or '').strip() \
                or placeholder_entity_id(prospect_id, n)
            terms = _load_terms(rel.get('terms'))
            params = {
                'n': rel.get('name') or entity_id,
                'pid': entity_id,
                'parent': vid,
                'pct': float(rel.get('slice_pct') or 0),
                'tj': json.dumps(terms),
            }
            rid = rel.get('id')
            if rid and rid in old_ids:
                c.execute(text(
                    "UPDATE prospect_entities SET entity_name = :n, "
                    "planned_entity_id = :pid, parent_entity_id = :parent, "
                    "ownership_pct = :pct, terms_json = :tj, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = :i"),
                    {**params, 'i': rid})
            else:
                rid = c.execute(text(
                    "INSERT INTO prospect_entities (prospect_id, entity_name, "
                    "entity_type, planned_entity_id, parent_entity_id, "
                    "ownership_pct, role, terms_json) "
                    "VALUES (:p, :n, 'ppi_relationship', :pid, :parent, "
                    ":pct, :r, :tj) RETURNING id"),
                    {**params, 'p': prospect_id,
                     'r': RELATIONSHIP_ROLE}).scalar()
            kept_ids.append(rid)

            # participants: replace wholesale per relationship (small lists)
            c.execute(text(
                "DELETE FROM prospect_investors WHERE entity_id = :e"),
                {'e': rid})
            for p_row in (rel.get('participants') or []):
                c.execute(text(
                    "INSERT INTO prospect_investors (entity_id, investor_name, "
                    "planned_investor_id, ownership_pct, commitment, "
                    "investor_type) VALUES (:e, :n, :pid, :pct, :com, :t)"),
                    {'e': rid, 'n': p_row.get('name'),
                     'pid': p_row.get('investor_id'),
                     'pct': float(p_row.get('share_pct') or 0),
                     'com': p_row.get('commitment'),
                     't': (p_row.get('type') or 'lp').lower()})

        removed = [i for i in old_ids if i not in kept_ids]
        for rid in removed:
            c.execute(text(
                "DELETE FROM prospect_investors WHERE entity_id = :e"),
                {'e': rid})
            c.execute(text(
                "DELETE FROM prospect_entities WHERE id = :e"), {'e': rid})

    logger.info("PPI stack saved for prospect %s by %s: %d relationships "
                "(%d removed)", prospect_id, username, len(rels), len(removed))
    saved = get_stack(engine, prospect_id)
    saved_check = validate_stack(saved)
    return {'ok': True, 'stack': saved, 'validation': saved_check}
