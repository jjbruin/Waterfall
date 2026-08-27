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

# ---------------------------------------------------------------------------
# Phase 2: generate the waterfall steps from the declared stack.
#
# Template (the standard Peaceable deal, per Jim Aug 27; TGA22 is the AM
# reference): CF and capital shared pro-rata between PSC and the investor(s);
# the AM fee is deducted from the investor's distributions (AMFee rows at
# iOrder 900+, vNotes = source investor, nPercent = raw percent, mAmount =
# periods/yr); PSC earns a promote only after the investor reaches the
# minimum IRR (IRR gate, computed net of fees by the engine). PSCKOC-style
# catch-ups are deliberately NOT templated.
#
# Rate conventions follow the AM rows: Pref/IRR rates as decimals (0.08),
# AMFee as raw percent (0.95).
# ---------------------------------------------------------------------------

def _post_gate_splits(participants, terms):
    """Post-promote tier percentages (decimals summing to 1.0).

    promote_pct of the excess is the promote pool; promote_shared_pct of
    that pool goes back to the LPs pro-rata; the rest of the pool is PSC's.
    The remaining (1 - promote) splits pro-rata across ALL participants.
    """
    promote = float(terms.get('promote_pct') or 0) / 100.0
    shared = float(terms.get('promote_shared_pct') or 0) / 100.0
    lps = [p for p in participants if (p.get('type') or '').lower() != 'psc']
    psc = next((p for p in participants
                if (p.get('type') or '').lower() == 'psc'), None)
    lp_total = sum(float(p.get('share_pct') or 0) for p in lps) or 1.0

    out = {}
    for p_row in lps:
        share = float(p_row.get('share_pct') or 0) / 100.0
        lp_frac = float(p_row.get('share_pct') or 0) / lp_total
        out[p_row['investor_id']] = (1 - promote) * share \
            + promote * shared * lp_frac
    if psc is not None:
        share = float(psc.get('share_pct') or 0) / 100.0
        out[psc['investor_id']] = (1 - promote) * share \
            + promote * (1 - shared)
    return out


def _participant_ids(rel, prospect_id, n):
    """Resolve participant ids, defaulting placeholders that keep LP vs PSC
    distinguishable until MRI assigns real ids at closing."""
    parts = []
    for j, p_row in enumerate(rel.get('participants') or [], 1):
        pid = (p_row.get('investor_id') or '').strip()
        if not pid:
            kind = 'PSC' if (p_row.get('type') or '').lower() == 'psc' else 'INV'
            pid = f"{kind}{prospect_id:04d}{n}{j}"
        parts.append({**p_row, 'investor_id': pid})
    return parts


def build_relationship_steps(vcode, rel, prospect_id, n):
    """Waterfall rows (CF_WF + Cap_WF) for one relationship, keyed by vcode."""
    from datetime import date as dt_date
    terms = _load_terms(rel.get('terms'))
    participants = _participant_ids(rel, prospect_id, n)
    lps = [p for p in participants if (p.get('type') or '').lower() != 'psc']
    psc = next((p for p in participants
                if (p.get('type') or '').lower() == 'psc'), None)
    lps.sort(key=lambda p_row: -float(p_row.get('share_pct') or 0))
    ordered = lps + ([psc] if psc else [])

    def _row(wf, order, pc, state, fx, npct, mamt, notes, trans):
        return {'vcode': vcode, 'vmisc': wf, 'iOrder': order, 'vAmtType': '',
                'vNotes': notes, 'PropCode': pc, 'nmisc': 0,
                'dteffective': dt_date(2020, 1, 1), 'vtranstype': trans,
                'mAmount': mamt, 'nPercent': npct, 'FXRate': fx,
                'vState': state}

    rows = []
    pref = terms.get('pref_rate_pct')
    pref_dec = float(pref) / 100.0 if pref else None
    fee = float(terms.get('am_fee_pct') or 0)
    periods = int(terms.get('fee_periods_per_year') or 4)
    min_irr = terms.get('min_irr_pct')
    gate_dec = float(min_irr) / 100.0 if min_irr else None
    promote = float(terms.get('promote_pct') or 0)

    def _pro_rata_tier(wf, order, lead_state, trans):
        for k, p_row in enumerate(ordered):
            fx = float(p_row.get('share_pct') or 0) / 100.0
            state = lead_state if k == 0 else 'Tag'
            rows.append(_row(wf, order, p_row['investor_id'], state,
                             fx, 0, 0, '', trans))

    # ---- CF_WF: (pref) -> pro-rata split; fee from investor distributions
    if pref_dec:
        for k, p_row in enumerate(ordered):
            fx = float(p_row.get('share_pct') or 0) / 100.0
            state = 'Pref' if k == 0 else 'Tag'
            rows.append(_row('CF_WF', 10, p_row['investor_id'], state,
                             fx, pref_dec if k == 0 else 0, 0, '',
                             'Preferred Return'))
    _pro_rata_tier('CF_WF', 20, 'Share', 'Excess Cash Flow')

    # ---- Cap_WF: (pref) -> return of capital -> IRR gate -> promote tier
    order = 10
    if pref_dec:
        for k, p_row in enumerate(ordered):
            fx = float(p_row.get('share_pct') or 0) / 100.0
            state = 'Pref' if k == 0 else 'Tag'
            rows.append(_row('Cap_WF', order, p_row['investor_id'], state,
                             fx, pref_dec if k == 0 else 0, 0, '',
                             'Preferred Return'))
        order += 10
    _pro_rata_tier('Cap_WF', order, 'Initial', 'Return of Capital')
    order += 10
    if gate_dec and promote > 0:
        # gate: lead LP's IRR (net of fees) releases pro-rata until the
        # minimum IRR; with several LPs the largest gates for the tier
        for k, p_row in enumerate(ordered):
            fx = float(p_row.get('share_pct') or 0) / 100.0
            state = 'IRR' if k == 0 else 'Tag'
            rows.append(_row('Cap_WF', order, p_row['investor_id'], state,
                             fx, gate_dec if k == 0 else 0, 0, '',
                             'IRR Threshold'))
        order += 10
        splits = _post_gate_splits(participants, terms)
        post = sorted(splits.items(), key=lambda kv: -kv[1])
        for k, (pid, fx) in enumerate(post):
            state = 'Share' if k == 0 else 'Tag'
            rows.append(_row('Cap_WF', order, pid, state, round(fx, 6),
                             0, 0, '', 'Promote Split'))
    else:
        _pro_rata_tier('Cap_WF', order, 'Share', 'Excess Cash Flow')

    # ---- AM fee rows (both waterfalls, one per LP, quarterly-capped)
    if fee > 0 and psc is not None:
        for j, p_row in enumerate(lps):
            for wf in ('CF_WF', 'Cap_WF'):
                rows.append(_row(wf, 900 + j, psc['investor_id'], 'AMFee',
                                 1.0, fee, periods, p_row['investor_id'],
                                 'AM Fee'))
    return rows


def build_stack_waterfalls(engine, prospect_id: int,
                           username: str = '') -> Dict[str, Any]:
    """Generate and SAVE the stack's waterfalls (explicit write, mirroring
    the deal Builder's rule that only Build & Save touches stored steps).

    Multi-relationship stacks get a pro-rata Share/Tag split on the vehicle
    id routing into the relationship entities; a single-relationship stack
    puts the relationship waterfall directly on the vehicle id.
    Returns {'ok', 'steps': {vcode: [rows]}, 'validation'}.
    """
    import pandas as pd
    from datetime import date as dt_date
    from database import save_waterfall_steps

    stack = get_stack(engine, prospect_id)
    check = validate_stack(stack)
    if check['errors']:
        return {'ok': False, 'validation': check}
    rels = stack.get('relationships') or []
    vehicle = stack.get('vehicle') or {}
    vehicle_id = (vehicle.get('entity_id') or '').strip()
    if not rels:
        return {'ok': False, 'validation': {
            'errors': ['No relationships declared.'], 'warnings': []}}
    if not vehicle_id:
        return {'ok': False, 'validation': {
            'errors': ['The PE vehicle needs an entity id.'], 'warnings': []}}

    steps_by_vcode: Dict[str, list] = {}
    warnings = list(check['warnings'])

    if len(rels) == 1:
        steps_by_vcode[vehicle_id] = build_relationship_steps(
            vehicle_id, rels[0], prospect_id, 1)
    else:
        split_rows = []
        ordered = sorted(rels, key=lambda r: -float(r.get('slice_pct') or 0))
        for wf in ('CF_WF', 'Cap_WF'):
            for k, rel in enumerate(ordered):
                split_rows.append({
                    'vcode': vehicle_id, 'vmisc': wf, 'iOrder': 10,
                    'vAmtType': '', 'vNotes': '',
                    'PropCode': rel['entity_id'],
                    'nmisc': 0, 'dteffective': dt_date(2020, 1, 1),
                    'vtranstype': 'Pro Rata Split',
                    'mAmount': 0, 'nPercent': 0,
                    'FXRate': float(rel.get('slice_pct') or 0) / 100.0,
                    'vState': 'Share' if k == 0 else 'Tag'})
        steps_by_vcode[vehicle_id] = split_rows
        for n, rel in enumerate(rels, 1):
            steps_by_vcode[rel['entity_id']] = build_relationship_steps(
                rel['entity_id'], rel, prospect_id, n)
        lps_multi = [r['name'] for r in rels
                     if len([p_row for p_row in r['participants']
                             if (p_row.get('type') or '').lower() != 'psc']) > 1]
        if lps_multi:
            warnings.append(
                "Relationships with several investors gate the promote on "
                "the largest investor's IRR: " + ", ".join(lps_multi))

    for vcode, rows in steps_by_vcode.items():
        save_waterfall_steps(vcode, pd.DataFrame(rows))
    logger.info("PPI waterfalls built for prospect %s by %s: %s",
                prospect_id, username,
                {k: len(v) for k, v in steps_by_vcode.items()})
    return {'ok': True, 'steps': steps_by_vcode,
            'validation': {'errors': [], 'warnings': warnings}}

# ---------------------------------------------------------------------------
# Phase 5: migration at close.
#
# MRI assigns the real entity ids when a deal closes. This re-keys the
# placeholder ids everywhere the stack used them (waterfall vcodes,
# PropCodes, AMFee vNotes sources, and the prospect rows) and writes the
# ownership rows that let the AM side's tree and Portfolio Analysis pick
# the stack up natively.
# ---------------------------------------------------------------------------

def _stack_ids(stack, prospect_id):
    """Every entity/participant id the stack references (placeholders
    resolved the same way the builder resolves them)."""
    ids = []
    veh = ((stack.get('vehicle') or {}).get('entity_id') or '').strip()
    if veh:
        ids.append(veh)
    for n, rel in enumerate(stack.get('relationships') or [], 1):
        rid = (rel.get('entity_id') or '').strip()
        if rid:
            ids.append(rid)
        for p_row in _participant_ids(rel, prospect_id, n):
            ids.append(p_row['investor_id'])
    return ids


def migrate_stack_at_close(engine, prospect_id: int,
                           id_map: Optional[Dict[str, str]] = None,
                           close_date: Optional[str] = None,
                           write_relationships: bool = True,
                           username: str = '') -> Dict[str, Any]:
    """Migrate the stack to its closing identities.

    id_map: {placeholder_or_old_id: final_MRI_id}. Only ids the stack
    actually references are honored. When a final id ALREADY carries a
    waterfall (the deal joined an existing JV), the existing waterfall is
    kept: the placeholder's steps are deleted, not copied over it, and this
    is reported.

    Ownership rows are written into `relationships` (InvestmentID = owned
    entity, InvestorID = owner, OwnershipPct as whole percent). That table
    is refreshed from MRI, so these rows are a BRIDGE until MRI carries the
    real ownership records after closing.
    """
    from sqlalchemy import text as sa_text

    stack = get_stack(engine, prospect_id)
    rels = stack.get('relationships') or []
    if not rels:
        return {'ok': False, 'notes': ['No PPI stack declared.']}
    known = set(_stack_ids(stack, prospect_id))
    id_map = {str(k).strip(): str(v).strip()
              for k, v in (id_map or {}).items()
              if str(k).strip() in known and str(v).strip()
              and str(v).strip() != str(k).strip()}

    notes: List[str] = []
    renames: List[Dict[str, str]] = []
    rel_rows_written = 0

    with engine.begin() as c:
        # A mapped final id that already carries a waterfall is a linked JV
        # whose waterfall we keep; its rows are never rewritten. Precomputed
        # so the scope below is order-independent.
        from sqlalchemy import text as _t
        kept_existing = [
            new for old, new in id_map.items()
            if c.execute(_t('SELECT COUNT(*) FROM waterfalls WHERE vcode = :v'),
                         {'v': new}).scalar()
            and c.execute(_t('SELECT COUNT(*) FROM waterfalls WHERE vcode = :v'),
                          {'v': old}).scalar()]
        # ---- 1. re-key waterfall rows -----------------------------------
        for old, new in id_map.items():
            existing = c.execute(sa_text(
                'SELECT COUNT(*) FROM waterfalls WHERE vcode = :v'),
                {'v': new}).scalar()
            mine = c.execute(sa_text(
                'SELECT COUNT(*) FROM waterfalls WHERE vcode = :v'),
                {'v': old}).scalar()
            if mine and existing:
                c.execute(sa_text(
                    'DELETE FROM waterfalls WHERE vcode = :v'), {'v': old})
                notes.append(
                    f"{new} already has a waterfall (existing JV) — kept it; "
                    f"the {mine} steps built under {old} were removed.")
            elif mine:
                c.execute(sa_text(
                    'UPDATE waterfalls SET vcode = :n WHERE vcode = :o'),
                    {'n': new, 'o': old})
            # PropCode + AMFee source references, scoped to the stack's
            # waterfalls so unrelated entities are never touched — and never
            # a linked JV whose existing waterfall was kept
            scope = [id_map.get(i, i) for i in known
                     if id_map.get(i, i) not in kept_existing]
            c.execute(sa_text(
                'UPDATE waterfalls SET "PropCode" = :n '
                'WHERE "PropCode" = :o AND vcode = ANY(:s)'),
                {'n': new, 'o': old, 's': scope})
            c.execute(sa_text(
                'UPDATE waterfalls SET "vNotes" = :n '
                'WHERE "vNotes" = :o AND vcode = ANY(:s)'),
                {'n': new, 'o': old, 's': scope})
            renames.append({'from': old, 'to': new})

        # ---- 2. prospect rows follow ------------------------------------
        for old, new in id_map.items():
            c.execute(sa_text(
                'UPDATE prospect_entities SET planned_entity_id = :n '
                'WHERE prospect_id = :p AND planned_entity_id = :o'),
                {'n': new, 'o': old, 'p': prospect_id})
            c.execute(sa_text(
                'UPDATE prospect_investors SET planned_investor_id = :n '
                'WHERE planned_investor_id = :o AND entity_id IN '
                '(SELECT id FROM prospect_entities WHERE prospect_id = :p)'),
                {'n': new, 'o': old, 'p': prospect_id})

        # ---- 3. ownership rows for the AM tree --------------------------
        if write_relationships:
            import pandas as pd
            start = pd.Timestamp(close_date).to_pydatetime() if close_date                 else None
            final = get_stack(engine, prospect_id)
            veh = ((final.get('vehicle') or {}).get('entity_id') or '').strip()
            veh = id_map.get(veh, veh)
            multi = len(final.get('relationships') or []) > 1

            def _own(owned, owner, pct, name):
                nonlocal rel_rows_written
                exists = c.execute(sa_text(
                    'SELECT COUNT(*) FROM relationships '
                    'WHERE "InvestmentID" = :a AND TRIM("InvestorID") = :b '
                    'AND "EndDate" IS NULL'),
                    {'a': owned, 'b': owner}).scalar()
                if exists:
                    notes.append(f"ownership row {owned} <- {owner} already "
                                 f"exists — left as is")
                    return
                c.execute(sa_text(
                    'INSERT INTO relationships ("InvestmentID", "InvestorID", '
                    '"OwnershipPct", "Name", "StartDate", "EndDate") '
                    'VALUES (:a, :b, :p, :n, :d, NULL)'),
                    {'a': owned, 'b': owner, 'p': float(pct or 0),
                     'n': name, 'd': start})
                rel_rows_written += 1

            for n, rel in enumerate(final.get('relationships') or [], 1):
                rid = (rel.get('entity_id') or '').strip()
                rid = id_map.get(rid, rid)
                rname = rel.get('name') or rid
                if multi and veh:
                    _own(veh, rid, rel.get('slice_pct'), rname)
                owner_of = rid if multi else veh
                for p_row in _participant_ids(rel, prospect_id, n):
                    pid = id_map.get(p_row['investor_id'], p_row['investor_id'])
                    _own(owner_of, pid, p_row.get('share_pct'), rname)

    if rel_rows_written:
        notes.append(
            f"{rel_rows_written} ownership rows written to `relationships` — "
            f"note this table is refreshed from MRI, so these rows are a "
            f"bridge until MRI carries the real ownership records.")
    logger.info("PPI stack migrated for prospect %s by %s: %s",
                prospect_id, username, renames)
    return {'ok': True, 'renames': renames,
            'relationships_written': rel_rows_written, 'notes': notes}

