"""
ppi_upstream_service.py
Run the PPI ownership waterfalls for a New Business deal.

Takes the deal-level allocations the analysis already produced, feeds every
dollar the PE vehicle received through the stack's stored waterfalls
(vehicle pro-rata split -> per-relationship fee/pref/gate/promote), and
assembles per-participant economics.

CF and Cap events are interleaved chronologically through
run_upstream_waterfall_period with ONE shared state dict and ONE quarterly
fee tracker — so the capital-side IRR gates see the investors' CF
distribution history (the minimum-IRR promote test is a whole-deal lookback,
net of AM fees, which the engine nets into the investor's cashflows by
construction), and the quarterly fee cap spans both waterfalls.
"""

import copy
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from models import InvestorState
from waterfall import (run_upstream_waterfall_period,
                       get_upstream_waterfall_entities)
from loaders import load_waterfalls
from metrics import xirr
from flask_app.services.ppi_stack_service import (
    get_stack, _load_terms, _participant_ids)

logger = logging.getLogger(__name__)


def _load_stack_steps(engine, vcodes: List[str]) -> pd.DataFrame:
    from sqlalchemy import text as sa_text
    frames = []
    with engine.connect() as c:
        for vc in vcodes:
            rows = [dict(r) for r in c.execute(sa_text(
                'SELECT * FROM waterfalls WHERE vcode = :v'),
                {'v': vc}).mappings()]
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return load_waterfalls(pd.concat(frames, ignore_index=True))


def build_ppi_results(engine, prospect_id: int,
                      result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Returns the ppi_waterfalls payload, or None when no stack is declared.

    A declared stack without built steps (or a vehicle id that never received
    deal cash) returns {'notes': [...]} so the UI can say why it is empty.
    """
    stack = get_stack(engine, prospect_id)
    rels = stack.get('relationships') or []
    if not rels:
        return None
    notes: List[str] = []
    vehicle_id = ((stack.get('vehicle') or {}).get('entity_id') or '').strip()
    if not vehicle_id:
        return {'notes': ['The PE vehicle has no entity id — set one on the '
                          'PPI Ownership panel.']}

    vcodes = [vehicle_id] + [r['entity_id'] for r in rels if r.get('entity_id')]
    wf_steps = _load_stack_steps(engine, vcodes)
    if wf_steps.empty:
        return {'notes': ['No PPI waterfalls stored yet — use "Build & Save '
                          'PPI Waterfalls" on the setup panel.']}
    upstream_entities = get_upstream_waterfall_entities(wf_steps)

    # ---- deal cash into the vehicle -------------------------------------
    def _vehicle_events(alloc, wf_type):
        if alloc is None or getattr(alloc, 'empty', True):
            return []
        a = alloc[(alloc['PropCode'].astype(str) == vehicle_id)
                  & (pd.to_numeric(alloc['Allocated'], errors='coerce')
                     .fillna(0) > 0)]
        out = []
        for dt, grp in a.groupby('event_date'):
            d = pd.Timestamp(dt).date()
            out.append((d, float(grp['Allocated'].sum()), wf_type))
        return out

    events = (_vehicle_events(result.get('cf_alloc'), 'CF_WF')
              + _vehicle_events(result.get('cap_alloc'), 'Cap_WF'))
    if not events:
        return {'notes': [f'The deal waterfall allocated no cash to '
                          f'{vehicle_id} — check that the vehicle entity id '
                          f'matches the deal waterfall PropCode.']}
    events.sort(key=lambda e: (e[0], 0 if e[2] == 'CF_WF' else 1))

    # ---- seed participant capital at close - 1 day ----------------------
    close = (result.get('prospect_assumptions') or {}).get('close_date')
    seed_date = (pd.Timestamp(close).date() - timedelta(days=1)) if close \
        else events[0][0]
    vehicle_contrib = 0.0
    for pr in (result.get('partner_results') or []):
        if str(pr.get('partner')) == vehicle_id:
            vehicle_contrib = abs(float(pr.get('contributions') or 0))
    if vehicle_contrib <= 0:
        notes.append(f'{vehicle_id} shows no contributions at the deal '
                     f'level; participant capital could not be seeded.')

    registry: Dict[str, Dict[str, Any]] = {}   # pid -> {name, type, rels}
    seeded: Dict[str, InvestorState] = {}
    for n, rel in enumerate(rels, 1):
        rel_amt = vehicle_contrib * float(rel.get('slice_pct') or 0) / 100.0
        for p_row in _participant_ids(rel, prospect_id, n):
            pid = p_row['investor_id']
            amt = rel_amt * float(p_row.get('share_pct') or 0) / 100.0
            reg = registry.setdefault(pid, {
                'name': p_row.get('name') or pid,
                'type': (p_row.get('type') or 'lp').lower(),
                'relationships': [], 'contributions': 0.0})
            reg['relationships'].append(rel.get('name') or rel['entity_id'])
            reg['contributions'] += amt
            if amt <= 0:
                continue
            st = seeded.get(pid)
            if st is None:
                st = InvestorState(propcode=pid)
                st.get_pool('initial').last_accrual_date = seed_date
                seeded[pid] = st
            st.get_pool('initial').capital_outstanding += amt
            st.cashflows.append((seed_date, -amt))
            st.cashflow_labels.append('Contribution')
            st.cashflow_types.append('C')

    # ---- interleaved upstream run ---------------------------------------
    entity_states: Dict[str, InvestorState] = copy.deepcopy(seeded)
    allocation_rows: List[dict] = []
    tracker: Dict = {}
    empty_rel = pd.DataFrame()
    for d, cash, wf_type in events:
        try:
            run_upstream_waterfall_period(
                vehicle_id, cash, d, wf_steps, empty_rel, entity_states,
                upstream_entities, wf_type=wf_type,
                allocation_rows=allocation_rows,
                amt_quarterly_tracker=tracker)
        except Exception as e:
            logger.exception("PPI upstream period failed")
            notes.append(f'{d} {wf_type}: upstream run failed ({e})')

    alloc = pd.DataFrame(allocation_rows)

    # ---- assemble -------------------------------------------------------
    def _rows(mask):
        return alloc[mask] if not alloc.empty else alloc

    participants_out = []
    for pid, reg in registry.items():
        st = entity_states.get(pid)
        cfs = list(st.cashflows) if st is not None else []
        lbls = (list(st.cashflow_labels)
                if st is not None and len(st.cashflow_labels) == len(cfs)
                else [''] * len(cfs))
        # negatives are either the seeded contribution or an AM fee
        # deduction (the engine labels both); keep them separate
        contribs = -sum(a for (_, a), l in zip(cfs, lbls)
                        if a < 0 and str(l) == 'Contribution')
        fees_paid = -sum(a for (_, a), l in zip(cfs, lbls)
                         if a < 0 and str(l) != 'Contribution')
        dists = sum(a for _, a in cfs if a > 0)
        irr = None
        try:
            if cfs and min(a for _, a in cfs) < 0 and dists > 0:
                irr = xirr(sorted(cfs, key=lambda x: x[0]))
        except Exception:
            irr = None
        fees_received = 0.0
        promote_received = 0.0
        if not alloc.empty:
            mine = alloc[alloc['PropCode'].astype(str) == pid]
            fees_received = float(
                mine.loc[mine['vState'] == 'AMFee', 'Allocated'].sum())
            post = mine[mine['vtranstype'] == 'Promote Split']
            promote_received = float(post['Allocated'].sum())
        participants_out.append({
            'investor_id': pid,
            'name': reg['name'],
            'type': reg['type'],
            'relationships': reg['relationships'],
            'contributions': contribs,
            'distributions': dists,
            'am_fees_paid': fees_paid,
            'am_fees_received': fees_received,
            'post_gate_distributions': promote_received,
            'net_total': dists - contribs,
            'moic': (dists / contribs) if contribs > 0 else None,
            'irr': irr,
            'cashflows': [(str(d), float(a)) for d, a in
                          sorted(cfs, key=lambda x: x[0])],
        })
    participants_out.sort(key=lambda p_row: (p_row['type'] != 'lp',
                                             -p_row['contributions']))

    # per-relationship detail: distribution breakdown + fee schedule
    rels_out = []
    for rel in rels:
        rid = rel['entity_id']
        r_alloc = _rows(alloc['Entity'].astype(str) == rid) \
            if not alloc.empty and 'Entity' in alloc.columns else pd.DataFrame()
        breakdown = []
        fee_schedule = []
        if not r_alloc.empty:
            grp = (r_alloc[r_alloc['vState'] != 'AMFee']
                   .groupby(['vtranstype', 'PropCode'])['Allocated']
                   .sum().reset_index())
            breakdown = [{'category': g['vtranstype'],
                          'participant': g['PropCode'],
                          'amount': float(g['Allocated'])}
                         for _, g in grp.iterrows() if g['Allocated']]
            for _, f in r_alloc[r_alloc['vState'] == 'AMFee'].iterrows():
                if float(f['Allocated']):
                    fee_schedule.append({
                        'date': str(f['event_date'])[:10],
                        'recipient': f['PropCode'],
                        'fee': float(f['Allocated']),
                        'waterfall': f.get('WaterfallType') or '',
                    })
        fee_schedule.sort(key=lambda x: x['date'])
        rels_out.append({
            'name': rel.get('name') or rid,
            'entity_id': rid,
            'slice_pct': rel.get('slice_pct'),
            'terms': _load_terms(rel.get('terms')),
            'breakdown': breakdown,
            'fee_schedule': fee_schedule,
        })

    psc = [p_row for p_row in participants_out if p_row['type'] == 'psc']
    psc_summary = None
    if psc:
        psc_summary = {
            'total_fees': sum(p_row['am_fees_received'] for p_row in psc),
            'total_contributions': sum(p_row['contributions'] for p_row in psc),
            'total_distributions': sum(p_row['distributions'] for p_row in psc),
            'irr': psc[0]['irr'] if len(psc) == 1 else None,
        }

    return {
        'vehicle': vehicle_id,
        'seed_date': str(seed_date),
        'vehicle_contributions': vehicle_contrib,
        'events': [{'date': str(d), 'amount': amt, 'waterfall': w}
                   for d, amt, w in events],
        'participants': participants_out,
        'relationships': rels_out,
        'psc_summary': psc_summary,
        'notes': notes,
    }
