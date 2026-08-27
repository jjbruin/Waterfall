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


def _load_all_steps(engine) -> pd.DataFrame:
    """Every stored waterfall. A linked JV chain (vehicle -> existing JV ->
    fund -> investors) traverses entities far beyond the declared level, so
    the whole (small) table is loaded, exactly as PSCKOC does."""
    from sqlalchemy import text as sa_text
    with engine.connect() as c:
        rows = [dict(r) for r in c.execute(sa_text(
            'SELECT * FROM waterfalls')).mappings()]
    if not rows:
        return pd.DataFrame()
    return load_waterfalls(pd.DataFrame(rows))


def _load_relationships(engine) -> pd.DataFrame:
    """The real ownership tree, so entities without waterfalls (e.g. a fund
    feeder) pass cash through to their owners by OwnershipPct."""
    from sqlalchemy import text as sa_text
    from ownership_tree import load_relationships
    try:
        with engine.connect() as c:
            rows = [dict(r) for r in c.execute(sa_text(
                'SELECT * FROM relationships')).mappings()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # only active rows: an ownership generation that ended (EndDate set)
        # must not double-count alongside its replacement
        if 'EndDate' in df.columns:
            end = df['EndDate']
            df = df[end.isna() | (end.astype(str).str.strip() == '')]
        return load_relationships(df)
    except Exception:
        logger.exception("relationships load failed")
        return pd.DataFrame()


def _stack_closure(vehicle_id, declared_ids, wf_steps, relationships):
    """Bound the run to the deal's subtree.

    BFS downward from the vehicle: an entity's waterfall runs only if the
    entity is the vehicle, a declared relationship, or was reached as a
    relationship-passthrough target (a fund under a feeder). A
    waterfall-bearing entity reached as a DISTRIBUTION CHILD is a terminal
    here — its own structure (e.g. PSC1's house waterfall) is out of scope
    for this deal's view. Returns (wf_steps_bounded, relationships_bounded,
    wf_entity_set)."""
    if wf_steps is None or wf_steps.empty:
        return wf_steps, relationships, set()
    have = {}
    for vc, grp in wf_steps.groupby('vcode'):
        have[str(vc)] = grp
    rel_by_inv = {}
    if relationships is not None and not relationships.empty:
        for inv, grp in relationships.groupby('InvestmentID'):
            rel_by_inv[str(inv)] = grp

    # fee (and promote) recipients are the manager: terminal by
    # definition — its owners' structures are out of this deal's scope
    fee_recipients = set()
    for grp in have.values():
        fr = grp[grp['vState'].astype(str) == 'AMFee']
        fee_recipients |= set(fr['PropCode'].astype(str))

    wf_allowed = set()
    pass_allowed = set()
    frontier = [(vehicle_id, True)] + [(d, True) for d in declared_ids]
    visited = set()
    while frontier:
        eid, may_run_wf = frontier.pop(0)
        if eid in visited:
            continue
        visited.add(eid)
        if may_run_wf and eid in have:
            wf_allowed.add(eid)
            if eid in rel_by_inv:
                # its own ownership rows feed the capital cascade only;
                # the engine prefers the waterfall over passthrough
                pass_allowed.add(eid)
            children = set(have[eid]['PropCode'].astype(str))
            for ch in children:
                if ch in wf_allowed or ch in visited or ch in fee_recipients:
                    continue
                if ch in have and ch not in declared_ids:
                    continue  # terminal: its own structure is out of scope
                if ch in rel_by_inv:
                    frontier.append((ch, False))
                    pass_allowed.add(ch)
                elif ch in declared_ids:
                    frontier.append((ch, True))
        elif eid in rel_by_inv:
            pass_allowed.add(eid)
            for _, o in rel_by_inv[eid].iterrows():
                oid = str(o['InvestorID']).strip()
                if oid and oid not in visited and oid not in fee_recipients:
                    # a passthrough target's fund may carry its waterfall
                    frontier.append((oid, True))

    wf_bounded = wf_steps[wf_steps['vcode'].astype(str).isin(wf_allowed)]
    if relationships is not None and not relationships.empty:
        rel_bounded = relationships[
            relationships['InvestmentID'].astype(str).isin(pass_allowed)]
    else:
        rel_bounded = relationships
    return wf_bounded, rel_bounded, wf_allowed


def _cascade_seed(seeded, registry, relationships, seed_date, notes,
                  max_depth=5):
    """Propagate seeded capital down the ownership tree.

    A declared participant that is itself an owned entity (INV6, owned by
    AMB6, owned by 13 investors) carries its capital through to its owners
    pro-rata by OwnershipPct — so fee bases and IRRs exist at every level
    the cash will reach. Zero-percent owners (a non-equity manager) are
    skipped."""
    if relationships is None or relationships.empty:
        return
    rel = relationships
    inv_col = 'InvestmentID' if 'InvestmentID' in rel.columns else None
    own_col = 'InvestorID' if 'InvestorID' in rel.columns else None
    pct_col = 'OwnershipPct' if 'OwnershipPct' in rel.columns else None
    if not (inv_col and own_col and pct_col):
        return
    frontier = list(seeded.keys())
    visited = set()
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        next_frontier = []
        for pid in frontier:
            if pid in visited:
                continue
            visited.add(pid)
            owners = rel[rel[inv_col].astype(str).str.strip() == pid]
            if owners.empty:
                continue
            base = seeded[pid].get_pool('initial').capital_outstanding
            if base <= 0:
                continue
            for _, o in owners.iterrows():
                oid = str(o[own_col]).strip()
                pct = float(o[pct_col] or 0)
                # load_relationships normalizes to decimals; raw tables carry
                # percentages -- accept either
                frac = pct if pct <= 1.0 else pct / 100.0
                if not oid or frac <= 0:
                    continue
                amt = base * frac
                st = seeded.get(oid)
                if st is None:
                    st = InvestorState(propcode=oid)
                    st.get_pool('initial').last_accrual_date = seed_date
                    seeded[oid] = st
                st.get_pool('initial').capital_outstanding += amt
                st.cashflows.append((seed_date, -amt))
                st.cashflow_labels.append('Contribution')
                st.cashflow_types.append('C')
                kind = 'psc' if oid.upper().startswith('PSC') else 'indirect'
                reg = registry.setdefault(oid, {
                    'name': oid, 'type': kind,
                    'relationships': [], 'contributions': 0.0})
                reg['contributions'] += amt
                if f'via {pid}' not in reg['relationships']:
                    reg['relationships'].append(f'via {pid}')
                next_frontier.append(oid)
        frontier = next_frontier


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

    wf_steps = _load_all_steps(engine)
    have = set(wf_steps['vcode'].astype(str)) if not wf_steps.empty else set()
    if vehicle_id not in have:
        return {'notes': ['No PPI waterfalls stored yet — use "Build & Save '
                          'PPI Waterfalls" on the setup panel.']}
    relationships = _load_relationships(engine)
    declared_ids = [str(r.get('entity_id') or '').strip()
                    for r in rels if r.get('entity_id')]
    wf_steps, relationships, wf_entities = _stack_closure(
        vehicle_id, declared_ids, wf_steps, relationships)
    have = wf_entities
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

    def _seed(pid, amt):
        st = seeded.get(pid)
        if st is None:
            st = InvestorState(propcode=pid)
            st.get_pool('initial').last_accrual_date = seed_date
            seeded[pid] = st
        st.get_pool('initial').capital_outstanding += amt
        st.cashflows.append((seed_date, -amt))
        st.cashflow_labels.append('Contribution')
        st.cashflow_types.append('C')

    intermediates = {vehicle_id}
    for n, rel in enumerate(rels, 1):
        rid = (rel.get('entity_id') or '').strip()
        rel_amt = vehicle_contrib * float(rel.get('slice_pct') or 0) / 100.0
        linked = bool(rid and rid in have)
        if linked:
            # existing JV: seed the entity once; the ownership cascade
            # derives every level below it (declared participants included),
            # so nothing is double-counted
            intermediates.add(rid)
            if rel_amt > 0:
                _seed(rid, rel_amt)
            for p_row in _participant_ids(rel, prospect_id, n):
                pid = p_row['investor_id']
                reg = registry.setdefault(pid, {
                    'name': p_row.get('name') or pid,
                    'type': (p_row.get('type') or 'lp').lower(),
                    'relationships': [], 'contributions': 0.0})
                reg['relationships'].append(rel.get('name') or rid)
            continue
        for p_row in _participant_ids(rel, prospect_id, n):
            pid = p_row['investor_id']
            amt = rel_amt * float(p_row.get('share_pct') or 0) / 100.0
            reg = registry.setdefault(pid, {
                'name': p_row.get('name') or pid,
                'type': (p_row.get('type') or 'lp').lower(),
                'relationships': [], 'contributions': 0.0})
            reg['relationships'].append(rel.get('name') or rel['entity_id'])
            reg['contributions'] += amt
            if amt > 0:
                _seed(pid, amt)

    # capital cascades down the real ownership tree so every level the
    # cash will reach has a fee base and a meaningful IRR. Waterfall-bearing
    # intermediates are excluded from the participant display.
    _cascade_seed(seeded, registry, relationships, seed_date, notes)
    intermediates |= {e for e in have if e in registry or e in seeded}
    # passthrough feeders (a wholly-owned investment vehicle between the JV
    # and the fund) pass their money onward — show the people, not the pipe
    if relationships is not None and not relationships.empty:
        for feeder in relationships['InvestmentID'].astype(str).unique():
            if feeder in registry and feeder not in have:
                intermediates.add(feeder)

    # ---- interleaved upstream run ---------------------------------------
    entity_states: Dict[str, InvestorState] = copy.deepcopy(seeded)
    allocation_rows: List[dict] = []
    tracker: Dict = {}
    for d, cash, wf_type in events:
        try:
            run_upstream_waterfall_period(
                vehicle_id, cash, d, wf_steps, relationships, entity_states,
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

    # recipients the run discovered that were never declared or seeded
    # (a non-equity manager earning fees / promote)
    if not alloc.empty:
        for pid in alloc['PropCode'].astype(str).unique():
            if pid not in registry and pid in entity_states:
                registry[pid] = {'name': pid, 'type': 'mgr',
                                 'relationships': [], 'contributions': 0.0}

    participants_out = []
    for pid, reg in registry.items():
        if pid in intermediates:
            continue
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

    psc = [p_row for p_row in participants_out
           if p_row['type'] in ('psc', 'mgr')]
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
