"""
prospects.py
API endpoints for the New Business deal pipeline.
"""

from flask import Blueprint, g, jsonify, request
from flask_app.auth.routes import login_required, role_required
from flask_app.db import get_engine
from flask_app.services.prospect_service import (
    ensure_prospect_tables,
    list_deals, get_deal, create_deal, update_deal, delete_deal,
    list_properties, create_property, update_property, delete_property,
    create_entity, update_entity, delete_entity,
    create_investor, update_investor, delete_investor,
    get_activity, add_activity_note,
    create_lease_review_for_property,
    list_assumptions, get_assumption, save_assumptions, delete_assumptions,
    import_property_cashflows, get_property_cashflows, delete_property_cashflows,
    get_deal_cashflows_by_property, get_property_cashflow_status,
)
import logging

logger = logging.getLogger(__name__)

prospects_bp = Blueprint('prospects', __name__, url_prefix='/api/prospects')


@prospects_bp.before_app_request
def _ensure_tables():
    """Ensure prospect tables exist on first request (runs once)."""
    if not getattr(_ensure_tables, '_done', False):
        try:
            ensure_prospect_tables(get_engine())
            _ensure_tables._done = True
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Deal CRUD
# ---------------------------------------------------------------------------

@prospects_bp.route('', methods=['GET'])
@login_required
def list_prospect_deals():
    """List prospect deals with optional stage/assigned_to filters."""
    stage = request.args.get('stage')
    assigned_to = request.args.get('assigned_to')
    deals = list_deals(get_engine(), stage=stage, assigned_to=assigned_to)
    return jsonify(deals)


@prospects_bp.route('', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def create_prospect_deal():
    """Create a new prospect deal."""
    data = request.json
    if not data or not data.get('deal_name'):
        return jsonify({'error': 'deal_name is required'}), 400
    username = g.current_user.get('username', 'unknown')
    result = create_deal(get_engine(), data, username)
    return jsonify({'id': result['id'], 'vcode': result['vcode'], 'status': 'created'}), 201


@prospects_bp.route('/<int:deal_id>', methods=['GET'])
@login_required
def get_prospect_deal(deal_id):
    """Get full deal detail including properties, entities, and investors."""
    result = get_deal(get_engine(), deal_id)
    if not result:
        return jsonify({'error': 'Deal not found'}), 404
    return jsonify(result)


@prospects_bp.route('/<int:deal_id>', methods=['PUT'])
@login_required
@role_required('admin', 'analyst')
def update_prospect_deal(deal_id):
    """Update a prospect deal."""
    data = request.json
    username = g.current_user.get('username', 'unknown')
    if not update_deal(get_engine(), deal_id, data, username):
        return jsonify({'error': 'Deal not found'}), 404
    return jsonify({'status': 'updated'})


@prospects_bp.route('/<int:deal_id>', methods=['DELETE'])
@login_required
@role_required('admin')
def delete_prospect_deal(deal_id):
    """Delete a prospect deal and all related data."""
    if not delete_deal(get_engine(), deal_id):
        return jsonify({'error': 'Deal not found'}), 404
    return jsonify({'status': 'deleted'})


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/properties', methods=['GET'])
@login_required
def get_properties(deal_id):
    """List properties for a prospect deal."""
    props = list_properties(get_engine(), deal_id)
    return jsonify(props)


@prospects_bp.route('/<int:deal_id>/properties', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def add_property(deal_id):
    """Add a property to a prospect deal."""
    data = request.json
    if not data or not data.get('property_name'):
        return jsonify({'error': 'property_name is required'}), 400
    username = g.current_user.get('username', 'unknown')
    result = create_property(get_engine(), deal_id, data, username)
    return jsonify({'id': result['id'], 'vcode': result['vcode'], 'status': 'created'}), 201


@prospects_bp.route('/<int:deal_id>/properties/<int:prop_id>', methods=['PUT'])
@login_required
@role_required('admin', 'analyst')
def edit_property(deal_id, prop_id):
    """Update a prospect property."""
    data = request.json
    if not update_property(get_engine(), prop_id, data):
        return jsonify({'error': 'Property not found'}), 404
    return jsonify({'status': 'updated'})


@prospects_bp.route('/<int:deal_id>/properties/<int:prop_id>', methods=['DELETE'])
@login_required
@role_required('admin', 'analyst')
def remove_property(deal_id, prop_id):
    """Remove a property from a prospect deal."""
    if not delete_property(get_engine(), prop_id):
        return jsonify({'error': 'Property not found'}), 404
    return jsonify({'status': 'deleted'})


# ---------------------------------------------------------------------------
# Entities & Investors
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/entities', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def add_entity(deal_id):
    """Add an entity to a prospect deal."""
    data = request.json
    if not data or not data.get('entity_name'):
        return jsonify({'error': 'entity_name is required'}), 400
    entity_id = create_entity(get_engine(), deal_id, data)
    return jsonify({'id': entity_id, 'status': 'created'}), 201


@prospects_bp.route('/<int:deal_id>/entities/<int:entity_id>', methods=['PUT'])
@login_required
@role_required('admin', 'analyst')
def edit_entity(deal_id, entity_id):
    """Update a prospect entity."""
    data = request.json
    if not update_entity(get_engine(), entity_id, data):
        return jsonify({'error': 'Entity not found'}), 404
    return jsonify({'status': 'updated'})


@prospects_bp.route('/<int:deal_id>/entities/<int:entity_id>', methods=['DELETE'])
@login_required
@role_required('admin', 'analyst')
def remove_entity(deal_id, entity_id):
    """Remove an entity and its investors."""
    if not delete_entity(get_engine(), entity_id):
        return jsonify({'error': 'Entity not found'}), 404
    return jsonify({'status': 'deleted'})


@prospects_bp.route('/<int:deal_id>/entities/<int:entity_id>/investors', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def add_investor(deal_id, entity_id):
    """Add an investor to an entity."""
    data = request.json
    if not data or not data.get('investor_name'):
        return jsonify({'error': 'investor_name is required'}), 400
    inv_id = create_investor(get_engine(), entity_id, data)
    return jsonify({'id': inv_id, 'status': 'created'}), 201


@prospects_bp.route('/<int:deal_id>/investors/<int:inv_id>', methods=['PUT'])
@login_required
@role_required('admin', 'analyst')
def edit_investor(deal_id, inv_id):
    """Update an investor."""
    data = request.json
    if not update_investor(get_engine(), inv_id, data):
        return jsonify({'error': 'Investor not found'}), 404
    return jsonify({'status': 'updated'})


@prospects_bp.route('/<int:deal_id>/investors/<int:inv_id>', methods=['DELETE'])
@login_required
@role_required('admin', 'analyst')
def remove_investor(deal_id, inv_id):
    """Remove an investor."""
    if not delete_investor(get_engine(), inv_id):
        return jsonify({'error': 'Investor not found'}), 404
    return jsonify({'status': 'deleted'})


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/activity', methods=['GET'])
@login_required
def get_deal_activity(deal_id):
    """Get activity log for a prospect deal."""
    activity = get_activity(get_engine(), deal_id)
    return jsonify(activity)


@prospects_bp.route('/<int:deal_id>/activity', methods=['POST'])
@login_required
def add_deal_note(deal_id):
    """Add a note to the activity log."""
    data = request.json
    if not data or not data.get('note'):
        return jsonify({'error': 'note is required'}), 400
    username = g.current_user.get('username', 'unknown')
    note_id = add_activity_note(get_engine(), deal_id, username, data['note'])
    return jsonify({'id': note_id, 'status': 'created'}), 201


# ---------------------------------------------------------------------------
# Lease Review Link
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/properties/<int:prop_id>/lease-review', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def create_property_lease_review(deal_id, prop_id):
    """Create a lease review linked to a prospect property."""
    username = g.current_user.get('username', 'unknown')
    try:
        review_id = create_lease_review_for_property(
            get_engine(), deal_id, prop_id, username
        )
        return jsonify({'review_id': review_id, 'status': 'created'}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@prospects_bp.route('/<int:deal_id>/lease-reviews', methods=['GET'])
@login_required
def get_deal_lease_reviews(deal_id):
    """List all lease reviews for properties in this deal."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT lr.id, lr.property_name, lr.status, lr.total_tenants,
                   lr.total_gla, lr.total_annual_rent, lr.created_at,
                   pp.id as property_id, pp.property_name as prop_name
            FROM lease_reviews lr
            JOIN prospect_properties pp ON pp.id = lr.prospect_property_id
            WHERE pp.prospect_id = :did
            ORDER BY pp.sort_order, pp.property_name
        """), {'did': deal_id}).fetchall()

    return jsonify([{
        'review_id': r[0], 'property_name': r[1], 'status': r[2],
        'total_tenants': r[3], 'total_gla': r[4], 'total_annual_rent': r[5],
        'created_at': str(r[6]) if r[6] else None,
        'property_id': r[7], 'prospect_property_name': r[8],
    } for r in rows])


# ---------------------------------------------------------------------------
# Assumptions CRUD
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/assumptions', methods=['GET'])
@login_required
def get_deal_assumptions(deal_id):
    """List assumption versions for a deal."""
    versions = list_assumptions(get_engine(), deal_id)
    return jsonify(versions)


@prospects_bp.route('/<int:deal_id>/assumptions', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def save_deal_assumptions(deal_id):
    """Create or update an assumption version."""
    data = request.json or {}
    result = save_assumptions(get_engine(), deal_id, data)
    return jsonify(result), 201


@prospects_bp.route('/<int:deal_id>/assumptions/<int:assumption_id>', methods=['GET'])
@login_required
def get_deal_assumption(deal_id, assumption_id):
    """Get a single assumption version."""
    result = get_assumption(get_engine(), assumption_id)
    if not result or result.get('prospect_id') != deal_id:
        return jsonify({'error': 'Assumption not found'}), 404
    return jsonify(result)


@prospects_bp.route('/<int:deal_id>/assumptions/<int:assumption_id>', methods=['DELETE'])
@login_required
@role_required('admin', 'analyst')
def delete_deal_assumption(deal_id, assumption_id):
    """Delete an assumption version."""
    if not delete_assumptions(get_engine(), assumption_id):
        return jsonify({'error': 'Assumption not found'}), 404
    return jsonify({'status': 'deleted'})


# ---------------------------------------------------------------------------
# Deal Analysis (compute returns from assumptions)
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/analyze', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def analyze_deal(deal_id):
    """Run deal analysis using assumptions and the shared compute engine.

    Accepts optional assumption overrides in request body.
    If 'assumption_id' is provided, loads saved assumptions as base.
    """
    from flask_app.serializers import safe_json
    from prospect_analysis import build_prospect_analysis

    deal_data = get_deal(get_engine(), deal_id)
    if not deal_data:
        return jsonify({'error': 'Deal not found'}), 404

    body = request.json or {}

    # Load saved assumptions or use request body
    assumption_id = body.get('assumption_id')
    if assumption_id:
        assumptions = get_assumption(get_engine(), int(assumption_id))
        if not assumptions:
            return jsonify({'error': 'Assumption version not found'}), 404
        # Allow body to override individual fields
        for k, v in body.items():
            if k in assumptions and v is not None:
                assumptions[k] = v
    else:
        assumptions = body

    # Apply acquisition overrides from request body
    for field in ['purchase_price', 'closing_cost_pct', 'capex_at_close']:
        override = body.get(f'{field}_override')
        if override is not None:
            deal_data['deal'][field] = override

    # Apply property-level price overrides from capital budget
    prop_prices = body.get('property_prices')
    if prop_prices and isinstance(prop_prices, dict):
        for prop in deal_data.get('properties', []):
            pid = str(prop['id'])
            if pid in prop_prices and prop_prices[pid] is not None:
                prop['property_price'] = prop_prices[pid]

    # Check for property-level cash flows (Argus or Excel uploads)
    # Priority: Argus imports > prospect_cashflows > NOI growth assumptions
    argus_forecast_df = None
    property_cashflows = None
    property_ids = [p['id'] for p in deal_data.get('properties', [])]

    if property_ids:
        deal_vcode = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"
        target_close = deal_data['deal'].get('target_close')
        try:
            import pandas as pd
            close_yr = pd.to_datetime(target_close).year if target_close else 2026
        except Exception:
            close_yr = 2026

        # Try Argus first
        try:
            from flask_app.services.argus_service import get_property_rollup_forecast_df
            argus_forecast_df = get_property_rollup_forecast_df(
                get_engine(), deal_vcode, property_ids, close_yr - 1,
            )
        except Exception as e:
            logger.debug("Argus property rollup check: %s", e)

        # Fall back to prospect_cashflows if no Argus data
        if argus_forecast_df is None:
            try:
                by_prop = get_deal_cashflows_by_property(get_engine(), deal_id)
                if by_prop:
                    # Flatten all property cashflows into one list
                    all_cfs = []
                    for pid, cfs in by_prop.items():
                        all_cfs.extend(cfs)
                    if all_cfs:
                        property_cashflows = all_cfs
            except Exception as e:
                logger.debug("Property cashflows check: %s", e)

    # Check for a real waterfall in the DB for this prospect's vcode
    waterfall_df = None
    try:
        from loaders import load_waterfalls
        from sqlalchemy import text as sa_text
        wf_vcode = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"
        with get_engine().connect() as conn:
            wf_rows = conn.execute(sa_text(
                "SELECT * FROM waterfalls WHERE vcode = :v"
            ), {"v": wf_vcode}).fetchall()
        if wf_rows:
            import pandas as pd
            cols = [c for c in wf_rows[0]._mapping.keys()]
            wf_raw = pd.DataFrame([dict(r._mapping) for r in wf_rows], columns=cols)
            waterfall_df = load_waterfalls(wf_raw)
    except Exception as e:
        logger.debug("Waterfall DB check for prospect: %s", e)

    try:
        result = build_prospect_analysis(
            deal=deal_data['deal'],
            properties=deal_data['properties'],
            entities=deal_data['entities'],
            assumptions=assumptions,
            cashflows=property_cashflows,
            argus_forecast_df=argus_forecast_df,
            waterfall_df=waterfall_df,
        )
    except Exception as e:
        logger.exception("Prospect analysis failed for deal %d", deal_id)
        return jsonify({'error': str(e)}), 500

    if 'error' in result and 'partner_results' not in result:
        return jsonify({'error': result['error']}), 400

    # Build annual forecast table for display
    annual_forecast = None
    try:
        from reporting import annual_aggregation_table
        fc_display = result.get('fc_deal_display') or result.get('fc_deal_modeled')
        if fc_display is not None and not fc_display.empty:
            hold_years = assumptions.get('hold_years', 7)
            start_year = result.get('model_start')
            if start_year and hasattr(start_year, 'year'):
                start_year = start_year.year
            else:
                start_year = 2026
            aat = annual_aggregation_table(
                fc_display, start_year, hold_years,
                cf_alloc=result.get('cf_alloc'),
                cap_alloc=result.get('cap_alloc'),
                cash_schedule=result.get('cash_schedule'),
            )
            if aat is not None and not aat.empty:
                years = [int(y) for y in aat.index.tolist()]
                rows = []
                pct_rows = {'Debt Service Coverage Ratio'}
                underline_rows = {'Expenses', 'Capital Expenditures', 'Other Below-the-Line'}
                topline_rows = set()
                for col in aat.columns:
                    vals = {}
                    for yr in years:
                        v = aat.loc[yr, col] if yr in aat.index else None
                        if v is not None and not (isinstance(v, float) and (v != v)):
                            vals[yr] = v
                    rows.append({
                        'label': col,
                        'values': safe_json(vals),
                        'is_pct': col in pct_rows,
                        'is_header': col.endswith(':') or col == '',
                        'underline': col in underline_rows,
                        'topline': col in topline_rows,
                    })
                annual_forecast = {'years': years, 'rows': rows}
    except Exception as e:
        logger.warning("Annual forecast build failed: %s", e)

    # Build debt service summary
    debt_service = None
    loan_sched = result.get('loan_sched')
    if loan_sched is not None and not loan_sched.empty:
        try:
            ls = loan_sched.copy()
            ls['Year'] = ls['event_date'].apply(lambda d: d.year if hasattr(d, 'year') else d)
            annual_ds = ls.groupby('Year').agg(
                interest=('interest', 'sum'),
                principal=('principal', 'sum'),
            ).reset_index()
            annual_ds['total'] = annual_ds['interest'] + annual_ds['principal']
            debt_service = safe_json(annual_ds.to_dict('records'))
        except Exception:
            pass

    return jsonify(safe_json({
        'vcode': result.get('prospect_assumptions', {}).get('close_date', ''),
        'partner_results': result.get('partner_results', []),
        'deal_summary': result.get('deal_summary', {}),
        'debug_msgs': result.get('debug_msgs', []),
        'prospect_assumptions': result.get('prospect_assumptions', {}),
        'annual_forecast': annual_forecast,
        'debt_service': debt_service,
        'sale_dbg': result.get('sale_dbg'),
        'cap_data': result.get('cap_data', {}),
    }))


# ---------------------------------------------------------------------------
# Property Cashflows (Excel/CSV upload)
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/properties/<int:property_id>/cashflows/upload',
                    methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def upload_property_cashflows(deal_id, property_id):
    """Upload a partner Excel/CSV cash flow model for a property.

    Parses the file, auto-detects columns, and stores monthly cash flows
    in prospect_cashflows.
    """
    from cashflow_parser import parse_cashflow_excel

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'No file selected'}), 400

    file_bytes = f.read()
    filename = f.filename

    parsed = parse_cashflow_excel(file_bytes, filename)
    if 'error' in parsed:
        return jsonify(parsed), 400

    cashflows = parsed['cashflows']
    if not cashflows:
        return jsonify({'error': 'No cash flow rows parsed'}), 400

    try:
        result = import_property_cashflows(
            get_engine(), deal_id, property_id,
            cashflows, source='excel',
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.exception("Cashflow import failed for property %d", property_id)
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'status': 'imported',
        'rows_imported': result['rows_imported'],
        'frequency': parsed['frequency'],
        'periods': parsed['periods'],
        'columns_detected': parsed['columns_detected'],
        'metadata': parsed['metadata'],
        'cashflows': cashflows,
    })


@prospects_bp.route('/<int:deal_id>/properties/<int:property_id>/cashflows',
                    methods=['GET'])
@login_required
def get_cashflows(deal_id, property_id):
    """Get stored cashflows for a property."""
    version = request.args.get('version', 1, type=int)
    rows = get_property_cashflows(get_engine(), deal_id, property_id, version)
    return jsonify({'cashflows': rows, 'count': len(rows)})


@prospects_bp.route('/<int:deal_id>/properties/<int:property_id>/cashflows',
                    methods=['DELETE'])
@login_required
@role_required('admin', 'analyst')
def clear_cashflows(deal_id, property_id):
    """Delete stored cashflows for a property."""
    version = request.args.get('version', 1, type=int)
    deleted = delete_property_cashflows(get_engine(), deal_id, property_id, version)
    return jsonify({'status': 'deleted' if deleted else 'nothing_to_delete'})


@prospects_bp.route('/<int:deal_id>/cashflow-status', methods=['GET'])
@login_required
def cashflow_status(deal_id):
    """Get cashflow load status (source + timestamp) for all properties in a deal."""
    engine = get_engine()
    props = list_properties(engine, deal_id)
    property_ids = [p['id'] for p in props]
    status = get_property_cashflow_status(engine, deal_id, property_ids)
    return jsonify(status)


# ---------------------------------------------------------------------------
# Waterfall Builder (streamlined structure creation)
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/waterfall', methods=['GET'])
@login_required
def get_deal_waterfall(deal_id):
    """Get waterfall steps for a prospect deal."""
    deal_data = get_deal(get_engine(), deal_id)
    if not deal_data:
        return jsonify({'error': 'Deal not found'}), 404

    vcode = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"

    from sqlalchemy import text as sa_text
    with get_engine().connect() as conn:
        rows = conn.execute(sa_text(
            "SELECT vcode, vmisc, \"iOrder\", \"PropCode\", \"vState\", "
            "\"FXRate\", \"nPercent\", \"mAmount\", vtranstype, \"vAmtType\", "
            "\"vNotes\" FROM waterfalls WHERE vcode = :v ORDER BY vmisc, \"iOrder\""
        ), {"v": vcode}).fetchall()

    steps = []
    for r in rows:
        steps.append({
            'vcode': r[0], 'vmisc': r[1], 'iOrder': r[2],
            'PropCode': r[3], 'vState': r[4], 'FXRate': r[5],
            'nPercent': r[6], 'mAmount': r[7], 'vtranstype': r[8],
            'vAmtType': r[9], 'vNotes': r[10],
        })

    return jsonify({
        'vcode': vcode,
        'steps': steps,
        'has_cf': any(s['vmisc'] == 'CF_WF' for s in steps),
        'has_cap': any(s['vmisc'] == 'Cap_WF' for s in steps),
    })


@prospects_bp.route('/<int:deal_id>/waterfall/build', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def build_deal_waterfall(deal_id):
    """Generate and save waterfall from independent CF and Cap step inputs.

    Body: {
        cf_steps: [{entity_id, step_type, rate, amount}, ...],
        cap_steps: [{entity_id, step_type, rate, amount}, ...],
    }

    Also supports legacy format with 'investors' key for backward compat.
    """
    from database import save_waterfall_steps
    from flask_app.services.data_service import get_data_service

    deal_data = get_deal(get_engine(), deal_id)
    if not deal_data:
        return jsonify({'error': 'Deal not found'}), 404

    vcode = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"
    body = request.json or {}

    import pandas as pd
    from datetime import date as dt_date

    def _convert_step_inputs(step_inputs, wf_name):
        """Convert UI step inputs to waterfall rows."""
        rows = []
        order = 10
        lead_found = False
        for s in step_inputs:
            eid = s.get('entity_id', '')
            stype = s.get('step_type', '')
            if not eid:
                continue
            if stype == 'pref':
                rate = float(s.get('rate') or 0)
                rows.append({
                    'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
                    'PropCode': eid, 'vState': 'Pref',
                    'FXRate': 1.0, 'nPercent': rate,
                    'mAmount': 0, 'vtranstype': 'Preferred Return',
                    'vAmtType': '', 'vNotes': '',
                    'dteffective': dt_date(2020, 1, 1), 'nmisc': 0,
                })
                order += 10
            elif stype == 'return_of_capital':
                rows.append({
                    'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
                    'PropCode': eid, 'vState': 'Initial',
                    'FXRate': 1.0, 'nPercent': 0, 'mAmount': 0,
                    'vtranstype': 'Return of Capital',
                    'vAmtType': '', 'vNotes': '',
                    'dteffective': dt_date(2020, 1, 1), 'nmisc': 0,
                })
                order += 10
            elif stype == 'residual':
                share = float(s.get('rate') or 0) / 100
                state = 'Share' if not lead_found else 'Tag'
                lead_found = True
                rows.append({
                    'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
                    'PropCode': eid, 'vState': state,
                    'FXRate': share, 'nPercent': 0, 'mAmount': 0,
                    'vtranstype': 'Excess Cash Flow',
                    'vAmtType': '', 'vNotes': '',
                    'dteffective': dt_date(2020, 1, 1), 'nmisc': 0,
                })
                order += 10
            elif stype == 'fixed_amount':
                amt = float(s.get('amount') or 0)
                rows.append({
                    'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
                    'PropCode': eid, 'vState': 'Amt',
                    'FXRate': 0, 'nPercent': 0, 'mAmount': amt,
                    'vtranstype': 'Fixed Amount',
                    'vAmtType': '', 'vNotes': '',
                    'dteffective': dt_date(2020, 1, 1), 'nmisc': 0,
                })
                order += 10
            elif stype == 'irr_lookback':
                rate = float(s.get('rate') or 0)
                rows.append({
                    'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
                    'PropCode': eid, 'vState': 'IRR',
                    'FXRate': 0, 'nPercent': rate, 'mAmount': 0,
                    'vtranstype': 'IRR Hurdle',
                    'vAmtType': '', 'vNotes': '',
                    'dteffective': dt_date(2020, 1, 1), 'nmisc': 0,
                })
                order += 10
        return rows

    cf_inputs = body.get('cf_steps', [])
    cap_inputs = body.get('cap_steps', [])

    if not cf_inputs and not cap_inputs:
        return jsonify({'error': 'At least one waterfall step required'}), 400

    steps = _convert_step_inputs(cf_inputs, 'CF_WF') + _convert_step_inputs(cap_inputs, 'Cap_WF')

    # Save to database
    df = pd.DataFrame(steps)
    try:
        save_waterfall_steps(vcode, df)
        try:
            ds = get_data_service()
            if ds:
                ds.refresh_table('waterfalls')
                ds.clear_cache()
        except Exception:
            pass
    except Exception as e:
        logger.exception("Failed to save waterfall for %s", vcode)
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'status': 'saved',
        'vcode': vcode,
        'step_count': len(steps),
        'steps': steps,
    })


@prospects_bp.route('/<int:deal_id>/waterfall', methods=['DELETE'])
@login_required
@role_required('admin', 'analyst')
def delete_deal_waterfall(deal_id):
    """Delete waterfall steps for a prospect deal."""
    deal_data = get_deal(get_engine(), deal_id)
    if not deal_data:
        return jsonify({'error': 'Deal not found'}), 404

    vcode = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"

    from sqlalchemy import text as sa_text
    with get_engine().connect() as conn:
        result = conn.execute(sa_text(
            "DELETE FROM waterfalls WHERE vcode = :v"
        ), {"v": vcode})
        conn.commit()

    return jsonify({'status': 'deleted', 'rows': result.rowcount})
