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
    import_property_line_items, get_property_line_items,
    get_deal_line_items_by_property,
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
    data['_username'] = getattr(g, 'current_user', {}).get('username', '')
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

    prepared = _run_prospect_analysis(deal_id)
    if isinstance(prepared, tuple):
        return prepared  # an error response
    result = prepared['result']
    deal_data = prepared['deal_data']
    assumptions = prepared['assumptions']

    if 'error' in result and 'partner_results' not in result:
        return jsonify({'error': result['error']}), 400

    return _continue_analyze(result, deal_data, assumptions)


def _run_prospect_analysis(deal_id):
    """The compute half of /analyze, shared with the Excel export.

    Returns {'result', 'deal_data', 'assumptions', 'scenario'} with the raw
    DataFrames intact, or a (response, status) tuple on error -- so the audit
    workbook is built from exactly the computation the screen shows.
    """
    from prospect_analysis import build_prospect_analysis

    deal_data = get_deal(get_engine(), deal_id)
    if not deal_data:
        return jsonify({'error': 'Deal not found'}), 404

    body = request.get_json(silent=True) or {}

    # Load saved assumptions or use request body
    assumption_id = body.get('assumption_id')
    if assumption_id:
        assumptions = get_assumption(get_engine(), int(assumption_id))
        if not assumptions:
            return jsonify({'error': 'Assumption version not found'}), 404
        # Allow body to override individual fields or add new ones
        for k, v in body.items():
            if v is not None:
                assumptions[k] = v
    else:
        assumptions = body
        # A bare call (the Excel export, an API client) carries no form
        # state. Running on empty assumptions silently produces a deal with
        # no debt and phantom equity, so fall back to the deal's most recent
        # saved version -- the thing an audit should reflect anyway.
        from flask_app.services.prospect_service import ASSUMPTION_FIELDS
        if not any(body.get(f) is not None for f in ASSUMPTION_FIELDS):
            from sqlalchemy import text as _sa_text
            with get_engine().connect() as _conn:
                _arow = _conn.execute(_sa_text(
                    "SELECT * FROM prospect_assumptions WHERE prospect_id = :d "
                    "ORDER BY id DESC LIMIT 1"), {"d": deal_id}).mappings().fetchone()
            if _arow:
                assumptions = {**dict(_arow), **{k: v for k, v in body.items() if v is not None}}

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
    # Scenario: a named binding of cash flow source, assumption overrides
    # and adjustment events. Loaded here so its source choice can steer the
    # cascade below.
    scenario = None
    scenario_id = (request.get_json(silent=True) or {}).get('scenario_id')
    if scenario_id:
        from flask_app.services import scenario_service
        scenario = scenario_service.get_scenario(get_engine(), int(scenario_id))
        if not scenario or scenario.get('prospect_id') != deal_id:
            return jsonify({'error': f'Scenario {scenario_id} not found on this deal'}), 404

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
            from flask_app.services.argus_service import (
                get_property_rollup_forecast_df, get_forecast_df_by_id)
            # A scenario can pin a specific Argus import per property
            # ("OP Model" vs "Base Case UW"); otherwise the active one wins.
            pinned = (scenario or {}).get('argus_import_ids') or {}
            if pinned:
                frames = []
                for pid in property_ids:
                    imp_id = pinned.get(str(pid)) or pinned.get(pid)
                    if not imp_id:
                        continue
                    pdf = get_forecast_df_by_id(
                        get_engine(), f"NP{pid:06d}", int(imp_id), close_yr - 1)
                    if pdf is not None and not pdf.empty:
                        pdf = pdf.copy()
                        pdf["vcode"] = deal_vcode
                        frames.append(pdf)
                if frames:
                    import pandas as _pd
                    argus_forecast_df = _pd.concat(frames, ignore_index=True)
            if argus_forecast_df is None:
                argus_forecast_df = get_property_rollup_forecast_df(
                get_engine(), deal_vcode, property_ids, close_yr - 1,
            )
        except Exception as e:
            logger.debug("Argus property rollup check: %s", e)

        # Fall back to prospect_cashflows if no Argus data
        # Priority: line-item data (with vAccount) > old summary-format
        if argus_forecast_df is None:
            try:
                line_items_by_prop = get_deal_line_items_by_property(
                    get_engine(), deal_id)
                if line_items_by_prop:
                    # Flatten all line-item cashflows into one list
                    all_line_items = []
                    for pid, items in line_items_by_prop.items():
                        all_line_items.extend(items)
                    if all_line_items:
                        property_cashflows = all_line_items
                        assumptions['_has_line_items'] = True
            except Exception as e:
                logger.debug("Line-item cashflows check: %s", e)

            # Fall back to old summary-format
            if property_cashflows is None:
                try:
                    by_prop = get_deal_cashflows_by_property(
                        get_engine(), deal_id)
                    if by_prop:
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
            # ORDER BY is load-bearing: when the PE flag is ambiguous the
            # equity split falls back to investor order, so it must be stable.
            wf_rows = conn.execute(sa_text(
                'SELECT * FROM waterfalls WHERE vcode = :v '
                'ORDER BY vmisc, "iOrder"'
            ), {"v": wf_vcode}).fetchall()
        if wf_rows:
            import pandas as pd
            cols = [c for c in wf_rows[0]._mapping.keys()]
            wf_raw = pd.DataFrame([dict(r._mapping) for r in wf_rows], columns=cols)
            waterfall_df = load_waterfalls(wf_raw)
    except Exception as e:
        logger.debug("Waterfall DB check for prospect: %s", e)

    try:
        # Parcel sales saved for this deal feed the projection (property-bound
        # via property_vcode; effects aggregate at deal level).
        parcel_sales = None
        _pv = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"
        try:
            from flask_app.services import parcel_sale_service as _pss
            parcel_sales = _pss.list_parcel_sales(get_engine(), _pv)
        except Exception as _pe:
            logger.debug("parcel sales load for %s: %s", _pv, _pe)

        result = build_prospect_analysis(
            deal=deal_data['deal'],
            properties=deal_data['properties'],
            entities=deal_data['entities'],
            assumptions=assumptions,
            cashflows=property_cashflows,
            argus_forecast_df=argus_forecast_df,
            waterfall_df=waterfall_df,
            scenario=scenario,
            parcel_sales=parcel_sales,
        )
    except Exception as e:
        logger.exception("Prospect analysis failed for deal %d", deal_id)
        return jsonify({'error': str(e)}), 500

    return {'result': result, 'deal_data': deal_data,
            'assumptions': assumptions, 'scenario': scenario}


def _continue_analyze(result, deal_data, assumptions):
    from flask_app.serializers import safe_json

    # Build anniversary-based annual forecast table for display
    annual_forecast = None
    try:
        import pandas as pd
        from dateutil.relativedelta import relativedelta
        from config import (IS_ACCOUNTS, REVENUE_ACCTS, EXPENSE_ACCTS,
                            INTEREST_ACCTS, PRINCIPAL_ACCTS, CAPEX_ACCTS,
                            TAX_ABATEMENT_ACCTS, OTHER_EXCLUDED_ACCTS)

        fc_display = result.get('fc_deal_display')
        if fc_display is None:
            fc_display = result.get('fc_deal_modeled')

        if fc_display is not None and not fc_display.empty:
            hold_years = int(assumptions.get('hold_years') or 7)
            close_date_str = assumptions.get('target_close') or deal_data['deal'].get('target_close')
            if close_date_str:
                closing_dt = pd.Timestamp(close_date_str)
            else:
                closing_dt = pd.Timestamp(result.get('model_start') or '2026-01-01')

            # Anniversary year mapping: Year N covers months [closing + (N-1)*12 .. closing + N*12 - 1]
            # Extra year beyond hold for terminal NOI visibility
            num_years = hold_years + 1
            anniv_years = list(range(1, num_years + 1))

            # Build column metadata: year number + month-end label
            columns_meta = []
            columns_meta.append({'year': 0, 'label': 'In-Place NOI', 'sublabel': ''})
            for n in anniv_years:
                period_end = closing_dt + relativedelta(years=n, months=-1)
                columns_meta.append({
                    'year': n,
                    'label': str(n),
                    'sublabel': period_end.strftime('%b-%Y'),
                })
            all_year_keys = [0] + anniv_years  # 0 = In-Place NOI placeholder

            def _anniv_year(dt):
                """Map a date to anniversary year number (1-based)."""
                months_diff = (dt.year - closing_dt.year) * 12 + (dt.month - closing_dt.month)
                if months_diff < 0:
                    return 0
                return months_diff // 12 + 1

            # Prepare forecast data with anniversary year
            fc = fc_display.copy()
            fc['_dt'] = pd.to_datetime(fc['event_date'])
            fc['_ayr'] = fc['_dt'].apply(_anniv_year)
            fc['_acct'] = pd.to_numeric(fc['vAccount'], errors='coerce').astype('Int64')
            # Include data through num_years (hold + 1 extra for terminal NOI)
            fc = fc[fc['_ayr'].between(1, num_years)]

            def _sum_accts(acct_set):
                """Sum mAmount_norm by anniversary year for given accounts."""
                subset = fc[fc['_acct'].isin(acct_set)]
                if subset.empty:
                    return {}
                s = subset.groupby('_ayr')['mAmount_norm'].sum()
                return {int(k): v for k, v in s.items() if not (isinstance(v, float) and v != v)}

            rev_sums = _sum_accts(REVENUE_ACCTS)
            exp_sums = _sum_accts(EXPENSE_ACCTS)
            interest_sums = _sum_accts(INTEREST_ACCTS)
            principal_sums = _sum_accts(PRINCIPAL_ACCTS)
            capex_sums = _sum_accts(CAPEX_ACCTS)
            tax_abate_sums = _sum_accts(TAX_ABATEMENT_ACCTS)
            other_btl_sums = _sum_accts(OTHER_EXCLUDED_ACCTS)

            noi_sums = {}
            tds_sums = {}
            fad_sums = {}
            dscr_sums = {}
            for yr in anniv_years:
                rev = rev_sums.get(yr, 0)
                exp = exp_sums.get(yr, 0)
                noi = rev + exp
                noi_sums[yr] = noi
                intr = interest_sums.get(yr, 0)
                princ = principal_sums.get(yr, 0)
                tds = intr + princ
                tds_sums[yr] = tds
                capex = capex_sums.get(yr, 0)
                tax_a = tax_abate_sums.get(yr, 0)
                other = other_btl_sums.get(yr, 0)
                fad = noi + tax_a + tds + other + capex
                fad_sums[yr] = fad
                if abs(tds) > 0:
                    dscr_sums[yr] = noi / abs(tds)

            # Detail revenue/expense line items
            rev_details = []
            exp_details = []
            try:
                for section, detail_list, acct_set in [
                    ('REVENUES', rev_details, REVENUE_ACCTS),
                    ('EXPENSES', exp_details, EXPENSE_ACCTS),
                ]:
                    for label, accts in IS_ACCOUNTS.get(section, {}).items():
                        target = {int(a) for a in accts} & acct_set
                        if not target:
                            continue
                        subset = fc[fc['_acct'].isin(target)]
                        if subset.empty:
                            continue
                        sums = subset.groupby('_ayr')['mAmount_norm'].sum()
                        vals = {}
                        for yr in anniv_years:
                            v = sums.get(yr)
                            if v is not None and not (isinstance(v, float) and v != v) and abs(v) > 0.5:
                                vals[yr] = v
                        if vals:
                            detail_list.append({
                                'label': f'  {label}',
                                'values': safe_json(vals),
                                'is_pct': False, 'is_header': False,
                                'underline': False, 'topline': False,
                            })
            except Exception as detail_err:
                logger.warning("Detail line items failed: %s", detail_err)

            # Build waterfall allocation rows by anniversary year
            cf_alloc = result.get('cf_alloc')
            cap_alloc = result.get('cap_alloc')

            def _build_wf_rows(alloc_df, section_label):
                """Build waterfall step rows grouped by anniversary year."""
                wf_rows = []
                if alloc_df is None or alloc_df.empty:
                    return wf_rows
                adf = alloc_df.copy()
                adf['_dt'] = pd.to_datetime(adf['event_date'])
                adf['_ayr'] = adf['_dt'].apply(_anniv_year)
                adf = adf[adf['_ayr'].between(1, hold_years)]  # waterfall stops at hold period

                step_cols = ['iOrder', 'vAmtType', 'PropCode', 'vState']
                avail = [c for c in step_cols if c in adf.columns]
                if not avail:
                    return wf_rows

                adf['_key'] = [
                    f"  {int(r.get('iOrder', 0)):>2} | {r.get('vAmtType', '')} | {r.get('PropCode', '')} | {r.get('vState', '')}"
                    for _, r in adf.iterrows()
                ]

                wf_rows.append({'label': '', 'values': {}, 'is_header': True})
                wf_rows.append({'label': f'{section_label}:', 'values': {yr: 0 for yr in anniv_years[:hold_years]},
                                'is_header': True, 'isBold': True})

                sorted_df = adf.drop_duplicates('_key').sort_values('iOrder')
                for _, srow in sorted_df.iterrows():
                    key = srow['_key']
                    step_data = adf[adf['_key'] == key]
                    sums = step_data.groupby('_ayr')['Allocated'].sum()
                    vals = {}
                    for yr in anniv_years[:hold_years]:
                        vals[yr] = sums.get(yr, 0.0)
                    wf_rows.append({
                        'label': key, 'values': safe_json(vals),
                        'is_pct': False, 'is_header': False,
                    })
                return wf_rows

            cf_wf_rows = _build_wf_rows(cf_alloc, 'CF Waterfall')

            # Proceeds row between CF and Cap waterfall
            proceeds_row = {'label': 'Proceeds from Sale or Refinancing', 'values': {}}
            sale_dbg = result.get('sale_dbg')
            if sale_dbg:
                sale_yr = _anniv_year(pd.Timestamp(sale_dbg.get('sale_date', '2030-12-31')))
                net_proceeds = sale_dbg.get('net_sale_proceeds', 0)
                if sale_yr >= 1 and net_proceeds:
                    proceeds_row['values'] = {sale_yr: net_proceeds}

            cap_wf_rows = _build_wf_rows(cap_alloc, 'Capital Waterfall')

            # Partner totals by anniversary year (contributions negative, distributions positive)
            partner_total_rows = []
            partner_results = result.get('partner_results', [])
            if partner_results:
                partner_total_rows.append({'label': '', 'values': {}, 'is_header': True})
                partner_total_rows.append({'label': 'Partner Totals:', 'values': {},
                                           'is_header': True, 'isBold': True})

                deal_totals = {}
                for pr in partner_results:
                    pid = pr.get('partner', '')
                    cfs = pr.get('cashflow_details', [])
                    vals = {}
                    for cf in cfs:
                        d = cf.get('Date')
                        if d is None:
                            continue
                        dt = pd.Timestamp(d)
                        ayr = _anniv_year(dt)
                        amt = cf.get('Amount', 0)
                        # Contributions are already negative in cashflow_details
                        if ayr == 0:
                            # Pre-closing contributions go under In-Place NOI column
                            vals[0] = vals.get(0, 0) + amt
                        elif 1 <= ayr <= hold_years:
                            vals[ayr] = vals.get(ayr, 0) + amt
                    for yr, v in vals.items():
                        deal_totals[yr] = deal_totals.get(yr, 0) + v
                    partner_total_rows.append({
                        'label': f'  {pid} Total', 'values': safe_json(vals),
                        'is_pct': False, 'is_header': False,
                    })
                partner_total_rows.append({
                    'label': '  Total Distributions', 'values': safe_json(deal_totals),
                    'is_pct': False, 'is_header': False, 'isBold': True,
                })

            # Assemble all rows
            rows = []
            # Revenue details + total
            if rev_details:
                rows.extend(rev_details)
            rows.append({'label': 'Total Revenues', 'values': safe_json(rev_sums),
                         'isBold': True, 'is_pct': False, 'is_header': False, 'underline': False})
            # Expense details + total
            if exp_details:
                rows.extend(exp_details)
            rows.append({'label': 'Total Expenses', 'values': safe_json(exp_sums),
                         'isBold': True, 'is_pct': False, 'is_header': False, 'underline': True})
            # NOI through other items
            rows.append({'label': 'NOI', 'values': safe_json(noi_sums)})
            rows.append({'label': 'Tax Abatement', 'values': safe_json(tax_abate_sums)})
            rows.append({'label': 'Interest', 'values': safe_json(interest_sums)})
            rows.append({'label': 'Principal', 'values': safe_json(principal_sums)})
            rows.append({'label': 'Total Debt Service', 'values': safe_json(tds_sums)})
            rows.append({'label': 'Other Below-the-Line', 'values': safe_json(other_btl_sums)})
            rows.append({'label': 'Capital Expenditures', 'values': safe_json(capex_sums),
                         'underline': True})
            rows.append({'label': 'Funds Available for Distribution', 'values': safe_json(fad_sums)})
            rows.append({'label': 'Debt Service Coverage Ratio', 'values': safe_json(dscr_sums),
                         'is_pct': True})
            # CF Waterfall
            rows.extend(cf_wf_rows)
            # Proceeds
            rows.append(proceeds_row)
            # Cap Waterfall
            rows.extend(cap_wf_rows)
            # Partner Totals
            rows.extend(partner_total_rows)

            annual_forecast = {
                'years': all_year_keys,
                'columns': columns_meta,
                'rows': rows,
            }
    except Exception as e:
        logger.exception("Annual forecast build failed: %s", e)

    # Build debt service summary by anniversary year
    debt_service = None
    loan_sched = result.get('loan_sched')
    if loan_sched is not None and not loan_sched.empty:
        try:
            import pandas as pd
            from dateutil.relativedelta import relativedelta
            close_date_str = assumptions.get('target_close') or deal_data['deal'].get('target_close')
            if close_date_str:
                closing_dt_ds = pd.Timestamp(close_date_str)
            else:
                closing_dt_ds = pd.Timestamp(result.get('model_start') or '2026-01-01')

            def _anniv_year_ds(dt):
                months_diff = (dt.year - closing_dt_ds.year) * 12 + (dt.month - closing_dt_ds.month)
                if months_diff < 0:
                    return 0
                return months_diff // 12 + 1

            ls = loan_sched.copy()
            ls['_dt'] = pd.to_datetime(ls['event_date'])
            ls['Year'] = ls['_dt'].apply(_anniv_year_ds)
            hold_yrs = int(assumptions.get('hold_years') or 7)
            ls = ls[ls['Year'].between(1, hold_yrs)]
            annual_ds = ls.groupby('Year').agg(
                interest=('interest', 'sum'),
                principal=('principal', 'sum'),
            ).reset_index()
            annual_ds['total'] = annual_ds['interest'] + annual_ds['principal']
            debt_service = safe_json(annual_ds.to_dict('records'))
        except Exception:
            logger.exception("Debt service build failed")

    # Build cash management schedule
    cash_mgmt = None
    cash_schedule = result.get('cash_schedule')
    if cash_schedule is not None and not cash_schedule.empty:
        try:
            import pandas as pd
            cs = cash_schedule.copy()
            cs_records = []
            for _, row in cs.iterrows():
                rec = {}
                for col in cs.columns:
                    v = row[col]
                    if hasattr(v, 'isoformat'):
                        rec[col] = v.isoformat()
                    else:
                        rec[col] = v
                cs_records.append(rec)
            beginning_cash = result.get('beginning_cash', 0)
            # For prospects, beginning cash = CapEx Reserve from capital uses
            cap_uses_json = assumptions.get('capital_uses_json')
            if cap_uses_json and isinstance(cap_uses_json, str):
                import json as _json
                try:
                    cap_uses = _json.loads(cap_uses_json)
                except Exception:
                    cap_uses = None
            elif isinstance(cap_uses_json, list):
                cap_uses = cap_uses_json
            else:
                cap_uses = None
            if cap_uses and isinstance(cap_uses, list):
                for item in cap_uses:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get('label', '')).lower()
                    if 'cap ex reserve' in label or 'capex reserve' in label:
                        try:
                            beginning_cash = float(item.get('amount') or 0)
                        except (ValueError, TypeError):
                            pass
                        break
            cash_mgmt = safe_json({
                'schedule': cs_records,
                'summary': result.get('cash_summary', {}),
                'beginning_cash': beginning_cash,
            })
        except Exception:
            logger.exception("Cash management build failed")

    # Build XIRR cashflows
    xirr_cashflows = None
    try:
        partner_results = result.get('partner_results', [])
        if partner_results:
            cf_data = {}
            for pr in partner_results:
                details = pr.get('cashflow_details', [])
                cfs = []
                for row in details:
                    d = row.get('Date')
                    cfs.append({
                        'date': d.isoformat() if hasattr(d, 'isoformat') else str(d),
                        'description': row.get('Description', ''),
                        'amount': row.get('Amount', 0),
                        'source': row.get('Source', ''),
                    })
                cf_data[pr['partner']] = cfs
            xirr_cashflows = safe_json(cf_data)
    except Exception:
        logger.exception("XIRR cashflows build failed")

    # Build ROE / MOIC audits
    roe_audit = None
    moic_audit = None
    try:
        from flask_app.services import compute_service
        partner_results = result.get('partner_results', [])
        deal_summary = result.get('deal_summary', {})
        sale_me = result.get('sale_me')
        if partner_results:
            roe_audit = safe_json(compute_service.build_roe_audit(
                partner_results, deal_summary, sale_me,
                wf_steps=result.get('wf_steps'),
                vcode=result.get('prospect_assumptions', {}).get('close_date', ''),
            ))
            moic_audit = safe_json(compute_service.build_moic_audit(
                partner_results, deal_summary, sale_me,
            ))
    except Exception:
        logger.exception("ROE/MOIC audit build failed")

    return jsonify(safe_json({
        'vcode': result.get('prospect_assumptions', {}).get('close_date', ''),
        'partner_results': result.get('partner_results', []),
        'deal_summary': result.get('deal_summary', {}),
        'debug_msgs': result.get('debug_msgs', []),
        'prospect_assumptions': result.get('prospect_assumptions', {}),
        'annual_forecast': annual_forecast,
        'debt_service': debt_service,
        'cash_management': cash_mgmt,
        'xirr_cashflows': xirr_cashflows,
        'roe_audit': roe_audit,
        'moic_audit': moic_audit,
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


@prospects_bp.route('/<int:deal_id>/properties/<int:property_id>/cashflows/parse',
                    methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def parse_property_cashflows(deal_id, property_id):
    """Parse an Excel/CSV file and return ALL line items with suggested COA mappings.

    Step 1 of the two-step import flow. Returns line items for analyst review
    before importing. Does NOT store anything.
    """
    from cashflow_parser import parse_cashflow_line_items

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'No file selected'}), 400

    file_bytes = f.read()
    filename = f.filename

    # Load COA table for better matching
    coa_rows = []
    try:
        from sqlalchemy import text as sa_text
        with get_engine().connect() as conn:
            rows = conn.execute(sa_text(
                "SELECT * FROM coa ORDER BY vcode"
            )).fetchall()
            if rows:
                coa_rows = [dict(r._mapping) for r in rows]
    except Exception as e:
        logger.debug("COA table load for mapping: %s", e)

    parsed = parse_cashflow_line_items(file_bytes, filename, coa_rows)
    if 'error' in parsed:
        return jsonify(parsed), 400

    return jsonify(parsed)


@prospects_bp.route('/<int:deal_id>/properties/<int:property_id>/cashflows/confirm',
                    methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def confirm_property_cashflows(deal_id, property_id):
    """Import line-item cash flows with analyst-confirmed COA mappings.

    Step 2 of the two-step import flow. Expects JSON body:
    {
        line_items: [{label, vaccount, category, values: [{period_date, amount}]}],
        frequency: 'monthly' | 'annual'
    }
    """
    body = request.get_json()
    if not body or 'line_items' not in body:
        return jsonify({'error': 'Missing line_items in request body'}), 400

    line_items = body['line_items']
    frequency = body.get('frequency', 'monthly')

    if not line_items:
        return jsonify({'error': 'No line items to import'}), 400

    try:
        result = import_property_line_items(
            get_engine(), deal_id, property_id,
            line_items, source='excel', frequency=frequency,
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.exception("Line-item import failed for property %d", property_id)
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'status': 'imported',
        'rows_imported': result['rows_imported'],
        'line_items_count': len([li for li in line_items if li.get('vaccount')]),
    })


@prospects_bp.route('/<int:deal_id>/properties/<int:property_id>/cashflows/line-items',
                    methods=['GET'])
@login_required
def get_line_items(deal_id, property_id):
    """Get stored line-item cashflows for a property."""
    version = request.args.get('version', 1, type=int)
    rows = get_property_line_items(get_engine(), deal_id, property_id, version)
    return jsonify({'line_items': rows, 'count': len(rows)})


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

# ---------------------------------------------------------------------------
# Scenarios -- named bindings of cash flow source + overrides + adjustments
# ---------------------------------------------------------------------------

@prospects_bp.route('/<int:deal_id>/analyze/excel', methods=['POST'])
@login_required
def analyze_deal_excel(deal_id):
    """Run the analysis and return the audit workbook.

    Same body as /analyze (scenario_id, overrides); the response is an xlsx
    whose tabs carry the inputs and every supporting calculation, with live
    =XIRR formulas on the partner cash flows so a third party's Excel
    recomputes the returns from the same cash flows the model used.
    """
    from flask import send_file
    from io import BytesIO
    from flask_app.services.prospect_excel import generate_prospect_analysis_excel

    # The same compute half /analyze uses, with the raw DataFrames intact,
    # so the workbook can never diverge from what the screen shows.
    prepared = _run_prospect_analysis(deal_id)
    if isinstance(prepared, tuple):
        return prepared
    result = prepared['result']
    if 'error' in result and 'partner_results' not in result:
        return jsonify({'error': result['error']}), 400
    deal_data = prepared['deal_data']
    assumptions = prepared['assumptions'] or {}
    scenario = prepared.get('scenario')
    engine = get_engine()

    wf_steps = []
    try:
        vcode = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"
        from sqlalchemy import text as sa_text
        with engine.connect() as conn:
            wf_steps = [dict(r) for r in conn.execute(sa_text(
                'SELECT vmisc, "iOrder", "PropCode", "vState", "FXRate", '
                '"nPercent", "mAmount", vtranstype FROM waterfalls '
                'WHERE vcode = :v ORDER BY vmisc, "iOrder"'), {"v": vcode}).mappings()]
    except Exception as e:
        logger.debug("wf steps for excel: %s", e)

    # The exact anniversary-year forecast the app renders, so the Annual
    # Forecast tab ties to the screen by construction.
    annual_forecast = None
    try:
        payload = _continue_analyze(result, deal_data, assumptions).get_json()
        annual_forecast = (payload or {}).get('annual_forecast')
    except Exception as e:
        logger.warning("annual forecast for excel: %s", e)

    xlsx = generate_prospect_analysis_excel(
        result, deal_data['deal'], assumptions, wf_steps, scenario,
        annual_forecast=annual_forecast)
    name = (deal_data['deal'].get('deal_name') or 'deal').replace(' ', '_')
    if scenario:
        name += '_' + str(scenario.get('name', '')).replace(' ', '_')
    return send_file(BytesIO(xlsx), as_attachment=True,
                     download_name=f"deal_analysis_{name}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@prospects_bp.route('/<int:deal_id>/scenarios', methods=['GET'])
@login_required
def list_deal_scenarios(deal_id):
    from flask_app.services import scenario_service
    return jsonify({'scenarios': scenario_service.list_scenarios(get_engine(), deal_id)})


@prospects_bp.route('/<int:deal_id>/scenarios', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def create_deal_scenario(deal_id):
    from flask_app.services import scenario_service
    body = request.get_json(silent=True) or {}
    if not (body.get('name') or '').strip():
        return jsonify({'error': 'Scenario name is required'}), 400
    username = getattr(g, 'current_user', {}).get('username', '')
    s = scenario_service.create_scenario(get_engine(), deal_id, body, username)
    return jsonify({'scenario': s}), 201


@prospects_bp.route('/<int:deal_id>/scenarios/<int:scenario_id>', methods=['PUT'])
@login_required
@role_required('admin', 'analyst')
def update_deal_scenario(deal_id, scenario_id):
    from flask_app.services import scenario_service
    engine = get_engine()
    existing = scenario_service.get_scenario(engine, scenario_id)
    if not existing or existing['prospect_id'] != deal_id:
        return jsonify({'error': f'Scenario {scenario_id} not found on deal {deal_id}'}), 404
    username = getattr(g, 'current_user', {}).get('username', '')
    s = scenario_service.update_scenario(engine, scenario_id,
                                         request.get_json(silent=True) or {}, username)
    return jsonify({'scenario': s})


@prospects_bp.route('/<int:deal_id>/scenarios/<int:scenario_id>', methods=['DELETE'])
@login_required
@role_required('admin', 'analyst')
def delete_deal_scenario(deal_id, scenario_id):
    from flask_app.services import scenario_service
    engine = get_engine()
    existing = scenario_service.get_scenario(engine, scenario_id)
    if not existing or existing['prospect_id'] != deal_id:
        return jsonify({'error': f'Scenario {scenario_id} not found on deal {deal_id}'}), 404
    scenario_service.delete_scenario(engine, scenario_id)
    return jsonify({'message': 'Scenario deleted'})


@prospects_bp.route('/<int:deal_id>/scenarios/risk-candidates', methods=['GET'])
@login_required
def scenario_risk_candidates(deal_id):
    """Tenants worth a downside scenario, from the deal's lease reviews."""
    from flask_app.services import scenario_service
    return jsonify({'candidates': scenario_service.get_risk_candidates(get_engine(), deal_id)})


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
    from flask_app.services.data_service import refresh_table, reload

    deal_data = get_deal(get_engine(), deal_id)
    if not deal_data:
        return jsonify({'error': 'Deal not found'}), 404

    vcode = deal_data['deal'].get('vcode') or f"N{deal_id:07d}"
    body = request.json or {}

    import pandas as pd
    from datetime import date as dt_date

    def _convert_step_inputs(step_inputs, wf_name):
        """Convert UI step inputs to waterfall rows.

        The engine ties a level's steps together by identical iOrder: a
        Share (or a gated IRR lead) pairs with the Tags at its order. Each
        input row may carry an explicit `level` (the Tie # shown in the
        builder); rows sharing a level pair together, exactly like editing
        iOrder in the AM Waterfall Setup grid. Rows without one fall back to
        the close-at-100 heuristic: residuals join the open level as Tags
        until its shares reach 100%, a gated IRR opens a level with its own
        share, and any other step closes it.
        """
        # ---- pass 1: assign a level to every step -------------------------
        assigned = []           # (step, level)
        auto = 10               # next auto level
        open_level = None       # heuristic split level currently open
        open_sum = 0.0

        def _explicit(s):
            lvl = s.get('level')
            if lvl in (None, ''):
                return None
            try:
                return int(float(lvl))
            except (TypeError, ValueError):
                return None

        for s in step_inputs:
            eid = (s.get('entity_id') or '').strip()
            stype = s.get('step_type', '')
            if not eid:
                continue
            lvl = _explicit(s)
            if stype == 'residual':
                pct = float(s.get('rate') or 0)
                if lvl is None:
                    if open_level is not None and open_sum < 99.5:
                        lvl = open_level
                    else:
                        lvl = auto
                        auto += 10
                        open_sum = 0.0
                    open_level = lvl
                    open_sum += pct
                else:
                    open_level, open_sum = lvl, open_sum + pct
            elif stype == 'irr_lookback' and float(s.get('share') or 0) > 0:
                if lvl is None:
                    lvl = auto
                    auto += 10
                open_level = lvl
                open_sum = float(s.get('share') or 0)
            else:
                if lvl is None:
                    lvl = auto
                    auto += 10
                open_level = None
                open_sum = 0.0
            assigned.append((s, lvl))
            # keep auto levels clear of explicit ones
            if lvl >= auto:
                auto = lvl + 10

        # ---- pass 2: emit rows; Share vs Tag decided per level ------------
        level_has_lead = {}
        for s, lvl in assigned:
            stype = s.get('step_type', '')
            if stype == 'irr_lookback' and float(s.get('share') or 0) > 0:
                level_has_lead[lvl] = True

        rows = []
        for s, lvl in assigned:
            eid = (s.get('entity_id') or '').strip()
            stype = s.get('step_type', '')
            base = {
                'vcode': vcode, 'vmisc': wf_name, 'iOrder': lvl,
                'PropCode': eid, 'mAmount': 0, 'vAmtType': '', 'vNotes': '',
                'dteffective': dt_date(2020, 1, 1), 'nmisc': 0,
            }
            if stype == 'pref':
                rows.append({**base, 'vState': 'Pref', 'FXRate': 1.0,
                             'nPercent': float(s.get('rate') or 0),
                             'vtranstype': 'Preferred Return'})
            elif stype == 'return_of_capital':
                rows.append({**base, 'vState': 'Initial', 'FXRate': 1.0,
                             'nPercent': 0, 'vtranstype': 'Return of Capital'})
            elif stype == 'residual':
                if level_has_lead.get(lvl):
                    state = 'Tag'
                else:
                    state = 'Share'
                    level_has_lead[lvl] = True
                rows.append({**base, 'vState': state,
                             'FXRate': float(s.get('rate') or 0) / 100,
                             'nPercent': 0, 'vtranstype': 'Excess Cash Flow'})
            elif stype == 'fixed_amount':
                rows.append({**base, 'vState': 'Amt', 'FXRate': 0, 'nPercent': 0,
                             'mAmount': float(s.get('amount') or 0),
                             'vtranstype': 'Fixed Amount'})
            elif stype == 'irr_lookback':
                gate_share = float(s.get('share') or 0)
                rows.append({**base, 'vState': 'IRR',
                             'FXRate': gate_share / 100 if gate_share > 0 else 0,
                             'nPercent': float(s.get('rate') or 0),
                             'vtranstype': ('IRR Threshold' if gate_share > 0
                                            else 'IRR Hurdle')})
        return rows

    cf_inputs = body.get('cf_steps', [])
    cap_inputs = body.get('cap_steps', [])

    if not cf_inputs and not cap_inputs:
        return jsonify({'error': 'At least one waterfall step required'}), 400

    # Every step must name an entity declared on the deal.  Returns are
    # attributed by this ID and it is the key the deal is onboarded to Asset
    # Management on, so an undeclared or placeholder ID would assign capital
    # to a partner that does not exist.
    # Declared IDs come from BOTH places Pipeline models a participant:
    # entity records (planned_entity_id) and the investor records nested
    # under them (planned_investor_id). Waterfall capital is usually
    # attributed to investors, so leaving them out rejected correctly
    # modelled deals.
    declared = {
        (e.get('planned_entity_id') or '').strip()
        for e in (deal_data.get('entities') or [])
        if (e.get('planned_entity_id') or '').strip()
    }
    for e in (deal_data.get('entities') or []):
        for inv in (e.get('investors') or []):
            pid = (inv.get('planned_investor_id') or '').strip()
            if pid:
                declared.add(pid)
    used = {
        (s.get('entity_id') or '').strip()
        for s in list(cf_inputs) + list(cap_inputs)
        if (s.get('entity_id') or '').strip()
    }
    undeclared = sorted(used - declared)
    if undeclared:
        if declared:
            msg = (
                "These waterfall entities are not set up on the deal: "
                + ", ".join(undeclared)
                + ". Declared entity IDs are: "
                + ", ".join(sorted(declared))
                + "."
            )
        else:
            msg = (
                "This deal has no entities with an entity ID, so the waterfall "
                "cannot be attributed to anyone. Add each investing entity in "
                "Pipeline with the ID it will carry in MRI (for example PPI35 "
                "or OPPEGA), then build the waterfall."
            )
        return jsonify({'error': msg, 'undeclared_entities': undeclared}), 400

    steps = _convert_step_inputs(cf_inputs, 'CF_WF') + _convert_step_inputs(cap_inputs, 'Cap_WF')

    # Save to database
    df = pd.DataFrame(steps)
    try:
        save_waterfall_steps(vcode, df)
        try:
            refresh_table('waterfalls')
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
