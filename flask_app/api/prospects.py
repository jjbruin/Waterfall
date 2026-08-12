"""
prospects.py
API endpoints for the New Business deal pipeline.
"""

from flask import Blueprint, jsonify, request
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
    username = request.user.get('username', 'unknown')
    deal_id = create_deal(get_engine(), data, username)
    return jsonify({'id': deal_id, 'status': 'created'}), 201


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
    username = request.user.get('username', 'unknown')
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
    username = request.user.get('username', 'unknown')
    prop_id = create_property(get_engine(), deal_id, data, username)
    return jsonify({'id': prop_id, 'status': 'created'}), 201


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
    username = request.user.get('username', 'unknown')
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
    username = request.user.get('username', 'unknown')
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
