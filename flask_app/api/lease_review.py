"""
lease_review.py
API endpoints for lease review and due diligence.
"""

from flask import Blueprint, g, jsonify, request, send_file
from flask_app.auth.routes import login_required, role_required
from flask_app.db import get_engine
from flask_app.services.lease_review_service import (
    ensure_lease_tables,
    ingest_property,
    scan_property_folder,
    extract_all_documents,
    validate_rent_roll,
    get_expiration_histogram,
    get_cotenancy_matrix,
    get_scenario_analysis,
    generate_lease_review_excel,
    parse_rent_roll_flexible,
    import_rent_roll_to_review,
    create_review_manual,
)
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

lease_review_bp = Blueprint('lease_review', __name__, url_prefix='/api/lease-review')


@lease_review_bp.route('/prospect-properties', methods=['GET'])
@login_required
def list_prospect_properties():
    """List all prospect properties across all deals for lease review linking."""
    from sqlalchemy import text
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT pp.id, pp.property_name, pp.address, pp.city, pp.state,
                       pp.gla_sf, pd.deal_name, pd.id as deal_id,
                       (SELECT lr.id FROM lease_reviews lr
                        WHERE lr.prospect_property_id = pp.id LIMIT 1) as lease_review_id
                FROM prospect_properties pp
                JOIN prospect_deals pd ON pd.id = pp.prospect_id
                ORDER BY pd.deal_name, pp.sort_order, pp.property_name
            """)).fetchall()

        return jsonify([{
            'id': r[0], 'property_name': r[1], 'address': r[2],
            'city': r[3], 'state': r[4], 'gla_sf': r[5],
            'deal_name': r[6], 'deal_id': r[7],
            'lease_review_id': r[8],
        } for r in rows])
    except Exception:
        return jsonify([])


@lease_review_bp.route('/reviews', methods=['GET'])
@login_required
def list_reviews():
    """List all lease reviews."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, property_name, property_address, total_gla,
                   total_annual_rent, total_tenants, status,
                   created_by, created_at
            FROM lease_reviews
            ORDER BY created_at DESC
        """)).fetchall()

    return jsonify([{
        'id': r[0], 'property_name': r[1], 'property_address': r[2],
        'total_gla': r[3], 'total_annual_rent': r[4],
        'total_tenants': r[5], 'status': r[6],
        'created_by': r[7], 'created_at': str(r[8]),
    } for r in rows])


@lease_review_bp.route('/reviews', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def create_review():
    """Create a new lease review by scanning a property folder.

    Body: {
        "base_path": "...",
        "property_name": "...",
        "property_address": "...",
        "rent_roll_path": "..." (optional)
    }
    """
    data = request.json
    engine = get_engine()

    # Ensure tables exist
    ensure_lease_tables(engine)

    try:
        review_id = ingest_property(
            engine=engine,
            base_path=data['base_path'],
            property_name=data['property_name'],
            property_address=data.get('property_address', ''),
            rent_roll_path=data.get('rent_roll_path'),
            created_by=g.current_user.get('username', 'unknown'),
        )
        return jsonify({'review_id': review_id, 'status': 'created'})
    except Exception as e:
        logger.error(f"Error creating review: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/create', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def create_manual_review():
    """Create a lease review manually (no folder scanning required).

    Body: { "property_name": "...", "property_address": "...", "total_gla": 0 }
    """
    data = request.json
    if not data or not data.get('property_name'):
        return jsonify({'error': 'property_name is required'}), 400

    engine = get_engine()
    ensure_lease_tables(engine)

    try:
        review_id = create_review_manual(
            engine=engine,
            property_name=data['property_name'],
            property_address=data.get('property_address', ''),
            total_gla=float(data.get('total_gla', 0) or 0),
            created_by=g.current_user.get('username', 'unknown'),
            prospect_property_id=data.get('prospect_property_id'),
        )
        return jsonify({'review_id': review_id, 'status': 'created'}), 201
    except Exception as e:
        logger.error(f"Error creating manual review: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/upload-rent-roll', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def upload_rent_roll(review_id):
    """Upload a rent roll Excel/CSV file to populate tenants for a review.

    Accepts multipart/form-data with a 'file' field.
    Replaces any existing tenants in the review.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    engine = get_engine()
    ensure_lease_tables(engine)

    try:
        file_bytes = file.read()
        rr_df = parse_rent_roll_flexible(file_bytes, file.filename)
        count = import_rent_roll_to_review(engine, review_id, rr_df)
        return jsonify({
            'status': 'imported',
            'tenant_count': count,
            'total_gla': float(rr_df['square_feet'].sum()),
            'total_annual_rent': float(rr_df['annual_rent'].sum()),
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Rent roll upload error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>', methods=['GET'])
@login_required
def get_review(review_id):
    """Get full review data including tenants and documents."""
    from sqlalchemy import text
    engine = get_engine()

    with engine.connect() as conn:
        review = conn.execute(text("""
            SELECT id, property_name, property_address, total_gla,
                   total_annual_rent, total_tenants, status,
                   created_by, created_at
            FROM lease_reviews WHERE id = :rid
        """), {'rid': review_id}).fetchone()

        if not review:
            return jsonify({'error': 'Review not found'}), 404

        tenants = conn.execute(text("""
            SELECT id, tenant_name, suite, square_feet, lease_type,
                   lease_start, lease_end, term_months, monthly_rent,
                   annual_rent, rent_per_sf, security_deposit,
                   is_vacant, is_material, has_cotenancy, has_exclusive_use,
                   extraction_status
            FROM lease_tenants
            WHERE review_id = :rid
            ORDER BY suite
        """), {'rid': review_id}).fetchall()

        doc_counts = conn.execute(text("""
            SELECT tenant_id, COUNT(*) as cnt,
                   SUM(CASE WHEN extraction_status = 'extracted' THEN 1 ELSE 0 END) as extracted
            FROM lease_documents
            WHERE review_id = :rid
            GROUP BY tenant_id
        """), {'rid': review_id}).fetchall()

    doc_map = {r[0]: {'total': r[1], 'extracted': r[2]} for r in doc_counts}

    return jsonify({
        'review': {
            'id': review[0], 'property_name': review[1],
            'property_address': review[2], 'total_gla': review[3],
            'total_annual_rent': review[4], 'total_tenants': review[5],
            'status': review[6], 'created_by': review[7],
            'created_at': str(review[8]),
        },
        'tenants': [{
            'id': t[0], 'tenant_name': t[1], 'suite': t[2],
            'square_feet': t[3], 'lease_type': t[4],
            'lease_start': t[5], 'lease_end': t[6],
            'term_months': t[7], 'monthly_rent': t[8],
            'annual_rent': t[9], 'rent_per_sf': t[10],
            'security_deposit': t[11], 'is_vacant': bool(t[12]),
            'is_material': bool(t[13]), 'has_cotenancy': bool(t[14]),
            'has_exclusive_use': bool(t[15]),
            'extraction_status': t[16],
            'documents': doc_map.get(t[0], {'total': 0, 'extracted': 0}),
        } for t in tenants],
    })


@lease_review_bp.route('/reviews/<int:review_id>/scan', methods=['GET'])
@login_required
def scan_folder(review_id):
    """Preview what would be ingested from a folder (dry run)."""
    from sqlalchemy import text
    engine = get_engine()

    with engine.connect() as conn:
        review = conn.execute(text("""
            SELECT source_folder FROM lease_reviews WHERE id = :rid
        """), {'rid': review_id}).fetchone()

    if not review or not review[0]:
        return jsonify({'error': 'No source folder configured'}), 404

    scan = scan_property_folder(review[0], '')
    return jsonify({
        'tenant_count': len(scan['tenants']),
        'tenants': [{
            'folder_name': t['folder_name'],
            'doc_count': t['doc_count'],
        } for t in scan['tenants']],
        'abstracts': len(scan['abstracts']),
        'cotenancy_file': scan['cotenancy_file'] is not None,
        'rent_roll_files': len(scan['rent_roll_files']),
    })


@lease_review_bp.route('/reviews/<int:review_id>/extract', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def run_extraction(review_id):
    """Run Claude API extraction on all pending documents."""
    engine = get_engine()
    try:
        extract_all_documents(engine, review_id)
        return jsonify({'status': 'complete'})
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/validate', methods=['POST'])
@login_required
def run_validation(review_id):
    """Run rent roll validation against extracted lease terms."""
    engine = get_engine()
    try:
        results = validate_rent_roll(engine, review_id)
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/expirations', methods=['GET'])
@login_required
def get_expirations(review_id):
    """Get lease expiration histogram data."""
    years = request.args.get('years', 10, type=int)
    engine = get_engine()
    try:
        data = get_expiration_histogram(engine, review_id, years)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Expiration histogram error: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/cotenancy', methods=['GET'])
@login_required
def get_cotenancy(review_id):
    """Get co-tenancy cross-reference matrix."""
    engine = get_engine()
    try:
        data = get_cotenancy_matrix(engine, review_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Cotenancy matrix error: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/scenarios', methods=['GET'])
@login_required
def get_scenarios(review_id):
    """Get cascading co-tenancy scenario analysis."""
    engine = get_engine()
    try:
        data = get_scenario_analysis(engine, review_id)
        return jsonify({'scenarios': data})
    except Exception as e:
        logger.error(f"Scenario analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/excel', methods=['GET'])
@login_required
def download_excel(review_id):
    """Download the comprehensive lease review Excel workbook."""
    engine = get_engine()
    try:
        excel_bytes = generate_lease_review_excel(engine, review_id)

        from sqlalchemy import text
        with engine.connect() as conn:
            name = conn.execute(text(
                "SELECT property_name FROM lease_reviews WHERE id = :rid"
            ), {'rid': review_id}).fetchone()

        filename = f"Lease_Review_{name[0].replace(' ', '_')}.xlsx" if name else "Lease_Review.xlsx"
        return send_file(
            BytesIO(excel_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        logger.error(f"Excel generation error: {e}")
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/validation', methods=['GET'])
@login_required
def get_validation(review_id):
    """Get persisted validation results."""
    from sqlalchemy import text
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT t.tenant_name, t.suite, v.field_name, v.source_type,
                   v.seller_value, v.lease_value, v.status, v.notes
            FROM lease_validation v
            JOIN lease_tenants t ON t.id = v.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name, v.source_type, v.field_name
        """), {'rid': review_id}).fetchall()

    return jsonify([{
        'tenant': r[0], 'suite': r[1], 'field': r[2],
        'source_type': r[3], 'seller_value': r[4],
        'lease_value': r[5], 'status': r[6], 'notes': r[7],
    } for r in rows])


@lease_review_bp.route('/reviews/<int:review_id>/tenants/<int:tenant_id>/documents', methods=['GET'])
@login_required
def get_tenant_documents(review_id, tenant_id):
    """Get all documents for a specific tenant."""
    from sqlalchemy import text
    engine = get_engine()

    with engine.connect() as conn:
        docs = conn.execute(text("""
            SELECT id, filename, doc_type, doc_date, page_count,
                   extraction_status
            FROM lease_documents
            WHERE review_id = :rid AND tenant_id = :tid
            ORDER BY doc_date
        """), {'rid': review_id, 'tid': tenant_id}).fetchall()

    return jsonify([{
        'id': d[0], 'filename': d[1], 'doc_type': d[2],
        'doc_date': d[3], 'page_count': d[4],
        'extraction_status': d[5],
    } for d in docs])
