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
    merge_rent_roll_to_review,
    upload_documents_to_review,
    assign_document_to_tenant,
    approve_tenant,
    get_workflow_progress,
    update_workflow_step,
    ensure_resolution_table,
    resolve_field,
    clear_resolution,
    get_risk_analysis_data,
    RESOLVABLE_FIELDS,
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
                   created_by, created_at, workflow_step
            FROM lease_reviews WHERE id = :rid
        """), {'rid': review_id}).fetchone()

        if not review:
            return jsonify({'error': 'Review not found'}), 404

        tenants = conn.execute(text("""
            SELECT id, tenant_name, suite, square_feet, lease_type,
                   lease_start, lease_end, term_months, monthly_rent,
                   annual_rent, rent_per_sf, security_deposit,
                   is_vacant, is_material, has_cotenancy, has_exclusive_use,
                   extraction_status, approval_status, analyst_notes,
                   rent_roll_source
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
            'workflow_step': review[9] or 'setup',
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
            'approval_status': t[17] or 'pending',
            'analyst_notes': t[18],
            'rent_roll_source': t[19],
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


@lease_review_bp.route('/reviews/<int:review_id>/merge-rent-roll', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def merge_rent_roll(review_id):
    """Non-destructive rent roll import — merges into existing tenants.

    Fuzzy-matches by (suite, tenant_name). Updates matched tenants without
    touching extraction data. Adds new tenants. Flags missing tenants.
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
        report = merge_rent_roll_to_review(
            engine, review_id, rr_df,
            source_label=request.form.get('source_label', 'seller_rent_roll'),
        )
        return jsonify({'status': 'merged', **report})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Rent roll merge error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/upload-documents', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def upload_documents(review_id):
    """Upload multiple PDF documents with dedup and auto-tenant matching.

    Accepts multipart/form-data with one or more 'files' fields.
    """
    if not request.files:
        return jsonify({'error': 'No files provided'}), 400

    files_list = request.files.getlist('files')
    if not files_list:
        return jsonify({'error': 'No files provided'}), 400

    engine = get_engine()
    ensure_lease_tables(engine)

    try:
        file_tuples = []
        for f in files_list:
            if f.filename:
                file_tuples.append((f.filename, f.read()))

        if not file_tuples:
            return jsonify({'error': 'No valid files'}), 400

        report = upload_documents_to_review(
            engine, review_id, file_tuples,
            uploaded_by=g.current_user.get('username', 'unknown'),
        )
        return jsonify({'status': 'uploaded', **report})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Document upload error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/documents/<int:doc_id>/assign-tenant', methods=['POST'])
@login_required
@role_required('admin', 'analyst')
def assign_doc_tenant(review_id, doc_id):
    """Assign an unmatched document to a specific tenant.

    Body: { "tenant_id": 123 }
    """
    data = request.json
    if not data or not data.get('tenant_id'):
        return jsonify({'error': 'tenant_id required'}), 400

    engine = get_engine()
    try:
        assign_document_to_tenant(engine, review_id, doc_id, data['tenant_id'])
        return jsonify({'status': 'assigned'})
    except Exception as e:
        logger.error(f"Document assignment error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/tenants/<int:tenant_id>/approve', methods=['PUT'])
@login_required
@role_required('admin', 'analyst')
def approve_tenant_endpoint(review_id, tenant_id):
    """Set approval status for a tenant.

    Body: { "status": "approved|flagged|pending", "notes": "..." }
    """
    data = request.json or {}
    status = data.get('status', 'approved')
    notes = data.get('notes')

    engine = get_engine()
    try:
        result = approve_tenant(
            engine, review_id, tenant_id, status,
            approved_by=g.current_user.get('username', 'unknown'),
            notes=notes,
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Tenant approval error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/progress', methods=['GET'])
@login_required
def get_progress(review_id):
    """Get workflow step progress metrics."""
    engine = get_engine()
    try:
        progress = get_workflow_progress(engine, review_id)
        return jsonify(progress)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Progress error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/reviews/<int:review_id>/workflow-step', methods=['PUT'])
@login_required
@role_required('admin', 'analyst')
def set_workflow_step(review_id):
    """Update the current workflow step.

    Body: { "step": "setup|rent_roll|documents|extraction|validation|review|complete" }
    """
    data = request.json or {}
    step = data.get('step')
    if not step:
        return jsonify({'error': 'step required'}), 400

    engine = get_engine()
    try:
        update_workflow_step(engine, review_id, step)
        return jsonify({'status': 'updated', 'step': step})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Workflow step error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ── Lease Risk Analysis endpoints ──────────────────────────────────────

@lease_review_bp.route('/reviews/<int:review_id>/risk-analysis', methods=['GET'])
@login_required
def get_risk_analysis(review_id):
    """Get complete risk analysis data bundle for a review."""
    try:
        engine = get_engine()
        ensure_resolution_table(engine)
        data = get_risk_analysis_data(engine, review_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Risk analysis error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route(
    '/reviews/<int:review_id>/tenants/<int:tenant_id>/resolve',
    methods=['PUT'],
)
@login_required
@role_required('admin', 'analyst')
def resolve_tenant_field(review_id, tenant_id):
    """Resolve a field value for a tenant (analyst override)."""
    try:
        body = request.get_json(force=True)
        field_name = body.get('field_name')
        value = body.get('value')
        source = body.get('source', 'analyst')

        if not field_name:
            return jsonify({'error': 'field_name is required'}), 400
        if field_name not in RESOLVABLE_FIELDS:
            return jsonify({
                'error': f'Invalid field. Must be one of: {sorted(RESOLVABLE_FIELDS)}'
            }), 400

        engine = get_engine()
        ensure_resolution_table(engine)
        resolve_field(
            engine, tenant_id, field_name, value, source,
            resolved_by=g.user.get('username', 'unknown'),
        )
        return jsonify({'status': 'ok', 'field': field_name, 'value': value})
    except Exception as e:
        logger.error(f"Resolve field error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route(
    '/reviews/<int:review_id>/tenants/<int:tenant_id>/resolve/<field_name>',
    methods=['DELETE'],
)
@login_required
@role_required('admin', 'analyst')
def clear_tenant_resolution(review_id, tenant_id, field_name):
    """Clear a field resolution, reverting to base data."""
    try:
        engine = get_engine()
        ensure_resolution_table(engine)
        clear_resolution(engine, tenant_id, field_name)
        return jsonify({'status': 'ok', 'field': field_name, 'cleared': True})
    except Exception as e:
        logger.error(f"Clear resolution error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@lease_review_bp.route('/seed', methods=['POST'])
@login_required
@role_required('admin')
def seed_review():
    """Bulk-seed a full lease review with all related data (admin only).

    Accepts JSON with: review, tenants, cotenancy, cotenancy_refs,
    exclusive_use, options, rent_steps, validation.
    """
    from sqlalchemy import text
    engine = get_engine()
    ensure_lease_tables(engine)
    data = request.json
    if not data or 'review' not in data or 'tenants' not in data:
        return jsonify({'error': 'review and tenants required'}), 400

    rv = data['review']
    prospect_property_id = rv.get('prospect_property_id')

    def safe_float(v):
        """Convert to float or None if not numeric."""
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    try:
        with engine.begin() as conn:
            # Insert review
            row = conn.execute(text("""
                INSERT INTO lease_reviews
                    (property_name, property_address, total_gla, total_annual_rent,
                     total_tenants, rent_roll_date, prospect_property_id, status,
                     source_folder, created_by)
                VALUES (:pn, :pa, :gla, :rent, :tc, :rrd, :ppid, :status, :sf, :cb)
                RETURNING id
            """), {
                'pn': rv.get('property_name', ''),
                'pa': rv.get('property_address', ''),
                'gla': rv.get('total_gla', 0),
                'rent': rv.get('total_annual_rent', 0),
                'tc': rv.get('total_tenants', 0),
                'rrd': rv.get('rent_roll_date'),
                'ppid': prospect_property_id,
                'status': rv.get('status', 'in_progress'),
                'sf': rv.get('source_folder'),
                'cb': g.current_user.get('username', 'seed'),
            }).fetchone()
            review_id = row[0]
            logger.info(f"Seeded review id={review_id}")

            # Insert tenants, mapping local IDs to new IDs
            local_to_new = {}
            for t in data['tenants']:
                local_id = t.get('id')
                trow = conn.execute(text("""
                    INSERT INTO lease_tenants
                        (review_id, tenant_name, suite, square_feet, lease_type,
                         lease_start, lease_end, term_months, monthly_rent,
                         annual_rent, rent_per_sf, security_deposit,
                         is_vacant, is_material, has_cotenancy, has_exclusive_use,
                         extraction_status)
                    VALUES (:rid, :tn, :su, :sf, :lt, :ls, :le, :tm, :mr,
                            :ar, :rpsf, :sd, :iv, :im, :hc, :heu, :es)
                    RETURNING id
                """), {
                    'rid': review_id,
                    'tn': t.get('tenant_name', ''),
                    'su': t.get('suite', ''),
                    'sf': t.get('square_feet', 0),
                    'lt': t.get('lease_type', ''),
                    'ls': t.get('lease_start', ''),
                    'le': t.get('lease_end', ''),
                    'tm': t.get('term_months', 0),
                    'mr': t.get('monthly_rent', 0),
                    'ar': t.get('annual_rent', 0),
                    'rpsf': t.get('rent_per_sf', 0),
                    'sd': t.get('security_deposit', 0),
                    'iv': bool(t.get('is_vacant')),
                    'im': bool(t.get('is_material')),
                    'hc': bool(t.get('has_cotenancy')),
                    'heu': bool(t.get('has_exclusive_use')),
                    'es': t.get('extraction_status', 'pending'),
                }).fetchone()
                local_to_new[local_id] = trow[0]

            logger.info(f"Seeded {len(local_to_new)} tenants")

            # Cotenancy
            cot_id_map = {}
            for c in data.get('cotenancy', []):
                new_tid = local_to_new.get(c.get('tenant_id'))
                if not new_tid:
                    continue
                crow = conn.execute(text("""
                    INSERT INTO lease_cotenancy
                        (tenant_id, review_id, clause_text, trigger_description,
                         trigger_threshold, cure_period_days, alt_rent_formula,
                         termination_right, termination_notice_days, sunset_provision,
                         is_curable, waiver_mechanism, source_doc, source_page, notes)
                    VALUES (:tid, :rid, :ct, :td, :tt, :cpd, :arf, :tr, :tnd,
                            :sp, :ic, :wm, :sd, :spage, :n)
                    RETURNING id
                """), {
                    'tid': new_tid, 'rid': review_id,
                    'ct': c.get('clause_text'), 'td': c.get('trigger_description'),
                    'tt': c.get('trigger_threshold'), 'cpd': c.get('cure_period_days'),
                    'arf': c.get('alt_rent_formula'),
                    'tr': bool(c.get('termination_right')),
                    'tnd': c.get('termination_notice_days'),
                    'sp': c.get('sunset_provision'),
                    'ic': bool(c.get('is_curable', True)),
                    'wm': c.get('waiver_mechanism'),
                    'sd': c.get('source_doc'), 'spage': c.get('source_page'),
                    'n': c.get('notes'),
                }).fetchone()
                cot_id_map[c.get('id')] = crow[0]

            # Cotenancy refs
            for cr in data.get('cotenancy_refs', []):
                new_cot_id = cot_id_map.get(cr.get('cotenancy_id'))
                new_tid = local_to_new.get(cr.get('tenant_id'))
                ref_tid = local_to_new.get(cr.get('referenced_tenant_id'))
                if not new_cot_id or not new_tid:
                    continue
                conn.execute(text("""
                    INSERT INTO lease_cotenancy_refs
                        (cotenancy_id, tenant_id, referenced_tenant_name,
                         referenced_tenant_id, reference_type, notes)
                    VALUES (:cid, :tid, :rtn, :rtid, :rt, :n)
                """), {
                    'cid': new_cot_id, 'tid': new_tid,
                    'rtn': cr.get('referenced_tenant_name', ''),
                    'rtid': ref_tid, 'rt': cr.get('reference_type', 'named'),
                    'n': cr.get('notes'),
                })

            # Exclusive use
            for eu in data.get('exclusive_use', []):
                new_tid = local_to_new.get(eu.get('tenant_id'))
                if not new_tid:
                    continue
                conn.execute(text("""
                    INSERT INTO lease_exclusive_use
                        (tenant_id, restriction_text, restricted_use, radius_feet, source_doc)
                    VALUES (:tid, :rt, :ru, :rf, :sd)
                """), {
                    'tid': new_tid, 'rt': eu.get('restriction_text'),
                    'ru': eu.get('restricted_use'), 'rf': safe_float(eu.get('radius_feet')),
                    'sd': eu.get('source_doc'),
                })

            # Options
            for o in data.get('options', []):
                new_tid = local_to_new.get(o.get('tenant_id'))
                if not new_tid:
                    continue
                conn.execute(text("""
                    INSERT INTO lease_options
                        (tenant_id, option_type, option_number, total_options,
                         term_years, notice_days, notice_deadline, rent_terms,
                         auto_renewal, exercised, source_doc)
                    VALUES (:tid, :ot, :on, :to_, :ty, :nd, :ndl, :rt,
                            :ar, :ex, :sd)
                """), {
                    'tid': new_tid, 'ot': o.get('option_type', ''),
                    'on': o.get('option_number'), 'to_': o.get('total_options'),
                    'ty': safe_float(o.get('term_years')), 'nd': o.get('notice_days'),
                    'ndl': o.get('notice_deadline'), 'rt': o.get('rent_terms'),
                    'ar': bool(o.get('auto_renewal')),
                    'ex': bool(o.get('exercised')),
                    'sd': o.get('source_doc'),
                })

            # Rent steps
            for rs in data.get('rent_steps', []):
                new_tid = local_to_new.get(rs.get('tenant_id'))
                if not new_tid:
                    continue
                conn.execute(text("""
                    INSERT INTO lease_rent_steps
                        (tenant_id, effective_date, monthly_rent, annual_rent,
                         rent_per_sf, source_doc, source_page)
                    VALUES (:tid, :ed, :mr, :ar, :rpsf, :sd, :sp)
                """), {
                    'tid': new_tid, 'ed': rs.get('effective_date'),
                    'mr': safe_float(rs.get('monthly_rent')),
                    'ar': safe_float(rs.get('annual_rent')),
                    'rpsf': safe_float(rs.get('rent_per_sf')),
                    'sd': rs.get('source_doc'), 'sp': rs.get('source_page'),
                })

            # Validation
            for v in data.get('validation', []):
                new_tid = local_to_new.get(v.get('tenant_id'))
                if not new_tid:
                    continue
                conn.execute(text("""
                    INSERT INTO lease_validation
                        (tenant_id, field_name, source_type, seller_value,
                         lease_value, status, source_doc, notes)
                    VALUES (:tid, :fn, :st, :sv, :lv, :s, :sd, :n)
                """), {
                    'tid': new_tid, 'fn': v.get('field_name', ''),
                    'st': v.get('source_type', 'rent_roll'),
                    'sv': v.get('seller_value'), 'lv': v.get('lease_value'),
                    's': v.get('status', 'pending'),
                    'sd': v.get('source_doc'), 'n': v.get('notes'),
                })

            logger.info(f"Seed complete: review={review_id}, "
                        f"tenants={len(local_to_new)}, "
                        f"cotenancy={len(cot_id_map)}, "
                        f"refs={len(data.get('cotenancy_refs', []))}, "
                        f"exclusive={len(data.get('exclusive_use', []))}, "
                        f"options={len(data.get('options', []))}, "
                        f"rent_steps={len(data.get('rent_steps', []))}, "
                        f"validation={len(data.get('validation', []))}")

        return jsonify({
            'review_id': review_id,
            'tenants': len(local_to_new),
            'cotenancy': len(cot_id_map),
            'cotenancy_refs': len(data.get('cotenancy_refs', [])),
            'exclusive_use': len(data.get('exclusive_use', [])),
            'options': len(data.get('options', [])),
            'rent_steps': len(data.get('rent_steps', [])),
            'validation': len(data.get('validation', [])),
        }), 201
    except Exception as e:
        logger.error(f"Seed error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
