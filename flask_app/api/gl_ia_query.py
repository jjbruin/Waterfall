"""GL / IA Query endpoints — the CFO's Spreadsheet Server filters, in the app.

    GET  /api/gl-ia-query/gl/options    what can be picked (from the data itself)
    POST /api/gl-ia-query/gl            run it
    POST /api/gl-ia-query/gl/excel      the same result as a workbook
    GET  /api/gl-ia-query/ia/options
    POST /api/gl-ia-query/ia
    POST /api/gl-ia-query/ia/excel

READS ARE OPEN TO ANY SIGNED-IN USER, matching the rest of the accounting section
(see CLAUDE.md, "Who may edit the Accounting section": reads are open, writes are
gated). Nothing here writes. Worth Jim's eye all the same: this is a bulk export of
entity-level GL, which is a wider read than looking at one entity's statement, and
narrowing it to `ACCOUNTING_ROLES` is one decorator if he wants that.
"""
import logging

from flask import Blueprint, Response, jsonify, request

from flask_app.auth.routes import login_required
from flask_app.db import get_engine
from flask_app.serializers import safe_json
from flask_app.services import gl_ia_query_service as qs

logger = logging.getLogger(__name__)

gl_ia_query_bp = Blueprint('gl_ia_query', __name__, url_prefix='/api/gl-ia-query')


def _body() -> dict:
    return request.get_json(silent=True) or {}


@gl_ia_query_bp.route('/gl/options', methods=['GET'])
@login_required
def gl_options():
    try:
        return jsonify(safe_json(qs.gl_filter_options(get_engine())))
    except Exception as e:
        logger.error(f"gl_options failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@gl_ia_query_bp.route('/ia/options', methods=['GET'])
@login_required
def ia_options():
    try:
        return jsonify(safe_json(qs.ia_filter_options(get_engine())))
    except Exception as e:
        logger.error(f"ia_options failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def _run_gl(body):
    return qs.run_gl_query(
        get_engine(),
        entities=body.get('entities'),
        period_from=body.get('period_from'),
        period_to=body.get('period_to'),
        accounts=body.get('accounts'),
        bases=body.get('bases'),
        limit=int(body.get('limit') or qs.MAX_ROWS),
    )


def _run_ia(body):
    return qs.run_ia_query(
        get_engine(),
        investments=body.get('investments'),
        investors=body.get('investors'),
        date_from=body.get('date_from'),
        date_to=body.get('date_to'),
        date_field=body.get('date_field') or 'TransactionDate',
        major_types=body.get('major_types'),
        sub_types=body.get('sub_types'),
        limit=int(body.get('limit') or qs.MAX_ROWS),
    )


@gl_ia_query_bp.route('/gl', methods=['POST'])
@login_required
def gl_query():
    try:
        return jsonify(safe_json(_run_gl(_body())))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"gl_query failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@gl_ia_query_bp.route('/ia', methods=['POST'])
@login_required
def ia_query():
    try:
        return jsonify(safe_json(_run_ia(_body())))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"ia_query failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def _criteria(body, pairs) -> list:
    """The filters, as sentences, so the workbook records what produced it."""
    out = []
    for label, key in pairs:
        val = body.get(key)
        if isinstance(val, list):
            val = ', '.join(str(v) for v in val) if val else None
        if val in (None, ''):
            continue
        out.append(f'{label}: {val}')
    return out or ['No filters — every row.']


@gl_ia_query_bp.route('/gl/excel', methods=['POST'])
@login_required
def gl_excel():
    body = _body()
    try:
        # The export is NOT capped at the screen's row limit -- the cap exists so a
        # browser grid stays usable, and the reason to export is to get everything.
        body = dict(body, limit=qs.EXPORT_MAX_ROWS)
        result = _run_gl(body)
        crit = _criteria(body, [('Entities', 'entities'),
                                ('Period from', 'period_from'),
                                ('Period to', 'period_to'),
                                ('Accounts', 'accounts'),
                                ('Basis', 'bases')])
        crit.append(f"Rows: {result.get('row_count', 0):,}")
        if result.get('data_as_of'):
            crit.append(f"MRI refresh completed: {result['data_as_of']}")
        data = qs.to_excel(result, 'GL Detail', crit)
        return Response(
            data,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': 'attachment; filename=GL_Detail.xlsx'})
    except Exception as e:
        logger.error(f"gl_excel failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@gl_ia_query_bp.route('/ia/excel', methods=['POST'])
@login_required
def ia_excel():
    body = _body()
    try:
        body = dict(body, limit=qs.EXPORT_MAX_ROWS)
        result = _run_ia(body)
        crit = _criteria(body, [('Investments', 'investments'),
                                ('Investors', 'investors'),
                                ('Date field', 'date_field'),
                                ('Date from', 'date_from'),
                                ('Date to', 'date_to'),
                                ('Major types', 'major_types'),
                                ('Sub types', 'sub_types')])
        crit.append(f"Rows: {result.get('row_count', 0):,}")
        if result.get('data_as_of'):
            crit.append(f"MRI refresh completed: {result['data_as_of']}")
        data = qs.to_excel(result, 'IA Detail', crit)
        return Response(
            data,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': 'attachment; filename=IA_Detail.xlsx'})
    except Exception as e:
        logger.error(f"ia_excel failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
