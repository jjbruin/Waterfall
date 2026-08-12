"""Data API — load, reload, import CSVs, export database, config."""

from flask import Blueprint, request, jsonify, current_app, send_file
import io
import os

from flask_app.auth.routes import login_required, role_required
from flask_app.services import data_service
from flask_app.services import compute_service
from flask_app.services.sold_service import clear_sold_cache
from flask_app.api.dashboard import clear_dashboard_cache
from flask_app.serializers import df_to_response, safe_json
from database import (
    import_csvs_to_database, import_single_csv, import_csv_dataframe,
    import_csv_stream, export_all_tables_to_zip, TABLE_DEFINITIONS,
    PROTECTED_TABLES,
)
import pandas as pd

data_bp = Blueprint("data", __name__)


@data_bp.route("/version", methods=["GET"])
def get_version():
    """Return build hash for client-side version detection (no auth required)."""
    return jsonify({"version": current_app.config.get("BUILD_HASH", "dev")})


def _get_data():
    return data_service.get_data()


@data_bp.route("/deals", methods=["GET"])
@login_required
def list_deals():
    """List all deals (investment map)."""
    data = _get_data()
    inv_disp = data_service.get_inv_display(data["inv"])
    return jsonify(df_to_response(inv_disp, "deals"))


@data_bp.route("/deals/all", methods=["GET"])
@login_required
def list_all_deals():
    """List all deals including sold."""
    data = _get_data()
    return jsonify(df_to_response(data["inv"], "deals"))


@data_bp.route("/import", methods=["POST"])
@login_required
@role_required("admin")
def import_csvs():
    """Import CSVs into database from a folder path (admin only).

    Body: { folder_path: str, table_name?: str }
    If table_name is provided, imports only that single table's CSV.
    Otherwise imports all CSVs.
    """
    body = request.get_json(silent=True) or {}
    folder = body.get("folder_path", "")
    table_name = body.get("table_name", "")
    if not folder:
        return jsonify({"error": "folder_path required"}), 400

    db_path = current_app.config["DB_PATH"]

    if table_name:
        results = import_single_csv(folder, table_name, db_path)
    else:
        results = import_csvs_to_database(folder, db_path)

    # Clear caches after import
    data_service.reload()
    compute_service.clear_cache()
    clear_sold_cache()
    clear_dashboard_cache()

    return jsonify({"results": results})


@data_bp.route("/list-csvs", methods=["POST"])
@login_required
def list_csvs():
    """List available CSVs in a folder, matched against TABLE_DEFINITIONS.

    Body: { folder_path: str }
    Returns: { csvs: [{ table_name, csv_file, description, found, protected }] }
    """
    body = request.get_json(silent=True) or {}
    folder = body.get("folder_path", "")
    if not folder:
        return jsonify({"error": "folder_path required"}), 400

    from pathlib import Path
    data_path = Path(folder)
    if not data_path.is_dir():
        return jsonify({"error": f"Folder not found: {folder}"}), 400

    csvs = []
    for table_name, table_info in TABLE_DEFINITIONS.items():
        csv_file = table_info["csv"]
        found = (data_path / csv_file).exists()
        csvs.append({
            "table_name": table_name,
            "csv_file": csv_file,
            "description": table_info.get("description", ""),
            "found": found,
            "protected": table_name in PROTECTED_TABLES,
        })

    # Sort: found first, then alphabetical
    csvs.sort(key=lambda c: (not c["found"], c["table_name"]))
    return jsonify({"csvs": csvs})


@data_bp.route("/upload-import", methods=["POST"])
@login_required
@role_required("admin")
def upload_import():
    """Import CSV files uploaded directly (admin only).

    Accepts multipart/form-data with one or more CSV files.
    File names are matched to TABLE_DEFINITIONS by csv filename.
    Returns: { results: { table_name: {status, rows, error?} } }
    """
    if not request.files:
        return jsonify({"error": "No files uploaded"}), 400

    # Build reverse lookup: csv filename → table name
    csv_to_table = {}
    for table_name, table_info in TABLE_DEFINITIONS.items():
        csv_to_table[table_info["csv"].lower()] = table_name

    results = {}
    for key, file in request.files.items(multi=True):
        filename = file.filename or key
        # Match by exact CSV filename or by table name
        table_name = csv_to_table.get(filename.lower())
        if not table_name:
            # Try matching by table name directly
            bare = filename.rsplit(".", 1)[0].lower() if "." in filename else filename.lower()
            if bare in TABLE_DEFINITIONS:
                table_name = bare

        if not table_name:
            results[filename] = {'rows': 0, 'status': 'error',
                                 'error': f'No matching table for "{filename}"'}
            continue

        try:
            result = import_csv_stream(table_name, file.stream, source=f"upload:{filename}")
            results[table_name] = result
        except Exception as e:
            results[table_name or filename] = {'rows': 0, 'status': 'error', 'error': str(e)}

    # Clear caches after import
    data_service.reload()
    compute_service.clear_cache()
    clear_sold_cache()
    clear_dashboard_cache()

    return jsonify({"results": results})


@data_bp.route("/table-definitions", methods=["GET"])
@login_required
def table_definitions():
    """Return table definitions for the upload UI (filename mapping)."""
    tables = []
    for table_name, table_info in TABLE_DEFINITIONS.items():
        tables.append({
            "table_name": table_name,
            "csv_file": table_info["csv"],
            "description": table_info.get("description", ""),
            "protected": table_name in PROTECTED_TABLES,
        })
    tables.sort(key=lambda t: t["table_name"])
    return jsonify({"tables": tables})


@data_bp.route("/export", methods=["GET"])
@login_required
def export_database():
    """Export entire database as a zip of CSVs."""
    db_path = current_app.config["DB_PATH"]
    zip_bytes = export_all_tables_to_zip(db_path)
    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name="waterfall_db_export.zip",
    )


@data_bp.route("/tables", methods=["GET"])
@login_required
def list_tables():
    """List all database tables with row counts."""
    from database import list_all_tables
    tables = list_all_tables()
    return jsonify({"tables": safe_json(tables)})


@data_bp.route("/tables/<table_name>/rows", methods=["GET"])
@login_required
def table_rows(table_name):
    """Return paginated, sortable, filterable rows from a database table.

    Query params:
        page (int): Page number, 1-based (default 1)
        page_size (int): Rows per page, max 500 (default 100)
        sort (str): Column name to sort by
        order (str): 'asc' or 'desc' (default 'asc')
        filter__<col> (str): Filter value for column <col> (case-insensitive contains)
    """
    import sqlalchemy as sa
    from flask_app.db import get_engine

    engine = get_engine()

    # Validate table exists
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            check = conn.execute(sa.text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name=:t"
            ), {"t": table_name})
        else:
            check = conn.execute(sa.text(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:t"
            ), {"t": table_name})
        if not check.fetchone():
            return jsonify({"error": f"Table '{table_name}' not found"}), 404

    page = max(1, int(request.args.get("page", 1)))
    page_size = min(500, max(1, int(request.args.get("page_size", 100))))
    sort_col = request.args.get("sort", "")
    sort_order = request.args.get("order", "asc").lower()

    # Build column filters from query params like filter__vcode=P000
    filters = {}
    for key, val in request.args.items():
        if key.startswith("filter__") and val.strip():
            col_name = key[len("filter__"):]
            filters[col_name] = val.strip()

    with engine.connect() as conn:
        # Get columns
        if engine.dialect.name == "postgresql":
            col_rows = conn.execute(sa.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=:t ORDER BY ordinal_position"
            ), {"t": table_name}).fetchall()
            columns = [r[0] for r in col_rows]
        else:
            col_rows = conn.execute(sa.text(f'PRAGMA table_info("{table_name}")')).fetchall()
            columns = [r[1] for r in col_rows]

        # Build query with filters
        where_clauses = []
        params = {}
        for i, (col, val) in enumerate(filters.items()):
            if col not in columns:
                continue
            param_name = f"fv{i}"
            if engine.dialect.name == "postgresql":
                where_clauses.append(f'CAST("{col}" AS TEXT) ILIKE :{param_name}')
            else:
                where_clauses.append(f'CAST("{col}" AS TEXT) LIKE :{param_name} COLLATE NOCASE')
            params[param_name] = f"%{val}%"

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Total count (filtered)
        count_row = conn.execute(
            sa.text(f'SELECT COUNT(*) FROM "{table_name}"{where_sql}'), params
        ).fetchone()
        total = count_row[0]

        # Sort
        order_sql = ""
        if sort_col and sort_col in columns:
            direction = "DESC" if sort_order == "desc" else "ASC"
            order_sql = f' ORDER BY "{sort_col}" {direction}'

        # Paginate
        offset = (page - 1) * page_size
        limit_sql = f" LIMIT {page_size} OFFSET {offset}"

        rows = conn.execute(
            sa.text(f'SELECT * FROM "{table_name}"{where_sql}{order_sql}{limit_sql}'),
            params,
        ).fetchall()

        data = [dict(zip(columns, row)) for row in rows]

    return jsonify({
        "table": table_name,
        "columns": columns,
        "rows": safe_json(data),
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, -(-total // page_size)),
    })


@data_bp.route("/config", methods=["GET"])
@login_required
def get_config():
    """Get current application configuration."""
    return jsonify({
        "start_year": current_app.config["DEFAULT_START_YEAR"],
        "horizon_years": current_app.config["DEFAULT_HORIZON_YEARS"],
        "pro_yr_base": current_app.config["PRO_YR_BASE_DEFAULT"],
        "actuals_through": current_app.config.get("ACTUALS_THROUGH"),
        "db_path": current_app.config["DB_PATH"],
    })


@data_bp.route("/config", methods=["PUT"])
@login_required
@role_required("admin", "analyst")
def update_config():
    """Update application configuration (admin/analyst only, runtime, not persisted).

    Body: { start_year?, horizon_years?, pro_yr_base? }
    """
    body = request.get_json(force=True)

    if "start_year" in body:
        current_app.config["DEFAULT_START_YEAR"] = int(body["start_year"])
    if "horizon_years" in body:
        current_app.config["DEFAULT_HORIZON_YEARS"] = int(body["horizon_years"])
    if "pro_yr_base" in body:
        current_app.config["PRO_YR_BASE_DEFAULT"] = int(body["pro_yr_base"])
        # Pro_yr_base affects data loading — clear data cache
        data_service.reload()
    if "actuals_through" in body:
        current_app.config["ACTUALS_THROUGH"] = body["actuals_through"]  # ISO date string or None
        # Actuals cutoff changes computation results — clear compute cache
        compute_service.clear_cache()
    clear_sold_cache()
    clear_dashboard_cache()

    return jsonify({"message": "Config updated", **{
        "start_year": current_app.config["DEFAULT_START_YEAR"],
        "horizon_years": current_app.config["DEFAULT_HORIZON_YEARS"],
        "pro_yr_base": current_app.config["PRO_YR_BASE_DEFAULT"],
        "actuals_through": current_app.config.get("ACTUALS_THROUGH"),
    }})


@data_bp.route("/reload", methods=["POST"])
@login_required
def reload_all():
    """Clear ALL caches (data, compute, PSCKOC) and reload."""
    data_service.reload()
    compute_service.clear_cache()
    clear_sold_cache()
    clear_dashboard_cache()

    # Clear PSCKOC cache too
    from flask_app.services.psckoc_service import clear_cache as clear_psckoc
    clear_psckoc()

    return jsonify({"message": "All caches cleared"})


@data_bp.route("/sources", methods=["GET"])
@login_required
def data_sources():
    """Show active data source for each table (database or API)."""
    from flask_app.services.data_adapters import get_source_info
    return jsonify({"sources": get_source_info()})


# ── MRI Query Endpoints ──────────────────────────────────────────────────

@data_bp.route("/mri/status", methods=["GET"])
@login_required
def mri_status():
    """Test MRI server connectivity (VPN required)."""
    from flask_app.services.mri_service import test_connection, MRI_SERVERS
    results = {}
    for key in MRI_SERVERS:
        results[key] = test_connection(key)
    return jsonify({"servers": results})


@data_bp.route("/mri/queries", methods=["GET"])
@login_required
def mri_list_queries():
    """List all available MRI queries."""
    from flask_app.services.mri_service import list_queries
    return jsonify({"queries": list_queries()})


@data_bp.route("/mri/queries/<query_name>/run", methods=["POST"])
@login_required
def mri_run_query(query_name):
    """Run a query and save CSV to the network downloads folder.

    Any authenticated user can run queries and download results.
    Body (optional): { server: "pmx"|"im" } — override server for unregistered queries.
    Returns: { query, server, rows, columns, elapsed_seconds, csv_path }
    """
    from flask_app.services.mri_service import run_query

    body = request.get_json(silent=True) or {}
    server_key = body.get("server")

    try:
        result = run_query(query_name, server_key=server_key, save_csv=True)
        # Remove internal DataFrame from response
        result.pop("_dataframe", None)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import logging, traceback
        logging.getLogger(__name__).error(f"MRI query '{query_name}' failed:\n{traceback.format_exc()}")
        return jsonify({"error": f"Query failed: {str(e)[:300]}"}), 500


@data_bp.route("/mri/queries/<query_name>/download", methods=["GET"])
@login_required
def mri_download_query(query_name):
    """Run a query and return results directly as CSV download (no file save)."""
    from flask_app.services.mri_service import run_query

    try:
        result = run_query(query_name, save_csv=False)
        df = result["_dataframe"]
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{query_name}.csv",
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)[:300]}"}), 500


@data_bp.route("/mri/refresh", methods=["POST"])
@login_required
@role_required("admin")
def mri_refresh_all():
    """Refresh all app data from MRI (admin only).

    Runs all importable queries against MRI databases and imports results
    directly into the app's PostgreSQL database. Replaces CSV upload workflow.
    Clears all caches after import.
    """
    from flask_app.services.mri_service import refresh_all

    try:
        results = refresh_all()

        # Clear all caches after import
        data_service.reload()
        compute_service.clear_cache()
        clear_sold_cache()
        clear_dashboard_cache()
        from flask_app.services.psckoc_service import clear_cache as clear_psckoc
        clear_psckoc()

        return jsonify(results)
    except Exception as e:
        import logging, traceback
        logging.getLogger(__name__).error(f"MRI refresh_all failed:\n{traceback.format_exc()}")
        return jsonify({"error": f"Refresh failed: {str(e)[:300]}"}), 500


@data_bp.route("/mri/refresh/<query_name>", methods=["POST"])
@login_required
@role_required("admin")
def mri_refresh_single(query_name):
    """Refresh a single table from MRI (admin only).

    Runs one query and imports directly into the app database.
    """
    from flask_app.services.mri_service import import_query_to_database

    try:
        result = import_query_to_database(query_name)

        # Clear caches
        data_service.reload()
        compute_service.clear_cache()
        clear_sold_cache()
        clear_dashboard_cache()

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import logging, traceback
        logging.getLogger(__name__).error(f"MRI import '{query_name}' failed:\n{traceback.format_exc()}")
        return jsonify({"error": f"Import failed: {str(e)[:300]}"}), 500
