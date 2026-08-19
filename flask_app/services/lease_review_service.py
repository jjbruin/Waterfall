"""
lease_review_service.py
Lease review service for due diligence — extraction, validation, and analysis.

Supports multi-property portfolio reviews. Data stored in PostgreSQL/SQLite
via the standard database layer.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATERIAL_LEASE_SF_THRESHOLD = 10_000  # SF threshold for "material" lease
MATERIAL_LEASE_RENT_THRESHOLD = 100_000  # Annual rent threshold

# Document type classification by filename pattern
DOC_TYPE_PATTERNS = [
    (r'(?i)lease(?!.*(?:abstract|amend|memo))', 'Original Lease'),
    (r'(?i)(first|second|third|fourth|fifth|\d+)\s*(amendment|amend)', 'Amendment'),
    (r'(?i)amend', 'Amendment'),
    (r'(?i)commencement', 'Commencement Letter'),
    (r'(?i)option\s*letter', 'Option Letter'),
    (r'(?i)snda', 'SNDA'),
    (r'(?i)memo\s*of\s*lease', 'Memorandum of Lease'),
    (r'(?i)waiver', 'Waiver Letter'),
    (r'(?i)estoppel', 'Estoppel'),
    (r'(?i)consent', 'Consent Letter'),
    (r'(?i)move.?in', 'Move-In Notice'),
    (r'(?i)delivery', 'Delivery Notice'),
    (r'(?i)rent\s*commencement', 'Rent Commencement'),
    (r'(?i)change.*(?:notice|contact|office|address)', 'Notice Change'),
    (r'(?i)coi|certificate.*insurance', 'COI'),
    (r'(?i)open.*business', 'Opening Notice'),
    (r'(?i)short\s*form', 'Short Form Lease'),
    (r'(?i)co.?tenancy.*(?:cure|notice)', 'Co-Tenancy Notice'),
    (r'(?i)cam\s*dispute', 'CAM Dispute'),
    (r'(?i)architect|certif', 'Certification'),
    (r'(?i)cctv|install', 'Installation Notice'),
]


def classify_document(filename: str) -> str:
    """Classify a lease document by its filename."""
    for pattern, doc_type in DOC_TYPE_PATTERNS:
        if re.search(pattern, filename):
            return doc_type
    return 'Other'


def parse_doc_date(filename: str) -> Optional[str]:
    """Extract date from filename like '2024.03.28_Bealls-Lease.pdf'."""
    m = re.match(r'(\d{4})[.\-](\d{2})[.\-](\d{2})', filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# ---------------------------------------------------------------------------
# Database DDL — called from ensure_pg_tables / create_additional_tables
# ---------------------------------------------------------------------------

LEASE_DDL_PG = [
    """
    CREATE TABLE IF NOT EXISTS lease_reviews (
        id              SERIAL PRIMARY KEY,
        property_name   TEXT NOT NULL,
        property_address TEXT,
        total_gla       DOUBLE PRECISION,
        total_annual_rent DOUBLE PRECISION,
        total_tenants   INTEGER,
        rent_roll_date  TEXT,
        prospect_property_id INTEGER,
        status          TEXT DEFAULT 'in_progress',
        source_folder   TEXT,
        created_by      TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_tenants (
        id              SERIAL PRIMARY KEY,
        review_id       INTEGER NOT NULL REFERENCES lease_reviews(id),
        tenant_name     TEXT NOT NULL,
        suite           TEXT,
        square_feet     DOUBLE PRECISION,
        lease_type      TEXT,
        lease_start     TEXT,
        lease_end       TEXT,
        term_months     INTEGER,
        monthly_rent    DOUBLE PRECISION,
        monthly_rent_per_sf DOUBLE PRECISION,
        annual_rent     DOUBLE PRECISION,
        annual_rent_per_sf DOUBLE PRECISION,
        annual_recoveries_per_sf DOUBLE PRECISION,
        annual_misc_per_sf DOUBLE PRECISION,
        security_deposit DOUBLE PRECISION,
        is_vacant       BOOLEAN DEFAULT FALSE,
        is_material     BOOLEAN DEFAULT FALSE,
        has_cotenancy   BOOLEAN DEFAULT FALSE,
        has_exclusive_use BOOLEAN DEFAULT FALSE,
        extraction_status TEXT DEFAULT 'pending',
        extraction_json TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_documents (
        id              SERIAL PRIMARY KEY,
        tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
        review_id       INTEGER NOT NULL REFERENCES lease_reviews(id),
        filename        TEXT NOT NULL,
        file_path       TEXT,
        doc_type        TEXT,
        doc_date        TEXT,
        page_count      INTEGER,
        extracted_text   TEXT,
        extraction_status TEXT DEFAULT 'pending',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_rent_steps (
        id              SERIAL PRIMARY KEY,
        tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
        effective_date  TEXT,
        monthly_rent    DOUBLE PRECISION,
        annual_rent     DOUBLE PRECISION,
        rent_per_sf     DOUBLE PRECISION,
        source_doc      TEXT,
        source_page     TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_cotenancy (
        id              SERIAL PRIMARY KEY,
        tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
        review_id       INTEGER NOT NULL REFERENCES lease_reviews(id),
        clause_text     TEXT,
        trigger_description TEXT,
        trigger_threshold TEXT,
        cure_period_days INTEGER,
        alt_rent_formula TEXT,
        termination_right BOOLEAN DEFAULT FALSE,
        termination_notice_days INTEGER,
        sunset_provision TEXT,
        is_curable      BOOLEAN DEFAULT TRUE,
        waiver_mechanism TEXT,
        source_doc      TEXT,
        source_page     TEXT,
        notes           TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_cotenancy_refs (
        id              SERIAL PRIMARY KEY,
        cotenancy_id    INTEGER NOT NULL REFERENCES lease_cotenancy(id),
        tenant_id       INTEGER NOT NULL,
        referenced_tenant_name TEXT NOT NULL,
        referenced_tenant_id INTEGER,
        reference_type  TEXT DEFAULT 'named',
        notes           TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_exclusive_use (
        id              SERIAL PRIMARY KEY,
        tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
        restriction_text TEXT,
        restricted_use  TEXT,
        radius_feet     DOUBLE PRECISION,
        source_doc      TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_options (
        id              SERIAL PRIMARY KEY,
        tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
        option_type     TEXT NOT NULL,
        option_number   INTEGER,
        total_options   INTEGER,
        term_years      DOUBLE PRECISION,
        notice_days     INTEGER,
        notice_deadline TEXT,
        rent_terms      TEXT,
        auto_renewal    BOOLEAN DEFAULT FALSE,
        exercised       BOOLEAN DEFAULT FALSE,
        option_start    TEXT,
        option_end      TEXT,
        source_doc      TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_validation (
        id              SERIAL PRIMARY KEY,
        tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
        field_name      TEXT NOT NULL,
        source_type     TEXT DEFAULT 'rent_roll',
        seller_value    TEXT,
        lease_value     TEXT,
        status          TEXT DEFAULT 'pending',
        source_doc      TEXT,
        notes           TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_abstract_sections (
        id              SERIAL PRIMARY KEY,
        tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
        section_key     TEXT NOT NULL,
        section_title   TEXT NOT NULL,
        content         TEXT,
        lease_ref       TEXT,
        sort_order      INTEGER DEFAULT 0,
        updated_by      TEXT,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(tenant_id, section_key)
    )
    """,
]

# SQLite variants (no SERIAL, use INTEGER PRIMARY KEY AUTOINCREMENT)
LEASE_DDL_SQLITE = [
    ddl.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
       .replace('DOUBLE PRECISION', 'REAL')
       .replace('BOOLEAN', 'INTEGER')
       .replace('REFERENCES lease_reviews(id)', '')
       .replace('REFERENCES lease_tenants(id)', '')
       .replace('REFERENCES lease_cotenancy(id)', '')
    for ddl in LEASE_DDL_PG
]


def _migrate_add_column(engine, table: str, column: str, col_type: str):
    """Add a column to a table if it doesn't exist. Works on both PG and SQLite."""
    from sqlalchemy import text
    with engine.begin() as conn:
        if engine.dialect.name == 'postgresql':
            row = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = :tbl AND column_name = :col
            """), {'tbl': table, 'col': column}).fetchone()
            if row is None:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
        else:
            cols = [r[1] for r in conn.execute(
                text(f"PRAGMA table_info({table})")).fetchall()]
            if column not in cols:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))


def ensure_lease_tables(engine):
    """Create lease review tables and migrate missing columns."""
    from sqlalchemy import text, inspect
    with engine.connect() as conn:
        # Detect dialect for correct DDL
        dialect = engine.dialect.name
        if dialect == 'sqlite':
            for ddl in LEASE_DDL_SQLITE:
                conn.execute(text(ddl))
        else:
            for ddl in LEASE_DDL_PG:
                conn.execute(text(ddl))

        conn.commit()

    # Migrate: add prospect_property_id and rent_roll_date if missing
    with engine.begin() as conn:
        if engine.dialect.name == 'postgresql':
            for col in ['prospect_property_id', 'rent_roll_date']:
                row = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'lease_reviews' AND column_name = :col
                """), {'col': col}).fetchone()
                if row is None:
                    ctype = 'INTEGER' if col == 'prospect_property_id' else 'TEXT'
                    conn.execute(text(
                        f"ALTER TABLE lease_reviews ADD COLUMN {col} {ctype}"))
        else:
            cols = [r[1] for r in conn.execute(text("PRAGMA table_info(lease_reviews)")).fetchall()]
            if 'prospect_property_id' not in cols:
                conn.execute(text(
                    "ALTER TABLE lease_reviews ADD COLUMN prospect_property_id INTEGER"))
            if 'rent_roll_date' not in cols:
                conn.execute(text(
                    "ALTER TABLE lease_reviews ADD COLUMN rent_roll_date TEXT"))

    # Phase 1D: workflow_step + step_data on lease_reviews
    _migrate_add_column(engine, 'lease_reviews', 'workflow_step', "TEXT DEFAULT 'setup'")
    _migrate_add_column(engine, 'lease_reviews', 'step_data', 'TEXT')

    # Phase 1A: rent_roll_source on lease_tenants
    _migrate_add_column(engine, 'lease_tenants', 'rent_roll_source', 'TEXT')

    # Phase 1E: per-tenant approval columns on lease_tenants
    _migrate_add_column(engine, 'lease_tenants', 'approval_status', "TEXT DEFAULT 'pending'")
    _migrate_add_column(engine, 'lease_tenants', 'approved_by', 'TEXT')
    _migrate_add_column(engine, 'lease_tenants', 'approved_at', 'TIMESTAMP')
    _migrate_add_column(engine, 'lease_tenants', 'analyst_notes', 'TEXT')

    # Rent roll extended columns on lease_tenants
    _migrate_add_column(engine, 'lease_tenants', 'monthly_rent_per_sf', 'DOUBLE PRECISION')
    _migrate_add_column(engine, 'lease_tenants', 'annual_rent_per_sf', 'DOUBLE PRECISION')
    _migrate_add_column(engine, 'lease_tenants', 'annual_recoveries_per_sf', 'DOUBLE PRECISION')
    _migrate_add_column(engine, 'lease_tenants', 'annual_misc_per_sf', 'DOUBLE PRECISION')

    # Phase 2A: option_start / option_end on lease_options
    _migrate_add_column(engine, 'lease_options', 'option_start', 'TEXT')
    _migrate_add_column(engine, 'lease_options', 'option_end', 'TEXT')

    # Phase 1B: file_hash + uploaded_by on lease_documents; file_data for PDF storage
    _migrate_add_column(engine, 'lease_documents', 'file_hash', 'TEXT')
    _migrate_add_column(engine, 'lease_documents', 'uploaded_by', 'TEXT')
    _migrate_add_column(engine, 'lease_documents', 'file_data',
                        'BYTEA' if engine.dialect.name == 'postgresql' else 'BLOB')

    logger.info("Lease review tables ensured")


def create_lease_tables_sqlite(conn):
    """Create lease review tables on SQLite."""
    cursor = conn.cursor()
    for ddl in LEASE_DDL_SQLITE:
        cursor.execute(ddl)
    conn.commit()
    logger.info("Lease review tables created on SQLite")


# ---------------------------------------------------------------------------
# Folder scanning — discover properties, tenants, and lease PDFs
# ---------------------------------------------------------------------------

def scan_property_folder(
    base_path: str,
    property_name: str,
    property_address: str = '',
) -> Dict[str, Any]:
    """Scan a property's DD folder to discover tenants and lease documents.

    Expected folder structure:
        base_path/
            Tenant Leases/
                <Tenant Name>/
                    YYYY.MM.DD_Tenant-DocType.pdf
                    ...
                ! Tenant Lease Abstracts/
                    ...
            Rent Roll/
                *.xlsx
    """
    base = Path(base_path)
    leases_dir = base / 'Tenant Leases'
    abstracts_dir = leases_dir / '! Tenant Lease Abstracts'

    result = {
        'property_name': property_name,
        'property_address': property_address,
        'tenants': [],
        'abstracts': [],
        'cotenancy_file': None,
        'rent_roll_files': [],
    }

    # Find rent roll files
    rent_roll_dir = base / 'Rent Roll'
    if rent_roll_dir.exists():
        result['rent_roll_files'] = [
            str(f) for f in rent_roll_dir.glob('*.xlsx')
            if not f.name.startswith('~$')
        ]

    # Find cotenancy file
    if abstracts_dir.exists():
        for f in abstracts_dir.glob('*Co-Tenancy*'):
            result['cotenancy_file'] = str(f)
            break

    # Scan tenant folders
    if leases_dir.exists():
        for tenant_dir in sorted(leases_dir.iterdir()):
            if not tenant_dir.is_dir():
                continue
            if tenant_dir.name.startswith('!') or tenant_dir.name == 'REA Docs':
                continue

            pdfs = []
            for pdf in tenant_dir.rglob('*.pdf'):
                # Skip COI and Prior Owner Docs subfolders
                rel = pdf.relative_to(tenant_dir)
                parts = rel.parts
                if any(p.upper() in ('COI', 'PRIOR OWNER DOCS') for p in parts[:-1]):
                    continue
                pdfs.append({
                    'filename': pdf.name,
                    'path': str(pdf),
                    'doc_type': classify_document(pdf.name),
                    'doc_date': parse_doc_date(pdf.name),
                })

            result['tenants'].append({
                'folder_name': tenant_dir.name,
                'documents': sorted(pdfs, key=lambda d: d.get('doc_date') or ''),
                'doc_count': len(pdfs),
            })

    # Scan abstracts
    if abstracts_dir.exists():
        for f in abstracts_dir.glob('*.xlsx'):
            if not f.name.startswith('~$') and 'Co-Tenancy' not in f.name:
                result['abstracts'].append(str(f))

    return result


# ---------------------------------------------------------------------------
# Rent roll parsing
# ---------------------------------------------------------------------------

def parse_rent_roll(file_path: str) -> pd.DataFrame:
    """Parse a Windsor-style rent roll Excel file into a DataFrame.

    Returns columns: suite, tenant_name, lease_type, square_feet,
    lease_start, lease_end, term_months, monthly_rent, rent_per_sf_month,
    annual_rent, rent_per_sf_year, annual_recoveries_per_sf,
    annual_misc_per_sf, security_deposit, is_vacant.

    Derives missing gross/per-SF values from what's provided.
    """
    import openpyxl
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb[wb.sheetnames[0]]

    rows = []
    for r in range(1, ws.max_row + 1):
        col1 = ws.cell(r, 1).value
        col2 = ws.cell(r, 2).value
        col3 = ws.cell(r, 3).value

        # Skip header/summary rows — data rows start with property code
        if col1 is None or str(col1).startswith('Total') or str(col1).startswith('Future'):
            continue
        if str(col1).strip() in ('Rent Roll', '', 'Occupied', 'Vacant', 'Total'):
            continue
        if col2 is None:
            continue
        # Skip header rows
        if str(col2).strip() in ('Unit(s)', ''):
            continue
        # Check for property code pattern (e.g. f4wind)
        if not re.match(r'^[a-zA-Z]\d', str(col1).strip()):
            continue

        tenant_name = str(col3) if col3 else ''
        is_vacant = 'VACANT' in tenant_name.upper()

        def to_float(v):
            try:
                return float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        def to_date_str(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v.strftime('%Y-%m-%d')
            if isinstance(v, date):
                return v.isoformat()
            s = str(v).strip()
            if s in ('None', 'TBD', ''):
                return None
            return s

        sf = to_float(ws.cell(r, 5).value)
        monthly_rent = to_float(ws.cell(r, 9).value)
        monthly_rent_per_sf = to_float(ws.cell(r, 10).value)
        annual_rent = to_float(ws.cell(r, 11).value)
        annual_rent_per_sf = to_float(ws.cell(r, 12).value)
        annual_rec_per_sf = to_float(ws.cell(r, 13).value)
        annual_misc_per_sf = to_float(ws.cell(r, 14).value)
        security_deposit = to_float(ws.cell(r, 15).value)

        # Derive missing gross/per-SF values from what's provided
        if sf and sf > 0:
            if annual_rent and not annual_rent_per_sf:
                annual_rent_per_sf = annual_rent / sf
            elif annual_rent_per_sf and not annual_rent:
                annual_rent = annual_rent_per_sf * sf
            if monthly_rent and not monthly_rent_per_sf:
                monthly_rent_per_sf = monthly_rent / sf
            elif monthly_rent_per_sf and not monthly_rent:
                monthly_rent = monthly_rent_per_sf * sf
        # Cross-derive monthly/annual when one is missing
        if annual_rent and not monthly_rent:
            monthly_rent = annual_rent / 12
        elif monthly_rent and not annual_rent:
            annual_rent = monthly_rent * 12
        if sf and sf > 0:
            if annual_rent and not annual_rent_per_sf:
                annual_rent_per_sf = annual_rent / sf
            if monthly_rent and not monthly_rent_per_sf:
                monthly_rent_per_sf = monthly_rent / sf

        rows.append({
            'property_code': str(col1).strip(),
            'suite': str(col2).strip() if col2 else '',
            'tenant_name': tenant_name.strip(),
            'lease_type': str(ws.cell(r, 4).value or '').strip(),
            'square_feet': sf,
            'lease_start': to_date_str(ws.cell(r, 6).value),
            'lease_end': to_date_str(ws.cell(r, 7).value),
            'term_months': int(to_float(ws.cell(r, 8).value)),
            'monthly_rent': monthly_rent,
            'rent_per_sf_month': monthly_rent_per_sf,
            'annual_rent': annual_rent,
            'rent_per_sf_year': annual_rent_per_sf,
            'annual_recoveries_per_sf': annual_rec_per_sf,
            'annual_misc_per_sf': annual_misc_per_sf,
            'security_deposit': security_deposit,
            'is_vacant': is_vacant,
        })

    return pd.DataFrame(rows)


def _parse_pdf_rent_roll(file_obj) -> pd.DataFrame:
    """Extract rent roll table from a PDF using pdfplumber.

    Scans all pages, extracts tables, and returns the best candidate —
    the largest table whose headers contain a tenant/name keyword.
    Multi-page tables with matching columns are concatenated.
    """
    import io
    import pdfplumber

    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)

    best_df = None
    best_size = 0

    with pdfplumber.open(file_obj) as pdf:
        # Collect all tables across all pages
        all_tables = []
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables:
                if tbl and len(tbl) >= 2:
                    all_tables.append(tbl)

    if not all_tables:
        raise ValueError("No tables found in PDF. The rent roll may need "
                         "to be converted to Excel or CSV first.")

    def _has_tenant_header(headers):
        lower_h = [str(h).lower() for h in headers if h]
        return any(kw in h for h in lower_h
                   for kw in ['tenant', 'name', 'lessee', 'occupant'])

    # Try to find and concatenate tables with matching headers
    groups = {}  # header_key -> list of row lists
    for tbl in all_tables:
        header = tuple(str(c).strip() if c else '' for c in tbl[0])
        key = header
        if key not in groups:
            groups[key] = {'header': list(header), 'rows': []}
        groups[key]['rows'].extend(tbl[1:])

    for key, grp in groups.items():
        headers = grp['header']
        if not _has_tenant_header(headers):
            continue
        rows = grp['rows']
        size = len(rows)
        if size > best_size:
            best_size = size
            best_df = pd.DataFrame(rows, columns=headers)

    # Fallback: use the largest table regardless of header match
    if best_df is None:
        for tbl in sorted(all_tables, key=len, reverse=True):
            headers = [str(c).strip() if c else f'col_{i}'
                       for i, c in enumerate(tbl[0])]
            best_df = pd.DataFrame(tbl[1:], columns=headers)
            break

    if best_df is None or best_df.empty:
        raise ValueError("Could not extract rent roll data from PDF tables.")

    # Clean up: strip whitespace, drop fully empty rows
    for col in best_df.columns:
        if best_df[col].dtype == object:
            best_df[col] = best_df[col].apply(
                lambda x: str(x).strip() if x and str(x).strip() not in ('None', '') else None)
    best_df = best_df.dropna(how='all').reset_index(drop=True)

    logger.info(f"PDF rent roll: extracted {len(best_df)} rows, "
                f"{len(best_df.columns)} columns")
    return best_df


def parse_rent_roll_flexible(file_obj, filename: str = '') -> pd.DataFrame:
    """Parse an uploaded rent roll from Excel, CSV, or PDF using flexible column matching.

    Handles Argus-format rent rolls, generic Excel exports, CSVs, and PDF tables.
    Returns a DataFrame with standardized columns matching parse_rent_roll output.

    Column matching is fuzzy — looks for keywords in header row:
      tenant/name, suite/unit, sf/area/sqft, start, end/expir,
      base rent/annual rent, monthly rent, lease type/status, deposit
    """
    import io

    # Read into DataFrame
    lower_fn = (filename or '').lower()
    if lower_fn.endswith('.pdf'):
        df_raw = _parse_pdf_rent_roll(file_obj)
    elif lower_fn.endswith('.csv'):
        if isinstance(file_obj, (str, bytes)):
            df_raw = pd.read_csv(io.BytesIO(file_obj) if isinstance(file_obj, bytes)
                                 else io.StringIO(file_obj))
        else:
            df_raw = pd.read_csv(file_obj)
    else:
        # Excel — try each sheet until we find one with tenant data
        import openpyxl
        if isinstance(file_obj, bytes):
            file_obj = io.BytesIO(file_obj)
        try:
            wb = openpyxl.load_workbook(file_obj, data_only=True, read_only=True)
        except Exception:
            # Fallback: try pandas directly
            file_obj.seek(0)
            df_raw = pd.read_excel(file_obj, engine='openpyxl')
            wb = None

        if wb is not None:
            df_raw = None
            # Prioritize sheets with rent roll / tenant in the name
            sheet_order = sorted(wb.sheetnames,
                key=lambda s: (0 if any(kw in s.lower()
                    for kw in ['rent roll', 'tenant', 'rent_roll']) else 1))
            for sheet_name in sheet_order:
                ws = wb[sheet_name]
                data = list(ws.values)
                if not data:
                    continue
                # Find header row — look for individual cells containing column names
                # Argus exports use two-row headers; we want the row with names
                header_idx = None
                for i, row in enumerate(data[:30]):
                    if row is None:
                        continue
                    # Check individual cells (not concatenated row string)
                    # to avoid matching "Tenant Rent Roll" as both tenant+rent
                    cell_vals = [str(c).lower().strip()
                                 for c in row if c is not None]
                    # Best: a cell that is exactly or contains "tenant name"
                    if any('tenant name' in cv for cv in cell_vals):
                        header_idx = i
                        break
                    # Require "tenant" in one cell AND a data keyword in
                    # a DIFFERENT cell
                    has_tenant = any('tenant' in cv and len(cv) < 30
                                     for cv in cell_vals)
                    if has_tenant:
                        other_cells = [cv for cv in cell_vals
                                       if 'tenant' not in cv]
                        if any(kw in cv for cv in other_cells
                               for kw in ['suite', 'unit', 'rent',
                                          'sf', 'sqft', 'area']):
                            header_idx = i
                            break
                if header_idx is not None:
                    # Check if prior row is a sub-header (Argus two-row header)
                    # Argus format: row N-1 has category prefixes (Potential,
                    # Scheduled, etc.), row N has the column names (Base Rent,
                    # Start Date, etc.). Concatenate them to get full names
                    # like "Potential Base Rent", "Scheduled Base Rent".
                    raw_headers = list(data[header_idx])
                    if header_idx > 0:
                        prior = data[header_idx - 1]
                        if prior:
                            prior_str = ' '.join(
                                str(c).lower() for c in prior if c)
                            if any(kw in prior_str for kw in [
                                    'lease', 'rent', 'potential', 'expense',
                                    'scheduled', 'absorption']):
                                for j, c in enumerate(prior):
                                    if c:
                                        p = str(c).strip()
                                        if raw_headers[j]:
                                            # Concatenate: "Potential" + "Base Rent"
                                            raw_headers[j] = (
                                                p + ' ' + str(raw_headers[j]).strip())
                                        else:
                                            raw_headers[j] = p
                    headers = [str(c).strip() if c else f'col_{j}'
                               for j, c in enumerate(raw_headers)]
                    rows = data[header_idx + 1:]
                    df_raw = pd.DataFrame(rows, columns=headers)
                    # If this sheet has enough rows, use it
                    if len(df_raw.dropna(how='all')) >= 2:
                        break
                    df_raw = None
            wb.close()
            if df_raw is None:
                raise ValueError("No rent roll data found in any sheet")

    if df_raw is None or df_raw.empty:
        raise ValueError("No data found in uploaded file")

    # Drop fully empty rows
    df_raw = df_raw.dropna(how='all').reset_index(drop=True)

    # Fuzzy column matching
    col_map = {}
    cols_lower = {c: c.lower().strip() for c in df_raw.columns}

    def _find_col(*keywords, exclude=None):
        for col, cl in cols_lower.items():
            if col in col_map.values():
                continue
            if exclude and any(e in cl for e in exclude):
                continue
            if any(kw in cl for kw in keywords):
                return col
        return None

    col_map['tenant_name'] = _find_col('tenant', 'name', 'lessee', exclude=['group'])
    col_map['suite'] = _find_col('suite', 'unit', 'space')
    col_map['square_feet'] = _find_col('area', 'sqft', 'sq ft', 'square', 'sf', 'gla')
    col_map['lease_type'] = _find_col('lease type', 'type', 'lease status', 'status')
    col_map['lease_start'] = _find_col('start date', 'lease start', 'commence', 'begin')
    if not col_map['lease_start']:
        col_map['lease_start'] = _find_col('start', exclude=['date'])
    col_map['lease_end'] = _find_col('end date', 'lease end', 'expir', 'termin',
                                     'maturity')
    col_map['annual_rent'] = _find_col('scheduled base', 'potential base',
                                       'annual rent', 'annual', 'base rent',
                                       exclude=['monthly', 'per sf', 'turnover',
                                                'free', 'miscellaneous',
                                                'percentage', 'absorption'])
    col_map['monthly_rent'] = _find_col('monthly', 'month rent',
                                        exclude=['annual', 'per sf'])
    col_map['annual_rent_per_sf'] = _find_col('annual rent per', 'annual base per',
                                               'annual $/sf', 'rent per sf',
                                               'rent/sf', 'per area',
                                               exclude=['monthly', 'recover',
                                                        'misc', 'expense'])
    col_map['monthly_rent_per_sf'] = _find_col('monthly rent per', 'monthly $/sf',
                                                'monthly base per',
                                                exclude=['annual'])
    col_map['annual_recoveries_per_sf'] = _find_col('recover', 'cam',
                                                     'reimburse',
                                                     exclude=['misc', 'annual rent',
                                                              'base rent'])
    col_map['annual_misc_per_sf'] = _find_col('misc', 'other per',
                                               exclude=['recover', 'annual rent',
                                                        'base rent'])
    col_map['security_deposit'] = _find_col('deposit', 'security')

    if not col_map.get('tenant_name'):
        raise ValueError(
            f"Cannot find tenant name column. Headers: {list(df_raw.columns)}")

    def _safe_float(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0
        try:
            return float(str(val).replace(',', '').replace('$', '').strip())
        except (ValueError, TypeError):
            return 0.0

    def _safe_date(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, (datetime, date)):
            return val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val)
        s = str(val).strip()
        if not s or s.lower() in ('none', 'nan', 'nat', 'tbd', ''):
            return None
        try:
            return pd.to_datetime(s).strftime('%Y-%m-%d')
        except Exception:
            return s

    result_rows = []
    for _, row in df_raw.iterrows():
        tname = row.get(col_map['tenant_name']) if col_map.get('tenant_name') else None
        if tname is None or (isinstance(tname, float) and pd.isna(tname)):
            continue
        tname = str(tname).strip()
        if not tname or tname.lower() in ('total', 'totals', 'subtotal', 'grand total',
                                           'future', '', 'nan'):
            continue

        is_vacant = 'VACANT' in tname.upper()
        sf = _safe_float(row.get(col_map.get('square_feet', ''), 0))
        ann_rent = _safe_float(row.get(col_map.get('annual_rent', ''), 0))
        mon_rent = _safe_float(row.get(col_map.get('monthly_rent', ''), 0))
        ann_rent_psf = _safe_float(row.get(col_map.get('annual_rent_per_sf', ''), 0))
        mon_rent_psf = _safe_float(row.get(col_map.get('monthly_rent_per_sf', ''), 0))
        ann_rec_psf = _safe_float(row.get(col_map.get('annual_recoveries_per_sf', ''), 0))
        ann_misc_psf = _safe_float(row.get(col_map.get('annual_misc_per_sf', ''), 0))

        # Derive missing gross/per-SF values
        if sf > 0:
            if ann_rent and not ann_rent_psf:
                ann_rent_psf = ann_rent / sf
            elif ann_rent_psf and not ann_rent:
                ann_rent = ann_rent_psf * sf
            if mon_rent and not mon_rent_psf:
                mon_rent_psf = mon_rent / sf
            elif mon_rent_psf and not mon_rent:
                mon_rent = mon_rent_psf * sf
        if ann_rent > 0 and mon_rent == 0:
            mon_rent = ann_rent / 12
        elif mon_rent > 0 and ann_rent == 0:
            ann_rent = mon_rent * 12
        if sf > 0:
            if ann_rent and not ann_rent_psf:
                ann_rent_psf = ann_rent / sf
            if mon_rent and not mon_rent_psf:
                mon_rent_psf = mon_rent / sf

        result_rows.append({
            'tenant_name': tname,
            'suite': str(row.get(col_map.get('suite', ''), '')).strip()
                    if col_map.get('suite') else '',
            'lease_type': str(row.get(col_map.get('lease_type', ''), '')).strip()
                         if col_map.get('lease_type') else 'Retail',
            'square_feet': sf,
            'lease_start': _safe_date(row.get(col_map.get('lease_start', '')))
                          if col_map.get('lease_start') else None,
            'lease_end': _safe_date(row.get(col_map.get('lease_end', '')))
                        if col_map.get('lease_end') else None,
            'term_months': 0,
            'monthly_rent': mon_rent,
            'rent_per_sf_month': mon_rent_psf,
            'annual_rent': ann_rent,
            'rent_per_sf_year': ann_rent_psf,
            'annual_recoveries_per_sf': ann_rec_psf,
            'annual_misc_per_sf': ann_misc_psf,
            'security_deposit': _safe_float(
                row.get(col_map.get('security_deposit', ''), 0))
                if col_map.get('security_deposit') else 0,
            'is_vacant': is_vacant,
        })

    if not result_rows:
        raise ValueError("No tenant rows found in uploaded file")

    return pd.DataFrame(result_rows)


def import_rent_roll_to_review(engine, review_id: int, rr_df: pd.DataFrame) -> int:
    """Import parsed rent roll data into an existing lease review.

    Clears existing tenants for the review and replaces with new data.
    Returns count of tenants imported.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Verify review exists
        rev = conn.execute(text(
            "SELECT id FROM lease_reviews WHERE id = :rid"
        ), {'rid': review_id}).fetchone()
        if not rev:
            raise ValueError(f"Review {review_id} not found")

        # Clear existing tenants and related data
        conn.execute(text(
            "DELETE FROM lease_validation WHERE tenant_id IN "
            "(SELECT id FROM lease_tenants WHERE review_id = :rid)"), {'rid': review_id})
        conn.execute(text(
            "DELETE FROM lease_cotenancy_refs WHERE cotenancy_id IN "
            "(SELECT id FROM lease_cotenancy WHERE review_id = :rid)"), {'rid': review_id})
        conn.execute(text(
            "DELETE FROM lease_cotenancy WHERE review_id = :rid"), {'rid': review_id})
        conn.execute(text(
            "DELETE FROM lease_documents WHERE review_id = :rid"), {'rid': review_id})
        conn.execute(text(
            "DELETE FROM lease_rent_steps WHERE tenant_id IN "
            "(SELECT id FROM lease_tenants WHERE review_id = :rid)"), {'rid': review_id})
        conn.execute(text(
            "DELETE FROM lease_options WHERE tenant_id IN "
            "(SELECT id FROM lease_tenants WHERE review_id = :rid)"), {'rid': review_id})
        conn.execute(text(
            "DELETE FROM lease_exclusive_use WHERE tenant_id IN "
            "(SELECT id FROM lease_tenants WHERE review_id = :rid)"), {'rid': review_id})
        conn.execute(text(
            "DELETE FROM lease_tenants WHERE review_id = :rid"), {'rid': review_id})

        # Insert tenants
        count = 0
        for _, row in rr_df.iterrows():
            tname = row.get('tenant_name', '')
            sf = float(row.get('square_feet', 0) or 0)
            ann_rent = float(row.get('annual_rent', 0) or 0)
            is_material = (sf >= MATERIAL_LEASE_SF_THRESHOLD or
                           ann_rent >= MATERIAL_LEASE_RENT_THRESHOLD)

            conn.execute(text("""
                INSERT INTO lease_tenants
                    (review_id, tenant_name, suite, square_feet, lease_type,
                     lease_start, lease_end, term_months, monthly_rent,
                     monthly_rent_per_sf, annual_rent, annual_rent_per_sf,
                     rent_per_sf, annual_recoveries_per_sf, annual_misc_per_sf,
                     security_deposit,
                     is_vacant, is_material, has_cotenancy, has_exclusive_use)
                VALUES (:rid, :tn, :su, :sf, :lt,
                        :ls, :le, :tm, :mr,
                        :mrpsf, :ar, :arpsf,
                        :rpsf, :arecpsf, :amiscpsf,
                        :sd,
                        :iv, :im, FALSE, FALSE)
            """), {
                'rid': review_id,
                'tn': tname,
                'su': row.get('suite', ''),
                'sf': sf,
                'lt': row.get('lease_type', 'Retail'),
                'ls': row.get('lease_start'),
                'le': row.get('lease_end'),
                'tm': int(row.get('term_months', 0) or 0),
                'mr': float(row.get('monthly_rent', 0) or 0),
                'mrpsf': float(row.get('rent_per_sf_month', 0) or 0),
                'ar': ann_rent,
                'arpsf': float(row.get('rent_per_sf_year', 0) or 0),
                'rpsf': float(row.get('rent_per_sf_year', 0) or 0),
                'arecpsf': float(row.get('annual_recoveries_per_sf', 0) or 0),
                'amiscpsf': float(row.get('annual_misc_per_sf', 0) or 0),
                'sd': float(row.get('security_deposit', 0) or 0),
                'iv': bool(row.get('is_vacant', False)),
                'im': is_material,
            })
            count += 1

        # Update review totals
        total_gla = float(rr_df['square_feet'].sum()) if 'square_feet' in rr_df else 0
        total_rent = float(rr_df['annual_rent'].sum()) if 'annual_rent' in rr_df else 0
        conn.execute(text("""
            UPDATE lease_reviews
            SET total_gla = :gla, total_annual_rent = :rent,
                total_tenants = :cnt, updated_at = CURRENT_TIMESTAMP
            WHERE id = :rid
        """), {'gla': total_gla, 'rent': total_rent, 'cnt': count, 'rid': review_id})

        conn.commit()

    logger.info(f"Imported {count} tenants into review {review_id}")
    return count


# ---------------------------------------------------------------------------
# Phase 1A: Non-destructive rent roll merge
# ---------------------------------------------------------------------------

def _fuzzy_match_tenant(suite_a: str, name_a: str, suite_b: str, name_b: str) -> bool:
    """Check if two tenant identifiers refer to the same tenant.

    Match by (suite exact) OR (name fuzzy containment when both non-empty).
    """
    sa = str(suite_a or '').strip().lower()
    sb = str(suite_b or '').strip().lower()
    na = str(name_a or '').strip().lower()
    nb = str(name_b or '').strip().lower()

    # Exact suite match (when suites are non-empty)
    if sa and sb and sa == sb:
        return True

    # Name containment match
    if na and nb and (na in nb or nb in na):
        return True

    return False


def merge_rent_roll_to_review(
    engine,
    review_id: int,
    rr_df: pd.DataFrame,
    source_label: str = 'seller_rent_roll',
) -> Dict[str, Any]:
    """Merge rent roll data into an existing review without destroying extraction data.

    Fuzzy-matches uploaded rows to existing tenants by (suite, tenant_name).
    - Matched tenants: Update tenant-level fields only; extraction data preserved.
    - New tenants: Insert new lease_tenants rows.
    - Missing tenants: Flag existing tenants not found in the upload (not deleted).

    Returns merge report: {matched, added, not_in_upload, details}.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        rev = conn.execute(text(
            "SELECT id FROM lease_reviews WHERE id = :rid"
        ), {'rid': review_id}).fetchone()
        if not rev:
            raise ValueError(f"Review {review_id} not found")

        # Load existing tenants
        existing = conn.execute(text("""
            SELECT id, tenant_name, suite FROM lease_tenants
            WHERE review_id = :rid
        """), {'rid': review_id}).fetchall()

        existing_list = [{'id': r[0], 'name': r[1], 'suite': r[2]} for r in existing]

        matched_ids = set()
        matched_count = 0
        added_count = 0
        details = []

        for _, row in rr_df.iterrows():
            rr_name = str(row.get('tenant_name', '')).strip()
            rr_suite = str(row.get('suite', '')).strip()
            if not rr_name:
                continue

            # Try to match to an existing tenant
            match = None
            for ex in existing_list:
                if ex['id'] in matched_ids:
                    continue
                if _fuzzy_match_tenant(rr_suite, rr_name, ex['suite'], ex['name']):
                    match = ex
                    break

            sf = float(row.get('square_feet', 0) or 0)
            ann_rent = float(row.get('annual_rent', 0) or 0)
            mon_rent = float(row.get('monthly_rent', 0) or 0)
            rpsf = float(row.get('rent_per_sf_year', 0) or 0)
            mrpsf = float(row.get('rent_per_sf_month', 0) or 0)
            arecpsf = float(row.get('annual_recoveries_per_sf', 0) or 0)
            amiscpsf = float(row.get('annual_misc_per_sf', 0) or 0)
            is_material = (sf >= MATERIAL_LEASE_SF_THRESHOLD or
                           ann_rent >= MATERIAL_LEASE_RENT_THRESHOLD)

            if match:
                # Update tenant-level fields; DO NOT touch extraction data
                matched_ids.add(match['id'])
                conn.execute(text("""
                    UPDATE lease_tenants
                    SET tenant_name = :tn, suite = :su, square_feet = :sf,
                        lease_type = :lt, lease_start = :ls, lease_end = :le,
                        monthly_rent = :mr, monthly_rent_per_sf = :mrpsf,
                        annual_rent = :ar, annual_rent_per_sf = :arpsf,
                        rent_per_sf = :rpsf,
                        annual_recoveries_per_sf = :arecpsf,
                        annual_misc_per_sf = :amiscpsf,
                        security_deposit = :sd, is_vacant = :iv, is_material = :im,
                        rent_roll_source = :src, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :tid
                """), {
                    'tn': rr_name, 'su': rr_suite, 'sf': sf,
                    'lt': row.get('lease_type', 'Retail'),
                    'ls': row.get('lease_start'), 'le': row.get('lease_end'),
                    'mr': mon_rent, 'mrpsf': mrpsf,
                    'ar': ann_rent, 'arpsf': rpsf, 'rpsf': rpsf,
                    'arecpsf': arecpsf, 'amiscpsf': amiscpsf,
                    'sd': float(row.get('security_deposit', 0) or 0),
                    'iv': bool(row.get('is_vacant', False)),
                    'im': is_material, 'src': source_label,
                    'tid': match['id'],
                })
                matched_count += 1
                details.append({'action': 'updated', 'tenant': rr_name, 'suite': rr_suite})
            else:
                # Insert new tenant
                conn.execute(text("""
                    INSERT INTO lease_tenants
                        (review_id, tenant_name, suite, square_feet, lease_type,
                         lease_start, lease_end, term_months, monthly_rent,
                         monthly_rent_per_sf, annual_rent, annual_rent_per_sf,
                         rent_per_sf, annual_recoveries_per_sf, annual_misc_per_sf,
                         security_deposit,
                         is_vacant, is_material, has_cotenancy, has_exclusive_use,
                         rent_roll_source)
                    VALUES (:rid, :tn, :su, :sf, :lt,
                            :ls, :le, :tm, :mr,
                            :mrpsf, :ar, :arpsf,
                            :rpsf, :arecpsf, :amiscpsf,
                            :sd,
                            :iv, :im, FALSE, FALSE, :src)
                """), {
                    'rid': review_id, 'tn': rr_name, 'su': rr_suite, 'sf': sf,
                    'lt': row.get('lease_type', 'Retail'),
                    'ls': row.get('lease_start'), 'le': row.get('lease_end'),
                    'tm': int(row.get('term_months', 0) or 0),
                    'mr': mon_rent, 'mrpsf': mrpsf,
                    'ar': ann_rent, 'arpsf': rpsf, 'rpsf': rpsf,
                    'arecpsf': arecpsf, 'amiscpsf': amiscpsf,
                    'sd': float(row.get('security_deposit', 0) or 0),
                    'iv': bool(row.get('is_vacant', False)),
                    'im': is_material, 'src': source_label,
                })
                added_count += 1
                details.append({'action': 'added', 'tenant': rr_name, 'suite': rr_suite})

        # Flag tenants not in the upload
        not_in_upload = []
        for ex in existing_list:
            if ex['id'] not in matched_ids:
                not_in_upload.append({'tenant': ex['name'], 'suite': ex['suite']})

        # Update review totals from current state
        totals = conn.execute(text("""
            SELECT COALESCE(SUM(square_feet), 0),
                   COALESCE(SUM(annual_rent), 0),
                   COUNT(*)
            FROM lease_tenants WHERE review_id = :rid
        """), {'rid': review_id}).fetchone()

        conn.execute(text("""
            UPDATE lease_reviews
            SET total_gla = :gla, total_annual_rent = :rent,
                total_tenants = :cnt, updated_at = CURRENT_TIMESTAMP
            WHERE id = :rid
        """), {'gla': totals[0], 'rent': totals[1], 'cnt': totals[2], 'rid': review_id})

        conn.commit()

    report = {
        'matched': matched_count,
        'added': added_count,
        'not_in_upload': len(not_in_upload),
        'not_in_upload_tenants': not_in_upload,
        'details': details,
    }
    logger.info(f"Merge rent roll review {review_id}: {report['matched']} matched, "
                f"{report['added']} added, {report['not_in_upload']} not in upload")
    return report


# ---------------------------------------------------------------------------
# Phase 1B: Incremental document upload
# ---------------------------------------------------------------------------

def upload_documents_to_review(
    engine,
    review_id: int,
    files: List[Tuple[str, bytes]],
    uploaded_by: str = 'system',
    folder_hints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Upload multiple PDF documents to a review with dedup and auto-matching.

    Args:
        files: List of (filename, file_bytes) tuples.
        folder_hints: Optional list of subfolder names (one per file, same order)
                      used as additional signal for tenant matching.

    Returns dict with counts: {added, skipped_duplicate, unmatched, details}.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        rev = conn.execute(text(
            "SELECT id FROM lease_reviews WHERE id = :rid"
        ), {'rid': review_id}).fetchone()
        if not rev:
            raise ValueError(f"Review {review_id} not found")

        # Load existing document hashes for dedup
        existing_hashes = set()
        rows = conn.execute(text("""
            SELECT file_hash FROM lease_documents
            WHERE review_id = :rid AND file_hash IS NOT NULL
        """), {'rid': review_id}).fetchall()
        for r in rows:
            existing_hashes.add(r[0])

        # Load tenants for auto-matching
        tenants = conn.execute(text("""
            SELECT id, tenant_name, suite FROM lease_tenants
            WHERE review_id = :rid AND is_vacant = false
        """), {'rid': review_id}).fetchall()

        added = 0
        skipped = 0
        unmatched = 0
        details = []

        for i, (filename, file_bytes) in enumerate(files):
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            # Dedup check
            if file_hash in existing_hashes:
                skipped += 1
                details.append({'filename': filename, 'action': 'skipped_duplicate'})
                continue

            # Classify document type
            doc_type = classify_document(filename)
            doc_date = parse_doc_date(filename)

            # Fuzzy-match to tenant by filename + optional folder hint
            folder_hint = (folder_hints[i]
                           if folder_hints and i < len(folder_hints)
                           else None)
            tenant_id = _match_file_to_tenant(filename, tenants,
                                              folder_hint=folder_hint)

            if tenant_id is None:
                unmatched += 1
                # Still insert with tenant_id pointing to a placeholder
                # Use the first tenant as a fallback (unmatched docs need manual assignment)
                # Actually, store with tenant_id=0 temporarily — but FK constraint prevents this.
                # Instead, we'll need to create an "Unassigned" approach.
                # For now, skip unmatched docs and report them.
                details.append({
                    'filename': filename, 'action': 'unmatched',
                    'doc_type': doc_type,
                })
                continue

            conn.execute(text("""
                INSERT INTO lease_documents
                    (tenant_id, review_id, filename, doc_type, doc_date,
                     extraction_status, file_hash, uploaded_by, file_data)
                VALUES (:tid, :rid, :fn, :dt, :dd,
                        'pending', :fh, :ub, :fd)
            """), {
                'tid': tenant_id, 'rid': review_id,
                'fn': filename, 'dt': doc_type, 'dd': doc_date,
                'fh': file_hash, 'ub': uploaded_by, 'fd': file_bytes,
            })
            existing_hashes.add(file_hash)
            added += 1
            details.append({
                'filename': filename, 'action': 'added',
                'doc_type': doc_type, 'tenant_id': tenant_id,
            })

        conn.commit()

    report = {
        'added': added,
        'skipped_duplicate': skipped,
        'unmatched': unmatched,
        'details': details,
    }
    logger.info(f"Uploaded docs to review {review_id}: {added} added, "
                f"{skipped} deduped, {unmatched} unmatched")
    return report


def _match_file_to_tenant(
    filename: str,
    tenants: list,
    folder_hint: Optional[str] = None,
) -> Optional[int]:
    """Match a PDF filename to a tenant by name containment.

    If *folder_hint* is provided (the subfolder name the file was stored in),
    it is used as an additional matching signal.  A folder-hint match is tried
    first — if the subfolder name matches a tenant, that wins.

    Returns tenant_id or None.
    """
    fname_lower = filename.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    # Remove common prefixes/suffixes like date stamps
    fname_clean = re.sub(r'^\d{4}[.\-]\d{2}[.\-]\d{2}[_\s]*', '', fname_lower).strip()

    # --- Try folder hint first (subfolder name = tenant name) ---
    if folder_hint and folder_hint.strip():
        hint = folder_hint.strip().lower().replace('_', ' ').replace('-', ' ')
        hint_match = None
        hint_len = 0
        for t in tenants:
            tname = str(t[1]).strip().lower()
            if not tname:
                continue
            if tname in hint or hint in tname:
                if len(tname) > hint_len:
                    hint_match = t[0]
                    hint_len = len(tname)
        if hint_match is not None:
            return hint_match

    # --- Fall back to filename matching ---
    best_match = None
    best_len = 0

    for t in tenants:
        tname = str(t[1]).strip().lower()
        if not tname:
            continue
        # Check if tenant name appears in filename or vice versa
        if tname in fname_clean or fname_clean in tname:
            if len(tname) > best_len:
                best_match = t[0]  # tenant_id
                best_len = len(tname)
        # Try first word match for multi-word names
        first_word = tname.split()[0] if tname.split() else ''
        if first_word and len(first_word) > 3 and first_word in fname_clean:
            if len(tname) > best_len:
                best_match = t[0]
                best_len = len(tname)

    return best_match


def assign_document_to_tenant(
    engine,
    review_id: int,
    doc_id: int,
    tenant_id: int,
) -> None:
    """Assign an unmatched document to a specific tenant."""
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE lease_documents
            SET tenant_id = :tid
            WHERE id = :did AND review_id = :rid
        """), {'tid': tenant_id, 'did': doc_id, 'rid': review_id})


# ---------------------------------------------------------------------------
# Phase 1E: Per-tenant approval
# ---------------------------------------------------------------------------

def approve_tenant(
    engine,
    review_id: int,
    tenant_id: int,
    status: str,
    approved_by: str,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Set approval status for a tenant.

    Args:
        status: 'approved', 'flagged', or 'pending'
    """
    from sqlalchemy import text

    if status not in ('approved', 'flagged', 'pending'):
        raise ValueError(f"Invalid approval status: {status}")

    with engine.connect() as conn:
        # Verify tenant belongs to this review
        t = conn.execute(text("""
            SELECT id FROM lease_tenants
            WHERE id = :tid AND review_id = :rid
        """), {'tid': tenant_id, 'rid': review_id}).fetchone()
        if not t:
            raise ValueError(f"Tenant {tenant_id} not found in review {review_id}")

        update_params = {
            'status': status,
            'by': approved_by if status == 'approved' else None,
            'at': datetime.utcnow().isoformat() if status == 'approved' else None,
            'tid': tenant_id,
        }
        if notes is not None:
            conn.execute(text("""
                UPDATE lease_tenants
                SET approval_status = :status, approved_by = :by,
                    approved_at = :at, analyst_notes = :notes,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :tid
            """), {**update_params, 'notes': notes})
        else:
            conn.execute(text("""
                UPDATE lease_tenants
                SET approval_status = :status, approved_by = :by,
                    approved_at = :at, updated_at = CURRENT_TIMESTAMP
                WHERE id = :tid
            """), update_params)

        conn.commit()

    return {'tenant_id': tenant_id, 'status': status}


# ---------------------------------------------------------------------------
# Phase 1D: Workflow progress
# ---------------------------------------------------------------------------

def get_workflow_progress(engine, review_id: int) -> Dict[str, Any]:
    """Get workflow step progress metrics for a review."""
    from sqlalchemy import text

    with engine.connect() as conn:
        rev = conn.execute(text("""
            SELECT workflow_step, step_data, total_tenants
            FROM lease_reviews WHERE id = :rid
        """), {'rid': review_id}).fetchone()
        if not rev:
            raise ValueError(f"Review {review_id} not found")

        workflow_step = rev[0] or 'setup'
        step_data = json.loads(rev[1]) if rev[1] else {}
        total_tenants = rev[2] or 0

        # Count metrics
        tenant_counts = conn.execute(text("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_vacant = false THEN 1 ELSE 0 END) as occupied,
                SUM(CASE WHEN approval_status = 'approved' THEN 1 ELSE 0 END) as approved,
                SUM(CASE WHEN approval_status = 'flagged' THEN 1 ELSE 0 END) as flagged
            FROM lease_tenants WHERE review_id = :rid
        """), {'rid': review_id}).fetchone()

        doc_counts = conn.execute(text("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN extraction_status = 'extracted' THEN 1 ELSE 0 END) as extracted,
                SUM(CASE WHEN extraction_status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM lease_documents WHERE review_id = :rid
        """), {'rid': review_id}).fetchone()

        val_count = conn.execute(text("""
            SELECT COUNT(*) FROM lease_validation
            WHERE tenant_id IN (SELECT id FROM lease_tenants WHERE review_id = :rid)
        """), {'rid': review_id}).fetchone()

    return {
        'current_step': workflow_step,
        'step_data': step_data,
        'tenants_imported': tenant_counts[0] or 0,
        'tenants_occupied': tenant_counts[1] or 0,
        'tenants_approved': tenant_counts[2] or 0,
        'tenants_flagged': tenant_counts[3] or 0,
        'docs_uploaded': doc_counts[0] or 0,
        'docs_extracted': doc_counts[1] or 0,
        'docs_pending': doc_counts[2] or 0,
        'validations_run': val_count[0] or 0,
    }


def update_workflow_step(engine, review_id: int, step: str) -> None:
    """Update the current workflow step for a review."""
    from sqlalchemy import text

    valid_steps = ['setup', 'rent_roll', 'documents', 'extraction',
                   'validation', 'review', 'complete']
    if step not in valid_steps:
        raise ValueError(f"Invalid step: {step}. Must be one of {valid_steps}")

    with engine.begin() as conn:
        # Load current step_data and add timestamp
        rev = conn.execute(text(
            "SELECT step_data FROM lease_reviews WHERE id = :rid"
        ), {'rid': review_id}).fetchone()
        step_data = json.loads(rev[0]) if rev and rev[0] else {}
        step_data[step] = datetime.utcnow().isoformat()

        conn.execute(text("""
            UPDATE lease_reviews
            SET workflow_step = :step, step_data = :sd,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :rid
        """), {'step': step, 'sd': json.dumps(step_data), 'rid': review_id})


def create_review_manual(engine, property_name: str, property_address: str = '',
                         total_gla: float = 0, created_by: str = 'system',
                         prospect_property_id: int = None) -> int:
    """Create a lease review without folder scanning.

    Returns the review_id.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        params = {
            'pn': property_name, 'pa': property_address,
            'gla': total_gla, 'ppid': prospect_property_id,
            'cb': created_by,
        }
        result = conn.execute(text("""
            INSERT INTO lease_reviews
                (property_name, property_address, total_gla,
                 prospect_property_id, status, created_by)
            VALUES (:pn, :pa, :gla, :ppid, 'in_progress', :cb)
            RETURNING id
        """), params)
        review_id = result.fetchone()[0]
        conn.commit()

    logger.info(f"Created manual review '{property_name}': review_id={review_id}")
    return review_id


# ---------------------------------------------------------------------------
# Cotenancy spreadsheet parsing
# ---------------------------------------------------------------------------

def parse_cotenancy_spreadsheet(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse the Co-Tenancy and Exclusives spreadsheet.

    Returns (cotenancy_records, exclusive_use_records).
    """
    import openpyxl
    wb = openpyxl.load_workbook(file_path, data_only=True)

    cotenancy = []
    if 'Co-Tenancy' in wb.sheetnames:
        ws = wb['Co-Tenancy']
        for r in range(4, ws.max_row + 1):
            tenant = ws.cell(r, 1).value
            if tenant is None:
                continue
            tenant = str(tenant).strip()
            clause = str(ws.cell(r, 4).value or '')
            has_clause = clause and 'No Lease provision' not in clause and clause != 'None'
            cotenancy.append({
                'tenant_name': tenant,
                'suite': str(ws.cell(r, 2).value or '').strip(),
                'square_feet': float(ws.cell(r, 3).value or 0),
                'clause_text': clause if has_clause else None,
                'has_cotenancy': has_clause,
            })

    exclusive = []
    if 'Exclusive Use' in wb.sheetnames:
        ws = wb['Exclusive Use']
        for r in range(4, ws.max_row + 1):
            tenant = ws.cell(r, 1).value
            if tenant is None:
                continue
            tenant = str(tenant).strip()
            restriction = str(ws.cell(r, 4).value or '')
            has_restriction = (restriction and
                               'No lease provision' not in restriction.lower() and
                               restriction != 'None')
            if has_restriction:
                exclusive.append({
                    'tenant_name': tenant,
                    'suite': str(ws.cell(r, 2).value or '').strip(),
                    'square_feet': float(ws.cell(r, 3).value or 0),
                    'restriction_text': restriction,
                })

    return cotenancy, exclusive


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(file_path: str) -> Tuple[str, int]:
    """Extract text from a PDF using PyMuPDF. Returns (text, page_count)."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    text_parts = []
    doc = pymupdf.open(file_path)
    page_count = len(doc)
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return '\n'.join(text_parts), page_count


# ---------------------------------------------------------------------------
# Claude API extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a commercial real estate lease analyst. Extract the following structured information from this lease document.

TENANT: {tenant_name}
SUITE: {suite}
DOCUMENT TYPE: {doc_type}

Return a JSON object with these fields (use null for fields not found):

{{
  "tenant_legal_name": "...",
  "suite": "...",
  "square_feet": number,
  "permitted_use": "...",
  "lease_commencement": "YYYY-MM-DD",
  "rent_commencement": "YYYY-MM-DD",
  "lease_expiration": "YYYY-MM-DD",
  "holdover_rate": "...",
  "rent_steps": [
    {{"effective_date": "YYYY-MM-DD", "monthly_rent": number, "annual_rent": number, "rent_per_sf": number}}
  ],
  "escalation_structure": "fixed $ / fixed % / CPI / other",
  "security_deposit": number,
  "cam_structure": "pro rata / fixed / gross",
  "cam_cap_pct": number or null,
  "admin_fee_pct": number or null,
  "tax_pass_through": true/false,
  "insurance_pass_through": true/false,
  "cotenancy": {{
    "has_clause": true/false,
    "named_cotenants": ["Tenant A", "Tenant B"],
    "trigger_threshold": "description of what triggers",
    "cure_period_days": number,
    "alt_rent_formula": "e.g. 50% of base rent or 2% of gross sales",
    "termination_right": true/false,
    "termination_notice_days": number,
    "sunset_or_waiver": "description of how right expires",
    "is_curable": true/false,
    "uncurable_scenario": "description if applicable"
  }},
  "exclusive_use": {{
    "has_clause": true/false,
    "restriction": "..."
  }},
  "renewal_options": [
    {{
      "option_number": 1,
      "total_options": number,
      "term_years": number,
      "option_start": "YYYY-MM-DD or null",
      "option_end": "YYYY-MM-DD or null",
      "notice_days": number,
      "notice_deadline": "YYYY-MM-DD or null",
      "auto_renewal": true/false,
      "exercised": true/false,
      "rent_terms": "fair market / fixed increase / CPI"
    }}
  ],
  "termination_options": [
    {{
      "option_number": 1,
      "total_options": number,
      "earliest_termination_date": "YYYY-MM-DD or null",
      "notice_days": number,
      "notice_deadline": "YYYY-MM-DD or null",
      "exercised": true/false,
      "termination_fee": "description of fee or penalty if any",
      "conditions": "description of conditions required to exercise"
    }}
  ],
  "assignment": {{
    "consent_required": true/false,
    "tenant_released": true/false
  }},
  "ti_allowance": number or null,
  "percentage_rent": {{
    "has_clause": true/false,
    "breakpoint": number or null,
    "rate_pct": number or null
  }},
  "go_dark_provision": "...",
  "key_dates": {{
    "next_notice_deadline": "YYYY-MM-DD or null",
    "description": "..."
  }}
}}

IMPORTANT:
- Extract rent steps from the rent schedule if present
- For amendments, only return CHANGED fields; unchanged fields should be null
- Dates must be YYYY-MM-DD format
- Dollar amounts should be numbers (no $ signs)
- If this is an amendment, note which fields were modified
- For renewal_options: option_start/option_end are the beginning and ending dates of each renewal period. If not explicitly stated, derive from the prior term's expiration + term_years. Mark exercised=true if an amendment or exercise notice confirms the option was exercised.
- For termination_options: extract early termination rights, kick-out clauses, and similar provisions. earliest_termination_date is when the tenant can first terminate. Mark exercised=true if a termination notice was exercised.

DOCUMENT TEXT:
{text}"""


def extract_lease_terms_via_api(
    text: str,
    tenant_name: str,
    suite: str,
    doc_type: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Claude API to extract structured lease terms from PDF text.

    Uses Haiku for cost efficiency on high-volume extraction.
    """
    import anthropic

    key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=key)

    # Truncate very long documents to stay within context
    max_chars = 180_000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED — document exceeds extraction limit]"

    prompt = EXTRACTION_PROMPT.format(
        tenant_name=tenant_name,
        suite=suite,
        doc_type=doc_type,
        text=text,
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Parse JSON from response
    try:
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON for {tenant_name}: {response_text[:200]}")

    return {'_raw_response': response_text, '_parse_error': True}


# ---------------------------------------------------------------------------
# Ingestion pipeline — load property into database
# ---------------------------------------------------------------------------

def ingest_property(
    engine,
    base_path: str,
    property_name: str,
    property_address: str = '',
    rent_roll_path: Optional[str] = None,
    created_by: str = 'system',
) -> int:
    """Ingest a property's lease data into the database.

    Steps:
    1. Scan folder structure
    2. Parse rent roll
    3. Parse cotenancy spreadsheet
    4. Create database records
    5. Catalog all PDF documents

    Returns the review_id.
    """
    from sqlalchemy import text

    scan = scan_property_folder(base_path, property_name, property_address)

    # Parse rent roll
    rr_df = pd.DataFrame()
    rr_path = rent_roll_path
    if not rr_path and scan['rent_roll_files']:
        # Use most recent rent roll
        rr_path = sorted(scan['rent_roll_files'])[-1]
    if rr_path:
        rr_df = parse_rent_roll(rr_path)

    # Parse cotenancy
    cotenancy_data = []
    exclusive_data = []
    if scan['cotenancy_file']:
        cotenancy_data, exclusive_data = parse_cotenancy_spreadsheet(
            scan['cotenancy_file']
        )

    # Build cotenancy lookup by tenant name (fuzzy)
    cot_lookup = {}
    for rec in cotenancy_data:
        cot_lookup[rec['tenant_name'].lower().strip()] = rec

    exc_lookup = {}
    for rec in exclusive_data:
        exc_lookup[rec['tenant_name'].lower().strip()] = rec

    with engine.connect() as conn:
        # Create review
        total_gla = float(rr_df['square_feet'].sum()) if len(rr_df) else 0
        total_rent = float(rr_df['annual_rent'].sum()) if len(rr_df) else 0
        total_tenants = len(rr_df) if len(rr_df) else len(scan['tenants'])

        result = conn.execute(text("""
            INSERT INTO lease_reviews
                (property_name, property_address, total_gla,
                 total_annual_rent, total_tenants, source_folder, created_by)
            VALUES (:pn, :pa, :gla, :rent, :cnt, :sf, :cb)
            RETURNING id
        """), {
            'pn': property_name, 'pa': property_address,
            'gla': total_gla, 'rent': total_rent,
            'cnt': total_tenants, 'sf': base_path, 'cb': created_by,
        })
        review_id = result.fetchone()[0]

        # Insert tenants from rent roll
        tenant_id_map = {}  # suite -> tenant_id
        if len(rr_df):
            for _, row in rr_df.iterrows():
                tname = row['tenant_name']
                suite = row['suite']
                sf = row['square_feet']
                ann_rent = row['annual_rent']

                # Check if material
                is_material = (sf >= MATERIAL_LEASE_SF_THRESHOLD or
                               ann_rent >= MATERIAL_LEASE_RENT_THRESHOLD)

                # Check cotenancy (fuzzy match)
                has_cot = _fuzzy_match_cotenancy(tname, cot_lookup)
                has_exc = _fuzzy_match_cotenancy(tname, exc_lookup)

                r = conn.execute(text("""
                    INSERT INTO lease_tenants
                        (review_id, tenant_name, suite, square_feet, lease_type,
                         lease_start, lease_end, term_months, monthly_rent,
                         annual_rent, rent_per_sf, security_deposit,
                         is_vacant, is_material, has_cotenancy, has_exclusive_use)
                    VALUES (:rid, :tn, :su, :sf, :lt,
                            :ls, :le, :tm, :mr,
                            :ar, :rpsf, :sd,
                            :iv, :im, :hc, :heu)
                    RETURNING id
                """), {
                    'rid': review_id, 'tn': tname, 'su': suite,
                    'sf': sf, 'lt': row['lease_type'],
                    'ls': row['lease_start'], 'le': row['lease_end'],
                    'tm': row['term_months'], 'mr': row['monthly_rent'],
                    'ar': ann_rent, 'rpsf': row['rent_per_sf_year'],
                    'sd': row['security_deposit'],
                    'iv': row['is_vacant'], 'im': is_material,
                    'hc': has_cot is not None, 'heu': has_exc is not None,
                })
                tenant_id = r.fetchone()[0]
                tenant_id_map[suite] = tenant_id

                # Insert cotenancy record if applicable
                if has_cot and has_cot.get('clause_text'):
                    conn.execute(text("""
                        INSERT INTO lease_cotenancy
                            (tenant_id, review_id, clause_text)
                        VALUES (:tid, :rid, :ct)
                    """), {
                        'tid': tenant_id, 'rid': review_id,
                        'ct': has_cot['clause_text'],
                    })

                # Insert exclusive use if applicable
                if has_exc:
                    conn.execute(text("""
                        INSERT INTO lease_exclusive_use
                            (tenant_id, restriction_text)
                        VALUES (:tid, :rt)
                    """), {
                        'tid': tenant_id,
                        'rt': has_exc['restriction_text'],
                    })

        # Match tenant folders to tenant records and catalog documents
        for tenant_folder in scan['tenants']:
            folder_name = tenant_folder['folder_name']
            # Match folder to a tenant_id via suite or name
            tenant_id = _match_folder_to_tenant(
                folder_name, tenant_id_map, rr_df
            )
            if tenant_id is None:
                logger.warning(f"No tenant match for folder: {folder_name}")
                continue

            for doc in tenant_folder['documents']:
                conn.execute(text("""
                    INSERT INTO lease_documents
                        (tenant_id, review_id, filename, file_path,
                         doc_type, doc_date)
                    VALUES (:tid, :rid, :fn, :fp, :dt, :dd)
                """), {
                    'tid': tenant_id, 'rid': review_id,
                    'fn': doc['filename'], 'fp': doc['path'],
                    'dt': doc['doc_type'], 'dd': doc['doc_date'],
                })

        conn.commit()

    logger.info(
        f"Ingested property '{property_name}': review_id={review_id}, "
        f"{len(tenant_id_map)} tenants, {total_gla:.0f} GLA"
    )
    return review_id


def _fuzzy_match_cotenancy(
    tenant_name: str,
    lookup: Dict[str, Dict],
) -> Optional[Dict]:
    """Fuzzy match tenant name to cotenancy lookup."""
    key = tenant_name.lower().strip()
    # Direct match
    if key in lookup:
        return lookup[key]
    # Partial match — check if lookup key is contained in tenant name or vice versa
    for lk, rec in lookup.items():
        # Strip numbers and common suffixes for comparison
        clean_key = re.sub(r'[#\d]+$', '', lk).strip()
        clean_name = re.sub(r'[#\d]+$', '', key).strip()
        if clean_key and clean_name:
            if clean_key in clean_name or clean_name in clean_key:
                return rec
    return None


def _match_folder_to_tenant(
    folder_name: str,
    tenant_id_map: Dict[str, int],
    rr_df: pd.DataFrame,
) -> Optional[int]:
    """Match a folder name like 'Ross Dress For Less 884' to a tenant_id."""
    if len(rr_df) == 0:
        return None

    # Known aliases: folder name -> rent roll name patterns
    FOLDER_ALIASES = {
        'afl': 'autism',
        'cornerstone church': 'cornerstone',
        'creamy rolls': 'creamy rolls',
        'hibachi grill & supreme buffet': 'hibachi',
        'hibachi grill': 'hibachi',
        'high five indoor playgournd': 'high five',
        'high five indoor playground': 'high five',
        'kohls': "kohl's",
        'velva nail spa': 'velva nail',
        'le nails': 'velva nail',
    }

    # Clean folder name — remove trailing numbers and parenthetical notes
    clean = re.sub(r'\s+\d+$', '', folder_name).strip()
    clean = re.sub(r'\s*\(former.*?\)', '', clean, flags=re.IGNORECASE).strip()
    clean_lower = clean.lower()

    # Try alias lookup
    for alias_key, alias_val in FOLDER_ALIASES.items():
        if alias_key in clean_lower:
            for _, row in rr_df.iterrows():
                rr_name = str(row['tenant_name']).lower().strip()
                if alias_val in rr_name:
                    return tenant_id_map.get(row['suite'])

    for _, row in rr_df.iterrows():
        rr_name = str(row['tenant_name']).lower().strip()
        suite = row['suite']
        # Direct name match
        if clean_lower in rr_name or rr_name in clean_lower:
            return tenant_id_map.get(suite)
        # Check folder name contains tenant or vice versa
        rr_clean = re.sub(r'[#\d]+$', '', rr_name).strip()
        if rr_clean and (rr_clean in clean_lower or clean_lower in rr_clean):
            return tenant_id_map.get(suite)

    return None


# ---------------------------------------------------------------------------
# Reset extraction data for re-extraction
# ---------------------------------------------------------------------------

def reset_extraction_data(engine, review_id: int) -> Dict[str, int]:
    """Clear all AI-extracted data for a review and reset documents to pending.

    Preserves: tenant roster (from rent roll), uploaded documents (PDFs),
    field resolutions, abstract sections.

    Clears: rent_steps, cotenancy + refs, exclusive_use, options,
    validation, tenant extraction_json/status, document extraction_status.
    """
    from sqlalchemy import text

    counts: Dict[str, int] = {}

    with engine.begin() as conn:
        # Get tenant IDs for this review
        tid_rows = conn.execute(text(
            "SELECT id FROM lease_tenants WHERE review_id = :rid"
        ), {'rid': review_id}).fetchall()
        tenant_ids = [r[0] for r in tid_rows]

        if not tenant_ids:
            return {'tenants': 0}

        # Build placeholder list for IN clause
        placeholders = ','.join(f':t{i}' for i in range(len(tenant_ids)))
        tid_params = {f't{i}': tid for i, tid in enumerate(tenant_ids)}
        base_params = {'rid': review_id, **tid_params}

        # Delete cotenancy refs (must go before cotenancy due to FK)
        r = conn.execute(text(f"""
            DELETE FROM lease_cotenancy_refs
            WHERE cotenancy_id IN (
                SELECT id FROM lease_cotenancy
                WHERE tenant_id IN ({placeholders})
            )
        """), tid_params)
        counts['cotenancy_refs'] = r.rowcount

        # Delete cotenancy
        r = conn.execute(text(f"""
            DELETE FROM lease_cotenancy
            WHERE tenant_id IN ({placeholders})
        """), tid_params)
        counts['cotenancy'] = r.rowcount

        # Delete rent steps
        r = conn.execute(text(f"""
            DELETE FROM lease_rent_steps
            WHERE tenant_id IN ({placeholders})
        """), tid_params)
        counts['rent_steps'] = r.rowcount

        # Delete options
        r = conn.execute(text(f"""
            DELETE FROM lease_options
            WHERE tenant_id IN ({placeholders})
        """), tid_params)
        counts['options'] = r.rowcount

        # Delete exclusive use
        r = conn.execute(text(f"""
            DELETE FROM lease_exclusive_use
            WHERE tenant_id IN ({placeholders})
        """), tid_params)
        counts['exclusive_use'] = r.rowcount

        # Delete validation
        r = conn.execute(text(f"""
            DELETE FROM lease_validation
            WHERE tenant_id IN ({placeholders})
        """), tid_params)
        counts['validation'] = r.rowcount

        # Reset tenant extraction status and JSON
        conn.execute(text(f"""
            UPDATE lease_tenants
            SET extraction_status = 'pending',
                extraction_json = NULL,
                has_cotenancy = FALSE,
                has_exclusive_use = FALSE,
                updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
        """), tid_params)

        # Reset document extraction status (keep extracted_text for re-use)
        r = conn.execute(text("""
            UPDATE lease_documents
            SET extraction_status = CASE
                WHEN extracted_text IS NOT NULL THEN 'text_extracted'
                ELSE 'pending'
            END
            WHERE review_id = :rid
        """), {'rid': review_id})
        counts['documents_reset'] = r.rowcount
        counts['tenants'] = len(tenant_ids)

    logger.info(f"Reset extraction data for review {review_id}: {counts}")
    return counts


# ---------------------------------------------------------------------------
# Extract lease terms via Claude API (batch)
# ---------------------------------------------------------------------------

def extract_all_documents(engine, review_id: int, api_key: Optional[str] = None):
    """Extract text from all PDFs and run Claude extraction for key documents.

    Prioritizes Original Lease and Amendment documents.
    """
    from sqlalchemy import text as sql_text

    with engine.connect() as conn:
        # Get all documents for this review (pending or text_extracted)
        docs = conn.execute(sql_text("""
            SELECT d.id, d.tenant_id, d.filename, d.file_path,
                   d.doc_type, d.doc_date,
                   t.tenant_name, t.suite,
                   d.extraction_status, d.extracted_text
            FROM lease_documents d
            JOIN lease_tenants t ON t.id = d.tenant_id
            WHERE d.review_id = :rid
            AND d.extraction_status IN ('pending', 'text_extracted')
            ORDER BY d.tenant_id, d.doc_date
        """), {'rid': review_id}).fetchall()

        logger.info(f"Extracting {len(docs)} documents for review {review_id}")

        for doc in docs:
            doc_id = doc[0]
            tenant_id = doc[1]
            file_path = doc[3]
            doc_type = doc[4]
            tenant_name = doc[6]
            suite = doc[7]
            current_status = doc[8]
            existing_text = doc[9]

            try:
                # Step 1: Extract PDF text (skip if already extracted)
                if current_status == 'text_extracted' and existing_text:
                    pdf_text = existing_text
                else:
                    pdf_text, page_count = extract_pdf_text(file_path)

                    conn.execute(sql_text("""
                        UPDATE lease_documents
                        SET extracted_text = :txt, page_count = :pc,
                            extraction_status = 'text_extracted'
                        WHERE id = :did
                    """), {'txt': pdf_text, 'pc': page_count, 'did': doc_id})

                # Step 2: Run Claude extraction for leases and amendments
                if doc_type in ('Original Lease', 'Amendment'):
                    terms = extract_lease_terms_via_api(
                        pdf_text, tenant_name, suite, doc_type, api_key
                    )

                    if not terms.get('_parse_error'):
                        conn.execute(sql_text("""
                            UPDATE lease_documents
                            SET extraction_status = 'extracted'
                            WHERE id = :did
                        """), {'did': doc_id})

                        # Store extraction JSON on tenant
                        conn.execute(sql_text("""
                            UPDATE lease_tenants
                            SET extraction_json = COALESCE(extraction_json, '[]'),
                                extraction_status = 'extracted',
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = :tid
                        """), {'tid': tenant_id})

                        # Store rent steps (with dedup)
                        if terms.get('rent_steps'):
                            for step in terms['rent_steps']:
                                # Dedup: skip if (tenant_id, effective_date) already exists
                                dup = conn.execute(sql_text("""
                                    SELECT id FROM lease_rent_steps
                                    WHERE tenant_id = :tid AND effective_date = :ed
                                    LIMIT 1
                                """), {
                                    'tid': tenant_id,
                                    'ed': step.get('effective_date'),
                                }).fetchone()
                                if dup:
                                    continue
                                conn.execute(sql_text("""
                                    INSERT INTO lease_rent_steps
                                        (tenant_id, effective_date,
                                         monthly_rent, annual_rent,
                                         rent_per_sf, source_doc)
                                    VALUES (:tid, :ed, :mr, :ar, :rpsf, :sd)
                                """), {
                                    'tid': tenant_id,
                                    'ed': step.get('effective_date'),
                                    'mr': step.get('monthly_rent'),
                                    'ar': step.get('annual_rent'),
                                    'rpsf': step.get('rent_per_sf'),
                                    'sd': doc[2],  # filename
                                })

                        # Store cotenancy from extraction (with dedup by source_doc)
                        cot = terms.get('cotenancy', {})
                        if cot and cot.get('has_clause') and not conn.execute(sql_text("""
                            SELECT id FROM lease_cotenancy
                            WHERE tenant_id = :tid AND source_doc = :sd LIMIT 1
                        """), {'tid': tenant_id, 'sd': doc[2]}).fetchone():
                            cot_result = conn.execute(sql_text("""
                                INSERT INTO lease_cotenancy
                                    (tenant_id, review_id,
                                     trigger_description, trigger_threshold,
                                     cure_period_days, alt_rent_formula,
                                     termination_right, termination_notice_days,
                                     sunset_provision, is_curable,
                                     waiver_mechanism, source_doc)
                                VALUES (:tid, :rid,
                                        :td, :tt, :cpd, :arf,
                                        :tr, :tnd, :sp, :ic,
                                        :wm, :sd)
                                RETURNING id
                            """), {
                                'tid': tenant_id, 'rid': review_id,
                                'td': cot.get('trigger_threshold'),
                                'tt': cot.get('trigger_threshold'),
                                'cpd': cot.get('cure_period_days'),
                                'arf': cot.get('alt_rent_formula'),
                                'tr': cot.get('termination_right', False),
                                'tnd': cot.get('termination_notice_days'),
                                'sp': cot.get('sunset_or_waiver'),
                                'ic': cot.get('is_curable', True),
                                'wm': cot.get('sunset_or_waiver'),
                                'sd': doc[2],
                            })
                            cot_id = cot_result.fetchone()[0]

                            # Insert named co-tenant references
                            for ref_name in (cot.get('named_cotenants') or []):
                                conn.execute(sql_text("""
                                    INSERT INTO lease_cotenancy_refs
                                        (cotenancy_id, tenant_id,
                                         referenced_tenant_name)
                                    VALUES (:cid, :tid, :rtn)
                                """), {
                                    'cid': cot_id, 'tid': tenant_id,
                                    'rtn': ref_name,
                                })

                        # Store renewal options (with dedup by source_doc + option_number)
                        for opt in (terms.get('renewal_options') or []):
                            opt_dup = conn.execute(sql_text("""
                                SELECT id FROM lease_options
                                WHERE tenant_id = :tid AND source_doc = :sd
                                  AND option_type = 'renewal'
                                  AND option_number = :on
                                LIMIT 1
                            """), {
                                'tid': tenant_id, 'sd': doc[2],
                                'on': opt.get('option_number'),
                            }).fetchone()
                            if opt_dup:
                                # Update exercised status if exercise notice confirms it
                                if opt.get('exercised'):
                                    conn.execute(sql_text("""
                                        UPDATE lease_options SET exercised = TRUE
                                        WHERE id = :id
                                    """), {'id': opt_dup[0]})
                                continue
                            conn.execute(sql_text("""
                                INSERT INTO lease_options
                                    (tenant_id, option_type, option_number,
                                     total_options, term_years, notice_days,
                                     notice_deadline, rent_terms,
                                     auto_renewal, exercised,
                                     option_start, option_end, source_doc)
                                VALUES (:tid, 'renewal', :on, :to, :ty,
                                        :nd, :ndl, :rt, :ar, :ex,
                                        :os, :oe, :sd)
                            """), {
                                'tid': tenant_id,
                                'on': opt.get('option_number'),
                                'to': opt.get('total_options'),
                                'ty': opt.get('term_years'),
                                'nd': opt.get('notice_days'),
                                'ndl': opt.get('notice_deadline'),
                                'rt': opt.get('rent_terms'),
                                'ar': opt.get('auto_renewal', False),
                                'ex': opt.get('exercised', False),
                                'os': opt.get('option_start'),
                                'oe': opt.get('option_end'),
                                'sd': doc[2],
                            })

                        # Store termination options (with dedup by source_doc + option_number)
                        for opt in (terms.get('termination_options') or []):
                            opt_dup = conn.execute(sql_text("""
                                SELECT id FROM lease_options
                                WHERE tenant_id = :tid AND source_doc = :sd
                                  AND option_type = 'termination'
                                  AND option_number = :on
                                LIMIT 1
                            """), {
                                'tid': tenant_id, 'sd': doc[2],
                                'on': opt.get('option_number'),
                            }).fetchone()
                            if opt_dup:
                                if opt.get('exercised'):
                                    conn.execute(sql_text("""
                                        UPDATE lease_options SET exercised = TRUE
                                        WHERE id = :id
                                    """), {'id': opt_dup[0]})
                                continue
                            conn.execute(sql_text("""
                                INSERT INTO lease_options
                                    (tenant_id, option_type, option_number,
                                     total_options, term_years, notice_days,
                                     notice_deadline, rent_terms,
                                     exercised, option_start, option_end,
                                     source_doc)
                                VALUES (:tid, 'termination', :on, :to, NULL,
                                        :nd, :ndl, :rt,
                                        :ex, :os, NULL,
                                        :sd)
                            """), {
                                'tid': tenant_id,
                                'on': opt.get('option_number'),
                                'to': opt.get('total_options'),
                                'nd': opt.get('notice_days'),
                                'ndl': opt.get('notice_deadline'),
                                'rt': opt.get('conditions') or opt.get('termination_fee') or '',
                                'ex': opt.get('exercised', False),
                                'os': opt.get('earliest_termination_date'),
                                'sd': doc[2],
                            })

                conn.commit()
                logger.info(f"Extracted: {doc[2]}")

            except Exception as e:
                logger.error(f"Error extracting {doc[2]}: {e}")
                conn.execute(sql_text("""
                    UPDATE lease_documents
                    SET extraction_status = 'error'
                    WHERE id = :did
                """), {'did': doc_id})
                conn.commit()


# ---------------------------------------------------------------------------
# Argus rent roll validation parsing
# ---------------------------------------------------------------------------

def parse_argus_validation(file_path: str) -> pd.DataFrame:
    """Parse the Argus Rent Roll Validation spreadsheet.

    This is the seller/broker's representation of Argus inputs.
    Returns a DataFrame with: suite, tenant_name, source, commencement,
    expiration, sq_ft, rent_per_sf.
    """
    import openpyxl
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb['RR Validation']

    rows = []
    current_suite = None
    current_tenant = None

    for r in range(4, ws.max_row + 1):
        col_b = ws.cell(r, 2).value  # Unit column
        col_c = ws.cell(r, 3).value  # Tenant name or empty
        col_d = ws.cell(r, 4).value  # Source (Argus / Rent Roll / Lease)

        if col_b and str(col_b).strip():
            suite = str(col_b).strip()
            # Check if this is a unit code (e.g. A200)
            if re.match(r'^[A-Z]\d', suite):
                current_suite = suite

        if col_c and str(col_c).strip() and str(col_c) != 'None':
            current_tenant = str(col_c).strip()

        if col_d and str(col_d).strip() in ('Argus', 'Rent Roll', 'Lease'):
            source = str(col_d).strip()

            def to_str(v):
                if v is None:
                    return None
                if isinstance(v, datetime):
                    return v.strftime('%Y-%m-%d')
                return str(v).strip()

            rows.append({
                'suite': current_suite,
                'tenant_name': current_tenant,
                'source': source,
                'commencement': to_str(ws.cell(r, 5).value),
                'expiration': to_str(ws.cell(r, 6).value),
                'sq_ft': ws.cell(r, 7).value,
                'rent_per_sf': to_str(ws.cell(r, 9).value),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation — compare seller sources to lease-extracted ground truth
# ---------------------------------------------------------------------------

def validate_rent_roll(
    engine,
    review_id: int,
    argus_file: Optional[str] = None,
) -> List[Dict]:
    """Compare seller-provided documents to extracted lease terms (ground truth).

    The actual lease PDFs are ground truth. The rent roll, Argus model,
    and cotenancy schedule are seller-provided representations that must
    be validated against the lease documents.

    Validation sources compared against lease:
    - Rent Roll (MRI/Yardi export): tenant, suite, SF, dates, rent
    - Argus (broker's DCF model): commencement, expiration, SF, rent/SF
    - Cotenancy Schedule (seller abstract): clause summaries

    Stores results in lease_validation with source_type field.
    """
    from sqlalchemy import text

    results = []

    # Get the rent roll date for finding the correct rent step
    with engine.connect() as conn:
        review = conn.execute(text(
            "SELECT rent_roll_date FROM lease_reviews WHERE id = :rid"
        ), {'rid': review_id}).fetchone()
        rr_date = review[0] if review and review[0] else None

    # Parse Argus file if provided
    argus_df = pd.DataFrame()
    if argus_file and os.path.exists(argus_file):
        argus_df = parse_argus_validation(argus_file)

    with engine.connect() as conn:
        # Clear prior validation results for this review
        conn.execute(text("""
            DELETE FROM lease_validation
            WHERE tenant_id IN (
                SELECT id FROM lease_tenants WHERE review_id = :rid
            )
        """), {'rid': review_id})

        tenants = conn.execute(text("""
            SELECT id, tenant_name, suite, square_feet, lease_start,
                   lease_end, monthly_rent, annual_rent, rent_per_sf,
                   security_deposit, extraction_json, extraction_status
            FROM lease_tenants
            WHERE review_id = :rid AND is_vacant = false
            ORDER BY suite
        """), {'rid': review_id}).fetchall()

        for t in tenants:
            tenant_id = t[0]
            tenant_name = t[1]
            suite = t[2]
            rr_sf = t[3]
            rr_start = t[4]
            rr_end = t[5]
            rr_monthly = t[6]
            rr_annual = t[7]
            rr_rpsf = t[8]
            extraction_status = t[11]

            # --- Find the correct rent step from lease (ground truth) ---
            # Strategy: find the step effective on the rent roll date.
            # Many extracted steps have non-date effective_dates (e.g.
            # "Lease Year 7", "Rent Commencement Date"), so we need
            # a fallback when date-based matching fails.
            current_step = None
            step_match_method = None

            # 1. Try date-based match: step effective on or before RR date
            if rr_date:
                current_step = conn.execute(text("""
                    SELECT effective_date, monthly_rent, annual_rent, rent_per_sf
                    FROM lease_rent_steps
                    WHERE tenant_id = :tid
                    AND effective_date <= :rrd
                    AND effective_date GLOB '[0-9][0-9][0-9][0-9]-*'
                    ORDER BY effective_date DESC
                    LIMIT 1
                """), {'tid': tenant_id, 'rrd': rr_date}).fetchone()
                if current_step:
                    step_match_method = 'date'

            # 2. Fallback: find the step whose annual rent is closest
            #    to the rent roll amount (best match by value)
            if not current_step and rr_annual:
                all_steps = conn.execute(text("""
                    SELECT effective_date, monthly_rent, annual_rent, rent_per_sf
                    FROM lease_rent_steps
                    WHERE tenant_id = :tid AND annual_rent IS NOT NULL
                """), {'tid': tenant_id}).fetchall()
                if all_steps:
                    best = min(all_steps, key=lambda s: abs(
                        (float(s[2]) if s[2] else 0) - float(rr_annual)
                    ))
                    current_step = best
                    step_match_method = 'closest_rent'

            # 3. Last fallback: any step at all
            if not current_step:
                current_step = conn.execute(text("""
                    SELECT effective_date, monthly_rent, annual_rent, rent_per_sf
                    FROM lease_rent_steps
                    WHERE tenant_id = :tid
                    ORDER BY effective_date DESC
                    LIMIT 1
                """), {'tid': tenant_id}).fetchone()
                if current_step:
                    step_match_method = 'fallback'

            # Get extraction JSON for date/SF comparisons
            ext = {}
            if t[10]:
                try:
                    ext = json.loads(t[10]) if isinstance(t[10], str) else t[10]
                except (json.JSONDecodeError, TypeError):
                    pass

            # --- 1. Rent Roll vs Lease Validation ---
            if current_step:
                lease_monthly = current_step[1]
                lease_annual = current_step[2]
                lease_rpsf = current_step[3]
                step_date = current_step[0]

                rr_validations = [
                    ('monthly_rent', rr_monthly, lease_monthly,
                     _compare_amounts(rr_monthly, lease_monthly)),
                    ('annual_rent', rr_annual, lease_annual,
                     _compare_amounts(rr_annual, lease_annual)),
                    ('rent_per_sf', rr_rpsf, lease_rpsf,
                     _compare_amounts(rr_rpsf, lease_rpsf, tolerance=0.05)),
                ]

                # Validate SF
                lease_sf = ext.get('square_feet')
                if lease_sf:
                    rr_validations.append(
                        ('square_feet', rr_sf, lease_sf,
                         _compare_amounts(rr_sf, lease_sf, tolerance=10))
                    )

                # Validate lease expiration
                lease_exp = ext.get('lease_expiration')
                if lease_exp and rr_end:
                    rr_validations.append(
                        ('lease_expiration', rr_end, lease_exp,
                         _compare_dates(rr_end, lease_exp))
                    )

                for field, seller_val, lease_val, status in rr_validations:
                    if field.startswith(('monthly', 'annual', 'rent_per')):
                        note = f"Step {step_date} (matched by {step_match_method})"
                    else:
                        note = None
                    conn.execute(text("""
                        INSERT INTO lease_validation
                            (tenant_id, field_name, source_type,
                             seller_value, lease_value, status,
                             source_doc, notes)
                        VALUES (:tid, :fn, 'rent_roll',
                                :sv, :lv, :st, :sd, :notes)
                    """), {
                        'tid': tenant_id, 'fn': field,
                        'sv': str(seller_val) if seller_val is not None else None,
                        'lv': str(lease_val) if lease_val is not None else None,
                        'st': status, 'sd': 'rent_roll',
                        'notes': note,
                    })
                    results.append({
                        'tenant': tenant_name, 'suite': suite,
                        'field': field, 'source_type': 'rent_roll',
                        'seller_value': str(seller_val) if seller_val is not None else None,
                        'lease_value': str(lease_val) if lease_val is not None else None,
                        'status': status,
                        'notes': note,
                    })
            elif extraction_status != 'extracted':
                # No extraction yet — mark as pending
                results.append({
                    'tenant': tenant_name, 'suite': suite,
                    'field': 'all', 'source_type': 'rent_roll',
                    'seller_value': None, 'lease_value': None,
                    'status': 'pending',
                    'notes': 'Lease not yet extracted',
                })

            # --- 2. Argus vs Lease Validation ---
            if len(argus_df) and ext:
                # Find Argus rows for this suite
                argus_rows = argus_df[
                    (argus_df['suite'] == suite) &
                    (argus_df['source'] == 'Argus')
                ]
                if len(argus_rows):
                    ar = argus_rows.iloc[0]
                    # Argus expiration vs lease expiration
                    argus_exp = ar.get('expiration')
                    lease_exp = ext.get('lease_expiration')
                    if argus_exp and lease_exp:
                        status = _compare_dates(argus_exp, lease_exp)
                        conn.execute(text("""
                            INSERT INTO lease_validation
                                (tenant_id, field_name, source_type,
                                 seller_value, lease_value, status, source_doc)
                            VALUES (:tid, 'lease_expiration', 'argus',
                                    :sv, :lv, :st, 'argus')
                        """), {
                            'tid': tenant_id,
                            'sv': str(argus_exp), 'lv': str(lease_exp),
                            'st': status,
                        })
                        results.append({
                            'tenant': tenant_name, 'suite': suite,
                            'field': 'lease_expiration',
                            'source_type': 'argus',
                            'seller_value': str(argus_exp),
                            'lease_value': str(lease_exp),
                            'status': status,
                        })

                    # Argus SF vs lease SF
                    argus_sf = ar.get('sq_ft')
                    lease_sf = ext.get('square_feet')
                    if argus_sf and lease_sf:
                        status = _compare_amounts(argus_sf, lease_sf, tolerance=10)
                        conn.execute(text("""
                            INSERT INTO lease_validation
                                (tenant_id, field_name, source_type,
                                 seller_value, lease_value, status, source_doc)
                            VALUES (:tid, 'square_feet', 'argus',
                                    :sv, :lv, :st, 'argus')
                        """), {
                            'tid': tenant_id,
                            'sv': str(argus_sf), 'lv': str(lease_sf),
                            'st': status,
                        })
                        results.append({
                            'tenant': tenant_name, 'suite': suite,
                            'field': 'square_feet',
                            'source_type': 'argus',
                            'seller_value': str(argus_sf),
                            'lease_value': str(lease_sf),
                            'status': status,
                        })

                    # Argus rent/SF vs lease rent/SF
                    argus_rpsf = ar.get('rent_per_sf')
                    if argus_rpsf and current_step:
                        lease_rpsf = current_step[3]
                        try:
                            argus_rpsf_f = float(argus_rpsf)
                        except (ValueError, TypeError):
                            argus_rpsf_f = None
                        if argus_rpsf_f is not None and lease_rpsf:
                            status = _compare_amounts(
                                argus_rpsf_f, lease_rpsf, tolerance=0.05
                            )
                            conn.execute(text("""
                                INSERT INTO lease_validation
                                    (tenant_id, field_name, source_type,
                                     seller_value, lease_value, status,
                                     source_doc)
                                VALUES (:tid, 'rent_per_sf', 'argus',
                                        :sv, :lv, :st, 'argus')
                            """), {
                                'tid': tenant_id,
                                'sv': str(argus_rpsf_f),
                                'lv': str(lease_rpsf),
                                'st': status,
                            })
                            results.append({
                                'tenant': tenant_name, 'suite': suite,
                                'field': 'rent_per_sf',
                                'source_type': 'argus',
                                'seller_value': str(argus_rpsf_f),
                                'lease_value': str(lease_rpsf),
                                'status': status,
                            })

            # --- 3. Cotenancy Schedule vs Lease Validation ---
            cot_row = conn.execute(text("""
                SELECT clause_text, trigger_description, alt_rent_formula,
                       termination_right, is_curable
                FROM lease_cotenancy
                WHERE tenant_id = :tid
            """), {'tid': tenant_id}).fetchone()

            cot_ext = ext.get('cotenancy', {}) if ext else {}
            if cot_row and cot_ext and cot_ext.get('has_clause'):
                # Compare trigger descriptions
                if cot_row[1] and cot_ext.get('trigger_threshold'):
                    results.append({
                        'tenant': tenant_name, 'suite': suite,
                        'field': 'cotenancy_trigger',
                        'source_type': 'cotenancy_schedule',
                        'seller_value': cot_row[1],
                        'lease_value': cot_ext.get('trigger_threshold'),
                        'status': 'review',
                        'notes': 'Manual review recommended — compare trigger language',
                    })

                # Compare alt rent
                if cot_row[2] and cot_ext.get('alt_rent_formula'):
                    results.append({
                        'tenant': tenant_name, 'suite': suite,
                        'field': 'cotenancy_alt_rent',
                        'source_type': 'cotenancy_schedule',
                        'seller_value': cot_row[2],
                        'lease_value': cot_ext.get('alt_rent_formula'),
                        'status': 'review',
                        'notes': 'Manual review recommended — compare alt rent formulas',
                    })

        conn.commit()
    return results


def _compare_dates(val1: str, val2: str) -> str:
    """Compare two date strings. Returns 'match', 'minor', or 'mismatch'."""
    try:
        d1 = pd.to_datetime(val1)
        d2 = pd.to_datetime(val2)
    except (ValueError, TypeError):
        return 'pending'
    diff = abs((d1 - d2).days)
    if diff == 0:
        return 'match'
    if diff <= 31:
        return 'minor'
    return 'mismatch'


def _compare_amounts(
    rr_val: float,
    lease_val: float,
    tolerance: float = 1.0,
) -> str:
    """Compare two amounts. Returns 'match', 'minor', or 'mismatch'."""
    try:
        rr = float(rr_val) if rr_val else 0
        lv = float(lease_val) if lease_val else 0
    except (ValueError, TypeError):
        return 'pending'
    if rr == 0 and lv == 0:
        return 'match'
    if abs(rr - lv) <= tolerance:
        return 'match'
    if rr != 0 and abs(rr - lv) / abs(rr) < 0.02:
        return 'minor'
    return 'mismatch'


# ---------------------------------------------------------------------------
# Analysis — expiration histogram, cotenancy matrix, scenarios
# ---------------------------------------------------------------------------

def get_expiration_histogram(
    engine,
    review_id: int,
    years: int = 10,
) -> Dict[str, Any]:
    """Build lease expiration histogram by year.

    Returns:
        - yearly_data: list of dicts per year (year, expiring_sf, expiring_rent,
          pct_of_total_rent, avg_rent_per_sf, tenant_count)
        - material_leases: list of material leases per year with cotenancy flags
        - totals: total_gla, total_annual_rent
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Get review totals
        review = conn.execute(text("""
            SELECT total_gla, total_annual_rent
            FROM lease_reviews WHERE id = :rid
        """), {'rid': review_id}).fetchone()
        total_gla = review[0] or 0
        total_rent = review[1] or 0

        # Get all non-vacant tenants with lease end dates
        tenants = conn.execute(text("""
            SELECT id, tenant_name, suite, square_feet, lease_end,
                   annual_rent, rent_per_sf, is_material, has_cotenancy
            FROM lease_tenants
            WHERE review_id = :rid AND is_vacant = false
            AND lease_end IS NOT NULL
            ORDER BY lease_end
        """), {'rid': review_id}).fetchall()

    current_year = datetime.now().year
    end_year = current_year + years

    yearly = {}
    for yr in range(current_year, end_year + 1):
        yearly[yr] = {
            'year': yr,
            'tenants': [],
            'expiring_sf': 0,
            'expiring_rent': 0,
            'pct_of_total_rent': 0,
            'avg_rent_per_sf': 0,
            'tenant_count': 0,
        }

    material_by_year = {}

    for t in tenants:
        try:
            lease_end = pd.to_datetime(t[4])
            exp_year = lease_end.year
        except Exception:
            continue

        if exp_year < current_year or exp_year > end_year:
            continue

        sf = t[3] or 0
        rent = t[5] or 0
        rpsf = t[6] or 0

        yearly[exp_year]['tenants'].append(t[1])
        yearly[exp_year]['expiring_sf'] += sf
        yearly[exp_year]['expiring_rent'] += rent
        yearly[exp_year]['tenant_count'] += 1

        # Track material leases
        if t[7]:  # is_material
            if exp_year not in material_by_year:
                material_by_year[exp_year] = []
            material_by_year[exp_year].append({
                'tenant_name': t[1],
                'suite': t[2],
                'square_feet': sf,
                'annual_rent': rent,
                'rent_per_sf': rpsf,
                'lease_end': t[4],
                'has_cotenancy': bool(t[8]),
                'cotenancy_implication': (
                    'Departure may trigger co-tenancy clauses in other leases'
                    if t[8] else None
                ),
            })

    # Calculate percentages and averages
    yearly_data = []
    for yr in range(current_year, end_year + 1):
        d = yearly[yr]
        if total_rent > 0:
            d['pct_of_total_rent'] = round(d['expiring_rent'] / total_rent * 100, 1)
        if d['expiring_sf'] > 0:
            d['avg_rent_per_sf'] = round(d['expiring_rent'] / d['expiring_sf'], 2)
        del d['tenants']
        yearly_data.append(d)

    return {
        'yearly_data': yearly_data,
        'material_leases': material_by_year,
        'totals': {
            'total_gla': total_gla,
            'total_annual_rent': total_rent,
        },
    }


def get_cotenancy_matrix(engine, review_id: int) -> Dict[str, Any]:
    """Build the co-tenancy cross-reference matrix.

    Returns which tenants are named as co-tenants in which leases,
    plus the "who depends on whom" reverse lookup.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Get all cotenancy records with refs
        cotenancy = conn.execute(text("""
            SELECT c.id, c.tenant_id, t.tenant_name, t.suite,
                   t.annual_rent, c.clause_text,
                   c.trigger_description, c.alt_rent_formula,
                   c.termination_right, c.cure_period_days,
                   c.sunset_provision, c.is_curable
            FROM lease_cotenancy c
            JOIN lease_tenants t ON t.id = c.tenant_id
            WHERE c.review_id = :rid
        """), {'rid': review_id}).fetchall()

        refs = conn.execute(text("""
            SELECT cr.cotenancy_id, cr.referenced_tenant_name
            FROM lease_cotenancy_refs cr
            JOIN lease_cotenancy c ON c.id = cr.cotenancy_id
            WHERE c.review_id = :rid
        """), {'rid': review_id}).fetchall()

    # Build forward map: tenant -> list of named cotenants
    forward = {}
    cot_details = {}
    for c in cotenancy:
        tenant_name = c[2]
        forward[tenant_name] = []
        cot_details[tenant_name] = {
            'suite': c[3],
            'annual_rent': c[4],
            'clause_summary': c[5][:200] if c[5] else '',
            'trigger': c[6],
            'alt_rent': c[7],
            'termination_right': c[8],
            'cure_days': c[9],
            'sunset': c[10],
            'is_curable': c[11],
        }

    # Map refs
    cot_id_to_tenant = {c[0]: c[2] for c in cotenancy}
    for ref in refs:
        cot_id = ref[0]
        ref_name = ref[1]
        tenant_name = cot_id_to_tenant.get(cot_id)
        if tenant_name:
            forward[tenant_name].append(ref_name)

    # Build reverse map: referenced tenant -> list of tenants that depend on them
    reverse = {}
    for tenant, named_cotenants in forward.items():
        for cotenant in named_cotenants:
            if cotenant not in reverse:
                reverse[cotenant] = []
            detail = cot_details.get(tenant, {})
            reverse[cotenant].append({
                'dependent_tenant': tenant,
                'annual_rent': detail.get('annual_rent', 0),
                'alt_rent': detail.get('alt_rent', ''),
                'termination_right': detail.get('termination_right', False),
            })

    # Calculate rent at risk per referenced tenant
    rent_at_risk = {}
    for cotenant, dependents in reverse.items():
        total_rent = sum(d.get('annual_rent', 0) or 0 for d in dependents)
        term_count = sum(1 for d in dependents if d.get('termination_right'))
        rent_at_risk[cotenant] = {
            'total_dependent_rent': total_rent,
            'dependent_count': len(dependents),
            'termination_eligible_count': term_count,
            'dependents': dependents,
        }

    return {
        'forward': forward,
        'reverse': reverse,
        'rent_at_risk': rent_at_risk,
        'details': cot_details,
    }


def get_scenario_analysis(engine, review_id: int) -> List[Dict]:
    """Model cascading co-tenancy scenarios for major tenant departures.

    For each tenant that is referenced as a co-tenant by others,
    model what happens if they close.
    """
    matrix = get_cotenancy_matrix(engine, review_id)

    scenarios = []
    for cotenant, risk in matrix['rent_at_risk'].items():
        if risk['dependent_count'] == 0:
            continue

        scenario = {
            'departing_tenant': cotenant,
            'dependent_count': risk['dependent_count'],
            'total_dependent_rent': risk['total_dependent_rent'],
            'termination_eligible': risk['termination_eligible_count'],
            'impacts': [],
        }

        for dep in risk['dependents']:
            detail = matrix['details'].get(dep['dependent_tenant'], {})
            scenario['impacts'].append({
                'tenant': dep['dependent_tenant'],
                'annual_rent': dep.get('annual_rent', 0),
                'alt_rent_formula': dep.get('alt_rent', ''),
                'can_terminate': dep.get('termination_right', False),
                'cure_days': detail.get('cure_days'),
                'sunset': detail.get('sunset'),
                'is_curable': detail.get('is_curable', True),
            })

        scenarios.append(scenario)

    # Sort by total rent at risk descending
    scenarios.sort(key=lambda s: s['total_dependent_rent'], reverse=True)
    return scenarios


# ---------------------------------------------------------------------------
# Excel report generation
# ---------------------------------------------------------------------------

def generate_lease_review_excel(engine, review_id: int) -> bytes:
    """Generate the comprehensive lease review Excel workbook."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import BarChart, Reference
    from openpyxl.utils import get_column_letter
    from io import BytesIO

    wb = openpyxl.Workbook()

    # Styles
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4",
                              fill_type="solid")
    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE",
                             fill_type="solid")
    yellow_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C",
                              fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE",
                           fill_type="solid")
    currency_fmt = '#,##0'
    pct_fmt = '0.0%'
    thin_border = Border(
        bottom=Side(style='thin'),
    )

    from sqlalchemy import text

    with engine.connect() as conn:
        review = conn.execute(text("""
            SELECT property_name, property_address, total_gla,
                   total_annual_rent, total_tenants
            FROM lease_reviews WHERE id = :rid
        """), {'rid': review_id}).fetchone()

        tenants = conn.execute(text("""
            SELECT tenant_name, suite, square_feet, lease_type,
                   lease_start, lease_end, term_months, monthly_rent,
                   annual_rent, rent_per_sf, security_deposit,
                   is_vacant, is_material, has_cotenancy
            FROM lease_tenants
            WHERE review_id = :rid
            ORDER BY suite
        """), {'rid': review_id}).fetchall()

    # --- Sheet 1: Executive Summary ---
    ws = wb.active
    ws.title = "Executive Summary"
    ws['A1'] = f"Lease Review: {review[0]}"
    ws['A1'].font = Font(bold=True, size=16)
    ws['A2'] = review[1] or ''
    ws['A4'] = "Total GLA"
    ws['B4'] = review[2]
    ws['B4'].number_format = '#,##0'
    ws['A5'] = "Total Annual Rent"
    ws['B5'] = review[3]
    ws['B5'].number_format = '#,##0'
    ws['A6'] = "Total Tenants"
    ws['B6'] = review[4]
    ws['A7'] = "Tenants with Co-Tenancy"
    ws['B7'] = sum(1 for t in tenants if t[13])
    ws['A8'] = "Vacant Suites"
    ws['B8'] = sum(1 for t in tenants if t[11])
    for r in range(4, 9):
        ws.cell(r, 1).font = Font(bold=True)
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 18

    # --- Sheet 2: Lease Expiration Histogram ---
    histogram = get_expiration_histogram(engine, review_id)
    ws2 = wb.create_sheet("Lease Expirations")

    # Header
    ws2['A1'] = f"Lease Expiration Schedule — {review[0]}"
    ws2['A1'].font = Font(bold=True, size=14)
    ws2['A2'] = f"Total Annual Rent: ${review[3]:,.0f}   |   Total GLA: {review[2]:,.0f} SF"

    headers = ['Year', 'Expiring SF', 'Expiring Rent', '% of Total Rent',
               'Avg Rent/SF', '# Tenants']
    for c, h in enumerate(headers, 1):
        cell = ws2.cell(4, c, h)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')

    for i, yr_data in enumerate(histogram['yearly_data']):
        row = 5 + i
        ws2.cell(row, 1, yr_data['year'])
        ws2.cell(row, 2, yr_data['expiring_sf']).number_format = '#,##0'
        ws2.cell(row, 3, yr_data['expiring_rent']).number_format = '#,##0'
        ws2.cell(row, 4, yr_data['pct_of_total_rent'] / 100).number_format = '0.0%'
        ws2.cell(row, 5, yr_data['avg_rent_per_sf']).number_format = '#,##0.00'
        ws2.cell(row, 6, yr_data['tenant_count'])

    # Add bar chart
    chart = BarChart()
    chart.title = "Annual Lease Revenue Expiring"
    chart.y_axis.title = "Annual Rent ($)"
    chart.x_axis.title = "Year"
    chart.style = 10
    chart.width = 20
    chart.height = 12

    data_rows = len(histogram['yearly_data'])
    if data_rows > 0:
        data_ref = Reference(ws2, min_col=3, min_row=4,
                             max_row=4 + data_rows)
        cats = Reference(ws2, min_col=1, min_row=5,
                         max_row=4 + data_rows)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats)
        chart.series[0].graphicalProperties.solidFill = "4472C4"
        ws2.add_chart(chart, f"A{5 + data_rows + 2}")

    # Material leases below chart
    mat_start = 5 + data_rows + 18
    ws2.cell(mat_start, 1, "Material Leases Maturing by Year").font = Font(
        bold=True, size=12)
    mat_headers = ['Year', 'Tenant', 'Suite', 'Square Feet', 'Annual Rent',
                   '$/SF', 'Lease End', 'Co-Tenancy Risk']
    for c, h in enumerate(mat_headers, 1):
        cell = ws2.cell(mat_start + 1, c, h)
        cell.font = header_font_white
        cell.fill = header_fill
    r = mat_start + 2
    for yr in sorted(histogram['material_leases'].keys()):
        for lease in histogram['material_leases'][yr]:
            ws2.cell(r, 1, yr)
            ws2.cell(r, 2, lease['tenant_name'])
            ws2.cell(r, 3, lease['suite'])
            ws2.cell(r, 4, lease['square_feet']).number_format = '#,##0'
            ws2.cell(r, 5, lease['annual_rent']).number_format = '#,##0'
            ws2.cell(r, 6, lease['rent_per_sf']).number_format = '#,##0.00'
            ws2.cell(r, 7, lease['lease_end'])
            cot_text = lease.get('cotenancy_implication', '')
            ws2.cell(r, 8, cot_text or 'None')
            if cot_text:
                ws2.cell(r, 8).fill = yellow_fill
            r += 1

    # Auto-width
    for col in range(1, 9):
        ws2.column_dimensions[get_column_letter(col)].width = 18

    # --- Sheet 3: Rent Roll Validation ---
    ws3 = wb.create_sheet("Rent Roll Validation")
    ws3['A1'] = "Seller Document Validation vs Lease Terms (Ground Truth)"
    ws3['A1'].font = Font(bold=True, size=14)
    ws3['A2'] = "Lease PDFs are the authoritative source. Rent Roll and Argus are seller representations validated against actual lease terms."

    # Pull validation results from DB
    with engine.connect() as conn:
        val_rows = conn.execute(text("""
            SELECT t.tenant_name, t.suite, v.field_name, v.source_type,
                   v.seller_value, v.lease_value, v.status, v.notes
            FROM lease_validation v
            JOIN lease_tenants t ON t.id = v.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name, v.source_type, v.field_name
        """), {'rid': review_id}).fetchall()

    val_headers = ['Tenant', 'Suite', 'Field', 'Source', 'Seller Value',
                   'Lease Value', 'Status', 'Notes']
    for c, h in enumerate(val_headers, 1):
        cell = ws3.cell(4, c, h)
        cell.font = header_font_white
        cell.fill = header_fill

    for i, v in enumerate(val_rows):
        row = 5 + i
        ws3.cell(row, 1, v[0])  # tenant_name
        ws3.cell(row, 2, v[1])  # suite
        ws3.cell(row, 3, v[2])  # field_name
        ws3.cell(row, 4, v[3])  # source_type
        # Format numeric values
        for col_idx, val_idx in [(5, 4), (6, 5)]:
            val = v[val_idx]
            try:
                num = float(val) if val else None
                if num is not None:
                    ws3.cell(row, col_idx, num).number_format = '#,##0.00'
                else:
                    ws3.cell(row, col_idx, val or '')
            except (ValueError, TypeError):
                ws3.cell(row, col_idx, val or '')
        ws3.cell(row, 7, v[6])  # status
        ws3.cell(row, 8, v[7] or '')  # notes

        # Color code status
        status = v[6]
        if status == 'match':
            ws3.cell(row, 7).fill = green_fill
        elif status == 'mismatch':
            ws3.cell(row, 7).fill = red_fill
        elif status == 'minor':
            ws3.cell(row, 7).fill = yellow_fill
        elif status == 'review':
            ws3.cell(row, 7).fill = yellow_fill

    col_widths = [30, 12, 18, 18, 18, 18, 12, 45]
    for c, w in enumerate(col_widths, 1):
        ws3.column_dimensions[get_column_letter(c)].width = w

    # --- Sheet 4: Co-Tenancy Detail ---
    ws4 = wb.create_sheet("Co-Tenancy Detail")
    ws4['A1'] = "Co-Tenancy Clause Analysis"
    ws4['A1'].font = Font(bold=True, size=14)
    ws4['A2'] = "Lease PDFs are ground truth. Seller cotenancy schedule validated against actual lease terms."

    cot_headers = ['Tenant', 'Suite', 'SF', 'Annual Rent',
                   'Trigger', 'Cure Period', 'Alt Rent Formula',
                   'Termination Right', 'Sunset/Waiver', 'Curable?',
                   'Named Co-Tenants']
    for c, h in enumerate(cot_headers, 1):
        cell = ws4.cell(4, c, h)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', wrap_text=True)

    # Get cotenancy details from DB
    with engine.connect() as conn:
        cot_rows = conn.execute(text("""
            SELECT t.tenant_name, t.suite, t.square_feet, t.annual_rent,
                   c.trigger_description, c.cure_period_days,
                   c.alt_rent_formula, c.termination_right,
                   c.sunset_provision, c.is_curable, c.waiver_mechanism,
                   c.id
            FROM lease_cotenancy c
            JOIN lease_tenants t ON t.id = c.tenant_id
            WHERE c.review_id = :rid
            ORDER BY t.annual_rent DESC
        """), {'rid': review_id}).fetchall()

        # Get refs for each cotenancy
        all_refs = conn.execute(text("""
            SELECT cr.cotenancy_id, cr.referenced_tenant_name
            FROM lease_cotenancy_refs cr
            JOIN lease_cotenancy c ON c.id = cr.cotenancy_id
            WHERE c.review_id = :rid
        """), {'rid': review_id}).fetchall()

    ref_map = {}
    for r in all_refs:
        ref_map.setdefault(r[0], []).append(r[1])

    for i, c in enumerate(cot_rows):
        row = 5 + i
        ws4.cell(row, 1, c[0])   # tenant_name
        ws4.cell(row, 2, c[1])   # suite
        ws4.cell(row, 3, c[2]).number_format = '#,##0'
        ws4.cell(row, 4, c[3]).number_format = '#,##0'
        ws4.cell(row, 5, c[4] or 'See lease')  # trigger
        ws4.cell(row, 6, f"{c[5]} days" if c[5] else 'N/A')  # cure
        ws4.cell(row, 7, c[6] or 'N/A')  # alt rent
        ws4.cell(row, 8, 'Yes' if c[7] else 'No')  # termination
        ws4.cell(row, 9, c[8] or 'N/A')  # sunset
        curable = c[9]
        ws4.cell(row, 10, 'Yes' if curable else 'NO - UNCURABLE')
        if not curable:
            ws4.cell(row, 10).fill = red_fill
            ws4.cell(row, 10).font = Font(bold=True, color="9C0006")
        # Named cotenants
        refs = ref_map.get(c[11], [])
        ws4.cell(row, 11, ', '.join(refs) if refs else 'N/A')

        # Wrap text for readability
        for col in (5, 7, 9, 11):
            ws4.cell(row, col).alignment = Alignment(wrap_text=True,
                                                      vertical='top')

    col_widths = [25, 10, 10, 14, 35, 12, 30, 14, 35, 16, 35]
    for c, w in enumerate(col_widths, 1):
        ws4.column_dimensions[get_column_letter(c)].width = w

    # Add risk summary below
    risk_start = 5 + len(cot_rows) + 3
    ws4.cell(risk_start, 1, "Departing Tenant Risk Summary").font = Font(
        bold=True, size=12)
    risk_headers = ['Departing Tenant', 'Affected Count', 'Rent at Risk',
                    'Can Terminate']
    for c, h in enumerate(risk_headers, 1):
        cell = ws4.cell(risk_start + 1, c, h)
        cell.font = header_font_white
        cell.fill = header_fill

    matrix = get_cotenancy_matrix(engine, review_id)
    r = risk_start + 2
    for cotenant, risk in sorted(matrix['rent_at_risk'].items(),
                                  key=lambda x: x[1]['total_dependent_rent'],
                                  reverse=True):
        ws4.cell(r, 1, cotenant)
        ws4.cell(r, 2, risk['dependent_count'])
        ws4.cell(r, 3, risk['total_dependent_rent']).number_format = '#,##0'
        ws4.cell(r, 4, risk['termination_eligible_count'])
        r += 1

    # --- Sheet 5: Exclusive Use ---
    ws5 = wb.create_sheet("Exclusive Use")
    ws5['A1'] = "Exclusive Use Restrictions"
    ws5['A1'].font = Font(bold=True, size=14)

    with engine.connect() as conn:
        exc_rows = conn.execute(text("""
            SELECT t.tenant_name, t.suite, e.restriction_text
            FROM lease_exclusive_use e
            JOIN lease_tenants t ON t.id = e.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name
        """), {'rid': review_id}).fetchall()

    exc_headers = ['Tenant', 'Suite', 'Exclusive Use Restriction']
    for c, h in enumerate(exc_headers, 1):
        cell = ws5.cell(3, c, h)
        cell.font = header_font_white
        cell.fill = header_fill
    for i, e in enumerate(exc_rows):
        row = 4 + i
        ws5.cell(row, 1, e[0])
        ws5.cell(row, 2, e[1])
        ws5.cell(row, 3, e[2]).alignment = Alignment(wrap_text=True)
    ws5.column_dimensions['A'].width = 30
    ws5.column_dimensions['B'].width = 12
    ws5.column_dimensions['C'].width = 60

    # --- Sheet 6: Option Schedule ---
    ws6 = wb.create_sheet("Option Schedule")
    ws6['A1'] = "Renewal / Termination Options"
    ws6['A1'].font = Font(bold=True, size=14)

    with engine.connect() as conn:
        opt_rows = conn.execute(text("""
            SELECT t.tenant_name, t.suite, o.option_type,
                   o.option_number, o.total_options,
                   o.option_start, o.option_end, o.term_years,
                   o.notice_days, o.notice_deadline, o.rent_terms,
                   o.auto_renewal, o.exercised, o.source_doc
            FROM lease_options o
            JOIN lease_tenants t ON t.id = o.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name, o.option_type, o.option_number
        """), {'rid': review_id}).fetchall()

    opt_headers = ['Tenant', 'Suite', 'Type', 'Option #', 'Total Options',
                   'Start', 'End', 'Term (Yrs)',
                   'Notice (Days)', 'Notice Deadline',
                   'Rent Terms / Conditions', 'Auto-Renew', 'Exercised', 'Source']
    for c, h in enumerate(opt_headers, 1):
        cell = ws6.cell(3, c, h)
        cell.font = header_font_white
        cell.fill = header_fill
    for i, o in enumerate(opt_rows):
        row = 4 + i
        for c, val in enumerate(o, 1):
            cell = ws6.cell(row, c, val)
            if c == 11:  # rent_terms
                cell.alignment = Alignment(wrap_text=True)

    opt_widths = [30, 12, 12, 10, 10, 14, 14, 10, 12, 16, 40, 10, 10, 30]
    for c, w in enumerate(opt_widths, 1):
        ws6.column_dimensions[get_column_letter(c)].width = w

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Lease Risk Analysis — field resolutions + resolved data
# ---------------------------------------------------------------------------

LEASE_RESOLUTION_DDL_PG = """
CREATE TABLE IF NOT EXISTS lease_field_resolutions (
    id              SERIAL PRIMARY KEY,
    tenant_id       INTEGER NOT NULL REFERENCES lease_tenants(id),
    field_name      TEXT NOT NULL,
    resolved_value  TEXT,
    resolved_source TEXT DEFAULT 'analyst_override',
    resolved_by     TEXT,
    resolved_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (tenant_id, field_name)
)
"""

LEASE_RESOLUTION_DDL_SQLITE = (
    LEASE_RESOLUTION_DDL_PG
    .replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
    .replace('REFERENCES lease_tenants(id)', '')
)


def ensure_resolution_table(engine):
    """Create the lease_field_resolutions table if it doesn't exist."""
    from sqlalchemy import text
    ddl = (LEASE_RESOLUTION_DDL_SQLITE
           if engine.dialect.name == 'sqlite'
           else LEASE_RESOLUTION_DDL_PG)
    with engine.begin() as conn:
        conn.execute(text(ddl))


# Resolvable fields and which columns they map to on lease_tenants
RESOLVABLE_FIELDS = {
    'square_feet', 'annual_rent', 'monthly_rent', 'rent_per_sf',
    'lease_start', 'lease_end', 'security_deposit',
}


def resolve_field(
    engine,
    tenant_id: int,
    field_name: str,
    resolved_value: str,
    resolved_source: str = 'analyst_override',
    resolved_by: str = 'system',
) -> Dict[str, Any]:
    """Save an analyst's resolution for a specific field on a tenant.

    Uses UPSERT (INSERT ON CONFLICT UPDATE) to handle re-resolution.
    """
    from sqlalchemy import text

    if field_name not in RESOLVABLE_FIELDS:
        raise ValueError(f"Field '{field_name}' is not resolvable. "
                         f"Valid: {sorted(RESOLVABLE_FIELDS)}")

    ensure_resolution_table(engine)

    with engine.begin() as conn:
        if engine.dialect.name == 'postgresql':
            conn.execute(text("""
                INSERT INTO lease_field_resolutions
                    (tenant_id, field_name, resolved_value, resolved_source,
                     resolved_by, resolved_at)
                VALUES (:tid, :fn, :rv, :rs, :rb, CURRENT_TIMESTAMP)
                ON CONFLICT (tenant_id, field_name)
                DO UPDATE SET resolved_value = :rv, resolved_source = :rs,
                              resolved_by = :rb, resolved_at = CURRENT_TIMESTAMP
            """), {
                'tid': tenant_id, 'fn': field_name,
                'rv': resolved_value, 'rs': resolved_source, 'rb': resolved_by,
            })
        else:
            conn.execute(text("""
                INSERT OR REPLACE INTO lease_field_resolutions
                    (tenant_id, field_name, resolved_value, resolved_source,
                     resolved_by, resolved_at)
                VALUES (:tid, :fn, :rv, :rs, :rb, CURRENT_TIMESTAMP)
            """), {
                'tid': tenant_id, 'fn': field_name,
                'rv': resolved_value, 'rs': resolved_source, 'rb': resolved_by,
            })

    return {'tenant_id': tenant_id, 'field_name': field_name,
            'resolved_value': resolved_value, 'resolved_source': resolved_source}


def clear_resolution(engine, tenant_id: int, field_name: str) -> None:
    """Remove a field resolution, reverting to default data."""
    from sqlalchemy import text
    ensure_resolution_table(engine)
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM lease_field_resolutions
            WHERE tenant_id = :tid AND field_name = :fn
        """), {'tid': tenant_id, 'fn': field_name})


def get_resolved_tenants(engine, review_id: int) -> List[Dict[str, Any]]:
    """Get all tenants for a review with analyst resolutions applied.

    For each field, returns:
    - The analyst's resolved value if one exists
    - Otherwise the default from lease_tenants

    Each tenant dict includes a 'resolutions' sub-dict showing
    which fields have analyst overrides.
    """
    from sqlalchemy import text
    ensure_resolution_table(engine)

    with engine.connect() as conn:
        tenants = conn.execute(text("""
            SELECT id, tenant_name, suite, square_feet, lease_type,
                   lease_start, lease_end, term_months, monthly_rent,
                   annual_rent, rent_per_sf, security_deposit,
                   is_vacant, is_material, has_cotenancy, has_exclusive_use,
                   extraction_status, approval_status
            FROM lease_tenants
            WHERE review_id = :rid
            ORDER BY suite
        """), {'rid': review_id}).fetchall()

        # Load all resolutions for this review's tenants
        tenant_ids = [t[0] for t in tenants]
        resolutions = {}
        if tenant_ids:
            # Build parameterized IN clause
            placeholders = ', '.join(f':t{i}' for i in range(len(tenant_ids)))
            params = {f't{i}': tid for i, tid in enumerate(tenant_ids)}
            params['rid'] = review_id
            res_rows = conn.execute(text(f"""
                SELECT tenant_id, field_name, resolved_value, resolved_source,
                       resolved_by
                FROM lease_field_resolutions
                WHERE tenant_id IN ({placeholders})
            """), params).fetchall()
            for r in res_rows:
                resolutions.setdefault(r[0], {})[r[1]] = {
                    'value': r[2], 'source': r[3], 'by': r[4],
                }

    result = []
    for t in tenants:
        tid = t[0]
        tenant_res = resolutions.get(tid, {})

        def resolved(field_name, default_val, idx=None):
            """Return resolved value if exists, else default."""
            if field_name in tenant_res:
                raw = tenant_res[field_name]['value']
                # Try numeric conversion for numeric fields
                if field_name in ('square_feet', 'annual_rent', 'monthly_rent',
                                  'rent_per_sf', 'security_deposit'):
                    try:
                        return float(raw) if raw else default_val
                    except (ValueError, TypeError):
                        return default_val
                return raw
            return default_val

        row = {
            'id': tid,
            'tenant_name': t[1],
            'suite': t[2],
            'square_feet': resolved('square_feet', t[3]),
            'lease_type': t[4],
            'lease_start': resolved('lease_start', t[5]),
            'lease_end': resolved('lease_end', t[6]),
            'term_months': t[7],
            'monthly_rent': resolved('monthly_rent', t[8]),
            'annual_rent': resolved('annual_rent', t[9]),
            'rent_per_sf': resolved('rent_per_sf', t[10]),
            'security_deposit': resolved('security_deposit', t[11]),
            'is_vacant': bool(t[12]),
            'is_material': bool(t[13]),
            'has_cotenancy': bool(t[14]),
            'has_exclusive_use': bool(t[15]),
            'extraction_status': t[16],
            'approval_status': t[17] or 'pending',
            'resolutions': {
                fn: {'value': info['value'], 'source': info['source']}
                for fn, info in tenant_res.items()
            },
        }
        result.append(row)

    return result


def get_resolved_expiration_histogram(
    engine, review_id: int, years: int = 10,
) -> Dict[str, Any]:
    """Expiration histogram using analyst-resolved tenant data."""
    resolved = get_resolved_tenants(engine, review_id)
    occupied = [t for t in resolved if not t['is_vacant'] and t['lease_end']]

    total_gla = sum(t['square_feet'] or 0 for t in resolved if not t['is_vacant'])
    total_rent = sum(t['annual_rent'] or 0 for t in resolved if not t['is_vacant'])

    current_year = datetime.now().year
    end_year = current_year + years

    yearly = {}
    for yr in range(current_year, end_year + 1):
        yearly[yr] = {
            'year': yr, 'expiring_sf': 0, 'expiring_rent': 0,
            'pct_of_total_rent': 0, 'avg_rent_per_sf': 0, 'tenant_count': 0,
        }

    material_by_year = {}

    for t in occupied:
        try:
            lease_end = pd.to_datetime(t['lease_end'])
            exp_year = lease_end.year
        except Exception:
            continue

        if exp_year < current_year or exp_year > end_year:
            continue

        sf = t['square_feet'] or 0
        rent = t['annual_rent'] or 0

        yearly[exp_year]['expiring_sf'] += sf
        yearly[exp_year]['expiring_rent'] += rent
        yearly[exp_year]['tenant_count'] += 1

        if t['is_material']:
            material_by_year.setdefault(exp_year, []).append({
                'tenant_name': t['tenant_name'],
                'suite': t['suite'],
                'square_feet': sf,
                'annual_rent': rent,
                'rent_per_sf': t['rent_per_sf'] or 0,
                'lease_end': t['lease_end'],
                'has_cotenancy': t['has_cotenancy'],
                'cotenancy_implication': (
                    'Departure may trigger co-tenancy clauses in other leases'
                    if t['has_cotenancy'] else None
                ),
            })

    yearly_data = []
    for yr in range(current_year, end_year + 1):
        d = yearly[yr]
        if total_rent > 0:
            d['pct_of_total_rent'] = round(d['expiring_rent'] / total_rent * 100, 1)
        if d['expiring_sf'] > 0:
            d['avg_rent_per_sf'] = round(d['expiring_rent'] / d['expiring_sf'], 2)
        yearly_data.append(d)

    return {
        'yearly_data': yearly_data,
        'material_leases': material_by_year,
        'totals': {'total_gla': total_gla, 'total_annual_rent': total_rent},
    }


def get_resolved_cotenancy_matrix(engine, review_id: int) -> Dict[str, Any]:
    """Cotenancy matrix using analyst-resolved rent data for impact sizing."""
    from sqlalchemy import text

    # Get resolved tenants for rent overrides
    resolved = get_resolved_tenants(engine, review_id)
    rent_by_tid = {t['id']: t['annual_rent'] or 0 for t in resolved}

    with engine.connect() as conn:
        cotenancy = conn.execute(text("""
            SELECT c.id, c.tenant_id, t.tenant_name, t.suite,
                   c.clause_text, c.trigger_description, c.alt_rent_formula,
                   c.termination_right, c.cure_period_days,
                   c.sunset_provision, c.is_curable
            FROM lease_cotenancy c
            JOIN lease_tenants t ON t.id = c.tenant_id
            WHERE c.review_id = :rid
        """), {'rid': review_id}).fetchall()

        refs = conn.execute(text("""
            SELECT cr.cotenancy_id, cr.referenced_tenant_name
            FROM lease_cotenancy_refs cr
            JOIN lease_cotenancy c ON c.id = cr.cotenancy_id
            WHERE c.review_id = :rid
        """), {'rid': review_id}).fetchall()

    forward = {}
    cot_details = {}
    for c in cotenancy:
        tenant_name = c[2]
        resolved_rent = rent_by_tid.get(c[1], 0)
        forward[tenant_name] = []
        cot_details[tenant_name] = {
            'suite': c[3],
            'annual_rent': resolved_rent,
            'clause_summary': c[4][:200] if c[4] else '',
            'trigger': c[5],
            'alt_rent': c[6],
            'termination_right': c[7],
            'cure_days': c[8],
            'sunset': c[9],
            'is_curable': c[10],
        }

    cot_id_to_tenant = {c[0]: c[2] for c in cotenancy}
    for ref in refs:
        tenant_name = cot_id_to_tenant.get(ref[0])
        if tenant_name:
            forward[tenant_name].append(ref[1])

    reverse = {}
    for tenant, named_cotenants in forward.items():
        for cotenant in named_cotenants:
            reverse.setdefault(cotenant, [])
            detail = cot_details.get(tenant, {})
            reverse[cotenant].append({
                'dependent_tenant': tenant,
                'annual_rent': detail.get('annual_rent', 0),
                'alt_rent': detail.get('alt_rent', ''),
                'termination_right': detail.get('termination_right', False),
            })

    rent_at_risk = {}
    for cotenant, dependents in reverse.items():
        total_rent = sum(d.get('annual_rent', 0) or 0 for d in dependents)
        term_count = sum(1 for d in dependents if d.get('termination_right'))
        rent_at_risk[cotenant] = {
            'total_dependent_rent': total_rent,
            'dependent_count': len(dependents),
            'termination_eligible_count': term_count,
            'dependents': dependents,
        }

    return {
        'forward': forward, 'reverse': reverse,
        'rent_at_risk': rent_at_risk, 'details': cot_details,
    }


def get_resolved_scenario_analysis(engine, review_id: int) -> List[Dict]:
    """Scenario analysis using resolved rent data."""
    matrix = get_resolved_cotenancy_matrix(engine, review_id)

    scenarios = []
    for cotenant, risk in matrix['rent_at_risk'].items():
        if risk['dependent_count'] == 0:
            continue

        scenario = {
            'departing_tenant': cotenant,
            'dependent_count': risk['dependent_count'],
            'total_dependent_rent': risk['total_dependent_rent'],
            'termination_eligible': risk['termination_eligible_count'],
            'impacts': [],
        }

        for dep in risk['dependents']:
            detail = matrix['details'].get(dep['dependent_tenant'], {})
            scenario['impacts'].append({
                'tenant': dep['dependent_tenant'],
                'annual_rent': dep.get('annual_rent', 0),
                'alt_rent_formula': dep.get('alt_rent', ''),
                'can_terminate': dep.get('termination_right', False),
                'cure_days': detail.get('cure_days'),
                'sunset': detail.get('sunset'),
                'is_curable': detail.get('is_curable', True),
            })

        scenarios.append(scenario)

    scenarios.sort(key=lambda s: s['total_dependent_rent'], reverse=True)
    return scenarios


def get_risk_analysis_data(engine, review_id: int) -> Dict[str, Any]:
    """Get complete risk analysis data bundle for a review.

    Returns resolved tenants, validation results, expirations,
    cotenancy matrix, and scenario analysis — all using analyst-resolved data.
    """
    from sqlalchemy import text

    resolved = get_resolved_tenants(engine, review_id)
    expirations = get_resolved_expiration_histogram(engine, review_id)
    cotenancy = get_resolved_cotenancy_matrix(engine, review_id)
    scenarios = get_resolved_scenario_analysis(engine, review_id)

    # Get validation results
    with engine.connect() as conn:
        val_rows = conn.execute(text("""
            SELECT t.id, t.tenant_name, t.suite, v.field_name, v.source_type,
                   v.seller_value, v.lease_value, v.status, v.notes
            FROM lease_validation v
            JOIN lease_tenants t ON t.id = v.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name, v.source_type, v.field_name
        """), {'rid': review_id}).fetchall()

    validation = [{
        'tenant_id': r[0], 'tenant': r[1], 'suite': r[2],
        'field': r[3], 'source_type': r[4],
        'seller_value': r[5], 'lease_value': r[6],
        'status': r[7], 'notes': r[8],
    } for r in val_rows]

    # Build cotenancy clauses for display
    clauses = []
    if cotenancy.get('details'):
        for tenant_name, detail in cotenancy['details'].items():
            clauses.append({
                'tenant_name': tenant_name,
                **detail,
                'trigger_description': detail.get('trigger'),
                'alt_rent_formula': detail.get('alt_rent'),
                'cure_period_days': detail.get('cure_days'),
                'named_cotenants': cotenancy['forward'].get(tenant_name, []),
            })

    # Get exclusive use data
    with engine.connect() as conn:
        exc_rows = conn.execute(text("""
            SELECT t.tenant_name, t.suite, e.restriction_text, e.restricted_use
            FROM lease_exclusive_use e
            JOIN lease_tenants t ON t.id = e.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name
        """), {'rid': review_id}).fetchall()

    exclusive_use = [{
        'tenant_name': r[0], 'suite': r[1],
        'restriction_text': r[2], 'restricted_use': r[3],
    } for r in exc_rows]

    # Get options
    with engine.connect() as conn:
        opt_rows = conn.execute(text("""
            SELECT o.id, t.tenant_name, t.suite, o.option_type,
                   o.option_number, o.total_options,
                   o.option_start, o.option_end, o.term_years,
                   o.notice_days, o.notice_deadline, o.rent_terms,
                   o.auto_renewal, o.exercised, o.source_doc
            FROM lease_options o
            JOIN lease_tenants t ON t.id = o.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name, o.option_type, o.option_number
        """), {'rid': review_id}).fetchall()

    options = [{
        'id': r[0], 'tenant_name': r[1], 'suite': r[2], 'option_type': r[3],
        'option_number': r[4], 'total_options': r[5],
        'option_start': r[6], 'option_end': r[7], 'term_years': r[8],
        'notice_days': r[9], 'notice_deadline': r[10], 'rent_terms': r[11],
        'auto_renewal': bool(r[12]), 'exercised': bool(r[13]),
        'source_doc': r[14],
    } for r in opt_rows]

    # Get documents grouped by tenant_id for linking
    with engine.connect() as conn:
        doc_rows = conn.execute(text("""
            SELECT d.id, d.tenant_id, d.filename, d.doc_type, d.doc_date,
                   CASE WHEN d.file_data IS NOT NULL THEN 1 ELSE 0 END as has_file
            FROM lease_documents d
            JOIN lease_tenants t ON t.id = d.tenant_id
            WHERE t.review_id = :rid
            ORDER BY d.tenant_id, d.doc_type, d.doc_date
        """), {'rid': review_id}).fetchall()

    documents = {}
    for r in doc_rows:
        tid = r[1]
        if tid not in documents:
            documents[tid] = []
        documents[tid].append({
            'id': r[0], 'filename': r[2], 'doc_type': r[3],
            'doc_date': r[4], 'has_file': bool(r[5]),
        })

    return {
        'tenants': resolved,
        'validation': validation,
        'expirations': expirations,
        'cotenancy': {**cotenancy, 'clauses': clauses},
        'scenarios': scenarios,
        'exclusive_use': exclusive_use,
        'options': options,
        'documents': documents,
    }


# ---------------------------------------------------------------------------
# Lease Abstract — assemble & persist per-tenant abstracts
# ---------------------------------------------------------------------------

# Standard abstract sections in display order (matches Word template)
ABSTRACT_SECTIONS = [
    ('tenant', 'Tenant', 1),
    ('documents_reviewed', 'Documents Reviewed', 2),
    ('unit_address', 'Unit/Address', 3),
    ('square_feet', 'Square Feet', 4),
    ('prs', 'PRS (Proportionate Share)', 5),
    ('term', 'Term', 6),
    ('percentage_rent', 'Percentage Rent', 7),
    ('renewal_options', 'Renewal Options', 8),
    ('termination_options', 'Termination Options', 9),
    ('use', 'Use', 10),
    ('exclusive_use', 'Exclusive Use', 11),
    ('cam', 'CAM', 12),
    ('insurance', 'Insurance', 13),
    ('real_estate_taxes', 'Real Estate Taxes', 14),
    ('parking', 'Parking', 15),
    ('roof', 'Roof', 16),
    ('structural', 'Structural', 17),
    ('signage', 'Signage', 18),
    ('utilities', 'Utilities', 19),
    ('hvac', 'HVAC', 20),
    ('sales_reporting', 'Sales Reporting', 21),
    ('estoppel', 'Estoppel', 22),
    ('security_deposit', 'Security Deposit', 23),
    ('operating_covenants', 'Operating Covenants', 24),
    ('go_dark_termination', 'Go-Dark Termination', 25),
    ('business_termination', 'Landlord/Tenant Business Termination Rights', 26),
    ('casualty_termination', 'Landlord/Tenant Casualty/Default/Eminent Domain Termination Rights', 27),
    ('relocation_rights', 'Landlord/Tenant Relocation Rights', 28),
    ('rofr_rofo', 'ROFR/ROFO Options', 29),
    ('cotenancy', 'Co-Tenancy Provisions', 30),
    ('radius_restriction', 'Radius Restriction', 31),
    ('ti_allowance', 'Tenant Improvement Allowance', 32),
    ('lease_inducements', 'Lease Inducements', 33),
    ('assignment', 'Assignment and Subletting', 34),
    ('merchants_association', "Merchants' Association", 35),
    ('redevelopment', 'Redevelopment', 36),
    ('guarantor', 'Guarantor(s)', 37),
    ('notes', 'Notes', 38),
    ('date_of_review', 'Date of Review', 39),
]


def _fmt_money(val) -> str:
    """Format a number as currency."""
    if val is None:
        return ''
    try:
        v = float(val)
        if v == int(v):
            return f"${int(v):,}"
        return f"${v:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_date(val) -> str:
    """Format YYYY-MM-DD to M/D/YYYY."""
    if not val:
        return ''
    try:
        m = re.match(r'(\d{4})-(\d{2})-(\d{2})', str(val))
        if m:
            return f"{int(m.group(2))}/{int(m.group(3))}/{m.group(1)}"
    except Exception:
        pass
    return str(val)


def _assemble_abstract_from_data(
    engine, tenant_id: int, review_id: int
) -> Dict[str, Dict[str, str]]:
    """Build abstract section content from existing extracted data.

    Returns {section_key: {'content': str, 'lease_ref': str}}.
    Sections with no data return empty content (user can fill in manually).
    """
    from sqlalchemy import text

    sections: Dict[str, Dict[str, str]] = {}

    with engine.connect() as conn:
        # Tenant info
        t = conn.execute(text("""
            SELECT tenant_name, suite, square_feet, lease_type,
                   lease_start, lease_end, term_months,
                   monthly_rent, annual_rent, rent_per_sf,
                   security_deposit, extraction_json
            FROM lease_tenants WHERE id = :tid
        """), {'tid': tenant_id}).fetchone()
        if not t:
            return sections

        tenant_name = t[0] or ''
        suite = t[1] or ''
        sq_ft = t[2]
        lease_start = t[4]
        lease_end = t[5]
        term_months = t[6]
        monthly_rent = t[7]
        annual_rent = t[8]
        rent_per_sf = t[9]
        security_deposit = t[10]
        extraction_raw = t[11]

        # Parse extraction JSON (may be a list of extractions from multiple docs)
        ext = {}
        if extraction_raw:
            try:
                parsed = json.loads(extraction_raw)
                if isinstance(parsed, list) and parsed:
                    ext = parsed[0] if isinstance(parsed[0], dict) else {}
                elif isinstance(parsed, dict):
                    ext = parsed
            except (json.JSONDecodeError, IndexError):
                pass

        # Documents reviewed
        docs = conn.execute(text("""
            SELECT filename, doc_type, doc_date
            FROM lease_documents
            WHERE tenant_id = :tid
            ORDER BY doc_date, filename
        """), {'tid': tenant_id}).fetchall()

        doc_lines = []
        for d in docs:
            line = d[1] or d[0]
            if d[2]:
                line += f" (dated {_fmt_date(d[2])})"
            elif d[0] != d[1]:
                line += f" — {d[0]}"
            doc_lines.append(line)

        # Rent steps
        rent_steps = conn.execute(text("""
            SELECT effective_date, monthly_rent, annual_rent, rent_per_sf
            FROM lease_rent_steps WHERE tenant_id = :tid
            ORDER BY effective_date
        """), {'tid': tenant_id}).fetchall()

        # Renewal options
        renewals = conn.execute(text("""
            SELECT option_number, total_options, term_years,
                   notice_days, notice_deadline, rent_terms,
                   auto_renewal, exercised, option_start, option_end
            FROM lease_options
            WHERE tenant_id = :tid AND option_type = 'renewal'
            ORDER BY option_number
        """), {'tid': tenant_id}).fetchall()

        # Termination options
        terminations = conn.execute(text("""
            SELECT option_number, total_options, term_years,
                   notice_days, notice_deadline, rent_terms,
                   exercised, option_start, option_end
            FROM lease_options
            WHERE tenant_id = :tid AND option_type = 'termination'
            ORDER BY option_number
        """), {'tid': tenant_id}).fetchall()

        # Cotenancy
        cot_rows = conn.execute(text("""
            SELECT c.trigger_description, c.trigger_threshold,
                   c.cure_period_days, c.alt_rent_formula,
                   c.termination_right, c.termination_notice_days,
                   c.sunset_provision
            FROM lease_cotenancy c
            WHERE c.tenant_id = :tid
        """), {'tid': tenant_id}).fetchall()

        cot_refs = conn.execute(text("""
            SELECT cr.referenced_tenant_name
            FROM lease_cotenancy_refs cr
            JOIN lease_cotenancy c ON c.id = cr.cotenancy_id
            WHERE c.tenant_id = :tid
        """), {'tid': tenant_id}).fetchall()

        # Exclusive use
        exc_rows = conn.execute(text("""
            SELECT restriction_text, restricted_use
            FROM lease_exclusive_use WHERE tenant_id = :tid
        """), {'tid': tenant_id}).fetchall()

    # -- Build section content --

    # Tenant
    sections['tenant'] = {
        'content': ext.get('tenant_legal_name') or tenant_name,
        'lease_ref': '',
    }

    # Documents Reviewed
    sections['documents_reviewed'] = {
        'content': '\n'.join(doc_lines) if doc_lines else '',
        'lease_ref': '',
    }

    # Unit/Address
    sections['unit_address'] = {
        'content': suite,
        'lease_ref': '',
    }

    # Square Feet
    sections['square_feet'] = {
        'content': f"{int(sq_ft):,} Square Feet" if sq_ft else '',
        'lease_ref': '',
    }

    # PRS
    sections['prs'] = {'content': '', 'lease_ref': ''}

    # Term
    term_parts = []
    if term_months:
        years = term_months // 12
        months = term_months % 12
        if years and months:
            term_parts.append(f"{years} year(s), {months} month(s)")
        elif years:
            term_parts.append(f"{years} year(s)")
        else:
            term_parts.append(f"{months} month(s)")
    if lease_start:
        term_parts.append(f"Commencing {_fmt_date(lease_start)}")
    if lease_end:
        term_parts.append(f"Expiring {_fmt_date(lease_end)}")
    # Rent schedule
    if rent_steps:
        term_parts.append('')
        term_parts.append('Rent Schedule:')
        for idx, rs in enumerate(rent_steps):
            date_label = _fmt_date(rs[0])
            if not date_label:
                date_label = f"Step {idx + 1}"
            line = f"  {date_label}: {_fmt_money(rs[1])}/mo"
            if rs[2]:
                line += f" ({_fmt_money(rs[2])}/yr)"
            if rs[3]:
                line += f" — {_fmt_money(rs[3])}/SF"
            term_parts.append(line)
    elif monthly_rent or annual_rent:
        term_parts.append(
            f"Base Rent: {_fmt_money(monthly_rent)}/mo "
            f"({_fmt_money(annual_rent)}/yr)"
        )
        if rent_per_sf:
            term_parts.append(f"Rent/SF: {_fmt_money(rent_per_sf)}")

    sections['term'] = {
        'content': '\n'.join(term_parts),
        'lease_ref': '',
    }

    # Percentage Rent
    pct = ext.get('percentage_rent', {})
    if pct and pct.get('has_clause'):
        pct_text = f"Rate: {pct.get('rate_pct')}%"
        if pct.get('breakpoint'):
            pct_text += f", Breakpoint: {_fmt_money(pct['breakpoint'])}"
        sections['percentage_rent'] = {'content': pct_text, 'lease_ref': ''}
    else:
        sections['percentage_rent'] = {
            'content': 'No language noted.' if ext else '',
            'lease_ref': '',
        }

    # Renewal Options
    if renewals:
        lines = []
        for r in renewals:
            line = f"Option {r[0]} of {r[1]}: {r[2]} year(s)"
            if r[3]:
                line += f", {r[3]} days notice"
            if r[5]:
                line += f" at {r[5]}"
            if r[6]:
                line += ' (auto-renewal)'
            if r[7]:
                line += ' [EXERCISED]'
            if r[8] or r[9]:
                line += f" ({_fmt_date(r[8])} — {_fmt_date(r[9])})"
            lines.append(line)
        sections['renewal_options'] = {
            'content': '\n'.join(lines), 'lease_ref': '',
        }
    else:
        sections['renewal_options'] = {
            'content': 'No language noted.' if ext else '',
            'lease_ref': '',
        }

    # Termination Options
    if terminations:
        lines = []
        for r in terminations:
            line = f"Option {r[0]} of {r[1]}"
            if r[3]:
                line += f", {r[3]} days notice"
            if r[5]:
                line += f" — {r[5]}"
            if r[6]:
                line += ' [EXERCISED]'
            if r[7]:
                line += f", earliest: {_fmt_date(r[7])}"
            lines.append(line)
        sections['termination_options'] = {
            'content': '\n'.join(lines), 'lease_ref': '',
        }
    else:
        sections['termination_options'] = {
            'content': 'No language noted.' if ext else '',
            'lease_ref': '',
        }

    # Use
    sections['use'] = {
        'content': ext.get('permitted_use') or '',
        'lease_ref': '',
    }

    # Exclusive Use
    if exc_rows:
        lines = [f"{r[1]}: {r[0]}" if r[1] else r[0] for r in exc_rows]
        sections['exclusive_use'] = {
            'content': '\n'.join(lines), 'lease_ref': '',
        }
    else:
        sections['exclusive_use'] = {
            'content': 'No language noted.' if ext else '',
            'lease_ref': '',
        }

    # CAM
    cam_parts = []
    if ext.get('cam_structure'):
        cam_parts.append(f"Structure: {ext['cam_structure']}")
    if ext.get('cam_cap_pct'):
        cam_parts.append(f"Cap: {ext['cam_cap_pct']}%")
    if ext.get('admin_fee_pct'):
        cam_parts.append(f"Admin Fee: {ext['admin_fee_pct']}%")
    sections['cam'] = {
        'content': ', '.join(cam_parts) if cam_parts else (
            'No language noted.' if ext else ''
        ),
        'lease_ref': '',
    }

    # Insurance
    ins_parts = []
    if ext.get('insurance_pass_through') is True:
        ins_parts.append('Insurance pass-through to tenant.')
    elif ext.get('insurance_pass_through') is False:
        ins_parts.append('No insurance pass-through.')
    sections['insurance'] = {
        'content': ' '.join(ins_parts) if ins_parts else (
            'No language noted.' if ext else ''
        ),
        'lease_ref': '',
    }

    # Real Estate Taxes
    tax_parts = []
    if ext.get('tax_pass_through') is True:
        tax_parts.append('Real estate tax pass-through to tenant.')
    elif ext.get('tax_pass_through') is False:
        tax_parts.append('No real estate tax pass-through.')
    sections['real_estate_taxes'] = {
        'content': ' '.join(tax_parts) if tax_parts else (
            'No language noted.' if ext else ''
        ),
        'lease_ref': '',
    }

    # Sections that are typically blank until manually populated
    for key in [
        'parking', 'roof', 'structural', 'signage', 'utilities',
        'hvac', 'sales_reporting', 'estoppel', 'operating_covenants',
        'business_termination', 'casualty_termination',
        'relocation_rights', 'rofr_rofo', 'radius_restriction',
        'lease_inducements', 'merchants_association', 'redevelopment',
        'guarantor', 'notes',
    ]:
        sections[key] = {'content': '', 'lease_ref': ''}

    # Security Deposit
    sections['security_deposit'] = {
        'content': _fmt_money(security_deposit) if security_deposit else (
            'No language noted.' if ext else ''
        ),
        'lease_ref': '',
    }

    # Go-Dark Termination
    sections['go_dark_termination'] = {
        'content': ext.get('go_dark_provision') or (
            'No language noted.' if ext else ''
        ),
        'lease_ref': '',
    }

    # Co-Tenancy
    if cot_rows:
        lines = []
        ref_names = [r[0] for r in cot_refs]
        if ref_names:
            lines.append(f"Named Co-Tenants: {', '.join(ref_names)}")
        for c in cot_rows:
            if c[0]:
                lines.append(f"Trigger: {c[0]}")
            if c[2]:
                lines.append(f"Cure Period: {c[2]} days")
            if c[3]:
                lines.append(f"Alt Rent: {c[3]}")
            if c[4]:
                lines.append(
                    f"Termination Right: Yes"
                    + (f" ({c[5]} days notice)" if c[5] else '')
                )
            if c[6]:
                lines.append(f"Sunset: {c[6]}")
        sections['cotenancy'] = {
            'content': '\n'.join(lines), 'lease_ref': '',
        }
    else:
        sections['cotenancy'] = {
            'content': 'No language noted.' if ext else '',
            'lease_ref': '',
        }

    # TI Allowance
    sections['ti_allowance'] = {
        'content': _fmt_money(ext.get('ti_allowance')) if ext.get('ti_allowance') else (
            'No language noted.' if ext else ''
        ),
        'lease_ref': '',
    }

    # Assignment
    asgn = ext.get('assignment', {})
    if asgn:
        parts = []
        if asgn.get('consent_required') is True:
            parts.append('Landlord consent required.')
        elif asgn.get('consent_required') is False:
            parts.append('No landlord consent required.')
        if asgn.get('tenant_released') is True:
            parts.append('Tenant released upon assignment.')
        elif asgn.get('tenant_released') is False:
            parts.append('Tenant NOT released upon assignment.')
        sections['assignment'] = {
            'content': ' '.join(parts) if parts else 'No language noted.',
            'lease_ref': '',
        }
    else:
        sections['assignment'] = {
            'content': 'No language noted.' if ext else '',
            'lease_ref': '',
        }

    # Date of Review
    sections['date_of_review'] = {
        'content': datetime.now().strftime('%m/%d/%Y'),
        'lease_ref': '',
    }

    return sections


def get_tenant_abstract(
    engine, review_id: int, tenant_id: int
) -> Dict[str, Any]:
    """Get the abstract for a tenant, assembling from data if needed.

    Returns {tenant_name, suite, review_name, property_name, sections: [...]}.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Tenant + review info
        info = conn.execute(text("""
            SELECT t.tenant_name, t.suite, r.property_name,
                   t.review_id
            FROM lease_tenants t
            JOIN lease_reviews r ON r.id = t.review_id
            WHERE t.id = :tid AND t.review_id = :rid
        """), {'tid': tenant_id, 'rid': review_id}).fetchone()
        if not info:
            return {'error': 'Tenant not found'}

        tenant_name = info[0]
        suite = info[1]
        property_name = info[2]

        # Check for saved abstract sections
        saved = conn.execute(text("""
            SELECT section_key, section_title, content, lease_ref, sort_order
            FROM lease_abstract_sections
            WHERE tenant_id = :tid
            ORDER BY sort_order
        """), {'tid': tenant_id}).fetchall()

    # If we have saved sections, use them (merging with template for any missing)
    saved_map = {r[0]: {
        'section_key': r[0], 'section_title': r[1],
        'content': r[2] or '', 'lease_ref': r[3] or '',
        'sort_order': r[4],
    } for r in saved}

    # Assemble from data for any section not yet saved
    if not saved_map:
        assembled = _assemble_abstract_from_data(engine, tenant_id, review_id)
    else:
        assembled = {}

    # Build final section list in template order
    result_sections = []
    for key, title, order in ABSTRACT_SECTIONS:
        if key in saved_map:
            result_sections.append(saved_map[key])
        elif key in assembled:
            result_sections.append({
                'section_key': key,
                'section_title': title,
                'content': assembled[key].get('content', ''),
                'lease_ref': assembled[key].get('lease_ref', ''),
                'sort_order': order,
            })
        else:
            result_sections.append({
                'section_key': key,
                'section_title': title,
                'content': '',
                'lease_ref': '',
                'sort_order': order,
            })

    return {
        'tenant_name': tenant_name,
        'suite': suite,
        'property_name': property_name,
        'tenant_id': tenant_id,
        'review_id': review_id,
        'sections': result_sections,
    }


def save_abstract_sections(
    engine, tenant_id: int, sections: List[Dict], username: str = ''
) -> None:
    """Save (upsert) abstract sections for a tenant."""
    from sqlalchemy import text

    with engine.begin() as conn:
        for s in sections:
            key = s.get('section_key')
            if not key:
                continue
            title = s.get('section_title', key)
            content = s.get('content', '')
            lease_ref = s.get('lease_ref', '')
            sort_order = s.get('sort_order', 0)

            if engine.dialect.name == 'postgresql':
                conn.execute(text("""
                    INSERT INTO lease_abstract_sections
                        (tenant_id, section_key, section_title, content,
                         lease_ref, sort_order, updated_by, updated_at)
                    VALUES (:tid, :sk, :st, :c, :lr, :so, :ub, CURRENT_TIMESTAMP)
                    ON CONFLICT (tenant_id, section_key)
                    DO UPDATE SET section_title = :st, content = :c,
                        lease_ref = :lr, sort_order = :so,
                        updated_by = :ub, updated_at = CURRENT_TIMESTAMP
                """), {
                    'tid': tenant_id, 'sk': key, 'st': title,
                    'c': content, 'lr': lease_ref, 'so': sort_order,
                    'ub': username,
                })
            else:
                conn.execute(text("""
                    INSERT OR REPLACE INTO lease_abstract_sections
                        (tenant_id, section_key, section_title, content,
                         lease_ref, sort_order, updated_by, updated_at)
                    VALUES (:tid, :sk, :st, :c, :lr, :so, :ub, CURRENT_TIMESTAMP)
                """), {
                    'tid': tenant_id, 'sk': key, 'st': title,
                    'c': content, 'lr': lease_ref, 'so': sort_order,
                    'ub': username,
                })


def get_review_abstracts_list(engine, review_id: int) -> List[Dict]:
    """Get a list of tenants and whether they have abstract data."""
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT t.id, t.tenant_name, t.suite, t.is_vacant,
                   (SELECT COUNT(*) FROM lease_abstract_sections a
                    WHERE a.tenant_id = t.id) as section_count
            FROM lease_tenants t
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name
        """), {'rid': review_id}).fetchall()

    return [{
        'tenant_id': r[0], 'tenant_name': r[1], 'suite': r[2],
        'is_vacant': bool(r[3]),
        'has_abstract': r[4] > 0,
        'section_count': r[4],
    } for r in rows]
