"""
lease_review_service.py
Lease review service for due diligence — extraction, validation, and analysis.

Supports multi-property portfolio reviews. Data stored in PostgreSQL/SQLite
via the standard database layer.
"""

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
        annual_rent     DOUBLE PRECISION,
        rent_per_sf     DOUBLE PRECISION,
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
    annual_rent, rent_per_sf_year, security_deposit, is_vacant
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

        rows.append({
            'property_code': str(col1).strip(),
            'suite': str(col2).strip() if col2 else '',
            'tenant_name': tenant_name.strip(),
            'lease_type': str(ws.cell(r, 4).value or '').strip(),
            'square_feet': to_float(ws.cell(r, 5).value),
            'lease_start': to_date_str(ws.cell(r, 6).value),
            'lease_end': to_date_str(ws.cell(r, 7).value),
            'term_months': int(to_float(ws.cell(r, 8).value)),
            'monthly_rent': to_float(ws.cell(r, 9).value),
            'rent_per_sf_month': to_float(ws.cell(r, 10).value),
            'annual_rent': to_float(ws.cell(r, 11).value),
            'rent_per_sf_year': to_float(ws.cell(r, 12).value),
            'security_deposit': to_float(ws.cell(r, 13).value),
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

        # Derive missing fields
        if ann_rent > 0 and mon_rent == 0:
            mon_rent = ann_rent / 12
        elif mon_rent > 0 and ann_rent == 0:
            ann_rent = mon_rent * 12

        rent_per_sf = ann_rent / sf if sf > 0 else 0

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
            'annual_rent': ann_rent,
            'rent_per_sf_year': rent_per_sf,
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
                     annual_rent, rent_per_sf, security_deposit,
                     is_vacant, is_material, has_cotenancy, has_exclusive_use)
                VALUES (:rid, :tn, :su, :sf, :lt,
                        :ls, :le, :tm, :mr,
                        :ar, :rpsf, :sd,
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
                'ar': ann_rent,
                'rpsf': float(row.get('rent_per_sf_year', 0) or 0),
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
      "notice_days": number,
      "notice_deadline": "YYYY-MM-DD or null",
      "auto_renewal": true/false,
      "rent_terms": "fair market / fixed increase / CPI"
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
# Extract lease terms via Claude API (batch)
# ---------------------------------------------------------------------------

def extract_all_documents(engine, review_id: int, api_key: Optional[str] = None):
    """Extract text from all PDFs and run Claude extraction for key documents.

    Prioritizes Original Lease and Amendment documents.
    """
    from sqlalchemy import text as sql_text

    with engine.connect() as conn:
        # Get all documents for this review
        docs = conn.execute(sql_text("""
            SELECT d.id, d.tenant_id, d.filename, d.file_path,
                   d.doc_type, d.doc_date,
                   t.tenant_name, t.suite
            FROM lease_documents d
            JOIN lease_tenants t ON t.id = d.tenant_id
            WHERE d.review_id = :rid
            AND d.extraction_status = 'pending'
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

            try:
                # Step 1: Extract PDF text
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

                        # Store rent steps
                        if terms.get('rent_steps'):
                            for step in terms['rent_steps']:
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

                        # Store cotenancy from extraction
                        cot = terms.get('cotenancy', {})
                        if cot and cot.get('has_clause'):
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

                        # Store renewal options
                        for opt in (terms.get('renewal_options') or []):
                            conn.execute(sql_text("""
                                INSERT INTO lease_options
                                    (tenant_id, option_type, option_number,
                                     total_options, term_years, notice_days,
                                     notice_deadline, rent_terms,
                                     auto_renewal, source_doc)
                                VALUES (:tid, 'renewal', :on, :to, :ty,
                                        :nd, :ndl, :rt, :ar, :sd)
                            """), {
                                'tid': tenant_id,
                                'on': opt.get('option_number'),
                                'to': opt.get('total_options'),
                                'ty': opt.get('term_years'),
                                'nd': opt.get('notice_days'),
                                'ndl': opt.get('notice_deadline'),
                                'rt': opt.get('rent_terms'),
                                'ar': opt.get('auto_renewal', False),
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
                   o.option_number, o.total_options, o.term_years,
                   o.notice_days, o.notice_deadline, o.rent_terms,
                   o.auto_renewal, o.source_doc
            FROM lease_options o
            JOIN lease_tenants t ON t.id = o.tenant_id
            WHERE t.review_id = :rid
            ORDER BY t.tenant_name, o.option_number
        """), {'rid': review_id}).fetchall()

    opt_headers = ['Tenant', 'Suite', 'Type', 'Option #', 'Total Options',
                   'Term (Yrs)', 'Notice (Days)', 'Notice Deadline',
                   'Rent Terms', 'Auto-Renew', 'Source']
    for c, h in enumerate(opt_headers, 1):
        cell = ws6.cell(3, c, h)
        cell.font = header_font_white
        cell.fill = header_fill
    for i, o in enumerate(opt_rows):
        row = 4 + i
        for c, val in enumerate(o, 1):
            cell = ws6.cell(row, c, val)
            if c == 9:  # rent_terms
                cell.alignment = Alignment(wrap_text=True)

    opt_widths = [30, 12, 10, 10, 10, 10, 12, 16, 40, 10, 30]
    for c, w in enumerate(opt_widths, 1):
        ws6.column_dimensions[get_column_letter(c)].width = w

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
