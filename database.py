"""
database.py
SQLite database management for waterfall model

Provides:
- Database initialization from CSVs
- Connection management
- Table refresh from CSV
- CSV export from tables
- Schema management
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import io
import zipfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "waterfall.db"

# SQLAlchemy engine — set by Flask app when DATABASE_URL is configured
_sa_engine = None


def set_engine(engine):
    """Set the SQLAlchemy engine for PostgreSQL support."""
    global _sa_engine
    _sa_engine = engine

# Table definitions with their CSV sources
TABLE_DEFINITIONS = {
    'deals': {
        'csv': 'investment_map.csv',
        'description': 'Investment properties and deal information',
        'key_columns': ['vcode', 'InvestmentID']
    },
    'forecasts': {
        'csv': 'forecast_feed.csv',
        'description': 'Monthly operating forecasts',
        'key_columns': ['vcode', 'event_date', 'vAccount']
    },
    'waterfalls': {
        'csv': 'waterfalls.csv',
        'description': 'Waterfall distribution logic',
        'key_columns': ['vcode', 'vmisc', 'iOrder']
    },
    'accounting': {
        'csv': 'accounting_feed.csv',
        'description': 'Historical accounting transactions',
        'key_columns': ['InvestmentID', 'InvestorID', 'EffectiveDate']
    },
    'coa': {
        'csv': 'coa.csv',
        'description': 'Chart of accounts',
        'key_columns': ['vcode']
    },
    'relationships': {
        'csv': 'MRI_IA_Relationship.csv',
        'description': 'Ownership relationships and structures',
        'key_columns': ['InvestmentID', 'InvestorID']
    },
    'loans': {
        'csv': 'MRI_Loans.csv',
        'description': 'Existing loan structures',
        'key_columns': ['vCode', 'LoanID']
    },
    'valuations': {
        'csv': 'MRI_Val.csv',
        'description': 'Property valuations and cap rates',
        'key_columns': ['vcode', 'dtVal']
    },
    'planned_loans': {
        'csv': 'MRI_Supp.csv',
        'description': 'Planned second mortgage parameters',
        'key_columns': ['vCode', 'Orig_Date']
    },
    'capital_calls': {
        'csv': 'MRI_Capital_Calls.csv',
        'description': 'Planned capital calls at entity level',
        'key_columns': ['EntityID', 'CallDate']
    },
    'commitments': {
        'csv': 'MRI_Commitments.csv',
        'description': 'Investor commitments to entities',
        'key_columns': ['CommitmentUID', 'EntityID', 'InvestorID']
    },
    'fund_deals': {
        'csv': 'fund_deals.csv',
        'description': 'Fund to deal mappings',
        'key_columns': ['FundID', 'vcode']
    },
    'investor_waterfalls': {
        'csv': 'investor_waterfalls.csv',
        'description': 'LP/GP waterfall definitions',
        'key_columns': ['FundID', 'iOrder']
    },
    'investor_accounting': {
        'csv': 'investor_accounting.csv',
        'description': 'LP/GP historical transactions',
        'key_columns': ['FundID', 'InvestorID', 'EffectiveDate']
    },
    'occupancy': {
        'csv': 'MRI_Occupancy_Download.csv',
        'description': 'Quarterly occupancy data by property',
        'key_columns': ['vCode', 'Qtr']
    },
    'isbs': {
        'csv': 'ISBS_Download.csv',
        'description': 'Income statement and balance sheet data (legacy monolithic)',
        'key_columns': ['vcode', 'dtEntry', 'vSource', 'vAccount']
    },
    'isbs_interim_is': {
        'csv': 'ISBS_Interim_IS.csv',
        'description': 'ISBS Actuals — YTD cumulative trial balance (2025+)',
        'key_columns': ['vcode', 'dtEntry', 'vAccount']
    },
    'isbs_interim_is_historical': {
        'csv': 'ISBS_Interim_IS_Historical.csv',
        'description': 'ISBS Actuals — YTD cumulative trial balance (pre-2025)',
        'key_columns': ['vcode', 'dtEntry', 'vAccount']
    },
    'isbs_interim_bs': {
        'csv': 'ISBS_Interim_BS.csv',
        'description': 'ISBS Balance Sheet — current outstanding balances',
        'key_columns': ['vcode', 'dtEntry', 'vAccount']
    },
    'isbs_budget_is': {
        'csv': 'ISBS_Budget_IS.csv',
        'description': 'ISBS Budget — periodic monthly amounts',
        'key_columns': ['vcode', 'dtEntry', 'vAccount']
    },
    'isbs_projected_is': {
        'csv': 'ISBS_Projected_IS.csv',
        'description': 'ISBS Underwriting — YTD cumulative trial balance',
        'key_columns': ['vcode', 'dtEntry', 'vAccount']
    },
    'isbs_valuation_is': {
        'csv': 'ISBS_Valuation_IS.csv',
        'description': 'ISBS Valuation — periodic monthly from forecast_feed',
        'key_columns': ['vcode', 'dtEntry', 'vAccount']
    },
    'one_pager_comments': {
        'csv': 'OnePager_Comments.csv',
        'description': 'One Pager report comments by deal and period',
        'key_columns': ['vcode', 'reporting_period']
    },
    'tenants': {
        'csv': 'Tenant_Report.csv',
        'description': 'Commercial tenant roster by property',
        'key_columns': ['Code', 'Tenant Code']
    }
}


def get_db_connection():
    """
    Get database connection with optimizations.

    Returns SQLAlchemy connection if engine is set (PostgreSQL),
    otherwise falls back to sqlite3.
    """
    if _sa_engine is not None:
        return _sa_engine.connect()

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries

    # Performance optimizations
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
    conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
    conn.execute("PRAGMA cache_size=10000")  # Larger cache
    conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory

    return conn


def _exec(conn, sql, params=None):
    """Execute SQL on either sqlite3 or SQLAlchemy connection.

    Converts ``?`` positional placeholders to ``:p0, :p1, …`` named params
    when running against SQLAlchemy (PostgreSQL).  sqlite3 connections are
    passed through unchanged.
    """
    if _sa_engine is not None:
        from sqlalchemy import text
        if params:
            named_sql = []
            param_dict = {}
            idx = 0
            for ch in sql:
                if ch == '?':
                    name = f"p{idx}"
                    named_sql.append(f":{name}")
                    param_dict[name] = params[idx] if isinstance(params, (list, tuple)) else params
                    idx += 1
                else:
                    named_sql.append(ch)
            return conn.execute(text("".join(named_sql)), param_dict)
        return conn.execute(text(sql))
    else:
        if params:
            return conn.execute(sql, params)
        return conn.execute(sql)


def _pg_fix_column_types(conn, text, table, col_types):
    """ALTER columns to correct types if they are currently TEXT.

    Safe to run repeatedly — only alters columns whose current type is 'text'.
    Uses ALTER COLUMN ... TYPE ... USING to cast existing data.
    """
    for col, target_type in col_types.items():
        try:
            cur_type = conn.execute(text(
                "SELECT data_type FROM information_schema.columns "
                f"WHERE table_name = '{table}' AND column_name = '{col}'"
            )).scalar()
            if cur_type and cur_type.lower() == "text":
                conn.execute(text(
                    f'ALTER TABLE "{table}" ALTER COLUMN "{col}" '
                    f"TYPE {target_type} USING NULLIF(\"{col}\", '')::{ target_type}"
                ))
                logger.info(f"Altered {table}.{col} from TEXT to {target_type}")
        except Exception as e:
            logger.warning(f"Could not alter {table}.{col}: {e}")


def ensure_pg_tables(engine):
    """Ensure app-managed tables exist on PostgreSQL with correct schema.

    Called once at startup when DATABASE_URL is set.  Uses SERIAL for
    auto-increment columns (SQLite AUTOINCREMENT is not valid on PG).
    Also backfills any rows that have NULL id from prior inserts.
    """
    from sqlalchemy import text

    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS prospective_loans (
            id              SERIAL PRIMARY KEY,
            vcode           TEXT NOT NULL,
            loan_name       TEXT,
            status          TEXT DEFAULT 'draft',
            refi_date       TEXT NOT NULL,
            existing_loan_id TEXT,
            loan_amount     DOUBLE PRECISION,
            lender_uw_noi   DOUBLE PRECISION,
            max_ltv         DOUBLE PRECISION,
            min_dscr        DOUBLE PRECISION,
            min_debt_yield  DOUBLE PRECISION,
            interest_rate   DOUBLE PRECISION,
            rate_spread_bps BIGINT,
            rate_index      TEXT,
            term_years      BIGINT,
            amort_years     BIGINT,
            io_years        DOUBLE PRECISION,
            int_type        TEXT DEFAULT 'Fixed',
            closing_costs   DOUBLE PRECISION DEFAULT 0,
            reserve_holdback DOUBLE PRECISION DEFAULT 0,
            notes           TEXT,
            created_by      TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by      TEXT,
            updated_at      TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prospective_loans_audit (
            id          SERIAL PRIMARY KEY,
            loan_id     BIGINT NOT NULL,
            action      TEXT NOT NULL,
            vcode       TEXT,
            loan_name   TEXT,
            status      TEXT,
            loan_amount DOUBLE PRECISION,
            all_fields  TEXT,
            changed_by  TEXT,
            changed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS waterfall_audit (
            audit_id        SERIAL PRIMARY KEY,
            audit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action          TEXT,
            vcode           TEXT,
            vmisc           TEXT,
            "iOrder"        INTEGER,
            "vAmtType"      TEXT,
            "vNotes"        TEXT,
            "PropCode"      TEXT,
            nmisc           DOUBLE PRECISION,
            dteffective     TEXT,
            vtranstype      TEXT,
            "mAmount"       DOUBLE PRECISION,
            "nPercent"      DOUBLE PRECISION,
            "FXRate"        DOUBLE PRECISION,
            "vState"        TEXT
        )
        """,
    ]

    with engine.connect() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))

        # Fix column types — the table may have been created with all-TEXT
        # columns via pandas to_sql or SQLite DDL passed through.
        # Migrate to proper types so PostgreSQL returns ints/floats.
        _pg_fix_column_types(conn, text, "prospective_loans", {
            "id": "INTEGER",
            "loan_amount": "DOUBLE PRECISION",
            "lender_uw_noi": "DOUBLE PRECISION",
            "max_ltv": "DOUBLE PRECISION",
            "min_dscr": "DOUBLE PRECISION",
            "min_debt_yield": "DOUBLE PRECISION",
            "interest_rate": "DOUBLE PRECISION",
            "rate_spread_bps": "INTEGER",
            "term_years": "INTEGER",
            "amort_years": "INTEGER",
            "io_years": "DOUBLE PRECISION",
            "closing_costs": "DOUBLE PRECISION",
            "reserve_holdback": "DOUBLE PRECISION",
        })

        # Ensure id has a SERIAL sequence
        try:
            has_seq = conn.execute(text(
                "SELECT pg_get_serial_sequence('prospective_loans', 'id')"
            )).scalar()
            if not has_seq:
                conn.execute(text(
                    "CREATE SEQUENCE IF NOT EXISTS prospective_loans_id_seq"
                ))
                conn.execute(text(
                    "SELECT setval('prospective_loans_id_seq', COALESCE((SELECT MAX(id) FROM prospective_loans), 0))"
                ))
                conn.execute(text(
                    "UPDATE prospective_loans SET id = nextval('prospective_loans_id_seq') WHERE id IS NULL"
                ))
                conn.execute(text(
                    "ALTER TABLE prospective_loans ALTER COLUMN id SET DEFAULT nextval('prospective_loans_id_seq')"
                ))
                conn.execute(text(
                    "ALTER TABLE prospective_loans ALTER COLUMN id SET NOT NULL"
                ))
                conn.execute(text(
                    "ALTER SEQUENCE prospective_loans_id_seq OWNED BY prospective_loans.id"
                ))
                logger.info("Fixed prospective_loans.id to use SERIAL sequence")
        except Exception as e:
            logger.warning(f"Could not fix prospective_loans id sequence: {e}")

        conn.commit()


def create_additional_tables(conn: sqlite3.Connection):
    """
    Create tables that don't come from CSVs
    
    These are for app-specific data like narratives, report templates, etc.
    """
    
    # Narratives table for report text sections
    conn.execute("""
        CREATE TABLE IF NOT EXISTS narratives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vcode TEXT NOT NULL,
            section_name TEXT NOT NULL,
            content TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by TEXT,
            UNIQUE(vcode, section_name)
        )
    """)
    
    # Report templates
    conn.execute("""
        CREATE TABLE IF NOT EXISTS report_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_name TEXT UNIQUE NOT NULL,
            sections TEXT NOT NULL,  -- JSON array of section names
            format TEXT DEFAULT 'PDF',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Data import log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS import_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            rows_imported INTEGER,
            import_mode TEXT,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            imported_by TEXT,
            source_file TEXT
        )
    """)
    
    # Calculation cache (for expensive calculations)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calculation_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vcode TEXT NOT NULL,
            calculation_type TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(vcode, calculation_type, input_hash)
        )
    """)

    # One Pager comments table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS one_pager_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vcode TEXT NOT NULL,
            reporting_period TEXT NOT NULL,
            econ_comments TEXT,
            business_plan_comments TEXT,
            accrued_pref_comment TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(vcode, reporting_period)
        )
    """)

    # Capital calls (may also be populated from CSV import)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS capital_calls (
            Vcode TEXT NOT NULL,
            PropCode TEXT,
            CallDate TEXT,
            Amount REAL,
            CallType TEXT,
            FundingSource TEXT,
            Notes TEXT,
            Typename TEXT DEFAULT 'Contribution: Investments'
        )
    """)

    # Prospective loans — refinance / new mortgage proposals
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prospective_loans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            vcode           TEXT NOT NULL,
            loan_name       TEXT,
            status          TEXT DEFAULT 'draft',
            refi_date       TEXT NOT NULL,
            existing_loan_id TEXT,
            loan_amount     REAL,
            lender_uw_noi   REAL,
            max_ltv         REAL,
            min_dscr        REAL,
            min_debt_yield  REAL,
            interest_rate   REAL,
            rate_spread_bps INTEGER,
            rate_index      TEXT,
            term_years      INTEGER,
            amort_years     INTEGER,
            io_years        REAL,
            int_type        TEXT DEFAULT 'Fixed',
            closing_costs   REAL DEFAULT 0,
            reserve_holdback REAL DEFAULT 0,
            notes           TEXT,
            created_by      TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by      TEXT,
            updated_at      TIMESTAMP
        )
    """)

    # Prospective loans audit trail
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prospective_loans_audit (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_id     INTEGER NOT NULL,
            action      TEXT NOT NULL,
            vcode       TEXT,
            loan_name   TEXT,
            status      TEXT,
            loan_amount REAL,
            all_fields  TEXT,
            changed_by  TEXT,
            changed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


def run_migrations(conn: sqlite3.Connection):
    """Run any pending schema migrations."""
    _migrate_capital_calls_typename(conn)


def _migrate_capital_calls_typename(conn: sqlite3.Connection):
    """Add Typename column to capital_calls table if it does not exist."""
    try:
        conn.execute("SELECT Typename FROM capital_calls LIMIT 1")
    except Exception:
        try:
            conn.execute(
                "ALTER TABLE capital_calls ADD COLUMN Typename TEXT "
                "DEFAULT 'Contribution: Investments'"
            )
            conn.commit()
            logger.info("Migration: added Typename column to capital_calls")
        except Exception as e:
            # Table may not exist yet — that's fine, column will come from CSV
            logger.debug(f"Migration skip (capital_calls Typename): {e}")


def init_database(data_folder: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize database from CSV files
    
    Args:
        data_folder: Path to folder containing CSV files (default: current directory)
    
    Returns:
        Dictionary with results: {table_name: {'rows': count, 'status': 'success'|'skipped'|'error'}}
    """
    results = {}
    
    if data_folder:
        data_path = Path(data_folder)
    else:
        data_path = Path(".")
    
    conn = get_db_connection()
    
    logger.info("=" * 80)
    logger.info("DATABASE INITIALIZATION")
    logger.info("=" * 80)
    
    # Load tables from CSVs
    for table_name, table_info in TABLE_DEFINITIONS.items():
        csv_file = data_path / table_info['csv']
        
        if not csv_file.exists():
            logger.warning(f"⚠️  Skipped {table_name}: {csv_file} not found")
            results[table_name] = {'rows': 0, 'status': 'skipped', 'file': str(csv_file)}
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            # Normalize column names (strip whitespace)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Load to database
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            logger.info(f"✅ Loaded {table_name}: {len(df):,} rows from {csv_file.name}")
            results[table_name] = {'rows': len(df), 'status': 'success', 'file': str(csv_file)}
            
        except Exception as e:
            logger.error(f"❌ Error loading {table_name}: {e}")
            results[table_name] = {'rows': 0, 'status': 'error', 'error': str(e)}
    
    # Create additional tables
    logger.info("\nCreating application tables...")
    create_additional_tables(conn)
    logger.info("✅ Created additional tables")
    
    # Run schema migrations
    logger.info("\nRunning migrations...")
    run_migrations(conn)
    logger.info("✅ Migrations complete")

    # Create indexes for performance
    logger.info("\nCreating indexes...")
    create_indexes(conn)
    logger.info("✅ Created indexes")

    conn.close()
    
    logger.info("=" * 80)
    logger.info("DATABASE INITIALIZATION COMPLETE")
    logger.info("=" * 80)
    
    return results


def create_indexes(conn: sqlite3.Connection):
    """Create indexes for common queries"""
    
    indexes = [
        # Forecasts
        "CREATE INDEX IF NOT EXISTS idx_forecasts_vcode ON forecasts(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(event_date)",
        "CREATE INDEX IF NOT EXISTS idx_forecasts_account ON forecasts(vAccount)",
        
        # Waterfalls
        "CREATE INDEX IF NOT EXISTS idx_waterfalls_vcode ON waterfalls(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_waterfalls_vmisc ON waterfalls(vmisc)",
        
        # Accounting
        "CREATE INDEX IF NOT EXISTS idx_accounting_investment ON accounting(InvestmentID)",
        "CREATE INDEX IF NOT EXISTS idx_accounting_investor ON accounting(InvestorID)",
        "CREATE INDEX IF NOT EXISTS idx_accounting_date ON accounting(EffectiveDate)",
        
        # Relationships
        "CREATE INDEX IF NOT EXISTS idx_relationships_investment ON relationships(InvestmentID)",
        "CREATE INDEX IF NOT EXISTS idx_relationships_investor ON relationships(InvestorID)",
        
        # Loans
        "CREATE INDEX IF NOT EXISTS idx_loans_vcode ON loans(vCode)",
        
        # Capital calls
        "CREATE INDEX IF NOT EXISTS idx_capital_calls_vcode ON capital_calls(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_capital_calls_date ON capital_calls(CallDate)",

        # ISBS (income statement / balance sheet) — legacy monolithic table
        "CREATE INDEX IF NOT EXISTS idx_isbs_vcode ON isbs(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_source ON isbs(vSource)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_date ON isbs(dtEntry)",

        # ISBS split tables
        "CREATE INDEX IF NOT EXISTS idx_isbs_interim_is_vcode ON isbs_interim_is(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_interim_is_date ON isbs_interim_is(dtEntry)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_interim_is_hist_vcode ON isbs_interim_is_historical(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_interim_is_hist_date ON isbs_interim_is_historical(dtEntry)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_interim_bs_vcode ON isbs_interim_bs(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_interim_bs_date ON isbs_interim_bs(dtEntry)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_budget_is_vcode ON isbs_budget_is(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_budget_is_date ON isbs_budget_is(dtEntry)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_projected_is_vcode ON isbs_projected_is(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_projected_is_date ON isbs_projected_is(dtEntry)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_valuation_is_vcode ON isbs_valuation_is(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_valuation_is_date ON isbs_valuation_is(dtEntry)",

        # Occupancy
        "CREATE INDEX IF NOT EXISTS idx_occupancy_vcode ON occupancy(vCode)",
        "CREATE INDEX IF NOT EXISTS idx_occupancy_qtr ON occupancy(Qtr)",

        # One Pager comments
        "CREATE INDEX IF NOT EXISTS idx_one_pager_vcode ON one_pager_comments(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_one_pager_period ON one_pager_comments(reporting_period)",
    ]
    
    for idx_sql in indexes:
        try:
            conn.execute(idx_sql)
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    conn.commit()


def refresh_table_from_csv(
    table_name: str, 
    csv_path: str, 
    mode: str = 'replace',
    user: str = None
) -> Dict[str, Any]:
    """
    Refresh a table from CSV file
    
    Args:
        table_name: Database table name
        csv_path: Path to CSV file
        mode: 'replace' (delete all, insert new) or 'append' (add to existing)
        user: Username for logging (optional)
    
    Returns:
        Dictionary with import results
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        
        conn = get_db_connection()
        
        # Import data
        df.to_sql(table_name, conn, if_exists=mode, index=False)
        
        # Log import
        conn.execute("""
            INSERT INTO import_log (table_name, rows_imported, import_mode, imported_by, source_file)
            VALUES (?, ?, ?, ?, ?)
        """, (table_name, len(df), mode, user or 'system', csv_path))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Refreshed {table_name}: {len(df):,} rows ({mode} mode)")
        
        return {
            'status': 'success',
            'table': table_name,
            'rows': len(df),
            'mode': mode
        }
        
    except Exception as e:
        logger.error(f"❌ Error refreshing {table_name}: {e}")
        return {
            'status': 'error',
            'table': table_name,
            'error': str(e)
        }


def export_table_to_csv(table_name: str, csv_path: str) -> Dict[str, Any]:
    """
    Export database table to CSV
    
    Args:
        table_name: Database table name
        csv_path: Output CSV path
    
    Returns:
        Dictionary with export results
    """
    try:
        conn = get_db_connection()
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        df.to_csv(csv_path, index=False)
        
        logger.info(f"✅ Exported {table_name}: {len(df):,} rows to {csv_path}")
        
        return {
            'status': 'success',
            'table': table_name,
            'rows': len(df),
            'file': csv_path
        }
        
    except Exception as e:
        logger.error(f"❌ Error exporting {table_name}: {e}")
        return {
            'status': 'error',
            'table': table_name,
            'error': str(e)
        }


def get_table_info(table_name: str) -> Dict[str, Any]:
    """
    Get information about a table
    
    Returns:
        Dictionary with table metadata
    """
    conn = get_db_connection()
    
    try:
        # Get row count
        count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", conn)
        row_count = count['cnt'].iloc[0]
        
        # Get schema
        schema = pd.read_sql(f"PRAGMA table_info({table_name})", conn)
        
        # Get sample data
        sample = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        
        conn.close()
        
        return {
            'table': table_name,
            'rows': row_count,
            'columns': len(schema),
            'schema': schema.to_dict('records'),
            'sample': sample.to_dict('records')
        }
        
    except Exception as e:
        conn.close()
        return {
            'table': table_name,
            'error': str(e)
        }


def list_all_tables() -> List[Dict[str, Any]]:
    """
    List all tables in database
    
    Returns:
        List of table metadata dictionaries
    """
    conn = get_db_connection()
    
    tables = pd.read_sql("""
        SELECT name, sql 
        FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
    """, conn)
    
    table_list = []
    
    for _, row in tables.iterrows():
        table_name = row['name']
        
        # Get row count
        count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", conn)
        row_count = count['cnt'].iloc[0]
        
        # Get description from definitions if available
        description = TABLE_DEFINITIONS.get(table_name, {}).get('description', '')
        
        table_list.append({
            'name': table_name,
            'rows': row_count,
            'description': description
        })
    
    conn.close()
    
    return table_list


def backup_database(backup_dir: str = "backups") -> Dict[str, Any]:
    """
    Create timestamped backup of all tables as CSVs
    
    Args:
        backup_dir: Directory for backups
    
    Returns:
        Dictionary with backup results
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    conn = get_db_connection()
    
    # Get all tables
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", 
        conn
    )
    
    results = {}
    
    for table_name in tables['name']:
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            
            filename = f"{table_name}_{timestamp}.csv"
            filepath = backup_path / filename
            
            df.to_csv(filepath, index=False)
            
            results[table_name] = {
                'status': 'success',
                'rows': len(df),
                'file': str(filepath)
            }
            
        except Exception as e:
            results[table_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    conn.close()
    
    logger.info(f"✅ Backup created in {backup_dir}/ with timestamp {timestamp}")
    
    return {
        'timestamp': timestamp,
        'backup_dir': backup_dir,
        'tables': results
    }


def validate_database() -> Dict[str, Any]:
    """
    Validate database integrity and structure
    
    Returns:
        Dictionary with validation results
    """
    conn = get_db_connection()
    issues = []
    
    # Check each expected table exists
    for table_name in TABLE_DEFINITIONS.keys():
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", conn)
            row_count = count['cnt'].iloc[0]
            
            if row_count == 0:
                issues.append({
                    'severity': 'warning',
                    'table': table_name,
                    'message': 'Table exists but is empty'
                })
        except Exception as e:
            issues.append({
                'severity': 'error',
                'table': table_name,
                'message': f'Table missing or error: {e}'
            })
    
    conn.close()
    
    return {
        'valid': len([i for i in issues if i['severity'] == 'error']) == 0,
        'issues': issues
    }


def execute_query(query: str, params: tuple = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as DataFrame.

    Supports both SQLite (?) and PostgreSQL (%s) parameter styles.
    """
    conn = get_db_connection()

    # Convert SQLite-style ? params to %s for PostgreSQL
    if _sa_engine is not None and params and "?" in query:
        query = query.replace("?", "%s")

    try:
        if params:
            df = pd.read_sql(query, conn, params=params)
        else:
            df = pd.read_sql(query, conn)
    finally:
        conn.close()

    return df


def save_waterfall_steps(vcode: str, steps_df: pd.DataFrame):
    """Replace all waterfall steps for a vcode with the provided DataFrame.

    Backs up existing rows to waterfall_audit first, then DELETE + INSERT
    within a transaction.
    """
    conn = get_db_connection()
    try:
        # Ensure audit table exists (SQLite only; PostgreSQL handled by migration)
        if _sa_engine is None:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS waterfall_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT,
                    vcode TEXT,
                    vmisc TEXT,
                    iOrder INTEGER,
                    vAmtType TEXT,
                    vNotes TEXT,
                    PropCode TEXT,
                    nmisc REAL,
                    dteffective TEXT,
                    vtranstype TEXT,
                    mAmount REAL,
                    nPercent REAL,
                    FXRate REAL,
                    vState TEXT
                )
            """)

        # Backup existing rows
        if _sa_engine is not None:
            from sqlalchemy import text
            existing = pd.read_sql(
                text("SELECT * FROM waterfalls WHERE vcode = :vc"), conn, params={"vc": vcode}
            )
        else:
            existing = pd.read_sql(
                "SELECT * FROM waterfalls WHERE vcode = ?", conn, params=(vcode,)
            )
        if not existing.empty:
            audit_cols = [
                "vcode", "vmisc", "iOrder", "vAmtType", "vNotes", "PropCode",
                "nmisc", "dteffective", "vtranstype", "mAmount", "nPercent",
                "FXRate", "vState",
            ]
            # Quote column names for PostgreSQL (lowercases unquoted identifiers)
            if _sa_engine is not None:
                audit_col_sql = ', '.join(f'"{c}"' for c in ["action"] + audit_cols)
            else:
                audit_col_sql = "action, " + ", ".join(audit_cols)
            placeholders = ", ".join(["?"] * (1 + len(audit_cols)))
            for _, row in existing.iterrows():
                vals = ["backup"] + [row.get(c) for c in audit_cols]
                _exec(
                    conn,
                    f"INSERT INTO waterfall_audit ({audit_col_sql}) "
                    f"VALUES ({placeholders})",
                    vals,
                )

        # Delete existing rows for this vcode
        _exec(conn, "DELETE FROM waterfalls WHERE vcode = ?", (vcode,))

        # Insert new rows
        wf_cols = [
            "vcode", "vmisc", "iOrder", "vAmtType", "vNotes", "PropCode",
            "nmisc", "dteffective", "vtranstype", "mAmount", "nPercent",
            "FXRate", "vState",
        ]
        # Quote column names for PostgreSQL
        if _sa_engine is not None:
            wf_col_sql = ', '.join(f'"{c}"' for c in wf_cols)
        else:
            wf_col_sql = ", ".join(wf_cols)
        placeholders = ", ".join(["?"] * len(wf_cols))
        for _, row in steps_df.iterrows():
            vals = [row.get(c) for c in wf_cols]
            _exec(
                conn,
                f"INSERT INTO waterfalls ({wf_col_sql}) "
                f"VALUES ({placeholders})",
                vals,
            )

        conn.commit()
        logger.info(f"Saved {len(steps_df)} waterfall steps for {vcode}")
    finally:
        conn.close()


def delete_waterfall_steps(vcode: str, wf_type: str = None):
    """Delete waterfall steps for a vcode (optionally filtered by vmisc type)."""
    conn = get_db_connection()
    try:
        if wf_type:
            _exec(
                conn,
                "DELETE FROM waterfalls WHERE vcode = ? AND vmisc = ?",
                (vcode, wf_type),
            )
        else:
            _exec(conn, "DELETE FROM waterfalls WHERE vcode = ?", (vcode,))
        conn.commit()
        logger.info(
            f"Deleted waterfall steps for {vcode}"
            + (f" (type={wf_type})" if wf_type else "")
        )
    finally:
        conn.close()


# Tables managed exclusively via the app (never overwritten by CSV import)
PROTECTED_TABLES = {'waterfalls', 'one_pager_comments', 'waterfall_audit', 'review_roles', 'review_submissions', 'review_notes', 'prospective_loans', 'prospective_loans_audit', 'planned_loans'}


def _get_import_connection():
    """Get a connection for import operations (PostgreSQL or SQLite)."""
    if _sa_engine is not None:
        return _sa_engine.connect(), True  # (conn, is_postgres)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn, False


def _log_import(conn, is_postgres, table_name, row_count, source):
    """Log an import to import_log table."""
    try:
        if is_postgres:
            from sqlalchemy import text
            conn.execute(
                text("INSERT INTO import_log (table_name, rows_imported, import_mode, imported_by, source_file) "
                     "VALUES (:t, :r, :m, :u, :f)"),
                {"t": table_name, "r": row_count, "m": "replace", "u": "csv_import", "f": source},
            )
        else:
            conn.execute(
                "INSERT INTO import_log (table_name, rows_imported, import_mode, imported_by, source_file) "
                "VALUES (?, ?, ?, ?, ?)",
                (table_name, row_count, 'replace', 'csv_import', source),
            )
    except Exception:
        pass


def _import_dataframe(conn, is_postgres, table_name, df):
    """Import a DataFrame into a table, replacing existing data."""
    if is_postgres:
        # pd.to_sql needs the engine, not a raw connection
        df.to_sql(table_name, _sa_engine, if_exists='replace', index=False)
    else:
        df.to_sql(table_name, conn, if_exists='replace', index=False)


def import_csv_dataframe(
    table_name: str,
    df: pd.DataFrame,
    source: str = "upload",
) -> Dict[str, Any]:
    """Import a DataFrame into a database table (works with SQLite and PostgreSQL).

    Args:
        table_name: Key in TABLE_DEFINITIONS to import.
        df: DataFrame with the CSV data.
        source: Description of where the data came from.

    Returns:
        Result dict: {status, rows, error?}
    """
    if table_name not in TABLE_DEFINITIONS:
        return {'rows': 0, 'status': 'error', 'error': f'Unknown table: {table_name}'}

    if table_name in PROTECTED_TABLES:
        return {'rows': 0, 'status': 'protected'}

    conn, is_postgres = _get_import_connection()

    try:
        df.columns = [str(c).strip() for c in df.columns]
        _import_dataframe(conn, is_postgres, table_name, df)
        _log_import(conn, is_postgres, table_name, len(df), source)

        if is_postgres:
            conn.commit()
        else:
            create_indexes(conn)
            conn.commit()

        logger.info(f"Imported {table_name}: {len(df):,} rows from {source}")
        return {'rows': len(df), 'status': 'success'}

    except Exception as e:
        logger.error(f"Error importing {table_name}: {e}")
        return {'rows': 0, 'status': 'error', 'error': str(e)}
    finally:
        conn.close()


def import_csv_stream(
    table_name: str,
    file_stream,
    source: str = "upload",
    chunk_size: int = 50_000,
) -> Dict[str, Any]:
    """Import a CSV file stream in chunks to avoid OOM on large files.

    Reads the CSV in chunks and writes each chunk to the database.
    First chunk replaces the table; subsequent chunks append.

    Args:
        table_name: Key in TABLE_DEFINITIONS to import.
        file_stream: File-like object (e.g., request file stream).
        source: Description of where the data came from.
        chunk_size: Number of rows per chunk.

    Returns:
        Result dict: {status, rows, error?}
    """
    if table_name not in TABLE_DEFINITIONS:
        return {'rows': 0, 'status': 'error', 'error': f'Unknown table: {table_name}'}

    if table_name in PROTECTED_TABLES:
        return {'rows': 0, 'status': 'protected'}

    conn, is_postgres = _get_import_connection()

    try:
        total_rows = 0
        first_chunk = True
        engine = _sa_engine if is_postgres else conn

        for chunk in pd.read_csv(file_stream, chunksize=chunk_size, low_memory=False,
                                  dtype=str):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            mode = 'replace' if first_chunk else 'append'
            chunk.to_sql(table_name, engine, if_exists=mode, index=False)
            total_rows += len(chunk)
            first_chunk = False
            logger.info(f"  {table_name}: imported {total_rows:,} rows so far...")

        _log_import(conn, is_postgres, table_name, total_rows, source)

        if is_postgres:
            conn.commit()
        else:
            create_indexes(conn)
            conn.commit()

        logger.info(f"Imported {table_name}: {total_rows:,} rows from {source}")
        return {'rows': total_rows, 'status': 'success'}

    except Exception as e:
        logger.error(f"Error importing {table_name}: {e}")
        return {'rows': 0, 'status': 'error', 'error': str(e)}
    finally:
        conn.close()


def import_single_csv(
    data_folder: str,
    table_name: str,
    db_path: str = DB_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Import a single CSV file into its database table.

    Args:
        data_folder: Path to folder containing the CSV file.
        table_name: Key in TABLE_DEFINITIONS to import.
        db_path: Path to SQLite database file.

    Returns:
        Per-table results dict (same format as import_csvs_to_database).
    """
    if table_name not in TABLE_DEFINITIONS:
        return {table_name: {'rows': 0, 'status': 'error', 'error': f'Unknown table: {table_name}'}}

    if table_name in PROTECTED_TABLES:
        return {table_name: {'rows': 0, 'status': 'protected'}}

    data_path = Path(data_folder)
    if not data_path.is_dir():
        return {'_error': {'status': 'error', 'error': f'Folder not found: {data_folder}'}}

    table_info = TABLE_DEFINITIONS[table_name]
    csv_file = data_path / table_info['csv']

    if not csv_file.exists():
        return {table_name: {'rows': 0, 'status': 'skipped', 'file': str(csv_file)}}

    try:
        df = pd.read_csv(csv_file)
        result = import_csv_dataframe(table_name, df, source=str(csv_file))
        return {table_name: result}
    except Exception as e:
        logger.error(f"Error importing {table_name}: {e}")
        return {table_name: {'rows': 0, 'status': 'error', 'error': str(e)}}


def import_csvs_to_database(
    data_folder: str,
    db_path: str = DB_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Import CSV files into the database, protecting DB-managed tables.

    Iterates ``TABLE_DEFINITIONS``, reads each CSV from *data_folder*, and
    replaces the corresponding table **except** for protected tables
    which are managed exclusively via the application UI.

    Args:
        data_folder: Path to folder containing MRI CSV source files.
        db_path: Path to SQLite database file.

    Returns:
        Per-table results dict: ``{table_name: {status, rows, file?, error?}}``.
    """
    results: Dict[str, Dict[str, Any]] = {}
    data_path = Path(data_folder)

    if not data_path.is_dir():
        return {'_error': {'status': 'error', 'error': f'Folder not found: {data_folder}'}}

    for table_name, table_info in TABLE_DEFINITIONS.items():
        if table_name in PROTECTED_TABLES:
            results[table_name] = {'rows': 0, 'status': 'protected'}
            continue

        csv_file = data_path / table_info['csv']

        if not csv_file.exists():
            results[table_name] = {'rows': 0, 'status': 'skipped', 'file': str(csv_file)}
            continue

        try:
            df = pd.read_csv(csv_file)
            result = import_csv_dataframe(table_name, df, source=str(csv_file))
            results[table_name] = result
        except Exception as e:
            logger.error(f"Error importing {table_name}: {e}")
            results[table_name] = {'rows': 0, 'status': 'error', 'error': str(e)}

    # For SQLite, ensure app-managed tables and indexes exist
    if _sa_engine is None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        create_additional_tables(conn)
        run_migrations(conn)
        create_indexes(conn)
        conn.commit()
        conn.close()

    return results


# ============================================================
# Prospective Loans CRUD
# ============================================================

def get_prospective_loans_for_deal(vcode: str, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Get all prospective loans for a deal, ordered by created_at desc."""
    conn = get_db_connection()
    try:
        rows = _exec(
            conn,
            "SELECT * FROM prospective_loans WHERE vcode = ? ORDER BY created_at DESC",
            (vcode,),
        ).fetchall()
        return [dict(r._mapping) if hasattr(r, '_mapping') else dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def get_prospective_loan_by_id(loan_id: int, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Get a single prospective loan by ID."""
    conn = get_db_connection()
    try:
        row = _exec(
            conn,
            "SELECT * FROM prospective_loans WHERE id = ?", (loan_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
    except Exception:
        return None
    finally:
        conn.close()


def save_prospective_loan(row_dict: Dict[str, Any], username: str = "system", db_path: str = DB_PATH) -> int:
    """Insert or update a prospective loan. Returns the row ID."""
    conn = get_db_connection()
    try:
        if "status" not in row_dict or not row_dict["status"]:
            row_dict["status"] = "draft"
        loan_id = row_dict.get("id")
        cols = [
            "vcode", "loan_name", "status", "refi_date", "existing_loan_id",
            "loan_amount", "lender_uw_noi", "max_ltv", "min_dscr", "min_debt_yield",
            "interest_rate", "rate_spread_bps", "rate_index",
            "term_years", "amort_years", "io_years", "int_type",
            "closing_costs", "reserve_holdback", "notes",
        ]

        if loan_id:
            # Update
            set_clause = ", ".join(f"{c} = ?" for c in cols)
            vals = [row_dict.get(c) for c in cols]
            vals += [username, loan_id]
            _exec(
                conn,
                f"UPDATE prospective_loans SET {set_clause}, updated_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                vals,
            )
            conn.commit()
            _audit_prospective_loan(conn, loan_id, "update", row_dict, username)
        else:
            # Insert
            placeholders = ", ".join("?" for _ in cols)
            col_names = ", ".join(cols)
            vals = [row_dict.get(c) for c in cols]
            vals += [username]
            if _sa_engine is not None:
                # PostgreSQL: use RETURNING id to get the new row ID
                cur = _exec(
                    conn,
                    f"INSERT INTO prospective_loans ({col_names}, created_by) VALUES ({placeholders}, ?) RETURNING id",
                    vals,
                )
                loan_id = cur.fetchone()[0]
            else:
                cur = conn.execute(
                    f"INSERT INTO prospective_loans ({col_names}, created_by) VALUES ({placeholders}, ?)",
                    vals,
                )
                loan_id = cur.lastrowid
            conn.commit()
            _audit_prospective_loan(conn, loan_id, "create", row_dict, username)

        return loan_id
    finally:
        conn.close()


def delete_prospective_loan(loan_id: int, username: str = "system", db_path: str = DB_PATH) -> bool:
    """Delete a prospective loan by ID."""
    conn = get_db_connection()
    try:
        row = _exec(conn, "SELECT * FROM prospective_loans WHERE id = ?", (loan_id,)).fetchone()
        if not row:
            return False
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        _audit_prospective_loan(conn, loan_id, "delete", row_dict, username)
        _exec(conn, "DELETE FROM prospective_loans WHERE id = ?", (loan_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def accept_prospective_loan(loan_id: int, username: str = "system", db_path: str = DB_PATH) -> bool:
    """Accept a prospective loan: sets status='accepted', rejects all others for the same vcode."""
    conn = get_db_connection()
    try:
        row = _exec(conn, "SELECT * FROM prospective_loans WHERE id = ?", (loan_id,)).fetchone()
        if not row:
            return False
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        vcode = row_dict["vcode"]

        # Reject all other loans for this deal
        _exec(
            conn,
            "UPDATE prospective_loans SET status = 'rejected', updated_by = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE vcode = ? AND id != ? AND status != 'rejected'",
            (username, vcode, loan_id),
        )
        # Accept this one
        _exec(
            conn,
            "UPDATE prospective_loans SET status = 'accepted', updated_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (username, loan_id),
        )
        conn.commit()
        _audit_prospective_loan(conn, loan_id, "accept", row_dict, username)
        return True
    finally:
        conn.close()


def revert_prospective_loan(loan_id: int, username: str = "system", db_path: str = DB_PATH) -> bool:
    """Revert an accepted loan back to draft status."""
    conn = get_db_connection()
    try:
        row = _exec(conn, "SELECT * FROM prospective_loans WHERE id = ?", (loan_id,)).fetchone()
        if not row:
            return False
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        _exec(
            conn,
            "UPDATE prospective_loans SET status = 'draft', updated_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (username, loan_id),
        )
        conn.commit()
        _audit_prospective_loan(conn, loan_id, "revert", row_dict, username)
        return True
    finally:
        conn.close()


def _audit_prospective_loan(conn, loan_id: int, action: str, row_dict: dict, username: str):
    """Write audit trail for prospective loan change."""
    import json
    try:
        _exec(
            conn,
            "INSERT INTO prospective_loans_audit (loan_id, action, vcode, loan_name, status, loan_amount, all_fields, changed_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                loan_id, action,
                row_dict.get("vcode"), row_dict.get("loan_name"),
                row_dict.get("status"), row_dict.get("loan_amount"),
                json.dumps({k: str(v) if v is not None else None for k, v in row_dict.items()}),
                username,
            ),
        )
        conn.commit()
    except Exception as e:
        logger.debug(f"Audit write failed for prospective_loan {loan_id}: {e}")


def export_all_tables_to_zip(db_path: str = DB_PATH) -> bytes:
    """Export every database table as CSV files inside a zip archive.

    Each table is written as ``{table_name}_db_export.csv`` to distinguish
    exports from the original MRI source CSVs.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        Zip archive as ``bytes``, suitable for ``st.download_button(data=...)``.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        conn,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for table_name in tables['name']:
            try:
                df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                zf.writestr(f"{table_name}_db_export.csv", csv_bytes)
            except Exception as e:
                logger.warning(f"Export skip {table_name}: {e}")

    conn.close()
    buf.seek(0)
    return buf.read()


# ============================================================
# ISBS Split Migration
# ============================================================

# Mapping from new table name to the vSource value it holds
# isbs_interim_is and isbs_interim_is_historical both map to 'Interim IS'
# but are split by date (pre-2025 vs 2025+)
ISBS_SPLIT_TABLES = {
    'isbs_interim_is': 'Interim IS',
    'isbs_interim_is_historical': 'Interim IS',
    'isbs_interim_bs': 'Interim BS',
    'isbs_budget_is': 'Budget IS',
    'isbs_projected_is': 'Projected IS',
    'isbs_valuation_is': 'Valuation IS',
}

# Date cutoff for Interim IS historical split
_INTERIM_IS_CUTOFF = '2025-01-01'


def split_isbs_table(db_path: str = DB_PATH):
    """Migrate monolithic isbs table into 6 split tables by vSource.

    Interim IS is further split into historical (pre-2025) and current (2025+).
    Idempotent — skips tables that already have data.
    Works with both SQLite and PostgreSQL (via _sa_engine).
    """
    from flask_app.db import get_engine

    engine = get_engine()
    try:
        isbs = pd.read_sql("SELECT * FROM isbs", engine)
    except Exception as e:
        logger.warning(f"split_isbs_table: cannot read isbs table: {e}")
        return

    if isbs.empty:
        logger.info("split_isbs_table: isbs table is empty, nothing to split")
        return

    # Parse dates for the Interim IS historical/current split
    isbs['_dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], format='mixed', dayfirst=False, errors='coerce')

    # Define splits: (table_name, filter_function)
    splits = {
        'isbs_interim_is': lambda df: df[
            (df['vSource'] == 'Interim IS') &
            (df['_dtEntry_parsed'] >= _INTERIM_IS_CUTOFF)
        ],
        'isbs_interim_is_historical': lambda df: df[
            (df['vSource'] == 'Interim IS') &
            (df['_dtEntry_parsed'] < _INTERIM_IS_CUTOFF)
        ],
        'isbs_interim_bs': lambda df: df[df['vSource'] == 'Interim BS'],
        'isbs_budget_is': lambda df: df[df['vSource'] == 'Budget IS'],
        'isbs_projected_is': lambda df: df[df['vSource'] == 'Projected IS'],
        'isbs_valuation_is': lambda df: df[df['vSource'] == 'Valuation IS'],
    }

    for table_name, filter_fn in splits.items():
        # Check if target table already has data
        try:
            existing = pd.read_sql(f"SELECT 1 FROM {table_name} LIMIT 1", engine)
            if not existing.empty:
                logger.info(f"split_isbs_table: {table_name} already populated, skipping")
                continue
        except Exception:
            pass  # Table doesn't exist yet — will be created by to_sql

        subset = filter_fn(isbs).copy()
        # Drop helper column
        subset = subset.drop(columns=['_dtEntry_parsed'], errors='ignore')

        if subset.empty:
            logger.info(f"split_isbs_table: no rows for {table_name}, skipping")
            continue

        # Drop vSource column — it's implicit in the table name
        if 'vSource' in subset.columns:
            subset = subset.drop(columns=['vSource'])

        row_count = len(subset)
        logger.info(f"split_isbs_table: writing {row_count:,} rows to {table_name}")
        subset.to_sql(table_name, engine, if_exists='replace', index=False)

    logger.info("split_isbs_table: migration complete")
