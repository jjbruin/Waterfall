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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "waterfall.db"

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
    'investor_roe_feed': {
        'csv': 'MRI_Investor_ROE_Feed.csv',
        'description': 'Investor-level financial activity for ROE calculations',
        'key_columns': ['InvestmentID', 'InvestorID', 'EffectiveDate']
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
        'description': 'Income statement and balance sheet data',
        'key_columns': ['vcode', 'dtEntry', 'vSource', 'vAccount']
    },
    'one_pager_comments': {
        'csv': 'OnePager_Comments.csv',
        'description': 'One Pager report comments by deal and period',
        'key_columns': ['vcode', 'reporting_period']
    }
}


def get_db_connection() -> sqlite3.Connection:
    """
    Get database connection with optimizations
    
    Returns:
        sqlite3.Connection with row_factory set to Row
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    
    # Performance optimizations
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
    conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
    conn.execute("PRAGMA cache_size=10000")  # Larger cache
    conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
    
    return conn


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

    conn.commit()


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

        # ISBS (income statement / balance sheet)
        "CREATE INDEX IF NOT EXISTS idx_isbs_vcode ON isbs(vcode)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_source ON isbs(vSource)",
        "CREATE INDEX IF NOT EXISTS idx_isbs_date ON isbs(dtEntry)",

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
    Execute a SQL query and return results as DataFrame
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
    
    Returns:
        DataFrame with query results
    """
    conn = get_db_connection()
    
    if params:
        df = pd.read_sql(query, conn, params=params)
    else:
        df = pd.read_sql(query, conn)
    
    conn.close()
    
    return df
