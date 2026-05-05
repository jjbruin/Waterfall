"""MRI Query Service — run SQL queries against MRI databases via VPN.

Two workflows:
- Any user: run a query → download CSV to network data-downloads folder
- Admin only: refresh app data → run queries → import directly to Azure PostgreSQL

Requires VPN connection to MRI SQL Server instances.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── MRI Server Configuration ─────────────────────────────────────────────

MRI_SERVERS = {
    "pmx": {
        "server": "10.219.226.17,1433",
        "database": "BV6899900001",
        "description": "PMX — Accounting, Relationships, Commitments",
    },
    "im": {
        "server": "10.219.226.18,1433",
        "database": "PSC",
        "description": "IM — ISBS, Valuations, Loans, Occupancy, Tenants",
    },
}

MRI_USERNAME = "PSCVPN"
MRI_PASSWORD = "NVc8MkB^PlRuv*"

# ── Query Registry ────────────────────────────────────────────────────────
# Maps query filename (without .sql) to server, target table, and description.
# Queries with target_table=None are download-only (not importable to app DB).

QUERY_REGISTRY = {
    "accounting_feed": {
        "server": "pmx",
        "target_table": "accounting",
        "description": "Accounting feed — contributions & distributions",
    },
    "MRI_IA_Relationship": {
        "server": "pmx",
        "target_table": "relationships",
        "description": "Entity relationships with ownership %",
    },
    "MRI_Commitments": {
        "server": "pmx",
        "target_table": "commitments",
        "description": "Active investor commitments",
    },
    "ISBS_Download": {
        "server": "im",
        "target_table": "_isbs_split",  # special: split by vSource into 5 tables
        "description": "ISBS journal entries — splits into 5 tables by vSource",
        "post_process": "_post_process_isbs",
    },
    "coa": {
        "server": "im",
        "target_table": None,  # COA view on IM is permission-denied; local copy is authoritative
        "description": "Chart of accounts (local only — IM COA view not accessible)",
    },
    "MRI_Loans": {
        "server": "im",
        "target_table": "loans",
        "description": "Loan structures with dates",
    },
    "MRI_VAL": {
        "server": "im",
        "target_table": "valuations",
        "description": "Property valuations & cap rates",
    },
    "MRI_Occupancy_Download": {
        "server": "im",
        "target_table": "occupancy",
        "description": "Occupancy data with computed Occ%",
    },
    "Tenant_Report": {
        "server": "im",
        "target_table": "tenants",
        "description": "Commercial tenant roster",
    },
    "ROE_Download": {
        "server": "pmx",
        "target_table": None,  # download-only
        "description": "ROE analysis — contributions & distributions with capital tracking",
    },
}

# Network folder paths (SharePoint-synced)
QUERIES_FOLDER = r"C:\Users\jbruin\peaceablestreet.com\Peaceable Street Capital - Documents\Asset Mgmt\7. Azure App\queries"
DOWNLOADS_FOLDER = r"C:\Users\jbruin\peaceablestreet.com\Peaceable Street Capital - Documents\Asset Mgmt\7. Azure App\data-downloads"


def _get_connection(server_key: str):
    """Get a pyodbc connection to an MRI server."""
    import pyodbc

    server_info = MRI_SERVERS[server_key]
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server_info['server']};"
        f"DATABASE={server_info['database']};"
        f"UID={MRI_USERNAME};"
        f"PWD={MRI_PASSWORD};"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)


def _load_query_sql(query_name: str) -> str:
    """Load SQL from the network queries folder."""
    sql_path = Path(QUERIES_FOLDER) / f"{query_name}.sql"
    if not sql_path.exists():
        # Fallback to local queries folder
        sql_path = Path(__file__).resolve().parent.parent.parent / "queries" / f"{query_name}.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_name}.sql")
    text = sql_path.read_text(encoding="utf-8-sig")  # utf-8-sig strips BOM
    return text


def _post_process_isbs(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Val_IS_2025 into Valuation IS."""
    if "vSource" in df.columns:
        df.loc[df["vSource"] == "Val_IS_2025", "vSource"] = "Valuation IS"
    return df


def test_connection(server_key: str) -> dict:
    """Test connectivity to an MRI server. Returns status dict."""
    try:
        t0 = time.time()
        conn = _get_connection(server_key)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return {
            "server": server_key,
            "status": "ok",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {"server": server_key, "status": "error", "error": str(e)[:200]}


def list_queries() -> list[dict]:
    """List all available queries with metadata."""
    queries = []
    for name, info in QUERY_REGISTRY.items():
        # Check if .sql file exists
        sql_exists = (Path(QUERIES_FOLDER) / f"{name}.sql").exists()
        queries.append({
            "name": name,
            "server": info["server"],
            "server_description": MRI_SERVERS[info["server"]]["description"],
            "target_table": info["target_table"],
            "description": info["description"],
            "importable": info["target_table"] is not None,
            "sql_exists": sql_exists,
        })
    # Also list any .sql files in the network folder that aren't in the registry
    try:
        for sql_file in Path(QUERIES_FOLDER).glob("*.sql"):
            name = sql_file.stem
            if name not in QUERY_REGISTRY:
                queries.append({
                    "name": name,
                    "server": None,
                    "server_description": "Not registered — select server to run",
                    "target_table": None,
                    "description": "Custom query (not in registry)",
                    "importable": False,
                    "sql_exists": True,
                })
    except Exception:
        pass
    return queries


def run_query(query_name: str, server_key: str = None, save_csv: bool = True) -> dict:
    """Run a query and optionally save CSV to the network downloads folder.

    Args:
        query_name: Name of the .sql file (without extension)
        server_key: 'pmx' or 'im'. If None, uses registry default.
        save_csv: Whether to save results to the downloads folder.

    Returns:
        dict with keys: rows, columns, csv_path (if saved), elapsed_seconds, dataframe
    """
    info = QUERY_REGISTRY.get(query_name, {})
    if not server_key:
        server_key = info.get("server")
    if not server_key:
        raise ValueError(f"No server specified for query '{query_name}' and it's not in the registry")

    sql = _load_query_sql(query_name)

    logger.info(f"Running query '{query_name}' on {server_key} ({MRI_SERVERS[server_key]['server']})...")
    t0 = time.time()
    conn = _get_connection(server_key)
    conn.timeout = 600  # 10-minute query timeout for large queries like ISBS
    df = pd.read_sql(sql, conn)
    conn.close()
    elapsed = time.time() - t0
    logger.info(f"Query '{query_name}': {len(df):,} rows in {elapsed:.1f}s")

    # Apply post-processing if defined
    post_fn_name = info.get("post_process")
    if post_fn_name and post_fn_name in globals():
        df = globals()[post_fn_name](df)

    result = {
        "query": query_name,
        "server": server_key,
        "rows": len(df),
        "columns": list(df.columns),
        "elapsed_seconds": round(elapsed, 1),
    }

    if save_csv:
        try:
            downloads = Path(DOWNLOADS_FOLDER)
            downloads.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = downloads / f"{query_name}_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            result["csv_path"] = str(csv_path)
            logger.info(f"Saved CSV: {csv_path}")
        except OSError as e:
            # Network folder not available (e.g., running on Azure)
            logger.warning(f"Could not save CSV to network folder: {e}")
            result["csv_path"] = None

    # Attach DataFrame for callers that need it (not serialized to JSON)
    result["_dataframe"] = df
    return result


def import_query_to_database(query_name: str, engine=None) -> dict:
    """Run a query and import results directly into the app database.

    For ISBS, splits data into 5 tables by vSource.
    Returns dict with import status per table.
    """
    import sqlalchemy as sa
    from flask_app.db import get_engine

    if engine is None:
        engine = get_engine()

    info = QUERY_REGISTRY.get(query_name)
    if not info:
        raise ValueError(f"Query '{query_name}' not in registry")
    if not info["target_table"]:
        raise ValueError(f"Query '{query_name}' is download-only (no target table)")

    # Run the query
    result = run_query(query_name, save_csv=False)
    df = result["_dataframe"]

    if len(df) == 0:
        return {"query": query_name, "status": "empty", "rows": 0}

    import_results = {}

    if info["target_table"] == "_isbs_split":
        # Special handling: split ISBS by vSource
        isbs_split_map = {
            "Interim IS": "isbs_interim_is",
            "Interim BS": "isbs_interim_bs",
            "Budget IS": "isbs_budget_is",
            "Projected IS": "isbs_projected_is",
            "Valuation IS": "isbs_valuation_is",
        }
        # Keep only needed columns
        keep_cols = [c for c in ["vcode", "dtEntry", "vSource", "vAccount", "mAmount", "vInput", "statement_id"] if c in df.columns]
        df = df[keep_cols]

        with engine.begin() as conn:
            for vsource, table_name in isbs_split_map.items():
                subset = df[df["vSource"] == vsource]
                conn.execute(sa.text(f"DROP TABLE IF EXISTS {table_name}"))
                if len(subset) > 0:
                    subset.to_sql(table_name, conn, if_exists="replace", index=False)
                else:
                    # Create empty table
                    cols_sql = ", ".join(f'"{c}" TEXT' for c in keep_cols)
                    conn.execute(sa.text(f"CREATE TABLE {table_name} ({cols_sql})"))
                import_results[table_name] = {"rows": len(subset), "status": "ok"}
                logger.info(f"  {table_name}: {len(subset):,} rows")

            # Ensure historical table exists
            conn.execute(sa.text("DROP TABLE IF EXISTS isbs_interim_is_historical"))
            cols_sql = ", ".join(f'"{c}" TEXT' for c in keep_cols)
            conn.execute(sa.text(f"CREATE TABLE isbs_interim_is_historical ({cols_sql})"))
            import_results["isbs_interim_is_historical"] = {"rows": 0, "status": "ok"}

    else:
        # Standard single-table import
        table_name = info["target_table"]
        with engine.begin() as conn:
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {table_name}"))
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        import_results[table_name] = {"rows": len(df), "status": "ok"}
        logger.info(f"  {table_name}: {len(df):,} rows")

    return {
        "query": query_name,
        "status": "ok",
        "elapsed_seconds": result["elapsed_seconds"],
        "tables": import_results,
    }


def refresh_all(engine=None) -> dict:
    """Run all importable queries and refresh the app database.

    This is the admin "Refresh Data" action — replaces CSV upload entirely.
    Returns summary of all imports.
    """
    results = {}
    total_t0 = time.time()

    for query_name, info in QUERY_REGISTRY.items():
        if not info["target_table"]:
            continue  # Skip download-only queries

        try:
            result = import_query_to_database(query_name, engine=engine)
            results[query_name] = result
        except Exception as e:
            logger.error(f"Failed to refresh '{query_name}': {e}")
            results[query_name] = {"query": query_name, "status": "error", "error": str(e)[:200]}

    total_elapsed = time.time() - total_t0
    return {
        "status": "ok",
        "elapsed_seconds": round(total_elapsed, 1),
        "queries": results,
    }
