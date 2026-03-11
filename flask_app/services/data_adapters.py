"""Data source adapters — pluggable per-table data loading.

Each adapter implements load(config) -> pd.DataFrame for a specific data table.
The registry maps table names to their active adapter. Default is DatabaseAdapter
(reads from SQLite/PostgreSQL). API adapters can be registered per-table when
MRI API credentials are configured.

Usage in data_service.py:
    from flask_app.services.data_adapters import get_adapter
    acct = get_adapter("accounting").load(config)
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

from flask_app.db import get_engine


class DataAdapter(ABC):
    """Base class for data source adapters."""

    @abstractmethod
    def load(self, config: dict) -> pd.DataFrame:
        """Load data and return a DataFrame."""
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source description."""
        ...


class DatabaseAdapter(DataAdapter):
    """Load from local database (SQLite or PostgreSQL)."""

    def __init__(self, table_name: str):
        self.table_name = table_name

    def load(self, config: dict) -> pd.DataFrame:
        engine = get_engine()
        try:
            return pd.read_sql(f"SELECT * FROM {self.table_name}", engine)
        except Exception:
            return pd.DataFrame()

    def source_name(self) -> str:
        return f"database:{self.table_name}"


class MriApiAdapter(DataAdapter):
    """Load from MRI accounting system API.

    Requires MRI_API_BASE and MRI_API_TOKEN in config/environment.
    Falls back to DatabaseAdapter if API is unreachable.
    """

    def __init__(self, table_name: str, endpoint: str, fallback: bool = True):
        self.table_name = table_name
        self.endpoint = endpoint
        self.fallback = fallback
        self._db_adapter = DatabaseAdapter(table_name)

    def load(self, config: dict) -> pd.DataFrame:
        api_base = config.get("MRI_API_BASE") or os.environ.get("MRI_API_BASE")
        api_token = config.get("MRI_API_TOKEN") or os.environ.get("MRI_API_TOKEN")

        if not api_base or not api_token:
            if self.fallback:
                return self._db_adapter.load(config)
            return pd.DataFrame()

        try:
            import requests
            url = f"{api_base.rstrip('/')}/{self.endpoint.lstrip('/')}"
            headers = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Handle both list-of-dicts and {data: [...]} response formats
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                return pd.DataFrame(data["data"])
            else:
                return pd.DataFrame([data])

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"MRI API error for {self.table_name}: {e}. "
                f"{'Falling back to database.' if self.fallback else 'Returning empty.'}"
            )
            if self.fallback:
                return self._db_adapter.load(config)
            return pd.DataFrame()

    def source_name(self) -> str:
        return f"mri_api:{self.endpoint}"


# ── Adapter Registry ──────────────────────────────────────────────────────

# Default: all tables from database
_registry: dict[str, DataAdapter] = {}


def register_adapter(table_name: str, adapter: DataAdapter):
    """Register a custom adapter for a table."""
    _registry[table_name] = adapter


def get_adapter(table_name: str) -> DataAdapter:
    """Get the adapter for a table. Falls back to DatabaseAdapter."""
    if table_name not in _registry:
        _registry[table_name] = DatabaseAdapter(table_name)
    return _registry[table_name]


def get_source_info() -> dict[str, str]:
    """Return current data sources for all registered tables."""
    return {name: adapter.source_name() for name, adapter in _registry.items()}


def configure_from_env():
    """Auto-configure API adapters based on environment variables.

    If MRI_API_BASE is set, registers MRI adapters for supported tables.
    Tables without API endpoints keep their database adapter.
    """
    api_base = os.environ.get("MRI_API_BASE")
    if not api_base:
        return

    # Map table names to their MRI API endpoints
    MRI_ENDPOINTS = {
        "accounting": "/accounting/feed",
        "deals": "/deals",
        "coa": "/chart-of-accounts",
        "forecasts": "/forecasts",
        "loans": "/loans",
        "valuations": "/valuations",
        "relationships": "/relationships",
        "capital_calls": "/capital-calls",
        "isbs": "/isbs",
        "occupancy": "/occupancy",
        "commitments": "/commitments",
        "tenants": "/tenants",
    }

    for table, endpoint in MRI_ENDPOINTS.items():
        register_adapter(table, MriApiAdapter(table, endpoint, fallback=True))
