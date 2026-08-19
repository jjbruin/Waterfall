"""Flask application configuration."""

import os

from pathlib import Path

# Project root is one level up from flask_app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-waterfall-xirr-32b")
    JWT_SECRET = os.environ.get("JWT_SECRET", SECRET_KEY)
    JWT_EXPIRATION_HOURS = 24

    # Database — PostgreSQL via DATABASE_URL, or SQLite via DB_PATH
    DATABASE_URL = os.environ.get("DATABASE_URL")  # e.g. postgresql://user:pass@host/dbname
    DB_PATH = os.environ.get("DB_PATH", str(PROJECT_ROOT / "waterfall.db"))

    # CORS
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

    # Email / SendGrid
    SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
    SENDGRID_FROM = os.environ.get("SENDGRID_FROM", "")
    APP_URL = os.environ.get("APP_URL", "https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io")

    # Shared network folders (SharePoint-synced, per-developer OneDrive path)
    DATA_DIR = os.environ.get("DATA_DIR", "")          # CSV imports folder
    QUERIES_DIR = os.environ.get("QUERIES_DIR", "")     # MRI SQL query files
    DOWNLOADS_DIR = os.environ.get("DOWNLOADS_DIR", "")  # MRI query result downloads

    # Upload limits
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max request body

    # Cache
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 300

    # Defaults matching Streamlit sidebar
    DEFAULT_START_YEAR = 2026
    DEFAULT_HORIZON_YEARS = 10
    PRO_YR_BASE_DEFAULT = 2025
    ACTUALS_THROUGH = "2026-07-31"  # None = full forecast; ISO date string = actuals cutoff


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
