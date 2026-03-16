"""Flask application configuration."""

import os
from datetime import date
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

    # Cache
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 300

    # Defaults matching Streamlit sidebar
    DEFAULT_START_YEAR = date.today().year
    DEFAULT_HORIZON_YEARS = 10
    PRO_YR_BASE_DEFAULT = date.today().year - 1
    ACTUALS_THROUGH = None  # None = full forecast; ISO date string = actuals cutoff


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
