"""Flask application factory."""

import os
import sys
from flask import Flask

# Add project root to path so we can import existing modules (waterfall.py, etc.)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def create_app(config_name: str = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_name: 'development' or 'production'. Defaults to FLASK_ENV env var.
    """
    app = Flask(__name__)

    # Load config
    from flask_app.config import config_by_name
    config_name = config_name or os.environ.get("FLASK_ENV", "development")
    app.config.from_object(config_by_name[config_name])

    # Initialize extensions
    from flask_app.extensions import cors, cache
    cors.init_app(app, origins=app.config["CORS_ORIGINS"], supports_credentials=True)
    cache.init_app(app)

    # Custom JSON encoder for numpy types
    import json
    import numpy as np
    import pandas as pd

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return super().default(obj)

    app.json.encoder = NumpyEncoder  # type: ignore

    # Set DB_PATH for legacy database.py (used by import/export/save)
    import database
    database.DB_PATH = app.config["DB_PATH"]

    # Initialize SQLAlchemy engine management
    from flask_app.db import init_app as init_db
    init_db(app)

    # Configure data adapters (MRI API if env vars set, else database)
    with app.app_context():
        from flask_app.services.data_adapters import configure_from_env
        configure_from_env()

    # Register blueprints
    from flask_app.auth.routes import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/auth")

    # SSO (optional — only active if SSO_CLIENT_ID is set)
    from flask_app.auth.sso import sso_bp, init_sso
    app.register_blueprint(sso_bp, url_prefix="/auth/sso")
    init_sso(app)

    from flask_app.api.data import data_bp
    app.register_blueprint(data_bp, url_prefix="/api/data")

    from flask_app.api.deals import deals_bp
    app.register_blueprint(deals_bp, url_prefix="/api/deals")

    from flask_app.api.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    from flask_app.api.financials import financials_bp
    app.register_blueprint(financials_bp, url_prefix="/api/financials")

    from flask_app.api.ownership import ownership_bp
    app.register_blueprint(ownership_bp, url_prefix="/api/ownership")

    from flask_app.api.waterfall_setup import waterfall_setup_bp
    app.register_blueprint(waterfall_setup_bp, url_prefix="/api/waterfall-setup")

    from flask_app.api.reports import reports_bp
    app.register_blueprint(reports_bp, url_prefix="/api/reports")

    from flask_app.api.sold_portfolio import sold_portfolio_bp
    app.register_blueprint(sold_portfolio_bp, url_prefix="/api/sold-portfolio")

    from flask_app.api.psckoc import psckoc_bp
    app.register_blueprint(psckoc_bp, url_prefix="/api/psckoc")

    # Health check
    @app.route("/health")
    def health():
        return {"status": "ok"}

    # Serve Vue SPA in production (static/ directory from Docker build)
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    if os.path.isdir(static_dir):
        from flask import send_from_directory

        @app.route("/", defaults={"path": ""})
        @app.route("/<path:path>")
        def serve_spa(path):
            # Serve actual files (JS, CSS, images) if they exist
            file_path = os.path.join(static_dir, path)
            if path and os.path.isfile(file_path):
                return send_from_directory(static_dir, path)
            # Otherwise serve index.html (Vue Router handles client-side routes)
            return send_from_directory(static_dir, "index.html")
    else:
        # Development mode — Vue dev server handles frontend
        @app.route("/")
        def index():
            rules = sorted(set(
                r.rule for r in app.url_map.iter_rules()
                if not r.rule.startswith("/static")
            ))
            return {
                "app": "Waterfall XIRR API",
                "status": "running",
                "endpoints": len(rules),
            }

    return app
