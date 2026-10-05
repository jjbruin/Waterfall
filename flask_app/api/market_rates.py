"""Market rates endpoints -- Bank of Canada and NY Fed, stored in `market_rates`.

    GET  /api/market-rates/series                 every series, coverage, latest value
    GET  /api/market-rates/observations?series=   one series' history (from/to optional)
    GET  /api/market-rates/rate?series=&date=     the rate in force on a date, and which date it used
    POST /api/market-rates/refresh                bring every series up to date from its publisher

Data Management's section. Refreshing pulls public data and overwrites only
rows the publisher itself supplies, so any signed-in user of the section may
press it; the same refresh also runs at the end of "Refresh All Data from MRI".
"""
import logging

from flask import Blueprint, jsonify, request

from flask_app.auth.routes import login_required
from flask_app.db import get_engine
from flask_app.serializers import safe_json
from flask_app.services import market_rates_service as mr

logger = logging.getLogger(__name__)

market_rates_bp = Blueprint("market_rates", __name__, url_prefix="/api/market-rates")


def _run(name, fn, *a, **k):
    try:
        return jsonify(safe_json(fn(*a, **k)))
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("market rates %s failed: %s", name, e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@market_rates_bp.route("/series", methods=["GET"])
@login_required
def get_series():
    return _run("series", lambda: {"series": mr.series_summary(get_engine())})


@market_rates_bp.route("/observations", methods=["GET"])
@login_required
def get_observations():
    key = request.args.get("series", "")
    if key not in mr.SERIES:
        return jsonify({"error": "unknown series %r" % key}), 400
    return _run("observations", lambda: {
        "series": key, "spec": mr.SERIES[key],
        "observations": mr.observations(get_engine(), key, request.args.get("from"),
                                        request.args.get("to"))})


@market_rates_bp.route("/rate", methods=["GET"])
@login_required
def get_rate():
    key = request.args.get("series", "")
    if key not in mr.SERIES:
        return jsonify({"error": "unknown series %r" % key}), 400
    return _run("rate", lambda: {"rate": mr.rate_on(get_engine(), key,
                                                    request.args.get("date"))})


@market_rates_bp.route("/refresh", methods=["POST"])
@login_required
def post_refresh():
    return _run("refresh", lambda: {"results": mr.refresh(get_engine())})
