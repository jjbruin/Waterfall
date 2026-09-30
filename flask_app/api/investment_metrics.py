"""Investment Metrics API — the Current and Sold portfolio summary.

One GET. The report is a pure read across ``deals``, ``accounting``,
``commitments``, ``deal_terms``, ``loans`` and the ISBS balance sheet; nothing
here writes, and there is no DDL behind it.
"""
import datetime as _dt

from flask import Blueprint, jsonify, request

from flask_app.auth.routes import login_required
from flask_app.serializers import safe_json
from flask_app.services import data_service

investment_metrics_bp = Blueprint("investment_metrics", __name__)


@investment_metrics_bp.route("/api/investment-metrics", methods=["GET"])
@login_required
def investment_metrics():
    """Build the report.

    ``as_of`` (``YYYY-MM-DD``) is optional and defaults to the latest quarter
    end that has actually finished. A date that will not parse is REFUSED with
    the reason rather than silently falling back to the default — a report
    quietly dated to a different quarter than the one asked for is worse than
    an error, because nothing on the page would say so.
    """
    import investment_metrics as engine

    as_of = None
    raw = (request.args.get("as_of") or "").strip()
    if raw:
        try:
            as_of = _dt.date.fromisoformat(raw)
        except ValueError:
            return jsonify({
                "error": f"as_of must be YYYY-MM-DD; got {raw!r}",
            }), 400

    data = data_service.get_data()
    out = engine.build_investment_metrics(
        data["inv"],
        data["acct"],
        commitments=data.get("commitments_raw"),
        deal_terms=data.get("deal_terms_raw"),
        loans=data.get("mri_loans_all"),
        isbs_interim_bs=data.get("isbs_interim_bs"),
        waterfalls=data.get("wf"),
        isbs_raw=data.get("isbs_raw"),
        as_of=as_of,
    )
    return jsonify(safe_json(out))


@investment_metrics_bp.route("/api/investment-metrics/quarters", methods=["GET"])
@login_required
def quarters():
    """Quarter ends the report can be run at, newest first, with the default.

    The screen must not pin a quarter of its own — five literal ``2026-Q2``
    spellings across two views and two scripts is the failure ``v530`` had to
    unpick. The server answers what is available and which one opens.
    """
    import investment_metrics as engine

    default = engine.latest_quarter_end()
    out = []
    d = default
    for _ in range(16):
        out.append(d.isoformat())
        d = engine.latest_quarter_end(d)
    return jsonify({"quarters": out, "default": default.isoformat()})
