"""Investment Metrics API — the Current and Sold portfolio summary.

One GET. The report is a pure read across ``deals``, ``accounting``,
``commitments``, ``deal_terms``, ``loans`` and the ISBS balance sheet; nothing
here writes, and there is no DDL behind it.
"""
import datetime as _dt

from flask import Blueprint, jsonify, request

from flask_app.auth.routes import login_required
from flask_app.services import investment_metrics_service

investment_metrics_bp = Blueprint("investment_metrics", __name__)

#: Built reports, keyed by as-of date.
#:
#: WHY THIS EXISTS. Measured on frames at live row counts the build takes about
#: ten seconds — it prices 76 deals, and the CoC columns go through the One
#: Pager's PE engine once per deal because that engine is the one that owns
#: those numbers. Ten seconds is tolerable once and not tolerable on every
#: quarter the reader flips between.
#:
#: The payload is cached in ``services/investment_metrics_service.py`` -- by object
#: identity of the frames it reads -- so the Board's pages read the same one.


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
    as_of = None
    raw = (request.args.get("as_of") or "").strip()
    if raw:
        try:
            as_of = _dt.date.fromisoformat(raw)
        except ValueError:
            return jsonify({
                "error": f"as_of must be YYYY-MM-DD; got {raw!r}",
            }), 400

    return jsonify(investment_metrics_service.get_report(as_of))


@investment_metrics_bp.route("/api/investment-metrics/quarters", methods=["GET"])
@login_required
def quarters():
    """Quarter ends the report can be run at, newest first, with the default.

    The screen must not pin a quarter of its own — five literal ``2026-Q2``
    spellings across two views and two scripts is the failure ``v530`` had to
    unpick. The server answers what is available and which one opens.
    """
    import investment_metrics as engine

    # THE LIST AND THE DEFAULT ARE SEPARATE QUESTIONS. The list is every quarter
    # end that has finished; the default is the one the screen opens on, which
    # `cfg.DEFAULT_QUARTER` pins because the quarter that closed two days ago has
    # no closed accounting behind it. Deriving the list from the default would
    # hide the newer quarters, which stay selectable on purpose.
    default = engine.default_as_of()
    newest = engine.latest_quarter_end()
    out = []
    d = max(newest, default)
    for _ in range(16):
        out.append(d.isoformat())
        d = engine.latest_quarter_end(d)
    if default.isoformat() not in out:
        out.append(default.isoformat())
        out.sort(reverse=True)
    return jsonify({"quarters": out, "default": default.isoformat()})
