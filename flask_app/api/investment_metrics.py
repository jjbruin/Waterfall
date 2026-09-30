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

#: Built reports, keyed by as-of date.
#:
#: WHY THIS EXISTS. Measured on frames at live row counts the build takes about
#: ten seconds — it prices 76 deals, and the CoC columns go through the One
#: Pager's PE engine once per deal because that engine is the one that owns
#: those numbers. Ten seconds is tolerable once and not tolerable on every
#: quarter the reader flips between.
#:
#: INVALIDATION IS BY OBJECT IDENTITY, NOT BY A VERSION NUMBER. `load_all` is
#: LRU-cached and hands back the SAME DataFrame objects until something clears
#: it; a refresh or a single-table reload builds new ones. So the cache keeps a
#: reference to the frames it was built from and compares with `is`. Holding
#: the reference is what makes that safe — `id()` alone can be reused by a
#: later object at the same address, and row counts can repeat.
#:
#: The cost of holding them is nil: these are the same objects `data_service`
#: is already keeping alive, not copies.
_CACHE: dict = {}
_CACHE_MAX = 8


def _cache_token(data: dict):
    """The frame objects this report reads. Identity is the whole test."""
    return tuple(id(data.get(k)) for k in
                 ("inv", "acct", "commitments_raw", "deal_terms_raw",
                  "mri_loans_all", "isbs_interim_bs", "wf", "isbs_raw"))


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
    key = as_of.isoformat() if as_of else "__default__"
    token = _cache_token(data)
    hit = _CACHE.get(key)
    # `frames` is the tuple of objects the entry was built from — kept so the
    # identity test below is against something still alive, not a stale id().
    if hit and hit["token"] == token and hit["frames"] is not None:
        return jsonify(hit["payload"])

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
    payload = safe_json(out)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = {
        "token": token,
        "payload": payload,
        # Hold the frames so their ids cannot be reused by a later object.
        "frames": tuple(data.get(k) for k in
                        ("inv", "acct", "commitments_raw", "deal_terms_raw",
                         "mri_loans_all", "isbs_interim_bs", "wf", "isbs_raw")),
    }
    return jsonify(payload)


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
