"""Investment Metrics, built once per quarter and data load.

The cache that used to sit in the route (``api/investment_metrics.py``), moved here
so a second caller -- the Board's investment summaries and performance pages --
reads the SAME payload the Investment Metrics tab shows, rather than building
its own. ONE NUMBER, ONE ENGINE: the engine is ``investment_metrics.build_
investment_metrics``; this module only remembers its answer.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

from flask_app.serializers import safe_json

#: The frames the report reads. Identity is the whole cache test: a data reload
#: creates new frames, so a refresh invalidates every entry by itself.
FRAMES = ("inv", "acct", "commitments_raw", "deal_terms_raw",
          "mri_loans_all", "isbs_interim_bs", "wf", "isbs_raw")

_CACHE: dict = {}
_CACHE_MAX = 8


def get_report(as_of: Optional[_dt.date] = None, data: Optional[dict] = None) -> dict:
    """The JSON-safe Investment Metrics payload at ``as_of`` (None = the default quarter)."""
    import investment_metrics as engine

    if data is None:
        from flask_app.services import data_service
        data = data_service.get_data()
    key = as_of.isoformat() if as_of else "__default__"
    token = tuple(id(data.get(k)) for k in FRAMES)
    hit = _CACHE.get(key)
    if hit and hit["token"] == token and hit["frames"] is not None:
        return hit["payload"]

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
    # Holding the frames keeps their ids from being reused while the entry lives.
    _CACHE[key] = {"token": token, "payload": payload,
                   "frames": tuple(data.get(k) for k in FRAMES)}
    return payload
