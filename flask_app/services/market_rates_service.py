"""Public market rates, stored once and read by every screen that needs one.

Jim, Oct 5 2026: "build the rates table with Bank of Canada and NY Fed." Until
now the app held NO exchange rate and NO reference interest rate -- intercompany
says so in terms ("a CAD row is shown, never computed"), the PE tracker typed
CAD x 0.7286 by hand, and every screen that needed a rate either asked a person
or did without. A rate typed into one screen and a different one typed into
another is the same defect as two engines for one number (CLAUDE.md "ONE NUMBER,
ONE ENGINE"), so the rule here is: rates come from this table, and this table
comes from the publisher.

THE SOURCES (both official, free, no API key; measured Oct 5 2026):
  * Bank of Canada Valet -- https://www.bankofcanada.ca/valet/ -- USD/CAD daily
    average (FXUSDCAD, CAD per 1 USD), CORRA, and the policy rate target.
  * Federal Reserve Bank of New York Markets API -- https://markets.newyorkfed.org/
    -- SOFR, EFFR, OBFR, and the 30/90/180-day SOFR averages and SOFR index.

WHAT THIS DOES NOT HAVE, deliberately: TERM SOFR. The 1- and 3-month Term SOFR
most floating loans reference is CME's, and licensed. Overnight SOFR and its
compounded averages are not the same rate, so nothing here is labelled as one.

A RATE IS NEVER INTERPOLATED OR INVENTED. ``rate_on`` returns the last published
observation on or before the date asked for -- a weekend or a holiday falls back
to the prior business day -- and SAYS which date it used. Past ``max_lag_days``
it returns None: a stale rate silently used is worse than a missing one, which
announces itself.

Each stored value carries its source and when it was fetched, so a figure on a
report can be traced to the publisher's own page.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

TABLE = "market_rates"

BOC_URL = "https://www.bankofcanada.ca/valet/observations/{code}/json?start_date={start}&end_date={end}"
NYFED_URL = "https://markets.newyorkfed.org/api/rates/{path}/search.json?startDate={start}&endDate={end}"

#: Every series the app stores. ``key`` is what callers ask for; the rest says
#: where it comes from and what the number means. The unit is part of the
#: definition: FXUSDCAD is CAD PER USD, so USD = CAD / rate, never CAD x rate.
SERIES: Dict[str, dict] = {
    "USDCAD": {"label": "USD/CAD (Bank of Canada daily average)",
               "unit": "CAD per USD", "source": "Bank of Canada", "provider": "boc",
               "code": "FXUSDCAD", "first": "2017-01-03"},
    "CORRA": {"label": "CORRA (Canadian Overnight Repo Rate Average)",
              "unit": "percent", "source": "Bank of Canada", "provider": "boc",
              "code": "AVG.INTWO", "first": "2017-01-03"},
    "BOC_POLICY": {"label": "Bank of Canada target for the overnight rate",
                   "unit": "percent", "source": "Bank of Canada", "provider": "boc",
                   "code": "V39079", "first": "2017-01-03"},
    "SOFR": {"label": "SOFR (Secured Overnight Financing Rate)",
             "unit": "percent", "source": "Federal Reserve Bank of New York",
             "provider": "nyfed", "path": "secured/sofr", "field": "percentRate",
             "first": "2018-04-02"},
    "SOFR_30D": {"label": "30-day average SOFR", "unit": "percent",
                 "source": "Federal Reserve Bank of New York", "provider": "nyfed",
                 "path": "secured/sofrai", "field": "average30day", "first": "2018-04-02"},
    "SOFR_90D": {"label": "90-day average SOFR", "unit": "percent",
                 "source": "Federal Reserve Bank of New York", "provider": "nyfed",
                 "path": "secured/sofrai", "field": "average90day", "first": "2018-04-02"},
    "SOFR_180D": {"label": "180-day average SOFR", "unit": "percent",
                  "source": "Federal Reserve Bank of New York", "provider": "nyfed",
                  "path": "secured/sofrai", "field": "average180day", "first": "2018-04-02"},
    "SOFR_INDEX": {"label": "SOFR Index", "unit": "index",
                   "source": "Federal Reserve Bank of New York", "provider": "nyfed",
                   "path": "secured/sofrai", "field": "index", "first": "2018-04-02"},
    "EFFR": {"label": "Effective Federal Funds Rate", "unit": "percent",
             "source": "Federal Reserve Bank of New York", "provider": "nyfed",
             "path": "unsecured/effr", "field": "percentRate", "first": "2017-01-03"},
    "OBFR": {"label": "Overnight Bank Funding Rate", "unit": "percent",
             "source": "Federal Reserve Bank of New York", "provider": "nyfed",
             "path": "unsecured/obfr", "field": "percentRate", "first": "2017-01-03"},
}

#: A refresh re-reads this many days before the last stored date, because the
#: NY Fed revises a published rate (its ``revisionIndicator``) and Bank of
#: Canada occasionally corrects one. Re-reading costs one small request.
REVISION_WINDOW_DAYS = 10

#: The NY Fed search endpoint is asked one year at a time: a single request for
#: eight years of SOFR is large enough to time out on a cold container.
NYFED_CHUNK_DAYS = 366


def _iso(d) -> str:
    return d.isoformat() if isinstance(d, (date, datetime)) else str(d)[:10]


def _as_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()


def ensure_tables(engine) -> None:
    from sqlalchemy import inspect
    if inspect(engine).has_table(TABLE):
        return
    with engine.begin() as c:
        c.execute(text(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            series TEXT NOT NULL,
            rate_date TEXT NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            unit TEXT,
            source TEXT,
            fetched_at TEXT,
            PRIMARY KEY (series, rate_date))"""))


# ---------------------------------------------------------------- fetching

def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "waterfall-xirr market-rates"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_boc(spec: dict, start: date, end: date) -> List[tuple]:
    data = _http_json(BOC_URL.format(code=spec["code"], start=_iso(start), end=_iso(end)))
    out = []
    for o in data.get("observations") or []:
        v = (o.get(spec["code"]) or {}).get("v")
        if v in (None, ""):
            continue                     # a published blank is not a zero
        out.append((o["d"], float(v)))
    return out


def _fetch_nyfed(spec: dict, start: date, end: date, cache: Optional[dict] = None) -> List[tuple]:
    out = []
    a = start
    while a <= end:
        b = min(end, a + timedelta(days=NYFED_CHUNK_DAYS - 1))
        url = NYFED_URL.format(path=spec["path"], start=_iso(a), end=_iso(b))
        # The four SOFR averages are four fields of ONE response; asked once.
        if cache is not None and url in cache:
            data = cache[url]
        else:
            data = _http_json(url)
            if cache is not None:
                cache[url] = data
        for o in data.get("refRates") or []:
            v = o.get(spec["field"])
            if v is None:
                continue
            out.append((o["effectiveDate"][:10], float(v)))
        a = b + timedelta(days=1)
    return out


def fetch(key: str, start: date, end: date, cache: Optional[dict] = None) -> List[tuple]:
    """[(iso date, value)] for one series straight from its publisher.

    ASCENDING, ONE VALUE PER DATE. The NY Fed returns newest first, and the
    refresh replaces the span from the first observation to the last -- read off
    an unsorted list that span is backwards, the DELETE matches nothing and the
    INSERT collides with the rows it should have replaced (found by the
    incremental test, Oct 5 2026). Where a date appears twice the later entry
    in the response wins; the publisher lists a revision after the original.
    """
    spec = SERIES[key]
    raw = (_fetch_boc(spec, start, end) if spec["provider"] == "boc"
           else _fetch_nyfed(spec, start, end, cache))
    return sorted(dict(raw).items())


def last_stored(engine, key: str) -> Optional[date]:
    ensure_tables(engine)
    with engine.connect() as c:
        v = c.execute(text(f"SELECT MAX(rate_date) FROM {TABLE} WHERE series = :s"),
                      {"s": key}).scalar()
    return _as_date(v) if v else None


def refresh(engine, keys: Optional[List[str]] = None, start: Optional[date] = None,
            end: Optional[date] = None) -> dict:
    """Bring each series up to date from its publisher.

    Incremental: from the last stored date less REVISION_WINDOW_DAYS, or from
    the series' first published date when nothing is stored. Each series is its
    own transaction and its own result, so a publisher being down costs that
    publisher's series and nothing else -- and is REPORTED, not swallowed.
    """
    ensure_tables(engine)
    end = end or date.today()
    results, cache = {}, {}
    for key in keys or list(SERIES):
        if key not in SERIES:
            results[key] = {"status": "error", "error": "unknown series"}
            continue
        spec = SERIES[key]
        last = last_stored(engine, key)
        a = start or ((last - timedelta(days=REVISION_WINDOW_DAYS)) if last
                      else _as_date(spec["first"]))
        try:
            obs = fetch(key, a, end, cache)
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning("market rates: %s fetch failed", key, exc_info=True)
            results[key] = {"status": "error", "error": str(e)[:200], "from": _iso(a)}
            continue
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with engine.begin() as c:
            if obs:
                c.execute(text(f"DELETE FROM {TABLE} WHERE series = :s AND rate_date >= :a "
                               f"AND rate_date <= :b"),
                          {"s": key, "a": obs[0][0], "b": obs[-1][0]})
                c.execute(text(f"INSERT INTO {TABLE} (series, rate_date, value, unit, source, "
                               f"fetched_at) VALUES (:s, :d, :v, :u, :src, :f)"),
                          [{"s": key, "d": d, "v": v, "u": spec["unit"],
                            "src": spec["source"], "f": now} for d, v in obs])
        results[key] = {"status": "ok", "rows": len(obs), "from": _iso(a), "to": _iso(end),
                        "latest": obs[-1][0] if obs else (_iso(last) if last else None)}
    return results


# ---------------------------------------------------------------- reading

def rate_on(engine, key: str, on, max_lag_days: int = 7) -> Optional[dict]:
    """The published rate in force on ``on``: the last observation on or before it.

    Returns {"series", "value", "date", "unit", "source", "asked", "lag_days"}
    or None when nothing was published within ``max_lag_days`` -- never a guess.
    """
    ensure_tables(engine)
    asked = _as_date(on)
    with engine.connect() as c:
        r = c.execute(text(f"SELECT rate_date, value, unit, source FROM {TABLE} "
                           f"WHERE series = :s AND rate_date <= :d "
                           f"ORDER BY rate_date DESC LIMIT 1"),
                      {"s": key, "d": _iso(asked)}).first()
    if not r:
        return None
    used = _as_date(r[0])
    lag = (asked - used).days
    if lag > max_lag_days:
        return None
    return {"series": key, "value": float(r[1]), "date": _iso(used), "unit": r[2],
            "source": r[3], "asked": _iso(asked), "lag_days": lag}


def series_summary(engine) -> List[dict]:
    """Every series with its coverage and latest value, for the screen."""
    ensure_tables(engine)
    with engine.connect() as c:
        rows = {r[0]: r for r in c.execute(text(
            f"SELECT series, COUNT(*), MIN(rate_date), MAX(rate_date), MAX(fetched_at) "
            f"FROM {TABLE} GROUP BY series"))}
    out = []
    for key, spec in SERIES.items():
        r = rows.get(key)
        latest = rate_on(engine, key, r[3], max_lag_days=0) if r else None
        out.append({"key": key, "label": spec["label"], "unit": spec["unit"],
                    "source": spec["source"], "count": int(r[1]) if r else 0,
                    "first": r[2] if r else None, "last": r[3] if r else None,
                    "fetched_at": r[4] if r else None,
                    "latest_value": latest["value"] if latest else None})
    return out


def observations(engine, key: str, start=None, end=None) -> List[dict]:
    ensure_tables(engine)
    q = f"SELECT rate_date, value FROM {TABLE} WHERE series = :s"
    p = {"s": key}
    if start:
        q += " AND rate_date >= :a"
        p["a"] = _iso(_as_date(start))
    if end:
        q += " AND rate_date <= :b"
        p["b"] = _iso(_as_date(end))
    with engine.connect() as c:
        return [{"date": r[0], "value": float(r[1])}
                for r in c.execute(text(q + " ORDER BY rate_date"), p)]
