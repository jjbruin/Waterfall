"""Guardrail: the market rates table (Bank of Canada, NY Fed).

Makes NO network calls -- the publishers are stubbed -- so it runs anywhere,
the container included. What it pins:

  1. A publisher answering NEWEST FIRST is stored correctly and a second
     refresh REPLACES its window rather than colliding. The first version read
     the replace span off the unsorted list, so the DELETE matched nothing and
     the INSERT raised -- found by an incremental run, Oct 5 2026.
  2. A date published twice keeps one row (the later entry).
  3. ``rate_on`` falls back to the prior publication and SAYS which date it
     used; past the lag it returns None, never a stale value.
  4. One publisher failing is reported and costs the others nothing.
  5. The four SOFR averages are one download, asked once.
  6. The table is protected from the CSV import, and the API is Data
     Management's.
  7. Treasury's par yields are read by COLUMN NAME from a per-year CSV whose
     columns changed over time; a tenor a year lacks contributes nothing, a
     blank is not a zero, and all thirteen tenors share one download per year.
"""
import json
import os
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sqlalchemy import create_engine, text  # noqa: E402

_passed, _failed = [], []


def chk(label, cond, detail=None):
    (_passed if cond else _failed).append(label)
    print(("   ok   " if cond else "   FAIL ") + label + ("" if cond or detail is None else "   [%s]" % (detail,)))


def main() -> int:
    from flask_app.services import market_rates_service as mr

    calls = []
    down = {"boc": False}

    def fake_http(url):
        calls.append(url)
        if "bankofcanada" in url:
            if down["boc"]:
                raise OSError("Bank of Canada unreachable (stub)")
            code = url.split("/observations/")[1].split("/")[0]
            return {"observations": [
                {"d": "2026-06-26", code: {"v": "1.4186"}},
                {"d": "2026-06-29", code: {"v": "1.4206"}},
                {"d": "2026-06-30", code: {"v": "1.4210"}},
                {"d": "2026-07-01", code: {"v": ""}},          # a published blank
            ]}
        # NY Fed: NEWEST FIRST, with 06-29 listed twice (a revision after it).
        return {"refRates": [
            {"effectiveDate": "2026-06-30", "percentRate": 3.68, "average30day": 3.63179,
             "average90day": 3.63459, "average180day": 3.67232, "index": 1.24910242},
            {"effectiveDate": "2026-06-29", "percentRate": 3.60, "average30day": 3.6,
             "average90day": 3.6, "average180day": 3.6, "index": 1.249},
            {"effectiveDate": "2026-06-29", "percentRate": 3.62, "average30day": 3.62,
             "average90day": 3.62, "average180day": 3.62, "index": 1.2491},
            {"effectiveDate": "2026-06-26", "percentRate": 3.65, "average30day": 3.62,
             "average90day": 3.62, "average180day": 3.62, "index": 1.2490},
        ]}

    mr._http_json = fake_http

    # Treasury: newest first, and the two years carry DIFFERENT columns -- 2021
    # has no "4 Mo" (it began in Oct 2022) and puts "10 Yr" in another position.
    ust = {
        2021: 'Date,"1 Mo","3 Mo","2 Yr","10 Yr","30 Yr"\n'
              '12/31/2021,0.06,0.05,0.73,1.52,1.90\n12/30/2021,0.06,0.05,0.73,1.51,1.93\n',
        2026: 'Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr",'
              '"7 Yr","10 Yr","20 Yr","30 Yr"\n'
              '06/30/2026,4.20,4.21,4.22,4.25,4.27,4.30,4.20,4.14,4.18,4.25,4.35,4.44,4.80,4.91\n'
              '06/29/2026,4.21,4.21,4.23,4.26,,4.31,4.21,4.15,4.19,4.26,4.36,4.45,4.81,4.92\n',
    }

    def fake_text(url):
        calls.append(url)
        year = int(url.split("daily-treasury-rates.csv/")[1].split("/")[0])
        return ust.get(year, 'Date,"1 Mo"\n')

    mr._http_text = fake_text
    eng = create_engine("sqlite:///" + os.path.join(tempfile.mkdtemp(), "rates.db"))
    start, end = date(2026, 6, 20), date(2026, 7, 2)

    print("\n1. Newest-first is stored in order, and a second refresh replaces its window")
    r1 = mr.refresh(eng, start=start, end=end)
    chk("every series refreshed", all(v["status"] == "ok" for v in r1.values()), r1)
    chk("SOFR's latest is the newest date, not the first one listed",
        r1["SOFR"]["latest"] == "2026-06-30", r1["SOFR"])
    r2 = mr.refresh(eng, start=start, end=end)
    chk("a second refresh over the same window succeeds", all(v["status"] == "ok" for v in r2.values()), r2)
    with eng.connect() as c:
        n = dict(c.execute(text("SELECT series, COUNT(*) FROM market_rates GROUP BY series")).all())
    chk("...and leaves one row per date (SOFR 3, USD/CAD 3)", n.get("SOFR") == 3 and n.get("USDCAD") == 3, n)

    print("\n2. A date published twice keeps the later entry; a published blank is not a zero")
    chk("SOFR 2026-06-29 is the revision, 3.62", mr.rate_on(eng, "SOFR", "2026-06-29")["value"] == 3.62)
    with eng.connect() as c:
        blank = c.execute(text("SELECT COUNT(*) FROM market_rates WHERE series='USDCAD' "
                               "AND rate_date='2026-07-01'")).scalar()
    chk("USD/CAD's blank 2026-07-01 was not stored", blank == 0, blank)

    print("\n3. The rate in force on a date, and which date it came from")
    r = mr.rate_on(eng, "USDCAD", "2026-06-30")
    chk("6/30 is the 6/30 publication, CAD per USD", r and r["value"] == 1.4210 and r["date"] == "2026-06-30"
        and r["unit"] == "CAD per USD" and r["lag_days"] == 0, r)
    r = mr.rate_on(eng, "USDCAD", "2026-06-28")
    chk("a Sunday uses Friday and says so", r and r["date"] == "2026-06-26" and r["lag_days"] == 2, r)
    chk("past the lag it is None, not a stale rate",
        mr.rate_on(eng, "USDCAD", "2026-07-20") is None)
    chk("before anything was published it is None", mr.rate_on(eng, "USDCAD", "2026-01-01") is None)

    print("\n4. One publisher down costs the others nothing")
    down["boc"] = True
    r3 = mr.refresh(eng, start=start, end=end)
    chk("Bank of Canada series report the error", all(r3[k]["status"] == "error" for k in ("USDCAD", "CORRA", "BOC_POLICY")), r3)
    chk("NY Fed series still refreshed", all(r3[k]["status"] == "ok" for k in ("SOFR", "EFFR", "SOFR_30D")), r3)
    chk("...and the stored Bank of Canada rates are untouched",
        mr.rate_on(eng, "USDCAD", "2026-06-30")["value"] == 1.4210)
    down["boc"] = False

    print("\n5. The four SOFR averages are one download")
    calls.clear()
    mr.refresh(eng, keys=["SOFR_30D", "SOFR_90D", "SOFR_180D", "SOFR_INDEX"], start=start, end=end)
    chk("asked once, not four times", len([c for c in calls if "sofrai" in c]) == 1, calls)
    chk("each field went to its own series",
        mr.rate_on(eng, "SOFR_90D", "2026-06-30")["value"] == 3.63459
        and mr.rate_on(eng, "SOFR_INDEX", "2026-06-30")["value"] == 1.24910242)

    print("\n6. Protected, and Data Management's")
    import database
    chk("market_rates is in PROTECTED_TABLES", "market_rates" in database.PROTECTED_TABLES)
    from flask_app.auth import sections
    chk("/api/market-rates belongs to Data Management",
        sections.sections_for_api("/api/market-rates/series") == ("data_management",),
        sections.sections_for_api("/api/market-rates/series"))
    chk("/market-rates is a Data Management screen",
        sections.section_for_route("/market-rates") == "data_management",
        sections.section_for_route("/market-rates"))

    print("\n7. Treasury par yields, read by column name")
    calls.clear()
    ukeys = [k for k in mr.SERIES if k.startswith("UST_")]
    ru = mr.refresh(eng, keys=ukeys, start=date(2021, 12, 1), end=date(2026, 7, 2))
    chk("all thirteen tenors refreshed", len(ukeys) == 13 and all(ru[k]["status"] == "ok" for k in ukeys), ru)
    chk("one download per YEAR, shared by every tenor (6 years, 6 downloads)",
        len([c for c in calls if "treasury.gov" in c]) == 6, len(calls))
    chk("the 10-year on 6/30/26 is 4.44 -- found by name, not position",
        mr.rate_on(eng, "UST_10Y", "2026-06-30")["value"] == 4.44)
    chk("...and in 2021, where the column sits elsewhere, 1.52",
        mr.rate_on(eng, "UST_10Y", "2021-12-31")["value"] == 1.52)
    chk("a tenor the year did not publish contributes nothing (no 4-month in 2021)",
        mr.rate_on(eng, "UST_4M", "2021-12-31") is None)
    r4 = mr.rate_on(eng, "UST_4M", "2026-06-30")
    chk("a blank cell is not stored: 6/29's empty 4-month is absent, 6/30 is 4.27",
        r4["value"] == 4.27 and mr.rate_on(eng, "UST_4M", "2026-06-29", max_lag_days=0) is None, r4)
    chk("the source is named", r4["source"] == "U.S. Treasury" and r4["unit"] == "percent")

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
