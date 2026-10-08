"""Interest-rate cap terms of a floating loan, read from MRI's loan record.

NO ENGINE OWNED THIS before the Board's page 28 (Oct 8 2026). MRI keeps a cap in
two places, neither complete on its own:

  * ``vIntRatereset`` -- the cap STRIKE on the index, as a decimal (Belleville
    0.05, Middle Island 0.02, Poplar Prairie 0.06). The engine's loan model
    already reads this field as the cap (``models.Loan``), so it is the primary
    source here too.
  * ``vHedgedStrat`` -- free text kept by asset management, e.g.
    "2.00% (eff. 5.70%), exp: 10/26", "5.00%, exp: 6/27", "6.00% (eff. 8.85%),
    full term". It carries the EXPIRY, and sometimes the only statement of the
    strike (Trolley Square "4.00%, exp: 8/28" has no ``vIntRatereset``).

READ, NEVER GUESSED. The text is read only when it has the plain shape above --
a leading strike, optionally "(eff. x%)", then "exp: m/yy" or "full term".
Anything else ("5.00% for $5.3M, 5yr $825k" -- a cap on part of the loan) is
returned as ``readable: False`` with the raw text, so a page shows the words
rather than a strike the text does not state. Where the field and the text both
state a strike and disagree, the field is used and the disagreement reported.

MAX RATE = strike + spread: what the borrower pays at most while the cap runs.
"""
from __future__ import annotations

import re
from typing import Optional

_PLAIN = re.compile(
    r"^\s*(?P<strike>\d+(?:\.\d+)?)\s*%\s*"
    r"(?:\(\s*eff\.?\s*(?P<eff>\d+(?:\.\d+)?)\s*%\s*\)\s*)?"
    r"(?:,\s*(?:exp\.?:?\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{2,4})|(?P<full>full\s+term)))?\s*$",
    re.IGNORECASE)


def _rate(v) -> Optional[float]:
    """A rate as a decimal, or None. MRI mixes 0.05 and 5 for five percent."""
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f / 100.0 if f >= 1 else f


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip().lower() in ("", "nan", "none")


def cap_terms(strike_field, hedged, text, spread, maturity: Optional[str] = None) -> dict:
    """The cap on one floating loan.

    Returns ``{capped, strike, expiry, max_rate, readable, text, source, problem}``:
    ``capped`` is True / False, or None when the record says there is a hedge but
    its terms cannot be read. ``expiry`` is "m/yy"; "full term" resolves to the
    loan's maturity when given.
    """
    raw = None if _blank(text) else str(text).strip()
    is_hedged = str(hedged or "").strip().lower() in ("yes", "y", "true", "1")
    strike = _rate(strike_field) if not _blank(strike_field) else None
    spread_d = _rate(spread)
    out = {"capped": False, "strike": None, "expiry": None, "max_rate": None, "readable": True,
           "text": raw, "source": None, "problem": None}

    m = _PLAIN.match(raw) if raw else None
    text_strike = _rate(m.group("strike")) if m else None
    if raw and not m:
        out["readable"] = False

    if strike is not None:
        out.update(capped=True, strike=strike, source="vIntRatereset")
        if text_strike is not None and abs(text_strike - strike) > 1e-9:
            out["problem"] = (f"vIntRatereset says {strike:.2%}, the cap text says {text_strike:.2%}; "
                              f"the field is used")
    elif text_strike is not None:
        out.update(capped=True, strike=text_strike, source="vHedgedStrat")
    elif raw or is_hedged:
        # A hedge is recorded but no strike can be read from it.
        out.update(capped=None, readable=False)
        out["problem"] = ("hedge recorded with no readable terms" if not raw
                          else "cap text is not a plain strike / expiry")

    if m and m.group("m"):
        yy = m.group("y")[-2:]
        out["expiry"] = f"{int(m.group('m'))}/{yy}"
    elif m and m.group("full"):
        out["expiry"] = maturity
    if out["strike"] is not None and spread_d is not None:
        out["max_rate"] = out["strike"] + spread_d
    return out
