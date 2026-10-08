"""Investment Metrics — the Current and Sold portfolio summary, from MRI data.

This is the app's answer to the quarterly ``PSC Investment Summary`` that has
been maintained by hand in a workbook. The workbook is the reference for
LAYOUT and WORDING (see ``investment_metrics_config``); it is deliberately
**not** the source of any figure here. Almost every data column in it is typed
in — of the twenty-one columns on the Current page, seventeen are constants on
every row — so reproducing it means deriving those figures, not copying them.

ONE NUMBER, ONE ENGINE (CLAUDE.md). Nothing here re-derives a figure the app
already owns:

===========================  ====================================================
Column                       Engine
===========================  ====================================================
Act. CoC Since Close         ``one_pager.get_pe_performance`` -> ``roe_to_date``
Proj. CoC Since Close        ``one_pager.get_pe_performance`` -> ``uw_roe_to_date``
Realized Final IRR           ``metrics.xirr``
Funded to date               ``one_pager.get_pe_performance`` -> ``funded_to_date``
Child properties of a deal   ``one_pager._child_vcodes_for_parent``
===========================  ====================================================

What this module owns, because nothing else computes it: the population and
the twin pairing, the capitalization stack (first lien / PSC pref /
first-loss), proceeds to date, Year-1 CoC, and the pref-weighted averages.

THE TWIN RULE
-------------
A sold investment is carried in ``deals`` as **two rows**, and neither is
complete:

* the **letter** row (``P3RDAVE``) holds ``InvestmentID``, ``Sale_Status``,
  ``Sale_Date``, ``Currency`` and ``Property_Count``;
* the **numeric** row (``P0000002``) holds ``City``, ``State``,
  ``Asset_Type``, ``Operating_Partner`` and ``Lifecycle``, and its
  ``InvestmentID`` is NULL.

Nineteen investments are shaped this way. **Both rows are load-bearing and the
report must merge them**, which was not obvious and is worth stating:

* ``accounting`` and ``commitments`` key on ``InvestmentID``, so only the
  LETTER row can reach a cash flow. ``loaders.build_investmentid_to_vcode``
  builds ``dict(zip(InvestmentID, vcode))``, and every NULL id stringifies to
  the same ``'NAN'`` key — so all nineteen numeric rows collapse onto one
  another and none of them can ever receive accounting.
* ``deal_terms``, ``loans``, ``waterfalls`` and the ISBS tables key on
  ``vcode``, and every one of them uses the NUMERIC vcode. Measured on
  production: ``deal_terms`` holds 81 rows, all ``P00000NN``; not one letter
  vcode appears in any of the three.

So the numeric vcode is the deal's identity and the letter row's
``InvestmentID`` is its key into cash. The reference workbook keys only on the
numeric vcode, which is exactly why its own ``Prop Lists`` lookup returns
``#N/A`` for all nineteen and its sold rows are hand-typed.

Pairing is on the normalised ``Investment_Name``, never on the vcode stem:
``PASTONC`` is *Jefferson Centura* (InvestmentID ``JEFFRC``), and stripping its
leading ``P`` would match ``ASTONC`` — *Aston Center*, an unrelated Giant-7
child.

A row with no ``InvestmentID`` and no name twin is an **orphan**. It is
dropped from the report and named in ``diagnostics['orphans']`` rather than
silently disappearing.
"""
from __future__ import annotations

import datetime as _dt
from committed_pref import resolve_committed_pref  # ONE ENGINE
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import investment_metrics_config as cfg
from metrics import xirr

MILLION = 1_000_000.0

CURRENT = "current"
SOLD = "sold"


# ══════════════════════════════════════════════════════════════════════════
# keys
# ══════════════════════════════════════════════════════════════════════════
_NULLISH = {"", "NAN", "NONE", "NULL", "NAT", "<NA>"}


def norm_id(value: Any) -> str:
    """Strip and upper-case an MRI identifier; nullish spellings become ``''``.

    Every join in this module goes through here. It is not defensive tidying:
    the live ``accounting`` table carries BOTH ``'BARN'`` (1 row) and
    ``'BARN  '`` (14 rows) for Barnbeck, ``'CREEK'``/``'CREEK '``,
    ``'DEVON'``/``'DEVON '``, ``'PMAT'``/``'PMAT  '`` and a lower-case
    ``'Centre'`` against 79 ``'CENTRE'``. Joining raw finds one row and looks
    like a deal with almost no history.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip().upper()
    return "" if s in _NULLISH else s


def norm_name(value: Any) -> str:
    """Fold a deal name for twin matching: case, punctuation and spacing."""
    s = str(value or "").strip().lower()
    keep = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in s]
    return " ".join("".join(keep).split())


def _to_date(value: Any) -> Optional[_dt.date]:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _to_float(value: Any) -> Optional[float]:
    v = pd.to_numeric(value, errors="coerce")
    if v is None or pd.isna(v):
        return None
    return float(v)


def _to_int(value: Any) -> Optional[int]:
    v = _to_float(value)
    return None if v is None else int(v)


# ══════════════════════════════════════════════════════════════════════════
# as-of date
# ══════════════════════════════════════════════════════════════════════════
def plus_months(d: _dt.date, n: int) -> _dt.date:
    """``d`` plus ``n`` CALENDAR months, clamped to the end of the month.

    Calendar months, not 365 days, because footnote (5) says "less than 1 year"
    and a year is what a reader means by it. The two agree on every live deal at
    as-of 2026-06-30 — both select exactly the nine the reference marks — but
    they part company on a 29 February and on any deal whose anniversary falls
    within a day of the as-of date, and the calendar reading is the one the
    footnote states.
    """
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    last = [31, 29 if (y % 4 == 0 and (y % 100 or y % 400 == 0)) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return _dt.date(y, m, min(d.day, last))


def default_as_of(today: Optional[_dt.date] = None) -> _dt.date:
    """The quarter the report OPENS on. See ``cfg.DEFAULT_QUARTER``."""
    pinned = getattr(cfg, "DEFAULT_QUARTER", None)
    if pinned:
        return _dt.date.fromisoformat(pinned)
    return latest_quarter_end(today)


def latest_quarter_end(today: Optional[_dt.date] = None) -> _dt.date:
    """The most recent quarter end STRICTLY BEFORE ``today``.

    Strictly before, not on-or-before, so that running this on 30 Sep — a
    quarter end — reports the quarter that has finished rather than one that
    ends at midnight tonight with no closed figures behind it. This is the same
    convention the Portfolio Snapshot's quarter picker uses.
    """
    today = today or _dt.date.today()
    ends = [
        _dt.date(today.year, 3, 31), _dt.date(today.year, 6, 30),
        _dt.date(today.year, 9, 30), _dt.date(today.year, 12, 31),
    ]
    earlier = [d for d in ends if d < today]
    if earlier:
        return max(earlier)
    return _dt.date(today.year - 1, 12, 31)


# ══════════════════════════════════════════════════════════════════════════
# population and twin pairing
# ══════════════════════════════════════════════════════════════════════════
class DealIdentity:
    """One logical investment, assembled from one or two ``deals`` rows.

    ``vcode`` is always the numeric row's when a pair exists, because that is
    what every vcode-keyed table uses. ``investment_id`` is always the letter
    row's, because that is the only key into ``accounting``.
    """

    __slots__ = ("vcode", "investment_id", "name", "asset_class", "city",
                 "state", "partner", "invest_date", "sale_status", "sale_date",
                 "currency", "property_count", "lifecycle", "vcodes",
                 "shadow_vcode", "key_vcode")

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    @property
    def is_sold(self) -> bool:
        return norm_id(self.sale_status) == "SOLD"

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


def _pick(*values):
    """First value that is not blank/null."""
    for v in values:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        if str(v).strip() == "":
            continue
        return v
    return None


def resolve_deal_identities(inv: pd.DataFrame) -> Tuple[List[DealIdentity], Dict[str, Any]]:
    """Pair the twins and return one identity per logical investment.

    Returns ``(identities, diagnostics)``. Diagnostics name every row that did
    NOT become an identity and why — children, orphans, and rows that merged
    into another. Nothing is dropped silently.
    """
    diag: Dict[str, Any] = {
        "children_excluded": [], "orphans": [], "twins_merged": [],
        "id_collisions": [],
    }
    if inv is None or inv.empty:
        return [], diag

    df = inv.copy()
    if "vcode" not in df.columns and "vCode" in df.columns:
        df = df.rename(columns={"vCode": "vcode"})

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "vcode": str(r.get("vcode", "")).strip(),
            "investment_id": norm_id(r.get("InvestmentID")),
            "name": str(_pick(r.get("Investment_Name")) or "").strip(),
            "asset_class": _pick(r.get("Asset_Type")),
            "city": _pick(r.get("City")),
            "state": _pick(r.get("State")),
            "partner": _pick(r.get("Operating_Partner")),
            "invest_date": _to_date(r.get("Acquisition_Date")),
            "sale_status": _pick(r.get("Sale_Status")),
            "sale_date": _to_date(r.get("Sale_Date")),
            "currency": _pick(r.get("Currency")),
            "property_count": _to_int(r.get("Property_Count")),
            "lifecycle": _pick(r.get("Lifecycle")),
            "portfolio_name": str(_pick(r.get("Portfolio_Name")) or "").strip(),
        })

    keyed = [r for r in rows if r["investment_id"]]
    shadows = [r for r in rows if not r["investment_id"]]

    # ── group the keyed rows by InvestmentID ──────────────────────────────
    # An InvestmentID can appear twice (MCCORD is on both P0000049, a parent,
    # and P0000073, a child carrying the SOLD marker). The parent is the deal;
    # the sale status is taken from whichever row in the group has one, or the
    # pair reports a sold deal as live.
    groups: Dict[str, List[dict]] = {}
    for r in keyed:
        groups.setdefault(r["investment_id"], []).append(r)

    identities: List[DealIdentity] = []
    by_name: Dict[str, DealIdentity] = {}

    for iid, members in sorted(groups.items()):
        parents = [m for m in members if (m["property_count"] or 0) >= 1]
        if len(members) > 1:
            diag["id_collisions"].append({
                "investment_id": iid,
                "vcodes": [m["vcode"] for m in members],
                "chosen": (parents[0]["vcode"] if parents else None),
            })
        if not parents:
            for m in members:
                diag["children_excluded"].append({
                    "vcode": m["vcode"], "name": m["name"],
                    "reason": "Property_Count == 0 (child property)",
                })
            continue
        head = parents[0]
        sale_status = _pick(*[m["sale_status"] for m in members])
        sale_date = _pick(*[m["sale_date"] for m in members])
        ident = DealIdentity(
            vcode=head["vcode"], key_vcode=head["vcode"], investment_id=iid,
            name=head["name"], asset_class=head["asset_class"],
            city=head["city"], state=head["state"], partner=head["partner"],
            invest_date=_pick(*[m["invest_date"] for m in members]),
            sale_status=sale_status, sale_date=sale_date,
            currency=_pick(*[m["currency"] for m in members]),
            property_count=head["property_count"], lifecycle=head["lifecycle"],
            vcodes=[m["vcode"] for m in members], shadow_vcode=None,
        )
        identities.append(ident)
        by_name.setdefault(norm_name(ident.name), ident)

    # ── attach the shadows ────────────────────────────────────────────────
    for s in shadows:
        target = _match_shadow(s, by_name)
        if target is None:
            diag["orphans"].append({
                "vcode": s["vcode"], "name": s["name"],
                "reason": "no InvestmentID and no name twin — unreachable from "
                          "accounting, commitments, deal_terms and loans",
            })
            continue
        # THE NUMERIC ROW BECOMES THE VCODE. deal_terms / loans / waterfalls /
        # ISBS all key on it; the letter row appears in none of them.
        target.shadow_vcode = target.vcode
        target.vcode = s["vcode"]
        target.asset_class = _pick(s["asset_class"], target.asset_class)
        target.city = _pick(s["city"], target.city)
        target.state = _pick(s["state"], target.state)
        target.partner = _pick(s["partner"], target.partner)
        target.lifecycle = _pick(s["lifecycle"], target.lifecycle)
        target.invest_date = _pick(target.invest_date, s["invest_date"])
        target.vcodes = list(target.vcodes) + [s["vcode"]]
        diag["twins_merged"].append({
            "numeric_vcode": s["vcode"], "letter_vcode": target.shadow_vcode,
            "investment_id": target.investment_id, "name": target.name,
        })

    return identities, diag


def _match_shadow(shadow: dict, by_name: Dict[str, DealIdentity]) -> Optional[DealIdentity]:
    """Find the keyed identity a NULL-InvestmentID row describes.

    Exact normalised name first, then containment in either direction — the
    live table carries ``Declan & Walton Apartments`` against its twin's
    ``Declan & Walton``. Containment is only accepted when exactly one
    candidate matches; two would mean guessing.
    """
    key = norm_name(shadow["name"])
    if not key:
        return None
    if key in by_name:
        return by_name[key]
    hits = [ident for name, ident in by_name.items()
            if (name and (name.startswith(key) or key.startswith(name)))]
    return hits[0] if len(hits) == 1 else None


def resolved_inv_frame(inv: pd.DataFrame,
                       identities: List[DealIdentity]) -> pd.DataFrame:
    """``deals``, with the twin rule applied — one row per investment.

    THIS IS WHAT LETS THE ONE PAGER'S PE ENGINE SEE A SOLD DEAL AT ALL, and
    it is worth spelling out because the failure is silent. ``get_pe_performance``
    finds a deal's cash by inverting ``build_investmentid_to_vcode``, which is
    ``dict(zip(InvestmentID, vcode))`` over the raw frame. On a twin pair that
    inversion cannot work either way round:

    * ask it for the NUMERIC vcode and you get nothing, because the numeric row's
      ``InvestmentID`` is NULL — and every NULL stringifies to the same
      ``'NAN'`` key, so all nineteen numeric rows overwrite one another;
    * ask it for the LETTER vcode and you get the accounting, but
      ``deal_terms``, ``waterfalls`` and ISBS hold nothing under that vcode.

    Either way the engine returns its zero-filled defaults and the screen shows
    a deal with no history rather than an error.

    So the frame handed to it says the true thing: the numeric row IS the deal
    AND it carries the letter row's ``InvestmentID``. The letter rows are
    dropped, because leaving them in restores the collision — ``dict(zip(...))``
    keeps the LAST row for a duplicated id, and ``PHOMEW`` sorts after
    ``P0000020``.

    Child properties are kept: ``_child_vcodes_for_parent`` reads this frame.
    """
    df = inv.copy()
    if "vcode" not in df.columns and "vCode" in df.columns:
        df = df.rename(columns={"vCode": "vcode"})
    df["vcode"] = df["vcode"].astype(str).str.strip()

    drop = {i.shadow_vcode for i in identities if i.shadow_vcode}
    df = df[~df["vcode"].isin(drop)].copy()

    ids = {i.vcode: i.investment_id for i in identities}
    df["InvestmentID"] = [
        ids.get(v, norm_id(cur)) for v, cur in zip(df["vcode"], df.get("InvestmentID", ""))
    ]
    return df


def classify(ident: DealIdentity, as_of: _dt.date) -> str:
    """Current or Sold.

    A deal is SOLD at a quarter when it carries the SOLD marker AND its sale date
    is on or before the as-of. A deal sold AFTER the as-of was still held at that
    quarter end, so it is CURRENT and every figure on its row is as of the
    quarter -- values in a quarter stay in that quarter. (The reference prints
    Clima Secur, 30 Bearfoot and 870 Donald Lynch in Sold at 30 Jun 26 under a
    footnote (4) "sold after June 2026"; that was true of the day it was
    produced and not of the quarter, and it is the one place this report
    deliberately departs from the reference page.)

    A SOLD marker with NO sale date stays SOLD: nothing says it was held. A deal
    carrying a sale date but no SOLD marker is still CURRENT -- the marker is what
    accounting sets when the deal is closed out. Deals moved to Current by this
    rule are listed in ``diagnostics["sold_after_as_of_shown_current"]``.
    """
    if ident.is_sold and (ident.sale_date is None or ident.sale_date <= as_of):
        return SOLD
    return CURRENT


def is_young_deal(ident: DealIdentity, as_of: _dt.date) -> bool:
    """Footnote (5): under a year of operating history at the as-of date."""
    if ident.invest_date is None:
        return False
    return plus_months(ident.invest_date, cfg.YOUNG_DEAL_MONTHS) > as_of


def row_markers(ident: DealIdentity, table: str, as_of: _dt.date) -> List[int]:
    """The footnote numbers printed after a deal's name, hardcoded + derived.

    Three markers are DERIVED and the rest transcribed — see the note on
    ``cfg.ROW_MARKERS_CURRENT`` for which and why. Merged and sorted, so a
    hardcoded ``[3]`` plus a derived currency note and a derived young-deal note
    print as ``(2)(3)(5)``, which is the order the reference prints them in.

    SORTING IS THE WHOLE MERGE RULE. The reference prints markers in ascending
    numeric order on every one of its 76 rows, so there is nothing else to
    preserve, and a hand-ordered list would be one more thing to keep in step.
    """
    if table == CURRENT:
        marks = set(cfg.ROW_MARKERS_CURRENT.get(ident.vcode, []))
        if norm_id(ident.currency) not in ("", "USD"):
            marks.add(cfg.NON_USD_MARKER)
        if is_young_deal(ident, as_of):
            marks.add(cfg.YOUNG_DEAL_MARKER)
    else:
        marks = set(cfg.ROW_MARKERS_SOLD.get(ident.vcode, []))
    return sorted(marks)


# ══════════════════════════════════════════════════════════════════════════
# capitalization
# ══════════════════════════════════════════════════════════════════════════
#: An accounting row is PSC's side when its investor is not an operating
#: partner. Two spellings of the same test exist in the data — ``Partner ==
#: 'Preferred Equity'`` (the reference workbook's) and an ``InvestorID`` that
#: does not start with ``OP`` (the app's, used by
#: ``one_pager.get_pe_performance``). The app's is used here so this report
#: and the One Pager cannot disagree; ``diagnostics`` reports any row where
#: the two tests differ.
def is_psc_side(investor_id: Any) -> bool:
    return not norm_id(investor_id).startswith("OP")


#: Column name for the pre-normalised InvestmentID. Added once in
#: `build_investment_metrics`; `_deal_accounting` uses it when present.
_NORM_ID_COL = "_im_iid"


def _deal_accounting(acct: pd.DataFrame, investment_id: str) -> pd.DataFrame:
    """The deal's accounting rows.

    Normalising the id column is O(rows) in PYTHON, and this is called eight
    times per deal across seventy-six deals — eight million `norm_id` calls on
    the live feed. `build_investment_metrics` normalises once up front and
    leaves the result in `_im_iid`; this reads it when it is there and falls
    back to normalising on the spot when a caller passes a raw frame.
    """
    if acct is None or acct.empty or not investment_id:
        return pd.DataFrame()
    if _NORM_ID_COL in acct.columns:
        return acct[acct[_NORM_ID_COL] == investment_id]
    ids = acct["InvestmentID"].map(norm_id)
    return acct[ids == investment_id]


def narrow_isbs_for_pe(isbs_raw: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """The only ISBS rows `get_pe_performance` can read, selected ONCE.

    THIS IS THE DIFFERENCE BETWEEN 90 SECONDS AND THREE. `_get_uw_7073_signed`
    and `_get_uw_pe_periodic` each open with `isbs_raw.copy()` — a full copy of
    the frame — and `get_pe_performance` calls both, once per deal. On the live
    `isbs_raw` (797,660 rows, 325 MB) that is 152 copies of a 325 MB frame to
    reach about five thousand rows.

    The predicate here is the SAME one those functions apply per deal
    (`vSource == 'Projected IS'` and `vAccount` in {7071, 7073}), so the rows
    they see are unchanged — including the case where the comparison matches
    nothing because the column's dtype is wrong, which narrows to empty here
    and returned empty there. Equivalent, not merely similar.

    `one_pager` is deliberately NOT modified: it is shared with the One Pager
    screen and the Portfolio Snapshot, and making the copies cheaper there is a
    change to code this report does not own.
    """
    if isbs_raw is None or isbs_raw.empty:
        return isbs_raw
    from one_pager import UW_PE_DIST_ACCT, UW_PE_ROC_ACCT

    df = isbs_raw
    if "vSource" in df.columns:
        df = df[df["vSource"] == "Projected IS"]
    if "vAccount" in df.columns:
        df = df[df["vAccount"].isin([UW_PE_DIST_ACCT, UW_PE_ROC_ACCT])]
    return df.copy()


def pref_and_first_loss(
    ident: DealIdentity,
    acct: pd.DataFrame,
    commitments: Optional[pd.DataFrame],
    sources: Optional[List[Tuple[str, Optional[float], Optional[float]]]] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """PSC Pref. Equity and First-Loss Equity, in dollars, plus the basis used.

    Three sources, in order, and the basis travels with the figure so a reader
    can tell a commitment from a funding:

    1. ``commitments`` (MRI's IA_Commitment) — the contractual amount;
    2. accounting ``Commitment`` rows (``SubtypeUID`` 1026) — the same fact
       recorded on the journal where the commitments table has no row;
    3. funded to date — what actually went in.

    Falling through to (3) is not a failure, but it is a DIFFERENT quantity: a
    deal part-way through its draw shows less than it has committed. That is
    why the basis is returned rather than assumed.
    """
    levels = (sources if sources is not None
              else capitalization_sources(ident, acct, commitments))

    pref = floss = None
    pref_basis = floss_basis = "unavailable"
    for label, p, o in levels:
        if pref is None and p:
            pref, pref_basis = p, label
        # FIRST-LOSS FALLS THROUGH ON ITS OWN. The two sides are recorded
        # separately in MRI and one can be present where the other is not —
        # the commitments table carries PSC's side for every sold twin and the
        # operating partner's for only ten of them. Stopping both at the level
        # that answered the pref would print a dash for a first-loss figure
        # that is sitting one level down, so each side takes the first source
        # that has it and says which.
        if floss is None and o:
            floss, floss_basis = o, label
        if pref is not None and floss is not None:
            break

    basis = pref_basis if pref_basis == floss_basis else (
        f"PSC pref: {pref_basis}; first-loss: {floss_basis}")
    return pref, floss, basis


def capitalization_sources(
    ident: DealIdentity,
    acct: pd.DataFrame,
    commitments: Optional[pd.DataFrame],
    as_of: Any = None,
) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """Every source for (PSC pref, first-loss), in precedence order.

    Returned whole rather than consumed internally so the report can publish
    what the sources it did NOT use would have said. Measured against the
    reference across all 76 deals: the commitments table lands on the printed
    PSC pref for 64 and on first-loss for 49, accounting Commitment rows for
    42/41, and funded-to-date for 62/48 — so the order below is the best
    available, not merely the first one tried.
    """
    iid = ident.investment_id
    rows = _deal_accounting(acct, iid)
    out: List[Tuple[str, Optional[float], Optional[float]]] = []
    q = _to_date(as_of) if as_of is not None else None

    def upto(df):
        """Accounting rows dated on or before the as-of. QUARTER INTEGRITY: a
        figure read at 26Q2 must not contain a July row. No as-of, no cut."""
        if q is None or df is None or df.empty or "EffectiveDate" not in df.columns:
            return df
        when = pd.to_datetime(df["EffectiveDate"], errors="coerce")
        return df[when <= pd.Timestamp(q)]

    def split(df, col, id_col="InvestorID"):
        if df is None or df.empty:
            return None, None
        psc = df[df[id_col].map(is_psc_side)]
        op = df[~df[id_col].map(is_psc_side)]
        p = _to_float(psc[col].abs().sum()) if not psc.empty else None
        o = _to_float(op[col].abs().sum()) if not op.empty else None
        return (p or None), (o or None)

    if commitments is not None and not commitments.empty and "EntityID" in commitments.columns:
        # Pre-normalised once by `build_investment_metrics` where possible.
        key = (_NORM_ID_COL if _NORM_ID_COL in commitments.columns else None)
        m = (commitments[commitments[key] == iid] if key
             else commitments[commitments["EntityID"].map(norm_id) == iid])
        # ONE ENGINE for the PSC pref side. `committed_pref.resolve_committed_pref`
        # is what the One Pager's cap stack and PE block now read; calling it here
        # too is what stops this report and that page disagreeing about the same
        # deal, which they did on twelve deals until 2026-10-01. With no as-of
        # supplied the whole chain is offered, which is this report's own basis.
        # CURRENT ROWS ONLY on the no-as-of path. Once MRI_Commitments.sql
        # stopped filtering EndDate IS NULL (Jim, 2026-10-01) this frame carries
        # SUPERSEDED REVISIONS, and summing them reads every past version of a
        # pledge as a live one: measured on the real table, Pontchartrain
        # 10,847,420 -> 42,823,260, Middle Island 7,896,655 -> 29,978,275,
        # Belleville 4,752,161 -> 21,533,305. Neither side has an as-of here, so
        # both take the current row and this report is unmoved by the data
        # change. Pass `as_of` to get the quarter-aware figure instead.
        if "EndDate" in m.columns:
            m_cur = m[m["EndDate"].isna()]
        else:
            m_cur = m
        if as_of is not None:
            psc, _basis = resolve_committed_pref(m, iid, as_of)
            # THE OPERATING PARTNER'S SIDE IS READ AS OF THE QUARTER TOO. It was
            # summed over the current rows whatever quarter was asked for, so
            # First-Loss Equity (and with it Total Size and every % of Cap.)
            # quietly carried today's pledge into every earlier quarter.
            op_side = _op_side_in_effect(m, q)
        else:
            psc = split(m_cur, "Amount")[0]
            _, op_side = split(m_cur, "Amount")
        out.append(("commitments (IA_Commitment)", psc, op_side))
    else:
        out.append(("commitments (IA_Commitment)", None, None))

    cm = (rows[rows["is_commitment"].fillna(False)]
          if (not rows.empty and "is_commitment" in rows.columns) else None)
    out.append(("accounting Commitment rows (SubtypeUID 1026)",) + split(upto(cm), "Amt"))

    cb = (rows[rows["is_contribution"].fillna(False)]
          if (not rows.empty and "is_contribution" in rows.columns) else None)
    out.append(("funded to date",) + split(upto(cb), "Amt"))
    return out


def _op_side_in_effect(m: Optional[pd.DataFrame], q: Optional[_dt.date]) -> Optional[float]:
    """The operating partner's commitment in force on ``q``, or None.

    The same range rule ``committed_pref.row_in_effect`` applies to the PSC side
    -- ``StartDate <= q AND (EndDate IS NULL OR EndDate >= q)``, one row per
    (entity, investor) chain, the latest start winning where revisions overlap --
    applied to the investors that are NOT the PSC side. None when no chain has a
    row in effect, so the caller falls through to the next source instead of
    printing a zero.
    """
    if m is None or getattr(m, "empty", True) or q is None:
        return None
    if "InvestorID" not in m.columns or "EntityID" not in m.columns:
        return None
    ops = m[~m["InvestorID"].map(is_psc_side)]
    if ops.empty:
        return None
    chains: Dict[Tuple[str, str], list] = {}
    for _, r in ops.iterrows():
        sd, ed = _to_date(r.get("StartDate")), _to_date(r.get("EndDate"))
        if sd is None or sd > q or (ed is not None and ed < q):
            continue
        amt = abs(_to_float(r.get("Amount")) or 0.0)
        chains.setdefault((norm_id(r.get("EntityID")), norm_id(r.get("InvestorID"))),
                          []).append((sd, str(r.get("CommitmentUID")), amt))
    total = sum(sorted(rows)[-1][2] for rows in chains.values())
    return total or None


#: ``vDateType`` spellings. MRI carries exactly three on the live table:
#: ``Maturity`` (83 rows), ``Origination`` (4) and ``Paid Off`` (4).
_ORIGINATION = "origination"


def _loan_columns(loans: Optional[pd.DataFrame]):
    """``(vcode, LoanID, vDateType, dtEvent, mOrigLoanAmt)``, or None."""
    if loans is None or loans.empty:
        return None
    cols = {c.lower(): c for c in loans.columns}
    need = ("loanid", "vdatetype", "dtevent", "morigloanamt")
    vc = cols.get("vcode")
    if not vc or any(k not in cols for k in need):
        return None
    return (vc, cols["loanid"], cols["vdatetype"], cols["dtevent"],
            cols["morigloanamt"])


def first_lien_by_origination(
    loans: Optional[pd.DataFrame], want: set,
) -> Tuple[Optional[float], str, List[Any]]:
    """The loans originated FIRST, summed. ``(value, how, loans_without_a_date)``.

    Returns ``(None, reason, missing)`` whenever the deal's loans cannot settle
    the question, and the caller falls back. ``missing`` names the LoanIDs with
    no Origination row so the gap is reportable rather than inferred from a
    silent fallback.

    THE RAW FRAME IS REQUIRED and the caller supplies it. This reads
    ``mri_loans_all`` — MRI's rows before ``_filter_paid_off_loans`` and
    ``_collapse_loan_date_events``, both of which would destroy the input: the
    first drops a repaid facility that was still part of the capitalization at
    stabilization, and the second KEEPS ONE ROW PER FACILITY AND OVERWRITES ITS
    ``dtEvent`` WITH A MATURITY DATE, which is the one value this must never
    read as an origination. Neither function is touched; this just reads
    upstream of them, which the report already did.
    """
    cc = _loan_columns(loans)
    if cc is None:
        return None, "loans frame has no date-event columns", []
    vc, lid, dtc, evc, amtc = cc
    m = loans[loans[vc].map(norm_id).isin(want)]
    if m.empty:
        return None, "no loans on record", []

    ids = list(dict.fromkeys(m[lid].tolist()))
    if len(ids) == 1:
        total = pd.to_numeric(
            m.drop_duplicates(subset=[lid])[amtc], errors="coerce"
        ).fillna(0.0).sum()
        return (float(total) or None), "the deal's only loan", []

    kind = m[dtc].astype(str).str.strip().str.lower()
    orig = m[kind == _ORIGINATION].copy()
    orig["_d"] = pd.to_datetime(orig[evc], errors="coerce")
    orig = orig.dropna(subset=["_d"])
    dated = set(orig[lid].tolist())
    missing = [i for i in ids if i not in dated]
    if missing:
        return (None,
                f"{len(missing)} of {len(ids)} loans carry no origination date",
                missing)

    first = orig["_d"].min()
    take = orig[orig["_d"] == first].drop_duplicates(subset=[lid])
    total = pd.to_numeric(take[amtc], errors="coerce").fillna(0.0).sum()
    return (float(total) or None,
            f"{len(take)} of {len(ids)} loans originated "
            f"{first.date().isoformat()}, summed", [])


def first_lien_origination_variants(
    ident: DealIdentity, loans: Optional[pd.DataFrame],
    child_vcodes: Optional[List[str]],
) -> Dict[str, Any]:
    """Both child-rollup readings of the origination rule, measured together.

    ``deal`` takes the earliest origination across the parent and its children
    as ONE date; ``property`` takes each property's own earliest and sums those.
    They differ only for a portfolio whose properties closed on different days,
    and which is right is a question for the data — so both are computed on
    every row and published. ``cfg.FIRST_LIEN_CHILD_BASIS`` names the one in use.
    """
    own = {norm_id(v) for v in (ident.vcode, ident.shadow_vcode) if v}
    kids = [norm_id(v) for v in (child_vcodes or []) if v and norm_id(v) not in own]

    deal_v, deal_how, deal_missing = first_lien_by_origination(
        loans, own | set(kids))

    per: List[Tuple[str, Optional[float], str, list]] = []
    for group in [own] + [{k} for k in kids]:
        per.append((sorted(group)[0],) + first_lien_by_origination(loans, group))
    have = [p[1] for p in per if p[1] is not None]
    prop_v = sum(have) if (have and len(have) == len(per)) else None

    return {
        "deal": {"value": deal_v, "how": deal_how, "missing": deal_missing},
        "property": {"value": prop_v,
                     "how": f"per-property earliest origination, {len(per)} "
                            f"propert{'y' if len(per) == 1 else 'ies'} summed",
                     "parts": [{"vcode": v, "value": x, "how": h}
                               for v, x, h, _ in per]},
        "in_use": cfg.FIRST_LIEN_CHILD_BASIS,
    }


def first_lien(
    ident: DealIdentity,
    loans: Optional[pd.DataFrame],
    isbs_interim_bs: Optional[pd.DataFrame],
    child_vcodes: Optional[List[str]] = None,
    diag: Optional[dict] = None,
) -> Tuple[Optional[float], str]:
    """The first mortgage at stabilization, in dollars, plus its basis.

    A DEVELOPMENT deal takes the COMMITTED facility (``loans.mOrigLoanAmt``):
    its balance sheet shows whatever has been drawn so far, which on a deal
    still building is not its capitalization and would understate the column
    by most of the loan.

    ONE GLOBAL RULE, NOT A DEV / NON-DEV SPLIT. The split this used to apply
    — development deals take the committed facility, everything else the
    earliest balance-sheet row — sounds right and the data does not support
    it. Measured against the reference on all 76 deals it scores **34/76**,
    where using the summed committed facility on EVERY deal scores **42/76**.
    The earliest balance-sheet row wins on no development deal at all and on
    only 27 of the other 66.

    So: the summed committed facility first, then the earliest loan record,
    then nothing. The earliest balance-sheet row used to be a third fallback;
    ``cfg.FIRST_LIEN_ISBS_FALLBACK`` (False) switches it off because it tied the
    reference on none of the seven deals that reached it. The order is worth
    stating, because the fallbacks are about COVERAGE and not accuracy — a
    later basis is tried only when the one before it yields no figure at all.

    CHILD PROPERTIES ARE ROLLED UP FOR LOANS AND NOT FOR ISBS. ``loans``
    records one row per property, so a portfolio's facility is only complete
    once the children are added; ``isbs_interim_bs`` already carries the
    parent's consolidated balance sheet, so adding the children counts the
    same debt twice — measured, it doubled Giant-7 ($97.0m to $194.0m), OREI
    ($34.2m to $68.4m) and Burton ($75.3m to $150.6m), and it takes the ISBS
    basis from 27/76 to 25/76.

    THIS IS STILL THE WEAKEST COLUMN IN THE REPORT and the reason is
    structural: "underwritten capitalization at stabilization" is an
    underwriting ASSUMPTION, and MRI records loans and balances, not
    assumptions. Every basis is published on the row so the gap stays visible.

    ORIGINATION FIRST, WHERE THE DATA CAN SAY SO. "First lien" means the senior
    mortgage and seniority is settled at origination, so a deal whose loans all
    carry an Origination date is answered by the loans sharing the earliest one
    — see ``first_lien_by_origination`` and the note on
    ``cfg.FIRST_LIEN_FROM_ORIGINATION``. A DEVELOPMENT deal is exempt: it takes
    the committed facility, because a construction loan's origination says when
    the draw began and not what was committed. On the live table today no deal
    with several loans reaches this path, so no printed figure moves; the ones
    that cannot answer are named in ``first_lien_origination_missing`` rather
    than falling back silently.
    """
    if cfg.FIRST_LIEN_FROM_ORIGINATION and ident.vcode not in cfg.DEV_DEALS:
        variants = first_lien_origination_variants(ident, loans, child_vcodes)
        chosen = variants.get(cfg.FIRST_LIEN_CHILD_BASIS) or variants["deal"]
        if chosen.get("value") is not None:
            return chosen["value"], f"first lien by origination — {chosen['how']}"
        if diag is not None and variants["deal"].get("missing"):
            diag.setdefault("first_lien_origination_missing", []).append({
                "vcode": ident.vcode, "name": ident.name,
                "loan_ids": [str(i) for i in variants["deal"]["missing"]],
                "reason": variants["deal"]["how"],
                "fell_back_to": cfg.FIRST_LIEN_BASIS,
            })

    withheld = None
    for basis in (cfg.FIRST_LIEN_BASIS,) + tuple(cfg.FIRST_LIEN_FALLBACKS):
        if basis == "earliest_isbs" and not cfg.FIRST_LIEN_ISBS_FALLBACK:
            # Computed so the withheld figure can be REPORTED, never printed.
            withheld = _lien_basis(basis, ident, loans, isbs_interim_bs,
                                   child_vcodes)
            continue
        value, note = _lien_basis(basis, ident, loans, isbs_interim_bs,
                                  child_vcodes)
        if value is not None:
            return value, note
    if diag is not None and withheld is not None and withheld[0] is not None:
        diag.setdefault("first_lien_isbs_withheld", []).append({
            "vcode": ident.vcode, "name": ident.name,
            "isbs_value_usd": withheld[0], "how": withheld[1],
            "reason": "no usable loan record; cfg.FIRST_LIEN_ISBS_FALLBACK is "
                      "False, so the balance-sheet figure is not printed",
        })
    return None, "none"


#: The three candidate bases, each returning (value, how it was reached).
_LIEN_BASES = ("summed_facility", "earliest_loan", "earliest_isbs")


def _lien_basis(basis, ident, loans, isbs_interim_bs, child_vcodes):
    own = {norm_id(v) for v in (ident.vcode, ident.shadow_vcode) if v}
    with_kids = own | {norm_id(v) for v in (child_vcodes or []) if v}
    if basis == "summed_facility":
        v = _loan_facility(loans, with_kids)
        return v, "loans.mOrigLoanAmt, summed per facility (children included)"
    if basis == "earliest_loan":
        v = _earliest_loan(loans, with_kids)
        return v, "loans.mOrigLoanAmt of the earliest loan record by date"
    if basis == "earliest_isbs":
        v, when = _earliest_isbs_debt(isbs_interim_bs, own)
        return v, f"earliest ISBS Interim BS debt row ({when})"
    raise ValueError(f"unknown first-lien basis {basis!r}")


def first_lien_alternates(ident, loans, isbs_interim_bs, child_vcodes, to_m):
    """Every basis, so the one in use can be checked against the others."""
    out = []
    for basis in _LIEN_BASES:
        v, note = _lien_basis(basis, ident, loans, isbs_interim_bs, child_vcodes)
        out.append({"basis": basis, "value": to_m(v), "how": note,
                    "in_use": basis == cfg.FIRST_LIEN_BASIS})
    return out


def _loan_facility(loans: Optional[pd.DataFrame], want: set) -> Optional[float]:
    if loans is None or loans.empty:
        return None
    col = "vCode" if "vCode" in loans.columns else ("vcode" if "vcode" in loans.columns else None)
    if col is None or "mOrigLoanAmt" not in loans.columns:
        return None
    m = loans[loans[col].map(norm_id).isin(want)]
    if m.empty:
        return None
    # One facility can appear once per date event (v481's Loan_Date fan-out),
    # so sum one row per LoanID rather than every row.
    if "LoanID" in m.columns:
        m = m.drop_duplicates(subset=["LoanID"])
    total = pd.to_numeric(m["mOrigLoanAmt"], errors="coerce").fillna(0.0).sum()
    return float(total) if total else None


def _earliest_loan(loans: Optional[pd.DataFrame], want: set) -> Optional[float]:
    """The committed amount on the deal's earliest loan record.

    NOTE WHAT "EARLIEST" MEANS HERE. ``dtEvent`` carries a MATURITY date on 83
    of the 91 live rows (``vDateType``) and an origination date on four, so
    this is the earliest-MATURING facility, not the first-originated one. MRI
    holds no origination date for most loans. It is measured as a candidate
    rather than presented as "the original first mortgage".
    """
    if loans is None or loans.empty:
        return None
    col = "vCode" if "vCode" in loans.columns else ("vcode" if "vcode" in loans.columns else None)
    if col is None or "mOrigLoanAmt" not in loans.columns or "dtEvent" not in loans.columns:
        return None
    m = loans[loans[col].map(norm_id).isin(want)].copy()
    if m.empty:
        return None
    m["_d"] = pd.to_datetime(m["dtEvent"], errors="coerce")
    m = m.dropna(subset=["_d"])
    if m.empty:
        return None
    sort_cols = ["_d"] + (["LoanID"] if "LoanID" in m.columns else [])
    v = _to_float(m.sort_values(sort_cols).iloc[0]["mOrigLoanAmt"])
    return v or None


#: ISBS balance-sheet accounts that carry mortgage debt. Same set as
#: ``config.DEBT_BS_ACCTS`` — imported rather than restated so the two cannot
#: drift.
def _debt_accounts() -> set:
    from config import DEBT_BS_ACCTS
    return {str(a).strip() for a in DEBT_BS_ACCTS}


def narrow_isbs_bs_for_lien(isbs: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """The balance-sheet debt rows, selected and normalised ONCE.

    THIS WAS 53 OF 64 SECONDS. `_earliest_isbs_debt` normalised the whole
    `vcode` column with a Python `map` on every call — and it is called for the
    default basis and again for each alternate, ninety-two times over 233,072
    rows: **21.6 million `norm_id` calls**, more than four fifths of the build.
    Nothing about the result changed between calls.

    So the account filter and the id normalisation happen here, once, and the
    per-deal path becomes a column comparison. The rows selected are identical
    — same account set, same `norm_id` — it is only the repetition that goes.
    """
    if isbs is None or isbs.empty:
        return isbs
    vcol = next((c for c in ("vcode", "vCode") if c in isbs.columns), None)
    acol = next((c for c in ("vAccount", "vaccount") if c in isbs.columns), None)
    if not (vcol and acol):
        return isbs
    accts = _debt_accounts()
    m = isbs[isbs[acol].astype(str).str.strip().str.split(".").str[0].isin(accts)]
    return m.assign(**{_NORM_ID_COL: m[vcol].map(norm_id)})


def _earliest_isbs_debt(isbs: Optional[pd.DataFrame], want: set) -> Tuple[Optional[float], str]:
    if isbs is None or isbs.empty:
        return None, "no ISBS data"
    df = isbs
    vcol = next((c for c in ("vcode", "vCode") if c in df.columns), None)
    acol = next((c for c in ("vAccount", "vaccount") if c in df.columns), None)
    dcol = next((c for c in ("dtEntry", "dtentry") if c in df.columns), None)
    mcol = next((c for c in ("mAmount", "mamount") if c in df.columns), None)
    if not all((vcol, acol, dcol, mcol)):
        return None, "ISBS columns missing"
    # `narrow_isbs_bs_for_lien` has usually done the account filter and the id
    # normalisation once, for the whole frame. Fall back to doing it here when
    # a caller passes a raw frame, so the function stands alone.
    pre = _NORM_ID_COL in df.columns
    m = df[df[_NORM_ID_COL].isin(want)] if pre else df[df[vcol].map(norm_id).isin(want)]
    if m.empty:
        return None, "no ISBS rows for this deal"
    if not pre:
        accts = _debt_accounts()
        m = m[m[acol].astype(str).str.strip().str.split(".").str[0].isin(accts)]
        if m.empty:
            return None, "no debt accounts on this deal's balance sheet"
    when = pd.to_datetime(m[dcol], errors="coerce", format="mixed")
    m = m.assign(_when=when).dropna(subset=["_when"])
    if m.empty:
        return None, "no parseable ISBS dates"
    first = m["_when"].min()
    take = m[m["_when"] == first]
    total = abs(pd.to_numeric(take[mcol], errors="coerce").fillna(0.0).sum())
    if not total:
        return None, "earliest balance-sheet debt row is zero"
    return float(total), first.date().isoformat()


# ══════════════════════════════════════════════════════════════════════════
# proceeds and Year-1 CoC
# ══════════════════════════════════════════════════════════════════════════
def proceeds_to_date(ident: DealIdentity, acct: pd.DataFrame, table: str,
                     as_of: Optional[_dt.date] = None) -> Optional[float]:
    """Cash returned to PSC, in dollars, through the as-of date.

    The column is headed *To-Date*, and "to date" means to the quarter being
    reported. It used to carry NO cutoff, so a 26Q2 report counted July and
    August. On a CURRENT deal it is the four distribution subtypes footnote (1)
    names; on a SOLD deal it is every distribution, because the deal is finished
    and the question is the total -- and a deal is only Sold once its sale is on
    or before the as-of, so the cutoff never removes a sale distribution.
    """
    rows = _deal_accounting(acct, ident.investment_id)
    if rows.empty:
        return None
    rows = rows[rows["InvestorID"].map(is_psc_side)]
    if as_of is not None and not rows.empty and "EffectiveDate" in rows.columns:
        when = pd.to_datetime(rows["EffectiveDate"], errors="coerce")
        rows = rows[when <= pd.Timestamp(as_of)]
    if rows.empty:
        return None
    if table == SOLD:
        m = rows[rows.get("is_distribution", pd.Series(False, index=rows.index)).fillna(False)]
    else:
        sub = pd.to_numeric(rows.get("SubtypeUID"), errors="coerce")
        m = rows[sub.isin(cfg.PROCEEDS_SUBTYPES_CURRENT)]
    if m.empty:
        return None
    return float(pd.to_numeric(m["Amt"], errors="coerce").fillna(0.0).sum())


def act_year_one_coc(
    ident: DealIdentity, acct: pd.DataFrame, funded: Optional[float],
) -> Tuple[Optional[float], str]:
    """Preferred return received in the deal's first 365 days, over funded.

    The window opens at the deal's FIRST accounting record of any kind — the
    reference workbook uses ``MINIFS`` over every row for the investment, not
    only PSC's — so a deal whose operating partner funded first is measured
    from that date.

    DENOMINATOR: funded-to-date, per the agreed data rules. **The reference
    workbook divides by the PSC Pref. Equity COMMITMENT instead**, and the two
    differ on any deal not fully drawn. Both are carried on every row under
    ``alternates.act_yr1_coc`` (``on_funded`` and ``on_commitment``), so the
    difference stays measurable rather than argued about.

    **NEITHER IS RENDERED TODAY.** ``cfg.UNLOADED_FIGURES['act_yr1_coc']`` is
    in ``"none"`` mode, so the column prints an em dash and what this function
    returns only reaches diagnostics. Flipping that one switch to ``computed``
    publishes it; flipping it to ``mri`` reads an MRI field instead.

    **THE REFERENCE'S FIGURES IN THIS COLUMN ARE NOT DERIVED, AND NO WINDOW
    REPRODUCES THEM.** The workbook carries this formula on exactly ONE of its
    76 rows; the other 75 cells are typed-in constants. Measured against the
    feed:

    * Merle Hay — the reference prints 9.2%, i.e. $690,000 of preferred return.
      Day 365 reaches $602,877 (8.0%), day 378 $660,205 and day 409 $717,534.
      No cutoff lands on the printed figure.
    * 5-15 Broad St — the reference prints 14.0%, i.e. $140,000. Every
      preferred-return payment the deal has EVER made totals $136,000, so no
      window of any length can produce it.
    * Evergreen Plaza — 12.0% needs $1,968,000; day 365 reaches $1,307,429 and
      the figure is not passed until roughly day 560.

    So the gap in this column is not a rule that needs tuning. What is computed
    here is a defined, reproducible quantity; the reference's is an editorial
    one. Both are reported.
    """
    rows = _deal_accounting(acct, ident.investment_id)
    if rows.empty:
        return None, "no accounting rows"
    dates = pd.to_datetime(rows["EffectiveDate"], errors="coerce").dropna()
    if dates.empty:
        return None, "no dated accounting rows"
    first = dates.min()
    window_end = first + pd.Timedelta(days=365)
    psc = rows[rows["InvestorID"].map(is_psc_side)]
    sub = pd.to_numeric(psc.get("SubtypeUID"), errors="coerce")
    when = pd.to_datetime(psc["EffectiveDate"], errors="coerce")
    m = psc[(sub == cfg.SUBTYPE_PREFERRED_RETURN) & (when <= window_end)]
    pref_paid = float(pd.to_numeric(m["Amt"], errors="coerce").fillna(0.0).sum()) if not m.empty else 0.0
    if not funded:
        return None, "no funded amount to divide by"
    return pref_paid / funded, (
        f"preferred return through {window_end.date().isoformat()} "
        f"(365 days from {first.date().isoformat()}) over funded-to-date"
    )


#: The ROE engine's look-forward window for a pref payment that lands late.
#: Mirrors ``one_pager.get_pe_performance``; see ``_pe_roe_events``.
ROE_GRACE_DAYS = 45


def _pe_roe_events(
    ident: DealIdentity, acct: pd.DataFrame, through: _dt.date,
) -> Tuple[List[Tuple[_dt.date, float]], List[Tuple[_dt.date, float]]]:
    """``(capital_events, cf_distributions)`` on the One Pager's definitions.

    THIS IS A TRANSCRIPTION OF ``one_pager.get_pe_performance``, NOT A SECOND
    OPINION, and the reason it is a transcription rather than a call is worth
    stating. That function takes a QUARTER STRING and derives its window from
    it; Year-1 CoC needs a window that ends on the deal's own anniversary, which
    is almost never a quarter end. The alternative was to widen
    ``get_pe_performance``'s signature — code the One Pager, the Portfolio
    Snapshot and the CoC Since Close columns all read — so the narrower change
    is to restate the classification here and PROVE the two agree.

    ``scripts/investment_metrics_check.py`` does exactly that: for a window that
    IS a quarter end it asserts this function's ROE equals
    ``get_pe_performance``'s ``roe_to_date`` to the cent, on fixtures carrying
    every branch below. A divergence fails the build rather than printing.

    The rules, each one load-bearing:

    * **PSC's side only** — an operating partner's cash is not PSC's return.
    * **Commitment rows are pledges**, not cash, and never enter.
    * **Contributions are signed**: a POSITIVE contribution row is a correction
      and must reduce capital, not add to it.
    * **Return of capital and realized gain** are capital events but not income
      — they reach ``capital_events`` and never ``cf_distributions``.
    * **An acquisition fee is in neither.** It is cash PSC received (so it is a
      proceed, and ``proceeds_to_date`` counts it) and it is not a return on
      equity.
    * **A negative CF distribution** reduces income but must NOT reach
      ``capital_events``, where it would inflate weighted average capital.
    * **The 45-day grace**: a pref payment contractually due within 30 days of
      the window close routinely lands after it. It is counted, dated AT the
      window close so the annualisation uses the boundary and not the payment
      date.
    """
    rows = _deal_accounting(acct, ident.investment_id)
    if rows.empty:
        return [], []
    r = rows.copy()
    r["_d"] = pd.to_datetime(r["EffectiveDate"], errors="coerce")
    r = r.dropna(subset=["_d"])
    if r.empty:
        return [], []
    r = r[r["InvestorID"].map(is_psc_side)]
    if "is_commitment" in r.columns:
        r = r[~r["is_commitment"].fillna(False)]
    if r.empty:
        return [], []

    r["_mt"] = r["MajorType"].fillna("").astype(str).str.strip().str.lower()
    name_col = "TypeName" if "TypeName" in r.columns else (
        "Typename" if "Typename" in r.columns else None)
    r["_tn"] = (r[name_col].fillna("").astype(str).str.strip().str.lower()
                if name_col else "")
    r["_amt"] = pd.to_numeric(r["Amt"], errors="coerce").fillna(0.0)
    r["_tid"] = (pd.to_numeric(r["TypeID"], errors="coerce").fillna(0.0)
                 if "TypeID" in r.columns else 0.0)

    capital: List[Tuple[_dt.date, float]] = []
    cf: List[Tuple[_dt.date, float]] = []

    for _, x in r[r["_d"].dt.date <= through].iterrows():
        d, a, mt, tn = x["_d"].date(), float(x["_amt"]), x["_mt"], x["_tn"]
        if "contrib" in mt:
            capital.append((d, a))
        elif "distri" in mt:
            if "return of capital" in tn or "realized gain" in tn:
                capital.append((d, a))
            elif "acquisition fee" not in tn:
                if a >= 0:
                    capital.append((d, a))
                cf.append((d, a))

    grace_end = through + _dt.timedelta(days=ROE_GRACE_DAYS)
    late = r[(r["_d"].dt.date > through) & (r["_d"].dt.date <= grace_end)]
    for _, x in late.iterrows():
        if "distri" not in x["_mt"]:
            continue
        tn = x["_tn"]
        if not (float(x["_tid"]) == 1019.0 or "preferred return" in tn
                or "pref return" in tn):
            continue
        a = float(x["_amt"])
        if a >= 0:
            capital.append((through, a))
        cf.append((through, a))

    return capital, cf


def act_year_one_coc_roe(
    ident: DealIdentity, acct: pd.DataFrame, as_of: _dt.date,
) -> Tuple[Optional[float], str]:
    """Return on equity over the deal's FIRST TWELVE MONTHS. ONE ENGINE.

    The arithmetic is ``metrics.calculate_roe_detailed`` — the same call the One
    Pager's ROE to Date goes through — and the only thing this changes is the
    window: it opens at the PSC Invest. Date and closes twelve calendar months
    later, or at the as-of date if that comes first.

    THE WINDOW OPENS AT THE EARLIER OF THE INVEST DATE AND THE FIRST CASH EVENT,
    and that is not a detail. ``deals.Acquisition_Date`` is overwritten at load
    time with the earliest accounting entry for the investment across ALL
    investors (see ``data_service._enrich_acquisition_dates``), and on six live
    deals PSC's own first contribution is dated the day BEFORE it — Evergreen
    Plaza, Giant-7, Mount Prospect, OREI, Pontchartrain and 870 Donald Lynch.
    Opening strictly on the invest date drops that contribution, which takes
    total contributions in the window to zero, which takes the whole figure to
    an em dash. Measured: starting on the invest date alone produced no figure
    at all on seven of 76 deals and sent Cocoplum to 36.0% against a reference
    4.98%, because the denominator had lost nearly all its capital.

    A PARTIAL WINDOW IS STILL RETURNED, and the caller decides. A deal under a
    year old has a year-1 CoC that is arithmetically true and useless, and
    footnote (5) is what handles it — see ``_apply_young_deal_substitution``.
    Returning the stub here and suppressing it there keeps the suppression in
    one place instead of two.
    """
    if ident.invest_date is None:
        return None, "no PSC invest date"
    end = min(plus_months(ident.invest_date, cfg.YOUNG_DEAL_MONTHS), as_of)
    capital, cf = _pe_roe_events(ident, acct, end)
    if not capital:
        return None, "no PSC capital events in the first twelve months"
    start = min(ident.invest_date, min(d for d, _ in capital))
    from metrics import calculate_roe_detailed
    detail = calculate_roe_detailed(capital, cf, start, end)
    # A ZERO WEIGHTED AVERAGE IS "NOT PRICED", NOT "RETURNED NOTHING".
    # calculate_roe_detailed zero-fills when the window holds no contribution,
    # and 0.0% is a real figure on this column — Plaza Del Mar prints it.
    if not detail.get("weighted_avg_capital"):
        return None, f"no capital at risk between {start} and {end}"
    return detail["roe"], (
        f"ROE over {start.isoformat()}..{end.isoformat()} "
        f"({cfg.YOUNG_DEAL_MONTHS} months from the PSC invest date, capped at "
        f"the as-of date), {ROE_GRACE_DAYS}-day pref grace"
    )


def realized_flows(ident: DealIdentity, acct: pd.DataFrame,
                   cutoff: Optional[_dt.date] = None) -> List[Tuple[_dt.date, float]]:
    """PSC's contributions and distributions for one investment, dated.

    ``cutoff`` is the quarter's as-of: flows dated after it are not in this
    quarter's figure.

    Commitment rows are pledges and are excluded — including them dates the
    first cash flow at the signing rather than the funding and shifts the IRR.
    """
    rows = _deal_accounting(acct, ident.investment_id)
    if rows.empty:
        return []
    rows = rows[rows["InvestorID"].map(is_psc_side)]
    if "is_commitment" in rows.columns:
        rows = rows[~rows["is_commitment"].fillna(False)]
    flows: List[Tuple[_dt.date, float]] = []
    for _, r in rows.iterrows():
        d = _to_date(r.get("EffectiveDate"))
        a = _to_float(r.get("Amt"))
        if d is None or a is None:
            continue
        if cutoff is not None and d > cutoff:
            continue
        flows.append((d, a))
    return flows


def realized_irr(ident: DealIdentity, acct: pd.DataFrame,
                 cutoff: Optional[_dt.date] = None) -> Optional[float]:
    """XIRR over PSC's contributions and distributions. ONE ENGINE: ``metrics.xirr``."""
    flows = realized_flows(ident, acct, cutoff)
    if len(flows) < 2:
        return None
    return xirr(flows)


def pooled_realized_irr(idents: List[DealIdentity], acct: pd.DataFrame,
                        cutoff: Optional[_dt.date] = None) -> Optional[float]:
    """ONE XIRR over every listed investment's PSC flows, pooled.

    This is the method of Reports > Sold Portfolio's "Portfolio Total" IRR
    (``sold_service.compute_all_sold_returns``), and of the reference workbook's
    Sold Total (18.2254% at 6/30/26 is exactly this over its 26 deals). It is NOT
    an average of the deals' IRRs, which weights a deal by its pref and gives a
    different number: the Sold Total used to be that average (20.15% at 26Q2
    against the pooled 18.14%).

    Every investment in the table goes in, including the ones whose own IRR
    cell is a dash (City West, lost to foreclosure; Adirondack, whose flows do
    not solve): footnote (2) says the foreclosure is "included in IRR
    calculations", and the pooled flows solve even where a single deal's do not.
    """
    flows: List[Tuple[_dt.date, float]] = []
    for ident in idents:
        flows.extend(realized_flows(ident, acct, cutoff))
    if len(flows) < 2:
        return None
    return xirr(flows)


# ══════════════════════════════════════════════════════════════════════════
# averages
# ══════════════════════════════════════════════════════════════════════════
def pref_weighted_average(rows: List[dict], field: str) -> Optional[float]:
    """SUMPRODUCT(field, pref) / SUM(pref), over rows that HAVE both.

    A row showing ``Dev.`` or ``Inf.`` instead of a figure contributes NOTHING
    — neither to the numerator nor to the weight. Letting a label row through
    as a zero would drag the average down by the weight of a large development
    deal while looking like an ordinary average; excluding its weight too is
    what makes the result "the average of the deals that have one".
    """
    num = 0.0
    den = 0.0
    for r in rows:
        if r.get("labels", {}).get(field):
            continue
        v = r.get(field)
        w = r.get("pref")
        if v is None or w is None:
            continue
        num += v * w
        den += w
    return (num / den) if den else None


def total_of(rows: List[dict], field: str) -> Optional[float]:
    """Sum of RAW values. Never of the rounded display figures.

    The reference's Grand Total foots from raw: its First Lien total is
    ``$2,668.8`` where the two displayed subtotals add to ``$2,668.9``.
    """
    vals = [r[field] for r in rows if r.get(field) is not None]
    return sum(vals) if vals else None


# ══════════════════════════════════════════════════════════════════════════
# the report
# ══════════════════════════════════════════════════════════════════════════
def build_investment_metrics(
    inv: pd.DataFrame,
    acct: pd.DataFrame,
    *,
    commitments: Optional[pd.DataFrame] = None,
    deal_terms: Optional[pd.DataFrame] = None,
    loans: Optional[pd.DataFrame] = None,
    isbs_interim_bs: Optional[pd.DataFrame] = None,
    waterfalls: Optional[pd.DataFrame] = None,
    isbs_raw: Optional[pd.DataFrame] = None,
    as_of: Optional[_dt.date] = None,
    today: Optional[_dt.date] = None,
) -> Dict[str, Any]:
    """Assemble the Current and Sold tables, their totals and the grand total."""
    as_of = as_of or default_as_of(today)

    # ── done ONCE, not once per deal ──────────────────────────────────────
    # Both of these are pure narrowing: the rows every consumer below sees are
    # the rows it would have selected for itself. Measured on frames at live
    # row counts, they take the build from ~90s to a few seconds.
    if acct is not None and not acct.empty and _NORM_ID_COL not in acct.columns:
        acct = acct.assign(**{_NORM_ID_COL: acct["InvestmentID"].map(norm_id)})
    isbs_pe = narrow_isbs_for_pe(isbs_raw)
    isbs_interim_bs = narrow_isbs_bs_for_lien(isbs_interim_bs)
    if (commitments is not None and not commitments.empty
            and "EntityID" in commitments.columns
            and _NORM_ID_COL not in commitments.columns):
        commitments = commitments.assign(
            **{_NORM_ID_COL: commitments["EntityID"].map(norm_id)})

    identities, diag = resolve_deal_identities(inv)
    # Every downstream engine reads the RESOLVED frame, not the raw one — see
    # resolved_inv_frame for why a twin is invisible to the PE engine otherwise.
    inv_resolved = resolved_inv_frame(inv, identities)

    dt_index = _deal_terms_index(deal_terms)
    rows_by_table: Dict[str, List[dict]] = {CURRENT: [], SOLD: []}
    ident_by_vcode: Dict[str, DealIdentity] = {}

    for ident in identities:
        # Excluded BY NAME, with the reason, and reported. A deal the
        # reference carries in neither table cannot be reached by any
        # population rule — "sold but not yet moved across" is a judgement
        # about final distributions, not a state MRI records — so the
        # alternative to naming it is a rule contorted until it happens to
        # drop this one deal, which nobody could later read.
        # Dated: before ``from`` the reason was not yet true, so the deal is
        # reported like any other at that quarter.
        excluded = cfg.EXCLUDED_DEALS.get(ident.vcode)
        if excluded and as_of >= _dt.date.fromisoformat(excluded["from"]):
            diag.setdefault("excluded_deals", []).append(
                {"vcode": ident.vcode, "name": ident.name,
                 "from": excluded["from"], "reason": excluded["reason"]})
            continue
        # QUARTER INTEGRITY: a deal not yet invested at the as-of is not in that
        # quarter's report. Reported, not dropped silently.
        if ident.invest_date is not None and ident.invest_date > as_of:
            diag.setdefault("not_yet_invested", []).append(
                {"vcode": ident.vcode, "name": ident.name,
                 "invest_date": ident.invest_date.isoformat(),
                 "as_of": as_of.isoformat()})
            continue
        table = classify(ident, as_of)
        if table == CURRENT and ident.is_sold:
            diag.setdefault("sold_after_as_of_shown_current", []).append(
                {"vcode": ident.vcode, "name": ident.name,
                 "sale_date": ident.sale_date.isoformat() if ident.sale_date else None,
                 "as_of": as_of.isoformat(),
                 "reason": "carries the SOLD marker but the sale is after the "
                           "as-of: held at the quarter end, so shown as Current"})
        ident_by_vcode[ident.vcode] = ident
        rows_by_table[table].append(
            _build_row(ident, table, as_of, acct, commitments, dt_index,
                       loans, isbs_interim_bs, inv_resolved, isbs_pe,
                       waterfalls, diag)
        )

    out: Dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "as_of_display": _display_as_of(as_of),
        "fx_rate": cfg.CAD_TO_USD,
        "fx_rate_workbook": cfg.CAD_TO_USD_WORKBOOK,
        "units_note": cfg.UNITS_NOTE,
        "disclaimer": cfg.DISCLAIMER,
        "page": cfg.PAGE,
        # The draft gate travels ON THE PAYLOAD rather than being read again
        # client-side, so the screen, the printed sheet and the sidebar cannot
        # disagree about whether this report is signed off.
        "draft": bool(cfg.INVESTMENT_METRICS_DRAFT),
        "draft_banner": cfg.DRAFT_BANNER,
        "draft_mark": cfg.DRAFT_MARK,
        "diagnostics": diag,
    }

    for table, order, title, footnotes in (
        (CURRENT, cfg.ROW_ORDER_CURRENT, cfg.TITLE_CURRENT,
         cfg.FOOTNOTES_CURRENT),
        (SOLD, cfg.ROW_ORDER_SOLD, cfg.TITLE_SOLD, cfg.FOOTNOTES_SOLD),
    ):
        rows = _ordered(rows_by_table[table], order, diag, table)
        other_table = SOLD if table == CURRENT else CURRENT
        other_vcodes = [{"vcode": r["vcode"]} for r in rows_by_table[other_table]]
        for r in rows:
            r["markers"] = row_markers(ident_by_vcode[r["vcode"]], table, as_of)
            _apply_young_deal_substitution(r, diag)
        _check_config_population(rows, table, footnotes, diag, other_vcodes)
        total = _total_row(rows)
        if table == SOLD:
            # The Sold Total's realized IRR is the POOLED XIRR (the Reports
            # method), not the pref-weighted average `_total_row` computed.
            total["realized_irr"] = pooled_realized_irr(
                [ident_by_vcode[r["vcode"]] for r in rows], acct, as_of)
            diag["sold_total_realized_irr"] = {
                "method": "pooled XIRR over the Sold table's PSC flows (Reports > Sold Portfolio method)",
                "deals": len(rows)}
        out[table] = {
            "title": title,
            "rows": rows,
            "total": total,
            "footnotes": [{"n": n, "text": t} for n, t in footnotes],
            # The view draws the table from this, so the headings, the column
            # widths, the alignment and the vertical rules all come off the
            # server. The alternative is a second copy of the reference's
            # geometry in TypeScript, and two copies drift.
            "columns": [
                {"key": k, "row1": r1, "row2": r2, "row3": r3,
                 "width": w, "align": a}
                for k, r1, r2, r3, w, a in
                (cfg.COLUMNS_CURRENT if table == CURRENT else cfg.COLUMNS_SOLD)
            ],
            "cap_group": {"heading": cfg.CAP_GROUP_HEADING,
                          **cfg.CAP_GROUP_SPAN},
            "cap_pairs": cfg.CAP_PAIR_HEADINGS,
            "vertical_rules": (cfg.VERTICAL_RULES_CURRENT if table == CURRENT
                               else cfg.VERTICAL_RULES_SOLD),
        }
    out[SOLD]["total_markers"] = cfg.SOLD_TOTAL_MARKERS
    all_rows = out[CURRENT]["rows"] + out[SOLD]["rows"]
    _check_config_population_any(all_rows, diag)
    for vcode in cfg.EXCLUDED_DEALS:
        if vcode not in {i.vcode for i in identities}:
            diag.setdefault("config_entries_without_a_deal", []).append(
                {"config": "EXCLUDED_DEALS", "vcode": vcode,
                 "reason": "named for exclusion but no such deal exists"})
    out["grand_total"] = _grand_total(all_rows)
    return out


def resolve_unloaded(key: str, terms: dict, computed: Dict[str, Optional[float]],
                     diag: dict, vcode: str,
                     columns: Optional[frozenset] = None) -> Tuple[Optional[float], str]:
    """What to PRINT for a column whose source Alay has not loaded yet.

    One switch, in one place (``cfg.UNLOADED_FIGURES``). Three columns go
    through here and none of them decides for itself, so turning one on when
    the data arrives is a config edit rather than a hunt through the engine.

    Returns ``(value, basis)``. ``value`` is None whenever the source is
    absent — never 0.0, which would print as a real zero return.

    A ``field`` that is named but missing from the table is reported rather
    than treated as NULL: "Alay has not loaded it" and "the column name in the
    config is wrong" are different problems and only one of them is waiting on
    somebody else.
    """
    spec = cfg.UNLOADED_FIGURES.get(key)
    if spec is None:
        return computed.get("default"), "computed"

    mode = spec.get("mode", "none")
    if mode == "computed":
        variant = spec.get("variant", "default")
        return computed.get(variant), f"computed ({variant})"

    if mode == "mri":
        field = spec.get("field")
        if not field:
            diag.setdefault("unloaded_figure_misconfigured", []).append(
                {"column": key, "reason": "mode is 'mri' but no field is named"})
            return None, "no MRI field named"
        # `columns` is the table's column set. A deal with no row at all in a
        # table that HAS the column is NULL for that deal, not an absent column.
        if field not in terms and field not in (columns or ()):
            diag.setdefault("unloaded_figure_field_absent", []).append(
                {"column": key, "table": spec.get("table"), "field": field,
                 "vcode": vcode})
            return None, f"{spec.get('table')}.{field} is not present"
        value = _as_rate(terms.get(field))
        if value is None:
            # Present but empty: MRI holds no figure for THIS deal. Distinct
            # from "absent" above, which means the table has not been
            # refreshed with the column at all.
            nulls = diag.setdefault("unloaded_figure_value_null", {}).setdefault(
                key, {"table": spec.get("table"), "field": field, "vcodes": []})
            nulls["vcodes"].append(vcode)
            return None, f"{spec.get('table')}.{field} is NULL for this deal"
        return value, f"{spec.get('table')}.{field}"

    # mode == "none" — pending, and deliberately not rendered.
    diag.setdefault("unloaded_figures_pending", {}).setdefault(
        key, {"label": spec.get("label"), "note": spec.get("note"),
              "deals": 0})["deals"] += 1
    return None, f"pending Alay — {spec.get('note')}"


def _apply_young_deal_substitution(row: dict, diag: dict) -> None:
    """Do what footnote (5) says, rather than only printing it.

    A deal three months old has an actual cash-on-cash return that is
    arithmetically true and useless — it annualises a stub period — so the
    reference substitutes the PROJECTED year-1 figure and says so in a
    footnote. Printing the footnote while leaving the actual figure in place
    would be the worst of both: the note tells the reader one thing and the cell
    shows another.

    THE RULE IS THE DATE TEST, NOT THE MARKER. ``row['young_deal']`` is set from
    ``is_young_deal``; the marker is derived from the same test, so the two
    cannot disagree, and a deal ages out of the rule on its own rather than when
    somebody remembers to edit a list.

    PRECEDENCE: ``Dev.`` > ``Lease up`` > this rule > the computed figure. A
    cell carrying a label in ``cfg.CELL_LABELS_*`` is left completely alone —
    Trolley Square and Jefferson Stephens are both development deals AND under a
    year old, and the reference prints ``Dev.`` for them, not a CoC.

    WITH NOTHING TO SUBSTITUTE, THE CELL IS BLANKED. This is the one judgement
    here and it reverses what this function used to do. Where ``proj_yr1_coc``
    is absent or NULL (MRI holds no figure for the deal, or the column has not
    been refreshed in yet) there is no projected figure to put in — and
    the alternative is to leave a stub-period actual sitting under a heading the
    footnote has just told the reader means something else. Presidential Arms
    funded in May and its first twelve months close in May 2027; its ROE over
    seven weeks is not a year-1 return, and 8.0% on the page is a number
    somebody will quote. An em dash says the app has no figure, which is true.

    Every substitution that could not be made is still recorded in
    ``young_deal_substitution_unavailable``, so the gap stays countable, and the
    moment ``proj_yr1_coc`` is switched on the projected figure flows into these
    same three cells with no further change.
    """
    if not row.get("young_deal"):
        return
    src = row.get("proj_yr1_coc")
    labels = row.get("labels") or {}
    fields = [f for f in cfg.YOUNG_DEAL_SUBSTITUTED_COLUMNS if not labels.get(f)]
    if not fields:
        return
    note = "(6)" if 6 in (row.get("markers") or ()) else "(5)"
    if src is None:
        for f in fields:
            row[f] = None
            row.setdefault("basis", {})[f] = (
                f"under a year of operating history — footnote {note} "
                f"substitutes the projected Year-1 CoC, which is not loaded")
        diag.setdefault("young_deal_substitution_unavailable", []).append(
            {"vcode": row["vcode"], "name": row["name"],
             "markers": row.get("markers"), "columns": list(fields),
             "reason": "projected Year-1 CoC is not loaded into MRI; the "
                       "cells are blank rather than showing a partial year"})
        return
    for f in fields:
        row[f] = src
        row.setdefault("basis", {})[f] = (
            f"projected Year-1 CoC, substituted under footnote {note}")


#: Every config dict keyed by vcode that this report reads, by table.
_VCODE_CONFIGS = {
    CURRENT: (("ROW_MARKERS_CURRENT", "ROW_MARKERS_CURRENT"),
              ("CELL_LABELS_CURRENT", "CELL_LABELS_CURRENT"),
              ("ROW_ORDER_CURRENT", "ROW_ORDER_CURRENT")),
    SOLD: (("ROW_MARKERS_SOLD", "ROW_MARKERS_SOLD"),
           ("CELL_LABELS_SOLD", "CELL_LABELS_SOLD"),
           ("ROW_ORDER_SOLD", "ROW_ORDER_SOLD"),
           ("REALIZED_IRR_SUPPRESSED", "REALIZED_IRR_SUPPRESSED")),
}

#: Keyed by vcode but not by table — checked against BOTH populations at once.
_VCODE_CONFIGS_ANY = ("DEV_DEALS", "LEASE_UP_DEALS", "EXCLUDED_DEALS")


def _check_config_population(rows: List[dict], table: str, footnotes,
                             diag: dict,
                             other_rows: Optional[List[dict]] = None) -> None:
    """A transcribed vcode that no longer names a deal, and a marker with no note.

    A REPORTED FINDING, NOT AN ERROR. Every entry in this file's hand-maintained
    config was correct the day it was written, and the way it goes wrong is
    silent: a deal is renamed, re-keyed, sold, or dropped from MRI, and the
    entry simply stops matching. Nothing on the page says so — the footnote
    marker just stops printing, or the ``Dev.`` label quietly becomes a number.
    That is precisely the failure mode this whole file exists to avoid, so the
    config is checked against the population it claims to describe, every build.

    Two questions, both answered against THIS table's rows:

    * does every vcode named in the config appear in the report's population?
    * does every footnote number printed after a name exist in this table's
      footnote list?

    The second is not hypothetical: the Current and Sold pages have different
    footnote (2)s and different (4)s, and a marker moved from one table's config
    to the other would print a number that refers to the wrong note or to no
    note at all.
    """
    # A deal the quarter put in the OTHER table is not a stale entry: its config
    # is still correct, it is simply not on this page this quarter.
    present = {r["vcode"] for r in rows} | {r["vcode"] for r in (other_rows or [])}
    known = {n for n, _ in footnotes}
    stale: List[dict] = []
    for attr, label in _VCODE_CONFIGS[table]:
        for vcode in getattr(cfg, attr, {}) or {}:
            if vcode not in present:
                stale.append({"config": label, "vcode": vcode, "table": table})
    for r in rows:
        unknown = [n for n in (r.get("markers") or []) if n not in known]
        if unknown:
            stale.append({"config": "row markers", "vcode": r["vcode"],
                          "table": table, "name": r["name"],
                          "footnotes_not_in_this_table": unknown})
    if stale:
        diag.setdefault("config_entries_without_a_deal", []).extend(stale)


def _check_config_population_any(all_rows: List[dict], diag: dict) -> None:
    """The table-agnostic vcode configs, against the whole population."""
    present = {r["vcode"] for r in all_rows}
    stale = [{"config": attr, "vcode": v}
             for attr in _VCODE_CONFIGS_ANY
             for v in (getattr(cfg, attr, ()) or ())
             # EXCLUDED_DEALS is the one config whose vcodes are SUPPOSED to be
             # absent from the tables — it is what removed them. It is checked
             # against the identities instead, by the caller.
             if attr != "EXCLUDED_DEALS" and v not in present]
    if stale:
        diag.setdefault("config_entries_without_a_deal", []).extend(stale)


def _display_as_of(d: _dt.date) -> str:
    """``30-Jun-26``, as the reference prints it."""
    return f"{d.day:02d}-{d.strftime('%b')}-{d.strftime('%y')}"


class _TermsIndex(dict):
    """``{vcode: row}`` that remembers which COLUMNS the table has, so a deal
    with no row can be told apart from a column that was never loaded."""
    columns: frozenset = frozenset()


def _deal_terms_index(deal_terms: Optional[pd.DataFrame]) -> Dict[str, dict]:
    if deal_terms is None or deal_terms.empty:
        return {}
    out = _TermsIndex()
    out.columns = frozenset(deal_terms.columns)
    for _, r in deal_terms.iterrows():
        out[norm_id(r.get("vcode"))] = r.to_dict()
    return out


def _as_rate(value: Any) -> Optional[float]:
    """Contract rates are stored as either 0.085 or 8.5. Both mean 8.5%."""
    v = _to_float(value)
    if v is None:
        return None
    return v if abs(v) < 1 else v / 100.0


def _blank(value: Any) -> bool:
    """MRI's spellings of "no value": None, NaN, and the empty-ish strings."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() in ("", "nan", "None", "NaT")


def _identity_terms(ident, dt_index, diag: dict) -> dict:
    """The deal terms for a LOGICAL investment, across all of its vcodes.

    An investment carried as two vcodes can hold different fields of its terms
    on each: Donald Lynch keeps ``uw_irr`` and ``proj_yr1_coc`` under
    ``P0000049`` and ``pe_coupon``, ``irr_lookback`` and ``pe_split_capital``
    under its sold twin ``P0000073``. Reading only ``ident.vcode`` printed three
    dashes for terms MRI does hold, and the MRI gaps file listed them as
    missing. The identity merge exists so that one investment is read as one;
    this makes the terms follow it, as the loan and the accounting already do.

    THE PRIMARY VCODE WINS. A twin only fills a field the primary leaves blank,
    so a deal with one vcode, or whose twin says nothing, reads exactly as
    before. Where both carry a value and they differ, the primary stands and
    the disagreement is NAMED in diagnostics, never resolved silently.

    Measured on production over every investment with more than one vcode (20):
    one fills anything, Donald Lynch, four fields; none conflicts.
    """
    primary = dt_index.get(norm_id(ident.vcode), {})
    twins = [v for v in (getattr(ident, "vcodes", None) or [])
             if norm_id(v) != norm_id(ident.vcode)]
    if not twins:
        return primary
    merged = dict(primary)
    for v in twins:
        row = dt_index.get(norm_id(v))
        if not row:
            continue
        for field, value in row.items():
            if field == "vcode" or _blank(value):
                continue
            if _blank(merged.get(field)):
                merged[field] = value
                diag.setdefault("terms_from_twin", []).append(
                    {"vcode": ident.vcode, "field": field, "from_vcode": v,
                     "value": str(value)})
            elif str(merged[field]) != str(value):
                diag.setdefault("terms_twin_conflict", []).append(
                    {"vcode": ident.vcode, "field": field, "kept": str(merged[field]),
                     "twin_vcode": v, "twin_value": str(value)})
    return merged


def _build_row(ident, table, as_of, acct, commitments, dt_index, loans,
               isbs_interim_bs, inv, isbs_raw, waterfalls, diag) -> dict:
    from one_pager import _child_vcodes_for_parent

    try:
        children = _child_vcodes_for_parent(ident.vcode, inv)
    except Exception:
        children = []

    # Computed ONCE and shared: `pref_and_first_loss` consumes this list and
    # the row publishes it, and building it twice per deal was measurably the
    # third-largest cost in the profile.
    cap_alts = capitalization_sources(ident, acct, commitments, as_of=as_of)
    pref_usd, floss_usd, cap_basis = pref_and_first_loss(
        ident, acct, commitments, sources=cap_alts)
    lien_usd, lien_basis = first_lien(ident, loans, isbs_interim_bs, children,
                                      diag=diag)
    lien_orig = first_lien_origination_variants(ident, loans, children)

    fx = cfg.CAD_TO_USD if norm_id(ident.currency) == "CAD" else 1.0
    to_m = lambda v: None if v is None else (v * fx) / MILLION  # noqa: E731

    lien_alts = first_lien_alternates(ident, loans, isbs_interim_bs, children, to_m)

    pref = to_m(pref_usd)
    first_lien_m = to_m(lien_usd)
    first_loss = to_m(floss_usd)

    # THE TOTAL IS ONLY A TOTAL IF EVERY PIECE IS KNOWN. Summing the pieces
    # that happen to be present treats an unknown first lien as a zero, and
    # the damage is not confined to one cell: the total shrinks to the equity,
    # and every "% of Cap." is then a share of that, so a sold deal with no
    # balance-sheet history came out reading "PSC Pref. Equity — 100.0% of
    # capitalization" on a deal that was 70% levered. Each of those cells looks
    # perfectly ordinary on its own.
    #
    # A genuine zero is different and still counts: Barnbeck really did carry
    # no first lien, and the reference prints $0.0 / 0.0% for it.
    pieces = (first_lien_m, pref, first_loss)
    total_size = sum(pieces) if all(p is not None for p in pieces) else None
    pct = lambda v: (None if (v is None or not total_size) else v / total_size)  # noqa: E731

    funded = _funded_to_date(ident, acct, as_of)
    proceeds = to_m(proceeds_to_date(ident, acct, table, as_of))
    act_yr1_roe, yr1_roe_basis = act_year_one_coc_roe(ident, acct, as_of)
    act_yr1, yr1_basis = act_year_one_coc(ident, acct, funded)
    act_yr1_on_commit = (None if (pref_usd in (None, 0) or funded in (None, 0))
                         else (act_yr1 * funded / pref_usd if act_yr1 is not None else None))

    roe, uw_roe = _coc_since_close(ident, as_of, acct, waterfalls, inv, isbs_raw)

    terms = _identity_terms(ident, dt_index, diag)
    # A label (Dev., N/A, Inf.) describes the DEAL, so it follows the deal when a
    # quarter puts it in the other table than the reference does. The table's
    # own entry wins; the other table's is the fallback.
    own, other = ((cfg.CELL_LABELS_CURRENT, cfg.CELL_LABELS_SOLD) if table == CURRENT
                  else (cfg.CELL_LABELS_SOLD, cfg.CELL_LABELS_CURRENT))
    labels = own.get(ident.vcode) or other.get(ident.vcode) or {}

    # The three columns with no source in the app go through ONE switch. The
    # derived Year-1 figures are computed either way and kept below in
    # `alternates`, so turning the column on later needs no new arithmetic.
    dt_columns = getattr(dt_index, "columns", None)
    uw_irr_v, uw_irr_basis = resolve_unloaded(
        "uw_irr", terms, {}, diag, ident.vcode, dt_columns)
    proj_yr1_v, proj_yr1_basis = resolve_unloaded(
        "proj_yr1_coc", terms, {}, diag, ident.vcode, dt_columns)
    act_yr1_v, act_yr1_mode = resolve_unloaded(
        "act_yr1_coc", terms,
        {"roe_window": act_yr1_roe, "funded": act_yr1,
         "commitment": act_yr1_on_commit, "default": act_yr1_roe},
        diag, ident.vcode)

    row = {
        "vcode": ident.vcode,
        "investment_id": ident.investment_id,
        "name": ident.name,
        "asset_class": ident.asset_class,
        "dma": _dma(ident),
        "invest_date": ident.invest_date.isoformat() if ident.invest_date else None,
        "invest_date_display": _month_year(ident.invest_date),
        "partner": ident.partner,
        "currency": ident.currency,
        "total_size": total_size,
        "first_lien": first_lien_m,
        "first_lien_pct": pct(first_lien_m),
        "pref": pref,
        "pref_pct": pct(pref),
        "first_loss": first_loss,
        "first_loss_pct": pct(first_loss),
        # These three come from `cfg.UNLOADED_FIGURES`, not from the engine's
        # own opinion. None, never 0 — a zero would print as a real 0.0%.
        "uw_irr": uw_irr_v,
        "proceeds": proceeds,
        "proj_yr1_coc": proj_yr1_v,
        "act_yr1_coc": act_yr1_v,
        "proj_coc_since_close": uw_roe,
        "act_coc_since_close": roe,
        "pref_coupon": _as_rate(terms.get("pe_coupon")),
        # The reference's column is headed "Residual CF Split" and the value
        # comes from pe_split_CAPITAL: pe_split_cf is NULL on all 81 rows of
        # the live table, so the CF column has never had a CF source. Flagged,
        # not silently relabelled.
        "residual_cf_split": _as_rate(terms.get("pe_split_capital")),
        "residual_cf_split_source": "deal_terms.pe_split_capital",
        "irr_lookback": _as_rate(terms.get("irr_lookback")),
        "labels": labels,
        # Footnote (5)'s population, from the DATE and not from a list. The
        # marker is derived from the same test in `row_markers`, so the note
        # printed after the name and the cells that get blanked cannot disagree.
        "young_deal": is_young_deal(ident, as_of),
        "basis": {
            "capitalization": cap_basis,
            "first_lien": lien_basis,
            "act_yr1_coc": act_yr1_mode,
            "uw_irr": uw_irr_basis,
            "proj_yr1_coc": proj_yr1_basis,
            "fx": (f"CAD converted at {cfg.CAD_TO_USD}" if fx != 1.0 else "USD"),
        },
        # What the sources this report did NOT use would have said. Published,
        # not discarded: the first-lien and capitalization rules were chosen
        # ahead of the data, and these are what make the choice reviewable.
        "alternates": {
            # THE YEAR-1 FIGURES ARE COMPUTED AND KEPT, NOT SHOWN. The column
            # prints an em dash (see cfg.UNLOADED_FIGURES) because the
            # reference's figures in it are not reproducible from the feed
            # under any window, so a derived number under the same heading
            # would be read as the same quantity. Both defined denominators
            # are here so the question stays measurable: `funded` is the
            # agreed data rule, `commitment` is what the reference workbook
            # divides by.
            "act_yr1_coc": {
                "on_roe_window": act_yr1_roe,
                "roe_basis": yr1_roe_basis,
                "on_funded": act_yr1,
                "on_commitment": act_yr1_on_commit,
                "basis": yr1_basis,
            },
            # Both child-rollup readings of the origination rule — see
            # `first_lien_origination_variants`. Published on every row so the
            # default (`deal`) can be checked against the alternative rather
            # than argued about.
            "first_lien_origination": lien_orig,
            # ALL THREE bases, every row, with the one in use flagged. The
            # column is the report's weakest and the choice between them was
            # made on a count — publishing only the runner-up would hide the
            # basis that happens to be right for this deal.
            "first_lien": lien_alts,
            "capitalization": [
                {"basis": label, "pref": to_m(p), "first_loss": to_m(o)}
                for label, p, o in cap_alts
            ],
        },
        "funded_to_date": None if funded is None else funded / MILLION,
        "sale_date": ident.sale_date.isoformat() if ident.sale_date else None,
        "twin_vcodes": ident.vcodes,
    }

    if not terms:
        diag.setdefault("missing_deal_terms", []).append(
            {"vcode": ident.vcode, "name": ident.name})

    if table == SOLD:
        if ident.vcode in cfg.REALIZED_IRR_SUPPRESSED:
            row["realized_irr"] = None
            row["basis"]["realized_irr"] = cfg.REALIZED_IRR_SUPPRESSED[ident.vcode]
        else:
            row["realized_irr"] = realized_irr(ident, acct, as_of)
            row["basis"]["realized_irr"] = "XIRR over PSC contributions and distributions"
    return row


def _dma(ident: DealIdentity) -> Optional[str]:
    """The market the deal sits in.

    THE REFERENCE PRINTS A MARKET AND MRI STORES A CITY, and they are not the
    same field: the reference reads ``Springfield`` where MRI's city is
    ``Lake Ozark``, ``Chicago`` for ``Evergreen Park``, ``New York`` for
    ``Stamford``, ``Washington DC`` for ``Adelphi``. The workbook's own legend
    names its source as ``MSA``, a One-Pager field that has no column in
    ``deals`` and no table behind it.

    So the city is what the app can honestly say, and the difference is
    reported rather than papered over with a lookup somebody would have to
    maintain by hand.
    """
    city = str(ident.city or "").strip()
    state = str(ident.state or "").strip()
    if city and state and state.lower() not in city.lower():
        return f"{city}, {state}"
    return city or state or None


def _month_year(d: Optional[_dt.date]) -> Optional[str]:
    return None if d is None else f"{d.strftime('%b')}-{d.year}"


def _funded_to_date(ident: DealIdentity, acct: pd.DataFrame,
                    as_of: _dt.date) -> Optional[float]:
    """PSC capital in, through the as-of date.

    The same definition ``one_pager.get_pe_performance`` uses — contributions
    are signed, so a positive contribution row (a correction) reduces funded
    rather than adding to it.
    """
    rows = _deal_accounting(acct, ident.investment_id)
    if rows.empty:
        return None
    rows = rows[rows["InvestorID"].map(is_psc_side)]
    if "is_contribution" not in rows.columns:
        return None
    rows = rows[rows["is_contribution"].fillna(False)]
    when = pd.to_datetime(rows["EffectiveDate"], errors="coerce")
    rows = rows[when <= pd.Timestamp(as_of)]
    if rows.empty:
        return None
    total = -float(pd.to_numeric(rows["Amt"], errors="coerce").fillna(0.0).sum())
    return total if total else None


def _coc_since_close(ident, as_of, acct, waterfalls, inv, isbs_raw):
    """Act. and Proj. CoC since close, from the One Pager's PE engine.

    ONE ENGINE. ``get_pe_performance`` is what the One Pager renders as ROE to
    Date and U/W ROE to Date; calling it here means the two screens cannot show
    different numbers for the same deal, which is the whole point of the rule.

    IT IS HANDED ONLY THIS DEAL'S ACCOUNTING ROWS, and that is equivalence
    rather than a shortcut: the engine's first act is to derive
    ``deal_investment_ids`` from the frame it was given and filter the feed to
    ``isin(deal_investment_ids)``, twice — once for the period and once for the
    45-day grace window. Pre-filtering to the same InvestmentID makes that
    filter a no-op. What it avoids is the engine's ``acct.copy()`` and its
    ``to_datetime`` over the WHOLE 13,000-row feed, once per deal.
    """
    from one_pager import get_pe_performance

    quarter = f"{as_of.year}-Q{(as_of.month - 1) // 3 + 1}"
    deal_acct = _deal_accounting(acct, ident.investment_id)
    try:
        pe = get_pe_performance(
            ident.vcode, quarter, deal_acct, waterfalls, inv,
            isbs_raw=isbs_raw, deal_terms=None,
        )
    except Exception:
        return None, None
    # A ZERO IS ONLY A ZERO WHEN THE ENGINE ACTUALLY RAN. ``get_pe_performance``
    # initialises both figures to 0.0 and every path that finds no data leaves
    # them at 0.0, so the scalar alone cannot tell "this deal has returned
    # nothing yet" from "this deal was never priced" — and the first prints as
    # a real 0.0% while the second must print as a dash.
    #
    # ``roe_components`` is the witness: the engine writes it ONLY inside the
    # branch that computed the figure. Present -> the number is real, zero
    # included. Absent -> nothing was computed.
    roe = pe.get("roe_to_date") if pe.get("roe_components") else None
    uw = pe.get("uw_roe_to_date") if pe.get("uw_roe_components") else None
    return roe, uw


def _ordered(rows: List[dict], order: List[str], diag: dict, table: str) -> List[dict]:
    """Reference order first, then anything new by investment date.

    A deal the reference has never carried is APPENDED and named in
    diagnostics, not dropped — a report that silently omits a new deal is
    indistinguishable from one that has none.
    """
    index = {v: i for i, v in enumerate(order)}
    known = [r for r in rows if r["vcode"] in index]
    extra = [r for r in rows if r["vcode"] not in index]
    known.sort(key=lambda r: index[r["vcode"]])
    extra.sort(key=lambda r: (r["invest_date"] or "9999"))
    # EXACTLY the deals `classify` moved from Sold to Current this quarter -- no
    # broader. They are still in the reference's order (in the Sold list), so
    # appending them here is not "a deal the reference has never carried", and
    # their absence from the Sold list is not a missing row; the diagnostic
    # `sold_after_as_of_shown_current` already names them. A deal the reference
    # puts in one table and the data puts in the other, for any other reason, is
    # still reported.
    moved = {x["vcode"] for x in diag.get("sold_after_as_of_shown_current", [])}
    reported = [r for r in extra if r["vcode"] not in moved]
    if reported:
        diag.setdefault("not_in_reference_order", []).extend(
            {"table": table, "vcode": r["vcode"], "name": r["name"]} for r in reported)
    not_yet = {x["vcode"] for x in diag.get("not_yet_invested", [])}
    missing = [v for v in order
               if v not in {r["vcode"] for r in rows}
               and v not in not_yet and v not in moved]
    if missing:
        diag.setdefault("reference_rows_absent", []).extend(
            {"table": table, "vcode": v} for v in missing)
    return known + extra


_SUM_FIELDS = ("total_size", "first_lien", "pref", "first_loss", "proceeds")
_AVG_FIELDS = ("first_lien_pct", "pref_pct", "first_loss_pct", "uw_irr",
               "realized_irr", "proj_yr1_coc", "act_yr1_coc",
               "proj_coc_since_close", "act_coc_since_close", "pref_coupon",
               "residual_cf_split", "irr_lookback")


def _total_row(rows: List[dict]) -> dict:
    out = {"label": cfg.TOTAL_LABEL}
    for f in _SUM_FIELDS:
        out[f] = total_of(rows, f)
    for f in _AVG_FIELDS:
        out[f] = pref_weighted_average(rows, f)
    # The three % of Cap. cells foot from the RAW totals rather than being an
    # average of the rows' percentages — they are a share of one total, and an
    # average of shares is a different number.
    if out.get("total_size"):
        for f, src in (("first_lien_pct", "first_lien"), ("pref_pct", "pref"),
                       ("first_loss_pct", "first_loss")):
            out[f] = (None if out.get(src) is None
                      else out[src] / out["total_size"])
    return out


def _grand_total(all_rows: List[dict]) -> dict:
    """Current + Sold, from RAW values. Only the columns the reference prints."""
    out = {"label": cfg.GRAND_TOTAL_LABEL}
    for f in _SUM_FIELDS:
        out[f] = total_of(all_rows, f)
    if out.get("total_size"):
        for f, src in (("first_lien_pct", "first_lien"), ("pref_pct", "pref"),
                       ("first_loss_pct", "first_loss")):
            out[f] = (None if out.get(src) is None
                      else out[src] / out["total_size"])
    return out
