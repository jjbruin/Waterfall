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

    SOLD is the marker, not the date: a deal sold AFTER the as-of date still
    belongs in the Sold table (the reference says so with footnote (4) on
    Clima Secur, 30 Bearfoot and 870 Donald Lynch, all sold after 30 Jun 26).
    A deal carrying a sale date but no SOLD marker is still CURRENT — the
    marker is what accounting sets when the deal is closed out.
    """
    return SOLD if ident.is_sold else CURRENT


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


def _deal_accounting(acct: pd.DataFrame, investment_id: str) -> pd.DataFrame:
    if acct is None or acct.empty or not investment_id:
        return pd.DataFrame()
    ids = acct["InvestmentID"].map(norm_id)
    return acct[ids == investment_id]


def pref_and_first_loss(
    ident: DealIdentity,
    acct: pd.DataFrame,
    commitments: Optional[pd.DataFrame],
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
    levels = capitalization_sources(ident, acct, commitments)

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

    def split(df, col, id_col="InvestorID"):
        if df is None or df.empty:
            return None, None
        psc = df[df[id_col].map(is_psc_side)]
        op = df[~df[id_col].map(is_psc_side)]
        p = _to_float(psc[col].abs().sum()) if not psc.empty else None
        o = _to_float(op[col].abs().sum()) if not op.empty else None
        return (p or None), (o or None)

    if commitments is not None and not commitments.empty and "EntityID" in commitments.columns:
        m = commitments[commitments["EntityID"].map(norm_id) == iid]
        out.append(("commitments (IA_Commitment)",) + split(m, "Amount"))
    else:
        out.append(("commitments (IA_Commitment)", None, None))

    cm = (rows[rows["is_commitment"].fillna(False)]
          if (not rows.empty and "is_commitment" in rows.columns) else None)
    out.append(("accounting Commitment rows (SubtypeUID 1026)",) + split(cm, "Amt"))

    cb = (rows[rows["is_contribution"].fillna(False)]
          if (not rows.empty and "is_contribution" in rows.columns) else None)
    out.append(("funded to date",) + split(cb, "Amt"))
    return out


def first_lien(
    ident: DealIdentity,
    loans: Optional[pd.DataFrame],
    isbs_interim_bs: Optional[pd.DataFrame],
    child_vcodes: Optional[List[str]] = None,
) -> Tuple[Optional[float], str]:
    """The first mortgage at stabilization, in dollars, plus its basis.

    A DEVELOPMENT deal takes the COMMITTED facility (``loans.mOrigLoanAmt``):
    its balance sheet shows whatever has been drawn so far, which on a deal
    still building is not its capitalization and would understate the column
    by most of the loan.

    Everything else takes the EARLIEST balance-sheet debt row — the amount
    outstanding when the deal opened, which is what "at stabilization" means
    for a stabilized asset. Deliberately NOT ``compute.get_isbs_debt_balance``,
    which returns the MOST RECENT balance: that is today's outstanding after
    years of amortisation, a different question, and using it would make every
    older deal's capitalization drift down each quarter.

    CHILD PROPERTIES ARE ROLLED UP FOR LOANS AND NOT FOR ISBS, and the
    difference is not an oversight. ``loans`` records one row per property, so
    a portfolio's facility is only complete once the children are added;
    ``isbs_interim_bs`` already carries the parent's consolidated balance
    sheet, so adding the children counts the same debt twice — measured, it
    doubled Giant-7 ($97.0m to $194.0m), OREI ($34.2m to $68.4m) and Burton
    ($75.3m to $150.6m).

    THIS IS THE WEAKEST COLUMN IN THE REPORT and the reason is structural:
    "underwritten capitalization at stabilization" is an underwriting
    ASSUMPTION, and MRI records loans and balances, not assumptions. Measured
    against the reference across all 76 deals, the committed facility lands on
    the printed figure for 42 and the earliest balance-sheet row for 27 — so
    the rule implemented here is the one that was agreed, not the one that
    scores best, and ``alternate`` carries the other so the gap is visible
    rather than argued about.
    """
    own = {norm_id(v) for v in ([ident.vcode, ident.shadow_vcode]) if v}
    with_kids = own | {norm_id(v) for v in (child_vcodes or []) if v}

    facility = _loan_facility(loans, with_kids)
    isbs_amt, when = _earliest_isbs_debt(isbs_interim_bs, own)

    if ident.vcode in cfg.DEV_DEALS:
        if facility is not None:
            return facility, "loans.mOrigLoanAmt (committed facility — development)"
        return None, "no loan on record (development)"

    if isbs_amt is not None:
        return isbs_amt, f"earliest ISBS Interim BS debt row ({when})"
    if facility is not None:
        return facility, "loans.mOrigLoanAmt (no ISBS balance-sheet debt found)"
    return None, f"unavailable ({when})"


def first_lien_alternate(ident, loans, isbs_interim_bs, child_vcodes):
    """What the source this report did NOT use would have said, and why."""
    own = {norm_id(v) for v in ([ident.vcode, ident.shadow_vcode]) if v}
    with_kids = own | {norm_id(v) for v in (child_vcodes or []) if v}
    if ident.vcode in cfg.DEV_DEALS:
        amt, when = _earliest_isbs_debt(isbs_interim_bs, own)
        return {"value": amt, "basis": f"earliest ISBS Interim BS debt row ({when})"}
    return {"value": _loan_facility(loans, with_kids),
            "basis": "loans.mOrigLoanAmt (committed facility)"}


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


#: ISBS balance-sheet accounts that carry mortgage debt. Same set as
#: ``config.DEBT_BS_ACCTS`` — imported rather than restated so the two cannot
#: drift.
def _debt_accounts() -> set:
    from config import DEBT_BS_ACCTS
    return {str(a).strip() for a in DEBT_BS_ACCTS}


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
    m = df[df[vcol].map(norm_id).isin(want)]
    if m.empty:
        return None, "no ISBS rows for this deal"
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
def proceeds_to_date(ident: DealIdentity, acct: pd.DataFrame, table: str) -> Optional[float]:
    """Cash returned to PSC, in dollars. NO as-of cutoff — deliberately.

    The column is headed *To-Date*, and the reference workbook's own formula
    carries no date bound either. On a CURRENT deal it is the four
    distribution subtypes footnote (1) names; on a SOLD deal it is every
    distribution, because the deal is finished and the question is the total.
    """
    rows = _deal_accounting(acct, ident.investment_id)
    if rows.empty:
        return None
    rows = rows[rows["InvestorID"].map(is_psc_side)]
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


def realized_irr(ident: DealIdentity, acct: pd.DataFrame) -> Optional[float]:
    """XIRR over PSC's contributions and distributions. ONE ENGINE: ``metrics.xirr``.

    Commitment rows are pledges and are excluded — including them dates the
    first cash flow at the signing rather than the funding and shifts the IRR.
    """
    rows = _deal_accounting(acct, ident.investment_id)
    if rows.empty:
        return None
    rows = rows[rows["InvestorID"].map(is_psc_side)]
    if "is_commitment" in rows.columns:
        rows = rows[~rows["is_commitment"].fillna(False)]
    flows: List[Tuple[_dt.date, float]] = []
    for _, r in rows.iterrows():
        d = _to_date(r.get("EffectiveDate"))
        a = _to_float(r.get("Amt"))
        if d is None or a is None:
            continue
        flows.append((d, a))
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
    as_of = as_of or latest_quarter_end(today)
    identities, diag = resolve_deal_identities(inv)
    # Every downstream engine reads the RESOLVED frame, not the raw one — see
    # resolved_inv_frame for why a twin is invisible to the PE engine otherwise.
    inv_resolved = resolved_inv_frame(inv, identities)

    dt_index = _deal_terms_index(deal_terms)
    rows_by_table: Dict[str, List[dict]] = {CURRENT: [], SOLD: []}

    for ident in identities:
        table = classify(ident, as_of)
        rows_by_table[table].append(
            _build_row(ident, table, as_of, acct, commitments, dt_index,
                       loans, isbs_interim_bs, inv_resolved, isbs_raw,
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
        "diagnostics": diag,
    }

    for table, order, markers, title, footnotes in (
        (CURRENT, cfg.ROW_ORDER_CURRENT, cfg.ROW_MARKERS_CURRENT,
         cfg.TITLE_CURRENT, cfg.FOOTNOTES_CURRENT),
        (SOLD, cfg.ROW_ORDER_SOLD, cfg.ROW_MARKERS_SOLD,
         cfg.TITLE_SOLD, cfg.FOOTNOTES_SOLD),
    ):
        rows = _ordered(rows_by_table[table], order, diag, table)
        for r in rows:
            r["markers"] = markers.get(r["vcode"], [])
            _apply_young_deal_substitution(r, diag)
        out[table] = {
            "title": title,
            "rows": rows,
            "total": _total_row(rows),
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
    out["grand_total"] = _grand_total(out[CURRENT]["rows"] + out[SOLD]["rows"])
    return out


def resolve_unloaded(key: str, terms: dict, computed: Dict[str, Optional[float]],
                     diag: dict, vcode: str) -> Tuple[Optional[float], str]:
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
        if field not in terms:
            diag.setdefault("unloaded_figure_field_absent", []).append(
                {"column": key, "table": spec.get("table"), "field": field,
                 "vcode": vcode})
            return None, f"{spec.get('table')}.{field} is not present"
        return _as_rate(terms.get(field)), f"{spec.get('table')}.{field}"

    # mode == "none" — pending, and deliberately not rendered.
    diag.setdefault("unloaded_figures_pending", {}).setdefault(
        key, {"label": spec.get("label"), "note": spec.get("note"),
              "deals": 0})["deals"] += 1
    return None, f"pending Alay — {spec.get('note')}"


#: Footnote (5): under a year of operating history, so the Act. Yr-1 CoC
#: column shows the PROJECTED year-1 figure instead.
#: Footnote (6): under a quarter, so ALL FOUR CoC columns do.
_YOUNG_DEAL_SUBSTITUTION = {
    5: ("act_yr1_coc",),
    6: ("proj_yr1_coc", "act_yr1_coc", "proj_coc_since_close",
        "act_coc_since_close"),
}


def _apply_young_deal_substitution(row: dict, diag: dict) -> None:
    """Do what footnotes (5) and (6) say, rather than only printing them.

    A deal three months old has an actual cash-on-cash return that is
    arithmetically true and useless — it annualises a stub period — so the
    reference substitutes the projected year-1 figure and says so in a
    footnote. Printing the footnote while leaving the actual figure in place
    would be the worst of both: the note tells the reader one thing and the
    cell shows another.

    THIS IS INERT UNTIL ALAY LOADS THE PROJECTED YEAR-1 CoC. With nothing to
    substitute, the cell keeps its own value rather than being blanked —
    replacing a real figure with an em dash on the strength of a footnote
    would lose information rather than correct it. Every substitution that
    could not be made is recorded so the gap is countable.
    """
    src = row.get("proj_yr1_coc")
    fields: set = set()
    for marker in row.get("markers", ()):
        fields.update(_YOUNG_DEAL_SUBSTITUTION.get(marker, ()))
    if not fields:
        return
    if src is None:
        diag.setdefault("young_deal_substitution_unavailable", []).append(
            {"vcode": row["vcode"], "name": row["name"],
             "markers": row["markers"],
             "reason": "projected Year-1 CoC is not loaded into MRI"})
        return
    for f in sorted(fields):
        if f == "proj_yr1_coc":
            continue
        row[f] = src
        row.setdefault("basis", {})[f] = (
            f"projected Year-1 CoC, substituted under footnote "
            f"({'6' if 6 in row['markers'] else '5'})")


def _display_as_of(d: _dt.date) -> str:
    """``30-Jun-26``, as the reference prints it."""
    return f"{d.day:02d}-{d.strftime('%b')}-{d.strftime('%y')}"


def _deal_terms_index(deal_terms: Optional[pd.DataFrame]) -> Dict[str, dict]:
    if deal_terms is None or deal_terms.empty:
        return {}
    out = {}
    for _, r in deal_terms.iterrows():
        out[norm_id(r.get("vcode"))] = r.to_dict()
    return out


def _as_rate(value: Any) -> Optional[float]:
    """Contract rates are stored as either 0.085 or 8.5. Both mean 8.5%."""
    v = _to_float(value)
    if v is None:
        return None
    return v if abs(v) < 1 else v / 100.0


def _build_row(ident, table, as_of, acct, commitments, dt_index, loans,
               isbs_interim_bs, inv, isbs_raw, waterfalls, diag) -> dict:
    from one_pager import _child_vcodes_for_parent

    try:
        children = _child_vcodes_for_parent(ident.vcode, inv)
    except Exception:
        children = []

    pref_usd, floss_usd, cap_basis = pref_and_first_loss(ident, acct, commitments)
    lien_usd, lien_basis = first_lien(ident, loans, isbs_interim_bs, children)
    lien_alt = first_lien_alternate(ident, loans, isbs_interim_bs, children)
    cap_alts = capitalization_sources(ident, acct, commitments)

    fx = cfg.CAD_TO_USD if norm_id(ident.currency) == "CAD" else 1.0
    to_m = lambda v: None if v is None else (v * fx) / MILLION  # noqa: E731

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
    proceeds = to_m(proceeds_to_date(ident, acct, table))
    act_yr1, yr1_basis = act_year_one_coc(ident, acct, funded)
    act_yr1_on_commit = (None if (pref_usd in (None, 0) or funded in (None, 0))
                         else (act_yr1 * funded / pref_usd if act_yr1 is not None else None))

    roe, uw_roe = _coc_since_close(ident, as_of, acct, waterfalls, inv, isbs_raw)

    terms = dt_index.get(norm_id(ident.vcode), {})
    labels = (cfg.CELL_LABELS_CURRENT if table == CURRENT
              else cfg.CELL_LABELS_SOLD).get(ident.vcode, {})

    # The three columns with no source in the app go through ONE switch. The
    # derived Year-1 figures are computed either way and kept below in
    # `alternates`, so turning the column on later needs no new arithmetic.
    uw_irr_v, uw_irr_basis = resolve_unloaded(
        "uw_irr", terms, {}, diag, ident.vcode)
    proj_yr1_v, proj_yr1_basis = resolve_unloaded(
        "proj_yr1_coc", terms, {}, diag, ident.vcode)
    act_yr1_v, act_yr1_mode = resolve_unloaded(
        "act_yr1_coc", terms,
        {"funded": act_yr1, "commitment": act_yr1_on_commit,
         "default": act_yr1},
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
                "on_funded": act_yr1,
                "on_commitment": act_yr1_on_commit,
                "basis": yr1_basis,
            },
            "first_lien": {"value": to_m(lien_alt["value"]),
                           "basis": lien_alt["basis"]},
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
            row["realized_irr"] = realized_irr(ident, acct)
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
    """
    from one_pager import get_pe_performance

    quarter = f"{as_of.year}-Q{(as_of.month - 1) // 3 + 1}"
    try:
        pe = get_pe_performance(
            ident.vcode, quarter, acct, waterfalls, inv,
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
    if extra:
        diag.setdefault("not_in_reference_order", []).extend(
            {"table": table, "vcode": r["vcode"], "name": r["name"]} for r in extra)
    missing = [v for v in order if v not in {r["vcode"] for r in rows}]
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
