"""Ownership chains from the commitments table, for waterfall setup.

WHAT THIS ANSWERS. Starting at the entity through which we hold preferred
equity in a property, who owns it, who owns them, and so on up to OWPSC --
and at every level, is there a waterfall set up yet?

WHY COMMITMENTS AND NOT `relationships`. ``ownership_tree.py`` walks
MRI_IA_Relationship and reads a stored ``OwnershipPct``. That field is right
only once somebody fills it in and silently wrong until then. On Sep 14 2026
``statement_service`` established the precedent this module follows: PPIECH's
commitment to EASTCH is 29,390,000 of 44,085,000 committed -- 66.67%, which is
what the audited SOI carries -- while ``relationships`` claimed PPIECH held
100% and ``commitments.CapitalPercent`` read 0.00 on both rows. The AMOUNTS
were right and BOTH percentage fields were wrong. So the percentage here is
always derived from committed dollars and never read from a stored field.

That also makes the "must total 100%" requirement structural rather than a
check: every owner's share is its amount over the sum of amounts at the same
entity, so a level sums to 100% by construction. What CAN be wrong is the
population -- a missing commitment row makes the survivors look larger. The
level therefore reports its own total committed dollars, so a level whose
total is not what the analyst expects announces itself.

DIRECTION. ``commitments.EntityID`` is the entity invested IN;
``commitments.InvestorID`` is the owner. Walking "up" means: given an entity,
find the rows whose EntityID is that entity, and the InvestorIDs are its
owners.

WHERE THE WATERFALL LIVES. Both deal-level and upstream waterfalls sit in the
same ``waterfalls`` table keyed by ``vcode`` -- but the key differs by level.
A deal's waterfall is keyed by its PROPERTY code (P0000085), while an upstream
entity's is keyed by the ENTITY id (PPIECH, AMB23, OWPSC). Level 0 therefore
looks itself up by vcode and every level above it by entity id. Getting this
backwards would report every deal as having no waterfall.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import pandas as pd
from sqlalchemy import text

from flask_app.db import get_engine

logger = logging.getLogger(__name__)

# The top of the house. The chain stops here: OWPSC is reported as a terminal
# node -- with its own waterfall status, because it has one -- but its owners
# are never expanded. Jim's instruction, Sep 15 2026.
TERMINAL_ENTITY = "OWPSC"

# A cycle in the data would otherwise recurse forever. Depth is also capped so
# that a pathological chain degrades into a reported truncation rather than a
# stack overflow in a request handler.
MAX_DEPTH = 12


def _norm(s) -> str:
    return str(s).strip().upper() if s is not None else ""


class _Source:
    """The four tables this module reads, loaded once per request.

    Held together rather than passed around separately because every level of
    the walk needs all of them, and reloading per level turned a 6-level chain
    into 24 queries.
    """

    def __init__(self, engine=None):
        engine = engine or get_engine()
        self.load_errors: List[str] = []
        with engine.connect() as conn:
            self.com = self._read(conn, "commitments")
            self.ent = self._read(conn, "entities")
            self.deals = self._read(conn, "deals")
            self.wf = self._read(conn, "waterfalls")
            self.acct = self._read(conn, "accounting")

        # WHAT ARRIVED, BEFORE ANYTHING IS DONE TO IT. Reported in the health
        # block so "no commitments for this entity" can be told apart from "the
        # table did not load" and from "the filter ate it" without a developer
        # having to reproduce the database. Sep 15 2026: 601 rows were visibly
        # present in production and the screen still said none, and none of the
        # three hypotheses could be distinguished from the outside.
        self.raw_commitment_rows = int(len(self.com))
        self.commitment_columns = [str(c) for c in self.com.columns]
        self.superseded_rows = 0

        # Commitments: the one table whose shape we depend on.
        if not self.com.empty:
            # CASE-INSENSITIVE, because PostgreSQL folds unquoted identifiers to
            # lower case and SQLite does not. The same table is `EntityID` here
            # and `entityid` there, and matching one spelling silently produced
            # either an empty screen or an AttributeError depending on which
            # column happened to miss. CLAUDE.md carries the same warning for
            # SQL identifiers; it applies just as much to DataFrame lookups.
            self.com = self._canonicalise(self.com, (
                "EntityID", "InvestorID", "Amount", "CapitalPercent",
                "StartDate", "EndDate"))
            for c in ("EntityID", "InvestorID"):
                if c in self.com.columns:
                    self.com[c] = self.com[c].map(_norm)
            if "Amount" in self.com.columns:
                self.com["Amount"] = pd.to_numeric(
                    self.com["Amount"], errors="coerce").fillna(0.0)
            else:
                self.load_errors.append(
                    "commitments has no Amount column (found: "
                    + ", ".join(self.commitment_columns[:12]) + ")")
                self.com["Amount"] = 0.0
            if "EntityID" not in self.com.columns:
                self.load_errors.append(
                    "commitments has no EntityID column (found: "
                    + ", ".join(self.commitment_columns[:12]) + ")")
            self.com, self.superseded_rows = self._current_only(self.com)

        # Entity id -> display name.
        self.names: Dict[str, str] = {}
        if not self.ent.empty:
            self.ent = self._canonicalise(self.ent, ("ENTITYID", "NAME"))
            idc = self._col(self.ent, "entityid")
            nmc = self._col(self.ent, "name")
            if idc and nmc:
                for _, r in self.ent.iterrows():
                    self.names[_norm(r[idc])] = str(r[nmc] or "").strip()

        # Deal InvestmentID -> (vcode, investment name). This is the PE
        # investment level and the only level keyed by vcode.
        self.deal_by_investment: Dict[str, dict] = {}
        if not self.deals.empty:
            self.deals = self._canonicalise(self.deals, (
                "InvestmentID", "vcode", "Investment_Name", "Portfolio_Name"))
            if "InvestmentID" not in self.deals.columns:
                self.load_errors.append(
                    "deals has no InvestmentID column, so no PE investments "
                    "can be listed")
        if not self.deals.empty and "InvestmentID" in self.deals.columns:
            for _, r in self.deals.iterrows():
                iid = _norm(r.get("InvestmentID"))
                if not iid:
                    continue
                self.deal_by_investment[iid] = {
                    "vcode": str(r.get("vcode") or "").strip(),
                    "name": str(r.get("Investment_Name") or "").strip(),
                    "portfolio": str(r.get("Portfolio_Name") or "").strip(),
                }

        # ── Capital outstanding per (entity, investor) ──────────────────
        #
        # A commitment is what was PROMISED. The balance is what is actually
        # outstanding against it, and the two answer different questions: a
        # fully returned commitment still reads at its committed amount.
        #
        # THE CLASSIFIER IS BORROWED, NOT INVENTED. `reports_service` already
        # decides which accounting rows touch capital, and `loaders.capital_after`
        # already decides which way. CLAUDE.md §1.3 records that four different
        # classifiers for this already exist on the same rows; a fifth written
        # here would be the defect, not the feature. Both open questions about
        # that rule -- §1.7 ("contrib" in MajorType is over-broad) and §2.3
        # (the Capital flag vs the Typename rule, $73.6M apart) -- therefore
        # apply to this column exactly as they apply to the reports, which is
        # the point: it moves when they move.
        self.balances: Dict[tuple, float] = {}
        self.balance_detail: Dict[tuple, list] = {}
        self.balance_by_flag: Dict[tuple, float] = {}
        if not self.acct.empty:
            self.acct = self._canonicalise(self.acct, (
                "InvestmentID", "InvestorID", "Amt", "MajorType", "Typename"))
            need = {"InvestmentID", "InvestorID", "Amt"}
            if need.issubset(set(self.acct.columns)):
                a = self.acct
                major = a.get("MajorType", pd.Series("", index=a.index)).astype(str).str.lower()
                tname = a.get("Typename", pd.Series("", index=a.index)).astype(str).str.lower()
                touches = major.str.contains("contrib") | (
                    major.str.contains("distri")
                    & (tname.str.contains("return of capital")
                       | tname.str.contains("realized gain")))
                cap = a[touches].copy()

                # The same rows as the `Capital` flag sees them, for comparison
                # only -- never used as the figure.
                flagged = None
                if "Capital" in a.columns:
                    flagged = a[a["Capital"].astype(str).str.strip().str.upper() == "Y"].copy()
                    if not flagged.empty:
                        flagged["Amt"] = pd.to_numeric(flagged["Amt"], errors="coerce").fillna(0.0)
                        flagged["_e"] = flagged["InvestmentID"].map(_norm)
                        flagged["_i"] = flagged["InvestorID"].map(_norm)

                if not cap.empty:
                    cap["Amt"] = pd.to_numeric(cap["Amt"], errors="coerce").fillna(0.0)
                    cap["_e"] = cap["InvestmentID"].map(_norm)
                    cap["_i"] = cap["InvestorID"].map(_norm)
                    # capital_after(c, amt) == c - amt, so the balance is the
                    # NEGATED sum. Not floored here: a negative running total is
                    # a real finding and the caller reports it.
                    for (e, i), grp in cap.groupby(["_e", "_i"], sort=False):
                        self.balances[(e, i)] = -float(grp["Amt"].sum())
                        # WHAT THE NUMBER IS MADE OF, so a balance that looks
                        # wrong can be argued with instead of guessed at. Jim,
                        # Sep 15 2026: 30BEAR had its equity fully returned and
                        # PPI27 still showed $1,347,797. Locally the same rows
                        # net to exactly 0.00 under BOTH this classifier and
                        # the Capital flag, so the difference is in rows this
                        # machine does not have -- and no amount of reasoning
                        # from here will name them. The breakdown travels with
                        # the figure so the next person reads it instead.
                        by = grp.groupby("Typename")["Amt"].agg(["count", "sum"])
                        self.balance_detail[(e, i)] = [
                            {"typename": str(t),
                             "rows": int(r["count"]),
                             # negated to match the balance's direction: a
                             # positive line ADDS to capital outstanding
                             "effect": -float(r["sum"])}
                            for t, r in by.iterrows()
                        ]
                        # The Capital flag's answer for the same pair, when the
                        # column exists. CLAUDE.md §2.3 has these two
                        # classifiers $73.6M apart across the portfolio and the
                        # question unsettled; where they agree the figure is
                        # uncontested, and where they do not the screen should
                        # say so rather than pick a winner silently.
                        if flagged is not None:
                            f = flagged[(flagged["_e"] == e) & (flagged["_i"] == i)]
                            self.balance_by_flag[(e, i)] = -float(f["Amt"].sum())
            else:
                self.load_errors.append(
                    "accounting is missing " + ", ".join(sorted(need - set(self.acct.columns)))
                    + " — capital balances cannot be shown")

        # Every code that has at least one waterfall row, deal or entity.
        self.wf_codes: Set[str] = set()
        self.wf_step_counts: Dict[str, int] = {}
        if not self.wf.empty:
            self.wf = self._canonicalise(self.wf, ("vcode",))
        if not self.wf.empty and "vcode" in self.wf.columns:
            codes = self.wf["vcode"].map(_norm)
            self.wf_step_counts = codes.value_counts().to_dict()
            self.wf_codes = {c for c in codes if c}

    def _read(self, conn, table) -> pd.DataFrame:
        """Load a table, RECORDING a failure rather than swallowing it.

        This returned an empty frame on any exception and logged a warning
        nobody reads. A failed load then rendered as "No commitments recorded
        into this entity" -- the screen reporting a fact about the data when
        the truth was that the query never ran. A missing table and an empty
        one must not look the same to the reader.
        """
        try:
            return pd.read_sql(text(f"SELECT * FROM {table}"), conn)
        except Exception as e:
            logger.warning("ownership chain: %s not loaded", table, exc_info=True)
            self.load_errors.append(f"{table} could not be read: {str(e)[:160]}")
            return pd.DataFrame()

    @staticmethod
    def _canonicalise(df: pd.DataFrame, wanted) -> pd.DataFrame:
        """Rename columns to the spelling this module expects, ignoring case."""
        lower = {str(c).lower(): c for c in df.columns}
        ren = {lower[w.lower()]: w for w in wanted
               if w.lower() in lower and lower[w.lower()] != w}
        return df.rename(columns=ren) if ren else df

    @staticmethod
    def _col(df: pd.DataFrame, want: str) -> Optional[str]:
        for c in df.columns:
            if c.lower() == want:
                return c
        return None

    @staticmethod
    def _current_only(com: pd.DataFrame):
        """Reduce the commitment history to the CURRENT commitment per pair.

        COMMITMENTS ARE AN AMENDMENT HISTORY, NOT A LEDGER OF ADDITIONS. When a
        commitment changes, the old row is closed with an EndDate and a new row
        opens with a later StartDate. So the current commitment of an investor
        into an entity is the amount on **the most recent StartDate that has no
        EndDate** -- one row, not a total. (Jim, Sep 15 2026, reading the
        deployed tree against MRI.)

        The first version of this summed every row that had not yet ended,
        which is wrong in both directions at once: an amended commitment was
        counted once per amendment, inflating that investor, and because every
        share is that investor's amount over the level's total, EVERY OTHER
        owner at the level was correspondingly understated. The level still
        summed to 100%, so nothing looked broken -- the same blind spot the
        CapitalPercent cross-check exists for.

        It also treated a row with a FUTURE EndDate as open. A dated end is a
        dated end; the rule is "no ending date", not "not ended yet".

        Rows sharing the same latest StartDate are summed, since two genuine
        co-equal commitments starting the same day are one position. Returns
        the reduced frame and the number of rows it set aside, so the caller
        can say how much history was considered rather than silently dropping
        it.
        """
        if com.empty or "EntityID" not in com.columns:
            return com, 0
        before = len(com)

        if "EndDate" in com.columns:
            # PANDAS' OWN NULL TEST, never a rendered string.
            #
            # This asked `astype(str)` what the cell looked like and matched
            # the result against ("", "none", "nan", "nat", "null"). Four of
            # the five flavours of null render into that list; `pd.NA` renders
            # as "<NA>" and does not. PostgreSQL produces pd.NA where SQLite
            # produces None, so on Azure EVERY row failed the test and all 601
            # commitments were discarded, while every local test passed. The
            # screen then reported "no commitments" for every deal in the
            # portfolio. (Found Sep 15 2026, after two wrong fixes.)
            #
            # `.isna()` is the primitive that answers this question for all of
            # them. The original intent survives unchanged: a non-null value
            # that merely fails to parse is NOT absent -- somebody entered it,
            # and guessing it away would resurrect a closed commitment -- and
            # `.isna()` is False for exactly those.
            com = com[com["EndDate"].isna()].copy()

        if com.empty:
            return com, before

        if "StartDate" in com.columns:
            # Normalised to tz-NAIVE before anything compares them. PostgreSQL
            # returns timestamptz and SQLite returns strings, and a frame
            # carrying both kinds cannot be compared or filled without either
            # raising or quietly never matching. `utc=True` puts every value on
            # one clock; stripping the zone then makes it comparable to
            # `Timestamp.min` below, which is naive.
            st = pd.to_datetime(com["StartDate"], errors="coerce", utc=True)
            try:
                st = st.dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            com["_start"] = st
            # A row with no usable StartDate cannot lose a recency contest it
            # was never in, so it sorts last -- but it still counts if it is
            # the only row for the pair.
            com["_rank"] = com["_start"].fillna(pd.Timestamp.min)
            latest = com.groupby(["EntityID", "InvestorID"])["_rank"].transform("max")
            com = com[com["_rank"] == latest].copy()
            com.drop(columns=["_rank"], inplace=True)
        else:
            com["_start"] = pd.NaT

        return com, before - len(com)

    def display_name(self, eid: str) -> str:
        """Best available name, never blank -- the id is the last resort."""
        return (self.names.get(eid)
                or (self.deal_by_investment.get(eid) or {}).get("name")
                or eid)


def _waterfall_status(src: _Source, code: str) -> dict:
    """Is a waterfall set up for this code, and where does the link go?

    ``code`` is a vcode at the PE investment level and an entity id above it;
    the caller decides which, because only the caller knows the level.
    """
    code = _norm(code)
    steps = int(src.wf_step_counts.get(code, 0))
    return {
        "waterfall_code": code,
        "has_waterfall": steps > 0,
        "step_count": steps,
        "waterfall_url": f"/waterfall-setup?vcode={code}" if code else None,
    }


def _owners_of(src: _Source, entity_id: str) -> List[dict]:
    """The owners of one entity, with each share derived from committed $."""
    if src.com.empty:
        return []
    rows = src.com[src.com["EntityID"] == entity_id]
    if rows.empty:
        return []

    total = float(rows["Amount"].sum())
    owners = []
    # One owner may hold several commitments into the same entity; they are
    # one owner with a combined position, not several rows on screen.
    for investor_id, grp in rows.groupby("InvestorID", sort=False):
        amt = float(grp["Amount"].sum())
        derived = (100.0 * amt / total) if total else None

        # CROSS-CHECK AGAINST THE STORED PERCENTAGE, WHERE THERE IS ONE.
        #
        # Derived shares always sum to 100%, so a level totalling 100% proves
        # nothing -- if a commitment row is missing, the remaining owners
        # simply absorb its share and the arithmetic still looks perfect. That
        # is the one failure that would silently produce a WRONG waterfall,
        # which is what this screen exists to help set up.
        #
        # CapitalPercent is unreliable as a source (it reads 0.00 on both
        # EASTCH rows, which is why the amount is authoritative) but where it
        # IS populated it is an independent witness. PPIECH's three
        # commitments carry 15.31 / 16.64 / 68.05 and match their implied
        # shares to the cent. So: never use it, always compare it, and say so
        # when it disagrees -- the same "report both, flag the difference"
        # posture statement_service takes on the same table.
        stated = None
        if "CapitalPercent" in grp.columns:
            cp = pd.to_numeric(grp["CapitalPercent"], errors="coerce").sum()
            if pd.notna(cp) and cp > 0:
                stated = float(cp)
        disagrees = (stated is not None and derived is not None
                     and abs(stated - derived) > 0.5)

        # The date the current commitment took effect. Shown because the
        # figure is "the amount on the latest open StartDate" -- without the
        # date, a reader cannot tell which amendment they are looking at.
        since = None
        if "_start" in grp.columns:
            s = grp["_start"].max()
            if pd.notna(s):
                since = s.date().isoformat()

        # What is actually outstanding against this commitment, as opposed to
        # what was promised. None -- never 0.0 -- when the feed has no rows for
        # the pair: a zero balance and no data are different facts and only one
        # of them means "fully returned".
        bal = src.balances.get((entity_id, investor_id))
        bal_flag = src.balance_by_flag.get((entity_id, investor_id))
        detail = src.balance_detail.get((entity_id, investor_id)) or []

        owners.append({
            "entity_id": investor_id,
            "name": src.display_name(investor_id),
            "committed": amt,
            "balance": bal,
            "balance_detail": detail,
            "balance_by_flag": bal_flag,
            # The two classifiers disagreeing is news, not noise.
            "balance_disputed": (bal is not None and bal_flag is not None
                                 and abs(bal - bal_flag) > 1.0),
            "pct": derived,
            "pct_stated": stated,
            "pct_disagrees": disagrees,
            "commitment_count": int(len(grp)),
            "since": since,
        })
    owners.sort(key=lambda o: o["committed"], reverse=True)
    return owners


def _build_level(src: _Source, entity_id: str, depth: int,
                 seen: Set[str], parent_eff: float = 1.0,
                 parent_lt: Optional[float] = None,
                 parent_lt_bal: Optional[float] = None) -> List[dict]:
    """Owners of `entity_id`, each with its own owners, recursively.

    TWO DIFFERENT DOLLAR FIGURES, AND CONFUSING THEM MISREPRESENTS THE DEAL.
    ``committed`` is what this owner put into the entity DIRECTLY BELOW IT,
    which above the first level is a commitment to a fund, not to this
    property. Jim's example, Sep 15 2026: OWPSC's commitment to PSC3 may be
    $64M, and none of that is its share of the $3M PPI27 committed to 30BEAR —
    the $64M is spread across everything PSC3 holds. Printing it in this chain
    reads as $64M sitting in 30BEAR.

    ``look_through`` is the figure that belongs in a chain about one deal: this
    owner's share OF THIS DEAL, the percentages multiplied down. The first
    level is the anchor, because its commitment really is into the investment;
    every level above is its parent's look-through times its own share.
    ``effective_pct`` is the same thing as a percentage of the deal.

    Both are carried and both are labelled. The direct figure is still the
    right answer to "what did this entity commit to its subsidiary" — it is
    only the wrong answer to "how much of this deal is theirs".
    """
    if depth >= MAX_DEPTH:
        return []

    nodes = []
    for o in _owners_of(src, entity_id):
        eid = o["entity_id"]
        node = dict(o)
        node["level"] = depth + 1
        # Which entity the direct commitment was INTO, so the card can say
        # "$64,000,000 into PSC3" rather than leaving a number that looks like
        # it belongs to this deal.
        node["into_entity_id"] = entity_id
        node["into_name"] = src.display_name(entity_id)
        node.update(_waterfall_status(src, eid))

        share = (o["pct"] / 100.0) if o["pct"] is not None else None
        if depth == 0:
            # Level 1: the commitment IS into the investment, so it anchors
            # every look-through above it.
            node["effective_pct"] = o["pct"]
            node["look_through"] = o["committed"]
            node["look_through_balance"] = o["balance"]
        else:
            node["effective_pct"] = (parent_eff * share * 100.0
                                     if share is not None else None)
            node["look_through"] = (parent_lt * share
                                    if share is not None and parent_lt is not None
                                    else None)
            node["look_through_balance"] = (
                parent_lt_bal * share
                if share is not None and parent_lt_bal is not None else None)

        if eid == TERMINAL_ENTITY:
            # Reported, with its waterfall status, but never expanded.
            node["terminal"] = True
            node["owners"] = []
            node["truncated_reason"] = f"Chain stops at {TERMINAL_ENTITY}."
        elif eid in seen:
            # A cycle. Show the node so the loop is visible rather than
            # silently pruned -- a circular ownership record is a data defect
            # somebody needs to see.
            node["terminal"] = True
            node["owners"] = []
            node["truncated_reason"] = "Already appears higher in this chain (circular ownership)."
        else:
            node["owners"] = _build_level(
                src, eid, depth + 1, seen | {eid},
                parent_eff=((node["effective_pct"] or 0.0) / 100.0),
                parent_lt=node["look_through"],
                parent_lt_bal=node["look_through_balance"])
            # NOTHING ABOVE IT MEANS IT IS THE TOP, NOT THAT SOMETHING IS
            # MISSING. Jim, Sep 15 2026: an entity with no owner above it is
            # the ultimate beneficial owner -- that record IS the owner. So it
            # has no absent commitment and needs no waterfall, and reporting
            # either as a gap sent the reader looking for data that does not
            # exist and inflated the missing-waterfall count with rows that can
            # never be filled.
            node["terminal"] = not node["owners"]
            node["ultimate_owner"] = not node["owners"]
        nodes.append(node)
    return nodes


def _count(nodes: List[dict]) -> tuple:
    """(entities, levels_missing_waterfall) across a subtree.

    An ultimate beneficial owner is NOT counted as missing a waterfall. There
    is nothing beneath it to distribute and no owners to distribute to, so a
    waterfall there is not a gap somebody can close.
    """
    n = missing = 0
    for x in nodes:
        n += 1
        if not x["has_waterfall"] and not x.get("ultimate_owner"):
            missing += 1
        sub_n, sub_missing = _count(x.get("owners") or [])
        n += sub_n
        missing += sub_missing
    return n, missing


def list_pe_investments(engine=None) -> List[dict]:
    """The PE investment level -- one row per deal, the left edge of the tree.

    Every deal with an InvestmentID is listed, including those with no
    commitments recorded. A deal missing from this list because it has no
    ownership data would be invisible exactly when somebody most needs to
    notice it, so ``owner_count`` reports 0 instead.
    """
    src = _Source(engine)
    out = []
    for iid, d in src.deal_by_investment.items():
        owners = _owners_of(src, iid)
        wf = _waterfall_status(src, d["vcode"] or iid)
        out.append({
            "entity_id": iid,
            "name": d["name"] or src.display_name(iid),
            "vcode": d["vcode"],
            "portfolio": d["portfolio"],
            "owner_count": len(owners),
            "total_committed": sum(o["committed"] for o in owners),
            **wf,
        })
    out.sort(key=lambda r: (r["name"] or r["entity_id"]).lower())
    return out


def build_chain(investment_id: str, engine=None) -> dict:
    """The full ownership chain above one PE investment."""
    src = _Source(engine)
    iid = _norm(investment_id)
    deal = src.deal_by_investment.get(iid)

    # Level 0 is keyed by VCODE, every level above it by entity id. See the
    # module docstring -- reversing this reports every deal as unconfigured.
    wf_code = (deal or {}).get("vcode") or iid
    owners = _build_level(src, iid, 0, {iid})
    entities, missing = _count(owners)

    root = {
        "entity_id": iid,
        "name": (deal or {}).get("name") or src.display_name(iid),
        "vcode": (deal or {}).get("vcode"),
        "portfolio": (deal or {}).get("portfolio"),
        "level": 0,
        "is_pe_investment": True,
        "effective_pct": 100.0,
        "look_through": sum(o["committed"] for o in owners),
        "look_through_balance": None,
        "committed": sum(o["committed"] for o in owners),
        "pct": None,          # the root is not a share of anything
        "terminal": False,
        "owners": owners,
        **_waterfall_status(src, wf_code),
    }
    if not owners:
        # At the PE investment level this is a real gap: an investment we hold
        # must have been funded by somebody, so no commitments naming it is
        # missing data rather than a beneficial owner.
        root["truncated_reason"] = (
            "No commitments name this investment as the invested entity.")

    return {
        "root": root,
        "summary": {
            "entities_above": entities,
            "levels_missing_waterfall": missing,
            "max_depth_reached": _depth(root),
            "terminal_entity": TERMINAL_ENTITY,
        },
        "data_health": _health(src, iid, owners),
    }


def _depth(node: dict) -> int:
    subs = node.get("owners") or []
    return 1 + max((_depth(s) for s in subs), default=0)


def _health(src: _Source, iid: str, owners: List[dict]) -> dict:
    """What a reader needs in order to trust -- or distrust -- the chain.

    Shares are derived from amounts, so a level ALWAYS sums to 100% and a
    100% total proves nothing about whether every owner is present. These are
    the things that can actually be wrong.
    """
    notes = []
    if src.com.empty:
        notes.append("The commitments table is empty; no ownership can be derived.")
    if not owners:
        notes.append(f"No open commitments name {iid} as the invested entity. "
                     f"An investment we hold should have been funded by somebody, "
                     f"so this is missing data rather than a top of the chain.")
    zero = [o["entity_id"] for o in owners if o["committed"] == 0]
    if zero:
        notes.append("Zero-dollar commitment from " + ", ".join(zero)
                     + " — shown, but it carries no ownership share.")

    # The disagreements anywhere in the chain, not just at the top level.
    # These are the levels where a waterfall built from this screen would be
    # built on the wrong split.
    bad = _collect_disagreements(owners)
    for b in bad:
        notes.append(
            f"{b['entity_id']} derives {b['pct']:.2f}% of {b['parent']} from committed "
            f"dollars, but the stored CapitalPercent says {b['pct_stated']:.2f}%. The "
            f"usual cause is a commitment row missing from this entity, which makes "
            f"every remaining owner look larger. Confirm before setting up a waterfall "
            f"on this split.")
    # LOUD ABOUT LOAD FAILURES. A table that did not load must never read as
    # a table that was empty.
    for err in getattr(src, "load_errors", []):
        notes.append("DATA LOAD PROBLEM — " + err)

    raw = int(getattr(src, "raw_commitment_rows", 0))
    kept = int(len(src.com))
    if raw and not kept:
        notes.append(
            f"The commitments table returned {raw} rows and none survived "
            f"filtering to the current commitment. That is a defect in this "
            f"screen, not a fact about the data.")

    return {
        "commitment_rows": kept,
        "commitment_rows_loaded": raw,
        "commitment_columns": getattr(src, "commitment_columns", [])[:20],
        "superseded_rows": int(getattr(src, "superseded_rows", 0)),
        "deals_with_investment_id": len(src.deal_by_investment),
        "entities_named": len(src.names),
        "disagreement_count": len(bad),
        "load_errors": list(getattr(src, "load_errors", [])),
        "notes": notes,
    }


def _collect_disagreements(nodes: List[dict], parent: str = "") -> List[dict]:
    out = []
    for n in nodes:
        if n.get("pct_disagrees"):
            out.append({"entity_id": n["entity_id"], "parent": parent or "the level below",
                        "pct": n["pct"], "pct_stated": n["pct_stated"]})
        out.extend(_collect_disagreements(n.get("owners") or [], n["entity_id"]))
    return out
