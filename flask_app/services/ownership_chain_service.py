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
        with engine.connect() as conn:
            self.com = self._read(conn, "commitments")
            self.ent = self._read(conn, "entities")
            self.deals = self._read(conn, "deals")
            self.wf = self._read(conn, "waterfalls")

        # Commitments: the one table whose shape we depend on.
        if not self.com.empty:
            for c in ("EntityID", "InvestorID"):
                if c in self.com.columns:
                    self.com[c] = self.com[c].map(_norm)
            self.com["Amount"] = pd.to_numeric(
                self.com.get("Amount"), errors="coerce").fillna(0.0)
            self.com = self._open_only(self.com)

        # Entity id -> display name.
        self.names: Dict[str, str] = {}
        if not self.ent.empty:
            idc = self._col(self.ent, "entityid")
            nmc = self._col(self.ent, "name")
            if idc and nmc:
                for _, r in self.ent.iterrows():
                    self.names[_norm(r[idc])] = str(r[nmc] or "").strip()

        # Deal InvestmentID -> (vcode, investment name). This is the PE
        # investment level and the only level keyed by vcode.
        self.deal_by_investment: Dict[str, dict] = {}
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

        # Every code that has at least one waterfall row, deal or entity.
        self.wf_codes: Set[str] = set()
        self.wf_step_counts: Dict[str, int] = {}
        if not self.wf.empty and "vcode" in self.wf.columns:
            codes = self.wf["vcode"].map(_norm)
            self.wf_step_counts = codes.value_counts().to_dict()
            self.wf_codes = {c for c in codes if c}

    @staticmethod
    def _read(conn, table) -> pd.DataFrame:
        try:
            return pd.read_sql(text(f"SELECT * FROM {table}"), conn)
        except Exception:
            logger.warning("ownership chain: %s not loaded", table, exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def _col(df: pd.DataFrame, want: str) -> Optional[str]:
        for c in df.columns:
            if c.lower() == want:
                return c
        return None

    @staticmethod
    def _open_only(com: pd.DataFrame) -> pd.DataFrame:
        """Drop commitments that have already closed.

        A closed commitment is history, not ownership -- leaving it in would
        both dilute the live owners and make the level's total committed
        dollars disagree with the balance sheet. Rows with no EndDate are open,
        which is the overwhelming majority.
        """
        if "EndDate" not in com.columns:
            return com
        end = pd.to_datetime(com["EndDate"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        return com[end.isna() | (end > today)].copy()

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

        owners.append({
            "entity_id": investor_id,
            "name": src.display_name(investor_id),
            "committed": amt,
            "pct": derived,
            "pct_stated": stated,
            "pct_disagrees": disagrees,
            "commitment_count": int(len(grp)),
        })
    owners.sort(key=lambda o: o["committed"], reverse=True)
    return owners


def _build_level(src: _Source, entity_id: str, depth: int,
                 seen: Set[str]) -> List[dict]:
    """Owners of `entity_id`, each with its own owners, recursively."""
    if depth >= MAX_DEPTH:
        return []

    nodes = []
    for o in _owners_of(src, entity_id):
        eid = o["entity_id"]
        node = dict(o)
        node["level"] = depth + 1
        node.update(_waterfall_status(src, eid))

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
            node["terminal"] = False
            node["owners"] = _build_level(src, eid, depth + 1, seen | {eid})
            if not node["owners"] and depth + 1 < MAX_DEPTH:
                node["truncated_reason"] = "No commitments recorded into this entity."
        nodes.append(node)
    return nodes


def _count(nodes: List[dict]) -> tuple:
    """(entities, levels_missing_waterfall) across a subtree."""
    n = missing = 0
    for x in nodes:
        n += 1
        if not x["has_waterfall"]:
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
        "committed": sum(o["committed"] for o in owners),
        "pct": None,          # the root is not a share of anything
        "terminal": False,
        "owners": owners,
        **_waterfall_status(src, wf_code),
    }
    if not owners:
        root["truncated_reason"] = "No commitments recorded into this entity."

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
        notes.append(f"No open commitments name {iid} as the invested entity.")
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
    return {
        "commitment_rows": int(len(src.com)),
        "entities_named": len(src.names),
        "disagreement_count": len(bad),
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
