"""Accounting's allocation overrides: one entity's split, for one investment, from a date.

Accounting, Oct 5 2026, reviewing the PSC Preferred Equity Exposure report:

  BRNERD -- restructured mid-hold; the commitment ratios were modified and the
            commitments are not yet fully funded, so ratios misstate who funded.
            Overrides at PPIBPA and INVBPS.
  JBFAIR -- DCXVIA and DCXVIB opted out of the investment and their initial
            commitment was returned; PSC3's ratios no longer apply to it.
  NOTTNV -- an increase funded by every investor except DCXVIA/DCXVIB, whose share
            OWPSC funded.

MRI's commitments are ENTITY-level: PSC3 has one set of ratios for everything it
holds, so no correction to the commitments can say "for JBFAIR only". This is the
record accounting keeps instead -- data they enter, with a reason, not a per-deal
constant in code.

THE RULES (accounting's answers, Oct 5 2026):
- **Entered as AMOUNTS** -- what each investor funded -- and the share is each
  amount over their total. Amounts are what accounting can check against funding.
- **Dated: applies on the effective date and after.** On any date the set in force
  is the latest one effective on or before it; earlier quarters are untouched.
- **One entity, one investment.** Above that entity the walk goes back to
  commitments; the entity's OTHER investments are untouched.
- **Refused, with the reason**: amounts that are not numbers, negative, or total
  nothing; an investor named twice; an entity that is not in the investment's
  ownership chain on the effective date (an override there would change nothing,
  silently); an entity the report already stops at (a named holder).
- **Never deleted**: removing a set marks it, and the record stays.

Read by ``ownership_chain_service.group_shares`` -- the one engine for ultimate
ownership -- so every consumer of that engine sees the same split.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import text

_READY = set()


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def _norm(s) -> str:
    return str(s or "").strip().upper()


def ensure(engine=None):
    engine = engine or _engine()
    if id(engine) in _READY:
        return engine
    pk = ("id SERIAL PRIMARY KEY" if engine.dialect.name == "postgresql"
          else "id INTEGER PRIMARY KEY AUTOINCREMENT")
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS ownership_overrides (
                {pk},
                entity_id TEXT NOT NULL,
                investment_id TEXT NOT NULL,
                effective_date DATE NOT NULL,
                reason TEXT NOT NULL,
                entered_by TEXT,
                entered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP,
                deleted_by TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ownership_override_lines (
                override_id INTEGER NOT NULL,
                investor_id TEXT NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (override_id, investor_id)
            )"""))
    _READY.add(id(engine))
    return engine


def _parse_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v or "").strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Effective date must be a date (YYYY-MM-DD), not %r" % (v,))


def _d(v):
    return str(v)[:10] if v is not None else None


def list_overrides(engine=None, include_deleted: bool = True) -> list[dict]:
    engine = ensure(engine)
    with engine.connect() as conn:
        heads = conn.execute(text(
            "SELECT id, entity_id, investment_id, effective_date, reason, entered_by, entered_at, "
            "deleted_at, deleted_by FROM ownership_overrides ORDER BY investment_id, entity_id, "
            "effective_date, id")).mappings().fetchall()
        lines = conn.execute(text(
            "SELECT override_id, investor_id, amount FROM ownership_override_lines "
            "ORDER BY override_id, investor_id")).mappings().fetchall()
    by = {}
    for ln in lines:
        by.setdefault(ln["override_id"], []).append({"investor_id": ln["investor_id"],
                                                     "amount": float(ln["amount"])})
    out = []
    for h in heads:
        if h["deleted_at"] is not None and not include_deleted:
            continue
        ls = by.get(h["id"], [])
        tot = sum(x["amount"] for x in ls)
        for x in ls:
            x["pct"] = (x["amount"] / tot * 100.0) if tot else None
        out.append({"id": h["id"], "entity_id": h["entity_id"], "investment_id": h["investment_id"],
                    "effective_date": _d(h["effective_date"]), "reason": h["reason"],
                    "entered_by": h["entered_by"], "entered_at": str(h["entered_at"] or ""),
                    "deleted_at": str(h["deleted_at"]) if h["deleted_at"] else None,
                    "deleted_by": h["deleted_by"], "total": tot, "lines": ls})
    return out


def in_effect(as_of: date, engine=None) -> dict:
    """{(ENTITY, INVESTMENT): override} -- per key, the latest set effective on or
    before ``as_of``; removed sets are ignored. Shares are percentages, as the
    commitment walk's owners carry them."""
    out = {}
    for o in list_overrides(engine, include_deleted=False):
        if o["effective_date"] > as_of.isoformat():
            continue
        key = (_norm(o["entity_id"]), _norm(o["investment_id"]))
        cur = out.get(key)
        if cur is None or (o["effective_date"], o["id"]) > (cur["effective_date"], cur["id"]):
            out[key] = o
    return out


def version(engine=None) -> tuple:
    """Changes whenever a set is added or removed -- for report caches."""
    engine = ensure(engine)
    with engine.connect() as conn:
        r = conn.execute(text("SELECT COUNT(*), MAX(id), MAX(deleted_at) FROM ownership_overrides")).fetchone()
    return tuple(str(x) for x in r)


def _chain_paths(investment: str, on: date, engine) -> list[list[str]]:
    from flask_app.services import ownership_chain_service as oc
    from flask_app.services import pe_exposure_service as pe
    walk = oc.group_shares(investment, on, pe.STOPS, context=pe._context,
                           default_group=pe.DEFAULT_GROUP, engine=engine, use_overrides=False)
    return [rt["path"] for rt in walk["routes"]]


def create(entity_id, investment_id, effective_date, lines, reason, actor, engine=None) -> dict:
    """Add a set. Every rule is checked before anything is written."""
    from flask_app.services import pe_exposure_service as pe
    ent, inv = _norm(entity_id), _norm(investment_id)
    if not ent or not inv:
        raise ValueError("Name the entity and the investment")
    eff = _parse_date(effective_date)
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("Say why -- the reason is what the next reviewer reads")
    clean, seen = [], set()
    for ln in lines or []:
        inv_id = _norm((ln or {}).get("investor_id"))
        if not inv_id:
            raise ValueError("Every line needs an investor")
        if inv_id in seen:
            raise ValueError("%s is named twice" % inv_id)
        seen.add(inv_id)
        try:
            amt = float(str(ln.get("amount")).replace(",", ""))
        except (TypeError, ValueError):
            raise ValueError("%s: the amount must be a number" % inv_id)
        if amt < 0:
            raise ValueError("%s: an amount funded cannot be negative" % inv_id)
        clean.append((inv_id, amt))
    if not clean or sum(a for _, a in clean) <= 0:
        raise ValueError("The amounts total nothing, so they say nothing about the split")
    if ent in pe.STOPS:
        raise ValueError("%s is a named holder (%s); the report stops there, so an override "
                         "would change nothing" % (ent, pe.STOPS[ent]))
    engine = ensure(engine)
    paths = _chain_paths(inv, eff, engine)
    if not any(ent in p[1:] or (p and p[0] == ent) for p in paths):
        raise ValueError("%s is not in %s's ownership chain on %s, so an override there would "
                         "change nothing" % (ent, inv, eff.isoformat()))
    for o in list_overrides(engine, include_deleted=False):
        if (_norm(o["entity_id"]), _norm(o["investment_id"]), o["effective_date"]) == (ent, inv, eff.isoformat()):
            raise ValueError("%s for %s already has a set effective %s -- remove it first"
                             % (ent, inv, eff.isoformat()))
    # Odd but possible: an investor the entity's commitments do not carry.
    from flask_app.services import ownership_chain_service as oc
    src = oc._Source(engine, as_of=eff, with_balances=False)
    owners = {_norm(o["entity_id"]) for o in oc._owners_of(src, ent)}
    warnings = ["%s is not among %s's investors in MRI on %s" % (i, ent, eff.isoformat())
                for i, _ in clean if owners and i not in owners]
    with engine.begin() as conn:
        oid = conn.execute(text(
            "INSERT INTO ownership_overrides (entity_id, investment_id, effective_date, reason, entered_by) "
            "VALUES (:e, :i, :d, :r, :by) RETURNING id"),
            {"e": ent, "i": inv, "d": eff, "r": reason, "by": actor}).scalar()
        for i, a in clean:
            conn.execute(text("INSERT INTO ownership_override_lines (override_id, investor_id, amount) "
                              "VALUES (:o, :i, :a)"), {"o": oid, "i": i, "a": a})
    out = next(o for o in list_overrides(engine) if o["id"] == oid)
    out["warnings"] = warnings
    return out


def remove(override_id: int, actor: str, engine=None) -> dict:
    """Mark a set removed. The record stays: who entered it, why, and who removed it."""
    engine = ensure(engine)
    with engine.begin() as conn:
        n = conn.execute(text(
            "UPDATE ownership_overrides SET deleted_at = CURRENT_TIMESTAMP, deleted_by = :by "
            "WHERE id = :i AND deleted_at IS NULL"), {"i": int(override_id), "by": actor}).rowcount
    if not n:
        raise LookupError("No active override %s" % override_id)
    return next(o for o in list_overrides(engine) if o["id"] == int(override_id))
