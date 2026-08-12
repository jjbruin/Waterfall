"""
prospect_service.py
Business logic for the New Business deal pipeline.

Manages prospect deals, properties, entities/investors, assumptions,
and activity log. Properties link to lease reviews for due diligence.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL — PostgreSQL (SERIAL, DOUBLE PRECISION)
# ---------------------------------------------------------------------------

PROSPECT_DDL_PG = [
    """
    CREATE TABLE IF NOT EXISTS prospect_deals (
        id              SERIAL PRIMARY KEY,
        deal_name       TEXT NOT NULL,
        location        TEXT,
        asset_type      TEXT,
        partner_name    TEXT,
        source_broker   TEXT,
        assigned_to     TEXT,
        stage           TEXT DEFAULT 'lead',
        pass_reason     TEXT,
        target_close    TEXT,
        purchase_price  DOUBLE PRECISION,
        closing_cost_pct DOUBLE PRECISION DEFAULT 0.02,
        capex_at_close  DOUBLE PRECISION DEFAULT 0,
        notes           TEXT,
        onboarded_vcode TEXT,
        created_by      TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prospect_properties (
        id              SERIAL PRIMARY KEY,
        prospect_id     INTEGER NOT NULL REFERENCES prospect_deals(id) ON DELETE CASCADE,
        property_name   TEXT NOT NULL,
        address         TEXT,
        city            TEXT,
        state           TEXT,
        zip             TEXT,
        asset_type      TEXT,
        gla_sf          DOUBLE PRECISION,
        units           INTEGER,
        year_built      INTEGER,
        acreage         DOUBLE PRECISION,
        property_price  DOUBLE PRECISION,
        occupancy_pct   DOUBLE PRECISION,
        noi_in_place    DOUBLE PRECISION,
        notes           TEXT,
        onboarded_vcode TEXT,
        sort_order      INTEGER DEFAULT 0,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prospect_entities (
        id              SERIAL PRIMARY KEY,
        prospect_id     INTEGER NOT NULL REFERENCES prospect_deals(id) ON DELETE CASCADE,
        entity_name     TEXT NOT NULL,
        entity_type     TEXT,
        planned_entity_id TEXT,
        parent_entity_id INTEGER,
        ownership_pct   DOUBLE PRECISION,
        role            TEXT,
        notes           TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prospect_investors (
        id              SERIAL PRIMARY KEY,
        entity_id       INTEGER NOT NULL REFERENCES prospect_entities(id) ON DELETE CASCADE,
        investor_name   TEXT NOT NULL,
        planned_investor_id TEXT,
        commitment      DOUBLE PRECISION,
        ownership_pct   DOUBLE PRECISION,
        investor_type   TEXT,
        notes           TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prospect_assumptions (
        id              SERIAL PRIMARY KEY,
        prospect_id     INTEGER NOT NULL REFERENCES prospect_deals(id) ON DELETE CASCADE,
        version         INTEGER DEFAULT 1,
        version_label   TEXT DEFAULT 'Base Case',
        debt_amount     DOUBLE PRECISION,
        debt_rate       DOUBLE PRECISION,
        debt_term_months INTEGER DEFAULT 84,
        io_months       INTEGER DEFAULT 60,
        amort_months    INTEGER DEFAULT 360,
        origination_fee DOUBLE PRECISION DEFAULT 0.0025,
        psc_equity_pct  DOUBLE PRECISION DEFAULT 0.90,
        pref_rate       DOUBLE PRECISION DEFAULT 0.08,
        promote_pct     DOUBLE PRECISION DEFAULT 0.20,
        am_fee_pct      DOUBLE PRECISION DEFAULT 0.0095,
        annual_expenses DOUBLE PRECISION DEFAULT 7500,
        exit_cap_rate   DOUBLE PRECISION,
        selling_cost_pct DOUBLE PRECISION DEFAULT 0.02,
        hold_years      INTEGER DEFAULT 7,
        capex_reserve_psf DOUBLE PRECISION DEFAULT 0.80,
        noi_year1       DOUBLE PRECISION,
        noi_growth_rate DOUBLE PRECISION DEFAULT 0.02,
        crossed_vcodes  TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prospect_cashflows (
        id              SERIAL PRIMARY KEY,
        prospect_id     INTEGER NOT NULL REFERENCES prospect_deals(id) ON DELETE CASCADE,
        property_id     INTEGER REFERENCES prospect_properties(id) ON DELETE SET NULL,
        version         INTEGER DEFAULT 1,
        period_date     TEXT,
        revenue         DOUBLE PRECISION,
        expenses        DOUBLE PRECISION,
        noi             DOUBLE PRECISION,
        capex           DOUBLE PRECISION,
        other           DOUBLE PRECISION,
        source          TEXT DEFAULT 'manual'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prospect_activity (
        id              SERIAL PRIMARY KEY,
        prospect_id     INTEGER NOT NULL REFERENCES prospect_deals(id) ON DELETE CASCADE,
        username        TEXT,
        action          TEXT,
        note            TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]

# SQLite variants
PROSPECT_DDL_SQLITE = [
    ddl.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
       .replace('DOUBLE PRECISION', 'REAL')
       .replace('BOOLEAN', 'INTEGER')
       .replace('ON DELETE CASCADE', '')
       .replace('ON DELETE SET NULL', '')
       .replace('REFERENCES prospect_deals(id)', '')
       .replace('REFERENCES prospect_entities(id)', '')
       .replace('REFERENCES prospect_properties(id)', '')
    for ddl in PROSPECT_DDL_PG
]

PIPELINE_STAGES = [
    'lead', 'screening', 'loi', 'due_diligence',
    'ic_review', 'closing', 'closed', 'passed',
]


def ensure_prospect_tables(engine):
    """Create prospect tables if they don't exist."""
    from sqlalchemy import text
    is_pg = 'postgresql' in str(engine.url)
    ddl_list = PROSPECT_DDL_PG if is_pg else PROSPECT_DDL_SQLITE
    with engine.connect() as conn:
        for ddl in ddl_list:
            conn.execute(text(ddl))

        # Migrate: add prospect_property_id to lease_reviews if missing
        try:
            if is_pg:
                conn.execute(text("""
                    ALTER TABLE lease_reviews
                    ADD COLUMN IF NOT EXISTS prospect_property_id INTEGER
                """))
            else:
                # SQLite: check if column exists first
                cols = conn.execute(text(
                    "PRAGMA table_info(lease_reviews)"
                )).fetchall()
                col_names = [c[1] for c in cols]
                if 'prospect_property_id' not in col_names:
                    conn.execute(text(
                        "ALTER TABLE lease_reviews ADD COLUMN prospect_property_id INTEGER"
                    ))
        except Exception as e:
            logger.debug(f"prospect_property_id column migration: {e}")

        conn.commit()
    logger.info("Prospect tables ensured")


# ---------------------------------------------------------------------------
# Deal CRUD
# ---------------------------------------------------------------------------

def list_deals(engine, stage: Optional[str] = None,
               assigned_to: Optional[str] = None) -> List[Dict]:
    """List prospect deals with optional filters."""
    sql = """
        SELECT d.id, d.deal_name, d.location, d.asset_type, d.partner_name,
               d.stage, d.assigned_to, d.target_close, d.purchase_price,
               d.source_broker, d.onboarded_vcode,
               d.created_by, d.created_at, d.updated_at,
               (SELECT COUNT(*) FROM prospect_properties WHERE prospect_id = d.id) as property_count
        FROM prospect_deals d
        WHERE 1=1
    """
    params: Dict[str, Any] = {}
    if stage:
        sql += " AND d.stage = :stage"
        params['stage'] = stage
    if assigned_to:
        sql += " AND d.assigned_to = :assigned_to"
        params['assigned_to'] = assigned_to
    sql += " ORDER BY d.updated_at DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    return [{
        'id': r[0], 'deal_name': r[1], 'location': r[2], 'asset_type': r[3],
        'partner_name': r[4], 'stage': r[5], 'assigned_to': r[6],
        'target_close': r[7], 'purchase_price': r[8], 'source_broker': r[9],
        'onboarded_vcode': r[10], 'created_by': r[11],
        'created_at': str(r[12]) if r[12] else None,
        'updated_at': str(r[13]) if r[13] else None,
        'property_count': r[14],
    } for r in rows]


def get_deal(engine, deal_id: int) -> Optional[Dict]:
    """Get full deal detail including properties and entities."""
    with engine.connect() as conn:
        deal = conn.execute(text("""
            SELECT id, deal_name, location, asset_type, partner_name,
                   source_broker, assigned_to, stage, pass_reason,
                   target_close, purchase_price, closing_cost_pct,
                   capex_at_close, notes, onboarded_vcode,
                   created_by, created_at, updated_at
            FROM prospect_deals WHERE id = :did
        """), {'did': deal_id}).fetchone()
        if not deal:
            return None

        props = conn.execute(text("""
            SELECT p.id, p.property_name, p.address, p.city, p.state, p.zip,
                   p.asset_type, p.gla_sf, p.units, p.year_built, p.acreage,
                   p.property_price, p.occupancy_pct, p.noi_in_place,
                   p.notes, p.onboarded_vcode, p.sort_order,
                   (SELECT lr.id FROM lease_reviews lr
                    WHERE lr.prospect_property_id = p.id LIMIT 1) as lease_review_id
            FROM prospect_properties p
            WHERE p.prospect_id = :did
            ORDER BY p.sort_order, p.property_name
        """), {'did': deal_id}).fetchall()

        entities = conn.execute(text("""
            SELECT id, entity_name, entity_type, planned_entity_id,
                   parent_entity_id, ownership_pct, role, notes
            FROM prospect_entities
            WHERE prospect_id = :did
            ORDER BY entity_type, entity_name
        """), {'did': deal_id}).fetchall()

        # Investors grouped by entity
        entity_ids = [e[0] for e in entities]
        investors = []
        if entity_ids:
            placeholders = ','.join(str(eid) for eid in entity_ids)
            investors = conn.execute(text(f"""
                SELECT id, entity_id, investor_name, planned_investor_id,
                       commitment, ownership_pct, investor_type, notes
                FROM prospect_investors
                WHERE entity_id IN ({placeholders})
                ORDER BY investor_name
            """)).fetchall()

    inv_by_entity: Dict[int, list] = {}
    for inv in investors:
        eid = inv[1]
        if eid not in inv_by_entity:
            inv_by_entity[eid] = []
        inv_by_entity[eid].append({
            'id': inv[0], 'entity_id': inv[1], 'investor_name': inv[2],
            'planned_investor_id': inv[3], 'commitment': inv[4],
            'ownership_pct': inv[5], 'investor_type': inv[6], 'notes': inv[7],
        })

    return {
        'deal': {
            'id': deal[0], 'deal_name': deal[1], 'location': deal[2],
            'asset_type': deal[3], 'partner_name': deal[4],
            'source_broker': deal[5], 'assigned_to': deal[6],
            'stage': deal[7], 'pass_reason': deal[8],
            'target_close': deal[9], 'purchase_price': deal[10],
            'closing_cost_pct': deal[11], 'capex_at_close': deal[12],
            'notes': deal[13], 'onboarded_vcode': deal[14],
            'created_by': deal[15],
            'created_at': str(deal[16]) if deal[16] else None,
            'updated_at': str(deal[17]) if deal[17] else None,
        },
        'properties': [{
            'id': p[0], 'property_name': p[1], 'address': p[2],
            'city': p[3], 'state': p[4], 'zip': p[5],
            'asset_type': p[6], 'gla_sf': p[7], 'units': p[8],
            'year_built': p[9], 'acreage': p[10], 'property_price': p[11],
            'occupancy_pct': p[12], 'noi_in_place': p[13],
            'notes': p[14], 'onboarded_vcode': p[15], 'sort_order': p[16],
            'lease_review_id': p[17],
        } for p in props],
        'entities': [{
            'id': e[0], 'entity_name': e[1], 'entity_type': e[2],
            'planned_entity_id': e[3], 'parent_entity_id': e[4],
            'ownership_pct': e[5], 'role': e[6], 'notes': e[7],
            'investors': inv_by_entity.get(e[0], []),
        } for e in entities],
    }


def create_deal(engine, data: Dict, username: str) -> int:
    """Create a new prospect deal and log activity."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO prospect_deals
                (deal_name, location, asset_type, partner_name,
                 source_broker, assigned_to, stage, target_close,
                 purchase_price, closing_cost_pct, capex_at_close,
                 notes, created_by)
            VALUES (:deal_name, :location, :asset_type, :partner_name,
                    :source_broker, :assigned_to, :stage, :target_close,
                    :purchase_price, :closing_cost_pct, :capex_at_close,
                    :notes, :created_by)
            RETURNING id
        """), {
            'deal_name': data['deal_name'],
            'location': data.get('location', ''),
            'asset_type': data.get('asset_type', ''),
            'partner_name': data.get('partner_name', ''),
            'source_broker': data.get('source_broker', ''),
            'assigned_to': data.get('assigned_to', ''),
            'stage': data.get('stage', 'lead'),
            'target_close': data.get('target_close'),
            'purchase_price': data.get('purchase_price'),
            'closing_cost_pct': data.get('closing_cost_pct', 0.02),
            'capex_at_close': data.get('capex_at_close', 0),
            'notes': data.get('notes', ''),
            'created_by': username,
        })
        deal_id = result.fetchone()[0]

        # Log creation
        conn.execute(text("""
            INSERT INTO prospect_activity (prospect_id, username, action, note)
            VALUES (:pid, :user, 'created', :note)
        """), {'pid': deal_id, 'user': username,
               'note': f"Deal created: {data['deal_name']}"})

        conn.commit()
    return deal_id


def update_deal(engine, deal_id: int, data: Dict, username: str) -> bool:
    """Update a prospect deal. Logs stage changes."""
    with engine.connect() as conn:
        # Get current stage for comparison
        current = conn.execute(text(
            "SELECT stage FROM prospect_deals WHERE id = :did"
        ), {'did': deal_id}).fetchone()
        if not current:
            return False

        old_stage = current[0]
        new_stage = data.get('stage', old_stage)

        conn.execute(text("""
            UPDATE prospect_deals SET
                deal_name = COALESCE(:deal_name, deal_name),
                location = COALESCE(:location, location),
                asset_type = COALESCE(:asset_type, asset_type),
                partner_name = COALESCE(:partner_name, partner_name),
                source_broker = COALESCE(:source_broker, source_broker),
                assigned_to = COALESCE(:assigned_to, assigned_to),
                stage = COALESCE(:stage, stage),
                pass_reason = :pass_reason,
                target_close = :target_close,
                purchase_price = :purchase_price,
                closing_cost_pct = :closing_cost_pct,
                capex_at_close = :capex_at_close,
                notes = :notes,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :did
        """), {
            'did': deal_id,
            'deal_name': data.get('deal_name'),
            'location': data.get('location'),
            'asset_type': data.get('asset_type'),
            'partner_name': data.get('partner_name'),
            'source_broker': data.get('source_broker'),
            'assigned_to': data.get('assigned_to'),
            'stage': data.get('stage'),
            'pass_reason': data.get('pass_reason'),
            'target_close': data.get('target_close'),
            'purchase_price': data.get('purchase_price'),
            'closing_cost_pct': data.get('closing_cost_pct'),
            'capex_at_close': data.get('capex_at_close'),
            'notes': data.get('notes'),
        })

        # Log stage change
        if new_stage and new_stage != old_stage:
            conn.execute(text("""
                INSERT INTO prospect_activity (prospect_id, username, action, note)
                VALUES (:pid, :user, 'stage_change', :note)
            """), {'pid': deal_id, 'user': username,
                   'note': f"Stage: {old_stage} → {new_stage}"})

        conn.commit()
    return True


def delete_deal(engine, deal_id: int) -> bool:
    """Delete a prospect deal and all related data (CASCADE)."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "DELETE FROM prospect_deals WHERE id = :did"
        ), {'did': deal_id})
        conn.commit()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Property CRUD
# ---------------------------------------------------------------------------

def list_properties(engine, deal_id: int) -> List[Dict]:
    """List properties for a prospect deal."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.id, p.property_name, p.address, p.city, p.state, p.zip,
                   p.asset_type, p.gla_sf, p.units, p.year_built, p.acreage,
                   p.property_price, p.occupancy_pct, p.noi_in_place,
                   p.notes, p.onboarded_vcode, p.sort_order,
                   (SELECT lr.id FROM lease_reviews lr
                    WHERE lr.prospect_property_id = p.id LIMIT 1) as lease_review_id
            FROM prospect_properties p
            WHERE p.prospect_id = :did
            ORDER BY p.sort_order, p.property_name
        """), {'did': deal_id}).fetchall()

    return [{
        'id': r[0], 'property_name': r[1], 'address': r[2],
        'city': r[3], 'state': r[4], 'zip': r[5],
        'asset_type': r[6], 'gla_sf': r[7], 'units': r[8],
        'year_built': r[9], 'acreage': r[10], 'property_price': r[11],
        'occupancy_pct': r[12], 'noi_in_place': r[13],
        'notes': r[14], 'onboarded_vcode': r[15], 'sort_order': r[16],
        'lease_review_id': r[17],
    } for r in rows]


def create_property(engine, deal_id: int, data: Dict, username: str) -> int:
    """Add a property to a prospect deal."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO prospect_properties
                (prospect_id, property_name, address, city, state, zip,
                 asset_type, gla_sf, units, year_built, acreage,
                 property_price, occupancy_pct, noi_in_place, notes, sort_order)
            VALUES (:pid, :property_name, :address, :city, :state, :zip,
                    :asset_type, :gla_sf, :units, :year_built, :acreage,
                    :property_price, :occupancy_pct, :noi_in_place, :notes, :sort_order)
            RETURNING id
        """), {
            'pid': deal_id,
            'property_name': data['property_name'],
            'address': data.get('address', ''),
            'city': data.get('city', ''),
            'state': data.get('state', ''),
            'zip': data.get('zip', ''),
            'asset_type': data.get('asset_type', ''),
            'gla_sf': data.get('gla_sf'),
            'units': data.get('units'),
            'year_built': data.get('year_built'),
            'acreage': data.get('acreage'),
            'property_price': data.get('property_price'),
            'occupancy_pct': data.get('occupancy_pct'),
            'noi_in_place': data.get('noi_in_place'),
            'notes': data.get('notes', ''),
            'sort_order': data.get('sort_order', 0),
        })
        prop_id = result.fetchone()[0]

        conn.execute(text("""
            INSERT INTO prospect_activity (prospect_id, username, action, note)
            VALUES (:pid, :user, 'property_added', :note)
        """), {'pid': deal_id, 'user': username,
               'note': f"Property added: {data['property_name']}"})

        conn.commit()
    return prop_id


def update_property(engine, property_id: int, data: Dict) -> bool:
    """Update a prospect property."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            UPDATE prospect_properties SET
                property_name = COALESCE(:property_name, property_name),
                address = :address, city = :city, state = :state, zip = :zip,
                asset_type = :asset_type, gla_sf = :gla_sf, units = :units,
                year_built = :year_built, acreage = :acreage,
                property_price = :property_price, occupancy_pct = :occupancy_pct,
                noi_in_place = :noi_in_place, notes = :notes,
                sort_order = COALESCE(:sort_order, sort_order),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :pid
        """), {
            'pid': property_id,
            'property_name': data.get('property_name'),
            'address': data.get('address'),
            'city': data.get('city'),
            'state': data.get('state'),
            'zip': data.get('zip'),
            'asset_type': data.get('asset_type'),
            'gla_sf': data.get('gla_sf'),
            'units': data.get('units'),
            'year_built': data.get('year_built'),
            'acreage': data.get('acreage'),
            'property_price': data.get('property_price'),
            'occupancy_pct': data.get('occupancy_pct'),
            'noi_in_place': data.get('noi_in_place'),
            'notes': data.get('notes'),
            'sort_order': data.get('sort_order'),
        })
        conn.commit()
    return result.rowcount > 0


def delete_property(engine, property_id: int) -> bool:
    """Remove a property from a prospect deal."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "DELETE FROM prospect_properties WHERE id = :pid"
        ), {'pid': property_id})
        conn.commit()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Entity / Investor CRUD
# ---------------------------------------------------------------------------

def create_entity(engine, deal_id: int, data: Dict) -> int:
    """Add an entity to a prospect deal."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO prospect_entities
                (prospect_id, entity_name, entity_type, planned_entity_id,
                 parent_entity_id, ownership_pct, role, notes)
            VALUES (:pid, :entity_name, :entity_type, :planned_entity_id,
                    :parent_entity_id, :ownership_pct, :role, :notes)
            RETURNING id
        """), {
            'pid': deal_id,
            'entity_name': data['entity_name'],
            'entity_type': data.get('entity_type', ''),
            'planned_entity_id': data.get('planned_entity_id', ''),
            'parent_entity_id': data.get('parent_entity_id'),
            'ownership_pct': data.get('ownership_pct'),
            'role': data.get('role', ''),
            'notes': data.get('notes', ''),
        })
        entity_id = result.fetchone()[0]
        conn.commit()
    return entity_id


def update_entity(engine, entity_id: int, data: Dict) -> bool:
    """Update a prospect entity."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            UPDATE prospect_entities SET
                entity_name = COALESCE(:entity_name, entity_name),
                entity_type = :entity_type,
                planned_entity_id = :planned_entity_id,
                parent_entity_id = :parent_entity_id,
                ownership_pct = :ownership_pct,
                role = :role,
                notes = :notes,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :eid
        """), {
            'eid': entity_id,
            'entity_name': data.get('entity_name'),
            'entity_type': data.get('entity_type'),
            'planned_entity_id': data.get('planned_entity_id'),
            'parent_entity_id': data.get('parent_entity_id'),
            'ownership_pct': data.get('ownership_pct'),
            'role': data.get('role'),
            'notes': data.get('notes'),
        })
        conn.commit()
    return result.rowcount > 0


def delete_entity(engine, entity_id: int) -> bool:
    """Remove an entity (and its investors via CASCADE)."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "DELETE FROM prospect_entities WHERE id = :eid"
        ), {'eid': entity_id})
        conn.commit()
    return result.rowcount > 0


def create_investor(engine, entity_id: int, data: Dict) -> int:
    """Add an investor to an entity."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO prospect_investors
                (entity_id, investor_name, planned_investor_id,
                 commitment, ownership_pct, investor_type, notes)
            VALUES (:eid, :investor_name, :planned_investor_id,
                    :commitment, :ownership_pct, :investor_type, :notes)
            RETURNING id
        """), {
            'eid': entity_id,
            'investor_name': data['investor_name'],
            'planned_investor_id': data.get('planned_investor_id', ''),
            'commitment': data.get('commitment'),
            'ownership_pct': data.get('ownership_pct'),
            'investor_type': data.get('investor_type', ''),
            'notes': data.get('notes', ''),
        })
        inv_id = result.fetchone()[0]
        conn.commit()
    return inv_id


def update_investor(engine, investor_id: int, data: Dict) -> bool:
    """Update an investor."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            UPDATE prospect_investors SET
                investor_name = COALESCE(:investor_name, investor_name),
                planned_investor_id = :planned_investor_id,
                commitment = :commitment,
                ownership_pct = :ownership_pct,
                investor_type = :investor_type,
                notes = :notes,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :iid
        """), {
            'iid': investor_id,
            'investor_name': data.get('investor_name'),
            'planned_investor_id': data.get('planned_investor_id'),
            'commitment': data.get('commitment'),
            'ownership_pct': data.get('ownership_pct'),
            'investor_type': data.get('investor_type'),
            'notes': data.get('notes'),
        })
        conn.commit()
    return result.rowcount > 0


def delete_investor(engine, investor_id: int) -> bool:
    """Remove an investor."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "DELETE FROM prospect_investors WHERE id = :iid"
        ), {'iid': investor_id})
        conn.commit()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Activity Log
# ---------------------------------------------------------------------------

def get_activity(engine, deal_id: int) -> List[Dict]:
    """Get activity log for a prospect deal."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, username, action, note, created_at
            FROM prospect_activity
            WHERE prospect_id = :pid
            ORDER BY created_at DESC
        """), {'pid': deal_id}).fetchall()

    return [{
        'id': r[0], 'username': r[1], 'action': r[2],
        'note': r[3], 'created_at': str(r[4]) if r[4] else None,
    } for r in rows]


def add_activity_note(engine, deal_id: int, username: str, note: str) -> int:
    """Add a note to the activity log."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO prospect_activity (prospect_id, username, action, note)
            VALUES (:pid, :user, 'note', :note)
            RETURNING id
        """), {'pid': deal_id, 'user': username, 'note': note})
        note_id = result.fetchone()[0]
        conn.commit()
    return note_id


# ---------------------------------------------------------------------------
# Lease Review Link
# ---------------------------------------------------------------------------

def create_lease_review_for_property(engine, deal_id: int, property_id: int,
                                     username: str) -> int:
    """Create a lease review linked to a prospect property.

    Copies property info into the lease review's denormalized fields.
    Returns the new lease_review id.
    """
    with engine.connect() as conn:
        prop = conn.execute(text("""
            SELECT property_name, address, city, state, gla_sf
            FROM prospect_properties
            WHERE id = :pid AND prospect_id = :did
        """), {'pid': property_id, 'did': deal_id}).fetchone()

        if not prop:
            raise ValueError(f"Property {property_id} not found in deal {deal_id}")

        full_address = ', '.join(filter(None, [prop[1], prop[2], prop[3]]))

        result = conn.execute(text("""
            INSERT INTO lease_reviews
                (property_name, property_address, total_gla,
                 prospect_property_id, status, created_by)
            VALUES (:name, :addr, :gla, :ppid, 'in_progress', :user)
            RETURNING id
        """), {
            'name': prop[0],
            'addr': full_address,
            'gla': prop[4],
            'ppid': property_id,
            'user': username,
        })
        review_id = result.fetchone()[0]

        conn.execute(text("""
            INSERT INTO prospect_activity (prospect_id, username, action, note)
            VALUES (:did, :user, 'lease_review_created', :note)
        """), {'did': deal_id, 'user': username,
               'note': f"Lease review created for {prop[0]}"})

        conn.commit()
    return review_id
