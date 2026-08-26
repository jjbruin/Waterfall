"""
prospect_service.py
Business logic for the New Business deal pipeline.

Manages prospect deals, properties, entities/investors, assumptions,
and activity log. Properties link to lease reviews for due diligence.

Vcodes: Auto-generated N-series codes (N0000001, N0000002, ...) for deals.
Properties within a portfolio deal get child codes (N0000001-01, N0000001-02, ...).
"""

import json
import logging
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


def _next_deal_vcode(conn) -> str:
    """Generate the next available N-series vcode (N0000001, N0000002, ...)."""
    row = conn.execute(text(
        "SELECT vcode FROM prospect_deals "
        "WHERE vcode IS NOT NULL AND vcode LIKE 'N%' "
        "ORDER BY vcode DESC LIMIT 1"
    )).fetchone()
    if row and row[0]:
        num = int(row[0][1:]) + 1
    else:
        num = 1
    return f"N{num:07d}"


def _next_property_vcode(conn, deal_vcode: str) -> str:
    """Generate the next child vcode for a property within a deal (N0000001-01, -02, ...)."""
    row = conn.execute(text(
        "SELECT vcode FROM prospect_properties "
        "WHERE vcode IS NOT NULL AND vcode LIKE :prefix "
        "ORDER BY vcode DESC LIMIT 1"
    ), {'prefix': f'{deal_vcode}-%'}).fetchone()
    if row and row[0]:
        suffix = int(row[0].split('-')[-1]) + 1
    else:
        suffix = 1
    return f"{deal_vcode}-{suffix:02d}"


# ---------------------------------------------------------------------------
# DDL — PostgreSQL (SERIAL, DOUBLE PRECISION)
# ---------------------------------------------------------------------------

PROSPECT_DDL_PG = [
    """
    CREATE TABLE IF NOT EXISTS prospect_deals (
        id              SERIAL PRIMARY KEY,
        vcode           TEXT UNIQUE,
        deal_name       TEXT NOT NULL,
        deal_structure  TEXT DEFAULT 'single_property',
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
        vcode           TEXT UNIQUE,
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
        lender          TEXT,
        rate_type       TEXT DEFAULT 'fixed',
        rate_index      TEXT,
        rate_index_term TEXT,
        rate_spread_bps DOUBLE PRECISION,
        rate_cushion_bps DOUBLE PRECISION,
        extension_count INTEGER,
        extension_months INTEGER DEFAULT 12,
        extension_conditions TEXT,
        prepay_type     TEXT,
        prepay_schedule TEXT,
        max_ltv         DOUBLE PRECISION,
        max_ltc         DOUBLE PRECISION,
        min_dscr        DOUBLE PRECISION,
        dscr_test_start TEXT,
        min_debt_yield  DOUBLE PRECISION,
        origination_fee_bps DOUBLE PRECISION,
        earnout_notes   TEXT,
        guarantor_notes TEXT,
        capital_uses_json TEXT,
        capital_sources_json TEXT,
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
        source          TEXT DEFAULT 'manual',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        # Migrate: add vcode and deal_structure columns to prospect tables
        _migrate_columns = [
            ('prospect_deals', 'vcode', 'TEXT'),
            ('prospect_deals', 'deal_structure', "TEXT DEFAULT 'single_property'"),
            ('prospect_properties', 'vcode', 'TEXT'),
            ('prospect_cashflows', 'created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            # New assumption fields (Aug 2026)
            ('prospect_assumptions', 'lender', 'TEXT'),
            ('prospect_assumptions', 'rate_type', "TEXT DEFAULT 'fixed'"),
            ('prospect_assumptions', 'rate_index', 'TEXT'),
            ('prospect_assumptions', 'rate_index_term', 'TEXT'),
            ('prospect_assumptions', 'rate_spread_bps', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'rate_cushion_bps', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'extension_count', 'INTEGER'),
            ('prospect_assumptions', 'extension_months', 'INTEGER DEFAULT 12'),
            ('prospect_assumptions', 'extension_conditions', 'TEXT'),
            ('prospect_assumptions', 'prepay_type', 'TEXT'),
            ('prospect_assumptions', 'prepay_schedule', 'TEXT'),
            ('prospect_assumptions', 'max_ltv', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'max_ltc', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'min_dscr', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'dscr_test_start', 'TEXT'),
            ('prospect_assumptions', 'min_debt_yield', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'origination_fee_bps', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'earnout_notes', 'TEXT'),
            ('prospect_assumptions', 'guarantor_notes', 'TEXT'),
            ('prospect_assumptions', 'capital_uses_json', 'TEXT'),
            ('prospect_assumptions', 'capital_sources_json', 'TEXT'),
            # Operating override fields (Aug 2026)
            ('prospect_assumptions', 'mgmt_fee_pct', 'DOUBLE PRECISION'),
            ('prospect_assumptions', 'replacement_reserve_psf', 'DOUBLE PRECISION'),
            # Planned refinancing within the hold (Aug 2026)
            ('prospect_assumptions', 'planned_refi_json', 'TEXT'),
            # Line-item cashflow columns
            ('prospect_cashflows', 'vaccount', 'INTEGER'),
            ('prospect_cashflows', 'line_item', 'TEXT'),
            ('prospect_cashflows', 'amount', 'DOUBLE PRECISION'),
        ]
        for table, col, col_type in _migrate_columns:
            try:
                if is_pg:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type}"
                    ))
                else:
                    cols = conn.execute(text(
                        f"PRAGMA table_info({table})"
                    )).fetchall()
                    if col not in [c[1] for c in cols]:
                        conn.execute(text(
                            f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                        ))
            except Exception as e:
                logger.debug(f"Migration {table}.{col}: {e}")

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
               (SELECT COUNT(*) FROM prospect_properties WHERE prospect_id = d.id) as property_count,
               d.vcode, d.deal_structure
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
        'vcode': r[15], 'deal_structure': r[16],
    } for r in rows]


def get_deal(engine, deal_id: int) -> Optional[Dict]:
    """Get full deal detail including properties and entities."""
    with engine.connect() as conn:
        deal = conn.execute(text("""
            SELECT id, deal_name, location, asset_type, partner_name,
                   source_broker, assigned_to, stage, pass_reason,
                   target_close, purchase_price, closing_cost_pct,
                   capex_at_close, notes, onboarded_vcode,
                   created_by, created_at, updated_at,
                   vcode, deal_structure
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
                    WHERE lr.prospect_property_id = p.id LIMIT 1) as lease_review_id,
                   p.vcode
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
            'vcode': deal[18], 'deal_structure': deal[19],
        },
        'properties': [{
            'id': p[0], 'property_name': p[1], 'address': p[2],
            'city': p[3], 'state': p[4], 'zip': p[5],
            'asset_type': p[6], 'gla_sf': p[7], 'units': p[8],
            'year_built': p[9], 'acreage': p[10], 'property_price': p[11],
            'occupancy_pct': p[12], 'noi_in_place': p[13],
            'notes': p[14], 'onboarded_vcode': p[15], 'sort_order': p[16],
            'lease_review_id': p[17], 'vcode': p[18],
        } for p in props],
        'entities': [{
            'id': e[0], 'entity_name': e[1], 'entity_type': e[2],
            'planned_entity_id': e[3], 'parent_entity_id': e[4],
            'ownership_pct': e[5], 'role': e[6], 'notes': e[7],
            'investors': inv_by_entity.get(e[0], []),
        } for e in entities],
    }


def create_deal(engine, data: Dict, username: str) -> Dict:
    """Create a new prospect deal with auto-generated vcode and log activity.

    Returns dict with 'id' and 'vcode'.
    """
    with engine.connect() as conn:
        vcode = _next_deal_vcode(conn)

        result = conn.execute(text("""
            INSERT INTO prospect_deals
                (vcode, deal_name, deal_structure, location, asset_type,
                 partner_name, source_broker, assigned_to, stage, target_close,
                 purchase_price, closing_cost_pct, capex_at_close,
                 notes, created_by)
            VALUES (:vcode, :deal_name, :deal_structure, :location, :asset_type,
                    :partner_name, :source_broker, :assigned_to, :stage, :target_close,
                    :purchase_price, :closing_cost_pct, :capex_at_close,
                    :notes, :created_by)
            RETURNING id
        """), {
            'vcode': vcode,
            'deal_name': data['deal_name'],
            'deal_structure': data.get('deal_structure', 'single_property'),
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

        conn.execute(text("""
            INSERT INTO prospect_activity (prospect_id, username, action, note)
            VALUES (:pid, :user, 'created', :note)
        """), {'pid': deal_id, 'user': username,
               'note': f"Deal created: {data['deal_name']} ({vcode})"})

        conn.commit()
    return {'id': deal_id, 'vcode': vcode}


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
                deal_structure = COALESCE(:deal_structure, deal_structure),
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
            'deal_structure': data.get('deal_structure'),
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
                    WHERE lr.prospect_property_id = p.id LIMIT 1) as lease_review_id,
                   p.vcode
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
        'lease_review_id': r[17], 'vcode': r[18],
    } for r in rows]


def create_property(engine, deal_id: int, data: Dict, username: str) -> Dict:
    """Add a property to a prospect deal with auto-generated child vcode.

    Returns dict with 'id' and 'vcode'.
    """
    with engine.connect() as conn:
        # Get parent deal vcode for child vcode generation
        deal_row = conn.execute(text(
            "SELECT vcode FROM prospect_deals WHERE id = :did"
        ), {'did': deal_id}).fetchone()
        deal_vcode = deal_row[0] if deal_row and deal_row[0] else f"N{deal_id:07d}"
        prop_vcode = _next_property_vcode(conn, deal_vcode)

        result = conn.execute(text("""
            INSERT INTO prospect_properties
                (prospect_id, vcode, property_name, address, city, state, zip,
                 asset_type, gla_sf, units, year_built, acreage,
                 property_price, occupancy_pct, noi_in_place, notes, sort_order)
            VALUES (:pid, :vcode, :property_name, :address, :city, :state, :zip,
                    :asset_type, :gla_sf, :units, :year_built, :acreage,
                    :property_price, :occupancy_pct, :noi_in_place, :notes, :sort_order)
            RETURNING id
        """), {
            'pid': deal_id,
            'vcode': prop_vcode,
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
               'note': f"Property added: {data['property_name']} ({prop_vcode})"})

        conn.commit()
    return {'id': prop_id, 'vcode': prop_vcode}


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


# ---------------------------------------------------------------------------
# Assumptions CRUD
# ---------------------------------------------------------------------------

ASSUMPTION_FIELDS = [
    'debt_amount', 'debt_rate', 'debt_term_months', 'io_months',
    'amort_months', 'origination_fee', 'psc_equity_pct', 'pref_rate',
    'promote_pct', 'am_fee_pct', 'annual_expenses', 'exit_cap_rate',
    'selling_cost_pct', 'hold_years', 'capex_reserve_psf',
    'noi_year1', 'noi_growth_rate', 'crossed_vcodes',
    'lender', 'rate_type', 'rate_index', 'rate_index_term',
    'rate_spread_bps', 'rate_cushion_bps',
    'extension_count', 'extension_months', 'extension_conditions',
    'prepay_type', 'prepay_schedule',
    'max_ltv', 'max_ltc', 'min_dscr', 'dscr_test_start', 'min_debt_yield',
    'origination_fee_bps', 'earnout_notes', 'guarantor_notes',
    'capital_uses_json', 'capital_sources_json',
    'mgmt_fee_pct', 'replacement_reserve_psf',
    'planned_refi_json',
]


def _assumption_row_to_dict(r, columns: list) -> Dict:
    """Convert a row tuple to dict using column name list."""
    d = {}
    for i, col in enumerate(columns):
        v = r[i]
        if col in ('created_at', 'updated_at'):
            d[col] = str(v) if v else None
        else:
            d[col] = v
    return d


def list_assumptions(engine, deal_id: int) -> List[Dict]:
    """List assumption versions for a prospect deal."""
    fields_sql = ', '.join(ASSUMPTION_FIELDS)
    cols = ['id', 'version', 'version_label'] + list(ASSUMPTION_FIELDS) + ['created_at', 'updated_at']
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, version, version_label, {fields_sql}, created_at, updated_at
            FROM prospect_assumptions
            WHERE prospect_id = :pid
            ORDER BY version
        """), {'pid': deal_id}).fetchall()

    return [_assumption_row_to_dict(r, cols) for r in rows]


def get_assumption(engine, assumption_id: int) -> Optional[Dict]:
    """Get a single assumption version."""
    fields_sql = ', '.join(ASSUMPTION_FIELDS)
    cols = ['id', 'prospect_id', 'version', 'version_label'] + list(ASSUMPTION_FIELDS) + ['created_at', 'updated_at']
    with engine.connect() as conn:
        r = conn.execute(text(f"""
            SELECT id, prospect_id, version, version_label, {fields_sql}, created_at, updated_at
            FROM prospect_assumptions
            WHERE id = :aid
        """), {'aid': assumption_id}).fetchone()
    if not r:
        return None
    return _assumption_row_to_dict(r, cols)


def save_assumptions(engine, deal_id: int, data: Dict) -> Dict:
    """Create or update assumption version. Returns {'id': ..., 'version': ...}."""
    assumption_id = data.get('id')
    with engine.connect() as conn:
        if assumption_id:
            # Snapshot the row being overwritten into the activity log first.
            # An UPDATE here has destroyed work before (a form holding
            # defaults auto-saved over a real capital budget); with the
            # snapshot, any overwrite is recoverable instead of final.
            try:
                prev = conn.execute(text(
                    "SELECT * FROM prospect_assumptions WHERE id = :aid"
                ), {'aid': assumption_id}).mappings().fetchone()
                if prev:
                    snapshot = json.dumps({k: str(v) for k, v in dict(prev).items()
                                           if v is not None})
                    conn.execute(text("""
                        INSERT INTO prospect_activity (prospect_id, username, action, note)
                        VALUES (:pid, :u, 'assumptions_overwritten', :note)
                    """), {
                        'pid': deal_id,
                        'u': data.get('_username') or 'system',
                        'note': f'Pre-update snapshot of assumptions id={assumption_id}: {snapshot}',
                    })
            except Exception as snap_err:  # never block a save on the snapshot
                logger.warning("Assumptions snapshot failed for id=%s: %s",
                               assumption_id, snap_err)

            # Update existing
            sets = ', '.join(f"{f} = :{f}" for f in ASSUMPTION_FIELDS)
            conn.execute(text(f"""
                UPDATE prospect_assumptions SET
                    version_label = :version_label,
                    {sets},
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :aid
            """), {
                'aid': assumption_id,
                'version_label': data.get('version_label', 'Base Case'),
                **{f: data.get(f) for f in ASSUMPTION_FIELDS},
            })
            version = data.get('version', 1)
        else:
            # Get next version number
            row = conn.execute(text(
                "SELECT COALESCE(MAX(version), 0) FROM prospect_assumptions WHERE prospect_id = :pid"
            ), {'pid': deal_id}).fetchone()
            version = (row[0] or 0) + 1

            fields_sql = ', '.join(ASSUMPTION_FIELDS)
            placeholders = ', '.join(f':{f}' for f in ASSUMPTION_FIELDS)
            result = conn.execute(text(f"""
                INSERT INTO prospect_assumptions
                    (prospect_id, version, version_label, {fields_sql})
                VALUES (:pid, :version, :version_label, {placeholders})
                RETURNING id
            """), {
                'pid': deal_id,
                'version': version,
                'version_label': data.get('version_label', 'Base Case'),
                **{f: data.get(f) for f in ASSUMPTION_FIELDS},
            })
            assumption_id = result.fetchone()[0]

        conn.commit()
    return {'id': assumption_id, 'version': version}


def delete_assumptions(engine, assumption_id: int) -> bool:
    """Delete an assumption version."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "DELETE FROM prospect_assumptions WHERE id = :aid"
        ), {'aid': assumption_id})
        conn.commit()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Property Cashflows CRUD
# ---------------------------------------------------------------------------

def import_property_cashflows(
    engine, deal_id: int, property_id: int,
    cashflows: List[Dict], source: str = 'excel',
    version: int = 1,
) -> Dict:
    """Import cash flow rows for a property, replacing existing rows for that version.

    cashflows: list of {'period_date', 'revenue', 'expenses', 'capex', 'noi'}
    """
    with engine.connect() as conn:
        # Verify property belongs to deal
        prop = conn.execute(text(
            "SELECT id FROM prospect_properties WHERE id = :pid AND prospect_id = :did"
        ), {'pid': property_id, 'did': deal_id}).fetchone()
        if not prop:
            raise ValueError(f"Property {property_id} not found in deal {deal_id}")

        # Delete existing cashflows for this property + version
        conn.execute(text("""
            DELETE FROM prospect_cashflows
            WHERE prospect_id = :did AND property_id = :pid AND version = :v
        """), {'did': deal_id, 'pid': property_id, 'v': version})

        # Insert new rows
        count = 0
        for cf in cashflows:
            conn.execute(text("""
                INSERT INTO prospect_cashflows
                    (prospect_id, property_id, version, period_date,
                     revenue, expenses, noi, capex, source)
                VALUES (:did, :pid, :v, :period_date,
                        :revenue, :expenses, :noi, :capex, :source)
            """), {
                'did': deal_id, 'pid': property_id, 'v': version,
                'period_date': cf.get('period_date'),
                'revenue': cf.get('revenue'),
                'expenses': cf.get('expenses'),
                'noi': cf.get('noi'),
                'capex': cf.get('capex'),
                'source': source,
            })
            count += 1

        conn.commit()
    return {'rows_imported': count, 'property_id': property_id, 'version': version}


def get_property_cashflows(engine, deal_id: int, property_id: int,
                           version: int = 1) -> List[Dict]:
    """Get cash flow rows for a property."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, period_date, revenue, expenses, noi, capex, other, source, created_at
            FROM prospect_cashflows
            WHERE prospect_id = :did AND property_id = :pid AND version = :v
            ORDER BY period_date
        """), {'did': deal_id, 'pid': property_id, 'v': version}).fetchall()

    return [{
        'id': r[0], 'period_date': r[1],
        'revenue': r[2], 'expenses': r[3], 'noi': r[4],
        'capex': r[5], 'other': r[6], 'source': r[7],
        'created_at': str(r[8]) if r[8] else None,
    } for r in rows]


def delete_property_cashflows(engine, deal_id: int, property_id: int,
                              version: int = 1) -> bool:
    """Delete cash flow rows for a property."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            DELETE FROM prospect_cashflows
            WHERE prospect_id = :did AND property_id = :pid AND version = :v
        """), {'did': deal_id, 'pid': property_id, 'v': version})
        conn.commit()
    return result.rowcount > 0


def import_property_line_items(engine, deal_id: int, property_id: int,
                               line_items: List[Dict], source: str = 'excel',
                               version: int = 1, frequency: str = 'monthly') -> Dict:
    """Import line-item cash flows with COA account assignments.

    Each line_item has: {label, vaccount, category, values: [{period_date, amount}]}
    Stores one row per (period_date, vAccount) in prospect_cashflows.
    Annual data is auto-spread to monthly.
    """
    from utils import month_end
    from datetime import date as date_cls

    with engine.connect() as conn:
        # Verify property belongs to deal
        prop = conn.execute(text(
            "SELECT id FROM prospect_properties WHERE id = :pid AND prospect_id = :did"
        ), {'pid': property_id, 'did': deal_id}).fetchone()
        if not prop:
            raise ValueError(f"Property {property_id} not found in deal {deal_id}")

        # Delete existing line-item rows for this property + version
        conn.execute(text("""
            DELETE FROM prospect_cashflows
            WHERE prospect_id = :did AND property_id = :pid AND version = :v
        """), {'did': deal_id, 'pid': property_id, 'v': version})

        count = 0
        for item in line_items:
            vaccount = item.get('vaccount')
            label = item.get('label', '')
            if not vaccount:
                continue  # Skip unmapped items

            for val in item.get('values', []):
                period_date = val.get('period_date')
                amount = val.get('amount', 0)
                if not period_date:
                    continue

                if frequency == 'annual':
                    # Spread annual to 12 monthly rows
                    try:
                        dt = pd.to_datetime(period_date)
                        year = dt.year
                    except Exception:
                        continue
                    monthly_amount = round(float(amount) / 12, 2)
                    for m in range(1, 13):
                        m_date = str(month_end(date_cls(year, m, 1)))
                        conn.execute(text("""
                            INSERT INTO prospect_cashflows
                                (prospect_id, property_id, version, period_date,
                                 vaccount, line_item, amount, source)
                            VALUES (:did, :pid, :v, :period_date,
                                    :vaccount, :line_item, :amount, :source)
                        """), {
                            'did': deal_id, 'pid': property_id, 'v': version,
                            'period_date': m_date,
                            'vaccount': int(vaccount),
                            'line_item': label,
                            'amount': monthly_amount,
                            'source': source,
                        })
                        count += 1
                else:
                    conn.execute(text("""
                        INSERT INTO prospect_cashflows
                            (prospect_id, property_id, version, period_date,
                             vaccount, line_item, amount, source)
                        VALUES (:did, :pid, :v, :period_date,
                                :vaccount, :line_item, :amount, :source)
                    """), {
                        'did': deal_id, 'pid': property_id, 'v': version,
                        'period_date': period_date,
                        'vaccount': int(vaccount),
                        'line_item': label,
                        'amount': float(amount),
                        'source': source,
                    })
                    count += 1

        conn.commit()
    return {'rows_imported': count, 'property_id': property_id, 'version': version}


def get_property_line_items(engine, deal_id: int, property_id: int,
                            version: int = 1) -> List[Dict]:
    """Get line-item cash flow rows for a property."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, period_date, vaccount, line_item, amount, source, created_at
            FROM prospect_cashflows
            WHERE prospect_id = :did AND property_id = :pid AND version = :v
                AND vaccount IS NOT NULL
            ORDER BY vaccount, period_date
        """), {'did': deal_id, 'pid': property_id, 'v': version}).fetchall()

    return [{
        'id': r[0], 'period_date': r[1], 'vaccount': r[2],
        'line_item': r[3], 'amount': r[4], 'source': r[5],
        'created_at': str(r[6]) if r[6] else None,
    } for r in rows]


def get_deal_line_items_by_property(engine, deal_id: int,
                                     version: int = 1) -> Dict[int, List[Dict]]:
    """Get all line-item cashflows for a deal grouped by property_id."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT property_id, period_date, vaccount, line_item, amount, source
            FROM prospect_cashflows
            WHERE prospect_id = :did AND version = :v
                AND property_id IS NOT NULL AND vaccount IS NOT NULL
            ORDER BY property_id, period_date, vaccount
        """), {'did': deal_id, 'v': version}).fetchall()

    result: Dict[int, List[Dict]] = {}
    for r in rows:
        pid = r[0]
        if pid not in result:
            result[pid] = []
        result[pid].append({
            'period_date': r[1], 'vaccount': r[2], 'line_item': r[3],
            'amount': r[4], 'source': r[5],
        })
    return result


def get_deal_cashflows_by_property(engine, deal_id: int,
                                   version: int = 1) -> Dict[int, List[Dict]]:
    """Get all cashflows for a deal grouped by property_id."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT property_id, period_date, revenue, expenses, noi, capex, source
            FROM prospect_cashflows
            WHERE prospect_id = :did AND version = :v AND property_id IS NOT NULL
            ORDER BY property_id, period_date
        """), {'did': deal_id, 'v': version}).fetchall()

    result: Dict[int, List[Dict]] = {}
    for r in rows:
        pid = r[0]
        if pid not in result:
            result[pid] = []
        result[pid].append({
            'period_date': r[1], 'revenue': r[2], 'expenses': r[3],
            'noi': r[4], 'capex': r[5], 'source': r[6],
        })
    return result


def get_property_cashflow_status(engine, deal_id: int,
                                 property_ids: List[int]) -> Dict[int, Dict]:
    """Get cashflow load status for each property: source type and timestamp.

    Checks both prospect_cashflows (Excel uploads) and argus_imports (Argus).
    Returns {property_id: {excel: {loaded_at}, argus: {loaded_at, filename}}}.
    """
    status: Dict[int, Dict] = {}
    if not property_ids:
        return status

    with engine.connect() as conn:
        # Excel cashflows — get most recent created_at per property
        placeholders = ', '.join(f':pid{i}' for i in range(len(property_ids)))
        params: Dict = {'did': deal_id}
        params.update({f'pid{i}': pid for i, pid in enumerate(property_ids)})

        excel_rows = conn.execute(text(f"""
            SELECT property_id, source, MAX(created_at) as loaded_at, COUNT(*) as row_count
            FROM prospect_cashflows
            WHERE prospect_id = :did AND property_id IN ({placeholders})
            GROUP BY property_id, source
        """), params).fetchall()

        for r in excel_rows:
            pid = r[0]
            if pid not in status:
                status[pid] = {}
            status[pid]['excel'] = {
                'source': r[1] or 'excel',
                'loaded_at': str(r[2]) if r[2] else None,
                'rows': r[3],
            }

        # Argus imports — check argus_imports table per property vcode
        for pid in property_ids:
            prop_vcode = f"NP{pid:06d}"
            argus_row = conn.execute(text("""
                SELECT id, import_label, original_filename, is_active, created_at
                FROM argus_imports
                WHERE vcode = :v
                ORDER BY created_at DESC
                LIMIT 1
            """), {'v': prop_vcode}).fetchone()

            if argus_row:
                if pid not in status:
                    status[pid] = {}
                status[pid]['argus'] = {
                    'import_id': argus_row[0],
                    'label': argus_row[1],
                    'filename': argus_row[2],
                    'is_active': bool(argus_row[3]),
                    'loaded_at': str(argus_row[4]) if argus_row[4] else None,
                }

    return status
