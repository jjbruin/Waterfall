"""
argus_service.py
Database-aware service layer for Argus Enterprise imports.

Handles import sessions, cashflow storage, tenant detail,
projection scenario management, and forecast DataFrame generation.
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
from sqlalchemy import text

from argus_parser import (
    parse_monthly_cashflow,
    parse_rent_roll_summary,
    parse_revenue_assumptions,
    cashflow_to_forecast_df,
    map_to_coa,
)

logger = logging.getLogger(__name__)


# ============================================================
# IMPORT FUNCTIONS
# ============================================================

def import_argus_cashflow(
    engine,
    vcode: str,
    file_bytes: bytes,
    filename: str,
    import_label: str,
    import_type: str,
    username: str,
) -> Dict[str, Any]:
    """Import an Argus Monthly Cash Flow Excel export.

    Args:
        engine: SQLAlchemy engine.
        vcode: Deal vcode.
        file_bytes: Raw Excel file bytes.
        filename: Original filename.
        import_label: Label for this projection (e.g. "Partner Projection").
        import_type: 'new_business' or 'asset_management'.
        username: Importing user.

    Returns:
        Import summary dict with import_id, line_item_count, mapped/unmapped counts.
    """
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check for duplicate import
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT id FROM argus_imports WHERE vcode = :v AND file_hash = :h"),
            {"v": vcode, "h": file_hash},
        ).fetchone()
        if existing:
            return {
                "status": "duplicate",
                "import_id": existing[0],
                "message": f"This file has already been imported (import #{existing[0]})",
            }

    # Parse the Excel file
    parsed = parse_monthly_cashflow(file_bytes, filename)
    if parsed["metadata"].get("error"):
        return {"status": "error", "message": parsed["metadata"]["error"]}

    with engine.begin() as conn:
        # Create import record
        result = conn.execute(
            text("""
                INSERT INTO argus_imports (vcode, import_label, import_type,
                    original_filename, file_hash, is_active, imported_by)
                VALUES (:vcode, :label, :itype, :fname, :fhash, TRUE, :user)
                RETURNING id
            """),
            {
                "vcode": vcode, "label": import_label, "itype": import_type,
                "fname": filename, "fhash": file_hash, "user": username,
            },
        )
        import_id = result.fetchone()[0]

        # Insert cashflow rows
        periods = parsed["periods"]
        for li in parsed["line_items"]:
            coa_account = li.get("coa_account")
            category = li.get("category")
            for i, period_str in enumerate(periods):
                if i >= len(li["amounts"]):
                    break
                amount = li["amounts"][i]
                if amount == 0.0:
                    continue

                # Compute normalized amount
                amount_norm = _normalize_amount(coa_account, amount)

                conn.execute(
                    text("""
                        INSERT INTO argus_cashflows
                            (import_id, vcode, period_date, line_item,
                             coa_account, amount, amount_norm, category)
                        VALUES (:iid, :v, :pd, :li, :coa, :amt, :norm, :cat)
                    """),
                    {
                        "iid": import_id, "v": vcode, "pd": period_str,
                        "li": li["name"], "coa": coa_account, "amt": amount,
                        "norm": amount_norm, "cat": category,
                    },
                )

    logger.info(f"Imported Argus cashflow for {vcode}: {len(parsed['line_items'])} line items, "
                f"{parsed['metadata']['mapped_count']} mapped")

    return {
        "status": "success",
        "import_id": import_id,
        "total_line_items": parsed["metadata"]["total_line_items"],
        "mapped_count": parsed["metadata"]["mapped_count"],
        "unmapped_items": parsed["metadata"]["unmapped_items"],
        "total_periods": parsed["metadata"]["total_periods"],
        "parsed": parsed,
    }


def import_argus_rent_roll(
    engine,
    vcode: str,
    file_bytes: bytes,
    filename: str,
    import_id: int,
    username: str,
) -> Dict[str, Any]:
    """Import an Argus Rent Roll Summary Excel export.

    Args:
        engine: SQLAlchemy engine.
        vcode: Deal vcode.
        file_bytes: Raw Excel file bytes.
        filename: Original filename.
        import_id: FK to argus_imports.
        username: Importing user.

    Returns:
        Summary with tenant count.
    """
    parsed = parse_rent_roll_summary(file_bytes, filename)
    tenants = parsed.get("tenants", [])

    with engine.begin() as conn:
        for tenant in tenants:
            result = conn.execute(
                text("""
                    INSERT INTO argus_tenants
                        (import_id, vcode, tenant_name, suite, square_feet,
                         lease_type, lease_start, lease_end, term_months,
                         base_rent_annual, base_rent_psf, recovery_type,
                         ret_recovery_psf, ins_recovery_psf, cam_recovery_psf,
                         ti_psf, lc_psf, renewal_probability, cpi_pct,
                         free_rent_months, pct_rent_breakpoint, pct_rent_rate,
                         security_deposit, is_vacant)
                    VALUES (:iid, :v, :tn, :su, :sf,
                            :lt, :ls, :le, :tm,
                            :bra, :brp, :rt,
                            :retr, :insr, :camr,
                            :ti, :lc, :rp, :cpi,
                            :frm, :prb, :prr,
                            :sd, :iv)
                    RETURNING id
                """),
                {
                    "iid": import_id, "v": vcode,
                    "tn": tenant["tenant_name"], "su": tenant["suite"],
                    "sf": tenant["square_feet"], "lt": tenant["lease_type"],
                    "ls": tenant["lease_start"], "le": tenant["lease_end"],
                    "tm": tenant.get("term_months", 0),
                    "bra": tenant["base_rent_annual"], "brp": tenant["base_rent_psf"],
                    "rt": tenant["recovery_type"],
                    "retr": tenant["ret_recovery_psf"], "insr": tenant["ins_recovery_psf"],
                    "camr": tenant["cam_recovery_psf"],
                    "ti": tenant["ti_psf"], "lc": tenant["lc_psf"],
                    "rp": tenant["renewal_probability"], "cpi": tenant["cpi_pct"],
                    "frm": tenant["free_rent_months"],
                    "prb": tenant["pct_rent_breakpoint"], "prr": tenant["pct_rent_rate"],
                    "sd": tenant["security_deposit"],
                    "iv": tenant["is_vacant"],
                },
            )
            tenant_id = result.fetchone()[0]

            # Insert rent steps for this tenant
            for step in parsed.get("rent_steps", []):
                if step.get("tenant_index") == tenants.index(tenant):
                    conn.execute(
                        text("""
                            INSERT INTO argus_rent_steps
                                (argus_tenant_id, effective_date, annual_rent,
                                 rent_psf, step_type, step_pct)
                            VALUES (:tid, :ed, :ar, :rp, :st, :sp)
                        """),
                        {
                            "tid": tenant_id, "ed": step["effective_date"],
                            "ar": step["annual_rent"], "rp": step["rent_psf"],
                            "st": step["step_type"], "sp": step["step_pct"],
                        },
                    )

    logger.info(f"Imported Argus rent roll for {vcode}: {len(tenants)} tenants")
    return {
        "status": "success",
        "tenant_count": len(tenants),
        "summary": parsed.get("summary", {}),
    }


def import_argus_revenue_assumptions(
    engine,
    vcode: str,
    file_bytes: bytes,
    filename: str,
    import_id: int,
    username: str,
) -> Dict[str, Any]:
    """Import Argus Revenue Assumptions Excel export.

    Args:
        engine: SQLAlchemy engine.
        vcode: Deal vcode.
        file_bytes: Raw Excel file bytes.
        filename: Original filename.
        import_id: FK to argus_imports.
        username: Importing user.

    Returns:
        Summary with profile count.
    """
    parsed = parse_revenue_assumptions(file_bytes, filename)
    profiles = parsed.get("profiles", [])

    with engine.begin() as conn:
        for profile in profiles:
            conn.execute(
                text("""
                    INSERT INTO argus_market_profiles
                        (import_id, vcode, profile_name, base_rent_psf,
                         term_months, renewal_probability, vacancy_months,
                         ti_new_psf, ti_renewal_psf, lc_new_pct, lc_renewal_pct,
                         fixed_step_pct, cpi_pct, recovery_type)
                    VALUES (:iid, :v, :pn, :brp,
                            :tm, :rp, :vm,
                            :tin, :tir, :lcn, :lcr,
                            :fsp, :cpi, :rt)
                """),
                {
                    "iid": import_id, "v": vcode, "pn": profile["profile_name"],
                    "brp": profile["base_rent_psf"], "tm": profile["term_months"],
                    "rp": profile["renewal_probability"], "vm": profile["vacancy_months"],
                    "tin": profile["ti_new_psf"], "tir": profile["ti_renewal_psf"],
                    "lcn": profile["lc_new_pct"], "lcr": profile["lc_renewal_pct"],
                    "fsp": profile["fixed_step_pct"], "cpi": profile["cpi_pct"],
                    "rt": profile["recovery_type"],
                },
            )

    logger.info(f"Imported Argus revenue assumptions for {vcode}: {len(profiles)} profiles")
    return {"status": "success", "profile_count": len(profiles)}


# ============================================================
# PROJECTION SCENARIO MANAGEMENT
# ============================================================

def get_projection_scenarios(engine, vcode: str) -> List[Dict[str, Any]]:
    """Get all Argus import sessions for a deal."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, vcode, import_label, import_type, original_filename,
                       is_active, imported_by, created_at, updated_at
                FROM argus_imports
                WHERE vcode = :v
                ORDER BY created_at DESC
            """),
            {"v": vcode},
        ).fetchall()

    return [
        {
            "id": r[0], "vcode": r[1], "import_label": r[2], "import_type": r[3],
            "original_filename": r[4], "is_active": bool(r[5]),
            "imported_by": r[6],
            "created_at": str(r[7]) if r[7] else None,
            "updated_at": str(r[8]) if r[8] else None,
        }
        for r in rows
    ]


def set_active_projection(engine, vcode: str, import_id: int):
    """Activate one projection and deactivate others for a deal."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE argus_imports SET is_active = FALSE WHERE vcode = :v"),
            {"v": vcode},
        )
        conn.execute(
            text("UPDATE argus_imports SET is_active = TRUE, updated_at = :now WHERE id = :id AND vcode = :v"),
            {"id": import_id, "v": vcode, "now": datetime.utcnow()},
        )


def delete_projection(engine, vcode: str, import_id: int):
    """Delete a projection and all related data."""
    with engine.begin() as conn:
        # Delete rent steps via tenant IDs
        conn.execute(
            text("""
                DELETE FROM argus_rent_steps
                WHERE argus_tenant_id IN (
                    SELECT id FROM argus_tenants WHERE import_id = :iid
                )
            """),
            {"iid": import_id},
        )
        conn.execute(
            text("DELETE FROM argus_tenants WHERE import_id = :iid"),
            {"iid": import_id},
        )
        conn.execute(
            text("DELETE FROM argus_market_profiles WHERE import_id = :iid"),
            {"iid": import_id},
        )
        conn.execute(
            text("DELETE FROM argus_cashflows WHERE import_id = :iid"),
            {"iid": import_id},
        )
        conn.execute(
            text("DELETE FROM argus_imports WHERE id = :iid AND vcode = :v"),
            {"iid": import_id, "v": vcode},
        )

    logger.info(f"Deleted Argus projection {import_id} for {vcode}")


# ============================================================
# FORECAST GENERATION
# ============================================================

def get_active_forecast_df(engine, vcode: str, pro_yr_base: int) -> Optional[pd.DataFrame]:
    """Get forecast DataFrame from the active Argus projection.

    Returns None if no active projection exists.
    Returns a DataFrame in the same format as load_forecast() output.
    """
    with engine.connect() as conn:
        # Find active import
        active = conn.execute(
            text("""
                SELECT id FROM argus_imports
                WHERE vcode = :v AND is_active = TRUE
                ORDER BY created_at DESC LIMIT 1
            """),
            {"v": vcode},
        ).fetchone()

        if not active:
            return None

        import_id = active[0]

        # Load cashflows
        rows = conn.execute(
            text("""
                SELECT period_date, coa_account, amount, amount_norm
                FROM argus_cashflows
                WHERE import_id = :iid AND coa_account IS NOT NULL
                ORDER BY period_date
            """),
            {"iid": import_id},
        ).fetchall()

    if not rows:
        return None

    fc_rows = []
    for r in rows:
        period_date = pd.to_datetime(r[0]).date()
        coa_account = int(r[1])
        amount = float(r[2]) if r[2] else 0.0
        amount_norm = float(r[3]) if r[3] else 0.0
        pro_yr = period_date.year - pro_yr_base

        fc_rows.append({
            "vcode": vcode,
            "event_date": period_date,
            "vAccount": coa_account,
            "mAmount": amount,
            "Pro_Yr": pro_yr,
            "vAccountType": "",
            "mAmount_norm": amount_norm,
        })

    return pd.DataFrame(fc_rows)


def get_forecast_df_by_id(engine, vcode: str, import_id: int, pro_yr_base: int) -> Optional[pd.DataFrame]:
    """Get forecast DataFrame from a specific Argus projection (not necessarily active)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT period_date, coa_account, amount, amount_norm
                FROM argus_cashflows
                WHERE import_id = :iid AND vcode = :v AND coa_account IS NOT NULL
                ORDER BY period_date
            """),
            {"iid": import_id, "v": vcode},
        ).fetchall()

    if not rows:
        return None

    fc_rows = []
    for r in rows:
        period_date = pd.to_datetime(r[0]).date()
        coa_account = int(r[1])
        amount = float(r[2]) if r[2] else 0.0
        amount_norm = float(r[3]) if r[3] else 0.0
        pro_yr = period_date.year - pro_yr_base

        fc_rows.append({
            "vcode": vcode,
            "event_date": period_date,
            "vAccount": coa_account,
            "mAmount": amount,
            "Pro_Yr": pro_yr,
            "vAccountType": "",
            "mAmount_norm": amount_norm,
        })

    return pd.DataFrame(fc_rows)


# ============================================================
# TENANT DETAIL
# ============================================================

def get_argus_tenants(engine, vcode: str, import_id: int) -> List[Dict[str, Any]]:
    """Get tenant detail with rent steps for a specific import."""
    with engine.connect() as conn:
        tenants = conn.execute(
            text("""
                SELECT id, tenant_name, suite, square_feet, lease_type,
                       lease_start, lease_end, term_months,
                       base_rent_annual, base_rent_psf, recovery_type,
                       ret_recovery_psf, ins_recovery_psf, cam_recovery_psf,
                       ti_psf, lc_psf, renewal_probability, cpi_pct,
                       free_rent_months, pct_rent_breakpoint, pct_rent_rate,
                       security_deposit, is_vacant, lease_tenant_id
                FROM argus_tenants
                WHERE import_id = :iid AND vcode = :v
                ORDER BY suite, tenant_name
            """),
            {"iid": import_id, "v": vcode},
        ).fetchall()

        result = []
        for t in tenants:
            tenant_id = t[0]
            steps = conn.execute(
                text("""
                    SELECT effective_date, annual_rent, rent_psf, step_type, step_pct
                    FROM argus_rent_steps
                    WHERE argus_tenant_id = :tid
                    ORDER BY effective_date
                """),
                {"tid": tenant_id},
            ).fetchall()

            result.append({
                "id": tenant_id,
                "tenant_name": t[1], "suite": t[2], "square_feet": t[3],
                "lease_type": t[4], "lease_start": t[5], "lease_end": t[6],
                "term_months": t[7], "base_rent_annual": t[8], "base_rent_psf": t[9],
                "recovery_type": t[10],
                "ret_recovery_psf": t[11], "ins_recovery_psf": t[12],
                "cam_recovery_psf": t[13],
                "ti_psf": t[14], "lc_psf": t[15],
                "renewal_probability": t[16], "cpi_pct": t[17],
                "free_rent_months": t[18],
                "pct_rent_breakpoint": t[19], "pct_rent_rate": t[20],
                "security_deposit": t[21], "is_vacant": bool(t[22]),
                "lease_tenant_id": t[23],
                "rent_steps": [
                    {
                        "effective_date": s[0], "annual_rent": s[1],
                        "rent_psf": s[2], "step_type": s[3], "step_pct": s[4],
                    }
                    for s in steps
                ],
            })

    return result


# ============================================================
# COA MAPPING MANAGEMENT
# ============================================================

def get_coa_mapping(engine, import_id: int) -> Dict[str, Any]:
    """Get COA mapping for an import — shows mapped and unmapped items."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT line_item, coa_account, category
                FROM argus_cashflows
                WHERE import_id = :iid
                ORDER BY line_item
            """),
            {"iid": import_id},
        ).fetchall()

    mapped = []
    unmapped = []
    for r in rows:
        item = {"line_item": r[0], "coa_account": r[1], "category": r[2]}
        if r[1]:
            mapped.append(item)
        else:
            unmapped.append(item)

    return {"mapped": mapped, "unmapped": unmapped}


def update_coa_mapping(engine, import_id: int, mappings: List[Dict[str, Any]]):
    """Override COA mappings for specific line items.

    Args:
        engine: SQLAlchemy engine.
        import_id: Import session ID.
        mappings: List of {'line_item': str, 'coa_account': int, 'category': str}.
    """
    with engine.begin() as conn:
        for m in mappings:
            coa = m.get("coa_account")
            cat = m.get("category")
            line_item = m.get("line_item")

            # Recompute normalized amount for the new mapping
            conn.execute(
                text("""
                    UPDATE argus_cashflows
                    SET coa_account = :coa, category = :cat,
                        amount_norm = CASE
                            WHEN :coa IS NULL THEN amount
                            ELSE amount
                        END
                    WHERE import_id = :iid AND line_item = :li
                """),
                {"coa": coa, "cat": cat, "iid": import_id, "li": line_item},
            )

        # Now recompute amount_norm based on new COA mapping
        rows = conn.execute(
            text("""
                SELECT id, coa_account, amount
                FROM argus_cashflows
                WHERE import_id = :iid
            """),
            {"iid": import_id},
        ).fetchall()

        for r in rows:
            norm = _normalize_amount(r[1], float(r[2]) if r[2] else 0.0)
            conn.execute(
                text("UPDATE argus_cashflows SET amount_norm = :norm WHERE id = :id"),
                {"norm": norm, "id": r[0]},
            )


# ============================================================
# MIGRATION (NB → AM)
# ============================================================

def migrate_projection_to_forecast(
    engine,
    old_vcode: str,
    new_vcode: str,
    import_id: int,
    pro_yr_base: int,
) -> Dict[str, Any]:
    """Migrate an Argus projection to the forecasts table for AM onboarding.

    Re-keys from N-series vcode to P-series vcode and inserts into
    the forecasts table (same format as forecast_feed.csv).
    """
    fc_df = get_forecast_df_by_id(engine, old_vcode, import_id, pro_yr_base)
    if fc_df is None or fc_df.empty:
        return {"status": "error", "message": "No cashflow data found for this projection"}

    # Re-key to new vcode
    fc_df["vcode"] = new_vcode

    # Write to forecasts table
    fc_df.to_sql("forecasts", engine, if_exists="append", index=False)

    # Copy argus_imports, argus_tenants, argus_rent_steps with new vcode
    with engine.begin() as conn:
        # Copy import record
        conn.execute(
            text("""
                INSERT INTO argus_imports (vcode, import_label, import_type,
                    original_filename, file_hash, is_active, imported_by)
                SELECT :new_v, import_label, 'asset_management',
                    original_filename, file_hash, is_active, imported_by
                FROM argus_imports WHERE id = :iid
            """),
            {"new_v": new_vcode, "iid": import_id},
        )
        new_import = conn.execute(
            text("SELECT id FROM argus_imports WHERE vcode = :v ORDER BY id DESC LIMIT 1"),
            {"v": new_vcode},
        ).fetchone()
        new_import_id = new_import[0] if new_import else None

        if new_import_id:
            # Copy tenants
            conn.execute(
                text("""
                    INSERT INTO argus_tenants
                        (import_id, vcode, tenant_name, suite, square_feet,
                         lease_type, lease_start, lease_end, term_months,
                         base_rent_annual, base_rent_psf, recovery_type,
                         ret_recovery_psf, ins_recovery_psf, cam_recovery_psf,
                         ti_psf, lc_psf, renewal_probability, cpi_pct,
                         free_rent_months, pct_rent_breakpoint, pct_rent_rate,
                         security_deposit, is_vacant, lease_tenant_id)
                    SELECT :niid, :new_v, tenant_name, suite, square_feet,
                         lease_type, lease_start, lease_end, term_months,
                         base_rent_annual, base_rent_psf, recovery_type,
                         ret_recovery_psf, ins_recovery_psf, cam_recovery_psf,
                         ti_psf, lc_psf, renewal_probability, cpi_pct,
                         free_rent_months, pct_rent_breakpoint, pct_rent_rate,
                         security_deposit, is_vacant, lease_tenant_id
                    FROM argus_tenants WHERE import_id = :iid
                """),
                {"niid": new_import_id, "new_v": new_vcode, "iid": import_id},
            )

    logger.info(f"Migrated Argus projection {import_id} from {old_vcode} to {new_vcode}")
    return {
        "status": "success",
        "new_vcode": new_vcode,
        "new_import_id": new_import_id,
        "forecast_rows": len(fc_df),
    }


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _normalize_amount(coa_account: Optional[int], amount: float) -> float:
    """Normalize sign convention for a COA account + amount.

    Same logic as prospect_analysis.py:_fc_row() and argus_parser._fc_row().
    """
    from config import (
        GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS,
        EXPENSE_ACCTS, ALL_EXCLUDED, TAX_ABATEMENT_ACCTS,
    )

    if coa_account is None:
        return amount

    acct = int(coa_account)
    if acct in GROSS_REVENUE_ACCTS:
        return abs(amount)
    elif acct in CONTRA_REVENUE_ACCTS:
        return -abs(amount)
    elif acct in EXPENSE_ACCTS:
        return -abs(amount)
    elif acct in ALL_EXCLUDED:
        return -abs(amount)
    elif acct in TAX_ABATEMENT_ACCTS:
        return abs(amount)
    return amount
