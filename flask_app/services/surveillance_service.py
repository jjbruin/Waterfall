"""Surveillance Service — portfolio monitoring with covenant tracking.

Reads from Waterfall's existing data sources (inv, occ, isbs_raw, loans)
and a thin editable table (surveillance_properties) for manual fields.
"""

import logging
from datetime import date, datetime

import pandas as pd
from sqlalchemy import text

from flask_app.db import get_engine
from flask_app.services import data_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

_SURVEILLANCE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS surveillance_properties (
        vcode               TEXT PRIMARY KEY,
        dscr_val            DOUBLE PRECISION,
        dscr_min            DOUBLE PRECISION,
        dy_val              DOUBLE PRECISION,
        dy_min              DOUBLE PRECISION,
        ltv_val             DOUBLE PRECISION,
        ltv_min             DOUBLE PRECISION,
        working_capital     DOUBLE PRECISION,
        tax_due             TEXT,
        ins_renewal         TEXT,
        tenant_exp          TEXT,
        comments            TEXT,
        updated_at          TIMESTAMP,
        updated_by          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS insurance (
        id                  {pk_type},
        vcode               TEXT NOT NULL,
        ins_type            TEXT NOT NULL,
        carrier             TEXT,
        policy_number       TEXT,
        coverage_amount     DOUBLE PRECISION,
        expiration_date     TEXT,
        notes               TEXT,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at          TIMESTAMP,
        UNIQUE(vcode, ins_type)
    )
    """,
]


def ensure_tables(engine=None):
    """Create surveillance tables if they don't exist."""
    if engine is None:
        engine = get_engine()

    is_pg = "postgresql" in str(engine.url)
    pk_type = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"

    with engine.begin() as conn:
        for ddl in _SURVEILLANCE_DDL:
            conn.execute(text(ddl.replace("{pk_type}", pk_type)))


# ---------------------------------------------------------------------------
# Surveillance table — main query
# ---------------------------------------------------------------------------

def get_surveillance_table() -> list[dict]:
    """Build the surveillance table by joining Waterfall data sources.

    Returns one row per active deal with live occupancy, NOI, debt,
    and editable surveillance fields.
    """
    data = data_service.get_data()
    inv = data.get("inv", pd.DataFrame())
    occ = data.get("occupancy_raw", pd.DataFrame())
    isbs_raw = data.get("isbs_raw", pd.DataFrame())
    mri_loans_raw = data.get("mri_loans_raw", pd.DataFrame())

    if inv.empty:
        return []

    # --- Filter to active deals (exclude sold) ---
    inv_col = {c.lower(): c for c in inv.columns}
    sale_col = inv_col.get("sale_status", inv_col.get("salestatus"))
    lifecycle_col = inv_col.get("lifecycle")
    mask = pd.Series(True, index=inv.index)
    if sale_col and sale_col in inv.columns:
        mask &= inv[sale_col].fillna("").str.upper() != "SOLD"
    if lifecycle_col and lifecycle_col in inv.columns:
        mask &= inv[lifecycle_col].fillna("").str.lower() != "sold"
    active = inv[mask].copy()

    vcode_col = _find_col(active, ["vcode", "Vcode", "PropCode"])
    name_col = _find_col(active, ["Investment_Name", "Property_Name", "PropertyName", "Deal_Name"])
    type_col = _find_col(active, ["Asset_Type", "AssetType", "asset_type"])
    city_col = _find_col(active, ["City", "city"])
    units_col = _find_col(active, ["Units", "units", "iResidentialUnits"])
    partner_col = _find_col(active, ["Partner", "partner", "PSC_Asset_Manager"])
    lifecycle_disp = _find_col(active, ["Lifecycle", "lifecycle"])
    portfolio_col = _find_col(active, ["Portfolio_Name", "portfolio_name"])

    # --- Latest occupancy per deal ---
    occ_latest = _latest_occupancy(occ)

    # --- Latest NOI from ISBS actuals ---
    noi_latest = _latest_noi(isbs_raw)

    # --- Loan summary per deal ---
    loan_summary = _loan_summary(mri_loans_raw)

    # --- Load editable surveillance fields ---
    surv_fields = _load_surveillance_properties()

    # --- Load insurance ---
    insurance = _load_insurance_summary()

    # --- Build result rows ---
    rows = []
    for _, deal in active.iterrows():
        vc = str(deal.get(vcode_col, "")).strip()
        if not vc:
            continue
        vc_lower = vc.lower()

        occ_row = occ_latest.get(vc_lower, {})
        noi_row = noi_latest.get(vc_lower, {})
        loan_row = loan_summary.get(vc_lower, {})
        surv = surv_fields.get(vc, {})
        ins = insurance.get(vc, {})

        row = {
            "vcode": vc,
            "name": deal.get(name_col, ""),
            "asset_type": deal.get(type_col, ""),
            "city": deal.get(city_col, ""),
            "units": _safe_int(deal.get(units_col)),
            "partner": deal.get(partner_col, ""),
            "lifecycle": deal.get(lifecycle_disp, ""),
            "portfolio_name": deal.get(portfolio_col, ""),
            # Live data
            "occ_pct": occ_row.get("occ_pct"),
            "occ_period": occ_row.get("period"),
            "noi_monthly": noi_row.get("noi"),
            "revenue_monthly": noi_row.get("revenue"),
            "fin_period": noi_row.get("period"),
            # Loan data
            "loan_balance": loan_row.get("loan_balance"),
            "loan_rate": loan_row.get("loan_rate"),
            "maturity_date": loan_row.get("maturity_date"),
            "loan_type": loan_row.get("loan_type"),
            # Editable surveillance fields
            "dscr_val": surv.get("dscr_val"),
            "dscr_min": surv.get("dscr_min"),
            "dy_val": surv.get("dy_val"),
            "dy_min": surv.get("dy_min"),
            "ltv_val": surv.get("ltv_val"),
            "ltv_min": surv.get("ltv_min"),
            "working_capital": surv.get("working_capital"),
            "tax_due": surv.get("tax_due"),
            "ins_renewal": ins.get("nearest_expiration"),
            "tenant_exp": surv.get("tenant_exp"),
            "comments": surv.get("comments"),
            "updated_at": surv.get("updated_at"),
            # Insurance
            "has_property_ins": ins.get("has_property", False),
            "has_gl_ins": ins.get("has_gl", False),
        }

        # Flag logic: covenant breach or comments present
        row["flagged"] = bool(
            row["comments"]
            or (row["dscr_val"] is not None and row["dscr_min"] is not None
                and row["dscr_val"] < row["dscr_min"])
            or (row["ltv_val"] is not None and row["ltv_min"] is not None
                and row["ltv_val"] > row["ltv_min"])
        )

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Dashboard KPIs
# ---------------------------------------------------------------------------

def get_dashboard() -> dict:
    """Portfolio surveillance KPIs and chart data."""
    rows = get_surveillance_table()
    if not rows:
        return {"total": 0}

    total = len(rows)
    total_debt = sum(r["loan_balance"] or 0 for r in rows)
    occ_vals = [r["occ_pct"] for r in rows if r["occ_pct"] is not None]
    avg_occ = sum(occ_vals) / len(occ_vals) if occ_vals else None
    total_noi = sum(r["noi_monthly"] or 0 for r in rows)
    flagged = sum(1 for r in rows if r["flagged"])

    # Maturing within 12 months
    today = date.today()
    mat_12 = 0
    for r in rows:
        md = r.get("maturity_date")
        if md:
            try:
                mat = pd.to_datetime(md).date()
                if mat <= today.replace(year=today.year + 1):
                    mat_12 += 1
            except Exception:
                pass

    # By asset type
    by_type = {}
    for r in rows:
        t = r.get("asset_type") or "Unknown"
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "total": total,
        "total_debt": total_debt,
        "avg_occ": round(avg_occ, 1) if avg_occ else None,
        "total_noi_monthly": total_noi,
        "flagged": flagged,
        "maturing_12mo": mat_12,
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# CRUD — surveillance properties (editable fields)
# ---------------------------------------------------------------------------

def update_surveillance_property(vcode: str, fields: dict, username: str = None) -> dict:
    """Upsert editable surveillance fields for a deal."""
    allowed = {
        "dscr_val", "dscr_min", "dy_val", "dy_min",
        "ltv_val", "ltv_min", "working_capital",
        "tax_due", "ins_renewal", "tenant_exp", "comments",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return {"status": "no_changes"}

    updates["updated_at"] = datetime.utcnow().isoformat()
    if username:
        updates["updated_by"] = username

    engine = get_engine()
    with engine.begin() as conn:
        # Check if row exists
        existing = conn.execute(
            text("SELECT vcode FROM surveillance_properties WHERE vcode = :vc"),
            {"vc": vcode}
        ).fetchone()

        if existing:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            conn.execute(
                text(f"UPDATE surveillance_properties SET {set_clause} WHERE vcode = :vcode"),
                {**updates, "vcode": vcode}
            )
        else:
            updates["vcode"] = vcode
            cols = ", ".join(updates.keys())
            vals = ", ".join(f":{k}" for k in updates.keys())
            conn.execute(
                text(f"INSERT INTO surveillance_properties ({cols}) VALUES ({vals})"),
                updates
            )

    return {"status": "ok", "vcode": vcode}


# ---------------------------------------------------------------------------
# CRUD — insurance
# ---------------------------------------------------------------------------

def get_insurance_list() -> list[dict]:
    """All insurance records with days-to-expiration."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM insurance ORDER BY vcode, ins_type"
        )).mappings().all()

    today = date.today()
    result = []
    for r in rows:
        d = dict(r)
        exp = d.get("expiration_date")
        if exp:
            try:
                exp_date = pd.to_datetime(exp).date()
                d["days_to_expiration"] = (exp_date - today).days
            except Exception:
                d["days_to_expiration"] = None
        else:
            d["days_to_expiration"] = None
        result.append(d)
    return result


def upsert_insurance(vcode: str, ins_type: str, fields: dict) -> dict:
    """Create or update an insurance record."""
    allowed = {"carrier", "policy_number", "coverage_amount", "expiration_date", "notes"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    updates["updated_at"] = datetime.utcnow().isoformat()

    engine = get_engine()
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM insurance WHERE vcode = :vc AND ins_type = :it"),
            {"vc": vcode, "it": ins_type}
        ).fetchone()

        if existing:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            conn.execute(
                text(f"UPDATE insurance SET {set_clause} WHERE vcode = :vc AND ins_type = :it"),
                {**updates, "vc": vcode, "it": ins_type}
            )
            return {"status": "updated", "id": existing[0]}
        else:
            updates["vcode"] = vcode
            updates["ins_type"] = ins_type
            cols = ", ".join(updates.keys())
            vals = ", ".join(f":{k}" for k in updates.keys())
            conn.execute(
                text(f"INSERT INTO insurance ({cols}) VALUES ({vals})"),
                updates
            )
            return {"status": "created"}


def delete_insurance(ins_id: int) -> dict:
    """Delete an insurance record by ID."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM insurance WHERE id = :id"), {"id": ins_id})
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Helpers — data extraction from Waterfall sources
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching column name (case-insensitive)."""
    col_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    return candidates[0]  # fallback


def _safe_int(val):
    """Convert to int, return None on failure."""
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _latest_occupancy(occ: pd.DataFrame) -> dict:
    """Get latest occupancy per deal from occ DataFrame."""
    if occ.empty:
        return {}

    col_map = {c.lower(): c for c in occ.columns}
    vcode_col = col_map.get("vcode", col_map.get("propcode", "vcode"))
    period_col = col_map.get("dtreported", col_map.get("period", "dtReported"))
    occ_col = col_map.get("occ%", col_map.get("occ_pct", col_map.get("focc", "Occ%")))

    df = occ.copy()
    df["_vcode"] = df[vcode_col].astype(str).str.strip().str.lower()
    df["_period"] = pd.to_datetime(df[period_col], errors="coerce")
    df = df.dropna(subset=["_period"])

    # Latest period per deal
    idx = df.groupby("_vcode")["_period"].idxmax()
    latest = df.loc[idx]

    result = {}
    for _, row in latest.iterrows():
        try:
            occ_val = float(row[occ_col])
        except (TypeError, ValueError):
            occ_val = None
        result[row["_vcode"]] = {
            "occ_pct": round(occ_val, 1) if occ_val is not None else None,
            "period": row["_period"].strftime("%Y-%m"),
        }
    return result


def _latest_noi(isbs_raw: pd.DataFrame) -> dict:
    """Get latest monthly NOI and Revenue from ISBS Interim IS actuals.

    ISBS columns: vcode, dtEntry (or dtEntry_parsed), vSource, vAccount,
    vAccountType, mAmount, iNOI.
    """
    if isbs_raw.empty:
        return {}

    col_map = {c.lower(): c for c in isbs_raw.columns}
    source_col = col_map.get("vsource", "vSource")
    vcode_col = col_map.get("vcode", "vcode")
    # dtEntry_parsed is pre-parsed datetime; dtEntry is raw string
    period_col = col_map.get("dtentry_parsed", col_map.get("dtentry", "dtEntry"))
    amount_col = col_map.get("mamount", "mAmount")
    acct_col = col_map.get("vaccount", "vAccount")
    inoi_col = col_map.get("inoi", "iNOI")

    # Filter to Interim IS (actuals)
    df = isbs_raw.copy()
    if source_col in df.columns:
        df = df[df[source_col].astype(str).str.strip() == "Interim IS"]
    if df.empty:
        return {}

    df["_vcode"] = df[vcode_col].astype(str).str.strip().str.lower()
    df["_period"] = pd.to_datetime(df[period_col], format="mixed", errors="coerce")
    df["_amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
    df = df.dropna(subset=["_period"])

    # Get latest period per deal
    latest_period = df.groupby("_vcode")["_period"].max()

    result = {}
    for vc, max_period in latest_period.items():
        deal_df = df[(df["_vcode"] == vc) & (df["_period"] == max_period)]
        if deal_df.empty:
            continue

        # Revenue = 4xxx accounts (negative in MRI, negate to positive)
        # Expenses = 5xxx accounts
        revenue = 0.0
        expenses = 0.0
        for _, row in deal_df.iterrows():
            acct = str(row.get(acct_col, "")).strip()
            amt = row["_amount"]
            if acct.startswith("4"):
                revenue += abs(amt)
            elif acct.startswith("5"):
                expenses += abs(amt)

        result[vc] = {
            "noi": round(revenue - expenses, 2),
            "revenue": round(revenue, 2),
            "period": max_period.strftime("%Y-%m"),
        }
    return result


def _loan_summary(mri_loans: pd.DataFrame) -> dict:
    """Summarize loan data per deal from MRI_Loans."""
    if mri_loans.empty:
        return {}

    col_map = {c.lower(): c for c in mri_loans.columns}
    vcode_col = col_map.get("vcode", "vCode")
    balance_col = col_map.get("morigloanamt", "mOrigLoanAmt")
    rate_col = col_map.get("nrate", "nRate")
    maturity_col = col_map.get("dtmaturity", "dtMaturity")
    type_col = col_map.get("vinttype", "vIntType")

    df = mri_loans.copy()
    df["_vcode"] = df[vcode_col].astype(str).str.strip().str.lower()
    df["_balance"] = pd.to_numeric(df.get(balance_col, pd.Series(dtype=float)), errors="coerce").fillna(0)

    result = {}
    for vc, group in df.groupby("_vcode"):
        total_balance = group["_balance"].sum()
        # Use the largest loan for rate/maturity/type
        primary = group.loc[group["_balance"].idxmax()]
        result[vc] = {
            "loan_balance": round(total_balance, 2),
            "loan_rate": _safe_float(primary.get(rate_col)),
            "maturity_date": _safe_date_str(primary.get(maturity_col)),
            "loan_type": str(primary.get(type_col, "")).strip() or None,
        }
    return result


def _safe_float(val):
    try:
        v = float(val)
        return round(v, 4) if v == v else None  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_date_str(val):
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d")
    except Exception:
        return None


def _load_surveillance_properties() -> dict:
    """Load all editable surveillance fields keyed by vcode."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM surveillance_properties")).mappings().all()
        return {r["vcode"]: dict(r) for r in rows}
    except Exception:
        return {}


def _load_insurance_summary() -> dict:
    """Load insurance summary per vcode (has_property, has_gl, nearest_expiration)."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM insurance")).mappings().all()
    except Exception:
        return {}

    result = {}
    for r in rows:
        vc = r["vcode"]
        if vc not in result:
            result[vc] = {"has_property": False, "has_gl": False, "nearest_expiration": None}
        entry = result[vc]
        if r["ins_type"] == "Property":
            entry["has_property"] = True
        elif r["ins_type"] == "General Liability":
            entry["has_gl"] = True
        exp = r.get("expiration_date")
        if exp:
            if entry["nearest_expiration"] is None or exp < entry["nearest_expiration"]:
                entry["nearest_expiration"] = exp
    return result
