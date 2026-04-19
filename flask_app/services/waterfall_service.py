"""Waterfall Setup service — validation, templates, preview, CRUD.

Extracts pure logic from waterfall_setup_ui.py (no Streamlit dependency).
"""

import pandas as pd
import numpy as np
import copy
from datetime import date
from typing import Optional

from database import save_waterfall_steps, delete_waterfall_steps


# ── Constants ────────────────────────────────────────────────────────────

WF_COLUMNS = [
    "iOrder", "PropCode", "vState", "FXRate", "nPercent", "mAmount",
    "vtranstype", "vAmtType", "vNotes",
]

DB_COLUMNS = [
    "vcode", "vmisc", "iOrder", "vAmtType", "vNotes", "PropCode",
    "nmisc", "dteffective", "vtranstype", "mAmount", "nPercent",
    "FXRate", "vState",
]

VSTATE_OPTIONS = [
    "Pref", "Initial", "Add", "Tag", "Share", "IRR",
    "Amt", "Def&Int", "Def_Int", "Default", "AMFee", "Promote",
]

_PARI_PASSU_DEFAULTS = {
    "entity_expenses_quarterly": 15625.0,
    "asset_mgmt_fee_pct": 0.5,
    "asset_mgmt_propcode": "PSCMAN",
}


# ── Validation ───────────────────────────────────────────────────────────

def validate_steps(df: pd.DataFrame, wf_type: str,
                   valid_entities: set = None) -> tuple[list[str], list[str]]:
    """Validate waterfall steps. Returns (errors, warnings).

    Extracted from waterfall_setup_ui.py::_validate_steps().
    """
    errors = []
    warnings = []

    if df is None or df.empty:
        return errors, warnings

    # 1. FXRate sum per iOrder
    if "FXRate" in df.columns and "iOrder" in df.columns:
        for order, grp in df.groupby("iOrder"):
            fx_sum = grp["FXRate"].sum()
            if abs(fx_sum - 1.0) > 0.02 and fx_sum > 0:
                warnings.append(
                    f"iOrder {int(order)}: FXRate sum = {fx_sum:.4f} (expected ~1.0)"
                )

    # 2. Operating Capital must be Add, not Tag
    if "vtranstype" in df.columns and "vState" in df.columns:
        oc_rows = df[
            df["vtranstype"].astype(str).str.contains("Operating Capital", case=False, na=False)
        ]
        tag_oc = oc_rows[oc_rows["vState"] == "Tag"]
        for _, r in tag_oc.iterrows():
            errors.append(
                f"iOrder {int(r['iOrder'])} {r['PropCode']}: Operating Capital "
                "MUST use Add, not Tag"
            )

    # 3. Pref steps should have FX=1.0
    if "vState" in df.columns and "FXRate" in df.columns:
        pref_rows = df[df["vState"] == "Pref"]
        for _, r in pref_rows.iterrows():
            if r["FXRate"] != 1.0:
                warnings.append(
                    f"iOrder {int(r['iOrder'])} {r['PropCode']}: Pref step FXRate = "
                    f"{r['FXRate']:.4f} (typically 1.0)"
                )

    # 4. Each iOrder with Tags must have at least one lead
    if "vState" in df.columns and "iOrder" in df.columns:
        for order, grp in df.groupby("iOrder"):
            has_tag = (grp["vState"] == "Tag").any()
            has_lead = (grp["vState"] != "Tag").any()
            if has_tag and not has_lead:
                errors.append(
                    f"iOrder {int(order)}: has Tag step(s) but no lead step"
                )

    # 5. AMFee steps require source PropCode in vNotes
    if "vState" in df.columns:
        amfee_rows = df[df["vState"] == "AMFee"]
        for _, r in amfee_rows.iterrows():
            vnotes = str(r.get("vNotes", "")).strip()
            if not vnotes:
                errors.append(
                    f"iOrder {int(r['iOrder'])} {r['PropCode']}: AMFee requires "
                    "source investor PropCode in vNotes"
                )

    # 6. Promote steps require capital investor PropCodes in vNotes
    if "vState" in df.columns:
        promote_rows = df[df["vState"] == "Promote"]
        for _, r in promote_rows.iterrows():
            vnotes = str(r.get("vNotes", "")).strip()
            if not vnotes:
                errors.append(
                    f"iOrder {int(r['iOrder'])} {r['PropCode']}: Promote requires "
                    "capital investor PropCodes in vNotes (comma-separated)"
                )

    # 7. Required fields
    for idx, row in df.iterrows():
        if not str(row.get("vState", "")).strip():
            errors.append(f"Row {idx + 1}: vState is required")
        if not str(row.get("PropCode", "")).strip():
            errors.append(f"Row {idx + 1}: PropCode is required")

    # 8. PropCode must exist in relationships table (if valid_entities provided)
    if valid_entities is not None and "PropCode" in df.columns:
        for idx, row in df.iterrows():
            pc = str(row.get("PropCode", "")).strip()
            if pc and pc not in valid_entities:
                errors.append(
                    f"iOrder {int(row.get('iOrder', 0))} {pc}: PropCode does not "
                    "exist in the relationships table"
                )

    return errors, warnings


# ── CRUD ─────────────────────────────────────────────────────────────────

def save_steps(vcode: str, steps_df: pd.DataFrame, full_wf: pd.DataFrame = None) -> dict:
    """Save waterfall steps to database with audit trail.

    steps_df should contain rows for one wf_type (CF_WF or Cap_WF).
    full_wf is the current complete waterfall DataFrame used to preserve
    the other wf_type when saving only one.
    """
    try:
        # Ensure DB columns exist
        for col in DB_COLUMNS:
            if col not in steps_df.columns:
                steps_df[col] = None

        save_waterfall_steps(vcode, steps_df)
        return {"success": True, "message": f"Saved {len(steps_df)} steps for {vcode}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def delete_steps(vcode: str, wf_type: Optional[str] = None) -> dict:
    """Delete waterfall steps for an entity."""
    try:
        delete_waterfall_steps(vcode, wf_type)
        return {"success": True, "message": f"Deleted steps for {vcode}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_waterfall_steps(wf: pd.DataFrame, vcode: str,
                        valid_entities: set = None) -> dict:
    """Get CF_WF and Cap_WF steps for an entity.

    Returns dict with cf_wf and cap_wf as lists of dicts.
    Note: valid_entities is accepted but no longer used for filtering —
    waterfall steps are returned as-is (matching Deal Analysis behavior).
    Validation of PropCodes against relationships is done at save time only.
    """
    entity_wf = wf[wf["vcode"] == vcode] if "vcode" in wf.columns else pd.DataFrame()

    cf_wf = entity_wf[entity_wf["vmisc"] == "CF_WF"] if not entity_wf.empty else pd.DataFrame()
    cap_wf = entity_wf[entity_wf["vmisc"] == "Cap_WF"] if not entity_wf.empty else pd.DataFrame()

    def to_records(df):
        if df.empty:
            return []
        cols = [c for c in WF_COLUMNS if c in df.columns]
        out = df[cols].sort_values("iOrder")
        # Replace NaN with 0.0 for numeric fields, "" for strings
        # (NaN is not valid JSON and breaks frontend parsing)
        for col in ["FXRate", "nPercent", "mAmount"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        for col in ["PropCode", "vState", "vtranstype", "vAmtType", "vNotes"]:
            if col in out.columns:
                out[col] = out[col].fillna("").astype(str)
        return out.to_dict(orient="records")

    return {
        "cf_wf": to_records(cf_wf),
        "cap_wf": to_records(cap_wf),
        "has_cf": not cf_wf.empty,
        "has_cap": not cap_wf.empty,
    }


def get_entities_with_waterfalls(wf: pd.DataFrame, inv: pd.DataFrame) -> list[dict]:
    """Get list of entities that have waterfall definitions.

    Returns list of dicts with vcode, name, has_cf, has_cap.
    """
    if wf is None or wf.empty:
        return []

    vcodes = wf["vcode"].unique() if "vcode" in wf.columns else []
    result = []
    for vc in vcodes:
        entity_wf = wf[wf["vcode"] == vc]
        name = vc
        if inv is not None and not inv.empty:
            deal_row = inv[inv["vcode"] == vc]
            if not deal_row.empty:
                name = deal_row.iloc[0].get("Investment_Name", vc)
        result.append({
            "vcode": vc,
            "name": name,
            "has_cf": "CF_WF" in entity_wf["vmisc"].values if "vmisc" in entity_wf.columns else False,
            "has_cap": "Cap_WF" in entity_wf["vmisc"].values if "vmisc" in entity_wf.columns else False,
        })
    return sorted(result, key=lambda x: x["name"])


def get_entity_nav_data(wf: pd.DataFrame, inv: pd.DataFrame, relationships_raw: pd.DataFrame) -> dict:
    """Build entity navigation data: options list, ownership tree info, investor lists.

    Returns dict with entities list (each with vcode, name, has_wf, investors).
    """
    wf_vcodes = set(wf["vcode"].dropna().unique().tolist()) if wf is not None and not wf.empty else set()

    inv_names = {}
    inv_id_to_vcode = {}
    if inv is not None and not inv.empty:
        for _, r in inv.iterrows():
            vc = str(r.get("vcode", ""))
            nm = str(r.get("Investment_Name", ""))
            iid = str(r.get("InvestmentID", "")).strip()
            if vc:
                inv_names[vc] = nm
                if iid:
                    inv_id_to_vcode[iid] = vc
                    if iid not in inv_names:
                        inv_names[iid] = nm

    # Normalise relationship InvestmentIDs to vcodes
    rel_vcodes = set()
    if relationships_raw is not None and not relationships_raw.empty:
        for eid in relationships_raw["InvestmentID"].astype(str).str.strip().unique():
            rel_vcodes.add(inv_id_to_vcode.get(eid, eid))

    # Include entities from relationships AND any with waterfalls
    all_ids = sorted(rel_vcodes | wf_vcodes)

    entities = []
    for eid in all_ids:
        has_wf = eid in wf_vcodes
        name = inv_names.get(eid, "")
        label = f"{name} ({eid})" if name else eid
        if has_wf:
            label += " [WF]"
        entities.append({
            "vcode": eid,
            "name": name or eid,
            "label": label,
            "has_wf": has_wf,
        })

    return {"entities": entities}


def get_entity_investors(entity_id: str, relationships_raw: pd.DataFrame,
                         acct: pd.DataFrame, inv: pd.DataFrame) -> list[dict]:
    """Return list of investors for an entity with ownership percentages."""
    investors = _get_investors_for_entity(entity_id, relationships_raw, acct, inv)
    return [{"investor_id": iid, "ownership_pct": pct} for iid, pct in investors]


# ── Preview ──────────────────────────────────────────────────────────────

def preview_waterfall(vcode: str, wf_type: str, steps: list[dict]) -> dict:
    """Run $100k test waterfall through draft steps.

    Returns dict with allocations list or error message.
    """
    try:
        from waterfall import run_waterfall

        test_wf = pd.DataFrame(steps)
        if test_wf.empty:
            return {"success": False, "error": "No steps to preview"}

        test_wf["vcode"] = vcode
        test_wf["vmisc"] = wf_type
        for col in ["nmisc", "dteffective"]:
            if col not in test_wf.columns:
                test_wf[col] = None

        # Ensure numeric types
        for col in ["iOrder"]:
            if col in test_wf.columns:
                test_wf[col] = pd.to_numeric(test_wf[col], errors="coerce").fillna(0).astype(int)
        for col in ["FXRate", "nPercent", "mAmount"]:
            if col in test_wf.columns:
                test_wf[col] = pd.to_numeric(test_wf[col], errors="coerce").fillna(0.0)
        for col in ["vState", "PropCode", "vtranstype", "vAmtType", "vNotes"]:
            if col in test_wf.columns:
                test_wf[col] = test_wf[col].fillna("").astype(str).str.strip()

        # Add nPercent_dec (waterfall engine expects this)
        p = pd.to_numeric(test_wf["nPercent"], errors="coerce").fillna(0.0)
        test_wf["nPercent_dec"] = np.where(p > 1.0, p / 100.0, p)

        test_cash = pd.DataFrame([{
            "event_date": date(2025, 12, 31),
            "cash_available": 100_000.0,
        }])

        alloc, states = run_waterfall(
            wf_steps=test_wf,
            vcode=vcode,
            wf_name=wf_type,
            period_cash=test_cash,
            initial_states={},
        )

        if alloc.empty:
            return {"success": True, "allocations": [], "message": "No allocations produced. Check step definitions."}

        summary = alloc.groupby(["PropCode", "vState"])["Allocated"].sum().reset_index()
        summary["Allocated"] = summary["Allocated"].round(2)
        return {
            "success": True,
            "allocations": summary.to_dict(orient="records"),
            "total": float(summary["Allocated"].sum()),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Copy CF_WF -> Cap_WF ────────────────────────────────────────────────

def copy_cf_to_cap(wf: pd.DataFrame, vcode: str) -> dict:
    """Copy CF_WF steps to Cap_WF for an entity.

    Returns the new Cap_WF steps as list of dicts.
    """
    entity_wf = wf[wf["vcode"] == vcode] if "vcode" in wf.columns else pd.DataFrame()
    cf_rows = entity_wf[entity_wf["vmisc"] == "CF_WF"]

    if cf_rows.empty:
        return {"success": False, "error": "No CF_WF steps to copy"}

    cols = [c for c in WF_COLUMNS if c in cf_rows.columns]
    cap_steps = cf_rows[cols].sort_values("iOrder").to_dict(orient="records")
    return {"success": True, "cap_wf": cap_steps}


# ── Copy from another entity ────────────────────────────────────────────

def copy_waterfall_from_entity(source_vcode: str, wf: pd.DataFrame) -> dict:
    """Copy CF_WF and Cap_WF steps from a source entity.

    Returns dict with cf_wf and cap_wf as lists of dicts.
    """
    src = wf[wf["vcode"].astype(str) == str(source_vcode)].copy()
    result = {}
    for wf_type in ("cf_wf", "cap_wf"):
        vmisc = "CF_WF" if wf_type == "cf_wf" else "Cap_WF"
        rows = src[src["vmisc"] == vmisc]
        if not rows.empty:
            cols = [c for c in WF_COLUMNS if c in rows.columns]
            result[wf_type] = rows[cols].sort_values("iOrder").to_dict(orient="records")
        else:
            result[wf_type] = []
    return result


# ── New waterfall templates ──────────────────────────────────────────────

def prefill_new_waterfall(entity_id: str, relationships_raw: pd.DataFrame,
                          acct: pd.DataFrame, inv: pd.DataFrame) -> dict:
    """Create template CF_WF and Cap_WF pre-populated with investors.

    Returns dict with cf_wf and cap_wf as lists of dicts.
    """
    investors = _get_investors_for_entity(entity_id, relationships_raw, acct, inv)

    if not investors:
        blank = {c: (0 if c == "iOrder" else 0.0 if c in ("FXRate", "nPercent", "mAmount") else "")
                 for c in WF_COLUMNS}
        blank["iOrder"] = 1
        blank["FXRate"] = 1.0
        return {"cf_wf": [blank], "cap_wf": [blank]}

    cf_rows = []
    cap_rows = []
    order = 1

    # Pref steps per investor
    for inv_id, pct in investors:
        cf_rows.append({
            "iOrder": order, "PropCode": inv_id, "vState": "Pref",
            "FXRate": 1.0, "nPercent": 0.08, "mAmount": 0.0,
            "vtranstype": "Pref on Initial Capital",
            "vAmtType": "", "vNotes": "Pref" if pct >= 0.5 else "OP",
        })

        # Cap_WF: Initial before Pref
        cap_rows.append({
            "iOrder": order, "PropCode": inv_id, "vState": "Initial",
            "FXRate": 1.0, "nPercent": 0.0, "mAmount": 0.0,
            "vtranstype": "Initial Capital",
            "vAmtType": "", "vNotes": "Pref" if pct >= 0.5 else "OP",
        })
        cap_rows.append({
            "iOrder": order + 1, "PropCode": inv_id, "vState": "Pref",
            "FXRate": 1.0, "nPercent": 0.08, "mAmount": 0.0,
            "vtranstype": "Pref on Initial Capital",
            "vAmtType": "", "vNotes": "Pref" if pct >= 0.5 else "OP",
        })
        order += 2

    # Residual sharing
    sorted_inv = sorted(investors, key=lambda x: x[1], reverse=True)
    for i, (inv_id, pct) in enumerate(sorted_inv):
        row = {
            "iOrder": order, "PropCode": inv_id,
            "vState": "Share" if i == 0 else "Tag",
            "FXRate": round(pct, 4), "nPercent": 0.0, "mAmount": 0.0,
            "vtranstype": "Residual Sharing Ratio",
            "vAmtType": "", "vNotes": "Pref" if pct >= 0.5 else "OP",
        }
        cf_rows.append(row)
        cap_rows.append(dict(row))

    return {"cf_wf": cf_rows, "cap_wf": cap_rows}


def prefill_pari_passu_waterfall(entity_id: str, relationships_raw: pd.DataFrame,
                                 acct: pd.DataFrame, inv: pd.DataFrame) -> dict:
    """Create pari passu waterfall: expenses -> AM fee -> pro-rata distribution.

    Returns dict with cf_wf and cap_wf as lists of dicts.
    """
    investors = _get_investors_for_entity(entity_id, relationships_raw, acct, inv)
    defaults = _PARI_PASSU_DEFAULTS
    expense_propcode = defaults["asset_mgmt_propcode"]

    shared_rows = []

    # iOrder 0: Entity Expenses
    shared_rows.append({
        "iOrder": 0, "PropCode": expense_propcode, "vState": "Amt",
        "FXRate": 1.0, "nPercent": 0.0,
        "mAmount": defaults["entity_expenses_quarterly"],
        "vtranstype": "Entity Expenses",
        "vAmtType": "EXP",
        "vNotes": f"Quarterly: ${defaults['entity_expenses_quarterly'] * 4:,.0f}/yr",
    })

    # iOrder 1: Asset Management Fee
    shared_rows.append({
        "iOrder": 1, "PropCode": defaults["asset_mgmt_propcode"],
        "vState": "Amt", "FXRate": 1.0,
        "nPercent": defaults["asset_mgmt_fee_pct"], "mAmount": 0.0,
        "vtranstype": "Asset Management Fee",
        "vAmtType": "AM_FEE",
        "vNotes": f"{defaults['asset_mgmt_fee_pct']}%/yr of total invested capital",
    })

    # iOrder 2: Pro Rata Distribution
    if investors:
        sorted_inv = sorted(investors, key=lambda x: x[1], reverse=True)
        for i, (inv_id, pct) in enumerate(sorted_inv):
            shared_rows.append({
                "iOrder": 2, "PropCode": inv_id,
                "vState": "Share" if i == 0 else "Tag",
                "FXRate": round(pct, 4), "nPercent": 0.0, "mAmount": 0.0,
                "vtranstype": "Pro Rata Distribution",
                "vAmtType": "ProRata", "vNotes": "",
            })
    else:
        shared_rows.append({
            "iOrder": 2, "PropCode": "", "vState": "Share",
            "FXRate": 1.0, "nPercent": 0.0, "mAmount": 0.0,
            "vtranstype": "Pro Rata Distribution",
            "vAmtType": "ProRata", "vNotes": "",
        })

    # CF_WF and Cap_WF are identical for pari passu
    return {
        "cf_wf": [dict(r) for r in shared_rows],
        "cap_wf": [dict(r) for r in shared_rows],
    }


# ── Internal helpers ─────────────────────────────────────────────────────

def _get_investors_for_entity(entity_id, relationships_raw, acct, inv):
    """Return list of (InvestorID, ownership_fraction) for entity."""
    investors = {}

    if relationships_raw is not None and not relationships_raw.empty:
        rel_inv_ids = relationships_raw["InvestmentID"].astype(str).str.strip()
        rel_investor_ids = relationships_raw["InvestorID"].astype(str).str.strip()
        mask = rel_inv_ids == str(entity_id)
        for idx in mask[mask].index:
            inv_id = rel_investor_ids[idx]
            pct = float(relationships_raw.at[idx, "OwnershipPct"]) if "OwnershipPct" in relationships_raw.columns else 0
            if pct > 1:
                pct = pct / 100.0
            investors[inv_id] = pct

    # Also check accounting feed for additional investors
    if acct is not None and not acct.empty and inv is not None:
        inv_map = {}
        if "InvestmentID" in inv.columns:
            for _, r in inv.iterrows():
                inv_map[str(r["InvestmentID"])] = str(r["vcode"])

        acct_vc = acct["InvestmentID"].astype(str).map(inv_map)
        acct_match = acct[acct_vc == str(entity_id)]
        for inv_id in acct_match["InvestorID"].astype(str).unique():
            if inv_id not in investors:
                investors[inv_id] = 0.0

    result = list(investors.items())
    # Normalise if percentages don't sum to ~1
    total = sum(p for _, p in result)
    if total > 0 and abs(total - 1.0) > 0.01:
        result = [(i, p / total) for i, p in result]
    elif total == 0 and result:
        equal = 1.0 / len(result)
        result = [(i, equal) for i, _ in result]

    return result


def get_ownership_tree_data(vcode: str, inv: pd.DataFrame,
                            relationships_raw: pd.DataFrame) -> dict:
    """Get structured ownership tree for an entity.

    Returns dict with:
      - owners: list of dicts (investors who own this entity)
      - selected: dict (the entity itself)
      - investments: list of dicts (entities this entity invests in)
      - investors: list of dicts (detailed investor info with names)
    """
    result = {"owners": [], "selected": None, "investments": [], "investors": []}

    # Build vcode -> InvestmentID mapping
    vcode_to_inv_id = {}
    inv_names = {}
    if inv is not None and not inv.empty:
        for _, r in inv.iterrows():
            vc = str(r.get("vcode", ""))
            nm = str(r.get("Investment_Name", ""))
            iid = str(r.get("InvestmentID", "")).strip()
            if vc:
                inv_names[vc] = nm
                if iid:
                    vcode_to_inv_id[vc] = iid
                    inv_names[iid] = nm

    tree_id = vcode_to_inv_id.get(vcode, vcode)

    if relationships_raw is None or relationships_raw.empty:
        result["selected"] = {"id": vcode, "name": inv_names.get(vcode, vcode)}
        return result

    try:
        from ownership_tree import build_ownership_tree, load_relationships
        rels = load_relationships(relationships_raw)
        nodes = build_ownership_tree(rels)

        node = nodes.get(tree_id)

        # Selected entity
        entity_name = node.name if node else inv_names.get(vcode, vcode)
        result["selected"] = {"id": tree_id, "name": entity_name}

        if node:
            # Owners: investors who own this entity (sorted by pct desc)
            for inv_id, pct in sorted(node.investors, key=lambda x: x[1], reverse=True):
                inv_node = nodes.get(inv_id)
                inv_name = inv_node.name if inv_node else ""
                tags = []
                if inv_node and not inv_node.investors:
                    tags.append("ULTIMATE OWNER")
                if inv_node and inv_node.is_passthrough:
                    tags.append("PASSTHROUGH")
                result["owners"].append({
                    "id": inv_id,
                    "name": inv_name,
                    "pct": pct,
                    "tags": tags,
                })

            # Investments: entities this entity invests in
            for child_id in sorted(node.investments):
                child_node = nodes.get(child_id)
                child_name = child_node.name if child_node else inv_names.get(child_id, "")
                # Find this entity's ownership stake in the child
                stake = 0.0
                if child_node:
                    for iid, p in child_node.investors:
                        if iid == tree_id:
                            stake = p
                            break
                tags = []
                if child_node and child_node.is_passthrough:
                    tags.append("PASSTHROUGH")
                if child_node and child_node.needs_waterfall:
                    tags.append("NEEDS WATERFALL")
                result["investments"].append({
                    "id": child_id,
                    "name": child_name,
                    "pct": stake,
                    "tags": tags,
                })

            # Detailed investor list (for the collapsible panel)
            for inv_id, pct in node.investors:
                inv_name = nodes[inv_id].name if inv_id in nodes else ""
                label = f"{inv_id} ({inv_name})" if inv_name else inv_id
                result["investors"].append({
                    "investor_id": inv_id,
                    "name": inv_name,
                    "label": label,
                    "ownership_pct": pct,
                })
        else:
            result["selected"] = {"id": vcode, "name": inv_names.get(vcode, vcode)}
    except Exception:
        result["selected"] = {"id": vcode, "name": inv_names.get(vcode, vcode)}

    return result


def build_save_df(vcode: str, wf_type: str, steps: list[dict],
                  other_type_steps: list[dict] = None,
                  full_wf: pd.DataFrame = None) -> pd.DataFrame:
    """Build a combined DataFrame for saving one wf_type while preserving the other.

    steps: the rows being saved (for wf_type)
    other_type_steps: if provided, use these for the other wf_type
    full_wf: fallback for the other wf_type if other_type_steps not provided
    """
    save_df = pd.DataFrame(steps)
    save_df["vcode"] = vcode
    save_df["vmisc"] = wf_type
    for col in DB_COLUMNS:
        if col not in save_df.columns:
            save_df[col] = None

    other_type = "Cap_WF" if wf_type == "CF_WF" else "CF_WF"

    if other_type_steps:
        other_df = pd.DataFrame(other_type_steps)
        other_df["vcode"] = vcode
        other_df["vmisc"] = other_type
        for col in DB_COLUMNS:
            if col not in other_df.columns:
                other_df[col] = None
    elif full_wf is not None and not full_wf.empty:
        other_df = full_wf[
            (full_wf["vcode"].astype(str) == str(vcode))
            & (full_wf["vmisc"] == other_type)
        ].copy()
    else:
        other_df = pd.DataFrame(columns=DB_COLUMNS)

    # Also preserve Promote_WF if it exists
    promote_df = pd.DataFrame(columns=DB_COLUMNS)
    if full_wf is not None and not full_wf.empty:
        promote_df = full_wf[
            (full_wf["vcode"].astype(str) == str(vcode))
            & (full_wf["vmisc"] == "Promote_WF")
        ].copy()

    parts = [save_df[DB_COLUMNS]]
    if not other_df.empty:
        for col in DB_COLUMNS:
            if col not in other_df.columns:
                other_df[col] = None
        parts.append(other_df[DB_COLUMNS])
    if not promote_df.empty:
        parts.append(promote_df[DB_COLUMNS])

    return pd.concat(parts, ignore_index=True)
