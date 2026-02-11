"""
waterfall_setup_ui.py
Waterfall Setup tab — view, edit, and create waterfall structures.

Provides:
- Entity/ownership tree navigation
- Editable CF_WF and Cap_WF waterfall step tables
- On-demand validation (FXRate sums, Add vs Tag, etc.) via Save/Validate buttons
- New-waterfall prefill from relationships/accounting
- Session-draft workflow with explicit "Save to Database"
- Guidance panel from waterfall_setup_rules.txt
"""

import streamlit as st
import pandas as pd
import copy
from datetime import date
from typing import Dict, List, Optional, Tuple

from database import save_waterfall_steps, delete_waterfall_steps
from ownership_tree import (
    build_ownership_tree,
    load_relationships,
    visualize_ownership_tree,
    identify_waterfall_requirements,
)


# ── Constants ────────────────────────────────────────────────────────────
VSTATE_OPTIONS = [
    "Pref", "Initial", "Add", "Tag", "Share", "IRR",
    "Amt", "Def&Int", "Def_Int", "Default", "AMFee", "Promote",
]

WF_COLUMNS = [
    "iOrder", "vAmtType", "vNotes", "PropCode", "vState", "FXRate", "nPercent",
    "mAmount", "vtranstype",
]

# Columns persisted to DB (superset of WF_COLUMNS)
DB_COLUMNS = [
    "vcode", "vmisc", "iOrder", "vAmtType", "vNotes", "PropCode",
    "nmisc", "dteffective", "vtranstype", "mAmount", "nPercent",
    "FXRate", "vState",
]


# ── Public entry point ───────────────────────────────────────────────────

def render_waterfall_setup(wf, inv, relationships_raw, acct):
    """Render the Waterfall Setup tab.

    Delegates to _waterfall_setup_fragment (decorated with @st.fragment)
    so that widget interactions inside the tab (entity selectbox, data_editor
    cell edits, buttons) only rerun this fragment — not the entire app.
    """
    st.header("Waterfall Setup")

    if wf is None or wf.empty:
        st.warning("No waterfall data loaded.")
        return

    _waterfall_setup_fragment(wf, inv, relationships_raw, acct)


@st.fragment
def _waterfall_setup_fragment(wf, inv, relationships_raw, acct):
    """Fragment-isolated body of the Waterfall Setup tab.

    Widgets here (selectbox, data_editor, buttons) trigger only a fragment
    rerun (~200 ms) instead of a full-app rerun (~20 s).
    """
    # Normalise column types once
    wf = _normalise_wf(wf)

    # ── Entity selection ──────────────────────────────────────────────
    entity_id, entity_label = _render_entity_nav(wf, inv, relationships_raw)

    # ── Waterfall editor (full width, below selector) ─────────────────
    if entity_id:
        _render_editor_panel(entity_id, entity_label, wf, inv, relationships_raw, acct)

    # ── Guidance panel (full width, collapsible) ──────────────────────
    _render_guidance_panel()


# ── Entity navigation (left column) ─────────────────────────────────────

def _get_nav_data(wf, inv, relationships_raw):
    """Build and cache entity options, labels, lookup dicts, and ownership tree.

    Results are cached in session_state keyed by DataFrame shapes so they
    are only recomputed when the underlying data actually changes — not on
    every cell-edit rerun.
    """
    cache_key = (
        len(wf),
        len(inv) if inv is not None else 0,
        len(relationships_raw) if relationships_raw is not None else 0,
    )
    cached = st.session_state.get("_wf_nav_cache")
    if cached and cached["key"] == cache_key:
        return cached["data"]

    # ── Expensive work: only runs when data changes ──────────────────
    wf_vcodes = sorted(wf["vcode"].dropna().unique().tolist())

    inv_names = {}
    vcode_to_inv_id = {}
    inv_id_to_vcode = {}
    if inv is not None and not inv.empty:
        for _, r in inv.iterrows():
            vc = str(r.get("vcode", ""))
            nm = str(r.get("Investment_Name", ""))
            iid = str(r.get("InvestmentID", "")).strip()
            if vc:
                inv_names[vc] = nm
                if iid:
                    vcode_to_inv_id[vc] = iid
                    inv_id_to_vcode[iid] = vc
                    # Also let InvestmentID resolve to the same name
                    if iid not in inv_names:
                        inv_names[iid] = nm

    rel_entities = set()
    if relationships_raw is not None and not relationships_raw.empty:
        rel_entities = set(relationships_raw["InvestmentID"].astype(str).str.strip().unique())

    # Normalise relationship InvestmentIDs → vcodes where a mapping exists,
    # so "P0000001" collapses to "30BEAR" and the dropdown is deduplicated.
    rel_vcodes = set()
    for eid in rel_entities:
        rel_vcodes.add(inv_id_to_vcode.get(eid, eid))

    all_entity_ids = sorted(set(wf_vcodes) | rel_vcodes)

    options: List[Tuple[str, str]] = []
    labels: List[str] = []
    for eid in all_entity_ids:
        has_wf = eid in wf_vcodes
        name = inv_names.get(eid, "")
        badge = " [WF]" if has_wf else ""
        lbl = f"{name} ({eid}){badge}" if name else f"{eid}{badge}"
        labels.append(lbl)
        options.append((eid, lbl))

    nodes = {}
    if relationships_raw is not None and not relationships_raw.empty:
        try:
            rels = load_relationships(relationships_raw)
            nodes = build_ownership_tree(rels)
        except Exception:
            pass  # tree display is best-effort

    data = {
        "options": options,
        "labels": labels,
        "vcode_to_inv_id": vcode_to_inv_id,
        "nodes": nodes,
    }
    st.session_state["_wf_nav_cache"] = {"key": cache_key, "data": data}
    return data


def _render_entity_nav(wf, inv, relationships_raw):
    """Build entity selector + mini tree view. Returns (entity_id, label)."""

    st.subheader("Select Entity")

    nav = _get_nav_data(wf, inv, relationships_raw)
    options = nav["options"]
    labels = nav["labels"]
    vcode_to_inv_id = nav["vcode_to_inv_id"]
    nodes = nav["nodes"]

    if not options:
        st.info("No entities found.")
        return None, None

    # Default selection: try to match current deal (vcode first, then label)
    current_vcode = st.session_state.get("_current_deal_vcode", "")
    current_deal = st.session_state.get("deal_selector", "")
    default_idx = 0
    if current_vcode:
        for i, (eid, _) in enumerate(options):
            if eid == current_vcode:
                default_idx = i
                break
    if default_idx == 0 and current_deal:
        for i, (eid, _) in enumerate(options):
            if f"({eid})" in current_deal:
                default_idx = i
                break
        else:
            for i, (_, lbl) in enumerate(options):
                if current_deal in lbl:
                    default_idx = i
                    break

    selected_label = st.selectbox(
        "Entity", labels, index=default_idx, key="_wf_entity_sel"
    )
    selected_idx = labels.index(selected_label)
    entity_id = options[selected_idx][0]

    # Resolve entity_id (vcode) to InvestmentID used in the ownership tree
    tree_id = vcode_to_inv_id.get(entity_id, entity_id)

    # Show tree and investors side-by-side below the selectbox
    if tree_id in nodes:
        node = nodes[tree_id]
        col_tree, col_inv = st.columns(2, gap="medium")

        with col_tree:
            tree_text = visualize_ownership_tree(tree_id, nodes, max_depth=2)
            with st.expander("Ownership Tree", expanded=False):
                st.code(tree_text, language=None)

        with col_inv:
            if node.investors:
                with st.expander(f"Investors ({len(node.investors)})", expanded=False):
                    for inv_id, pct in node.investors:
                        inv_name = nodes[inv_id].name if inv_id in nodes else ""
                        label = f"{inv_id} ({inv_name})" if inv_name else inv_id
                        st.caption(f"{label} — {pct:.2%}")

    return entity_id, selected_label


# ── Editor panel (right column) ─────────────────────────────────────────

def _render_editor_panel(entity_id, entity_label, wf, inv, relationships_raw, acct):
    """Show editable waterfall tables for selected entity."""

    st.subheader(f"Waterfall Steps — {entity_label}")

    # Filter waterfall rows for this entity
    entity_wf = wf[wf["vcode"].astype(str) == str(entity_id)]
    has_cf = not entity_wf[entity_wf["vmisc"] == "CF_WF"].empty
    has_cap = not entity_wf[entity_wf["vmisc"] == "Cap_WF"].empty

    # Check if drafts exist (from create-new or prior edits)
    cf_key = f"_wf_draft|{entity_id}|CF_WF"
    cap_key = f"_wf_draft|{entity_id}|Cap_WF"
    has_draft = cf_key in st.session_state or cap_key in st.session_state

    if not has_cf and not has_cap and not has_draft:
        st.info("No waterfall defined for this entity.")
        btn_new, btn_pari = st.columns(2)
        with btn_new:
            if st.button("Create New Waterfall", key="_wf_create_new"):
                draft = _prefill_new_waterfall(entity_id, relationships_raw, acct, inv)
                st.session_state[cf_key] = draft["CF_WF"]
                st.session_state[cap_key] = draft["Cap_WF"]
                st.rerun()
        with btn_pari:
            if st.button("Create Pari Passu Waterfall", key="_wf_create_pari"):
                draft = _prefill_pari_passu_waterfall(entity_id, relationships_raw, acct, inv)
                st.session_state[cf_key] = draft["CF_WF"]
                st.session_state[cap_key] = draft["Cap_WF"]
                st.rerun()

        # Copy from existing deal
        source_options = _get_entities_with_waterfalls(wf, inv, exclude_entity=entity_id)
        if source_options:
            st.markdown("---")
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                source_label = st.selectbox(
                    "Copy waterfall from",
                    options=[lbl for lbl, _ in source_options],
                    index=None,
                    placeholder="Select a deal to copy from…",
                    key="_wf_copy_source_main",
                )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # align with selectbox
                copy_clicked = st.button("Copy", key="_wf_copy_btn_main")
            if copy_clicked and source_label:
                source_vcode = dict(source_options)[source_label]
                copied = _copy_waterfall_from_entity(source_vcode, wf)
                if not copied["CF_WF"].empty:
                    st.session_state[cf_key] = copied["CF_WF"]
                if not copied["Cap_WF"].empty:
                    st.session_state[cap_key] = copied["Cap_WF"]
                st.rerun()
        return

    tab_cf, tab_cap = st.tabs(["CF_WF", "Cap_WF"])

    with tab_cf:
        _render_wf_type_editor(entity_id, "CF_WF", entity_wf, cf_key, wf, inv, relationships_raw, acct)

    with tab_cap:
        _render_wf_type_editor(entity_id, "Cap_WF", entity_wf, cap_key, wf, inv, relationships_raw, acct)


def _render_wf_type_editor(entity_id, wf_type, entity_wf, draft_key, full_wf, inv, relationships_raw, acct):
    """Render editable data_editor for one waterfall type (CF_WF or Cap_WF)."""

    db_rows = entity_wf[entity_wf["vmisc"] == wf_type].copy()

    # Initialise draft from DB if needed
    if draft_key not in st.session_state:
        if db_rows.empty:
            st.info(f"No {wf_type} steps defined.")
            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"Create {wf_type}", key=f"_wf_create_{wf_type}_{entity_id}"):
                    draft = _prefill_new_waterfall(entity_id, relationships_raw, acct, inv)
                    st.session_state[draft_key] = draft[wf_type]
                    st.rerun()
            with b2:
                if st.button(f"Create Pari Passu {wf_type}", key=f"_wf_create_pari_{wf_type}_{entity_id}"):
                    draft = _prefill_pari_passu_waterfall(entity_id, relationships_raw, acct, inv)
                    st.session_state[draft_key] = draft[wf_type]
                    st.rerun()
            # Copy single type from another deal
            source_options = _get_entities_with_waterfalls(full_wf, inv, exclude_entity=entity_id)
            if source_options:
                col_sel, col_btn = st.columns([3, 1])
                with col_sel:
                    src_lbl = st.selectbox(
                        f"Copy {wf_type} from",
                        options=[lbl for lbl, _ in source_options],
                        index=None,
                        placeholder="Select a deal to copy from…",
                        key=f"_wf_copy_src_{wf_type}_{entity_id}",
                    )
                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Copy", key=f"_wf_copy_btn_{wf_type}_{entity_id}") and src_lbl:
                        src_vc = dict(source_options)[src_lbl]
                        copied = _copy_waterfall_from_entity(src_vc, full_wf)
                        if not copied[wf_type].empty:
                            st.session_state[draft_key] = copied[wf_type]
                            st.rerun()
                        else:
                            st.warning(f"{src_vc} has no {wf_type} steps to copy.")
            return
        else:
            st.session_state[draft_key] = _db_to_editor_df(db_rows)

    editor_df = st.session_state[draft_key]

    # Build column config — PropCode is a text field so users can enter
    # any entity code (expense entities, management co, etc.)
    col_config = {
        "iOrder": st.column_config.NumberColumn("iOrder", min_value=0, max_value=30, step=1),
        "PropCode": st.column_config.TextColumn("PropCode"),
        "vState": st.column_config.SelectboxColumn("vState", options=VSTATE_OPTIONS, required=True),
        "FXRate": st.column_config.NumberColumn("FXRate", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"),
        "nPercent": st.column_config.NumberColumn("nPercent", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"),
        "mAmount": st.column_config.NumberColumn("mAmount", min_value=0.0, format="%.0f"),
        "vtranstype": st.column_config.TextColumn("vtranstype"),
        "vAmtType": st.column_config.TextColumn("vAmtType"),
        "vNotes": st.column_config.TextColumn("vNotes"),
    }

    edited = st.data_editor(
        editor_df,
        column_config=col_config,
        num_rows="dynamic",
        use_container_width=True,
        key=f"_wf_editor_{wf_type}_{entity_id}",
    )
    st.session_state[draft_key] = edited
    current_df = edited

    # ── Action buttons ────────────────────────────────────────────────
    btn_cols = st.columns(6)

    with btn_cols[0]:
        if st.button("Save to Database", key=f"_wf_save_{wf_type}_{entity_id}", type="primary"):
            errs, warns = _validate_steps(current_df, wf_type)
            if errs:
                _show_validation(errs, warns)
                st.error("Fix errors above before saving.")
            else:
                _show_validation(errs, warns)  # show warnings (non-blocking)
                _save_draft(entity_id, wf_type, current_df, full_wf)

    with btn_cols[1]:
        if st.button("Validate", key=f"_wf_validate_{wf_type}_{entity_id}"):
            errs, warns = _validate_steps(current_df, wf_type)
            _show_validation(errs, warns)
            if not errs and not warns:
                st.success("All checks passed.")

    with btn_cols[2]:
        if st.button("Reset to Saved", key=f"_wf_reset_{wf_type}_{entity_id}"):
            if not db_rows.empty:
                st.session_state[draft_key] = _db_to_editor_df(db_rows)
            else:
                st.session_state.pop(draft_key, None)

    with btn_cols[3]:
        if wf_type == "CF_WF":
            if st.button("Copy CF_WF -> Cap_WF", key=f"_wf_copy_{entity_id}"):
                cap_draft = current_df.copy()
                st.session_state[f"_wf_draft|{entity_id}|Cap_WF"] = cap_draft
                st.success("Copied CF_WF steps to Cap_WF draft.")

    with btn_cols[4]:
        csv_data = current_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export CSV",
            csv_data,
            file_name=f"{entity_id}_{wf_type}.csv",
            mime="text/csv",
            key=f"_wf_export_{wf_type}_{entity_id}",
        )

    with btn_cols[5]:
        if st.button("Preview Waterfall", key=f"_wf_preview_{wf_type}_{entity_id}"):
            _preview_waterfall(entity_id, wf_type, current_df)


# ── Validation ───────────────────────────────────────────────────────────

def _validate_steps(df, wf_type):
    """Run validation rules on the editor DataFrame.

    Returns (errors: list[str], warnings: list[str]).
    """
    if df.empty:
        return [], []

    warnings = []
    errors = []

    # 1. FXRate sum per iOrder
    for order, grp in df.groupby("iOrder"):
        fx_sum = grp["FXRate"].sum()
        if abs(fx_sum - 1.0) > 0.02 and fx_sum > 0:
            warnings.append(f"iOrder {int(order)}: FXRate sum = {fx_sum:.4f} (expected ~1.0)")

    # 2. Operating Capital must be Add, not Tag
    oc_rows = df[
        df["vtranstype"].astype(str).str.contains("Operating Capital", case=False, na=False)
    ]
    tag_oc = oc_rows[oc_rows["vState"] == "Tag"]
    if not tag_oc.empty:
        for _, r in tag_oc.iterrows():
            errors.append(
                f"iOrder {int(r['iOrder'])} {r['PropCode']}: Operating Capital "
                "MUST use Add, not Tag"
            )

    # 3. Pref steps should have FX=1.0
    pref_rows = df[df["vState"] == "Pref"]
    for _, r in pref_rows.iterrows():
        if r["FXRate"] != 1.0:
            warnings.append(
                f"iOrder {int(r['iOrder'])} {r['PropCode']}: Pref step FXRate = "
                f"{r['FXRate']:.4f} (typically 1.0)"
            )

    # 4. Each iOrder with Tags must have at least one lead
    for order, grp in df.groupby("iOrder"):
        has_tag = (grp["vState"] == "Tag").any()
        has_lead = (grp["vState"] != "Tag").any()
        if has_tag and not has_lead:
            errors.append(
                f"iOrder {int(order)}: has Tag step(s) but no lead step"
            )

    # 5. AMFee steps require source PropCode in vNotes
    amfee_rows = df[df["vState"] == "AMFee"]
    for _, r in amfee_rows.iterrows():
        vnotes = str(r.get("vNotes", "")).strip()
        if not vnotes:
            errors.append(
                f"iOrder {int(r['iOrder'])} {r['PropCode']}: AMFee requires "
                "source investor PropCode in vNotes"
            )

    # 6. Promote steps require capital investor PropCodes in vNotes
    promote_rows = df[df["vState"] == "Promote"]
    for _, r in promote_rows.iterrows():
        vnotes = str(r.get("vNotes", "")).strip()
        if not vnotes:
            errors.append(
                f"iOrder {int(r['iOrder'])} {r['PropCode']}: Promote requires "
                "capital investor PropCodes in vNotes (comma-separated)"
            )

    return errors, warnings


def _show_validation(errors, warnings):
    """Display validation results in Streamlit."""
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)


# ── Persistence helpers ──────────────────────────────────────────────────

def _save_draft(entity_id, wf_type, editor_df, full_wf):
    """Save the draft to the database."""
    if editor_df.empty:
        st.warning("Nothing to save — table is empty.")
        return

    # Build full rows for DB
    save_df = editor_df.copy()
    save_df["vcode"] = entity_id
    save_df["vmisc"] = wf_type
    # Fill missing columns with defaults
    for col in DB_COLUMNS:
        if col not in save_df.columns:
            save_df[col] = None

    # We only save this wf_type — keep the other type intact.
    # Prefer session-state draft (latest edits) over full_wf (stale startup data).
    other_type = "Cap_WF" if wf_type == "CF_WF" else "CF_WF"
    other_draft_key = f"_wf_draft|{entity_id}|{other_type}"
    if other_draft_key in st.session_state and not st.session_state[other_draft_key].empty:
        other_rows = st.session_state[other_draft_key].copy()
        other_rows["vcode"] = entity_id
        other_rows["vmisc"] = other_type
        for col in DB_COLUMNS:
            if col not in other_rows.columns:
                other_rows[col] = None
    else:
        other_rows = full_wf[
            (full_wf["vcode"].astype(str) == str(entity_id))
            & (full_wf["vmisc"] == other_type)
        ].copy()

    # Also grab Promote_WF if it exists
    promote_rows = full_wf[
        (full_wf["vcode"].astype(str) == str(entity_id))
        & (full_wf["vmisc"] == "Promote_WF")
    ].copy()

    combined = pd.concat([save_df[DB_COLUMNS], other_rows[DB_COLUMNS] if not other_rows.empty else pd.DataFrame(columns=DB_COLUMNS),
                          promote_rows[DB_COLUMNS] if not promote_rows.empty else pd.DataFrame(columns=DB_COLUMNS)],
                         ignore_index=True)

    try:
        save_waterfall_steps(entity_id, combined)
        st.success(f"Saved {len(save_df)} {wf_type} steps for {entity_id}. "
                   "Use **Reload Data** in the sidebar to refresh other tabs.")
    except Exception as e:
        st.error(f"Save failed: {e}")


def _preview_waterfall(entity_id, wf_type, editor_df):
    """Run a $100k test waterfall through the draft steps."""
    try:
        from waterfall import run_waterfall
        import numpy as np

        # Build wf_steps DataFrame matching expected format
        test_wf = editor_df.copy()
        test_wf["vcode"] = entity_id
        test_wf["vmisc"] = wf_type
        for col in ["nmisc", "dteffective"]:
            if col not in test_wf.columns:
                test_wf[col] = None

        # Add nPercent_dec (waterfall engine expects this column)
        p = pd.to_numeric(test_wf["nPercent"], errors="coerce").fillna(0.0)
        test_wf["nPercent_dec"] = np.where(p > 1.0, p / 100.0, p)

        test_cash = pd.DataFrame([{
            "event_date": date(2025, 12, 31),
            "cash_available": 100_000.0,
        }])

        alloc, states = run_waterfall(
            wf_steps=test_wf,
            vcode=entity_id,
            wf_name=wf_type,
            period_cash=test_cash,
            initial_states={},
        )

        if alloc.empty:
            st.info("Waterfall produced no allocations. Check step definitions.")
        else:
            st.markdown("**Preview: $100,000 allocation**")
            summary = alloc.groupby(["PropCode", "vState"])["Allocated"].sum().reset_index()
            summary["Allocated"] = summary["Allocated"].map(lambda x: f"${x:,.2f}")
            st.dataframe(summary, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Preview failed: {e}")


# ── New waterfall prefill logic ──────────────────────────────────────────

def _prefill_new_waterfall(entity_id, relationships_raw, acct, inv):
    """Create template CF_WF and Cap_WF DataFrames pre-populated with investors."""

    investors = _get_investors_for_entity(entity_id, relationships_raw, acct, inv)

    cf_rows = []
    cap_rows = []
    order = 1

    if not investors:
        # Minimal empty template
        blank = {c: None for c in WF_COLUMNS}
        blank["iOrder"] = 1
        blank["FXRate"] = 1.0
        return {"CF_WF": pd.DataFrame([blank]), "Cap_WF": pd.DataFrame([blank])}

    # Pref steps — one per investor, each gets its own iOrder, FX=1.0
    for inv_id, pct in investors:
        row = _blank_row()
        row["iOrder"] = order
        row["PropCode"] = inv_id
        row["vState"] = "Pref"
        row["FXRate"] = 1.0
        row["nPercent"] = 0.08
        row["vtranstype"] = "Pref on Initial Capital"
        row["vNotes"] = "Pref" if pct >= 0.5 else "OP"
        cf_rows.append(row)

        # Cap_WF also gets Initial step before Pref
        cap_initial = _blank_row()
        cap_initial["iOrder"] = order
        cap_initial["PropCode"] = inv_id
        cap_initial["vState"] = "Initial"
        cap_initial["FXRate"] = 1.0
        cap_initial["vtranstype"] = "Initial Capital"
        cap_initial["vNotes"] = row["vNotes"]
        cap_rows.append(cap_initial)

        cap_pref = row.copy()
        cap_pref["iOrder"] = order + 1
        cap_rows.append(cap_pref)

        order += 2

    # Residual sharing — lead = Share (largest investor), rest = Tag
    sorted_inv = sorted(investors, key=lambda x: x[1], reverse=True)
    for i, (inv_id, pct) in enumerate(sorted_inv):
        row = _blank_row()
        row["iOrder"] = order
        row["PropCode"] = inv_id
        row["vState"] = "Share" if i == 0 else "Tag"
        row["FXRate"] = round(pct, 4)
        row["vtranstype"] = "Residual Sharing Ratio"
        row["vNotes"] = "Pref" if pct >= 0.5 else "OP"
        cf_rows.append(row)
        cap_rows.append(row.copy())

    return {
        "CF_WF": pd.DataFrame(cf_rows)[WF_COLUMNS],
        "Cap_WF": pd.DataFrame(cap_rows)[WF_COLUMNS],
    }


# Default expense settings for pari passu waterfalls
_PARI_PASSU_DEFAULTS = {
    "entity_expenses_quarterly": 15625.0,   # $62,500/yr ÷ 4
    "asset_mgmt_fee_pct": 0.5,              # 0.5% of invested capital
    "asset_mgmt_propcode": "PSCMAN",        # Management entity
}


def _prefill_pari_passu_waterfall(entity_id, relationships_raw, acct, inv):
    """Create pari passu CF_WF and Cap_WF: expenses first, then pro-rata distribution.

    Structure:
      iOrder 0  — Entity Expenses (Amt, fixed quarterly deduction)
      iOrder 1  — Asset Management Fee (Amt, % of invested capital)
      iOrder 2  — Pro Rata Distribution (Share + Tag by ownership %)
    Cap_WF mirrors CF_WF exactly for pari passu.
    """
    investors = _get_investors_for_entity(entity_id, relationships_raw, acct, inv)
    defaults = _PARI_PASSU_DEFAULTS

    expense_propcode = f"{entity_id}_EXP"

    shared_rows = []

    # ── iOrder 0: Entity Expenses ─────────────────────────────────────
    exp_row = _blank_row()
    exp_row["iOrder"] = 0
    exp_row["vAmtType"] = "EXP"
    exp_row["vNotes"] = f"Quarterly: ${defaults['entity_expenses_quarterly'] * 4:,.0f}/yr"
    exp_row["PropCode"] = expense_propcode
    exp_row["vState"] = "Amt"
    exp_row["FXRate"] = 1.0
    exp_row["mAmount"] = defaults["entity_expenses_quarterly"]
    exp_row["vtranstype"] = "Entity Expenses"
    shared_rows.append(exp_row)

    # ── iOrder 1: Asset Management Fee ────────────────────────────────
    amf_row = _blank_row()
    amf_row["iOrder"] = 1
    amf_row["vAmtType"] = "AM_FEE"
    amf_row["vNotes"] = f"{defaults['asset_mgmt_fee_pct']}%/yr of total invested capital"
    amf_row["PropCode"] = defaults["asset_mgmt_propcode"]
    amf_row["vState"] = "Amt"
    amf_row["FXRate"] = 1.0
    amf_row["nPercent"] = defaults["asset_mgmt_fee_pct"]
    amf_row["vtranstype"] = "Asset Management Fee"
    shared_rows.append(amf_row)

    # ── iOrder 2: Pro Rata Distribution ───────────────────────────────
    if investors:
        sorted_inv = sorted(investors, key=lambda x: x[1], reverse=True)
        for i, (inv_id, pct) in enumerate(sorted_inv):
            row = _blank_row()
            row["iOrder"] = 2
            row["vAmtType"] = "ProRata"
            row["PropCode"] = inv_id
            row["vState"] = "Share" if i == 0 else "Tag"
            row["FXRate"] = round(pct, 4)
            row["vtranstype"] = "Pro Rata Distribution"
            shared_rows.append(row)
    else:
        # Placeholder row if no investors found
        row = _blank_row()
        row["iOrder"] = 2
        row["vAmtType"] = "ProRata"
        row["vState"] = "Share"
        row["FXRate"] = 1.0
        row["vtranstype"] = "Pro Rata Distribution"
        shared_rows.append(row)

    # CF_WF and Cap_WF are identical for pari passu
    import copy
    cf_rows = [r.copy() for r in shared_rows]
    cap_rows = [r.copy() for r in shared_rows]

    return {
        "CF_WF": pd.DataFrame(cf_rows)[WF_COLUMNS],
        "Cap_WF": pd.DataFrame(cap_rows)[WF_COLUMNS],
    }


def _get_investors_for_entity(entity_id, relationships_raw, acct, inv):
    """Return list of (InvestorID, ownership_fraction) for entity."""
    investors = {}

    if relationships_raw is not None and not relationships_raw.empty:
        rel_inv_ids = relationships_raw["InvestmentID"].astype(str).str.strip()
        rel_investor_ids = relationships_raw["InvestorID"].astype(str).str.strip()
        mask = rel_inv_ids == str(entity_id)
        for idx in mask[mask].index:
            inv_id = rel_investor_ids[idx]
            pct = float(relationships_raw.at[idx, "OwnershipPct"] if "OwnershipPct" in relationships_raw.columns else 0)
            if pct > 1:
                pct = pct / 100.0
            investors[inv_id] = pct

    # Also check accounting feed for additional investors
    if acct is not None and not acct.empty and inv is not None:
        # Map InvestmentID -> vcode
        inv_map = {}
        if "InvestmentID" in inv.columns:
            for _, r in inv.iterrows():
                inv_map[str(r["InvestmentID"])] = str(r["vcode"])

        acct_vc = acct["InvestmentID"].astype(str).map(inv_map)
        acct_match = acct[acct_vc == str(entity_id)]
        for inv_id in acct_match["InvestorID"].astype(str).unique():
            if inv_id not in investors:
                investors[inv_id] = 0.0  # unknown pct

    result = list(investors.items())
    # Normalise if percentages don't sum to ~1
    total = sum(p for _, p in result)
    if total > 0 and abs(total - 1.0) > 0.01:
        result = [(i, p / total) for i, p in result]
    elif total == 0 and result:
        equal = 1.0 / len(result)
        result = [(i, equal) for i, _ in result]

    return result


# ── DataFrame helpers ────────────────────────────────────────────────────

def _normalise_wf(wf):
    """Ensure consistent dtypes on waterfall DataFrame.

    Uses a lightweight session_state cache keyed by DataFrame identity (id + shape)
    to avoid re-normalising on every Streamlit rerun.  Much faster than
    @st.cache_data which must hash the entire DataFrame to check cache validity.
    """
    cache_key = (id(wf), len(wf))
    cached = st.session_state.get("_wf_normalised_cache")
    if cached is not None and cached["key"] == cache_key:
        return cached["data"]

    out = wf.copy()
    out["vcode"] = out["vcode"].astype(str).str.strip()
    out["vmisc"] = out["vmisc"].astype(str).str.strip()
    for col in ["iOrder"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["FXRate", "nPercent", "mAmount", "nmisc"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in ["vState", "PropCode", "vtranstype", "vAmtType", "vNotes"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.strip()

    st.session_state["_wf_normalised_cache"] = {"key": cache_key, "data": out}
    return out


def _db_to_editor_df(rows):
    """Convert DB rows to editor-ready DataFrame (WF_COLUMNS only)."""
    df = rows[WF_COLUMNS].copy().sort_values("iOrder").reset_index(drop=True)
    return df


def _get_entities_with_waterfalls(full_wf, inv, exclude_entity=None):
    """Return sorted list of (label, vcode) for entities that have waterfalls."""
    vcodes = full_wf["vcode"].dropna().unique()
    # Build label lookup from deals table
    label_map = {}
    if inv is not None and not inv.empty:
        for _, r in inv.iterrows():
            vc = str(r.get("vcode", "")).strip()
            nm = str(r.get("Investment_Name", "")).strip()
            if vc and nm:
                label_map[vc] = nm
    result = []
    for vc in sorted(vcodes):
        vc = str(vc).strip()
        if vc and vc != str(exclude_entity):
            lbl = f"{label_map[vc]} ({vc})" if vc in label_map else vc
            result.append((lbl, vc))
    return result


def _copy_waterfall_from_entity(source_vcode, full_wf):
    """Copy CF_WF and Cap_WF steps from a source entity as editor DataFrames."""
    src = full_wf[full_wf["vcode"].astype(str) == str(source_vcode)].copy()
    result = {}
    for wf_type in ("CF_WF", "Cap_WF"):
        rows = src[src["vmisc"] == wf_type]
        if not rows.empty:
            result[wf_type] = _db_to_editor_df(rows)
        else:
            result[wf_type] = pd.DataFrame(columns=WF_COLUMNS)
    return result


def _blank_row():
    """Return a blank row dict with WF_COLUMNS keys."""
    return {c: (0 if c in ("iOrder",) else 0.0 if c in ("FXRate", "nPercent", "mAmount") else "") for c in WF_COLUMNS}


def _get_propcode_options(entity_id, relationships_raw, entity_wf):
    """Collect possible PropCode values for selectbox."""
    opts = set()

    # From existing waterfall steps
    if not entity_wf.empty:
        opts.update(entity_wf["PropCode"].dropna().astype(str).str.strip().unique())

    # From relationships
    if relationships_raw is not None and not relationships_raw.empty:
        rel = relationships_raw.copy()
        rel["InvestmentID"] = rel["InvestmentID"].astype(str).str.strip()
        investors = rel[rel["InvestmentID"] == str(entity_id)]
        opts.update(investors["InvestorID"].astype(str).str.strip().unique())

    opts.discard("")
    return sorted(opts) if opts else [""]


# ── Guidance panel ───────────────────────────────────────────────────────

def _render_guidance_panel():
    """Static guidance from waterfall_setup_rules.txt."""

    with st.expander("Waterfall Setup Guide", expanded=False):
        st.markdown("### vState Reference")
        st.markdown("""
| vState | Description |
|--------|-------------|
| **Pref** | Pay accrued preferred return. `nPercent` = annual rate. Accrues daily Act/365. |
| **Initial** | Return initial capital. Cap_WF reduces capital; CF_WF skips. |
| **Add** | Route to capital pool per `vtranstype`. Independent cap/tracking. |
| **Tag** | Proportional follower of lead step at same iOrder. No pool routing. |
| **Share** | Residual distribution. `FXRate` = sharing percentage. |
| **IRR** | IRR-targeted distribution (hurdle gate). `nPercent` = target IRR. |
| **Amt** | Fixed-amount distribution. `mAmount` = dollar amount per period. |
| **Def&Int** | Default interest + principal. CF_WF: interest only; Cap_WF: both. |
| **Def_Int** | Default interest only. Does not reduce capital. |
| **Default** | Default principal return. Reduces capital_outstanding. |
| **AMFee** | Post-distribution AM fee. Deducts from source (vNotes) investor, pays to PropCode. Pool-neutral. `nPercent` = annual rate, `mAmount` = periods/yr. |
| **Promote** | Cumulative catch-up. `FXRate` = carry share, `nPercent` = target carry %. `vNotes` = comma-separated capital investor PropCodes. Paired with Tags. |
""")

        st.warning(
            "**ADD vs TAG Rule:** If an investor contributed capital to a pool "
            "that needs tracked and returned, that investor MUST have an **Add** "
            "step, NEVER a Tag."
        )

        st.error(
            "**Operating Capital Rule:** Operating capital MUST always use **Add** "
            "for each investor. Tag steps bypass pool routing — capital_outstanding "
            "is never reduced and cumulative caps are never enforced."
        )

        st.markdown("### Pool Routing Table (`vtranstype` -> pool)")
        st.markdown("""
| vtranstype contains | Pool | Action (CF_WF) | Action (Cap_WF) |
|---------------------|------|----------------|-----------------|
| Operating Capital | operating | pay_capital_capped | pay_capital_capped |
| Cost Overrun + Pref | cost_overrun | pay_pref | pay_pref |
| Cost Overrun | cost_overrun | skip | pay_capital |
| Special Capital + Pref | special | pay_pref | pay_pref |
| Special Capital | special | skip | pay_capital |
| Pref (other) | additional | pay_pref | pay_pref |
| (default) | additional | skip | pay_capital |
""")

        st.markdown("### Common Patterns")
        st.info("""**Pattern A — Simple two-partner deal:**
- iOrder 1: Lead + Tag for Default/Def&Int (prorata)
- iOrder 3-4: Separate Pref steps per investor (FX=1.0 each)
- iOrder 5: Share (lead) + Tag (partner) for residual

**Pattern B — Operating capital deal:**
- iOrder 1: Default step(s)
- iOrder 2: **Add + Add** for operating capital (separate mAmount caps)
- iOrder 3-4: Pref steps per investor
- iOrder 5: Share + Tag for residual""")

        st.markdown("### Modeling Checklist")
        st.markdown("""
1. Identify all capital pools (initial, operating, additional, special, cost overrun, default)
2. For each pool with multiple investors: independent cap -> separate Add; shared prorata -> Add + Tag
3. Operating capital MUST always use Add (never Tag) for each investor
4. Set `mAmount` correctly for operating capital = cumulative cap
5. Verify FXRates at each iOrder sum to ~1.0
6. Pref steps: own iOrder per investor, FX=1.0
7. Residual sharing: lead = Share, partner = Tag
8. Set up both CF_WF and Cap_WF; CF_WF never reduces non-operating capital
9. Verify `vtranstype` text matches routing rules
""")
