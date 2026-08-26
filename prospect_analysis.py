"""
prospect_analysis.py
Synthetic data builder for New Business deal analysis.

Converts prospect_assumptions form inputs into the DataFrames that
compute_deal_analysis() expects, then calls the shared engine.
This is a TRANSLATION LAYER, not a new engine.
"""

import logging
import json
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, Any, Optional, List

from utils import month_end, add_months
from config import (
    REVENUE_ACCTS, EXPENSE_ACCTS, CAPEX_ACCTS,
    DEFAULT_HORIZON_YEARS, DEFAULT_START_YEAR, PRO_YR_BASE_DEFAULT,
)

logger = logging.getLogger(__name__)
from models import Loan
from loans import build_loans_from_mri_loans, amortize_monthly_schedule
from compute import compute_deal_analysis


def compute_capital_budget(assumptions: dict, pe_pct_fallback: float = 0.7) -> dict:
    """Replicate the app's Capital Budget arithmetic from the saved JSON.

    Percentage-based use items store their pct, not an amount, and the PSC
    Origination Fee is a grossed-up 1% of PE equity -- so the saved amounts
    alone understate total uses. This mirrors getUseAmount()/pscOrigFee()/
    totalUses() in ProspectAnalysisView exactly, and is the single source the
    engine and the audit workbook both read, so they cannot disagree with
    the screen.
    """
    import json as _json

    def _load(blob):
        if not blob:
            return None
        try:
            return _json.loads(blob) if isinstance(blob, str) else blob
        except (TypeError, ValueError):
            return None

    uses = _load((assumptions or {}).get('capital_uses_json')) or []
    srcs = _load((assumptions or {}).get('capital_sources_json')) or {}

    debt_rows = [d for d in (srcs.get('debt') or []) if isinstance(d, dict)]
    total_debt = sum(float(d.get('amount') or 0) for d in debt_rows)

    pp = 0.0
    for u in uses:
        if isinstance(u, dict) and u.get('id') == 'purchase_price':
            pp = float(u.get('amount') or 0)

    resolved = []          # (id, label, amount)
    base_total = 0.0
    for u in uses:
        if not isinstance(u, dict) or u.get('isFixed'):
            continue
        amt = 0.0
        pct = u.get('pct')
        if u.get('pctBase') and pct not in (None, 0, ''):
            base = pp if u['pctBase'] == 'purchase_price' else total_debt
            amt = float(pct) / 100.0 * base
        else:
            amt = float(u.get('amount') or 0)
        if amt:
            resolved.append((u.get('id'), u.get('label') or u.get('id'), amt))
        base_total += amt

    pe_pct = float(srcs.get('pe_pct') or pe_pct_fallback * 100) / 100.0
    orig_pct = 0.01
    denom = 1 - orig_pct * pe_pct
    fee = 0.0
    if denom > 0 and base_total > total_debt:
        fee = orig_pct * pe_pct * (base_total - total_debt) / denom
    if fee:
        resolved.append(('psc_orig_fee', 'PSC Origination Fee', fee))

    total_uses = base_total + fee
    equity = max(0.0, total_uses - total_debt)
    capex_reserve = next((a for i, _, a in resolved if i == 'capex_reserve'), 0.0)

    equity_rows = [e for e in (srcs.get('equity') or []) if isinstance(e, dict)]

    return {
        'uses': resolved,
        'total_uses': total_uses,
        'debt_rows': debt_rows,
        'total_debt': total_debt,
        'equity': equity,
        'pe_pct': pe_pct,
        'pe_amount': equity * pe_pct,
        'op_amount': equity * (1 - pe_pct),
        'extra_equity_rows': equity_rows,
        'capex_reserve': capex_reserve,
        'psc_orig_fee': fee,
    }


def apply_scenario_adjustments(fc, adjustments, debug_msgs=None):
    """Apply a scenario's income adjustments to the forecast.

    Each adjustment is {'label', 'start_date', 'end_date' (optional),
    'revenue': {acct: annual $}, 'expense': {acct: annual $}}. Amounts are
    annual, spread across the months inside the window. Positive removes the
    line (a tenant leaves, an expense is saved); negative adds it back (space
    re-leased, a new cost arrives). Only mAmount_norm is touched -- the
    column every NOI/FAD/DSCR/terminal-value path reads -- so the effect
    flows through debt sizing checks, the exit value and the waterfall.
    """
    if fc is None or fc.empty or not adjustments:
        return fc
    msgs = debug_msgs if debug_msgs is not None else []
    out = fc.copy()
    acct_str = out["vAccount"].astype(str).str.strip()
    dates = pd.to_datetime(out["event_date"])

    for adj in adjustments:
        label = adj.get("label") or "Adjustment"
        start = pd.to_datetime(adj.get("start_date"), errors="coerce")
        if pd.isna(start):
            msgs.append(f"Scenario adjustment '{label}' skipped: no usable start date")
            continue
        end = pd.to_datetime(adj.get("end_date"), errors="coerce") if adj.get("end_date") else None
        window = dates >= start
        if end is not None and not pd.isna(end):
            window &= dates <= end
        span = f"from {start.date()}" + (f" to {end.date()}" if end is not None and not pd.isna(end) else " onward")

        for kind, sign in (("revenue", -1.0), ("expense", 1.0)):
            for acct, annual in (adj.get(kind) or {}).items():
                try:
                    annual_amt = float(annual or 0)
                except (TypeError, ValueError):
                    continue
                if annual_amt == 0:
                    continue
                monthly = annual_amt / 12.0
                mask = window & (acct_str == str(acct).strip())
                n_rows = int(mask.sum())
                if n_rows == 0:
                    msgs.append(
                        f"Scenario adjustment '{label}': account {acct} has no "
                        f"forecast rows {span}; ${annual_amt:,.0f}/yr of {kind} "
                        f"not applied"
                    )
                    continue
                out.loc[mask, "mAmount_norm"] = out.loc[mask, "mAmount_norm"] + sign * monthly
                verb = "removed" if annual_amt > 0 else "added"
                msgs.append(
                    f"Scenario adjustment '{label}': {verb} "
                    f"${abs(annual_amt):,.0f}/yr of {kind} on account {acct} "
                    f"{span} ({n_rows} months)"
                )
    return out


def build_prospect_analysis(
    deal: dict,
    properties: list,
    entities: list,
    assumptions: dict,
    cashflows: Optional[list] = None,
    argus_forecast_df: Optional[pd.DataFrame] = None,
    waterfall_df: Optional[pd.DataFrame] = None,
    scenario: Optional[dict] = None,
    parcel_sales: Optional[list] = None,
) -> dict:
    """Build synthetic DataFrames and run compute_deal_analysis().

    Args:
        deal: prospect_deals row dict (deal_name, vcode, purchase_price, etc.)
        properties: list of prospect_properties row dicts
        entities: list of prospect_entities row dicts (with nested 'investors')
        assumptions: prospect_assumptions row dict
        cashflows: optional list of prospect_cashflows row dicts

    Returns:
        Full result dict from compute_deal_analysis(), or dict with 'error' key.
    """
    vcode = deal.get('vcode') or f"N{deal['id']:07d}"

    # Scenario: overlay its assumption overrides before anything reads them.
    scenario_msgs = []
    if scenario:
        overrides = scenario.get('assumption_overrides') or {}
        if overrides:
            assumptions = {**(assumptions or {}), **overrides}
            scenario_msgs.append(
                "Scenario '" + str(scenario.get('name')) + "': assumption "
                "overrides applied: " + ", ".join(sorted(overrides.keys()))
            )
    deal_name = deal.get('deal_name', 'Prospect')

    # Resolve key assumptions with defaults
    purchase_price = float(deal.get('purchase_price') or 0)
    closing_cost_pct = float(deal.get('closing_cost_pct') or 0.02)
    capex_at_close = float(deal.get('capex_at_close') or 0)

    debt_amount = float(assumptions.get('debt_amount') or 0)
    debt_rate = float(assumptions.get('debt_rate') or 0.05)
    debt_term_months = int(assumptions.get('debt_term_months') or 84)
    io_months = int(assumptions.get('io_months') or 60)
    amort_months = int(assumptions.get('amort_months') or 360)
    psc_equity_pct = float(assumptions.get('psc_equity_pct') or 0.90)
    pref_rate = float(assumptions.get('pref_rate') or 0.08)
    promote_pct = float(assumptions.get('promote_pct') or 0.20)
    hold_years = int(assumptions.get('hold_years') or 7)
    noi_year1 = float(assumptions.get('noi_year1') or 0)
    noi_growth_rate = float(assumptions.get('noi_growth_rate') or 0.02)
    exit_cap_rate = float(assumptions.get('exit_cap_rate') or 0.06)
    selling_cost_pct = float(assumptions.get('selling_cost_pct') or 0.02)
    capex_reserve_psf = float(assumptions.get('capex_reserve_psf') or 0.80)

    # The Capital Budget is the source of truth for cost and equity when it
    # exists; the purchase-price formula only covers deals without one. The
    # budget replicates the app's own arithmetic (pct-based items, grossed-up
    # PSC fee), so partner contributions tie to the Sources table on screen.
    budget = compute_capital_budget(assumptions, pe_pct_fallback=psc_equity_pct)
    if budget['total_uses'] > 0:
        total_cost = budget['total_uses']
        equity_needed = budget['equity']
        scenario_msgs.append(
            f"Equity from Capital Budget: total uses ${budget['total_uses']:,.0f} "
            f"- debt ${budget['total_debt']:,.0f} = ${budget['equity']:,.0f}"
        )
    else:
        total_cost = purchase_price + (purchase_price * closing_cost_pct) + capex_at_close
        equity_needed = total_cost - debt_amount
    # Prefer explicit PE/OP amounts from Capital Budget over percentage split
    pe_equity_override = assumptions.get('pe_equity_amount')
    op_equity_override = assumptions.get('op_equity_amount')
    logger.info("Equity overrides: pe=%s op=%s psc_equity_pct=%s equity_needed=%s",
                pe_equity_override, op_equity_override, psc_equity_pct, equity_needed)
    if pe_equity_override is not None and op_equity_override is not None:
        pe_equity = float(pe_equity_override)
        op_equity = float(op_equity_override)
        equity_needed = pe_equity + op_equity
    else:
        pe_equity = equity_needed * psc_equity_pct
        op_equity = equity_needed * (1 - psc_equity_pct)
    logger.info("Final equity: pe=%s op=%s total=%s", pe_equity, op_equity, equity_needed)

    # Derive timing
    target_close = deal.get('target_close')
    close_date = None
    if target_close:
        try:
            close_date = pd.to_datetime(target_close).date()
        except Exception:
            close_date = None
    if close_date is None:
        # No close date on the deal: anchor to the start of the actual cash
        # flow source, so the sale clock (close + hold) cannot drift from the
        # forecast clock and truncate the final hold year mid-stream.
        fc_start = None
        if argus_forecast_df is not None and not argus_forecast_df.empty:
            fc_start = pd.to_datetime(argus_forecast_df['event_date']).min().date()
        elif cashflows:
            try:
                fc_start = min(pd.to_datetime(cf['period_date']).date()
                               for cf in cashflows if cf.get('period_date'))
            except (TypeError, ValueError):
                fc_start = None
        if fc_start is not None:
            close_date = fc_start
            scenario_msgs.append(
                f"Close date defaulted to the cash flow start {fc_start} - "
                f"set Target Close on the deal to override."
            )
        else:
            close_date = date(DEFAULT_START_YEAR, 1, 1)

    start_year = close_date.year
    model_start = date(start_year, 1, 31)
    # Sale date = end of month, hold_years after close (e.g., 9/24/2026 + 5yr → 9/30/2031)
    sale_date = month_end(add_months(close_date, hold_years * 12))
    pro_yr_base = start_year - 1

    # Contribution date must be within the accounting seed window
    # (default cutoff = Dec 31 of start_year - 1)
    seed_date = date(start_year - 1, 12, 31)

    # Investor IDs from entities or defaults
    pe_investor_id, op_investor_id = _resolve_investors(entities, psc_equity_pct)

    # --- Build synthetic DataFrames ---

    inv = _build_inv(vcode, deal_name, deal, close_date, properties)
    identity_warnings: list = []

    # Investor records declared in Pipeline: {planned_investor_id: info}.
    # When the waterfall's PropCodes all match declared investors, the equity
    # split comes from their ownership percentages (or commitments) instead of
    # being inferred from the waterfall's shape -- inference cannot tell the
    # PE partner apart when both sides carry Pref or IRR steps.
    declared_investors = {}
    for ent in (entities or []):
        for dinv in (ent.get('investors') or []):
            pid = str(dinv.get('planned_investor_id') or '').strip()
            if not pid:
                continue
            itype = str(dinv.get('investor_type') or '').lower()
            declared_investors[pid] = {
                'share': float(dinv.get('ownership_pct') or 0),
                'commitment': float(dinv.get('commitment') or 0),
                'is_pe': 'pref' in itype or itype == 'pe',
            }

    if waterfall_df is not None and not waterfall_df.empty:
        wf = waterfall_df
        # Build accounting from waterfall investors so IDs match
        wf_investors = _get_waterfall_investors(wf)
        wf_ids = [iid for iid, _ in wf_investors]
        if wf_investors and declared_investors and \
                all(iid in declared_investors for iid in wf_ids):
            # Declared beats inferred: split by ownership %, falling back to
            # commitment proportions when percentages are absent.
            weights = {iid: declared_investors[iid]['share'] for iid in wf_ids}
            basis = 'ownership percentages'
            if sum(weights.values()) <= 0:
                weights = {iid: declared_investors[iid]['commitment'] for iid in wf_ids}
                basis = 'commitments'
            total_w = sum(weights.values())
            rows = []
            if total_w > 0:
                for iid in wf_ids:
                    amt = equity_needed * weights[iid] / total_w
                    if amt > 0:
                        rows.append({
                            'InvestmentID': vcode, 'InvestorID': iid,
                            'EffectiveDate': seed_date, 'MajorType': 'Contributions',
                            'Amt': -abs(amt), 'Capital': 'Y',
                            'Typename': 'Investments', 'TypeID': 1001, 'Partner': iid,
                        })
            if rows:
                acct = pd.DataFrame(rows)
                identity_warnings.append(
                    "Equity split taken from declared investor records ("
                    + basis + "): "
                    + ", ".join(f"{iid} {weights[iid] / total_w:.0%}" for iid in wf_ids)
                )
                # PE/OP flags also come from the declaration
                wf_investors = [(iid, declared_investors[iid]['is_pe'])
                                for iid in wf_ids]
            else:
                acct = _build_accounting_from_waterfall(
                    vcode, deal_name, wf_investors, equity_needed,
                    psc_equity_pct, seed_date, warn=identity_warnings)
        elif wf_investors:
            acct = _build_accounting_from_waterfall(
                vcode, deal_name, wf_investors, equity_needed,
                psc_equity_pct, seed_date, warn=identity_warnings)
        if wf_investors:
            # Set pe/op IDs for the investment map (first PE, first non-PE)
            pe_ids = [iid for iid, is_pe in wf_investors if is_pe]
            op_ids = [iid for iid, is_pe in wf_investors if not is_pe]
            pe_investor_id = pe_ids[0] if pe_ids else pe_investor_id
            op_investor_id = op_ids[0] if op_ids else op_investor_id
        else:
            acct = _build_accounting(vcode, deal_name, pe_investor_id,
                                     op_investor_id, pe_equity, op_equity, seed_date)
    else:
        wf = _build_waterfall(vcode, pe_investor_id, op_investor_id,
                              psc_equity_pct, pref_rate, promote_pct)
        acct = _build_accounting(vcode, deal_name, pe_investor_id, op_investor_id,
                                 pe_equity, op_equity, seed_date)
    if argus_forecast_df is not None and not argus_forecast_df.empty:
        fc = argus_forecast_df
        forecast_source = 'argus'
    else:
        forecast_source = 'cashflows' if cashflows else 'noi_growth'
        fc = _build_forecast(vcode, start_year, hold_years, noi_year1,
                             noi_growth_rate, capex_reserve_psf, properties,
                             cashflows, pro_yr_base, assumptions)

    if scenario and (scenario.get('adjustments') or []):
        fc = apply_scenario_adjustments(fc, scenario['adjustments'], scenario_msgs)
    coa = _build_coa()

    # Build synthetic MRI_Val with exit cap rate so compute.py uses
    # twelve_month_noi_after_date() / cap_rate consistently for both
    # the displayed terminal NOI and the sale price calculation.
    # Set dtValuation = sale_date so there's no annual escalation.
    mri_val = pd.DataFrame([{
        'vcode': vcode,
        'fCapRate': exit_cap_rate,
        'dtValuation': sale_date,
    }]) if exit_cap_rate > 0 else pd.DataFrame()

    debt_sources = None
    try:
        _srcs = assumptions.get('capital_sources_json')
        if _srcs:
            _parsed = json.loads(_srcs) if isinstance(_srcs, str) else _srcs
            debt_sources = _parsed.get('debt') if isinstance(_parsed, dict) else None
    except (TypeError, ValueError):
        debt_sources = None

    mri_loans_raw = _build_loans(vcode, debt_amount, debt_rate,
                                 debt_term_months, io_months, amort_months,
                                 close_date, debt_sources=debt_sources)

    # Planned refinancing within the hold. Reuses the AM prospective-loan
    # machinery end to end: the old loans retire at the refi date, the new
    # loan amortises from there, net proceeds flow through the capital
    # waterfall, and a shortfall raises the capital-call flag.
    prospective_loans_raw = None
    try:
        _refi_raw = assumptions.get('planned_refi_json')
        refi = json.loads(_refi_raw) if isinstance(_refi_raw, str) else (_refi_raw or {})
    except (TypeError, ValueError):
        refi = {}
    if refi.get('enabled') and refi.get('refi_date') and float(refi.get('loan_amount') or 0) > 0:
        # The loans being replaced: every loan _build_loans just created,
        # named the same way it names them.
        replace_ids = []
        if mri_loans_raw is not None and not mri_loans_raw.empty:
            replace_ids = [str(x) for x in mri_loans_raw['LoanID'].tolist()]
        prospective_loans_raw = pd.DataFrame([{
            'vcode': vcode, 'status': 'accepted',
            'refi_date': refi['refi_date'],
            'loan_amount': float(refi.get('loan_amount') or 0),
            'interest_rate': float(refi.get('rate') or 0),
            'term_years': float(refi.get('term_years') or 10),
            'amort_years': float(refi.get('amort_years') or 30),
            'io_years': float(refi.get('io_years') or 0),
            'closing_costs': float(refi.get('closing_costs') or 0),
            'reserve_holdback': float(refi.get('holdback') or 0),
            'existing_loan_id': ",".join(replace_ids),
            'max_ltv': 0, 'min_dscr': 0, 'min_debt_yield': 0,
            'lender_uw_noi': 0,
            'int_type': 'Fixed', 'loan_name': 'PLANNED-REFI',
        }])
        scenario_msgs.append(
            f"Planned refinancing at {refi['refi_date']}: new loan "
            f"${float(refi['loan_amount']):,.0f} @ "
            f"{float(refi.get('rate') or 0) * 100:.2f}% replaces "
            f"{', '.join(replace_ids) or 'no existing loans'}"
        )
        # The engine extends the sale date to the new loan's maturity. A
        # maturity past the end of the cash flow forecast means the terminal
        # NOI -- and with it the exit value -- computes from no data at all.
        try:
            from utils import add_months as _am
            _refi_maturity = _am(pd.to_datetime(refi['refi_date']).date(),
                                 int(round(float(refi.get('term_years') or 10) * 12)))
            _fc_end = (pd.to_datetime(argus_forecast_df['event_date']).max().date() if argus_forecast_df is not None and not argus_forecast_df.empty else None)
            if _fc_end and _refi_maturity > _fc_end:
                scenario_msgs.append(
                    f"WARNING: the refi loan matures {_refi_maturity}, after the cash "
                    f"flow forecast ends ({_fc_end}). The sale date follows the new "
                    f"maturity, so the exit value will compute from empty data and "
                    f"understate returns badly. Shorten the refi term or import a "
                    f"longer Argus projection."
                )
        except Exception:
            pass
    if mri_loans_raw is not None and len(mri_loans_raw) > 1:
        parts = ", ".join(
            f"{r['LoanID']}: ${r['mOrigLoanAmt']:,.0f} @ {r['nRate']:.2f}%"
            for _, r in mri_loans_raw.iterrows())
        scenario_msgs.append(f"Individually financed: {parts}")

    result = compute_deal_analysis(
        deal_vcode=vcode,
        deal_investment_id=vcode,
        sale_date_raw=sale_date,
        inv=inv,
        wf=wf,
        acct=acct,
        fc=fc,
        coa=coa,
        mri_loans_raw=mri_loans_raw,
        mri_supp=pd.DataFrame(),
        mri_val=mri_val,
        relationships_raw=pd.DataFrame(),
        prospective_loans_raw=prospective_loans_raw,
        capital_calls_raw=pd.DataFrame(),
        isbs_raw=pd.DataFrame(),
        start_year=start_year,
        horizon_years=(sale_date.year - start_year + 2),  # cover through terminal NOI year
        pro_yr_base=pro_yr_base,
        actuals_through=None,
        contract_sale_price=None,  # let compute.py derive from NOI / cap rate
        selling_cost_override=selling_cost_pct,
        selling_cost_type='pct',
        parcel_sales=parcel_sales,
        beginning_cash_override=(budget['capex_reserve'] if budget['capex_reserve'] > 0 else None),
    )

    # Derive terminal NOI and exit value from compute result (single source of truth)
    sale_dbg = result.get('sale_dbg') or {}
    terminal_noi = sale_dbg.get('NOI_12m_After_Sale', 0)
    exit_value = sale_dbg.get('Implied_Value', 0)

    result['forecast_source'] = forecast_source
    result['capital_budget'] = budget

    if scenario or scenario_msgs:
        existing = result.get('debug_msgs') or []
        header = (["Scenario: " + str(scenario.get('name') or scenario.get('id'))]
                  if scenario else [])
        result['debug_msgs'] = header + scenario_msgs + existing
        if scenario:
            result['scenario'] = {'id': scenario.get('id'), 'name': scenario.get('name')}

    # Surface investor-identity warnings in the Diagnostics panel.  These
    # describe misattributed capital, so they lead the list.
    if identity_warnings:
        existing = result.get('debug_msgs') or []
        result['debug_msgs'] = [f"WARNING: {w}" for w in identity_warnings] + existing

    # Attach assumptions summary for the UI
    result['prospect_assumptions'] = {
        'purchase_price': purchase_price,
        'closing_costs': purchase_price * closing_cost_pct,
        'capex_at_close': capex_at_close,
        'total_cost': total_cost,
        'debt_amount': debt_amount,
        'equity_needed': equity_needed,
        'pe_equity': pe_equity,
        'op_equity': op_equity,
        'ltv': debt_amount / purchase_price if purchase_price else 0,
        'debt_rate': debt_rate,
        'pref_rate': pref_rate,
        'promote_pct': promote_pct,
        'noi_year1': noi_year1,
        'noi_growth_rate': noi_growth_rate,
        'exit_cap_rate': exit_cap_rate,
        'hold_years': hold_years,
        'close_date': str(close_date),
        'sale_date': str(sale_date),
        'terminal_noi': terminal_noi,
        'exit_value': exit_value,
    }

    return result


# ---------------------------------------------------------------------------
# Synthetic DataFrame builders
# ---------------------------------------------------------------------------

def _resolve_investors_from_waterfall(wf, fallback_pe, fallback_op):
    """Extract investor IDs from waterfall PropCode values.

    Returns (pe_id, op_id) where pe_id is the first investor with a Pref
    step and op_id is the first investor without one.  When more than two
    investors exist, _build_accounting_from_waterfall should be used instead.
    """
    if wf is None or wf.empty:
        return fallback_pe, fallback_op

    pc_col = 'PropCode' if 'PropCode' in wf.columns else 'propcode'
    st_col = 'vState' if 'vState' in wf.columns else 'vstate'
    if pc_col not in wf.columns or st_col not in wf.columns:
        return fallback_pe, fallback_op

    all_ids = list(dict.fromkeys(wf[pc_col].dropna()))  # unique, preserving order
    pref_ids = set(wf.loc[wf[st_col].str.lower() == 'pref', pc_col].dropna().unique())
    non_pref_ids = [i for i in all_ids if i not in pref_ids]

    pe_id = list(pref_ids)[0] if pref_ids else fallback_pe
    op_id = non_pref_ids[0] if non_pref_ids else fallback_op

    return pe_id, op_id


def _get_waterfall_investors(wf):
    """Get all unique investor IDs from waterfall PropCode, with PE flag."""
    if wf is None or wf.empty:
        return []
    pc_col = 'PropCode' if 'PropCode' in wf.columns else 'propcode'
    st_col = 'vState' if 'vState' in wf.columns else 'vstate'
    if pc_col not in wf.columns:
        return []

    all_ids = list(dict.fromkeys(wf[pc_col].dropna()))
    pref_ids = set()
    if st_col in wf.columns:
        pref_ids = set(wf.loc[wf[st_col].str.lower() == 'pref', pc_col].dropna().unique())

    # Also treat IRR lookback investors as PE (they have capital at risk)
    irr_ids = set()
    if st_col in wf.columns:
        irr_ids = set(wf.loc[wf[st_col].str.upper() == 'IRR', pc_col].dropna().unique())

    return [(iid, iid in pref_ids or iid in irr_ids) for iid in all_ids]


def _resolve_investors(entities: list, psc_equity_pct: float):
    """Extract PE and OP investor IDs from entity structure."""
    pe_id = "PE"
    op_id = "OP"
    for ent in (entities or []):
        role = (ent.get('role') or '').lower()
        eid = ent.get('planned_entity_id') or ent.get('entity_name', '')
        if 'investor' in role or 'lp' in role or psc_equity_pct >= 0.5:
            if role in ('investor', 'lp', 'pe'):
                pe_id = eid or pe_id
            elif role in ('operator', 'gp', 'op'):
                op_id = eid or op_id
    return pe_id, op_id


def _build_inv(vcode, deal_name, deal, close_date, properties):
    """Build the investment map (deals) DataFrame."""
    rows = [{
        'vcode': vcode,
        'InvestmentID': vcode,
        'Investment_Name': deal_name,
        'Asset_Type': deal.get('asset_type', 'Commercial'),
        'Acquisition_Date': close_date,
        'Acquisition_Price': deal.get('purchase_price'),
        'Sale_Status': None,
        'Lifecycle': 'Active',
        'Portfolio_Name': None,
    }]
    # Add child properties for portfolio deals
    if len(properties) > 1:
        rows[0]['Portfolio_Name'] = None  # parent has no portfolio
        for i, prop in enumerate(properties):
            pv = prop.get('vcode') or f"{vcode}-{i+1:02d}"
            rows.append({
                'vcode': pv,
                'InvestmentID': pv,
                'Investment_Name': prop.get('property_name', f'Property {i+1}'),
                'Asset_Type': prop.get('asset_type') or deal.get('asset_type', 'Commercial'),
                'Acquisition_Date': close_date,
                'Acquisition_Price': prop.get('property_price'),
                'Sale_Status': None,
                'Lifecycle': 'Active',
                'Portfolio_Name': deal_name,
            })

    return pd.DataFrame(rows)


def _build_waterfall(vcode, pe_id, op_id, pe_pct, pref_rate, promote_pct):
    """Build a standard two-partner waterfall (CF_WF + Cap_WF).

    Structure:
        1. Pref return to PE investor
        2. Initial capital return (Cap_WF only)
        3. Residual split: promote to OP, remainder to PE
    """
    op_pct = 1.0 - pe_pct
    pref_pct = pref_rate  # already decimal

    steps = []

    for wf_name in ['CF_WF', 'Cap_WF']:
        order = 10
        # Pref to PE
        steps.append({
            'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
            'PropCode': pe_id, 'dteffective': date(2020, 1, 1),
            'mAmount': 0, 'nPercent': pref_pct * 100, 'FXRate': 1.0,
            'vState': 'Pref', 'vtranstype': 'Preferred Return',
            'vAmtType': '', 'vNotes': '',
        })
        order += 10

        # Initial capital return (Cap_WF only)
        if wf_name == 'Cap_WF':
            steps.append({
                'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
                'PropCode': pe_id, 'dteffective': date(2020, 1, 1),
                'mAmount': 0, 'nPercent': 0, 'FXRate': 1.0,
                'vState': 'Initial', 'vtranstype': 'Return of Capital',
                'vAmtType': '', 'vNotes': '',
            })
            order += 10
            steps.append({
                'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
                'PropCode': op_id, 'dteffective': date(2020, 1, 1),
                'mAmount': 0, 'nPercent': 0, 'FXRate': 1.0,
                'vState': 'Initial', 'vtranstype': 'Return of Capital',
                'vAmtType': '', 'vNotes': '',
            })
            order += 10

        # Residual split — PE share (lead)
        steps.append({
            'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
            'PropCode': pe_id, 'dteffective': date(2020, 1, 1),
            'mAmount': 0, 'nPercent': 0, 'FXRate': pe_pct,
            'vState': 'Share', 'vtranstype': 'Excess Cash Flow',
            'vAmtType': '', 'vNotes': '',
        })
        order += 10

        # Residual split — OP share (tag), includes promote
        steps.append({
            'vcode': vcode, 'vmisc': wf_name, 'iOrder': order,
            'PropCode': op_id, 'dteffective': date(2020, 1, 1),
            'mAmount': 0, 'nPercent': 0, 'FXRate': op_pct,
            'vState': 'Tag', 'vtranstype': 'Promote / Residual',
            'vAmtType': '', 'vNotes': '',
        })

    df = pd.DataFrame(steps)
    # Normalize like load_waterfalls()
    df['nPercent_dec'] = np.where(
        pd.to_numeric(df['nPercent'], errors='coerce') > 1.0,
        pd.to_numeric(df['nPercent'], errors='coerce') / 100.0,
        pd.to_numeric(df['nPercent'], errors='coerce'),
    )
    df['iOrder'] = df['iOrder'].astype(int)
    df['FXRate'] = df['FXRate'].astype(float)
    df['mAmount'] = df['mAmount'].astype(float)
    return df


def _build_accounting_from_waterfall(vcode, deal_name, wf_investors,
                                     total_equity, psc_equity_pct, close_date,
                                     warn=None):
    """Build accounting feed with contributions for all waterfall investors.

    Splits equity among investors using psc_equity_pct for the PE vs OP
    split.  The PE flag is inferred from the waterfall (a Pref or IRR step),
    which is ambiguous when no investor carries one or when several do --
    both cases misattribute capital, so they are reported through `warn`
    (a list of diagnostic strings) rather than passing silently.

    Whatever the split, the apportioned contributions are reconciled against
    total_equity before returning: a mismatch corrupts every deal-level
    return metric, so it is scaled back to the capital budget and reported.
    """
    warn = warn if warn is not None else []
    rows = []
    n_investors = len(wf_investors)
    if n_investors == 0:
        return pd.DataFrame(columns=[
            'InvestmentID', 'InvestorID', 'EffectiveDate', 'MajorType',
            'Amt', 'Capital', 'Typename', 'TypeID', 'Partner',
        ])

    if n_investors == 1:
        iid, _ = wf_investors[0]
        rows.append({
            'InvestmentID': vcode, 'InvestorID': iid,
            'EffectiveDate': close_date, 'MajorType': 'Contributions',
            'Amt': -abs(total_equity), 'Capital': 'Y',
            'Typename': 'Investments', 'TypeID': 1001, 'Partner': iid,
        })
    elif n_investors == 2:
        # Two investors: use psc_equity_pct to split
        pe_ids = [iid for iid, is_pe in wf_investors if is_pe]
        if not pe_ids:
            warn.append(
                f"Equity split is a guess: no waterfall investor has a Pref or "
                f"IRR step, so no PE partner could be identified. "
                f"{wf_investors[0][0]} was given the "
                f"{psc_equity_pct:.0%} PE share because it is first in "
                f"waterfall order. Set the investor's planned entity ID and "
                f"capital role to fix the attribution."
            )
        elif len(pe_ids) > 1:
            warn.append(
                f"Equity split is unreliable: {len(pe_ids)} investors "
                f"({', '.join(pe_ids)}) each have a Pref or IRR step, so each "
                f"was treated as the PE partner and given the "
                f"{psc_equity_pct:.0%} PE share. Contributions have been "
                f"scaled back to the capital budget, but the split between "
                f"partners should not be relied on."
            )
        for iid, is_pe in wf_investors:
            if is_pe or (not pe_ids and iid == wf_investors[0][0]):
                share = psc_equity_pct
            else:
                share = 1 - psc_equity_pct
            amt = total_equity * share
            if amt > 0:
                rows.append({
                    'InvestmentID': vcode, 'InvestorID': iid,
                    'EffectiveDate': close_date, 'MajorType': 'Contributions',
                    'Amt': -abs(amt), 'Capital': 'Y',
                    'Typename': 'Investments', 'TypeID': 1001, 'Partner': iid,
                })
    else:
        # N investors: split equally (user should refine via capital budget)
        per_investor = total_equity / n_investors
        for iid, _ in wf_investors:
            rows.append({
                'InvestmentID': vcode, 'InvestorID': iid,
                'EffectiveDate': close_date, 'MajorType': 'Contributions',
                'Amt': -abs(per_investor), 'Capital': 'Y',
                'Typename': 'Investments', 'TypeID': 1001, 'Partner': iid,
            })

    # Reconcile against the capital budget.  A total that does not match
    # total_equity understates or overstates invested capital, which throws
    # off deal-level IRR, ROE and MOIC as well as the partner split, so
    # scale the contributions back and say so.
    apportioned = sum(abs(r['Amt']) for r in rows)
    if rows and total_equity and abs(apportioned - abs(total_equity)) > 1.0:
        scale = abs(total_equity) / apportioned
        warn.append(
            f"Contributions totalled ${apportioned:,.0f} against "
            f"${abs(total_equity):,.0f} of equity in the capital budget "
            f"({apportioned / abs(total_equity):.2f}x); each investor's "
            f"contribution was scaled by {scale:.3f} to reconcile."
        )
        for r in rows:
            r['Amt'] = -abs(abs(r['Amt']) * scale)

    return pd.DataFrame(rows)


def _build_accounting(vcode, deal_name, pe_id, op_id, pe_equity, op_equity, close_date):
    """Build synthetic accounting feed with initial contributions."""
    rows = []
    if pe_equity > 0:
        rows.append({
            'InvestmentID': vcode,
            'InvestorID': pe_id,
            'EffectiveDate': close_date,
            'MajorType': 'Contributions',
            'Amt': -abs(pe_equity),
            'Capital': 'Y',
            'Typename': 'Investments',
            'TypeID': 1001,
            'Partner': pe_id,
        })
    if op_equity > 0:
        rows.append({
            'InvestmentID': vcode,
            'InvestorID': op_id,
            'EffectiveDate': close_date,
            'MajorType': 'Contributions',
            'Amt': -abs(op_equity),
            'Capital': 'Y',
            'Typename': 'Investments',
            'TypeID': 1001,
            'Partner': op_id,
        })

    if not rows:
        return pd.DataFrame(columns=[
            'InvestmentID', 'InvestorID', 'EffectiveDate', 'MajorType',
            'Amt', 'Capital', 'Typename', 'TypeID', 'Partner',
        ])

    return pd.DataFrame(rows)


def _build_forecast(vcode, start_year, hold_years, noi_year1,
                    noi_growth_rate, capex_reserve_psf, properties,
                    cashflows, pro_yr_base, assumptions=None):
    """Build synthetic forecast from NOI growth assumptions or explicit cashflows.

    Supports three modes:
    1. Line-item cashflows (vaccount present) — each row has a COA account
    2. Summary cashflows (revenue/expenses/capex) — legacy format
    3. NOI growth rate assumptions — generate from scratch

    Operating overrides (from assumptions):
    - mgmt_fee_pct: If set, replaces imported 5040 (mgmt fee) with gross_rev × pct
    - replacement_reserve_psf: If set, adds monthly 5092 expense = total_sf × psf / 12
    """
    assumptions = assumptions or {}
    mgmt_fee_pct = None
    replacement_reserve_psf = None
    try:
        v = assumptions.get('mgmt_fee_pct')
        if v is not None and v != '' and v != 0:
            mgmt_fee_pct = float(v)
    except (ValueError, TypeError):
        pass
    try:
        v = assumptions.get('replacement_reserve_psf')
        if v is not None and v != '' and v != 0:
            replacement_reserve_psf = float(v)
    except (ValueError, TypeError):
        pass

    total_gla = sum(float(p.get('gla_sf') or 0) for p in (properties or []))
    has_line_items = assumptions.get('_has_line_items', False)

    rows = []

    if cashflows and has_line_items:
        # Mode 1: Line-item cashflows with vAccount codes
        for cf in cashflows:
            period_date = pd.to_datetime(cf['period_date']).date()
            vaccount = cf.get('vaccount')
            amount = float(cf.get('amount') or 0)
            if not vaccount or amount == 0:
                continue

            vaccount = int(vaccount)
            # Skip imported mgmt fee if override is set
            if mgmt_fee_pct is not None and vaccount == 5040:
                continue

            rows.append(_fc_row(vcode, period_date, vaccount, amount, pro_yr_base))

        # Apply mgmt fee override: gross_rev × mgmt_fee_pct per period
        if mgmt_fee_pct is not None and rows:
            _apply_mgmt_fee_override(rows, vcode, mgmt_fee_pct, pro_yr_base)

        # Apply replacement reserve: total_sf × psf / 12 per month
        if replacement_reserve_psf is not None and total_gla > 0 and rows:
            _apply_replacement_reserve(rows, vcode, replacement_reserve_psf,
                                       total_gla, pro_yr_base)

    elif cashflows:
        # Mode 2: Legacy summary cashflows (revenue/expenses/capex)
        for cf in cashflows:
            period_date = pd.to_datetime(cf['period_date']).date()
            rev = float(cf.get('revenue') or 0)
            exp = float(cf.get('expenses') or 0)
            capex = float(cf.get('capex') or 0)

            if rev:
                rows.append(_fc_row(vcode, period_date, 4010, rev, pro_yr_base))
            if exp:
                rows.append(_fc_row(vcode, period_date, 5040, -abs(exp), pro_yr_base))
            if capex:
                rows.append(_fc_row(vcode, period_date, 7050, -abs(capex), pro_yr_base))

        # Apply overrides to summary-format too
        if mgmt_fee_pct is not None and rows:
            # Remove 5040 rows and replace with % of gross revenue
            rows = [r for r in rows if r['vAccount'] != 5040]
            _apply_mgmt_fee_override(rows, vcode, mgmt_fee_pct, pro_yr_base)

        if replacement_reserve_psf is not None and total_gla > 0 and rows:
            _apply_replacement_reserve(rows, vcode, replacement_reserve_psf,
                                       total_gla, pro_yr_base)
    else:
        # Mode 3: Generate from NOI growth rate — monthly periods
        annual_capex = total_gla * capex_reserve_psf if total_gla > 0 else 0

        for yr_offset in range(hold_years + 1):  # +1 for terminal NOI year
            year = start_year + yr_offset
            annual_noi = noi_year1 * ((1 + noi_growth_rate) ** yr_offset)

            # Split NOI into revenue and expense (assume 60/40 gross margin)
            annual_revenue = annual_noi / 0.60  # gross revenue
            annual_expenses = annual_revenue - annual_noi

            monthly_rev = annual_revenue / 12
            monthly_exp = annual_expenses / 12
            monthly_capex = annual_capex / 12

            for month in range(1, 13):
                period_date = month_end(date(year, month, 1))
                rows.append(_fc_row(vcode, period_date, 4010, monthly_rev, pro_yr_base))
                rows.append(_fc_row(vcode, period_date, 5040, -abs(monthly_exp), pro_yr_base))
                if monthly_capex > 0:
                    rows.append(_fc_row(vcode, period_date, 7050, -abs(monthly_capex), pro_yr_base))

    if not rows:
        return pd.DataFrame(columns=[
            'vcode', 'event_date', 'vAccount', 'mAmount', 'Pro_Yr',
            'vAccountType', 'mAmount_norm',
        ])

    df = pd.DataFrame(rows)
    return df


def _apply_mgmt_fee_override(rows, vcode, mgmt_fee_pct, pro_yr_base):
    """Replace management fee with gross_revenue × mgmt_fee_pct per period.

    Gross revenue = sum of all GROSS_REVENUE_ACCTS amounts per period.
    """
    from config import GROSS_REVENUE_ACCTS
    # Group existing rows by period to find gross revenue
    period_rev = {}
    for r in rows:
        acct = r['vAccount']
        if acct in GROSS_REVENUE_ACCTS:
            dt = r['event_date']
            period_rev[dt] = period_rev.get(dt, 0) + abs(r['mAmount_norm'])

    # Add mgmt fee rows (negative expense)
    for dt, gross_rev in period_rev.items():
        fee = gross_rev * mgmt_fee_pct
        if fee > 0:
            rows.append(_fc_row(vcode, dt, 5040, -fee, pro_yr_base))


def _apply_replacement_reserve(rows, vcode, reserve_psf, total_gla, pro_yr_base):
    """Add monthly replacement reserve expense (5092) = total_sf × psf / 12."""
    # Find all unique periods from existing rows
    periods = sorted(set(r['event_date'] for r in rows))
    monthly_reserve = total_gla * reserve_psf / 12

    for dt in periods:
        rows.append(_fc_row(vcode, dt, 5092, -monthly_reserve, pro_yr_base))


def _fc_row(vcode, period_date, account, amount, pro_yr_base):
    """Create a single forecast row with normalized amount."""
    from config import (GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS,
                        EXPENSE_ACCTS, ALL_EXCLUDED, TAX_ABATEMENT_ACCTS)
    acct = int(account)
    raw = float(amount)

    # Normalize sign convention (same as normalize_forecast_signs)
    if acct in GROSS_REVENUE_ACCTS:
        norm = abs(raw)
    elif acct in CONTRA_REVENUE_ACCTS:
        norm = -abs(raw)
    elif acct in EXPENSE_ACCTS:
        norm = -abs(raw)
    elif acct in ALL_EXCLUDED:
        norm = -abs(raw)
    elif acct in TAX_ABATEMENT_ACCTS:
        norm = abs(raw)
    else:
        norm = raw

    pro_yr = period_date.year - pro_yr_base
    return {
        'vcode': vcode,
        'event_date': period_date,
        'vAccount': acct,
        'mAmount': raw,
        'Pro_Yr': pro_yr,
        'vAccountType': '',
        'mAmount_norm': norm,
    }


def _build_coa():
    """Build minimal chart of accounts covering standard accounts."""
    from config import (REVENUE_ACCTS, EXPENSE_ACCTS, INTEREST_ACCTS,
                        PRINCIPAL_ACCTS, CAPEX_ACCTS, TAX_ABATEMENT_ACCTS,
                        OTHER_EXCLUDED_ACCTS)
    rows = []
    for a in REVENUE_ACCTS:
        rows.append({'vAccount': a, 'vAccountType': 'Revenue'})
    for a in EXPENSE_ACCTS:
        rows.append({'vAccount': a, 'vAccountType': 'Expense'})
    for a in INTEREST_ACCTS:
        rows.append({'vAccount': a, 'vAccountType': 'Interest'})
    for a in PRINCIPAL_ACCTS:
        rows.append({'vAccount': a, 'vAccountType': 'Principal'})
    for a in CAPEX_ACCTS:
        rows.append({'vAccount': a, 'vAccountType': 'CapEx'})
    for a in TAX_ABATEMENT_ACCTS:
        rows.append({'vAccount': a, 'vAccountType': 'Tax Abatement'})
    for a in OTHER_EXCLUDED_ACCTS:
        rows.append({'vAccount': a, 'vAccountType': 'Other'})

    df = pd.DataFrame(rows)
    df['vAccount'] = df['vAccount'].astype('Int64')
    return df


def _build_loans(vcode, debt_amount, debt_rate, debt_term_months,
                 io_months, amort_months, close_date, debt_sources=None):
    """Build synthetic MRI loans DataFrame.

    Deals are financed two ways in practice: one loan against the group, or
    each property financed individually. A debt source row that carries its
    own rate becomes its own loan (terms falling back to the deal-level
    values where blank); whatever debt remains after the individually-termed
    rows is one blended loan at the deal-level terms. With no per-row terms
    this reduces to the original single-loan behaviour.
    """
    _cols = ['vCode', 'LoanID', 'dtEvent', 'mOrigLoanAmt', 'iAmortTerm',
             'mNominalPenalty', 'iLoanTerm', 'nRate', 'vSpread', 'nFloor',
             'vIntRatereset', 'vIntType', 'vIndex']
    if debt_amount <= 0:
        return pd.DataFrame(columns=_cols)

    def _row(loan_id, amount, rate, term_m, io_m, amort_m):
        return {
            'vCode': vcode,
            'LoanID': loan_id,
            'dtEvent': add_months(close_date, int(term_m)),
            'mOrigLoanAmt': float(amount),
            'iAmortTerm': int(amort_m),
            'mNominalPenalty': int(io_m),
            'iLoanTerm': int(term_m),
            'nRate': rate * 100 if rate < 1 else rate,
            'vSpread': 0, 'nFloor': 0, 'vIntRatereset': 0,
            'vIntType': 'Fixed', 'vIndex': '',
        }

    rows = []
    individual_total = 0.0
    for src in (debt_sources or []):
        if not isinstance(src, dict):
            continue
        try:
            amt = float(src.get('amount') or 0)
            rate = src.get('rate')
            rate = float(rate) if rate not in (None, '') else None
        except (TypeError, ValueError):
            continue
        if amt <= 0 or rate is None or rate <= 0:
            continue  # no own terms -> folds into the blended loan
        term_m = int(float(src.get('term_months') or 0) or debt_term_months)
        io_m = int(float(src.get('io_months') or 0) or io_months)
        amort_m = int(float(src.get('amort_months') or 0) or amort_months)
        rows.append(_row(f"{vcode}-{src.get('id') or f'L{len(rows) + 1}'}",
                         amt, rate, term_m, io_m, amort_m))
        individual_total += amt

    remainder = debt_amount - individual_total
    if remainder > 0.5:
        rows.append(_row(f'{vcode}-L1', remainder, debt_rate,
                         debt_term_months, io_months, amort_months))
    if not rows:
        return pd.DataFrame(columns=_cols)
    return pd.DataFrame(rows)
