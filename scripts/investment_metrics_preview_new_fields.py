"""Preview what UW Proj. IRR / Proj Yr-1 CoC would change, WITHOUT a refresh.

Read-only. Builds the Investment Metrics report from the local waterfall.db
twice: as the live deal_terms stands (no new columns) and with `uw_irr` /
`proj_yr1_coc` joined on from MRI-side values you supply. Nothing is written,
no MRI connection is made, no refresh is triggered.

    python scripts/investment_metrics_preview_new_fields.py values.csv|xlsx

The file needs columns vcode, uw_irr, proj_yr1_coc (fractions, 0.11 = 11%) --
the output of the new Prop_Info_DealTerms.sql run in SSMS. A raw
txfinancial_IC export (vCode, vTransType, dtEffective, nPercent, UID) is also
accepted: the latest dtEffective per (vCode, type) is taken, as the SQL does.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import investment_metrics as im                     # noqa: E402
from flask_app.services import data_service        # noqa: E402

THREE = ("act_yr1_coc", "proj_coc_since_close", "act_coc_since_close")
TYPES = {"U/W IRR": "uw_irr", "Projected Yr 1 CoC Returns": "proj_yr1_coc"}


def load_values(path):
    df = (pd.read_excel(path) if path.lower().endswith(("xlsx", "xls"))
          else pd.read_csv(path))
    df.columns = [str(c) for c in df.columns]
    if "vTransType" in df.columns:
        df = df[df["vTransType"].isin(TYPES)].copy()
        df["dtEffective"] = pd.to_datetime(df["dtEffective"])
        sort = ["dtEffective"] + (["UID"] if "UID" in df.columns else [])
        df = df.sort_values(sort).groupby(["vCode", "vTransType"]).tail(1)
        df["field"] = df["vTransType"].map(TYPES)
        df = (df.pivot(index="vCode", columns="field", values="nPercent")
                .reset_index().rename(columns={"vCode": "vcode"}))
    for c in ("uw_irr", "proj_yr1_coc"):
        if c not in df.columns:
            df[c] = None
    return df[["vcode", "uw_irr", "proj_yr1_coc"]]


def build(data, terms):
    return im.build_investment_metrics(
        data["inv"], data["acct"], commitments=data.get("commitments_raw"),
        deal_terms=terms, loans=data.get("mri_loans_all"),
        isbs_interim_bs=data.get("isbs_interim_bs"),
        waterfalls=data.get("wf"), isbs_raw=data.get("isbs_raw"))


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def main(path):
    vals = load_values(path)
    from flask_app import create_app
    with create_app().app_context():
        data = data_service.get_data()
    base_terms = data["deal_terms_raw"]
    new_terms = base_terms.merge(vals, on="vcode", how="left")
    before, after = build(data, base_terms), build(data, new_terms)
    print(f"as of {after['as_of']}; values supplied for {len(vals)} deals "
          f"({vals.uw_irr.notna().sum()} UW IRR, "
          f"{vals.proj_yr1_coc.notna().sum()} Proj Yr-1 CoC)")
    for t in ("current", "sold"):
        rb = {r["vcode"]: r for r in before[t]["rows"]}
        print(f"\n== {t.upper()} ==")
        shown = [r for r in after[t]["rows"]
                 if r["uw_irr"] is not None or r["proj_yr1_coc"] is not None]
        print(f"deals that would show UW IRR / Proj Yr-1: {len(shown)}")
        for r in shown:
            print(f"  {r['vcode']} {r['name'][:28]:28} UW IRR {pct(r['uw_irr']):>6}"
                  f"  Proj Yr-1 {pct(r['proj_yr1_coc']):>6}")
        young = [r for r in after[t]["rows"] if r.get("young_deal")]
        print(f"young deals (footnote 5): {len(young)}")
        for r in young:
            b = rb[r["vcode"]]
            print(f"  {r['vcode']} {r['name'][:28]:28} proj {pct(r['proj_yr1_coc']):>6}"
                  + "".join(f" | {f}: {pct(b[f])}->{pct(r[f])}" for f in THREE))
        tb, ta = before[t]["total"], after[t]["total"]
        for f in ("proj_yr1_coc", "uw_irr") + THREE:
            if tb.get(f) != ta.get(f):
                print(f"  Total row {f}: {pct(tb.get(f))} -> {pct(ta.get(f))}")
        moved = [f for f in ("proj_yr1_coc", "uw_irr") + THREE
                 if tb.get(f) == ta.get(f)]
        print(f"  Total row unchanged: {moved}")
    d = after["diagnostics"]
    print("\ndiagnostics: absent =", len(d.get("unloaded_figure_field_absent", [])),
          "| NULL =", {k: len(v["vcodes"]) for k, v in
                       d.get("unloaded_figure_value_null", {}).items()})


if __name__ == "__main__":
    main(sys.argv[1])
