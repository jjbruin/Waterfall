-- Prop_Info_DealTerms.sql
-- PE deal terms from txfinancial_IC (Investment Checklist).
-- Server: IM
-- One row per vCode with pivoted deal term fields.
-- These are authoritative MRI-entered values for coupon, IRR hurdle, PE split, etc.

SELECT
    vCode                                                                   AS vcode,
    MAX(CASE WHEN vtranstype = 'PE Coupon'              THEN npercent END)  AS pe_coupon,
    MAX(CASE WHEN vtranstype = 'IRR Lookback'           THEN npercent END)  AS irr_lookback,
    MAX(CASE WHEN vtranstype = 'PE Split (Capital Event)' THEN npercent END) AS pe_split_capital,
    MAX(CASE WHEN vtranstype = 'PE Split (Cash Flow)'   THEN npercent END)  AS pe_split_cf,
    MAX(CASE WHEN vtranstype = 'Ecc. Occ. at Close'    THEN npercent END)  AS econ_occ_at_close
FROM
    txfinancial_IC
WHERE
    vCode LIKE 'P%'
GROUP BY
    vCode
ORDER BY
    vCode
