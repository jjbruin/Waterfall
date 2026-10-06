-- Prop_Info_DealTerms.sql
-- PE deal terms from txfinancial_IC (Investment Checklist).
-- Server: IM
-- One row per vCode with pivoted deal term fields.
-- These are authoritative MRI-entered values for coupon, IRR hurdle, PE split, etc.
--
-- uw_irr and proj_yr1_coc are UNDATED underwriting figures: the row with the
-- LATEST dtEffective wins (UID breaks a tie), not MAX(npercent), and nothing is
-- filtered against an as-of date. Both are stored as fractions (0.11 = 11%),
-- like pe_coupon and irr_lookback. The exact vtranstype strings are
-- 'U/W IRR' and 'Projected Yr 1 CoC Returns'; 'UW IRR' and 'Projected Yr 1 CoC'
-- are different (older) types and are deliberately NOT read.

WITH latest AS (
    SELECT
        vCode,
        vtranstype,
        npercent,
        ROW_NUMBER() OVER (
            PARTITION BY vCode, vtranstype
            ORDER BY dtEffective DESC, UID DESC
        ) AS rn
    FROM
        txfinancial_IC
    WHERE
        vCode LIKE 'P%'
        AND vtranstype IN ('U/W IRR', 'Projected Yr 1 CoC Returns')
),
uw AS (
    SELECT
        vCode,
        MAX(CASE WHEN vtranstype = 'U/W IRR'                    THEN npercent END) AS uw_irr,
        MAX(CASE WHEN vtranstype = 'Projected Yr 1 CoC Returns' THEN npercent END) AS proj_yr1_coc
    FROM
        latest
    WHERE
        rn = 1
    GROUP BY
        vCode
)
SELECT
    d.vcode,
    d.pe_coupon,
    d.irr_lookback,
    d.pe_split_capital,
    d.pe_split_cf,
    d.econ_occ_at_close,
    uw.uw_irr,
    uw.proj_yr1_coc
FROM (
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
) AS d
LEFT JOIN uw
    ON uw.vCode = d.vcode
ORDER BY
    d.vcode
