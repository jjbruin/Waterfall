-- Prop_Info_Core.sql
-- Core property metadata for the deals (investment_map) table.
-- Server: IM
-- One row per vCode (P-series property).
-- Columns align with existing deals table column names.
--
-- NOTE: InvestmentID, Portfolio_Name, Sale_Status are NOT available in MRI IM.
-- They are preserved during upsert import (see mri_service.py).

WITH PropMeta AS (
    SELECT
        t.vCode,
        t.vPropType,
        t.vStatus,
        t.vYearBuiltRange,
        ed.acq_closing_act,
        ed.uw_exit
    FROM TXProp t
    LEFT JOIN (
        SELECT
            vcode,
            MAX(CASE WHEN veventtype = 'Acquisition' AND vevent = 'Closing' AND vdatetype = 'Actual' THEN dtevent END) AS acq_closing_act,
            MAX(CASE WHEN veventtype = 'Asset Management' AND vevent = 'U/W Exit' THEN dtevent END) AS uw_exit
        FROM event_dates
        WHERE delete_flag IS NULL
        GROUP BY vcode
    ) ed ON t.vCode = ed.vcode
    WHERE t.delete_flag IS NULL
),
PropIntAgg AS (
    SELECT
        pint.vCode,
        MAX(CASE WHEN pint.vIntType = 'Total Units' THEN pint.iInt END) AS totunits,
        MAX(CASE WHEN pint.vIntType = 'Rentable SF' THEN pint.iInt END) AS rentablesf,
        -- Two labels carry the purchase price. Records created from mid-2025
        -- store it as 'Purchase Price'; everything before that used
        -- 'Original Purchase Price'. Reading only the latter returned NULL for
        -- the newer deals, and _upsert_deals overwrites on non-null only
        -- (mri_service.py), so a refresh could never repair it.
        --
        -- COALESCE, not IN (...), so the precedence is explicit. A single
        -- MAX over both labels would return the LARGER NUMBER rather than the
        -- preferred label, silently changing any deal that carries both.
        -- This form cannot: the first argument is exactly the previous
        -- expression, so the second is reached only where the result was
        -- already NULL.
        COALESCE(
            MAX(CASE WHEN pint.vIntType = 'Original Purchase Price' THEN pint.iInt END),
            MAX(CASE WHEN pint.vIntType = 'Purchase Price' THEN pint.iInt END)
        ) AS original_purchase_price
    FROM PropInt pint
    WHERE pint.delete_flag IS NULL
    GROUP BY pint.vCode
),
PropPartiesAgg AS (
    SELECT
        pp.vcode,
        MAX(CASE WHEN pp.vtype = 'Sponsor' THEN ent.vName END) AS sponsor
    FROM PropParties pp
    INNER JOIN (
        SELECT vCode, vname FROM Vendor WHERE delete_flag IS NULL
        UNION ALL
        SELECT vCode, vname FROM Owner WHERE delete_flag IS NULL
    ) ent ON pp.vparty = ent.vcode
    WHERE CURRENT_TIMESTAMP BETWEEN ISNULL(pp.dtStart, '1/1/1950') AND ISNULL(pp.dtEnd, '1/1/2049')
        AND pp.delete_flag IS NULL
    GROUP BY pp.vcode
)

SELECT
    p.vCode                         AS vcode,
    p.vPropertyName                 AS Investment_Name,
    p.vCity                         AS City,
    p.vState                        AS State,
    meta.vPropType                  AS Asset_Type,
    parties.sponsor                 AS Operating_Partner,
    pint.totunits                   AS Total_Units,
    pint.rentablesf                 AS Size_Sqf,
    meta.acq_closing_act            AS Acquisition_Date,
    meta.vStatus                    AS Lifecycle,
    meta.vYearBuiltRange            AS Year_Built,
    pint.original_purchase_price    AS Acquisition_Price,
    meta.uw_exit                    AS Anticipated_Exit
FROM
    Property p
    LEFT JOIN PropMeta meta ON p.vCode = meta.vCode
    LEFT JOIN PropIntAgg pint ON p.vCode = pint.vCode
    LEFT JOIN PropPartiesAgg parties ON p.vCode = parties.vcode
WHERE
    p.delete_flag IS NULL
    AND p.vCode LIKE 'P%'
ORDER BY p.vCode
