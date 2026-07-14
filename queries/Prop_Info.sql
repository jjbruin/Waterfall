WITH RankedOccupancy AS (
    SELECT
        UID,
        vCode,
        dtReported,
        iResidentialUnits,
        iOccupiedUnits,
        iVacantUnits,
        iCommSqft,
        vType,
        delete_flag,
        ROW_NUMBER() OVER (PARTITION BY vCode ORDER BY dtReported DESC) AS rnCurrent
    FROM
        occupancy
    WHERE
        delete_flag IS NULL
        AND vCode LIKE 'P%'
),
TotalNSFLeased AS (
    SELECT
        OccID,
        SUM(nSFLeased) AS TotalNSFLeased
    FROM
        Occupancy_Tenants
    GROUP BY
        OccID
),
LeasedNSFNonVacant AS (
    SELECT
        OccID,
        SUM(nSFLeased) AS TotalNSFLeasedNonVacant
    FROM
        Occupancy_Tenants
    WHERE
        vVendorCode <> 'vacant'
    GROUP BY
        OccID
),
OccupancyCalculation AS (
    SELECT
        ro.vCode,
        CASE
            WHEN ro.iResidentialUnits > 0 THEN ROUND((ro.iOccupiedUnits * 100.0 / ro.iResidentialUnits), 1)
            ELSE 0
        END AS ResidentialOcc,
        lsnv.TotalNSFLeasedNonVacant,
        tnsf.TotalNSFLeased,
        CASE
            WHEN tnsf.TotalNSFLeased > 0 THEN ROUND((lsnv.TotalNSFLeasedNonVacant * 100.0 / tnsf.TotalNSFLeased), 1)
            ELSE NULL
        END AS OccupancyPercent,
        CASE
            WHEN
                CASE
                    WHEN ro.iResidentialUnits > 0 THEN ROUND((ro.iOccupiedUnits * 100.0 / ro.iResidentialUnits), 1)
                    ELSE 0
                END
                >
                CASE
                    WHEN tnsf.TotalNSFLeased > 0 THEN ROUND((lsnv.TotalNSFLeasedNonVacant * 100.0 / tnsf.TotalNSFLeased), 1)
                    ELSE 0
                END
            THEN
                CASE
                    WHEN ro.iResidentialUnits > 0 THEN ROUND((ro.iOccupiedUnits * 100.0 / ro.iResidentialUnits), 1)
                    ELSE 0
                END
            ELSE
                CASE
                    WHEN tnsf.TotalNSFLeased > 0 THEN ROUND((lsnv.TotalNSFLeasedNonVacant * 100.0 / tnsf.TotalNSFLeased), 1)
                    ELSE 0
                END
        END AS OccPercent
    FROM
        RankedOccupancy ro
    LEFT JOIN LeasedNSFNonVacant lsnv ON ro.UID = lsnv.OccID
    LEFT JOIN TotalNSFLeased tnsf ON ro.UID = tnsf.OccID
    WHERE
        ro.rnCurrent = 1
),
LatestEntryPerInvestment AS (
    SELECT
        sje.vCode,
        MAX(sje.dtEntry) AS MostRecentDate
    FROM
        vSJE sje
    WHERE
        TRY_CAST(sje.vAccount AS INT) IN (2150, 2152)
        AND sje.vSource = 'Interim BS'
    GROUP BY
        sje.vCode
),
SummedLoanBalances AS (
    SELECT
        sje.vCode,
        le.MostRecentDate,
        SUM(sje.mAmount) AS TotalLoanBalance
    FROM
        vSJE sje
    INNER JOIN LatestEntryPerInvestment le ON sje.vCode = le.vCode AND sje.dtEntry = le.MostRecentDate
    WHERE
        TRY_CAST(sje.vAccount AS INT) IN (2150, 2152)
        AND sje.vSource = 'Interim BS'
    GROUP BY
        sje.vCode, le.MostRecentDate
),
AtCloseNOI AS (
    SELECT
        sje.vCode,
        SUM(CASE WHEN sje.vAccount LIKE '4%' THEN sje.mAmount ELSE 0 END) AS AtCloseRevenue,
        SUM(CASE WHEN sje.vAccount LIKE '5%' OR sje.vAccount LIKE '7%' THEN sje.mAmount ELSE 0 END) AS AtCloseExpenses
    FROM
        vstaging_journal_entry sje
    WHERE
        sje.vSource = 'Projected IS'
        AND sje.dtEntry = '12/31/2015'
    GROUP BY
        sje.vCode
),
PropMeta AS (
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
        MAX(CASE WHEN pint.vIntType = 'Original Purchase Price' THEN pint.iInt END) AS original_purchase_price
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

-- Main Query starts here
SELECT
    P.vCode AS 'Investment Code',
    p.vPropertyName AS 'Investment Name',
    CONCAT(p.vCity, ', ', p.vState) AS 'City/Town/District',
    meta.vPropType AS 'Asset Type',
    parties.sponsor AS 'Operating Partner',
    pint.totunits AS 'Total Units',
    pint.rentablesf AS 'Size (SQF)',
    meta.acq_closing_act AS 'Acquisition Date',
    meta.vStatus AS 'Lifecycle',
    meta.vYearBuiltRange AS 'Year Built',
    pint.original_purchase_price AS 'Original Purchase Price',
    meta.uw_exit AS 'Anticipated Exit',
    l.vIntType AS 'Loan VIntType',
    l.vindex AS 'Loan Index',  
    l.vspread AS 'Loan Spread',
    l.nRate AS 'Loan Rate',
    MAX(ld.dtEvent) AS 'Maturity Date',
    COALESCE(MAX(pe_coupon.npercent), 0) AS 'PE Coupon',  
    COALESCE(MAX(irr_lookback.npercent), 0) AS 'IRR Lookback',
    COALESCE(MAX(pe_split.npercent), 0) AS 'PE Split (Capital Event)',
    COALESCE(slb.TotalLoanBalance, 0) AS 'Loan Balance',
    COALESCE(MAX(occ_at_close.npercent), 0) AS 'Ecc. Occ. at Close',
    COALESCE(acn.AtCloseRevenue, 0) AS 'At-Close Revenue',
    COALESCE(acn.AtCloseExpenses, 0) AS 'At-Close Expenses',
    COALESCE(SUM(ic.mAmount), 0) AS 'Initial Equity Commitment',
    l.vAmortAmt AS 'Loan Extension'
FROM
    Property p
    LEFT JOIN PropMeta meta ON p.vCode = meta.vCode
    LEFT JOIN PropIntAgg pint ON p.vCode = pint.vCode
    LEFT JOIN PropPartiesAgg parties ON p.vCode = parties.vcode
    LEFT JOIN txfinancial_IC pe_coupon ON p.vCode = pe_coupon.vCode AND pe_coupon.vtranstype = 'PE Coupon'
    LEFT JOIN txfinancial_IC irr_lookback ON p.vCode = irr_lookback.vCode AND irr_lookback.vtranstype = 'IRR Lookback'
    LEFT JOIN txfinancial_IC pe_split ON p.vCode = pe_split.vCode AND pe_split.vtranstype = 'PE Split (Capital Event)'
    LEFT JOIN txfinancial_IC occ_at_close ON p.vCode = occ_at_close.vCode AND occ_at_close.vtranstype = 'Ecc. Occ. at Close'
    LEFT JOIN Loan l ON p.vCode = l.vCode AND l.delete_flag IS NULL
    LEFT JOIN Loan_Date ld ON l.UID = ld.LoanID AND ld.vDateType = 'Maturity'
    LEFT JOIN SummedLoanBalances slb ON p.vCode = slb.vCode
    LEFT JOIN AtCloseNOI acn ON p.vCode = acn.vCode
    LEFT JOIN (
        SELECT
            vCode,
            SUM(mAmount) AS mAmount
        FROM
            Commitment
        WHERE
            vCode LIKE 'P%' AND vownercode = 'O0000002'
            AND delete_flag IS NULL
        GROUP BY vCode
    ) ic ON p.vCode = ic.vCode
WHERE
    p.delete_flag IS NULL
GROUP BY
    P.vCode, p.vPropertyName, p.vCity, p.vState, meta.vPropType, parties.sponsor, pint.totunits, pint.rentablesf,
    meta.acq_closing_act, meta.vStatus, meta.vYearBuiltRange, pint.original_purchase_price, meta.uw_exit,
    l.vIntType, l.vindex, l.vspread, l.nRate, ld.dtEvent, slb.TotalLoanBalance, occ_at_close.npercent,
    acn.AtCloseRevenue, acn.AtCloseExpenses, ic.mAmount, l.vAmortAmt;

