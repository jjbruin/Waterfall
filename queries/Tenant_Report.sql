-- Tenant Report (Detail) — source for Tenant_Report.csv
-- Parameter: @filterdt (DATETIME) — report as-of date; defaults to 50 years out if NULL

DECLARE @filterdt DATETIME
SET @filterdt = COALESCE(@filterdt, DATEADD(YEAR, 50, CURRENT_TIMESTAMP))

SELECT p.vcode
, p.vPropertyName
, inv.vPartnershipName
, inv.fOwnership
, pint.iInt
, occ.dtReported
, occ.iCommSqft
, occ.vtype2
, occ.vVendorCode
, occ.vName
, occ.dtLeasest
, occ.dtLeaseEnd
, occ.nSFLeased
, occ.mRent
, CASE WHEN occ.iVacated=1 THEN 'Yes' END AS iVacated
, CASE WHEN occ.iMonthToMonth=1 THEN 'Yes' END AS iMonthtoMonth
FROM property p
INNER JOIN txprop txp ON p.vcode = txp.vcode
    AND txp.delete_flag IS NULL
INNER JOIN (
    SELECT o.vcode
        , o.dtReported
        , o.iCommSqft
        , o.vtype2
        , ot.vVendorCode
        , v.vName
        , ot.dtLeasest
        , ot.dtLeaseEnd
        , ot.nSFLeased
        , ot.mRent
        , ot.iVacated
        , ot.iMonthToMonth
    FROM Occupancy o
    LEFT JOIN Occupancy_Tenants ot ON o.[uid] = ot.OccID
        AND ot.delete_flag IS NULL
    LEFT JOIN Vendor v ON ot.vVendorCode = v.vcode
        AND v.delete_flag IS NULL
    WHERE o.delete_flag IS NULL
        AND o.vType = 'Retail'
        AND o.dtReported = (
            SELECT MAX(dtReported)
            FROM Occupancy
            WHERE delete_flag IS NULL
                AND vcode = o.vcode
                AND vtype = 'Retail'
                AND dtReported <= @filterdt
            GROUP BY vcode
        )
) occ ON occ.vCode = p.vCode
LEFT JOIN (
    SELECT inv.vcode
        , inv.fOwnership
        , f.vPartnershipName
    FROM Investors inv
    INNER JOIN fund f ON inv.vInvestor = f.vcode
        AND f.delete_flag IS NULL
    WHERE inv.delete_flag IS NULL
        AND inv.iyear = (
            SELECT MAX(iyear)
            FROM Investors
            WHERE vcode = inv.vcode
                AND delete_flag IS NULL
                AND iyear <= @filterdt
            GROUP BY vCode
        )
) inv ON inv.vcode = p.vcode
LEFT JOIN (
    SELECT pr.vcode
        , pr.iInt
    FROM propint pr
    WHERE pr.delete_flag IS NULL
        AND pr.vIntType = 'Rentable SF'
) pint ON pint.vCode = p.vcode
LEFT JOIN (
    SELECT ed.vcode
        , ed.vEvent
    FROM Event_Dates ed
    WHERE ed.delete_flag IS NULL
        AND ed.vEventType = 'Status Change'
        AND ISNULL(ed.vEvent, '') NOT IN ('Sold', 'Preclosing')
        AND ed.dtEvent = (
            SELECT MAX(ed1.dtEvent)
            FROM Event_Dates ed1
            WHERE ed1.delete_flag IS NULL
                AND ed1.vCode = ed.vCode
                AND ed1.vEventType = 'Status Change'
                AND ISNULL(ed1.vEvent, '') NOT IN ('Sold', 'Preclosing')
                AND ed1.dtEvent <= @filterdt
            GROUP BY ed1.vCode
        )
) ed ON ed.vCode = p.vcode
WHERE p.delete_flag IS NULL
    AND COALESCE(ed.vEvent, txp.vstatus, '') NOT IN ('Sold', 'Preclosing')