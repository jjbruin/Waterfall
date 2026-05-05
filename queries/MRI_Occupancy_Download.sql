WITH RankedOccupancy AS (
    SELECT 
        vCode,
        UID,
        dtReported,
        iResidentialUnits,
        iOccupiedUnits,
        iVacantUnits,
        iCommSqft,
        vType,
        delete_flag,
        ROW_NUMBER() OVER (PARTITION BY vCode ORDER BY dtReported DESC) AS rn
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
)

SELECT 
    ro.vCode,
    ro.UID,
    ro.dtReported,
    CONCAT(YEAR(ro.dtReported), '-Q', DATEPART(QUARTER, ro.dtReported)) AS Qtr,
    ro.iResidentialUnits,
    ro.iOccupiedUnits,
    ro.iVacantUnits,
    ro.iCommSqft,
    ro.vType,
    ro.delete_flag,
    CASE 
        WHEN ro.iResidentialUnits > 0 THEN FORMAT(ROUND((ro.iOccupiedUnits * 100.0 / ro.iResidentialUnits), 1), 'N1')
        ELSE '0.0'
    END AS [ResidentialOcc%],
    l.TotalNSFLeasedNonVacant,
    t.TotalNSFLeased,
    CASE 
        WHEN t.TotalNSFLeased > 0 THEN ROUND((l.TotalNSFLeasedNonVacant * 100.0 / t.TotalNSFLeased), 1)
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
                WHEN t.TotalNSFLeased > 0 THEN ROUND((l.TotalNSFLeasedNonVacant * 100.0 / t.TotalNSFLeased), 1)
                ELSE 0
            END
        THEN 
            CASE 
                WHEN ro.iResidentialUnits > 0 THEN ROUND((ro.iOccupiedUnits * 100.0 / ro.iResidentialUnits), 1)
                ELSE 0
            END
        ELSE 
            CASE 
                WHEN t.TotalNSFLeased > 0 THEN ROUND((l.TotalNSFLeasedNonVacant * 100.0 / t.TotalNSFLeased), 1)
                ELSE 0
            END
    END AS [Occ%]
FROM 
    RankedOccupancy ro
LEFT JOIN 
    LeasedNSFNonVacant l ON ro.UID = l.OccID
LEFT JOIN 
    TotalNSFLeased t ON ro.UID = t.OccID
ORDER BY 
    ro.vCode, ro.dtReported;