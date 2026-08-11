-- MRI_Inspection.sql
-- Construction draw inspection data for development deals.
-- Server: IM
-- Returns inspection records with draw amounts per property.
--
-- Output table name: inspection
-- Key columns:
--   vCode        - Property code (P-series)
--   InspectionID - Unique inspection record (UID)
--   dtInspect    - Date of draw inspection
--   mHardCosts   - Hard cost amount drawn
--   mSoftCosts   - Soft cost amount drawn
--   mTotalDraw   - Total draw amount (hard + soft)
--   mAmtToDate   - Cumulative amount to date
--   nPctComplete - Percent complete

SELECT
    I.vCode,
    P.vPropertyName,
    I.UID                           AS InspectionID,
    I.vSiteVisitNo,
    I.dtSiteVisit,
    I.dtInspect,
    I.nPctComplete,
    I.mHardCosts,
    I.mSoftCosts,
    ISNULL(I.mHardCosts, 0)
        + ISNULL(I.mSoftCosts, 0)   AS mTotalDraw,
    I.mContingency,
    I.mRetainage,
    I.mAmtToDate,
    I.mCertAmt,
    I.mChgOrders,
    I.fRequestNo,
    I.vNotes
FROM
    Inspection I
    LEFT JOIN Property P ON I.vCode = P.vCode
WHERE
    I.delete_flag IS NULL
    AND I.vCode LIKE 'P%'
ORDER BY
    I.vCode, I.dtInspect
