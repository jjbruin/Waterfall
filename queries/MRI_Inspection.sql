-- MRI_Inspection.sql
-- Construction draw inspection data for development deals.
-- Server: IM
-- Returns inspection records with drawn amounts per property.
-- Used to determine actual drawn balance on construction loans
-- (vs mOrigLoanAmt which is the full commitment amount).
--
-- Output table name: inspection
-- Key columns:
--   vCode        - Property code (P-series)
--   LoanID       - Links to MRI_Loans for loan identification
--   InspectionID - Unique inspection record
--   dtInspection - Date of draw inspection
--   mHardCost    - Hard cost amount drawn
--   mSoftCost    - Soft cost amount drawn
--   mTotalDraw   - Total draw amount (hard + soft)
--   mCumDrawn    - Cumulative amount drawn to date
--   mCommitment  - Total loan commitment
--   pctDrawn     - Percent of commitment drawn

SELECT
    IL.vCode,
    IL.LoanID,
    P.vPropertyName,
    IL.UID                          AS InspectionID,
    IL.dtInspection,
    IL.mHardCost,
    IL.mSoftCost,
    ISNULL(IL.mHardCost, 0)
        + ISNULL(IL.mSoftCost, 0)   AS mTotalDraw,
    IL.mRetainage,
    IL.mCumDisbursed                AS mCumDrawn,
    L.mOrigLoanAmt                  AS mCommitment,
    CASE
        WHEN L.mOrigLoanAmt > 0
        THEN IL.mCumDisbursed / L.mOrigLoanAmt * 100
        ELSE NULL
    END                             AS pctDrawn,
    IL.vStatus,
    IL.vNotes
FROM
    Inspection IL
    INNER JOIN Loan L ON IL.LoanID = L.UID AND L.delete_flag IS NULL
    LEFT JOIN Property P ON IL.vCode = P.vCode
WHERE
    IL.delete_flag IS NULL
    AND IL.vCode LIKE 'P%'
ORDER BY
    IL.vCode, IL.LoanID, IL.dtInspection
