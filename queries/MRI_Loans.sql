-- NOTE: the Loan_Date join below returns ONE ROW PER DATE EVENT, so a loan with
-- both an Origination and a Maturity event appears twice with every other column
-- repeated. Consumers must not treat those rows as separate facilities — the app
-- collapses them to one row per LoanID in
-- flask_app/services/data_service.py::_collapse_loan_date_events, which also
-- resolves the maturity (dtMaturity is NULL on every live row, so it falls back to
-- the Maturity event's dtEvent, or origination + iLoanTerm when there is none).
-- Fixing it there rather than here is deliberate: a change to this query only
-- takes effect after a full MRI refresh, and the `loans` table is also populated
-- by CSV import, which bypasses this file entirely.
SELECT
    L.vCode,
	LD.LoanID,
	P.vPropertyName,
    L.mOrigLoanAmt,
    L.iAmortTerm,
	L.mNominalPenalty,
    L.iLoanTerm,
    L.vIntType,
    L.vIndex,
    L.nRate,
    L.vSpread,
    L.nFloor,
	L.vIntRatereset,
    L.nRequiredDCR,
    L.nReqDSR,
    L.nDY,
    L.nRequiredDY,
    L.nLTV,
    L.nRequiredLTV,
    L.vAmortAmt AS ExtensionOptions,
    L.dtMaturity,
    L.vNotes,
    L.vHedged,
    L.vHedgedStrat,
    LD.vDateType,
    LD.dtEvent
FROM 
    Loan L
LEFT JOIN 
    Loan_Date LD ON L.UID = LD.LoanID
LEFT JOIN
	Property P ON L.vCode = P.vCode
WHERE 
    L.delete_flag IS NULL and
	L.mOrigLoanAmt > 0
Order by L.vCode,LD.dtEvent;