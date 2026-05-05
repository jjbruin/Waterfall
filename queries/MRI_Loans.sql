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
    L.dtMaturity,
    L.vNotes,
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