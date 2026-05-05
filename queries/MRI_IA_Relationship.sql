SELECT  
	R.InvestmentID,
    R.InvestorID,
    R.OwnershipPct,
    E.Name,
    R.StartDate,
    R.EndDate
FROM IA_RELATIONSHIP R
Left Join entity E ON R.InvestmentID = E.EntityID
