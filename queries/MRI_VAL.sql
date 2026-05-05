select 
	Val.vCode,
	P.vPropertyName,
	Val.dtValuation,
	Val.vMethod,
	Val.mAnnualNOI,
	Val.fCapRate,
	Val.nTermCapRate,
	Val.nDiscountRateForEquityInterest,
	Val.mIncomeCapConcludedValue,
	Val.mDebtValue,
	Val.mEquityValue,
	Val.mMezzanineValue,
	Val.nCostSaleRate

from Valuation Val
	
INNER JOIN property p ON p.vCode = Val.vCode

where val.delete_flag is null
and p.delete_flag is null