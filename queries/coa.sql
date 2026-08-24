-- Chart of Accounts from IM vCOA view
-- Maps to: coa table (vAccount, vAccountType)
select Distinct
vcode,
vDescription,
vNotes,
vType,
VMisc,
vAccountType
from COA
where vcode <> 'M%'
order by vcode
