select vaccount as vcode, vAccountType
from vCOA
where ISNUMERIC(vaccount)=1