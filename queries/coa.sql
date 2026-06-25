select vcode,vdescription,vtype,iNOI,vMisc,vAccountType
from vCOA
where ISNUMERIC(vcode)=1
and vcode not like 'M%'