-- MRI_Entity_GroupID.sql
-- Server: PMX (BV6899900001)
-- Maps to: entity_groups table
--
-- Entity group membership. Accounting tags an entity with ENTGRPID = 'REP'
-- when it requires a quarterly workpaper package, so this table is what
-- answers "which entities do we produce packages for" -- a list the app
-- otherwise has no way to know, and one that accounting maintains in MRI
-- rather than telling us.
--
-- NOT FILTERED TO 'REP'. An entity can belong to several groups, and pulling
-- only REP would make "this entity is not tagged REP" indistinguishable from
-- "the refresh did not load it". The app filters `where ENTGRPID = 'REP'`;
-- the other groups cost four characters a row and show what else exists:
--   select ENTGRPID, count(*) from entity_groups group by ENTGRPID
--
-- TRIMMED. MRI's char columns come back space-padded -- the accounting feed
-- returns InvestorID as 'PPI10        ' -- and an untrimmed ENTITYID here
-- would not match the trimmed one in gl_detail, so the join would silently
-- find nothing. Both columns are trimmed for that reason.

SELECT
    RTRIM(LTRIM(ENTITYID))  AS ENTITYID,
    RTRIM(LTRIM(ENTGRPID))  AS ENTGRPID
FROM ENTITYGRPD
ORDER BY ENTITYID, ENTGRPID
