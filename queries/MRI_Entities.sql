-- MRI_Entities.sql
-- Server: PMX (BV6899900001)
-- Maps to: entities table
--
-- The ENTITY master -- every reporting entity in the accounting system, by
-- EntityID and name.
--
-- The app has no other list of these. `deals` holds properties (P-codes, plus
-- the hand-maintained InvestmentID beside them); `relationships` holds entity
-- pairs with ownership %, but only those appearing in a relationship. Neither
-- answers "which entities exist, and what is each one called". PPIECH -- PPI
-- Eastchase (TX) LLC -- is a fund entity and is in neither.
--
-- ENTITYID IS TRIMMED, AND THAT IS THE POINT. This query was first written as
-- SELECT *, which returned ENTITYID space-padded to the column width. Every
-- other MRI query here trims, so the join from entity_groups ('AMB6') to
-- entities ('AMB6  ') found nothing and AMB6 -- PSC Ambassadors Fund TGA VI
-- LLC, one of only two entities accounting has tagged REP -- came back with a
-- blank name. Six-character ids matched by luck, because MRI pads to 6.
-- Nothing said anything was wrong; the name was simply empty.
--
-- WHY A COLUMN LIST AND NOT SELECT *. Trimming the key requires naming it, and
-- ENTITY is 147 columns wide -- most of them AP/AR/FX/tax GL wiring that no
-- part of this app will read, and unreadable in Data Explorer. The set below is
-- identity, location, classification, status, dates and the accounting config
-- worth seeing. Anything dropped is one ad-hoc query against MRI away, and
-- widening this list is a one-line edit plus a refresh -- nothing is lost
-- permanently. Add columns here when something actually needs them.

SELECT
    RTRIM(LTRIM(ENTITYID))      AS ENTITYID,
    RTRIM(LTRIM(NAME))          AS NAME,
    RTRIM(LTRIM(ADDR1))         AS ADDR1,
    RTRIM(LTRIM(CITY))          AS CITY,
    RTRIM(LTRIM(STATE))         AS STATE,
    RTRIM(LTRIM(ZIPCODE))       AS ZIPCODE,
    RTRIM(LTRIM(COUNTRY))       AS COUNTRY,
    RTRIM(LTRIM(ENTTYPE))       AS ENTTYPE,
    RTRIM(LTRIM(IAENTTYPE))     AS IAENTTYPE,
    RTRIM(LTRIM(ACTIVE))        AS ACTIVE,
    RTRIM(LTRIM(PROPTYPE))      AS PROPTYPE,
    RTRIM(LTRIM(PROPSUBTYPE))   AS PROPSUBTYPE,
    RTRIM(LTRIM(CLASSID))       AS CLASSID,
    RTRIM(LTRIM(INVTYPE))       AS INVTYPE,
    RTRIM(LTRIM(LIFECODE))      AS LIFECODE,
    RTRIM(LTRIM(INVESTFLAG))    AS INVESTFLAG,
    RTRIM(LTRIM(OWNERID))       AS OWNERID,
    ACQUIRED                    AS ACQUIRED,
    DISPOSED                    AS DISPOSED,
    RTRIM(LTRIM(CURRCODE))      AS CURRCODE,
    RTRIM(LTRIM(LEDGCODE))      AS LEDGCODE,
    RTRIM(LTRIM(BASIS))         AS BASIS,
    YEAREND                     AS YEAREND,
    CURPED                      AS CURPED,
    UNITS                       AS UNITS,
    FEET                        AS FEET
FROM ENTITY
ORDER BY ENTITYID
