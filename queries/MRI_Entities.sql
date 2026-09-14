-- MRI_Entities.sql
-- Server: PMX (BV6899900001)
-- Maps to: entities table
--
-- The ENTITY master -- every reporting entity in the accounting system, by
-- EntityID and name.
--
-- The app currently has no list of these. `deals` holds properties (P-codes,
-- and the hand-maintained InvestmentID beside them); `relationships` holds
-- entity pairs with ownership %, but only those that appear in a relationship.
-- Neither answers "which entities get a quarterly workpaper, and what is each
-- one called". PPIECH -- PPI Eastchase (TX) LLC, the subject of this package --
-- is a fund entity and is in neither list.
--
-- Needed for: the entity picker on any workpaper screen, resolving ENTITYID to
-- a readable name on the GL detail and trial balance, and -- the reason it is
-- worth pulling on its own -- checking that an ENTITYID appearing in the GL is
-- a known entity rather than a typo. A hand-typed id that matches nothing is
-- exactly how Village Square and Jefferson Centura sat missing from the Sold
-- Portfolio report (Sep 14 2026): the join found nothing and said nothing.
--
-- SELECT * because ENTITY's other columns (entity type, status, currency,
-- parent) decide what the picker can group and filter by, and it is a small
-- table. Look at what comes back before depending on a column.

SELECT *
FROM ENTITY
ORDER BY EntityID
