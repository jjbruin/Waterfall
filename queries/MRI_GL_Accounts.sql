-- MRI_GL_Accounts.sql
-- Server: PMX (BV6899900001)
-- Maps to: gl_accounts table
--
-- The GL chart of accounts (GACC) for the entity ledger.
--
-- THIS IS NOT THE `coa` TABLE. `coa` comes from the IM server's COA view and
-- holds the PROPERTY chart -- four-digit vAccounts (4010 Rental Income, 5090
-- Real Estate Taxes) that every NOI / FAD / DSCR path in the app reads. GACC
-- is the ENTITY ledger's own chart, keyed MR10001000 / MR20000003 style, and
-- the two do not map one-to-one. Keeping them in separate tables is deliberate;
-- a join between them needs a mapping somebody has to write and own.
--
-- MRI_GL_Detail already carries ACCTNAME per row from its GACC join, so this
-- table is not needed to read the ledger. It is here so the app can render an
-- account picker, group a trial balance, and show accounts that carry no
-- activity in the window -- an account absent from the detail is invisible
-- otherwise, and on a trial balance a missing account and a zero account are
-- not the same statement.
--
-- SELECT * rather than a column list: GACC's shape drives what grouping and
-- statement-tagging the app can offer later (account type, normal balance,
-- roll-up parent, active flag), and it is a small table. Check what came back
-- before building anything on a particular column.

SELECT *
FROM GACC
ORDER BY ACCTNUM
