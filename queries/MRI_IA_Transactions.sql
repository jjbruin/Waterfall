-- MRI_IA_Transactions.sql
-- Server: PMX (BV6899900001)
-- Maps to: ia_transactions table
--
-- The investor-activity feed AS ACCOUNTING SEES IT -- a faithful copy of the
-- Spreadsheet Server "IA QUERY" behind the workpaper's Investor Detail,
-- Investment Detail and IA Query RF tabs.
--
-- WHY THIS EXISTS ALONGSIDE accounting_feed.sql. `accounting_feed` is the APP's
-- view of the same three MRI tables: it re-signs the amounts to the app's
-- cashflow convention, derives Cum_Amt / Capital / ROE_Income / Partner, and
-- every waterfall, XIRR and ROE path in the app reads it. It is not wrong, but
-- it is not the ledger either, and three differences make it unable to
-- reproduce a workpaper:
--
--   1. IT DROPS MOST NON-CASH ACTIVITY. accounting_feed filters IA_noncashtrans
--      to `Typename LIKE 'Transfer of Ownership%'`, which discards everything
--      else with MajorType='Other'. In this one package that is 40 of the 82
--      Investor Detail rows -- 34 "Income/Loss before Fees" and 6 "Unrealized
--      Gain/Loss" -- INCLUDING the whole of the rollforward's current-period
--      movement (-11,745.08 across three members). Nearly half the tab, and the
--      rollforward cannot be reproduced from accounting_feed at all. This query
--      filters nothing.
--   2. IT KEEPS ONLY EffectiveDate. Accounting's parameter is the TRANSACTION
--      date (contributiondate / distributiondate / transactiondate), so a
--      quarter-end cut taken on the effective date is a cut on a different
--      column than the workpaper's. NOTE: the two agree on all 99 rows of the
--      06.30.2026 PPIECH package, so how often they diverge portfolio-wide is
--      UNMEASURED -- it cannot be measured from the app today, because
--      accounting_feed never loaded the transaction date. Both are carried here
--      so the question becomes answerable:
--        select count(*) from ia_transactions
--        where TransactionDate <> EffectiveDate
--   3. IT CARRIES NO NAMES. A workpaper is read by people; InvestorID ERIBPI
--      has to print as "PPI Ederville Road Investors". ENTITY and IA_Investor
--      are joined here for both sides.
--
-- SIGNS ARE ACCOUNTING'S, NOT THE APP'S. Distributions are negated
-- (`Amount * -1`) exactly as Spreadsheet Server does, so a column sums to the
-- entity's capital movement. The app's opposite convention (negative =
-- contribution) is applied in accounting_feed and is NOT applied here. Do not
-- feed this table into a waterfall without re-signing it.
--
-- THE ENTITY IS ON BOTH SIDES. The workpaper runs this query twice for one
-- entity: InvestmentID=PPIECH, InvestorID=* gives who invested INTO the fund
-- (ERIBPI, TGA23, INVECH); InvestmentID=*, InvestorID=PPIECH gives what the
-- fund invested into (EASTCH). Same rows, different filter. No scope clause
-- here -- the app filters the loaded table both ways.
--
-- FOLLOW-UP WORTH TAKING: accounting_feed re-queries MRI for the same three
-- tables this one reads, so the two can drift on a schema change. Deriving
-- accounting_feed's columns FROM ia_transactions in Python would leave one
-- pull and one definition. Not done here -- it changes the feed every deal's
-- numbers come from, and that is its own piece of work with its own checks.

SELECT
    RTRIM(LTRIM(C.InvestmentID))    AS InvestmentID,
    RTRIM(LTRIM(E.Name))            AS InvestmentName,
    RTRIM(LTRIM(I.InvestorID))      AS InvestorID,
    RTRIM(LTRIM(I.Name))            AS InvestorName,
    C.ContributionDate              AS TransactionDate,
    C.EffectiveDate                 AS EffectiveDate,
    S.MajorType                     AS MajorType,
    S.Typename                      AS Typename,
    S.SubtypeUID                    AS SubtypeUID,
    C.Amount                        AS Amount
FROM IA_Contribution C
INNER JOIN IA_Subtype  S ON C.ContributionType = S.SubtypeUID
INNER JOIN IA_Investor I ON C.InvestorID       = I.InvestorID
INNER JOIN ENTITY      E ON C.InvestmentID     = E.EntityID
                        AND S.MajorType        = 'Contribution'

UNION ALL

SELECT
    RTRIM(LTRIM(D.InvestmentID)),
    RTRIM(LTRIM(E.Name)),
    RTRIM(LTRIM(I.InvestorID)),
    RTRIM(LTRIM(I.Name)),
    D.DistributionDate,
    D.EffectiveDate,
    S.MajorType,
    S.Typename,
    S.SubtypeUID,
    D.Amount * -1
FROM IA_Distribution D
INNER JOIN IA_Subtype  S ON D.DistributionTypeID = S.SubtypeUID
INNER JOIN IA_Investor I ON D.InvestorID         = I.InvestorID
INNER JOIN ENTITY      E ON D.InvestmentID       = E.EntityID
                        AND S.MajorType          = 'Distribution'

UNION ALL

SELECT
    RTRIM(LTRIM(N.InvestmentID)),
    RTRIM(LTRIM(E.Name)),
    RTRIM(LTRIM(I.InvestorID)),
    RTRIM(LTRIM(I.Name)),
    N.TransactionDate,
    N.EffectiveDate,
    S.MajorType,
    S.Typename,
    S.SubtypeUID,
    N.Amount
FROM IA_NonCashTrans N
INNER JOIN IA_Subtype  S ON N.NonCashTransTypeID = S.SubtypeUID
INNER JOIN IA_Investor I ON N.InvestorID         = I.InvestorID
INNER JOIN ENTITY      E ON N.InvestmentID       = E.EntityID
                        AND S.MajorType          = 'Other'
;

-- UNION ALL, not SS's UNION: two investors can contribute the same amount to
-- the same investment on the same day under the same subtype, and dedupe would
-- drop the second. See the same note in MRI_GL_Detail.sql.
--
-- MajorType is asserted in the ENTITY join (`AND S.MajorType = 'Contribution'`)
-- rather than in a WHERE, copying Spreadsheet Server exactly. On an INNER JOIN
-- the two are equivalent; it is written this way so a line-by-line diff against
-- accounting's query stays clean.
