-- Prop_Info_AtClose.sql
-- At-close underwriting NOI from Projected IS (due diligence audit at closing).
-- Server: IM
-- Dynamic: uses the EARLIEST December 31 date per deal (not hardcoded).
-- One row per vCode.
--
-- ACCOUNT 7083 IS PART OF THE EXPENSE BUCKET -- DO NOT "TIDY" IT OUT.
--
-- 7083 is OPERATING RESERVE RELEASE: a credit (negative mAmount) that funds the
-- gross 5xxx operating costs booked in the same period, so summing it into the
-- expense bucket nets the two to the real cost borne by the property. Without
-- it the '5%' prefix captured the gross costs while the offset -- which is a
-- 7xxx account and so invisible to both the '4%' and '5%' buckets -- was
-- silently dropped. 30 Bearfoot (P0000001) at 2020-12-31 is the case: three
-- cost rows totalling 60,100.00 against 7083 of -60,100.00, printing $0.1M of
-- expenses and a -$0.1M NOI on a deal whose at-close rows sum to exactly zero.
--
-- NAMED ACCOUNT, NOT A '7%' PREFIX RULE, and deliberately only this one.
-- 7083 exists on exactly one deal in the portfolio. The other 7xxx accounts
-- that land at an at-close date must STAY excluded: 7073/7074 are capital
-- proceeds (netting them would add 22.2M to Jefferson Addison Heights alone),
-- 7071/7072 are distributions, 7050 is capex, 7075 is reserves for
-- replacement (below NOI by convention, 61 deals), and 7080 offsets 7010 debt
-- service rather than operating cost. Adding any of them here is a bug.
--
-- Mirrored in one_pager.py AT_CLOSE_RESERVE_RELEASE_ACCTS, which does the same
-- netting on the fallback path. Change both or the two disagree.

WITH EarliestProjectedDec AS (
    SELECT
        vCode,
        MIN(dtEntry) AS at_close_date
    FROM
        vstaging_journal_entry
    WHERE
        vSource = 'Projected IS'
        AND MONTH(dtEntry) = 12
        AND DAY(dtEntry) = 31
        AND vCode LIKE 'P%'
    GROUP BY
        vCode
)
SELECT
    sje.vCode                                                                       AS vcode,
    epd.at_close_date,
    SUM(CASE WHEN sje.vAccount LIKE '4%' THEN sje.mAmount ELSE 0 END)              AS at_close_revenue,
    -- 5xxx gross operating cost NET of the 7083 operating reserve release.
    SUM(CASE WHEN sje.vAccount LIKE '5%'
                  OR TRY_CAST(sje.vAccount AS INT) = 7083
             THEN sje.mAmount ELSE 0 END)                                           AS at_close_expenses,
    SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) IN (5190, 7030)
             THEN sje.mAmount ELSE 0 END)                                           AS at_close_interest,
    SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) = 7060
             THEN sje.mAmount ELSE 0 END)                                           AS at_close_principal,
    -- NOI = revenue + expenses, so it carries the same 7083 netting.
    SUM(CASE WHEN sje.vAccount LIKE '4%' THEN sje.mAmount ELSE 0 END)
      + SUM(CASE WHEN sje.vAccount LIKE '5%'
                      OR TRY_CAST(sje.vAccount AS INT) = 7083
                 THEN sje.mAmount ELSE 0 END)                                       AS at_close_noi,
    CASE
        WHEN ABS(SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) IN (5190, 7030) THEN sje.mAmount ELSE 0 END)
               + SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) = 7060 THEN sje.mAmount ELSE 0 END)) > 0
        THEN (SUM(CASE WHEN sje.vAccount LIKE '4%' THEN sje.mAmount ELSE 0 END)
            + SUM(CASE WHEN sje.vAccount LIKE '5%'
                            OR TRY_CAST(sje.vAccount AS INT) = 7083
                       THEN sje.mAmount ELSE 0 END))
            / ABS(SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) IN (5190, 7030) THEN sje.mAmount ELSE 0 END)
                + SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) = 7060 THEN sje.mAmount ELSE 0 END))
        ELSE NULL
    END                                                                             AS at_close_dscr
FROM
    vstaging_journal_entry sje
    INNER JOIN EarliestProjectedDec epd
        ON sje.vCode = epd.vCode
        AND sje.dtEntry = epd.at_close_date
WHERE
    sje.vSource = 'Projected IS'
GROUP BY
    sje.vCode, epd.at_close_date
ORDER BY
    sje.vCode
