-- Prop_Info_AtClose.sql
-- At-close underwriting NOI from Projected IS (due diligence audit at closing).
-- Server: IM
-- Dynamic: uses the EARLIEST December 31 date per deal (not hardcoded).
-- One row per vCode.

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
    SUM(CASE WHEN sje.vAccount LIKE '5%' THEN sje.mAmount ELSE 0 END)              AS at_close_expenses,
    SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) IN (5190, 7030)
             THEN sje.mAmount ELSE 0 END)                                           AS at_close_interest,
    SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) = 7060
             THEN sje.mAmount ELSE 0 END)                                           AS at_close_principal,
    SUM(CASE WHEN sje.vAccount LIKE '4%' THEN sje.mAmount ELSE 0 END)
      + SUM(CASE WHEN sje.vAccount LIKE '5%' THEN sje.mAmount ELSE 0 END)          AS at_close_noi,
    CASE
        WHEN ABS(SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) IN (5190, 7030) THEN sje.mAmount ELSE 0 END)
               + SUM(CASE WHEN TRY_CAST(sje.vAccount AS INT) = 7060 THEN sje.mAmount ELSE 0 END)) > 0
        THEN (SUM(CASE WHEN sje.vAccount LIKE '4%' THEN sje.mAmount ELSE 0 END)
            + SUM(CASE WHEN sje.vAccount LIKE '5%' THEN sje.mAmount ELSE 0 END))
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
