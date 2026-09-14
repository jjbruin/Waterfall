-- MRI_GL_Detail.sql
-- Server: PMX (BV6899900001)
-- Maps to: gl_detail table
--
-- Entity-level GENERAL LEDGER detail. This is the feed behind accounting's
-- quarterly workpaper packages (the "GL Detail" tab, and every Trial Balance,
-- Intercompany, Accrued Expenses and Accrued/Cash support tab derived from it).
-- The app has never pulled GL data: ISBS is PROPERTY statement data on the IM
-- server, and IA_* is investor activity. Neither can produce an entity trial
-- balance.
--
-- Replicates the Spreadsheet Server "NEW JOURNAL" query one-for-one, with the
-- &SPARM smart parameters removed -- run_query() calls pd.read_sql(sql, conn)
-- with no parameter binding, so the scope is set in the WHERE clause below and
-- the app filters the loaded table by ENTITYID / PERIOD / BASIS.
--
-- TWO SOURCES, BOTH REQUIRED:
--   JOURNAL - the open period(s). Every row is a normal entry, so the SS query
--             hardcodes BALFOR = 'N' and this one keeps that.
--   GHIS    - closed history, INCLUDING the balance-forward rows (BALFOR='B')
--             that carry each year's opening balance. A balance-forward row has
--             no real entry date, so ENTRDATE is built from the period as
--             MM/01/YYYY -- exactly what Spreadsheet Server does.
-- Dropping either one loses half the trial balance.
--
-- OBSERVED IN THE 06.30.2026 PPIECH PACKAGE (verify before relying on it):
--   the opening-balance rows came back BASIS='A' and the period activity
--   BASIS='B'. That is why accounting's parameter is the RANGE `A.B` rather
--   than a single value. Do NOT hardcode a basis anywhere downstream; carry the
--   column and let the caller choose. Confirm what A and B mean with accounting:
--     select BASIS, BALFOR, count(*) from JOURNAL group by BASIS, BALFOR
--     union all
--     select BASIS, BALFOR, count(*) from GHIS group by BASIS, BALFOR
--
-- SCOPE: PERIOD >= '202401'. There is no parameter plumbing, so this pull is
-- every entity for that window -- widen or narrow the literal in BOTH halves of
-- the UNION together. Size it before widening: GHIS is full company history and
-- ISBS_Download already had to be split at 800K rows.

SELECT
    RTRIM(LTRIM(JOURNAL.ENTITYID))              AS ENTITYID,
    JOURNAL.PERIOD                              AS PERIOD,
    JOURNAL.ENTRDATE                            AS ENTRDATE,
    RTRIM(LTRIM(GACC.ACCTNAME))                 AS ACCTNAME,
    JOURNAL.ACCTNUM                             AS ACCTNUM,
    JOURNAL.BASIS                               AS BASIS,
    'N'                                         AS BALFOR,
    JOURNAL.ITEM                                AS ITEM,
    JOURNAL.REF                                 AS REF,
    RTRIM(LTRIM(JOURNAL.DESCRPN))               AS DESCRPN,
    RTRIM(LTRIM(JOURNAL.SegmentID))             AS SEGMENTID,
    RTRIM(LTRIM(ADVGL_SEGMENTDATA.RLTDENTITY))  AS RLTDENTITY,
    GLSegment_RLTDENTITY.DESCRIPTION            AS RLTDENTITY_NAME,
    JOURNAL.AMT                                 AS AMT
FROM GACC
INNER JOIN JOURNAL
    ON GACC.ACCTNUM = JOURNAL.ACCTNUM
LEFT JOIN ADVGL_SEGMENTDATA
    ON ADVGL_SEGMENTDATA.SegmentID = JOURNAL.SegmentID
LEFT JOIN GLSegment_RLTDENTITY
    ON GLSegment_RLTDENTITY.RLTDENTITY = ADVGL_SEGMENTDATA.RLTDENTITY
WHERE JOURNAL.PERIOD >= '202401'

UNION ALL

SELECT
    RTRIM(LTRIM(GHIS.ENTITYID))                 AS ENTITYID,
    GHIS.PERIOD                                 AS PERIOD,
    CASE GHIS.BALFOR
        WHEN 'B' THEN SUBSTRING(GHIS.PERIOD, 5, 2) + '/01/' + SUBSTRING(GHIS.PERIOD, 1, 4)
        ELSE GHIS.ENTRDATE
    END                                         AS ENTRDATE,
    RTRIM(LTRIM(GACC.ACCTNAME))                 AS ACCTNAME,
    GHIS.ACCTNUM                                AS ACCTNUM,
    GHIS.BASIS                                  AS BASIS,
    GHIS.BALFOR                                 AS BALFOR,
    GHIS.ITEM                                   AS ITEM,
    GHIS.REF                                    AS REF,
    RTRIM(LTRIM(GHIS.DESCRPN))                  AS DESCRPN,
    RTRIM(LTRIM(GHIS.SegmentID))                AS SEGMENTID,
    RTRIM(LTRIM(ADVGL_SEGMENTDATA.RLTDENTITY))  AS RLTDENTITY,
    GLSegment_RLTDENTITY.DESCRIPTION            AS RLTDENTITY_NAME,
    GHIS.AMT                                    AS AMT
FROM GACC
INNER JOIN GHIS
    ON GACC.ACCTNUM = GHIS.ACCTNUM
LEFT JOIN ADVGL_SEGMENTDATA
    ON ADVGL_SEGMENTDATA.SegmentID = GHIS.SegmentID
LEFT JOIN GLSegment_RLTDENTITY
    ON GLSegment_RLTDENTITY.RLTDENTITY = ADVGL_SEGMENTDATA.RLTDENTITY
WHERE GHIS.PERIOD >= '202401'
;

-- NOTE ON UNION vs UNION ALL. Spreadsheet Server writes UNION, which DEDUPES.
-- A general ledger legitimately carries identical rows -- the same account, the
-- same amount, the same description, posted twice in a period -- and UNION
-- silently collapses them into one, understating the balance. This is the same
-- trap as the ISBS drop_duplicates measured taking isbs_raw from 797,660 rows
-- to 439,268 (see CLAUDE.md). JOURNAL and GHIS are disjoint by construction
-- (open vs closed), so there is nothing to dedupe ACROSS the halves either.
-- UNION ALL is deliberate. If a row count ever has to tie to a Spreadsheet
-- Server sheet exactly, the difference is SS's dedupe, not a defect here --
-- prove it with:
--   select ENTITYID, PERIOD, ACCTNUM, BASIS, ITEM, REF, DESCRPN, AMT, count(*)
--   from gl_detail group by ... having count(*) > 1
