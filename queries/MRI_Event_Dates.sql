-- MRI_Event_Dates.sql
-- Full Event_Dates table export for import into the Waterfall app.
-- Server: IM
-- Used by: Prop_Info_Core.sql (Acquisition Date, U/W Exit),
--          Tenant_Report.sql (lease events)
--
-- Output table name: event_dates

SELECT
    vCode,
    vEventType,
    vEvent,
    vDateType,
    dtEvent,
    vNotes
FROM
    Event_Dates
WHERE
    delete_flag IS NULL
ORDER BY
    vCode, vEventType, vEvent, dtEvent
