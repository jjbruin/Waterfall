SELECT
    vs.vcode,
    vs.dtEntry,
    vs.vSource,
    vs.vAccount,
    vs.vType,
    vs.vInput,
    vs.mAmount,
    vs.statement_id

FROM vstaging_journal_entry vs

WHERE vs.vCode LIKE 'P%'

ORDER BY vs.vcode, vs.vsource, vs.dtEntry
