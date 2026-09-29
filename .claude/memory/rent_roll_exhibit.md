# Rent roll: new business objectives, gaps, and the IC exhibit (Sep 29 2026)

New business sent a 31-section specification ("Rent Roll Application – Lease
Analysis & Extraction Instructions") and an example exhibit,
`Downloads\Claude - Market at Poplar Rent Roll.xlsx`, range **B2:H111**, to be
produced by the app for the final investment committee approval memo "as a result
of the lease review and analyst validation". Their core principle (§31): the app is
a **lease interpretation and timeline-building tool, not a field extractor**.

## Measured: the app against their exhibit (Market at Poplar, review 3)

27 of the exhibit's 34 tenants matched an app tenant by name (the other 7 differ in
naming; not yet reconciled). Rent roll as-of date 2026-09-01 in both.

| Field | Matches | What the misses are |
|---|---|---|
| SF | 26/27 | Tropical Smoothie 1,300 vs 1,307 |
| Current annual rent | 26/27 | Matches because the app's figure IS the validated rent roll; the lease-derived rent in force is used only by validation |
| Expiration | 21–22/27 | **Exercised options never applied**: Hobby Lobby 2024 vs 2032, Mattress Firm 2027 vs 2035, Muddy Paws 2025 vs 2028 |
| Start | 12/27 | **Consolidation's "latest document governs" overwrites the lease start**; the Sep 29 re-read moved BooYa's 2019 -> 2024 and Outback 1991 -> 2016. §28: Start is the original commencement |
| Future step dates | 21/27 | Steps stored as start dates only -- no end dates, no gap/overlap check, stale extras |
| Option count | 19/27 | **115 renewal-option rows stored vs 39 in the exhibit** (Patton 6 vs 1, Hobby Lobby 5 vs 2): options dedupe on (source_doc, option_number) and never replace -- the stale-row shape the exclusives had before `v532` |
| **Option rent** | **0/39** | Captured only as free text in `rent_terms` ("fixed increase – $10,075.47 per month"), never as annual rent / PSF; termination-option text also lands in that column |

## The 31 sections, against the app

**Doing**: every document type read, COI alone excluded (§1, §30.1–2); chronological
layering, later governs (§3, §14); an as-of date per review (§2, used only by
validation); lease vs rent commencement kept apart, "Months 1-12" dated from rent
commencement (§9); rent normalisation (§15); additional-space amendments add rent
(§16 partly); source document per value (§22 partly); unread documents flagged (§23
partly); scans read as page images.

**Partial / wrong**: Start (§28); exercised options (§11); remaining option count
(§13); continuous timeline (§4, §7, §8); SF over time (§16); abatement vs
contractual rent (§18).

**Missing**: option rent amounts, option rent steps, FMV flags (§10, §12, §20);
timeline validation checks (§24); conflict / missing-document / missing-exhibit
flags (§21, §23); page and section per value (§22); the exhibit itself (§26–27); and
**a way for the analyst to settle a step, option, start or expiration** -- validation
settles only eight scalar fields today, and the exhibit is meant to come out of
analyst validation.

## The exhibit, exactly

- Times New Roman 12 throughout. Columns B:H = Tenant, SF, Start, Expiration, Annual
  Rent, Annual PSF, Option(s). Widths B 40, C 10, D 13.54, E (default), F 14, G 12,
  H 23.45; column A 2 as a gutter. Gridlines off.
- **Header row** (row 2): bold, centred, fill = theme white darkened 5% (F2F2F2);
  MEDIUM top, MEDIUM left on B, MEDIUM right on H, THIN bottom.
- **Frame**: MEDIUM left border down column B and MEDIUM right down column H for
  every row.
- **Tenant groups**: a THIN bottom rule across B:H under the last row of each tenant.
- **Tenant row**: name left-aligned; SF `#,##0`; Start and Expiration `mm-dd-yy`;
  Annual Rent `#,##0`; PSF `#,##0.00`; Option(s) right-aligned text, "n x 5 Years"
  or "None". All values right-aligned.
- **Subordinate rows**: "Rent Step Dates" (first future step only) and "Option n" in
  ITALIC, right-aligned in column B; later step rows unlabelled; D:G carry start,
  end, annual rent, PSF; C and H blank.
- **Total row**: "Total / Wtd. Avg." bold, same grey fill, MEDIUM bottom (closes the
  frame); SF = sum of tenant SF; Annual Rent = sum of tenant rows' current rent;
  PSF = rent / SF; D, E, H blank.
- **Order**: alphabetical by tenant. Vacant suites excluded (228,119 SF shown; the
  sheet's side note adds 1,600 vacant to reach the GLA).
- **Paging**: the header row is repeated at the page break (row 56); a tenant group
  is not split across pages.
- Tenant row Start = original lease commencement; Expiration = current contractual
  expiration; rent = rent in force on the as-of date (§5, §28).
- Rows 116–152 of their sheet are the analyst's tie-out to their model (Model SqFt,
  Model Rent, variances) -- outside the exhibit; offer as a separate check tab.

## The plan (agreed order, Sep 29 2026)

1. **The three re-read failures** -- DONE in the working tree, see below.
2. **Governing terms**: keep the original lease start; apply exercised options to the
   expiration and the remaining count; option rows replace per document (like
   `_write_document_clause_rows`); extract option rent as numbers, with option rent
   steps and an FMV flag.
3. **One timeline engine**: governing terms -> continuous periods (end = next start −
   1 day, last = expiration, options after), the §24 checks, the §21/§23 flags. ONE
   engine: every screen and the exhibit read it.
4. **Analyst settlement of timeline rows** (step, option, start, expiration) with a
   reason and a cited document, like rent-roll findings.
5. **The exhibit** in Excel and print; acceptance = reproduce the Market at Poplar
   exhibit cell by cell and list every difference.

## Step 1 -- the three failures from the Sep 29 re-read

| Tenant / document | Cause | Fix |
|---|---|---|
| Sam's Club, CenturyLink agreement (50 pp) | 26 NUL characters in the text layer; PostgreSQL refuses NUL | stripped in `extract_pdf_text` |
| Tropical Smoothie, original lease | 27 MB raw passed a RAW 32 MB check; ~36 MB once base64-encoded -> 413 | `_pdf_fits` checks the ENCODED request; an over-size scan goes as page images |
| Perkins, Assignment & 1st Amendment | intermittent -- failed twice on the same text, read cleanly on a later run | a failed TEXT reading is retried from the PDF (or images), not the same text |

The Sep 29 re-read otherwise: Outback's 2026 Option Letter and BooYa's 2014 Renewal
Notice read via page images.

## Step 2 -- the governing terms (built Sep 29 2026)

Read off the per-document extractions on production first. The causes were
CONSOLIDATION RULES, not reading errors:

| Tenant | Documents say | Why the app was wrong | Rule |
|---|---|---|---|
| Mattress Firm | 2024 3rd Amendment -> 2035-03-21 | an UNDATED "Commencement Date (Exhibit)" sorts last and reset it to 2027 | an undated non-amendment document only FILLS GAPS (it may still set the commencement dates if it is a commencement document) |
| Outback | 2026 option letter exercises -> 2031-12-09 | an undated third-party "Abstracts_Misc" reset expiry to 2026 and start to 2016 | same |
| BooYa's | original commencement 2008-03-15 | Addendum #4 / 5th Amendment state their RENEWAL term's start | only an Original Lease or Commencement Letter sets `lease_commencement`; others -> `current_term_commencement` |
| Muddy Paws | original lease option marked "exercised" | a lease cannot record its own exercise | exercised flags from an Original Lease are discarded |
| Hobby Lobby | 3rd Amendment restates options 1-5 | merge by number mixed eras | a document listing dated unexercised options REPLACES the set; an all-exercised list marks the matching option; undated restatements merge by number |

Plus: an exercised option ending after the expiration carries the expiration
(`_expiration_basis` says so); `_remaining_options` / `_options_summary` ("2 x 5
Years") computed after consolidation; option rows replaced per document
(`_write_document_option_rows`, also in `rebuild_clause_rows`); "Addendum" typed
Amendment; the prompt asks for option rent as figures (`rent_basis`,
`rent_schedule` with per-period amounts or `escalation_pct`), never estimating FMV.

NOT fixed by rules, needs a person or a re-read: Hobby Lobby's 2024 4th Amendment
was read with nothing extracted (the exhibit's 2032 comes from it); Muddy Paws has
no document for the extension to 2028 the exhibit shows.

Guardrail `scripts/lease_governing_terms_check.py` (21), the five real cases as
fixtures; proved against reversing each of the three core rules.
