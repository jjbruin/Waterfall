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

## Deployed and measured (Sep 29 2026)

`v535` step 1, `v536` step 2, `v537` step 3. After `v536`, all 77 tenants with
readings re-consolidated and `rebuild_clause_rows` run for reviews 2 and 3 (no API
calls). Windsor Square had the same duplicate problem as Poplar: 1,134 exclusive
rows -> 337. CORRECTION: the 115 -> 39 option "duplicates" were mostly each document
legitimately restating the options (119 -> 117 after the rebuild); the governing
count now comes from `_options_summary`.

Step 1's re-read: Sam's Club and Perkins now read in full. **Tropical Smoothie's
original lease (60 pages, 27 MB) still fails** -- it renders and was sent as images,
and the model answered in prose. Not yet diagnosed.

### Against the exhibit, 27 matched tenants

| | After v537 |
|---|---|
| Total SF / current rent | 228,122 / $3,125,144 vs 228,119 / $3,112,833 (0.4%) |
| Current rent | 26 |
| Future steps exact | 20 |
| Expiration | 22 |
| Option(s) summary | 21 (was 0) |
| Option rows count | 22 |
| **Option rent** | **0 of 34 -- needs a re-extraction with the new prompt** |
| Whole block exact | 1 |

### THE START COLUMN IS A DEFINITION QUESTION -- ask new business
Their spec §28 says Start = lease commencement. Their exhibit does not follow it:
of 27, 5 match the original commencement, 8 rent commencement, and 14 match no date
the app holds (BooYa's 2014 = its FIRST renewal, Marco's 2016-10-17 = apparently the
execution date, Mattress Firm 2025-03-22, Chapultepec/Ciao Baby/Collierville/Peak
Potential/Perkins/USA Karate = the current term's start). No rule reproduces it.

### Flags to tune before the exhibit
`psf_mismatch` (25) probably compares a historical step's stated PSF with today's SF;
`step_after_expiration` (10) is probably option-period rent the lease already states
(usable as option rent). Also: Patton's options are 30-day rolling, which
`term_years` cannot express ("6 x term not stated").

## Acceptance on production data (v538, Sep 29 2026)

Full Market at Poplar re-read with option rent: 33/33 tenants, option rent now
captured (Starbucks 4/4, Pure Barre, Peach Cobbler, Tropical Smoothie, USA Karate).
Tropical Smoothie's original lease finally read. The single re-read job was killed
by memory at tenant 27 -- run long re-reads ONE SUBPROCESS PER TENANT.

Paired 32 of 33 (6 by identical SF where names differ). Matches: SF 31, current
rent 30, expiration 27, Option(s) 25, subordinate rows 17, whole block 4.
Workbooks sent to Jim: `Downloads\Acceptance - Market at Poplar Rent Roll.xlsx`
(summary, differences explained, tenant by tenant, flags) and the app's exhibit.

The differences, by kind:
- **Start definition** (26/32) -- new business's question, not the engine's.
- **OLD ANALYST SETTLEMENTS OUTRANK THE CORRECTED DOCUMENTS**: Mattress Firm's
  lease_end was settled 2027-09-30 and Hobby Lobby's 2027-07-31 when the app still
  read the documents wrongly; the documents now say 2035-03-21 (Mattress). The
  timeline flags a settled RENT that differs from the lease, but not a settled
  start/expiry -- build that flag, and have analysts revisit.
- Documents not captured: Hobby Lobby's 2024 4th Amendment, Muddy Paws' extension,
  Perkins' options.
- Option rent detail: annual option steps (Peak Potential +2%/yr) collapsed to one
  amount per option.
- Engine: an option-rent period dated before the option start produced a stray row
  (Little Petals) -- clip option periods to the option.
- 30-day rolling options (Patton): term unit unsupported; their exhibit collapses
  six 30-day options into one row.

## The three acceptance fixes, and the regression they caused (v539-v541, Sep 30 2026)

Built from the v538 findings: the stale-settlement flag, rent steps replaced
per document, month-of-term re-anchoring to the commencement letter, and
30-day rolling options.

- **v539** shipped all three with Charlene's PR #5. The rebuild on production
  took rent steps from 241 to 180 (Poplar) and 673 to 509 (Windsor), idempotent.
  Little Petals' option began 2031-03-01 as their exhibit shows. **Patton was
  NOT fixed**: its "Thirty (30) day option" wording sits in each option's
  `rent_schedule` period, the timeline's copy of an option dropped the schedule,
  and the summary shown was the one STORED at consolidation. The first fixture
  put the wording in `rent_terms` and passed -- a fixture in the shape I
  expected, not the shape production stores.
- **v540** fixed Patton. **The acceptance re-run then showed a regression**:
  current rent 30 -> 26 of 32, total +$231k. Two causes, both mine:
  - **Re-anchoring guessed.** Unconfirmed, it fell back to rent commencement;
    Habitat (rent from month 1, rent commencement six months late) shifted six
    months. Now evidence-only: the anchor moves when counting from the lease
    commencement lands the first paying month on the rent commencement, and
    otherwise nothing moves.
  - **Replacing each document's steps exposed QUOTED schedules.** Hobby Lobby's
    2nd/3rd Amendments restate the original "months 1 / 61"; counted from each
    amendment's date, month 61 outranked the amendment's own $435,582. The old
    tenant-wide dedup had been hiding this by accident. `drop_restated_steps`
    keeps months with the document that first stated them.
- **v541** carries both. RUN THE ACCEPTANCE COMPARISON AFTER EVERY ENGINE
  DEPLOY -- the guardrails were green at v540 and the regression was only
  visible against new business's own exhibit.
- **v542** ends an add-on when a LATER document states the whole rent (Mattress
  Firm's 2020 +$6,125/mo under its 2024 restated $170,100), keeping Marco's rule
  that the original lease's own later steps do not end one. Every add-on applied
  is flagged (`additional_rent_added`) with its document.

Acceptance, Market at Poplar, 32 of 33 paired:

| | v538 | v540 | v541 | v542 |
|---|---|---|---|---|
| Current rent | 30 | 26 | 28 | **29** |
| Annual PSF | 28 | 24 | 26 | **27** |
| Expiration | 27 | 27 | 28 | **28** |
| Option(s) | 25 | 26 | 26 | **26** |
| Subordinate rows | 17 | 16 | 17 | **18** |
| Total rent (theirs 3,112,833) | 3,125,144 | 3,356,526 | 3,277,008 | **3,203,508** |

The three current-rent differences left at v542:
- **Habitat** $606,501 vs $518,931 -- a 2025 "Rent Reduction REQUEST" letter
  read as adding $87,570/yr. Flagged; an analyst decides whether it binds.
- **Martway** -- the rent roll's own figure moved $56,383 -> $47,177; not this
  work, and Martway is the one tenant unpaired with their exhibit.
- The third is the pre-existing difference from v538.
Stale-settlement flags now showing: A Perfect Bloom, Hobby Lobby, Muddy Paws.
Mattress Firm's old 2027-09-30 settlement is no longer applied (documents give
2035-03-21).

## New business's answers to the v542 acceptance (Sep 30 2026) -- BUILT, NOT DEPLOYED

- **Settlements outrank the documents** -- confirmed, no change. Mattress Firm's
  analyst REMOVED the old expiry settlement (only SF is settled now), so the
  timeline shows the documents' 2035-03-21 / $170,100. Bombay's settled $31,671.24
  is in force. Hobby Lobby's settled 2027-07-31 still outranks the documents and
  is flagged; after the re-read it will differ from the 4th Amendment -- the
  analyst should revisit it.
- **The three "missing documents" were PART-SCANNED FILES** read as text (measured
  per page on production): Hobby 4th Amend [54,54,54,289] (dotloop stamps only),
  Perkins 1st Amend [2722,0,0,0,0], Muddy Paws lease 22 of 26 pages empty. Fix:
  `PAGE_TEXT_MIN` / `pages_without_text` -- any page under 100 characters sends
  the PDF. 11 Poplar and 40 Windsor term-bearing documents are mixed, so their
  next re-read moves route: RUN THE ACCEPTANCE COMPARISON after re-reading.
- **A file holding several instruments** (Muddy Paws' lease + 1st/2nd Amendments):
  the prompt now returns `instruments` and the as-amended terms;
  `bundled_instruments` layers such a file at its last instrument's date, as an
  amendment (exercises kept), commencement dates fill-only.
- **Annual option increases** (Peak Potential, 2%/yr): `_option_periods` expands a
  percentage with `escalation_frequency: annual` (or "annual" in its wording) into
  yearly periods, each rounded to the dollar before the next, as their figures
  are. Takes effect on stored data without a re-read.
- **A Perfect Bloom, NOT fixed (analyst investigating)**: two Original Leases,
  Ste 6 (2024) and Ste 7 (2026 relocation), both read with no dates; Ste 7's rent
  steps count from Ste 6's commencement letter (2024-07-01). A later lease for
  new premises probably has to start a fresh lease -- wait for the analyst.
- Guardrail `scripts/lease_bundle_scan_check.py` (32).
- To do after deploy: re-read Hobby Lobby, Muddy Paws, Perkins (one subprocess
  per tenant), then the acceptance comparison.
