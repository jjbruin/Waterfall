# Lease review — the rules behind the extraction, the terms and the validation

Moved verbatim out of CLAUDE.md on Oct 5 2026. Code: `lease_review_service.py`,
`lease_terms.py`, `api/lease_review.py`. Related: `rent_roll_exhibit.md` for new
business's rent-roll specification and the IC exhibit.

## The five lease-review rule sets (moved from CLAUDE.md, Oct 5 2026)

### Lease review — settling a finding, and the change report
Live at `v518`. Jim, Sep 20 2026: "how does the analyst clear the mismatches on
this page? If the analyst determines that the applicable rent is different from
the rent roll, we need a clear report showing the changes with the reasons for the
change citing the lease document that was used."

- **THE REASON IS REQUIRED AND THE DOCUMENT IS CHECKED.** Settle records the
  figure that applies, why, and the document it was read from (defaulted to the
  one that governs). A citation naming ANOTHER tenant's document is refused — an
  unchecked citation is worse than none, because it reads as evidence.
- **A settled finding stops asking, and survives a re-validation.** Validation
  rows are deleted and rebuilt on every run, so the decision lives in
  `lease_field_resolutions`, keyed by tenant and field. `prior_value` is stored
  WITH it because the rent roll gets re-imported.
- **Confirming the rent roll is a decision too.** The report separates changes
  from confirmations rather than dropping the latter: "23 findings, 6 changed, 4
  confirmed" is the shape of an answer.
- **A validation field is not a tenant field.** `VALIDATION_FIELD_TO_RESOLVABLE`
  maps them in one place — `lease_expiration` is stored as `lease_end`, and
  `rent_step_in_force` is not a value at all but a decision about the annual rent.
- **Formatting follows the FIELD, not the column.** One table carries rents, $/SF
  figures, square feet and dates in the same two columns. Whole dollars with
  commas for a rent, dollars and cents for $/SF, and the sign OUTSIDE the dollar
  sign.
- **`RESOLVABLE_FIELDS` must reach something.** `annual_recoveries_per_sf` was
  added to `get_resolved_tenants` at the same time; a resolution that changes
  nothing downstream is a silent no-op.

### Lease review — adding a document to a scanned tenant
Live at `v518`. Jim: "rerun the extraction just on that tenant's set of leases...
Ask the user if any other files will be loaded before running."

- **THE WHOLE SET, NOT THE FILE THAT ARRIVED.** Reading only the new document
  leaves a tenant's terms assembled from a mixture of prompt versions, and the
  prompt moves. `rerun_tenant_extraction` resets every TERM-BEARING document
  (a COI is still excluded) and runs extract → consolidate → refresh the abstract
  → re-validate, each feeding the next. Scoped by a PARAMETER, never a second
  extractor.
- **Nothing runs until the analyst says there are no more files.** Files stage;
  uploading on choose started a re-read per drop, so three files arriving one at a
  time meant three runs over the same tenant. A tenant can also be re-read with no
  upload, for a document assigned from the unmatched list.
- **THE ABSTRACT WAS FROZEN THE MOMENT ANYONE SAVED IT.** `get_tenant_abstract`
  assembles from data only when NO section is stored. A section this code wrote
  (`updated_by = 'extraction'`) is refreshed; a section A PERSON wrote is never
  overwritten — it is marked `stale_at` with `proposed_content` beside it, and
  saving clears the mark whichever text they keep.
- **THE RISK ANALYSIS DOES NOT TAKE THE LEASE'S DATE, AND MUST NOT.**
  `lease_tenants.lease_end` is the RENT ROLL's figure and the extraction's
  `lease_expiration` is the LEASE's; the histogram reads the former with
  resolutions on top. Copying the lease's over it would make the rent roll agree
  by construction and no expiry mismatch could ever be reported — the `v503`
  failure. The re-read surfaces the disagreement; SETTLING it moves the analysis.
- Guardrails: `scripts/lease_validation_resolve_check.py` (106),
  `scripts/lease_tenant_rerun_check.py` (36),
  `scripts/lease_validation_context_check.py` (35).

### Lease review — exclusives, co-tenancy and the analysts' review
Built Sep 29 2026, see `open_items.md` §15 (two production steps after deploy).

- **A re-read REPLACES a document's clause rows** (`_write_document_clause_rows`,
  scoped to tenant + source_doc). Exclusives used to dedupe on the model's wording,
  which moves between runs: 291 rows for 35 tenant/document pairs on Market at
  Poplar. Rows with no source_doc (seller spreadsheet) are never touched.
- **"Bound by" is not "holds".** A lease's exhibit listing OTHER tenants' existing
  exclusives is stored `clause_role = 'subject'`. The export used to drop the role,
  so Firehouse Subs read as holding CiCi's pizza exclusive. Never ship a view of
  these rows without the role.
- **The analysts' reading lives in `lease_clause_reviews`**, per tenant and
  section, apart from the rows (which re-reads rebuild); a re-read after sign-off
  sets `reread_at`. A flag needs a note.
- **A failed reading is `error` with `extraction_error`**, never "Extracted". One
  retry when no JSON returns. Caps are the model's: 64K output, 2M text characters,
  and a cut document says `_truncated`.

### Lease review — fixed recoveries (CAM)
Live at `v518`. Jim, having read the AT&T Mobility 4th Amendment: "one of the
lease amendments was stating a fixed CAM charge for the lease. Is this situation
part of the lease review and validation to the rent roll?" **It was not.**

- The extraction captured `cam_structure = 'fixed'` — the WORD — and never the
  AMOUNT, so the one figure the rent roll could be checked against did not exist.
- **A SCHEDULE ONLY COUNTS IF `cam_structure` SAYS FIXED.** Under a pro-rata lease
  the monthly figure is an ESTIMATE trued up at the annual reconciliation; holding
  the rent roll to it as though the lease capped it compares two different things.
  That was 3 of the first 9 findings (USA Karate, CPR, and DSW/Kohls behind them).
- **`cam_fixed` is CAM only** — a separate water-and-sewer reimbursement is not a
  recovery (Pure Barre, $0.20/SF against a rent roll of $3.64). The prompt says so;
  no structural rule can tell.
- **A difference is a QUESTION when the lease passes tax or insurance through
  separately**, because our rent roll carries ONE combined recoveries figure. Only
  where it passes NEITHER through are the two sides the same quantity and a
  difference a real mismatch.
- **Calendar years, lease years, or a verbatim period** — `cam_fixed_in_force`
  resolves all three. A LEASE-YEAR schedule is placed against rent commencement
  with `month_to_date`, the rent steps' own primitive: lease year 1 begins ON rent
  commencement, so lease year 6 begins on the fifth anniversary. Without that date
  it is REPORTED, never approximated from the calendar year.
- **A stated escalation is compounded by the app** (`fill_cam_escalations`), not
  by the model — $1.96 → $2.156 → $2.3716 — and a derived figure says so. An
  escalation with nothing before it stays empty.
- **A fixed amount with NO period applies throughout**, but only when it is the
  sole row: among dated rows an undated one would beat all of them at every date.
- **`cam_fixed` is a LIST**, so `_merge_extraction_terms` needed its own case —
  the scalar/object whitelist would have dropped it silently, and in the wrong
  direction, since the schedule is stated BY the amendment.
- Live findings (Sep 21 2026): Starbucks $7.70 vs $2.16, AT&T $4.63 vs $2.38,
  BooYa's $3.64 vs $3.21, O'Reilly cannot be dated — no rent commencement date.

### Lease review — rent PSF, amendments, and rent by month of term
Live at `v510`. New business, Sep 19 2026, via Jim.

- **Rent PSF is ANNUAL rent over square feet, always.** A monthly rent is ANNUALISED
  before dividing, never divided as-is — that yields a plausible figure a twelfth of
  the right one, which is the 12x error `v495` already shipped once. A per-SF figure
  the file or lease STATES is kept as stated; only derivation is constrained.
- **The most recent amendment governs.** Consolidation layers documents in order, and
  the order used to come from `doc_date`, which `parse_doc_date` only found at the
  START of a filename. A folder of "First/Second/Third/Fourth Amendment.pdf" had no
  dates, so it fell through to UPLOAD ORDER — measured applying 4, 1, 3, 2. The
  ordinal is read from the filename and stored in `lease_documents.doc_ordinal`.
- **A rent step stated as "Months 1-12" is placed against the rent commencement date.**
  `period_start_month` / `period_end_month` are columns; `rent_commencement` is lifted
  out of `extraction_json` onto `lease_tenants`. **Month 1 begins ON rent
  commencement**, so month N is the ANNIVERSARY, not the first of that calendar month.
- **The validation no longer guesses.** When a step would not resolve it used to pick
  the step whose annual rent was CLOSEST to the rent roll's — the rent roll checked
  against whichever lease figure already agreed with it, so it could not report a
  mismatch. A tenant whose rent cannot be dated is now a `rent_step_in_force` finding.
- The extraction prompt is told NOT to convert a period into a date itself; the app
  does it, because a later commencement letter often carries the real date.
- **A DOCUMENT IS TYPED BY ITS FILE NAME, NOT ITS FOLDER** (`v510`).
  `classify_document` matched the whole stored path, and every document sits
  under `Tenant Leases/`, so the `lease` pattern matched the FOLDER and won:
  409 of 530 typed `Original Lease`, only 77 with "lease" in the file name.
  There is deliberately no fallback to the path when the basename yields
  `Other` — that fallback re-admits the 126 documents this fixes.
- **Extraction is gated on `is_term_bearing`, not on two type names.** Fixing
  the classifier alone would push 328 documents out of the old
  `('Original Lease', 'Amendment')` gate and strip the rent commencement date
  from **16 of the 38 tenants that have one**, 15 of those sources being
  Commencement Letters. `NON_TERM_TYPES` is `{'COI'}` alone and was MEASURED
  against all 500 extracted documents: excluding it costs no tenant a field and
  removes 108 certificates of insurance from the layering. An unknown or
  missing type counts as term-bearing.
- **The same function gates extraction AND filters the consolidation.** It must
  do both: a document extracted under the old classifier still carries its
  `extraction_json`, which is the consolidation's admission ticket.
- **THE BASE LEASE IS FIRST; EVERYTHING AFTER IT IS LAYERED BY DATE** (`v511`).
  The ordering used to return `originals + amendments + others`, applying every
  non-amendment AFTER every amendment — so a 2021 Acceptance of Premises
  overwrote a 2026 First Amendment. Undated documents sort last, amendments
  among them by number, so a folder of undated numbered amendments still
  applies 1,2,3,4. On an equal date the amendment wins.
- **The 70 stored consolidations were re-run at `v511`** — 392 documents
  applied, coverage unchanged on every field, nine expirations and five rent
  commencements corrected, and `lease_tenants.rent_commencement` populated for
  the first time (0 -> 37). A second pass moves nothing, which is the
  convergence proof. Re-consolidation makes NO API calls; it re-layers the
  existing per-document extractions.
- **A SCANNED PDF IS READ AS IMAGES, not as its empty text** (`v512`). 219 of
  the 419 term-bearing documents yield under 200 characters — 47 amendments and
  28 original leases among them — and were contributing nothing, silently. Below
  `SCAN_TEXT_THRESHOLD` the PDF goes to the model as a document block: no OCR
  stack, no Tesseract, no poppler. Over the 32 MB / 600-page caps it is refused
  with the reason, never attempted, and every result records
  `_extraction_source`.
- **Extraction runs on `claude-opus-5`** (Jim: best model for every scenario).
  Thinking is ON by default there, so the response's first block is a THINKING
  block — never read `content[0].text`. A refusal returns HTTP 200 with no text;
  check `stop_reason` first.
- **The re-extraction is DONE** (Sep 20 2026): 417 documents, converged.
  Coverage rose on every field — rent_commencement 38 -> 53, square_feet
  43 -> 61, escalation 45 -> 65 — with 110 fields newly populated, and
  `period_start_month` went 0 -> 305 with **208 rent steps dated from the term**,
  so the month-of-term feature is live on production for the first time.
- **A SECOND full re-extraction ran on `v517`** (Sep 21 2026) to pick up
  `cam_fixed` / `escalation_pct`: 419 documents in three hours, no failures.
  Coverage HELD on every field (rent_commencement 53, lease_expiration 57,
  square_feet 61, suite 71), security deposit +2 and escalation -1, with 92
  fields moved — 32 escalation descriptions, 15 rent-step counts, 11 rent
  commencements, 11 lease commencements, 7 suites and **one** lease expiration.
  Tenants carrying a fixed-recovery schedule went 4 -> 9. **Run it with a diff:**
  it is what found the pro-rata estimate defect `v518` fixed.
- **A ROW READ AS NOT-A-TENANT HAS NO EXPIRY** (`v514`). `v501` taught the
  roster and the headline totals to respect `tenant_status`; the expiration
  histogram was never updated, so the building banner, the subtotal rows and the
  vacant suites still reached it carrying the literal string `'NaN'` as
  `lease_end`. **`pd.to_datetime('NaN')` returns NaT WITHOUT raising**, so
  `.year` is `nan` and BOTH range comparisons are False — a range guard cannot
  catch NaN. It reached `yearly[nan]` and returned HTTP 500. Filtered on
  `COALESCE(tenant_status,'active')='active'` AND guarded with `pd.isna`,
  because the status filter cannot save an ACTIVE tenant with an unparseable
  date.
- **The four secondary panels load with `Promise.allSettled`, never `all`.**
  `all` rejects on the first failure, and validation was assigned last — so one
  500 left 23 validation rows in the database and a blank screen, with the catch
  logging "(expected for new reviews)". A panel that cannot load now says so.
- Guardrail: `scripts/lease_validation_blank_check.py` (15).
- Guardrails: `scripts/lease_terms_check.py` (129), which drives the shipping
  paths against a real database including an EXISTING schema migrated with rows
  in it, and `scripts/lease_doc_type_check.py` (37).
