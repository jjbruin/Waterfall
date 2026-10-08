# Board section (Phase 0 LIVE at v572, Oct 5 2026)

The plan is in Claude Docs, "Board Package — Development Plan":
https://claude.ai/code/artifact/71e5c89f-933f-4b88-9e3e-f1b2e1d7a5b9. It records the
seven decisions Jim made on Oct 5.

## Phase 0: the foundation
- **Board is an OPT-IN section** (`"opt_in": True` in `auth/sections.py`).
  - The storage is inverted: a GRANT is a TRUE row, and no row means denied. Deploying
    the section shows it to nobody, and a user created later doesn't have it.
  - A grant may carry an end date (`user_section_access.expires_at`, added in place by
    an asked-first ALTER). It's for outside advisors; past the date the grant reads as
    absent.
  - The User Management column shows Board unticked for everyone.
- **Only the `admin` USERNAME grants it**, through User Management or Board > Access.
  The admin ROLE (Charlene, the developers) holds nothing here. The guardrail proves
  this past the section gate: it grants an admin-role user Board, then checks that
  user can do no more than a reader.
- **The permissions** (`auth/permissions.py`), opt-in, by username:

  | Permission | Allows | Implies |
  |---|---|---|
  | `board_edit` | narrative, as-of dates | — |
  | `board_build` | creates meetings | `board_edit` |
  | `comp_view` | salary planning: view | — |
  | `comp_edit` | salary planning: edit | `comp_view` |

  **A permission requires its section.** Remove Board, or let it expire, and every
  permission goes void although the rows remain. Implications are applied on read, so
  revoking the stronger permission never leaves the weaker one behind as an orphan.
- **The access log** (`auth/audit.py`, table `access_audit`) is append-only. It holds
  every section and permission change, and every meeting, schedule and narrative
  write. A failure to log RAISES. The admin username reads it on Board > Access log.
- **NEVER-exposed tables** (`sections.NEVER`): `comp_*`, `access_audit` and
  `user_permissions`.
  - They don't appear in Data Explorer, the export or the assistant, for anyone: the
    admin username and salary holders included.
  - `pull_production_db.py` skips them; `comp_*` only travels with
    `--include-compensation`.
  - `board_*` tables follow the Board section.
- **Meetings** (`services/board_service.py`, `api/board.py`):
  - A meeting carries every schedule in the deck's catalog, each with its OWN as-of
    date (the January deck mixed 12/31/25 and 9/30/25), plus seven narrative blocks.
  - Refused: an unknown key, a date that doesn't parse, or any edit once the meeting
    is no longer a draft.
  - Saved with a warning: an as-of date after the meeting date.
  - No figures yet. Each schedule's view arrives with its phase.
- Guardrail `scripts/board_access_check.py` (156 checks, including the race-safe column add). It enumerates every
  `/api/board` route from the running app and calls each one as no-token, ungranted,
  expired, revoked, the admin role, reader, editor, builder, salary holder and the
  admin username. It fails on each injected defect: `optin` (31 failures), `role` (5),
  `section` (1). `section_access_check` now expects "no rows" to mean every
  NON-opt-in section.

## Next phases
Phases 1, 2 and 4 can start now; their engines are live. Phase 3 needs the
originations spreadsheet. Phase 5 needs the CFO's model and an anonymized payroll copy.

## Phase 1 started Oct 7 2026: the PE exposure engine reconciled to the January deck at 12/31/25
Run on PRODUCTION data (`pe_exposure_service.build(2025-12-31)`), against the deck's own
source, `Source/Asset Management Supporting Materials/PSC Preferred Equity Tracker -
12312025 updated 01082026.xlsx` (Cost sheet), and, for asset class, `Source/PSC Investment
Summary Data_v34_12.31.25 - Acct 01082026.xlsx`, sheet "Asset Type Net Cost" (per-deal
classes typed by hand; its total 691,681,937.6 IS the deck's $691.6M).

| | Deck / tracker | Engine |
|---|---|---|
| Funded (p.26 total net pref) | 635,556,951 | 635.6M |
| Future funding (p.26 footnote "$56.1M") | 56,124,986 | 56.77M |
| Total incl. unfunded (p.23) | 691,681,937 | 692.4M |
| PSC incl. unfunded (p.23) | 120.2M | 120.9M |
| 3rd-party funded: KoC / F&F / Decl / Amb | 82.5 / 28.1 / 13.1 / 12.3 | identical |
| TIAA funded / PSC funded | 383.6 / 115.9 | 385.7 / 113.9 |

FUNDED ties holding by holding (54 holdings) except: **Brainerd** -- the engine gives TIAA
$2,075,057 the tracker gives PSC; it is one of accounting's allocation overrides and
`ownership_overrides` on production is EMPTY (none of BRNERD / JBFAIR / NOTTNV entered);
**Pontchartrain** +$57,000 in the engine; **Bel Air** $11.7K PSC vs F&F (the tracker's
typed constants); Apple only looks different (the tracker's Bales row has no ID; combined
10.91M vs 10.93M, CAD).

FUTURE FUNDING differs by +$0.65M, five deals: Pontchartrain (tracker 2,250,000, engine
none); 30 Bearfoot (engine 915,000) and Donald Lynch (engine 2,485,000), both held at 12/31/25
and sold 9/4/26, absent from the tracker; Apple Bales (-253,192) and Middle Island
(-233,312 vs tracker +16,688) -- the One Pager's remaining-to-fund goes NEGATIVE when a deal
is over-funded and the engine sums it.

ASSET CLASS (p.23) differs only by classification: Burton $26.6M (deck Grocery-Anchored,
app `Retail - Non Groc.`); Brainerd's future funding $13.3M (deck Non-Grocery Retail -- a
deck error, Brainerd is multifamily); Merle Hay + 5-15 Broad $8.5M (app plain `Retail`,
deck Non-Grocery); plus the future-funding items. Deck "Self Storage" = its "Other -
Self-Storage" rows; deck "Other" = Industrial, RV Park, Specialty.

Open, before the Phase 1 views are built (decisions, not code):
1. DONE for Brainerd (Oct 7). JB Fair / Nottingham: the 12/31/25 tracker used the standard
   ratios for both, so nothing to enter at 12/31/25; accounting's later corrections need
   their own dated sets.
2. Future funding: may remaining-to-fund be negative? Do Bearfoot / Lynch commitments count
   at 12/31/25? Why Pontchartrain shows $2.25M unfunded in the tracker and none in the app.
3. Asset class: fix `Asset_Type` in MRI (Burton, Merle Hay, 5-15 Broad) or keep a board
   class per deal in the app.
4. Page 24: a canonical operating-partner list (app names vary: JPI / JPI Companies,
   Vastgood / Vastgood Properties LLC, Apple / Apple Self Storage, Bertram and DiMarco).

## Phase 1 views: pages 26, 27, 29-31 (Oct 7 2026; live `v598`)
**Shown as a deck since `v600`** (`components/board/BoardDeck.vue` and the slide components):
each schedule is a page on the deck's 1100 x 825 canvas, scaled to the window; figures load
once per meeting + schedule + as-of and page turns never call the server. Layout decisions
(titles, the deck's narrower column set, 30 rows a page, footnotes) live in
`board_views_service`, not the screen. Pages 29-30 set at ~9.4px on the canvas because
Investment Metrics carries full legal partner names; the page-24 canonical partner list fixes it.
`GET /api/board/meetings/<id>/schedules/<key>/view`, at the meeting's as-of for that
schedule; `services/board_views_service.py` composes, computes nothing. pp.29-31 ARE the
Investment Metrics payload (through `investment_metrics_service.get_report`, the route's
cache moved there). Guardrail `board_views_check.py` (35).

The deck's own sources: p.26 gross cap / properties = `PSC Investment Summary Data_v34...`
sheet `Investments_1` and `# Deals & Properties`; pp.27, 29-31 = `Asset Management
Supporting Materials/PSC Investment Metrics 9.30.25 OW.xlsx` -- the 9/30/25 Investment
Metrics workbook with Q4 closings and Q4 sales moved across by hand. **The deck mixes
dates**: population and pref at 12/31/25, proceeds and CoC "through 9/30/25".

Production, 12/31/25, after the Apple fix (`066f747`):

| | Deck | App |
|---|---|---|
| p.26 PSC / 3rd party / total net pref | 115.9 / 519.6 / 635.5 | 115.97 / 519.66 / 635.63 |
| p.26 TIAA / KoC / F&F / Decl / Amb | 383.6 / 82.5 / 28.1 / 13.1 / 12.3 | 383.65 / 82.53 / 28.08 / 13.12 / 12.29 |
| p.26 wholly owned: deals / PSC | 6 / 24.5 | 6 / 24.53 (same six deals) |
| p.26 deals | 54 | 53 (deck counts Brainerd I and II as two transactions) |
| p.26 properties | 98 (WO 23) | 70 (WO 6): MRI `Property_Count` is 1 on Apple (16), PMAT Midwest (3), Prestige (12); with those, 98 and 23 exactly |
| p.26 gross cap | 3,190.79 | 3,170.92: IM Total Size; Apple, Bales and Life Storage have none (first lien unknown, open item 20.2) |
| p.27 current pref incl. unfunded | 691.6 | 692.40 (the future-funding items above) |
| p.27 proj IRR / proceeds / CoC | 15.7% / 198.6 / 6.4% | 15.6% / 201.2 / 7.3% (cash through 12/31, not 9/30) |
| p.27 exited pref / realized IRR / proceeds / CoC | 128.3 / 17.7% / 208.4 / 5.6% | 128.26 / 20.7% / 209.7 / 7.7% |
| p.27 combined CoC | 6.1% | 7.4% (IM's pref-weighted average over both tables) |

**DECIDED (Jim, Oct 7 2026): ONE AS-OF PER SCHEDULE, page 27 included.** No second "cash
through" date. Every figure on a page belongs to that schedule's as-of; where the latest
quarter's accounting is not closed by the meeting, set the schedule's as-of to the last closed
quarter rather than mixing dates as the January deck did. Do not add a cash-through date.

**DECIDED (Jim, Oct 7 2026): BRAINERD I AND II ARE ONE DEAL.** A deal is an MRI deal
(Investment Metrics' Current table), so Brainerd Place is counted once; p.26's 53 against the
January deck's 54 is this, not a gap. Do not split it into two transactions.

Jefferson Stephens (closed Oct 2025) has no funded capital at 12/31/25 -- the tracker
carries $1 -- so it is classed JV by its $22.7M unfunded commitment and adds nothing to
capital. The p.27 / pp.29-31 gaps are Investment Metrics' own against the 9/30 workbook
(per-deal first lien, CoC, realized IRR -- the workbook's sold 17.7% and 5.6% are typed
on its Total row); not re-litigated here. Scripts (scratchpad): im_vs_deck.py, bv_run.py.

## p.23 exposure by asset class (Oct 8 2026; live `v605`)
Total and PSC net preferred equity INCLUDING unfunded, by class: the PE exposure engine's
funded Cost + remaining to fund per deal (PSC by its split), grouped by MRI `Asset_Type`
through `board_views_service.ASSET_CLASS_OF` -- a deck grouping (Multifamily, Non-Grocery
Retail, Grocery-Anchored Retail, Self Storage, Other = Industrial / RV Park / Resort). An
Asset_Type the map does not name is its OWN row with a note, never absorbed.

MRI on Oct 8 2026: Burton is now `Retail - Grocery`; Merle Hay and 5-15 Broad were changed
to `Retail - Non-Grocery` -- a SECOND spelling beside the 23 deals' `Retail - Non Groc.`
(both map; MRI should settle on one). `Self Storage` / `Self-Storage` likewise.
`Property_Count` NOT yet changed; `Investment_Strategy` still blank on all 199.

Production, 12/31/25 ($M):

| Class | Deck total / PSC | App total / PSC | Why |
|---|---|---|---|
| Multifamily | 427.9 / 63.8 | 441.3 / 65.2 | the deck filed Brainerd's $13.3M unfunded under Non-Grocery (a deck error) |
| Non-Grocery Retail | 90.9 / 18.7 | 77.6 / 17.4 | the same, the other way: +13.3 ties it exactly |
| Grocery-Anchored | 92.8 / 11.0 | 92.8 / 11.1 | ties |
| Self Storage | 70.3 / 20.5 | 69.8 / 20.4 | Apple Bales, Middle Island over-funded (accounting Q) |
| Other | 9.6 / 6.2 | 10.8 / 6.9 | Pontchartrain -2.25, Bearfoot +0.92, Lynch +2.49 (accounting Q) |
| Total | 691.6 / 120.2 | 692.4 / 120.9 | = PE engine grand total, to the dollar |

## p.28 portfolio metrics (Oct 8 2026; live `v605`)
Schedule key `debt`. Deals = Investment Metrics' Current table at the as-of (as p.26).
- Occupancy / DSCR: One Pager `get_property_performance` per deal (YTD economic occupancy,
  YTD DSCR) via the Snapshot's memoised provider; rolled up with the Snapshot's own
  functions -- `_weighted` (NOI) and `_debt_weighted` (debt from `resolve_debt`); development
  deals (`config.is_dev_deal` on Investment_Strategy or Lifecycle) are out, named.
- Debt: `dashboard_service.get_loan_maturity_data` (facility amounts, fixed / floating; NULL
  rate type counts fixed -- its rule). Buckets: years from the as-of to maturity on the
  CALENDAR (<=3, <=6, >6). Caps: NEW `flask_app/services/loan_caps.py` (`vIntRatereset`, then
  the `vHedgedStrat` text for strike / expiry; unreadable text shown raw). One-engine row added.
- Loans are MRI's today; a loan repaid since the as-of is missing (noted on the page).

Production vs the deck (metrics as of 9/30/25):

| | Deck | App |
|---|---|---|
| Multifamily / Grocery occ, DSCR | 90% 1.4 / 96% 2.1 | identical |
| Non-Grocery / Self Storage | 88% 1.5 / 84% 2.1 | 93% 1.6 / 87% 2.7 |
| RV Park / Industrial | 52% 0.7 (annual avg) / 100% 2.0 | 66% 1.3 (YTD) / 100% 1.3; Resort (Old Kinderhook) its own row |
| Portfolio | 90% 1.6 | 90% 1.7 |
| Fixed total | 1,477.5 | 1,478.2 |
| Fixed 0-3 / 4-6 / 6+ | 502.2 / 615.2 / 360.2 | 393.1 / 814.1 / 271.0 -- no bucket rule reproduces the deck (tested six) |
| Capped <=2.5% / >2.5% | 56.8 / 48.8 | 20.1 / 55.3 (Nottingham 36.7 now fixed in MRI; Poplar Prairie 6.5 not in the deck) |
| No cap | 356.1 | 403.0 (JB Fair 77.4 and Town Fair Tire 20.0 are Variable in MRI, absent from the deck) |
| Cap not readable | -- | Mount Prospect 6.0 ("5.00% for $5.3M, 5yr $825k") |

## p.24 exposure by operating partner (Oct 8 2026; live `v605`)
p.23's figures grouped by MRI `Operating_Partner` through `board_views_service.PARTNER_NAMES`
-- SPELLING variants only, to the deck's short names (JPI / JPI Companies, Vastgood / Vastgood
Properties LLC, Apple / Apple Self Storage...); the same names print in pp.29-30's partner
column (display only). Deals = MRI deals (Jim's rule), so the deck's transaction counts
(Apple 7, Berger 8, total 61) are not this column. Production 12/31/25: totals = PE engine
(692.4 / 120.9); 14 partners tie the deck to $0.1M; the rest differ by WHO the deck assigns a
deal to -- MRI vs deck: Brainerd / Crowne ("Bertram and DiMarco", "Bertram/Pyramid" vs
"Bright Ravens"), JB Fair Park ("Dave West" vs "L. Allen"), The Gallery ("Manhattan Five" vs
"MFP"); and Pegasus 44.9 vs 12.5, JPI 99.9 vs 109.2, PMAT 6.1 vs 32.7, LBX PSC 5.5 vs 11.8 --
redistribution, since the grand total ties. Not mapped; AM's call.

## The package (Oct 8 2026; live `v605`)
Jim asked for: page numbers that match what prints; a FULL and an ABBREVIATED package; attachments in
narrative sections; narratives that run onto more pages at natural breaks; and editable footnotes and
disclosures on every page.
- **Structure** (`BoardDeck.vue`): cover, table of contents, then each PART (`board_service.PARTS`,
  the January deck's I-V; every schedule and narrative names its `part`) -- a divider page, then its
  pages in catalog order. ABBREVIATED = schedules with a view; FULL adds narratives with text or
  attachments. A part with nothing in it is not printed.
- **Page numbers are positions** (index + 1) after pagination; the contents prints the same numbers.
- **Narratives** (`narrative.ts`): text syntax -- blank line = paragraph, "-" lines = bullets, "## " =
  sub-heading; attachments follow the text. Blocks are MEASURED in a hidden copy of the page body
  (global `bd-narr` styles) and packed; a paragraph breaks at a sentence end, a list between items; the
  last page keeps room for the section's footnotes; continuation pages titled "(cont'd)".
- **Attachments** (`board_package_service`): images, or PDFs rendered to one PNG per page at 150 dpi
  (PyMuPDF), max 30 pages / 25 MB; type by the file's BYTES; unreadable files refused with the reason.
  Tables `board_attachments`, `board_attachment_pages`.
- **Footnotes / disclosures**: drawn by the frame (`BoardSlide`) on every page. Defaults come with the
  view; `board_page_notes` row per (meeting, page key) -- `footnotes` NULL = defaults, a list REPLACES
  them, [] = none; reset deletes the row. Editor panel under the page, live preview. Narrative notes show
  on the section's last page.
- **Print / PDF**: every page drawn at 1100 x 825 by the same `DeckPage` component, `@page` sized to the
  canvas; the button waits for all figures and images.
- Logo: the official file (`brand.md`).
- Guardrails: `board_package_check` 28, `board_access_check` 217 (the five new routes, every role),
  `board_views_check` 101. All `board_*` tables are now PROTECTED.

### Future-funding split, measured Oct 7 2026 (production data at 12/31/25, nothing changed)
The engine splits a deal's unfunded commitment by its holder's funded shares (commitment
ratios); a multi-holder deal is left unsplit. Against the 12/31/25 tracker, by deal:
- **Brainerd** is the only deal the tracker splits differently: PSC 1.33M / TIAA 11.98M
  (all unfunded is owed by the TIAA JV: INVBPA funded its whole 5,493,264). Engine: PSC 3.41M.
- **A -- split by unfunded commitment (committed - net funded from the books) at every
  level**: Brainerd exact; Giant 7 drifts (a shared fund's fund-wide unfunded); **Bel Air
  WRONG** (PSC 0.61 -> 0.41, KoC 0.72 -> 1.16) -- PSC1 has FUNDED MORE THAN IT COMMITTED
  at I1BAS2 (2.35M committed, 8.88M funded, carrying PIG6) and PIG5 (0.54M / 2.11M,
  carrying CFCNI), so "who still owes" is not who funds the deal.
- **B** (A, but commitment ratios at shared funds) and **C** (B, plus ratios wherever an
  investor over-funded): Giant 7 fixed; Bel Air still off (C: PSC 0.65, KoC 1.01, F&F 0.76
  vs 0.61 / 0.72 / 1.11) -- uneven funding inside PIG6 / PIG5 shifts the weights.
- **D -- unfunded split ONLY at an entity carrying an allocation override** (remaining =
  committed - the override's funded amount); ratios everywhere else, as today. Reproduces
  the tracker on every deal: Brainerd via its two overrides, all others unchanged.
  RECOMMENDED. With no overrides entered, D changes nothing.
Brainerd overrides -- ENTERED on production Oct 7 2026 after v594 (ids 1, 2), D is live;
the funded and future splits now match the tracker. Investment BRNERD, effective
2024-12-27, the last capital movement:
PPIBPA <- INVBPS 9,316,074.29 / TGA22 9,091,603.11; INVBPS <- INVBPA 5,493,264.01 / TGA22
3,822,810.30 (books: contributions less return of capital).
Scripts (scratchpad, not in repo): ff_measure.py, ff_measure_c.py, belair_chain.py.
