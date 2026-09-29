# Intercompany — design, and phase 1 (Sep 29 2026)

**Phase 1 BUILT Sep 29 2026, NOT DEPLOYED**: the reconciliation grid, checks, drilldown,
per-entity account settings and comments — `intercompany_service.py`, `api/intercompany.py`,
`IntercompanyView.vue`, Accounting → Intercompany. Guardrail `scripts/intercompany_check.py`
(53): a fixture plus the CFO's own GEXD rows reproduced to the cent, proved non-vacuous against
four injected defects (basis filter dropped, opening row lost, drilldown not narrowed to the
entity, the sheet's pay-0-when-short rule). **Phase 2, the Pay step and the JE file, is not
built.** New tables `ic_entity_settings` (seeded once, from the CFO's answers) and
`ic_recon_notes`, both in `PROTECTED_TABLES`.

Built in passing: `gl_ia_query_service._freshness` looked for refresh state `'complete'`,
which nothing writes (the refresh writes `'done'`), so the GL / IA Query screen has said
"freshness unknown" after every refresh since `v504`. Fixed there; this screen shares it.

The CFO asked for a new Accounting tab to manage intercompany journal entries. Source:
`BORG_Intercompany Template.xlsx` (Jim's OneDrive). What follows is read off that file and
measured against production, not inferred from its sheet names.

## What the package contains

| Tab | What it is |
|---|---|
| `DTM Recon Template` | **The mock-up.** Due to/from PSC Manager reconciliation, one row per entity |
| `JE Template` | MRI GL upload CSV — the reimbursement entries the reconciliation produces |
| `NEW Due to Manager GEXD` | Spreadsheet Server query: every entity, BASIS `A.B`, periods `202601.202612` |
| `New Manager Due to Interco GEXD` | Same query for `PSCMAN` only |
| `New Intercompany GEXD` | Same query, every entity — **byte-identical rows to the first**; used by nothing visible |
| 12 hidden tabs | Older versions of the same process (2023–24 summaries, pivots, a PNC cash lookup) |

All three queries are `NEW JOURNAL and GHIS Balance Forward.edq` — **the query
`queries/MRI_GL_Detail.sql` already replicates**, into `gl_detail`. Their `DEFINED_CODE` /
`DESCRIPTION` are our `RLTDENTITY` / `RLTDENTITY_NAME`. `USER_DEFINED_VAL1` is a line key and
is not used. So no new MRI query is needed: the three become three filters on one table.

## The reconciliation, as the mock-up defines it

Inputs: entity-side account `MR15000002` (Due To/From PSC Manager), manager `PSCMAN`,
manager-side account `MR15000001` (Due to/from Intercompany), tolerance $1.

Per entity:
- **A** entity balance = Σ AMT, ENTITYID = entity, ACCTNUM = `MR15000002`
- **B** alternate account, if the entity uses one. CFO, Sep 29 2026: **PEGASU uses
  `MR99991102` Due To/From Property Manager**, and there are no others. The sheet had
  PEGASU on `MR99982002`, which is wrong for it (see below). NOTTNV keeps `MR99982002`
  Accounts Payable - Other (-29,905.07), confirmed Sep 29 2026. Those two are the whole list.
- **C = A + B**
- **D** manager balance = Σ AMT, ENTITYID = `PSCMAN`, ACCTNUM = `MR15000001`,
  RLTDENTITY = entity
- **Variance = C + D**, which should be 0 (entity owes = negative, manager is owed = positive)
- **Status**: No Balance (C = D = 0) / Reconciled (|variance| ≤ tolerance) / Investigate
- **Cash balance** — typed by hand ("MR1*")
- **Amount to pay** = D if D < cash, else 0; typed over where needed; "Pay" creates the JE

Checks: entity total per source vs the rows (an entity missing from the list), manager total
per source vs the rows (a segment missing), manager balance with a BLANK segment, count to
investigate.

## MEASURED: our tables reproduce it to the cent

Production `gl_detail`, PERIOD 202601–202609, **BASIS A and B only**:

| | CFO's sheet | `gl_detail` | entities differing |
|---|---|---|---|
| Entity `MR15000002` | -308,316.59 | -308,316.59 | 0 of 61 (and 0 of 274 entity/period/balfor keys) |
| PSCMAN `MR15000001` by segment | 313,093.97 | 313,093.97 | 0 of 64 |
| NOTTNV `MR99982002` | -29,905.07 | -29,905.07 | — |

**The basis filter is load-bearing.** `gl_detail` also carries BASIS `C` (4,622 rows in 2026)
and `T` (54). Without the filter the same sums come out -160,919.94 and 101,804.92. The
screen must filter `A.B` explicitly and say so; never "all bases".

**Cash: 61 of 63 typed balances equal the GL cash accounts** (`MR1000*`) at period end. The
two that do not:
- **PSC1** 12,794,339.93 = PNC + Wells Fargo MM, **excluding Liberty Bank MM** 1,000,024.66.
  So which cash accounts count is a per-entity choice, not a prefix.
- **PEGASU** 313,530.29 is **`MR99991000` Cash - Bank** (CFO, Sep 29 2026) — a property-style
  cash account outside `MR1000*`. Measured: A 299,555.29 + B 13,975.00 = 313,530.29, exact.
  (Its BASIS C row repeats the A figure, one more reason the A.B filter is not optional.)

**PEGASU's alternate account moves its row.** On the sheet's `MR99982002` it reads 0 and
shows No Balance. On `MR99991102` it carries **-94,941.70** (A -14,889.51, B -80,052.19)
against a manager balance of 0, so it becomes **Investigate** — a real variance the sheet
could not see.

## What the app does differently from the sheet, on purpose

- **The entity list is derived, not typed.** Every entity with a `MR15000002` balance or
  movement, plus every manager segment. The sheet's first two checks exist only because
  column B is hand-maintained; here they become structural. The **blank-segment** check stays —
  it is a real data condition (4 blank-segment rows on PSCMAN today, net 0).
- **Cash is read from the GL**: `MR1000*` by default, a per-entity list of excluded cash
  accounts (PSC1 → Liberty MM) and of added ones (PEGASU → `MR99991000`). Where Treasury has the entity's bank account, its carried ledger is shown beside
  it; neither is typed.
- **Mock-up defects not carried over**: `C2` (as-of) is `#REF!`; `M76` is `#REF!`; INVF10's
  variance reads `=G18+L11` — another row's CASH — so it shows Investigate on a zero balance.
  That makes the sheet's "4 to investigate" really 3: NOTTNV -22,385.07, PSC2 -2,467.66,
  PPI2 -274.16.
- **A JE that has been generated but is not yet in the GL is shown as such.** Between
  downloading the file and the next MRI refresh the balance still reads unpaid, so without this
  the same reimbursement can be paid twice. Generated batches are stored.

## Screen

Accounting → **Intercompany** (`/intercompany`), below Treasury.

1. **Header**: as-of period (YYYYMM), accounts shown read-only with their names, tolerance,
   "GL as of the last MRI refresh <time>", basis A.B stated.
2. **Checks strip** first, above the grid: blank-segment manager balance, count to
   investigate, net variance, generated-not-posted batches.
3. **Grid**: the mock-up's columns A–N, sort/filter on any column (the GL / IA Query grid's
   pattern). Clicking A, B or D opens the GL lines behind it — the statement drilldown's
   pattern, same row selection as the figure so it always reconciles.
4. **Per row, stored**: comment, alternate account, excluded cash accounts, amount-to-pay
   override. Keyed by (period, entity) so they survive a refresh; the standing ones (alternate
   account, excluded cash) carry to the next period.
5. **Pay**: tick rows → preview of the JE lines → download the MRI CSV.

## The journal entry

Exactly `JE Template`'s shape, which is **the same MRI GL upload format Treasury already
writes** (`treasury_upload.GL_COLUMNS`, byte-for-byte the same ten columns). Reuse
`build_gl_csv`; do not write a second writer. Per paying entity, four lines:

```
<entity>  MR15000002   +amt   Intercompany Reimbursement to PSC Manager
<entity>  <its cash>   -amt   Intercompany Reimbursement to PSC Manager
PSCMAN    MR10005000   +amt   Intercompany Reimbursement from <entity>
PSCMAN    MR15000001   -amt   Intercompany Reimbursement from <entity>   RLTDENTITY=<entity>
```

Period, Basis `B`, and ENTRDATE **typed by the accountant — no default** (CFO, Sep 29 2026).
A batch with no entry date is refused.

**CAD entities (PPI2, PSC2) are reimbursed on PSC Manager's USD intercompany balance**
(CFO, Sep 29 2026). So the amount is D, the manager's `MR15000001` figure, never the
entity's own balance. Their cash is the TOTAL of `MR10006000` Cash - Canada (PNC) and
`MR10003000` Cash - Canada (CFO, Sep 29 2026).

Measured 202609, and it raises points the CFO has not yet seen:
- **PPI2 also holds USD**: `MR10005000` 23,136.50 — exactly the figure the sheet typed as its
  cash. Its CAD total is 344,688.53 (all in `MR10006000`; `MR10003000` is 0). The CFO's rule
  shows 344,688.53 CAD where the sheet showed 23,136.50 USD.
- **PSC2 holds no cash in any GL account** — `MR10005000` 0, no CAD rows. On any rule it can
  afford 0 of its 6,793.60.
- A JE credits ONE cash account. With two CAD accounts, which one the entry credits is still
  to settle (default `MR10006000`, where the money is, editable per row).
- The CAD total cannot be compared with a USD amount owed without a rate, and the app holds
  no FX rate. So for these two rows "what it can afford" is shown, not computed. Both are among the sheet's three real Investigates (PPI2
-274.16, PSC2 -2,467.66); whether that variance is FX or a genuine difference is NOT known —
do not label it FX until someone has looked. **`validate_gl` needs one addition: balance PER ENTITY.** Today it
checks the whole entry and only *warns* on several entities — a batch that balances overall
could leave one entity out of balance, which MRI rejects or, worse, posts.

**What gets paid** (CFO, Sep 29 2026): every row with a balance owed — Reconciled or
Investigate — is PROMPTED with what it can afford, `min(D, cash)`, and the accountant decides.
They may pay LESS than the prompt, never more. This replaces the sheet's rule (pay D in full
if cash covers it, otherwise 0), which paid nothing at all to an entity 1 dollar short.
An Investigate row carries its variance beside the prompt, since D is the manager's figure and
the entity's own balance disagrees with it. CAD rows: prompted with D and the CAD cash shown,
not capped (see above).

Refused: more than the prompt, a paying entity with no cash account, a missing entry date, a
row already in an unposted batch.

## Access

Writes `roles_exactly(*ACCOUNTING_ROLES)`; reads open, as the rest of the section. Routes
enumerated by `accounting_access_check.py` automatically.

## Guardrail (when built)

Rebuild the CFO's 9/28 figures from a fixture of his GEXD rows: the two totals above, the
per-entity table, and the JE CSV **byte-identical** to `JE Template` for the same selections —
the treasury `v492` method. Plus: the BASIS C/T rows present in the fixture and excluded; the
INVF10 case reads No Balance.

## Questions for the CFO

**Answered Sep 29 2026** (via Jim): Pegasus cash is `MR99991000`; alternate accounts are
PEGASU `MR99991102` and NOTTNV `MR99982002`, nothing else; PPI2/PSC2 are reimbursed on PSC
Manager's USD interco balance from the total of their two CAD cash accounts; the entry date is
a manual entry; every row is prompted with what it can afford and the accountant may pay less.
Folded in above.

Still open:

1. **The third query** (`New Intercompany GEXD`) returns the same rows and nothing visible uses
   it. Is a general intercompany reconciliation — every `MR15*` pair between entities, e.g.
   PSC1, PSS1, PPI2LP, AMB25 all carry `MR15000001` — the next phase? Phase 1 is the manager.
2. **PPI2's USD account** (`MR10005000` 23,136.50) — ignored under the CAD rule, or counted?
3. **PSC2 has no cash in the GL** — is it paid from somewhere else?
4. **Which CAD account the JE credits** when an entity holds both.
5. **Minimum balance** — should "what it can afford" leave a reserve?
6. **Description** — fixed wording as in the template?
