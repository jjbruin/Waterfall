# TGA6 + AMB6 waterfalls vs the executed LLC agreements (Sep 1 2026)

Sources read in full:
- **TGA6**: `OneDrive\Documents - Peaceable Accounting Team\a - Reporting Entities\2. Entity
  Documents\TGAVI\A&R LLC agreement for PSC TGA VI LLC 542026.docx.pdf` (67 pp, Docusign
  EE72F539-7D31-81EB-802A-BB56A2230A92)
- **AMB6**: `OneDrive\Documents - Peaceable Street Capital\Capital Campaign\Ambassadors Fund
  VI\PSC Ambassadors Fund TGA VI LLC - EXECUTED - Amended and Restated LLC Agreement.pdf`

**Verdict: the modeled waterfalls did NOT match either agreement. Six discrepancies, four
material.**

**STATUS (Jim, Sep 1 2026): TGA6 REBUILT to the agreement and live on Azure. AMB6 NOT
changed — §8.4(b) cannot be expressed in waterfall rows (see "AMB6 remains open" below).**

## Entity identities (settles a question we got wrong)
- TGA6 **"PSC"** = **PSC Investment TGA VI LLC** = the app's **INV6**.
- TGA6 **"PSC Manager"** = **PSC Manager LLC** = the app's **PSCMAN**.
- AMB6 **"PSC Sub"** = PSC Investment TGA VI LLC = **INV6**; **"PSC TGA"** = **TGA6**.
- AMB6 Promote Receipts are defined as amounts INV6 receives from TGA6 **"on account of PSC
  Sub's ownership of Promote Units"** — so **INV6 holds the Promote Units**, not PSCMAN.

## TGA6 — the agreement
Preferred Return: **9% per annum, compounding annually on DECEMBER 31**, average daily
balance of Adjusted Capital Contribution plus compounded unpaid pref, actual days.
(Contrast PSCKOC, which compounds 9/30.)

**§6.02 Distributions of Available Cash** (operating -> CF_WF), quarterly:
(a) unpaid Default Preferred Return · (b) **unpaid Preferred Return** · (c) Adjusted Default
Capital Contributions · (d) **20% to PSC holding Promote Units / 70% TIAA / 10% PSC holding
Capital Units**

**§6.03 Distributions of Capital Transaction Proceeds** (Cap_WF), quarterly:
(a) Default pref · (b) **unpaid Preferred Return** · (c) Adj. Default Capital Contributions ·
(d) **Adjusted Capital Contributions** (ROC) · (e) **20% Promote / 70% TIAA / 10% PSC Capital**

There is **no IRR hurdle anywhere in Article VI.**

## TGA6 — what is modeled (Azure, as deployed)
| | modeled | agreement |
|---|---|---|
| CF tie 1 | `Amt` TGA6_EXP 1,875/qtr | not in the agreement (see below) |
| CF tie 10 | TGAM Share .90 / INV6 Tag .10 | **§6.02(b) pref first, then 20/70/10** |
| Cap tie 10 | ROC TGAM .90 / INV6 .10 | §6.03(d) — but **after** (b) pref |
| Cap tie 20 | **`IRR` TGAM @ 9%** .90 / INV6 .10 | **NOT IN THE AGREEMENT** |
| Cap tie 30 | TGAM .70 / **PSCMAN .04** / **INV6 .26** | §6.03(e) TGAM .70 / **INV6 .30** |
| tie 900 | `AMFee` PSCMAN 0.95% on TGAM | §5.06 — rate not yet verified against the text |

### Material discrepancies
1. **No Preferred Return tier in either waterfall.** §6.02(b) and §6.03(b) both put unpaid
   9% pref ahead of everything else; the model has no `Pref` step at all. This is the single
   biggest gap. It also explains the Phase 0 finding: the source Excel *does* run the pref
   (its "Investor Due / Paid / Accrued" and "PSC Due / Paid / Accrued" rows), which is why
   the Excel's "Available to Share" is 0 every year and the app's is not.
2. **The `IRR` step at Cap tie 20 has no basis in the agreement.** Article VI contains no
   IRR hurdle. It was presumably added to approximate the Excel's promote formula (XNPV at
   9% compounded to exit x 20%) — but the *agreement's* promote is a straight 20/70/10 of
   the residual after pref and capital, with no hurdle. **Neither the app nor the Excel
   implements §6.03(e) literally**, and they differ from each other.
3. **CF_WF gives TIAA 90% of operating residual; the agreement gives 70%** (with 20% promote
   + 10% PSC capital). Does not bite on Windsor only because the pref would absorb all
   operating cash — but the pref is missing too, so today the model over-allocates operating
   cash to TIAA and pays PSC no operating promote.
4. **Cap tie 30 splits PSC's 30% as PSCMAN .04 / INV6 .26.** The agreement gives the whole
   30% to **PSC = INV6** (20 promote + 10 capital); PSCMAN's compensation is the §5.06 fee,
   not a promote share. **Jim's original instruction (Sep 1) was TGAM .70 / INV6 .30 and was
   correct**; the .04/.26 split was my suggestion, on the reasoning that AMB6 could not tell
   promote dollars from capital dollars. **That reasoning is wrong — see AMB6 below.**

Minor / to confirm:
5. `Amt` TGA6_EXP 1,875/qtr ($7,500/yr) is in the source Excel's structure legend as
   "Annual Costs — venture expense, paid first from Distributable Cash Flow", but I have not
   located it in the agreement text. §5.07 Expenses needs reading before this stays.
6. The 0.95% AMFee rate needs checking against §5.06 Fees (page 35 of the PDF).

## AMB6 — the agreement
**§8.4 Distributions of Available Cash**, quarterly, splits Available Cash into **two
separately-defined streams**:
- **(a) Available Cash – Investment Receipts** (everything that is not Promote Receipts) ->
  **Class A Members** pro rata on Adjusted Percentage Interests.
- **(b) Available Cash – Promote Receipts** (what INV6 receives from TGA6 on its Promote
  Units) -> **Class B Members and NON-Peaceable Class A Members** in the **Promote
  Percentages** at Exhibit C.

No Preferred Return. No promote catch-up. **§5.2 Quarterly Manager Fee: 0.5% per annum of
total aggregate Capital Contributions made by ALL Members**, measured on the last day of each
quarter, paid within 5 days.

Exhibit B (members, capital, class):
| member | app id | class | capital | % |
|---|---|---|---|---|
| PSC Manager LLC | PSCMAN | **Class B, 1 unit** | **$0** | Class B 100% |
| Peaceable Street Capital LLC | PSC1 | Class A | 4,700,000 | 42.73% |
| CCGS Investco | CCGSI | Class A | 2,250,000 | 20.45% |
| Impala Real Estate Partners | IREP | Class A | 1,000,000 | 9.09% |
| CWS Investment Partners | CWSPART | Class A | 500,000 | 4.55% |
| JJ&C Investments | JJCI | Class A | 500,000 | 4.55% |
| Atlas Malabar | ATLAS | Class A | 300,000 | 2.73% |
| Sheira Schacter, Batten, Ancora, + 4 more | SHEIRA/BATTEN/ANCORA/… | Class A | 250,000 ea | 2.27% ea |
| **total** | | | **11,000,000** | 100% |

**The model's 13 Class A percentages match Exhibit B exactly** (PSC1 .427273, CCGSI .204545,
IREP .090909, CWSPART/JJCI .045455, ATLAS .027273, seven at .022727). That part is right.

### Material discrepancies
1. **§8.4(b) is not implemented at all.** AMB6 has one pro-rata tier for both CF and Cap;
   the agreement requires promote receipts to be split on a completely different basis from
   investment receipts. **My earlier claim that AMB6 "receives a pooled inflow and cannot
   distinguish promote from capital dollars" is contradicted by the agreement** — the two
   streams are defined terms and the Manager is required to determine the split. The
   limitation is in the app, not the structure.
2. **PSC1 is a *Peaceable* Class A Member** ("any Member owning Membership Units that is an
   Affiliate of PSC Manager LLC") and §8.4(b) pays promote only to **Class B** and
   **NON-Peaceable Class A**. So **PSC1 is entitled to NO promote receipts** — today it
   receives its full 42.73% of everything, promote included. This directly overstates the
   PSC1 consolidated return shipped in v401/v402.
3. **The promote split is a 44-band sliding scale (Exhibit C)**, keyed to aggregate capital
   contributions made by non-Peaceable Class A Members, running from 1.82%/98.18% at
   $250,000 to 80.00%/20.00% at $11,000,000. Non-Peaceable Class A capital is
   **$6,300,000** of the $11,000,000 (obligation basis), which lands on the **$6,250,000
   band: ~45.45% to non-Peaceable Class A / ~54.55% to PSCMAN (Class B)**.
   Jim's stated convention — "20% to PSCMAN and the remaining 80% shared according to the
   capital contribution ratio" — is close to the **inverse** of the agreement at current
   capital, and the agreement's split is not fixed: it moves every time capital is raised.
   **CAVEAT**: Exhibit C keys off contributions *made*; Exhibit B gives *obligations*. If
   less than $6.3M is funded the band is lower. Needs the funded figure to be exact.
4. **Manager Fee base.** §5.2 is 0.5% on aggregate **Capital Contributions made** — a
   contributed-capital base that does not decline as capital is returned. The engine's
   `AMFee` charges `total_capital_outstanding`, which **does** fall with ROC. Same rate,
   different base. Also: the 13 per-member `AMFee` rows sum to 0.5% of total capital, which
   is economically equivalent to the single fee the agreement describes — that construction
   is fine.
5. **The `;accrue` modifier I added to AMB6's 26 AMFee rows (v401) is not supported by the
   agreement text** — §5.2 makes the fee payable within 5 days of quarter end with no
   subordination or accrual language. Jim instructed the accrual behaviour explicitly
   ("otherwise the fee accrues in a similar way to the way we do it with the other
   investment joint ventures"), so this is a question for him, not an error.

## Recommended order if these are to be fixed
1. Add the 9% `Pref` step to TGA6 CF_WF and Cap_WF (compounding 12/31 — the engine default).
2. Decide §6.03(e) vs the Excel's XNPV promote. They are different rules; the agreement is
   the authority but the Excel is what TIAA has been sent. **This supersedes the pending
   `IRRPromote` vState recommendation from Phase 0** — do not build it until this is settled.
3. Correct Cap tie 30 to TGAM .70 / INV6 .30 and delete the PSCMAN .04 row.
4. Correct CF residual to 20/70/10.
5. Implement AMB6 §8.4(b): a promote-receipts stream, PSC1 excluded, Exhibit C percentages.
   This needs the engine to carry a *provenance tag* on distributions (which upstream tier
   they came from) — the one genuinely new engine capability in this list.
6. Re-check the PSC1 consolidated return after (5); it is currently overstated.


## APPLIED — TGA6 rebuilt to the agreement (Sep 1 2026, live on Azure)

Written through `save_waterfall_steps` (audit trail); `dteffective` restored afterwards for
the known NULL defect. App validator: **no errors** on either waterfall (one warning, "Pref
FXRate 0.9 typically 1.0", which is the same shape TGA22 uses).

| | CF_WF | Cap_WF |
|---|---|---|
| tie 1 | `Amt` TGA6_EXP 1,875/qtr (§5.07) | same |
| tie 10 | **`Pref` TGAM .90 @ 9% + Tag INV6 .10** (§6.02(b)) | **same** (§6.03(b)) |
| tie 20 | **`Share` TGAM .70 + Tag INV6 .30** (§6.02(d)) | `Initial` TGAM .90 + Tag INV6 .10 (§6.03(d)) |
| tie 30 | — | **`Share` TGAM .70 + Tag INV6 .30** (§6.03(e)) |
| tie 900/901 | **`AMFee` PSCMAN 0.95% on TGAM AND on INV6** (§5.06) | same |

Removed: the `IRR` step (no basis in Article VI) and PSCMAN's .04 promote share.
Added: the 9% Pref tier on both waterfalls, and the second AMFee row — §5.06 charges 0.95%
on *the Members'* contributions, so the single TGAM row was collecting only 90% of the fee.

### Windsor rerun — before vs after
| participant | contrib | dists before | dists after | IRR before | IRR after |
|---|---|---|---|---|---|
| TGAM (TIAA) | 17,235,000 | 27,332,134 | **27,182,512** | 12.281% | **12.076%** |
| Fund Investors (12) | 1,096,773 | 2,186,092 | **2,369,858** | 18.992% | **21.917%** |
| PSC1 | 818,227 | 2,772,789 | **2,829,609** | 64.960% | **72.828%** |
| PSCMAN | 0 | 912,895 | **832,620** | n/a | n/a |

Tier totals over the hold, all tying to the agreement's ratios:
pref 7,118,803 (TGAM 6,406,923 / INV6 711,880 = 90/10) · CF residual 378,517 (70/30) ·
ROC 19,150,000 (90/10) · Cap residual 4,650,537 (70/30) · expenses 37,500 ·
AM fee 784,745 (was 693,782 — the missing 10% now charged).
Total distributed 31,335,357 against 31,357,858 of deal cash into the vehicle. Balances.

**TGAM annual-bucket IRR: 11.568% -> 11.383%.** The Excel/TIAA one-pager prints **11.740%**,
so conforming to the agreement moved the app *further* from the Excel (0.172pp below ->
0.357pp below). Expected: the agreement's §6.03(e) is a flat 20/70/10 with no hurdle, the
Excel takes 20% of the XNPV excess above 9%. **Three rules, and only one can govern the
investor-facing page — still Jim's call.**

### Two reporting artifacts introduced (cosmetic, not economic)
1. `TGA6_EXP` now shows in the participants table as a line with 37,500 of "distributions".
   It is an expense sink, not a participant, and should be filtered out of that table.
2. `psc_summary.total_promote` now reads **0**, because it keys off
   `vtranstype == 'Promote Split'` rows paid to PSCMAN, and the promote correctly flows to
   INV6 now. The promote is real (INV6's 1,395,161 at the Cap residual) — the label is wrong.

## AMB6 remains open — and PSC1's return is still overstated
§8.4(b) requires promote receipts to be split on a different basis from investment receipts
(Class B + non-Peaceable Class A on the Exhibit C sliding scale, **PSC1 excluded**). That
needs the engine to tag a distribution with the upstream tier it came from — provenance the
waterfall row format cannot carry. **Not attempted.** So PSC1's 72.83% still includes
promote dollars it is not entitled to under the agreement, and PSCMAN's Class B promote
share is still missing. Correcting it will move PSC1 down and PSCMAN up.
