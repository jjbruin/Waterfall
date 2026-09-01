# PSC KOC I LLC — waterfall from the executed agreement (Sep 1 2026)

Source of truth: `OneDrive\Documents - Peaceable Street Capital\Asset Mgmt\3. Investors\
1. KoC\A. Closing files\1. PSC KOC I LLC Agreement.pdf` (72 pp). Read Sep 1 2026.
Cross-checked against `~\Documents\KOC Belair Net Returns 2.14.2025.pdf`, whose printed
"Investment Structure" legend agrees on every rate.

## Prior-session status
The earlier session that read this agreement did **not** survive: there is **no PSCKOC
waterfall in Azure PG or local `waterfall.db`, and no waterfall row anywhere pays KCREIT
or PCBLE** (all 104 vcodes checked). So today PSCKOC's members split on bare `relationships`
percentages (KCREIT 85 / PSC1 15 / **PCBLE 0**) with **no fee, no pref and no catch-up**,
and PCBLE's carry is entirely unmodeled. What survived is the engine built from that
reading — commit `85492cd` (Feb 11 2026) added `AMFee` and `Promote` for this agreement.

## Section 6.02 — Distributions of Available Cash (quarterly, within 10 business days)
| Tie | Agreement | Basis | vState |
|---|---|---|---|
| (a) | Aggregate Unpaid **Default** Preferred Return | pro rata, Capital Units | `Pref` (defaulting-member edge case; skip unless a default occurs) |
| (b) | Aggregate Unpaid **Preferred Return** | pro rata to unpaid pref | `Pref` |
| (c) | Adjusted **Default** Capital Contributions | pro rata | `Initial` (edge case) |
| (d) | **Adjusted Capital Contributions** | pro rata | `Initial` |
| (e) | **80% Carry / 20% Capital** until Carry has received 20% of aggregate 6.02(b) + 6.02(e) + 6.06(d)(ii) | Carry pro rata by Carry Units; Capital pro rata by **Capital Units** | `Promote` |
| (f) | **20% Carry / 80% Capital** residual | same | `Share` + `Tag` |

**The catch-up formula ties exactly to the engine.** Let P = pref distributions (6.02(b))
and E = carry distributions (6.02(e)). The agreement's test `E = 20% x (P + E)` solves to
`E = 0.25P`, which is precisely `Promote`'s `E >= target/(1-target) x promote_base` with
target 0.20 — and it is why `promote_base` accumulates on `Pref` steps. The memory note
"PSCKOC's catch-up is unique — do not template it" is now fully explained.

## Section 6.06 — the per-deal branch, and what "fully caught up" actually means
On the sale of an entire Company Investment, after 6.02(a)-(c): compute the **Hypothetical
Sale Amount** (FMV of all Remaining Assets + Remaining Capital Transaction Proceeds), then
test whether a hypothetical 6.02(d) distribution of that amount would reduce the KOC
Member's Adjusted Capital Contributions to zero.
- **Not sufficient** -> normal portfolio-basis 6.02(d)-(f).
- **Sufficient** -> **6.06(d) instead**: (i) ROC pro rata on Adjusted Capital Contributions
  **"calculated solely with respect to Capital Contributions and Distributions allocable to
  the Sale Assets"** — i.e. PER DEAL; (ii) 80/20 catch-up; (iii) 20/80 residual.

**So Jim's "make the deal stand on its own and not factor in the history and performance of
the portfolio as a whole" IS Section 6.06(d).** It is the agreement's own mechanism, not a
modeling shortcut: the portfolio-level test is assumed passed, and the consequence is a
per-deal distribution. A deal-level net-returns run should be modeled on the 6.06(d) branch,
and the one-pager can cite it. Confirms that a standalone single-deal run is correct and
that the `gates_satisfied` run flag in the first draft plan is unnecessary.

## Section 5.06 — Management Fee (differs from the Belair legend; read this carefully)
- **1.5% per annum of total Capital Contributions made by the KOC MEMBER only** — not of
  both members' capital. Paid **quarterly in arrears at 0.375%** of KOC contributions
  through the last day of the quarter, prorated for partial quarters.
- Fee base is **reduced** by (A) KOC contributions not yet deployed into an Investment, and
  (B) KOC contributions for an Investment that has been sold or written off, in whole or
  part (pro rata for a partial sale).
- Treated as a **guaranteed payment** (Code s.707(c)); a Company obligation, not a
  waterfall tie.
- Encoding: `AMFee`, `nPercent` 1.50, `mAmount` 4, **`vNotes` = KCREIT** (source), and
  `;exclude:` for realized investments. `capital_outstanding` already falls as `Initial`
  ROC is paid, which covers most of (B) naturally.
- **DISCREPANCY to resolve with Jim**: the agreement makes the fee an off-the-top quarterly
  payable; the Belair model treats it as *"subordinated to Coupon rate, accrued when not
  paid."* Those differ in a cash-short quarter. The Belair treatment is the more
  conservative modeling convention and maps to the **`;accrue` modifier shipped in v401**.
  Ask which governs the net-returns presentation.

## Preferred Return definition — NOTE THE COMPOUNDING DATE
> 8% per annum, **compounding annually on September 30, 2018 and on September 30 of each
> subsequent year**, on the average daily balance of each Capital Unit's Adjusted Capital
> Contribution plus previously-compounded unpaid pref, actual days, cumulative to the
> extent not distributed under 6.02(b).

**The app compounds pref on 12/31** (InvestorState with a 45-day grace;
`build_pref_balance_detail` at 12/31). PSCKOC compounds on **9/30**. This is a real engine
gap: the compounding month must become configurable per waterfall/entity before a PSCKOC
net-returns page can be trusted. Day count is average-daily-balance on actual days.

## Adjusted Capital Contribution
Per Capital Unit: that Member's Capital Contributions (excluding Default Capital
Contributions) less amounts previously distributed under 6.02(c) [sic — economically the
ROC tie] and 6.06(d)(i). Contribution allocable to a unit = total contributions / number of
units held.

## THE TWO RATIOS — Jim's 90/10 nuance, and why it matters mechanically
Founding ratio was **85/15** (s.4.02(a): KOC contributed 85% of the Exhibit B asset values,
PSC 15%). **New deals go in at 90/10 while older deals retain 85/15** (Jim, Sep 1 2026), so
the blended Capital Unit count drifts with every new investment. The agreement accommodates
this through Schedule A, amended as capital is contributed.

The consequence is that **one PSCKOC waterfall run uses two different ratios**:
- **Pref (6.02(b)) and ROC (6.02(d) / 6.06(d)(i))** are pro rata to each member's own
  *Adjusted Capital Contributions* -> the **per-deal** ratio. Seeding KCREIT/PSC1 at 90/10
  for a new deal makes these correct automatically; no FXRate is involved in the arithmetic.
- **Catch-up and residual (6.02(e)/(f), 6.06(d)(ii)/(iii))** are pro rata to the **number of
  Capital Units held** -> the **blended portfolio-wide** ratio at the time of distribution,
  which is NOT the deal's own ratio and changes as deals are added.

`Share`/`Tag`/`Promote` FXRates are static numbers in the `waterfalls` table, so the blended
ratio would silently go stale. Options, cheapest first:
1. **Derive it at run time** from a capital-unit register (member x deal x contribution),
   and let the residual steps read it. Matches the agreement's own Schedule A mechanism and
   cannot go stale. Requires a small register table + an FXRate sentinel meaning "pro rata
   by capital units".
2. Store the blended ratio on the waterfall and update it on every new deal (a maintenance
   burden; a stale ratio misallocates silently — not recommended, but it is the zero-code
   option).
Either way this is a **new tracking requirement** the plan did not previously carry.

## Other terms
- Venture expenses: Belair legend shows **$15,000 startup (pari-passu)** and **$15,000
  annual, paid first from distributable cash flow** -> `Amt` at the top tie. The agreement
  (s.5.07) allocates Company Expenses pro rata across PSC Vehicles; the $15k is the
  practical per-deal estimate.
- Members: KCREIT (Capital Units, the KOC Member), PSC1 (Capital Units), **PCBLE (Carry
  Units, 0% capital)**.
- Tax Distributions (s.6.03) on Carry Units are advances that reduce future 6.02(e)/(f) and
  6.06(d)(ii)/(iii) carry distributions — not modeled; note only.
- Belair precedent: that deal was **already split across three JVs** (PSC TGA 2024 51.3% /
  PSC I/F&F 15.6% / PSC/KofC 33.1%), so the slice capability has precedent in Jim's output.

## Open with Jim
1. Fee presentation: agreement's off-the-top quarterly payable, or Belair's
   subordinated-and-accruing treatment?
2. Blended-unit-ratio tracking: build the capital-unit register (option 1) or hand-maintain
   the FXRates (option 2)?
3. Pref compounding on 9/30 needs the engine change — confirm it applies to all PSCKOC
   deals including the new 90/10 vintage.
