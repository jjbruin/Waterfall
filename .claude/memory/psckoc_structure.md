# PSC KOC I LLC structure — RECOVERED from the Belair net-returns PDF (Sep 1 2026)

## Why this file exists
Jim asked whether the prior session that read the PSC KOC LLC agreement and prepared
waterfall steps still exists. **It does not.** What survives and what does not:

- **GONE**: the PSCKOC waterfall step rows. There is **no PSCKOC waterfall anywhere** —
  not in Azure PG, not in local `waterfall.db`, and **no waterfall row in the entire table
  pays KCREIT or PCBLE** (checked all 104 vcodes). They were entered into a database that
  has since been replaced, and were never committed as code or a script.
- **GONE**: the agreement reading itself. `.claude/memory/` has no PSCKOC step table, and
  searchable session transcripts only reach back to ~Aug 25 2026 (no hit for "PCBLE").
- **SURVIVED**: the *engine* built from that reading — commit `85492cd` (Feb 11 2026) added
  the `AMFee` and `Promote` (cumulative catch-up) vStates specifically to express this
  agreement, plus `promote_base` tracking. `Amt` and `IRR` followed. So the vocabulary the
  agreement needs is all present and battle-tested; only the step rows are missing.
- **CONSEQUENCE**: PSCKOC's member returns today fall back to the `relationships`
  pro-rata (KCREIT 85 / PSC1 15 / **PCBLE 0**), so **PCBLE's carry is not modeled at all**
  and no PSCKOC-level fee or catch-up is being charged.

## The structure, recovered from a primary source
`~\OneDrive\Documents\KOC Belair Net Returns 2.14.2025.pdf` prints its own
**"Investment Structure"** legend. Verbatim terms:

| Term | Value | The PDF's own note |
|---|---|---|
| Investor (KofC) contribution to pref equity | 85.0% | |
| PSC contribution to pref equity | 15.0% | |
| Coupon | **8.0%** | "Priority cash flow distributed Pari-Pasu to Investor and PSC" |
| Asset Management Fee | **1.50%** | "Applied annually to invested capital, **subordinated to Coupon rate, accrued when not paid**" |
| Catch-up | **20.0%** | "Estimated based upon this investment's contribution to the PSC/KofC JV" |
| Startup Costs | **$15,000** | "Venture expense, Pari-Pasu" |
| Annual Costs | **$15,000** | "Venture expense, paid first from Distributable Cash Flow" |

Ownership rows confirm the members: KCREIT 85%, PSC1 15%, PCBLE 0% (carry units).

## Direct mapping to existing vStates — no new engine work
| Agreement term | vState | Encoding |
|---|---|---|
| $15,000 annual venture expense, paid first | `Amt` | iOrder 1, `mAmount` 15000/yr, recipient the expense code. Same shape as TGA22's quarterly `Amt`. |
| 8% coupon, pari-passu 85/15 | `Pref` | `nPercent` 0.08, `FXRate` .85 lead / .15 Tag |
| Return of capital, pari-passu | `Initial` + `Tag` | .85 / .15 (Cap_WF) |
| 1.50% AM fee on invested capital, subordinated, **accrued when not paid** | `AMFee` **with `;accrue`** | `nPercent` 1.50, `mAmount` periods/yr, `vNotes` = source;accrue. **This is exactly the modifier shipped in v401 today** — the agreement's accrual language and that feature are the same thing. |
| 20% catch-up | `Promote` | `FXRate` = carry share, `nPercent` 0.20, `vNotes` = capital investors |

## "Fully caught up in the promote" — Jim's definition (Sep 1 2026)
> "when evaluating the deal net returns we are making the deal stand on its own and not
> factoring in the history and performance of the portfolio as a whole."

This lines up with the Belair PDF's own catch-up note ("**Estimated based upon this
investment's contribution to the PSC/KofC JV**"): the catch-up is a portfolio-level
concept *estimated at the deal level*. So a deal-level net-returns run is **standalone** —
`promote_base` / `promote_carry` accumulate from this deal's cash flows only, with no
cross-deal carry-forward.

**This is what the engine already does when it runs one deal.** No `gates_satisfied` run
flag and no gate-skipping is needed — the earlier plan's Phase 2(b) was solving a problem
that does not exist. Keep the `Promote` step as written and run the slice standalone.

## Precedent worth knowing: Belair was already sliced three ways
The same PDF splits one deal across **three** JVs — PSC TGA 2024 LLC 51.3%, PSC I/F&F
15.6%, PSC/KofC JV 33.1% — of capital account units 25,827,976 at close / 35,453,000 at
full funding. The capability Jim is asking for is exactly this, and the multi-relationship
stack model already expresses it.

## Still needed
The Belair legend is a **derived** source. Before these steps are saved as the permanent
PSCKOC waterfall, confirm against the executed PSC KOC I LLC agreement: whether the AM fee
base is gross invested capital or net of ROC, the fee's compounding on accrual, the
catch-up's precise target definition, and the startup-cost treatment. The agreement PDF is
**not** in the searched OneDrive paths (only financial statements, PCAPs and invoices) —
Jim to point at it, or confirm the legend is authoritative.

## Other net-returns precedents in the same folder (output format references)
`KOC Apple Net Returns 9.26.22.pdf`, `KOC Ascent Net Returns 3.10.22.pdf`,
`KOC Belair Net Returns 2.14.2025.pdf`, plus the TIAA series (Windsor Square 9.1.2026,
Berger Glenmoore, Life Storage, Mt Prospect, Post Commons, Court at Deptford).
The KOC layout differs from the TIAA one: it prints the Investment Structure legend and a
combined "PSC Asset Mgmt Fee & Catch-up" line. **The one-pager needs both variants.**
