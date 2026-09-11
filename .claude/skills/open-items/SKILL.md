---
name: open-items
description: Triage the project's open work queue — show what is still open, brief one item on request, and close it out when it ships. Use when the user asks what is outstanding, what to work on next, or names an open item (7076, ROE_Income, Realized Gain, quarter dropdown, Berger loans, pro-rate, Pegasus TGA22, Excel serials, JWT rotation). Also the weekly open-items review.
---

# Open items — triage

The queue is `.claude/memory/open_items.md`. Read it, then follow this.

## Rule zero: verify before you present

Items go stale. That is why this queue exists — the last set lived in session narratives
and five of them had already been fixed by the time anyone read them. **Run each item's
stated check before showing it as open.** If a check now passes, say so and offer to close
it; do not present it as work.

Line numbers in the queue are hints. They drift. Grep for the symbol, not the line.

## Default output: the triage table, nothing more

One line per open item. No explanation, no history. Group by §1 code / §2 decisions /
§3 data-ops. Columns: **item · the number that makes it matter · owner · next action**.

Then one line: *"Say a number for a brief, or 'detail N' for the full entry."*

Keep it under 25 lines. If more than that is open, show §1 and §2 and say how many §3
items are waiting.

## On request: brief one item — six lines, hard limit

1. **What's wrong** — one sentence, plain.
2. **Evidence** — the check you just ran, and what it returned.
3. **Why it matters** — the dollar figure or the deal count. No adjectives.
4. **Recommended fix** — the shape, not the code.
5. **The trap** — the thing that makes the obvious fix wrong. Most items have one; if
   there is none, say "none known" rather than inventing one.
6. **How you'd verify** — what proves it worked.

Stop there. Offer detail; do not volunteer it.

## Only if asked: full detail

Point at the §-section in `open_items.md`, and at `session_log_aug2026.md` for the original
diagnostics and numbers. Quote what is relevant; do not paste whole sections.

## Blocked items are not work

§2 items are waiting on a human answer, not on code. Present the **question**, the options
and what each is worth — never a recommendation dressed as a fact. §2.1 (what metric U/W
ROE is meant to be) gates most of the other U/W ROE items; surface that first and say so.

## Closing one out

When an item ships or is dismissed, edit `open_items.md`:

- **Move it to §4 Resolved** with the commit SHA and one line on what was done. **Never
  delete it** — a reader needs to know the question was answered, not that it vanished.
- If it was decided rather than fixed, record the decision, who made it, and the date, and
  mark it *do not re-raise* the way the At-Close entry is.
- Update the verified-as-of date on anything else you re-checked while you were in there.
- Commit the queue change with the fix, not separately.

## Before a deploy

If an item is about to ship, apply the standing rule in CLAUDE.md → Deploying Changes:
review the diff for symptom repair, and check the span against the **running image**
(`git log <live-sha>..<target>`), not against local HEAD. That distinction cost a review on
v429 — three investor-facing commits shipped unreviewed because only the local span was
checked.
