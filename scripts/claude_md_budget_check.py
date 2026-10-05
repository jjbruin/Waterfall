"""Guardrail: CLAUDE.md stays a rule book, not an archive.

CLAUDE.md is loaded into EVERY session, so every line in it is paid for whether or not
the work needs it. It had grown to 4,133 lines -- 2,790 of them a per-revision deploy
narrative, and roughly a thousand more of per-feature detail that belongs beside the
feature. On Oct 5 2026 all of that moved to `.claude/memory/` and the file came back to
under 350 lines.

THE FAILURE THIS EXISTS TO CATCH IS GRADUAL AND NOBODY'S FAULT. No single paragraph is
wrong to add; the file simply never gives anything back. The deploy history is the
clearest case -- each entry was written by somebody being careful, and the sum was a
2,790-line block that every session carried and no session read.

TWO RULES, and they are different in kind:

  1. A LINE BUDGET. Blunt on purpose. A budget that negotiates is not a budget, and
     any rule subtle enough to distinguish "worth its space" from "not" is a rule
     this check cannot apply.

  2. NO REVISION SUFFIX (`vNNN`) OUTSIDE THE LESSONS LIST. This is the shape the
     archive grew back in, and it is the one that reads as current when it is not:
     "live at `v504`" is a fact that expires silently. The Lessons list is the one
     place a revision is allowed, because there it is a POINTER into
     deploy_history.md and not a claim about what is running.

WHY A PATTERN AND NOT A WORD LIST: `v16` (PostgreSQL) and `Vue 3` must not match, so
the pattern is three or more digits -- the suffixes run v349 upward. A two-digit
version is left alone deliberately.

The deploy command block names its suffix `<next-suffix>` rather than a literal, which
is both clearer (nobody copy-pastes a stale number) and what keeps rule 2 honest
without an exemption for code fences.

Run:      .venv\\Scripts\\python.exe scripts\\claude_md_budget_check.py
Inject:   .venv\\Scripts\\python.exe scripts\\claude_md_budget_check.py --inject=lines
          .venv\\Scripts\\python.exe scripts\\claude_md_budget_check.py --inject=stamp
          -- each re-creates a defect in a COPY and asserts this check fails on it,
          so "it passes" is never confused with "it looks".
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAUDE_MD = os.path.join(ROOT, 'CLAUDE.md')

#: The budget. Raising this is a decision, not a convenience -- the point of the
#: number is that it has to be argued for.
#:
#: 350 -> 330 -> 340. The first compaction set it at 350 with the file at 349, which is
#: no headroom at all: the next legitimate rule would have had to raise the limit to
#: land, and a limit raised reflexively is not a limit. It went to 330 against a file of
#: 316, then to 340 on Oct 5 2026 when an audit found the compaction had cut the REASONS
#: off four standing rules -- the date-stamp rule lost both of its worked examples, ONE
#: NUMBER lost two sentences of Jim's own words, and the `queries/` UNION ALL rule had
#: no pointer at all. Restoring them cost 20 lines and was worth every one: a rule
#: without its reason is a rule people argue with.
#:
#: 340 -> 372 on Oct 5 2026, PINNED EXACTLY TO THE FILE. The rebase onto main carried
#: `v565` across: its deploy entry went to deploy_history.md, its three ONE NUMBER rows
#: stayed in the table, and the PE Exposure section became pe_exposure.md with every
#: RULE kept here as a bullet, and the same for its Market rates section. That is 32
#: lines of rules, not of history.
#:
#: 372 -> 371 when the compaction's own transient archive was deleted and CLAUDE.md's
#: pointer row went with it. RE-PINNED rather than left slack, which is the whole
#: point: a budget one line above the file is a budget that has already been spent.
#:
#: AT EXACTLY THE FILE LENGTH there is no headroom, deliberately: the next line added
#: has to be argued for, and raising this number is that argument. Do not nudge it to
#: buy room -- find something that has stopped being a rule and move it.
MAX_LINES = 371

#: A revision suffix: `v349` .. `v563` and onward, optionally with a trailing letter
#: (`v556r`). Three digits minimum so `v16` and `Vue 3` cannot match.
REVISION = re.compile(r'\bv\d{3,}[a-z]?\b')

#: Revisions are allowed only under this heading, where they are pointers into
#: deploy_history.md rather than claims about what is live.
LESSONS_HEADING = '### Lessons'

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


def section(t):
    print('\n--- ' + t)


def lessons_span(lines):
    """(first, last) 0-indexed line numbers of the Lessons list, or None.

    Runs from the heading to the next heading at the SAME OR SHALLOWER depth, so a
    `####` inside Lessons would stay inside it and a following `##` ends it.
    """
    start = None
    for i, line in enumerate(lines):
        if line.rstrip() == LESSONS_HEADING:
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        m = re.match(r'(#{1,6})\s', lines[j])
        if m and len(m.group(1)) <= 3:
            return (start, j - 1)
    return (start, len(lines) - 1)


def audit(text):
    """Return (n_lines, [(lineno, revision, line)]) for revisions outside Lessons."""
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    span = lessons_span(lines)
    strays = []
    for i, line in enumerate(lines):
        if span and span[0] <= i <= span[1]:
            continue
        for m in REVISION.finditer(line):
            strays.append((i + 1, m.group(0), line.strip()))
    return len(lines), strays, span


def run(text, label):
    n, strays, span = audit(text)
    section(label)
    chk('CLAUDE.md is at most %d lines' % MAX_LINES, n <= MAX_LINES, '%d lines' % n)
    chk('the Lessons list is present, so revisions have somewhere legal to live',
        span is not None)
    chk('no revision suffix (vNNN) outside the Lessons list',
        not strays,
        '%d stray: %s' % (len(strays),
                          '; '.join('L%d %s' % (l, v) for l, v, _ in strays[:4]))
        if strays else '')
    for lineno, rev, line in strays[:8]:
        print('        line %d  %s  %s' % (lineno, rev, line[:90]))
    return not BAD


if not os.path.exists(CLAUDE_MD):
    print('  FAIL  CLAUDE.md is not in this tree')
    sys.exit(1)

src = open(CLAUDE_MD, encoding='utf-8').read()

inject = None
for a in sys.argv[1:]:
    if a.startswith('--inject='):
        inject = a.split('=', 1)[1]

if inject:
    # Prove the check is non-vacuous: break a COPY in memory and assert it fails.
    # Nothing is written to disk.
    if inject == 'lines':
        broken = src + '\nfiller\n' * (MAX_LINES + 10)
        expect = 'the line budget'
    elif inject == 'stamp':
        # Edits a line in place rather than adding one, so this injection isolates
        # rule 2 -- it must fail the vNNN check and PASS the line budget.
        broken = src.replace('## Domain invariants',
                             '## Domain invariants, live at `v504`', 1)
        expect = 'the vNNN rule'
    else:
        print('unknown injection: %s' % inject)
        sys.exit(2)
    run(broken, 'INJECTED (%s) -- this check MUST fail' % inject)
    failed = len(BAD)
    print('\ninjection "%s" produced %d failure(s); %s is live.'
          % (inject, failed, expect))
    sys.exit(0 if failed else 1)

run(src, 'CLAUDE.md')

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
    print("""
CLAUDE.md holds invariants and procedures only. WHERE THE CONTENT BELONGS INSTEAD:

  a deploy note -- what a revision shipped, what broke, what was measured
      -> .claude/memory/deploy_history.md   (never CLAUDE.md)
  the detail behind a calculation rule -- columns, account sets, fallbacks
      -> .claude/memory/engine_reference.md
  anything about ONE feature -- treasury, leases, the budget review, GL/IA query,
  section access, workpapers, expenses, intercompany
      -> that feature's own file; the pointer table at the foot of CLAUDE.md
         lists every one of them by topic
  work in flight, with an owner and a date stamp
      -> .claude/memory/open_items.md
  which function answers which question
      -> .claude/memory/function_index.md

If you are adding a genuine RULE and the file is full, something already in it has
stopped being a rule -- find that and move it, rather than raising MAX_LINES. The
budget is only worth having if it is occasionally inconvenient.""")
sys.exit(1 if BAD else 0)
