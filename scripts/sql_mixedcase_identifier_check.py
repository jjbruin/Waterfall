#!/usr/bin/env python
"""Guardrail: raw SQL must double-quote mixed-case column names.

WHY THIS EXISTS. PostgreSQL folds an unquoted identifier to lower case. The MRI-derived
tables were created by pandas `to_sql`, which QUOTES, so their columns really are
`dtValuation`, `vCode`, `Vcode`, `mAmount`. Unquoted SQL therefore looks for
`dtvaluation` and fails with `column "dtvaluation" does not exist`.

SQLite is case-insensitive and hides this completely. That asymmetry is the trap: the
valuation publish path worked on every local run and had NEVER ONCE SUCCEEDED against
Azure, because nothing exercised it there until someone tried to publish a real
valuation. The unit of failure is a production feature that looks tested.

CLAUDE.md already records the rule for iOrder / PropCode / FXRate on the prospective-loan
tables. This makes it checkable instead of remembered.

Scans SQLAlchemy text() blocks in the app for identifiers matching the project's
naming conventions, and fails on any that is not double-quoted.

Run:  python scripts/sql_mixedcase_identifier_check.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = (".claude", "worktrees", ".venv", "node_modules", ".git")

# The conventions actually used by the MRI-derived tables:
#   dtValuation vCode vSource vAccount mAmount nRate fCapRate iOrder  -> prefix + Capital
#   Vcode Date Pro_Yr Amt Typename                                    -> Capitalised words
PREFIXED = re.compile(r'(?<!["\w])((?:dt|v|m|n|f|i)[A-Z][A-Za-z_]+)(?!["\w])')
NAMED = re.compile(r'(?<!["\w])(Vcode|Pro_Yr|InvestmentID|InvestorID|PropCode|FXRate|'
                   r'LoanID|EndDate|StartDate|MajorType|Typename|TypeID|EffectiveDate)'
                   r'(?!["\w])')

# Words that look like identifiers but are SQL/param/text, not columns.
IGNORE = {"nValue", "vValue"}


def sql_blocks(text_src: str):
    """Yield (line_offset, sql) for every triple-quoted text() SQL block."""
    for m in re.finditer(r'text\(\s*"""(.*?)"""', text_src, re.S):
        yield text_src[: m.start()].count("\n") + 1, m.group(1)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    problems = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            if any(s in rel for s in SKIP_DIRS):
                continue
            try:
                src = open(path, encoding="utf-8").read()
            except Exception:
                continue
            scanned += 1
            for base_line, sql in sql_blocks(src):
                # Strip SQL comments so prose in a `--` line is not flagged.
                body = re.sub(r"--[^\n]*", "", sql)
                found = set()
                for rx in (PREFIXED, NAMED):
                    found |= {g for g in rx.findall(body) if g not in IGNORE}
                if found:
                    line = base_line + body[: body.find(sorted(found)[0])].count("\n")
                    problems.append((rel, line, sorted(found)))

    print(f"scanned {scanned} python files")
    if not problems:
        print("\nPASS — every mixed-case identifier in raw SQL is double-quoted.")
        return 0

    print(f"\nFAIL — {len(problems)} SQL block(s) use an UNQUOTED mixed-case identifier.")
    print("On PostgreSQL these resolve to lower case and raise "
          '`column "..." does not exist`. SQLite will not show you this.\n')
    for rel, line, ids in problems:
        print(f"  {rel}:{line}")
        print(f"      {', '.join(ids)}")
    print('\nFix: wrap each in double quotes — "dtValuation", "vCode", "Vcode".')
    return 1


if __name__ == "__main__":
    sys.exit(main())
