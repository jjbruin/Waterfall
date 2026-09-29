"""Reading the One Pager print population, with the quarter supplied, not baked in.

WHY THIS IS SHARED. Three scripts read `onepager_print_population.txt` and each
parsed it itself. Two of them carried their OWN `"2026-Q2"` fallback, so the
quarter was pinned in three places at once and a sweep could quietly render a
finished quarter while the comparison still looked clean. One reader now, and
the quarter is an argument.

A LEGACY TWO-COLUMN LINE STILL WINS. An older population file names its own
quarter per row, and honouring it is what stops this change from silently
re-pointing a file somebody is mid-way through using. `--quarter` fills in only
where the line does not say.

It REFUSES rather than guessing: a bare vcode with no `--quarter` is an error
naming the file, not a default. A guessed quarter renders a document that looks
entirely correct and is for the wrong period.
"""
from __future__ import annotations

import re
import sys

QUARTER_RE = re.compile(r"^\d{4}-Q[1-4]$")


def read_population(path: str, quarter: str | None, stream=sys.stderr):
    """[(vcode, quarter)] from a population file, or None when it cannot be read.

    Returns None (rather than raising) so callers can `return 2` and keep their
    own exit conventions.
    """
    if quarter is not None and not QUARTER_RE.match(quarter):
        print(f"--quarter {quarter!r} is not a quarter (expected YYYY-Qn)",
              file=stream)
        return None

    out, bare = [], 0
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) > 1 and QUARTER_RE.match(parts[1]):
                    out.append((parts[0], parts[1]))     # legacy line: it wins
                else:
                    bare += 1
                    out.append((parts[0], quarter))
    except OSError as exc:
        print(f"could not read {path}: {exc}", file=stream)
        return None

    if bare and quarter is None:
        print(f"{bare} line(s) in {path} name no quarter and --quarter was not "
              f"given. Pass --quarter YYYY-Qn; the population file deliberately "
              f"no longer pins one.", file=stream)
        return None
    return out
