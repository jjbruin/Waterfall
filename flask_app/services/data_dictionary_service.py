"""Data traceability — the field dictionary and the dependency map, read-only.

WHAT THIS IS FOR. The assistant could always fetch a VALUE; it had no way to say
where the value came from. Asked "what uses the ISBS balance sheet?" on
2026-09-21 it answered from "general knowledge of how ISBS balance sheet data is
typically wired in real estate investment platforms" — a confident sentence with
no source behind it, and wrong by omission (it named the cap stack and missed
DSCR principal, the Property Financials principal line, the dashboard KPI,
Surveillance and the NAV liability split).

So this module serves two committed reference files and nothing else:

    flask_app/reference/data_dictionary.json   what a field means, and its bases
    flask_app/reference/dependencies.json      source -> consumers, blast radius

READ-ONLY, AND DELIBERATELY NOT IN data_service._cache. That cache is cleared on
every MRI refresh and on every CSV import; these files are part of the image, not
part of the data, and re-reading them on a refresh would be work done for no
reason. They are loaded once per process into the module-level singletons below.

THE FILES ARE THE ONLY AUTHORITY. Nothing here computes, infers or falls back to
a default answer: a field that is not in the dictionary returns a miss with
suggestions, never a guess. The assistant is instructed to say "no verified
source" on a miss rather than fill the gap from its own knowledge — the whole
point of the module is that an ungrounded answer is worse than no answer.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Repo-relative, the same shape as ``mri_service._get_queries_folder`` uses for
#: ``queries/``: this file is ``flask_app/services/x.py``, so parent.parent is
#: ``flask_app/``. Shipped by ``COPY flask_app/ flask_app/`` (Dockerfile:35).
_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

#: Lazy singletons — loaded once per process, never invalidated. See module docstring.
_DICT: Optional[dict] = None
_DEPS: Optional[dict] = None


def _load(name: str) -> dict:
    """Read one reference file. Raises on missing/invalid — callers convert to a miss."""
    path = _REFERENCE_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dictionary() -> dict:
    global _DICT
    if _DICT is None:
        _DICT = _load("data_dictionary")
        logger.info("data_dictionary.json loaded: %d fields",
                    len(_DICT.get("fields", [])))
    return _DICT


def _dependencies() -> dict:
    global _DEPS
    if _DEPS is None:
        _DEPS = _load("dependencies")
        logger.info("dependencies.json loaded: %d sources, %d constants",
                    len(_DEPS.get("sources", {})),
                    len(_DEPS.get("shared_constants", {})))
    return _DEPS


# ── field dictionary ──────────────────────────────────────────────────────

def field_ids() -> list:
    """Every field id in the dictionary, for the tool's closed enum.

    Built from the FILE rather than written out in assistant_service, so the
    enum cannot drift from the dictionary it indexes. Fails soft: a missing or
    unreadable file leaves the enum off the schema rather than breaking import
    of the whole assistant.
    """
    try:
        return [f["field_id"] for f in _dictionary().get("fields", [])
                if f.get("field_id")]
    except Exception:
        logger.exception("field_ids() could not read the dictionary")
        return []


def _tab_of(field_id: str) -> str:
    """The tab segment of a ``tab.field`` id, or "" for an unqualified id."""
    fid = str(field_id or "").strip()
    return fid.rsplit(".", 1)[0].lower() if "." in fid else ""


def _bare_of(field_id: str) -> str:
    """The field segment of a ``tab.field`` id."""
    return str(field_id or "").strip().rsplit(".", 1)[-1].lower()


def _find_fields(field_id: str, tab: Optional[str] = None) -> list:
    """Every dictionary entry matching ``field_id``, most specific first.

    THE ID IS ``tab.field`` AND A BARE NAME IS AMBIGUOUS ON PURPOSE. "debt" is
    three different entries — one_pager.debt (which a development deal rebases
    onto hard costs), snapshot_financial.debt and snapshot_loan.debt (which do
    not, and must not). Returning a LIST rather than picking one is what stops
    the assistant answering for the wrong tab: with several hits it is told to
    show the variants, not to guess.

    ``tab`` narrows by the id prefix or by ``appears_on``. An unknown tab
    narrows to nothing, and the caller reports that rather than silently
    widening back to every tab.
    """
    fields = _dictionary().get("fields", []) or []
    want = str(field_id or "").strip().lower()
    if not want:
        return []

    exact = [f for f in fields
             if str(f.get("field_id", "")).strip().lower() == want]
    if exact:
        return exact

    bare = _bare_of(want)
    hits = [f for f in fields if _bare_of(f.get("field_id")) == bare]

    if tab:
        t = str(tab).strip().lower()
        hits = [f for f in hits
                if _tab_of(f.get("field_id")) == t
                or t in [str(a).strip().lower()
                         for a in (f.get("appears_on") or [])]]
    return hits


def _source_lines(entry: dict) -> tuple:
    """(source_line, source_value) for one field entry.

    ASSEMBLED FROM ``inputs``, NEVER INVENTED. A field with exactly one input
    has one source and says it outright. A field with several — ROE has four,
    economic occupancy six — has no single source, and collapsing them into one
    sentence would mean choosing which components to drop. So the line points at
    the Inputs section instead and every component keeps its own source there.
    """
    inputs = entry.get("inputs") or []
    if len(inputs) == 1 and inputs[0].get("source"):
        value = str(inputs[0]["source"])
    elif inputs:
        value = (f"{len(inputs)} components, each with its own source — "
                 f"see Inputs")
    else:
        value = "no source recorded in the dictionary"
    return f"Source: {value}", value


def _field_payload(entry: dict) -> dict:
    """One dictionary entry, flattened for the tool.

    ``formula_latex`` IS PASSED THROUGH VERBATIM and is never built here. The
    file is the only authority on how a formula is written; code that assembled
    LaTeX from a field name would be a second, unreviewed definition of the
    arithmetic — the exact failure the dictionary exists to prevent.
    """
    source_line, source_value = _source_lines(entry)
    return {
        "field_id": entry.get("field_id"),
        "display_label": entry.get("display_label"),
        "section": entry.get("section"),
        "tab": _tab_of(entry.get("field_id")),
        "appears_on": entry.get("appears_on"),
        "status": entry.get("status"),
        "definition": entry.get("definition"),
        "source_line": source_line,
        "source_value": source_value,
        "formula": entry.get("formula"),
        "formula_latex": entry.get("formula_latex"),
        "formula_is_arithmetic": bool(entry.get("formula_is_arithmetic")),
        "inputs": entry.get("inputs") or [],
    }


def lookup_field(field_id: str, tab: str = None, basis: str = None) -> dict:
    """What a field means, how it is computed, and where each input comes from.

    ``basis`` is the retired pre-2026-09-24 argument and is accepted only so an
    in-flight tool call from the old schema does not raise; it is treated as a
    tab hint. The dictionary no longer carries bases — it carries one entry per
    (tab, field), each decomposed into ``inputs``.
    """
    try:
        matches = _find_fields(field_id, tab or basis)
    except Exception as exc:
        logger.exception("lookup_field: dictionary unavailable")
        return {"error": f"Field dictionary unavailable: {exc}"}

    if not matches:
        ids = field_ids()
        bare = _bare_of(field_id)
        near = difflib.get_close_matches(str(field_id or "").strip().lower(),
                                         ids, n=5, cutoff=0.4)
        if not near:
            near = [i for i in ids if bare and bare in i.lower()][:5]
        return {
            "error": (f"No field '{field_id}'"
                      + (f" on tab '{tab}'" if tab else "")
                      + " in the dictionary."),
            "did_you_mean": near,
            "known_field_ids": ids,
        }

    if len(matches) == 1:
        return _field_payload(matches[0])

    # SEVERAL TABS REPORT THIS NAME AND THEY DO NOT AGREE. Surfaced as variants
    # rather than resolved here: which one applies is a property of the question
    # ("on the One Pager", "for a dev deal"), and the dictionary has no basis on
    # which to prefer one. The assistant is instructed to show them all.
    variants = [_field_payload(m) for m in matches]
    tabs = ", ".join(v["tab"] or "?" for v in variants)
    return {
        "field_id": _bare_of(field_id),
        "multi_tab": True,
        "definition": (f"'{_bare_of(field_id)}' is reported on {len(variants)} "
                       f"tabs ({tabs}) and is NOT the same figure on each — "
                       f"which applies depends on the tab."),
        "source_line": "Source: varies by tab — see each variant below.",
        "source_value": "varies by tab — see each variant below",
        "variants": variants,
        "tabs": [v["tab"] for v in variants],
    }


# ── dependency map ────────────────────────────────────────────────────────
#
# THE TWO COLLECTIONS ARE LISTS KEYED BY ``name`` as of 2026-09-24. They were
# objects keyed by a long prose string; a list of records reads the same in the
# file and stops the key doubling as data.

def _by_name(entries) -> dict:
    """{name: entry} from the new list form, tolerating the old dict form."""
    if isinstance(entries, dict):
        return dict(entries)
    out = {}
    for e in entries or []:
        if isinstance(e, dict) and e.get("name"):
            out[str(e["name"])] = e
    return out


#: Words that carry no discriminating power in a "what uses X" question.
_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to",
              "from", "data", "table", "field", "value", "app", "is", "it",
              "what", "breaks", "if", "i", "change", "depend", "depends",
              "many", "things", "how", "uses", "use"}


def _tokens(text: str) -> set:
    """Lowercase alphanumeric tokens, minus stopwords and 1-character noise."""
    out, cur = set(), []
    for ch in str(text or "").lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.add("".join(cur))
            cur = []
    if cur:
        out.add("".join(cur))
    return {t for t in out if len(t) > 1 and t not in _STOPWORDS}


def _match_key_strict(name: str, entries: dict) -> Optional[str]:
    """Exact, case-insensitive, then substring either way. No fuzzy pass.

    Run across sources AND constants before any token scoring, so a caller that
    names a constant outright — ``DEBT_BS_ACCTS`` — gets the constant and not
    the source whose prose happens to mention it.
    """
    want = str(name or "").strip()
    if not want:
        return None
    keys = list(entries.keys())
    for k in keys:
        if k.lower() == want.lower():
            return k
    for k in keys:
        if want.lower() in k.lower() or k.lower() in want.lower():
            return k
    return None


def _match_key(name: str, entries: dict) -> Optional[str]:
    """Best token-overlap match, or None below the confidence floor."""
    q = _tokens(name)
    if not q:
        return None
    best, best_score = None, 0.0
    for k, entry in entries.items():
        blob = k + " " + json.dumps(entry, ensure_ascii=False)
        key_hits = len(q & _tokens(k))
        score = (len(q & _tokens(blob)) + key_hits) / (2.0 * len(q))
        if score > best_score:
            best, best_score = k, score
    return best if best_score >= 0.25 else None


def _matching_notes(deps: dict, want: str) -> dict:
    """known_divergences / known_defects mentioning ``want``."""
    out = {}
    for key in ("known_divergences", "known_defects"):
        hits = [d for d in (deps.get(key) or [])
                if want and want in json.dumps(d, ensure_ascii=False).lower()]
        if hits:
            out[key] = hits
    return out


def impact(name: str) -> dict:
    """What reads a source or constant, and what moves if it changes.

    ``name`` may be a source table, a shared constant, a known divergence or
    defect, or a field. A field is answered backwards — which sources feed it —
    because "what breaks if I change the debt field" is really a question about
    the field's inputs.
    """
    try:
        deps = _dependencies()
    except Exception as exc:
        logger.exception("impact: dependency map unavailable")
        return {"error": f"Dependency map unavailable: {exc}"}

    sources = _by_name(deps.get("sources"))
    constants = _by_name(deps.get("shared_constants"))
    want = str(name or "").strip().lower()

    # An outright name — source or constant — wins before any fuzzy matching.
    strict_source = _match_key_strict(name, sources)
    strict_const = _match_key_strict(name, constants)
    key = None if (strict_const and not strict_source) else (
        strict_source or _match_key(name, sources))

    # 1. a source table
    if key:
        entry = sources[key]
        feeds = list(entry.get("feeds") or [])
        count = entry.get("consumer_count")
        if count is None:
            count = len(feeds)
        out = {
            "lead": (f"{key} feeds {count} consumer(s) across the app."),
            "source_line": f"Source: dependencies.json — sources[{key!r}]",
            "matched": key,
            "match_type": "source",
            "blast_radius_note": entry.get("blast_radius_note"),
            "consumer_count": count,
            "feeds": feeds,
            "consumers": feeds,
            "loads_to": entry.get("loads_to"),
            "provides": entry.get("provides"),
            "traps": entry.get("traps"),
        }
        out.update(_matching_notes(deps, want))
        return out

    # 2. a shared constant
    key = _match_key_strict(name, constants) or _match_key(name, constants)
    if key:
        entry = constants[key]
        read_by = list(entry.get("read_by") or [])
        changes = list(entry.get("fields_that_change") or [])
        count = entry.get("consumer_count")
        if count is None:
            count = len(read_by) + len(changes)
        out = {
            "lead": (f"{key} (defined at {entry.get('defined_at')}) is read in "
                     f"{len(read_by)} place(s) and moves {len(changes)} "
                     f"field(s) if it changes."),
            "source_line": f"Source: dependencies.json — shared_constants[{key!r}]",
            "matched": key,
            "match_type": "shared_constant",
            "definition": entry.get("value"),
            "value": entry.get("value"),
            "defined_at": entry.get("defined_at"),
            # NOT `duplicate_warning`. Reusing it here printed the same sentence
            # twice in one answer — once under **Breaks** and again under
            # **Duplicates** — because a constant entry has no blast-radius note
            # of its own. What breaks IS `fields_that_change`, and the template
            # renders that list; nothing is synthesised to fill the slot.
            "blast_radius_note": None,
            "duplicates": list(entry.get("duplicates") or []),
            "duplicate_warning": entry.get("duplicate_warning"),
            "consumer_count": count,
            "read_by": read_by,
            "fields_that_change": changes,
            "consumers": read_by + changes,
        }
        out.update(_matching_notes(deps, want))
        return out

    # 3. a known divergence or defect, named directly
    notes = _matching_notes(deps, want)
    if notes and want:
        first = (notes.get("known_divergences") or notes.get("known_defects"))[0]
        kind = "divergence" if "known_divergences" in notes else "defect"
        return {
            "lead": f"'{name}' matches a known {kind}: {first.get('what')}.",
            "source_line": (f"Source: dependencies.json — "
                            f"{'known_divergences' if kind == 'divergence' else 'known_defects'}"),
            "matched": first.get("what"),
            "match_type": kind,
            "blast_radius_note": first.get("effect"),
            "where": first.get("where"),
            "magnitude": first.get("magnitude"),
            "consumer_count": None,
            **notes,
        }

    # 4. a field — answered backwards, via the sources that feed it
    feeding, radius = [], []
    for src_key, entry in sources.items():
        blob = json.dumps(entry, ensure_ascii=False).lower()
        if want and want in blob:
            feeding.append(src_key)
            if entry.get("blast_radius_note"):
                radius.append(f"{src_key}: {entry['blast_radius_note']}")
    if feeding:
        out = {
            "lead": (f"'{name}' is fed by {len(feeding)} source(s): "
                     f"{', '.join(feeding)}."),
            "source_line": "Source: dependencies.json — sources[*].feeds",
            "matched": name,
            "match_type": "field",
            "blast_radius_note": " | ".join(radius) or None,
            "consumer_count": len(feeding),
            "feeds": feeding,
            "consumers": feeding,
        }
        out.update(_matching_notes(deps, want))
        return out

    return {
        "error": (f"No source, constant, divergence or defect matching "
                  f"'{name}' in the dependency map."),
        "did_you_mean": difflib.get_close_matches(
            str(name or ""), list(sources.keys()) + list(constants.keys()),
            n=5, cutoff=0.3),
        "known_sources": list(sources.keys()),
        "known_constants": list(constants.keys()),
    }
