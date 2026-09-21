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


def _find_field(field_id: str) -> Optional[dict]:
    want = str(field_id or "").strip().lower()
    for f in _dictionary().get("fields", []):
        if str(f.get("field_id", "")).strip().lower() == want:
            return f
    return None


def _origin_phrase(origin: str) -> str:
    """meta.origins maps an origin key to the phrase the answer should use."""
    origins = _dictionary().get("meta", {}).get("origins", {})
    return origins.get(str(origin or "").strip(), str(origin or "unknown origin"))


def _basis_payload(b: dict) -> dict:
    """One basis, with the Source line already assembled from meta.origins."""
    return {
        "basis": b.get("basis"),
        "plain": b.get("plain"),
        "origin": b.get("origin"),
        "source": b.get("source"),
        "caveats": b.get("caveats"),
        "source_line": f"Source: {_origin_phrase(b.get('origin'))} — {b.get('source')}",
    }


def lookup_field(field_id: str, basis: str = None) -> dict:
    """What a field means and where it comes from.

    ``basis`` is validated against THIS FIELD'S OWN bases — there is no global
    basis vocabulary, and there must not be one: 'isbs' is a debt basis,
    'manual_entry' is a Net ROE basis, and a shared enum would advertise
    combinations that do not exist. An unknown or omitted basis returns every
    basis the field has, which is the honest answer to "where does X come from"
    when X legitimately has more than one source.
    """
    try:
        field = _find_field(field_id)
    except Exception as exc:
        logger.exception("lookup_field: dictionary unavailable")
        return {"error": f"Field dictionary unavailable: {exc}"}

    if field is None:
        ids = field_ids()
        return {
            "error": f"No field '{field_id}' in the dictionary.",
            "did_you_mean": difflib.get_close_matches(
                str(field_id or "").strip().lower(), ids, n=5, cutoff=0.4),
            "known_field_ids": ids,
        }

    all_bases = field.get("bases", []) or []
    names = [b.get("basis") for b in all_bases]

    selected = None
    if basis:
        want = str(basis).strip().lower()
        selected = next(
            (b for b in all_bases
             if str(b.get("basis", "")).strip().lower() == want), None)
    # A field with exactly one basis has no ambiguity to preserve — answer it
    # directly rather than making the caller pick from a list of one.
    if selected is None and not basis and len(all_bases) == 1:
        selected = all_bases[0]

    shown = [selected] if selected else all_bases
    label = field.get("display_label") or field.get("field_id")
    tabs = ", ".join(field.get("appears_on", []) or []) or "this report"

    if selected:
        lead = selected.get("plain")
        source_line = _basis_payload(selected)["source_line"]
    else:
        lead = (f"{label} appears on {tabs} and is reported on "
                f"{len(all_bases)} different bases ({', '.join(str(n) for n in names)}) — "
                f"which one applies depends on the deal and the tab.")
        source_line = ("Source: varies by basis — see `bases` below, each with its "
                       "own Source line.")

    out = {
        "lead": lead,
        "source_line": source_line,
        "field_id": field.get("field_id"),
        "display_label": label,
        "appears_on": field.get("appears_on"),
        "owner": field.get("owner"),
        "status": field.get("status"),
        "basis_selection": field.get("basis_selection"),
        "bases": [_basis_payload(b) for b in shown],
        "caveats": (selected or {}).get("caveats") if selected
                   else [b.get("caveats") for b in all_bases if b.get("caveats")],
    }
    if field.get("tab_notes"):
        out["tab_notes"] = field["tab_notes"]
    if basis and selected is None:
        out["basis_warning"] = (
            f"'{basis}' is not a basis for {field.get('field_id')}. "
            f"Valid bases: {', '.join(str(n) for n in names)}. Showing all.")
    return out


# ── dependency map ────────────────────────────────────────────────────────

def _constant_duplicate_index(deps: dict) -> dict:
    """{bare constant name: [duplicate literal locations]} from shared_constants."""
    idx = {}
    for key, entry in (deps.get("shared_constants") or {}).items():
        dups = entry.get("duplicated_at") or []
        if not dups:
            continue
        for token in str(key).replace("/", " ").split():
            bare = token.split(".")[-1].strip()
            if len(bare) > 3:
                idx.setdefault(bare, []).extend(
                    [f"{key}: {d}" for d in dups])
    return idx


#: Words that carry no discriminating power in a "what uses X" question.
_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to",
              "from", "data", "table", "field", "value", "app", "is", "it"}


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
    names a constant outright — ``config.DEBT_BS_ACCTS`` — gets the constant and
    not the source whose prose happens to mention it.
    """
    want = str(name or "").strip()
    if not want:
        return None
    keys = list(entries.keys())
    for k in keys:
        if k == want:
            return k
    low = want.lower()
    for k in keys:
        if k.lower() == low:
            return k
    for k in keys:
        if low in k.lower() or k.lower() in low:
            return k
    return None


def _match_key(name: str, entries: dict) -> Optional[str]:
    """Strict pass first, then token overlap.

    THE TOKEN PASS IS WHAT MAKES THIS USABLE FROM A QUESTION. The assistant
    passes whatever the user said — "ISBS balance sheet" — and the source key is
    ``ISBS_Download[vSource='Interim BS', vAccount in 2150/2152/2210]``, which
    shares no substring with it. Scoring query tokens against the key AND the
    entry's own text matches it on {isbs, balance} without needing a hand-written
    synonym list that would then have to be kept in step with the file.
    """
    exact = _match_key_strict(name, entries)
    if exact:
        return exact
    want = str(name or "").strip()
    q = _tokens(want)
    if not q:
        return None
    best, best_score = None, 0.0
    for k in entries:
        key_hits = len(q & _tokens(k))
        # AT LEAST ONE WORD MUST LAND ON THE KEY ITSELF. Scoring against the
        # entry's prose alone matches anything: 'zzz_none' hit the ISBS source
        # because the word "None" appears in its gateway rules. The key is what
        # the caller is naming; the prose only breaks ties.
        if not key_hits:
            continue
        blob = k + " " + json.dumps(entries[k], ensure_ascii=False)
        # Key hits count double — matching the name beats matching prose.
        score = (len(q & _tokens(blob)) + key_hits) / (2.0 * len(q))
        if score > best_score:
            best, best_score = k, score
    return best if best_score >= 0.25 else None


def impact(name: str) -> dict:
    """What reads a source or constant, and what moves if it changes.

    ``name`` may be a source table, a shared constant, or a field id. A field id
    is answered backwards — which sources feed it — because "what breaks if I
    change the debt field" is really a question about the field's inputs.
    """
    try:
        deps = _dependencies()
    except Exception as exc:
        logger.exception("impact: dependency map unavailable")
        return {"error": f"Dependency map unavailable: {exc}"}

    sources = deps.get("sources") or {}
    constants = deps.get("shared_constants") or {}
    dup_idx = _constant_duplicate_index(deps)

    def _dup_warnings(blob: str, extra=None) -> list:
        """Duplicate-literal warnings for any constant named in this entry."""
        warnings = list(extra or [])
        for bare, locations in dup_idx.items():
            if bare in blob:
                warnings.extend(locations)
        # stable, de-duplicated
        seen, out = set(), []
        for w in warnings:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    # An outright name — source or constant — wins before any fuzzy matching.
    strict_source = _match_key_strict(name, sources)
    strict_const = _match_key_strict(name, constants)
    if strict_const and not strict_source:
        key = None
    else:
        key = strict_source or _match_key(name, sources)

    # 1. a source table
    if key:
        entry = sources[key]
        blob = json.dumps(entry, ensure_ascii=False)
        consumers = (list(entry.get("direct_consumers") or [])
                     + list(entry.get("direct_account_readers_bypassing_the_gateway") or [])
                     + list(entry.get("downstream_fields") or []))
        return {
            "lead": (f"{key} is read by {len(consumers)} consumers across the app "
                     f"(loaded as `{entry.get('data_key') or entry.get('target_table')}`)."),
            "source_line": f"Source: dependencies.json — sources[{key!r}]",
            "matched": key,
            "match_type": "source",
            "blast_radius_note": entry.get("blast_radius_note"),
            "duplicate_literal_warnings": _dup_warnings(blob),
            "consumers": consumers,
            "gateway": entry.get("gateway"),
            "gateway_rules": entry.get("gateway_rules"),
            "downstream_fields": entry.get("downstream_fields"),
        }

    # 2. a shared constant
    key = _match_key(name, constants)
    if key:
        entry = constants[key]
        consumers = (list(entry.get("read_by") or [])
                     + list(entry.get("fields_that_change_if_this_changes") or []))
        dups = [f"{key}: {d}" for d in (entry.get("duplicated_at") or [])]
        return {
            "lead": (f"{key} (defined at {entry.get('at')}) is read in "
                     f"{len(entry.get('read_by') or [])} places and moves "
                     f"{len(entry.get('fields_that_change_if_this_changes') or [])} "
                     f"fields if it changes."),
            "source_line": f"Source: dependencies.json — shared_constants[{key!r}]",
            "matched": key,
            "match_type": "shared_constant",
            "blast_radius_note": entry.get("blast_radius_note")
                                 or entry.get("consequence"),
            "duplicate_literal_warnings": dups,
            "consumers": consumers,
            "value": entry.get("value"),
            "defined_at": entry.get("at"),
        }

    # 3. a field id — answered backwards, via the sources that feed it
    want = str(name or "").strip().lower()
    feeding, notes, dups = [], [], []
    for src_key, entry in sources.items():
        blob = json.dumps(entry, ensure_ascii=False).lower()
        if want and want in blob:
            feeding.append(src_key)
            if entry.get("blast_radius_note"):
                notes.append(f"{src_key}: {entry['blast_radius_note']}")
            dups.extend(_dup_warnings(json.dumps(entry, ensure_ascii=False)))
    if feeding:
        divergences = [d for d in (deps.get("known_divergences") or [])
                       if want in json.dumps(d, ensure_ascii=False).lower()]
        return {
            "lead": (f"'{name}' is fed by {len(feeding)} source(s): "
                     f"{', '.join(feeding)}."),
            "source_line": "Source: dependencies.json — sources[*].downstream_fields",
            "matched": name,
            "match_type": "field",
            "blast_radius_note": " | ".join(notes) or None,
            "duplicate_literal_warnings": sorted(set(dups)),
            "consumers": feeding,
            "known_divergences": divergences or None,
        }

    return {
        "error": f"No source, constant or field matching '{name}' in the dependency map.",
        "did_you_mean": difflib.get_close_matches(
            str(name or ""), list(sources.keys()) + list(constants.keys()),
            n=5, cutoff=0.3),
        "known_sources": list(sources.keys()),
        "known_constants": list(constants.keys()),
    }
