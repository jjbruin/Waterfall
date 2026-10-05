# Board section (Phase 0 built Oct 5 2026, NOT deployed)

**Status, Oct 5 2026:** Phase 0 is built on `feat/board-phase0`. Delete this paragraph
when it ships. The plan is in Claude Docs, "Board Package — Development Plan":
https://claude.ai/code/artifact/71e5c89f-933f-4b88-9e3e-f1b2e1d7a5b9. It records the
seven decisions Jim made on Oct 5.

## Phase 0: the foundation
- **Board is an OPT-IN section** (`"opt_in": True` in `auth/sections.py`).
  - The storage is inverted: a GRANT is a TRUE row, and no row means denied. Deploying
    the section shows it to nobody, and a user created later doesn't have it.
  - A grant may carry an end date (`user_section_access.expires_at`, added in place by
    an asked-first ALTER). It's for outside advisors; past the date the grant reads as
    absent.
  - The User Management column shows Board unticked for everyone.
- **Only the `admin` USERNAME grants it**, through User Management or Board > Access.
  The admin ROLE (Charlene, the developers) holds nothing here. The guardrail proves
  this past the section gate: it grants an admin-role user Board, then checks that
  user can do no more than a reader.
- **The permissions** (`auth/permissions.py`), opt-in, by username:

  | Permission | Allows | Implies |
  |---|---|---|
  | `board_edit` | narrative, as-of dates | — |
  | `board_build` | creates meetings | `board_edit` |
  | `comp_view` | salary planning: view | — |
  | `comp_edit` | salary planning: edit | `comp_view` |

  **A permission requires its section.** Remove Board, or let it expire, and every
  permission goes void although the rows remain. Implications are applied on read, so
  revoking the stronger permission never leaves the weaker one behind as an orphan.
- **The access log** (`auth/audit.py`, table `access_audit`) is append-only. It holds
  every section and permission change, and every meeting, schedule and narrative
  write. A failure to log RAISES. The admin username reads it on Board > Access log.
- **NEVER-exposed tables** (`sections.NEVER`): `comp_*`, `access_audit` and
  `user_permissions`.
  - They don't appear in Data Explorer, the export or the assistant, for anyone: the
    admin username and salary holders included.
  - `pull_production_db.py` skips them; `comp_*` only travels with
    `--include-compensation`.
  - `board_*` tables follow the Board section.
- **Meetings** (`services/board_service.py`, `api/board.py`):
  - A meeting carries every schedule in the deck's catalog, each with its OWN as-of
    date (the January deck mixed 12/31/25 and 9/30/25), plus seven narrative blocks.
  - Refused: an unknown key, a date that doesn't parse, or any edit once the meeting
    is no longer a draft.
  - Saved with a warning: an as-of date after the meeting date.
  - No figures yet. Each schedule's view arrives with its phase.
- Guardrail `scripts/board_access_check.py` (156 checks, including the race-safe column add). It enumerates every
  `/api/board` route from the running app and calls each one as no-token, ungranted,
  expired, revoked, the admin role, reader, editor, builder, salary holder and the
  admin username. It fails on each injected defect: `optin` (31 failures), `role` (5),
  `section` (1). `section_access_check` now expects "no rows" to mean every
  NON-opt-in section.

## Next phases
Phases 1, 2 and 4 can start now; their engines are live. Phase 3 needs the
originations spreadsheet. Phase 5 needs the CFO's model and an anonymized payroll copy.
