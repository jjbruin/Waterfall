# Section access by username

Moved verbatim out of CLAUDE.md on Oct 5 2026. Registry:
`flask_app/auth/sections.py`. Guardrail: `scripts/section_access_check.py`,
whose `--static` half the pre-commit hook runs.

## Section access by username

### Section access by username
Built Oct 1 2026. **SHIPPED as `v552` = `11c3455`, Oct 2 2026** — the "NOT deployed"
this line used to carry was true for one day and then stale. (It was HELD a day:
`b93dd5f` was imaged Oct 1, but `v549`/`v550` had been deployed from an unpushed clone
in between, so shipping it would have rolled that back. Re-running pre-flight P1 before
the update caught it; `origin/main` was then merged in, giving `11c3455`.) Verified on
production: `user_section_access` had 0 rows at deploy, so nobody's access changed.
Registry: `flask_app/auth/sections.py`.

Jim: access is granted per SIDEBAR SECTION, by username, from checkboxes in
Settings > User Management that default to ticked. Users without Accounting
cannot see `gl_accounts`, `gl_detail`, `ia_transactions` or ANY `tr_*`,
`wp_*` or `ic_*` table (treasury, workpapers, intercompany -- matched by
PREFIX, so a new table in those families is covered when it is created). The
USERNAME `admin` always has every section; the admin ROLE has only what is
ticked. **Only the `admin` USERNAME can change the boxes** -- admin-role users
see them greyed out and the endpoint refuses them (403), so nobody can grant
themselves a section they were denied. **Asset Management and New Business
are granted together "for now"** (`LINKED_SECTIONS`): one click writes both,
and a request splitting them is refused. Delete the group to separate them.

- **A second axis, not a replacement for roles.** The section says where a user
  may GO; the role still says what they may DO there. Both must pass.
- **Only an untick is stored** (`user_section_access.allowed = FALSE`). No row
  means allowed, so a NEW section is ticked for every existing user, and every
  section for a new user, with no backfill. Re-ticking DELETES the row.
- **Enforced on the server**, one `before_request` gate over every `/api` path
  (`enforce_section_access`). The sidebar and router hide what is refused; they
  are not the control. Read from the database on every request, so an untick
  takes effect at the user's next click, not when their token expires.
- **Shared APIs name every section that uses them** -- `/api/deals` and
  `/api/argus` are Asset Management OR New Business, because both sections'
  screens call them. Most of `/api/data` (deal list, config, reload) is open
  plumbing; only Data Management's own tools are gated.
- **The three GL tables are blocked wherever raw rows leave the app**: Data
  Explorer (listed AND rows), the database export, the MRI query run/download
  for `MRI_GL_Detail` / `MRI_GL_Accounts` / `MRI_IA_Transactions`, and the
  assistant's `query_database` tool.
- **The assistant's tools are gated one by one** (`TOOL_SECTIONS` in
  `assistant_service.py`) -- it is on every screen, so otherwise a user could
  simply ask it for a section's data. An unmapped tool is refused.
- **Settings is open to everyone** -- it is the user's own password. It sits
  under Data Management, so the sidebar footer links it when that is unticked.

**THE RULE (Jim, Oct 1 2026): A NEW SECTION GOES IN THE REGISTRY.** Adding a
section to the sidebar means adding it to `SECTIONS` in `auth/sections.py` and
gating its block on `auth.hasSection('<key>')`; it then appears in User
Management, ticked for everyone, with no screen change. Likewise every new Vue
route, `/api` route and assistant tool must be assigned. Guardrail
`scripts/section_access_check.py` (256) fails until they are -- it enumerates
the RUNNING app's url_map rather than grepping -- and the pre-commit hook runs
its `--static` half whenever the sidebar, router, an API blueprint or the
assistant is staged. **It caught its first case before it shipped:** merging
`origin/main` brought in Investment Metrics (`v545`), whose `/investment-metrics`
screens and `/api/investment-metrics` endpoints belonged to no section. They are
Asset Management's (`b93dd5f`).

**Deploy status: LIVE.** This line used to read "built and imaged as `b93dd5f`, NOT
deployed" and said to delete it when it shipped. It shipped the next day as `v552` =
`11c3455` and the line sat here stale, which is the exact failure the date-stamp rule
at the top of CLAUDE.md exists to prevent. Corrected Oct 5 2026; see the header above
and the `v552` entry in `deploy_history.md`.
