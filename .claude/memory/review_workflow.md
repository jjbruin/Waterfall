---
name: One Pager Review Workflow
description: Sequential approval pipeline for One Pager investor reports — tables, API, Vue components, role management
type: project
---

## One Pager Review & Approval Workflow (Mar 2026)

### Architecture
- **Backend**: `flask_app/services/review_service.py` (business logic) + `flask_app/api/reviews.py` (9 endpoints at `/api/reviews`)
- **Frontend**: `ReviewPanel.vue` (status + actions + notes) embedded in `OnePagerView.vue`, `ReviewTrackingView.vue` (production dashboard)
- **Admin**: Review role management section in `SettingsView.vue`

### Review Steps (Sequential)
0. Draft (asset_manager) → 1. Head of AM → 2. President → 3. CCO → 4. CEO → 5. Approved

### Database Tables (all in PROTECTED_TABLES)
- `review_roles`: user_id + review_role (UNIQUE pair). A user can hold multiple roles.
- `review_submissions`: vcode + quarter (UNIQUE). Fields: status, current_step, submitted_by, returned_to_step, timestamps.
- `review_notes`: Audit trail — user_id, username, review_role, action (note/submit/approve/return), note_text, created_at.

### Status Values
`draft`, `pending_head_am`, `pending_president`, `pending_cco`, `pending_ceo`, `approved`, `returned`

### API Endpoints
- `GET /api/reviews/<vcode>/<quarter>` — status + notes + user permissions (can_submit, can_approve, can_return, is_editable)
- `POST .../submit` — asset_manager only, draft/returned → pending_head_am
- `POST .../approve` — matching step role, advances to next step
- `POST .../return` — matching step role, returns to draft (note required)
- `POST .../note` — any participant, adds discussion note
- `GET /api/reviews/tracking` — production pipeline data with quarter/status/investor filters
- `GET /api/reviews/investors` — distinct upstream investor IDs for filter dropdown
- `GET/POST/DELETE /api/reviews/roles` — admin-only role assignment CRUD

### Comment Editing Policy
- Comments are **editable throughout the entire review process** (draft, in review, returned)
- Comments are locked (read-only) **only after final approval** (status = `approved`)
- `financials.py` save_comments endpoint checks `is_editable(vcode, quarter)` — returns 403 only when `approved`
- Vue textareas get `:readonly="commentsLocked"`, "Save Comments" button hidden when locked
- **Bug fix (Aug 2026)**: `post_note` endpoint was missing `is_editable` in its response, causing `commentsLocked` to become `true` after posting a review note. Fixed by adding `result["is_editable"]` to the response (same as all other review endpoints).

### Vue Components
- `ReviewPanel.vue`: Status dot + label, Submit/Approve/Return buttons (role-gated), collapsible notes list, add-note input. Hidden in print (`no-print` class).
- `ReviewTrackingView.vue`: Summary cards (Draft/In Review/Returned/Approved counts, clickable to filter), quarter+investor+status filters, deal table with click→navigate to One Pager.
- `OnePagerView.vue`: Loads review status alongside one-pager data, handles route query params (`?vcode=X&quarter=Y`) for navigation from tracking view.

### Investor Filter (Jun 2026)
- **Recursive CTE**: Traverses ownership chains at any depth (e.g. PSCKOC → KOCTRS → PPI → deal)
- **EndDate filtering**: Only follows active relationships (`COALESCE(CAST(r."EndDate" AS TEXT), '') = ''`) — excludes ended relationships like PSC3→PSCMAN
- **Child property exclusion**: Excludes child properties (invested in by another deal in the same Portfolio_Name group) but keeps parent portfolio deals (Burton, Giant 7, Brainerd, etc.)
- **Sold deal exclusion**: Filters both `Sale_Status != 'SOLD'` and `Lifecycle != 'Sold'`
- **PostgreSQL compatibility**: Uses `CAST(EndDate AS TEXT)` instead of `TRIM(EndDate)` since EndDate is `timestamp without time zone` on PG (TRIM only works on text types)
- **Investor list**: `get_investor_list()` traces upward from deals recursively, excludes OP* and PPI* prefixes

### Settings
Admin "Review Roles" section: table of assignments + add/remove controls. Users dropdown + review role dropdown → Assign.
