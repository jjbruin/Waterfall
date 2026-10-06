# SharePoint import

Jim asked IT for one Entra app for Microsoft sign-in AND SharePoint/OneDrive file
access. IT configured it Oct 6 2026: app **Waterfall XIRR**, client id
`bcb65961-4196-41a5-9218-586ea8bde0b1`, tenant `101f2584-c3d5-4cfd-ac0f-5026d8b170c1`,
delegated Graph permissions, admin consent granted, a client secret (24 months),
"Assignment required = Yes". Jim chose (Oct 6): **the picker first, scheduled folder
pulls after.** First importers: expense receipts, Data CSVs, Treasury, Lease Review
documents, Valuations documents, Argus.

## Phase 1 -- the picker (branch `feat/sso-email-match`, not deployed as of Oct 6 2026)

A "From SharePoint" button beside each importer's own file input
(`components/common/SharePointPicker.vue`, `services/sharepoint.ts`).

- **Runs in the browser.** MSAL (`@azure/msal-browser`, pinned exactly) signs in by
  POPUP with DELEGATED, READ-ONLY scopes `Files.Read.All`, `Sites.Read.All`. The user
  sees only what they can already open in SharePoint. No Microsoft token reaches our
  server and no client secret is used -- so the picker can be on while password
  sign-in is still the only sign-in.
- **Our own Graph browser, not Microsoft's File Picker v8.** v8 needs SharePoint
  (not Graph) API permissions, which IT did not grant. Graph covers sites, libraries,
  folders, pasted links (`/shares/u!...`) and downloads.
- **ONE PATH PER IMPORTER.** The picker emits browser `File` objects into the very
  function the screen's own file input feeds. A folder pick sets `webkitRelativePath`,
  so Lease Review's folder-name tenant hint works. Over the importer's file limit is
  REFUSED with the count, never truncated.
- **Switched on by** `SHAREPOINT_CLIENT_ID` + `SHAREPOINT_TENANT_ID` (falling back to
  the `SSO_*` pair), served to signed-in users by `GET /auth/sso/sharepoint`.
- **MSAL 5 popups need a return page that RUNS the redirect bridge**
  (`broadcastResponseToMainFrame`) -- a blank page no longer works. It is
  `vue_app/msal-redirect.html` + `src/msalRedirect.ts`, a second Vite build input.
- MSAL is dynamically imported on hover/first use, so it is not in the main bundle.
- Guardrail: `scripts/sharepoint_picker_check.py` (read-only scopes, no token to our
  server, the return page in the build, one path per importer, the config route
  signed-in only); `--inject=scope|token|entry|fork|open` each fail it.

### What must be registered in Entra (SPA platform) before it works
**IT confirmed Oct 6 2026:** all four URIs below registered exactly; delegated Graph
`openid email profile offline_access User.Read Files.Read.All Sites.Read.All`, admin
consent granted; all 19 users assigned (Assignment required = Yes, so only they can
sign in -- a new user must be assigned in Entra AND exist in User Management). **The
client secret expires October 5, 2028**; IT sends the value by Mimecast to Jim, who
puts it in the Container App secrets himself. Rotate before that date or Microsoft
sign-in stops (the picker needs no secret and keeps working).
- `https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io/msal-redirect.html`
- `http://localhost:5173/msal-redirect.html`

Web platform (server sign-in, `sso.py`): `.../auth/sso/callback` on both hosts
(`http://localhost:5000/auth/sso/callback` locally).

### Not yet verified
The Microsoft sign-in itself: the in-app preview pane blocks popups, and the redirect
URIs above were not confirmed registered. Verified locally: config route, all
pickers render, dialog opens, popup-blocked message, return page served.

## Microsoft sign-in (same branch)
`sso.py` matches the EXISTING account by the `users.email` column, never opens the
`admin` username, creates no account (`no_account` / `ambiguous` refused). ProxyFix
makes `url_for(_external=True)` say https behind the Azure ingress. Guardrail:
`scripts/sso_email_match_check.py`. Needs the client secret as a Container App
secret plus `SSO_PROVIDER/SSO_CLIENT_ID/SSO_TENANT_ID/SSO_CLIENT_SECRET`.

## Phase 2 -- scheduled folder pulls (not started)
Needs per-user refresh tokens stored server-side (encrypted, a PROTECTED table) or an
application permission -- a decision for Jim and IT, since an app-only `Sites.Read.All`
reads every site in the tenant.
