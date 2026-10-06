// Entry point of msal-redirect.html, the page the Microsoft sign-in popup
// returns to. MSAL 5 no longer reads the popup's URL from the opening window:
// this page must broadcast the response back itself, then it closes.
import { broadcastResponseToMainFrame } from '@azure/msal-browser/redirect-bridge'

broadcastResponseToMainFrame().catch((e) => {
  document.body.textContent = 'Microsoft sign-in could not be completed: ' + (e?.message || e)
})
