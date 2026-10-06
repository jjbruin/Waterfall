"""Is the Microsoft sign-in client secret stored in Azure the right one?

Run it YOURSELF after storing or rotating the secret (IT's current one expires
October 5, 2028):

    .venv\\Scripts\\python scripts\\check_sso_secret.py

It reads `sso-client-secret` straight from the Container App, asks Microsoft for an
app-only token with it, and prints ONE line: OK, or Microsoft's reason for refusing.
The secret is never printed, logged or written anywhere, and the token Microsoft
issues is discarded -- the app has no application permissions, so it grants nothing;
it only proves Microsoft recognises the secret.

Why a script and not a PowerShell one-liner: on Jim's PC (Oct 6 2026) Windows
PowerShell 5.1 could not open a connection to login.microsoftonline.com at all
("Unable to connect to the remote server") while `az` reached it fine. This uses
Python's networking, which honours the same proxy settings `az` does.
"""
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

RG, APP, SECRET = "rg-waterfall-dev", "app-waterfall-dev-v2", "sso-client-secret"
TENANT = "101f2584-c3d5-4cfd-ac0f-5026d8b170c1"
CLIENT = "bcb65961-4196-41a5-9218-586ea8bde0b1"


def main() -> int:
    try:
        value = subprocess.run(
            f"az containerapp secret show -g {RG} -n {APP} --secret-name {SECRET} "
            f"--query value -o tsv",
            shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:                       # noqa: BLE001
        print(f"EMPTY - could not run az ({type(e).__name__}).")
        return 2
    if not value:
        print("EMPTY - the secret came back empty; check `az account show` is the "
              "PeaceableStreet-ThriveCSP-Dev-01 subscription.")
        return 2
    # Oct 6 2026: the first `secret set` command given wrapped the value in single
    # quotes inside double quotes, and az stored the quote marks as part of it.
    if value[:1] in "'\"" or value[-1:] in "'\"" or value != value.strip():
        print("STORED WITH EXTRA CHARACTERS - the value begins or ends with a quote "
              "mark or a space; store it again without them.")
        return 1
    if re.fullmatch(r"[0-9a-fA-F-]{36}", value):
        print("This looks like the Secret ID (a GUID), not the Secret Value - re-run "
              "the secret set command with the Value.")
        return 1

    body = urllib.parse.urlencode({
        "client_id": CLIENT, "client_secret": value,
        "grant_type": "client_credentials",
        "scope": "https://graph.microsoft.com/.default",
    }).encode()
    length = len(value)
    del value
    req = urllib.request.Request(
        f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token", data=body)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            json.load(r)                         # the token is read and dropped
        print(f"OK - Microsoft accepted the secret ({length} characters).")
        return 0
    except urllib.error.HTTPError as e:
        try:
            desc = json.load(e).get("error_description", "")
        except Exception:                        # noqa: BLE001
            desc = f"HTTP {e.code}"
        print("REJECTED - " + desc.splitlines()[0] if desc else f"REJECTED - HTTP {e.code}")
        return 1
    except Exception as e:                       # noqa: BLE001
        print(f"NO ANSWER FROM MICROSOFT - {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
