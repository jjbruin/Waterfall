"""Guardrail: the MRI connection string is one ODBC Driver 18 will parse.

WHY THIS EXISTS. `Connection Timeout=30;` sat in the string from 08df897
(May 5 2026). It is an ADO/OLE DB keyword, not an ODBC one, and Driver 18
rejects the entire string with SQLSTATE 08001 "Invalid connection string
attribute" BEFORE attempting any network connection. Every MRI query therefore
failed identically whether or not the VPN was up.

That is why it lasted four months: the MRI tunnel has its own long-running
troubles, the failure surfaced as 08001 — the same class a dead VPN produces —
and it read as a connectivity problem every single time. Measured Sep 15 2026,
removing that one clause changes the failure from "invalid attribute" to
"cannot reach the server": from a string the driver will not parse, to one it
parses and then acts on.

This checks the STRING, not the connection. It needs no VPN, no driver and no
credentials, so it runs anywhere — which is the point, since the thing it
guards is a parse error that no amount of network access would reveal.

Run:  .venv/Scripts/python.exe scripts/mri_connection_string_check.py
"""
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask_app.services import mri_service as M  # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


# Keywords ODBC Driver 18 for SQL Server accepts in a connection string.
# Deliberately a allow-list: a typo is far likelier than a legitimate exotic
# keyword, and a rejected string fails the whole refresh.
ALLOWED = {
    "driver", "server", "database", "uid", "pwd", "trusted_connection",
    "encrypt", "trustservercertificate", "app", "wsid", "language",
    "applicationintent", "multisubnetfailover", "failover_partner",
    "attachdbfilename", "authentication", "columnencryption",
    "connectretrycount", "connectretryinterval", "hostnameincertificate",
    "keystoreauthentication", "keystoreprincipalid", "keystoresecret",
    "longasmax", "mars_connection", "packetsize", "querylog_on",
    "querylogfile", "querylogtime", "regional", "replication",
    "statstimeout", "tracefile", "traceflags", "trace", "servercertificate",
    "ipaddresspreference", "transparentnetworkipresolution",
}

# NOT keywords, however plausible. Each of these is real — from ADO, OLE DB or
# .NET SqlClient — and each would take the whole string down with it.
FORBIDDEN = {
    "connection timeout": "ADO/OLE DB; use the pyodbc.connect(timeout=) argument",
    "connect timeout": ".NET SqlClient; use pyodbc.connect(timeout=)",
    "integrated security": ".NET SqlClient; ODBC uses Trusted_Connection",
    "initial catalog": ".NET SqlClient; ODBC uses DATABASE",
    "data source": ".NET SqlClient; ODBC uses SERVER",
    "user id": ".NET SqlClient; ODBC uses UID",
    "password": ".NET SqlClient; ODBC uses PWD",
    "persist security info": ".NET SqlClient; meaningless to ODBC",
    "provider": "OLE DB; meaningless to ODBC",
}


def keywords(cs: str):
    """The keyword of each clause, brace-quoted values stepped over."""
    out, i, n = [], 0, len(cs)
    while i < n:
        eq = cs.find("=", i)
        if eq < 0:
            break
        out.append(cs[i:eq].strip().lower())
        j = eq + 1
        if j < n and cs[j] == "{":          # brace-quoted value: skip to '}'
            close = cs.find("}", j + 1)
            while close + 1 < n and cs[close + 1] == "}":
                close = cs.find("}", close + 2)
            j = close + 1 if close > 0 else n
        semi = cs.find(";", j)
        i = (semi + 1) if semi >= 0 else n
    return out


for key in sorted(M.MRI_SERVERS):
    # Build the string the way _get_connection does, without connecting.
    info = M.MRI_SERVERS[key]
    cs = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={info['server']};"
        f"DATABASE={info['database']};"
        f"UID={M.MRI_USERNAME};"
        "PWD={" + M.MRI_PASSWORD.replace("}", "}}") + "};"
        "TrustServerCertificate=yes;"
    )
    for kw in keywords(cs):
        if kw in FORBIDDEN:
            check(False, f"[{key}] {kw!r} is not an ODBC keyword — {FORBIDDEN[kw]}. "
                         f"Driver 18 rejects the WHOLE string with 08001 before "
                         f"it tries the network.")
        elif kw and kw not in ALLOWED:
            check(False, f"[{key}] {kw!r} is not a recognised ODBC Driver 18 "
                         f"keyword; if it is genuinely valid, add it to ALLOWED "
                         f"with a reference.")

    # The password must be brace-quoted, or a ';' in it truncates the string.
    check("PWD={" in cs,
          f"[{key}] PWD is not brace-quoted; a ';' in the password would "
          f"silently truncate the connection string at that character")

# And the source itself must not reintroduce the clause.
src = pathlib.Path(M.__file__).read_text(encoding="utf-8")
live = "\n".join(l for l in src.splitlines()
                 if not l.lstrip().startswith("#"))
check(not re.search(r'["\']Connection Timeout\s*=', live),
      "mri_service still builds 'Connection Timeout=' into a connection string")
check("pyodbc.connect(conn_str, timeout=" in live,
      "the login timeout is no longer passed to pyodbc.connect(timeout=), so "
      "removing the clause dropped the behaviour instead of relocating it")

if FAIL:
    print("FAIL")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print("OK - every MRI connection-string keyword is one ODBC Driver 18 accepts, "
      "PWD is brace-quoted, and the login timeout is a pyodbc argument")
