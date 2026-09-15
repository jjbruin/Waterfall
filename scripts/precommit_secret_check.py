"""Guardrail: the pre-commit hook catches the shapes that actually got through.

WHY THIS EXISTS. The hook blocked `scheme://user:secret@host` and nothing else.
On Sep 15 2026 three credentials surfaced in one day and NOT ONE was URI-shaped:

  * MRI_PASSWORD = "..."      in mri_service.py, public since 2026-05-05
  * a SendGrid API key        as a plaintext container-app env var
  * `admin / <password>`      in a memory file

Every one walked past a hook whose whole job was to stop it. A rule nobody has
tested against the thing it missed is a rule nobody should trust, so this runs
the real hook, as git runs it, against a scratch repository.

EVERY SECRET BELOW IS SYNTHETIC. They have the SHAPE of the real ones and none
of the value — putting a live credential in a test is the defect being guarded
against.

Run:  .venv/Scripts/python.exe scripts/precommit_secret_check.py
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = ROOT / "scripts" / "hooks" / "pre-commit"

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def git(repo, *args, **kw):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, **kw)


def run_hook(filename: str, body: str):
    """Stage `body` as `filename` in a scratch repo and run the hook on it."""
    repo = tempfile.mkdtemp(prefix="hooktest-")
    try:
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "t@t")
        git(repo, "config", "user.name", "t")
        # A base commit, so `git diff --cached` has something to compare against.
        (pathlib.Path(repo) / ".keep").write_text("x\n", encoding="utf-8")
        git(repo, "add", ".keep")
        git(repo, "commit", "-q", "-m", "base")

        target = pathlib.Path(repo) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        git(repo, "add", filename)

        hook_dir = pathlib.Path(repo) / ".githooks"
        hook_dir.mkdir()
        dest = hook_dir / "pre-commit"
        shutil.copy(HOOK, dest)
        os.chmod(dest, 0o755)

        r = subprocess.run(["sh", str(dest)], cwd=repo,
                           capture_output=True, text=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def blocks(label, filename, body):
    code, out = run_hook(filename, body)
    check(code != 0, f"NOT BLOCKED — {label}\n      staged: {body.strip()[:88]}")
    if code != 0:
        # Whatever it prints must not contain the secret itself.
        for tok in ("Hx9vQ2mL", "aB3dE7gH", "Zq7WmT2p"):
            check(tok not in out,
                  f"{label}: the hook echoed the secret back instead of masking it")


def allows(label, filename, body):
    code, out = run_hook(filename, body)
    check(code == 0, f"FALSE POSITIVE — {label}\n      staged: "
                     f"{body.strip()[:88]}\n      hook said: {out.strip()[:140]}")


# ── The three that actually got through ──────────────────────────────────
blocks("a secret-named variable assigned a literal (the MRI password shape)",
       "svc.py", 'MRI_PASSWORD = "Hx9vQ2mL^Pw4z*"\n')

blocks("a SendGrid key by its own shape, whatever it is called",
       "conf.py", 'MAIL = "SG.aB3dE7gHiJkLmNoPqR.sTuVwXyZ0123456789abcdefgh"\n')

blocks("a username / password pair in prose",
       "notes.md", "Login: admin / Zq7WmT2pQ9\n")

# ── The one it already caught, still caught ──────────────────────────────
blocks("a connection string with an inline password",
       "m.py", 'URL = "postgresql://wfadmin:Hx9vQ2mLpw@host:5432/db"\n')

# ── Other self-identifying tokens ────────────────────────────────────────
blocks("an AWS access key id",
       "a.py", 'AWS = "AKIAIOSFODNN7EXAMPLE"\n')
blocks("an Anthropic key",
       "b.py", 'K = "sk-ant-aB3dE7gHiJkLmNoPqRsTuVwXyZ01234567"\n')
blocks("a GitHub personal access token",
       "c.py", 'T = "ghp_aB3dE7gHiJkLmNoPqRsTuVwXyZ0123456789"\n')

# ── Must NOT fire on the legitimate patterns this repo uses ──────────────
# A hook that cries wolf gets --no-verify'd reflexively, and then it protects
# nothing at all. These are all real lines from this codebase's own idiom.
allows("reading from the environment",
       "cfg.py", 'SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")\n')
allows("an empty default",
       "cfg.py", 'ACS_CONNECTION_STRING = ""\n')
allows("a container-app secret reference",
       "deploy.sh", 'ACS_CONNECTION_STRING=secretref:acs-connection-string\n')
allows("an angle-bracket placeholder (the azure-complete-setup.sh false positive)",
       "setup.sh", '#   DATABASE_URL=postgresql://wfadmin:<password>@host:5432/db\n')
allows("a documentation placeholder in caps",
       "doc.md", 'export DATABASE_URL="postgresql://USER:PASS@host/db"\n')
allows("a shell variable",
       "run.sh", 'PGPASSWORD="$DB_PASSWORD" psql -h host\n')
allows("a named env var with no value",
       "docs.md", "Set `MRI_PASSWORD` in the container app's secrets.\n")
allows("prose that merely mentions a password",
       "README.md", "The admin password must be rotated before release.\n")
allows("a file path that looks like a pair",
       "notes.md", "See admin/passwords-policy.md for the rotation schedule.\n")

if FAIL:
    print("FAIL")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print("OK - the hook blocks the three shapes that reached production, the "
      "URI form it already caught, and four self-identifying token formats, "
      "while leaving this repo's env-var and placeholder idioms alone")
