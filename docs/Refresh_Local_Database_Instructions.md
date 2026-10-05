# Refreshing your local database from production

Charlene,

Your local `waterfall.db` is your own copy. It isn't in git, so it only has the data from whenever you last loaded it. I've added a script that copies production's data into it, so local testing runs against the same data as the live app.

It takes about five minutes.

---

## Before you start

- **Get the latest main branch.** The script is `scripts/pull_production_db.py`.
  ```
  git checkout main
  git pull
  ```
- **Stop your local Flask server** if it's running. Windows won't replace a file that another program has open.
- **Be signed in to Azure** in your terminal. Run `az login` if you aren't sure.
- **Have about 1.5 GB of free disk space**, because the full copy includes stored PDFs. If you don't need the PDFs, see "Smaller copy" below.

---

## Run it

From the repo folder, paste **one** of these as a single line, in a single terminal window. Each terminal tab keeps its own settings, so don't split the line across tabs.

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL = (az containerapp secret show -g rg-waterfall-dev -n app-waterfall-dev-v2 --secret-name db-url --query value -o tsv); .venv\Scripts\python scripts\pull_production_db.py; Remove-Item Env:DATABASE_URL
```

**Mac (Terminal):**
```bash
DATABASE_URL="$(az containerapp secret show -g rg-waterfall-dev -n app-waterfall-dev-v2 --secret-name db-url --query value -o tsv)" .venv/bin/python scripts/pull_production_db.py
```

You'll see one line per table. When it's finished, it ends with:

```
Done: 132 tables, ... rows, every count matching production.
Previous local database kept as waterfall.db.bak-<date-time>.
```

Then start your local server again as usual.

---

## What it does (and doesn't do)

- **Copies every table from production**, except the login tables. Your local `admin` / `admin` login keeps working.
- **Keeps your old database** as `waterfall.db.bak-<date-time>`, in case you need anything from it.
- **Only swaps in the new copy if it's complete.** Every table's row count has to match production. If anything goes wrong, your existing database is left exactly as it was.
- **Never shows the database password.** It's read from your terminal for the length of the run and is never printed or saved.
- **Keeps your local table structure**, so adding records locally still works normally.

---

## Smaller copy

To skip stored documents (lease PDFs, statement PDFs, receipt images) and keep the file small, add `--no-files` to the end of the command:

```
... scripts\pull_production_db.py --no-files
```

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `DATABASE_URL is not set...` | The first part of the line didn't run in this window. Paste the whole single line again. |
| `Could not connect to production ... timed out` | Your computer's IP address needs adding to the database firewall (`psql-waterfall-dev` → Networking in the Azure portal). Let me know and we'll add it. |
| An error from the `az containerapp secret show` part | Run `az login` again, then retry. |
| `PermissionError` / file in use | Your local Flask server is still running. Stop it and retry. |
| `... table(s) did not copy cleanly` | Nothing was changed. Send me the last few lines it printed. |

---

## One caution

Don't use **Data Management → Database Tools → Export Database** for this. On the live app that button exports an empty local file, not the production database. It's a known bug and it's logged to be fixed.

---

The full technical notes are in `CLAUDE.md` under **"Refreshing local data from production."**

Jim
