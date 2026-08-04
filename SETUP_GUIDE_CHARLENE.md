# Claude Code Setup Guide — Waterfall XIRR

Hi Charlene! This guide will get you set up to use Claude Code to investigate bugs and fix calculations in the Waterfall XIRR codebase. You don't need to know how to code — Claude does the coding. You describe the problem, Claude finds and fixes it.

---

## Step 1: Get GitHub Access

Jim will add you as a collaborator to the repo. You'll get an email invitation from GitHub — click "Accept" to get access.

If you don't have a GitHub account yet:
1. Go to https://github.com/join
2. Create a free account
3. Send Jim your username so he can add you

## Step 2: Install Prerequisites

You need two things installed: **Git** and **Node.js**.

### Git (version control)
1. Download from https://git-scm.com/download/win
2. Run the installer — accept all defaults
3. Restart your terminal after installing

### Node.js (runs Claude Code)
1. Download the **LTS** version from https://nodejs.org
2. Run the installer — accept all defaults
3. Restart your terminal after installing

### Verify both installed
Open a new **Command Prompt** or **PowerShell** window and run:
```
git --version
node --version
```
Both should print a version number.

## Step 3: Install Claude Code

In your terminal:
```
npm install -g @anthropic-ai/claude-code
```

## Step 4: Set Up Your Anthropic API Key

Claude Code needs an API key to talk to Claude. Jim will provide this.

In your terminal, set it permanently:
```
setx ANTHROPIC_API_KEY "sk-ant-..."
```
Then **close and reopen** your terminal for it to take effect.

## Step 5: Clone the Repository

Pick a folder where you want the code to live (e.g., your Documents folder):
```
cd %USERPROFILE%\Documents
git clone https://github.com/jjbruin/Waterfall.git waterfall-xirr
cd waterfall-xirr
```

## Step 6: Set Up Python Virtual Environment

The codebase runs on Python. Set up the virtual environment so Claude can run and test code locally:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `python` isn't found, download Python 3.12 from https://www.python.org/downloads/ — during install, check "Add Python to PATH".

## Step 7: Launch Claude Code

From the project folder:
```
cd %USERPROFILE%\Documents\waterfall-xirr
claude
```

That's it! Claude loads the project context automatically and is ready to work.

---

## Daily Workflow

### Starting a session
```
cd %USERPROFILE%\Documents\waterfall-xirr
git pull
claude
```
Always `git pull` first to get the latest code.

### Describing a bug or issue
Just tell Claude what's wrong in plain English. Examples:

- "The NOI chart for Pontchartrain Landing shows $558K but the investor model shows $684K. The difference might be account 5092 being excluded. Investigate and fix."
- "The One Pager DSCR for Belleville is showing 1.234 but it should be 1.456. Check the debt service calculation."
- "The occupancy chart for Prestige Storage shows 654% which is obviously wrong. Find out why and fix it."

Claude will read the relevant code, explain what's happening, and propose a fix. You approve or deny each change.

### Saving your work
When Claude has made fixes you're happy with, tell it:
- "commit these changes" — saves locally with a description
- "push it" — sends to GitHub so Jim and the deploy pipeline can pick it up

### If something goes wrong
- Type `/undo` to reverse the last change Claude made
- Type "reset this file to what's in git" if a file gets messed up
- You can always `git pull` to get back to the latest clean version

---

## What You CAN Do
- Investigate and fix bugs in Python backend code
- Fix calculation logic (NOI, DSCR, occupancy, waterfall, etc.)
- Update account mappings in `config.py`
- Edit shared memory files (`.claude/memory/MEMORY.md`) to document findings
- Run the app locally to test changes

## What to Be Careful With
- **Deploying to Azure** — coordinate with Jim before deploying (the deploy commands are in CLAUDE.md if needed)
- **Database tables** — don't drop or delete tables; protected tables exist for a reason
- **Importing data** — use the app's CSV import UI rather than direct SQL

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start Claude | `claude` (from project folder) |
| Get latest code | `git pull` |
| See what changed | `git status` |
| Get help | Type `/help` in Claude |
| Exit Claude | `Ctrl+C` or type `/exit` |

---

## Need Help?

- **App login**: https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io (your admin credentials)
- **Code questions**: Ask Claude! It knows the entire codebase.
- **Access issues**: Message Jim
