# Azure DevOps Timesheet

A tiny, low-entry timesheet that syncs your **local time entries** to Azure DevOps Work Items by updating **Completed Work (hours)** — while giving you a clean local record with the details behind those numbers.

## Why this exists

Azure DevOps has great tracking fields, but it doesn’t feel like a timesheet:
- You often just want: “Log 1.25h on WI #1234 with a note”
- You also want context: **Original Estimate**, **Remaining Work**, your **delta** today, and what changed
- And you want a local record you can export, audit, or reuse for invoicing/time reporting

This tool is **offline-first** and **sync-optional**:
- **Local timesheet** is the source of truth for your detailed entries
- **Azure DevOps** gets updated with the rolled-up Completed Work (and optionally Remaining Work)

---

## Goals

### Primary
- Log time locally with *minimal typing*
- Sync to Azure DevOps:
  - Update `Microsoft.VSTS.Scheduling.CompletedWork` (Completed Work / CompletedHours)
  - Read and display:
    - `Microsoft.VSTS.Scheduling.OriginalEstimate`
    - `Microsoft.VSTS.Scheduling.RemainingWork`

### Secondary
- Be fast enough to use daily (sub-30 seconds to log time)
- Provide a “what changed?” audit trail
- Support export (CSV/JSON) and future integrations (e.g., invoicing tools)

---

## What “low entry” means (design principles)

- **One command / one screen** to log time
- Smart defaults:
  - today’s date
  - last-used work items
  - your assigned items / current iteration items
- No required “admin” structure up-front (projects/cost-centers optional)
- Sync can be done later (end of day/week)

---

## Core Features (MVP)

### 1) Local Timesheet Entries
Each entry captures:
- Date
- Work Item ID
- Hours
- Optional: note / activity
- Optional: category (support/dev/meeting)
- Optional: “billable” flag / customer code

### 2) Work Item Context (read-only)
When you pick a WI, show:
- Title, Type, State
- Original Estimate, Remaining Work, Completed Work
- Your last logged entry for that WI

### 3) Sync (write back to AzDo)
Sync turns local entries into updates to Azure DevOps:
- Adds `Δ hours` to Completed Work
- Optionally adjusts Remaining Work (configurable strategy)
- Records a local “sync receipt” so re-sync doesn’t double-count

---

## Cool Features (near-term ideas)

### Smart Remaining Work Strategies (configurable)
When logging time, you can choose how Remaining Work should behave:
1. **Do not change Remaining Work** (safe default)
2. **Decrease Remaining Work by logged hours** (common agile flow)
3. **Recalculate Remaining Work from estimate**:
   - `Remaining = max(OriginalEstimate - CompletedWork, 0)`
4. **Manual prompt** when you log time:
   - “Logged 1.5h — Remaining was 6.0h, set to: [4.5] (enter to accept)”

### “Timesheet from my work” picker
- List work items:
  - assigned to me
  - in current sprint/iteration
  - recently updated by me
  - from recent timesheet usage (“recents”)

### Daily/Weekly summary view
- Total hours per day
- Totals per work item
- Totals per project/team area path (if available)
- Notes roll-up (copy/paste-ready for status updates)

### Gentle safety rails
- **Dry-run** mode: show patch operations before applying
- Warnings when:
  - Remaining Work would go negative
  - Completed Work jumps unusually high vs recent patterns
  - WI is closed but you’re trying to add hours

### Field mapping & process compatibility
Allow mapping for custom processes:
- Completed Hours field name could differ in some organizations
- Mapping lives in config

### Export
- CSV export per week (for invoicing)
- JSON export (for automation / future Moneybird-like flows)

---

## Azure DevOps Fields Used

Default Azure DevOps scheduling fields:
- `Microsoft.VSTS.Scheduling.OriginalEstimate`
- `Microsoft.VSTS.Scheduling.RemainingWork`
- `Microsoft.VSTS.Scheduling.CompletedWork`

> Note: Many UIs label “Completed Work” as “Completed Hours”. This tool uses the field above unless configured otherwise.

---

## Data Storage

Local storage options (pick one; start simple):
- **Single file JSON** (lowest entry)
- **CSV** (easy export)
- **SQLite** (best long-term: fast queries + safe receipts)

Recommended starter: **SQLite** to avoid double-sync and to support summaries cleanly.

---

## Authentication

Support one (or more) of:
- **PAT (Personal Access Token)** (fastest to start)
- OAuth Device Code flow (later)

Minimum permissions typically needed:
- Work Items: read + write

---

## Quick Start (intended UX)

> The exact commands will depend on the implementation (PowerShell/.NET/Python), but the user journey is the point.

### 1) Configure
- org URL
- project (optional if you query by WI id directly)
- auth (PAT)
- storage location

### 2) Log time (fast path)
- pick work item from “assigned to me” or “recent”
- type hours
- optional note

### 3) Sync
- show what will change
- apply updates
- store receipts to prevent double counting

---

## Example Workflows

### End-of-day
1. Log time throughout the day quickly (local only)
2. Run `sync` at the end of the day
3. Export a summary for your daily status update

### End-of-week
1. Log locally all week
2. Sync Friday afternoon
3. Export CSV for invoicing/reporting

---

## Roadmap

### v0 (MVP)
- Local entries (add/list)
- Fetch WI details
- Sync to Completed Work
- Sync receipts to avoid double counting

### v1
- Remaining Work strategies
- Recents + assigned-to-me picker
- Weekly export

### vNext
- UI (TUI or minimal web)
- OAuth device login
- Rules for closed items, validations, policies

---

## Contributing

See `AGENT.md` for implementation notes, architecture decisions, and coding conventions.

---

## License

TBD
