# Azure DevOps Timesheet

A tiny, low-entry timesheet that syncs your **local time entries** to Azure DevOps Work Items by updating **Completed Work (hours)** while keeping a local record of every entry.

## What it does today

- **Local-first time entries** stored in SQLite (date, work item id, hours, optional note/category).
- **Work item cache** synced from Azure DevOps via a configured WIQL query (title, state, original/remaining/completed work).
- **Sync to Azure DevOps** using the Work Item PATCH API:
  - Adds logged hours to Completed Work.
  - Optionally adjusts Remaining Work (strategy-based).
  - Saves sync receipts to prevent double counting.
- **Export** weekly entries to CSV or JSON.
- **Edit/remove** unsynced entries.

---

## Azure DevOps Fields Used

Default Azure DevOps scheduling fields:
- `Microsoft.VSTS.Scheduling.OriginalEstimate`
- `Microsoft.VSTS.Scheduling.RemainingWork`
- `Microsoft.VSTS.Scheduling.CompletedWork`

> Note: Many UIs label “Completed Work” as “Completed Hours”. This tool uses the field above unless configured otherwise.

---

## Storage

By default, entries and receipts are stored in SQLite (`~/.azdo_timesheet/timesheet.sqlite`).

You can also opt into a **Markdown storage backend** for Azure DevOps Wiki workflows. This
creates a folder structure like:

```
timesheet/
  README.md
  entries.md
  entries/YYYY/MM/DD.md
  entries/YYYY.md
  entries/YYYY/MM.md
  receipts.md
  receipts/YYYY/MM.md
  receipts/YYYY.md
  work_items.json
```

Each daily entry file includes a consistent table for humans plus a fenced `jsonl` or `yaml`
block that holds the canonical entry data (`entry_id`, `date`, `work_item_id`, etc.). The
Markdown parser reads only the fenced block, so you can safely edit the table without
breaking parsing. The top-level README links to recent days/weeks for quick navigation in
the Wiki UI. Folder pages include `[[_TOSP_]]` (Azure DevOps Table of Subpages) to keep
navigation working for folder nodes. Year/month folder pages also include summary tables
of work item totals plus a grand total for that period. When `org_url` and `project` are
configured, work item IDs in tables link directly to Azure DevOps work items, and page
titles include full dates (`YYYY-MM-DD` or `YYYY-MM` for month pages).

---

## Wiki mode

Use **Wiki mode** when you want your timesheet files to live in a repo and render as an
Azure DevOps Wiki.

### Enable Markdown storage

Set `storage_backend: markdown` in your `config.json`, or initialize with:

```bash
azdo-timesheet init --storage-backend markdown --storage ~/.azdo_timesheet/timesheet
```

### Where files are written

Markdown storage writes to `~/.azdo_timesheet/timesheet` by default (or the `--storage`
path you provide). The layout matches Azure DevOps Wiki expectations:

```
timesheet/
  README.md
  entries.md
  entries/YYYY/MM/DD.md
  entries/YYYY.md
  entries/YYYY/MM.md
  receipts.md
  receipts/YYYY/MM.md
  receipts/YYYY.md
  work_items.json
```

### Publish as an Azure DevOps Wiki

1. Commit and push the `timesheet/` folder to a repo.
2. In Azure DevOps, go to **Wiki** → **Create Wiki** → **Publish code as wiki**.
3. Select the repo (and path) that contains `timesheet/`, or link the repo to a project
   wiki if you already have one.

> **Merge conflict guidance:** treat daily entry files as append-only. Avoid in-place edits
> to past entries, and prefer adding new rows for corrections. Pull/rebase before logging
> time if multiple people commit to the same wiki repo.

When `org_url` and `project` are configured in `config.json`, wiki tables turn work item IDs
into links to the Azure DevOps work items, and page titles include full dates (`YYYY-MM-DD`
or `YYYY-MM` for month pages) for better searchability.

---

## Wiki mode

Use **Wiki mode** when you want your timesheet files to live in a repo and render as an
Azure DevOps Wiki.

### Enable Markdown storage

Set `storage_backend: markdown` in your `config.json`, or initialize with:

```bash
azdo-timesheet init --storage-backend markdown --storage ~/.azdo_timesheet/timesheet
```

### Where files are written

Markdown storage writes to `~/.azdo_timesheet/timesheet` by default (or the `--storage`
path you provide). The layout matches Azure DevOps Wiki expectations:

```
timesheet/
  README.md
  entries/YYYY/MM/DD.md
  receipts/YYYY/MM.md
  work_items.json
```

### Publish as an Azure DevOps Wiki

1. Commit and push the `timesheet/` folder to a repo.
2. In Azure DevOps, go to **Wiki** → **Create Wiki** → **Publish code as wiki**.
3. Select the repo (and path) that contains `timesheet/`, or link the repo to a project
   wiki if you already have one.

> **Merge conflict guidance:** treat daily entry files as append-only. Avoid in-place edits
> to past entries, and prefer adding new rows for corrections. Pull/rebase before logging
> time if multiple people commit to the same wiki repo.

When `org_url` and `project` are configured in `config.json`, wiki tables turn work item IDs
into links to the Azure DevOps work items, and page titles include full dates (`YYYY-MM-DD`
or `YYYY-MM` for month pages) for better searchability.

---

## Authentication

The CLI uses a Personal Access Token (PAT). Set the token in the environment variable configured in your `config.json` (default: `AZDO_PAT`).

Minimum permissions typically needed:
- Work Items: read + write

---

## Remaining Work strategies

Choose how Remaining Work is updated during sync:
- `none` (default): do not change Remaining Work.
- `decrement`: subtract logged hours, clamped to zero.
- `recalc_from_original`: `Remaining = max(OriginalEstimate - CompletedWork, 0)`.
- `interactive`: prompt during dry-run with the current Remaining Work plus Completed Work before/after and a suggested default.

---

## Quick Start (current CLI)

### 1) Install (editable)
```bash
python -m pip install -e .
```

### 2) Initialize storage
```bash
azdo-timesheet init --org-url https://dev.azure.com/your-org --project MyProject --wiql "Select [System.Id] From WorkItems Where [System.AssignedTo] = @Me"
```

To use Markdown storage instead of SQLite:
```bash
azdo-timesheet init --storage-backend markdown --storage ~/.azdo_timesheet/timesheet
```

Set your PAT so WIQL sync can fetch work items:
```bash
export AZDO_PAT=your-token
```

### 3) Add an entry (fast path)
```bash
azdo-timesheet add --wi 1234 --t 1.25 --note "Bug triage"
```

If you omit `--wi`, the CLI will show recent work items to pick from.

### 4) Sync work items from WIQL (manual refresh)
```bash
azdo-timesheet wi sync
```

### 5) List entries
```bash
azdo-timesheet list --wi 1234
```

### 6) Dry-run sync (preview)
```bash
azdo-timesheet sync
```

The dry-run output shows Completed Work before/after, plus Remaining Work behavior (including any prompt inputs).

### 7) Apply sync updates to Azure DevOps
```bash
azdo-timesheet sync --apply
```

`sync --apply` updates the work items in Azure DevOps, stores receipts, and refreshes the local work item list from the configured WIQL query. Use `--skip-wi-sync` to skip the download step.

### 8) Edit or remove unsynced entries
```bash
azdo-timesheet edit --t 2.0 --note "Updated note"
azdo-timesheet remove
```

You can still pass `--id <entry-id>` if you want to target a specific entry directly.

### 9) Export a week
```bash
azdo-timesheet export --week 2024-06-24 --format csv --output week.csv
```

### 10) View cached work items
```bash
azdo-timesheet wi list
```

---

## Example Workflows

### End-of-day
1. Log time throughout the day.
2. Run `sync` to preview changes.
3. Run `sync --apply` to update Azure DevOps.

### End-of-week
1. Log locally all week.
2. Run `sync --apply` Friday afternoon.
3. Export CSV for invoicing/reporting.

### Azure DevOps Wiki (Markdown storage)
1. Initialize the Markdown storage backend.
2. Commit the `timesheet/` folder to a repo connected to an Azure DevOps Wiki.
3. Browse the Markdown UI to review daily entries and receipts. The `README.md` index links
   recent days and weeks.

---

## Contributing

See `AGENTS.MD` for implementation notes, architecture decisions, and coding conventions.

---

## License

TBD
