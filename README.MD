# Azure DevOps Timesheet

A tiny, low-entry timesheet that syncs your **local time entries** to Azure DevOps Work Items by updating **Completed Work (hours)** while keeping a local record of every entry.

## What it does today

- **Local-first time entries** stored in SQLite (date, work item id, hours, optional note/category).
- **Work item cache** synced from Azure DevOps via a configured WIQL query (including each item's parent work item id, title, tags, state, original/remaining/completed work).
- **Sync to Azure DevOps** using the Work Item PATCH API:
  - Adds logged hours to Completed Work.
  - Optionally adjusts Remaining Work (strategy-based).
  - Saves sync receipts to prevent double counting.
- **Export** weekly entries to CSV or JSON, with optional totals grouped by parent work item.
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

By default, entries and receipts are stored per profile in SQLite (for example:
`~/.azdo_timesheet/profiles/default/timesheet.sqlite`).

Use `azdo-timesheet config show` to inspect the active profile, config path, and current
storage path. Use `azdo-timesheet config edit` to open the config file and adjust
`storage_path` when you want portable or repo-backed timesheet storage.

You can also opt into a **Markdown storage backend** for Azure DevOps Wiki workflows. This
creates a folder structure like:

```
timesheet/
  README.md
  entries.md
  entries/YYYY/YYYY-MM/YYYY-MM-DD.md
  entries/YYYY.md
  entries/YYYY/YYYY-MM.md
  receipts.md
  receipts/YYYY/YYYY-MM.md
  receipts/YYYY.md
  work_items.json
```

Each daily entry file includes a consistent table for humans plus a fenced `jsonl` or `yaml`
block that holds the canonical entry data (`entry_id`, `date`, `work_item_id`, etc.). The
Markdown parser reads only the fenced block, so you can safely edit the table without
breaking parsing. The top-level README links to recent days/weeks for quick navigation in
the Wiki UI. Folder pages include `[[_TOSP_]]` (Azure DevOps Table of Subpages) to keep
navigation working for folder nodes. Year/month folder pages plus daily pages and the index include summary tables
of work item totals and parent work item totals, each with a grand total for that period. When `org_url` and `project` are
configured, work item IDs in tables link directly to Azure DevOps work items, and page
titles include full dates (`YYYY-MM-DD` or `YYYY-MM` for month pages).
Daily pages are regenerated when entries change so the work item links stay up to date.
If the storage directory does not exist yet, it is created automatically when syncing work
items or writing entries.

If someone edits the fenced `Canonical Entry Data` block directly and the visible Markdown
table gets out of sync, use the emergency repair command below to rebuild the table from
the canonical stored data:

```bash
azdo-timesheet repair markdown-tables
```

Use this only as an emergency procedure after direct edits to the canonical fenced data.

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

Markdown storage writes to a per-profile location (for example
`~/.azdo_timesheet/profiles/default/timesheet`) by default. The layout matches Azure DevOps
Wiki expectations:

```
timesheet/
  README.md
  entries.md
  entries/YYYY/YYYY-MM/YYYY-MM-DD.md
  entries/YYYY.md
  entries/YYYY/YYYY-MM.md
  receipts.md
  receipts/YYYY/YYYY-MM.md
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
  entries/YYYY/YYYY-MM/YYYY-MM-DD.md
  receipts/YYYY/YYYY-MM.md
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

## Moneybird time registration export

The CLI can export local timesheet totals to Moneybird time registrations with:

```bash
azdo-timesheet moneybird export --week 2026-04-01
azdo-timesheet moneybird export --week 2026-04-01 --apply
```

The export groups entries by **day** and by the Azure DevOps **parent work item**. The
Moneybird `project_id` is read from a tag on the cached parent work item using the `mb:`
prefix, for example `mb:459275493454120239`. Child work items without a cached parent,
or parents without an `mb:` tag, are skipped and reported before anything is created.
The regular `list` output and `list --summary-by-parent` include an `mb_project` column
so you can see which Moneybird project will be used before exporting.

Moneybird settings are stored in the existing profile configuration and can be set during
`init` or `profile add`, or edited later with `azdo-timesheet config edit`:

```json
{
  "moneybird_administration_id": "123456789",
  "moneybird_user_id": "987654321",
  "moneybird_contact_id": "456789123",
  "moneybird_token_env_var": "MONEYBIRD_TOKEN",
  "moneybird_start_time": "08:00",
  "moneybird_lunch_break_start": "12:00",
  "moneybird_lunch_break_end": "13:00",
  "moneybird_dinner_break_start": "18:00",
  "moneybird_dinner_break_end": "19:30",
  "moneybird_timezone": "Z",
  "moneybird_billable": null
}
```

Because Moneybird requires a start and end timestamp for every time registration, generated
registrations for the same day are scheduled back-to-back by parent work item id. The first
registration starts at the configured start time (default `08:00`), and each next
registration starts when the previous one ends. Time automatically skips the configured
lunch break (default `12:00`-`13:00`) and dinner break (default `18:00`-`19:30`). If a
registration crosses a break, the Moneybird request includes `paused_duration` for that
break, and the dry-run/Markdown previews show the break minutes. The description explicitly
marks the registration as an automated export from the local timesheets and references the
Azure DevOps parent plus aggregated child work items. Markdown storage pages also include
a **Moneybird Export Preview** on each daily page, showing the parent, Moneybird project,
total hours, computed start/end timestamps, break minutes, and generated description.
Parent summary tables in Markdown include the Moneybird project id as well.

`moneybird_timezone` is appended to generated timestamps. Use an explicit offset such as
`+01:00` for CET or `+02:00` for CEST/summer time. `Z` means UTC. The tool does not
currently auto-switch between CET and CEST, so choose the offset that matches the period
you are exporting.

Moneybird export is dry-run by default. Use `--apply` to create the registrations through
the Moneybird API. Set `moneybird_user_id` to the Moneybird user the registrations should
be booked for. Set `moneybird_contact_id` to book each generated registration against a
fixed Moneybird contact; the export sends this as `contact_id` in the Moneybird time entry.
Configure a Bearer token in the environment variable named by
`moneybird_token_env_var` (default `MONEYBIRD_TOKEN`). The Moneybird API endpoint used is
`POST /api/v2/{administration_id}/time_entries.json`.

`moneybird export --apply` only runs when every local entry in each exported day is already
synced to Azure DevOps. If any entry in the selected period is still unsynced, the command
stops before creating Moneybird registrations and lists the unsynced day/entry ids. Run
`azdo-timesheet sync --apply` first, then re-run the Moneybird apply command.

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

If a work item has no Remaining Work field value at all, dry-run sync prompts for a value
only in interactive mode. Non-interactive sync leaves Remaining Work unset and still
updates Completed Work.

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
The entry list is printed as an aligned table for quick scanning, includes a `parent_wi`
column, truncates long notes to 77 characters plus `...`, and always appends a per-day
hours summary with a total for the selected result set.

Filter list output by parent work item id:
```bash
azdo-timesheet list --parent_wi 5678
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

### 9) Export entries for a week or period
```bash
azdo-timesheet export --week 2024-06-24 --format csv --output week.csv
```

Include parent totals in the export (and include `parent_work_item_id` beside each `work_item_id` entry):
```bash
azdo-timesheet export --week 2024-06-24 --format json --summary-by-parent
```

Export an explicit date period instead of a week:
```bash
azdo-timesheet export --start 2024-06-01 --end 2024-06-30 --format csv
```

### 10) View cached work items
```bash
azdo-timesheet wi list
```
The work item list renders as an aligned table and includes the parent work item id and
cached tags columns.

### 11) Show or edit config
```bash
azdo-timesheet config show
azdo-timesheet config edit
```
Use these commands to inspect the resolved config file, active profile, and current
storage path, or to open the config for manual edits.

### 12) Emergency repair for Markdown tables
```bash
azdo-timesheet repair markdown-tables
```
This rebuilds the visible Markdown entry tables from the fenced `Canonical Entry Data`
for Markdown storage profiles only. Use it only after someone edited the canonical data
directly.

### 13) View parent summary from the CLI
```bash
azdo-timesheet list --summary-by-parent
```

List entries for an explicit period:
```bash
azdo-timesheet list --start 2024-06-01 --end 2024-06-30 --parent_wi 5678
```

---

## Multiple orgs/projects (profiles)

You can configure multiple Azure DevOps org/project pairs as named **profiles**. Each
profile has its own storage path so entries, work items, and receipts never overlap.

### Profile config format

```json
{
  "default_profile": "default",
  "profiles": {
    "default": {
      "org_url": "https://dev.azure.com/your-org",
      "project": "MyProject",
      "auth_mode": "pat",
      "pat_env_var": "AZDO_PAT",
      "remaining_work_strategy": "none",
      "allow_sync_closed_items": false,
      "max_hours_per_entry": 8,
      "storage_backend": "sqlite",
      "storage_path": "~/.azdo_timesheet/profiles/default/timesheet.sqlite",
      "wiql_query": "Select [System.Id] From WorkItems Where [System.AssignedTo] = @Me"
    },
    "another": {
      "org_url": "https://dev.azure.com/another-org",
      "project": "AnotherProject",
      "storage_backend": "markdown",
      "storage_path": "~/.azdo_timesheet/profiles/another/timesheet"
    }
  }
}
```

### Add a new profile

```bash
azdo-timesheet profile add another --org-url https://dev.azure.com/another-org --project AnotherProject
```

To initialize the first profile with a custom name:

```bash
azdo-timesheet --profile default init --org-url https://dev.azure.com/your-org --project MyProject
```

### Switch the default profile

```bash
azdo-timesheet profile use another
```

### List profiles

```bash
azdo-timesheet profile list
```

### Override the profile for a single command

```bash
azdo-timesheet --profile another add --wi 1234 --t 1.0
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
