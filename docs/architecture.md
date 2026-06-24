# Architecture Overview

Azure DevOps Timesheet is a CLI-first local timesheet. The detailed local
timesheet is the source of truth; Azure DevOps and Moneybird are downstream
systems updated from local entries.

## Layer Model

```mermaid
flowchart TB
    user["User"]
    cli["CLI UX Layer<br/>argparse commands, prompts, dry-run output"]
    app["Application Services<br/>entry formatting, exports, Moneybird planning"]
    sync["Sync Engine<br/>group entries, compute deltas, create receipts"]
    storage["Storage Layer<br/>SQLite or Markdown canonical data"]
    azdo["Azure DevOps Client<br/>WIQL, work item reads, JSON Patch"]
    moneybird["Moneybird Client<br/>time registration requests"]

    user --> cli
    cli --> app
    cli --> sync
    app --> storage
    sync --> storage
    sync --> azdo
    app --> moneybird
    azdo -->|"work item cache, revisions"| sync
    storage -->|"entries, work items, receipts"| app
    storage -->|"unsynced entries, receipts"| sync
```

### Layer Responsibilities

| Layer | Responsibilities |
| --- | --- |
| CLI UX Layer | Parse commands, keep entry logging fast, print trustable dry-run and list output. |
| Application Services | Format reports, plan CSV/JSON and Moneybird exports, apply report-hour rounding. |
| Sync Engine | Group unsynced entries by work item, compute Completed Work deltas, apply Remaining Work strategy, save receipts after successful sync. |
| Storage Layer | Persist entries, cached work items, and receipts. SQLite and Markdown backends expose the same storage behavior. |
| Azure DevOps Client | Read work items through WIQL/API calls and apply JSON Patch updates. |
| Moneybird Client | Create time registrations from locally planned day and parent-work-item totals. |

## Data Ownership

Local storage owns detailed time entry data. Azure DevOps owns work item fields
such as title, state, Completed Work, Original Estimate, and Remaining Work.
Moneybird owns created time registrations. Receipts connect local entries to the
Azure DevOps patch that applied them, which keeps sync idempotent.

Human-facing reports round each entry to the nearest quarter hour before
summing. Canonical entry data keeps the exact stored `hours` value so exports
and sync receipts can preserve the original input.

## Add And List Entries

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Storage
    participant AzDo
    participant Moneybird

    User->>CLI: azdo-timesheet add --wi 1234 --t 1.25 --note "..."
    CLI->>Storage: add_entry(entry)
    Storage-->>CLI: persisted entry
    CLI-->>User: Added entry id

    User->>CLI: azdo-timesheet list --start ... --end ...
    CLI->>Storage: list_entries_range(period)
    Storage-->>CLI: entries + cached work items
    CLI-->>User: aligned table + report totals

    Note over CLI,AzDo: Local add/list does not call Azure DevOps.
    Note over CLI,Moneybird: Local add/list does not call Moneybird.
```

## Work Item Cache Sync

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Storage
    participant AzDo
    participant Moneybird

    User->>CLI: azdo-timesheet wi sync
    CLI->>AzDo: run configured WIQL / fetch work item fields
    AzDo-->>CLI: work item states
    CLI->>Storage: upsert_work_items(items)
    Storage-->>CLI: cache updated
    CLI-->>User: synced work item count

    Note over CLI,Moneybird: Work item cache sync does not contact Moneybird.
```

## Azure DevOps Time Sync

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Storage
    participant AzDo
    participant Moneybird

    User->>CLI: azdo-timesheet sync
    CLI->>Storage: get_unsynced_entries()
    Storage-->>CLI: unsynced local entries
    CLI->>AzDo: read current work item fields
    AzDo-->>CLI: before snapshots
    CLI->>CLI: group by work item and compute deltas
    CLI-->>User: dry-run summary

    User->>CLI: azdo-timesheet sync --apply
    CLI->>Storage: get_unsynced_entries()
    Storage-->>CLI: unsynced local entries
    CLI->>AzDo: PATCH Completed Work / Remaining Work
    AzDo-->>CLI: updated work item revision/state
    CLI->>Storage: record_receipt(receipt) and mark entry synced
    Storage-->>CLI: receipt persisted
    CLI-->>User: success/failure per work item

    Note over CLI,Moneybird: Azure DevOps sync does not create Moneybird registrations.
```

## Moneybird Export

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Storage
    participant AzDo
    participant Moneybird

    User->>CLI: azdo-timesheet moneybird export --week ...
    CLI->>Storage: list_entries_range(period)
    Storage-->>CLI: entries + cached parent work items
    CLI->>CLI: group by day and parent work item
    CLI-->>User: dry-run registrations and missing mb: tag errors

    User->>CLI: azdo-timesheet moneybird export --week ... --apply
    CLI->>Storage: verify exported days have no unsynced entries
    Storage-->>CLI: sync status for local entries
    CLI->>Moneybird: POST time_entries.json
    Moneybird-->>CLI: created registration id
    CLI-->>User: created registration summary

    Note over CLI,AzDo: Moneybird export uses cached AzDo parent data from Storage.
```

## Safety Invariants

- Entry logging must stay local and fast.
- Dry-run output must describe planned changes before external writes.
- Azure DevOps sync must record receipts only after a successful patch.
- Synced entries must not be applied to Azure DevOps a second time.
- Moneybird apply requires the exported local days to be synced to Azure DevOps first.
