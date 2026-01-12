# Agent Guide: Azure DevOps Timesheet

This file explains the project’s intent, architecture, and “guardrails” so an AI agent (or human contributor) can extend the tool without breaking the core promise: **low-entry local timesheets + safe AzDo sync**.

## Product Promise (non-negotiables)
1. Logging time must be *faster* than editing a work item manually.
2. The local timesheet is the **detailed source of truth**.
3. Sync must be **idempotent** (no double-counting).
4. The user can always run in **dry-run** mode and see changes before applying.

---

## Key Concepts

### Time Entry
A single local record:
- `entry_id` (GUID)
- `date` (YYYY-MM-DD)
- `work_item_id` (int)
- `hours` (decimal)
- `note` (string, optional)
- `category` (optional)
- `created_at` / `updated_at`
- `synced` (boolean or derived)
- `sync_receipts[]` (0..n)

### Sync Receipt (idempotency)
A receipt ties local entries to a specific Azure DevOps update:
- `entry_id`
- `work_item_id`
- `delta_completed_work` (hours applied)
- `before` snapshot:
  - completed/original/remaining values
- `after` snapshot
- `azdo_revision` (if available)
- `synced_at`
- `patch_document` (stored for audit/debug)

Rule: **Never apply the same entry twice.**  
Implementation: mark entries as synced only after successful patch + receipt saved.

---

## Azure DevOps API Notes

### Fields (defaults)
- Completed Work: `Microsoft.VSTS.Scheduling.CompletedWork`
- Original Estimate: `Microsoft.VSTS.Scheduling.OriginalEstimate`
- Remaining Work: `Microsoft.VSTS.Scheduling.RemainingWork`

### Update mechanism
Use Work Item PATCH (JSON Patch):
- Read current field values
- Calculate intended deltas
- Apply patch operations
- Re-read or trust response to capture post-state

### Robustness requirements
- If field missing/null: treat as 0 for calculations, but record the null in receipts
- Handle closed states:
  - Default behavior: warn and require `--force` to sync hours to closed items (configurable)

---

## Remaining Work Strategies (design)
Make Remaining Work behavior explicit and configurable:

1. `none`  
   Do not change Remaining Work.

2. `decrement`  
   `RemainingWork = max(RemainingWork - logged_hours, 0)`

3. `recalc_from_original`  
   `RemainingWork = max(OriginalEstimate - (CompletedWork + logged_hours), 0)`

4. `prompt` (interactive)
   Suggest one of the above, allow manual override.

For non-interactive runs, `prompt` must fall back to a safe default (`none`).

---

## Suggested Modules / Layers

### 1) Storage
Responsibilities:
- CRUD time entries
- Query summaries (daily/weekly)
- Manage receipts
- Export formats

### 2) AzDo Client
Responsibilities:
- Auth
- Fetch work items
- Apply JSON Patch updates
- Handle retry/backoff for transient errors

### 3) Sync Engine
Responsibilities:
- Compute changes per work item:
  - group entries by work_item_id (per sync batch)
  - compute `Δ completed`
- Apply Remaining Work strategy
- Create receipts
- Ensure idempotency

### 4) UX Layer (CLI first)
Responsibilities:
- “Add entry” fast path
- “Pick a work item” sources:
  - recents
  - assigned to me
  - current iteration (optional)
- `dry-run` output should be extremely clear

---

## CLI Design (low-entry UX)

### Commands (recommended)
- `init`  
  Create config + storage
- `add`  
  Add a time entry (fast path)
- `list`  
  Show entries (filters: day/week/workitem)
- `wi`  
  Fetch and show WI details
- `sync`  
  Apply local deltas to AzDo + generate receipts
- `export`  
  CSV/JSON weekly

### Fast-path requirements for `add`
- Accept `--wi 1234 --h 1.25 --note "..."`  
- If `--wi` omitted:
  - show picker (recents first)
- If `--note` omitted:
  - no prompt (keep it frictionless)

---

## Output Requirements (make it trustable)
For `sync --dry-run`, print per WI:
- WI id + title
- Completed: `before -> after` (+delta)
- Remaining: `before -> after` (strategy shown)
- Number of entries included
- Receipt IDs that would be created

For real sync:
- same as dry-run, plus confirmation of success/failure per item
- store errors with enough detail to re-run safely

---

## Testing Approach
- Unit tests for:
  - remaining work strategies
  - grouping entries by work item
  - idempotency logic (receipts)
- Integration tests (optional) using a mocked AzDo endpoint:
  - verify JSON Patch formation
  - verify safe behavior when fields are null/missing

---

## Config (recommended)
Config should live in a single file (YAML/JSON):
- org URL
- default project (optional)
- auth mode (PAT/OAuth)
- field mappings (defaults above)
- remaining work strategy
- safety:
  - allow_sync_closed_items (false by default)
  - max_hours_per_entry warning threshold (e.g. 8)
- storage location

---

Regardless of stack:
- prioritize **speed of logging** and **sync safety**

---

## Instruction on edit
- Always update the README.md
- Always update the CLI help

