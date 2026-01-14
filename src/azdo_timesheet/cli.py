import argparse
import base64
import csv
import json
import os
import sys
import textwrap
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib import error, parse, request
from typing import Sequence

from .models import (
    Config,
    Entry,
    Receipt,
    WorkItem,
    WorkItemDelta,
    WorkItemState,
)
from .storage import MarkdownStorage, SQLiteStorage

DEFAULT_CONFIG_DIR = Path.home() / ".azdo_timesheet"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_DB_PATH = DEFAULT_CONFIG_DIR / "timesheet.sqlite"
DEFAULT_MARKDOWN_ROOT = DEFAULT_CONFIG_DIR / "timesheet"


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Run 'azdo-timesheet init' first."
        )
    data = json.loads(path.read_text())
    storage_backend = data.get("storage_backend", "sqlite")
    if data.get("storage_path"):
        storage_path = Path(data["storage_path"]).expanduser()
    else:
        storage_path = (
            DEFAULT_MARKDOWN_ROOT if storage_backend == "markdown" else DEFAULT_DB_PATH
        )
    return Config(
        org_url=data.get("org_url", ""),
        project=data.get("project"),
        auth_mode=data.get("auth_mode", "pat"),
        pat_env_var=data.get("pat_env_var", "AZDO_PAT"),
        remaining_work_strategy=data.get("remaining_work_strategy", "none"),
        allow_sync_closed_items=bool(data.get("allow_sync_closed_items", False)),
        max_hours_per_entry=float(data.get("max_hours_per_entry", 8)),
        storage_backend=storage_backend,
        storage_path=storage_path,
        wiql_query=data.get("wiql_query"),
    )


def save_config(path: Path, config: Config) -> None:
    payload = {
        "org_url": config.org_url,
        "project": config.project,
        "auth_mode": config.auth_mode,
        "pat_env_var": config.pat_env_var,
        "remaining_work_strategy": config.remaining_work_strategy,
        "allow_sync_closed_items": config.allow_sync_closed_items,
        "max_hours_per_entry": config.max_hours_per_entry,
        "storage_backend": config.storage_backend,
        "storage_path": str(config.storage_path),
        "wiql_query": config.wiql_query,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

def get_storage(config: Config) -> SQLiteStorage | MarkdownStorage:
    if config.storage_backend == "markdown":
        return MarkdownStorage(config.storage_path)
    return SQLiteStorage(config.storage_path)


def init_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if args.storage:
        storage_path = Path(args.storage).expanduser().resolve()
    else:
        storage_path = (
            DEFAULT_MARKDOWN_ROOT
            if args.storage_backend == "markdown"
            else DEFAULT_DB_PATH
        )
    config = Config(
        org_url=args.org_url or "",
        project=args.project,
        auth_mode="pat",
        pat_env_var=args.pat_env_var,
        remaining_work_strategy=args.remaining_work_strategy,
        allow_sync_closed_items=False,
        max_hours_per_entry=float(args.max_hours_per_entry),
        storage_backend=args.storage_backend,
        storage_path=storage_path,
        wiql_query=args.wiql_query,
    )
    save_config(config_path, config)
    storage = get_storage(config)
    storage.init()
    print(f"Initialized config at {config_path}")
    print(f"Storage: {storage_path}")
    return 0


def get_recent_work_items(
    storage: SQLiteStorage | MarkdownStorage, *, limit: int = 5
) -> list[tuple[int, str | None]]:
    return storage.get_recent_work_items(limit=limit)


def prompt_for_work_item(storage: SQLiteStorage | MarkdownStorage) -> int:
    if not sys.stdin.isatty():
        raise ValueError("Work item id required when running non-interactively.")
    recents = get_recent_work_items(storage)
    if not recents:
        raise ValueError("No recent work items found. Provide --wi.")
    print("Recent work items:")
    for index, (work_item_id, title) in enumerate(recents, start=1):
        title_text = f" - {title}" if title else ""
        print(f"  [{index}] {work_item_id}{title_text}")
    selection = input("Pick a work item number or enter an id: ").strip()
    if selection.isdigit():
        choice = int(selection)
        if 1 <= choice <= len(recents):
            return recents[choice - 1][0]
        return choice
    raise ValueError("Invalid selection. Provide a numeric work item id.")


def add_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    if args.hours <= 0:
        print("Hours must be greater than zero.", file=sys.stderr)
        return 2
    if args.hours > config.max_hours_per_entry:
        print(
            f"Warning: hours {args.hours} exceed warning threshold "
            f"({config.max_hours_per_entry}).",
            file=sys.stderr,
        )
    entry_date = args.date or date.today().isoformat()
    now = datetime.utcnow().isoformat()
    try:
        work_item_id = int(args.work_item_id) if args.work_item_id else None
    except ValueError:
        print("Work item id must be a number.", file=sys.stderr)
        return 2
    if work_item_id is None:
        try:
            work_item_id = prompt_for_work_item(storage)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    entry = Entry(
        entry_id=str(uuid.uuid4()),
        entry_date=entry_date,
        work_item_id=work_item_id,
        hours=float(args.hours),
        note=args.note,
        category=args.category,
        created_at=now,
        updated_at=now,
        synced=0,
    )
    storage.add_entry(entry)
    print(f"Added entry {entry.entry_id} for WI #{entry.work_item_id}")
    return 0


def format_entries(entries: Sequence[Entry]) -> str:
    if not entries:
        return "No entries found."
    lines = [
        "idx | entry_id | date | wi | hours | synced | note",
        "-" * 72,
    ]
    for idx, entry in enumerate(entries, start=1):
        note = (entry.note or "").replace("\n", " ")
        short_id = entry.entry_id.split("-")[0]
        lines.append(
            f"{idx:>3} | {short_id} | {entry.entry_date} | {entry.work_item_id} | "
            f"{entry.hours:.2f} | {entry.synced} | {note}"
        )
    return "\n".join(lines)


def select_entry_id(
    storage: SQLiteStorage | MarkdownStorage, *, include_synced: bool = False
) -> str:
    if not sys.stdin.isatty():
        raise ValueError("Entry id required when running non-interactively.")
    entries = storage.list_recent_entries(limit=20, include_synced=include_synced)
    if not entries:
        raise ValueError("No entries available to select.")
    print(format_entries(entries))
    selection = input("Pick an entry number: ").strip()
    if not selection.isdigit():
        raise ValueError("Invalid selection. Provide an entry number.")
    idx = int(selection)
    if not 1 <= idx <= len(entries):
        raise ValueError("Selection out of range.")
    return entries[idx - 1].entry_id


def list_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    entries = storage.list_entries(
        work_item_id=args.work_item_id,
        entry_date=args.date,
    )
    print(format_entries(entries))
    return 0


def edit_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    updates: dict[str, object] = {}
    if args.work_item_id is not None:
        updates["work_item_id"] = int(args.work_item_id)
    if args.hours is not None:
        updates["hours"] = float(args.hours)
    if args.note is not None:
        updates["note"] = args.note
    if args.category is not None:
        updates["category"] = args.category
    if args.date is not None:
        updates["entry_date"] = args.date
    if not updates:
        print("No fields provided to update.", file=sys.stderr)
        return 2
    now = datetime.utcnow().isoformat()
    updates["updated_at"] = now
    entry_id = args.entry_id
    if entry_id is None:
        try:
            entry_id = select_entry_id(storage)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    entry = storage.get_entry(entry_id)
    if entry is None:
        print("Entry not found.", file=sys.stderr)
        return 2
    if entry.synced:
        print("Cannot edit synced entries.", file=sys.stderr)
        return 2
    storage.update_entry(entry_id, updates)
    print(f"Updated entry {entry_id}.")
    return 0


def remove_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    entry_ids = args.entry_id or []
    if not entry_ids:
        try:
            entry_ids = [select_entry_id(storage)]
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    for entry_id in entry_ids:
        entry = storage.get_entry(entry_id)
        if entry is None:
            print(f"Entry {entry_id} not found.", file=sys.stderr)
            return 2
        if entry.synced:
            print(f"Entry {entry_id} is synced and cannot be removed.", file=sys.stderr)
            return 2
    storage.remove_entries(entry_ids)
    print(f"Removed {len(entry_ids)} entries.")
    return 0


def format_work_items(work_items: Sequence[WorkItem]) -> str:
    if not work_items:
        return "No work items found."
    lines = [
        "wi | title | state | original | remaining | completed",
        "-" * 88,
    ]
    for item in work_items:
        lines.append(
            f"{item.work_item_id} | {item.title or ''} | {item.state or ''} | "
            f"{item.original_estimate if item.original_estimate is not None else ''} | "
            f"{item.remaining_work if item.remaining_work is not None else ''} | "
            f"{item.completed_work if item.completed_work is not None else ''}"
        )
    return "\n".join(lines)

def work_item_list_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    work_items = storage.list_work_items()
    print(format_work_items(work_items))
    return 0


def pat_token(config: Config) -> str:
    token = os.environ.get(config.pat_env_var, "")
    if not token:
        raise ValueError(
            f"Missing PAT. Set {config.pat_env_var} in the environment to sync work items."
        )
    return token


def azdo_request(
    *,
    config: Config,
    method: str,
    path: str,
    payload: dict | list | None = None,
    content_type: str = "application/json",
) -> dict:
    if not config.org_url:
        raise ValueError("org_url is required to sync work items.")
    if not config.project:
        raise ValueError("project is required to sync work items.")
    token = pat_token(config)
    auth = base64.b64encode(f":{token}".encode("utf-8")).decode("utf-8")
    url = f"{config.org_url.rstrip('/')}/{config.project}/{path}"
    body = None
    headers = {"Authorization": f"Basic {auth}", "Content-Type": content_type}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise ValueError(f"Azure DevOps request failed: {exc.code} {detail}") from exc


def work_item_fields() -> list[str]:
    return [
        "System.Title",
        "System.State",
        "Microsoft.VSTS.Scheduling.OriginalEstimate",
        "Microsoft.VSTS.Scheduling.RemainingWork",
        "Microsoft.VSTS.Scheduling.CompletedWork",
    ]


def parse_work_item(work_item: dict) -> WorkItemState:
    fields_data = work_item.get("fields", {})
    return WorkItemState(
        item=WorkItem(
            work_item_id=work_item.get("id"),
            title=fields_data.get("System.Title"),
            state=fields_data.get("System.State"),
            original_estimate=fields_data.get(
                "Microsoft.VSTS.Scheduling.OriginalEstimate"
            ),
            remaining_work=fields_data.get("Microsoft.VSTS.Scheduling.RemainingWork"),
            completed_work=fields_data.get("Microsoft.VSTS.Scheduling.CompletedWork"),
            updated_at=datetime.utcnow().isoformat(),
        ),
        has_original_estimate="Microsoft.VSTS.Scheduling.OriginalEstimate"
        in fields_data,
        has_remaining_work="Microsoft.VSTS.Scheduling.RemainingWork" in fields_data,
        has_completed_work="Microsoft.VSTS.Scheduling.CompletedWork" in fields_data,
    )


def fetch_work_items_from_azdo(
    *, config: Config, work_item_ids: Sequence[int]
) -> dict[int, WorkItemState]:
    if not work_item_ids:
        return {}
    fields = ",".join(work_item_fields())
    ids_param = ",".join(str(item_id) for item_id in work_item_ids)
    response = azdo_request(
        config=config,
        method="GET",
        path=(
            f"_apis/wit/workitems?ids={parse.quote(ids_param)}"
            f"&fields={parse.quote(fields)}&api-version=7.0"
        ),
    )
    result: dict[int, WorkItemState] = {}
    for item in response.get("value", []):
        state = parse_work_item(item)
        result[state.item.work_item_id] = state
    return result


def sync_work_items_from_wiql(
    *, storage: SQLiteStorage | MarkdownStorage, config: Config
) -> int:
    if not config.wiql_query:
        raise ValueError("wiql_query is not configured. Set it in config.json.")
    wiql_encoded = parse.quote(config.wiql_query)
    wiql_response = azdo_request(
        config=config,
        method="GET",
        path=f"_apis/wit/wiql/{wiql_encoded}",
    )
    work_items = wiql_response.get("workItems", [])
    ids = [item["id"] for item in work_items]
    if not ids:
        return 0
    work_items_response = fetch_work_items_from_azdo(
        config=config,
        work_item_ids=ids,
    )
    now = datetime.utcnow().isoformat()
    records = []
    for item in work_items_response.values():
        records.append(
            (
                item.item.work_item_id,
                item.item.title,
                item.item.state,
                item.item.original_estimate,
                item.item.remaining_work,
                item.item.completed_work,
                now,
            )
        )
    storage.upsert_work_items(
        [
            WorkItem(
                work_item_id=record[0],
                title=record[1],
                state=record[2],
                original_estimate=record[3],
                remaining_work=record[4],
                completed_work=record[5],
                updated_at=record[6],
            )
            for record in records
        ]
    )
    return len(records)


def work_item_sync_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    try:
        count = sync_work_items_from_wiql(storage=storage, config=config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Synced {count} work items from WIQL.")
    return 0


def normalize_remaining_strategy(strategy: str) -> str:
    if strategy == "prompt":
        return "interactive"
    return strategy


def compute_remaining_after(
    *,
    strategy: str,
    remaining_before: float | None,
    original_estimate: float | None,
    completed_after: float,
    hours_logged: float,
    allow_interactive: bool,
    work_item_id: int,
    work_item_title: str | None,
) -> float | None:
    if remaining_before is None and original_estimate is None:
        return None
    remaining_before = remaining_before or 0.0
    original_estimate = original_estimate or 0.0
    if strategy == "none":
        return remaining_before
    if strategy == "decrement":
        return max(remaining_before - hours_logged, 0.0)
    if strategy == "recalc_from_original":
        return max(original_estimate - completed_after, 0.0)
    if strategy == "interactive":
        if not allow_interactive:
            return remaining_before
        default_value = max(remaining_before - hours_logged, 0.0)
        title_label = f" - {work_item_title}" if work_item_title else ""
        completed_before = max(completed_after - hours_logged, 0.0)
        raw = input(
            f"Remaining work for WI #{work_item_id}{title_label} "
            f"(current {remaining_before:.2f}, "
            f"completed {completed_before:.2f} -> {completed_after:.2f}, "
            f"default {default_value:.2f}): "
        ).strip()
        if not raw:
            return default_value
        try:
            return max(float(raw), 0.0)
        except ValueError:
            print("Invalid number, keeping previous remaining work.")
            return remaining_before
    return remaining_before


def plan_deltas(
    entries: Sequence[Entry],
    *,
    work_items: dict[int, WorkItemState],
    remaining_work_strategy: str,
    allow_interactive_remaining: bool,
) -> list[WorkItemDelta]:
    grouped: dict[int, list[Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.work_item_id, []).append(entry)

    deltas: list[WorkItemDelta] = []
    for work_item_id, group in grouped.items():
        total_hours = sum(item.hours for item in group)
        state = work_items.get(work_item_id)
        work_item = state.item if state else None
        completed_before = work_item.completed_work if work_item else 0.0
        remaining_before = work_item.remaining_work if work_item else None
        original_estimate = work_item.original_estimate if work_item else None
        completed_after = (completed_before or 0.0) + total_hours
        remaining_after = compute_remaining_after(
            strategy=remaining_work_strategy,
            remaining_before=remaining_before,
            original_estimate=original_estimate,
            completed_after=completed_after,
            hours_logged=total_hours,
            allow_interactive=allow_interactive_remaining,
            work_item_id=work_item_id,
            work_item_title=work_item.title if work_item else None,
        )
        deltas.append(
            WorkItemDelta(
                work_item_id=work_item_id,
                entries=group,
                total_hours=total_hours,
                completed_before=completed_before or 0.0,
                completed_after=completed_after,
                remaining_before=remaining_before,
                remaining_after=remaining_after,
                remaining_strategy=remaining_work_strategy,
            )
        )
    return deltas


def is_closed_state(state: str | None) -> bool:
    if not state:
        return False
    closed_states = {"closed", "done", "removed", "resolved"}
    return state.strip().lower() in closed_states


def build_patch_operations(
    *, delta: WorkItemDelta, state: WorkItemState
) -> list[dict[str, object]]:
    operations: list[dict[str, object]] = []
    completed_path = "/fields/Microsoft.VSTS.Scheduling.CompletedWork"
    remaining_path = "/fields/Microsoft.VSTS.Scheduling.RemainingWork"
    operations.append(
        {
            "op": "replace" if state.has_completed_work else "add",
            "path": completed_path,
            "value": delta.completed_after,
        }
    )
    if (
        delta.remaining_after is not None
        and delta.remaining_before != delta.remaining_after
        and delta.remaining_strategy != "none"
    ):
        operations.append(
            {
                "op": "replace" if state.has_remaining_work else "add",
                "path": remaining_path,
                "value": delta.remaining_after,
            }
        )
    return operations


def sync_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    sync_work_items = args.sync_work_items
    if sync_work_items is None:
        sync_work_items = args.apply
    if sync_work_items:
        try:
            count = sync_work_items_from_wiql(storage=storage, config=config)
            print(f"Synced {count} work items from WIQL.")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    entries = storage.get_unsynced_entries()
    if not entries:
        print("No unsynced entries.")
        return 0
    work_item_ids = sorted({entry.work_item_id for entry in entries})
    try:
        work_items = fetch_work_items_from_azdo(
            config=config,
            work_item_ids=work_item_ids,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    storage.upsert_work_items([state.item for state in work_items.values()])
    remaining_strategy = normalize_remaining_strategy(
        args.remaining_work_strategy or config.remaining_work_strategy
    )
    allow_interactive_remaining = not args.apply and sys.stdin.isatty()
    deltas = plan_deltas(
        entries,
        work_items=work_items,
        remaining_work_strategy=remaining_strategy,
        allow_interactive_remaining=allow_interactive_remaining,
    )

    for delta in deltas:
        remaining_line = "Remaining Work: (no data)"
        if delta.remaining_before is not None or delta.remaining_after is not None:
            remaining_line = (
                "Remaining Work: "
                f"{delta.remaining_before or 0.0:.2f} -> "
                f"{delta.remaining_after or 0.0:.2f} "
                f"({delta.remaining_strategy})"
            )
        print(
            textwrap.dedent(
                f"""
                Work Item #{delta.work_item_id}
                  Entries: {len(delta.entries)}
                  Completed Work: {delta.completed_before:.2f} -> {delta.completed_after:.2f} (+{delta.total_hours:.2f})
                  {remaining_line}
                """
            ).strip()
        )
        print("-")

    apply_now = False
    if (
        not args.apply
        and remaining_strategy == "interactive"
        and allow_interactive_remaining
    ):
        response = input("Apply these updates now? [y/N]: ").strip().lower()
        apply_now = response in {"y", "yes"}

    if args.apply or apply_now:
        now = datetime.utcnow().isoformat()
        errors = 0
        for delta in deltas:
            state = work_items.get(delta.work_item_id)
            if not state:
                print(
                    f"Work item {delta.work_item_id} not found in Azure DevOps.",
                    file=sys.stderr,
                )
                errors += 1
                continue
            if is_closed_state(state.item.state) and not config.allow_sync_closed_items:
                print(
                    f"Skipping closed work item {delta.work_item_id}. "
                    "Enable allow_sync_closed_items in config to override.",
                    file=sys.stderr,
                )
                errors += 1
                continue
            patch_operations = build_patch_operations(delta=delta, state=state)
            if not patch_operations:
                print(
                    f"No changes to apply for work item {delta.work_item_id}.",
                    file=sys.stderr,
                )
                continue
            try:
                patch_response = azdo_request(
                    config=config,
                    method="PATCH",
                    path=f"_apis/wit/workitems/{delta.work_item_id}?api-version=7.0",
                    payload=patch_operations,
                    content_type="application/json-patch+json",
                )
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                errors += 1
                continue
            updated_state = parse_work_item(patch_response)
            storage.upsert_work_items([updated_state.item])
            for entry in delta.entries:
                receipt_id = str(uuid.uuid4())
                patch_payload = {
                    "work_item_id": delta.work_item_id,
                    "completed_before": delta.completed_before,
                    "completed_after": delta.completed_after,
                    "remaining_before": delta.remaining_before,
                    "remaining_after": delta.remaining_after,
                    "strategy": delta.remaining_strategy,
                    "azdo_revision": patch_response.get("rev"),
                    "patch_operations": patch_operations,
                }
                receipt = Receipt(
                    receipt_id=receipt_id,
                    entry_id=entry.entry_id,
                    work_item_id=entry.work_item_id,
                    delta_completed_work=entry.hours,
                    synced_at=now,
                    patch_document=json.dumps(patch_payload),
                )
                storage.record_receipt(receipt)
        if errors:
            print(
                f"Sync completed with {errors} error(s).",
                file=sys.stderr,
            )
            return 2
        print("Synced entries to Azure DevOps.")
    else:
        print("Dry run only. Use --apply to mark entries synced locally.")
    return 0


def week_bounds(day: date) -> tuple[date, date]:
    start = day - timedelta(days=day.weekday())
    end = start + timedelta(days=6)
    return start, end


def export_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    storage = get_storage(config)
    target_day = date.fromisoformat(args.week)
    start, end = week_bounds(target_day)
    entries = storage.list_entries_range(start=start, end=end)

    output = sys.stdout
    if args.output:
        output = Path(args.output).expanduser().open("w", newline="", encoding="utf-8")

    try:
        if args.format == "json":
            payload = [entry.__dict__ for entry in entries]
            json.dump(payload, output, indent=2)
            output.write("\n")
        else:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "entry_id",
                    "date",
                    "work_item_id",
                    "hours",
                    "note",
                    "category",
                    "synced",
                ]
            )
            for entry in entries:
                writer.writerow(
                    [
                        entry.entry_id,
                        entry.entry_date,
                        entry.work_item_id,
                        f"{entry.hours:.2f}",
                        entry.note or "",
                        entry.category or "",
                        entry.synced,
                    ]
                )
    finally:
        if output is not sys.stdout:
            output.close()

    print(
        f"Exported {len(entries)} entries for week {start.isoformat()} to {end.isoformat()}."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="azdo-timesheet",
        description="Low-entry Azure DevOps timesheet (local-first).",
        epilog=textwrap.dedent(
            """\
            Wiki mode:
              Set storage_backend: markdown in config.json (or use init --storage-backend).
              Files live under ~/.azdo_timesheet/timesheet by default:
                entries/YYYY/MM/DD.md, receipts/YYYY/MM.md, and a README.md index.
              Publish by committing the folder to a repo, then in Azure DevOps Wiki
              choose "Publish code as wiki" (or link the repo to a project wiki).
              To avoid merge conflicts, treat daily files as append-only and avoid
              in-place edits once pushed.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config file (default: ~/.azdo_timesheet/config.json)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create config + storage")
    init_parser.add_argument("--org-url", help="Azure DevOps org URL")
    init_parser.add_argument("--project", help="Default project name")
    init_parser.add_argument(
        "--storage-backend",
        choices=["sqlite", "markdown"],
        default="sqlite",
        help=(
            "Storage backend (sqlite or markdown). Markdown mode is optimized for "
            "Azure DevOps Wiki publishing and uses fenced jsonl/yaml blocks."
        ),
    )
    init_parser.add_argument(
        "--storage",
        help=(
            "Storage path (SQLite file path or Markdown root directory). "
            "Defaults depend on the selected backend."
        ),
    )
    init_parser.add_argument(
        "--pat-env-var",
        default="AZDO_PAT",
        help="Environment variable name containing a PAT",
    )
    init_parser.add_argument(
        "--remaining-work-strategy",
        default="none",
        choices=["none", "decrement", "recalc_from_original", "interactive", "prompt"],
        help=(
            "Remaining Work strategy (default: none). "
            "Use interactive for dry-run prompts."
        ),
    )
    init_parser.add_argument(
        "--wiql",
        dest="wiql_query",
        help="WIQL query for syncing work items",
    )
    init_parser.add_argument(
        "--max-hours-per-entry",
        default=8,
        type=float,
        help="Warning threshold for large entries",
    )
    init_parser.set_defaults(func=init_command)

    add_parser = subparsers.add_parser("add", help="Add a time entry")
    add_parser.add_argument("--wi", dest="work_item_id")
    add_parser.add_argument("--t", dest="hours", required=True, type=float)
    add_parser.add_argument("--note")
    add_parser.add_argument("--category")
    add_parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    add_parser.set_defaults(func=add_command)

    list_parser = subparsers.add_parser("list", help="List time entries")
    list_parser.add_argument("--wi", dest="work_item_id", type=int)
    list_parser.add_argument("--date", help="YYYY-MM-DD")
    list_parser.set_defaults(func=list_command)

    edit_parser = subparsers.add_parser("edit", help="Edit an unsynced entry")
    edit_parser.add_argument(
        "--id",
        dest="entry_id",
        help="Entry id (omit to pick from a list)",
    )
    edit_parser.add_argument("--wi", dest="work_item_id")
    edit_parser.add_argument("--t", dest="hours", type=float)
    edit_parser.add_argument("--note")
    edit_parser.add_argument("--category")
    edit_parser.add_argument("--date", help="YYYY-MM-DD")
    edit_parser.set_defaults(func=edit_command)

    remove_parser = subparsers.add_parser("remove", help="Remove unsynced entries")
    remove_parser.add_argument(
        "--id",
        dest="entry_id",
        action="append",
        help="Entry id (omit to pick from a list)",
    )
    remove_parser.set_defaults(func=remove_command)

    work_item_parser = subparsers.add_parser("wi", help="Manage local work items")
    wi_subparsers = work_item_parser.add_subparsers(dest="wi_command", required=True)

    wi_sync = wi_subparsers.add_parser("sync", help="Sync work items from WIQL")
    wi_sync.set_defaults(func=work_item_sync_command)

    wi_list = wi_subparsers.add_parser("list", help="List work items")
    wi_list.set_defaults(func=work_item_list_command)

    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync work items and entries (dry-run by default)",
    )
    sync_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates to Azure DevOps and mark entries as synced",
    )
    sync_parser.add_argument(
        "--remaining-work-strategy",
        choices=["none", "decrement", "recalc_from_original", "interactive", "prompt"],
        help="Override remaining work strategy for this sync",
    )
    sync_parser.add_argument(
        "--sync-work-items",
        action="store_true",
        default=None,
        help="Sync work items from WIQL before syncing entries",
    )
    sync_parser.add_argument(
        "--skip-wi-sync",
        action="store_false",
        dest="sync_work_items",
        help="Skip WIQL work item sync before syncing entries",
    )
    sync_parser.set_defaults(func=sync_command)

    export_parser = subparsers.add_parser("export", help="Export weekly entries")
    export_parser.add_argument(
        "--week",
        default=date.today().isoformat(),
        help="Any date in the target week (YYYY-MM-DD)",
    )
    export_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
    )
    export_parser.add_argument("--output", help="Write to file instead of stdout")
    export_parser.set_defaults(func=export_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
