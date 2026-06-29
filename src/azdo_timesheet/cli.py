import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import textwrap
import uuid
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib import error, parse, request
from typing import Sequence

from .models import (
    AppConfig,
    Config,
    Entry,
    Receipt,
    WorkItem,
    WorkItemDelta,
    WorkItemState,
)
from .hours import format_report_hours, round_report_hours, sum_report_hours
from .storage import MarkdownStorage, SQLiteStorage

DEFAULT_CONFIG_DIR = Path.home() / ".azdo_timesheet"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_PROFILE_NAME = "default"
DEFAULT_PROFILES_ROOT = DEFAULT_CONFIG_DIR / "profiles"
LEGACY_DB_PATH = DEFAULT_CONFIG_DIR / "timesheet.sqlite"
LEGACY_MARKDOWN_ROOT = DEFAULT_CONFIG_DIR / "timesheet"


def _sanitize_profile_name(name: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in name.strip()
    )
    return cleaned or "profile"


def _default_storage_path(profile_name: str, storage_backend: str) -> Path:
    safe_name = _sanitize_profile_name(profile_name)
    root = DEFAULT_PROFILES_ROOT / safe_name
    return root / ("timesheet" if storage_backend == "markdown" else "timesheet.sqlite")


def _profile_from_payload(
    name: str, payload: dict, *, use_legacy_defaults: bool = False
) -> Config:
    storage_backend = payload.get("storage_backend", "sqlite")
    if payload.get("storage_path"):
        storage_path = Path(payload["storage_path"]).expanduser()
    else:
        if use_legacy_defaults:
            storage_path = (
                LEGACY_MARKDOWN_ROOT
                if storage_backend == "markdown"
                else LEGACY_DB_PATH
            )
        else:
            storage_path = _default_storage_path(name, storage_backend)
    return Config(
        profile_name=name,
        org_url=payload.get("org_url", ""),
        project=payload.get("project"),
        auth_mode=payload.get("auth_mode", "pat"),
        pat_env_var=payload.get("pat_env_var", "AZDO_PAT"),
        remaining_work_strategy=payload.get("remaining_work_strategy", "none"),
        allow_sync_closed_items=bool(payload.get("allow_sync_closed_items", False)),
        max_hours_per_entry=float(payload.get("max_hours_per_entry", 8)),
        storage_backend=storage_backend,
        storage_path=storage_path,
        wiql_query=payload.get("wiql_query"),
        moneybird_administration_id=payload.get("moneybird_administration_id"),
        moneybird_user_id=payload.get("moneybird_user_id"),
        moneybird_contact_id=payload.get("moneybird_contact_id")
        or payload.get("moneybird_client_id"),
        moneybird_token_env_var=payload.get("moneybird_token_env_var", "MONEYBIRD_TOKEN"),
        moneybird_start_time=payload.get("moneybird_start_time", "08:00"),
        moneybird_lunch_break_start=payload.get("moneybird_lunch_break_start", "12:00"),
        moneybird_lunch_break_end=payload.get("moneybird_lunch_break_end", "13:00"),
        moneybird_dinner_break_start=payload.get("moneybird_dinner_break_start", "18:00"),
        moneybird_dinner_break_end=payload.get("moneybird_dinner_break_end", "19:30"),
        moneybird_timezone=payload.get("moneybird_timezone", "Z"),
        moneybird_billable=payload.get("moneybird_billable"),
    )


def load_app_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Run 'azdo-timesheet init' first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if "profiles" in data:
        profiles_payload = data.get("profiles", {})
        if not profiles_payload:
            raise ValueError("Config must contain at least one profile.")
        profiles = {
            name: _profile_from_payload(name, payload)
            for name, payload in profiles_payload.items()
        }
        default_profile = data.get("default_profile") or next(iter(profiles))
    else:
        profile = _profile_from_payload(
            DEFAULT_PROFILE_NAME, data, use_legacy_defaults=True
        )
        profiles = {DEFAULT_PROFILE_NAME: profile}
        default_profile = DEFAULT_PROFILE_NAME
    if default_profile not in profiles:
        raise ValueError(
            f"Default profile '{default_profile}' not found in config profiles."
        )
    storage_paths: dict[Path, str] = {}
    for name, profile in profiles.items():
        resolved = profile.storage_path.expanduser().resolve()
        if resolved in storage_paths:
            other = storage_paths[resolved]
            raise ValueError(
                f"Profiles '{name}' and '{other}' share the same storage path: "
                f"{resolved}. Each profile must have its own storage."
            )
        storage_paths[resolved] = name
    return AppConfig(default_profile=default_profile, profiles=profiles)


def save_app_config(path: Path, app_config: AppConfig) -> None:
    payload = {
        "default_profile": app_config.default_profile,
        "profiles": {
            name: {
                "org_url": profile.org_url,
                "project": profile.project,
                "auth_mode": profile.auth_mode,
                "pat_env_var": profile.pat_env_var,
                "remaining_work_strategy": profile.remaining_work_strategy,
                "allow_sync_closed_items": profile.allow_sync_closed_items,
                "max_hours_per_entry": profile.max_hours_per_entry,
                "storage_backend": profile.storage_backend,
                "storage_path": str(profile.storage_path),
                "wiql_query": profile.wiql_query,
                "moneybird_administration_id": profile.moneybird_administration_id,
                "moneybird_user_id": profile.moneybird_user_id,
                "moneybird_contact_id": profile.moneybird_contact_id,
                "moneybird_token_env_var": profile.moneybird_token_env_var,
                "moneybird_start_time": profile.moneybird_start_time,
                "moneybird_lunch_break_start": profile.moneybird_lunch_break_start,
                "moneybird_lunch_break_end": profile.moneybird_lunch_break_end,
                "moneybird_dinner_break_start": profile.moneybird_dinner_break_start,
                "moneybird_dinner_break_end": profile.moneybird_dinner_break_end,
                "moneybird_timezone": profile.moneybird_timezone,
                "moneybird_billable": profile.moneybird_billable,
            }
            for name, profile in app_config.profiles.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_profile_config(path: Path, profile_name: str | None = None) -> Config:
    app_config = load_app_config(path)
    active_name = profile_name or app_config.default_profile
    if active_name not in app_config.profiles:
        raise ValueError(f"Profile '{active_name}' not found in config.")
    return app_config.profiles[active_name]


def resolve_active_profile(
    path: Path, profile_name: str | None = None
) -> tuple[AppConfig, str, Config]:
    app_config = load_app_config(path)
    active_name = profile_name or app_config.default_profile
    if active_name not in app_config.profiles:
        raise ValueError(f"Profile '{active_name}' not found in config.")
    return app_config, active_name, app_config.profiles[active_name]


def get_storage(config: Config) -> SQLiteStorage | MarkdownStorage:
    if config.storage_backend == "markdown":
        return MarkdownStorage(
            config.storage_path,
            org_url=config.org_url,
            project=config.project,
            moneybird_start_time=config.moneybird_start_time,
            moneybird_lunch_break_start=config.moneybird_lunch_break_start,
            moneybird_lunch_break_end=config.moneybird_lunch_break_end,
            moneybird_dinner_break_start=config.moneybird_dinner_break_start,
            moneybird_dinner_break_end=config.moneybird_dinner_break_end,
            moneybird_timezone=config.moneybird_timezone,
        )
    return SQLiteStorage(config.storage_path)


def init_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    profile_name = args.profile or DEFAULT_PROFILE_NAME
    if args.storage:
        storage_path = Path(args.storage).expanduser().resolve()
    else:
        storage_path = _default_storage_path(profile_name, args.storage_backend)
    config = Config(
        profile_name=profile_name,
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
        moneybird_administration_id=args.moneybird_administration_id,
        moneybird_user_id=args.moneybird_user_id,
        moneybird_contact_id=args.moneybird_contact_id,
        moneybird_token_env_var=args.moneybird_token_env_var,
        moneybird_start_time=args.moneybird_start_time,
        moneybird_lunch_break_start=args.moneybird_lunch_break_start,
        moneybird_lunch_break_end=args.moneybird_lunch_break_end,
        moneybird_dinner_break_start=args.moneybird_dinner_break_start,
        moneybird_dinner_break_end=args.moneybird_dinner_break_end,
        moneybird_timezone=args.moneybird_timezone,
        moneybird_billable=args.moneybird_billable,
    )
    app_config = AppConfig(
        default_profile=profile_name,
        profiles={profile_name: config},
    )
    save_app_config(config_path, app_config)
    storage = get_storage(config)
    storage.init()
    print(f"Initialized config at {config_path}")
    print(f"Storage: {storage_path}")
    return 0


def profile_list_command(args: argparse.Namespace) -> int:
    try:
        app_config = load_app_config(Path(args.config).expanduser())
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    default = app_config.default_profile
    for name in sorted(app_config.profiles):
        profile = app_config.profiles[name]
        marker = "*" if name == default else " "
        project_label = profile.project or "(no project)"
        print(f"{marker} {name}: {profile.org_url} / {project_label}")
    return 0


def profile_add_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    try:
        app_config = load_app_config(config_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    profile_name = args.name
    if profile_name in app_config.profiles:
        print(f"Profile '{profile_name}' already exists.", file=sys.stderr)
        return 2
    if args.storage:
        storage_path = Path(args.storage).expanduser().resolve()
    else:
        storage_path = _default_storage_path(profile_name, args.storage_backend)
    for existing in app_config.profiles.values():
        if existing.storage_path.expanduser().resolve() == storage_path:
            print(
                "Storage path already in use by another profile. "
                "Each profile must have its own storage.",
                file=sys.stderr,
            )
            return 2
    profile = Config(
        profile_name=profile_name,
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
        moneybird_administration_id=args.moneybird_administration_id,
        moneybird_user_id=args.moneybird_user_id,
        moneybird_contact_id=args.moneybird_contact_id,
        moneybird_token_env_var=args.moneybird_token_env_var,
        moneybird_start_time=args.moneybird_start_time,
        moneybird_lunch_break_start=args.moneybird_lunch_break_start,
        moneybird_lunch_break_end=args.moneybird_lunch_break_end,
        moneybird_dinner_break_start=args.moneybird_dinner_break_start,
        moneybird_dinner_break_end=args.moneybird_dinner_break_end,
        moneybird_timezone=args.moneybird_timezone,
        moneybird_billable=args.moneybird_billable,
    )
    profiles = dict(app_config.profiles)
    profiles[profile_name] = profile
    default_profile = (
        profile_name if args.set_default else app_config.default_profile
    )
    save_app_config(
        config_path,
        AppConfig(default_profile=default_profile, profiles=profiles),
    )
    storage = get_storage(profile)
    storage.init()
    print(f"Added profile '{profile_name}'.")
    print(f"Storage: {storage_path}")
    return 0


def profile_use_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    try:
        app_config = load_app_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.name not in app_config.profiles:
        print(f"Profile '{args.name}' not found.", file=sys.stderr)
        return 2
    updated = AppConfig(default_profile=args.name, profiles=app_config.profiles)
    save_app_config(config_path, updated)
    print(f"Default profile set to '{args.name}'.")
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
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
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


def truncate_note(note: str | None, *, max_length: int = 77) -> str:
    value = (note or "").replace("\n", " ")
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}..."


def summarize_hours_by_day(entries: Sequence[Entry]) -> list[tuple[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    for entry in entries:
        totals[entry.entry_date] += entry.hours
    return sorted(totals.items())


def format_entries(
    entries: Sequence[Entry],
    work_items: dict[int, WorkItem] | None = None,
) -> str:
    if not entries:
        return "No entries found."
    headers = ["idx", "entry_id", "date", "wi", "parent_wi", "mb_project", "hours", "synced", "note"]
    work_items = work_items or {}
    rows: list[list[str]] = []
    for idx, entry in enumerate(entries, start=1):
        note = truncate_note(entry.note)
        short_id = entry.entry_id.split("-")[0]
        parent_work_item_id = work_items.get(entry.work_item_id)
        parent_value = (
            str(parent_work_item_id.parent_work_item_id)
            if parent_work_item_id and parent_work_item_id.parent_work_item_id is not None
            else ""
        )
        rows.append(
            [
                str(idx),
                short_id,
                str(entry.entry_date),
                str(entry.work_item_id),
                parent_value,
                moneybird_project_id_for_entry(entry, work_items) or "",
                format_report_hours(entry.hours),
                str(entry.synced),
                note,
            ]
        )
    widths = [len(header) for header in headers]
    widths[2] = max(widths[2], len("selection total"))
    for row in rows:
        for col_idx, value in enumerate(row):
            widths[col_idx] = max(widths[col_idx], len(value))
    align_right = {0, 3, 4, 6, 7}

    def format_row(values: Sequence[str]) -> str:
        padded = []
        for col_idx, value in enumerate(values):
            if col_idx in align_right:
                padded.append(value.rjust(widths[col_idx]))
            else:
                padded.append(value.ljust(widths[col_idx]))
        return " | ".join(padded)

    header_line = format_row(headers)
    lines = [header_line, "-" * len(header_line)]
    current_day = None
    current_day_total = 0.0
    overall_total = 0.0
    for row, entry in zip(rows, entries):
        row_day = row[2]
        row_hours = round_report_hours(entry.hours)
        if current_day is None:
            current_day = row_day
        elif row_day != current_day:
            lines.append(
                format_row(
                    ["", "", current_day, "", "", "", f"{current_day_total:.2f}", "", "daily total"]
                )
            )
            current_day = row_day
            current_day_total = 0.0
        lines.append(format_row(row))
        current_day_total += row_hours
        overall_total += row_hours
    if current_day is not None:
        lines.append(
            format_row(
                ["", "", current_day, "", "", "", f"{current_day_total:.2f}", "", "daily total"]
            )
        )
    lines.append("-" * len(header_line))
    lines.append(
        format_row(
            [
                "",
                "",
                "selection total",
                "",
                "",
                "",
                f"{overall_total:.2f}",
                "",
                "",
            ]
        )
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
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
    storage = get_storage(config)
    work_items = storage.list_work_items()
    work_item_map = {item.work_item_id: item for item in work_items}
    try:
        if args.start or args.end:
            if args.date:
                raise ValueError("Use either --date or --start/--end, not both.")
            start_day, end_day = resolve_period(week=None, start=args.start, end=args.end)
            entries = storage.list_entries_range(start=start_day, end=end_day)
            if args.work_item_id is not None:
                entries = [entry for entry in entries if entry.work_item_id == args.work_item_id]
            entries.sort(key=lambda item: (item.entry_date, item.created_at), reverse=True)
        else:
            entries = storage.list_entries(
                work_item_id=args.work_item_id,
                entry_date=args.date,
            )
        if args.parent_work_item_id is not None:
            entries = [
                entry
                for entry in entries
                if (
                    work_item_map.get(entry.work_item_id)
                    and work_item_map[entry.work_item_id].parent_work_item_id
                    == args.parent_work_item_id
                )
            ]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.summary_by_parent:
        print(format_parent_summary(entries, work_items))
    else:
        print(format_entries(entries, work_item_map))
    return 0


def edit_command(args: argparse.Namespace) -> int:
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
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
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
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
    headers = [
        "wi",
        "parent",
        "title",
        "tags",
        "state",
        "original",
        "remaining",
        "completed",
    ]
    rows: list[list[str]] = []
    for item in work_items:
        rows.append(
            [
                str(item.work_item_id),
                str(item.parent_work_item_id) if item.parent_work_item_id is not None else "",
                item.title or "",
                item.tags or "",
                item.state or "",
                str(item.original_estimate)
                if item.original_estimate is not None
                else "",
                str(item.remaining_work) if item.remaining_work is not None else "",
                str(item.completed_work) if item.completed_work is not None else "",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    align_right = {0, 1, 5, 6, 7}

    def format_row(values: Sequence[str]) -> str:
        padded = []
        for idx, value in enumerate(values):
            if idx in align_right:
                padded.append(value.rjust(widths[idx]))
            else:
                padded.append(value.ljust(widths[idx]))
        return " | ".join(padded)

    header_line = format_row(headers)
    lines = [header_line, "-" * len(header_line)]
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)

def work_item_list_command(args: argparse.Namespace) -> int:
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
    storage = get_storage(config)
    work_items = storage.list_work_items()
    print(format_work_items(work_items))
    return 0


def config_show_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    try:
        _, active_name, config = resolve_active_profile(config_path, args.profile)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = {
        "profile_name": config.profile_name,
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
        "moneybird_administration_id": config.moneybird_administration_id,
        "moneybird_user_id": config.moneybird_user_id,
        "moneybird_contact_id": config.moneybird_contact_id,
        "moneybird_token_env_var": config.moneybird_token_env_var,
        "moneybird_start_time": config.moneybird_start_time,
        "moneybird_lunch_break_start": config.moneybird_lunch_break_start,
        "moneybird_lunch_break_end": config.moneybird_lunch_break_end,
        "moneybird_dinner_break_start": config.moneybird_dinner_break_start,
        "moneybird_dinner_break_end": config.moneybird_dinner_break_end,
        "moneybird_timezone": config.moneybird_timezone,
        "moneybird_billable": config.moneybird_billable,
    }
    print(f"Config path: {config_path}")
    print(f"Active profile: {active_name}")
    print(f"Storage path: {config.storage_path}")
    print(json.dumps(payload, indent=2))
    return 0


def config_edit_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(
            f"Config not found at {config_path}. Run 'azdo-timesheet init' first.",
            file=sys.stderr,
        )
        return 2
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(config_path))
        else:
            subprocess.Popen(["xdg-open", str(config_path)])
    except OSError as exc:
        print(f"Unable to open config: {exc}", file=sys.stderr)
        return 2
    print(f"Opened config: {config_path}")
    return 0


def repair_markdown_tables_command(args: argparse.Namespace) -> int:
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
    if config.storage_backend != "markdown":
        print(
            "repair markdown-tables only works with storage_backend=markdown.",
            file=sys.stderr,
        )
        return 2
    storage = get_storage(config)
    rebuilt_days = storage.rebuild_tables_from_canonical_data()
    print(
        "Rebuilt Markdown entry tables from Canonical Entry Data. "
        "Emergency use only after direct canonical edits."
    )
    print(f"Updated day pages: {rebuilt_days}")
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
        "System.Tags",
        "System.State",
        "System.Parent",
        "Microsoft.VSTS.Scheduling.OriginalEstimate",
        "Microsoft.VSTS.Scheduling.RemainingWork",
        "Microsoft.VSTS.Scheduling.CompletedWork",
    ]


def parse_work_item(work_item: dict) -> WorkItemState:
    fields_data = work_item.get("fields", {})
    return WorkItemState(
        item=WorkItem(
            work_item_id=work_item.get("id"),
            parent_work_item_id=fields_data.get("System.Parent"),
            title=fields_data.get("System.Title"),
            tags=fields_data.get("System.Tags"),
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
                item.item.parent_work_item_id,
                item.item.title,
                item.item.tags,
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
                parent_work_item_id=record[1],
                title=record[2],
                tags=record[3],
                state=record[4],
                original_estimate=record[5],
                remaining_work=record[6],
                completed_work=record[7],
                updated_at=record[8],
            )
            for record in records
        ]
    )
    return len(records)


def work_item_sync_command(args: argparse.Namespace) -> int:
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
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
    remaining_field_missing: bool,
) -> float | None:
    if remaining_field_missing:
        if not allow_interactive:
            return None
        default_value = (
            max((original_estimate or 0.0) - completed_after, 0.0)
            if strategy == "recalc_from_original"
            else 0.0
        )
        title_label = f" - {work_item_title}" if work_item_title else ""
        completed_before = max(completed_after - hours_logged, 0.0)
        raw = input(
            f"Remaining work is not set for WI #{work_item_id}{title_label} "
            f"(completed {completed_before:.2f} -> {completed_after:.2f}, "
            f"suggested {default_value:.2f}; leave blank to keep it unset): "
        ).strip()
        if not raw:
            return None
        try:
            return max(float(raw), 0.0)
        except ValueError:
            print("Invalid number, leaving Remaining Work unset.")
            return None
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
        remaining_field_missing = bool(state and not state.has_remaining_work)
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
            remaining_field_missing=remaining_field_missing,
        )
        deltas.append(
            WorkItemDelta(
                work_item_id=work_item_id,
                entries=group,
                total_hours=total_hours,
                completed_before=completed_before or 0.0,
                completed_after=completed_after,
                remaining_field_missing=remaining_field_missing,
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
        and (
            delta.remaining_field_missing
            or delta.remaining_before != delta.remaining_after
        )
        and (delta.remaining_strategy != "none" or delta.remaining_field_missing)
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
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
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
        if delta.remaining_field_missing:
            if delta.remaining_after is None:
                remaining_line = "Remaining Work: missing -> left unset"
            else:
                remaining_line = (
                    "Remaining Work: missing -> "
                    f"{delta.remaining_after:.2f} (prompted)"
                )
        elif delta.remaining_before is not None or delta.remaining_after is not None:
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
        and (
            remaining_strategy == "interactive"
            or any(delta.remaining_field_missing for delta in deltas)
        )
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


def resolve_period(*, week: str | None, start: str | None, end: str | None) -> tuple[date, date]:
    if start or end:
        if not start or not end:
            raise ValueError("Both --start and --end must be provided together.")
        start_day = date.fromisoformat(start)
        end_day = date.fromisoformat(end)
        if start_day > end_day:
            raise ValueError("--start cannot be after --end.")
        return start_day, end_day
    if not week:
        raise ValueError("--week is required when --start/--end are not used.")
    return week_bounds(date.fromisoformat(week))


def _parent_key(work_item: WorkItem | None) -> tuple[int | None, str]:
    if not work_item or work_item.parent_work_item_id is None:
        return (None, "(no parent)")
    parent = work_item.parent_work_item_id
    return (parent, str(parent))


def summarize_by_parent(
    entries: Sequence[Entry],
    work_items: dict[int, WorkItem],
    *,
    report_hours: bool = False,
) -> list[tuple[int | None, str, float]]:
    totals: dict[tuple[int | None, str], float] = defaultdict(float)
    for entry in entries:
        work_item = work_items.get(entry.work_item_id)
        key = _parent_key(work_item)
        totals[key] += round_report_hours(entry.hours) if report_hours else entry.hours
    ordered = sorted(totals.items(), key=lambda item: (item[0][0] is None, item[0][1]))
    return [(parent_id, label, hours) for (parent_id, label), hours in ordered]


def format_parent_summary(entries: Sequence[Entry], work_items: Sequence[WorkItem]) -> str:
    if not entries:
        return "No entries found."
    work_item_map = {item.work_item_id: item for item in work_items}
    rows = summarize_by_parent(entries, work_item_map, report_hours=True)
    headers = ["parent", "mb_project", "hours"]
    widths = [len(h) for h in headers]
    for parent_id, parent_label, hours in rows:
        widths[0] = max(widths[0], len(parent_label))
        widths[1] = max(widths[1], len(moneybird_project_id_for_parent(parent_id, work_item_map) or ""))
        widths[2] = max(widths[2], len(f"{hours:.2f}"))

    def fmt(values: Sequence[str]) -> str:
        return (
            f"{values[0].ljust(widths[0])} | "
            f"{values[1].ljust(widths[1])} | "
            f"{values[2].rjust(widths[2])}"
        )

    header = fmt(headers)
    lines = [header, "-" * len(header)]
    total = 0.0
    for parent_id, parent_label, hours in rows:
        total += hours
        lines.append(
            fmt(
                [
                    parent_label,
                    moneybird_project_id_for_parent(parent_id, work_item_map) or "",
                    f"{hours:.2f}",
                ]
            )
        )
    lines.append("-" * len(header))
    lines.append(fmt(["total", "", f"{total:.2f}"]))
    return "\n".join(lines)


def _parse_clock(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError:
        parts = value.strip().split(":")
        if len(parts) != 2:
            raise
        hour_text, minute_text = parts
        if not hour_text.isdigit() or not minute_text.isdigit():
            raise
        try:
            return time(hour=int(hour_text), minute=int(minute_text))
        except ValueError:
            raise


def _minutes(value: str) -> int:
    parsed = _parse_clock(value)
    return parsed.hour * 60 + parsed.minute


def normalize_clock_argument(value: str, option_name: str) -> str:
    try:
        parsed = _parse_clock(value)
    except ValueError:
        raise ValueError(f"Invalid {option_name} value '{value}'. Use HH:MM.") from None
    if parsed.second or parsed.microsecond:
        raise ValueError(f"Invalid {option_name} value '{value}'. Use HH:MM.")
    return f"{parsed.hour:02d}:{parsed.minute:02d}"


def _format_moneybird_timestamp(entry_date: str, minutes_after_midnight: int, timezone: str) -> str:
    hour, minute = divmod(minutes_after_midnight, 60)
    suffix = timezone if timezone else "Z"
    return f"{entry_date}T{hour:02d}:{minute:02d}:00{suffix}"


def add_work_minutes(start_minute: int, duration_minutes: int, breaks: Sequence[tuple[int, int]]) -> int:
    ended_minute, _ = add_work_minutes_with_breaks(start_minute, duration_minutes, breaks)
    return ended_minute


def add_work_minutes_with_breaks(
    start_minute: int, duration_minutes: int, breaks: Sequence[tuple[int, int]]
) -> tuple[int, int]:
    if duration_minutes <= 0:
        return start_minute, 0
    current = start_minute
    remaining = duration_minutes
    break_minutes = 0
    for break_start, break_end in sorted(breaks):
        if current >= break_end:
            continue
        if current < break_start:
            available = break_start - current
            if remaining <= available:
                return current + remaining, break_minutes
            remaining -= available
            break_minutes += break_end - break_start
            current = break_end
        elif break_start <= current < break_end:
            break_minutes += break_end - current
            current = break_end
    return current + remaining, break_minutes


def moneybird_project_id_from_tags(tags: str | None) -> str | None:
    if not tags:
        return None
    for raw_tag in tags.replace(",", ";").split(";"):
        tag = raw_tag.strip()
        if tag.lower().startswith("mb:"):
            project_id = tag[3:].strip()
            return project_id or None
    return None


def moneybird_project_id_for_entry(
    entry: Entry, work_items: dict[int, WorkItem]
) -> str | None:
    child = work_items.get(entry.work_item_id)
    if not child or child.parent_work_item_id is None:
        return None
    parent = work_items.get(child.parent_work_item_id)
    return moneybird_project_id_from_tags(parent.tags if parent else None)


def moneybird_project_id_for_parent(
    parent_id: int | None, work_items: dict[int, WorkItem]
) -> str | None:
    if parent_id is None:
        return None
    parent = work_items.get(parent_id)
    return moneybird_project_id_from_tags(parent.tags if parent else None)


def build_moneybird_description(
    *, entry_date: str, parent_id: int, parent_title: str | None, child_ids: Sequence[int]
) -> str:
    title = f" - {parent_title}" if parent_title else ""
    children = ", ".join(str(item) for item in sorted(set(child_ids)))
    return (
        "Automated export from azdo-timesheet local timesheets. "
        f"Date: {entry_date}. Azure DevOps parent work item: {parent_id}{title}. "
        f"Aggregated child work items: {children}."
    )


def plan_moneybird_time_entries(
    entries: Sequence[Entry],
    work_items: Sequence[WorkItem],
    config: Config,
) -> list[dict[str, object]]:
    work_item_map = {item.work_item_id: item for item in work_items}
    grouped: dict[tuple[str, int], list[Entry]] = defaultdict(list)
    for entry in entries:
        child = work_item_map.get(entry.work_item_id)
        if not child or child.parent_work_item_id is None:
            continue
        grouped[(entry.entry_date, child.parent_work_item_id)].append(entry)

    start_minute = _minutes(config.moneybird_start_time)
    breaks = [
        (_minutes(config.moneybird_lunch_break_start), _minutes(config.moneybird_lunch_break_end)),
        (_minutes(config.moneybird_dinner_break_start), _minutes(config.moneybird_dinner_break_end)),
    ]
    planned: list[dict[str, object]] = []
    current_minute_by_date: dict[str, int] = {}
    for (entry_date, parent_id), group in sorted(grouped.items()):
        started_minute = current_minute_by_date.get(entry_date, start_minute)
        duration_minutes = int(round(sum_report_hours(item.hours for item in group) * 60))
        ended_minute, break_minutes = add_work_minutes_with_breaks(
            started_minute,
            duration_minutes,
            breaks,
        )
        current_minute_by_date[entry_date] = ended_minute
        parent = work_item_map.get(parent_id)
        project_id = moneybird_project_id_from_tags(parent.tags if parent else None)
        if not project_id:
            planned.append(
                {
                    "date": entry_date,
                    "parent_work_item_id": parent_id,
                    "hours": duration_minutes / 60,
                    "error": "missing_moneybird_project_tag",
                }
            )
            continue
        payload: dict[str, object] = {
            "started_at": _format_moneybird_timestamp(entry_date, started_minute, config.moneybird_timezone),
            "ended_at": _format_moneybird_timestamp(entry_date, ended_minute, config.moneybird_timezone),
            "description": build_moneybird_description(
                entry_date=entry_date,
                parent_id=parent_id,
                parent_title=parent.title if parent else None,
                child_ids=[entry.work_item_id for entry in group],
            ),
            "project_id": project_id,
        }
        if break_minutes:
            payload["paused_duration"] = break_minutes * 60
        if config.moneybird_user_id:
            payload["user_id"] = config.moneybird_user_id
        if config.moneybird_contact_id:
            payload["contact_id"] = config.moneybird_contact_id
        if config.moneybird_billable is not None:
            payload["billable"] = config.moneybird_billable
        planned.append(
            {
                "date": entry_date,
                "parent_work_item_id": parent_id,
                "hours": duration_minutes / 60,
                "time_entry": payload,
            }
        )
    return planned


def moneybird_token(config: Config) -> str:
    token = os.environ.get(config.moneybird_token_env_var, "")
    if not token:
        raise ValueError(
            f"Missing Moneybird token. Set {config.moneybird_token_env_var} in the environment."
        )
    return token


def moneybird_request(*, config: Config, payload: dict[str, object]) -> dict:
    if not config.moneybird_administration_id:
        raise ValueError("moneybird_administration_id is required in config.json.")
    url = f"https://moneybird.com/api/v2/{config.moneybird_administration_id}/time_entries.json"
    body = json.dumps({"time_entry": payload}).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {moneybird_token(config)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise ValueError(f"Moneybird request failed: {exc.code} {detail}") from exc


def unsynced_entries_by_day(entries: Sequence[Entry]) -> dict[str, list[Entry]]:
    grouped: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        if not entry.synced:
            grouped[entry.entry_date].append(entry)
    return dict(sorted(grouped.items()))


def moneybird_export_command(args: argparse.Namespace) -> int:
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
    starttime_override = getattr(args, "starttime", None)
    if starttime_override:
        try:
            config = replace(
                config,
                moneybird_start_time=normalize_clock_argument(
                    starttime_override, "--starttime"
                ),
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    storage = get_storage(config)
    try:
        start, end = resolve_period(week=args.week, start=args.start, end=args.end)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    entries = storage.list_entries_range(start=start, end=end)
    planned = plan_moneybird_time_entries(entries, storage.list_work_items(), config)
    if not planned:
        print("No parent work item totals found for Moneybird export.")
        return 0
    errors = [item for item in planned if "error" in item]
    for item in planned:
        if "error" in item:
            print(
                f"{item['date']} parent WI #{item['parent_work_item_id']}: "
                f"{item['hours']:.2f}h skipped ({item['error']})"
            )
            continue
        time_entry = item["time_entry"]
        user_id = time_entry.get("user_id") or "(missing user_id)"
        contact_note = (
            f" contact {time_entry['contact_id']}"
            if time_entry.get("contact_id")
            else ""
        )
        paused_seconds = int(time_entry.get("paused_duration") or 0)
        break_note = (
            f", including {paused_seconds // 60} minutes break"
            if paused_seconds
            else ""
        )
        print(
            f"{item['date']} parent WI #{item['parent_work_item_id']}: "
            f"{item['hours']:.2f}h -> Moneybird project {time_entry['project_id']} "
            f"user {user_id}{contact_note} "
            f"({time_entry['started_at']} - {time_entry['ended_at']}{break_note})"
        )
    if errors:
        print("Fix missing parent work item tags before applying.", file=sys.stderr)
        return 2
    apply_now = args.apply
    can_prompt = sys.stdin.isatty()
    if not apply_now and can_prompt:
        try:
            response = input("Apply these Moneybird registrations now? [y/N]: ").strip().lower()
        except EOFError:
            response = ""
        apply_now = response in {"y", "yes"}
    if not apply_now:
        if can_prompt:
            print("Dry run only. No Moneybird time registrations created.")
        else:
            print(
                "Dry run only. Re-run from an interactive terminal and answer yes, "
                "or use --apply to create Moneybird time registrations."
            )
        if not config.moneybird_user_id:
            print("Set moneybird_user_id in config before applying Moneybird export.")
        return 0
    unsynced_by_day = unsynced_entries_by_day(entries)
    if unsynced_by_day:
        print(
            "Moneybird apply requires every entry in each exported day to be "
            "synced to Azure DevOps first.",
            file=sys.stderr,
        )
        for entry_date, day_entries in unsynced_by_day.items():
            entry_ids = ", ".join(entry.entry_id for entry in day_entries)
            print(
                f"{entry_date}: {len(day_entries)} unsynced entr"
                f"{'y' if len(day_entries) == 1 else 'ies'} ({entry_ids})",
                file=sys.stderr,
            )
        print("Run azdo-timesheet sync --apply before Moneybird apply.", file=sys.stderr)
        return 2
    if not config.moneybird_user_id:
        print(
            "moneybird_user_id is required in config.json before applying Moneybird export.",
            file=sys.stderr,
        )
        return 2
    created = 0
    for item in planned:
        try:
            response = moneybird_request(config=config, payload=item["time_entry"])
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        created += 1
        print(f"Created Moneybird time entry {response.get('id', '(unknown id)')}.")
    print(f"Created {created} Moneybird time registration(s).")
    return 0


def export_command(args: argparse.Namespace) -> int:
    config = load_profile_config(Path(args.config).expanduser(), args.profile)
    storage = get_storage(config)
    try:
        start, end = resolve_period(week=args.week, start=args.start, end=args.end)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    entries = storage.list_entries_range(start=start, end=end)
    work_items = storage.list_work_items()
    work_item_map = {item.work_item_id: item for item in work_items}
    parent_summary = summarize_by_parent(entries, work_item_map)

    export_entries = []
    for entry in entries:
        work_item = work_item_map.get(entry.work_item_id)
        export_entries.append(
            {
                **entry.__dict__,
                "parent_work_item_id": work_item.parent_work_item_id if work_item else None,
            }
        )

    output = sys.stdout
    if args.output:
        output = Path(args.output).expanduser().open("w", newline="", encoding="utf-8")

    try:
        if args.format == "json":
            if args.summary_by_parent:
                payload = {
                    "entries": export_entries,
                    "parent_summary": [
                        {"parent_work_item_id": parent_id, "hours": hours}
                        for parent_id, _, hours in parent_summary
                    ],
                }
            else:
                payload = export_entries
            json.dump(payload, output, indent=2)
            output.write("\n")
        else:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "entry_id",
                    "date",
                    "work_item_id",
                    "parent_work_item_id",
                    "hours",
                    "note",
                    "category",
                    "synced",
                ]
            )
            for item in export_entries:
                writer.writerow(
                    [
                        item["entry_id"],
                        item["entry_date"],
                        item["work_item_id"],
                        item["parent_work_item_id"] if item["parent_work_item_id"] is not None else "",
                        f"{item['hours']:.2f}",
                        item["note"] or "",
                        item["category"] or "",
                        item["synced"],
                    ]
                )
            if args.summary_by_parent:
                writer.writerow([])
                writer.writerow(["parent_work_item_id", "hours"])
                for parent_id, _, hours in parent_summary:
                    writer.writerow([parent_id if parent_id is not None else "", f"{hours:.2f}"])
    finally:
        if output is not sys.stdout:
            output.close()

    print(
        f"Exported {len(entries)} entries for period {start.isoformat()} to {end.isoformat()}."
    )
    return 0


def add_moneybird_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--moneybird-administration-id", help="Moneybird administration id for time registration export")
    parser.add_argument("--moneybird-user-id", help="Moneybird user id for created time registrations")
    parser.add_argument("--moneybird-contact-id", help="Moneybird contact id for created time registrations")
    parser.add_argument("--moneybird-token-env-var", default="MONEYBIRD_TOKEN", help="Environment variable containing the Moneybird Bearer token")
    parser.add_argument("--moneybird-start-time", default="08:00", help="Start time for generated Moneybird registrations (HH:MM)")
    parser.add_argument("--moneybird-lunch-break-start", default="12:00", help="Lunch break start time (HH:MM)")
    parser.add_argument("--moneybird-lunch-break-end", default="13:00", help="Lunch break end time (HH:MM)")
    parser.add_argument("--moneybird-dinner-break-start", default="18:00", help="Dinner break start time (HH:MM)")
    parser.add_argument("--moneybird-dinner-break-end", default="19:30", help="Dinner break end time (HH:MM)")
    parser.add_argument("--moneybird-timezone", default="Z", help="Timezone suffix for Moneybird timestamps, e.g. Z or +02:00")
    parser.add_argument("--moneybird-billable", action=argparse.BooleanOptionalAction, default=None, help="Set billable on created Moneybird time registrations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="azdo-timesheet",
        description="Low-entry Azure DevOps timesheet (local-first).",
        epilog=textwrap.dedent(
            """\
            Wiki mode:
              Set storage_backend: markdown in config.json (or use init --storage-backend).
              Files live under ~/.azdo_timesheet/timesheet by default:
                entries/YYYY/YYYY-MM/YYYY-MM-DD.md, receipts/YYYY/YYYY-MM.md, folder pages
                (entries.md, entries/YYYY.md, entries/YYYY/YYYY-MM.md, receipts.md,
                receipts/YYYY.md),
                and a README.md index. Folder pages include [[_TOSP_]] for navigation
                plus summary tables (per year/month) with work item totals. When
                org_url/project are configured, work item IDs are linked to Azure
                DevOps work items and page titles include full dates (YYYY-MM-DD or
                YYYY-MM for month pages). Daily pages are refreshed as entries change
                to keep links current.
              Publish by committing the folder to a repo, then in Azure DevOps Wiki
              choose "Publish code as wiki" (or link the repo to a project wiki).
              To avoid merge conflicts, treat daily files as append-only and avoid
              in-place edits once pushed.

            Architecture docs:
              See docs/architecture.md for layer and sequence diagrams covering
              CLI, storage, Azure DevOps, and Moneybird flows.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config file (default: ~/.azdo_timesheet/config.json)",
    )
    parser.add_argument(
        "--profile",
        help="Profile name to use (defaults to default_profile in config)",
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
            "Azure DevOps Wiki publishing and uses fenced jsonl/yaml blocks with "
            "year/month summary pages."
        ),
    )
    init_parser.add_argument(
        "--storage",
        help=(
            "Storage path (SQLite file path or Markdown root directory). "
            "Defaults depend on the selected backend. Missing directories are "
            "created automatically. Review or change it later with "
            "'config show' / 'config edit'."
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
    add_moneybird_config_arguments(init_parser)
    init_parser.set_defaults(func=init_command)

    add_parser = subparsers.add_parser("add", help="Add a time entry")
    add_parser.add_argument("--wi", dest="work_item_id")
    add_parser.add_argument(
        "--t",
        dest="hours",
        required=True,
        type=float,
        help="Hours to store exactly; reports round each entry to the nearest 0.25h",
    )
    add_parser.add_argument("--note")
    add_parser.add_argument("--category")
    add_parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    add_parser.set_defaults(func=add_command)

    list_parser = subparsers.add_parser(
        "list",
        help="List entries with truncated notes and quarter-hour report summaries",
    )
    list_parser.add_argument("--wi", dest="work_item_id", type=int)
    list_parser.add_argument("--parent_wi", dest="parent_work_item_id", type=int)
    list_parser.add_argument("--date", help="YYYY-MM-DD")
    list_parser.add_argument("--start", help="Start date YYYY-MM-DD (requires --end)")
    list_parser.add_argument("--end", help="End date YYYY-MM-DD (requires --start)")
    list_parser.add_argument(
        "--summary-by-parent",
        action="store_true",
        help="Show a parent work item summary instead of individual entries",
    )
    list_parser.set_defaults(func=list_command)

    edit_parser = subparsers.add_parser("edit", help="Edit an unsynced entry")
    edit_parser.add_argument(
        "--id",
        dest="entry_id",
        help="Entry id (omit to pick from a list)",
    )
    edit_parser.add_argument("--wi", dest="work_item_id")
    edit_parser.add_argument(
        "--t",
        dest="hours",
        type=float,
        help="Hours to store exactly; reports round each entry to the nearest 0.25h",
    )
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

    wi_list = wi_subparsers.add_parser(
        "list", help="List work items in an aligned table, including cached tags"
    )
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

    export_parser = subparsers.add_parser("export", help="Export entries")
    export_parser.add_argument(
        "--week",
        default=date.today().isoformat(),
        help="Any date in the target week (YYYY-MM-DD). Ignored when --start/--end are provided.",
    )
    export_parser.add_argument("--start", help="Start date YYYY-MM-DD (requires --end)")
    export_parser.add_argument("--end", help="End date YYYY-MM-DD (requires --start)")
    export_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
    )
    export_parser.add_argument("--output", help="Write to file instead of stdout")
    export_parser.add_argument(
        "--summary-by-parent",
        action="store_true",
        help="Include totals grouped by parent work item",
    )
    export_parser.set_defaults(func=export_command)

    moneybird_parser = subparsers.add_parser(
        "moneybird",
        help="Export parent work item day totals to Moneybird time registrations",
    )
    moneybird_subparsers = moneybird_parser.add_subparsers(
        dest="moneybird_command",
        required=True,
    )
    moneybird_export = moneybird_subparsers.add_parser(
        "export",
        help="Create Moneybird time registrations from day totals grouped by parent work item",
        description=(
            "Preview or create Moneybird time registrations from parent work item "
            "day totals. Interactive runs print the preview and then ask whether "
            "to create the registrations. Report totals round each entry to the "
            "nearest 0.25h. Registrations for the same day are scheduled "
            "back-to-back from the configured Moneybird start time, or from "
            "--starttime when provided, sending paused_duration when configured "
            "breaks are crossed. Before applying, configure "
            "moneybird_administration_id, moneybird_user_id, optional "
            "moneybird_contact_id, and the Moneybird token environment variable "
            "on the active profile. Applying also requires "
            "every entry in each exported day to be synced to Azure DevOps first."
        ),
    )
    moneybird_export.add_argument("--week", default=date.today().isoformat(), help="Any date in the target week (YYYY-MM-DD). Ignored when --start/--end are provided.")
    moneybird_export.add_argument("--start", help="Start date YYYY-MM-DD (requires --end)")
    moneybird_export.add_argument("--end", help="End date YYYY-MM-DD (requires --start)")
    moneybird_export.add_argument("--starttime", help="Override moneybird_start_time for this export only (HH:MM)")
    moneybird_export.add_argument("--apply", action="store_true", help="Create time registrations without prompting; requires all exported day entries to be synced to Azure DevOps first")
    moneybird_export.set_defaults(func=moneybird_export_command)

    config_parser = subparsers.add_parser(
        "config",
        help="Show or open the config file",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        required=True,
    )

    config_show = config_subparsers.add_parser(
        "show",
        help="Show the resolved config path and active profile config",
    )
    config_show.set_defaults(func=config_show_command)

    config_edit = config_subparsers.add_parser(
        "edit",
        help="Open the config file in the default editor/application",
    )
    config_edit.set_defaults(func=config_edit_command)

    repair_parser = subparsers.add_parser(
        "repair",
        help="Emergency repair commands for storage-backed content",
    )
    repair_subparsers = repair_parser.add_subparsers(
        dest="repair_command",
        required=True,
    )

    repair_markdown_tables = repair_subparsers.add_parser(
        "markdown-tables",
        help="Rebuild Markdown entry tables from Canonical Entry Data after direct canonical edits",
    )
    repair_markdown_tables.set_defaults(func=repair_markdown_tables_command)

    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage Azure DevOps org/project profiles",
    )
    profile_subparsers = profile_parser.add_subparsers(
        dest="profile_command",
        required=True,
    )

    profile_list = profile_subparsers.add_parser(
        "list",
        help="List configured profiles",
    )
    profile_list.set_defaults(func=profile_list_command)

    profile_add = profile_subparsers.add_parser(
        "add",
        help="Add a new profile",
    )
    profile_add.add_argument("name", help="Profile name")
    profile_add.add_argument("--org-url", help="Azure DevOps org URL")
    profile_add.add_argument("--project", help="Default project name")
    profile_add.add_argument(
        "--storage-backend",
        choices=["sqlite", "markdown"],
        default="sqlite",
        help="Storage backend for this profile",
    )
    profile_add.add_argument(
        "--storage",
        help=(
            "Storage path (SQLite file or Markdown root). Defaults per profile. "
            "Missing directories are created automatically. Review or change it "
            "later with 'config show' / 'config edit'."
        ),
    )
    profile_add.add_argument(
        "--pat-env-var",
        default="AZDO_PAT",
        help="Environment variable name containing a PAT",
    )
    profile_add.add_argument(
        "--remaining-work-strategy",
        default="none",
        choices=["none", "decrement", "recalc_from_original", "interactive", "prompt"],
        help="Remaining Work strategy (default: none).",
    )
    profile_add.add_argument(
        "--wiql",
        dest="wiql_query",
        help="WIQL query for syncing work items",
    )
    profile_add.add_argument(
        "--max-hours-per-entry",
        default=8,
        type=float,
        help="Warning threshold for large entries",
    )
    profile_add.add_argument(
        "--set-default",
        action="store_true",
        help="Make this the default profile",
    )
    add_moneybird_config_arguments(profile_add)
    profile_add.set_defaults(func=profile_add_command)

    profile_use = profile_subparsers.add_parser(
        "use",
        help="Set the default profile",
    )
    profile_use.add_argument("name", help="Profile name")
    profile_use.set_defaults(func=profile_use_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
