from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from azdo_timesheets import __version__
from azdo_timesheets.storage import (
    SyncReceipt,
    TimeEntry,
    connect,
    fetch_entries,
    fetch_unsynced,
    init_db,
    insert_entry,
    insert_receipts,
    mark_entries_synced,
    utc_now,
)

DEFAULT_CONFIG_PATH = Path.home() / ".azdo-timesheet" / "config.json"
DEFAULT_DB_FILENAME = "timesheet.sqlite3"


class ConfigError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Config not found at {path}. Run 'init' to create it first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def resolve_db_path(config: dict[str, Any]) -> Path:
    db_path = config.get("storage_path")
    if not db_path:
        raise ConfigError("storage_path missing from config. Re-run 'init'.")
    return Path(db_path).expanduser()


def format_table(rows: Iterable[Iterable[str]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    widths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]
    lines = []
    for row in rows:
        padded = [str(cell).ljust(width) for cell, width in zip(row, widths)]
        lines.append("  ".join(padded))
    return "\n".join(lines)


def cmd_init(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser()
    storage_dir = (
        Path(args.storage_dir).expanduser()
        if args.storage_dir
        else config_path.parent
    )
    storage_path = storage_dir / DEFAULT_DB_FILENAME

    config = {
        "org_url": args.org_url,
        "project": args.project,
        "pat": args.pat,
        "remaining_work_strategy": "none",
        "storage_path": str(storage_path),
    }

    save_config(config_path, config)
    conn = connect(storage_path)
    init_db(conn)
    conn.close()

    print(f"Initialized config at {config_path}")
    print(f"Storage database at {storage_path}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    db_path = resolve_db_path(config)
    conn = connect(db_path)
    init_db(conn)

    entry_date = args.date or date.today().isoformat()
    entry = TimeEntry(
        entry_id=str(uuid.uuid4()),
        entry_date=entry_date,
        work_item_id=args.work_item,
        hours=args.hours,
        note=args.note,
        category=args.category,
        created_at=utc_now(),
        updated_at=utc_now(),
        synced=False,
    )
    insert_entry(conn, entry)
    conn.close()

    print(
        f"Logged {entry.hours:.2f}h on WI {entry.work_item_id} for {entry.entry_date}"
    )
    if entry.note:
        print(f"Note: {entry.note}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    conn = connect(resolve_db_path(config))
    init_db(conn)

    entries = fetch_entries(
        conn,
        entry_date=args.date,
        work_item_id=args.work_item,
        limit=args.limit,
    )
    conn.close()

    if not entries:
        print("No entries found.")
        return 0

    rows = [
        (
            "Date",
            "WI",
            "Hours",
            "Note",
            "Category",
            "Synced",
        )
    ]
    for entry in entries:
        rows.append(
            (
                entry.entry_date,
                str(entry.work_item_id),
                f"{entry.hours:.2f}",
                entry.note or "",
                entry.category or "",
                "yes" if entry.synced else "no",
            )
        )
    print(format_table(rows))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    conn = connect(resolve_db_path(config))
    init_db(conn)

    entries = fetch_entries(conn, limit=args.limit)
    conn.close()

    if args.format == "json":
        payload = [asdict(entry) for entry in entries]
        output = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        writer = csv.writer(sys.stdout)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(output_path.open("w", encoding="utf-8", newline=""))

    writer.writerow(
        [
            "entry_id",
            "date",
            "work_item_id",
            "hours",
            "note",
            "category",
            "created_at",
            "updated_at",
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
                entry.created_at,
                entry.updated_at,
                "true" if entry.synced else "false",
            ]
        )
    return 0


def format_sync_preview(entries: list[TimeEntry]) -> str:
    grouped: dict[int, list[TimeEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.work_item_id].append(entry)

    lines = []
    for work_item_id, items in grouped.items():
        total = sum(item.hours for item in items)
        lines.append(f"WI {work_item_id}: +{total:.2f}h ({len(items)} entries)")
    return "\n".join(lines)


def build_receipts(entries: list[TimeEntry]) -> list[SyncReceipt]:
    receipts = []
    for entry in entries:
        receipts.append(
            SyncReceipt(
                receipt_id=str(uuid.uuid4()),
                entry_id=entry.entry_id,
                work_item_id=entry.work_item_id,
                delta_completed_work=entry.hours,
                synced_at=utc_now(),
                patch_document=json.dumps(
                    {
                        "action": "local-only",
                        "note": "AzDo sync not yet implemented",
                        "entry_id": entry.entry_id,
                        "hours": entry.hours,
                    }
                ),
            )
        )
    return receipts


def cmd_sync(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    conn = connect(resolve_db_path(config))
    init_db(conn)

    entries = fetch_unsynced(conn)
    if not entries:
        print("No unsynced entries.")
        conn.close()
        return 0

    print("Planned updates (local-only preview):")
    print(format_sync_preview(entries))

    if not args.apply:
        print("\nRun again with --apply to mark entries as synced locally.")
        conn.close()
        return 0

    receipts = build_receipts(entries)
    insert_receipts(conn, receipts)
    mark_entries_synced(conn, [entry.entry_id for entry in entries])
    conn.close()

    print(f"Synced {len(entries)} entries locally.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="azdo-timesheet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            A fast, low-entry CLI for logging local time against Azure DevOps work items.
            """
        ).strip(),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create config + storage")
    init_parser.add_argument("--org-url", default="", help="Azure DevOps org URL")
    init_parser.add_argument("--project", default="", help="Default project")
    init_parser.add_argument("--pat", default="", help="Personal access token")
    init_parser.add_argument(
        "--storage-dir", default="", help="Directory for the local SQLite DB"
    )
    init_parser.set_defaults(func=cmd_init)

    add_parser = subparsers.add_parser("add", help="Log a time entry")
    add_parser.add_argument("--wi", dest="work_item", type=int, required=True)
    add_parser.add_argument("--hours", type=float, required=True)
    add_parser.add_argument("--note", default=None)
    add_parser.add_argument("--category", default=None)
    add_parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    add_parser.set_defaults(func=cmd_add)

    list_parser = subparsers.add_parser("list", help="List entries")
    list_parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    list_parser.add_argument("--wi", dest="work_item", type=int, default=None)
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.set_defaults(func=cmd_list)

    export_parser = subparsers.add_parser("export", help="Export entries")
    export_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    export_parser.add_argument("--output", default=None)
    export_parser.add_argument("--limit", type=int, default=500)
    export_parser.set_defaults(func=cmd_export)

    sync_parser = subparsers.add_parser("sync", help="Preview or mark sync locally")
    sync_parser.add_argument(
        "--apply", action="store_true", help="Mark entries as synced locally"
    )
    sync_parser.set_defaults(func=cmd_sync)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "add" and args.hours <= 0:
        parser.error("--hours must be greater than 0")

    return args.func(args)
