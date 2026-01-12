import argparse
import json
import sqlite3
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_CONFIG_DIR = Path.home() / ".azdo_timesheet"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_DB_PATH = DEFAULT_CONFIG_DIR / "timesheet.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    entry_id TEXT PRIMARY KEY,
    entry_date TEXT NOT NULL,
    work_item_id INTEGER NOT NULL,
    hours REAL NOT NULL,
    note TEXT,
    category TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS receipts (
    receipt_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL,
    work_item_id INTEGER NOT NULL,
    delta_completed_work REAL NOT NULL,
    synced_at TEXT NOT NULL,
    patch_document TEXT,
    FOREIGN KEY(entry_id) REFERENCES entries(entry_id)
);
"""


@dataclass(frozen=True)
class Config:
    org_url: str
    project: str | None
    auth_mode: str
    pat_env_var: str
    remaining_work_strategy: str
    allow_sync_closed_items: bool
    max_hours_per_entry: float
    storage_path: Path


@dataclass(frozen=True)
class Entry:
    entry_id: str
    entry_date: str
    work_item_id: int
    hours: float
    note: str | None
    category: str | None
    created_at: str
    updated_at: str
    synced: int


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Run 'azdo-timesheet init' first."
        )
    data = json.loads(path.read_text())
    return Config(
        org_url=data.get("org_url", ""),
        project=data.get("project"),
        auth_mode=data.get("auth_mode", "pat"),
        pat_env_var=data.get("pat_env_var", "AZDO_PAT"),
        remaining_work_strategy=data.get("remaining_work_strategy", "none"),
        allow_sync_closed_items=bool(data.get("allow_sync_closed_items", False)),
        max_hours_per_entry=float(data.get("max_hours_per_entry", 8)),
        storage_path=Path(data.get("storage_path", DEFAULT_DB_PATH)),
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
        "storage_path": str(config.storage_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def init_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    db_path = Path(args.storage).expanduser().resolve() if args.storage else DEFAULT_DB_PATH
    config = Config(
        org_url=args.org_url or "",
        project=args.project,
        auth_mode="pat",
        pat_env_var=args.pat_env_var,
        remaining_work_strategy="none",
        allow_sync_closed_items=False,
        max_hours_per_entry=float(args.max_hours_per_entry),
        storage_path=db_path,
    )
    save_config(config_path, config)
    connect_db(db_path).close()
    print(f"Initialized config at {config_path}")
    print(f"Storage: {db_path}")
    return 0


def add_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
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
    entry = Entry(
        entry_id=str(uuid.uuid4()),
        entry_date=entry_date,
        work_item_id=int(args.work_item_id),
        hours=float(args.hours),
        note=args.note,
        category=args.category,
        created_at=now,
        updated_at=now,
        synced=0,
    )
    with connect_db(config.storage_path) as connection:
        connection.execute(
            """
            INSERT INTO entries (
                entry_id, entry_date, work_item_id, hours, note, category,
                created_at, updated_at, synced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.entry_date,
                entry.work_item_id,
                entry.hours,
                entry.note,
                entry.category,
                entry.created_at,
                entry.updated_at,
                entry.synced,
            ),
        )
    print(f"Added entry {entry.entry_id} for WI #{entry.work_item_id}")
    return 0


def list_entries(
    connection: sqlite3.Connection, *, work_item_id: int | None, entry_date: str | None
) -> Iterable[Entry]:
    query = "SELECT * FROM entries"
    conditions: list[str] = []
    params: list[object] = []
    if work_item_id is not None:
        conditions.append("work_item_id = ?")
        params.append(work_item_id)
    if entry_date is not None:
        conditions.append("entry_date = ?")
        params.append(entry_date)
    if conditions:
        query = f"{query} WHERE {' AND '.join(conditions)}"
    query = f"{query} ORDER BY entry_date DESC, created_at DESC"
    for row in connection.execute(query, params):
        yield Entry(**row)


def format_entries(entries: Sequence[Entry]) -> str:
    if not entries:
        return "No entries found."
    lines = [
        "entry_id | date | wi | hours | synced | note",
        "-" * 72,
    ]
    for entry in entries:
        note = (entry.note or "").replace("\n", " ")
        lines.append(
            f"{entry.entry_id} | {entry.entry_date} | {entry.work_item_id} | "
            f"{entry.hours:.2f} | {entry.synced} | {note}"
        )
    return "\n".join(lines)


def list_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    with connect_db(config.storage_path) as connection:
        entries = list(
            list_entries(
                connection,
                work_item_id=args.work_item_id,
                entry_date=args.date,
            )
        )
    print(format_entries(entries))
    return 0


def sync_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    with connect_db(config.storage_path) as connection:
        rows = connection.execute(
            "SELECT * FROM entries WHERE synced = 0 ORDER BY work_item_id, entry_date"
        ).fetchall()
        if not rows:
            print("No unsynced entries.")
            return 0
        entries = [Entry(**row) for row in rows]
        grouped: dict[int, list[Entry]] = {}
        for entry in entries:
            grouped.setdefault(entry.work_item_id, []).append(entry)

        for work_item_id, group in grouped.items():
            total_hours = sum(item.hours for item in group)
            print(
                textwrap.dedent(
                    f"""
                    Work Item #{work_item_id}
                      Entries: {len(group)}
                      Completed Work: +{total_hours:.2f} (local-only preview)
                    """
                ).strip()
            )
            print("-")

        if args.apply:
            now = datetime.utcnow().isoformat()
            for entry in entries:
                receipt_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO receipts (
                        receipt_id, entry_id, work_item_id,
                        delta_completed_work, synced_at, patch_document
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt_id,
                        entry.entry_id,
                        entry.work_item_id,
                        entry.hours,
                        now,
                        json.dumps({"note": "local-only sync"}),
                    ),
                )
                connection.execute(
                    "UPDATE entries SET synced = 1, updated_at = ? WHERE entry_id = ?",
                    (now, entry.entry_id),
                )
            connection.commit()
            print("Marked entries as synced locally. (No Azure DevOps updates yet.)")
        else:
            print("Dry run only. Use --apply to mark entries synced locally.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="azdo-timesheet",
        description="Low-entry Azure DevOps timesheet (local-first).",
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
    init_parser.add_argument("--storage", help="SQLite file path")
    init_parser.add_argument(
        "--pat-env-var",
        default="AZDO_PAT",
        help="Environment variable name containing a PAT",
    )
    init_parser.add_argument(
        "--max-hours-per-entry",
        default=8,
        type=float,
        help="Warning threshold for large entries",
    )
    init_parser.set_defaults(func=init_command)

    add_parser = subparsers.add_parser("add", help="Add a time entry")
    add_parser.add_argument("--wi", dest="work_item_id", required=True)
    add_parser.add_argument("--h", dest="hours", required=True, type=float)
    add_parser.add_argument("--note")
    add_parser.add_argument("--category")
    add_parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    add_parser.set_defaults(func=add_command)

    list_parser = subparsers.add_parser("list", help="List time entries")
    list_parser.add_argument("--wi", dest="work_item_id", type=int)
    list_parser.add_argument("--date", help="YYYY-MM-DD")
    list_parser.set_defaults(func=list_command)

    sync_parser = subparsers.add_parser(
        "sync", help="Preview sync output (local-only placeholder)"
    )
    sync_parser.add_argument(
        "--apply",
        action="store_true",
        help="Mark entries as synced and create receipts (local-only)",
    )
    sync_parser.set_defaults(func=sync_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
