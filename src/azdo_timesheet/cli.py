import argparse
import base64
import csv
import json
import os
import sqlite3
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib import error, parse, request
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

CREATE TABLE IF NOT EXISTS work_items (
    work_item_id INTEGER PRIMARY KEY,
    title TEXT,
    state TEXT,
    original_estimate REAL,
    remaining_work REAL,
    completed_work REAL,
    updated_at TEXT NOT NULL
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
    wiql_query: str | None


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


@dataclass(frozen=True)
class WorkItem:
    work_item_id: int
    title: str | None
    state: str | None
    original_estimate: float | None
    remaining_work: float | None
    completed_work: float | None
    updated_at: str


@dataclass(frozen=True)
class WorkItemDelta:
    work_item_id: int
    entries: list[Entry]
    total_hours: float
    completed_before: float
    completed_after: float
    remaining_before: float | None
    remaining_after: float | None
    remaining_strategy: str


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
        "storage_path": str(config.storage_path),
        "wiql_query": config.wiql_query,
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
        remaining_work_strategy=args.remaining_work_strategy,
        allow_sync_closed_items=False,
        max_hours_per_entry=float(args.max_hours_per_entry),
        storage_path=db_path,
        wiql_query=args.wiql_query,
    )
    save_config(config_path, config)
    connect_db(db_path).close()
    print(f"Initialized config at {config_path}")
    print(f"Storage: {db_path}")
    return 0


def get_recent_work_items(connection: sqlite3.Connection, *, limit: int = 5) -> list[int]:
    rows = connection.execute(
        """
        SELECT work_item_id, MAX(created_at) AS last_seen
        FROM entries
        GROUP BY work_item_id
        ORDER BY last_seen DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [int(row["work_item_id"]) for row in rows]


def prompt_for_work_item(connection: sqlite3.Connection) -> int:
    if not sys.stdin.isatty():
        raise ValueError("Work item id required when running non-interactively.")
    recents = get_recent_work_items(connection)
    if not recents:
        raise ValueError("No recent work items found. Provide --wi.")
    print("Recent work items:")
    for index, work_item_id in enumerate(recents, start=1):
        print(f"  [{index}] {work_item_id}")
    selection = input("Pick a work item number or enter an id: ").strip()
    if selection.isdigit():
        choice = int(selection)
        if 1 <= choice <= len(recents):
            return recents[choice - 1]
        return choice
    raise ValueError("Invalid selection. Provide a numeric work item id.")


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
    with connect_db(config.storage_path) as connection:
        try:
            work_item_id = int(args.work_item_id) if args.work_item_id else None
        except ValueError:
            print("Work item id must be a number.", file=sys.stderr)
            return 2
        if work_item_id is None:
            try:
                work_item_id = prompt_for_work_item(connection)
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


def edit_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    updates: list[str] = []
    params: list[object] = []
    if args.work_item_id is not None:
        updates.append("work_item_id = ?")
        params.append(int(args.work_item_id))
    if args.hours is not None:
        updates.append("hours = ?")
        params.append(float(args.hours))
    if args.note is not None:
        updates.append("note = ?")
        params.append(args.note)
    if args.category is not None:
        updates.append("category = ?")
        params.append(args.category)
    if args.date is not None:
        updates.append("entry_date = ?")
        params.append(args.date)
    if not updates:
        print("No fields provided to update.", file=sys.stderr)
        return 2
    now = datetime.utcnow().isoformat()
    updates.append("updated_at = ?")
    params.append(now)
    params.extend([args.entry_id])

    with connect_db(config.storage_path) as connection:
        row = connection.execute(
            "SELECT synced FROM entries WHERE entry_id = ?",
            (args.entry_id,),
        ).fetchone()
        if row is None:
            print("Entry not found.", file=sys.stderr)
            return 2
        if row["synced"]:
            print("Cannot edit synced entries.", file=sys.stderr)
            return 2
        connection.execute(
            f"UPDATE entries SET {', '.join(updates)} WHERE entry_id = ?",
            params,
        )
    print(f"Updated entry {args.entry_id}.")
    return 0


def remove_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    with connect_db(config.storage_path) as connection:
        for entry_id in args.entry_id:
            row = connection.execute(
                "SELECT synced FROM entries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            if row is None:
                print(f"Entry {entry_id} not found.", file=sys.stderr)
                return 2
            if row["synced"]:
                print(f"Entry {entry_id} is synced and cannot be removed.", file=sys.stderr)
                return 2
        connection.executemany(
            "DELETE FROM entries WHERE entry_id = ?",
            [(entry_id,) for entry_id in args.entry_id],
        )
    print(f"Removed {len(args.entry_id)} entries.")
    return 0


def fetch_work_item(connection: sqlite3.Connection, work_item_id: int) -> WorkItem | None:
    row = connection.execute(
        "SELECT * FROM work_items WHERE work_item_id = ?",
        (work_item_id,),
    ).fetchone()
    return WorkItem(**row) if row else None


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
    with connect_db(config.storage_path) as connection:
        rows = connection.execute(
            "SELECT * FROM work_items ORDER BY updated_at DESC"
        ).fetchall()
    work_items = [WorkItem(**row) for row in rows]
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
    payload: dict | None = None,
) -> dict:
    if not config.org_url:
        raise ValueError("org_url is required to sync work items.")
    if not config.project:
        raise ValueError("project is required to sync work items.")
    token = pat_token(config)
    auth = base64.b64encode(f":{token}".encode("utf-8")).decode("utf-8")
    url = f"{config.org_url.rstrip('/')}/{config.project}/{path}"
    body = None
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise ValueError(f"Azure DevOps request failed: {exc.code} {detail}") from exc


def sync_work_items_from_wiql(
    *, connection: sqlite3.Connection, config: Config
) -> int:
    if not config.wiql_query:
        raise ValueError("wiql_query is not configured. Set it in config.json.")
    wiql_payload = {"query": config.wiql_query}
    wiql_response = azdo_request(
        config=config,
        method="POST",
        path="_apis/wit/wiql?api-version=7.0",
        payload=wiql_payload,
    )
    work_items = wiql_response.get("workItems", [])
    ids = [item["id"] for item in work_items]
    if not ids:
        return 0
    fields = ",".join(
        [
            "System.Title",
            "System.State",
            "Microsoft.VSTS.Scheduling.OriginalEstimate",
            "Microsoft.VSTS.Scheduling.RemainingWork",
            "Microsoft.VSTS.Scheduling.CompletedWork",
        ]
    )
    ids_param = ",".join(str(item_id) for item_id in ids)
    work_items_response = azdo_request(
        config=config,
        method="GET",
        path=f"_apis/wit/workitems?ids={parse.quote(ids_param)}&fields={parse.quote(fields)}&api-version=7.0",
    )
    now = datetime.utcnow().isoformat()
    records = []
    for item in work_items_response.get("value", []):
        fields_data = item.get("fields", {})
        records.append(
            (
                item.get("id"),
                fields_data.get("System.Title"),
                fields_data.get("System.State"),
                fields_data.get("Microsoft.VSTS.Scheduling.OriginalEstimate"),
                fields_data.get("Microsoft.VSTS.Scheduling.RemainingWork"),
                fields_data.get("Microsoft.VSTS.Scheduling.CompletedWork"),
                now,
            )
        )
    connection.executemany(
        """
        INSERT INTO work_items (
            work_item_id, title, state, original_estimate, remaining_work,
            completed_work, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(work_item_id) DO UPDATE SET
            title = excluded.title,
            state = excluded.state,
            original_estimate = excluded.original_estimate,
            remaining_work = excluded.remaining_work,
            completed_work = excluded.completed_work,
            updated_at = excluded.updated_at
        """,
        records,
    )
    return len(records)


def work_item_sync_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    with connect_db(config.storage_path) as connection:
        try:
            count = sync_work_items_from_wiql(connection=connection, config=config)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    print(f"Synced {count} work items from WIQL.")
    return 0


def compute_remaining_after(
    *,
    strategy: str,
    remaining_before: float | None,
    original_estimate: float | None,
    completed_after: float,
    hours_logged: float,
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
    if strategy == "prompt":
        if not sys.stdin.isatty():
            return remaining_before
        default_value = max(remaining_before - hours_logged, 0.0)
        raw = input(f"Remaining work (default {default_value:.2f}): ").strip()
        if not raw:
            return default_value
        try:
            return max(float(raw), 0.0)
        except ValueError:
            print("Invalid number, keeping previous remaining work.")
            return remaining_before
    return remaining_before


def plan_deltas(
    connection: sqlite3.Connection,
    entries: Sequence[Entry],
    remaining_work_strategy: str,
) -> list[WorkItemDelta]:
    grouped: dict[int, list[Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.work_item_id, []).append(entry)

    deltas: list[WorkItemDelta] = []
    for work_item_id, group in grouped.items():
        total_hours = sum(item.hours for item in group)
        work_item = fetch_work_item(connection, work_item_id)
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


def sync_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    with connect_db(config.storage_path) as connection:
        sync_work_items = args.sync_work_items
        if sync_work_items is None:
            sync_work_items = args.apply
        if sync_work_items:
            try:
                count = sync_work_items_from_wiql(connection=connection, config=config)
                print(f"Synced {count} work items from WIQL.")
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
        rows = connection.execute(
            "SELECT * FROM entries WHERE synced = 0 ORDER BY work_item_id, entry_date"
        ).fetchall()
        if not rows:
            print("No unsynced entries.")
            return 0
        entries = [Entry(**row) for row in rows]
        deltas = plan_deltas(
            connection,
            entries,
            remaining_work_strategy=args.remaining_work_strategy
            or config.remaining_work_strategy,
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

        if args.apply:
            now = datetime.utcnow().isoformat()
            for delta in deltas:
                work_item = fetch_work_item(connection, delta.work_item_id)
                if work_item:
                    connection.execute(
                        """
                        UPDATE work_items
                        SET completed_work = ?, remaining_work = ?, updated_at = ?
                        WHERE work_item_id = ?
                        """,
                        (
                            delta.completed_after,
                            delta.remaining_after
                            if delta.remaining_after is not None
                            else work_item.remaining_work,
                            now,
                            delta.work_item_id,
                        ),
                    )
                for entry in delta.entries:
                    receipt_id = str(uuid.uuid4())
                    patch_payload = {
                        "work_item_id": delta.work_item_id,
                        "completed_before": delta.completed_before,
                        "completed_after": delta.completed_after,
                        "remaining_before": delta.remaining_before,
                        "remaining_after": delta.remaining_after,
                        "strategy": delta.remaining_strategy,
                    }
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
                            json.dumps(patch_payload),
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


def week_bounds(day: date) -> tuple[date, date]:
    start = day - timedelta(days=day.weekday())
    end = start + timedelta(days=6)
    return start, end


def export_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).expanduser())
    target_day = date.fromisoformat(args.week)
    start, end = week_bounds(target_day)
    with connect_db(config.storage_path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM entries
            WHERE entry_date BETWEEN ? AND ?
            ORDER BY entry_date, created_at
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    entries = [Entry(**row) for row in rows]

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
        "--remaining-work-strategy",
        default="none",
        choices=["none", "decrement", "recalc_from_original", "prompt"],
        help="Remaining Work strategy (default: none)",
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
    add_parser.add_argument("--h", dest="hours", required=True, type=float)
    add_parser.add_argument("--note")
    add_parser.add_argument("--category")
    add_parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    add_parser.set_defaults(func=add_command)

    list_parser = subparsers.add_parser("list", help="List time entries")
    list_parser.add_argument("--wi", dest="work_item_id", type=int)
    list_parser.add_argument("--date", help="YYYY-MM-DD")
    list_parser.set_defaults(func=list_command)

    edit_parser = subparsers.add_parser("edit", help="Edit an unsynced entry")
    edit_parser.add_argument("--id", dest="entry_id", required=True)
    edit_parser.add_argument("--wi", dest="work_item_id")
    edit_parser.add_argument("--h", dest="hours", type=float)
    edit_parser.add_argument("--note")
    edit_parser.add_argument("--category")
    edit_parser.add_argument("--date", help="YYYY-MM-DD")
    edit_parser.set_defaults(func=edit_command)

    remove_parser = subparsers.add_parser("remove", help="Remove unsynced entries")
    remove_parser.add_argument("--id", dest="entry_id", action="append", required=True)
    remove_parser.set_defaults(func=remove_command)

    work_item_parser = subparsers.add_parser("wi", help="Manage local work items")
    wi_subparsers = work_item_parser.add_subparsers(dest="wi_command", required=True)

    wi_sync = wi_subparsers.add_parser("sync", help="Sync work items from WIQL")
    wi_sync.set_defaults(func=work_item_sync_command)

    wi_list = wi_subparsers.add_parser("list", help="List work items")
    wi_list.set_defaults(func=work_item_list_command)

    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync work items (download) and entries (local-only placeholder)",
    )
    sync_parser.add_argument(
        "--apply",
        action="store_true",
        help="Mark entries as synced and create receipts (local-only)",
    )
    sync_parser.add_argument(
        "--remaining-work-strategy",
        choices=["none", "decrement", "recalc_from_original", "prompt"],
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
