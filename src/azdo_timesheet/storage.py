from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import yaml

from .models import Entry, Receipt, WorkItem

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


class SQLiteStorage:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    def init(self) -> None:
        self.connect().close()

    def add_entry(self, entry: Entry) -> None:
        with self.connect() as connection:
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

    def list_entries(
        self,
        *,
        work_item_id: int | None = None,
        entry_date: str | None = None,
    ) -> list[Entry]:
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
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [Entry(**row) for row in rows]

    def list_entries_range(self, *, start: date, end: date) -> list[Entry]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM entries
                WHERE entry_date BETWEEN ? AND ?
                ORDER BY entry_date, created_at
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [Entry(**row) for row in rows]

    def get_entry(self, entry_id: str) -> Entry | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM entries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
        return Entry(**row) if row else None

    def list_recent_entries(self, *, limit: int, include_synced: bool) -> list[Entry]:
        query = "SELECT * FROM entries"
        params: list[object] = []
        if not include_synced:
            query += " WHERE synced = 0"
        query += " ORDER BY entry_date DESC, created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [Entry(**row) for row in rows]

    def update_entry(self, entry_id: str, updates: dict[str, object]) -> bool:
        if not updates:
            return False
        fields = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [entry_id]
        with self.connect() as connection:
            connection.execute(
                f"UPDATE entries SET {fields} WHERE entry_id = ?",
                params,
            )
        return True

    def remove_entries(self, entry_ids: Sequence[str]) -> None:
        if not entry_ids:
            return
        with self.connect() as connection:
            connection.executemany(
                "DELETE FROM entries WHERE entry_id = ?",
                [(entry_id,) for entry_id in entry_ids],
            )

    def get_recent_work_items(self, *, limit: int = 5) -> list[tuple[int, str | None]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT entries.work_item_id, MAX(entries.created_at) AS last_seen, work_items.title
                FROM entries
                LEFT JOIN work_items ON entries.work_item_id = work_items.work_item_id
                GROUP BY entries.work_item_id
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(int(row["work_item_id"]), row["title"]) for row in rows]

    def get_unsynced_entries(self) -> list[Entry]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM entries WHERE synced = 0 ORDER BY work_item_id, entry_date"
            ).fetchall()
        return [Entry(**row) for row in rows]

    def upsert_work_items(self, items: Sequence[WorkItem]) -> None:
        if not items:
            return
        records = [
            (
                item.work_item_id,
                item.title,
                item.state,
                item.original_estimate,
                item.remaining_work,
                item.completed_work,
                item.updated_at,
            )
            for item in items
        ]
        with self.connect() as connection:
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

    def list_work_items(self) -> list[WorkItem]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM work_items ORDER BY updated_at DESC"
            ).fetchall()
        return [WorkItem(**row) for row in rows]

    def record_receipt(self, receipt: Receipt) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO receipts (
                    receipt_id, entry_id, work_item_id,
                    delta_completed_work, synced_at, patch_document
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    receipt.entry_id,
                    receipt.work_item_id,
                    receipt.delta_completed_work,
                    receipt.synced_at,
                    receipt.patch_document,
                ),
            )
            connection.execute(
                "UPDATE entries SET synced = 1, updated_at = ? WHERE entry_id = ?",
                (receipt.synced_at, receipt.entry_id),
            )


@dataclass(frozen=True)
class MarkdownIndex:
    recent_days: list[str]
    recent_weeks: list[str]


class MarkdownStorage:
    COLUMNS = [
        "Entry ID",
        "Date",
        "Work Item",
        "Hours",
        "Note",
        "Category",
        "Created At",
        "Updated At",
        "Synced",
        "Receipt IDs",
    ]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries_root = root / "entries"
        self.receipts_root = root / "receipts"
        self.work_items_path = root / "work_items.json"
        self.index_path = root / "README.md"

    def init(self) -> None:
        self.entries_root.mkdir(parents=True, exist_ok=True)
        self.receipts_root.mkdir(parents=True, exist_ok=True)
        if not self.work_items_path.exists():
            self._save_work_items({})
        if not self.index_path.exists():
            self.update_index([])
        self._ensure_root_folder_pages()

    def add_entry(self, entry: Entry) -> None:
        entries = self._load_entries_for_date(entry.entry_date)
        entries.append(entry)
        entries.sort(key=lambda item: item.created_at)
        self._write_entries_for_date(entry.entry_date, entries)
        self.update_index(self._load_all_entries())

    def list_entries(
        self,
        *,
        work_item_id: int | None = None,
        entry_date: str | None = None,
    ) -> list[Entry]:
        if entry_date:
            entries = self._load_entries_for_date(entry_date)
        else:
            entries = self._load_all_entries()
        if work_item_id is not None:
            entries = [entry for entry in entries if entry.work_item_id == work_item_id]
        entries.sort(key=lambda item: (item.entry_date, item.created_at), reverse=True)
        return entries

    def list_entries_range(self, *, start: date, end: date) -> list[Entry]:
        entries: list[Entry] = []
        current = start
        while current <= end:
            entries.extend(self._load_entries_for_date(current.isoformat()))
            current += timedelta(days=1)
        entries.sort(key=lambda item: (item.entry_date, item.created_at))
        return entries

    def get_entry(self, entry_id: str) -> Entry | None:
        result = self._find_entry(entry_id)
        return result.entry if result else None

    def list_recent_entries(self, *, limit: int, include_synced: bool) -> list[Entry]:
        entries = self._load_all_entries()
        if not include_synced:
            entries = [entry for entry in entries if entry.synced == 0]
        entries.sort(key=lambda item: (item.entry_date, item.created_at), reverse=True)
        return entries[:limit]

    def update_entry(self, entry_id: str, updates: dict[str, object]) -> bool:
        result = self._find_entry(entry_id)
        if not result:
            return False
        entry = result.entry
        updated_entry = Entry(
            entry_id=entry.entry_id,
            entry_date=str(updates.get("entry_date", entry.entry_date)),
            work_item_id=int(updates.get("work_item_id", entry.work_item_id)),
            hours=float(updates.get("hours", entry.hours)),
            note=updates.get("note", entry.note),
            category=updates.get("category", entry.category),
            created_at=str(updates.get("created_at", entry.created_at)),
            updated_at=str(updates.get("updated_at", entry.updated_at)),
            synced=int(updates.get("synced", entry.synced)),
            receipt_ids=tuple(updates.get("receipt_ids", entry.receipt_ids)),
        )
        self._replace_entry(result, updated_entry)
        self.update_index(self._load_all_entries())
        return True

    def remove_entries(self, entry_ids: Sequence[str]) -> None:
        for entry_id in entry_ids:
            result = self._find_entry(entry_id)
            if not result:
                continue
            entries = [
                entry for entry in result.entries if entry.entry_id != entry_id
            ]
            self._write_entries_for_date(result.entry.entry_date, entries)
        self.update_index(self._load_all_entries())

    def get_recent_work_items(self, *, limit: int = 5) -> list[tuple[int, str | None]]:
        entries = self._load_all_entries()
        last_seen: dict[int, str] = {}
        for entry in entries:
            last_seen[entry.work_item_id] = max(
                last_seen.get(entry.work_item_id, ""), entry.created_at
            )
        sorted_items = sorted(
            last_seen.items(), key=lambda item: item[1], reverse=True
        )[:limit]
        work_items = self._load_work_items()
        results: list[tuple[int, str | None]] = []
        for work_item_id, _ in sorted_items:
            cached = work_items.get(work_item_id)
            results.append((work_item_id, cached.title if cached else None))
        return results

    def get_unsynced_entries(self) -> list[Entry]:
        entries = [entry for entry in self._load_all_entries() if entry.synced == 0]
        entries.sort(key=lambda item: (item.work_item_id, item.entry_date))
        return entries

    def upsert_work_items(self, items: Sequence[WorkItem]) -> None:
        work_items = self._load_work_items()
        for item in items:
            work_items[item.work_item_id] = item
        self._save_work_items(work_items)

    def list_work_items(self) -> list[WorkItem]:
        work_items = list(self._load_work_items().values())
        work_items.sort(key=lambda item: item.updated_at, reverse=True)
        return work_items

    def record_receipt(self, receipt: Receipt) -> None:
        result = self._find_entry(receipt.entry_id)
        if not result:
            return
        entry = result.entry
        updated_entry = Entry(
            entry_id=entry.entry_id,
            entry_date=entry.entry_date,
            work_item_id=entry.work_item_id,
            hours=entry.hours,
            note=entry.note,
            category=entry.category,
            created_at=entry.created_at,
            updated_at=receipt.synced_at,
            synced=1,
            receipt_ids=tuple(list(entry.receipt_ids) + [receipt.receipt_id]),
        )
        self._replace_entry(result, updated_entry)
        self._append_receipt(receipt)
        self.update_index(self._load_all_entries())

    def update_index(self, entries: Sequence[Entry]) -> None:
        self._ensure_root_folder_pages()
        index = self._build_index(entries)
        lines = [
            "# Timesheet Index",
            "",
            "This directory is meant to be published to an Azure DevOps Wiki.",
            "",
            "## Recent Days",
        ]
        if index.recent_days:
            for day in index.recent_days:
                lines.append(f"- [{day}]({self._entry_link(day)})")
        else:
            lines.append("- No entries yet. Add one with `azdo-timesheet add`.")
        lines.extend(["", "## Recent Weeks"])
        if index.recent_weeks:
            for week_start in index.recent_weeks:
                lines.append(
                    f"- [Week of {week_start}]({self._entry_link(week_start)})"
                )
        else:
            lines.append("- No weekly data yet.")
        lines.extend(
            [
                "",
                "## Receipts",
                "Receipts are stored under `receipts/YYYY/MM.md`.",
            ]
        )
        self.index_path.write_text("\n".join(lines) + "\n")
        self._update_entry_summaries(entries)

    def _build_index(self, entries: Sequence[Entry]) -> MarkdownIndex:
        unique_dates = sorted(
            {entry.entry_date for entry in entries}, reverse=True
        )
        recent_days = unique_dates[:14]
        week_starts: list[str] = []
        for entry_date in unique_dates:
            day = date.fromisoformat(entry_date)
            start = day - timedelta(days=day.weekday())
            week_start = start.isoformat()
            if week_start not in week_starts:
                week_starts.append(week_start)
            if len(week_starts) >= 8:
                break
        return MarkdownIndex(recent_days=recent_days, recent_weeks=week_starts)

    def _entry_link(self, entry_date: str) -> str:
        day = date.fromisoformat(entry_date)
        return f"entries/{day:%Y/%m/%d}.md"

    def _entry_path(self, entry_date: str) -> Path:
        day = date.fromisoformat(entry_date)
        return self.entries_root / f"{day:%Y/%m/%d}.md"

    def _summary_path(self, *, year: int, month: int | None = None) -> Path:
        if month is None:
            return self.entries_root / f"{year}.md"
        return self.entries_root / f"{year}/{month:02d}.md"

    def _receipts_path(self, entry_date: str) -> Path:
        day = date.fromisoformat(entry_date)
        return self.receipts_root / f"{day:%Y/%m}.md"

    def _load_entries_for_date(self, entry_date: str) -> list[Entry]:
        path = self._entry_path(entry_date)
        if not path.exists():
            return []
        return self._parse_entries(path.read_text().splitlines())

    def _write_entries_for_date(self, entry_date: str, entries: Sequence[Entry]) -> None:
        path = self._entry_path(entry_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_entry_folder_pages(entry_date)
        lines = [
            f"# Timesheet Entries ({entry_date})",
            "",
            f"Date: {entry_date}",
            "",
            "## Entries",
            "",
        ]
        lines.append(self._format_table(entries))
        lines.extend(["", "## Canonical Entry Data", ""])
        lines.extend(self._format_fenced_entries(entries))
        path.write_text("\n".join(lines) + "\n")

    def _format_table(self, entries: Sequence[Entry]) -> str:
        header = "| " + " | ".join(self.COLUMNS) + " |"
        separator = "| " + " | ".join(["---"] * len(self.COLUMNS)) + " |"
        rows = [header, separator]
        for entry in entries:
            rows.append(
                "| "
                + " | ".join(
                    [
                        self._escape(entry.entry_id),
                        self._escape(entry.entry_date),
                        self._escape(str(entry.work_item_id)),
                        self._escape(f"{entry.hours:.2f}"),
                        self._escape(entry.note or ""),
                        self._escape(entry.category or ""),
                        self._escape(entry.created_at),
                        self._escape(entry.updated_at),
                        self._escape(str(entry.synced)),
                        self._escape(",".join(entry.receipt_ids)),
                    ]
                )
                + " |"
            )
        return "\n".join(rows)

    def _format_summary_table(
        self,
        entries: Sequence[Entry],
        work_items: dict[int, WorkItem],
    ) -> tuple[list[str], float]:
        totals: dict[int, float] = defaultdict(float)
        for entry in entries:
            totals[entry.work_item_id] += entry.hours
        header = "| Work Item ID | Title | Total Hours |"
        separator = "| --- | --- | --- |"
        rows = [header, separator]
        for work_item_id in sorted(totals):
            cached = work_items.get(work_item_id)
            title = cached.title if cached else ""
            rows.append(
                "| "
                + " | ".join(
                    [
                        self._escape(str(work_item_id)),
                        self._escape(title or ""),
                        self._escape(f"{totals[work_item_id]:.2f}"),
                    ]
                )
                + " |"
            )
        grand_total = sum(totals.values())
        return rows, grand_total

    def _parse_entries(self, lines: Sequence[str]) -> list[Entry]:
        block = self._extract_fenced_block(lines)
        if block:
            return self._parse_fenced_entries(block)
        return self._parse_legacy_table(lines)

    def _format_fenced_entries(self, entries: Sequence[Entry]) -> list[str]:
        lines = ["```jsonl"]
        for entry in entries:
            payload = self._serialize_entry(entry)
            lines.append(json.dumps(payload, ensure_ascii=False))
        lines.append("```")
        return lines

    def _serialize_entry(self, entry: Entry) -> dict[str, object]:
        return {
            "entry_id": entry.entry_id,
            "date": entry.entry_date,
            "work_item_id": entry.work_item_id,
            "hours": entry.hours,
            "note": entry.note,
            "category": entry.category,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "synced": entry.synced,
            "receipt_ids": list(entry.receipt_ids),
        }

    def _parse_fenced_entries(self, block: tuple[str, list[str]]) -> list[Entry]:
        language, content = block
        if language == "jsonl":
            entries: list[Entry] = []
            for line in content:
                if not line.strip():
                    continue
                payload = json.loads(line)
                entries.append(self._entry_from_payload(payload))
            return entries
        payload = yaml.safe_load("\n".join(content)) or []
        if isinstance(payload, dict):
            payload = payload.get("entries", [])
        if not isinstance(payload, list):
            return []
        return [self._entry_from_payload(item) for item in payload if isinstance(item, dict)]

    def _entry_from_payload(self, payload: dict[str, object]) -> Entry:
        entry_date = payload.get("date") or payload.get("entry_date") or ""
        receipt_ids = payload.get("receipt_ids") or []
        return Entry(
            entry_id=str(payload.get("entry_id", "")),
            entry_date=str(entry_date),
            work_item_id=int(payload.get("work_item_id", 0) or 0),
            hours=float(payload.get("hours", 0) or 0),
            note=payload.get("note") or None,
            category=payload.get("category") or None,
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            synced=int(payload.get("synced", 0) or 0),
            receipt_ids=tuple(receipt_ids),
        )

    def _extract_fenced_block(
        self, lines: Sequence[str]
    ) -> tuple[str, list[str]] | None:
        in_block = False
        language = ""
        content: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not in_block:
                if stripped.startswith("```"):
                    fence_language = stripped[3:].strip().lower()
                    if fence_language in {"jsonl", "yaml", "yml"}:
                        in_block = True
                        language = "yaml" if fence_language == "yml" else fence_language
                        continue
            else:
                if stripped.startswith("```"):
                    break
                content.append(line)
        if not in_block:
            return None
        return (language, content)

    def _parse_legacy_table(self, lines: Sequence[str]) -> list[Entry]:
        header = "| " + " | ".join(self.COLUMNS) + " |"
        entries: list[Entry] = []
        for idx, line in enumerate(lines):
            if line.strip() == header:
                start = idx + 2
                for row in lines[start:]:
                    if not row.strip().startswith("|"):
                        break
                    cells = [
                        cell.strip()
                        for cell in row.strip().strip("|").split("|")
                    ]
                    if len(cells) != len(self.COLUMNS):
                        continue
                    entries.append(
                        Entry(
                            entry_id=self._unescape(cells[0]),
                            entry_date=self._unescape(cells[1]),
                            work_item_id=int(self._unescape(cells[2]) or 0),
                            hours=float(self._unescape(cells[3]) or 0),
                            note=self._unescape(cells[4]) or None,
                            category=self._unescape(cells[5]) or None,
                            created_at=self._unescape(cells[6]),
                            updated_at=self._unescape(cells[7]),
                            synced=int(self._unescape(cells[8]) or 0),
                            receipt_ids=tuple(
                                filter(
                                    None,
                                    self._unescape(cells[9]).split(",")
                                    if cells[9].strip()
                                    else [],
                                )
                            ),
                        )
                    )
                break
        return entries

    def _escape(self, value: str) -> str:
        return value.replace("|", "&#124;").replace("\n", "<br>")

    def _unescape(self, value: str) -> str:
        return value.replace("&#124;", "|").replace("<br>", "\n")

    def _load_all_entries(self) -> list[Entry]:
        entries: list[Entry] = []
        if not self.entries_root.exists():
            return entries
        for path in self.entries_root.glob("*/*/*.md"):
            entries.extend(self._parse_entries(path.read_text().splitlines()))
        return entries

    def _load_work_items(self) -> dict[int, WorkItem]:
        if not self.work_items_path.exists():
            return {}
        data = json.loads(self.work_items_path.read_text())
        items: dict[int, WorkItem] = {}
        for item in data.get("items", []):
            work_item = WorkItem(**item)
            items[work_item.work_item_id] = work_item
        return items

    def _save_work_items(self, items: dict[int, WorkItem]) -> None:
        payload = {
            "items": [item.__dict__ for item in items.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.work_items_path.write_text(json.dumps(payload, indent=2))

    def _append_receipt(self, receipt: Receipt) -> None:
        path = self._receipts_path(receipt.synced_at[:10])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_receipts_folder_pages(receipt.synced_at[:10])
        if path.exists():
            lines = path.read_text().splitlines()
        else:
            month_label = f"{path.parent.name}-{path.stem}"
            lines = [
                f"# Receipts ({month_label})",
                "",
                "| Receipt ID | Entry ID | Work Item | Hours | Synced At | Patch Document |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        lines.append(
            "| "
            + " | ".join(
                [
                    self._escape(receipt.receipt_id),
                    self._escape(receipt.entry_id),
                    self._escape(str(receipt.work_item_id)),
                    self._escape(f"{receipt.delta_completed_work:.2f}"),
                    self._escape(receipt.synced_at),
                    self._escape(receipt.patch_document or ""),
                ]
            )
            + " |"
        )
        path.write_text("\n".join(lines) + "\n")

    def _ensure_root_folder_pages(self) -> None:
        self._ensure_folder_page(self.entries_root, "Entries")
        self._ensure_folder_page(self.receipts_root, "Receipts")

    def _update_entry_summaries(self, entries: Sequence[Entry]) -> None:
        if not entries:
            return
        work_items = self._load_work_items()
        entries_by_year: dict[int, list[Entry]] = defaultdict(list)
        entries_by_month: dict[tuple[int, int], list[Entry]] = defaultdict(list)
        for entry in entries:
            day = date.fromisoformat(entry.entry_date)
            entries_by_year[day.year].append(entry)
            entries_by_month[(day.year, day.month)].append(entry)
        for year, year_entries in entries_by_year.items():
            self._write_summary_page(year=year, entries=year_entries, work_items=work_items)
        for (year, month), month_entries in entries_by_month.items():
            self._write_summary_page(
                year=year,
                month=month,
                entries=month_entries,
                work_items=work_items,
            )

    def _write_summary_page(
        self,
        *,
        year: int,
        entries: Sequence[Entry],
        work_items: dict[int, WorkItem],
        month: int | None = None,
    ) -> None:
        page_path = self._summary_path(year=year, month=month)
        page_path.parent.mkdir(parents=True, exist_ok=True)
        label = f"{year}/{month:02d}" if month is not None else f"{year}"
        title = f"Entries {label}"
        table_lines, grand_total = self._format_summary_table(entries, work_items)
        lines = [
            f"# {title}",
            "",
            "## Summary",
            "",
            *table_lines,
            "",
            f"**Grand Total:** {grand_total:.2f} hours",
            "",
            "[[_TOSP_]]",
        ]
        page_path.write_text("\n".join(lines) + "\n")

    def _ensure_entry_folder_pages(self, entry_date: str) -> None:
        day = date.fromisoformat(entry_date)
        year_folder = self.entries_root / f"{day:%Y}"
        month_folder = year_folder / f"{day:%m}"
        self._ensure_folder_page(year_folder, f"Entries {day:%Y}")
        self._ensure_folder_page(month_folder, f"Entries {day:%Y/%m}")

    def _ensure_receipts_folder_pages(self, entry_date: str) -> None:
        day = date.fromisoformat(entry_date)
        year_folder = self.receipts_root / f"{day:%Y}"
        self._ensure_folder_page(year_folder, f"Receipts {day:%Y}")

    def _ensure_folder_page(self, folder: Path, title: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        page_path = folder.parent / f"{folder.name}.md"
        if page_path.exists():
            content = page_path.read_text()
            if "[[_TOSP_]]" in content:
                return
            content = content.rstrip()
            if content:
                content = f"{content}\n\n[[_TOSP_]]\n"
            else:
                content = f"# {title}\n\n[[_TOSP_]]\n"
            page_path.write_text(content)
            return
        page_path.write_text(f"# {title}\n\n[[_TOSP_]]\n")

    @dataclass(frozen=True)
    class _EntrySearchResult:
        entry: Entry
        entries: list[Entry]

    def _find_entry(self, entry_id: str) -> _EntrySearchResult | None:
        if not self.entries_root.exists():
            return None
        for path in self.entries_root.glob("*/*/*.md"):
            entries = self._parse_entries(path.read_text().splitlines())
            for entry in entries:
                if entry.entry_id == entry_id:
                    return self._EntrySearchResult(entry=entry, entries=entries)
        return None

    def _replace_entry(self, result: _EntrySearchResult, updated_entry: Entry) -> None:
        entries = [
            item for item in result.entries if item.entry_id != updated_entry.entry_id
        ]
        if updated_entry.entry_date != result.entry.entry_date:
            self._write_entries_for_date(result.entry.entry_date, entries)
            target_entries = self._load_entries_for_date(updated_entry.entry_date)
            target_entries.append(updated_entry)
            target_entries.sort(key=lambda item: item.created_at)
            self._write_entries_for_date(updated_entry.entry_date, target_entries)
        else:
            entries.append(updated_entry)
            entries.sort(key=lambda item: item.created_at)
            self._write_entries_for_date(updated_entry.entry_date, entries)
