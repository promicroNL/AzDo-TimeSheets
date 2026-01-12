from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

ISO_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class TimeEntry:
    entry_id: str
    entry_date: str
    work_item_id: int
    hours: float
    note: Optional[str]
    category: Optional[str]
    created_at: str
    updated_at: str
    synced: bool


@dataclass(frozen=True)
class SyncReceipt:
    receipt_id: str
    entry_id: str
    work_item_id: int
    delta_completed_work: float
    synced_at: str
    patch_document: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS time_entries (
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
        CREATE TABLE IF NOT EXISTS sync_receipts (
            receipt_id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            work_item_id INTEGER NOT NULL,
            delta_completed_work REAL NOT NULL,
            synced_at TEXT NOT NULL,
            patch_document TEXT NOT NULL,
            FOREIGN KEY(entry_id) REFERENCES time_entries(entry_id)
        );
        """
    )
    conn.commit()


def insert_entry(conn: sqlite3.Connection, entry: TimeEntry) -> None:
    conn.execute(
        """
        INSERT INTO time_entries (
            entry_id,
            entry_date,
            work_item_id,
            hours,
            note,
            category,
            created_at,
            updated_at,
            synced
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
            1 if entry.synced else 0,
        ),
    )
    conn.commit()


def fetch_entries(
    conn: sqlite3.Connection,
    *,
    entry_date: Optional[str] = None,
    work_item_id: Optional[int] = None,
    limit: int = 50,
) -> list[TimeEntry]:
    query = "SELECT * FROM time_entries"
    clauses = []
    params: list[object] = []
    if entry_date:
        clauses.append("entry_date = ?")
        params.append(entry_date)
    if work_item_id:
        clauses.append("work_item_id = ?")
        params.append(work_item_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY entry_date DESC, created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [
        TimeEntry(
            entry_id=row["entry_id"],
            entry_date=row["entry_date"],
            work_item_id=row["work_item_id"],
            hours=row["hours"],
            note=row["note"],
            category=row["category"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            synced=bool(row["synced"]),
        )
        for row in rows
    ]


def fetch_unsynced(conn: sqlite3.Connection) -> list[TimeEntry]:
    rows = conn.execute(
        "SELECT * FROM time_entries WHERE synced = 0 ORDER BY entry_date ASC"
    ).fetchall()
    return [
        TimeEntry(
            entry_id=row["entry_id"],
            entry_date=row["entry_date"],
            work_item_id=row["work_item_id"],
            hours=row["hours"],
            note=row["note"],
            category=row["category"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            synced=bool(row["synced"]),
        )
        for row in rows
    ]


def insert_receipts(conn: sqlite3.Connection, receipts: Iterable[SyncReceipt]) -> None:
    conn.executemany(
        """
        INSERT INTO sync_receipts (
            receipt_id,
            entry_id,
            work_item_id,
            delta_completed_work,
            synced_at,
            patch_document
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                receipt.receipt_id,
                receipt.entry_id,
                receipt.work_item_id,
                receipt.delta_completed_work,
                receipt.synced_at,
                receipt.patch_document,
            )
            for receipt in receipts
        ],
    )
    conn.commit()


def mark_entries_synced(conn: sqlite3.Connection, entry_ids: Iterable[str]) -> None:
    entry_ids = list(entry_ids)
    if not entry_ids:
        return
    conn.executemany(
        "UPDATE time_entries SET synced = 1, updated_at = ? WHERE entry_id = ?",
        [(utc_now(), entry_id) for entry_id in entry_ids],
    )
    conn.commit()
