from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    profile_name: str
    org_url: str
    project: str | None
    auth_mode: str
    pat_env_var: str
    remaining_work_strategy: str
    allow_sync_closed_items: bool
    max_hours_per_entry: float
    storage_backend: str
    storage_path: Path
    wiql_query: str | None


@dataclass(frozen=True)
class AppConfig:
    default_profile: str
    profiles: dict[str, Config]


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
    receipt_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WorkItem:
    work_item_id: int
    parent_work_item_id: int | None
    title: str | None
    state: str | None
    original_estimate: float | None
    remaining_work: float | None
    completed_work: float | None
    updated_at: str


@dataclass(frozen=True)
class WorkItemState:
    item: WorkItem
    has_original_estimate: bool
    has_remaining_work: bool
    has_completed_work: bool


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


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    entry_id: str
    work_item_id: int
    delta_completed_work: float
    synced_at: str
    patch_document: str | None
