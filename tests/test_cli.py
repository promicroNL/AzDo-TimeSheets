import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from azdo_timesheet.cli import (
    build_parser,
    build_patch_operations,
    compute_remaining_after,
    config_show_command,
    format_entries,
    format_work_items,
    parse_work_item,
    truncate_note,
)
from azdo_timesheet.models import WorkItem, WorkItemDelta, WorkItemState
from azdo_timesheet.storage import MarkdownStorage, SQLiteStorage


class CliFormattingTests(unittest.TestCase):
    def test_truncate_note_keeps_exact_limit(self) -> None:
        note = "x" * 77
        self.assertEqual(truncate_note(note), note)

    def test_truncate_note_appends_ellipsis_after_limit(self) -> None:
        note = "x" * 80
        self.assertEqual(truncate_note(note), ("x" * 77) + "...")

    def test_format_entries_includes_inline_daily_rollup_and_total(self) -> None:
        from azdo_timesheet.models import Entry

        entries = [
            Entry("1", "2026-04-01", 1, 1.5, None, None, "a", "a", 0),
            Entry("2", "2026-04-01", 1, 2.0, None, None, "b", "b", 0),
            Entry("3", "2026-04-02", 2, 3.0, None, None, "c", "c", 0),
        ]

        output = format_entries(entries)


        self.assertIn("2026-04-01", output)
        self.assertIn("3.50", output)
        self.assertIn("2026-04-02", output)
        self.assertIn("3.00", output)
        self.assertIn("daily total", output)
        self.assertIn("selection total", output)
        self.assertIn("6.50", output)
        self.assertLess(output.index("daily total"), output.index("2026-04-02"))
        self.assertLess(output.index("2026-04-02"), output.rindex("daily total"))

    def test_format_work_items_includes_tags_column(self) -> None:
        output = format_work_items(
            [
                WorkItem(
                    work_item_id=123,
                    parent_work_item_id=50,
                    title="Title",
                    tags="foo; bar",
                    state="Active",
                    original_estimate=5.0,
                    remaining_work=2.0,
                    completed_work=3.0,
                    updated_at="2026-04-02T09:00:00",
                )
            ]
        )

        self.assertIn("tags", output)
        self.assertIn("foo; bar", output)


class RemainingWorkTests(unittest.TestCase):
    def test_parse_work_item_reads_tags_and_missing_remaining_flag(self) -> None:
        state = parse_work_item(
            {
                "id": 123,
                "fields": {
                    "System.Title": "Example",
                    "System.Tags": "foo; bar",
                    "System.State": "Active",
                    "System.Parent": 10,
                    "Microsoft.VSTS.Scheduling.CompletedWork": 1.5,
                },
            }
        )

        self.assertEqual(state.item.tags, "foo; bar")
        self.assertFalse(state.has_remaining_work)

    def test_missing_remaining_stays_unset_when_not_interactive(self) -> None:
        remaining = compute_remaining_after(
            strategy="decrement",
            remaining_before=None,
            original_estimate=8.0,
            completed_after=3.0,
            hours_logged=1.0,
            allow_interactive=False,
            work_item_id=123,
            work_item_title="Example",
            remaining_field_missing=True,
        )

        self.assertIsNone(remaining)

    def test_missing_remaining_prompts_when_interactive(self) -> None:
        with patch("builtins.input", return_value="2.5"):
            remaining = compute_remaining_after(
                strategy="decrement",
                remaining_before=None,
                original_estimate=8.0,
                completed_after=3.0,
                hours_logged=1.0,
                allow_interactive=True,
                work_item_id=123,
                work_item_title="Example",
                remaining_field_missing=True,
            )

        self.assertEqual(remaining, 2.5)

    def test_patch_operations_add_remaining_when_field_was_missing(self) -> None:
        delta = WorkItemDelta(
            work_item_id=123,
            entries=[],
            total_hours=1.0,
            completed_before=2.0,
            completed_after=3.0,
            remaining_field_missing=True,
            remaining_before=None,
            remaining_after=4.0,
            remaining_strategy="none",
        )
        state = WorkItemState(
            item=WorkItem(
                work_item_id=123,
                parent_work_item_id=None,
                title="Example",
                tags=None,
                state="Active",
                original_estimate=8.0,
                remaining_work=None,
                completed_work=2.0,
                updated_at="2026-04-02T09:00:00",
            ),
            has_original_estimate=True,
            has_remaining_work=False,
            has_completed_work=True,
        )

        operations = build_patch_operations(delta=delta, state=state)

        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[1]["op"], "add")
        self.assertEqual(
            operations[1]["path"], "/fields/Microsoft.VSTS.Scheduling.RemainingWork"
        )


class StorageTests(unittest.TestCase):
    def test_sqlite_migration_adds_tags_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "timesheet.sqlite"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE work_items (
                    work_item_id INTEGER PRIMARY KEY,
                    parent_work_item_id INTEGER,
                    title TEXT,
                    state TEXT,
                    original_estimate REAL,
                    remaining_work REAL,
                    completed_work REAL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.close()

            storage = SQLiteStorage(db_path)
            migrated = storage.connect()
            columns = {
                row["name"]
                for row in migrated.execute("PRAGMA table_info(work_items)").fetchall()
            }
            migrated.close()

            self.assertIn("tags", columns)

    def test_markdown_storage_round_trips_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = MarkdownStorage(Path(tmp))
            storage.init()
            storage.upsert_work_items(
                [
                    WorkItem(
                        work_item_id=123,
                        parent_work_item_id=10,
                        title="Example",
                        tags="foo; bar",
                        state="Active",
                        original_estimate=8.0,
                        remaining_work=4.0,
                        completed_work=4.0,
                        updated_at="2026-04-02T09:00:00",
                    )
                ]
            )

            items = storage.list_work_items()

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].tags, "foo; bar")


class ConfigCommandTests(unittest.TestCase):
    def test_config_show_prints_active_profile_and_storage_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "default",
                        "profiles": {
                            "default": {
                                "org_url": "https://dev.azure.com/example",
                                "project": "Demo",
                                "auth_mode": "pat",
                                "pat_env_var": "AZDO_PAT",
                                "remaining_work_strategy": "none",
                                "allow_sync_closed_items": False,
                                "max_hours_per_entry": 8,
                                "storage_backend": "sqlite",
                                "storage_path": str(Path(tmp) / "portable.sqlite"),
                                "wiql_query": "Select [System.Id] From WorkItems",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = config_show_command(
                    SimpleNamespace(config=str(config_path), profile=None)
                )

            self.assertEqual(result, 0)
            output = stdout.getvalue()
            self.assertIn("Config path:", output)
            self.assertIn("Active profile: default", output)
            self.assertIn("Storage path:", output)
            self.assertIn("portable.sqlite", output)

    def test_build_parser_includes_config_commands(self) -> None:
        parser = build_parser()
        subparsers = parser._subparsers._group_actions[0].choices

        self.assertIn("config", subparsers)
        config_parser = subparsers["config"]
        config_choices = config_parser._subparsers._group_actions[0].choices
        self.assertIn("show", config_choices)
        self.assertIn("edit", config_choices)


if __name__ == "__main__":
    unittest.main()





