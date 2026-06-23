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
    format_parent_summary,
    format_work_items,
    add_work_minutes,
    moneybird_project_id_from_tags,
    plan_moneybird_time_entries,
    parse_work_item,
    repair_markdown_tables_command,
    truncate_note,
)
from azdo_timesheet.models import Config, Entry, WorkItem, WorkItemDelta, WorkItemState
from azdo_timesheet.storage import MarkdownStorage, SQLiteStorage


class CliFormattingTests(unittest.TestCase):
    def test_truncate_note_keeps_exact_limit(self) -> None:
        note = "x" * 77
        self.assertEqual(truncate_note(note), note)

    def test_truncate_note_appends_ellipsis_after_limit(self) -> None:
        note = "x" * 80
        self.assertEqual(truncate_note(note), ("x" * 77) + "...")

    def test_format_entries_includes_inline_daily_rollup_and_total(self) -> None:
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

    def test_format_entries_visualizes_moneybird_project(self) -> None:
        entries = [Entry("1", "2026-04-01", 101, 1.5, None, None, "a", "a", 0)]
        work_items = {
            101: WorkItem(101, 10, "Child", None, "Active", None, None, None, "u"),
            10: WorkItem(10, None, "Parent", "mb:459275493454120239", "Active", None, None, None, "u"),
        }

        output = format_entries(entries, work_items)

        self.assertIn("mb_project", output)
        self.assertIn("459275493454120239", output)

    def test_format_parent_summary_visualizes_moneybird_project(self) -> None:
        entries = [Entry("1", "2026-04-01", 101, 1.5, None, None, "a", "a", 0)]
        work_items = [
            WorkItem(101, 10, "Child", None, "Active", None, None, None, "u"),
            WorkItem(10, None, "Parent", "mb:459275493454120239", "Active", None, None, None, "u"),
        ]

        output = format_parent_summary(entries, work_items)

        self.assertIn("mb_project", output)
        self.assertIn("459275493454120239", output)


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


class MoneybirdExportTests(unittest.TestCase):
    def test_moneybird_project_id_is_read_from_parent_tags(self) -> None:
        self.assertEqual(
            moneybird_project_id_from_tags("foo; mb:459275493454120239; bar"),
            "459275493454120239",
        )

    def test_add_work_minutes_skips_lunch_and_dinner_breaks(self) -> None:
        end = add_work_minutes(8 * 60, 10 * 60, [(12 * 60, 13 * 60), (18 * 60, 19 * 60 + 30)])
        self.assertEqual(end, 20 * 60 + 30)

    def test_plan_moneybird_time_entries_groups_by_day_and_parent(self) -> None:
        config = Config(
            profile_name="default",
            org_url="",
            project=None,
            auth_mode="pat",
            pat_env_var="AZDO_PAT",
            remaining_work_strategy="none",
            allow_sync_closed_items=False,
            max_hours_per_entry=8,
            storage_backend="sqlite",
            storage_path=Path("timesheet.sqlite"),
            wiql_query=None,
            moneybird_administration_id="123",
            moneybird_token_env_var="MONEYBIRD_TOKEN",
            moneybird_start_time="08:00",
            moneybird_lunch_break_start="12:00",
            moneybird_lunch_break_end="13:00",
            moneybird_dinner_break_start="18:00",
            moneybird_dinner_break_end="19:30",
            moneybird_timezone="Z",
            moneybird_billable=False,
        )
        entries = [
            Entry("1", "2026-04-01", 101, 3.0, None, None, "a", "a", 0),
            Entry("2", "2026-04-01", 102, 2.0, None, None, "b", "b", 0),
        ]
        work_items = [
            WorkItem(101, 10, "Child 1", None, "Active", None, None, None, "u"),
            WorkItem(102, 10, "Child 2", None, "Active", None, None, None, "u"),
            WorkItem(10, None, "Parent", "mb:459275493454120239", "Active", None, None, None, "u"),
        ]

        planned = plan_moneybird_time_entries(entries, work_items, config)

        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["hours"], 5.0)
        time_entry = planned[0]["time_entry"]
        self.assertEqual(time_entry["project_id"], "459275493454120239")
        self.assertEqual(time_entry["started_at"], "2026-04-01T08:00:00Z")
        self.assertEqual(time_entry["ended_at"], "2026-04-01T14:00:00Z")
        self.assertFalse(time_entry["billable"])
        self.assertIn("Automated export", time_entry["description"])


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


    def test_markdown_daily_page_includes_moneybird_export_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = MarkdownStorage(Path(tmp))
            storage.init()
            storage.upsert_work_items(
                [
                    WorkItem(101, 10, "Child", None, "Active", None, None, None, "u"),
                    WorkItem(10, None, "Parent", "mb:459275493454120239", "Active", None, None, None, "u"),
                ]
            )
            storage.add_entry(
                Entry("entry-1", "2026-04-01", 101, 5.0, None, None, "a", "a", 0)
            )

            page_path = Path(tmp) / "entries" / "2026" / "2026-04" / "2026-04-01.md"
            content = page_path.read_text(encoding="utf-8")

            self.assertIn("## Moneybird Export Preview", content)
            self.assertIn("459275493454120239", content)
            self.assertIn("2026-04-01T08:00:00Z", content)
            self.assertIn("2026-04-01T14:00:00Z", content)

    def test_rebuild_tables_restores_markdown_table_from_canonical_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = MarkdownStorage(Path(tmp))
            storage.init()
            entry = Entry(
                entry_id="entry-1",
                entry_date="2026-04-07",
                work_item_id=123,
                hours=1.5,
                note="Recovered from canonical data",
                category=None,
                created_at="2026-04-07T08:00:00",
                updated_at="2026-04-07T08:00:00",
                synced=0,
            )
            storage.add_entry(entry)
            page_path = Path(tmp) / "entries" / "2026" / "2026-04" / "2026-04-07.md"
            original = page_path.read_text(encoding="utf-8")
            corrupted = original.replace(
                "| entry-1 | 2026-04-07 | 123 | 1.50 | Recovered from canonical data |  | 2026-04-07T08:00:00 | 2026-04-07T08:00:00 | 0 |  |",
                "| broken | broken | broken | broken | broken | broken | broken | broken | broken | broken |",
            )
            page_path.write_text(corrupted, encoding="utf-8")

            rebuilt_days = storage.rebuild_tables_from_canonical_data()
            rebuilt = page_path.read_text(encoding="utf-8")

            self.assertEqual(rebuilt_days, 1)
            self.assertIn(
                "| entry-1 | 2026-04-07 | 123 | 1.50 | Recovered from canonical data |",
                rebuilt,
            )
            self.assertNotIn("| broken | broken |", rebuilt)


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

    def test_build_parser_includes_repair_markdown_tables_command(self) -> None:
        parser = build_parser()
        subparsers = parser._subparsers._group_actions[0].choices

        self.assertIn("repair", subparsers)
        repair_parser = subparsers["repair"]
        repair_choices = repair_parser._subparsers._group_actions[0].choices
        self.assertIn("markdown-tables", repair_choices)

    def test_repair_markdown_tables_command_requires_markdown_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "default",
                        "profiles": {
                            "default": {
                                "org_url": "",
                                "project": None,
                                "auth_mode": "pat",
                                "pat_env_var": "AZDO_PAT",
                                "remaining_work_strategy": "none",
                                "allow_sync_closed_items": False,
                                "max_hours_per_entry": 8,
                                "storage_backend": "sqlite",
                                "storage_path": str(Path(tmp) / "timesheet.sqlite"),
                                "wiql_query": None,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = repair_markdown_tables_command(
                SimpleNamespace(config=str(config_path), profile=None)
            )

            self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
