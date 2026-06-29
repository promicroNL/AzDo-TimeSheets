import gc
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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
    moneybird_export_command,
    moneybird_project_id_from_tags,
    plan_moneybird_time_entries,
    parse_work_item,
    repair_markdown_tables_command,
    summarize_by_parent,
    truncate_note,
)
from azdo_timesheet.hours import round_report_hours, sum_report_hours
from azdo_timesheet.models import Config, Entry, WorkItem, WorkItemDelta, WorkItemState
from azdo_timesheet.storage import MarkdownStorage, SQLiteStorage


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


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

    def test_format_entries_rounds_report_hours_before_summing(self) -> None:
        entries = [
            Entry("1", "2026-04-01", 1, 0.45, None, None, "a", "a", 0),
            Entry("2", "2026-04-01", 1, 0.25, None, None, "b", "b", 0),
        ]

        output = format_entries(entries)

        self.assertIn("0.75", output)
        self.assertNotIn("0.70", output)

    def test_report_hours_round_half_up_to_nearest_quarter(self) -> None:
        self.assertEqual(round_report_hours(0.45), 0.5)
        self.assertEqual(round_report_hours(0.125), 0.25)
        self.assertEqual(sum_report_hours([0.45, 0.25]), 0.75)

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

    def test_format_parent_summary_rounds_report_hours_before_summing(self) -> None:
        entries = [
            Entry("1", "2026-04-01", 101, 0.45, None, None, "a", "a", 0),
            Entry("2", "2026-04-01", 101, 0.25, None, None, "b", "b", 0),
        ]
        work_items = [WorkItem(101, 10, "Child", None, "Active", None, None, None, "u")]

        output = format_parent_summary(entries, work_items)

        self.assertIn("0.75", output)
        self.assertNotIn("0.70", output)

    def test_summarize_by_parent_preserves_exact_hours_by_default(self) -> None:
        entries = [
            Entry("1", "2026-04-01", 101, 0.45, None, None, "a", "a", 0),
            Entry("2", "2026-04-01", 101, 0.25, None, None, "b", "b", 0),
        ]

        rows = summarize_by_parent(entries, {})

        self.assertEqual(rows[0][:2], (None, "(no parent)"))
        self.assertAlmostEqual(rows[0][2], 0.7)


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
            moneybird_user_id="987654321",
            moneybird_contact_id="456789123",
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
        self.assertEqual(time_entry["user_id"], "987654321")
        self.assertEqual(time_entry["contact_id"], "456789123")
        self.assertEqual(time_entry["started_at"], "2026-04-01T08:00:00Z")
        self.assertEqual(time_entry["ended_at"], "2026-04-01T14:00:00Z")
        self.assertFalse(time_entry["billable"])
        self.assertIn("Automated export", time_entry["description"])

    def test_plan_moneybird_time_entries_uses_report_hour_totals(self) -> None:
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
            moneybird_user_id="987654321",
            moneybird_contact_id=None,
            moneybird_token_env_var="MONEYBIRD_TOKEN",
            moneybird_start_time="08:00",
            moneybird_lunch_break_start="12:00",
            moneybird_lunch_break_end="13:00",
            moneybird_dinner_break_start="18:00",
            moneybird_dinner_break_end="19:30",
            moneybird_timezone="Z",
            moneybird_billable=None,
        )
        entries = [
            Entry("1", "2026-04-01", 101, 0.45, None, None, "a", "a", 0),
            Entry("2", "2026-04-01", 101, 0.25, None, None, "b", "b", 0),
        ]
        work_items = [
            WorkItem(101, 10, "Child", None, "Active", None, None, None, "u"),
            WorkItem(10, None, "Parent", "mb:459275493454120239", "Active", None, None, None, "u"),
        ]

        planned = plan_moneybird_time_entries(entries, work_items, config)

        self.assertEqual(planned[0]["hours"], 0.75)
        self.assertEqual(planned[0]["time_entry"]["ended_at"], "2026-04-01T08:45:00Z")

    def test_plan_moneybird_time_entries_schedules_day_totals_back_to_back(self) -> None:
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
            moneybird_user_id="987654321",
            moneybird_contact_id=None,
            moneybird_token_env_var="MONEYBIRD_TOKEN",
            moneybird_start_time="08:00",
            moneybird_lunch_break_start="12:00",
            moneybird_lunch_break_end="13:00",
            moneybird_dinner_break_start="18:00",
            moneybird_dinner_break_end="19:30",
            moneybird_timezone="Z",
            moneybird_billable=None,
        )
        entries = [
            Entry("1", "2026-06-22", 101, 1.0, None, None, "a", "a", 0),
            Entry("2", "2026-06-22", 102, 5.0, None, None, "b", "b", 0),
        ]
        work_items = [
            WorkItem(101, 6390, "Child 1", None, "Active", None, None, None, "u"),
            WorkItem(102, 6391, "Child 2", None, "Active", None, None, None, "u"),
            WorkItem(6390, None, "Parent 1", "mb:475528629432878107", "Active", None, None, None, "u"),
            WorkItem(6391, None, "Parent 2", "mb:459275493454120239", "Active", None, None, None, "u"),
        ]

        planned = plan_moneybird_time_entries(entries, work_items, config)

        self.assertEqual(len(planned), 2)
        first = planned[0]["time_entry"]
        second = planned[1]["time_entry"]
        self.assertEqual(first["started_at"], "2026-06-22T08:00:00Z")
        self.assertEqual(first["ended_at"], "2026-06-22T09:00:00Z")
        self.assertEqual(second["started_at"], "2026-06-22T09:00:00Z")
        self.assertEqual(second["ended_at"], "2026-06-22T15:00:00Z")
        self.assertNotIn("paused_duration", first)
        self.assertEqual(second["paused_duration"], 3600)

    def test_moneybird_export_dry_run_prints_back_to_back_break_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "timesheet.sqlite"
            config_path = Path(tmp) / "config.json"
            storage = SQLiteStorage(db_path)
            storage.init()
            storage.upsert_work_items(
                [
                    WorkItem(101, 6390, "Child 1", None, "Active", None, None, None, "u"),
                    WorkItem(102, 6391, "Child 2", None, "Active", None, None, None, "u"),
                    WorkItem(6390, None, "Parent 1", "mb:475528629432878107", "Active", None, None, None, "u"),
                    WorkItem(6391, None, "Parent 2", "mb:459275493454120239", "Active", None, None, None, "u"),
                ]
            )
            storage.add_entry(
                Entry("1", "2026-06-22", 101, 1.0, None, None, "a", "a", 0)
            )
            storage.add_entry(
                Entry("2", "2026-06-22", 102, 5.0, None, None, "b", "b", 0)
            )
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "default",
                        "profiles": {
                            "default": {
                                "storage_backend": "sqlite",
                                "storage_path": str(db_path),
                                "moneybird_administration_id": "123",
                                "moneybird_user_id": "419608190908368752",
                                "moneybird_contact_id": "419608190908368753",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = moneybird_export_command(
                    SimpleNamespace(
                        config=str(config_path),
                        profile=None,
                        week="2026-06-22",
                        start=None,
                        end=None,
                        apply=False,
                    )
                )

            gc.collect()
            self.assertEqual(result, 0)
            output = stdout.getvalue()
            self.assertIn(
                "2026-06-22 parent WI #6390: 1.00h -> Moneybird project "
                "475528629432878107 user 419608190908368752 "
                "contact 419608190908368753 "
                "(2026-06-22T08:00:00Z - 2026-06-22T09:00:00Z)",
                output,
            )
            self.assertIn(
                "2026-06-22 parent WI #6391: 5.00h -> Moneybird project "
                "459275493454120239 user 419608190908368752 "
                "contact 419608190908368753 "
                "(2026-06-22T09:00:00Z - 2026-06-22T15:00:00Z, "
                "including 60 minutes break)",
                output,
            )

    def test_moneybird_export_starttime_overrides_config_start_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "timesheet.sqlite"
            config_path = Path(tmp) / "config.json"
            storage = SQLiteStorage(db_path)
            storage.init()
            storage.upsert_work_items(
                [
                    WorkItem(101, 6390, "Child 1", None, "Active", None, None, None, "u"),
                    WorkItem(6390, None, "Parent 1", "mb:475528629432878107", "Active", None, None, None, "u"),
                ]
            )
            storage.add_entry(
                Entry("1", "2026-06-22", 101, 1.0, None, None, "a", "a", 0)
            )
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "default",
                        "profiles": {
                            "default": {
                                "storage_backend": "sqlite",
                                "storage_path": str(db_path),
                                "moneybird_administration_id": "123",
                                "moneybird_user_id": "419608190908368752",
                                "moneybird_start_time": "08:00",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = moneybird_export_command(
                    SimpleNamespace(
                        config=str(config_path),
                        profile=None,
                        week="2026-06-22",
                        start=None,
                        end=None,
                        starttime="13:00",
                        apply=False,
                    )
                )

            gc.collect()
            self.assertEqual(result, 0)
            self.assertIn(
                "(2026-06-22T13:00:00Z - 2026-06-22T14:00:00Z)",
                stdout.getvalue(),
            )

    def test_moneybird_export_prompts_and_applies_from_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "timesheet.sqlite"
            config_path = Path(tmp) / "config.json"
            storage = SQLiteStorage(db_path)
            storage.init()
            storage.upsert_work_items(
                [
                    WorkItem(101, 6390, "Child 1", None, "Active", None, None, None, "u"),
                    WorkItem(6390, None, "Parent 1", "mb:475528629432878107", "Active", None, None, None, "u"),
                ]
            )
            storage.add_entry(
                Entry("synced", "2026-06-22", 101, 1.0, None, None, "a", "a", 1)
            )
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "default",
                        "profiles": {
                            "default": {
                                "storage_backend": "sqlite",
                                "storage_path": str(db_path),
                                "moneybird_administration_id": "123",
                                "moneybird_user_id": "419608190908368752",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with patch(
                "azdo_timesheet.cli.moneybird_request",
                return_value={"id": "time-entry-1"},
            ) as request_mock:
                with patch("azdo_timesheet.cli.sys.stdin", TtyStringIO("y\n")):
                    with redirect_stdout(stdout):
                        result = moneybird_export_command(
                            SimpleNamespace(
                                config=str(config_path),
                                profile=None,
                                week="2026-06-22",
                                start=None,
                                end=None,
                                apply=False,
                            )
                        )

            gc.collect()
            self.assertEqual(result, 0)
            request_mock.assert_called_once()
            output = stdout.getvalue()
            self.assertIn("Apply these Moneybird registrations now? [y/N]:", output)
            self.assertIn("Created Moneybird time entry time-entry-1.", output)
            self.assertIn("Created 1 Moneybird time registration(s).", output)

    def test_moneybird_apply_requires_whole_day_synced_to_azdo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "timesheet.sqlite"
            config_path = Path(tmp) / "config.json"
            storage = SQLiteStorage(db_path)
            storage.init()
            storage.upsert_work_items(
                [
                    WorkItem(101, 6390, "Child 1", None, "Active", None, None, None, "u"),
                    WorkItem(102, 6391, "Child 2", None, "Active", None, None, None, "u"),
                    WorkItem(6390, None, "Parent 1", "mb:475528629432878107", "Active", None, None, None, "u"),
                    WorkItem(6391, None, "Parent 2", "mb:459275493454120239", "Active", None, None, None, "u"),
                ]
            )
            storage.add_entry(
                Entry("synced", "2026-06-22", 101, 1.0, None, None, "a", "a", 1)
            )
            storage.add_entry(
                Entry("unsynced", "2026-06-22", 102, 5.0, None, None, "b", "b", 0)
            )
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "default",
                        "profiles": {
                            "default": {
                                "storage_backend": "sqlite",
                                "storage_path": str(db_path),
                                "moneybird_administration_id": "123",
                                "moneybird_user_id": "419608190908368752",
                                "moneybird_contact_id": "419608190908368753",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("azdo_timesheet.cli.moneybird_request") as request_mock:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = moneybird_export_command(
                        SimpleNamespace(
                            config=str(config_path),
                            profile=None,
                            week="2026-06-22",
                            start=None,
                            end=None,
                            apply=True,
                        )
                    )

            gc.collect()
            self.assertEqual(result, 2)
            request_mock.assert_not_called()
            self.assertIn(
                "Moneybird apply requires every entry in each exported day",
                stderr.getvalue(),
            )
            self.assertIn("2026-06-22: 1 unsynced entry (unsynced)", stderr.getvalue())


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
                    WorkItem(101, 10, "Child 1", None, "Active", None, None, None, "u"),
                    WorkItem(102, 11, "Child 2", None, "Active", None, None, None, "u"),
                    WorkItem(10, None, "Parent 1", "mb:475528629432878107", "Active", None, None, None, "u"),
                    WorkItem(11, None, "Parent 2", "mb:459275493454120239", "Active", None, None, None, "u"),
                ]
            )
            storage.add_entry(
                Entry("entry-1", "2026-04-01", 101, 1.0, None, None, "a", "a", 0)
            )
            storage.add_entry(
                Entry("entry-2", "2026-04-01", 102, 5.0, None, None, "b", "b", 0)
            )

            page_path = Path(tmp) / "entries" / "2026" / "2026-04" / "2026-04-01.md"
            content = page_path.read_text(encoding="utf-8")

            self.assertIn("## Moneybird Export Preview", content)
            self.assertIn("459275493454120239", content)
            self.assertIn("2026-04-01T08:00:00Z", content)
            self.assertIn("2026-04-01T09:00:00Z", content)
            self.assertIn("2026-04-01T15:00:00Z", content)
            self.assertIn("| Parent Work Item ID | Moneybird Project ID | Total Hours | Started At | Ended At | Break Minutes | Export Description |", content)
            self.assertIn("| 11 | 459275493454120239 | 5.00 | 2026-04-01T09:00:00Z | 2026-04-01T15:00:00Z | 60 |", content)
            self.assertNotIn("2026-04-01T14:00:00Z", content)

    def test_markdown_pages_use_report_hours_but_keep_canonical_hours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = MarkdownStorage(Path(tmp))
            storage.init()
            storage.add_entry(
                Entry("entry-1", "2026-04-01", 101, 0.45, None, None, "a", "a", 0)
            )
            storage.add_entry(
                Entry("entry-2", "2026-04-01", 101, 0.25, None, None, "b", "b", 0)
            )

            page_path = Path(tmp) / "entries" / "2026" / "2026-04" / "2026-04-01.md"
            day_content = page_path.read_text(encoding="utf-8")
            month_content = (Path(tmp) / "entries" / "2026" / "2026-04.md").read_text(
                encoding="utf-8"
            )

            self.assertIn("| entry-1 | 2026-04-01 | 101 | 0.50 |", day_content)
            self.assertIn("**Grand Total:** 0.75 hours", day_content)
            self.assertIn("**Grand Total:** 0.75 hours", month_content)
            self.assertIn('"hours": 0.45', day_content)
            self.assertNotIn("**Grand Total:** 0.70 hours", day_content)

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

    def test_build_parser_includes_moneybird_config_help(self) -> None:
        parser = build_parser()
        subparsers = parser._subparsers._group_actions[0].choices

        self.assertIn("--moneybird-user-id", subparsers["init"].format_help())
        self.assertIn("--moneybird-contact-id", subparsers["init"].format_help())
        moneybird_parser = subparsers["moneybird"]
        moneybird_choices = moneybird_parser._subparsers._group_actions[0].choices
        self.assertIn("moneybird_user_id", moneybird_choices["export"].format_help())
        self.assertIn("moneybird_contact_id", moneybird_choices["export"].format_help())
        self.assertIn("--starttime", moneybird_choices["export"].format_help())
        self.assertIn("synced to Azure DevOps", moneybird_choices["export"].format_help())

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
