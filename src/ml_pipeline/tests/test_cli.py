"""Tests for cli.py – argument parsing and command routing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ml_pipeline.cli as cli_mod


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------

def test_cmd_list_no_processors(capsys):
    """cmd_list prints message when no processors found."""
    args = MagicMock()
    args.config = None
    with patch.object(cli_mod, "ProcessorRegistry") as MockReg:
        inst = MockReg.return_value
        inst.list_all.return_value = []
        cli_mod.cmd_list(args)
    captured = capsys.readouterr()
    assert "No processors found" in captured.out


def test_cmd_list_with_processors(capsys):
    """cmd_list enumerates discovered processors."""
    args = MagicMock()
    args.config = None
    with (
        patch.object(cli_mod, "ProcessorRegistry") as MockReg,
        patch.object(cli_mod, "load_config", side_effect=FileNotFoundError),
    ):
        inst = MockReg.return_value
        inst.list_all.return_value = ["segmentation", "classification"]
        inst.get.return_value = MagicMock
        cli_mod.cmd_list(args)
    captured = capsys.readouterr()
    assert "segmentation" in captured.out
    assert "classification" in captured.out


def test_cmd_list_with_enabled_processors(capsys):
    """Cover cli.py lines 75–101: enabled processors printed with metadata."""
    args = MagicMock()
    args.config = "test_config.yaml"

    mock_processor_class = MagicMock()
    mock_proc_inst = MagicMock()
    mock_proc_inst.get_metadata.return_value = MagicMock(version="2.0", description="test desc")
    mock_processor_class.return_value = mock_proc_inst

    with (
        patch.object(cli_mod, "ProcessorRegistry") as MockReg,
        patch.object(cli_mod, "load_config", return_value={
            "processors": {
                "segmentation": {"enabled": True, "config": {}},
                "classification": {"enabled": False},
            }
        }),
    ):
        inst = MockReg.return_value
        inst.list_all.return_value = ["segmentation", "classification"]
        inst.get.return_value = mock_processor_class
        cli_mod.cmd_list(args)

    captured = capsys.readouterr()
    assert "ENABLED" in captured.out
    assert "disabled" in captured.out
    assert "v2.0" in captured.out


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------

def test_cmd_status_success(capsys, tmp_path):
    """cmd_status prints processing stats."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("database:\n  path: db.sqlite\n")

    args = MagicMock()
    args.config = str(cfg_path)

    fake_db = MagicMock()
    fake_db.get_processing_stats.return_value = {
        "total": 100,
        "completed": 60,
        "pending": 30,
        "failed": 10,
    }

    with (
        patch.object(cli_mod, "load_config", return_value={"database": {"path": "db.sqlite"}}),
        patch.object(cli_mod, "resolve_path", return_value=tmp_path / "db.sqlite"),
        patch.object(cli_mod, "DatabaseManager", return_value=fake_db),
        patch.object(cli_mod, "setup_logging"),
    ):
        cli_mod.cmd_status(args)

    out = capsys.readouterr().out
    assert "100" in out
    assert "60" in out


def test_cmd_status_missing_config(capsys):
    """cmd_status exits when config file is missing."""
    args = MagicMock()
    args.config = "/nonexistent/config.yaml"

    with pytest.raises(SystemExit):
        cli_mod.cmd_status(args)


def test_cmd_status_db_error(tmp_path, capsys):
    """Cover cli.py lines 139-141: except block in cmd_status."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("database:\n  path: db.sqlite\n")
    args = MagicMock()
    args.config = str(cfg_path)

    with (
        patch.object(cli_mod, "load_config", return_value={"database": {"path": "db.sqlite"}}),
        patch.object(cli_mod, "resolve_path", return_value=tmp_path / "db.sqlite"),
        patch.object(cli_mod, "DatabaseManager", side_effect=RuntimeError("db broken")),
        pytest.raises(SystemExit),
    ):
        cli_mod.cmd_status(args)


# ---------------------------------------------------------------------------
# cmd_run
# ---------------------------------------------------------------------------

def test_cmd_run_missing_config(capsys):
    """cmd_run exits when config file is missing."""
    args = MagicMock()
    args.config = "/nonexistent/config.yaml"
    args.processors = None
    args.resume = False
    args.dry_run = False
    args.limit = None
    args.fragment_ids = None
    args.force = False
    args.collection = None

    with pytest.raises(SystemExit):
        cli_mod.cmd_run(args)


def test_cmd_run_delegates_to_orchestrator(tmp_path):
    """cmd_run creates an orchestrator and calls run()."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("dummy: true\n")

    args = MagicMock()
    args.config = str(cfg_path)
    args.processors = "seg,cls"
    args.resume = True
    args.dry_run = True
    args.limit = 5
    args.fragment_ids = "F1,F2"
    args.force = True
    args.collection = "BLL"

    mock_orch = MagicMock()

    with patch.object(cli_mod, "PipelineOrchestrator", return_value=mock_orch):
        cli_mod.cmd_run(args)

    mock_orch.run.assert_called_once_with(
        processor_names=["seg", "cls"],
        resume=True,
        dry_run=True,
        limit=5,
        fragment_ids=["F1", "F2"],
        force=True,
        collection="BLL",
    )


# ---------------------------------------------------------------------------
# cmd_migrate
# ---------------------------------------------------------------------------

def test_cmd_migrate_missing_config(capsys):
    args = MagicMock()
    args.config = "/nonexistent/config.yaml"
    with pytest.raises(SystemExit):
        cli_mod.cmd_migrate(args)


def test_cmd_migrate_db_error(tmp_path, capsys):
    """Cover cli.py lines 168-170: except block in cmd_migrate."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("database:\n  path: db.sqlite\n")
    args = MagicMock()
    args.config = str(cfg_path)

    with (
        patch.object(cli_mod, "load_config", return_value={"database": {"path": "db.sqlite"}, "logging": {"console": False}}),
        patch.object(cli_mod, "setup_logging", return_value=MagicMock()),
        patch.object(cli_mod, "resolve_path", return_value=tmp_path / "db.sqlite"),
        patch.object(cli_mod, "DatabaseManager", side_effect=RuntimeError("migrate fail")),
        pytest.raises(SystemExit),
    ):
        cli_mod.cmd_migrate(args)


def test_cmd_migrate_success(tmp_path, capsys):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("database:\n  path: db.sqlite\nlogging:\n  console: false\n  file: logs/p.log\n")

    args = MagicMock()
    args.config = str(cfg_path)

    fake_db = MagicMock()
    with (
        patch.object(cli_mod, "load_config", return_value={"database": {"path": "db.sqlite"}, "logging": {"console": False}}),
        patch.object(cli_mod, "setup_logging", return_value=MagicMock()),
        patch.object(cli_mod, "resolve_path", return_value=tmp_path / "db.sqlite"),
        patch.object(cli_mod, "DatabaseManager", return_value=fake_db),
    ):
        cli_mod.cmd_migrate(args)

    fake_db.run_migration.assert_called_once()
    out = capsys.readouterr().out
    assert "Migration completed" in out


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

def test_main_no_command(capsys):
    with patch("sys.argv", ["cli"]):
        with pytest.raises(SystemExit):
            cli_mod.main()


def test_main_dispatches_list(capsys):
    with (
        patch("sys.argv", ["cli", "list"]),
        patch.object(cli_mod, "cmd_list") as mock_list,
    ):
        cli_mod.main()
        mock_list.assert_called_once()


def test_main_dispatches_status():
    with (
        patch("sys.argv", ["cli", "status"]),
        patch.object(cli_mod, "cmd_status") as mock_status,
    ):
        cli_mod.main()
        mock_status.assert_called_once()


def test_main_dispatches_migrate():
    with (
        patch("sys.argv", ["cli", "migrate"]),
        patch.object(cli_mod, "cmd_migrate") as mock_mig,
    ):
        cli_mod.main()
        mock_mig.assert_called_once()


def test_main_dispatches_run():
    with (
        patch("sys.argv", ["cli", "--config", "c.yaml", "run", "--limit", "5"]),
        patch.object(cli_mod, "cmd_run") as mock_run,
    ):
        cli_mod.main()
        mock_run.assert_called_once()
