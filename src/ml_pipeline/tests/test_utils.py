from __future__ import annotations

from pathlib import Path

import pytest

from ml_pipeline.core.utils import load_config, resolve_path, setup_logging


def test_resolve_path_relative(tmp_path: Path) -> None:
    """test-ut-utils-1"""
    base_dir = tmp_path
    rel_path = "data/file.txt"
    resolved = resolve_path(rel_path, base_dir)
    assert resolved == (base_dir / rel_path).resolve()


def test_resolve_path_absolute(tmp_path: Path) -> None:
    """test-ut-utils-1"""
    abs_path = tmp_path / "abs.txt"
    resolved = resolve_path(str(abs_path), tmp_path)
    assert resolved == abs_path


def test_load_config_missing(tmp_path: Path) -> None:
    """test-ut-utils-2"""
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))


def test_load_config_success(tmp_path: Path) -> None:
    """test-ut-utils-3"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  path: db.sqlite\n"
        "logging:\n"
        "  console: false\n",
        encoding="utf-8",
    )
    config = load_config(str(config_path))
    assert config["database"]["path"] == "db.sqlite"


def test_setup_logging_creates_log_files(tmp_path: Path) -> None:
    """test-ut-utils-4"""
    log_file = tmp_path / "logs" / "pipeline.log"
    config = {"logging": {"file": str(log_file), "console": False}}

    logger = setup_logging(config)
    logger.info("test log entry")

    assert log_file.exists()
    assert (log_file.parent / "errors.log").exists()


def test_setup_logging_with_console_enabled(tmp_path: Path) -> None:
    """Cover utils.py lines 63-67 – console handler branch."""
    log_file = tmp_path / "logs" / "pipeline.log"
    config = {"logging": {"file": str(log_file), "console": True}}

    logger = setup_logging(config)
    # Should have at least 3 handlers: file, error-file, console
    assert len(logger.handlers) >= 3
    logger.info("console test")
