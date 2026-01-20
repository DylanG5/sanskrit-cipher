"""
Utility functions for the ML pipeline.

Provides logging setup, file handling, and other shared utilities.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import yaml


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Creates:
    - Console output with INFO level (optional)
    - File logging with DEBUG level for detailed debugging
    - Separate error log with ERROR level

    Args:
        config: Configuration dictionary with logging settings

    Returns:
        Configured logger instance
    """
    logging_config = config.get('logging', {})
    log_level = logging_config.get('level', 'INFO')
    log_file = logging_config.get('file', 'logs/pipeline.log')
    console_enabled = logging_config.get('console', True)

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('ml_pipeline')
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler for all logs
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Error file handler
    error_log_path = log_path.parent / 'errors.log'
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    # Console handler (optional)
    if console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(path: str, base_dir: Path) -> Path:
    """
    Resolve a path relative to a base directory.

    Args:
        path: Path string (may be relative or absolute)
        base_dir: Base directory to resolve relative paths from

    Returns:
        Absolute Path object
    """
    p = Path(path)
    if p.is_absolute():
        return p
    else:
        return (base_dir / p).resolve()
