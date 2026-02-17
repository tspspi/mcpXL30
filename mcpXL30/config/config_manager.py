"""Configuration loader and CLI helpers for mcpXL30."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

from mcpXL30.config.api_keys import (
    derive_argon2id_hash,
    ensure_kdf_defaults,
    ensure_kdf_salt,
    generate_random_api_key,
)
from mcpXL30.config.schema import Config, LoggingConfig


logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/mcpxl30/config.json")
_cached_config: Optional[Config] = None


def setup_logging(logging_config: LoggingConfig) -> None:
    """Install console/file logging using the supplied configuration."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if logging_config.logfile:
        try:
            log_dir = os.path.dirname(os.path.abspath(logging_config.logfile))
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(logging_config.logfile, mode="a")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info("Logging to file: %s", logging_config.logfile)
        except OSError as exc:
            logger.warning("Could not configure file logging at %s: %s", logging_config.logfile, exc)

    root_logger.setLevel(getattr(logging, logging_config.level.upper()))
    logger.info("Logging configured: level=%s file=%s", logging_config.level, logging_config.logfile or "stdout")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load, validate, and return the configuration document."""
    path = config_path or DEFAULT_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config_data = json.load(fh)
        logger.info("Loaded configuration from %s", path)
    except FileNotFoundError:
        logger.warning("Configuration file not found at %s. Using defaults.", path)
        config_data = {
            "instrument": {
                "port": "/dev/ttyUSB0",
                "log_level": "INFO",
                "retry_count": 3,
                "reconnect_count": 3,
            },
            "image_capture": {
                "remote_directory": r"C:\TEMP",
                "filename_prefix": "MCPIMG_",
            },
            "safety": {
                "max_high_tension_kv": 15.0,
                "allow_venting": False,
                "allow_pumping": True,
            },
            "logging": {
                "level": "INFO",
                "logfile": None,
            },
        }
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration file {path}: {exc}") from exc

    return Config(**config_data)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the MCP server."""
    parser = argparse.ArgumentParser(description="mcpXL30 FastMCP server")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override logging level",
    )
    parser.add_argument("--logfile", type=str, help="Path to log file (appends)")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "remotehttp"],
        help="Transport to expose (stdio for local MCP, remotehttp for FastAPI/uvicorn)",
    )
    parser.add_argument(
        "--genkey",
        action="store_true",
        help="Generate a new API key, store its hash in the config, then exit",
    )
    return parser.parse_args()


def get_config(parsed_args: Optional[argparse.Namespace] = None) -> Config:
    """Return a cached, validated config (respecting CLI overrides)."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    args = parsed_args or parse_arguments()
    config = load_config(args.config)

    if args.logfile:
        config.logging.logfile = args.logfile
    if args.log_level:
        config.logging.level = args.log_level.upper()

    setup_logging(config.logging)
    _cached_config = config
    return config


def create_default_config_file(path: str = DEFAULT_CONFIG_PATH) -> None:
    """Write a starter configuration file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    default_config = Config()  # relies on dataclass defaults
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(default_config.model_dump_json(indent=2))
        fh.write("\n")
    logger.info("Wrote default configuration to %s", path)


def generate_and_store_api_key(config_path: Optional[str] = None) -> str:
    """Generate an API key, store its Argon2id hash, and return the plain token."""
    path = config_path or DEFAULT_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config_data = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Configuration file not found at {path}. Provide --config or create one first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration file {path}: {exc}") from exc

    remote_block = config_data.get("remote_server")
    if not remote_block:
        raise ValueError("remote_server block must exist in configuration to generate an API key")

    kdf_block = ensure_kdf_defaults(remote_block.get("api_key_kdf"))
    ensure_kdf_salt(kdf_block)

    new_key = generate_random_api_key()
    kdf_block["hash"] = derive_argon2id_hash(new_key, kdf_block)
    remote_block["api_key_kdf"] = kdf_block
    remote_block.pop("api_key", None)
    config_data["remote_server"] = remote_block

    # Validate before persisting
    Config(**config_data)

    config_dir = os.path.dirname(path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config_data, fh, indent=2)
        fh.write("\n")

    logger.info("Updated API key hash in %s", path)
    return new_key
