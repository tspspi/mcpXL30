"""Console entrypoints for mcpXL30."""

from __future__ import annotations

from mcpXL30.app.mcp_server import run_mcp_server
from mcpXL30.config.config_manager import generate_and_store_api_key, parse_arguments


def main():
    """Entry point for the mcpxl30 executable."""
    run_mcp_server()


def generate_api_key():
    """Entry point for mcpxl30-genkey (prints token to stdout)."""
    args = parse_arguments()
    new_key = generate_and_store_api_key(args.config)
    print(new_key)
