"""``bodymesh-tool``: invoke one body-mesh tool without speaking MCP.

This is the subprocess/file-contract counterpart to the local-only MCP server:

    uv run --no-sync bodymesh-tool <tool-name> --args-json '<json-object>'

Successful stdout is the exact JSON string returned by the MCP implementation.
``BodyMeshError`` responses and CLI validation failures are emitted on stdout with
a nonzero exit code so callers can retain the structured failure payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any, NoReturn

from bodymesh_server import config
from bodymesh_server.schemas import BodyMeshError


def _dispatch() -> dict[str, Callable[..., str]]:
    """Return the same implementation functions registered by the MCP server."""
    from bodymesh_server.tools import (
        _create_body_mesh,
        _get_bodymesh_job,
        _inspect_bodymesh_runtime,
        _list_bodymesh_parameters,
        _prepare_bodymesh_job,
        _render_identity_set,
    )

    return {
        "inspect_bodymesh_runtime": _inspect_bodymesh_runtime,
        "list_bodymesh_parameters": _list_bodymesh_parameters,
        "prepare_bodymesh_job": _prepare_bodymesh_job,
        "create_body_mesh": _create_body_mesh,
        "get_bodymesh_job": _get_bodymesh_job,
        "render_identity_set": _render_identity_set,
    }


def _error(message: str, hint: str | None = None) -> str:
    return BodyMeshError(error=message, hint=hint).model_dump_json()


def _is_error(text: str) -> bool:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("error"), str)


def run_tool(tool: str, args: dict[str, Any]) -> tuple[str, int]:
    """Return ``(json_text, exit_code)`` for one registered body-mesh tool."""
    dispatch = _dispatch()
    fn = dispatch.get(tool)
    if fn is None:
        return (
            _error(f"unknown tool '{tool}'", f"available: {sorted(dispatch)}"),
            2,
        )

    try:
        config.ensure_data_dirs()
        text = fn(**args)
    except TypeError as exc:
        return _error(f"bad args for '{tool}': {exc}"), 2
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}"), 1

    if not isinstance(text, str):
        return _error(f"{tool} returned a non-JSON response"), 1
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return _error(f"{tool} returned invalid JSON: {exc}"), 1
    return text, 1 if _is_error(text) else 0


def _emit(text: str) -> None:
    sys.stdout.write(text + "\n")


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    parser = _JsonArgumentParser(
        prog="bodymesh-tool",
        description="Run one Blender body-mesh tool and print its JSON.",
    )
    parser.add_argument("tool", help="body-mesh tool name")
    parser.add_argument("--args-json", default="{}", help="JSON object of the tool's arguments")
    try:
        ns = parser.parse_args(argv)
    except ValueError as exc:
        _emit(_error(f"invalid command line: {exc}"))
        return 2

    try:
        args = json.loads(ns.args_json)
    except json.JSONDecodeError as exc:
        _emit(_error(f"--args-json is not valid JSON: {exc}"))
        return 2
    if not isinstance(args, dict):
        _emit(_error("--args-json must be a JSON object of the tool's arguments"))
        return 2

    text, code = run_tool(ns.tool, args)
    _emit(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
