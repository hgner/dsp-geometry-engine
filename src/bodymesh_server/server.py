"""Local stdio entry point for the Blender Body-Mesh MCP.

This server is deliberately not exposed over HTTP: it reads private reference
photos and executes a local Blender installation. All logging stays on stderr.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from bodymesh_server import config
from bodymesh_server.tools import register


def create_server() -> FastMCP:
    mcp = FastMCP("Blender-BodyMesh")
    register(mcp)
    return mcp


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config.ensure_data_dirs()
    create_server().run()


if __name__ == "__main__":
    main()
