"""MCP server entry point: a thin loader over the toolset registry.

Transport is env-selected (``DSP_TRANSPORT``): ``stdio`` for Claude Code/Desktop,
``streamable-http`` for remote/cloud clients. With ``DSP_AUTH_TOKEN`` set on the
HTTP transport, a static bearer token is enforced by wrapping FastMCP's Starlette
app in a tiny ASGI middleware and serving it with uvicorn ourselves (FastMCP
1.28's own ``run(transport="streamable-http")`` uses the same uvicorn +
``streamable_http_app()`` pair internally, so behavior is identical minus the
401 gate). This is minimum-viable auth — front it with the platform's real auth
(ALB / API Gateway authorizer) in production.

All logging goes to stderr: stdout is MCP JSON-RPC and must stay clean.
"""

from __future__ import annotations

import hmac
import logging
import sys

from mcp.server.fastmcp import FastMCP
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from dsp_server import config
from dsp_server.toolsets import TOOLSETS, AppContext

logger = logging.getLogger(__name__)


def create_server() -> FastMCP:
    """Build the FastMCP server with every enabled toolset registered.

    ``DSP_TOOLSETS`` (comma list) selects packs; unset means all. Unknown names
    are skipped with a warning so one typo cannot take the whole server down.
    """
    mcp = FastMCP("DSP-Geometry-Engine", host=config.http_host(), port=config.http_port())
    ctx = AppContext.from_config()
    enabled = config.enabled_toolsets()
    names = list(TOOLSETS) if enabled is None else enabled
    for name in names:
        register = TOOLSETS.get(name)
        if register is None:
            logger.warning("unknown toolset %r skipped (available: %s)", name, ", ".join(TOOLSETS))
            continue
        register(mcp, ctx)
        logger.info("registered toolset %r", name)
    return mcp


class _BearerAuthMiddleware:
    """Pure ASGI wrapper: 401 unless ``Authorization: Bearer <token>`` matches.

    Non-http scopes (notably ``lifespan``, which starts FastMCP's streamable-HTTP
    session manager) pass through untouched.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            supplied = headers.get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self._expected):
                await Response("unauthorized", status_code=401)(scope, receive, send)
                return
        await self._app(scope, receive, send)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config.ensure_data_dirs()
    mcp = create_server()

    transport = config.transport()
    if transport == "stdio":
        mcp.run()
        return
    if transport != "streamable-http":
        raise SystemExit(f"unknown DSP_TRANSPORT {transport!r} (expected stdio|streamable-http)")

    token = config.auth_token()
    if token is None:
        logger.warning("streamable-http without DSP_AUTH_TOKEN — do not expose this port without real auth")
        mcp.run(transport="streamable-http")
        return

    import uvicorn  # ships with mcp[cli]; needed only on this branch

    app = _BearerAuthMiddleware(mcp.streamable_http_app(), token)
    logger.info("serving streamable-http with bearer auth on %s:%d", config.http_host(), config.http_port())
    uvicorn.run(app, host=config.http_host(), port=config.http_port(), log_level="info")


if __name__ == "__main__":
    main()
