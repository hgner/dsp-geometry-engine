"""HTTP-transport security tests: the bearer gate, the fail-closed no-token branch,
and the loopback default that re-arms the MCP SDK's DNS-rebinding protection.

No port is ever bound and no server process is started. ``_BearerAuthMiddleware`` is
pure ASGI, so it is driven directly with a stub downstream app over ``asyncio.run``
(the same idiom the other MCP-layer tests use), and the fail-closed decision lives in
``_resolve_http_auth`` so it can be asserted without reaching ``uvicorn.run``.
"""

from __future__ import annotations

import asyncio

import pytest

from dsp_server import config
from dsp_server.server import _BearerAuthMiddleware, _resolve_http_auth, create_server

TOKEN = "correct-horse-battery-staple"  # test fixture only — never a real credential


class _StubApp:
    """Downstream ASGI app: records the scopes that got past the gate, answers 200."""

    def __init__(self) -> None:
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope)
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})


def _drive(app, headers: list[tuple[bytes, bytes]], scope_type: str = "http") -> list[dict]:
    """Push one ASGI request through ``app`` and return the messages it sent."""
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": scope_type, "headers": headers, "method": "POST", "path": "/mcp"}
    asyncio.run(app(scope, receive, send))
    return sent


def _status(sent: list[dict]) -> int | None:
    return next((m["status"] for m in sent if m["type"] == "http.response.start"), None)


# --------------------------------------------------------------------------- #
# _BearerAuthMiddleware: who gets through


def test_missing_authorization_header_is_401():
    downstream = _StubApp()
    sent = _drive(_BearerAuthMiddleware(downstream, TOKEN), headers=[])
    assert _status(sent) == 401
    assert downstream.scopes == []  # the request never reached the tools


@pytest.mark.parametrize(
    "supplied",
    [
        b"Bearer wrong-token",
        b"Bearer ",
        b"Bearer " + TOKEN.encode() + b"x",  # prefix of the expected value is not enough
        TOKEN.encode(),  # raw token without the scheme
        b"bearer " + TOKEN.encode(),  # scheme is case-sensitive here (constant-time compare)
        b"Basic " + TOKEN.encode(),
    ],
)
def test_wrong_credential_is_401(supplied: bytes):
    downstream = _StubApp()
    sent = _drive(_BearerAuthMiddleware(downstream, TOKEN), headers=[(b"authorization", supplied)])
    assert _status(sent) == 401
    assert downstream.scopes == []


def test_correct_bearer_token_reaches_the_app():
    downstream = _StubApp()
    header = [(b"authorization", f"Bearer {TOKEN}".encode())]
    sent = _drive(_BearerAuthMiddleware(downstream, TOKEN), headers=header)
    assert _status(sent) != 401
    assert _status(sent) == 200
    assert len(downstream.scopes) == 1  # delegated exactly once


def test_lifespan_scope_bypasses_the_gate():
    # FastMCP's streamable-HTTP session manager starts in the lifespan scope; gating it
    # would 401 the server's own startup rather than a client.
    downstream = _StubApp()
    _drive(_BearerAuthMiddleware(downstream, TOKEN), headers=[], scope_type="lifespan")
    assert [s["type"] for s in downstream.scopes] == ["lifespan"]


# --------------------------------------------------------------------------- #
# config: empty token is unset, host defaults to loopback


def test_auth_token_treats_empty_string_as_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DSP_AUTH_TOKEN", "")
    assert config.auth_token() is None  # "" must not become an always-matching credential
    monkeypatch.delenv("DSP_AUTH_TOKEN", raising=False)
    assert config.auth_token() is None
    monkeypatch.setenv("DSP_AUTH_TOKEN", TOKEN)
    assert config.auth_token() == TOKEN


def test_http_host_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DSP_HOST", raising=False)
    assert config.http_host() == "127.0.0.1"  # never 0.0.0.0 by default
    monkeypatch.setenv("DSP_HOST", "0.0.0.0")
    assert config.http_host() == "0.0.0.0"  # containers still opt in explicitly


def test_loopback_default_arms_sdk_dns_rebinding_protection(stub_engine_env, monkeypatch: pytest.MonkeyPatch):
    # FastMCP auto-enables transport_security only for 127.0.0.1/localhost/::1, so the
    # loopback default buys DNS-rebinding protection for free — assert the actual setting.
    monkeypatch.delenv("DSP_HOST", raising=False)
    monkeypatch.setenv("DSP_TOOLSETS", "netqueue")  # smallest pack; this test is about settings
    security = create_server().settings.transport_security
    assert security is not None
    assert security.enable_dns_rebinding_protection is True

    monkeypatch.setenv("DSP_HOST", "0.0.0.0")
    assert create_server().settings.transport_security is None  # regression canary for the default


# --------------------------------------------------------------------------- #
# _resolve_http_auth: fail closed


def test_http_without_token_refuses_to_serve(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DSP_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DSP_ALLOW_INSECURE_HTTP", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _resolve_http_auth()
    message = str(excinfo.value)
    assert "DSP_AUTH_TOKEN" in message  # names the fix
    assert "DSP_ALLOW_INSECURE_HTTP" in message  # ...and the escape hatch


def test_empty_token_also_refuses_to_serve(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DSP_AUTH_TOKEN", "")  # the old compose.yaml default
    monkeypatch.delenv("DSP_ALLOW_INSECURE_HTTP", raising=False)
    with pytest.raises(SystemExit):
        _resolve_http_auth()


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_escape_hatch_only_opts_in_on_exactly_1(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.delenv("DSP_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("DSP_ALLOW_INSECURE_HTTP", value)
    assert config.allow_insecure_http() is False
    with pytest.raises(SystemExit):
        _resolve_http_auth()


def test_escape_hatch_serves_unauthenticated(monkeypatch: pytest.MonkeyPatch, caplog):
    monkeypatch.delenv("DSP_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("DSP_ALLOW_INSECURE_HTTP", "1")
    with caplog.at_level("WARNING", logger="dsp_server.server"):
        assert _resolve_http_auth() is None  # no token to enforce, and no SystemExit
    assert "NO authentication" in caplog.text  # the loud warning is part of the contract


def test_token_wins_over_the_escape_hatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DSP_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("DSP_ALLOW_INSECURE_HTTP", "1")
    assert _resolve_http_auth() == TOKEN  # a set token is always enforced
