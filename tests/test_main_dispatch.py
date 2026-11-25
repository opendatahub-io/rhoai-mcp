"""Tests for the __main__ run dispatch: config values reach MCPServer.run."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import rhoai_mcp.server as server_module
from rhoai_mcp.__main__ import main


class TestStreamableHttpDispatch:
    """The streamable-http dispatch passes transport settings from config to run()."""

    def test_stateless_http_true_reaches_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RHOAI_MCP_STATELESS_HTTP=true is forwarded as stateless_http=True."""
        monkeypatch.setenv("RHOAI_MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("RHOAI_MCP_STATELESS_HTTP", "true")
        monkeypatch.setenv("RHOAI_MCP_MOCK_CLUSTER", "true")
        monkeypatch.setattr(sys, "argv", ["rhoai-mcp"])

        fake_mcp = MagicMock()
        monkeypatch.setattr(server_module, "create_server", lambda _config: fake_mcp)

        exit_code = main()

        assert exit_code == 0
        fake_mcp.run.assert_called_once_with(
            transport="streamable-http",
            host="127.0.0.1",
            port=8000,
            stateless_http=True,
        )

    def test_stateless_http_default_false_reaches_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the env var, run() is called with stateless_http=False (stateful default)."""
        monkeypatch.setenv("RHOAI_MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("RHOAI_MCP_MOCK_CLUSTER", "true")
        monkeypatch.delenv("RHOAI_MCP_STATELESS_HTTP", raising=False)
        monkeypatch.setattr(sys, "argv", ["rhoai-mcp"])

        fake_mcp = MagicMock()
        monkeypatch.setattr(server_module, "create_server", lambda _config: fake_mcp)

        exit_code = main()

        assert exit_code == 0
        fake_mcp.run.assert_called_once_with(
            transport="streamable-http",
            host="127.0.0.1",
            port=8000,
            stateless_http=False,
        )

    def test_stateless_with_stdio_exits_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stateless_http with stdio transport is rejected before the server starts."""
        monkeypatch.setenv("RHOAI_MCP_TRANSPORT", "stdio")
        monkeypatch.setenv("RHOAI_MCP_STATELESS_HTTP", "true")
        monkeypatch.setenv("RHOAI_MCP_MOCK_CLUSTER", "true")
        monkeypatch.setattr(sys, "argv", ["rhoai-mcp"])

        fake_mcp = MagicMock()
        monkeypatch.setattr(server_module, "create_server", lambda _config: fake_mcp)

        exit_code = main()

        assert exit_code == 1
        fake_mcp.run.assert_not_called()
