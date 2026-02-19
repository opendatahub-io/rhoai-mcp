"""MCP server lifecycle management for evaluations.

Runs the real RHOAI MCP server in-process, allowing mock injection
at the K8s client level while all domain logic, plugin loading,
and tool registration execute for real.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from rhoai_mcp.config import RHOAIConfig
from rhoai_mcp.server import RHOAIServer

from evals.config import ClusterMode, EvalConfig

logger = logging.getLogger(__name__)


class MCPHarness:
    """Wraps the RHOAI MCP server for evaluation use.

    Provides direct tool calling, tool listing, and lifecycle management.
    When cluster_mode is 'mock', injects a MockK8sClient before the
    server lifespan starts.
    """

    def __init__(self, server: RHOAIServer, eval_config: EvalConfig) -> None:
        self._server = server
        self._eval_config = eval_config

    @property
    def server(self) -> RHOAIServer:
        """Get the underlying RHOAI server."""
        return self._server

    @staticmethod
    @asynccontextmanager
    async def running(eval_config: EvalConfig) -> AsyncIterator[MCPHarness]:
        """Start the MCP server and yield a harness for calling tools.

        In mock mode, injects a MockK8sClient *before* create_mcp() so
        that startup() sees the pre-connected client and skips real K8s.
        """
        rhoai_config = RHOAIConfig(read_only_mode=True)
        server = RHOAIServer(config=rhoai_config)

        if eval_config.cluster_mode == ClusterMode.MOCK:
            from evals.mock_k8s.cluster_state import create_default_cluster_state
            from evals.mock_k8s.mock_client import MockK8sClient

            # Inject BEFORE create_mcp() so startup() skips real K8s
            state = create_default_cluster_state()
            mock_client = MockK8sClient(config_obj=rhoai_config, state=state)
            mock_client.connect()
            server._k8s_client = mock_client

            try:
                server.create_mcp()
                harness = MCPHarness(server, eval_config)
                yield harness
            finally:
                mock_client.disconnect()
                server._k8s_client = None
        else:
            # Live mode: create_mcp() connects to real K8s via startup()
            try:
                server.create_mcp()
                yield MCPHarness(server, eval_config)
            finally:
                server.shutdown()

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered MCP tools with their schemas.

        Returns a list of dicts with 'name', 'description', and 'parameters'.
        """
        tools = self._server.mcp._tool_manager.list_tools()
        result = []
        for tool in tools:
            tool_info: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            }
            result.append(tool_info)
        return result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool by name and return the result as a string.

        Args:
            name: Tool name.
            arguments: Tool arguments as a dict.

        Returns:
            String representation of the tool result.
        """
        try:
            result = await self._server.mcp.call_tool(name, arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})
