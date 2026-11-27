"""MCP tools for discovering AutoML input contracts."""

from typing import TYPE_CHECKING, Any, cast

from mcp.server.mcpserver import Context, MCPServer

from rhoai_mcp.domains.automl.client import AutoMLClient
from rhoai_mcp.utils.errors import OperationNotAllowedError

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def _deployment_response(deployment: Any) -> dict[str, Any]:
    return cast(dict[str, Any], deployment.model_dump(mode="json", by_alias=True))


def register_tools(mcp: MCPServer, server: "RHOAIServer") -> None:
    """Register AutoML discovery and prediction tools."""

    @mcp.tool()
    def list_automl_deployments(namespace: str, ctx: Context | None = None) -> list[dict[str, Any]]:
        """List explicitly exposed AutoGluon deployments in one project namespace."""
        return [
            _deployment_response(item)
            for item in AutoMLClient(server.get_k8s(ctx)).list_deployments(namespace)
        ]

    @mcp.tool()
    def get_automl_deployment(
        name: str, namespace: str, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Get the task type, input schema, example record, and status for a deployment."""
        return _deployment_response(
            AutoMLClient(server.get_k8s(ctx)).get_deployment(name, namespace)
        )

    @mcp.tool()
    async def predict_automl_deployment(
        name: str,
        namespace: str,
        records: list[dict[str, Any]],
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Run JSON records through an exposed AutoGluon deployment.

        Returns numeric values for regression. For classification, returns a class and,
        when the ServingRuntime enables ``PREDICT_PROBA=true``, class probabilities.
        """
        if server.config.read_only_mode:
            raise OperationNotAllowedError("prediction", reason="read-only mode is enabled")
        return await AutoMLClient(server.get_k8s(ctx)).predict_deployment(name, namespace, records)
