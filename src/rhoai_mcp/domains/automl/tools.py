"""MCP tools for discovering AutoML input contracts."""

from typing import TYPE_CHECKING, Any, cast

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.automl.client import AutoMLClient

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def _deployment_response(deployment: Any) -> dict[str, Any]:
    return cast(dict[str, Any], deployment.model_dump(mode="json", by_alias=True))


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register AutoML discovery and prediction tools."""

    @mcp.tool()
    def list_automl_deployments(namespace: str) -> list[dict[str, Any]]:
        """List explicitly exposed AutoGluon deployments in one project namespace."""
        return [
            _deployment_response(item)
            for item in AutoMLClient(server.k8s).list_deployments(namespace)
        ]

    @mcp.tool()
    def get_automl_deployment(name: str, namespace: str) -> dict[str, Any]:
        """Get the task type, input schema, example record, and status for a deployment."""
        return _deployment_response(AutoMLClient(server.k8s).get_deployment(name, namespace))

    @mcp.tool()
    async def predict_automl_deployment(
        name: str, namespace: str, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run JSON records through an exposed AutoGluon deployment.

        Returns numeric values for regression. For classification, returns a class and,
        when the ServingRuntime enables ``PREDICT_PROBA=true``, class probabilities.
        """
        return await AutoMLClient(server.k8s).predict_deployment(name, namespace, records)
