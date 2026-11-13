"""MCP tools for discovering AutoML input contracts."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.automl.client import AutoMLClient

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def _deployment_response(deployment: Any) -> dict[str, Any]:
    return deployment.model_dump(mode="json", by_alias=True)


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register AutoML discovery and payload-validation tools."""

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
    def prepare_automl_prediction(
        name: str, namespace: str, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate JSON records and return the KServe v2 payload the predictor will receive.

        This tool validates the contract and prepares the KServe v2 request.
        """
        deployment = AutoMLClient(server.k8s).get_deployment(name, namespace)
        if deployment.input_schema is None:
            raise ValueError("Deployment has no AutoML MCP input schema annotation")
        if not deployment.is_ready:
            raise ValueError(f"Deployment '{name}' is not ready for predictions")
        return deployment.input_schema.records_to_v2_request(records)
