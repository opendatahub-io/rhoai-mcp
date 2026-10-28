"""MCP Resources for project-level information."""

from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer

from rhoai_mcp.domains.projects.client import ProjectClient

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_resources(mcp: MCPServer, server: "RHOAIServer") -> None:
    """Register project-level MCP resources."""

    @mcp.resource("rhoai://projects/{name}/status")
    def project_status(name: str) -> dict[str, Any]:
        """Get comprehensive status of a Data Science Project.

        Returns project metadata and resource summary including counts
        of workbenches, models, connections, and storage.
        """
        project_client = ProjectClient(server.k8s)
        project = project_client.get_project(name, include_summary=True)

        result: dict[str, Any] = {
            "name": project.metadata.name,
            "display_name": project.display_name,
            "description": project.description,
            "status": project.status.value,
            "is_modelmesh_enabled": project.is_modelmesh_enabled,
            "created": (
                project.metadata.creation_timestamp.isoformat()
                if project.metadata.creation_timestamp
                else None
            ),
        }

        if project.resource_summary:
            result["resources"] = {
                "workbenches": {
                    "total": project.resource_summary.workbenches,
                    "running": project.resource_summary.workbenches_running,
                },
                "models": {
                    "total": project.resource_summary.models,
                    "ready": project.resource_summary.models_ready,
                },
                "pipelines": project.resource_summary.pipelines,
                "data_connections": project.resource_summary.data_connections,
                "storage": project.resource_summary.storage,
            }

        return result

    @mcp.resource("rhoai://projects/{name}/workbenches")
    def project_workbenches(name: str) -> list[dict[str, Any]]:
        """Get list of workbenches in a Data Science Project.

        Returns all workbenches with their current status, image,
        and access URL.
        """
        try:
            from rhoai_mcp.domains.notebooks.client import NotebookClient

            notebook_client = NotebookClient(server.k8s)
            workbenches = notebook_client.list_workbenches(name)

            return [
                {
                    "name": wb.metadata.name,
                    "display_name": wb.display_name,
                    "status": wb.status.value,
                    "image": wb.image_display_name or wb.image,
                    "size": wb.size,
                    "url": wb.url,
                    "stopped_time": (wb.stopped_time.isoformat() if wb.stopped_time else None),
                    "created": (
                        wb.metadata.creation_timestamp.isoformat()
                        if wb.metadata.creation_timestamp
                        else None
                    ),
                }
                for wb in workbenches
            ]
        except ImportError:
            return [{"error": "Notebooks domain not available"}]

    @mcp.resource("rhoai://projects/{name}/models")
    def project_models(name: str) -> list[dict[str, Any]]:
        """Get list of deployed models in a Data Science Project.

        Returns all InferenceServices with their status and endpoints.
        """
        try:
            from rhoai_mcp.domains.inference.client import InferenceClient

            inference_client = InferenceClient(server.k8s)
            return inference_client.list_inference_services(name)
        except ImportError:
            return [{"error": "Inference domain not available"}]
