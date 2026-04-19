"""MCP Tools for Quickstart operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.quickstarts.client import QuickstartClient

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register quickstart tools with the MCP server."""

    @mcp.tool()
    def list_available_quickstarts() -> dict[str, Any]:
        """List all available Red Hat AI Quickstarts.

        Returns a list of supported quickstarts with their names, descriptions,
        and repository URLs. These quickstarts provide ready-to-deploy AI/ML
        applications and patterns for OpenShift AI.

        Returns:
            JSON object with list of quickstarts and their metadata.
        """
        client = QuickstartClient()
        quickstarts = client.list_quickstarts()

        return {
            "quickstarts": [
                {
                    "name": qs.name,
                    "display_name": qs.display_name,
                    "description": qs.description,
                    "repo_url": qs.repo_url,
                    "tags": qs.tags,
                }
                for qs in quickstarts
            ],
            "total": len(quickstarts),
            "_source": {
                "kind": "QuickstartRegistry",
                "api_version": "rhoai-mcp/v1",
                "name": "rh-ai-quickstarts",
                "namespace": None,
                "uid": None,
            },
        }

    @mcp.tool()
    def get_quickstart_readme(quickstart_name: str) -> dict[str, Any]:
        """Get the README.md content for a specific quickstart.

        Use this to understand a quickstart's requirements, architecture,
        and deployment instructions before deploying it.

        Args:
            quickstart_name: Name of the quickstart (e.g., "llm-cpu-serving",
                "rag-chatbot", "product-recommender", "lemonade-stand").

        Returns:
            JSON object with README content and metadata.
        """
        client = QuickstartClient()

        try:
            readme = client.get_readme(quickstart_name)
            return {
                "quickstart_name": readme.quickstart_name,
                "repo_url": readme.repo_url,
                "content": readme.content,
                "_source": {
                    "kind": "QuickstartReadme",
                    "api_version": "rhoai-mcp/v1",
                    "name": readme.quickstart_name,
                    "namespace": None,
                    "uid": None,
                },
            }
        except ValueError as e:
            return {
                "error": str(e),
                "quickstart_name": quickstart_name,
                "_source": {
                    "kind": "QuickstartReadme",
                    "api_version": "rhoai-mcp/v1",
                    "name": quickstart_name,
                    "namespace": None,
                    "uid": None,
                },
            }

    @mcp.tool()
    def deploy_quickstart(
        quickstart_name: str,
        target_namespace: str = "rhoai-quickstarts",
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Deploy a Red Hat AI Quickstart to the cluster.

        Automatically detects the deployment method (Helm, Kustomize, or raw
        manifests) and deploys the quickstart to the specified namespace.

        IMPORTANT: By default, this runs in dry-run mode and only shows
        the command that would be executed. Set dry_run=False to actually deploy.

        Args:
            quickstart_name: Name of the quickstart to deploy (e.g., "llm-cpu-serving").
            target_namespace: Kubernetes namespace for deployment (default: "rhoai-quickstarts").
            dry_run: If True (default), only show the command without executing.
                Set to False to actually deploy to the cluster.

        Returns:
            JSON object with deployment results including:
            - method: Detected deployment method (helm/kustomize/manifests)
            - command: The deployment command
            - success: Whether deployment succeeded
            - stdout/stderr: Command output (if executed)
        """
        if not dry_run:
            allowed, reason = server.config.is_operation_allowed("create")
            if not allowed:
                return {
                    "error": reason,
                    "quickstart_name": quickstart_name,
                    "namespace": target_namespace,
                    "success": False,
                    "_source": {
                        "kind": "QuickstartDeployment",
                        "api_version": "rhoai-mcp/v1",
                        "name": quickstart_name,
                        "namespace": target_namespace,
                        "uid": None,
                    },
                }

        client = QuickstartClient()
        result = client.deploy(
            quickstart_name=quickstart_name,
            namespace=target_namespace,
            dry_run=dry_run,
        )

        response: dict[str, Any] = {
            "quickstart_name": result.quickstart_name,
            "namespace": result.namespace,
            "method": result.method.value,
            "command": result.command,
            "dry_run": result.dry_run,
            "success": result.success,
            "_source": {
                "kind": "QuickstartDeployment",
                "api_version": "rhoai-mcp/v1",
                "name": result.quickstart_name,
                "namespace": result.namespace,
                "uid": None,
            },
        }

        if result.stdout:
            response["stdout"] = result.stdout
        if result.stderr:
            response["stderr"] = result.stderr
        if result.error:
            response["error"] = result.error

        return response


def register_resources(mcp: FastMCP, server: "RHOAIServer") -> None:  # noqa: ARG001
    """Register quickstart resources with the MCP server."""

    @mcp.resource("rhoai://quickstarts/registry")
    def quickstarts_registry() -> dict[str, Any]:
        """Get the full quickstart registry metadata.

        Returns all registered quickstarts with their complete metadata
        for use as a reference resource.
        """
        client = QuickstartClient()
        quickstarts = client.list_quickstarts()

        return {
            "registry": "rh-ai-quickstarts",
            "quickstarts": {
                qs.name: {
                    "display_name": qs.display_name,
                    "description": qs.description,
                    "repo_url": qs.repo_url,
                    "tags": qs.tags,
                }
                for qs in quickstarts
            },
            "total": len(quickstarts),
        }
