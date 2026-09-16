"""MCP tools for quickstart discovery and deployment.

These tools let an agent (1) discover quickstarts from the registry, (2) read a
quickstart's manifest to learn its parameters, (3) run a deployment action by
creating the installer Job, and (4)/(5) poll that Job's status and logs.

Recommendation itself is left to the agent: it reasons over the registry and
manifest returned here, combined with cluster-resource facts from other RHOAI
MCP tools. This domain only supplies catalog data and mechanical execution.
"""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.quickstarts.client import DESTRUCTIVE_ACTIONS, QuickstartsClient
from rhoai_mcp.utils.errors import RHOAIError
from rhoai_mcp.utils.response import PaginatedResponse, paginate

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register quickstart tools with the MCP server."""

    def _client() -> QuickstartsClient:
        return QuickstartsClient(server.k8s, server.config)

    @mcp.tool()
    def list_quickstarts(
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List available quickstarts from the Red Hat AI quickstart registry.

        Reads the registry index (an OCI artifact on Quay). Each entry includes
        tags, industries and a short description an agent can use to recommend a
        quickstart for a user's goal. Use get_quickstart_manifest next to learn
        a specific quickstart's prerequisites and parameters.

        Args:
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).

        Returns:
            Paginated list of quickstart catalog entries.
        """
        try:
            summaries = _client().get_registry().quickstarts
        except RHOAIError as e:
            return {"error": str(e)}

        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        paginated, total = paginate(summaries, offset, effective_limit)
        items = [s.to_dict() for s in paginated]
        return PaginatedResponse.build(items, total, offset, effective_limit)

    @mcp.tool()
    def get_quickstart_manifest(
        name: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Get a quickstart's full manifest, including its parameters.

        Returns the deployment configuration (supported actions/modes, default
        namespace), the full parameter list (with per-parameter llm_guidance on
        how to elicit values from the user), and the rich prerequisites,
        classification and llm_context sections for recommendation reasoning.

        Args:
            name: The quickstart name (from list_quickstarts).
            version: Specific version; defaults to the registry's latest.

        Returns:
            The quickstart manifest details.
        """
        try:
            manifest = _client().get_manifest(name, version)
        except RHOAIError as e:
            return {"error": str(e)}

        params = manifest.parameters
        return {
            "name": manifest.name or name,
            "display_name": manifest.display_name,
            "version": manifest.version,
            "short_description": manifest.metadata.get("shortDescription"),
            "long_description": manifest.metadata.get("longDescription"),
            "supported_actions": manifest.deployment.supported_actions,
            "supported_modes": manifest.deployment.supported_modes,
            "default_namespace": manifest.deployment.default_namespace,
            "installer_image": manifest.deployment.installer.image,
            "parameters": {
                "secrets": [p.to_dict(is_secret=True) for p in params.secrets],
                "configuration": [p.to_dict(is_secret=False) for p in params.configuration],
            },
            "prerequisites": manifest.prerequisites,
            "classification": manifest.classification,
            "llm_context": manifest.llm_context,
        }

    @mcp.tool()
    def run_quickstart_action(
        name: str,
        action: str,
        target_namespace: str | None = None,
        mode: str = "demo",
        parameters: dict[str, Any] | None = None,
        version: str | None = None,
        source_version: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Run a quickstart deployment action by creating the installer Job.

        Mechanically creates a Kubernetes Job that runs the quickstart's
        installer image with the action, namespace, mode and parameter values.
        Secret parameters are written to a Secret and injected via secretKeyRef.
        Only call this when the user has explicitly asked to run the action, and
        after eliciting required parameter values (see get_quickstart_manifest).

        Args:
            name: The quickstart name.
            action: One of the manifest's supportedActions (e.g. INSTALL,
                CHECK_PRE_REQS, STATUS, UNINSTALL_KEEP_DATA, UNINSTALL_DELETE_ALL).
            target_namespace: Namespace to deploy into; defaults to the
                manifest's defaultNamespace.
            mode: Install mode (e.g. "demo"); validated against supportedModes.
            parameters: Mapping of parameter name -> value (names as declared in
                the manifest's parameters block).
            version: Quickstart version; defaults to the registry's latest.
            source_version: Current version, required for the UPGRADE action.
            confirm: Must be True for destructive actions (e.g.
                UNINSTALL_DELETE_ALL).

        Returns:
            Job provenance (job_name/job_namespace) for status and log polling.
        """
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        if action.upper() in DESTRUCTIVE_ACTIONS:
            if not server.config.enable_dangerous_operations:
                return {
                    "error": "Dangerous operations are disabled",
                    "message": (
                        f"Action '{action}' permanently deletes data. Enable "
                        "dangerous operations to allow it."
                    ),
                }
            if not confirm:
                return {
                    "error": "Action not confirmed",
                    "message": (
                        f"Action '{action}' permanently deletes data. Set "
                        "confirm=True to proceed."
                    ),
                }

        try:
            return _client().run_action(
                name=name,
                action=action,
                target_namespace=target_namespace,
                mode=mode,
                parameters=parameters,
                version=version,
                source_version=source_version,
            )
        except RHOAIError as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_quickstart_action_status(
        job_name: str,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Get the status of a quickstart action (installer Job).

        Args:
            job_name: The Job name returned by run_quickstart_action.
            namespace: The Job's namespace; defaults to the configured
                quickstart job namespace.

        Returns:
            Job phase and status details.
        """
        ns = namespace or server.config.quickstart_job_namespace
        try:
            return _client().get_action_status(job_name, ns)
        except RHOAIError as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_quickstart_action_logs(
        job_name: str,
        namespace: str | None = None,
        tail_lines: int = 200,
    ) -> dict[str, Any]:
        """Get installer logs for a quickstart action (installer Job).

        Args:
            job_name: The Job name returned by run_quickstart_action.
            namespace: The Job's namespace; defaults to the configured
                quickstart job namespace.
            tail_lines: Number of trailing log lines to return (default: 200).

        Returns:
            The installer pod's logs.
        """
        ns = namespace or server.config.quickstart_job_namespace
        try:
            return _client().get_action_logs(job_name, ns, tail_lines)
        except RHOAIError as e:
            return {"error": str(e)}
