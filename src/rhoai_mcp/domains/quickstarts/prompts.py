"""Quickstart workflow prompts for RHOAI MCP.

Provides a prompt that guides AI agents through the end-to-end quickstart
deployment loop: discover → inspect manifest → elicit parameters → run →
poll. Mirrors the structure of the prompts domain's workflow prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_prompts(mcp: FastMCP, server: RHOAIServer) -> None:  # noqa: ARG001
    """Register quickstart workflow prompts.

    Args:
        mcp: The FastMCP server instance to register prompts with.
        server: The RHOAI server instance (unused but required for interface).
    """

    @mcp.prompt(
        name="deploy-quickstart",
        description="Discover, configure and deploy a Red Hat AI quickstart",
    )
    def deploy_quickstart(
        quickstart: str = "",
        target_namespace: str = "",
        mode: str = "demo",
    ) -> str:
        """Generate guidance for the quickstart deployment workflow.

        Args:
            quickstart: Quickstart name if already known; leave empty to discover.
            target_namespace: Namespace to deploy into; leave empty to use the
                manifest's defaultNamespace.
            mode: Install mode (e.g. "demo" or "production").

        Returns:
            Workflow guidance as a string prompt.
        """
        chosen = quickstart or "(to be chosen from the catalog)"
        namespace = target_namespace or "(defaults to the manifest's defaultNamespace)"

        return f"""I want to deploy a Red Hat AI quickstart.

**Request:**
- Quickstart: {chosen}
- Target namespace: {namespace}
- Mode: {mode}

**Please guide me through these steps:**

1. **Discover** — If no quickstart is chosen yet, use `list_quickstarts` to
   browse the catalog and pick one that matches my goal. Confirm the name and
   the version to deploy.

2. **Inspect the manifest** — Use `get_quickstart_manifest` for the chosen
   quickstart to review:
   - `supportedActions` (confirm INSTALL is available) and `supportedModes`
     (confirm "{mode}" is supported)
   - `prerequisites` (OpenShift version, operators, cluster resources)
   - the `parameters` block (which values are required, their types, and any
     `llm_guidance`)

3. **Elicit parameters** — Collect every required parameter value from me
   before running anything. For secret/password parameters (type "password"),
   always ask me explicitly — never auto-generate or guess a value.

4. **Run the install** — Use `run_quickstart_action` with:
   - name="{quickstart or "<chosen name>"}"
   - action="INSTALL"
   - mode="{mode}"
   - target_namespace as appropriate (empty to use the default)
   - parameters={{...the values collected in step 3...}}
   Note the returned `job_name` and `job_namespace` for polling.

5. **Poll to completion** — Use `get_quickstart_action_status` with the returned
   job identifiers to watch the Job. If it fails, inspect `exit_code`,
   `termination_message` and `result` in the status, and use
   `get_quickstart_action_logs` for the full installer output.

**Teardown note:** UNINSTALL actions require `confirm=True`;
`UNINSTALL_DELETE_ALL` additionally destroys data and requires dangerous
operations to be enabled.

Please start with step 1 unless a quickstart is already chosen."""
