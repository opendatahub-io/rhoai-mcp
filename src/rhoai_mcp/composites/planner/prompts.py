"""MCP prompts for the recommendation-to-deployment workflow.

Provides prompts that guide AI agents through the end-to-end flow
from model recommendations to validated deployments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_prompts(mcp: FastMCP, server: RHOAIServer) -> None:  # noqa: ARG001
    """Register planner workflow prompts.

    Args:
        mcp: The FastMCP server instance to register prompts with.
        server: The RHOAI server instance (unused but required for interface).
    """

    @mcp.prompt(
        name="recommend-and-deploy",
        description="End-to-end workflow: recommend a model, plan deployment, and deploy it",
    )
    def recommend_and_deploy(
        description: str,
        namespace: str,
        priority: str = "balanced",
    ) -> str:
        """Guide an agent through recommendation to deployment.

        Args:
            description: Natural language description of the use case.
            namespace: Target namespace for deployment.
            priority: Optimization priority (balanced, cost, performance, quality).

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I want to find the best model for my use case and deploy it.

**My Requirements:**
- Description: {description}
- Target namespace: {namespace}
- Optimization priority: {priority}

**Please guide me through these steps:**

1. **Get Recommendations**
   - Use `recommend_model` with text="{description}"
   - Review the recommendations along with cluster GPU availability
   - Help me pick the best option based on {priority} priority
   - Show me the tradeoffs between the options

2. **Plan the Deployment**
   - Use `plan_deployment` with my chosen recommendation and namespace="{namespace}"
   - Review the resolved parameters (runtime, storage, resources)
   - Address any issues flagged in the plan before proceeding

3. **Deploy and Validate**
   - Only after I confirm the plan, use `execute_deployment`
   - Wait for the model to become Ready
   - Verify the endpoint is working

4. **Report Results**
   - Show me the endpoint URL
   - Compare predicted vs. actual configuration
   - Suggest monitoring and scaling options

Please start by getting model recommendations for my use case."""
