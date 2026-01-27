"""Plugin registration for Training component."""

from typing import TYPE_CHECKING

from rhoai_mcp_core.clients.base import CRDDefinition
from rhoai_mcp_core.plugin import BasePlugin, PluginMetadata

from rhoai_mcp_training import __version__
from rhoai_mcp_training.crds import TrainingCRDs

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp_core.server import RHOAIServer


class TrainingPlugin(BasePlugin):
    """RHOAI MCP plugin for Kubeflow Training Operator integration.

    Provides tools for managing TrainJob resources and training runtimes,
    enabling model fine-tuning on Kubernetes/OpenShift clusters.
    """

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="training",
                version=__version__,
                description="Kubeflow Training Operator integration for RHOAI",
                maintainer="training-team@redhat.com",
                requires_crds=["TrainJob", "ClusterTrainingRuntime"],
            )
        )

    def register_tools(self, mcp: "FastMCP", server: "RHOAIServer") -> None:
        """Register training management tools."""
        from rhoai_mcp_training.tools.discovery import (
            register_tools as register_discovery_tools,
        )
        from rhoai_mcp_training.tools.lifecycle import (
            register_tools as register_lifecycle_tools,
        )
        from rhoai_mcp_training.tools.monitoring import (
            register_tools as register_monitoring_tools,
        )
        from rhoai_mcp_training.tools.planning import (
            register_tools as register_planning_tools,
        )
        from rhoai_mcp_training.tools.runtimes import (
            register_tools as register_runtimes_tools,
        )
        from rhoai_mcp_training.tools.storage import (
            register_tools as register_storage_tools,
        )
        from rhoai_mcp_training.tools.training import (
            register_tools as register_training_tools,
        )

        # Register all tool modules
        register_discovery_tools(mcp, server)
        register_monitoring_tools(mcp, server)
        register_lifecycle_tools(mcp, server)
        register_training_tools(mcp, server)
        register_runtimes_tools(mcp, server)
        register_planning_tools(mcp, server)
        register_storage_tools(mcp, server)

    def register_resources(self, mcp: "FastMCP", server: "RHOAIServer") -> None:
        """Register training-related MCP resources."""
        pass

    def get_crd_definitions(self) -> list[CRDDefinition]:
        """Return CRD definitions used by this plugin."""
        return TrainingCRDs.all_crds()


def create_plugin() -> TrainingPlugin:
    """Factory function for plugin creation."""
    return TrainingPlugin()
