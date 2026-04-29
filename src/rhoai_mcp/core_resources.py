"""Core MCP resources for cluster information."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)


def register_core_resources(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register core MCP resources for cluster information."""
    from rhoai_mcp.clients.base import CRDs

    @mcp.resource("rhoai://cluster/status")
    def cluster_status() -> dict:
        """Get RHOAI cluster status and health.

        Returns overall cluster status including RHOAI operator status,
        available components, and loaded plugins.
        """
        k8s = server.k8s
        pm = server._plugin_manager

        result: dict[str, Any] = {
            "connected": k8s.is_connected,
            "rhoai_available": False,
            "components": {},
            "plugins": {
                "total": len(pm.registered_plugins) if pm else 0,
                "active": list(pm.healthy_plugins.keys()) if pm else [],
            },
            "accelerators": [],
        }

        # Check for DataScienceCluster
        try:
            dsc_list = k8s.list_resources(CRDs.DATA_SCIENCE_CLUSTER)
            if dsc_list:
                result["rhoai_available"] = True
                dsc = dsc_list[0]
                status = getattr(dsc, "status", None)
                if status:
                    installed = getattr(status, "installedComponents", {}) or {}
                    for component, state in installed.items():
                        result["components"][component] = state
        except Exception:
            logger.debug("Failed to list DataScienceCluster", exc_info=True)

        # Check for accelerator profiles
        try:
            accelerators = k8s.list_resources(CRDs.ACCELERATOR_PROFILE)
            result["accelerators"] = [
                {
                    "name": acc.metadata.name,
                    "display_name": (acc.metadata.annotations or {}).get(
                        "openshift.io/display-name", acc.metadata.name
                    ),
                    "enabled": getattr(acc.spec, "enabled", True) if hasattr(acc, "spec") else True,
                }
                for acc in accelerators
            ]
        except Exception:
            logger.debug("Failed to list AcceleratorProfiles", exc_info=True)

        return result

    @mcp.resource("rhoai://cluster/plugins")
    def cluster_plugins() -> dict:
        """Get information about loaded plugins.

        Returns details about all plugins with their health status.
        """
        pm = server._plugin_manager
        if not pm:
            return {"plugins": {}}

        plugin_info = {}
        for name, plugin in pm.registered_plugins.items():
            is_healthy = name in pm.healthy_plugins

            meta = None
            if hasattr(plugin, "rhoai_get_plugin_metadata"):
                meta = plugin.rhoai_get_plugin_metadata()

            plugin_info[name] = {
                "version": meta.version if meta else "unknown",
                "description": meta.description if meta else "No description",
                "maintainer": meta.maintainer if meta else "unknown",
                "requires_crds": meta.requires_crds if meta else [],
                "healthy": is_healthy,
            }

        return {
            "total": len(pm.registered_plugins),
            "active": len(pm.healthy_plugins),
            "plugins": plugin_info,
        }

    @mcp.resource("rhoai://cluster/accelerators")
    def cluster_accelerators() -> list[dict]:
        """Get available accelerator profiles (GPUs).

        Returns the list of AcceleratorProfile resources that define
        available GPU types and configurations.
        """
        k8s = server.k8s

        try:
            accelerators = k8s.list_resources(CRDs.ACCELERATOR_PROFILE)
            return [
                {
                    "name": acc.metadata.name,
                    "display_name": (acc.metadata.annotations or {}).get(
                        "openshift.io/display-name", acc.metadata.name
                    ),
                    "description": (acc.metadata.annotations or {}).get(
                        "openshift.io/description", ""
                    ),
                    "enabled": getattr(acc.spec, "enabled", True) if hasattr(acc, "spec") else True,
                    "identifier": getattr(acc.spec, "identifier", "nvidia.com/gpu")
                    if hasattr(acc, "spec")
                    else "nvidia.com/gpu",
                    "tolerations": getattr(acc.spec, "tolerations", [])
                    if hasattr(acc, "spec")
                    else [],
                }
                for acc in accelerators
            ]
        except Exception as e:
            logger.warning("Failed to list AcceleratorProfiles: %s", e)
            return [{"error": "Failed to retrieve accelerator profiles"}]

    logger.info("Registered core MCP resources")
