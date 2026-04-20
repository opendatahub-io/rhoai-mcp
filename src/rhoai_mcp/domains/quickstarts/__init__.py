"""Quickstarts domain module.

Provides tools for discovering and deploying Red Hat AI Quickstarts
to OpenShift clusters.
"""

from rhoai_mcp.domains.quickstarts.client import QuickstartClient
from rhoai_mcp.domains.quickstarts.models import (
    DeploymentMethod,
    DeploymentResult,
    Quickstart,
    QuickstartReadme,
)

__all__ = [
    "QuickstartClient",
    "DeploymentMethod",
    "DeploymentResult",
    "Quickstart",
    "QuickstartReadme",
]
