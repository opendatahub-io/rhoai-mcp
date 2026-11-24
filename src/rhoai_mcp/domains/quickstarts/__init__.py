"""Quickstarts domain — discover and deploy Red Hat AI quickstarts."""

from rhoai_mcp.domains.quickstarts.client import QuickstartsClient
from rhoai_mcp.domains.quickstarts.models import (
    QuickstartManifest,
    QuickstartParameter,
    QuickstartRegistry,
    QuickstartSummary,
)
from rhoai_mcp.domains.quickstarts.oci import OCIArtifactClient

__all__ = [
    "OCIArtifactClient",
    "QuickstartManifest",
    "QuickstartParameter",
    "QuickstartRegistry",
    "QuickstartSummary",
    "QuickstartsClient",
]
