"""CRD definitions for {{DOMAIN_DESCRIPTION}}.

Only needed if your domain uses custom Kubernetes CRDs.
Delete this file if your domain uses core API resources only.
"""

from rhoai_mcp.clients.base import CRDDefinition


class {{DOMAIN_CLASS}}CRDs:
    """CRD definitions for the {{DOMAIN_NAME}} domain."""

    {{CRD_KIND_UPPER}} = CRDDefinition(
        group="{{CRD_GROUP}}",
        version="{{CRD_VERSION}}",
        plural="{{CRD_PLURAL}}",
        kind="{{CRD_KIND}}",
    )

    @classmethod
    def all_crds(cls) -> list[CRDDefinition]:
        """Return all CRD definitions for this domain."""
        return [cls.{{CRD_KIND_UPPER}}]
