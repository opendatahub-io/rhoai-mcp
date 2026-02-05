"""CRD definitions for Model Registry domain."""

from rhoai_mcp.clients.base import CRDDefinition


class ModelRegistryCRDs:
    """CRD definitions for Model Registry auto-discovery.

    These CRDs are used to discover the Model Registry deployment
    in the cluster, not for direct Model Registry API operations
    (which use REST instead of Kubernetes CRDs).
    """

    # ModelRegistry component CRD from Open Data Hub / RHOAI
    # This defines the configuration for the Model Registry operator
    # and includes spec.registriesNamespace for where registries are deployed
    MODEL_REGISTRY_COMPONENT = CRDDefinition(
        group="components.platform.opendatahub.io",
        version="v1alpha1",
        plural="modelregistries",
        kind="ModelRegistry",
    )

    # OpenShift Route for external access to services
    ROUTE = CRDDefinition(
        group="route.openshift.io",
        version="v1",
        plural="routes",
        kind="Route",
    )
