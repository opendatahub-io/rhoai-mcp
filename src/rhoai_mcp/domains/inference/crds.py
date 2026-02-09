"""CRD definitions for Inference domain."""

from rhoai_mcp.clients.base import CRDDefinition

# Platform namespace where RHOAI stores templates and shared resources
PLATFORM_NAMESPACE = "redhat-ods-applications"


class InferenceCRDs:
    """KServe CRD definitions."""

    # KServe InferenceService
    INFERENCE_SERVICE = CRDDefinition(
        group="serving.kserve.io",
        version="v1beta1",
        plural="inferenceservices",
        kind="InferenceService",
    )

    # KServe ServingRuntime
    SERVING_RUNTIME = CRDDefinition(
        group="serving.kserve.io",
        version="v1alpha1",
        plural="servingruntimes",
        kind="ServingRuntime",
    )

    # OpenShift Template (for platform-level serving runtime templates)
    TEMPLATE = CRDDefinition(
        group="template.openshift.io",
        version="v1",
        plural="templates",
        kind="Template",
    )
