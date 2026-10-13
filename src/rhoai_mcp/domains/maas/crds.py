"""CRD definitions for the MaaS (Models-as-a-Service) domain.

MaaS serves models through a shared gateway (Kuadrant/Gateway API) that is
installed with the MaaS platform (see the rhoai-maas-guide quick start).
Deploying a model as a MaaS endpoint creates an LLMInferenceService that is
routed through that gateway plus MaaSModelRef/MaaSAuthPolicy/MaaSSubscription
resources that register and govern the model.
"""

from rhoai_mcp.clients.base import CRDDefinition

# Namespace where MaaS models are deployed by default
DEFAULT_MODEL_NAMESPACE = "llm"

# Namespaces where the MaaS API deployment lives (RHOAI 3.5+ first, 3.4 fallback)
MAAS_API_NAMESPACES = ("redhat-ai-gateway-infra", "redhat-ods-applications")

# Namespace where MaaS governance resources (AuthPolicy, Subscription) live
GOVERNANCE_NAMESPACE = "models-as-a-service"

# Shared gateway that fronts all MaaS models
GATEWAY_NAME = "maas-default-gateway"
GATEWAY_NAMESPACE = "openshift-ingress"

# Label that exposes a namespace's models through the MaaS gateway
GATEWAY_ACCESS_LABEL = "maas.opendatahub.io/gateway-access"

# RHOAI version layout for the MaaS API location
MAAS_API_NS_35_PLUS = MAAS_API_NAMESPACES[0]
MAAS_API_NS_34 = MAAS_API_NAMESPACES[1]


class MaasCRDs:
    """MaaS CRD definitions."""

    # KServe LLMInferenceService (llm-d based serving routed through the gateway)
    LLM_INFERENCE_SERVICE = CRDDefinition(
        group="serving.kserve.io",
        version="v1alpha1",
        plural="llminferenceservices",
        kind="LLMInferenceService",
    )

    # Registers a serving resource as a MaaS model
    MAAS_MODEL_REF = CRDDefinition(
        group="maas.opendatahub.io",
        version="v1alpha1",
        plural="maasmodelrefs",
        kind="MaaSModelRef",
    )

    # Grants subjects access to MaaS models
    MAAS_AUTH_POLICY = CRDDefinition(
        group="maas.opendatahub.io",
        version="v1alpha1",
        plural="maasauthpolicies",
        kind="MaaSAuthPolicy",
    )

    # Defines subscriptions (rate limits) for MaaS models
    MAAS_SUBSCRIPTION = CRDDefinition(
        group="maas.opendatahub.io",
        version="v1alpha1",
        plural="maassubscriptions",
        kind="MaaSSubscription",
    )

    # Gateway API Gateway that exposes MaaS models
    GATEWAY = CRDDefinition(
        group="gateway.networking.k8s.io",
        version="v1",
        plural="gateways",
        kind="Gateway",
    )

    # OpenShift cluster ingress configuration (for the cluster domain fallback)
    CLUSTER_INGRESS_CONFIG = CRDDefinition(
        group="config.openshift.io",
        version="v1",
        plural="ingresses",
        kind="Ingress",
    )

    # Deployment (used to locate the maas-api deployment)
    DEPLOYMENT = CRDDefinition(
        group="apps",
        version="v1",
        plural="deployments",
        kind="Deployment",
    )
