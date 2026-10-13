"""Pydantic models for MaaS (Models-as-a-Service) model deployment."""

from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.domains.maas.crds import DEFAULT_MODEL_NAMESPACE
from rhoai_mcp.models.common import Condition, ResourceMetadata

# CPU simulator image from the rhoai-maas-guide; lets users try MaaS without
# a real model or GPU. Override with the `image` parameter for real serving.
DEFAULT_SIMULATOR_IMAGE = "ghcr.io/llm-d/llm-d-inference-sim:v0.7.1"

# Default tag appended to tagless Red Hat AI Inference Server images. The
# registry.redhat.io rhaiis repositories do not publish a :latest tag and
# reject pulls without an explicit version.
DEFAULT_RHAIS_TAG = "3.3.0"

# Valid windows for MaaS token rate limits.
VALID_RATE_LIMIT_WINDOWS = ("1s", "1m", "1h", "1d")


def needs_registry_tag(image: str) -> bool:
    """Return True for registry.redhat.io images given without a tag or digest.

    registry.redhat.io repositories (e.g. rhaiis/vllm-cpu-rhel9) reject pulls
    with an implicit ':latest' tag, so a tagless reference would fail with
    ErrImagePull after deployment.
    """
    repo = image.rsplit("/", 1)[-1]
    return image.startswith("registry.redhat.io/") and ":" not in repo and "@" not in image


class MaasModelDeploy(BaseModel):
    """Request model for deploying a model as a MaaS endpoint."""

    name: str = Field(..., description="LLMInferenceService name (DNS-1123 subdomain)")
    namespace: str = Field(
        DEFAULT_MODEL_NAMESPACE,
        description="Namespace to deploy into (labeled for MaaS gateway access)",
    )
    model_name: str = Field(..., description="Served model name (e.g., 'facebook/opt-125m')")
    model_uri: str = Field(
        ...,
        description="Model location (hf://org/repo or oci://registry/repo:tag)",
    )
    image: str | None = Field(
        None,
        description=(
            "Serving container image. None deploys the CPU simulator "
            f"({DEFAULT_SIMULATOR_IMAGE}); pass a vLLM image for real "
            "inference (e.g. registry.redhat.io/rhaiis/vllm-cpu-rhel9:"
            f"{DEFAULT_RHAIS_TAG}; registry.redhat.io images require an "
            "explicit tag)."
        ),
    )
    display_name: str | None = Field(None, description="Human-readable display name")
    description: str | None = Field(None, description="Human-readable description")
    replicas: int = Field(1, ge=0, description="Number of replicas")
    cpu_request: str | None = Field(None, description="CPU request (None = preset default)")
    cpu_limit: str | None = Field(None, description="CPU limit (None = preset default)")
    memory_request: str | None = Field(None, description="Memory request (None = preset default)")
    memory_limit: str | None = Field(None, description="Memory limit (None = preset default)")
    grant_authenticated_access: bool = Field(
        False,
        description=(
            "Create an AuthPolicy + Subscription granting all authenticated users "
            "access with rate limits. Disabled by default (least privilege)."
        ),
    )
    token_rate_limit: int = Field(
        100, ge=1, description="Tokens per window when grant_authenticated_access is set"
    )
    rate_limit_window: str = Field(
        "1m",
        description="Rate limit window when grant_authenticated_access is set "
        f"(one of: {', '.join(VALID_RATE_LIMIT_WINDOWS)})",
    )


class MaasModelStatus(BaseModel):
    """Status of a MaaS model deployment."""

    metadata: ResourceMetadata
    model_name: str | None = Field(None, description="Served model name")
    model_uri: str | None = Field(None, description="Model location")
    replicas: int | None = Field(None, description="Configured replicas")
    ready_replicas: int | None = Field(None, description="Ready replicas")
    status: str = Field("Unknown", description="Deployment status")
    maas_registered: bool = Field(False, description="Whether MaaSModelRef exists")
    maas_phase: str | None = Field(None, description="MaaSModelRef status phase")
    maas_endpoint: str | None = Field(None, description="MaaS gateway endpoint URL")
    conditions: list[Condition] = Field(default_factory=list, description="Status conditions")

    @classmethod
    def from_llm_inference_service_cr(
        cls, isvc: Any, maas_endpoint: str | None = None
    ) -> "MaasModelStatus":
        """Build a MaasModelStatus from LLMInferenceService + MaaSModelRef CRs.

        Args:
            isvc: LLMInferenceService resource instance.
            maas_endpoint: MaaS gateway endpoint URL (None if gateway not found).
        """
        from rhoai_mcp.models.common import ResourceStatus

        metadata = isvc.metadata
        spec = isvc.spec or {}
        status_obj = getattr(isvc, "status", None)

        model = spec.get("model", {}) or {}

        conditions = []
        svc_status = ResourceStatus.UNKNOWN
        if status_obj:
            for cond in getattr(status_obj, "conditions", []) or []:
                conditions.append(Condition.from_k8s_condition(cond))
                if cond.type == "Ready":
                    svc_status = (
                        ResourceStatus.READY if cond.status == "True" else ResourceStatus.PENDING
                    )

        ready_replicas = getattr(status_obj, "readyReplicas", None) if status_obj else None

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                metadata,
                kind="LLMInferenceService",
                api_version="serving.kserve.io/v1alpha1",
            ),
            model_name=model.get("name"),
            model_uri=model.get("uri"),
            replicas=spec.get("replicas"),
            ready_replicas=ready_replicas,
            status=svc_status.value,
            maas_endpoint=maas_endpoint,
            conditions=conditions,
        )
