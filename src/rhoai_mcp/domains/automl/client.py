"""Kubernetes discovery for exposed AutoGluon InferenceServices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rhoai_mcp.domains.automl.models import AUTO_ML_EXPOSED_LABEL, AutoMLDeployment
from rhoai_mcp.domains.inference.crds import InferenceCRDs

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class AutoMLClient:
    """Read-only client for AutoML deployments opted into MCP exposure."""

    def __init__(self, k8s: K8sClient) -> None:
        self._k8s = k8s

    def list_deployments(self, namespace: str) -> list[AutoMLDeployment]:
        resources = self._k8s.list_resources(
            InferenceCRDs.INFERENCE_SERVICE,
            namespace=namespace,
            label_selector=f"{AUTO_ML_EXPOSED_LABEL}=true",
        )
        return [
            deployment
            for item in resources
            if (deployment := AutoMLDeployment.from_inference_service(item)) is not None
        ]

    def get_deployment(self, name: str, namespace: str) -> AutoMLDeployment:
        resource = self._k8s.get(InferenceCRDs.INFERENCE_SERVICE, name=name, namespace=namespace)
        deployment = AutoMLDeployment.from_inference_service(resource)
        if deployment is None:
            raise ValueError(f"InferenceService '{name}' is not an exposed AutoML deployment")
        return deployment
