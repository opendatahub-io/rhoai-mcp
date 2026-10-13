"""K8s client operations for the MaaS (Models-as-a-Service) domain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.maas.crds import (
    GATEWAY_ACCESS_LABEL,
    GATEWAY_NAME,
    GATEWAY_NAMESPACE,
    GOVERNANCE_NAMESPACE,
    MAAS_API_NAMESPACES,
    MaasCRDs,
)
from rhoai_mcp.domains.maas.models import (
    DEFAULT_SIMULATOR_IMAGE,
    MaasModelDeploy,
    MaasModelStatus,
)
from rhoai_mcp.utils.errors import (
    NotFoundError,
    NotManagedByMCPError,
    RHOAIError,
)
from rhoai_mcp.utils.labels import RHOAILabels

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

logger = logging.getLogger(__name__)


class MaasClient:
    """Client for MaaS platform inspection and model deployment.

    Wraps K8sClient operations for the resources that make up a MaaS
    endpoint: LLMInferenceService (serving), MaaSModelRef (registration),
    MaaSAuthPolicy/MaaSSubscription (governance), plus read-only inspection
    of the shared platform components (DataScienceCluster, Gateway, maas-api).
    """

    def __init__(self, k8s: K8sClient) -> None:
        self._k8s = k8s

    # ------------------------------------------------------------------
    # Platform inspection (read-only)
    # ------------------------------------------------------------------

    def get_dsc_maas_state(self) -> dict[str, Any] | None:
        """Return the DataScienceCluster MaaS component state.

        Checks both the RHOAI 3.5+ location (aigateway.modelsAsAService) and
        the 3.4 location (kserve.modelsAsService).

        Returns:
            Dict with name and management_state, or None if no
            DataScienceCluster exists.
        """
        from rhoai_mcp.clients.base import CRDs

        dscs = self._k8s.list_resources(CRDs.DATA_SCIENCE_CLUSTER)
        if not dscs:
            return None

        dsc = dscs[0]
        spec = getattr(dsc, "spec", None) or {}
        components = spec.get("components", {}) or {}
        state_35 = (components.get("aigateway", {}) or {}).get("modelsAsAService", {}) or {}
        state_34 = (components.get("kserve", {}) or {}).get("modelsAsService", {}) or {}
        management_state = state_35.get("managementState") or state_34.get("managementState")

        return {"name": dsc.metadata.name, "management_state": management_state}

    def get_gateway(self) -> Any | None:
        """Return the MaaS Gateway resource, or None if not installed."""
        try:
            return self._k8s.get(MaasCRDs.GATEWAY, GATEWAY_NAME, GATEWAY_NAMESPACE)
        except NotFoundError:
            return None

    def is_gateway_programmed(self, gateway: Any) -> bool | None:
        """Return the Gateway Programmed condition status, or None if unknown."""
        status = getattr(gateway, "status", None)
        if not status:
            return None
        for cond in getattr(status, "conditions", []) or []:
            if cond.get("type") == "Programmed":
                return bool(cond.get("status") == "True")
        return None

    def get_gateway_hostname(self) -> str | None:
        """Return the MaaS gateway hostname.

        Uses the Gateway listener hostname and falls back to
        maas.<cluster-domain> from the OpenShift ingress configuration.
        """
        gateway = self.get_gateway()
        if gateway is not None:
            spec = getattr(gateway, "spec", None) or {}
            for listener in spec.get("listeners") or []:
                hostname = listener.get("hostname")
                if hostname:
                    return str(hostname)

        domain = self._get_cluster_domain()
        if domain:
            return f"maas.{domain}"
        return None

    def _get_cluster_domain(self) -> str | None:
        """Return the OpenShift cluster domain from the ingress configuration."""
        try:
            ingress = self._k8s.get(MaasCRDs.CLUSTER_INGRESS_CONFIG, "cluster")
        except (NotFoundError, RHOAIError):
            return None
        spec = getattr(ingress, "spec", None) or {}
        return spec.get("domain") or None

    def find_maas_api(self) -> dict[str, Any] | None:
        """Find the maas-api deployment across known namespaces.

        Returns:
            Dict with namespace and ready_replicas, or None if not found.
        """
        for namespace in MAAS_API_NAMESPACES:
            try:
                deployment = self._k8s.get(MaasCRDs.DEPLOYMENT, "maas-api", namespace)
            except NotFoundError:
                continue
            status = getattr(deployment, "status", None)
            ready = getattr(status, "readyReplicas", None) if status else None
            return {"namespace": namespace, "ready_replicas": ready or 0}
        return None

    # ------------------------------------------------------------------
    # Namespace preparation
    # ------------------------------------------------------------------

    def ensure_model_namespace(self, namespace: str) -> dict[str, Any]:
        """Ensure the model namespace exists and is exposed to the MaaS gateway.

        Creates the namespace with the gateway-access label when missing, and
        attempts to label an existing namespace that lacks it (best effort).

        Returns:
            Dict with created, gateway_access, and warnings.
        """
        warnings: list[str] = []
        gateway_access = False

        try:
            ns = self._k8s.get_namespace(namespace)
            labels = dict(ns.metadata.labels or {})
            gateway_access = labels.get(GATEWAY_ACCESS_LABEL) == "true"
            created = False
        except NotFoundError:
            self._k8s.create_namespace(
                name=namespace,
                labels={
                    **RHOAILabels.managed_by_mcp_labels(),
                    GATEWAY_ACCESS_LABEL: "true",
                    RHOAILabels.DASHBOARD: "true",
                },
            )
            logger.info("Created MaaS model namespace '%s'", namespace)
            created = True
            gateway_access = True

        if not gateway_access:
            gateway_access = self._ensure_gateway_access_label(namespace, warnings)

        return {"created": created, "gateway_access": gateway_access, "warnings": warnings}

    def _ensure_gateway_access_label(self, namespace: str, warnings: list[str]) -> bool:
        """Try to stamp the gateway-access label on an existing namespace.

        The namespace was not created by this tool, so a successful label is
        always surfaced as a warning: it exposes the namespace's models to
        the shared MaaS gateway and the caller must know that happened.
        """
        try:
            self._k8s.patch_project(
                name=namespace,
                labels={
                    GATEWAY_ACCESS_LABEL: "true",
                    RHOAILabels.DASHBOARD: "true",
                },
            )
        except Exception as e:
            logger.warning("Could not label namespace '%s': %s", namespace, e)
            warnings.append(
                f"Could not set '{GATEWAY_ACCESS_LABEL}=true' on namespace "
                f"'{namespace}': {e}. Models in this namespace will not be "
                f"reachable through the MaaS gateway until an admin runs: "
                f"oc label namespace {namespace} {GATEWAY_ACCESS_LABEL}=true --overwrite"
            )
            return False
        warnings.append(
            f"Labeled existing namespace '{namespace}' with "
            f"'{GATEWAY_ACCESS_LABEL}=true' - models in it are now reachable "
            "through the shared MaaS gateway. Remove the label with: "
            f"oc label namespace {namespace} {GATEWAY_ACCESS_LABEL}-"
        )
        return True

    # ------------------------------------------------------------------
    # Model deployment
    # ------------------------------------------------------------------

    def deploy_model(self, request: MaasModelDeploy) -> Any:
        """Create the LLMInferenceService for a MaaS model."""
        body = self._build_llm_inference_service(request)
        return self._k8s.create(MaasCRDs.LLM_INFERENCE_SERVICE, body, namespace=request.namespace)

    def create_model_ref(self, request: MaasModelDeploy) -> Any:
        """Create the MaaSModelRef that registers the model with MaaS."""
        annotations: dict[str, str] = {}
        if request.display_name:
            annotations["openshift.io/display-name"] = request.display_name
        if request.description:
            annotations["openshift.io/description"] = request.description

        body: dict[str, Any] = {
            "apiVersion": MaasCRDs.MAAS_MODEL_REF.api_version,
            "kind": "MaaSModelRef",
            "metadata": {
                "name": request.name,
                "namespace": request.namespace,
                "labels": RHOAILabels.managed_by_mcp_labels(),
            },
            "spec": {
                "modelRef": {
                    "kind": "LLMInferenceService",
                    "name": request.name,
                },
            },
        }
        if annotations:
            body["metadata"]["annotations"] = annotations

        return self._k8s.create(MaasCRDs.MAAS_MODEL_REF, body, namespace=request.namespace)

    def create_auth_policy(self, request: MaasModelDeploy) -> Any:
        """Create an AuthPolicy granting all authenticated users access."""
        body: dict[str, Any] = {
            "apiVersion": MaasCRDs.MAAS_AUTH_POLICY.api_version,
            "kind": "MaaSAuthPolicy",
            "metadata": {
                "name": f"{request.name}-access",
                "namespace": GOVERNANCE_NAMESPACE,
                "labels": RHOAILabels.managed_by_mcp_labels(),
            },
            "spec": {
                "modelRefs": [{"name": request.name, "namespace": request.namespace}],
                "subjects": {
                    "groups": [{"name": "system:authenticated"}],
                    "users": [],
                },
            },
        }
        return self._k8s.create(MaasCRDs.MAAS_AUTH_POLICY, body, namespace=GOVERNANCE_NAMESPACE)

    def create_subscription(self, request: MaasModelDeploy) -> Any:
        """Create a Subscription with token rate limits for the model."""
        body: dict[str, Any] = {
            "apiVersion": MaasCRDs.MAAS_SUBSCRIPTION.api_version,
            "kind": "MaaSSubscription",
            "metadata": {
                "name": f"{request.name}-free",
                "namespace": GOVERNANCE_NAMESPACE,
                "labels": RHOAILabels.managed_by_mcp_labels(),
            },
            "spec": {
                "owner": {
                    "groups": [{"name": "system:authenticated"}],
                    "users": [],
                },
                "modelRefs": [
                    {
                        "name": request.name,
                        "namespace": request.namespace,
                        "tokenRateLimits": [
                            {
                                "limit": request.token_rate_limit,
                                "window": request.rate_limit_window,
                            },
                        ],
                    },
                ],
                "priority": 10,
            },
        }
        return self._k8s.create(MaasCRDs.MAAS_SUBSCRIPTION, body, namespace=GOVERNANCE_NAMESPACE)

    def get_model_status(
        self, name: str, namespace: str, maas_endpoint: str | None = None
    ) -> MaasModelStatus:
        """Get combined LLMInferenceService + MaaSModelRef status."""
        isvc = self._k8s.get(MaasCRDs.LLM_INFERENCE_SERVICE, name, namespace)
        status = MaasModelStatus.from_llm_inference_service_cr(isvc, maas_endpoint=maas_endpoint)

        try:
            model_ref = self._k8s.get(MaasCRDs.MAAS_MODEL_REF, name, namespace)
            status.maas_registered = True
            ref_status = getattr(model_ref, "status", None)
            status.maas_phase = getattr(ref_status, "phase", None)
        except NotFoundError:
            status.maas_registered = False

        return status

    # ------------------------------------------------------------------
    # Model deletion
    # ------------------------------------------------------------------

    def delete_model_resources(self, name: str, namespace: str) -> dict[str, list[str]]:
        """Delete all MaaS resources for a model, in the guide's cleanup order.

        Deletes the governance resources (MaaSSubscription, MaaSAuthPolicy),
        then the MaaSModelRef, then the LLMInferenceService. Resources that
        do not exist are skipped; governance resources not created by this
        MCP server are kept and reported in the result.

        Raises NotManagedByMCPError before touching anything if the
        LLMInferenceService itself was not created by this MCP server, and
        NotFoundError if it does not exist.

        Returns:
            Dict with the deleted, skipped, and kept resource kinds.
        """
        # Refuse before touching anything if the serving resource is not
        # MCP-managed: removing the governance around a live foreign isvc
        # would deregister it from MaaS while it keeps serving.
        isvc = self._k8s.get(MaasCRDs.LLM_INFERENCE_SERVICE, name, namespace)
        labels = dict(isvc.metadata.labels or {})
        if not RHOAILabels.is_managed_by_mcp(labels):
            raise NotManagedByMCPError("LLMInferenceService", name, namespace)

        deleted: list[str] = []
        skipped: list[str] = []
        kept: list[str] = []
        for kind, crd, res_name, res_ns in [
            (
                "MaaSSubscription",
                MaasCRDs.MAAS_SUBSCRIPTION,
                f"{name}-free",
                GOVERNANCE_NAMESPACE,
            ),
            (
                "MaaSAuthPolicy",
                MaasCRDs.MAAS_AUTH_POLICY,
                f"{name}-access",
                GOVERNANCE_NAMESPACE,
            ),
            ("MaaSModelRef", MaasCRDs.MAAS_MODEL_REF, name, namespace),
            ("LLMInferenceService", MaasCRDs.LLM_INFERENCE_SERVICE, name, namespace),
        ]:
            try:
                self._k8s.delete(crd, res_name, res_ns)
                deleted.append(kind)
            except NotFoundError:
                skipped.append(kind)
            except NotManagedByMCPError:
                kept.append(kind)
        return {"deleted": deleted, "skipped": skipped, "kept": kept}

    # ------------------------------------------------------------------
    # Manifest builders
    # ------------------------------------------------------------------

    def _build_llm_inference_service(self, request: MaasModelDeploy) -> dict[str, Any]:
        """Build the LLMInferenceService body routed through the MaaS gateway."""
        annotations = {"security.opendatahub.io/enable-auth": "true"}
        if request.display_name:
            annotations["openshift.io/display-name"] = request.display_name

        labels = {
            **RHOAILabels.managed_by_mcp_labels(),
            RHOAILabels.DASHBOARD: "true",
            "opendatahub.io/genai-asset": "true",
        }

        return {
            "apiVersion": MaasCRDs.LLM_INFERENCE_SERVICE.api_version,
            "kind": "LLMInferenceService",
            "metadata": {
                "name": request.name,
                "namespace": request.namespace,
                "labels": labels,
                "annotations": annotations,
            },
            "spec": {
                "model": {"name": request.model_name, "uri": request.model_uri},
                "replicas": request.replicas,
                "router": {
                    "route": {},
                    "gateway": {
                        "refs": [
                            {"name": GATEWAY_NAME, "namespace": GATEWAY_NAMESPACE},
                        ],
                    },
                },
                "template": {"containers": [self._build_container(request)]},
            },
        }

    def _build_container(self, request: MaasModelDeploy) -> dict[str, Any]:
        """Build the serving container spec.

        With no image, uses the llm-d CPU simulator preset from the
        rhoai-maas-guide so MaaS can be tried without a real model or GPU.
        With an image, uses the Red Hat AI Inference Server (vLLM) layout from
        the guide's real-model presets.
        """
        is_simulator = request.image is None
        # vLLM's CPU backend profiles against total node memory, not the
        # container limit, and pre-allocates a KV cache sized from it - a 16Gi
        # cgroup limit gets OOMKilled on ~32Gi nodes. For CPU images, bound the
        # KV cache explicitly, cap the context length, and default the limit
        # to 24Gi so the deployment fits CPU-only nodes.
        is_cpu_vllm = not is_simulator and "vllm-cpu" in (request.image or "")
        resources = {
            "requests": {
                "cpu": request.cpu_request or ("100m" if is_simulator else "4"),
                "memory": request.memory_request or ("256Mi" if is_simulator else "16Gi"),
            },
            "limits": {
                "cpu": request.cpu_limit or ("500m" if is_simulator else "8"),
                "memory": request.memory_limit
                or ("512Mi" if is_simulator else "24Gi" if is_cpu_vllm else "16Gi"),
            },
        }

        if is_simulator:
            return {
                "name": "main",
                "image": DEFAULT_SIMULATOR_IMAGE,
                "imagePullPolicy": "Always",
                "command": ["/app/llm-d-inference-sim"],
                "args": [
                    "--port",
                    "8000",
                    "--model",
                    request.model_name,
                    "--mode",
                    "random",
                    "--ssl-certfile",
                    "/var/run/kserve/tls/tls.crt",
                    "--ssl-keyfile",
                    "/var/run/kserve/tls/tls.key",
                ],
                "env": [
                    {
                        "name": "POD_NAME",
                        "valueFrom": {
                            "fieldRef": {
                                "apiVersion": "v1",
                                "fieldPath": "metadata.name",
                            },
                        },
                    },
                    {
                        "name": "POD_NAMESPACE",
                        "valueFrom": {
                            "fieldRef": {
                                "apiVersion": "v1",
                                "fieldPath": "metadata.namespace",
                            },
                        },
                    },
                ],
                "ports": [{"name": "https", "containerPort": 8000, "protocol": "TCP"}],
                "livenessProbe": {
                    "httpGet": {"path": "/health", "port": "https", "scheme": "HTTPS"},
                },
                "readinessProbe": {
                    "httpGet": {"path": "/ready", "port": "https", "scheme": "HTTPS"},
                },
                "resources": resources,
            }

        args = [
            # KServe LLMInferenceService template syntax - {{.Name}} is
            # rendered to the isvc name per pod by the runtime, not passed
            # to vLLM as a literal string.
            "--served-model-name={{.Name}}",
            "--model=/mnt/models",
            "--enable-ssl-refresh",
            "--ssl-certfile=/var/run/kserve/tls/tls.crt",
            "--ssl-keyfile=/var/run/kserve/tls/tls.key",
            "--enable-force-include-usage",
        ]
        env: list[dict[str, str]] = []
        if is_cpu_vllm:
            args.extend(["--max-model-len", "8192"])
            env.append({"name": "VLLM_CPU_KVCACHE_SPACE", "value": "8"})

        container: dict[str, Any] = {
            "name": "main",
            "image": request.image,
            "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
            "args": args,
            "ports": [{"containerPort": 8000, "name": "https", "protocol": "TCP"}],
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
                "initialDelaySeconds": 300,
                "periodSeconds": 30,
                "timeoutSeconds": 30,
                "failureThreshold": 5,
            },
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
                "initialDelaySeconds": 300,
                "periodSeconds": 15,
                "timeoutSeconds": 15,
                "failureThreshold": 30,
            },
            "resources": resources,
        }
        if env:
            container["env"] = env
        return container
