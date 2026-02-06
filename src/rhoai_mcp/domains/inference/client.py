"""InferenceService (Model Serving) client operations."""

import logging
from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.inference.crds import PLATFORM_NAMESPACE, InferenceCRDs
from rhoai_mcp.domains.inference.models import (
    InferenceService,
    InferenceServiceCreate,
)

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for InferenceService (Model Serving) operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_inference_services(self, namespace: str) -> list[dict[str, Any]]:
        """List all InferenceServices in a namespace."""
        try:
            isvc_list = self._k8s.list_resources(
                InferenceCRDs.INFERENCE_SERVICE, namespace=namespace
            )
        except Exception:
            return []

        results = []
        for isvc in isvc_list:
            model = InferenceService.from_inference_service_cr(isvc)
            results.append(
                {
                    "name": model.metadata.name,
                    "display_name": model.display_name,
                    "runtime": model.runtime,
                    "model_format": model.model_format,
                    "status": model.status.value,
                    "url": model.url,
                    "_source": model.metadata.to_source_dict(),
                }
            )
        return results

    def get_inference_service(self, name: str, namespace: str) -> InferenceService:
        """Get an InferenceService by name."""
        isvc = self._k8s.get(InferenceCRDs.INFERENCE_SERVICE, name=name, namespace=namespace)
        return InferenceService.from_inference_service_cr(isvc)

    def deploy_model(self, request: InferenceServiceCreate) -> InferenceService:
        """Deploy a model as an InferenceService."""
        body = self._build_inference_service_cr(request)
        isvc = self._k8s.create(
            InferenceCRDs.INFERENCE_SERVICE, body=body, namespace=request.namespace
        )
        return InferenceService.from_inference_service_cr(isvc)

    def delete_inference_service(self, name: str, namespace: str) -> None:
        """Delete an InferenceService."""
        self._k8s.delete(InferenceCRDs.INFERENCE_SERVICE, name=name, namespace=namespace)

    def list_serving_runtimes(
        self, namespace: str, include_templates: bool = True
    ) -> list[dict[str, Any]]:
        """List available ServingRuntimes in a namespace.

        Args:
            namespace: The namespace to list runtimes from.
            include_templates: If True, also include available runtime templates
                from the platform namespace that can be instantiated.

        Returns:
            List of available runtimes (both existing and templates).
        """
        results = []

        # Get existing ServingRuntimes in the namespace
        try:
            runtimes = self._k8s.list_resources(InferenceCRDs.SERVING_RUNTIME, namespace=namespace)
            for rt in runtimes:
                runtime_info = self._parse_serving_runtime(rt)
                runtime_info["source"] = "namespace"
                results.append(runtime_info)
        except Exception as e:
            logger.debug(f"Failed to list namespace runtimes: {e}")

        # Get available templates from platform namespace
        if include_templates:
            templates = self.list_serving_runtime_templates()
            for template in templates:
                # Check if this template is already instantiated in the namespace
                already_exists = any(r["name"] == template["creates_runtime"] for r in results)
                if not already_exists:
                    results.append(
                        {
                            "name": template["creates_runtime"],
                            "display_name": template["display_name"],
                            "supported_formats": template.get("supported_formats", []),
                            "multi_model": False,
                            "source": "template",
                            "template_name": template["name"],
                            "template_namespace": PLATFORM_NAMESPACE,
                            "requires_instantiation": True,
                        }
                    )

        return results

    def _parse_serving_runtime(self, rt: Any) -> dict[str, Any]:
        """Parse a ServingRuntime resource into a dict."""
        metadata = rt.metadata
        annotations = metadata.annotations or {}
        spec = rt.spec or {}

        # Get supported formats
        containers = spec.get("containers", [])
        supported_formats = []
        for container in containers:
            for model_format in container.get("supportedModelFormats", []):
                if isinstance(model_format, dict):
                    supported_formats.append(model_format.get("name", ""))
                else:
                    supported_formats.append(str(model_format))

        return {
            "name": metadata.name,
            "display_name": annotations.get("openshift.io/display-name", metadata.name),
            "supported_formats": supported_formats,
            "multi_model": spec.get("multiModel", False),
        }

    def list_serving_runtime_templates(self) -> list[dict[str, Any]]:
        """List available serving runtime templates from the platform namespace.

        Returns:
            List of templates that can be instantiated to create ServingRuntimes.
        """
        results = []
        try:
            templates = self._k8s.list_resources(
                InferenceCRDs.TEMPLATE,
                namespace=PLATFORM_NAMESPACE,
                label_selector="opendatahub.io/dashboard=true",
            )
        except Exception as e:
            logger.debug(f"Failed to list templates: {e}")
            return []

        for template in templates:
            metadata = template.metadata
            annotations = metadata.annotations or {}
            name = metadata.name

            # Only include serving runtime templates
            # These typically have names ending in -runtime-template or similar
            if not self._is_serving_runtime_template(template):
                continue

            # Extract the runtime name that would be created
            creates_runtime = self._get_runtime_name_from_template(template)

            results.append(
                {
                    "name": name,
                    "display_name": annotations.get("openshift.io/display-name", name),
                    "description": annotations.get("description", ""),
                    "creates_runtime": creates_runtime,
                    "supported_formats": self._get_formats_from_template(template),
                }
            )

        return results

    def _is_serving_runtime_template(self, template: Any) -> bool:
        """Check if a template creates a ServingRuntime."""
        objects = template.objects if hasattr(template, "objects") else []
        for obj in objects:
            kind = obj.get("kind", "") if isinstance(obj, dict) else getattr(obj, "kind", "")
            if kind == "ServingRuntime":
                return True
        return False

    def _get_runtime_name_from_template(self, template: Any) -> str:
        """Extract the ServingRuntime name from a template."""
        objects = template.objects if hasattr(template, "objects") else []
        fallback_name: str = str(template.metadata.name)
        for obj in objects:
            if isinstance(obj, dict):
                kind = obj.get("kind", "")
                if kind == "ServingRuntime":
                    metadata = obj.get("metadata", {})
                    return str(metadata.get("name", fallback_name))
            else:
                kind = getattr(obj, "kind", "")
                if kind == "ServingRuntime":
                    obj_metadata = getattr(obj, "metadata", {})
                    if isinstance(obj_metadata, dict):
                        return str(obj_metadata.get("name", fallback_name))
                    return str(getattr(obj_metadata, "name", fallback_name))
        return fallback_name

    def _get_formats_from_template(self, template: Any) -> list[str]:
        """Extract supported model formats from a template."""
        formats = []
        objects = template.objects if hasattr(template, "objects") else []
        for obj in objects:
            if isinstance(obj, dict):
                kind = obj.get("kind", "")
                if kind == "ServingRuntime":
                    spec = obj.get("spec", {})
                    containers = spec.get("containers", [])
                    for container in containers:
                        for model_format in container.get("supportedModelFormats", []):
                            if isinstance(model_format, dict):
                                fmt = model_format.get("name", "")
                                if fmt and fmt not in formats:
                                    formats.append(fmt)
        return formats

    def instantiate_serving_runtime_template(
        self,
        template_name: str,
        target_namespace: str,
        parameters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Instantiate a serving runtime template in the target namespace.

        Args:
            template_name: Name of the template in the platform namespace.
            target_namespace: Namespace where the runtime should be created.
            parameters: Optional template parameters.

        Returns:
            Information about the created ServingRuntime.
        """
        # Get the template
        template = self._k8s.get(
            InferenceCRDs.TEMPLATE,
            name=template_name,
            namespace=PLATFORM_NAMESPACE,
        )

        # Process the template and create the ServingRuntime directly
        # We extract the ServingRuntime object and create it in the target namespace
        objects = template.objects if hasattr(template, "objects") else []

        created_runtime = None
        for obj in objects:
            # Convert ResourceField objects to dicts (from kubernetes dynamic client)
            if hasattr(obj, "to_dict"):
                obj = obj.to_dict()
            if isinstance(obj, dict):
                kind = obj.get("kind", "")
                if kind == "ServingRuntime":
                    # Apply any parameter substitutions
                    runtime_body = self._substitute_parameters(obj, parameters or {})
                    # Set the target namespace
                    if "metadata" not in runtime_body:
                        runtime_body["metadata"] = {}
                    runtime_body["metadata"]["namespace"] = target_namespace

                    # Ensure apiVersion is set
                    if "apiVersion" not in runtime_body:
                        runtime_body["apiVersion"] = InferenceCRDs.SERVING_RUNTIME.api_version

                    # Create the ServingRuntime
                    created = self._k8s.create(
                        InferenceCRDs.SERVING_RUNTIME,
                        body=runtime_body,
                        namespace=target_namespace,
                    )
                    created_runtime = self._parse_serving_runtime(created)
                    break

        if not created_runtime:
            raise ValueError(f"Template '{template_name}' does not contain a ServingRuntime")

        return {
            "created": True,
            "runtime": created_runtime,
            "template": template_name,
            "namespace": target_namespace,
        }

    def _substitute_parameters(
        self, obj: dict[str, Any], parameters: dict[str, str]
    ) -> dict[str, Any]:
        """Substitute template parameters in an object."""
        import json
        import re

        # Convert to JSON string for easy substitution
        obj_str = json.dumps(obj)

        # Substitute ${PARAM_NAME} patterns
        for key, value in parameters.items():
            pattern = r"\$\{" + re.escape(key) + r"\}"
            obj_str = re.sub(pattern, value, obj_str)

        # Remove any remaining unsubstituted parameters (use defaults if available)
        # This is a simplified approach - in production you'd want to handle defaults
        obj_str = re.sub(r"\$\{[^}]+\}", "", obj_str)

        result: dict[str, Any] = json.loads(obj_str)
        return result

    def get_model_endpoint(self, name: str, namespace: str) -> dict[str, Any]:
        """Get the inference endpoint URL for a model."""
        isvc = self.get_inference_service(name, namespace)
        return {
            "name": isvc.metadata.name,
            "status": isvc.status.value,
            "url": isvc.url,
            "internal_url": isvc.internal_url,
        }

    # -------------------------------------------------------------------------
    # Pod and Log Operations
    # -------------------------------------------------------------------------

    def get_inference_service_logs(
        self,
        namespace: str,
        name: str,
        container: str | None = None,
        tail_lines: int = 100,
        previous: bool = False,
    ) -> str:
        """Get logs from an InferenceService's pods.

        Args:
            namespace: The namespace of the service.
            name: The name of the InferenceService.
            container: Container name to get logs from. If None, uses first container.
            tail_lines: Number of lines to return.
            previous: Get logs from previous container instance.

        Returns:
            Log content as string.
        """
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"serving.kserve.io/inferenceservice={name}",
        )

        if not pods.items:
            return f"No pods found for InferenceService '{name}'"

        pod = pods.items[0]
        try:
            kwargs: dict[str, Any] = {
                "name": pod.metadata.name,
                "namespace": namespace,
                "tail_lines": tail_lines,
                "previous": previous,
            }
            if container:
                kwargs["container"] = container
            logs: str = self._k8s.core_v1.read_namespaced_pod_log(**kwargs)
            return logs
        except Exception as e:
            return f"Error getting logs: {e}"

    def get_inference_service_events(self, namespace: str, name: str) -> list[dict[str, Any]]:
        """Get Kubernetes events for an InferenceService and its pods.

        Gathers events for both the InferenceService resource itself and any
        associated pods, since pod events contain most failure info (e.g.
        ImagePullBackOff, FailedScheduling).

        Args:
            namespace: The namespace of the service.
            name: The name of the InferenceService.

        Returns:
            List of event dictionaries.
        """
        result: list[dict[str, Any]] = []

        # Events for the InferenceService itself
        isvc_events = self._k8s.core_v1.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={name}",
        )
        for event in isvc_events.items:
            result.append(self._format_event(event))

        # Events for associated pods
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"serving.kserve.io/inferenceservice={name}",
        )
        for pod in pods.items:
            pod_events = self._k8s.core_v1.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.name={pod.metadata.name}",
            )
            for event in pod_events.items:
                result.append(self._format_event(event))

        return result

    def get_inference_service_pods(self, namespace: str, name: str) -> list[dict[str, Any]]:
        """List pods for an InferenceService.

        Args:
            namespace: The namespace of the service.
            name: The name of the InferenceService.

        Returns:
            List of pod info dictionaries.
        """
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"serving.kserve.io/inferenceservice={name}",
        )

        result = []
        for pod in pods.items:
            result.append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "node": getattr(pod.spec, "node_name", None),
                    "ready": self._is_pod_ready(pod),
                }
            )

        return result

    def _is_pod_ready(self, pod: Any) -> bool:
        """Check if a pod is ready."""
        if not pod.status.conditions:
            return False
        for condition in pod.status.conditions:
            if condition.type == "Ready" and condition.status == "True":
                return True
        return False

    def _format_event(self, event: Any) -> dict[str, Any]:
        """Format a Kubernetes event into a dict."""
        return {
            "type": event.type,
            "reason": event.reason,
            "message": event.message,
            "timestamp": str(event.last_timestamp) if event.last_timestamp else None,
            "count": getattr(event, "count", 1),
        }

    def _build_inference_service_cr(self, request: InferenceServiceCreate) -> dict[str, Any]:
        """Build the InferenceService CR body from request."""
        # Build annotations
        annotations: dict[str, str] = {}
        if request.display_name:
            annotations["openshift.io/display-name"] = request.display_name

        # Build resource requirements
        resources: dict[str, dict[str, str]] = {
            "requests": {
                "cpu": request.cpu_request,
                "memory": request.memory_request,
            },
            "limits": {
                "cpu": request.cpu_limit,
                "memory": request.memory_limit,
            },
        }
        if request.gpu_count > 0:
            resources["requests"]["nvidia.com/gpu"] = str(request.gpu_count)
            resources["limits"]["nvidia.com/gpu"] = str(request.gpu_count)

        return {
            "apiVersion": InferenceCRDs.INFERENCE_SERVICE.api_version,
            "kind": InferenceCRDs.INFERENCE_SERVICE.kind,
            "metadata": {
                "name": request.name,
                "namespace": request.namespace,
                "annotations": annotations if annotations else None,
            },
            "spec": {
                "predictor": {
                    "minReplicas": request.min_replicas,
                    "maxReplicas": request.max_replicas,
                    "model": {
                        "modelFormat": {"name": request.model_format},
                        "runtime": request.runtime,
                        "storageUri": request.storage_uri,
                        "resources": resources,
                    },
                },
            },
        }
