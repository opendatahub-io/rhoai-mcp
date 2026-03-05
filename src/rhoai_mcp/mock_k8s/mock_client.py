"""Mock Kubernetes client for running without a real cluster.

Subclasses K8sClient and overrides all methods to return data
from a ClusterState instance instead of making real API calls.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from rhoai_mcp.clients.base import CRDDefinition, K8sClient
from rhoai_mcp.config import RHOAIConfig
from rhoai_mcp.mock_k8s.cluster_state import ClusterState, MockMetadata, MockResource
from rhoai_mcp.utils.errors import NotFoundError

logger = logging.getLogger(__name__)


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase for K8s field name lookup."""
    parts = name.split("_")
    if len(parts) == 1:
        return name
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class _AttrDict(dict):
    """Recursively wraps dicts so attributes can be accessed with dot notation.

    Mimics kubernetes ResourceInstance attribute access patterns.
    Subclasses dict so isinstance(x, dict) checks pass and dict()
    conversion works naturally.
    """

    def __init__(self, data: dict[str, Any] | Any) -> None:
        if isinstance(data, dict):
            super().__init__(data)
            self._data = data
            for key, value in data.items():
                object.__setattr__(self, key, _wrap(value))
        else:
            super().__init__()
            self._data = data

    def __getattr__(self, name: str) -> Any:
        # Mimic K8s ResourceInstance: support snake_case → camelCase lookup.
        # The dynamic client exposes fields in camelCase; Python code often
        # uses snake_case (e.g., access_modes → accessModes).
        camel = _snake_to_camel(name)
        if camel != name:
            try:
                return object.__getattribute__(self, camel)
            except AttributeError:
                pass
        raise AttributeError(f"'_AttrDict' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"_AttrDict({self._data!r})"


def _wrap(value: Any) -> Any:
    """Wrap a value so dicts become attribute-accessible."""
    if isinstance(value, dict):
        return _AttrDict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _resource_to_instance(resource: MockResource) -> _AttrDict:
    """Convert a MockResource to an attribute-accessible object."""
    data: dict[str, Any] = {
        "metadata": {
            "name": resource.metadata.name,
            "namespace": resource.metadata.namespace,
            "uid": resource.metadata.uid,
            "creationTimestamp": resource.metadata.creation_timestamp,
            "labels": resource.metadata.labels or {},
            "annotations": resource.metadata.annotations or {},
        },
        "spec": resource.spec,
        "status": resource.status,
        "kind": resource.kind,
        "apiVersion": resource.api_version,
        **resource.extra,
    }
    return _AttrDict(data)


def _crd_key(crd: CRDDefinition) -> str:
    """Build the state lookup key for a CRD."""
    return f"{crd.api_version}/{crd.plural}"


class MockK8sClient(K8sClient):
    """K8s client that returns data from a ClusterState instead of a real cluster."""

    def __init__(
        self,
        config_obj: RHOAIConfig | None = None,
        state: ClusterState | None = None,
    ) -> None:
        super().__init__(config_obj)
        self._state = state or ClusterState()
        self._connected = False

    def connect(self) -> None:
        """No-op connect - set connected flag and mock internal clients."""
        self._connected = True
        self._api_client = MagicMock()
        self._core_v1 = self._build_core_v1_mock()
        self._dynamic_client = MagicMock()
        logger.info("MockK8sClient connected")

    def _build_core_v1_mock(self) -> MagicMock:
        """Build a core_v1 mock that returns realistic event and log data."""
        mock = MagicMock()

        # Mock list_namespaced_event to return realistic events
        def mock_list_events(namespace: str, **kwargs: Any) -> MagicMock:
            field_selector = kwargs.get("field_selector", "")
            events_result = MagicMock()
            events_result.items = []

            # Return events for known resources
            if "failed-training-001" in field_selector:
                event = MagicMock()
                event.type = "Warning"
                event.reason = "OOMKilled"
                event.message = "Container killed due to GPU out of memory"
                event.count = 3
                event.last_timestamp = "2025-01-15T12:30:00Z"
                events_result.items = [event]

            return events_result

        mock.list_namespaced_event = mock_list_events

        # Mock read_namespaced_pod_log to return realistic logs
        def mock_read_pod_log(name: str, namespace: str, **kwargs: Any) -> str:
            if "failed" in name:
                return (
                    "torch.cuda.OutOfMemoryError: CUDA out of memory. "
                    "Tried to allocate 2.00 GiB. GPU 0 has 79.15 GiB total capacity. "
                    "After allocating 71.23 GiB, only 1.48 GiB is free.\n"
                    "Consider using gradient checkpointing or reducing batch size."
                )
            return "Training completed successfully."

        mock.read_namespaced_pod_log = mock_read_pod_log

        # Mock list_namespaced_pod to return pod info
        def mock_list_pods(namespace: str, **kwargs: Any) -> MagicMock:
            label_selector = kwargs.get("label_selector", "")
            result = MagicMock()
            result.items = []

            if "failed-training-001" in label_selector:
                pod = MagicMock()
                pod.metadata.name = "failed-training-001-worker-0"
                pod.metadata.namespace = namespace
                pod.status.phase = "Failed"
                result.items = [pod]
            elif "llama-finetune" in label_selector:
                pod = MagicMock()
                pod.metadata.name = "llama-finetune-001-worker-0"
                pod.metadata.namespace = namespace
                pod.status.phase = "Succeeded"
                result.items = [pod]

            return result

        mock.list_namespaced_pod = mock_list_pods

        # Mock list_node to return nodes with GPU resources
        def mock_list_nodes(**kwargs: Any) -> MagicMock:
            result = MagicMock()
            # Create two GPU nodes
            gpu_node = MagicMock()
            gpu_node.metadata.name = "gpu-node-1"
            gpu_node.metadata.labels = {
                "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB",
                "node-role.kubernetes.io/worker": "",
            }
            gpu_node.status.capacity = {
                "cpu": "64",
                "memory": "512Gi",
                "nvidia.com/gpu": "4",
            }
            gpu_node.status.allocatable = {
                "cpu": "62",
                "memory": "500Gi",
                "nvidia.com/gpu": "4",
            }

            gpu_node2 = MagicMock()
            gpu_node2.metadata.name = "gpu-node-2"
            gpu_node2.metadata.labels = {
                "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB",
                "node-role.kubernetes.io/worker": "",
            }
            gpu_node2.status.capacity = {
                "cpu": "64",
                "memory": "512Gi",
                "nvidia.com/gpu": "4",
            }
            gpu_node2.status.allocatable = {
                "cpu": "62",
                "memory": "500Gi",
                "nvidia.com/gpu": "4",
            }

            result.items = [gpu_node, gpu_node2]
            return result

        mock.list_node = mock_list_nodes

        return mock

    def disconnect(self) -> None:
        """No-op disconnect."""
        self._connected = False
        self._core_v1 = None
        self._dynamic_client = None
        self._crd_cache.clear()
        logger.info("MockK8sClient disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # --- CRD resource operations ---

    def get_resource(self, crd: CRDDefinition) -> Any:
        """Return a mock resource handle. Always succeeds (all CRDs 'exist')."""
        cache_key = _crd_key(crd)
        if cache_key not in self._crd_cache:
            self._crd_cache[cache_key] = MagicMock(name=f"Resource({cache_key})")
        return self._crd_cache[cache_key]

    def get(
        self,
        crd: CRDDefinition,
        name: str,
        namespace: str | None = None,
    ) -> Any:
        """Get a single CRD resource by name."""
        key = _crd_key(crd)
        resources = self._state.resources.get(key, [])
        for r in resources:
            if r.metadata.name == name:
                if namespace is None or r.metadata.namespace == namespace:
                    return _resource_to_instance(r)
        raise NotFoundError(crd.kind, name, namespace)

    def list_resources(
        self,
        crd: CRDDefinition,
        namespace: str | None = None,
        label_selector: str | None = None,
        field_selector: str | None = None,
    ) -> list[Any]:
        """List CRD resources, optionally filtered by namespace."""
        key = _crd_key(crd)
        resources = self._state.resources.get(key, [])
        results = []
        for r in resources:
            if namespace and r.metadata.namespace and r.metadata.namespace != namespace:
                continue
            results.append(_resource_to_instance(r))
        return results

    def create(
        self,
        crd: CRDDefinition,
        body: dict[str, Any],
        namespace: str | None = None,
    ) -> Any:
        """Mock create - persist resource in state and return it."""
        metadata = body.get("metadata", {})
        metadata.setdefault("labels", {})
        metadata.setdefault("annotations", {})
        metadata.setdefault("uid", "mock-uid-created")
        metadata.setdefault("creationTimestamp", "2025-01-15T10:00:00Z")
        body["metadata"] = metadata

        # Add to state so subsequent get/list calls find it
        ns = namespace or metadata.get("namespace")
        resource = MockResource(
            metadata=MockMetadata(
                name=metadata.get("name", ""),
                namespace=ns,
                uid=metadata.get("uid", ""),
                labels=metadata.get("labels", {}),
                annotations=metadata.get("annotations", {}),
            ),
            spec=body.get("spec", {}),
            status=body.get("status", {}),
            kind=crd.kind,
            api_version=crd.api_version,
        )
        key = _crd_key(crd)
        self._state.resources.setdefault(key, []).append(resource)

        return _AttrDict(body)

    def delete(
        self,
        crd: CRDDefinition,
        name: str,
        namespace: str | None = None,
    ) -> None:
        """Mock delete - no-op."""

    def patch(
        self,
        crd: CRDDefinition,
        name: str,
        body: dict[str, Any],
        namespace: str | None = None,
    ) -> Any:
        """Mock patch - return a merged result."""
        try:
            existing = self.get(crd, name, namespace)
            return existing
        except NotFoundError:
            return _AttrDict(body)

    # --- Project operations ---

    def list_projects(
        self,
        label_selector: str | None = None,
    ) -> list[Any]:
        return [_resource_to_instance(p) for p in self._state.projects]

    def patch_project(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Any:
        for p in self._state.projects:
            if p.metadata.name == name:
                return _resource_to_instance(p)
        raise NotFoundError("Project", name)

    # --- Namespace operations ---

    def get_namespace(self, name: str) -> Any:
        for ns in self._state.namespaces:
            if ns.metadata.name == name:
                return _resource_to_instance(ns)
        raise NotFoundError("Namespace", name)

    def list_namespaces(
        self,
        label_selector: str | None = None,
    ) -> list[Any]:
        return [_resource_to_instance(ns) for ns in self._state.namespaces]

    def create_namespace(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Any:
        return _AttrDict(
            {"metadata": {"name": name, "labels": labels or {}, "annotations": annotations or {}}}
        )

    def delete_namespace(self, name: str) -> None:
        pass

    def patch_namespace(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Any:
        return self.get_namespace(name)

    # --- Secret operations ---

    def get_secret(self, name: str, namespace: str) -> Any:
        for s in self._state.secrets:
            if s.metadata.name == name and (
                s.metadata.namespace is None or s.metadata.namespace == namespace
            ):
                return _resource_to_instance(s)
        raise NotFoundError("Secret", name, namespace)

    def list_secrets(
        self,
        namespace: str,
        label_selector: str | None = None,
    ) -> list[Any]:
        return [
            _resource_to_instance(s)
            for s in self._state.secrets
            if s.metadata.namespace is None or s.metadata.namespace == namespace
        ]

    def create_secret(
        self,
        name: str,
        namespace: str,
        data: dict[str, str],
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        string_data: bool = True,
    ) -> Any:
        return _AttrDict(
            {"metadata": {"name": name, "namespace": namespace, "labels": labels or {}}}
        )

    def delete_secret(self, name: str, namespace: str) -> None:
        pass

    # --- PVC operations ---

    def get_pvc(self, name: str, namespace: str) -> Any:
        for p in self._state.pvcs:
            if p.metadata.name == name and (
                p.metadata.namespace is None or p.metadata.namespace == namespace
            ):
                return _resource_to_instance(p)
        raise NotFoundError("PersistentVolumeClaim", name, namespace)

    def list_pvcs(
        self,
        namespace: str,
        label_selector: str | None = None,
    ) -> list[Any]:
        return [
            _resource_to_instance(p)
            for p in self._state.pvcs
            if p.metadata.namespace is None or p.metadata.namespace == namespace
        ]

    def create_pvc(
        self,
        name: str,
        namespace: str,
        size: str,
        access_modes: list[str] | None = None,
        storage_class: str | None = None,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Any:
        return _AttrDict(
            {
                "metadata": {"name": name, "namespace": namespace},
                "spec": {
                    "accessModes": access_modes or ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": size}},
                },
            }
        )

    def delete_pvc(self, name: str, namespace: str) -> None:
        pass
