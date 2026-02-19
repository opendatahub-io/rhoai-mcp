"""Mock Kubernetes client for evaluation tests.

Subclasses K8sClient and overrides all methods to return data
from a ClusterState instance instead of making real API calls.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from rhoai_mcp.clients.base import CRDDefinition, K8sClient
from rhoai_mcp.config import RHOAIConfig
from rhoai_mcp.utils.errors import NotFoundError

from evals.mock_k8s.cluster_state import ClusterState, MockResource

logger = logging.getLogger(__name__)


class _AttrDict:
    """Recursively wraps dicts so attributes can be accessed with dot notation.

    Mimics kubernetes ResourceInstance attribute access patterns.
    """

    def __init__(self, data: dict[str, Any] | Any) -> None:
        if isinstance(data, dict):
            self._data = data
            for key, value in data.items():
                setattr(self, key, _wrap(value))
        else:
            self._data = data

    def __getattr__(self, name: str) -> Any:
        # Return None for missing attributes (like K8s ResourceInstance)
        return None

    def __repr__(self) -> str:
        return f"_AttrDict({self._data!r})"

    def __iter__(self) -> Any:
        if isinstance(self._data, dict):
            return iter(self._data)
        raise TypeError(f"_AttrDict wrapping {type(self._data)} is not iterable")

    def items(self) -> Any:
        if isinstance(self._data, dict):
            return self._data.items()
        raise TypeError("items() on non-dict _AttrDict")

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self._data, dict):
            val = self._data.get(key, default)
            return _wrap(val) if val is not default else default
        return default

    def __contains__(self, item: Any) -> bool:
        if isinstance(self._data, dict):
            return item in self._data
        return False

    def __bool__(self) -> bool:
        return bool(self._data)


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
        """No-op connect - just set connected flag."""
        self._connected = True
        self._core_v1 = MagicMock()
        self._dynamic_client = MagicMock()
        logger.info("MockK8sClient connected")

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
        """Mock create - just return the body as an instance."""
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
