"""Tests for Model Registry auto-discovery."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from kubernetes.client import ApiException  # type: ignore[import-untyped]

from rhoai_mcp.domains.model_registry.discovery import (
    COMMON_NAMESPACES,
    DiscoveredModelRegistry,
    ModelRegistryDiscovery,
)


class TestDiscoveredModelRegistry:
    """Tests for DiscoveredModelRegistry dataclass."""

    def test_str_representation(self) -> None:
        """Test string representation includes URL and source."""
        result = DiscoveredModelRegistry(
            url="http://model-catalog.rhoai-model-registries.svc:8080",
            namespace="rhoai-model-registries",
            service_name="model-catalog",
            port=8080,
            source="crd",
            requires_auth=False,
        )
        assert "http://model-catalog.rhoai-model-registries.svc:8080" in str(result)
        assert "crd" in str(result)

    def test_requires_auth_flag(self) -> None:
        """Test requires_auth flag for different ports."""
        # Port 8080 should not require auth
        result_8080 = DiscoveredModelRegistry(
            url="http://test.svc:8080",
            namespace="test",
            service_name="test",
            port=8080,
            source="test",
            requires_auth=False,
        )
        assert not result_8080.requires_auth

        # Port 8443 typically requires auth (kube-rbac-proxy)
        result_8443 = DiscoveredModelRegistry(
            url="https://test.svc:8443",
            namespace="test",
            service_name="test",
            port=8443,
            source="test",
            requires_auth=True,
        )
        assert result_8443.requires_auth


class TestModelRegistryDiscovery:
    """Tests for ModelRegistryDiscovery class."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        k8s = MagicMock()
        k8s.core_v1 = MagicMock()
        return k8s

    @pytest.fixture
    def discovery(self, mock_k8s: MagicMock) -> ModelRegistryDiscovery:
        """Create a ModelRegistryDiscovery instance."""
        return ModelRegistryDiscovery(mock_k8s)

    def _make_mock_service(
        self, name: str, namespace: str, ports: list[int]
    ) -> MagicMock:
        """Create a mock Kubernetes service."""
        svc = MagicMock()
        svc.metadata.name = name
        svc.metadata.namespace = namespace
        svc.spec.ports = [
            MagicMock(port=p) for p in ports
        ]
        return svc

    def _make_mock_component(self, registries_namespace: str | None) -> MagicMock:
        """Create a mock ModelRegistry component resource."""
        component = MagicMock()
        if registries_namespace:
            component.spec.registriesNamespace = registries_namespace
        else:
            component.spec = None
        return component

    def test_discover_from_crd_success(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test successful discovery from ModelRegistry CRD."""
        # Setup CRD response
        mock_k8s.list_resources.return_value = [
            self._make_mock_component("rhoai-model-registries")
        ]

        # Setup service response
        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-catalog", "rhoai-model-registries", [8080, 8443])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        assert result is not None
        assert result.url == "http://model-catalog.rhoai-model-registries.svc:8080"
        assert result.namespace == "rhoai-model-registries"
        assert result.service_name == "model-catalog"
        assert result.port == 8080
        assert result.source == "crd"
        assert not result.requires_auth

    def test_discover_port_8080_preferred_over_8443(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test that port 8080 is preferred over 8443."""
        mock_k8s.list_resources.return_value = [
            self._make_mock_component("test-ns")
        ]

        # Service with both ports (8443 first to test preference)
        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-registry", "test-ns", [8443, 8080])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        assert result is not None
        assert result.port == 8080
        assert result.url.startswith("http://")  # HTTP for 8080
        assert not result.requires_auth

    def test_discover_uses_8443_when_8080_unavailable(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test that 8443 is used when 8080 is not available."""
        mock_k8s.list_resources.return_value = [
            self._make_mock_component("test-ns")
        ]

        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-catalog", "test-ns", [8443])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        assert result is not None
        assert result.port == 8443
        assert result.url.startswith("https://")  # HTTPS for 8443
        assert result.requires_auth

    def test_discover_fallback_to_namespace_scan(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test fallback to namespace scan when CRD not found."""
        # CRD not found
        mock_k8s.list_resources.return_value = []

        # First namespace has no matching services
        def service_response(namespace: str) -> MagicMock:
            svc_list = MagicMock()
            if namespace == "rhoai-model-registries":
                svc_list.items = [
                    self._make_mock_service("model-catalog", namespace, [8080])
                ]
            else:
                svc_list.items = []
            return svc_list

        mock_k8s.core_v1.list_namespaced_service.side_effect = service_response

        result = discovery.discover()

        assert result is not None
        assert result.source == "namespace_scan"
        assert result.namespace == "rhoai-model-registries"

    def test_discover_fallback_to_configured_url(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test fallback to configured URL when discovery fails."""
        # No CRD
        mock_k8s.list_resources.return_value = []

        # No services in any namespace
        svc_list = MagicMock()
        svc_list.items = []
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        fallback = "http://custom-registry.custom-ns.svc:9000"
        result = discovery.discover(fallback_url=fallback)

        assert result is not None
        assert result.url == fallback
        assert result.source == "fallback"

    def test_discover_returns_none_when_no_registry_and_no_fallback(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test returns None when no registry found and no fallback."""
        mock_k8s.list_resources.return_value = []

        svc_list = MagicMock()
        svc_list.items = []
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover(fallback_url=None)

        assert result is None

    def test_discover_handles_crd_api_exception(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test graceful handling of CRD API errors."""
        mock_k8s.list_resources.side_effect = ApiException(status=404)

        # Should fall back to namespace scan
        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-registry", "odh-model-registries", [8080])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        assert result is not None
        assert result.source == "namespace_scan"

    def test_discover_handles_namespace_forbidden(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test handling of forbidden namespace access."""
        mock_k8s.list_resources.return_value = []

        def raise_forbidden(namespace: str) -> Any:
            if namespace == COMMON_NAMESPACES[0]:
                raise ApiException(status=403)
            svc_list = MagicMock()
            svc_list.items = [
                self._make_mock_service("model-catalog", namespace, [8080])
            ]
            return svc_list

        mock_k8s.core_v1.list_namespaced_service.side_effect = raise_forbidden

        result = discovery.discover()

        # Should skip forbidden namespace and try next
        assert result is not None
        assert result.namespace != COMMON_NAMESPACES[0]

    def test_discover_matches_service_name_patterns(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test that various service name patterns are matched."""
        mock_k8s.list_resources.return_value = []

        patterns_to_test = [
            "model-catalog",
            "my-model-registry",
            "modelregistry-service",
        ]

        for pattern in patterns_to_test:
            svc_list = MagicMock()
            svc_list.items = [
                self._make_mock_service(pattern, "rhoai-model-registries", [8080])
            ]
            mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

            result = discovery.discover()

            assert result is not None, f"Pattern '{pattern}' should be matched"
            assert result.service_name == pattern

    def test_discover_ignores_unrelated_services(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test that unrelated services are ignored."""
        mock_k8s.list_resources.return_value = []

        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("postgresql", "rhoai-model-registries", [5432]),
            self._make_mock_service("minio", "rhoai-model-registries", [9000]),
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover(fallback_url=None)

        assert result is None

    def test_discover_component_without_spec(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test handling of component without spec."""
        component = MagicMock()
        component.spec = None
        mock_k8s.list_resources.return_value = [component]

        # Should fall back to namespace scan
        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-catalog", "odh-model-registries", [8080])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        assert result is not None
        assert result.source == "namespace_scan"


class TestRouteDiscovery:
    """Tests for Route discovery when running outside the cluster."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        k8s = MagicMock()
        k8s.core_v1 = MagicMock()
        return k8s

    @pytest.fixture
    def discovery(self, mock_k8s: MagicMock) -> ModelRegistryDiscovery:
        """Create a ModelRegistryDiscovery instance."""
        return ModelRegistryDiscovery(mock_k8s)

    def _make_mock_service(
        self, name: str, namespace: str, ports: list[int]
    ) -> MagicMock:
        """Create a mock Kubernetes service."""
        svc = MagicMock()
        svc.metadata.name = name
        svc.metadata.namespace = namespace
        svc.spec.ports = [MagicMock(port=p) for p in ports]
        return svc

    def _make_mock_component(self, registries_namespace: str | None) -> MagicMock:
        """Create a mock ModelRegistry component resource."""
        component = MagicMock()
        if registries_namespace:
            component.spec.registriesNamespace = registries_namespace
        else:
            component.spec = None
        return component

    def _make_mock_route(
        self,
        name: str,
        namespace: str,
        target_service: str,
        host: str,
        is_admitted: bool = True,
        has_tls: bool = True,
    ) -> MagicMock:
        """Create a mock OpenShift Route.

        Args:
            name: Route name.
            namespace: Route namespace.
            target_service: Name of the service the Route points to.
            host: External hostname for the Route.
            is_admitted: Whether the Route is admitted (ready to serve).
            has_tls: Whether the Route uses TLS.

        Returns:
            Mock Route object.
        """
        route = MagicMock()
        route.metadata.name = name
        route.metadata.namespace = namespace

        # spec.to.kind and spec.to.name
        route.spec.to.kind = "Service"
        route.spec.to.name = target_service

        # spec.tls
        if has_tls:
            route.spec.tls = MagicMock()
        else:
            route.spec.tls = None

        # status.ingress[].conditions and host
        condition = MagicMock()
        condition.type = "Admitted"
        condition.status = "True" if is_admitted else "False"

        ingress = MagicMock()
        ingress.host = host
        ingress.conditions = [condition]

        route.status.ingress = [ingress]

        return route

    def test_discover_uses_route_when_outside_cluster(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery, monkeypatch: Any
    ) -> None:
        """Test that Route is used when running outside the cluster."""
        # Mock running outside cluster
        monkeypatch.setattr(
            "rhoai_mcp.domains.model_registry.discovery._is_running_in_cluster",
            lambda: False,
        )

        # Setup CRD response
        mock_k8s.list_resources.side_effect = lambda crd, ns=None: (
            [self._make_mock_component("rhoai-model-registries")]
            if ns is None
            else [
                self._make_mock_route(
                    "model-catalog",
                    "rhoai-model-registries",
                    "model-catalog",
                    "model-catalog.apps.cluster.example.com",
                )
            ]
        )

        # Setup service response
        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-catalog", "rhoai-model-registries", [8443])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        assert result is not None
        assert result.url == "https://model-catalog.apps.cluster.example.com"
        assert result.is_external is True
        assert result.route_name == "model-catalog"
        assert result.source == "crd_route"
        assert result.requires_auth is True

    def test_discover_uses_internal_url_when_in_cluster(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery, monkeypatch: Any
    ) -> None:
        """Test that internal URL is used when running inside the cluster."""
        # Mock running inside cluster
        monkeypatch.setattr(
            "rhoai_mcp.domains.model_registry.discovery._is_running_in_cluster",
            lambda: True,
        )

        # Setup CRD response
        mock_k8s.list_resources.return_value = [
            self._make_mock_component("rhoai-model-registries")
        ]

        # Setup service response
        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-catalog", "rhoai-model-registries", [8080])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        assert result is not None
        assert result.url == "http://model-catalog.rhoai-model-registries.svc:8080"
        assert result.is_external is False
        assert result.route_name is None
        assert result.source == "crd"

    def test_discover_falls_back_to_internal_when_no_route(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery, monkeypatch: Any
    ) -> None:
        """Test fallback to internal URL when running outside cluster but no Route exists."""
        # Mock running outside cluster
        monkeypatch.setattr(
            "rhoai_mcp.domains.model_registry.discovery._is_running_in_cluster",
            lambda: False,
        )

        # Setup CRD response - returns component for CRD query, empty for Route query
        mock_k8s.list_resources.side_effect = lambda crd, ns=None: (
            [self._make_mock_component("rhoai-model-registries")] if ns is None else []
        )

        # Setup service response
        svc_list = MagicMock()
        svc_list.items = [
            self._make_mock_service("model-catalog", "rhoai-model-registries", [8443])
        ]
        mock_k8s.core_v1.list_namespaced_service.return_value = svc_list

        result = discovery.discover()

        # Should fall back to internal URL with warning logged
        assert result is not None
        assert result.url == "https://model-catalog.rhoai-model-registries.svc:8443"
        assert result.is_external is False
        assert result.source == "crd"

    def test_find_route_for_service_skips_non_admitted(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test that non-admitted Routes are skipped."""
        # Route exists but is not admitted
        mock_k8s.list_resources.return_value = [
            self._make_mock_route(
                "model-catalog",
                "rhoai-model-registries",
                "model-catalog",
                "model-catalog.apps.cluster.example.com",
                is_admitted=False,
            )
        ]

        result = discovery._find_route_for_service(
            "model-catalog", "rhoai-model-registries"
        )

        assert result is None

    def test_find_route_for_service_handles_http_routes(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test that HTTP Routes (without TLS) produce http:// URLs."""
        # Route without TLS
        mock_k8s.list_resources.return_value = [
            self._make_mock_route(
                "model-catalog-http",
                "rhoai-model-registries",
                "model-catalog",
                "model-catalog.apps.cluster.example.com",
                has_tls=False,
            )
        ]

        result = discovery._find_route_for_service(
            "model-catalog", "rhoai-model-registries"
        )

        assert result is not None
        route_name, external_url = result
        assert external_url == "http://model-catalog.apps.cluster.example.com"
        assert route_name == "model-catalog-http"

    def test_find_route_for_service_skips_wrong_target(
        self, mock_k8s: MagicMock, discovery: ModelRegistryDiscovery
    ) -> None:
        """Test that Routes targeting a different service are skipped."""
        # Route targets a different service
        mock_k8s.list_resources.return_value = [
            self._make_mock_route(
                "other-service-route",
                "rhoai-model-registries",
                "other-service",  # Different service
                "other-service.apps.cluster.example.com",
            )
        ]

        result = discovery._find_route_for_service(
            "model-catalog", "rhoai-model-registries"
        )

        assert result is None

    def test_new_fields_have_correct_defaults(self) -> None:
        """Test that new fields is_external and route_name have correct defaults."""
        result = DiscoveredModelRegistry(
            url="http://test.svc:8080",
            namespace="test",
            service_name="test",
            port=8080,
            source="test",
        )
        assert result.is_external is False
        assert result.route_name is None
