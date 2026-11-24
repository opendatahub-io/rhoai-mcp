"""Tests for the QuickstartsPlugin health check."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from kubernetes.client import ApiException

from rhoai_mcp.domains.registry import (
    QUICKSTART_INSTALLER_RBAC_MANIFEST,
    QuickstartsPlugin,
)
from rhoai_mcp.utils.errors import NotFoundError


def _server(k8s: MagicMock) -> SimpleNamespace:
    """Build a server double with quickstart namespace/SA config and a k8s client."""
    return SimpleNamespace(
        config=SimpleNamespace(
            quickstart_job_namespace="openshift-quickstarts",
            quickstart_installer_service_account="quickstart-installer",
        ),
        k8s=k8s,
    )


class TestHealthCheck:
    def test_healthy_when_namespace_and_sa_present(self) -> None:
        k8s = MagicMock()
        server = _server(k8s)

        healthy, message = QuickstartsPlugin().rhoai_health_check(server)

        assert healthy
        assert "openshift-quickstarts" in message
        assert "quickstart-installer" in message
        k8s.get_namespace.assert_called_once_with("openshift-quickstarts")
        k8s.core_v1.read_namespaced_service_account.assert_called_once_with(
            name="quickstart-installer", namespace="openshift-quickstarts"
        )

    def test_unhealthy_when_namespace_missing(self) -> None:
        k8s = MagicMock()
        k8s.get_namespace.side_effect = NotFoundError("Namespace", "openshift-quickstarts")
        server = _server(k8s)

        healthy, message = QuickstartsPlugin().rhoai_health_check(server)

        assert not healthy
        assert "namespace" in message
        assert QUICKSTART_INSTALLER_RBAC_MANIFEST in message
        # Should not attempt the SA read once the namespace is known absent.
        k8s.core_v1.read_namespaced_service_account.assert_not_called()

    def test_unhealthy_when_service_account_missing(self) -> None:
        k8s = MagicMock()
        k8s.core_v1.read_namespaced_service_account.side_effect = ApiException(status=404)
        server = _server(k8s)

        healthy, message = QuickstartsPlugin().rhoai_health_check(server)

        assert not healthy
        assert "service account" in message
        assert QUICKSTART_INSTALLER_RBAC_MANIFEST in message

    def test_degrades_gracefully_when_namespace_unverifiable(self) -> None:
        k8s = MagicMock()
        k8s.get_namespace.side_effect = RuntimeError("Server not running")
        server = _server(k8s)

        healthy, message = QuickstartsPlugin().rhoai_health_check(server)

        assert healthy
        assert "could not verify" in message

    def test_degrades_gracefully_when_sa_read_forbidden(self) -> None:
        k8s = MagicMock()
        forbidden = ApiException(status=403)
        forbidden.reason = "Forbidden"
        k8s.core_v1.read_namespaced_service_account.side_effect = forbidden
        server = _server(k8s)

        healthy, message = QuickstartsPlugin().rhoai_health_check(server)

        assert healthy
        assert "could not verify" in message
