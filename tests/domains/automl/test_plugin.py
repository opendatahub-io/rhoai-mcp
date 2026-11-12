"""Tests for AutoML plugin integration details."""

from unittest.mock import MagicMock

from rhoai_mcp.domains.registry import AutoMLPlugin


def test_health_check_uses_shared_kubernetes_client() -> None:
    server = MagicMock()

    healthy, message = AutoMLPlugin().rhoai_health_check(server)

    assert healthy is True
    assert "available" in message
    server.get_shared_k8s.assert_called_once_with()
    server.get_k8s.assert_not_called()
