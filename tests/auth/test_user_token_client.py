"""Tests for K8sClient.create_user_token_client()."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from rhoai_mcp.clients.base import K8sClient
from rhoai_mcp.config import RHOAIConfig


@pytest.fixture(autouse=True)
def _patch_k8s_constructors():
    """Patch DynamicClient and CoreV1Api so they don't attempt real connections."""
    with (
        patch("rhoai_mcp.clients.base.DynamicClient"),
        patch("rhoai_mcp.clients.base.client.CoreV1Api"),
    ):
        yield


class TestCreateUserTokenClient:
    @pytest.fixture
    def connected_client(self):
        """Create a K8sClient with a mock API client."""
        config = RHOAIConfig(mock_cluster=True)
        k8s = K8sClient(config)

        mock_api_client = MagicMock()
        mock_api_client.configuration.host = "https://api.cluster.example.com:6443"
        mock_api_client.configuration.ssl_ca_cert = "/var/run/secrets/ca.crt"
        mock_api_client.configuration.verify_ssl = True
        k8s._api_client = mock_api_client

        return k8s

    def test_returns_client_with_user_token(self, connected_client):
        token = SecretStr("user-bearer-token-123")
        result = connected_client.create_user_token_client(token)

        assert result._api_client is not None
        assert result._api_client.configuration.api_key == {
            "authorization": "Bearer user-bearer-token-123"
        }

    def test_inherits_host(self, connected_client):
        token = SecretStr("token")
        result = connected_client.create_user_token_client(token)

        assert result._api_client.configuration.host == "https://api.cluster.example.com:6443"

    def test_inherits_ssl_ca_cert(self, connected_client):
        token = SecretStr("token")
        result = connected_client.create_user_token_client(token)

        assert result._api_client.configuration.ssl_ca_cert == "/var/run/secrets/ca.crt"

    def test_inherits_ssl_ca_cert_none(self):
        """Verify ssl_ca_cert=None doesn't break the new client."""
        config = RHOAIConfig(mock_cluster=True)
        k8s = K8sClient(config)

        mock_api_client = MagicMock()
        mock_api_client.configuration.host = "https://api.example.com:6443"
        mock_api_client.configuration.ssl_ca_cert = None
        mock_api_client.configuration.verify_ssl = True
        k8s._api_client = mock_api_client

        token = SecretStr("token")
        result = k8s.create_user_token_client(token)
        assert result._api_client.configuration.ssl_ca_cert is None

    def test_inherits_verify_ssl(self, connected_client):
        token = SecretStr("token")
        result = connected_client.create_user_token_client(token)

        assert result._api_client.configuration.verify_ssl is True

    def test_raises_when_disconnected(self):
        config = RHOAIConfig(mock_cluster=True)
        k8s = K8sClient(config)
        # _api_client is None by default

        token = SecretStr("token")
        with pytest.raises(RuntimeError, match="client not connected"):
            k8s.create_user_token_client(token)

    def test_returned_client_is_independent(self, connected_client):
        """Verify modifying the returned client doesn't affect the SA client."""
        token = SecretStr("token")
        result = connected_client.create_user_token_client(token)

        # The returned client should have its own api_client
        assert result._api_client is not connected_client._api_client
        # Modifying the new client's config shouldn't affect the original
        result._api_client.configuration.host = "https://different.host:6443"
        assert (
            connected_client._api_client.configuration.host
            == "https://api.cluster.example.com:6443"
        )

    def test_dynamic_client_is_created(self, connected_client):
        token = SecretStr("token")
        result = connected_client.create_user_token_client(token)

        assert result._dynamic_client is not None

    def test_core_v1_is_created(self, connected_client):
        token = SecretStr("token")
        result = connected_client.create_user_token_client(token)

        assert result._core_v1 is not None
