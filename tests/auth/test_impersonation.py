"""Tests for K8s client impersonation."""

from unittest.mock import MagicMock, patch

import pytest
from urllib3._collections import HTTPHeaderDict

from rhoai_mcp.clients.base import K8sClient
from rhoai_mcp.config import AuthMode, RHOAIConfig


@pytest.fixture
def connected_client():
    """Create a K8sClient with mocked internals."""
    cfg = RHOAIConfig(
        auth_mode=AuthMode.TOKEN,
        api_server="https://api.test:6443",
        api_token="sa-token",
    )
    k8s_client = K8sClient(cfg)
    # Mock the API client directly
    mock_api_client = MagicMock()
    mock_api_client.configuration = MagicMock()
    mock_api_client.configuration.host = "https://api.test:6443"
    mock_api_client.configuration.api_key = {"authorization": "Bearer sa-token"}
    mock_api_client.configuration.verify_ssl = True
    mock_api_client.default_headers = HTTPHeaderDict()
    k8s_client._api_client = mock_api_client
    k8s_client._dynamic_client = MagicMock()
    k8s_client._core_v1 = MagicMock()
    return k8s_client


@pytest.fixture(autouse=True)
def _patch_k8s_constructors():
    """Patch DynamicClient and CoreV1Api so copy-based impersonation works with mocks."""
    with (
        patch("rhoai_mcp.clients.base.DynamicClient"),
        patch("rhoai_mcp.clients.base.client.CoreV1Api"),
    ):
        yield


class TestImpersonation:
    def test_create_impersonating_client_sets_user_header(self, connected_client):
        imp = connected_client.create_impersonating_client("alice", ["team-a"])
        assert imp._api_client.default_headers["Impersonate-User"] == "alice"

    def test_create_impersonating_client_sets_separate_group_headers(self, connected_client):
        """Each group must be a separate Impersonate-Group header per K8s API spec."""
        imp = connected_client.create_impersonating_client("alice", ["team-a", "team-b"])
        headers = imp._api_client.default_headers
        # HTTPHeaderDict stores multi-value headers properly
        all_groups = headers.getlist("Impersonate-Group")
        assert all_groups == ["team-a", "team-b"]

    def test_create_impersonating_client_empty_groups(self, connected_client):
        imp = connected_client.create_impersonating_client("alice", [])
        assert imp._api_client.default_headers["Impersonate-User"] == "alice"
        assert "Impersonate-Group" not in imp._api_client.default_headers

    def test_create_impersonating_client_is_connected(self, connected_client):
        imp = connected_client.create_impersonating_client("alice", [])
        assert imp.is_connected

    def test_header_injection_rejected(self, connected_client):
        """CWE-113: Reject control characters in header values."""
        with pytest.raises(ValueError, match="control characters"):
            connected_client.create_impersonating_client("alice\r\nX-Evil: yes", [])

    def test_empty_username_rejected(self, connected_client):
        with pytest.raises(ValueError, match="empty value"):
            connected_client.create_impersonating_client("  ", [])

    def test_username_with_whitespace_rejected(self, connected_client):
        """CWE-20: Reject leading/trailing whitespace instead of silently stripping."""
        with pytest.raises(ValueError, match="whitespace"):
            connected_client.create_impersonating_client(" alice ", [])

    def test_header_injection_in_group_rejected(self, connected_client):
        """CWE-113: Reject control characters in group header values."""
        with pytest.raises(ValueError, match="control characters"):
            connected_client.create_impersonating_client("alice", ["team-a\r\nX-Evil: yes"])

    def test_empty_group_rejected(self, connected_client):
        """Reject empty group values."""
        with pytest.raises(ValueError, match="empty value"):
            connected_client.create_impersonating_client("alice", ["team-a", "  "])

    def test_inherited_impersonation_headers_cleared(self, connected_client):
        """CWE-269: Inherited impersonation headers must not carry over."""
        # Pre-set impersonation headers on the source client
        connected_client._api_client.default_headers["Impersonate-User"] = "old-user"
        connected_client._api_client.default_headers.add("Impersonate-Group", "old-group")

        imp = connected_client.create_impersonating_client("alice", ["team-a"])
        assert imp._api_client.default_headers["Impersonate-User"] == "alice"
        all_groups = imp._api_client.default_headers.getlist("Impersonate-Group")
        assert all_groups == ["team-a"]

    def test_create_impersonating_client_raises_when_not_connected(self):
        cfg = RHOAIConfig(
            auth_mode=AuthMode.TOKEN,
            api_server="https://api.test:6443",
            api_token="x",
        )
        k8s_client = K8sClient(cfg)
        with pytest.raises(RuntimeError, match="not connected"):
            k8s_client.create_impersonating_client("alice", [])
