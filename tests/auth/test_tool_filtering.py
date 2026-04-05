"""Tests for per-user tool list filtering."""

from unittest.mock import MagicMock, patch

from rhoai_mcp.auth.user_context import UserContext
from rhoai_mcp.config import RHOAIConfig, TransportMode
from rhoai_mcp.server import RHOAIServer


class TestToolFiltering:
    def test_get_allowed_tools_returns_all_when_oidc_disabled(self) -> None:
        config = RHOAIConfig(oidc_enabled=False, mock_cluster=True)
        server = RHOAIServer(config)
        server.create_mcp()
        # When OIDC disabled, all tools should be allowed
        assert server.get_allowed_tools() is None  # None means "all"

    def test_get_allowed_tools_checks_rbac_when_oidc_enabled(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_k8s = MagicMock()
        mock_k8s.is_connected = True
        server._k8s_client = mock_k8s

        # Set up a user context
        ctx = UserContext(username="alice", groups=["team-a"])
        token = UserContext.set_current(ctx)

        try:
            with patch("rhoai_mcp.auth.rbac.RBACChecker") as MockChecker:
                checker_instance = MagicMock()
                checker_instance.filter_tools.return_value = {
                    "list_data_science_projects"
                }
                MockChecker.return_value = checker_instance

                with patch.object(server, "_plugin_manager") as mock_pm:
                    mock_pm.collect_tool_permissions.return_value = {
                        "list_data_science_projects": [
                            {
                                "apiGroup": "project.openshift.io",
                                "resource": "projects",
                                "verb": "list",
                            }
                        ],
                        "delete_data_science_project": [
                            {
                                "apiGroup": "",
                                "resource": "namespaces",
                                "verb": "delete",
                            }
                        ],
                    }
                    result = server.get_allowed_tools()
                    assert result == {"list_data_science_projects"}
        finally:
            UserContext.reset_current(token)

    def test_get_allowed_tools_returns_empty_set_when_no_user_context(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        server._k8s_client = MagicMock()
        server._plugin_manager = MagicMock()
        # No user context set
        result = server.get_allowed_tools()
        assert result == set()

    def test_get_allowed_tools_raises_when_no_plugin_manager(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        server._k8s_client = MagicMock()

        ctx = UserContext(username="alice", groups=[])
        token = UserContext.set_current(ctx)
        try:
            with pytest.raises(RuntimeError, match="plugin_manager not initialized"):
                server.get_allowed_tools()
        finally:
            UserContext.reset_current(token)
