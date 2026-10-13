"""End-to-end MaaS tests against the mock cluster state.

Exercises the MaaS tools through MockK8sClient (the same path used by
`rhoai-mcp --mock-cluster` / start-mcp.sh) rather than MagicMock, proving
the check -> deploy -> status workflow works against the mock platform.
"""

from typing import Any
from unittest.mock import MagicMock

from rhoai_mcp.config import AuthMode, RHOAIConfig
from rhoai_mcp.mock_k8s.cluster_state import create_default_cluster_state
from rhoai_mcp.mock_k8s.mock_client import MockK8sClient


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register MaaS tools and return captured tool functions."""
    from rhoai_mcp.domains.maas.tools import register_tools

    mcp = MagicMock()
    registered_tools: dict[str, Any] = {}

    def capture_tool() -> Any:
        def decorator(func: Any) -> Any:
            registered_tools[func.__name__] = func
            return func

        return decorator

    mcp.tool = capture_tool
    register_tools(mcp, mock_server)
    return registered_tools


def _make_server() -> MagicMock:
    """Build a mock server backed by the real mock cluster state."""
    server = MagicMock()
    server.config = RHOAIConfig(
        auth_mode=AuthMode.KUBECONFIG,
        enable_dangerous_operations=True,
        read_only_mode=False,
    )
    server.k8s = MockK8sClient(state=create_default_cluster_state())
    return server


class TestMaasMockClusterWorkflow:
    """The check -> deploy -> status workflow against the mock cluster."""

    def test_readiness_passes_on_mock_cluster(self) -> None:
        tools = _register_tools(_make_server())

        result = tools["check_maas_readiness"]()

        assert result["ready"] is True
        assert result["issues"] is None
        assert result["maas_endpoint"] == "https://maas.apps.mock-cluster.local"

    def test_deploy_and_get_status_roundtrip(self) -> None:
        tools = _register_tools(_make_server())

        deploy = tools["deploy_maas_model"](
            name="test-model",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
            grant_authenticated_access=True,
        )

        assert "error" not in deploy
        assert deploy["created"]["llm_inference_service"] == "test-model"
        assert deploy["created"]["maas_model_ref"] == "test-model"
        assert deploy["created"]["maas_auth_policy"] == "test-model-access"
        assert deploy["created"]["maas_subscription"] == "test-model-free"
        assert deploy["namespace_created"] is False  # llm already exists in the mock
        assert deploy["maas_endpoint"] == "https://maas.apps.mock-cluster.local"

        status = tools["get_maas_model"](name="test-model", namespace="llm")

        assert "error" not in status
        assert status["maas_registered"] is True
        assert status["model_name"] == "facebook/opt-125m"
        assert status["maas_endpoint"] == "https://maas.apps.mock-cluster.local"

    def test_deploy_fails_closed_without_platform(self) -> None:
        # Cluster state without the MaaS pieces (empty state)
        server = _make_server()
        server.k8s = MockK8sClient()
        tools = _register_tools(server)

        result = tools["deploy_maas_model"](
            name="test-model",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert result["error"] == "MaaS platform is not ready"
        assert result["remediation"]  # points at the rhoai-maas-guide

    def test_readiness_reports_missing_platform(self) -> None:
        server = _make_server()
        server.k8s = MockK8sClient()
        tools = _register_tools(server)

        result = tools["check_maas_readiness"]()

        assert result["ready"] is False
        assert result["remediation"]  # points at the rhoai-maas-guide
