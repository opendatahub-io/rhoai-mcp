"""Tests for MaaS MCP tools."""

from typing import Any
from unittest.mock import MagicMock

from rhoai_mcp.utils.errors import NotFoundError, NotManagedByMCPError, RHOAIError


def _seed_isvc(k8s: MagicMock, name: str, labels: dict[str, str]) -> None:
    """Seed an LLMInferenceService with labels into the mock client's get."""
    original_get = k8s.get.side_effect
    isvc = MagicMock()
    isvc.metadata.labels = labels

    def get_side_effect(
        crd: Any, requested: str, namespace: str | None = None
    ) -> Any:
        if crd.kind == "LLMInferenceService" and requested == name:
            return isvc
        return original_get(crd, requested, namespace)

    k8s.get.side_effect = get_side_effect


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


def _make_k8s_client(
    *,
    dsc_state: str | None = "Managed",
    maas_component: bool = True,
    gateway: bool = True,
    maas_api: bool = True,
    maas_api_ready: int = 1,
    namespace_exists: bool = True,
    ns_has_label: bool = True,
) -> MagicMock:
    """Build a mock K8s client that mimics the MaaS platform resources."""
    k8s = MagicMock()

    # DataScienceCluster with MaaS component state
    if dsc_state is not None:
        dsc = MagicMock()
        dsc.metadata.name = "default-dsc"
        if maas_component:
            dsc.spec = {
                "components": {
                    "aigateway": {"modelsAsAService": {"managementState": dsc_state}},
                },
            }
        else:
            dsc.spec = {"components": {}}
        k8s.list_resources.return_value = [dsc]
    else:
        k8s.list_resources.return_value = []

    resources: dict[tuple[str, str, str | None], Any] = {}

    if gateway:
        gw = MagicMock()
        gw.spec = {"listeners": [{"hostname": "maas.example.com"}]}
        gw.status = MagicMock()
        gw.status.conditions = [{"type": "Programmed", "status": "True"}]
        resources[("Gateway", "maas-default-gateway", "openshift-ingress")] = gw

    if maas_api:
        dep = MagicMock()
        dep.status.readyReplicas = maas_api_ready
        resources[("Deployment", "maas-api", "redhat-ai-gateway-infra")] = dep

    # MaaSModelRef for get_maas_model status lookups
    model_ref = MagicMock()
    model_ref.status.phase = "Ready"
    resources[("MaaSModelRef", "sim", "llm")] = model_ref

    def get_side_effect(crd: Any, name: str, namespace: str | None = None) -> Any:
        key = (crd.kind, name, namespace)
        if key in resources:
            return resources[key]
        raise NotFoundError(crd.kind, name, namespace)

    k8s.get.side_effect = get_side_effect

    def create_side_effect(
        crd: Any, body: dict[str, Any], namespace: str | None = None  # noqa: ARG001
    ) -> Any:
        created = MagicMock()
        created.metadata.name = body.get("metadata", {}).get("name", "unknown")
        return created

    k8s.create.side_effect = create_side_effect

    # Namespace
    if namespace_exists:
        ns = MagicMock()
        ns.metadata.labels = (
            {"maas.opendatahub.io/gateway-access": "true"} if ns_has_label else {}
        )
        k8s.get_namespace.return_value = ns
    else:
        k8s.get_namespace.side_effect = NotFoundError("Namespace", "llm")

    return k8s


def _make_server(k8s: MagicMock, *, read_only: bool = False) -> MagicMock:
    """Build a mock RHOAIServer."""
    server = MagicMock()
    server.config.is_operation_allowed.return_value = (
        not read_only,
        "read-only" if read_only else None,
    )
    server.config.read_only_mode = read_only
    server.k8s = k8s
    return server


class TestCheckMaasReadiness:
    """Tests for check_maas_readiness."""

    def test_ready_when_all_checks_pass(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"](namespace="llm")

        assert result["ready"] is True
        assert result["issues"] is None
        assert result["maas_endpoint"] == "https://maas.example.com"
        passed = {c["name"] for c in result["checks"] if c["passed"]}
        assert {
            "DataScienceCluster",
            "MaaS Gateway",
            "maas-api",
            "Namespace gateway access",
        } <= passed

    def test_not_ready_when_dsc_not_managed(self) -> None:
        k8s = _make_k8s_client(dsc_state="Removed")
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"]()

        assert result["ready"] is False
        assert any("not enabled on DataScienceCluster" in i for i in result["issues"])

    def test_not_ready_when_gateway_missing(self) -> None:
        k8s = _make_k8s_client(gateway=False)
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"]()

        assert result["ready"] is False
        assert any("gateway is not installed" in i for i in result["issues"])

    def test_not_ready_when_maas_api_missing(self) -> None:
        k8s = _make_k8s_client(maas_api=False)
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"]()

        assert result["ready"] is False
        assert any("maas-api is not running" in i for i in result["issues"])

    def test_missing_namespace_is_informational(self) -> None:
        k8s = _make_k8s_client(namespace_exists=False)
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"]()

        # deploy_maas_model creates the namespace, so this must not block readiness
        assert result["ready"] is True
        assert any("does not exist" in i for i in result["issues"])

    def test_unlabeled_namespace_is_informational(self) -> None:
        k8s = _make_k8s_client(ns_has_label=False)
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"]()

        assert result["ready"] is True
        assert any("gateway" in i for i in result["issues"])

    def test_not_ready_when_maas_component_not_configured(self) -> None:
        # DataScienceCluster exists but has no MaaS component configured
        k8s = _make_k8s_client(maas_component=False)
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"]()

        assert result["ready"] is False
        assert any("not configured" in i for i in result["issues"])
        # The managementState must not be rendered as 'None'
        assert not any("'None'" in c["message"] for c in result["checks"])

    def test_not_ready_when_maas_api_has_zero_ready_replicas(self) -> None:
        k8s = _make_k8s_client(maas_api_ready=0)
        tools = _register_tools(_make_server(k8s))

        result = tools["check_maas_readiness"]()

        assert result["ready"] is False
        assert any("no ready replicas" in i for i in result["issues"])
        maas_api_check = next(
            c for c in result["checks"] if c["name"] == "maas-api"
        )
        assert maas_api_check["passed"] is False


class TestDeployMaasModel:
    """Tests for deploy_maas_model."""

    def test_creates_inference_service_and_model_ref(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="test-model",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert "error" not in result
        created = result["created"]
        assert created["llm_inference_service"] == "test-model"
        assert created["maas_model_ref"] == "test-model"
        assert created["maas_auth_policy"] is None
        assert created["maas_subscription"] is None
        assert result["maas_endpoint"] == "https://maas.example.com"

        # Inspect the created LLMInferenceService body
        isvc_call = next(
            c for c in k8s.create.call_args_list if c.args[0].kind == "LLMInferenceService"
        )
        body = isvc_call.args[1]
        assert body["spec"]["model"] == {
            "name": "facebook/opt-125m",
            "uri": "hf://sshleifer/tiny-gpt2",
        }
        # Routed through the shared MaaS gateway
        gateway_refs = body["spec"]["router"]["gateway"]["refs"]
        assert gateway_refs == [
            {"name": "maas-default-gateway", "namespace": "openshift-ingress"}
        ]
        # Managed-by label for the delete guard
        assert body["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "rhoai-mcp"

    def test_simulator_image_is_default(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        isvc_call = next(
            c for c in k8s.create.call_args_list if c.args[0].kind == "LLMInferenceService"
        )
        container = isvc_call.args[1]["spec"]["template"]["containers"][0]
        assert container["image"] == "ghcr.io/llm-d/llm-d-inference-sim:v0.7.1"
        assert "--mode" in container["args"]

    def test_custom_image_uses_vllm_layout(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        tools["deploy_maas_model"](
            name="qwen",
            model_name="qwen3-06b",
            model_uri="hf://Qwen/Qwen3-0.6B",
            image="registry.redhat.io/rhaiis/vllm-cpu-rhel9:latest",
        )

        isvc_call = next(
            c for c in k8s.create.call_args_list if c.args[0].kind == "LLMInferenceService"
        )
        container = isvc_call.args[1]["spec"]["template"]["containers"][0]
        assert container["image"] == "registry.redhat.io/rhaiis/vllm-cpu-rhel9:latest"
        assert container["command"] == ["python", "-m", "vllm.entrypoints.openai.api_server"]

    def test_tagless_registry_image_gets_default_tag(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="qwen",
            model_name="qwen3-06b",
            model_uri="hf://Qwen/Qwen3-0.6B",
            image="registry.redhat.io/rhaiis/vllm-cpu-rhel9",
        )

        assert "error" not in result
        # registry.redhat.io rejects implicit ':latest' pulls - the documented
        # default tag must be appended with a warning
        assert any("default tag" in w for w in (result["warnings"] or []))
        isvc_call = next(
            c for c in k8s.create.call_args_list if c.args[0].kind == "LLMInferenceService"
        )
        container = isvc_call.args[1]["spec"]["template"]["containers"][0]
        assert container["image"] == "registry.redhat.io/rhaiis/vllm-cpu-rhel9:3.3.0"

    def test_tagged_images_are_left_untouched(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="qwen",
            model_name="qwen3-06b",
            model_uri="hf://Qwen/Qwen3-0.6B",
            image="quay.io/org/vllm:custom",
        )

        assert "error" not in result
        assert result["warnings"] is None

    def test_cpu_vllm_image_gets_oom_guards(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        tools["deploy_maas_model"](
            name="qwen",
            model_name="qwen3-06b",
            model_uri="hf://Qwen/Qwen3-0.6B",
            image="registry.redhat.io/rhaiis/vllm-cpu-rhel9:3.3.0",
        )

        isvc_call = next(
            c for c in k8s.create.call_args_list if c.args[0].kind == "LLMInferenceService"
        )
        container = isvc_call.args[1]["spec"]["template"]["containers"][0]
        # vLLM's CPU backend profiles against node memory and OOMKills a 16Gi
        # limit - the KV cache must be bounded and the context capped
        assert {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "8"} in container["env"]
        assert container["args"][-2:] == ["--max-model-len", "8192"]
        assert container["resources"]["limits"]["memory"] == "24Gi"
        # The simulator and GPU presets are unchanged
        assert container["resources"]["limits"]["cpu"] == "8"

    def test_gpu_vllm_image_has_no_cpu_guards(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        tools["deploy_maas_model"](
            name="qwen",
            model_name="qwen3-06b",
            model_uri="hf://Qwen/Qwen3-0.6B",
            image="registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.3.0",
        )

        isvc_call = next(
            c for c in k8s.create.call_args_list if c.args[0].kind == "LLMInferenceService"
        )
        container = isvc_call.args[1]["spec"]["template"]["containers"][0]
        assert "env" not in container
        assert "--max-model-len" not in container["args"]
        assert container["resources"]["limits"]["memory"] == "16Gi"

    def test_grant_authenticated_access_creates_governance(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
            grant_authenticated_access=True,
            token_rate_limit=50,
        )

        created = result["created"]
        assert created["maas_auth_policy"] == "sim-access"
        assert created["maas_subscription"] == "sim-free"

        sub_call = next(
            c for c in k8s.create.call_args_list if c.args[0].kind == "MaaSSubscription"
        )
        model_ref = sub_call.args[1]["spec"]["modelRefs"][0]
        assert model_ref["name"] == "sim"
        assert model_ref["namespace"] == "llm"
        assert model_ref["tokenRateLimits"] == [{"limit": 50, "window": "1m"}]
        # Governance resources live in the models-as-a-service namespace
        assert sub_call.kwargs["namespace"] == "models-as-a-service"

    def test_no_governance_by_default(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert result["created"]["maas_auth_policy"] is None
        kinds = {c.args[0].kind for c in k8s.create.call_args_list}
        assert "MaaSAuthPolicy" not in kinds
        assert "MaaSSubscription" not in kinds
        # The agent is told how to grant access
        assert any("grant_authenticated_access" in s for s in result["next_steps"])

    def test_blocks_when_platform_not_ready(self) -> None:
        k8s = _make_k8s_client(dsc_state="Removed")
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert result["error"] == "MaaS platform is not ready"
        k8s.create.assert_not_called()

    def test_read_only_mode_rejected(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s, read_only=True))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert result["error"] == "read-only"
        k8s.create.assert_not_called()

    def test_invalid_name_rejected(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="Invalid_Name",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert "error" in result
        k8s.create.assert_not_called()

    def test_invalid_model_uri_rejected(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="ftp://example.com/model",
        )

        assert "error" in result
        assert "Unsupported model_uri" in result["error"]
        k8s.create.assert_not_called()

    def test_creates_namespace_when_missing(self) -> None:
        k8s = _make_k8s_client(namespace_exists=False)
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert result["namespace_created"] is True
        k8s.create_namespace.assert_called_once()
        labels = k8s.create_namespace.call_args.kwargs["labels"]
        assert labels["maas.opendatahub.io/gateway-access"] == "true"

    def test_labels_existing_namespace_with_warning(self) -> None:
        k8s = _make_k8s_client(ns_has_label=False)
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        k8s.patch_project.assert_called_once()
        labels = k8s.patch_project.call_args.kwargs["labels"]
        assert labels["maas.opendatahub.io/gateway-access"] == "true"
        # Labeling a namespace the tool did not create is always surfaced
        assert any("Labeled existing namespace" in w for w in (result["warnings"] or []))

    def test_rejects_protected_namespace(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
            namespace="kube-system",
        )

        assert "reserved by the platform" in result["error"]
        k8s.create_namespace.assert_not_called()
        k8s.create.assert_not_called()

    def test_rejects_trailing_newline_namespace(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
            namespace="llm\n",
        )

        assert "error" in result
        k8s.create.assert_not_called()

    def test_rejects_empty_model_name(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="  ",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert "non-empty" in result["error"]
        k8s.create.assert_not_called()

    def test_rejects_invalid_rate_limit_window(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
            grant_authenticated_access=True,
            rate_limit_window="1x",
        )

        assert "Invalid rate_limit_window" in result["error"]
        k8s.create.assert_not_called()

    def test_rejects_invalid_pydantic_params(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
            replicas=-1,
        )

        assert "error" in result
        k8s.create.assert_not_called()

    def test_model_ref_failure_reports_created_isvc(self) -> None:
        k8s = _make_k8s_client()

        def create_side_effect(
            crd: Any, body: dict[str, Any], namespace: str | None = None  # noqa: ARG001
        ) -> Any:
            if crd.kind == "MaaSModelRef":
                raise RHOAIError("forbidden: maasmodelrefs is forbidden")
            created = MagicMock()
            created.metadata.name = body.get("metadata", {}).get("name", "unknown")
            return created

        k8s.create.side_effect = create_side_effect
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="sim",
            model_name="facebook/opt-125m",
            model_uri="hf://sshleifer/tiny-gpt2",
        )

        assert "error" in result
        # The LLMInferenceService exists - the caller must not guess
        assert result["created"]["llm_inference_service"] == "sim"
        assert result["created"]["maas_model_ref"] is None
        assert "was created but MaaS registration failed" in result["message"]

    def test_oci_uri_warns_about_pull_secrets(self) -> None:
        k8s = _make_k8s_client()
        mock_sa = MagicMock()
        mock_sa.image_pull_secrets = None
        k8s.core_v1.read_namespaced_service_account.return_value = mock_sa
        tools = _register_tools(_make_server(k8s))

        result = tools["deploy_maas_model"](
            name="oci-model",
            model_name="my-model",
            model_uri="oci://quay.io/org/model:latest",
        )

        assert any("No image pull secrets" in w for w in (result["warnings"] or []))


class TestGetMaasModel:
    """Tests for get_maas_model."""

    def _make_isvc(self) -> MagicMock:
        isvc = MagicMock()
        isvc.metadata.name = "sim"
        isvc.metadata.namespace = "llm"
        isvc.metadata.uid = "isvc-uid"
        isvc.metadata.creation_timestamp = "2024-01-01T00:00:00Z"
        isvc.metadata.labels = {}
        isvc.metadata.annotations = {}
        isvc.spec = {
            "model": {"name": "facebook/opt-125m", "uri": "hf://sshleifer/tiny-gpt2"},
            "replicas": 1,
        }
        isvc.status = MagicMock()
        isvc.status.conditions = [
            MagicMock(
                type="Ready",
                status="True",
                reason=None,
                message=None,
                last_transition_time=None,
            )
        ]
        isvc.status.readyReplicas = 1
        return isvc

    def test_returns_status_and_endpoint(self) -> None:
        k8s = _make_k8s_client()
        # Override LLMInferenceService lookup with a realistic status
        original_get = k8s.get.side_effect

        def get_side_effect(crd: Any, name: str, namespace: str | None = None) -> Any:
            if crd.kind == "LLMInferenceService":
                return self._make_isvc()
            return original_get(crd, name, namespace)

        k8s.get.side_effect = get_side_effect
        tools = _register_tools(_make_server(k8s))

        result = tools["get_maas_model"](name="sim", namespace="llm")

        assert result["status"] == "Ready"
        assert result["maas_registered"] is True
        assert result["maas_phase"] == "Ready"
        assert result["maas_endpoint"] == "https://maas.example.com"
        assert "Ready and registered" in result["message"]

    def test_reports_unregistered_model(self) -> None:
        k8s = _make_k8s_client()
        original_get = k8s.get.side_effect

        def get_side_effect(crd: Any, name: str, namespace: str | None = None) -> Any:
            if crd.kind == "LLMInferenceService":
                return self._make_isvc()
            if crd.kind == "MaaSModelRef":
                raise NotFoundError(crd.kind, name, namespace)
            return original_get(crd, name, namespace)

        k8s.get.side_effect = get_side_effect
        tools = _register_tools(_make_server(k8s))

        result = tools["get_maas_model"](name="sim", namespace="llm")

        assert result["maas_registered"] is False
        assert "registration" in result["message"]

    def test_not_found_returns_error(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["get_maas_model"](name="missing", namespace="llm")

        assert "error" in result

    def test_invalid_name_rejected(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["get_maas_model"](name="Bad_Name", namespace="llm")

        assert "error" in result
        k8s.get.assert_not_called()


class TestDeleteMaasModel:
    """Tests for delete_maas_model."""

    MANAGED_LABELS = {"app.kubernetes.io/managed-by": "rhoai-mcp"}

    def _seed_managed_isvc(self, k8s: MagicMock, name: str = "sim") -> None:
        _seed_isvc(k8s, name, self.MANAGED_LABELS)

    def test_refuses_without_confirm(self) -> None:
        k8s = _make_k8s_client()
        self._seed_managed_isvc(k8s)
        tools = _register_tools(_make_server(k8s))

        result = tools["delete_maas_model"](name="sim", namespace="llm")

        assert result["error"] == "Deletion not confirmed"
        k8s.delete.assert_not_called()

    def test_deletes_all_resources(self) -> None:
        k8s = _make_k8s_client()
        self._seed_managed_isvc(k8s)
        tools = _register_tools(_make_server(k8s))

        result = tools["delete_maas_model"](name="sim", namespace="llm", confirm=True)

        assert "error" not in result
        assert result["deleted"] == [
            "MaaSSubscription",
            "MaaSAuthPolicy",
            "MaaSModelRef",
            "LLMInferenceService",
        ]
        assert result["kept"] is None
        # Governance resources are deleted from the governance namespace
        sub_call = next(
            c for c in k8s.delete.call_args_list if c.args[0].kind == "MaaSSubscription"
        )
        assert sub_call.args[1] == "sim-free"
        assert sub_call.args[2] == "models-as-a-service"

    def test_skips_missing_governance(self) -> None:
        # A model deployed without grant_authenticated_access has no
        # governance resources - deletion must still succeed
        k8s = _make_k8s_client()
        self._seed_managed_isvc(k8s)

        def delete_side_effect(
            crd: Any, name: str, namespace: str | None = None  # noqa: ARG001
        ) -> None:
            if crd.kind in ("MaaSSubscription", "MaaSAuthPolicy"):
                raise NotFoundError(crd.kind, name, namespace)

        k8s.delete.side_effect = delete_side_effect
        tools = _register_tools(_make_server(k8s))

        result = tools["delete_maas_model"](name="sim", namespace="llm", confirm=True)

        assert "error" not in result
        assert result["skipped"] == ["MaaSSubscription", "MaaSAuthPolicy"]
        assert result["deleted"] == ["MaaSModelRef", "LLMInferenceService"]

    def test_refuses_isvc_not_managed_by_mcp(self) -> None:
        k8s = _make_k8s_client()
        _seed_isvc(k8s, "sim", {"app.kubernetes.io/managed-by": "someone-else"})
        tools = _register_tools(_make_server(k8s))

        result = tools["delete_maas_model"](name="sim", namespace="llm", confirm=True)

        assert "was not created by this MCP server" in result["error"]
        k8s.delete.assert_not_called()

    def test_keeps_hand_created_governance(self) -> None:
        k8s = _make_k8s_client()
        self._seed_managed_isvc(k8s)

        def delete_side_effect(
            crd: Any, name: str, namespace: str | None = None  # noqa: ARG001
        ) -> None:
            if crd.kind == "MaaSAuthPolicy":
                raise NotManagedByMCPError(crd.kind, name, namespace)

        k8s.delete.side_effect = delete_side_effect
        tools = _register_tools(_make_server(k8s))

        result = tools["delete_maas_model"](name="sim", namespace="llm", confirm=True)

        assert "error" not in result
        assert result["kept"] == ["MaaSAuthPolicy"]
        assert any("Kept MaaSAuthPolicy" in w for w in (result["warnings"] or []))
        # Everything MCP-managed is still deleted
        assert result["deleted"] == [
            "MaaSSubscription",
            "MaaSModelRef",
            "LLMInferenceService",
        ]

    def test_not_found_returns_error(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["delete_maas_model"](name="missing", namespace="llm", confirm=True)

        assert "error" in result
        k8s.delete.assert_not_called()

    def test_read_only_mode_rejected(self) -> None:
        k8s = _make_k8s_client()
        self._seed_managed_isvc(k8s)
        tools = _register_tools(_make_server(k8s, read_only=True))

        result = tools["delete_maas_model"](name="sim", namespace="llm", confirm=True)

        assert result["error"] == "Read-only mode is enabled"
        k8s.delete.assert_not_called()

    def test_invalid_name_rejected(self) -> None:
        k8s = _make_k8s_client()
        tools = _register_tools(_make_server(k8s))

        result = tools["delete_maas_model"](name="Bad_Name", namespace="llm", confirm=True)

        assert "error" in result
        k8s.delete.assert_not_called()
