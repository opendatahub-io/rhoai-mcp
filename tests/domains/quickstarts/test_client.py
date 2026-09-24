"""Tests for QuickstartsClient."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.quickstarts.client import (
    INVALID_OPERATION_MESSAGE,
    JOB_CRD,
    QUICKSTART_NAME_LABEL,
    QUICKSTART_TARGET_NS_LABEL,
    QuickstartsClient,
)
from rhoai_mcp.domains.quickstarts.oci import MANIFEST_MEDIA_TYPE, REGISTRY_MEDIA_TYPE
from rhoai_mcp.utils.errors import NotFoundError, RHOAIError, ValidationError
from rhoai_mcp.utils.labels import RHOAILabels
from tests.domains.quickstarts.conftest import MANIFEST_YAML, REGISTRY_YAML


def _ns(labels: dict[str, str] | None = None) -> SimpleNamespace:
    """Build a namespace object shaped like the k8s client's read_namespace result."""
    return SimpleNamespace(metadata=SimpleNamespace(labels=labels or {}))


@pytest.fixture
def mock_config() -> SimpleNamespace:
    return SimpleNamespace(
        quickstart_registry_ref="quay.io/rh-ai-quickstart/quickstart-registry:latest",
        quickstart_job_namespace="openshift-quickstarts",
        quickstart_installer_service_account="quickstart-installer",
        quickstart_installer_cpu_request="100m",
        quickstart_installer_cpu_limit="1",
        quickstart_installer_memory_request="256Mi",
        quickstart_installer_memory_limit="2Gi",
        quickstart_job_ttl_seconds=3600,
        quickstart_job_active_deadline_seconds=7200,
        quickstart_allowed_repos=["quay.io/rh-ai-quickstart/"],
        read_only_mode=False,
    )


@pytest.fixture
def mock_oci() -> MagicMock:
    oci = MagicMock()

    def fetch(_ref: str, media_type: str) -> bytes:
        if media_type == REGISTRY_MEDIA_TYPE:
            return REGISTRY_YAML
        if media_type == MANIFEST_MEDIA_TYPE:
            return MANIFEST_YAML
        raise AssertionError(f"unexpected media type {media_type}")

    oci.fetch_layer.side_effect = fetch
    return oci


@pytest.fixture
def mock_k8s() -> MagicMock:
    k8s = MagicMock()
    k8s.create.return_value.to_dict.return_value = {"metadata": {"uid": "job-uid"}}
    # By default the target namespace does not exist, so INSTALL is allowed and
    # ownership checks for other actions must be configured explicitly per test.
    k8s.get_namespace.side_effect = NotFoundError("Namespace", "unset")
    return k8s


def _namespace_exists(k8s: MagicMock, labels: dict[str, str] | None = None) -> None:
    """Configure the mock so the target namespace exists with the given labels."""
    k8s.get_namespace.side_effect = None
    k8s.get_namespace.return_value = _ns(labels)


@pytest.fixture
def client(
    mock_k8s: MagicMock, mock_config: SimpleNamespace, mock_oci: MagicMock
) -> QuickstartsClient:
    return QuickstartsClient(mock_k8s, mock_config, oci=mock_oci)


class TestDiscovery:
    def test_get_registry(self, client: QuickstartsClient, mock_oci: MagicMock) -> None:
        registry = client.get_registry()
        assert registry.quickstarts[0].name == "peoplemesh"
        mock_oci.fetch_layer.assert_called_once_with(
            "quay.io/rh-ai-quickstart/quickstart-registry:latest", REGISTRY_MEDIA_TYPE
        )

    def test_get_manifest_resolves_ref_from_registry(
        self, client: QuickstartsClient, mock_oci: MagicMock
    ) -> None:
        manifest = client.get_manifest("peoplemesh")
        assert manifest.name == "peoplemesh"
        # Second fetch is for the manifest at the registry-resolved ref.
        mock_oci.fetch_layer.assert_any_call(
            "quay.io/rh-ai-quickstart/peoplemesh-manifest:1.0.0", MANIFEST_MEDIA_TYPE
        )

    def test_get_manifest_unknown_quickstart(self, client: QuickstartsClient) -> None:
        with pytest.raises(NotFoundError):
            client.get_manifest("does-not-exist")


class TestRunAction:
    def test_install_builds_job_and_secret(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        result = client.run_action(
            name="peoplemesh",
            action="install",
            parameters={
                "keycloak.realm.testUser.password": "s3cret",
                "ollama.gpu.enabled": True,
            },
        )

        # Secret created for the secret parameter, keyed by resolved env var, and
        # owned by the Job so it is garbage-collected together with it.
        secret_call = mock_k8s.core_v1.create_namespaced_secret.call_args
        assert secret_call.kwargs["namespace"] == "openshift-quickstarts"
        secret_body = secret_call.kwargs["body"]
        assert secret_body["stringData"] == {"PARAM_KEYCLOAK_REALM_TESTUSER_PASSWORD": "s3cret"}
        owner = secret_body["metadata"]["ownerReferences"][0]
        assert owner["kind"] == "Job"
        assert owner["uid"] == "job-uid"
        assert owner["controller"] is True

        # Job created via the batch/v1 Job CRD.
        create_args = mock_k8s.create.call_args
        assert create_args.args[0] is JOB_CRD
        body = create_args.kwargs["body"]
        assert create_args.kwargs["namespace"] == "openshift-quickstarts"

        container = body["spec"]["template"]["spec"]["containers"][0]
        env = {e["name"]: e for e in container["env"]}
        assert env["ACTION"]["value"] == "INSTALL"
        assert env["TARGET_NAMESPACE"]["value"] == "peoplemesh-quickstart"
        assert env["INSTALL_MODE"]["value"] == "demo"
        # Boolean config parameter rendered as a plain env value.
        assert env["PARAM_OLLAMA_GPU_ENABLED"]["value"] == "true"
        # Secret parameter injected via secretKeyRef, not inline.
        ref = env["PARAM_KEYCLOAK_REALM_TESTUSER_PASSWORD"]["valueFrom"]["secretKeyRef"]
        assert ref["key"] == "PARAM_KEYCLOAK_REALM_TESTUSER_PASSWORD"

        assert body["spec"]["template"]["spec"]["serviceAccountName"] == "quickstart-installer"
        assert body["metadata"]["labels"][QUICKSTART_NAME_LABEL] == "peoplemesh"

        assert result["action"] == "INSTALL"
        assert result["target_namespace"] == "peoplemesh-quickstart"
        assert result["_source"]["uid"] == "job-uid"

    def test_job_identity_env_injected(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # The installer gets its own Job name (literal) and namespace (downward
        # API) as additive env, distinct from TARGET_NAMESPACE.
        _namespace_exists(mock_k8s, RHOAILabels.managed_by_mcp_labels())
        result = client.run_action(name="peoplemesh", action="STATUS")

        env = {e["name"]: e for e in _container(mock_k8s)["env"]}
        assert env["JOB_NAME"]["value"] == result["job_name"]
        field_ref = env["JOB_NAMESPACE"]["valueFrom"]["fieldRef"]
        assert field_ref["fieldPath"] == "metadata.namespace"

    def test_job_active_deadline_seconds_set(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # A hung installer must not run forever; the Job carries a hard wall-clock
        # cap from config so Kubernetes terminates it and marks it Failed.
        _namespace_exists(mock_k8s, RHOAILabels.managed_by_mcp_labels())
        client.run_action(name="peoplemesh", action="STATUS")

        body = mock_k8s.create.call_args.kwargs["body"]
        assert body["spec"]["activeDeadlineSeconds"] == 7200

    def test_installer_container_has_resource_requests_and_limits(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # The installer container gets a guaranteed floor (requests) so it stays
        # schedulable and out of the eviction line, plus generous limits so a long
        # install is not OOM-killed while still capping any runaway.
        _namespace_exists(mock_k8s, RHOAILabels.managed_by_mcp_labels())
        client.run_action(name="peoplemesh", action="STATUS")

        resources = _container(mock_k8s)["resources"]
        assert resources["requests"] == {"cpu": "100m", "memory": "256Mi"}
        assert resources["limits"] == {"cpu": "1", "memory": "2Gi"}

    def test_install_missing_required_param(self, client: QuickstartsClient) -> None:
        with pytest.raises(ValidationError, match="missing required"):
            client.run_action(name="peoplemesh", action="install", parameters={})

    def test_unknown_parameter_rejected(self, client: QuickstartsClient) -> None:
        with pytest.raises(ValidationError, match="unknown parameter"):
            client.run_action(
                name="peoplemesh",
                action="install",
                parameters={
                    "keycloak.realm.testUser.password": "x",
                    "typo.param": "y",
                },
            )

    def test_unsupported_action_rejected(self, client: QuickstartsClient) -> None:
        with pytest.raises(ValidationError, match="not supported"):
            client.run_action(name="peoplemesh", action="UPGRADE")

    def test_unsupported_mode_rejected(self, client: QuickstartsClient) -> None:
        with pytest.raises(ValidationError, match="mode"):
            client.run_action(
                name="peoplemesh",
                action="install",
                mode="production",
                parameters={"keycloak.realm.testUser.password": "x"},
            )

    def test_action_without_secret_creates_no_secret(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        _namespace_exists(mock_k8s, RHOAILabels.managed_by_mcp_labels())
        client.run_action(name="peoplemesh", action="STATUS")
        mock_k8s.core_v1.create_namespaced_secret.assert_not_called()

    def test_job_cleaned_up_on_secret_failure(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # The Job is created first; if the params Secret cannot be created, the
        # Job must be deleted so it does not linger unable to start.
        mock_k8s.core_v1.create_namespaced_secret.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            client.run_action(
                name="peoplemesh",
                action="install",
                parameters={"keycloak.realm.testUser.password": "s3cret"},
            )

        mock_k8s.get_resource.return_value.delete.assert_called_once()


class TestStatusAndLogs:
    def test_get_action_status_complete(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        mock_k8s.get.return_value.to_dict.return_value = {
            "metadata": {
                "uid": "u1",
                "labels": {
                    QUICKSTART_NAME_LABEL: "peoplemesh",
                    "quickstart.redhat.com/action": "install",
                },
            },
            "status": {
                "succeeded": 1,
                "startTime": "t0",
                "completionTime": "t1",
                "conditions": [{"type": "Complete", "status": "True"}],
            },
        }

        status = client.get_action_status("qs-peoplemesh-install-abc", "openshift-quickstarts")

        mock_k8s.get.assert_called_once_with(
            JOB_CRD, "qs-peoplemesh-install-abc", "openshift-quickstarts"
        )
        assert status["phase"] == "Complete"
        assert status["succeeded"] == 1
        assert status["quickstart"] == "peoplemesh"

    def test_get_action_status_surfaces_failure_reason(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # A prerequisites failure: Job counters say "Failed", but the installer's
        # termination-log carries the real reason and a non-zero exit code.
        mock_k8s.get.return_value.to_dict.return_value = {
            "metadata": {"uid": "u1", "labels": {QUICKSTART_NAME_LABEL: "peoplemesh"}},
            "status": {"failed": 1, "conditions": [{"type": "Failed", "status": "True"}]},
        }
        message = '{"status": "prerequisites_failed", "missing": ["gpu-operator"]}'
        terminated = SimpleNamespace(exit_code=2, message=message)
        cs = SimpleNamespace(name="installer", state=SimpleNamespace(terminated=terminated))
        pod = SimpleNamespace(
            metadata=SimpleNamespace(name="pod-x", creation_timestamp="t0"),
            status=SimpleNamespace(container_statuses=[cs]),
        )
        mock_k8s.core_v1.list_namespaced_pod.return_value.items = [pod]

        status = client.get_action_status("qs-x", "openshift-quickstarts")

        assert status["phase"] == "Failed"
        assert status["exit_code"] == 2
        assert status["termination_message"] == message
        assert status["result"] == {"status": "prerequisites_failed", "missing": ["gpu-operator"]}

    def test_get_action_status_without_terminated_pod_omits_reason(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # Still running (or pod already GC'd): the richer detail degrades to None
        # rather than failing the status call.
        mock_k8s.get.return_value.to_dict.return_value = {
            "metadata": {},
            "status": {"active": 1},
        }
        mock_k8s.core_v1.list_namespaced_pod.return_value.items = []

        status = client.get_action_status("qs-x", "openshift-quickstarts")

        assert status["phase"] == "Running"
        assert status["exit_code"] is None
        assert status["termination_message"] is None
        assert status["result"] is None

    def test_get_action_logs(self, client: QuickstartsClient, mock_k8s: MagicMock) -> None:
        pod = MagicMock()
        pod.metadata.name = "qs-peoplemesh-install-abc-xyz"
        pod.metadata.creation_timestamp = "t0"
        mock_k8s.core_v1.list_namespaced_pod.return_value.items = [pod]
        mock_k8s.core_v1.read_namespaced_pod_log.return_value = "installer log output"

        logs = client.get_action_logs("qs-peoplemesh-install-abc", "openshift-quickstarts")

        assert logs["pod"] == "qs-peoplemesh-install-abc-xyz"
        assert logs["logs"] == "installer log output"
        mock_k8s.core_v1.list_namespaced_pod.assert_called_once_with(
            namespace="openshift-quickstarts",
            label_selector="job-name=qs-peoplemesh-install-abc",
        )

    def test_get_action_logs_no_pods(self, client: QuickstartsClient, mock_k8s: MagicMock) -> None:
        mock_k8s.core_v1.list_namespaced_pod.return_value.items = []
        with pytest.raises(NotFoundError):
            client.get_action_logs("missing-job", "openshift-quickstarts")

    def test_get_action_logs_handles_missing_timestamp(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # A pod without a creationTimestamp must not make max() compare a
        # datetime against a differently-typed fallback (TypeError).
        from datetime import datetime, timezone

        newer = SimpleNamespace(
            metadata=SimpleNamespace(
                name="newer", creation_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)
            )
        )
        no_ts = SimpleNamespace(metadata=SimpleNamespace(name="no-ts", creation_timestamp=None))
        mock_k8s.core_v1.list_namespaced_pod.return_value.items = [no_ts, newer]
        mock_k8s.core_v1.read_namespaced_pod_log.return_value = "log output"

        result = client.get_action_logs("qs-x", "openshift-quickstarts")

        assert result["pod"] == "newer"

    def test_get_action_logs_api_error_mapped(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        from kubernetes.client import ApiException

        mock_k8s.core_v1.list_namespaced_pod.side_effect = ApiException(
            status=403, reason="Forbidden"
        )
        # Mapped to RHOAIError so the tool's `except RHOAIError` handles it.
        with pytest.raises(RHOAIError, match="failed to list pods"):
            client.get_action_logs("qs-x", "openshift-quickstarts")

    def test_get_action_logs_namespace_not_found(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        from kubernetes.client import ApiException

        mock_k8s.core_v1.list_namespaced_pod.side_effect = ApiException(
            status=404, reason="Not Found"
        )
        with pytest.raises(NotFoundError):
            client.get_action_logs("qs-x", "openshift-quickstarts")


class TestNamespaceValidation:
    @pytest.mark.parametrize("namespace", ["openshift-monitoring", "kube-system", "kube-public"])
    def test_reserved_namespace_rejected(
        self, client: QuickstartsClient, mock_k8s: MagicMock, namespace: str
    ) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            client.run_action(name="peoplemesh", action="INSTALL", target_namespace=namespace)
        mock_k8s.create.assert_not_called()

    def test_destructive_action_on_reserved_namespace_rejected(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # The privilege-escalation case from the report: deleting a system namespace.
        with pytest.raises(ValidationError, match="reserved"):
            client.run_action(
                name="peoplemesh",
                action="UNINSTALL_DELETE_ALL",
                target_namespace="openshift-monitoring",
            )
        mock_k8s.create.assert_not_called()

    @pytest.mark.parametrize(
        "namespace",
        ["Bad-Namespace", "under_score", "trailing-", "-leading", "has space", "sym!bol"],
    )
    def test_malformed_namespace_rejected(self, client: QuickstartsClient, namespace: str) -> None:
        with pytest.raises(ValidationError, match="valid Kubernetes namespace"):
            client.run_action(name="peoplemesh", action="INSTALL", target_namespace=namespace)

    def test_overlong_namespace_rejected(self, client: QuickstartsClient) -> None:
        with pytest.raises(ValidationError, match="at most 63"):
            client.run_action(name="peoplemesh", action="INSTALL", target_namespace="a" * 64)

    def test_missing_namespace_rejected(
        self, mock_k8s: MagicMock, mock_config: SimpleNamespace
    ) -> None:
        # A manifest with no defaultNamespace and no override -> nothing to resolve.
        oci = MagicMock()

        def fetch(_ref: str, media_type: str) -> bytes:
            if media_type == REGISTRY_MEDIA_TYPE:
                return REGISTRY_YAML
            return MANIFEST_YAML.replace(b'  defaultNamespace: "peoplemesh-quickstart"\n', b"")

        oci.fetch_layer.side_effect = fetch
        client = QuickstartsClient(mock_k8s, mock_config, oci=oci)
        with pytest.raises(ValidationError, match="target_namespace is required"):
            client.run_action(name="peoplemesh", action="STATUS", target_namespace="")


class TestNamespaceOwnership:
    def test_install_rejects_existing_namespace(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        _namespace_exists(mock_k8s)
        with pytest.raises(ValidationError, match="already exists"):
            client.run_action(
                name="peoplemesh",
                action="INSTALL",
                target_namespace="my-quickstart",
                parameters={"keycloak.realm.testUser.password": "x"},
            )
        mock_k8s.create.assert_not_called()

    def test_managed_action_rejects_missing_namespace(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        # get_namespace raises NotFoundError by default (namespace absent).
        with pytest.raises(ValidationError, match="managed by this tool"):
            client.run_action(
                name="peoplemesh",
                action="UNINSTALL_DELETE_ALL",
                target_namespace="gone-ns",
            )
        mock_k8s.create.assert_not_called()

    def test_managed_action_rejects_unmanaged_namespace(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        _namespace_exists(mock_k8s, {"some": "label"})
        with pytest.raises(ValidationError, match="managed by this tool"):
            client.run_action(
                name="peoplemesh",
                action="UNINSTALL_DELETE_ALL",
                target_namespace="not-ours",
            )
        mock_k8s.create.assert_not_called()

    def test_managed_action_allowed_on_managed_namespace(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        _namespace_exists(mock_k8s, RHOAILabels.managed_by_mcp_labels())
        result = client.run_action(
            name="peoplemesh",
            action="UNINSTALL_DELETE_ALL",
            target_namespace="ours",
        )
        assert result["action"] == "UNINSTALL_DELETE_ALL"
        mock_k8s.create.assert_called_once()

    def test_install_records_target_namespace_label(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        client.run_action(
            name="peoplemesh",
            action="INSTALL",
            parameters={"keycloak.realm.testUser.password": "x"},
        )
        labels = mock_k8s.create.call_args.kwargs["body"]["metadata"]["labels"]
        assert labels[QUICKSTART_TARGET_NS_LABEL] == "peoplemesh-quickstart"


class TestSupplyChainAllowlist:
    """No quickstart artifact may be pulled from outside quickstart_allowed_repos."""

    def _client_with(
        self,
        mock_k8s: MagicMock,
        mock_config: SimpleNamespace,
        *,
        registry_yaml: bytes = REGISTRY_YAML,
        manifest_yaml: bytes = MANIFEST_YAML,
    ) -> tuple[QuickstartsClient, MagicMock]:
        oci = MagicMock()

        def fetch(_ref: str, media_type: str) -> bytes:
            if media_type == REGISTRY_MEDIA_TYPE:
                return registry_yaml
            return manifest_yaml

        oci.fetch_layer.side_effect = fetch
        return QuickstartsClient(mock_k8s, mock_config, oci=oci), oci

    def test_registry_ref_outside_allowlist_blocked(
        self, mock_k8s: MagicMock, mock_config: SimpleNamespace
    ) -> None:
        mock_config.quickstart_registry_ref = "quay.io/evil/quickstart-registry:latest"
        client, oci = self._client_with(mock_k8s, mock_config)
        with pytest.raises(ValidationError) as exc:
            client.get_registry()
        assert str(exc.value) == INVALID_OPERATION_MESSAGE
        # The blocked ref must never leak to the agent.
        assert "evil" not in str(exc.value)
        oci.fetch_layer.assert_not_called()

    def test_manifest_repo_outside_allowlist_blocked(
        self, mock_k8s: MagicMock, mock_config: SimpleNamespace
    ) -> None:
        registry = REGISTRY_YAML.replace(
            b'manifestRepo: "quay.io/rh-ai-quickstart/peoplemesh-manifest"',
            b'manifestRepo: "quay.io/evil/peoplemesh-manifest"',
        )
        client, oci = self._client_with(mock_k8s, mock_config, registry_yaml=registry)
        with pytest.raises(ValidationError) as exc:
            client.get_manifest("peoplemesh")
        assert str(exc.value) == INVALID_OPERATION_MESSAGE
        # The registry fetch happened; the manifest fetch must not have.
        oci.fetch_layer.assert_called_once()

    def test_installer_image_outside_allowlist_blocked(
        self, mock_k8s: MagicMock, mock_config: SimpleNamespace
    ) -> None:
        manifest = MANIFEST_YAML.replace(
            b'image: "quay.io/rh-ai-quickstart/peoplemesh-installer:1.0.0"',
            b'image: "quay.io/evil/peoplemesh-installer:1.0.0"',
        )
        client, _ = self._client_with(mock_k8s, mock_config, manifest_yaml=manifest)
        with pytest.raises(ValidationError) as exc:
            client.run_action(
                name="peoplemesh",
                action="INSTALL",
                parameters={"keycloak.realm.testUser.password": "x"},
            )
        assert str(exc.value) == INVALID_OPERATION_MESSAGE
        mock_k8s.create.assert_not_called()

    def test_sibling_prefix_bypass_blocked(
        self, mock_k8s: MagicMock, mock_config: SimpleNamespace
    ) -> None:
        # A namespace that merely starts with the allowed one must not slip through.
        mock_config.quickstart_registry_ref = (
            "quay.io/rh-ai-quickstart-evil/quickstart-registry:latest"
        )
        client, oci = self._client_with(mock_k8s, mock_config)
        with pytest.raises(ValidationError) as exc:
            client.get_registry()
        assert str(exc.value) == INVALID_OPERATION_MESSAGE
        oci.fetch_layer.assert_not_called()

    def test_unparseable_ref_blocked(
        self, mock_k8s: MagicMock, mock_config: SimpleNamespace
    ) -> None:
        mock_config.quickstart_registry_ref = "quay.io/:latest"
        client, oci = self._client_with(mock_k8s, mock_config)
        with pytest.raises(ValidationError) as exc:
            client.get_registry()
        assert str(exc.value) == INVALID_OPERATION_MESSAGE
        oci.fetch_layer.assert_not_called()

    def test_allowed_refs_pass(self, client: QuickstartsClient, mock_k8s: MagicMock) -> None:
        # The default fixtures all sit under quay.io/rh-ai-quickstart/ and succeed.
        result = client.run_action(
            name="peoplemesh",
            action="INSTALL",
            parameters={"keycloak.realm.testUser.password": "x"},
        )
        assert result["action"] == "INSTALL"
        mock_k8s.create.assert_called_once()


def _container(mock_k8s: MagicMock) -> dict[str, Any]:
    body = mock_k8s.create.call_args.kwargs["body"]
    return body["spec"]["template"]["spec"]["containers"][0]


class TestInstallerLabeling:
    def test_install_wraps_command_to_label_namespace(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        client.run_action(
            name="peoplemesh",
            action="INSTALL",
            parameters={"keycloak.realm.testUser.password": "x"},
        )
        command = _container(mock_k8s)["command"]
        assert command[:2] == ["/bin/sh", "-c"]
        script = command[2]
        # The declared installer command is run first...
        assert "/installer/entrypoint.sh" in script
        # ...then, only on success + existence, the namespace is labeled as ours...
        assert 'oc get namespace "$TARGET_NAMESPACE"' in script
        assert 'oc label namespace "$TARGET_NAMESPACE"' in script
        assert "app.kubernetes.io/managed-by=rhoai-mcp" in script
        # ...and the installer's exit code is preserved as the container result.
        assert 'exit "$rc"' in script

    def test_install_requires_declared_installer_command(
        self, mock_k8s: MagicMock, mock_config: SimpleNamespace
    ) -> None:
        oci = MagicMock()

        def fetch(_ref: str, media_type: str) -> bytes:
            if media_type == REGISTRY_MEDIA_TYPE:
                return REGISTRY_YAML
            return MANIFEST_YAML.replace(b'    command: ["/installer/entrypoint.sh"]\n', b"")

        oci.fetch_layer.side_effect = fetch
        client = QuickstartsClient(mock_k8s, mock_config, oci=oci)
        with pytest.raises(ValidationError, match="installer command"):
            client.run_action(
                name="peoplemesh",
                action="INSTALL",
                parameters={"keycloak.realm.testUser.password": "x"},
            )
        mock_k8s.create.assert_not_called()

    def test_non_labeling_action_uses_plain_command(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        _namespace_exists(mock_k8s, RHOAILabels.managed_by_mcp_labels())
        client.run_action(name="peoplemesh", action="STATUS")
        # STATUS neither creates nor owns the namespace, so it runs the installer
        # command unwrapped.
        assert _container(mock_k8s)["command"] == ["/installer/entrypoint.sh"]
