"""Tests for QuickstartsClient."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.quickstarts.client import (
    JOB_CRD,
    QUICKSTART_NAME_LABEL,
    QuickstartsClient,
)
from rhoai_mcp.domains.quickstarts.oci import MANIFEST_MEDIA_TYPE, REGISTRY_MEDIA_TYPE
from rhoai_mcp.utils.errors import NotFoundError, ValidationError
from tests.domains.quickstarts.conftest import MANIFEST_YAML, REGISTRY_YAML


@pytest.fixture
def mock_config() -> SimpleNamespace:
    return SimpleNamespace(
        quickstart_registry_ref="quay.io/rh-ai-quickstart/quickstart-registry:latest",
        quickstart_job_namespace="openshift-quickstarts",
        quickstart_installer_service_account="quickstart-installer",
        quickstart_job_ttl_seconds=3600,
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
    return k8s


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

        # Secret created for the secret parameter, keyed by resolved env var.
        secret_kwargs = mock_k8s.create_secret.call_args.kwargs
        assert secret_kwargs["namespace"] == "openshift-quickstarts"
        assert secret_kwargs["data"] == {"PARAM_KEYCLOAK_REALM_TESTUSER_PASSWORD": "s3cret"}

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
        client.run_action(name="peoplemesh", action="STATUS")
        mock_k8s.create_secret.assert_not_called()

    def test_secret_cleaned_up_on_job_failure(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        mock_k8s.create.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            client.run_action(
                name="peoplemesh",
                action="install",
                parameters={"keycloak.realm.testUser.password": "s3cret"},
            )

        mock_k8s.core_v1.delete_namespaced_secret.assert_called_once()


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

    def test_get_action_logs_no_pods(
        self, client: QuickstartsClient, mock_k8s: MagicMock
    ) -> None:
        mock_k8s.core_v1.list_namespaced_pod.return_value.items = []
        with pytest.raises(NotFoundError):
            client.get_action_logs("missing-job", "openshift-quickstarts")
