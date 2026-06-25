"""Tests for PipelineClient operations."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.pipelines.client import PipelineClient
from rhoai_mcp.domains.pipelines.crds import PipelinesCRDs
from rhoai_mcp.domains.pipelines.models import PipelineServerStatus


def _make_condition(
    type_name: str,
    status: str = "True",
    reason: str | None = None,
) -> MagicMock:
    """Create a mock K8s condition object."""
    cond = MagicMock()
    cond.type = type_name
    cond.status = status
    cond.reason = reason
    cond.message = None
    cond.last_transition_time = "2024-01-01T00:00:00Z"
    return cond


def _make_dspa_cr(
    conditions: list[MagicMock] | None = None,
    name: str = "dspa",
    namespace: str = "test-project",
) -> MagicMock:
    """Create a mock DSPA custom resource."""
    cr = MagicMock()
    cr.metadata.name = name
    cr.metadata.namespace = namespace
    cr.metadata.uid = "dspa-uid"
    cr.metadata.creation_timestamp = "2024-01-01T00:00:00Z"
    cr.metadata.labels = {}
    cr.metadata.annotations = {}
    if conditions is not None:
        cr.status.conditions = conditions
    else:
        cr.status = None
    return cr


ALL_READY_CONDITIONS = [
    _make_condition("APIServerReady"),
    _make_condition("PersistenceAgentReady"),
    _make_condition("ScheduledWorkflowReady"),
    _make_condition("DatabaseAvailable"),
    _make_condition("ObjectStoreAvailable"),
]


class TestGetPipelineServer:
    """Test PipelineClient.get_pipeline_server."""

    @pytest.fixture()
    def mock_k8s(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def client(self, mock_k8s: MagicMock) -> PipelineClient:
        return PipelineClient(mock_k8s)

    def test_get_pipeline_server_success(self, client: PipelineClient, mock_k8s: MagicMock) -> None:
        """Returning a DSPA CR with all-ready conditions gives a valid dict."""
        dspa_cr = _make_dspa_cr(conditions=ALL_READY_CONDITIONS)
        mock_k8s.list_resources.return_value = [dspa_cr]

        result = client.get_pipeline_server("test-project")

        assert result is not None
        assert result["name"] == "dspa"
        assert result["status"] == PipelineServerStatus.READY.value
        assert result["api_server_ready"] is True
        assert result["persistence_agent_ready"] is True
        assert result["scheduled_workflow_ready"] is True
        assert result["database_available"] is True
        assert result["object_store_available"] is True
        mock_k8s.list_resources.assert_called_once_with(
            PipelinesCRDs.DSPA, namespace="test-project"
        )

    def test_get_pipeline_server_no_dspa(self, client: PipelineClient, mock_k8s: MagicMock) -> None:
        """Empty list from K8s returns None."""
        mock_k8s.list_resources.return_value = []

        result = client.get_pipeline_server("empty-ns")

        assert result is None

    def test_get_pipeline_server_exception(
        self, client: PipelineClient, mock_k8s: MagicMock
    ) -> None:
        """Exception during list_resources returns None."""
        mock_k8s.list_resources.side_effect = RuntimeError("connection refused")

        result = client.get_pipeline_server("broken-ns")

        assert result is None


class TestCreatePipelineServer:
    """Test PipelineClient.create_pipeline_server."""

    @pytest.fixture()
    def mock_k8s(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def client(self, mock_k8s: MagicMock) -> PipelineClient:
        return PipelineClient(mock_k8s)

    def test_create_pipeline_server(self, client: PipelineClient, mock_k8s: MagicMock) -> None:
        """create_pipeline_server calls k8s.create with correct DSPA body."""
        from rhoai_mcp.domains.pipelines.models import PipelineServerCreate

        created_cr = _make_dspa_cr(conditions=[])
        mock_k8s.create.return_value = created_cr

        request = PipelineServerCreate(
            namespace="my-project",
            object_storage_secret="aws-connection-s3",
            object_storage_bucket="pipeline-artifacts",
            object_storage_endpoint="https://s3.amazonaws.com",
            object_storage_region="us-west-2",
        )
        result = client.create_pipeline_server(request)

        mock_k8s.create.assert_called_once()
        call_args = mock_k8s.create.call_args
        assert call_args[0][0] is PipelinesCRDs.DSPA
        body = call_args[1]["body"]
        assert body["kind"] == "DataSciencePipelinesApplication"
        assert body["metadata"]["namespace"] == "my-project"
        ext = body["spec"]["objectStorage"]["externalStorage"]
        assert ext["bucket"] == "pipeline-artifacts"
        assert ext["host"] == "https://s3.amazonaws.com"
        assert ext["region"] == "us-west-2"
        secret_ref = ext["s3CredentialsSecret"]
        assert secret_ref["secretName"] == "aws-connection-s3"
        assert call_args[1]["namespace"] == "my-project"

        assert result.metadata.name == "dspa"


class TestDeletePipelineServer:
    """Test PipelineClient.delete_pipeline_server."""

    @pytest.fixture()
    def mock_k8s(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def client(self, mock_k8s: MagicMock) -> PipelineClient:
        return PipelineClient(mock_k8s)

    def test_delete_pipeline_server(self, client: PipelineClient, mock_k8s: MagicMock) -> None:
        """delete_pipeline_server calls k8s.delete with correct args."""
        client.delete_pipeline_server("dspa", "test-project")

        mock_k8s.delete.assert_called_once_with(
            PipelinesCRDs.DSPA, name="dspa", namespace="test-project"
        )


class TestBuildDspaCr:
    """Test PipelineClient._build_dspa_cr structure."""

    def test_build_dspa_cr_structure(self) -> None:
        """CR body has correct apiVersion, kind, metadata, and spec."""
        from rhoai_mcp.domains.pipelines.models import PipelineServerCreate

        client = PipelineClient(MagicMock())
        request = PipelineServerCreate(
            namespace="prod-ns",
            object_storage_secret="my-s3-secret",
            object_storage_bucket="my-bucket",
            object_storage_endpoint="https://minio.example.com",
            object_storage_region="eu-west-1",
        )

        body: dict[str, Any] = client._build_dspa_cr(request)

        assert body["apiVersion"] == PipelinesCRDs.DSPA.api_version
        assert body["kind"] == "DataSciencePipelinesApplication"
        assert body["metadata"]["name"] == "dspa"
        assert body["metadata"]["namespace"] == "prod-ns"

        spec = body["spec"]
        ext = spec["objectStorage"]["externalStorage"]
        assert ext["bucket"] == "my-bucket"
        assert ext["host"] == "https://minio.example.com"
        assert ext["region"] == "eu-west-1"
        assert ext["s3CredentialsSecret"]["secretName"] == "my-s3-secret"
        assert ext["s3CredentialsSecret"]["accessKey"] == "AWS_ACCESS_KEY_ID"
        assert ext["s3CredentialsSecret"]["secretKey"] == "AWS_SECRET_ACCESS_KEY"

        db = spec["database"]
        assert db["mariaDB"]["deploy"] is True


class TestPipelineServerStatus:
    """Test PipelineServer.from_dspa_cr status determination."""

    def test_pipeline_server_status_ready(self) -> None:
        """5+ True conditions produce READY status."""
        from rhoai_mcp.domains.pipelines.models import PipelineServer

        cr = _make_dspa_cr(conditions=ALL_READY_CONDITIONS)
        server = PipelineServer.from_dspa_cr(cr)

        assert server.status == PipelineServerStatus.READY
        assert server.api_server_ready is True
        assert server.persistence_agent_ready is True
        assert server.scheduled_workflow_ready is True
        assert server.database_available is True
        assert server.object_store_available is True

    def test_pipeline_server_status_creating(self) -> None:
        """Some True conditions (but fewer than 5) produce CREATING status."""
        from rhoai_mcp.domains.pipelines.models import PipelineServer

        partial_conditions = [
            _make_condition("APIServerReady", status="True"),
            _make_condition("DatabaseAvailable", status="True"),
            _make_condition("PersistenceAgentReady", status="False"),
            _make_condition("ObjectStoreAvailable", status="False"),
        ]
        cr = _make_dspa_cr(conditions=partial_conditions)
        server = PipelineServer.from_dspa_cr(cr)

        assert server.status == PipelineServerStatus.CREATING
        assert server.api_server_ready is True
        assert server.database_available is True
        assert server.persistence_agent_ready is False
        assert server.object_store_available is False

    def test_pipeline_server_status_failed(self) -> None:
        """Condition with reason=Failed produces FAILED status."""
        from rhoai_mcp.domains.pipelines.models import PipelineServer

        failed_conditions = [
            _make_condition("APIServerReady", status="True"),
            _make_condition("DatabaseAvailable", status="False", reason="Failed"),
        ]
        cr = _make_dspa_cr(conditions=failed_conditions)
        server = PipelineServer.from_dspa_cr(cr)

        assert server.status == PipelineServerStatus.FAILED

    def test_pipeline_server_status_unknown_no_status(self) -> None:
        """No status object produces UNKNOWN status."""
        from rhoai_mcp.domains.pipelines.models import PipelineServer

        cr = _make_dspa_cr(conditions=None)
        server = PipelineServer.from_dspa_cr(cr)

        assert server.status == PipelineServerStatus.UNKNOWN

    def test_pipeline_server_status_unknown_empty_conditions(self) -> None:
        """Empty conditions list with no ready or failed produces UNKNOWN."""
        from rhoai_mcp.domains.pipelines.models import PipelineServer

        cr = _make_dspa_cr(conditions=[])
        server = PipelineServer.from_dspa_cr(cr)

        assert server.status == PipelineServerStatus.UNKNOWN
