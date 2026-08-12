"""Tests for ReadOnlyError and NotManagedByMCPError."""

from rhoai_mcp.utils.errors import (
    NotManagedByMCPError,
    OperationNotAllowedError,
    ReadOnlyError,
)


class TestReadOnlyError:
    """Test ReadOnlyError exception."""

    def test_is_operation_not_allowed(self) -> None:
        err = ReadOnlyError()
        assert isinstance(err, OperationNotAllowedError)

    def test_message_contains_read_only(self) -> None:
        err = ReadOnlyError()
        assert "read-only mode" in str(err)

    def test_operation_is_mutation(self) -> None:
        err = ReadOnlyError()
        assert err.details["operation"] == "mutation"


class TestNotManagedByMCPError:
    """Test NotManagedByMCPError exception."""

    def test_is_operation_not_allowed(self) -> None:
        err = NotManagedByMCPError("InferenceService", "my-model", "my-ns")
        assert isinstance(err, OperationNotAllowedError)

    def test_message_contains_kind_and_name(self) -> None:
        err = NotManagedByMCPError("InferenceService", "my-model", "my-ns")
        msg = str(err)
        assert "InferenceService" in msg
        assert "my-model" in msg
        assert "my-ns" in msg
        assert "not created by this MCP server" in msg

    def test_message_without_namespace(self) -> None:
        err = NotManagedByMCPError("ClusterTrainingRuntime", "my-runtime")
        msg = str(err)
        assert "ClusterTrainingRuntime" in msg
        assert "my-runtime" in msg
        assert "in namespace" not in msg

    def test_details_include_resource_info(self) -> None:
        err = NotManagedByMCPError("Secret", "my-secret", "ns")
        assert err.details["kind"] == "Secret"
        assert err.details["name"] == "my-secret"
        assert err.details["namespace"] == "ns"

    def test_operation_is_delete(self) -> None:
        err = NotManagedByMCPError("PVC", "vol", "ns")
        assert err.details["operation"] == "delete"
