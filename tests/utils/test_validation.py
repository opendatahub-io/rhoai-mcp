"""Tests for input validation utilities."""

import pytest

from rhoai_mcp.utils.errors import ValidationError
from rhoai_mcp.utils.validation import (
    validate_namespace,
    validate_resource_name,
    validate_writable_namespace,
)


class TestValidateNamespace:
    def test_valid(self) -> None:
        assert validate_namespace("ml-experiments") == "ml-experiments"

    def test_valid_simple(self) -> None:
        assert validate_namespace("default") == "default"

    def test_valid_with_numbers(self) -> None:
        assert validate_namespace("ns-123") == "ns-123"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("")

    def test_rejects_uppercase(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("MyNamespace")

    def test_rejects_shell_injection(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("foo; rm -rf /")

    def test_rejects_semicolons(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("ns;evil")

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("my namespace")

    def test_rejects_leading_hyphen(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("-bad-name")

    def test_rejects_trailing_hyphen(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("bad-name-")

    def test_rejects_trailing_newline(self) -> None:
        # '$' anchors match before a trailing newline; fullmatch must not.
        with pytest.raises(ValidationError, match="namespace"):
            validate_namespace("llm\n")


class TestValidateWritableNamespace:
    def test_allows_user_namespaces(self) -> None:
        assert validate_writable_namespace("llm") == "llm"
        assert validate_writable_namespace("ml-experiments") == "ml-experiments"

    def test_rejects_default(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            validate_writable_namespace("default")

    def test_rejects_kube_namespaces(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            validate_writable_namespace("kube-system")
        with pytest.raises(ValidationError, match="reserved"):
            validate_writable_namespace("kube-public")

    def test_rejects_openshift_namespaces(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            validate_writable_namespace("openshift-ingress")
        with pytest.raises(ValidationError, match="reserved"):
            validate_writable_namespace("openshift-ods-applications")
        with pytest.raises(ValidationError, match="reserved"):
            validate_writable_namespace("openshift")

    def test_rejects_invalid_names(self) -> None:
        with pytest.raises(ValidationError, match="namespace"):
            validate_writable_namespace("Bad Name")


class TestValidateResourceName:
    def test_valid(self) -> None:
        assert validate_resource_name("granite-serving") == "granite-serving"

    def test_valid_with_dots(self) -> None:
        assert validate_resource_name("my.model.v1") == "my.model.v1"

    def test_rejects_label_selector_injection(self) -> None:
        with pytest.raises(ValidationError, match="service_name"):
            validate_resource_name("foo,app=admin", field="service_name")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            validate_resource_name("")

    def test_rejects_uppercase(self) -> None:
        with pytest.raises(ValidationError):
            validate_resource_name("MyModel")

    def test_rejects_consecutive_dots(self) -> None:
        with pytest.raises(ValidationError):
            validate_resource_name("a..b")

    def test_rejects_leading_dot(self) -> None:
        with pytest.raises(ValidationError):
            validate_resource_name(".model")

    def test_rejects_empty_label_at_end(self) -> None:
        with pytest.raises(ValidationError):
            validate_resource_name("model.")

    def test_rejects_trailing_newline(self) -> None:
        with pytest.raises(ValidationError):
            validate_resource_name("granite-serving\n")

    def test_rejects_over_253_chars(self) -> None:
        with pytest.raises(ValidationError):
            validate_resource_name("a" * 254)
