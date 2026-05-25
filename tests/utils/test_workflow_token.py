"""Tests for workflow token HMAC signing and verification."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rhoai_mcp.utils.workflow_token import sign_step, verify_step, workflow_step


class TestSignStep:
    """Tests for sign_step function."""

    def test_returns_non_empty_string(self) -> None:
        """sign_step produces a non-empty token string."""
        token = sign_step("step_a", {"key": "value"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_contains_dot_separator(self) -> None:
        """Token format is payload.signature."""
        token = sign_step("step_a", {"key": "value"})
        parts = token.rsplit(".", 1)
        assert len(parts) == 2
        assert len(parts[0]) > 0
        assert len(parts[1]) > 0

    def test_different_data_produces_different_tokens(self) -> None:
        """Different input data produces different tokens."""
        token_a = sign_step("step_a", {"x": 1})
        token_b = sign_step("step_a", {"x": 2})
        assert token_a != token_b

    def test_different_steps_produce_different_tokens(self) -> None:
        """Different step names produce different tokens."""
        data = {"x": 1}
        token_a = sign_step("step_a", data)
        token_b = sign_step("step_b", data)
        assert token_a != token_b


class TestVerifyStepRoundTrip:
    """Tests for verify_step with valid tokens."""

    def test_valid_round_trip(self) -> None:
        """sign then verify returns the original data."""
        data = {"use_case": "chatbot", "user_count": 5000}
        token = sign_step("intent_extracted", data)
        result = verify_step(token, "intent_extracted")
        assert result == data

    def test_preserves_nested_data(self) -> None:
        """Nested dicts and lists survive round-trip."""
        data = {
            "specification": {
                "slo_targets": {"ttft_ms": 200, "itl_ms": 50},
                "gpu_types": ["H100", "A100-80"],
            },
            "count": 42,
        }
        token = sign_step("specs_prepared", data)
        result = verify_step(token, "specs_prepared")
        assert result == data


class TestVerifyStepRejection:
    """Tests for verify_step rejection cases."""

    def test_wrong_step_returns_error(self) -> None:
        """Verifying against wrong step name returns error dict."""
        token = sign_step("intent_extracted", {"x": 1})
        result = verify_step(token, "specs_prepared")
        assert "error" in result

    def test_tampered_payload_returns_error(self) -> None:
        """Modifying the payload invalidates the token."""
        token = sign_step("step_a", {"x": 1})
        parts = token.rsplit(".", 1)
        tampered = parts[0] + "XX." + parts[1]
        result = verify_step(tampered, "step_a")
        assert "error" in result

    def test_tampered_signature_returns_error(self) -> None:
        """Modifying the signature invalidates the token."""
        token = sign_step("step_a", {"x": 1})
        parts = token.rsplit(".", 1)
        tampered = parts[0] + "." + "bad" * 16
        result = verify_step(tampered, "step_a")
        assert "error" in result

    def test_malformed_token_no_dot_returns_error(self) -> None:
        """Token without dot separator returns error."""
        result = verify_step("nodothere", "step_a")
        assert "error" in result

    def test_empty_token_returns_error(self) -> None:
        """Empty string returns error."""
        result = verify_step("", "step_a")
        assert "error" in result

    def test_fabricated_token_returns_error(self) -> None:
        """A plausible but fabricated token is rejected."""
        import base64
        import json

        fake_payload = json.dumps({"step": "step_a", "data": {"x": 1}, "ts": 9999999999})
        encoded = base64.urlsafe_b64encode(fake_payload.encode()).decode()
        fake_token = f"{encoded}.{'a' * 64}"
        result = verify_step(fake_token, "step_a")
        assert "error" in result


class TestVerifyStepTTL:
    """Tests for token expiration."""

    def test_expired_token_returns_error(self) -> None:
        """Token older than TTL is rejected."""
        with (
            patch("rhoai_mcp.utils.workflow_token._get_ttl", return_value=3600),
            patch("rhoai_mcp.utils.workflow_token.time") as mock_time,
        ):
            mock_time.time.return_value = 1000.0
            token = sign_step("step_a", {"x": 1})

            mock_time.time.return_value = 4601.0  # 3601s later, past 1h default
            result = verify_step(token, "step_a")
            assert "error" in result
            assert "expired" in result["error"].lower()

    def test_token_within_ttl_succeeds(self) -> None:
        """Token within TTL is accepted."""
        with (
            patch("rhoai_mcp.utils.workflow_token._get_ttl", return_value=3600),
            patch("rhoai_mcp.utils.workflow_token.time") as mock_time,
        ):
            mock_time.time.return_value = 1000.0
            token = sign_step("step_a", {"x": 1})

            mock_time.time.return_value = 2800.0  # 1800s later, within 1h
            result = verify_step(token, "step_a")
            assert result == {"x": 1}


class TestWorkflowTokenConfig:
    """Tests for configuration."""

    def test_custom_secret_rejects_cross_secret_tokens(self) -> None:
        """Token signed with one secret is rejected under a different secret."""
        with patch("rhoai_mcp.utils.workflow_token._get_secret", return_value=b"secret-one"):
            token = sign_step("step_a", {"x": 1})

        with patch("rhoai_mcp.utils.workflow_token._get_secret", return_value=b"secret-two"):
            result = verify_step(token, "step_a")
            assert "error" in result
            assert "invalid" in result["error"].lower()

    def test_empty_hmac_secret_rejected_by_config(self) -> None:
        """Empty HMAC secret is rejected by pydantic validation."""
        from pydantic import ValidationError

        from rhoai_mcp.config import RHOAIConfig

        with pytest.raises(ValidationError, match="workflow_hmac_secret"):
            RHOAIConfig(workflow_hmac_secret="")

    def test_invalid_ttl_rejected_by_config(self) -> None:
        """Non-integer TTL value is rejected by pydantic validation."""
        from pydantic import ValidationError

        from rhoai_mcp.config import RHOAIConfig

        with pytest.raises(ValidationError, match="workflow_token_ttl"):
            RHOAIConfig(workflow_token_ttl="60s")  # type: ignore[arg-type]

    def test_custom_ttl(self) -> None:
        """Custom TTL is respected."""
        with (
            patch("rhoai_mcp.utils.workflow_token._get_ttl", return_value=60),
            patch("rhoai_mcp.utils.workflow_token.time") as mock_time,
        ):
            mock_time.time.return_value = 1000.0
            token = sign_step("step_a", {"x": 1})

            mock_time.time.return_value = 1061.0  # 61s later, past 60s TTL
            result = verify_step(token, "step_a")
            assert "error" in result
            assert "expired" in result["error"].lower()


class TestWorkflowStepProduces:
    """Tests for @workflow_step(produces=...)."""

    def test_adds_workflow_token_to_return(self) -> None:
        """Decorator adds workflow_token key to the return dict."""

        @workflow_step(produces="step_done")
        def my_tool(text: str) -> dict:
            return {"result": text.upper()}

        result = my_tool(text="hello")
        assert result["result"] == "HELLO"
        assert "workflow_token" in result

    def test_produced_token_is_verifiable(self) -> None:
        """Token added by decorator can be verified."""

        @workflow_step(produces="step_done")
        def my_tool(text: str) -> dict:
            return {"result": text.upper()}

        result = my_tool(text="hello")
        verified = verify_step(result["workflow_token"], "step_done")
        assert verified == {"result": "HELLO"}

    def test_does_not_sign_error_responses(self) -> None:
        """Decorator does not add token to error responses."""

        @workflow_step(produces="step_done")
        def my_tool(text: str) -> dict:
            return {"error": f"something went wrong with {text}"}

        result = my_tool(text="hello")
        assert "workflow_token" not in result
        assert result == {"error": "something went wrong with hello"}


class TestWorkflowStepRequires:
    """Tests for @workflow_step(requires=...)."""

    def test_verifies_and_replaces_token_with_data(self) -> None:
        """Decorator verifies token and passes data dict to function."""
        received = {}

        @workflow_step(requires="prev_step")
        def my_tool(workflow_token: str, extra: int = 0) -> dict:
            received["token_value"] = workflow_token
            return {"output": "done", "extra": extra}

        token = sign_step("prev_step", {"key": "val"})
        result = my_tool(workflow_token=token, extra=42)

        assert received["token_value"] == {"key": "val"}
        assert result == {"output": "done", "extra": 42}
        assert "workflow_token" not in result  # no produces = no signing

    def test_rejects_invalid_token(self) -> None:
        """Decorator returns error without calling the function."""
        called = False

        @workflow_step(requires="prev_step")
        def my_tool(workflow_token: str) -> dict:  # noqa: ARG001
            nonlocal called
            called = True
            return {"output": "done"}

        result = my_tool(workflow_token="fabricated")
        assert "error" in result
        assert not called

    def test_rejects_wrong_step(self) -> None:
        """Decorator returns error when token is from wrong step."""
        token = sign_step("wrong_step", {"x": 1})

        @workflow_step(requires="expected_step")
        def my_tool(workflow_token: str) -> dict:  # noqa: ARG001
            return {"output": "done"}

        result = my_tool(workflow_token=token)
        assert "error" in result

    def test_rejects_missing_token(self) -> None:
        """Decorator returns error when workflow_token is not passed."""

        @workflow_step(requires="prev_step")
        def my_tool(workflow_token: str = "") -> dict:  # noqa: ARG001
            return {"output": "done"}

        result = my_tool()
        assert "error" in result
