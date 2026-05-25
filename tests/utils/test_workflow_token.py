"""Tests for workflow token HMAC signing and verification."""

from __future__ import annotations

from rhoai_mcp.utils.workflow_token import sign_step, verify_step


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
