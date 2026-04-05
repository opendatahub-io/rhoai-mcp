"""Tests for TokenReview-based token validation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.auth.oidc import ValidatedIdentity
from rhoai_mcp.auth.token_review import TokenReviewError, TokenReviewValidator


def _make_token_review_response(
    authenticated: bool,
    username: str | None = None,
    uid: str | None = None,
) -> SimpleNamespace:
    """Build a mock TokenReview response."""
    user = SimpleNamespace(username=username, uid=uid, groups=None, extra=None)
    status = SimpleNamespace(authenticated=authenticated, user=user, error=None)
    return SimpleNamespace(status=status)


class TestTokenReviewValidator:
    """Tests for TokenReviewValidator."""

    @pytest.fixture
    def mock_api_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def validator(self, mock_api_client: MagicMock) -> TokenReviewValidator:
        return TokenReviewValidator(mock_api_client)

    async def test_validate_token_happy_path(self, validator: TokenReviewValidator) -> None:
        """Authenticated token with groups returns full ValidatedIdentity."""
        review_resp = _make_token_review_response(
            authenticated=True, username="alice", uid="uid-123"
        )

        with (
            patch.object(validator._authn_api, "create_token_review", return_value=review_resp),
            patch.object(
                validator._custom_api,
                "get_cluster_custom_object",
                return_value={"groups": ["team-a", "team-b"]},
            ),
        ):
            identity = await validator.validate_token("opaque-token-abc")

        assert isinstance(identity, ValidatedIdentity)
        assert identity.username == "alice"
        assert identity.uid == "uid-123"
        assert identity.groups == ["team-a", "team-b"]

    async def test_validate_token_unauthenticated_raises(
        self, validator: TokenReviewValidator
    ) -> None:
        """Unauthenticated token raises TokenReviewError."""
        review_resp = _make_token_review_response(authenticated=False)

        with (
            patch.object(validator._authn_api, "create_token_review", return_value=review_resp),
            pytest.raises(TokenReviewError, match="Token authentication failed"),
        ):
            await validator.validate_token("bad-token")

    async def test_validate_token_missing_username_raises(
        self, validator: TokenReviewValidator
    ) -> None:
        """Authenticated response without username raises TokenReviewError."""
        review_resp = _make_token_review_response(authenticated=True, username=None, uid="uid-456")

        with (
            patch.object(validator._authn_api, "create_token_review", return_value=review_resp),
            pytest.raises(TokenReviewError, match="No username"),
        ):
            await validator.validate_token("no-user-token")

    async def test_validate_token_empty_username_raises(
        self, validator: TokenReviewValidator
    ) -> None:
        """Authenticated response with empty username raises TokenReviewError."""
        review_resp = _make_token_review_response(authenticated=True, username="", uid="uid-789")

        with (
            patch.object(validator._authn_api, "create_token_review", return_value=review_resp),
            pytest.raises(TokenReviewError, match="No username"),
        ):
            await validator.validate_token("empty-user-token")

    async def test_validate_token_groups_lookup_failure_returns_empty(
        self, validator: TokenReviewValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Groups lookup failure falls back to empty groups with warning."""
        review_resp = _make_token_review_response(authenticated=True, username="bob", uid="uid-bob")

        with (
            patch.object(validator._authn_api, "create_token_review", return_value=review_resp),
            patch.object(
                validator._custom_api,
                "get_cluster_custom_object",
                side_effect=Exception("API not available"),
            ),
        ):
            identity = await validator.validate_token("bobs-token")

        assert identity.username == "bob"
        assert identity.groups == []
        assert "Failed to fetch OCP groups" in caplog.text

    async def test_validate_token_api_error_propagates(
        self, validator: TokenReviewValidator
    ) -> None:
        """TokenReview API error propagates as-is (not wrapped in TokenReviewError)."""
        with (
            patch.object(
                validator._authn_api,
                "create_token_review",
                side_effect=RuntimeError("connection refused"),
            ),
            pytest.raises(RuntimeError, match="connection refused"),
        ):
            await validator.validate_token("some-token")

    async def test_validate_token_non_list_groups_returns_empty(
        self, validator: TokenReviewValidator
    ) -> None:
        """Non-list groups value in OCP User response returns empty groups."""
        review_resp = _make_token_review_response(
            authenticated=True, username="carol", uid="uid-carol"
        )

        with (
            patch.object(validator._authn_api, "create_token_review", return_value=review_resp),
            patch.object(
                validator._custom_api,
                "get_cluster_custom_object",
                return_value={"groups": "not-a-list"},
            ),
        ):
            identity = await validator.validate_token("carols-token")

        assert identity.username == "carol"
        assert identity.groups == []
