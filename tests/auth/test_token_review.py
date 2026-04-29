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
    groups: list[str] | None = None,
) -> SimpleNamespace:
    """Build a mock TokenReview response."""
    user = SimpleNamespace(username=username, uid=uid, groups=groups, extra=None)
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
        """Authenticated token merges TokenReview and OCP User groups."""
        review_resp = _make_token_review_response(
            authenticated=True,
            username="alice",
            uid="uid-123",
            groups=["system:authenticated", "system:authenticated:oauth"],
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
        assert identity.groups == [
            "system:authenticated",
            "system:authenticated:oauth",
            "team-a",
            "team-b",
        ]

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

    async def test_validate_token_groups_lookup_failure_keeps_token_groups(
        self, validator: TokenReviewValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OCP groups lookup failure still returns TokenReview groups."""
        review_resp = _make_token_review_response(
            authenticated=True,
            username="bob",
            uid="uid-bob",
            groups=["system:authenticated"],
        )

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
        assert identity.groups == ["system:authenticated"]
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

    async def test_validate_token_non_list_groups_keeps_token_groups(
        self, validator: TokenReviewValidator
    ) -> None:
        """Non-list OCP User groups still returns TokenReview groups."""
        review_resp = _make_token_review_response(
            authenticated=True,
            username="carol",
            uid="uid-carol",
            groups=["system:authenticated"],
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
        assert identity.groups == ["system:authenticated"]

    async def test_validate_token_deduplicates_groups(
        self, validator: TokenReviewValidator
    ) -> None:
        """Groups appearing in both TokenReview and OCP User are deduplicated."""
        review_resp = _make_token_review_response(
            authenticated=True,
            username="dave",
            uid="uid-dave",
            groups=["system:authenticated", "team-a"],
        )

        with (
            patch.object(validator._authn_api, "create_token_review", return_value=review_resp),
            patch.object(
                validator._custom_api,
                "get_cluster_custom_object",
                return_value={"groups": ["team-a", "team-b"]},
            ),
        ):
            identity = await validator.validate_token("daves-token")

        assert identity.groups == ["system:authenticated", "team-a", "team-b"]
