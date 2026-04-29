"""Tests for OIDC auth middleware."""

from unittest.mock import AsyncMock

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from rhoai_mcp.auth.middleware import OIDCAuthMiddleware
from rhoai_mcp.auth.oidc import OIDCValidationError, ValidatedIdentity
from rhoai_mcp.auth.user_context import UserContext


def make_app(validator):
    """Create a test Starlette app with auth middleware."""

    async def protected(_request: Request) -> JSONResponse:
        ctx = UserContext.current()
        return JSONResponse({"user": ctx.username if ctx else None})

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/test", protected),
        ]
    )
    app.add_middleware(
        OIDCAuthMiddleware,
        validator=validator,
        exclude_paths=["/health"],
    )
    return app


@pytest.fixture
def mock_validator():
    v = AsyncMock()
    v.validate_token = AsyncMock(
        return_value=ValidatedIdentity(
            username="alice",
            groups=["team-a"],
            uid="user-123",
        )
    )
    return v


class TestOIDCAuthMiddleware:
    def test_excluded_path_passes_through(self, mock_validator):
        app = make_app(mock_validator)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_missing_auth_header_returns_401(self, mock_validator):
        app = make_app(mock_validator)
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers

    def test_invalid_auth_scheme_returns_401(self, mock_validator):
        app = make_app(mock_validator)
        client = TestClient(app)
        resp = client.get("/test", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401

    def test_valid_token_sets_user_context(self, mock_validator):
        app = make_app(mock_validator)
        client = TestClient(app)
        resp = client.get("/test", headers={"Authorization": "Bearer valid-token"})
        assert resp.status_code == 200
        assert resp.json()["user"] == "alice"

    def test_bearer_scheme_case_insensitive(self, mock_validator):
        app = make_app(mock_validator)
        client = TestClient(app)
        resp = client.get("/test", headers={"Authorization": "bearer valid-token"})
        assert resp.status_code == 200
        assert resp.json()["user"] == "alice"

    def test_invalid_token_returns_401(self, mock_validator):
        mock_validator.validate_token = AsyncMock(side_effect=OIDCValidationError("expired"))
        app = make_app(mock_validator)
        client = TestClient(app)
        resp = client.get("/test", headers={"Authorization": "Bearer bad-token"})
        assert resp.status_code == 401

    def test_unexpected_validator_error_returns_401(self, mock_validator):
        """Non-AuthenticationError exceptions should also result in 401 (fail-closed)."""
        mock_validator.validate_token = AsyncMock(side_effect=RuntimeError("network timeout"))
        app = make_app(mock_validator)
        client = TestClient(app)
        resp = client.get("/test", headers={"Authorization": "Bearer bad-token"})
        assert resp.status_code == 401
        # Should NOT leak internal error details
        assert "network timeout" not in resp.json().get("detail", "")

    def test_user_context_reset_after_request(self, mock_validator):
        app = make_app(mock_validator)
        client = TestClient(app)
        client.get("/test", headers={"Authorization": "Bearer valid-token"})
        # After request completes, context should be cleared
        assert UserContext.current() is None
