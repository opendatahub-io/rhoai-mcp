"""Starlette middleware for OIDC token validation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from rhoai_mcp.auth.user_context import UserContext

if TYPE_CHECKING:
    from starlette.types import ASGIApp

    from rhoai_mcp.auth.oidc import OIDCValidator
    from rhoai_mcp.auth.token_review import TokenReviewValidator

logger = logging.getLogger(__name__)


class OIDCAuthMiddleware(BaseHTTPMiddleware):
    """Validates Bearer tokens and sets UserContext for each request."""

    def __init__(
        self,
        app: ASGIApp,
        validator: OIDCValidator | TokenReviewValidator,
        exclude_paths: list[str] | None = None,
        resource_metadata_url: str | None = None,
    ) -> None:
        super().__init__(app)
        self._validator = validator
        self._exclude_paths = set(exclude_paths or [])
        if resource_metadata_url and any(
            c in resource_metadata_url for c in ('"', "\r", "\n", "\x00")
        ):
            raise ValueError(
                "resource_metadata_url contains invalid characters for HTTP header use"
            )
        self._resource_metadata_url = resource_metadata_url

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Skip auth for excluded paths
        if path in self._exclude_paths:
            return await call_next(request)

        # Extract Bearer token (case-insensitive per RFC 7235)
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return self._unauthorized("Missing or invalid Authorization header")

        token = auth_header[7:]  # Strip "Bearer " (7 chars regardless of case)

        # Validate token — fail-closed: any exception results in 401
        try:
            identity = await self._validator.validate_token(token)
        except Exception as e:
            logger.warning("Token validation failed: %s", e)
            return self._unauthorized("Token validation failed")

        # Set user context for the duration of the request
        ctx = UserContext(
            username=identity.username,
            groups=identity.groups,
            uid=identity.uid,
        )
        reset_token = UserContext.set_current(ctx)
        try:
            return await call_next(request)
        finally:
            UserContext.reset_current(reset_token)

    def _unauthorized(self, detail: str) -> JSONResponse:
        headers: dict[str, str] = {}
        www_auth = "Bearer"
        if self._resource_metadata_url:
            www_auth += f' resource_metadata="{self._resource_metadata_url}"'
        headers["WWW-Authenticate"] = www_auth
        return JSONResponse(
            {"error": "unauthorized", "detail": detail},
            status_code=401,
            headers=headers,
        )
