"""Pure ASGI middleware for OIDC/TokenReview token validation.

Uses raw ASGI instead of Starlette's BaseHTTPMiddleware because
BaseHTTPMiddleware wraps responses through body_stream which is
incompatible with SSE (Server-Sent Events) streaming responses.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from rhoai_mcp.auth.user_context import UserContext

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

    from rhoai_mcp.auth.oidc import OIDCValidator
    from rhoai_mcp.auth.token_review import TokenReviewValidator

logger = logging.getLogger(__name__)


class OIDCAuthMiddleware:
    """Validates Bearer tokens and sets UserContext for each request.

    Implemented as a pure ASGI middleware to support SSE streaming
    responses, which are incompatible with Starlette's BaseHTTPMiddleware.
    """

    def __init__(
        self,
        app: ASGIApp,
        validator: OIDCValidator | TokenReviewValidator,
        exclude_paths: list[str] | None = None,
        resource_metadata_url: str | None = None,
    ) -> None:
        self.app = app
        self._validator = validator
        self._exclude_paths = set(exclude_paths or [])
        if resource_metadata_url and any(
            c in resource_metadata_url for c in ('"', "\r", "\n", "\x00")
        ):
            raise ValueError(
                "resource_metadata_url contains invalid characters for HTTP header use"
            )
        self._resource_metadata_url = resource_metadata_url

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Skip auth for excluded paths
        if path in self._exclude_paths:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header from raw ASGI headers
        auth_header = ""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                auth_header = value.decode("latin-1")
                break

        if not auth_header.lower().startswith("bearer "):
            await self._send_401(send, "Missing or invalid Authorization header")
            return

        token = auth_header[7:]  # Strip "Bearer " (7 chars regardless of case)

        # Validate token — fail-closed: any exception results in 401
        try:
            identity = await self._validator.validate_token(token)
        except Exception as e:
            logger.warning("Token validation failed: %s", e)
            await self._send_401(send, "Token validation failed")
            return

        # Set user context for the duration of the request
        ctx = UserContext(
            username=identity.username,
            groups=identity.groups,
            uid=identity.uid,
        )
        reset_token = UserContext.set_current(ctx)
        try:
            await self.app(scope, receive, send)
        finally:
            UserContext.reset_current(reset_token)

    async def _send_401(self, send: Send, detail: str) -> None:
        www_auth = "Bearer"
        if self._resource_metadata_url:
            www_auth += f' resource_metadata="{self._resource_metadata_url}"'

        body = json.dumps({"error": "unauthorized", "detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", www_auth.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
