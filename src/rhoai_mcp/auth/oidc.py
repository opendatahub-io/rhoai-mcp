"""OIDC JWT validation for RHOAI MCP."""

import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
import jwt
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    PyJWTError,
)

from rhoai_mcp.utils.errors import AuthenticationError


class OIDCValidationError(AuthenticationError):
    """Error during OIDC JWT validation."""

    pass


@dataclass
class ValidatedIdentity:
    """Identity extracted from a validated JWT token."""

    username: str
    groups: list[str]
    uid: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)


class OIDCValidator:
    """Validates JWTs against an OIDC provider's JWKS."""

    def __init__(
        self,
        issuer_url: str,
        audience: str,
        username_claim: str = "preferred_username",
        groups_claim: str = "groups",
        jwks_cache_ttl: int = 3600,
    ) -> None:
        """Initialize OIDC validator.

        Args:
            issuer_url: OIDC provider issuer URL (trailing slash will be stripped)
            audience: Expected token audience
            username_claim: Token claim to extract username from
            groups_claim: Token claim to extract groups from
            jwks_cache_ttl: JWKS cache time-to-live in seconds
        """
        self.issuer_url = issuer_url.rstrip("/")
        if urlparse(self.issuer_url).scheme != "https":
            raise ValueError("issuer_url must use https")
        self.audience = audience
        self.username_claim = username_claim
        self.groups_claim = groups_claim
        self.jwks_cache_ttl = jwks_cache_ttl

        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: float | None = None
        self._last_forced_refresh_at: float | None = None

    async def discover(self) -> None:
        """Fetch OIDC discovery document and JWKS."""
        openid_config_url = f"{self.issuer_url}/.well-known/openid-configuration"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Fetch OpenID configuration
                config_resp = await client.get(openid_config_url)
                config_resp.raise_for_status()
                config = config_resp.json()

                # Fetch JWKS from the provided URI
                jwks_uri = config.get("jwks_uri")
                if not jwks_uri:
                    raise OIDCValidationError("JWKS URI not found in OIDC configuration")

                jwks_resp = await client.get(jwks_uri)
                jwks_resp.raise_for_status()
                self._jwks = jwks_resp.json()
                self._jwks_fetched_at = time.time()
        except OIDCValidationError:
            raise
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise OIDCValidationError(f"OIDC discovery failed: {e}") from e

    async def _ensure_jwks(self) -> None:
        """Ensure JWKS is loaded and fresh."""
        if self._jwks is None:
            await self.discover()
            return

        if self._jwks_fetched_at is None:
            await self.discover()
            return

        # Check if JWKS is stale
        if time.time() - self._jwks_fetched_at > self.jwks_cache_ttl:
            await self.discover()

    def _find_key(self, kid: str) -> Any:
        """Find a signing key by kid in the cached JWKS."""
        if self._jwks is None:
            return None
        jwks_set = jwt.PyJWKSet.from_dict(self._jwks)
        for jwk in jwks_set.keys:
            if jwk.key_id == kid:
                return jwk.key
        return None

    async def validate_token(self, token: str) -> ValidatedIdentity:
        """Validate a JWT token and extract identity.

        Args:
            token: JWT token string

        Returns:
            ValidatedIdentity with extracted username, groups, and claims

        Raises:
            OIDCValidationError: If token is invalid, expired, or has wrong aud/iss
        """
        # Ensure JWKS is loaded
        await self._ensure_jwks()

        try:
            # Decode header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                raise OIDCValidationError("Token missing 'kid' in header")

            # Get the signing key from JWKS
            key = self._find_key(kid)
            if key is None:
                # Key not found — refresh JWKS for key rotation, rate-limited (CWE-400)
                now = time.time()
                if self._last_forced_refresh_at is None or now - self._last_forced_refresh_at > 30:
                    await self.discover()
                    self._last_forced_refresh_at = now
                key = self._find_key(kid)
                if key is None:
                    raise OIDCValidationError(f"Key with id '{kid}' not found in JWKS")

            # Decode and validate token
            decoded = jwt.decode(
                token,
                key,
                algorithms=["RS256", "ES256"],
                audience=self.audience,
                issuer=self.issuer_url,
                options={"require": ["exp", "iss", "aud"]},
            )

        except ExpiredSignatureError:
            raise OIDCValidationError("Token has expired")
        except InvalidAudienceError:
            raise OIDCValidationError("Invalid token audience")
        except InvalidIssuerError:
            raise OIDCValidationError("Invalid token issuer")
        except PyJWTError as e:
            raise OIDCValidationError(f"Token validation failed: {e}")

        # Extract username
        username = decoded.get(self.username_claim)
        if not username or not isinstance(username, str):
            raise OIDCValidationError(
                f"Token missing or non-string claim '{self.username_claim}' (username)"
            )

        # Extract groups
        groups_value = decoded.get(self.groups_claim)
        if groups_value is None:
            groups: list[str] = []
        elif isinstance(groups_value, str):
            groups = [groups_value]
        elif isinstance(groups_value, list):
            if not all(isinstance(g, str) and g for g in groups_value):
                raise OIDCValidationError(
                    f"Token claim '{self.groups_claim}' must contain only non-empty strings"
                )
            groups = groups_value
        else:
            raise OIDCValidationError(
                f"Token claim '{self.groups_claim}' must be a string or list of strings"
            )

        # Validate extracted values are safe for downstream use (CWE-20/CWE-113)
        username = self._validate_identity_value(username, "username")
        groups = [self._validate_identity_value(g, "group") for g in groups]

        return ValidatedIdentity(
            username=username,
            groups=groups,
            uid=decoded.get("sub"),
            claims=decoded,
        )

    @staticmethod
    def _validate_identity_value(value: str, field: str) -> str:
        """Reject identity values containing control chars or leading/trailing whitespace."""
        if "\r" in value or "\n" in value or "\x00" in value:
            raise OIDCValidationError(f"Invalid {field}: contains control characters")
        stripped = value.strip()
        if not stripped:
            raise OIDCValidationError(f"Invalid {field}: empty value")
        if stripped != value:
            raise OIDCValidationError(f"Invalid {field}: contains leading or trailing whitespace")
        return value
