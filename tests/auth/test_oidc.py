"""Tests for OIDC JWT validation."""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWKSet
from jwt.algorithms import RSAAlgorithm

from rhoai_mcp.auth.oidc import OIDCValidationError, OIDCValidator, ValidatedIdentity


@pytest.fixture
def rsa_keypair():
    """Generate a test RSA keypair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def make_token(rsa_keypair):
    """Factory to create signed JWT tokens."""
    import jwt

    private_key, _ = rsa_keypair

    def _make_token(
        claims: dict | None = None,
        headers: dict | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        default_claims = {
            "iss": "https://example.com",
            "aud": "test-audience",
            "sub": "user-123",
            "preferred_username": "testuser",
            "groups": ["group1", "group2"],
            "exp": now + timedelta(hours=1),
            "iat": now,
        }
        if claims:
            default_claims.update(claims)

        default_headers = {"kid": "test-key-1"}
        if headers:
            default_headers.update(headers)

        return jwt.encode(
            default_claims,
            private_key,
            algorithm="RS256",
            headers=default_headers,
        )

    return _make_token


@pytest.fixture
def jwks_response(rsa_keypair):
    """Build a JWKS response from the test public key."""
    import json

    _, public_key = rsa_keypair
    jwk_json = RSAAlgorithm.to_jwk(public_key)
    jwk_dict = json.loads(jwk_json)
    return {
        "keys": [
            {
                "kid": "test-key-1",
                "use": "sig",
                "kty": "RSA",
                **jwk_dict,
            }
        ]
    }


@pytest.fixture
def validator(jwks_response):
    """Create an OIDCValidator instance with pre-set JWKS."""
    v = OIDCValidator(
        issuer_url="https://example.com",
        audience="test-audience",
        username_claim="preferred_username",
        groups_claim="groups",
    )
    # Pre-set JWKS to skip discovery
    v._jwks = jwks_response
    v._jwks_fetched_at = time.time()
    return v


class TestOIDCValidatorBasics:
    """Test basic validator functionality."""

    def test_init_strips_trailing_slash(self):
        """Issuer URL should have trailing slash stripped."""
        v = OIDCValidator(
            issuer_url="https://example.com/",
            audience="test-aud",
        )
        assert v.issuer_url == "https://example.com"

    def test_init_no_trailing_slash(self):
        """Issuer URL without trailing slash should be unchanged."""
        v = OIDCValidator(
            issuer_url="https://example.com",
            audience="test-aud",
        )
        assert v.issuer_url == "https://example.com"

    def test_init_with_custom_claims(self):
        """Constructor should accept custom claim names."""
        v = OIDCValidator(
            issuer_url="https://example.com",
            audience="test-aud",
            username_claim="sub",
            groups_claim="roles",
        )
        assert v.username_claim == "sub"
        assert v.groups_claim == "roles"

    def test_default_claim_names(self):
        """Default claim names should be set."""
        v = OIDCValidator(
            issuer_url="https://example.com",
            audience="test-aud",
        )
        assert v.username_claim == "preferred_username"
        assert v.groups_claim == "groups"


class TestValidateToken:
    """Test token validation."""

    @pytest.mark.asyncio
    async def test_validate_token_success(self, validator, make_token):
        """Valid token should be decoded and return ValidatedIdentity."""
        token = make_token()
        identity = await validator.validate_token(token)

        assert isinstance(identity, ValidatedIdentity)
        assert identity.username == "testuser"
        assert identity.groups == ["group1", "group2"]
        assert identity.uid == "user-123"
        assert "sub" in identity.claims
        assert identity.claims["sub"] == "user-123"

    @pytest.mark.asyncio
    async def test_validate_token_expired(self, validator, make_token):
        """Expired token should raise OIDCValidationError."""
        now = datetime.now(timezone.utc)
        token = make_token(
            claims={
                "exp": now - timedelta(hours=1),
                "iat": now - timedelta(hours=2),
            }
        )

        with pytest.raises(OIDCValidationError) as exc_info:
            await validator.validate_token(token)
        assert "expired" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_token_wrong_audience(self, validator, make_token):
        """Token with wrong audience should raise OIDCValidationError."""
        token = make_token(claims={"aud": "wrong-audience"})

        with pytest.raises(OIDCValidationError) as exc_info:
            await validator.validate_token(token)
        assert "audience" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_token_wrong_issuer(self, validator, make_token):
        """Token with wrong issuer should raise OIDCValidationError."""
        token = make_token(claims={"iss": "https://different.com"})

        with pytest.raises(OIDCValidationError) as exc_info:
            await validator.validate_token(token)
        assert "issuer" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_token_missing_username_claim(self, validator, make_token):
        """Token missing username claim should raise OIDCValidationError."""
        token = make_token(claims={"preferred_username": None})

        with pytest.raises(OIDCValidationError) as exc_info:
            await validator.validate_token(token)
        assert "username" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_token_groups_missing_returns_empty(self, validator, make_token):
        """Token without groups claim should return empty list."""
        token = make_token(claims={"groups": None})
        identity = await validator.validate_token(token)

        assert identity.groups == []

    @pytest.mark.asyncio
    async def test_validate_token_groups_string_wrapped_in_list(self, validator, make_token):
        """If groups is a string, it should be wrapped in a list."""
        token = make_token(claims={"groups": "single-group"})
        identity = await validator.validate_token(token)

        assert identity.groups == ["single-group"]

    @pytest.mark.asyncio
    async def test_validate_token_uses_configured_username_claim(self, jwks_response):
        """Should use the configured username_claim."""
        v = OIDCValidator(
            issuer_url="https://example.com",
            audience="test-audience",
            username_claim="sub",
        )
        v._jwks = jwks_response
        v._jwks_fetched_at = time.time()

        import jwt as pyjwt

        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        now = datetime.now(timezone.utc)
        token = pyjwt.encode(
            {
                "iss": "https://example.com",
                "aud": "test-audience",
                "sub": "user-from-sub",
                "exp": now + timedelta(hours=1),
                "iat": now,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key-1"},
        )

        # Update JWKS to match new key
        import json

        public_key = private_key.public_key()
        jwk_json = RSAAlgorithm.to_jwk(public_key)
        jwk_dict = json.loads(jwk_json)
        v._jwks = {
            "keys": [
                {
                    "kid": "test-key-1",
                    "use": "sig",
                    "kty": "RSA",
                    **jwk_dict,
                }
            ]
        }

        identity = await v.validate_token(token)
        assert identity.username == "user-from-sub"

    @pytest.mark.asyncio
    async def test_validate_token_uses_configured_groups_claim(self, jwks_response):
        """Should use the configured groups_claim."""
        v = OIDCValidator(
            issuer_url="https://example.com",
            audience="test-audience",
            groups_claim="roles",
        )
        v._jwks = jwks_response
        v._jwks_fetched_at = time.time()

        import jwt as pyjwt

        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        now = datetime.now(timezone.utc)
        token = pyjwt.encode(
            {
                "iss": "https://example.com",
                "aud": "test-audience",
                "sub": "user-123",
                "preferred_username": "testuser",
                "roles": ["admin", "developer"],
                "exp": now + timedelta(hours=1),
                "iat": now,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key-1"},
        )

        # Update JWKS to match new key
        import json

        public_key = private_key.public_key()
        jwk_json = RSAAlgorithm.to_jwk(public_key)
        jwk_dict = json.loads(jwk_json)
        v._jwks = {
            "keys": [
                {
                    "kid": "test-key-1",
                    "use": "sig",
                    "kty": "RSA",
                    **jwk_dict,
                }
            ]
        }

        identity = await v.validate_token(token)
        assert identity.groups == ["admin", "developer"]


class TestDiscovery:
    """Test OIDC discovery."""

    @pytest.mark.asyncio
    async def test_discover_fetches_openid_config(self, rsa_keypair, jwks_response):
        """discover() should fetch OpenID config and JWKS."""
        _, public_key = rsa_keypair

        v = OIDCValidator(
            issuer_url="https://example.com",
            audience="test-audience",
        )

        openid_config = {
            "issuer": "https://example.com",
            "jwks_uri": "https://example.com/.well-known/jwks.json",
        }

        with patch("rhoai_mcp.auth.oidc.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock responses - json() is sync, raise_for_status() is sync
            config_response = AsyncMock()
            config_response.json = lambda: openid_config
            config_response.raise_for_status = lambda: None

            jwks_resp = AsyncMock()
            jwks_resp.json = lambda: jwks_response
            jwks_resp.raise_for_status = lambda: None

            mock_client.get = AsyncMock(side_effect=[config_response, jwks_resp])

            await v.discover()

            # Verify calls were made
            assert mock_client.get.call_count == 2
            assert v._jwks is not None
            assert "keys" in v._jwks

    @pytest.mark.asyncio
    async def test_ensure_jwks_calls_discover_if_none(self, validator):
        """_ensure_jwks should call discover if JWKS is None."""
        validator._jwks = None
        validator._jwks_fetched_at = None

        with patch.object(validator, "discover", new_callable=AsyncMock) as mock_discover:
            await validator._ensure_jwks()
            mock_discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_jwks_calls_discover_if_stale(self, validator):
        """_ensure_jwks should call discover if JWKS is older than TTL."""
        validator._jwks_fetched_at = time.time() - 4000  # Older than default 3600s TTL

        with patch.object(validator, "discover", new_callable=AsyncMock) as mock_discover:
            await validator._ensure_jwks()
            mock_discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_jwks_skips_discover_if_fresh(self, validator):
        """_ensure_jwks should skip discover if JWKS is fresh."""
        with patch.object(validator, "discover", new_callable=AsyncMock) as mock_discover:
            await validator._ensure_jwks()
            mock_discover.assert_not_called()


class TestValidatedIdentity:
    """Test ValidatedIdentity dataclass."""

    def test_validated_identity_required_fields(self):
        """Required fields should be present."""
        identity = ValidatedIdentity(
            username="user1",
            groups=["group1"],
        )
        assert identity.username == "user1"
        assert identity.groups == ["group1"]

    def test_validated_identity_optional_uid(self):
        """uid should be optional."""
        identity = ValidatedIdentity(
            username="user1",
            groups=[],
            uid="uid-123",
        )
        assert identity.uid == "uid-123"

    def test_validated_identity_optional_claims(self):
        """claims should be optional dict."""
        identity = ValidatedIdentity(
            username="user1",
            groups=[],
            claims={"custom": "value"},
        )
        assert identity.claims["custom"] == "value"

    def test_validated_identity_claims_default(self):
        """claims should default to empty dict."""
        identity = ValidatedIdentity(
            username="user1",
            groups=[],
        )
        assert identity.claims == {}


class TestOIDCValidationError:
    """Test OIDCValidationError exception."""

    def test_oidc_validation_error_is_authentication_error(self):
        """OIDCValidationError should be subclass of AuthenticationError."""
        from rhoai_mcp.utils.errors import AuthenticationError

        err = OIDCValidationError("test message")
        assert isinstance(err, AuthenticationError)

    def test_oidc_validation_error_message(self):
        """Error message should be accessible."""
        err = OIDCValidationError("custom error")
        assert "custom error" in str(err)
