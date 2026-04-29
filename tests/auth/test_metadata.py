from rhoai_mcp.auth.metadata import build_protected_resource_metadata


class TestProtectedResourceMetadata:
    def test_includes_authorization_servers(self):
        meta = build_protected_resource_metadata(
            resource_url="https://mcp.example.com",
            issuer_url="https://idp.example.com",
        )
        assert meta["authorization_servers"] == ["https://idp.example.com"]

    def test_includes_resource_url(self):
        meta = build_protected_resource_metadata(
            resource_url="https://mcp.example.com",
            issuer_url="https://idp.example.com",
        )
        assert meta["resource"] == "https://mcp.example.com"

    def test_includes_bearer_methods(self):
        meta = build_protected_resource_metadata(
            resource_url="https://mcp.example.com",
            issuer_url="https://idp.example.com",
        )
        assert meta["bearer_methods_supported"] == ["header"]

    def test_includes_scopes_when_provided(self):
        meta = build_protected_resource_metadata(
            resource_url="https://mcp.example.com",
            issuer_url="https://idp.example.com",
            scopes=["openid", "profile"],
        )
        assert meta["scopes_supported"] == ["openid", "profile"]

    def test_default_scopes_when_none(self):
        meta = build_protected_resource_metadata(
            resource_url="https://mcp.example.com",
            issuer_url="https://idp.example.com",
        )
        assert meta["scopes_supported"] == ["openid"]

    def test_explicit_empty_scopes_preserved(self):
        meta = build_protected_resource_metadata(
            resource_url="https://mcp.example.com",
            issuer_url="https://idp.example.com",
            scopes=[],
        )
        assert meta["scopes_supported"] == []
