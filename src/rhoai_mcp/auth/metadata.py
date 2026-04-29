"""Protected Resource Metadata builder per RFC 9728."""

from typing import Any


def build_protected_resource_metadata(
    resource_url: str,
    issuer_url: str,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    """Build the Protected Resource Metadata document.

    Constructs an OAuth 2.0 Protected Resource Metadata document per RFC 9728
    that tells MCP clients where to find the authorization server for this
    MCP server.

    Args:
        resource_url: The URL of the protected resource (MCP server).
        issuer_url: The URL of the OIDC provider/authorization server.
        scopes: List of supported scopes. Defaults to ["openid"].

    Returns:
        A dictionary containing the Protected Resource Metadata document
        with the following fields:
        - resource: The protected resource URL
        - authorization_servers: List of authorization server URLs
        - scopes_supported: List of supported OAuth scopes
        - bearer_methods_supported: List of supported bearer token methods
    """
    return {
        "resource": resource_url,
        "authorization_servers": [issuer_url],
        "scopes_supported": scopes if scopes is not None else ["openid"],
        "bearer_methods_supported": ["header"],
    }
