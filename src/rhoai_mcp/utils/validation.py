"""Input validation for K8s resource names and MCP tool parameters."""

from __future__ import annotations

import re

from rhoai_mcp.utils.errors import ValidationError

# A DNS-1123 label: lowercase alphanumeric, hyphens inside, max 63 chars.
# Matched with re.fullmatch so no anchor/trailing-newline bypass is possible.
_DNS_LABEL_RE = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")

_MAX_SUBDOMAIN_LENGTH = 253

# Namespace names and prefixes reserved by Kubernetes and OpenShift. MCP
# tools must never create or write resources into platform namespaces.
_PROTECTED_NAMESPACE_PREFIXES = ("kube-", "openshift")
_PROTECTED_NAMESPACES = ("default",)


def validate_namespace(value: str) -> str:
    """Validate a Kubernetes namespace (DNS-1123 label).

    Raises ValidationError if the value is not a valid namespace name.
    Returns the validated value unchanged.
    """
    if not value or not _DNS_LABEL_RE.fullmatch(value):
        raise ValidationError(
            f"Invalid namespace: '{_truncate(value)}'. "
            "Must be a lowercase DNS label (a-z, 0-9, hyphens, max 63 chars).",
            field="namespace",
        )
    return value


def validate_writable_namespace(value: str) -> str:
    """Validate a namespace that MCP tools may create or write into.

    In addition to DNS-1123 label validation, rejects the namespaces
    reserved by Kubernetes (kube-*), OpenShift (openshift*), and 'default',
    so write tools never touch platform namespaces by accident.
    """
    validate_namespace(value)
    if value in _PROTECTED_NAMESPACES or value.startswith(_PROTECTED_NAMESPACE_PREFIXES):
        raise ValidationError(
            f"Namespace '{_truncate(value)}' is reserved by the platform and "
            "cannot be written to by this tool. Use a user project namespace.",
            field="namespace",
        )
    return value


def validate_resource_name(value: str, field: str = "name") -> str:
    """Validate a Kubernetes resource name (DNS-1123 subdomain).

    A subdomain is dot-separated DNS labels, each at most 63 chars, with a
    total length of at most 253 chars. Raises ValidationError if the value
    is not a valid resource name. Returns the validated value unchanged.
    """
    if (
        not value
        or len(value) > _MAX_SUBDOMAIN_LENGTH
        or not all(_DNS_LABEL_RE.fullmatch(label) for label in value.split("."))
    ):
        raise ValidationError(
            f"Invalid {field}: '{_truncate(value)}'. "
            "Must be a lowercase DNS subdomain (a-z, 0-9, hyphens, dots, "
            "max 253 chars, no empty or consecutive labels).",
            field=field,
        )
    return value


def _truncate(value: str, max_len: int = 40) -> str:
    """Truncate a value for safe inclusion in error messages."""
    if not value:
        return "<empty>"
    if len(value) > max_len:
        return value[:max_len] + "..."
    return value
