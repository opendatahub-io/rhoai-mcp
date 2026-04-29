"""RBAC permission checking via SubjectAccessReview."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kubernetes import client as k8s_client  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolPermission:
    """A single K8s API permission required by a tool."""

    api_group: str
    resource: str
    verb: str

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> ToolPermission:
        """Create a ToolPermission from a dict with camelCase keys."""
        required = ("apiGroup", "resource", "verb")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Permission dict missing required keys: {missing}")
        return cls(api_group=d["apiGroup"], resource=d["resource"], verb=d["verb"])

    @property
    def cache_key(self) -> str:
        """Return a unique key for deduplicating permission checks."""
        return f"{self.api_group}/{self.resource}/{self.verb}"


class RBACChecker:
    """Checks user permissions via K8s SubjectAccessReview."""

    def __init__(self, authz_api: k8s_client.AuthorizationV1Api) -> None:
        self._authz_api = authz_api

    def check_permission(
        self, username: str, groups: list[str], permission: ToolPermission
    ) -> bool:
        """Check if a user has a single permission."""
        review = k8s_client.V1SubjectAccessReview(
            spec=k8s_client.V1SubjectAccessReviewSpec(
                user=username,
                groups=groups,
                resource_attributes=k8s_client.V1ResourceAttributes(
                    verb=permission.verb,
                    resource=permission.resource,
                    group=permission.api_group,
                ),
            )
        )
        result = self._authz_api.create_subject_access_review(review)
        return bool(result.status.allowed)

    def filter_tools(
        self,
        username: str,
        groups: list[str],
        tool_permissions: dict[str, list[ToolPermission]],
    ) -> set[str]:
        """Return the set of tool names the user is allowed to use.

        Deduplicates SubjectAccessReview calls across tools sharing the same
        underlying permission.
        """
        # Collect unique permissions
        unique_perms: set[ToolPermission] = set()
        for perms in tool_permissions.values():
            unique_perms.update(perms)

        # Check each unique permission once
        perm_results: dict[str, bool] = {}
        for perm in unique_perms:
            if perm.cache_key not in perm_results:
                perm_results[perm.cache_key] = self.check_permission(username, groups, perm)

        # Filter tools: visible only if ALL permissions are allowed
        allowed_tools: set[str] = set()
        for tool_name, perms in tool_permissions.items():
            if all(perm_results.get(p.cache_key, False) for p in perms):
                allowed_tools.add(tool_name)

        logger.info(
            "RBAC check for %s: %d/%d tools allowed (%d unique permissions checked)",
            username,
            len(allowed_tools),
            len(tool_permissions),
            len(unique_perms),
        )
        return allowed_tools
