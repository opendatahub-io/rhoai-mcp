"""TokenReview-based token validation for OCP OAuth tokens."""

import asyncio
import logging

from kubernetes import client as k8s_client  # type: ignore[import-untyped]

from rhoai_mcp.auth.oidc import ValidatedIdentity
from rhoai_mcp.utils.errors import AuthenticationError

logger = logging.getLogger(__name__)


class TokenReviewError(AuthenticationError):
    """Error during TokenReview validation."""

    pass


class TokenReviewValidator:
    """Validates opaque bearer tokens via the K8s TokenReview API.

    After authentication, fetches OCP User groups for impersonation.
    Falls back to empty groups if the lookup fails (e.g. non-OCP cluster).
    """

    def __init__(self, api_client: k8s_client.ApiClient) -> None:
        self._authn_api = k8s_client.AuthenticationV1Api(api_client)
        self._custom_api = k8s_client.CustomObjectsApi(api_client)

    async def validate_token(self, token: str) -> ValidatedIdentity:
        """Validate a bearer token via TokenReview and return identity.

        Args:
            token: Opaque bearer token string.

        Returns:
            ValidatedIdentity with username, uid, and groups.

        Raises:
            TokenReviewError: If the token is invalid or missing username.
        """
        review = k8s_client.V1TokenReview(spec=k8s_client.V1TokenReviewSpec(token=token))

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._authn_api.create_token_review, review),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            raise TokenReviewError("TokenReview API call timed out")

        if not result.status.authenticated or result.status.user is None:
            raise TokenReviewError("Token authentication failed")

        username = result.status.user.username
        if not username:
            raise TokenReviewError("No username in TokenReview response")

        uid = result.status.user.uid

        # Start with system groups from TokenReview (e.g. system:authenticated)
        token_groups = result.status.user.groups or []

        # Merge with explicit OCP User groups
        ocp_groups = await self._fetch_ocp_groups(username)

        # Deduplicate while preserving order
        seen: set[str] = set()
        groups: list[str] = []
        for g in [*token_groups, *ocp_groups]:
            if g not in seen:
                seen.add(g)
                groups.append(g)

        return ValidatedIdentity(
            username=username,
            groups=groups,
            uid=uid,
        )

    async def _fetch_ocp_groups(self, username: str) -> list[str]:
        """Fetch group memberships from the OCP User API.

        Returns empty list if the lookup fails (non-OCP cluster, missing RBAC).
        """
        try:
            user_obj = await asyncio.wait_for(
                asyncio.to_thread(
                    self._custom_api.get_cluster_custom_object,
                    group="user.openshift.io",
                    version="v1",
                    plural="users",
                    name=username,
                ),
                timeout=10.0,
            )
            groups = user_obj.get("groups", [])
            if isinstance(groups, list):
                return groups
            return []
        except Exception:
            logger.warning("Failed to fetch OCP groups for authenticated user", exc_info=True)
            return []
