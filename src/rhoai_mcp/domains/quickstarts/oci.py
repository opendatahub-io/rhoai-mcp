"""Minimal OCI artifact client for fetching quickstart YAML payloads.

Quickstart catalog data is published to an OCI registry (Quay) as OCI
artifacts: a single YAML layer identified by a custom media type. This module
fetches that layer over the OCI Distribution v2 HTTP API using ``httpx`` — no
``oras`` dependency — following the project's minimal-dependency principle.

Access is anonymous (the public ``rh-ai-quickstart`` repositories require no
credentials). Anonymous bearer tokens are negotiated from the registry's
``WWW-Authenticate`` challenge, so this works against any standards-compliant
registry, not just Quay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from rhoai_mcp.utils.errors import RHOAIError

# Media types for quickstart OCI artifacts. These identify the single YAML
# layer inside each artifact and must match what the publisher (oras) sets.
REGISTRY_MEDIA_TYPE = "application/vnd.redhat.quickstart.registry.v1+yaml"
MANIFEST_MEDIA_TYPE = "application/vnd.redhat.quickstart.manifest.v1+yaml"

_MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ]
)

_INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}

_CHALLENGE_PARAM_RE = re.compile(r'(\w+)="([^"]*)"')


class OCIError(RHOAIError):
    """Raised when an OCI artifact cannot be fetched or parsed."""


@dataclass(frozen=True)
class ParsedRef:
    """A parsed OCI reference (registry / repository / tag-or-digest)."""

    registry: str
    repository: str
    reference: str


def parse_ref(ref: str) -> ParsedRef:
    """Parse ``registry/repository[:tag|@digest]`` into its components.

    Defaults the reference to ``latest`` when neither a tag nor a digest is
    present, matching the OCI convention.
    """
    if "://" in ref:
        ref = ref.split("://", 1)[1]

    reference = "latest"
    if "@" in ref:
        ref, digest = ref.rsplit("@", 1)
        reference = digest
    else:
        slash = ref.rfind("/")
        colon = ref.rfind(":")
        if colon > slash:
            reference = ref[colon + 1 :]
            ref = ref[:colon]

    parts = ref.split("/", 1)
    if len(parts) == 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry, repository = parts
    else:
        # Bare name (no registry host) — default to Docker Hub's registry.
        registry, repository = "registry-1.docker.io", ref

    if not repository:
        raise OCIError(f"invalid OCI reference (no repository): {ref!r}")

    return ParsedRef(registry=registry, repository=repository, reference=reference)


class OCIArtifactClient:
    """Fetches the YAML payload of a quickstart OCI artifact."""

    def __init__(self, timeout: int = 30, verify: bool = True) -> None:
        self._timeout = timeout
        self._verify = verify

    def fetch_layer(self, ref: str, media_type: str) -> bytes:
        """Return the bytes of the single layer whose media type equals ``media_type``.

        Args:
            ref: OCI reference, e.g. ``quay.io/org/name-manifest:1.0.0``.
            media_type: The layer media type to extract.

        Raises:
            OCIError: On any HTTP failure or if no matching layer exists.
        """
        parsed = parse_ref(ref)
        base = f"https://{parsed.registry}"
        token_box: dict[str, str] = {}

        with httpx.Client(
            timeout=self._timeout, verify=self._verify, follow_redirects=True
        ) as client:
            manifest = self._fetch_manifest(client, base, parsed, token_box)

            if manifest.get("mediaType") in _INDEX_MEDIA_TYPES or (
                "manifests" in manifest and "layers" not in manifest
            ):
                children = manifest.get("manifests") or []
                if not children:
                    raise OCIError(f"empty image index for {ref}")
                child_ref = ParsedRef(parsed.registry, parsed.repository, children[0]["digest"])
                manifest = self._fetch_manifest(client, base, child_ref, token_box)

            for layer in manifest.get("layers", []):
                if layer.get("mediaType") == media_type:
                    return self._fetch_blob(client, base, parsed, layer["digest"], token_box)

            raise OCIError(f"no layer with media type {media_type!r} in {ref}")

    def _fetch_manifest(
        self,
        client: httpx.Client,
        base: str,
        parsed: ParsedRef,
        token_box: dict[str, str],
    ) -> dict[str, Any]:
        url = f"{base}/v2/{parsed.repository}/manifests/{parsed.reference}"
        resp = self._authorized_get(client, url, {"Accept": _MANIFEST_ACCEPT}, parsed, token_box)
        if resp.status_code != 200:
            raise OCIError(
                f"failed to fetch manifest {parsed.repository}:{parsed.reference} "
                f"(HTTP {resp.status_code})"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise OCIError(f"invalid manifest JSON for {parsed.repository}: {exc}")
        if not isinstance(data, dict):
            raise OCIError(f"unexpected manifest JSON for {parsed.repository} (not an object)")
        return data

    def _fetch_blob(
        self,
        client: httpx.Client,
        base: str,
        parsed: ParsedRef,
        digest: str,
        token_box: dict[str, str],
    ) -> bytes:
        url = f"{base}/v2/{parsed.repository}/blobs/{digest}"
        resp = self._authorized_get(client, url, {}, parsed, token_box)
        if resp.status_code != 200:
            raise OCIError(f"failed to fetch blob {digest} (HTTP {resp.status_code})")
        return resp.content

    def _authorized_get(
        self,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        parsed: ParsedRef,
        token_box: dict[str, str],
    ) -> httpx.Response:
        """GET ``url``, negotiating an anonymous bearer token on a 401 challenge."""
        request_headers = dict(headers)
        if token_box.get("token"):
            request_headers["Authorization"] = f"Bearer {token_box['token']}"

        resp = client.get(url, headers=request_headers)
        if resp.status_code == 401 and "www-authenticate" in resp.headers:
            token = self._fetch_token(client, resp.headers["www-authenticate"], parsed)
            if token:
                token_box["token"] = token
                request_headers["Authorization"] = f"Bearer {token}"
                resp = client.get(url, headers=request_headers)
        return resp

    def _fetch_token(
        self, client: httpx.Client, challenge: str, parsed: ParsedRef
    ) -> str | None:
        """Exchange a ``WWW-Authenticate: Bearer`` challenge for an access token."""
        if not challenge.lower().startswith("bearer "):
            return None

        params = dict(_CHALLENGE_PARAM_RE.findall(challenge))
        realm = params.get("realm")
        if not realm:
            return None

        query: dict[str, str] = {"scope": params.get("scope") or f"repository:{parsed.repository}:pull"}
        if params.get("service"):
            query["service"] = params["service"]

        resp = client.get(realm, params=query)
        if resp.status_code != 200:
            raise OCIError(f"token request failed (HTTP {resp.status_code})")
        data = resp.json()
        token = data.get("token") or data.get("access_token")
        return token if isinstance(token, str) else None
