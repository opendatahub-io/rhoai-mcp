"""Tests for the minimal OCI artifact client."""

from typing import Any

import pytest

from rhoai_mcp.domains.quickstarts import oci as oci_module
from rhoai_mcp.domains.quickstarts.oci import (
    MANIFEST_MEDIA_TYPE,
    OCIArtifactClient,
    OCIError,
    parse_ref,
)


class TestParseRef:
    """Tests for OCI reference parsing."""

    def test_registry_repo_tag(self) -> None:
        parsed = parse_ref("quay.io/org/name-manifest:1.0.0")
        assert parsed.registry == "quay.io"
        assert parsed.repository == "org/name-manifest"
        assert parsed.reference == "1.0.0"

    def test_defaults_to_latest(self) -> None:
        parsed = parse_ref("quay.io/org/name")
        assert parsed.reference == "latest"

    def test_digest_reference(self) -> None:
        parsed = parse_ref("quay.io/org/name@sha256:abc123")
        assert parsed.reference == "sha256:abc123"

    def test_strips_scheme(self) -> None:
        parsed = parse_ref("https://quay.io/org/name:tag")
        assert parsed.registry == "quay.io"
        assert parsed.repository == "org/name"
        assert parsed.reference == "tag"


class _FakeResp:
    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        if self._json is None:
            raise ValueError("no json")
        return self._json


class _FakeClient:
    def __init__(self, routes: list[tuple[str, _FakeResp]]) -> None:
        self._routes = routes
        self.calls: list[str] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def get(self, url: str, **_kwargs: Any) -> _FakeResp:
        self.calls.append(url)
        for substring, resp in self._routes:
            if substring in url:
                return resp
        raise AssertionError(f"unexpected URL: {url}")


class TestFetchLayer:
    """Tests for OCIArtifactClient.fetch_layer."""

    def test_happy_path_returns_matching_layer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = _FakeResp(
            200,
            json_data={"layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": "sha256:deadbeef"}]},
        )
        blob = _FakeResp(200, content=b"payload-bytes")
        fake = _FakeClient([("/manifests/", manifest), ("/blobs/", blob)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        result = OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

        assert result == b"payload-bytes"
        assert any("/manifests/1.0.0" in c for c in fake.calls)
        assert any("/blobs/sha256:deadbeef" in c for c in fake.calls)

    def test_missing_layer_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = _FakeResp(
            200, json_data={"layers": [{"mediaType": "other/type", "digest": "sha256:x"}]}
        )
        fake = _FakeClient([("/manifests/", manifest)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        with pytest.raises(OCIError):
            OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

    def test_manifest_http_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeClient([("/manifests/", _FakeResp(404))])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        with pytest.raises(OCIError):
            OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

    def test_anonymous_token_negotiation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        challenge = 'Bearer realm="https://auth.example/token",service="registry",scope="repository:org/name:pull"'
        manifest_401 = _FakeResp(401, headers={"www-authenticate": challenge})
        manifest_ok = _FakeResp(
            200,
            json_data={"layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": "sha256:abc"}]},
        )
        token = _FakeResp(200, json_data={"token": "tok-123"})
        blob = _FakeResp(200, content=b"data")

        # First /manifests/ call 401s, then after token the retry succeeds.
        seq = {"manifests": [manifest_401, manifest_ok]}

        class SeqClient(_FakeClient):
            def get(self, url: str, **_kwargs: Any) -> _FakeResp:
                self.calls.append(url)
                if "/token" in url or "auth.example" in url:
                    return token
                if "/manifests/" in url:
                    return seq["manifests"].pop(0)
                if "/blobs/" in url:
                    return blob
                raise AssertionError(f"unexpected URL: {url}")

        fake = SeqClient([])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        result = OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)
        assert result == b"data"
