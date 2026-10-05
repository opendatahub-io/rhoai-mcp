"""Tests for the minimal OCI artifact client."""

import hashlib
import json
from typing import Any

import pytest

from rhoai_mcp.domains.quickstarts import oci as oci_module
from rhoai_mcp.domains.quickstarts.oci import (
    MANIFEST_MEDIA_TYPE,
    OCIArtifactClient,
    OCIError,
    parse_ref,
)


def _digest(content: bytes) -> str:
    """The ``sha256:hex`` digest of ``content``, as the registry would compute it."""
    return "sha256:" + hashlib.sha256(content).hexdigest()


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

    def test_tag_and_digest_reference(self) -> None:
        # Canonical repo:tag@digest: the digest wins and the tag is dropped from
        # the repository path.
        parsed = parse_ref("quay.io/org/name-manifest:1.0.0@sha256:abc123")
        assert parsed.registry == "quay.io"
        assert parsed.repository == "org/name-manifest"
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
        payload = b"payload-bytes"
        manifest = _FakeResp(
            200,
            json_data={"layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": _digest(payload)}]},
        )
        blob = _FakeResp(200, content=payload)
        fake = _FakeClient([("/manifests/", manifest), ("/blobs/", blob)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        result = OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

        assert result == payload
        assert any("/manifests/1.0.0" in c for c in fake.calls)
        assert any(f"/blobs/{_digest(payload)}" in c for c in fake.calls)

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
            json_data={"layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": _digest(b"data")}]},
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

    def test_transport_error_mapped_to_oci_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Connect/timeout errors must not escape raw past the tools' except RHOAIError.
        class BoomClient(_FakeClient):
            def get(self, _url: str, **_kwargs: Any) -> _FakeResp:
                raise oci_module.httpx.ConnectError("connection refused")

        fake = BoomClient([])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        with pytest.raises(OCIError, match="failed to fetch OCI artifact"):
            OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

    def test_index_entry_without_digest_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A child without a digest is malformed; it is skipped rather than failing
        # the whole index. With no usable child, no layer is found.
        index = _FakeResp(
            200,
            json_data={
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [{}],
            },
        )
        fake = _FakeClient([("/manifests/", index)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        with pytest.raises(OCIError, match="no layer with media type"):
            OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

    def test_index_child_not_first_is_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The registry may reorder index children, so the wanted YAML artifact is
        # not necessarily first. Binding to children[0] would fetch the wrong child
        # and raise; every child must be searched for the layer.
        payload = b"payload-bytes"
        wrong_json = {"layers": [{"mediaType": "other/type", "digest": "sha256:x"}]}
        wrong_bytes = json.dumps(wrong_json).encode()
        right_json = {"layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": _digest(payload)}]}
        right_bytes = json.dumps(right_json).encode()
        wrong_digest = _digest(wrong_bytes)
        right_digest = _digest(right_bytes)

        index = _FakeResp(
            200,
            json_data={
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [{"digest": wrong_digest}, {"digest": right_digest}],
            },
        )
        wrong_child = _FakeResp(200, json_data=wrong_json, content=wrong_bytes)
        right_child = _FakeResp(200, json_data=right_json, content=right_bytes)
        blob = _FakeResp(200, content=payload)

        class IndexClient(_FakeClient):
            def get(self, url: str, **_kwargs: Any) -> _FakeResp:
                self.calls.append(url)
                if f"/manifests/{wrong_digest}" in url:
                    return wrong_child
                if f"/manifests/{right_digest}" in url:
                    return right_child
                if "/manifests/" in url:
                    return index
                if "/blobs/" in url:
                    return blob
                raise AssertionError(f"unexpected URL: {url}")

        fake = IndexClient([])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        result = OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)
        assert result == payload

    def test_layer_without_digest_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = _FakeResp(200, json_data={"layers": [{"mediaType": MANIFEST_MEDIA_TYPE}]})
        fake = _FakeClient([("/manifests/", manifest)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        with pytest.raises(OCIError, match="no digest"):
            OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

    def test_invalid_token_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        challenge = 'Bearer realm="https://auth.example/token",service="registry"'
        manifest_401 = _FakeResp(401, headers={"www-authenticate": challenge})
        token = _FakeResp(200)  # no json_data -> .json() raises ValueError

        class SeqClient(_FakeClient):
            def get(self, url: str, **_kwargs: Any) -> _FakeResp:
                self.calls.append(url)
                if "auth.example" in url or "/token" in url:
                    return token
                return manifest_401

        fake = SeqClient([])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        with pytest.raises(OCIError, match="invalid token response JSON"):
            OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)


class TestDigestVerification:
    """Digest-addressed fetches must verify the returned bytes, failing closed."""

    def test_pinned_manifest_verified_and_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = b"the-yaml-payload"
        manifest_json = {"layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": _digest(payload)}]}
        manifest_bytes = json.dumps(manifest_json).encode()
        manifest = _FakeResp(200, json_data=manifest_json, content=manifest_bytes)
        blob = _FakeResp(200, content=payload)
        fake = _FakeClient([("/manifests/", manifest), ("/blobs/", blob)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        ref = f"quay.io/org/name@{_digest(manifest_bytes)}"
        result = OCIArtifactClient().fetch_layer(ref, MANIFEST_MEDIA_TYPE)

        assert result == payload

    def test_manifest_digest_mismatch_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest_json = {"layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": "sha256:x"}]}
        # Content the registry returns does not match the pinned digest.
        manifest = _FakeResp(200, json_data=manifest_json, content=b"tampered-manifest")
        fake = _FakeClient([("/manifests/", manifest)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        ref = f"quay.io/org/name@{_digest(b'original-manifest')}"
        with pytest.raises(OCIError, match="content digest mismatch"):
            OCIArtifactClient().fetch_layer(ref, MANIFEST_MEDIA_TYPE)

    def test_blob_digest_mismatch_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Manifest fetched by tag (not verified), but the blob must still match
        # the layer digest the manifest declared.
        manifest_json = {
            "layers": [{"mediaType": MANIFEST_MEDIA_TYPE, "digest": _digest(b"expected")}]
        }
        manifest = _FakeResp(200, json_data=manifest_json)
        blob = _FakeResp(200, content=b"tampered-blob")
        fake = _FakeClient([("/manifests/", manifest), ("/blobs/", blob)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        with pytest.raises(OCIError, match="content digest mismatch"):
            OCIArtifactClient().fetch_layer("quay.io/org/name:1.0.0", MANIFEST_MEDIA_TYPE)

    def test_malformed_digest_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = _FakeResp(200, json_data={"layers": []}, content=b"whatever")
        fake = _FakeClient([("/manifests/", manifest)])
        monkeypatch.setattr(oci_module.httpx, "Client", lambda *_a, **_k: fake)

        # "@sha256:" with no hex is a digest-shaped reference with an empty hash.
        with pytest.raises(OCIError, match="malformed content digest"):
            OCIArtifactClient().fetch_layer("quay.io/org/name@sha256:", MANIFEST_MEDIA_TYPE)
