"""Smoke tests for kustomize overlays.

Runs ``kubectl kustomize`` on specific overlay and validates the rendered output.
Skipped automatically when ``kubectl`` is not available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

KUSTOMIZE_ROOT = Path(__file__).resolve().parents[2] / "deploy" / "kustomize"
OVERLAY_DIR = KUSTOMIZE_ROOT / "overlays"

kubectl = shutil.which("kubectl")
skip_no_kubectl = pytest.mark.skipif(kubectl is None, reason="kubectl not found on PATH")


def _kustomize_build(overlay: str) -> list[dict]:
    result = subprocess.run(
        ["kubectl", "kustomize", str(OVERLAY_DIR / overlay)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"kustomize build failed for overlay '{overlay}':\n{result.stderr}"
    )
    return list(yaml.safe_load_all(result.stdout))


def _find_resource(docs: list[dict], kind: str, name: str) -> dict | None:
    for doc in docs:
        if doc and doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
            return doc
    return None


@skip_no_kubectl
class TestOpenshiftOidcOverlay:
    """Validate the rendered manifests for the openshift-oidc overlay."""

    @pytest.fixture(scope="class")
    def docs(self) -> list[dict]:
        return _kustomize_build("openshift-oidc")

    def test_clusterrole_has_only_auth_rules(self, docs: list[dict]) -> None:
        cr = _find_resource(docs, "ClusterRole", "rhoai-mcp")
        assert cr is not None, "ClusterRole 'rhoai-mcp' not found in rendered output"

        # Exact allow-list (4 total): only impersonation + token/access review rules
        expected = {
            "": (
                {"users", "groups", "serviceaccounts"},
                {"impersonate"},
            ),
            "authentication.k8s.io": (
                {"tokenreviews"},
                {"create"},
            ),
            "authorization.k8s.io": (
                {"subjectaccessreviews"},
                {"create"},
            ),
            "user.openshift.io": (
                {"users"},
                {"get"},
            ),
        }

        rules = cr.get("rules", [])
        assert len(rules) == len(expected)
        assert all(len(r["apiGroups"]) == 1 for r in rules), "multi-group rules not expected"

        actual = {r["apiGroups"][0]: (set(r["resources"]), set(r["verbs"])) for r in rules}
        assert actual == expected

    def test_configmap_has_oidc_enabled(self, docs: list[dict]) -> None:
        cm = _find_resource(docs, "ConfigMap", "rhoai-mcp-config")
        assert cm is not None, "ConfigMap 'rhoai-mcp-config' not found in rendered output"

        data = cm.get("data", {})
        assert data.get("RHOAI_MCP_OIDC_ENABLED") == "true"
        assert data.get("RHOAI_MCP_OIDC_TOKEN_MODE") == "token-review"
