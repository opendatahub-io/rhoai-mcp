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

        rules = cr.get("rules", [])
        assert len(rules) == 4

        api_groups = {r["apiGroups"][0] for r in rules}
        assert api_groups == {
            "",
            "authentication.k8s.io",
            "authorization.k8s.io",
            "user.openshift.io",
        }

    def test_clusterrole_has_impersonate(self, docs: list[dict]) -> None:
        cr = _find_resource(docs, "ClusterRole", "rhoai-mcp")
        assert cr is not None
        impersonate_rules = [r for r in cr["rules"] if "impersonate" in r.get("verbs", [])]
        assert len(impersonate_rules) == 1
        assert set(impersonate_rules[0]["resources"]) == {"users", "groups", "serviceaccounts"}

    def test_clusterrole_has_no_resource_level_rules(self, docs: list[dict]) -> None:
        cr = _find_resource(docs, "ClusterRole", "rhoai-mcp")
        assert cr is not None
        forbidden_resources = {
            "pods",
            "secrets",
            "persistentvolumeclaims",
            "namespaces",
            "notebooks",
            "inferenceservices",
            "trainjobs",
            "datasciencepipelinesapplications",
        }
        for rule in cr["rules"]:
            assert not forbidden_resources.intersection(rule.get("resources", [])), (
                f"ClusterRole contains resource-level rule: {rule}"
            )

    def test_configmap_has_oidc_enabled(self, docs: list[dict]) -> None:
        cm = _find_resource(docs, "ConfigMap", "rhoai-mcp-config")
        assert cm is not None, "ConfigMap 'rhoai-mcp-config' not found in rendered output"

        data = cm.get("data", {})
        assert data.get("RHOAI_MCP_OIDC_ENABLED") == "true"
        assert data.get("RHOAI_MCP_OIDC_TOKEN_MODE") == "token-review"
