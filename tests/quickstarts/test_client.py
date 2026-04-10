"""Tests for quickstarts client."""

from pathlib import Path

import pytest

from rhoai_mcp.domains.quickstarts.client import QuickstartClient, QuickstartRegistry
from rhoai_mcp.domains.quickstarts.models import DeploymentMethod


class TestQuickstartRegistry:
    """Tests for the QuickstartRegistry."""

    def test_list_all_returns_quickstarts(self) -> None:
        """Test listing all quickstarts."""
        quickstarts = QuickstartRegistry.list_all()

        assert len(quickstarts) >= 4
        names = [qs.name for qs in quickstarts]
        assert "llm-cpu-serving" in names
        assert "rag-chatbot" in names
        assert "product-recommender" in names
        assert "lemonade-stand" in names

    def test_get_existing_quickstart(self) -> None:
        """Test getting an existing quickstart by name."""
        qs = QuickstartRegistry.get("llm-cpu-serving")

        assert qs is not None
        assert qs.name == "llm-cpu-serving"
        assert "github.com" in qs.repo_url

    def test_get_nonexistent_quickstart(self) -> None:
        """Test getting a non-existent quickstart."""
        qs = QuickstartRegistry.get("nonexistent")

        assert qs is None

    def test_exists_true(self) -> None:
        """Test exists returns True for registered quickstart."""
        assert QuickstartRegistry.exists("rag-chatbot") is True

    def test_exists_false(self) -> None:
        """Test exists returns False for unregistered quickstart."""
        assert QuickstartRegistry.exists("unknown-qs") is False


class TestQuickstartClient:
    """Tests for the QuickstartClient."""

    def test_list_quickstarts(self) -> None:
        """Test listing quickstarts through client."""
        client = QuickstartClient()
        quickstarts = client.list_quickstarts()

        assert len(quickstarts) >= 4
        assert all(qs.name for qs in quickstarts)
        assert all(qs.repo_url for qs in quickstarts)

    def test_detect_helm_chart_yaml(self, tmp_path: Path) -> None:
        """Test detecting Helm deployment from Chart.yaml."""
        (tmp_path / "Chart.yaml").touch()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.HELM

    def test_detect_helm_directory(self, tmp_path: Path) -> None:
        """Test detecting Helm deployment from helm/ directory."""
        (tmp_path / "helm").mkdir()
        (tmp_path / "helm" / "Chart.yaml").touch()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.HELM

    def test_detect_kustomize_yaml(self, tmp_path: Path) -> None:
        """Test detecting Kustomize deployment."""
        (tmp_path / "kustomization.yaml").touch()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.KUSTOMIZE

    def test_detect_kustomize_yml(self, tmp_path: Path) -> None:
        """Test detecting Kustomize deployment with .yml extension."""
        (tmp_path / "kustomization.yml").touch()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.KUSTOMIZE

    def test_detect_kustomize_inside_manifest_dir(self, tmp_path: Path) -> None:
        """Test detecting Kustomize when kustomization.yaml is inside deploy/."""
        (tmp_path / "deploy").mkdir()
        (tmp_path / "deploy" / "kustomization.yaml").touch()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.KUSTOMIZE

    def test_detect_manifests_deploy_dir(self, tmp_path: Path) -> None:
        """Test detecting manifests from deploy/ directory."""
        (tmp_path / "deploy").mkdir()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.MANIFESTS

    def test_detect_manifests_kubernetes_dir(self, tmp_path: Path) -> None:
        """Test detecting manifests from kubernetes/ directory."""
        (tmp_path / "kubernetes").mkdir()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.MANIFESTS

    def test_detect_unknown(self, tmp_path: Path) -> None:
        """Test returning unknown for unrecognized structure."""
        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.UNKNOWN

    def test_deploy_unknown_quickstart(self) -> None:
        """Test deploying an unknown quickstart returns error."""
        client = QuickstartClient()
        result = client.deploy("nonexistent-quickstart")

        assert result.success is False
        assert "Unknown quickstart" in (result.error or "")
        assert "Available:" in (result.error or "")

    def test_get_readme_unknown_quickstart(self) -> None:
        """Test getting README for unknown quickstart."""
        client = QuickstartClient()

        with pytest.raises(ValueError, match="Unknown quickstart"):
            client.get_readme("nonexistent")


class TestDeploymentMethodPriority:
    """Test that deployment method detection has correct priority."""

    def test_helm_takes_priority_over_kustomize(self, tmp_path: Path) -> None:
        """Test Helm is detected over Kustomize when both present."""
        (tmp_path / "Chart.yaml").touch()
        (tmp_path / "kustomization.yaml").touch()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.HELM

    def test_kustomize_takes_priority_over_manifests(self, tmp_path: Path) -> None:
        """Test Kustomize is detected over manifests when both present."""
        (tmp_path / "kustomization.yaml").touch()
        (tmp_path / "deploy").mkdir()

        client = QuickstartClient()
        method = client.detect_deployment_method(tmp_path)

        assert method == DeploymentMethod.KUSTOMIZE
