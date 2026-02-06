"""Tests for quickstarts models."""

import pytest

from rhoai_mcp.domains.quickstarts.models import (
    DeploymentMethod,
    DeploymentResult,
    Quickstart,
    QuickstartReadme,
)


class TestQuickstartModel:
    """Tests for the Quickstart model."""

    def test_quickstart_creation(self) -> None:
        """Test creating a Quickstart with all fields."""
        qs = Quickstart(
            name="test-quickstart",
            display_name="Test Quickstart",
            description="A test quickstart",
            repo_url="https://github.com/test/repo",
            tags=["test", "demo"],
        )

        assert qs.name == "test-quickstart"
        assert qs.display_name == "Test Quickstart"
        assert qs.description == "A test quickstart"
        assert qs.repo_url == "https://github.com/test/repo"
        assert qs.tags == ["test", "demo"]

    def test_quickstart_default_tags(self) -> None:
        """Test that tags default to empty list."""
        qs = Quickstart(
            name="minimal",
            display_name="Minimal",
            description="Minimal quickstart",
            repo_url="https://github.com/test/minimal",
        )

        assert qs.tags == []


class TestQuickstartReadme:
    """Tests for the QuickstartReadme model."""

    def test_readme_creation(self) -> None:
        """Test creating a QuickstartReadme."""
        readme = QuickstartReadme(
            quickstart_name="test-qs",
            content="# Test README\n\nThis is a test.",
            repo_url="https://github.com/test/repo",
        )

        assert readme.quickstart_name == "test-qs"
        assert "# Test README" in readme.content
        assert readme.repo_url == "https://github.com/test/repo"


class TestDeploymentMethod:
    """Tests for the DeploymentMethod enum."""

    def test_deployment_methods(self) -> None:
        """Test all deployment method values."""
        assert DeploymentMethod.HELM.value == "helm"
        assert DeploymentMethod.KUSTOMIZE.value == "kustomize"
        assert DeploymentMethod.MANIFESTS.value == "manifests"
        assert DeploymentMethod.UNKNOWN.value == "unknown"


class TestDeploymentResult:
    """Tests for the DeploymentResult model."""

    def test_deployment_result_success(self) -> None:
        """Test successful deployment result."""
        result = DeploymentResult(
            quickstart_name="test-qs",
            namespace="test-ns",
            method=DeploymentMethod.HELM,
            command="helm install test .",
            dry_run=False,
            success=True,
            stdout="Deployed successfully",
        )

        assert result.quickstart_name == "test-qs"
        assert result.namespace == "test-ns"
        assert result.method == DeploymentMethod.HELM
        assert result.success is True
        assert result.error is None

    def test_deployment_result_failure(self) -> None:
        """Test failed deployment result."""
        result = DeploymentResult(
            quickstart_name="test-qs",
            namespace="test-ns",
            method=DeploymentMethod.KUSTOMIZE,
            command="oc apply -k .",
            dry_run=False,
            success=False,
            error="Namespace not found",
        )

        assert result.success is False
        assert result.error == "Namespace not found"

    def test_deployment_result_dry_run(self) -> None:
        """Test dry run deployment result."""
        result = DeploymentResult(
            quickstart_name="test-qs",
            namespace="default",
            method=DeploymentMethod.MANIFESTS,
            command="oc apply -f deploy/",
            dry_run=True,
            success=True,
        )

        assert result.dry_run is True
        assert result.success is True
