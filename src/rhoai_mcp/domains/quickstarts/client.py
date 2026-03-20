"""Client for Quickstart operations.

Handles repository cloning, README extraction, and deployment detection/execution.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from rhoai_mcp.domains.quickstarts.models import (
    DeploymentMethod,
    DeploymentResult,
    Quickstart,
    QuickstartReadme,
)

if TYPE_CHECKING:
    pass


class QuickstartRegistry:
    """Registry of supported Red Hat AI Quickstarts."""

    # Static registry of supported quickstarts
    QUICKSTARTS: dict[str, Quickstart] = {
        "llm-cpu-serving": Quickstart(
            name="llm-cpu-serving",
            display_name="LLM CPU Serving",
            description=(
                "Deploy and serve Large Language Models on CPU infrastructure. "
                "Ideal for environments without GPU resources or for cost-effective inference."
            ),
            repo_url="https://github.com/rh-ai-quickstart/llm-cpu-serving",
            tags=["llm", "inference", "cpu", "serving"],
        ),
        "rag-chatbot": Quickstart(
            name="rag-chatbot",
            display_name="RAG Chatbot",
            description=(
                "Build a Retrieval-Augmented Generation chatbot with document ingestion. "
                "Combines vector search with LLM for context-aware responses."
            ),
            repo_url="https://github.com/rh-ai-quickstart/rag-chatbot",
            tags=["rag", "chatbot", "vector-db", "llm"],
        ),
        "product-recommender": Quickstart(
            name="product-recommender",
            display_name="Product Recommender System",
            description=(
                "ML-powered product recommendation system. "
                "Uses collaborative filtering and embeddings for personalized recommendations."
            ),
            repo_url="https://github.com/rh-ai-quickstart/product-recommender-system",
            tags=["ml", "recommendations", "embeddings"],
        ),
        "lemonade-stand": Quickstart(
            name="lemonade-stand",
            display_name="Lemonade Stand Assistant",
            description=(
                "AI-powered business assistant demo application. "
                "Showcases multi-agent workflows and tool use patterns."
            ),
            repo_url="https://github.com/rh-ai-quickstart/lemonade-stand-assistant",
            tags=["agents", "demo", "tools", "workflow"],
        ),
    }

    @classmethod
    def list_all(cls) -> list[Quickstart]:
        """Return all registered quickstarts."""
        return list(cls.QUICKSTARTS.values())

    @classmethod
    def get(cls, name: str) -> Quickstart | None:
        """Get a quickstart by name."""
        return cls.QUICKSTARTS.get(name)

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a quickstart exists."""
        return name in cls.QUICKSTARTS


class QuickstartClient:
    """Client for quickstart operations."""

    def __init__(self, temp_dir: str = "/tmp/rhoai-quickstarts") -> None:
        """Initialize the quickstart client.

        Args:
            temp_dir: Base directory for cloning repositories.
        """
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    def list_quickstarts(self) -> list[Quickstart]:
        """List all available quickstarts."""
        return QuickstartRegistry.list_all()

    def get_readme(self, quickstart_name: str) -> QuickstartReadme:
        """Fetch and return the README.md content for a quickstart.

        Args:
            quickstart_name: Name of the quickstart.

        Returns:
            QuickstartReadme with the content.

        Raises:
            ValueError: If quickstart not found or README cannot be fetched.
        """
        quickstart = QuickstartRegistry.get(quickstart_name)
        if not quickstart:
            available = ", ".join(QuickstartRegistry.QUICKSTARTS.keys())
            raise ValueError(f"Unknown quickstart: '{quickstart_name}'. Available: {available}")

        # Clone the repository
        repo_path = self._clone_repo(quickstart)

        # Read README.md
        readme_path = repo_path / "README.md"
        if not readme_path.exists():
            # Try lowercase
            readme_path = repo_path / "readme.md"

        if not readme_path.exists():
            raise ValueError(f"No README.md found in repository: {quickstart.repo_url}")

        content = readme_path.read_text(encoding="utf-8")

        return QuickstartReadme(
            quickstart_name=quickstart_name,
            content=content,
            repo_url=quickstart.repo_url,
        )

    def detect_deployment_method(self, repo_path: Path) -> DeploymentMethod:
        """Detect the deployment method for a repository.

        Args:
            repo_path: Path to the cloned repository.

        Returns:
            Detected deployment method.
        """
        # Check for Helm
        if (repo_path / "Chart.yaml").exists():
            return DeploymentMethod.HELM
        if (repo_path / "helm").is_dir():
            return DeploymentMethod.HELM
        if (repo_path / "charts").is_dir():
            return DeploymentMethod.HELM

        # Check for Kustomize
        if (repo_path / "kustomization.yaml").exists():
            return DeploymentMethod.KUSTOMIZE
        if (repo_path / "kustomization.yml").exists():
            return DeploymentMethod.KUSTOMIZE

        # Check for manifest directories
        for manifest_dir in ["deploy", "manifests", "k8s", "kubernetes"]:
            if (repo_path / manifest_dir).is_dir():
                return DeploymentMethod.MANIFESTS

        # Check root for YAML files
        yaml_files = list(repo_path.glob("*.yaml")) + list(repo_path.glob("*.yml"))
        k8s_yamls = [f for f in yaml_files if self._is_k8s_manifest(f)]
        if k8s_yamls:
            return DeploymentMethod.MANIFESTS

        return DeploymentMethod.UNKNOWN

    def _is_k8s_manifest(self, file_path: Path) -> bool:
        """Check if a YAML file appears to be a Kubernetes manifest."""
        try:
            content = file_path.read_text(encoding="utf-8")
            # Simple heuristic: contains apiVersion and kind
            return "apiVersion:" in content and "kind:" in content
        except Exception:
            return False

    def deploy(
        self,
        quickstart_name: str,
        namespace: str = "rhoai-quickstarts",
        dry_run: bool = True,
    ) -> DeploymentResult:
        """Deploy a quickstart to the cluster.

        Args:
            quickstart_name: Name of the quickstart to deploy.
            namespace: Target namespace for deployment.
            dry_run: If True, only return the command without executing.

        Returns:
            DeploymentResult with command and execution results.
        """
        quickstart = QuickstartRegistry.get(quickstart_name)
        if not quickstart:
            available = ", ".join(QuickstartRegistry.QUICKSTARTS.keys())
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=DeploymentMethod.UNKNOWN,
                command="",
                dry_run=dry_run,
                success=False,
                error=f"Unknown quickstart: '{quickstart_name}'. Available: {available}",
            )

        # Check for required CLI tools
        tool_error = self._check_cli_tools()
        if tool_error:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=DeploymentMethod.UNKNOWN,
                command="",
                dry_run=dry_run,
                success=False,
                error=tool_error,
            )

        # Clone the repository
        try:
            repo_path = self._clone_repo(quickstart)
        except Exception as e:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=DeploymentMethod.UNKNOWN,
                command="",
                dry_run=dry_run,
                success=False,
                error=f"Failed to clone repository: {e}",
            )

        # Detect deployment method
        method = self.detect_deployment_method(repo_path)

        # Build deployment command
        command = self._build_deploy_command(repo_path, method, quickstart_name, namespace)

        if method == DeploymentMethod.UNKNOWN:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command,
                dry_run=dry_run,
                success=False,
                error=(
                    "Could not detect deployment method. "
                    "No Helm charts, Kustomization, or manifest directories found."
                ),
            )

        # Dry run - just return the command
        if dry_run:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command,
                dry_run=True,
                success=True,
                stdout=f"Dry run: would execute the following command:\n{command}",
            )

        # Execute the deployment
        return self._execute_deployment(
            command=command,
            quickstart_name=quickstart_name,
            namespace=namespace,
            method=method,
            repo_path=repo_path,
        )

    def _clone_repo(self, quickstart: Quickstart) -> Path:
        """Clone a quickstart repository (shallow clone).

        Args:
            quickstart: Quickstart to clone.

        Returns:
            Path to the cloned repository.

        Raises:
            RuntimeError: If cloning fails.
        """
        repo_path = self._temp_dir / quickstart.name

        # Remove existing clone if present
        if repo_path.exists():
            shutil.rmtree(repo_path)

        # Shallow clone
        result = subprocess.run(
            ["git", "clone", "--depth=1", quickstart.repo_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

        return repo_path

    def _check_cli_tools(self) -> str | None:
        """Check if required CLI tools are available.

        Returns:
            Error message if tools missing, None if all present.
        """
        missing = []

        # Check for oc or kubectl
        oc_available = shutil.which("oc") is not None
        kubectl_available = shutil.which("kubectl") is not None

        if not oc_available and not kubectl_available:
            missing.append("oc or kubectl")

        # Check for git
        if shutil.which("git") is None:
            missing.append("git")

        if missing:
            return f"Required CLI tools not found: {', '.join(missing)}"

        return None

    def _build_deploy_command(
        self,
        repo_path: Path,
        method: DeploymentMethod,
        quickstart_name: str,
        namespace: str,
    ) -> str:
        """Build the deployment command based on detected method.

        Args:
            repo_path: Path to the cloned repository.
            method: Detected deployment method.
            quickstart_name: Name of the quickstart.
            namespace: Target namespace.

        Returns:
            Shell command string.
        """
        # Prefer oc if available, fall back to kubectl
        kubectl_cmd = "oc" if shutil.which("oc") else "kubectl"

        if method == DeploymentMethod.HELM:
            # Find helm chart path
            chart_path = repo_path
            if (repo_path / "helm").is_dir():
                chart_path = repo_path / "helm"
            elif (repo_path / "charts").is_dir():
                # Find first chart in charts directory
                charts = list((repo_path / "charts").iterdir())
                if charts:
                    chart_path = charts[0]

            return (
                f"helm install {quickstart_name} {chart_path} "
                f"--namespace {namespace} --create-namespace"
            )

        elif method == DeploymentMethod.KUSTOMIZE:
            return f"{kubectl_cmd} apply -k {repo_path} -n {namespace}"

        elif method == DeploymentMethod.MANIFESTS:
            # Find manifest directory
            for manifest_dir in ["deploy", "manifests", "k8s", "kubernetes"]:
                if (repo_path / manifest_dir).is_dir():
                    return f"{kubectl_cmd} apply -f {repo_path / manifest_dir} -n {namespace}"

            # Fall back to root directory YAML files
            return f"{kubectl_cmd} apply -f {repo_path} -n {namespace}"

        return ""

    def _execute_deployment(
        self,
        command: str,
        quickstart_name: str,
        namespace: str,
        method: DeploymentMethod,
        repo_path: Path,
    ) -> DeploymentResult:
        """Execute a deployment command.

        Args:
            command: Shell command to execute.
            quickstart_name: Name of the quickstart.
            namespace: Target namespace.
            method: Deployment method.
            repo_path: Path to the cloned repository.

        Returns:
            DeploymentResult with execution results.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(repo_path),
            )

            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command,
                dry_run=False,
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command,
                dry_run=False,
                success=False,
                error="Deployment timed out after 300 seconds",
            )
        except Exception as e:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command,
                dry_run=False,
                success=False,
                error=f"Deployment failed: {e}",
            )
