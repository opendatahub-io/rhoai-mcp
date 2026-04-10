"""Client for Quickstart operations.

Handles repository cloning, README extraction, and deployment detection/execution.
"""

from __future__ import annotations

import os
import shlex
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
        # ── Red Hat AI BU quickstarts ──────────────────────────────────────
        "llm-cpu-serving": Quickstart(
            name="llm-cpu-serving",
            display_name="LLM CPU Serving",
            description=(
                "Deploy and serve a small language model on CPU infrastructure using "
                "vLLM inference runtime. Ideal for environments without GPU resources."
            ),
            repo_url="git@github.com:rh-ai-quickstart/llm-cpu-serving.git",
            tags=["llm", "inference", "cpu", "serving", "vllm"],
        ),
        "rag-chatbot": Quickstart(
            name="rag-chatbot",
            display_name="RAG Chatbot",
            description=(
                "Experiment with retrieval-augmented generation (RAG) in a streamlined "
                "chat environment. Combines vector search with LLM for context-aware responses."
            ),
            repo_url="git@github.com:rh-ai-quickstart/RAG.git",
            tags=["rag", "chatbot", "vector-db", "llm"],
        ),
        "product-recommender": Quickstart(
            name="product-recommender",
            display_name="Product Recommender System",
            description=(
                "ML-powered product recommendation system for interactions between users "
                "and products in an online store using collaborative filtering and embeddings."
            ),
            repo_url="git@github.com:rh-ai-quickstart/product-recommender-system.git",
            tags=["ml", "recommendations", "embeddings"],
        ),
        "lemonade-stand": Quickstart(
            name="lemonade-stand",
            display_name="Lemonade Stand Assistant",
            description=(
                "AI-powered customer service assistant with guardrails for safe, compliant "
                "interactions using an LLM and multiple detector models."
            ),
            repo_url="git@github.com:rh-ai-quickstart/lemonade-stand-assistant.git",
            tags=["agents", "guardrails", "safety", "demo"],
        ),
        "vllm-tool-calling": Quickstart(
            name="vllm-tool-calling",
            display_name="vLLM Tool Calling",
            description=(
                "Deploy an LLM with tool calling enabled on top of OpenShift AI. "
                "Demonstrates function calling and tool-use patterns with vLLM."
            ),
            repo_url="git@github.com:rh-ai-quickstart/vllm-tool-calling.git",
            tags=["llm", "tool-calling", "vllm", "inference"],
        ),
        "ai-virtual-agent": Quickstart(
            name="ai-virtual-agent",
            display_name="AI Virtual Agent",
            description=(
                "Deploy AI agents with knowledge bases and tools on OpenShift. "
                "Build virtual agents that can answer questions and perform actions."
            ),
            repo_url="git@github.com:rh-ai-quickstart/ai-virtual-agent.git",
            tags=["agents", "knowledge-base", "tools"],
        ),
        "ai-observability-summarizer": Quickstart(
            name="ai-observability-summarizer",
            display_name="AI Observability Summarizer",
            description=(
                "Interactive dashboard to analyze AI model performance and OpenShift "
                "metrics collected from Prometheus with AI-powered summarization."
            ),
            repo_url="git@github.com:rh-ai-quickstart/ai-observability-summarizer.git",
            tags=["observability", "monitoring", "prometheus", "dashboard"],
        ),
        "custom-workbench-images": Quickstart(
            name="custom-workbench-images",
            display_name="Custom Workbench Images",
            description=(
                "Quickly add useful community-provided custom workbench images "
                "to your OpenShift AI environment."
            ),
            repo_url="git@github.com:rh-ai-quickstart/custom-workbench-images-examples.git",
            tags=["workbench", "images", "customization"],
        ),
        "dynamic-model-router": Quickstart(
            name="dynamic-model-router",
            display_name="Dynamic Model Router",
            description=(
                "Dynamically route user prompts to LoRA adapters or a base LLM using "
                "semantic evaluation on Red Hat OpenShift AI with LiteLLM and vLLM."
            ),
            repo_url="git@github.com:rh-ai-quickstart/dynamic-model-router.git",
            tags=["llm", "routing", "lora", "litellm", "vllm"],
        ),
        "rhoai-metrics-dashboard": Quickstart(
            name="rhoai-metrics-dashboard",
            display_name="RHOAI Metrics Dashboard",
            description=(
                "Metrics dashboard for monitoring single serving models on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/rhoai-metrics-dashboard.git",
            tags=["monitoring", "metrics", "dashboard", "serving"],
        ),
        "llama-stack-mcp-server": Quickstart(
            name="llama-stack-mcp-server",
            display_name="Llama Stack MCP Server",
            description=(
                "Deploy Llama 3.2-3B on vLLM with Llama Stack and MCP servers "
                "in OpenShift AI for tool-augmented LLM workflows."
            ),
            repo_url="git@github.com:rh-ai-quickstart/llama-stack-mcp-server.git",
            tags=["llama-stack", "mcp", "vllm", "agents"],
        ),
        "llama-stack-observability": Quickstart(
            name="llama-stack-observability",
            display_name="Llama Stack Observability",
            description=(
                "Observability quickstart for Llama Stack deployments. "
                "Monitor and trace Llama Stack inference and agent workflows."
            ),
            repo_url="git@github.com:rh-ai-quickstart/lls-observability.git",
            tags=["llama-stack", "observability", "monitoring"],
        ),
        "llama-stack-react": Quickstart(
            name="llama-stack-react",
            display_name="Llama Stack ReAct Agent",
            description=(
                "Build a ReAct (Reasoning + Acting) agent using Llama Stack "
                "on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/llama-stack-ReAct.git",
            tags=["llama-stack", "agents", "react", "reasoning"],
        ),
        "byo-agentic-framework": Quickstart(
            name="byo-agentic-framework",
            display_name="BYO Agentic Framework",
            description=(
                "Bring your own or multi-agent framework into Red Hat AI with "
                "Llama Stack. Integrate custom agentic workflows."
            ),
            repo_url="git@github.com:rh-ai-quickstart/byo-agentic-framework.git",
            tags=["agents", "llama-stack", "multi-agent", "framework"],
        ),
        "ansible-log-analysis": Quickstart(
            name="ansible-log-analysis",
            display_name="Ansible Log Analysis Agent",
            description=(
                "AI agent for AAP clusters that detects Ansible log errors, suggests "
                "step-by-step fixes using cluster-wide logs, and routes issues to experts."
            ),
            repo_url="git@github.com:rh-ai-quickstart/ansible-log-analysis.git",
            tags=["agents", "ansible", "log-analysis", "aap"],
        ),
        "guardrailing-llms": Quickstart(
            name="guardrailing-llms",
            display_name="Guardrailing LLMs",
            description=(
                "Apply guardrails to LLM deployments on OpenShift AI. "
                "Ensure safe and compliant AI interactions with content filtering."
            ),
            repo_url="git@github.com:rh-ai-quickstart/guardrailing-llms.git",
            tags=["guardrails", "safety", "llm", "content-moderation"],
        ),
        "speech-to-text-whisper": Quickstart(
            name="speech-to-text-whisper",
            display_name="Speech to Text with Whisper",
            description=(
                "Set up OpenAI Whisper on Red Hat OpenShift AI to enable "
                "seamless speech-to-text transcription."
            ),
            repo_url="git@github.com:rh-ai-quickstart/basic-speech-to-text-with-whisper.git",
            tags=["speech-to-text", "whisper", "audio", "transcription"],
        ),
        "multi-skills-llm": Quickstart(
            name="multi-skills-llm",
            display_name="Multi-Skills LLM",
            description=(
                "A multi-skill customer support assistant for product questions, "
                "billing, shipping, and technical support."
            ),
            repo_url="git@github.com:rh-ai-quickstart/multi-skills-llm.git",
            tags=["llm", "customer-support", "multi-skill"],
        ),
        "spending-transaction-monitor": Quickstart(
            name="spending-transaction-monitor",
            display_name="Spending Transaction Monitor",
            description=(
                "AI-powered spending and transaction monitoring application "
                "on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/spending-transaction-monitor.git",
            tags=["finance", "monitoring", "transactions"],
        ),
        "it-self-service-agent": Quickstart(
            name="it-self-service-agent",
            display_name="IT Self-Service Agent",
            description=(
                "AI-powered IT self-service agent for automating common IT support "
                "tasks and ticket resolution on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/it-self-service-agent.git",
            tags=["agents", "it-support", "self-service", "automation"],
        ),
        # ── Partner and community quickstarts ──────────────────────────────
        "data-governance-co-pilot": Quickstart(
            name="data-governance-co-pilot",
            display_name="Data Governance Co-Pilot",
            description=(
                "Safe data discovery co-pilot for data governance. "
                "AI-assisted data classification and compliance on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/data-governance-co-pilot.git",
            tags=["data-governance", "compliance", "discovery"],
        ),
        "fraud-detection-lakefs": Quickstart(
            name="fraud-detection-lakefs",
            display_name="Fraud Detection with lakeFS",
            description=(
                "Demonstrates how lakeFS can be used for data versioning "
                "in a fraud detection application on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/Fraud-Detection-data-versioning-with-lakeFS.git",
            tags=["fraud-detection", "data-versioning", "lakefs", "ml"],
        ),
        "billing-extraction-groundx": Quickstart(
            name="billing-extraction-groundx",
            display_name="Billing Extraction with GroundX",
            description=(
                "Leverage GroundX to extract billing information from "
                "billing statements using AI on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/Billing-extraction-with-GroundX.git",
            tags=["document-extraction", "billing", "groundx"],
        ),
        "ppe-compliance-monitor": Quickstart(
            name="ppe-compliance-monitor",
            display_name="PPE Compliance Monitor",
            description=(
                "PPE compliance monitoring app that analyzes live video with a trained "
                "model and reports safety violations via a web UI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/ppe-compliance-monitor.git",
            tags=["computer-vision", "safety", "compliance", "video"],
        ),
        "lease-management-codvo": Quickstart(
            name="lease-management-codvo",
            display_name="Agentic Lease Management (Codvo)",
            description=(
                "AI-powered intelligence for lease management and compliance "
                "reconciliation with Codvo on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/Agentic-Lease-Management-and-Reconciliation-with-Codvo.git",
            tags=["agents", "lease-management", "compliance", "codvo"],
        ),
        "confidential-ai-inference": Quickstart(
            name="confidential-ai-inference",
            display_name="Confidential AI Inference",
            description=(
                "Protect your models and sensitive data from unauthorized access "
                "with confidential AI inference on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/confidential-ai-inference.git",
            tags=["security", "confidential-computing", "inference"],
        ),
        "secure-tool-planner": Quickstart(
            name="secure-tool-planner",
            display_name="Secure Tool Planner",
            description=(
                "Deploy a secure planner agentic app with secured MCP tools on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/secure-tool-planner.git",
            tags=["agents", "mcp", "security", "planner"],
        ),
        "first-line-support-alquimia": Quickstart(
            name="first-line-support-alquimia",
            display_name="First Line Support (Alquimia)",
            description=(
                "Red Hat AI partner quickstart with Alquimia Runtime to deliver "
                "operational value by integrating with legacy systems."
            ),
            repo_url="git@github.com:rh-ai-quickstart/First-line-support-with-Alquimia.git",
            tags=["customer-support", "alquimia", "integration"],
        ),
        "smart-telemetry-pipeline": Quickstart(
            name="smart-telemetry-pipeline",
            display_name="Smart Telemetry Pipeline",
            description=(
                "Intelligent observability pipeline that detects microservice errors, "
                "correlates logs and traces, and uses GenAI for actionable remediation."
            ),
            repo_url="git@github.com:rh-ai-quickstart/smart-telemetry-pipeline.git",
            tags=["observability", "telemetry", "opentelemetry", "sre"],
        ),
        "f5-api-security": Quickstart(
            name="f5-api-security",
            display_name="F5 API Security",
            description=(
                "AI-powered API security quickstart with F5 integration on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/f5-api-security.git",
            tags=["security", "api", "f5"],
        ),
        "f5-ai-guardrails": Quickstart(
            name="f5-ai-guardrails",
            display_name="F5 AI Guardrails",
            description=(
                "AI guardrails integration with F5 for securing AI applications "
                "on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/f5-ai-guardrails.git",
            tags=["guardrails", "security", "f5"],
        ),
        "elastic-3am-killer": Quickstart(
            name="elastic-3am-killer",
            display_name="Elastic 3AM Alert Killer",
            description=(
                "AI-powered alert triage and remediation with Elastic integration "
                "to reduce on-call alert fatigue on OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/elastic-3am-killer.git",
            tags=["observability", "alerting", "elastic", "sre"],
        ),
        "multi-agent-loan-origination": Quickstart(
            name="multi-agent-loan-origination",
            display_name="Multi-Agent Loan Origination",
            description=(
                "Multi-agent system for loan origination workflows on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/multi-agent-loan-origination.git",
            tags=["agents", "multi-agent", "finance", "loans"],
        ),
        "aml-rag-nvidia": Quickstart(
            name="aml-rag-nvidia",
            display_name="AML RAG with NVIDIA",
            description=(
                "Anti-money laundering RAG application with NVIDIA integration "
                "on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/aml-rag-nvidia.git",
            tags=["rag", "aml", "nvidia", "compliance"],
        ),
        "ai-supply-chain-agent": Quickstart(
            name="ai-supply-chain-agent",
            display_name="AI Supply Chain Agent",
            description=("AI-powered supply chain management agent on Red Hat OpenShift AI."),
            repo_url="git@github.com:rh-ai-quickstart/ai-supply-chain-agent.git",
            tags=["agents", "supply-chain", "automation"],
        ),
        "dagshub-ai-dev-platform": Quickstart(
            name="dagshub-ai-dev-platform",
            display_name="DagsHub AI Dev Platform",
            description=(
                "DagsHub integration for AI development platform support on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/dagshub-ai-dev-plaform-support.git",
            tags=["mlops", "dagshub", "experiment-tracking"],
        ),
        "maas-code-assistant": Quickstart(
            name="maas-code-assistant",
            display_name="MaaS Code Assistant",
            description=("Model-as-a-Service code assistant quickstart on Red Hat OpenShift AI."),
            repo_url="git@github.com:rh-ai-quickstart/maas-code-assistant.git",
            tags=["code-assistant", "maas", "llm"],
        ),
        "redis-ai-cost-reduction": Quickstart(
            name="redis-ai-cost-reduction",
            display_name="Reduce AI Costs with Redis",
            description=(
                "Reduce AI inference costs with Redis Labs using prompt caching "
                "and LLM routing on Red Hat OpenShift AI."
            ),
            repo_url="git@github.com:rh-ai-quickstart/Reducing-costs-of-AI-with-Redis-Labs.git",
            tags=["cost-optimization", "caching", "redis", "llm-routing"],
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

        try:
            repo_path = self._clone_repo(quickstart)
        except (RuntimeError, OSError) as e:
            raise ValueError(f"Failed to clone repository {quickstart.repo_url}: {e}") from e

        readme_path = repo_path / "README.md"
        if not readme_path.exists():
            readme_path = repo_path / "readme.md"

        if not readme_path.exists():
            raise ValueError(f"No README.md found in repository: {quickstart.repo_url}")

        try:
            content = readme_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to read README from {quickstart.repo_url}: {e}") from e

        return QuickstartReadme(
            quickstart_name=quickstart_name,
            content=content,
            repo_url=quickstart.repo_url,
        )

    def detect_deployment_method(self, repo_path: Path) -> DeploymentMethod:
        """Detect the deployment method for a repository.

        Priority: Helm > Kustomize > Manifests > Unknown.

        Args:
            repo_path: Path to the cloned repository.

        Returns:
            Detected deployment method.
        """
        # Check for Helm at root
        if (repo_path / "Chart.yaml").exists():
            return DeploymentMethod.HELM
        if (repo_path / "helm").is_dir():
            return DeploymentMethod.HELM
        if (repo_path / "charts").is_dir():
            return DeploymentMethod.HELM

        # Check for Kustomize at root
        if (repo_path / "kustomization.yaml").exists():
            return DeploymentMethod.KUSTOMIZE
        if (repo_path / "kustomization.yml").exists():
            return DeploymentMethod.KUSTOMIZE

        # Check manifest directories, preferring kustomize if present inside
        found_manifest_dir = False
        for manifest_dir in ["deploy", "manifests", "k8s", "kubernetes"]:
            dir_path = repo_path / manifest_dir
            if dir_path.is_dir():
                if (dir_path / "kustomization.yaml").exists() or (
                    dir_path / "kustomization.yml"
                ).exists():
                    return DeploymentMethod.KUSTOMIZE
                found_manifest_dir = True

        if found_manifest_dir:
            return DeploymentMethod.MANIFESTS

        # Check root for K8s YAML files
        yaml_files = list(repo_path.glob("*.yaml")) + list(repo_path.glob("*.yml"))
        k8s_yamls = [f for f in yaml_files if self._is_k8s_manifest(f)]
        if k8s_yamls:
            return DeploymentMethod.MANIFESTS

        return DeploymentMethod.UNKNOWN

    def _is_k8s_manifest(self, file_path: Path) -> bool:
        """Check if a YAML file appears to be a Kubernetes manifest."""
        try:
            content = file_path.read_text(encoding="utf-8")
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

        # Build deployment command as argv list
        argv = self._build_deploy_argv(repo_path, method, quickstart_name, namespace)
        command_display = shlex.join(argv) if argv else ""

        if method == DeploymentMethod.UNKNOWN:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command_display,
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
                command=command_display,
                dry_run=True,
                success=True,
                stdout=f"Dry run: would execute the following command:\n{command_display}",
            )

        # Execute the deployment
        return self._execute_deployment(
            argv=argv,
            command_display=command_display,
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

        # Shallow clone with SSH prompts disabled to avoid terminal hangs
        env = {
            **os.environ,
            "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
            "GIT_TERMINAL_PROMPT": "0",
        }
        result = subprocess.run(
            ["git", "clone", "--depth=1", quickstart.repo_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

        return repo_path

    def _build_deploy_argv(
        self,
        repo_path: Path,
        method: DeploymentMethod,
        quickstart_name: str,
        namespace: str,
    ) -> list[str]:
        """Build the deployment command as an argv list.

        Args:
            repo_path: Path to the cloned repository.
            method: Detected deployment method.
            quickstart_name: Name of the quickstart.
            namespace: Target namespace.

        Returns:
            Command as a list of arguments for subprocess.run.
        """
        # Use oc (pre-installed in Containerfile)
        kubectl_cmd = "oc"

        if method == DeploymentMethod.HELM:
            chart_path = str(repo_path)
            if (repo_path / "helm").is_dir():
                chart_path = str(repo_path / "helm")
            elif (repo_path / "charts").is_dir():
                charts = list((repo_path / "charts").iterdir())
                if charts:
                    chart_path = str(charts[0])

            return [
                "helm",
                "upgrade",
                "--install",
                quickstart_name,
                chart_path,
                "--namespace",
                namespace,
                "--create-namespace",
            ]

        elif method == DeploymentMethod.KUSTOMIZE:
            # Check manifest directories for kustomization files
            kustomize_path = str(repo_path)
            for manifest_dir in ["deploy", "manifests", "k8s", "kubernetes"]:
                dir_path = repo_path / manifest_dir
                if dir_path.is_dir() and (
                    (dir_path / "kustomization.yaml").exists()
                    or (dir_path / "kustomization.yml").exists()
                ):
                    kustomize_path = str(dir_path)
                    break
            return [kubectl_cmd, "apply", "-k", kustomize_path, "-n", namespace]

        elif method == DeploymentMethod.MANIFESTS:
            # Find manifest directory
            for manifest_dir in ["deploy", "manifests", "k8s", "kubernetes"]:
                if (repo_path / manifest_dir).is_dir():
                    return [
                        kubectl_cmd,
                        "apply",
                        "-f",
                        str(repo_path / manifest_dir),
                        "-n",
                        namespace,
                    ]

            # Fall back to individual K8s YAML files in root
            yaml_files = list(repo_path.glob("*.yaml")) + list(repo_path.glob("*.yml"))
            k8s_yamls = [str(f) for f in yaml_files if self._is_k8s_manifest(f)]
            if k8s_yamls:
                args = [kubectl_cmd, "apply", "-n", namespace]
                for f in k8s_yamls:
                    args.extend(["-f", f])
                return args

            return [kubectl_cmd, "apply", "-f", str(repo_path), "-n", namespace]

        return []

    def _execute_deployment(
        self,
        argv: list[str],
        command_display: str,
        quickstart_name: str,
        namespace: str,
        method: DeploymentMethod,
        repo_path: Path,
    ) -> DeploymentResult:
        """Execute a deployment command.

        Args:
            argv: Command as argument list for subprocess.
            command_display: Human-readable command string for display.
            quickstart_name: Name of the quickstart.
            namespace: Target namespace.
            method: Deployment method.
            repo_path: Path to the cloned repository.

        Returns:
            DeploymentResult with execution results.
        """
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(repo_path),
                stdin=subprocess.DEVNULL,
            )

            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command_display,
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
                command=command_display,
                dry_run=False,
                success=False,
                error="Deployment timed out after 300 seconds",
            )
        except Exception as e:
            return DeploymentResult(
                quickstart_name=quickstart_name,
                namespace=namespace,
                method=method,
                command=command_display,
                dry_run=False,
                success=False,
                error=f"Deployment failed: {e}",
            )
