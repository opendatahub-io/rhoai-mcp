"""Deployment planning — bridge from planner recommendations to rhoai-mcp deployments.

Maps a planner ModelRecommendation to the concrete parameters needed by
the inference domain's deploy_model tool, resolving serving runtime,
storage URI, and resource requirements against the live cluster.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from rhoai_mcp.composites.planner.models import (
    DeploymentPlan,
    DeploymentPlanIssue,
    DeploymentPlanStep,
    ResolvedDeployParams,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

GPU_RESOURCE_PROFILES: dict[str, tuple[str, str, str, str]] = {
    "H100": ("24", "48", "128Gi", "256Gi"),
    "H200": ("24", "48", "128Gi", "256Gi"),
    "B200": ("24", "48", "128Gi", "256Gi"),
    "A100-80": ("24", "48", "128Gi", "256Gi"),
    "A100-40": ("16", "32", "64Gi", "128Gi"),
    "L4": ("8", "16", "32Gi", "64Gi"),
}
_DEFAULT_RESOURCE_PROFILE = ("8", "16", "32Gi", "64Gi")

_VLLM_RUNTIME_PATTERNS = ["vllm", "tgis", "text-gen"]


def generate_deployment_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a DNS-1123 compatible name."""
    name = model_id.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name[:50]


async def plan_deployment(
    server: RHOAIServer,
    recommendation: dict[str, Any],
    namespace: str,
    name_override: str | None = None,
    storage_uri_override: str | None = None,
    runtime_override: str | None = None,
) -> DeploymentPlan:
    """Build a deployment plan from a planner recommendation.

    Steps:
    1. Parse the recommendation into deployment parameters
    2. Resolve the serving runtime (find compatible runtime on cluster)
    3. Resolve the storage URI (Model Catalog lookup if not provided)
    4. Compute resource requests from GPU config
    5. Run pre-flight checks (GPU availability, namespace exists)
    6. Return a DeploymentPlan with all resolved parameters and any issues
    """
    issues: list[DeploymentPlanIssue] = []
    warnings: list[str] = []

    # --- Step 1: Parse recommendation ---
    model_id = recommendation.get("model_id") or recommendation.get("model", "")
    model_name = recommendation.get("model_name") or recommendation.get("model") or model_id
    model_uri = recommendation.get("model_uri")
    gpu_config = recommendation.get("gpu_config", {})
    gpu_type = gpu_config.get("gpu_type", "L4") if isinstance(gpu_config, dict) else "L4"
    gpu_count = gpu_config.get("gpu_count", 1) if isinstance(gpu_config, dict) else 1
    tensor_parallel = gpu_config.get("tensor_parallel", 1) if isinstance(gpu_config, dict) else 1
    replicas = gpu_config.get("replicas", 1) if isinstance(gpu_config, dict) else 1
    deploy_name = name_override or generate_deployment_name(model_id)

    # --- Step 2: Resolve serving runtime ---
    runtime, runtime_issues = await _resolve_runtime(server, namespace, runtime_override)
    issues.extend(runtime_issues)
    requires_runtime_creation = runtime is None and not runtime_override

    # --- Step 3: Resolve storage URI ---
    storage_uri, storage_issues = await _resolve_storage_uri(
        server, model_id, model_uri, storage_uri_override
    )
    issues.extend(storage_issues)

    # --- Step 4: Compute resources from GPU config ---
    cpu_req, cpu_lim, mem_req, mem_lim = GPU_RESOURCE_PROFILES.get(
        gpu_type, _DEFAULT_RESOURCE_PROFILE
    )

    # --- Step 5: Pre-flight checks ---
    preflight_issues, preflight_warnings = await _run_preflight_checks(
        server, namespace, gpu_type, gpu_count * replicas, tensor_parallel
    )
    issues.extend(preflight_issues)
    warnings.extend(preflight_warnings)

    # --- Build resolved parameters ---
    effective_runtime = runtime_override or runtime or "vllm-cuda-runtime"
    params = ResolvedDeployParams(
        name=deploy_name,
        namespace=namespace,
        display_name=model_name if model_name != model_id else None,
        runtime=effective_runtime,
        model_format="pytorch",
        storage_uri=storage_uri or f"oci://{model_id}",
        min_replicas=replicas,
        max_replicas=max(replicas, replicas * 2),
        cpu_request=cpu_req,
        cpu_limit=cpu_lim,
        memory_request=mem_req,
        memory_limit=mem_lim,
        gpu_count=gpu_count,
    )

    # --- Build execution steps ---
    blocking_issues = [i for i in issues if i.blocking]
    if blocking_issues:
        steps = [
            DeploymentPlanStep(
                action="fix_issues",
                description="Resolve blocking issues before deployment",
            )
        ]
    else:
        steps = _build_execution_steps(
            deploy_name, namespace, effective_runtime, requires_runtime_creation
        )

    rec_summary: dict[str, Any] = {
        "model_id": model_id,
        "model_name": model_name,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "tensor_parallel": tensor_parallel,
        "replicas": replicas,
        "predicted_cost_month": recommendation.get("cost_per_month_usd")
        or recommendation.get("cost_usd_month"),
        "meets_slo": recommendation.get("meets_slo", False),
    }

    return DeploymentPlan(
        ready=len(blocking_issues) == 0,
        recommendation_summary=rec_summary,
        resolved_params=params,
        steps=steps,
        issues=issues,
        warnings=warnings,
    )


def _build_execution_steps(
    name: str,
    namespace: str,
    runtime: str,
    requires_runtime_creation: bool,
) -> list[DeploymentPlanStep]:
    """Build the ordered list of execution steps."""
    steps: list[DeploymentPlanStep] = []
    if requires_runtime_creation:
        steps.append(
            DeploymentPlanStep(
                action="create_runtime",
                description=f"Create serving runtime '{runtime}' from platform template",
            )
        )
    else:
        steps.append(
            DeploymentPlanStep(
                action="runtime_ready",
                description=f"Serving runtime '{runtime}' is available",
            )
        )
    steps.extend(
        [
            DeploymentPlanStep(
                action="deploy_model",
                description=f"Create InferenceService '{name}' in {namespace}",
            ),
            DeploymentPlanStep(
                action="wait_ready",
                description="Wait for model to become Ready",
            ),
            DeploymentPlanStep(
                action="test_endpoint",
                description="Validate the inference endpoint responds",
            ),
        ]
    )
    return steps


async def _resolve_runtime(
    server: RHOAIServer,
    namespace: str,
    override: str | None,
) -> tuple[str | None, list[DeploymentPlanIssue]]:
    """Find a vLLM-compatible serving runtime on the cluster."""
    issues: list[DeploymentPlanIssue] = []
    if override:
        return override, issues

    try:
        from rhoai_mcp.domains.inference.client import InferenceClient

        client = InferenceClient(server.k8s)
        runtimes = client.list_serving_runtimes(namespace)

        for pattern in _VLLM_RUNTIME_PATTERNS:
            for rt in runtimes:
                if pattern in rt.get("name", "").lower():
                    return rt["name"], issues

        for rt in runtimes:
            formats = rt.get("supported_formats", [])
            if "pytorch" in formats or "vllm" in formats:
                return rt["name"], issues

        issues.append(
            DeploymentPlanIssue(
                category="runtime",
                message="No vLLM/TGIS serving runtime found in namespace",
                blocking=False,
                suggestion=(
                    "A vLLM runtime will be created from the platform template, "
                    "or specify runtime_override"
                ),
            )
        )
    except Exception as e:
        logger.debug("Failed to list serving runtimes: %s", e)
        issues.append(
            DeploymentPlanIssue(
                category="runtime",
                message=f"Could not list serving runtimes: {e}",
                blocking=False,
                suggestion="Specify runtime_override parameter",
            )
        )
    return None, issues


async def _resolve_storage_uri(
    server: RHOAIServer,
    model_id: str,
    model_uri: str | None,
    override: str | None,
) -> tuple[str | None, list[DeploymentPlanIssue]]:
    """Resolve where the model artifacts live."""
    issues: list[DeploymentPlanIssue] = []
    if override:
        return override, issues
    if model_uri:
        return model_uri, issues

    try:
        from rhoai_mcp.domains.inference.tools import _resolve_catalog_storage_uri

        uri = await _resolve_catalog_storage_uri(server.config, server.k8s, model_id)
        if uri:
            return uri, issues
    except Exception as e:
        logger.debug("Model Catalog lookup failed for '%s': %s", model_id, e)

    issues.append(
        DeploymentPlanIssue(
            category="storage",
            message=f"Could not auto-resolve storage URI for '{model_id}'",
            blocking=True,
            suggestion=(
                "Provide storage_uri pointing to the model artifacts "
                "(oci://registry/model, s3://bucket/path, or pvc://name/path)"
            ),
        )
    )
    return None, issues


async def _run_preflight_checks(
    server: RHOAIServer,
    namespace: str,
    gpu_type: str,
    total_gpus_needed: int,
    tensor_parallel: int,
) -> tuple[list[DeploymentPlanIssue], list[str]]:
    """Check namespace existence and GPU availability."""
    issues: list[DeploymentPlanIssue] = []
    warnings: list[str] = []

    # Check namespace exists
    try:
        from rhoai_mcp.domains.projects.client import ProjectClient

        client = ProjectClient(server.k8s)
        projects = client.list_projects()
        project_names = [p.metadata.name for p in projects]
        if namespace not in project_names:
            issues.append(
                DeploymentPlanIssue(
                    category="namespace",
                    message=f"Namespace '{namespace}' does not exist",
                    blocking=True,
                    suggestion=f"Create the project first with create_data_science_project('{namespace}')",
                )
            )
    except Exception as e:
        logger.debug("Namespace check failed: %s", e)
        warnings.append(f"Could not verify namespace '{namespace}' exists: {e}")

    # Check GPU availability
    try:
        from rhoai_mcp.domains.training.client import TrainingClient

        training_client = TrainingClient(server.k8s)
        resources = training_client.get_cluster_resources()

        if not resources.has_gpus:
            issues.append(
                DeploymentPlanIssue(
                    category="gpu",
                    message="No GPUs detected on the cluster",
                    blocking=False,
                    suggestion="Verify GPU nodes are available and labeled correctly",
                )
            )
        elif resources.gpu_info:
            cluster_products = resources.gpu_info.products
            gpu_type_lower = gpu_type.lower().replace("-", "")
            matched = any(gpu_type_lower in p.lower().replace("-", "") for p in cluster_products)
            if not matched:
                issues.append(
                    DeploymentPlanIssue(
                        category="gpu",
                        message=(
                            f"Recommended GPU type '{gpu_type}' not found on cluster. "
                            f"Available: {', '.join(cluster_products)}"
                        ),
                        blocking=False,
                        suggestion="The deployment may still work if the scheduler can find matching nodes",
                    )
                )
            elif resources.gpu_info.available < total_gpus_needed:
                issues.append(
                    DeploymentPlanIssue(
                        category="gpu",
                        message=(
                            f"Need {total_gpus_needed} GPUs but only "
                            f"{resources.gpu_info.available} available"
                        ),
                        blocking=False,
                        suggestion="Wait for GPU capacity or choose a smaller configuration",
                    )
                )
    except Exception as e:
        logger.debug("GPU availability check failed: %s", e)
        warnings.append(f"Could not verify GPU availability: {e}")

    if tensor_parallel > 1:
        warnings.append(
            f"Tensor parallelism={tensor_parallel} requires NVLink/NVSwitch between GPUs"
        )

    return issues, warnings
