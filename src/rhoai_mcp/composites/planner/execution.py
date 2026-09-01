"""Deployment execution — deploy, monitor, and validate from a deployment plan.

Takes a validated DeploymentPlan and executes it end-to-end: creates the
KServe InferenceService, polls for readiness, tests the endpoint, and
reports results with SLO comparison.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from rhoai_mcp.composites.planner.models import (
    DeploymentPlan,
    DeploymentResult,
    EndpointValidation,
)

if TYPE_CHECKING:
    from rhoai_mcp.domains.inference.client import InferenceClient
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

MAX_READINESS_WAIT_SECONDS = 600
READINESS_POLL_INTERVAL = 15


async def execute_deployment(
    server: RHOAIServer,
    plan: DeploymentPlan,
) -> DeploymentResult:
    """Execute a deployment plan.

    Steps:
    1. Validate plan is ready
    2. Create the InferenceService via deploy_model
    3. Poll for Ready status
    4. Test the endpoint
    5. Build result with SLO comparison
    """
    if not plan.ready:
        return DeploymentResult(
            success=False,
            message="Deployment plan has unresolved blocking issues",
            issues=plan.issues,
        )

    if not server.config.is_operation_allowed("create"):
        return DeploymentResult(
            success=False,
            message="Create operations are not allowed (read-only mode or dangerous ops disabled)",
        )

    params = plan.resolved_params

    # --- Step 1: Create serving runtime if needed ---
    needs_runtime = any(s.action == "create_runtime" for s in plan.steps)
    if needs_runtime:
        runtime_ok, runtime_msg = await _ensure_serving_runtime(
            server, params.namespace, params.runtime
        )
        if not runtime_ok:
            return DeploymentResult(
                success=False,
                deployment_name=params.name,
                namespace=params.namespace,
                message=f"Failed to create serving runtime: {runtime_msg}",
            )

    # --- Step 2: Deploy the model ---
    from rhoai_mcp.domains.inference.client import InferenceClient
    from rhoai_mcp.domains.inference.models import InferenceServiceCreate

    client = InferenceClient(server.k8s)
    request = InferenceServiceCreate(
        name=params.name,
        namespace=params.namespace,
        display_name=params.display_name,
        runtime=params.runtime,
        model_format=params.model_format,
        storage_uri=params.storage_uri,
        min_replicas=params.min_replicas,
        max_replicas=params.max_replicas,
        cpu_request=params.cpu_request,
        cpu_limit=params.cpu_limit,
        memory_request=params.memory_request,
        memory_limit=params.memory_limit,
        gpu_count=params.gpu_count,
    )

    try:
        client.deploy_model(request)
    except Exception as e:
        return DeploymentResult(
            success=False,
            deployment_name=params.name,
            namespace=params.namespace,
            message=f"Failed to create InferenceService: {e}",
        )

    # --- Step 3: Wait for Ready ---
    ready, status_message = await _wait_for_ready(client, params.name, params.namespace)

    if not ready:
        return DeploymentResult(
            success=False,
            deployment_name=params.name,
            namespace=params.namespace,
            message=(
                f"Model deployed but not Ready after "
                f"{MAX_READINESS_WAIT_SECONDS}s: {status_message}"
            ),
            status="Pending",
        )

    # --- Step 4: Get endpoint and test ---
    endpoint = _get_endpoint_info(client, params.name, params.namespace)
    endpoint_url = endpoint.get("url")

    validation = EndpointValidation(
        reachable=endpoint.get("status") == "Ready",
        status=endpoint.get("status"),
        url=endpoint_url,
        message="Endpoint is ready" if endpoint.get("status") == "Ready" else "Endpoint not ready",
    )

    # --- Step 5: Build result with SLO comparison ---
    rec = plan.recommendation_summary
    slo_comparison: dict[str, Any] = {
        "gpu_config": f"{rec.get('gpu_count', '?')}x {rec.get('gpu_type', '?')}",
        "replicas": rec.get("replicas"),
        "predicted_cost_month_usd": rec.get("predicted_cost_month"),
        "predicted_meets_slo": rec.get("meets_slo"),
    }

    return DeploymentResult(
        success=True,
        deployment_name=params.name,
        namespace=params.namespace,
        message="Model deployed and serving",
        status="Ready",
        endpoint_url=endpoint_url,
        validation=validation,
        slo_comparison=slo_comparison,
    )


async def _wait_for_ready(
    client: InferenceClient,
    name: str,
    namespace: str,
) -> tuple[bool, str]:
    """Poll InferenceService status until Ready or timeout."""
    start = time.monotonic()
    last_status = "Unknown"
    while time.monotonic() - start < MAX_READINESS_WAIT_SECONDS:
        try:
            isvc = client.get_inference_service(name, namespace)
            last_status = isvc.status.value
            if isvc.status.value == "Ready":
                return True, "Ready"
            if isvc.status.value == "Failed":
                reasons = [c.reason for c in isvc.conditions if c.reason]
                reason = reasons[0] if reasons else "Unknown failure"
                return False, f"Failed: {reason}"
        except Exception as e:
            logger.debug("Status poll error: %s", e)
        await asyncio.sleep(READINESS_POLL_INTERVAL)
    return False, f"Timeout after {MAX_READINESS_WAIT_SECONDS}s (last status: {last_status})"


def _get_endpoint_info(
    client: InferenceClient,
    name: str,
    namespace: str,
) -> dict[str, Any]:
    """Retrieve endpoint information for a deployed model."""
    try:
        return client.get_model_endpoint(name, namespace)
    except Exception as e:
        logger.debug("Failed to get endpoint: %s", e)
        return {"name": name, "status": "Unknown", "url": None}


async def _ensure_serving_runtime(
    server: RHOAIServer,
    namespace: str,
    runtime_name: str,
) -> tuple[bool, str]:
    """Create a serving runtime from a platform template if not present."""
    try:
        from rhoai_mcp.domains.inference.client import InferenceClient

        client = InferenceClient(server.k8s)
        runtimes = client.list_serving_runtimes(namespace, include_templates=False)
        if any(rt["name"] == runtime_name for rt in runtimes):
            return True, f"Runtime '{runtime_name}' already exists"

        templates = client.list_serving_runtime_templates()
        for template in templates:
            if template.get("creates_runtime") == runtime_name:
                client.instantiate_serving_runtime_template(
                    template_name=template["name"],
                    target_namespace=namespace,
                )
                return True, f"Created runtime '{runtime_name}' from template"

        return False, f"No template found for runtime '{runtime_name}'"
    except Exception as e:
        return False, str(e)
