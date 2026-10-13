"""MCP Tools for MaaS (Models-as-a-Service) model deployment.

Deploys models as MaaS endpoints: an LLMInferenceService routed through the
shared MaaS gateway, registered and governed via MaaSModelRef (plus optional
AuthPolicy/Subscription). The MaaS platform itself (gateway, maas-api, DSC
modelsAsService) must already be installed - see check_maas_readiness or the
rhoai-maas-guide setup script.
"""

import logging
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError as PydanticValidationError

from rhoai_mcp.domains.inference.tools import check_pull_secrets, extract_oci_registry
from rhoai_mcp.domains.maas.client import MaasClient
from rhoai_mcp.domains.maas.crds import (
    DEFAULT_MODEL_NAMESPACE,
    GATEWAY_ACCESS_LABEL,
    GATEWAY_NAME,
    GATEWAY_NAMESPACE,
)
from rhoai_mcp.domains.maas.models import (
    DEFAULT_RHAIS_TAG,
    VALID_RATE_LIMIT_WINDOWS,
    MaasModelDeploy,
    needs_registry_tag,
)
from rhoai_mcp.utils.errors import (
    NotFoundError,
    NotManagedByMCPError,
    RHOAIError,
    ValidationError,
)
from rhoai_mcp.utils.validation import (
    validate_namespace,
    validate_resource_name,
    validate_writable_namespace,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

_MAAS_GUIDE_URL = (
    "https://rh-aiservices-bu.github.io/rhoai-maas-guide/modules/main/quick-start.html"
)


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register MaaS tools with the MCP server."""

    @mcp.tool()
    def check_maas_readiness(namespace: str = DEFAULT_MODEL_NAMESPACE) -> dict[str, Any]:
        """Check whether the cluster is ready to deploy models as MaaS endpoints.

        Verifies the MaaS platform components: the DataScienceCluster MaaS
        management state, the shared MaaS gateway, the maas-api deployment,
        and the target namespace's gateway access label.

        Args:
            namespace: Namespace where models will be deployed (default: llm).

        Returns:
            Readiness result with per-check details, issues, remediation, and
            the MaaS endpoint URL.
        """
        client = MaasClient(server.k8s)
        checks: list[dict[str, Any]] = []
        issues: list[str] = []
        ready = True

        try:
            validate_namespace(namespace)
        except ValidationError as e:
            return {"error": str(e)}

        # Check 1: DataScienceCluster MaaS component
        dsc_state = client.get_dsc_maas_state()
        if dsc_state is None:
            checks.append(
                {
                    "name": "DataScienceCluster",
                    "passed": False,
                    "message": "No DataScienceCluster found (RHOAI not installed)",
                }
            )
            issues.append("RHOAI is not installed - no DataScienceCluster found")
            ready = False
        elif dsc_state["management_state"] is None:
            checks.append(
                {
                    "name": "DataScienceCluster",
                    "passed": False,
                    "message": (
                        f"MaaS component is not configured on "
                        f"DataScienceCluster '{dsc_state['name']}'"
                    ),
                }
            )
            issues.append(
                f"MaaS is not configured on DataScienceCluster "
                f"'{dsc_state['name']}'. Set modelsAsService (3.4) or "
                "modelsAsAService (3.5) to Managed - see the rhoai-maas-guide, "
                "Phase 4."
            )
            ready = False
        elif dsc_state["management_state"] != "Managed":
            checks.append(
                {
                    "name": "DataScienceCluster",
                    "passed": False,
                    "message": (
                        f"MaaS managementState is "
                        f"'{dsc_state['management_state']}' (expected 'Managed')"
                    ),
                }
            )
            issues.append(
                f"MaaS is not enabled on DataScienceCluster '{dsc_state['name']}' "
                f"(managementState: '{dsc_state['management_state']}'). "
                "Set modelsAsService (3.4) or modelsAsAService (3.5) to Managed "
                "- see the rhoai-maas-guide, Phase 4."
            )
            ready = False
        else:
            checks.append(
                {
                    "name": "DataScienceCluster",
                    "passed": True,
                    "message": f"MaaS is Managed on '{dsc_state['name']}'",
                }
            )

        # Check 2: MaaS gateway
        gateway = client.get_gateway()
        hostname = client.get_gateway_hostname()
        if gateway is None:
            checks.append(
                {
                    "name": "MaaS Gateway",
                    "passed": False,
                    "message": f"Gateway '{GATEWAY_NAME}' not found in '{GATEWAY_NAMESPACE}'",
                }
            )
            issues.append("MaaS gateway is not installed - see the rhoai-maas-guide, Phase 2.")
            ready = False
        else:
            programmed = client.is_gateway_programmed(gateway)
            checks.append(
                {
                    "name": "MaaS Gateway",
                    "passed": True,
                    "message": (
                        f"Gateway '{GATEWAY_NAME}' exists "
                        f"(programmed: {programmed if programmed is not None else 'unknown'})"
                    ),
                }
            )
            if programmed is False:
                issues.append(
                    f"Gateway '{GATEWAY_NAME}' is not Programmed yet - the endpoint "
                    "may not be reachable until it gets an address."
                )

        # Check 3: maas-api deployment
        maas_api = client.find_maas_api()
        if maas_api is None:
            checks.append(
                {
                    "name": "maas-api",
                    "passed": False,
                    "message": "maas-api deployment not found",
                }
            )
            issues.append(
                "maas-api is not running - the MaaS API will not register models. "
                "See the rhoai-maas-guide, Phase 4."
            )
            ready = False
        elif not maas_api["ready_replicas"]:
            checks.append(
                {
                    "name": "maas-api",
                    "passed": False,
                    "message": (
                        f"maas-api found in '{maas_api['namespace']}' but has 0 ready replicas"
                    ),
                }
            )
            issues.append(
                "maas-api has no ready replicas - the MaaS API will not register "
                "models. Check the maas-api pod events. See the rhoai-maas-guide, "
                "Phase 4."
            )
            ready = False
        else:
            checks.append(
                {
                    "name": "maas-api",
                    "passed": True,
                    "message": (
                        f"maas-api running in '{maas_api['namespace']}' "
                        f"({maas_api['ready_replicas']} ready replica(s))"
                    ),
                }
            )

        # Check 4: target namespace gateway access (informational - deploy can fix it)
        try:
            ns = server.k8s.get_namespace(namespace)
            labels = dict(ns.metadata.labels or {})
            if labels.get(GATEWAY_ACCESS_LABEL) == "true":
                checks.append(
                    {
                        "name": "Namespace gateway access",
                        "passed": True,
                        "message": f"'{namespace}' is labeled for MaaS gateway access",
                    }
                )
            else:
                checks.append(
                    {
                        "name": "Namespace gateway access",
                        "passed": False,
                        "message": f"'{namespace}' is missing '{GATEWAY_ACCESS_LABEL}=true'",
                    }
                )
                issues.append(
                    f"Namespace '{namespace}' is not exposed to the MaaS gateway. "
                    "deploy_maas_model will attempt to label it; otherwise run: "
                    f"oc label namespace {namespace} {GATEWAY_ACCESS_LABEL}=true --overwrite"
                )
        except NotFoundError:
            checks.append(
                {
                    "name": "Namespace gateway access",
                    "passed": False,
                    "message": f"Namespace '{namespace}' does not exist",
                }
            )
            issues.append(
                f"Namespace '{namespace}' does not exist yet (deploy_maas_model will create it)"
            )

        endpoint = f"https://{hostname}" if hostname else None

        return {
            "ready": ready,
            "checks": checks,
            "issues": issues if issues else None,
            "maas_endpoint": endpoint,
            "remediation": (
                f"Install/configure MaaS with the rhoai-maas-guide setup script: {_MAAS_GUIDE_URL}"
            ),
        }

    @mcp.tool()
    def deploy_maas_model(
        name: str,
        model_name: str,
        model_uri: str,
        namespace: str = DEFAULT_MODEL_NAMESPACE,
        image: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        replicas: int = 1,
        cpu_request: str | None = None,
        cpu_limit: str | None = None,
        memory_request: str | None = None,
        memory_limit: str | None = None,
        grant_authenticated_access: bool = False,
        token_rate_limit: int = 100,
        rate_limit_window: str = "1m",
    ) -> dict[str, Any]:
        """Deploy a model as a Models-as-a-Service (MaaS) endpoint.

        Creates an LLMInferenceService routed through the shared MaaS gateway
        and registers it via MaaSModelRef so the MaaS API governs it. With no
        image, deploys the CPU simulator from the rhoai-maas-guide so MaaS can
        be tried without a real model or GPU; for real inference pass a vLLM
        image (e.g. registry.redhat.io/rhaiis/vllm-cpu-rhel9:3.3.0) with the
        model location - registry.redhat.io images require an explicit tag
        (a tagless reference gets the documented default appended).

        The MaaS platform (gateway, maas-api, DSC modelsAsService) must
        already be installed - run check_maas_readiness first, or install it
        with the rhoai-maas-guide setup script.

        Args:
            name: LLMInferenceService name (lowercase DNS name, e.g. 'qwen3-06b').
            model_name: Served model name (e.g. 'facebook/opt-125m').
            model_uri: Model location (hf://org/repo or oci://registry/repo:tag).
            namespace: Target namespace (default 'llm'; created and labeled for
                gateway access if missing).
            image: Serving container image. None deploys the CPU simulator.
            display_name: Human-readable display name.
            description: Human-readable description.
            replicas: Number of replicas.
            cpu_request: CPU request (None = preset default).
            cpu_limit: CPU limit (None = preset default).
            memory_request: Memory request (None = preset default).
            memory_limit: Memory limit (None = preset default).
            grant_authenticated_access: Grant all authenticated users access
                with token rate limits (creates AuthPolicy + Subscription).
                Default False (least privilege - access must be granted
                explicitly).
            token_rate_limit: Tokens per rate limit window when access is granted.
            rate_limit_window: Rate limit window (e.g. '1m').

        Returns:
            Created resources, the MaaS endpoint URL, and next steps.
        """
        # Safety gates
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        # Validate inputs
        try:
            validate_resource_name(name)
            validate_writable_namespace(namespace)
        except ValidationError as e:
            return {"error": str(e)}

        if not model_name or not model_name.strip():
            return {
                "error": (
                    "model_name must be a non-empty served model name (e.g. 'facebook/opt-125m')."
                ),
            }

        if not model_uri.startswith(("hf://", "oci://", "pvc://")):
            return {
                "error": (
                    f"Unsupported model_uri: '{model_uri}'. "
                    "Use hf://org/repo, oci://registry/repo:tag, or pvc://claim/path."
                ),
            }

        if grant_authenticated_access and rate_limit_window not in VALID_RATE_LIMIT_WINDOWS:
            return {
                "error": (
                    f"Invalid rate_limit_window: '{rate_limit_window}'. "
                    f"Must be one of: {', '.join(VALID_RATE_LIMIT_WINDOWS)}."
                ),
            }

        client = MaasClient(server.k8s)
        warnings: list[str] = []

        # Fail closed if the MaaS platform is not ready - a model deployed now
        # would never become reachable through the gateway.
        readiness_hints: list[str] = []
        dsc_state = client.get_dsc_maas_state()
        if dsc_state is None or dsc_state.get("management_state") != "Managed":
            readiness_hints.append(
                "DataScienceCluster MaaS component is not Managed - the model will "
                "not be served until MaaS is enabled (rhoai-maas-guide, Phase 4)."
            )
        if client.get_gateway() is None:
            readiness_hints.append(
                f"MaaS gateway '{GATEWAY_NAME}' not found - the endpoint will not be "
                "reachable until it is installed (rhoai-maas-guide, Phase 2)."
            )
        if readiness_hints:
            return {
                "error": "MaaS platform is not ready",
                "issues": readiness_hints,
                "message": (
                    "Run check_maas_readiness for details, or install MaaS with the "
                    "rhoai-maas-guide setup script first."
                ),
                "remediation": _MAAS_GUIDE_URL,
            }

        # registry.redhat.io repositories reject implicit ':latest' pulls, so a
        # tagless image would be undeployable - append the documented default
        # before the request is built.
        if image and needs_registry_tag(image):
            image = f"{image}:{DEFAULT_RHAIS_TAG}"
            warnings.append(
                f"Appended the documented default tag ':{DEFAULT_RHAIS_TAG}' to "
                f"'{image.rsplit(':', 1)[0]}' - registry.redhat.io repositories do "
                "not publish a ':latest' tag. Pin an explicit version for "
                "production deployments."
            )

        try:
            request = MaasModelDeploy(
                name=name,
                namespace=namespace,
                model_name=model_name,
                model_uri=model_uri,
                image=image,
                display_name=display_name,
                description=description,
                replicas=replicas,
                cpu_request=cpu_request,
                cpu_limit=cpu_limit,
                memory_request=memory_request,
                memory_limit=memory_limit,
                grant_authenticated_access=grant_authenticated_access,
                token_rate_limit=token_rate_limit,
                rate_limit_window=rate_limit_window,
            )
        except PydanticValidationError as e:
            return {"error": f"Invalid parameters: {e}"}

        # Warn (not block) about pull secrets for oci:// model URIs
        if model_uri.startswith("oci://"):
            registry = extract_oci_registry(model_uri)
            if registry:
                check_pull_secrets(server.k8s, namespace, registry, warnings)

        # Ensure the namespace exists and is exposed to the gateway
        ns_info = client.ensure_model_namespace(namespace)
        warnings.extend(ns_info["warnings"])

        # Create the MaaS resources, tracking what was created so a partial
        # failure is reported instead of leaving the caller guessing.
        created: dict[str, str | None] = {
            "llm_inference_service": None,
            "maas_model_ref": None,
            "maas_auth_policy": None,
            "maas_subscription": None,
        }
        try:
            isvc = client.deploy_model(request)
            created["llm_inference_service"] = isvc.metadata.name
        except RHOAIError as e:
            return {
                "error": str(e),
                "message": "MaaS model deployment failed - no resources were created",
                "warnings": warnings or None,
                "created": created or None,
            }

        try:
            model_ref = client.create_model_ref(request)
            created["maas_model_ref"] = model_ref.metadata.name
        except RHOAIError as e:
            return {
                "error": str(e),
                "message": (
                    f"LLMInferenceService '{name}' was created but MaaS registration "
                    "failed - the model will not be governed by MaaS until the "
                    "MaaSModelRef exists. Fix the error and re-run to retry "
                    "registration."
                ),
                "warnings": warnings or None,
                "created": created,
            }

        auth_policy: Any = None
        subscription: Any = None
        if grant_authenticated_access:
            try:
                auth_policy = client.create_auth_policy(request)
                subscription = client.create_subscription(request)
            except RHOAIError as e:
                warnings.append(
                    f"Model deployed but governance resources could not be created: {e}. "
                    "Only admins can access the model until an AuthPolicy and "
                    "Subscription grant access."
                )
        created["maas_auth_policy"] = auth_policy.metadata.name if auth_policy else None
        created["maas_subscription"] = subscription.metadata.name if subscription else None

        hostname = client.get_gateway_hostname()
        endpoint = f"https://{hostname}" if hostname else None

        next_steps = [
            f"Use get_maas_model(name='{name}', namespace='{namespace}') to check "
            "deployment status and MaaS registration.",
            f"Clients must pass model name '{name}' (the LLMInferenceService "
            "name, not the HF model name) in OpenAI requests.",
        ]
        if not grant_authenticated_access:
            next_steps.append(
                "No access was granted (least privilege default). Grant access by "
                "creating a MaaSAuthPolicy and MaaSSubscription for the model, or "
                "re-run with grant_authenticated_access=true to grant all "
                "authenticated users."
            )
        if endpoint:
            next_steps.append(
                f"Send inference requests to {endpoint}/v1/chat/completions with a MaaS API key."
            )

        return {
            "name": name,
            "namespace": namespace,
            "model_name": model_name,
            "model_uri": model_uri,
            "created": created,
            "namespace_created": ns_info["created"],
            "maas_endpoint": endpoint,
            "message": (
                f"MaaS model '{name}' deployment initiated. The model registers with "
                "the MaaS gateway via MaaSModelRef and may take a few minutes to "
                "become ready."
            ),
            "next_steps": next_steps,
            "warnings": warnings or None,
            "_source": {
                "kind": "LLMInferenceService",
                "api_version": "serving.kserve.io/v1alpha1",
                "name": name,
                "namespace": namespace,
                "uid": None,
            },
        }

    @mcp.tool()
    def get_maas_model(
        name: str,
        namespace: str = DEFAULT_MODEL_NAMESPACE,
    ) -> dict[str, Any]:
        """Get the status and MaaS endpoint of a model deployed as a MaaS endpoint.

        Combines the LLMInferenceService status with the MaaSModelRef
        registration state and the MaaS gateway endpoint URL.

        Args:
            name: LLMInferenceService name.
            namespace: Namespace the model is deployed in (default: llm).

        Returns:
            Deployment status, MaaS registration state, and endpoint URL.
        """
        try:
            validate_resource_name(name)
            validate_namespace(namespace)
        except ValidationError as e:
            return {"error": str(e)}

        client = MaasClient(server.k8s)
        hostname = client.get_gateway_hostname()
        endpoint = f"https://{hostname}" if hostname else None

        try:
            status = client.get_model_status(name, namespace, maas_endpoint=endpoint)
        except RHOAIError as e:
            return {"error": str(e)}

        if status.status == "Ready" and status.maas_phase == "Ready":
            message = f"MaaS model '{name}' is Ready and registered with the MaaS API" + (
                f" at {endpoint}" if endpoint else ""
            )
        elif status.status == "Ready":
            message = (
                f"Deployment '{name}' is Ready but MaaS registration is "
                f"'{status.maas_phase or 'pending'}' - the model may not appear in "
                "the MaaS API yet."
            )
        else:
            message = f"MaaS model '{name}' is {status.status} - endpoint may not be available"

        return {
            "name": status.metadata.name,
            "namespace": status.metadata.namespace,
            "model_name": status.model_name,
            "model_uri": status.model_uri,
            "status": status.status,
            "replicas": status.replicas,
            "ready_replicas": status.ready_replicas,
            "maas_registered": status.maas_registered,
            "maas_phase": status.maas_phase,
            "maas_endpoint": status.maas_endpoint,
            "conditions": [
                {
                    "type": c.type,
                    "status": c.status,
                    "reason": c.reason,
                    "message": c.message,
                }
                for c in status.conditions
            ],
            "message": message,
            "_source": status.metadata.to_source_dict(),
        }

    @mcp.tool()
    def delete_maas_model(
        name: str,
        namespace: str = DEFAULT_MODEL_NAMESPACE,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete a model deployed as a MaaS endpoint and its registration.

        Deletes the LLMInferenceService, the MaaSModelRef, and the MaaS
        governance resources (MaaSAuthPolicy, MaaSSubscription) created by
        deploy_maas_model, following the rhoai-maas-guide cleanup order
        (governance first, then registration, then serving). Only resources
        carrying the managed-by label can be deleted. Governance resources
        that do not exist are skipped; hand-created governance resources are
        kept and reported so no access is silently removed.

        Args:
            name: LLMInferenceService name.
            namespace: Namespace the model is deployed in (default: llm).
            confirm: Must be True to actually delete.

        Returns:
            Lists of deleted, skipped, and kept resources.
        """
        if server.config.read_only_mode:
            return {"error": "Read-only mode is enabled"}

        if not confirm:
            return {
                "error": "Deletion not confirmed",
                "message": f"To delete MaaS model '{name}', set confirm=True.",
            }

        try:
            validate_resource_name(name)
            validate_namespace(namespace)
        except ValidationError as e:
            return {"error": str(e)}

        client = MaasClient(server.k8s)
        try:
            result = client.delete_model_resources(name, namespace)
        except (NotFoundError, NotManagedByMCPError) as e:
            return {"error": str(e)}

        warnings: list[str] = []
        for kind in result["kept"]:
            warnings.append(
                f"Kept {kind} - it was not created by this MCP server "
                "(managed-by label missing) and may still grant access to "
                "the model name."
            )

        message = f"MaaS model '{name}' deleted"
        if result["kept"]:
            message = (
                f"MaaS model '{name}' deleted; kept non-MCP-managed: {', '.join(result['kept'])}"
            )

        return {
            "name": name,
            "namespace": namespace,
            "deleted": result["deleted"] or None,
            "skipped": result["skipped"] or None,
            "kept": result["kept"] or None,
            "message": message,
            "warnings": warnings or None,
            "_source": {
                "kind": "LLMInferenceService",
                "api_version": "serving.kserve.io/v1alpha1",
                "name": name,
                "namespace": namespace,
                "uid": None,
            },
        }
