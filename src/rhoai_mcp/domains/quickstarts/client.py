"""Client for quickstart discovery and deployment operations.

Discovery reads OCI artifacts (registry index + per-quickstart manifests) via
:class:`OCIArtifactClient`. Deployment mechanically translates a chosen action
and parameter values into a Kubernetes Job that runs the quickstart's installer
image — the agent supplies intent and values; this client does the plumbing.
"""

from __future__ import annotations

import logging
import secrets as secrets_mod
from typing import TYPE_CHECKING, Any

from kubernetes.client import ApiException  # type: ignore[import-untyped]

from rhoai_mcp.clients.base import CRDDefinition
from rhoai_mcp.domains.quickstarts.models import (
    QuickstartManifest,
    QuickstartRegistry,
)
from rhoai_mcp.domains.quickstarts.oci import (
    MANIFEST_MEDIA_TYPE,
    REGISTRY_MEDIA_TYPE,
    OCIArtifactClient,
)
from rhoai_mcp.utils.errors import NotFoundError, ValidationError
from rhoai_mcp.utils.labels import RHOAILabels

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient
    from rhoai_mcp.config import RHOAIConfig

logger = logging.getLogger(__name__)

# Job resource (no dedicated helper exists on K8sClient for batch/v1).
JOB_CRD = CRDDefinition(group="batch", version="v1", plural="jobs", kind="Job")

# Quickstart-specific labels applied to Jobs and their parameter Secrets.
QUICKSTART_NAME_LABEL = "quickstart.redhat.com/name"
QUICKSTART_ACTION_LABEL = "quickstart.redhat.com/action"

# Actions that destroy data and therefore require dangerous-operations + confirm.
DESTRUCTIVE_ACTIONS = {"UNINSTALL_DELETE_ALL"}

_MAX_NAME_LEN = 63


def _stringify(value: Any) -> str:
    """Render a parameter value as the string an installer env var expects."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class QuickstartsClient:
    """Client for quickstart catalog discovery and deployment actions."""

    def __init__(
        self,
        k8s: K8sClient,
        config: RHOAIConfig,
        oci: OCIArtifactClient | None = None,
    ) -> None:
        self._k8s = k8s
        self._config = config
        self._oci = oci or OCIArtifactClient(
            timeout=config.quickstart_oci_timeout,
            verify=not config.quickstart_oci_skip_tls_verify,
        )

    # -- discovery ----------------------------------------------------------

    def get_registry(self) -> QuickstartRegistry:
        """Fetch and parse the registry index artifact."""
        raw = self._oci.fetch_layer(self._config.quickstart_registry_ref, REGISTRY_MEDIA_TYPE)
        return QuickstartRegistry.from_yaml(raw)

    def get_manifest(self, name: str, version: str | None = None) -> QuickstartManifest:
        """Fetch and parse the manifest for a named quickstart.

        Resolves the manifest OCI reference from the registry entry (repo +
        version), so callers only need the quickstart name.
        """
        entry = self.get_registry().get(name)
        if entry is None:
            raise NotFoundError("Quickstart", name)
        raw = self._oci.fetch_layer(entry.manifest_ref(version), MANIFEST_MEDIA_TYPE)
        return QuickstartManifest.from_yaml(raw)

    # -- deployment ---------------------------------------------------------

    def run_action(
        self,
        name: str,
        action: str,
        target_namespace: str | None = None,
        mode: str = "demo",
        parameters: dict[str, Any] | None = None,
        version: str | None = None,
        source_version: str | None = None,
    ) -> dict[str, Any]:
        """Create a Job running the quickstart installer for ``action``.

        Validates the action/mode/parameters against the manifest, materialises
        secret parameters into a Secret in the Job's namespace, and submits the
        Job. Returns provenance the caller can poll with the status/logs tools.
        """
        parameters = parameters or {}
        manifest = self.get_manifest(name, version)
        action_upper = action.upper()

        if not manifest.supports_action(action_upper):
            raise ValidationError(
                f"action '{action_upper}' is not supported by '{name}'; "
                f"supported: {manifest.deployment.supported_actions}"
            )

        if action_upper in {"INSTALL", "UPGRADE"} and not manifest.supports_mode(mode):
            raise ValidationError(
                f"mode '{mode}' is not supported by '{name}'; "
                f"supported: {manifest.deployment.supported_modes}"
            )

        ns = target_namespace or manifest.deployment.default_namespace
        if not ns:
            raise ValidationError(
                "target_namespace is required (quickstart declares no defaultNamespace)"
            )

        if action_upper == "UPGRADE" and not source_version:
            raise ValidationError("source_version is required for the UPGRADE action")

        # Required parameters are only meaningful for actions that deploy or
        # upgrade the workload; STATUS/CHECK_PRE_REQS/UNINSTALL need no inputs.
        enforce_required = action_upper in {"INSTALL", "UPGRADE"}
        env, secret_data = self._build_parameter_env(manifest, parameters, enforce_required)

        env.extend(
            [
                {"name": "ACTION", "value": action_upper},
                {"name": "TARGET_NAMESPACE", "value": ns},
                {"name": "INSTALL_MODE", "value": mode.lower()},
            ]
        )
        if action_upper == "UPGRADE":
            env.append({"name": "SOURCE_VERSION", "value": source_version or ""})
            env.append({"name": "TARGET_VERSION", "value": manifest.version or version or ""})

        suffix = secrets_mod.token_hex(3)
        job_ns = self._config.quickstart_job_namespace
        job_name = self._job_name(name, action_upper, suffix)
        labels = {
            **RHOAILabels.managed_by_mcp_labels(),
            QUICKSTART_NAME_LABEL: name,
            QUICKSTART_ACTION_LABEL: action_upper.lower(),
        }

        secret_name: str | None = None
        if secret_data:
            secret_name = f"{job_name}-params"[:_MAX_NAME_LEN]
            self._k8s.create_secret(
                name=secret_name,
                namespace=job_ns,
                data=secret_data,
                labels=labels,
                string_data=True,
            )
            env.extend(
                {
                    "name": key,
                    "valueFrom": {"secretKeyRef": {"name": secret_name, "key": key}},
                }
                for key in secret_data
            )

        body = self._build_job_body(job_name, job_ns, manifest, env, labels)

        try:
            created = self._k8s.create(JOB_CRD, body=body, namespace=job_ns)
        except Exception:
            if secret_name:
                self._delete_secret_quietly(secret_name, job_ns)
            raise

        if secret_name:
            self._adopt_secret(secret_name, job_ns, job_name, created)

        return {
            "quickstart": name,
            "action": action_upper,
            "mode": mode.lower(),
            "target_namespace": ns,
            "job_name": job_name,
            "job_namespace": job_ns,
            "installer_image": manifest.deployment.installer.image,
            "parameters_secret": secret_name,
            "_source": {
                "kind": "Job",
                "api_version": "batch/v1",
                "name": job_name,
                "namespace": job_ns,
                "uid": self._resource_uid(created),
            },
        }

    def get_action_status(self, job_name: str, namespace: str) -> dict[str, Any]:
        """Return the status of a previously created installer Job."""
        job = self._k8s.get(JOB_CRD, job_name, namespace)
        data = job.to_dict() if hasattr(job, "to_dict") else dict(job)
        status = data.get("status") or {}
        meta = data.get("metadata") or {}
        labels = meta.get("labels") or {}

        succeeded = int(status.get("succeeded") or 0)
        failed = int(status.get("failed") or 0)
        active = int(status.get("active") or 0)
        if succeeded:
            phase = "Complete"
        elif failed:
            phase = "Failed"
        elif active:
            phase = "Running"
        else:
            phase = "Pending"

        return {
            "job_name": job_name,
            "namespace": namespace,
            "quickstart": labels.get(QUICKSTART_NAME_LABEL),
            "action": labels.get(QUICKSTART_ACTION_LABEL),
            "phase": phase,
            "active": active,
            "succeeded": succeeded,
            "failed": failed,
            "start_time": status.get("startTime"),
            "completion_time": status.get("completionTime"),
            "conditions": status.get("conditions") or [],
            "_source": {
                "kind": "Job",
                "api_version": "batch/v1",
                "name": job_name,
                "namespace": namespace,
                "uid": meta.get("uid"),
            },
        }

    def get_action_logs(
        self, job_name: str, namespace: str, tail_lines: int = 200
    ) -> dict[str, Any]:
        """Return logs from the installer pod for a Job."""
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace, label_selector=f"job-name={job_name}"
        ).items
        if not pods:
            raise NotFoundError("Pod", f"job-name={job_name}", namespace)

        pod = max(
            pods,
            key=lambda p: p.metadata.creation_timestamp or "",
        )
        pod_name = pod.metadata.name

        try:
            logs = self._k8s.core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=namespace, tail_lines=tail_lines
            )
        except ApiException as exc:
            logs = f"<logs unavailable: {exc.reason}>"

        return {
            "job_name": job_name,
            "namespace": namespace,
            "pod": pod_name,
            "logs": logs,
        }

    # -- internals ----------------------------------------------------------

    def _build_parameter_env(
        self,
        manifest: QuickstartManifest,
        parameters: dict[str, Any],
        enforce_required: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Validate provided parameters and split them into env vars and secrets."""
        known = {p.name for p in manifest.parameters.all()}
        unknown = sorted(set(parameters) - known)
        if unknown:
            raise ValidationError(f"unknown parameter(s): {', '.join(unknown)}")

        missing: list[str] = []
        env: list[dict[str, Any]] = []
        secret_data: dict[str, str] = {}

        for param in manifest.parameters.all():
            provided = param.name in parameters
            value = parameters[param.name] if provided else param.default

            if value is None:
                if param.required and enforce_required:
                    missing.append(param.name)
                continue

            if manifest.parameters.is_secret(param.name):
                secret_data[param.resolved_env_var()] = _stringify(value)
            else:
                env.append({"name": param.resolved_env_var(), "value": _stringify(value)})

        if missing:
            raise ValidationError(f"missing required parameter(s): {', '.join(sorted(missing))}")

        return env, secret_data

    def _build_job_body(
        self,
        job_name: str,
        namespace: str,
        manifest: QuickstartManifest,
        env: list[dict[str, Any]],
        labels: dict[str, str],
    ) -> dict[str, Any]:
        """Assemble the Job manifest for the installer."""
        container: dict[str, Any] = {
            "name": "installer",
            "image": manifest.deployment.installer.image,
            "env": env,
        }
        if manifest.deployment.installer.command:
            container["command"] = manifest.deployment.installer.command

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": self._config.quickstart_job_ttl_seconds,
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "serviceAccountName": self._config.quickstart_installer_service_account,
                        "restartPolicy": "Never",
                        "containers": [container],
                    },
                },
            },
        }

    def _adopt_secret(
        self, secret_name: str, namespace: str, job_name: str, created_job: Any
    ) -> None:
        """Set an ownerReference from the params Secret to its Job for GC."""
        uid = self._resource_uid(created_job)
        if not uid:
            return
        body = {
            "metadata": {
                "ownerReferences": [
                    {
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "name": job_name,
                        "uid": uid,
                        "controller": True,
                        "blockOwnerDeletion": True,
                    }
                ]
            }
        }
        try:
            self._k8s.core_v1.patch_namespaced_secret(
                name=secret_name, namespace=namespace, body=body
            )
        except ApiException as exc:
            logger.warning("could not set ownerReference on secret %s: %s", secret_name, exc.reason)

    def _delete_secret_quietly(self, secret_name: str, namespace: str) -> None:
        """Best-effort delete of an orphaned params Secret."""
        try:
            self._k8s.core_v1.delete_namespaced_secret(name=secret_name, namespace=namespace)
        except ApiException as exc:
            logger.warning("could not clean up secret %s: %s", secret_name, exc.reason)

    @staticmethod
    def _resource_uid(resource: Any) -> str | None:
        """Extract metadata.uid from a created dynamic resource."""
        data = resource.to_dict() if hasattr(resource, "to_dict") else resource
        if isinstance(data, dict):
            return (data.get("metadata") or {}).get("uid")
        return None

    @staticmethod
    def _job_name(name: str, action_upper: str, suffix: str) -> str:
        """Build a DNS-safe, unique Job name."""
        action_slug = action_upper.lower().replace("_", "-")
        reserved = len(suffix) + len(action_slug) + 5  # "qs-" + two "-" joiners
        stem = name[: max(1, _MAX_NAME_LEN - reserved)]
        return f"qs-{stem}-{action_slug}-{suffix}"
