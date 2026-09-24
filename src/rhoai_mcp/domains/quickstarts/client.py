"""Client for quickstart discovery and deployment operations.

Discovery reads OCI artifacts (registry index + per-quickstart manifests) via
:class:`OCIArtifactClient`. Deployment mechanically translates a chosen action
and parameter values into a Kubernetes Job that runs the quickstart's installer
image — the agent supplies intent and values; this client does the plumbing.
"""

from __future__ import annotations

import json
import logging
import re
import secrets as secrets_mod
import shlex
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
    OCIError,
    parse_ref,
)
from rhoai_mcp.utils.errors import NotFoundError, RHOAIError, ValidationError
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
# Records an action's target namespace on the Job; also the key the status
# reconcile uses to find which namespace to stamp as MCP-managed.
QUICKSTART_TARGET_NS_LABEL = "quickstart.redhat.com/target-namespace"

# Actions that destroy data and therefore additionally require dangerous-operations.
DESTRUCTIVE_ACTIONS = {"UNINSTALL_DELETE_ALL"}

# Teardown prefix: every UNINSTALL_* action tears down the deployed quickstart and
# so requires explicit confirmation, mirroring the project's delete_* tools.
UNINSTALL_ACTION_PREFIX = "UNINSTALL_"

# Actions permitted only on a namespace this tool already installed a quickstart
# into, identified by the managed-by-mcp label on the namespace itself.
MANAGED_NAMESPACE_ACTIONS = {
    "UNINSTALL_KEEP_DATA",
    "UNINSTALL_DELETE_ALL",
    "UPGRADE",
    "STATUS",
}

# Actions whose installer Job creates/owns the target namespace. For these the
# Job command is wrapped so that, on a successful installer run, it stamps the
# managed-by-mcp label onto that namespace; the guarded actions above then
# recognise it as ours. Because the label is written inside the Job, it does not
# depend on anyone polling the Job's status afterwards.
NAMESPACE_LABELING_ACTIONS = {"INSTALL", "UPGRADE"}

# Namespace prefixes that are always off-limits as deployment targets: deploying
# into (or, worse, deleting) a platform namespace is a privilege-escalation risk.
RESERVED_NAMESPACE_PREFIXES = ("openshift-", "kube-")

# Deliberately generic, detail-free message surfaced to the agent whenever a
# quickstart artifact (registry, manifest, or installer image) is requested from
# a repository outside the configured allowlist. The offending reference is only
# ever logged server-side, never returned, so a caller cannot probe the rule.
INVALID_OPERATION_MESSAGE = "An invalid operation was detected and stopped."

_MAX_NAME_LEN = 63
_MAX_NAMESPACE_LEN = 63
# Kubernetes namespaces must be a DNS-1123 label.
_NAMESPACE_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


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
        ref = self._config.quickstart_registry_ref
        self._ensure_allowed_ref(ref)
        raw = self._oci.fetch_layer(ref, REGISTRY_MEDIA_TYPE)
        return QuickstartRegistry.from_yaml(raw)

    def get_manifest(self, name: str, version: str | None = None) -> QuickstartManifest:
        """Fetch and parse the manifest for a named quickstart.

        Resolves the manifest OCI reference from the registry entry (repo +
        version), so callers only need the quickstart name.
        """
        entry = self.get_registry().get(name)
        if entry is None:
            raise NotFoundError("Quickstart", name)
        # manifestRepo comes from remote registry content; never trust it blindly.
        ref = entry.manifest_ref(version)
        self._ensure_allowed_ref(ref)
        raw = self._oci.fetch_layer(ref, MANIFEST_MEDIA_TYPE)
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

        # installer.image comes from remote manifest content and runs as
        # cluster-admin; refuse anything outside the allowlist before doing work.
        self._ensure_allowed_ref(manifest.deployment.installer.image)

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

        # INSTALL/UPGRADE wrap the installer command to label the target namespace
        # as MCP-managed on success, so a declared command is required.
        if action_upper in NAMESPACE_LABELING_ACTIONS and not manifest.deployment.installer.command:
            raise ValidationError(
                f"quickstart '{name}' does not declare an installer command; "
                f"'{action_upper}' requires one so the installer Job can mark the "
                "target namespace as managed by this tool"
            )

        ns = target_namespace or manifest.deployment.default_namespace
        if not ns:
            raise ValidationError(
                "target_namespace is required (quickstart declares no defaultNamespace)"
            )
        self._validate_namespace_name(ns)
        self._check_namespace_for_action(action_upper, ns)

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
            QUICKSTART_TARGET_NS_LABEL: ns,
        }

        secret_name: str | None = None
        if secret_data:
            secret_name = f"{job_name}-params"[:_MAX_NAME_LEN]
            env.extend(
                {
                    "name": key,
                    "valueFrom": {"secretKeyRef": {"name": secret_name, "key": key}},
                }
                for key in secret_data
            )

        body = self._build_job_body(job_name, job_ns, manifest, env, labels, action_upper)

        # Create the Job first so the params Secret can be created already owned by
        # it: the Secret is then garbage-collected together with the Job when its
        # TTL expires, with no fragile post-creation patch and no window in which
        # an unowned password Secret could be left behind.
        created = self._k8s.create(JOB_CRD, body=body, namespace=job_ns)
        if secret_name:
            self._create_owned_secret(secret_name, job_ns, secret_data, labels, job_name, created)

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
        """Return the status of a previously created installer Job.

        Job-level counters alone collapse every failure to "Failed". The
        installer instead writes a structured JSON result to its
        termination-log and exits with a meaningful code (e.g. 2 =
        prerequisites_failed), so this also surfaces the installer container's
        terminated ``exit_code`` and ``termination_message`` — the actual reason
        the agent needs to decide what to do next — plus the parsed ``result``
        when the message is JSON.
        """
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

        exit_code, termination_message = self._get_termination_details(job_name, namespace)

        return {
            "job_name": job_name,
            "namespace": namespace,
            "quickstart": labels.get(QUICKSTART_NAME_LABEL),
            "action": labels.get(QUICKSTART_ACTION_LABEL),
            "phase": phase,
            "active": active,
            "succeeded": succeeded,
            "failed": failed,
            "exit_code": exit_code,
            "termination_message": termination_message,
            "result": self._parse_json_result(termination_message),
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

    def _get_termination_details(
        self, job_name: str, namespace: str
    ) -> tuple[int | None, str | None]:
        """Best-effort read of the installer container's terminated exit code/message.

        Returns ``(None, None)`` if the pod is not (yet) terminated or has
        already been garbage-collected — status must degrade gracefully rather
        than fail when the richer detail is simply unavailable.
        """
        try:
            pods = self._k8s.core_v1.list_namespaced_pod(
                namespace=namespace, label_selector=f"job-name={job_name}"
            ).items
            if not pods:
                return None, None
            pod = max(pods, key=self._pod_creation_key)
            statuses = getattr(pod.status, "container_statuses", None) or []
            installer = next((s for s in statuses if getattr(s, "name", None) == "installer"), None)
            container = installer or (statuses[0] if statuses else None)
            terminated = getattr(getattr(container, "state", None), "terminated", None)
            if terminated is None:
                return None, None
            return getattr(terminated, "exit_code", None), getattr(terminated, "message", None)
        except Exception as exc:  # noqa: BLE001 - detail is optional; never fail status on it
            logger.debug("could not read termination details for job %s: %s", job_name, exc)
            return None, None

    @staticmethod
    def _pod_creation_key(pod: Any) -> tuple[bool, Any]:
        """Sort key selecting the newest pod, tolerant of a missing timestamp.

        Returning ``(has_timestamp, timestamp)`` keeps pods without a
        ``creationTimestamp`` strictly older than any that have one, so ``max``
        never compares a ``datetime`` against a fallback of a different type.
        """
        ts = getattr(getattr(pod, "metadata", None), "creation_timestamp", None)
        return (ts is not None, ts)

    @staticmethod
    def _parse_json_result(message: str | None) -> dict[str, Any] | None:
        """Parse a termination message into a structured result, if it is JSON."""
        if not message:
            return None
        try:
            parsed = json.loads(message)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def get_action_logs(
        self, job_name: str, namespace: str, tail_lines: int = 200
    ) -> dict[str, Any]:
        """Return logs from the installer pod for a Job."""
        try:
            pods = self._k8s.core_v1.list_namespaced_pod(
                namespace=namespace, label_selector=f"job-name={job_name}"
            ).items
        except ApiException as exc:
            if exc.status == 404:
                raise NotFoundError("Namespace", namespace)
            raise RHOAIError(f"failed to list pods for job '{job_name}': {exc.reason}")
        if not pods:
            raise NotFoundError("Pod", f"job-name={job_name}", namespace)

        pod = max(pods, key=self._pod_creation_key)
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

    # -- supply-chain safety ------------------------------------------------

    def _ensure_allowed_ref(self, ref: str) -> None:
        """Block any artifact reference outside the configured repository allowlist.

        The registry index, per-quickstart manifests, and installer images are all
        named by references that ultimately derive from remote (untrusted) content
        and run as cluster-admin. Only references whose ``registry/repository``
        falls under an allowlisted prefix are permitted. Matching is on whole path
        segments (trailing-slash boundary), so a sibling namespace such as
        ``quay.io/rh-ai-quickstart-evil/`` cannot pass by shared prefix. Any
        rejection surfaces a generic message and logs the real reference
        server-side only.
        """
        try:
            parsed = parse_ref(ref)
            candidate = f"{parsed.registry}/{parsed.repository}"
        except OCIError:
            candidate = None

        if candidate is not None:
            for allowed in self._config.quickstart_allowed_repos:
                base = allowed.rstrip("/")
                if candidate == base or candidate.startswith(base + "/"):
                    return

        logger.warning(
            "blocked quickstart artifact reference outside allowlist: %r (allowed: %s)",
            ref,
            self._config.quickstart_allowed_repos,
        )
        raise ValidationError(INVALID_OPERATION_MESSAGE)

    # -- namespace safety ---------------------------------------------------

    def _validate_namespace_name(self, ns: str) -> None:
        """Reject malformed or reserved target namespaces before any cluster call.

        Messages state the violated rule plainly (they only ever echo the
        caller's own input) and never disclose cluster contents.
        """
        if len(ns) > _MAX_NAMESPACE_LEN:
            raise ValidationError(
                f"target namespace '{ns}' is too long; it must be at most "
                f"{_MAX_NAMESPACE_LEN} characters"
            )
        if not _NAMESPACE_RE.fullmatch(ns):
            raise ValidationError(
                "target namespace must be a valid Kubernetes namespace name: only "
                "lowercase letters, digits and '-', starting and ending with a "
                "letter or digit"
            )
        if ns.startswith(RESERVED_NAMESPACE_PREFIXES):
            raise ValidationError(
                f"target namespace '{ns}' is reserved and cannot be used for "
                "quickstart actions; namespaces beginning with 'openshift-' or "
                "'kube-' are protected"
            )

    def _check_namespace_for_action(self, action_upper: str, ns: str) -> None:
        """Enforce per-action rules on whether the target namespace may exist.

        INSTALL creates the namespace, so it must not already exist. The
        data-affecting actions may only touch a namespace this tool installed a
        quickstart into (identified by the managed-by-mcp label).
        """
        if action_upper == "INSTALL":
            if self._get_namespace_or_none(ns) is not None:
                raise ValidationError(
                    f"target namespace '{ns}' already exists; INSTALL creates the "
                    "namespace, so choose one that does not yet exist"
                )
        elif action_upper in MANAGED_NAMESPACE_ACTIONS:
            existing = self._get_namespace_or_none(ns)
            managed = existing is not None and RHOAILabels.is_managed_by_mcp(
                self._namespace_labels(existing)
            )
            # One message for both "absent" and "not ours" so we never disclose
            # whether an arbitrary namespace exists on the cluster.
            if not managed:
                raise ValidationError(
                    f"target namespace '{ns}' is not a quickstart namespace managed "
                    f"by this tool; '{action_upper}' is only permitted on namespaces "
                    "where this MCP server installed a quickstart"
                )

    def _get_namespace_or_none(self, ns: str) -> Any:
        """Return the namespace object, or None if it does not exist."""
        try:
            return self._k8s.get_namespace(ns)
        except NotFoundError:
            return None

    @staticmethod
    def _namespace_labels(ns_obj: Any) -> dict[str, str]:
        """Extract metadata.labels from a namespace object or dict."""
        meta = getattr(ns_obj, "metadata", None)
        labels = getattr(meta, "labels", None)
        if labels is None and isinstance(ns_obj, dict):
            labels = (ns_obj.get("metadata") or {}).get("labels")
        return dict(labels or {})

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
        action_upper: str,
    ) -> dict[str, Any]:
        """Assemble the Job manifest for the installer."""
        # Additive job-identity env the installer may read: JOB_NAME (the Job the
        # MCP created) and JOB_NAMESPACE (the namespace the installer pod runs in,
        # via the downward API — distinct from TARGET_NAMESPACE). These make the
        # installer's own namespace/name explicit rather than relying on detection.
        job_identity_env = [
            {"name": "JOB_NAME", "value": job_name},
            {
                "name": "JOB_NAMESPACE",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
            },
        ]
        container: dict[str, Any] = {
            "name": "installer",
            "image": manifest.deployment.installer.image,
            "env": [*env, *job_identity_env],
            # The installer's own footprint (a shell/oc orchestration process), not
            # the deployed application's — small requests keep it schedulable and
            # out of the eviction line; generous limits avoid OOM-killing a long
            # install while capping any runaway well below what would harm a node.
            "resources": {
                "requests": {
                    "cpu": self._config.quickstart_installer_cpu_request,
                    "memory": self._config.quickstart_installer_memory_request,
                },
                "limits": {
                    "cpu": self._config.quickstart_installer_cpu_limit,
                    "memory": self._config.quickstart_installer_memory_limit,
                },
            },
        }
        command = manifest.deployment.installer.command
        if action_upper in NAMESPACE_LABELING_ACTIONS:
            # command is guaranteed present (validated in run_action).
            container["command"] = self._installer_with_labeling(command or [])
        elif command:
            container["command"] = command

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": self._config.quickstart_job_ttl_seconds,
                "activeDeadlineSeconds": self._config.quickstart_job_active_deadline_seconds,
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

    @staticmethod
    def _installer_with_labeling(command: list[str]) -> list[str]:
        """Wrap the installer command so a successful run labels the target namespace.

        Runs the installer, then — only if it exits 0 and the namespace exists —
        stamps the managed-by-mcp label so guarded actions (uninstall/upgrade/
        status) recognise the namespace as ours. Doing this inside the Job means
        it does not rely on anyone polling the Job afterwards. The installer's
        exit code is preserved as the container result so the Job still reflects
        success or failure. The namespace is read from $TARGET_NAMESPACE (already
        validated to DNS-1123 characters), so it is safe to reference in the shell.
        """
        installer = " ".join(shlex.quote(part) for part in command)
        label = f"{RHOAILabels.APP_KUBERNETES_MANAGED_BY}={RHOAILabels.MANAGED_BY_VALUE}"
        script = (
            f"{installer}\n"
            "rc=$?\n"
            'if [ "$rc" -eq 0 ] && oc get namespace "$TARGET_NAMESPACE" >/dev/null 2>&1; then\n'
            f'  oc label namespace "$TARGET_NAMESPACE" {label} --overwrite\n'
            "fi\n"
            'exit "$rc"\n'
        )
        return ["/bin/sh", "-c", script]

    def _create_owned_secret(
        self,
        secret_name: str,
        namespace: str,
        secret_data: dict[str, str],
        labels: dict[str, str],
        job_name: str,
        created_job: Any,
    ) -> None:
        """Create the params Secret owned by its Job so it is GC'd with the Job.

        The ownerReference is set at creation time — not patched afterward — so
        the plaintext parameter Secret can never end up unowned and outlive the
        Job it belongs to. If the Secret cannot be created, the already-created
        Job is deleted so it does not linger unable to start.
        """
        uid = self._resource_uid(created_job)
        metadata: dict[str, Any] = {
            "name": secret_name,
            "namespace": namespace,
            "labels": labels,
        }
        if uid:
            metadata["ownerReferences"] = [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": job_name,
                    "uid": uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ]
        body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": metadata,
            "stringData": secret_data,
        }
        try:
            self._k8s.core_v1.create_namespaced_secret(namespace=namespace, body=body)
        except Exception:
            self._delete_job_quietly(job_name, namespace)
            raise

    def _delete_job_quietly(self, job_name: str, namespace: str) -> None:
        """Best-effort delete of a Job whose params Secret could not be created."""
        try:
            self._k8s.get_resource(JOB_CRD).delete(name=job_name, namespace=namespace)
        except ApiException as exc:
            logger.warning("could not clean up job %s: %s", job_name, exc.reason)

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
