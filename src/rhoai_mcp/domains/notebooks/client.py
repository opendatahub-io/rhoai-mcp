"""Notebook (Workbench) client operations."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.notebooks.crds import NotebookCRDs
from rhoai_mcp.domains.notebooks.models import NotebookImage, Workbench, WorkbenchCreate
from rhoai_mcp.utils.annotations import RHOAIAnnotations
from rhoai_mcp.utils.labels import RHOAILabels

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

logger = logging.getLogger(__name__)


class NotebookClient:
    """Client for Notebook (Workbench) operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_workbenches(self, namespace: str) -> list[Workbench]:
        """List all workbenches in a namespace."""
        notebooks = self._k8s.list_resources(NotebookCRDs.NOTEBOOK, namespace=namespace)
        return [
            Workbench.from_notebook_cr(nb, url=self._get_workbench_url(nb.metadata.name, namespace))
            for nb in notebooks
        ]

    def get_workbench(self, name: str, namespace: str) -> Workbench:
        """Get a workbench by name."""
        notebook = self._k8s.get(NotebookCRDs.NOTEBOOK, name=name, namespace=namespace)
        return Workbench.from_notebook_cr(notebook, url=self._get_workbench_url(name, namespace))

    def create_workbench(self, request: WorkbenchCreate) -> Workbench:
        """Create a new workbench."""
        # Build the Notebook CR
        notebook_body = self._build_notebook_cr(request)

        # Create the workbench PVC first
        pvc_name = f"{request.name}-pvc"
        self._ensure_workbench_pvc(pvc_name, request.namespace, request.storage_size)

        # Create the notebook
        notebook = self._k8s.create(
            NotebookCRDs.NOTEBOOK, body=notebook_body, namespace=request.namespace
        )

        return Workbench.from_notebook_cr(
            notebook, url=self._get_workbench_url(request.name, request.namespace)
        )

    def delete_workbench(self, name: str, namespace: str) -> None:
        """Delete a workbench."""
        self._k8s.delete(NotebookCRDs.NOTEBOOK, name=name, namespace=namespace)

    def start_workbench(self, name: str, namespace: str) -> Workbench:
        """Start a stopped workbench by removing the stop annotation."""
        notebook = self._k8s.get(NotebookCRDs.NOTEBOOK, name=name, namespace=namespace)
        annotations = dict(notebook.metadata.annotations or {})

        if RHOAIAnnotations.NOTEBOOK_STOPPED in annotations:
            # Remove the stop annotation
            del annotations[RHOAIAnnotations.NOTEBOOK_STOPPED]

            patch_body = {"metadata": {"annotations": annotations}}
            notebook = self._k8s.patch(
                NotebookCRDs.NOTEBOOK, name=name, body=patch_body, namespace=namespace
            )

        return Workbench.from_notebook_cr(notebook, url=self._get_workbench_url(name, namespace))

    def stop_workbench(self, name: str, namespace: str) -> Workbench:
        """Stop a running workbench by adding the stop annotation."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        patch_body = {
            "metadata": {"annotations": RHOAIAnnotations.notebook_stopped_annotation(timestamp)}
        }

        notebook = self._k8s.patch(
            NotebookCRDs.NOTEBOOK, name=name, body=patch_body, namespace=namespace
        )

        return Workbench.from_notebook_cr(notebook, url=self._get_workbench_url(name, namespace))

    def list_notebook_images(self) -> list[NotebookImage]:
        """List available notebook images.

        Note: This returns a static list of common RHOAI images.
        In a real implementation, this would query the cluster's
        ImageStream resources or a configuration.
        """
        # Common RHOAI notebook images
        return [
            NotebookImage(
                name="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/jupyter-datascience-notebook:2024.1",
                display_name="Jupyter Data Science",
                description="Standard data science notebook with common ML libraries",
                recommended=True,
                order=1,
            ),
            NotebookImage(
                name="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/jupyter-pytorch-notebook:2024.1",
                display_name="PyTorch",
                description="Notebook with PyTorch deep learning framework",
                recommended=False,
                order=2,
            ),
            NotebookImage(
                name="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/jupyter-tensorflow-notebook:2024.1",
                display_name="TensorFlow",
                description="Notebook with TensorFlow deep learning framework",
                recommended=False,
                order=3,
            ),
            NotebookImage(
                name="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/code-server-notebook:2024.1",
                display_name="VS Code (Code Server)",
                description="VS Code-based development environment",
                recommended=False,
                order=4,
            ),
            NotebookImage(
                name="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/rstudio-notebook:2024.1",
                display_name="RStudio",
                description="RStudio development environment",
                recommended=False,
                order=5,
            ),
        ]

    def get_workbench_url(self, name: str, namespace: str) -> str | None:
        """Get the URL for accessing a workbench.

        Returns the OAuth-protected route URL for the workbench.
        """
        return self._get_workbench_url(name, namespace)

    # -------------------------------------------------------------------------
    # Pod and Log Operations
    # -------------------------------------------------------------------------

    def get_workbench_logs(
        self,
        namespace: str,
        name: str,
        container: str | None = None,
        tail_lines: int = 100,
        previous: bool = False,
    ) -> str:
        """Get logs from a workbench's pod.

        Args:
            namespace: The namespace of the workbench.
            name: The name of the workbench.
            container: Container name to get logs from. If None, uses first container.
            tail_lines: Number of lines to return.
            previous: Get logs from previous container instance.

        Returns:
            Log content as string.
        """
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"notebook-name={name}",
        )

        if not pods.items:
            return f"No pods found for workbench '{name}'"

        pod = pods.items[0]
        try:
            kwargs: dict[str, Any] = {
                "name": pod.metadata.name,
                "namespace": namespace,
                "tail_lines": tail_lines,
                "previous": previous,
            }
            if container:
                kwargs["container"] = container
            logs: str = self._k8s.core_v1.read_namespaced_pod_log(**kwargs)
            return logs
        except Exception as e:
            return f"Error getting logs: {e}"

    def get_workbench_events(self, namespace: str, name: str) -> list[dict[str, Any]]:
        """Get Kubernetes events for a workbench and its pods.

        Gathers events for both the Notebook resource itself and any
        associated pods, since pod events contain most failure info.

        Args:
            namespace: The namespace of the workbench.
            name: The name of the workbench.

        Returns:
            List of event dictionaries.
        """
        result: list[dict[str, Any]] = []

        # Events for the Notebook resource itself
        nb_events = self._k8s.core_v1.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={name}",
        )
        for event in nb_events.items:
            result.append(self._format_event(event))

        # Events for associated pods
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"notebook-name={name}",
        )
        for pod in pods.items:
            pod_events = self._k8s.core_v1.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.name={pod.metadata.name}",
            )
            for event in pod_events.items:
                result.append(self._format_event(event))

        return result

    def get_workbench_pods(self, namespace: str, name: str) -> list[dict[str, Any]]:
        """List pods for a workbench.

        Args:
            namespace: The namespace of the workbench.
            name: The name of the workbench.

        Returns:
            List of pod info dictionaries.
        """
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"notebook-name={name}",
        )

        result = []
        for pod in pods.items:
            result.append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "node": getattr(pod.spec, "node_name", None),
                    "ready": self._is_pod_ready(pod),
                }
            )

        return result

    def _is_pod_ready(self, pod: Any) -> bool:
        """Check if a pod is ready."""
        if not pod.status.conditions:
            return False
        for condition in pod.status.conditions:
            if condition.type == "Ready" and condition.status == "True":
                return True
        return False

    def _format_event(self, event: Any) -> dict[str, Any]:
        """Format a Kubernetes event into a dict."""
        return {
            "type": event.type,
            "reason": event.reason,
            "message": event.message,
            "timestamp": str(event.last_timestamp) if event.last_timestamp else None,
            "count": getattr(event, "count", 1),
        }

    def _get_workbench_url(self, name: str, namespace: str) -> str | None:
        """Internal method to construct workbench URL.

        In a real cluster, this would query the Route resource.
        For now, we construct a standard RHOAI URL pattern.
        """
        # Standard RHOAI workbench URL pattern
        # In practice, you'd query the Route CR to get the actual host
        return f"https://{name}-{namespace}.apps.cluster.example.com"

    def _ensure_workbench_pvc(self, name: str, namespace: str, size: str) -> None:
        """Ensure workbench PVC exists, create if not."""
        try:
            self._k8s.get_pvc(name, namespace)
        except Exception:
            # PVC doesn't exist, create it
            self._k8s.create_pvc(
                name=name,
                namespace=namespace,
                size=size,
                labels=RHOAILabels.dashboard_project_labels(),
            )

    def _build_notebook_cr(self, request: WorkbenchCreate) -> dict[str, Any]:
        """Build the Notebook CR body from request."""
        pvc_name = f"{request.name}-pvc"

        # Build annotations
        annotations: dict[str, str] = {}
        if request.inject_oauth:
            annotations.update(RHOAIAnnotations.oauth_annotations())
        if request.display_name:
            annotations["openshift.io/display-name"] = request.display_name
        annotations[RHOAIAnnotations.LAST_SIZE_SELECTION] = request.size

        # Build labels
        labels = RHOAILabels.notebook_labels(request.name)

        # Build resource requirements
        resources: dict[str, dict[str, str]] = {
            "requests": {
                "cpu": request.cpu_request,
                "memory": request.memory_request,
            },
            "limits": {
                "cpu": request.cpu_limit,
                "memory": request.memory_limit,
            },
        }
        if request.gpu_count > 0:
            resources["requests"]["nvidia.com/gpu"] = str(request.gpu_count)
            resources["limits"]["nvidia.com/gpu"] = str(request.gpu_count)

        # Build volumes
        volumes = [
            {
                "name": "workbench-storage",
                "persistentVolumeClaim": {"claimName": pvc_name},
            },
        ]
        volume_mounts = [
            {
                "name": "workbench-storage",
                "mountPath": "/opt/app-root/src",
            },
        ]

        # Add additional PVCs
        for idx, pvc in enumerate(request.additional_pvcs):
            vol_name = f"additional-pvc-{idx}"
            volumes.append(
                {
                    "name": vol_name,
                    "persistentVolumeClaim": {"claimName": pvc},
                }
            )
            volume_mounts.append(
                {
                    "name": vol_name,
                    "mountPath": f"/opt/app-root/data/{pvc}",
                }
            )

        # Build envFrom for data connections
        env_from = []
        for secret_name in request.data_connections:
            env_from.append({"secretRef": {"name": secret_name}})

        return {
            "apiVersion": NotebookCRDs.NOTEBOOK.api_version,
            "kind": NotebookCRDs.NOTEBOOK.kind,
            "metadata": {
                "name": request.name,
                "namespace": request.namespace,
                "annotations": annotations,
                "labels": labels,
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": request.name,
                                "image": request.image,
                                "resources": resources,
                                "volumeMounts": volume_mounts,
                                "envFrom": env_from if env_from else None,
                                "ports": [
                                    {"containerPort": 8888, "name": "notebook-port"},
                                ],
                            },
                        ],
                        "volumes": volumes,
                    },
                },
            },
        }
