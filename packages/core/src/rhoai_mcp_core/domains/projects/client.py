"""Project (namespace) client operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp_core.domains.projects.models import DataScienceProject, ProjectCreate
from rhoai_mcp_core.models.common import ResourceSummary
from rhoai_mcp_core.utils.annotations import RHOAIAnnotations
from rhoai_mcp_core.utils.labels import RHOAILabels

if TYPE_CHECKING:
    from rhoai_mcp_core.clients.base import K8sClient


class ProjectClient:
    """Client for Data Science Project operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_projects(self) -> list[DataScienceProject]:
        """List all Data Science Projects.

        Uses the OpenShift Projects API to list only projects the user has
        access to, then filters for those with the opendatahub.io/dashboard=true
        label indicating they are RHOAI Data Science Projects.
        """
        label_selector = RHOAILabels.filter_selector(**{RHOAILabels.DASHBOARD: "true"})
        # Use OpenShift Projects API which returns only user-accessible projects
        # This avoids requiring cluster-wide namespace list permissions
        projects = self._k8s.list_projects(label_selector=label_selector)
        return [DataScienceProject.from_project(p) for p in projects]

    def get_project(self, name: str, include_summary: bool = False) -> DataScienceProject:
        """Get a Data Science Project by name.

        Args:
            name: Project (namespace) name
            include_summary: Whether to include resource counts
        """
        namespace = self._k8s.get_namespace(name)

        # Verify it's a DS project
        labels = namespace.metadata.labels or {}
        if not RHOAILabels.is_dashboard_project(labels):
            from rhoai_mcp_core.utils.errors import NotFoundError

            raise NotFoundError("DataScienceProject", name)

        summary = None
        if include_summary:
            summary = self._get_resource_summary(name)

        return DataScienceProject.from_namespace(namespace, summary)

    def create_project(self, request: ProjectCreate) -> DataScienceProject:
        """Create a new Data Science Project."""
        # Build labels
        labels = RHOAILabels.dashboard_project_labels()
        if request.enable_modelmesh:
            labels.update(RHOAILabels.model_serving_labels(single_model=False))
        else:
            labels.update(RHOAILabels.model_serving_labels(single_model=True))

        # Build annotations
        annotations: dict[str, str] = {}
        if request.display_name:
            annotations[RHOAIAnnotations.DASHBOARD_DISPLAY_NAME] = request.display_name
        if request.description:
            annotations[RHOAIAnnotations.DASHBOARD_DESCRIPTION] = request.description

        namespace = self._k8s.create_namespace(
            name=request.name,
            labels=labels,
            annotations=annotations if annotations else None,
        )

        return DataScienceProject.from_namespace(namespace)

    def delete_project(self, name: str) -> None:
        """Delete a Data Science Project.

        This deletes the entire namespace and all resources within it.
        """
        # Verify it's a DS project first
        self.get_project(name)
        self._k8s.delete_namespace(name)

    def set_model_serving_mode(self, name: str, enable_modelmesh: bool) -> DataScienceProject:
        """Set the model serving mode for a project.

        Args:
            name: Project name
            enable_modelmesh: True for multi-model (ModelMesh), False for single-model (KServe)
        """
        # Verify it's a DS project
        self.get_project(name)

        labels = RHOAILabels.model_serving_labels(single_model=not enable_modelmesh)
        # Use OpenShift Projects API which regular users can modify,
        # unlike the Namespace API which requires cluster-admin permissions
        project = self._k8s.patch_project(name, labels=labels)

        return DataScienceProject.from_project(project)

    def update_project(
        self,
        name: str,
        display_name: str | None = None,
        description: str | None = None,
    ) -> DataScienceProject:
        """Update project display name and/or description."""
        # Verify it's a DS project
        self.get_project(name)

        annotations: dict[str, str] = {}
        if display_name is not None:
            annotations[RHOAIAnnotations.DASHBOARD_DISPLAY_NAME] = display_name
        if description is not None:
            annotations[RHOAIAnnotations.DASHBOARD_DESCRIPTION] = description

        if annotations:
            # Use OpenShift Projects API which regular users can modify,
            # unlike the Namespace API which requires cluster-admin permissions
            project = self._k8s.patch_project(name, annotations=annotations)
            return DataScienceProject.from_project(project)

        return self.get_project(name)

    def _get_resource_summary(self, namespace: str) -> ResourceSummary:
        """Get resource counts for a namespace."""
        # Import here to avoid circular imports when other domains not available
        workbenches = 0
        workbenches_running = 0
        models = 0
        models_ready = 0
        pipelines = 0
        data_connections = 0
        storage = 0

        # Count workbenches (use domain CRDs)
        try:
            from rhoai_mcp_core.domains.notebooks.crds import NotebookCRDs

            notebooks = self._k8s.list_resources(NotebookCRDs.NOTEBOOK, namespace=namespace)
            workbenches = len(notebooks)
            workbenches_running = sum(
                1
                for nb in notebooks
                if not RHOAIAnnotations.is_notebook_stopped(nb.metadata.annotations or {})
            )
        except Exception:
            pass

        # Count models (use domain CRDs)
        try:
            from rhoai_mcp_core.domains.inference.crds import InferenceCRDs

            isvc = self._k8s.list_resources(InferenceCRDs.INFERENCE_SERVICE, namespace=namespace)
            models = len(isvc)
            models_ready = sum(1 for svc in isvc if self._is_inference_service_ready(svc))
        except Exception:
            pass

        # Count data connections
        try:
            label_selector = RHOAILabels.filter_selector(**{RHOAILabels.DASHBOARD: "true"})
            secrets = self._k8s.list_secrets(namespace=namespace, label_selector=label_selector)
            # Filter to only data connections
            for secret in secrets:
                annotations = secret.metadata.annotations or {}
                if RHOAIAnnotations.CONNECTION_TYPE in annotations:
                    data_connections += 1
        except Exception:
            pass

        # Count PVCs
        try:
            pvcs = self._k8s.list_pvcs(namespace=namespace)
            storage = len(pvcs)
        except Exception:
            pass

        # Count pipelines (check if DSPA exists)
        try:
            from rhoai_mcp_core.domains.pipelines.crds import PipelinesCRDs

            dspas = self._k8s.list_resources(PipelinesCRDs.DSPA, namespace=namespace)
            pipelines = 1 if dspas else 0
        except Exception:
            pass

        return ResourceSummary(
            workbenches=workbenches,
            workbenches_running=workbenches_running,
            models=models,
            models_ready=models_ready,
            pipelines=pipelines,
            data_connections=data_connections,
            storage=storage,
        )

    @staticmethod
    def _is_inference_service_ready(isvc: Any) -> bool:
        """Check if an InferenceService is ready."""
        status = getattr(isvc, "status", None)
        if not status:
            return False

        conditions = getattr(status, "conditions", []) or []
        for cond in conditions:
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                return True
        return False
