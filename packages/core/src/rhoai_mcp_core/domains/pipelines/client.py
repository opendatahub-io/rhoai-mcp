"""Pipeline (DSPA) client operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp_core.domains.pipelines.crds import PipelinesCRDs
from rhoai_mcp_core.domains.pipelines.models import PipelineServer, PipelineServerCreate

if TYPE_CHECKING:
    from rhoai_mcp_core.clients.base import K8sClient


class PipelineClient:
    """Client for Data Science Pipeline operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def get_pipeline_server(self, namespace: str) -> dict[str, Any] | None:
        """Get the pipeline server (DSPA) in a namespace.

        Returns None if no DSPA exists in the namespace.
        """
        try:
            dspas = self._k8s.list_resources(PipelinesCRDs.DSPA, namespace=namespace)
            if not dspas:
                return None

            # Usually there's only one DSPA per namespace
            dspa = dspas[0]
            server = PipelineServer.from_dspa_cr(dspa)

            return {
                "name": server.metadata.name,
                "status": server.status.value,
                "api_server_ready": server.api_server_ready,
                "persistence_agent_ready": server.persistence_agent_ready,
                "scheduled_workflow_ready": server.scheduled_workflow_ready,
                "database_available": server.database_available,
                "object_store_available": server.object_store_available,
            }
        except Exception:
            return None

    def create_pipeline_server(self, request: PipelineServerCreate) -> PipelineServer:
        """Create a pipeline server (DSPA)."""
        body = self._build_dspa_cr(request)
        dspa = self._k8s.create(PipelinesCRDs.DSPA, body=body, namespace=request.namespace)
        return PipelineServer.from_dspa_cr(dspa)

    def delete_pipeline_server(self, name: str, namespace: str) -> None:
        """Delete a pipeline server (DSPA)."""
        self._k8s.delete(PipelinesCRDs.DSPA, name=name, namespace=namespace)

    def _build_dspa_cr(self, request: PipelineServerCreate) -> dict[str, Any]:
        """Build the DSPA CR body from request."""
        return {
            "apiVersion": PipelinesCRDs.DSPA.api_version,
            "kind": PipelinesCRDs.DSPA.kind,
            "metadata": {
                "name": "dspa",  # Standard name for DSPA
                "namespace": request.namespace,
            },
            "spec": {
                "objectStorage": {
                    "externalStorage": {
                        "bucket": request.object_storage_bucket,
                        "host": request.object_storage_endpoint,
                        "region": request.object_storage_region,
                        "s3CredentialsSecret": {
                            "accessKey": "AWS_ACCESS_KEY_ID",
                            "secretKey": "AWS_SECRET_ACCESS_KEY",
                            "secretName": request.object_storage_secret,
                        },
                    },
                },
                # Use MariaDB for database (simpler setup)
                "database": {
                    "mariaDB": {
                        "deploy": True,
                    },
                },
            },
        }
