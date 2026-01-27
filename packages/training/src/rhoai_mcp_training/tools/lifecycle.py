"""MCP Tools for training job lifecycle management."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp_training.client import TrainingClient
from rhoai_mcp_training.models import TrainJobStatus

if TYPE_CHECKING:
    from rhoai_mcp_core.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register training lifecycle tools with the MCP server."""

    @mcp.tool()
    def suspend_training_job(namespace: str, name: str) -> dict[str, Any]:
        """Suspend a running training job.

        Stops the training pods gracefully while preserving checkpoints
        and state. The job can be resumed later with resume_training_job.
        This frees up GPU and other compute resources.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.

        Returns:
            Confirmation of suspension.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("update")
        if not allowed:
            return {"error": reason}

        client = TrainingClient(server.k8s)
        client.suspend_training_job(namespace, name)

        return {
            "success": True,
            "job_name": name,
            "namespace": namespace,
            "message": f"Training job '{name}' has been suspended. Use resume_training_job to continue.",
        }

    @mcp.tool()
    def resume_training_job(namespace: str, name: str) -> dict[str, Any]:
        """Resume a suspended training job.

        Restarts the training pods for a previously suspended job.
        Training will continue from the last checkpoint if checkpointing
        was configured.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.

        Returns:
            Confirmation of resumption.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("update")
        if not allowed:
            return {"error": reason}

        client = TrainingClient(server.k8s)
        client.resume_training_job(namespace, name)

        return {
            "success": True,
            "job_name": name,
            "namespace": namespace,
            "message": f"Training job '{name}' has been resumed.",
        }

    @mcp.tool()
    def delete_training_job(
        namespace: str, name: str, confirm: bool = False
    ) -> dict[str, Any]:
        """Delete a training job.

        Permanently removes the training job and associated resources.
        This action cannot be undone. Checkpoints stored on PVCs will
        not be deleted.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.
            confirm: Must be True to actually delete.

        Returns:
            Confirmation of deletion.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("delete")
        if not allowed:
            return {"error": reason}

        if not confirm:
            return {
                "error": "Deletion not confirmed",
                "message": (
                    f"To delete training job '{name}', set confirm=True. "
                    "WARNING: This action cannot be undone."
                ),
            }

        client = TrainingClient(server.k8s)
        client.delete_training_job(namespace, name)

        return {
            "success": True,
            "deleted": True,
            "job_name": name,
            "namespace": namespace,
            "message": f"Training job '{name}' has been deleted.",
        }

    @mcp.tool()
    def wait_for_job_completion(
        namespace: str,
        name: str,
        target_status: str = "Completed",
        timeout_seconds: int = 3600,
        poll_interval: int = 10,
    ) -> dict[str, Any]:
        """Wait for a training job to reach a target status.

        Blocks until the job reaches the specified status or timeout.
        Useful in pipelines where you need to wait for training to
        complete before proceeding.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.
            target_status: Status to wait for (default: "Completed").
            timeout_seconds: Maximum time to wait (default: 3600 = 1 hour).
            poll_interval: Seconds between status checks (default: 10).

        Returns:
            Final status and whether target was reached.
        """
        client = TrainingClient(server.k8s)
        start_time = time.time()

        # Map target status to enum
        target_statuses = {"Completed", "Failed"}  # Terminal states
        if target_status:
            target_statuses = {target_status}

        while True:
            job = client.get_training_job(namespace, name)
            current_status = job.status.value

            # Check if we reached target
            if current_status in target_statuses:
                return {
                    "success": True,
                    "job_name": name,
                    "final_status": current_status,
                    "elapsed_seconds": int(time.time() - start_time),
                }

            # Check if we hit a terminal failure state
            if current_status == TrainJobStatus.FAILED.value:
                return {
                    "success": False,
                    "job_name": name,
                    "final_status": current_status,
                    "elapsed_seconds": int(time.time() - start_time),
                    "message": "Job failed before reaching target status.",
                }

            # Check timeout
            if time.time() - start_time > timeout_seconds:
                return {
                    "success": False,
                    "job_name": name,
                    "final_status": current_status,
                    "elapsed_seconds": int(time.time() - start_time),
                    "message": f"Timeout waiting for job to reach '{target_status}'.",
                }

            # Wait before next poll
            time.sleep(poll_interval)

    @mcp.tool()
    def get_job_spec(namespace: str, name: str) -> dict[str, Any]:
        """Get the complete specification of a training job.

        Returns the full YAML/JSON specification of a TrainJob resource.
        Useful for debugging configuration issues or understanding
        how a job was configured.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.

        Returns:
            Complete job specification.
        """
        from rhoai_mcp_training.crds import TrainingCRDs

        resource = server.k8s.get(TrainingCRDs.TRAIN_JOB, name, namespace=namespace)

        # Convert to dict, handling the dynamic resource format
        spec = dict(resource.spec) if hasattr(resource.spec, "items") else resource.spec
        metadata = {
            "name": resource.metadata.name,
            "namespace": resource.metadata.namespace,
            "uid": resource.metadata.uid,
            "creationTimestamp": str(resource.metadata.creation_timestamp),
            "labels": dict(resource.metadata.labels) if resource.metadata.labels else {},
            "annotations": dict(resource.metadata.annotations) if resource.metadata.annotations else {},
        }

        return {
            "apiVersion": TrainingCRDs.TRAIN_JOB.api_version,
            "kind": TrainingCRDs.TRAIN_JOB.kind,
            "metadata": metadata,
            "spec": spec,
        }
