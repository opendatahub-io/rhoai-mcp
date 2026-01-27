"""MCP Tools for training job monitoring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp_training.client import TrainingClient

if TYPE_CHECKING:
    from rhoai_mcp_core.server import RHOAIServer


# Checkpoint annotation key
CHECKPOINT_ANNOTATION = "trainer.opendatahub.io/checkpoint"


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register training monitoring tools with the MCP server."""

    @mcp.tool()
    def get_training_progress(namespace: str, name: str) -> dict[str, Any]:
        """Get real-time training progress for a job.

        Returns detailed progress information parsed from the trainer status
        annotation, including current epoch, step, loss, learning rate,
        throughput, and estimated time remaining.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.

        Returns:
            Training progress information with metrics.
        """
        client = TrainingClient(server.k8s)
        job = client.get_training_job(namespace, name)

        if not job.progress:
            return {
                "job_name": name,
                "state": "Unknown",
                "message": "No training progress information available. Job may not have started.",
            }

        progress = job.progress
        return {
            "job_name": name,
            "state": progress.state.value,
            "current_epoch": progress.current_epoch,
            "total_epochs": progress.total_epochs,
            "current_step": progress.current_step,
            "total_steps": progress.total_steps,
            "loss": progress.loss,
            "learning_rate": progress.learning_rate,
            "throughput": progress.throughput,
            "gradient_norm": progress.gradient_norm,
            "progress_percent": round(progress.progress_percent, 1),
            "progress_bar": progress.progress_bar(),
            "eta_seconds": progress.eta_seconds,
        }

    @mcp.tool()
    def get_training_logs(
        namespace: str,
        name: str,
        container: str = "trainer",
        tail_lines: int = 100,
        previous: bool = False,
    ) -> dict[str, Any]:
        """Get logs from a training job.

        Fetches container logs from the training job's pods. Supports
        getting logs from different containers and from previous
        container instances (useful for debugging crashes).

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.
            container: Container name (default: "trainer").
            tail_lines: Number of lines to return (default: 100).
            previous: Get logs from previous container instance.

        Returns:
            Training logs and any detected issues.
        """
        client = TrainingClient(server.k8s)
        logs = client.get_training_logs(
            namespace, name, container=container, tail_lines=tail_lines, previous=previous
        )

        # Analyze logs for common issues
        issues = _analyze_logs(logs)

        return {
            "job_name": name,
            "container": container,
            "logs": logs,
            "issues": issues if issues else None,
            "lines_returned": len(logs.split("\n")) if logs else 0,
        }

    @mcp.tool()
    def get_job_events(namespace: str, name: str) -> dict[str, Any]:
        """Get Kubernetes events for a training job.

        Returns events related to the training job, grouped by type.
        Useful for diagnosing scheduling issues, OOM conditions,
        image pull failures, and other infrastructure problems.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.

        Returns:
            List of events with issue detection and suggestions.
        """
        client = TrainingClient(server.k8s)
        events = client.get_job_events(namespace, name)

        # Group events by type
        warnings = [e for e in events if e.get("type") == "Warning"]

        # Analyze for common issues
        issues = []
        suggestions = []

        for event in warnings:
            reason = event.get("reason", "")
            message = event.get("message", "")

            if "OOMKilled" in reason or "OutOfMemory" in message:
                issues.append("Out of memory - pod was killed due to memory limits")
                suggestions.append(
                    "Increase memory limits or reduce batch size"
                )

            if "FailedScheduling" in reason:
                if "gpu" in message.lower():
                    issues.append("Insufficient GPU resources available")
                    suggestions.append(
                        "Wait for GPUs to become available or reduce GPU requirements"
                    )
                else:
                    issues.append("Pod scheduling failed")
                    suggestions.append("Check node resources and taints")

            if "ImagePullBackOff" in reason or "ErrImagePull" in reason:
                issues.append("Failed to pull container image")
                suggestions.append(
                    "Verify image exists and credentials are configured"
                )

        return {
            "job_name": name,
            "total_events": len(events),
            "events": events,
            "warnings": warnings,
            "has_warnings": len(warnings) > 0,
            "issues": issues if issues else None,
            "suggestions": suggestions if suggestions else None,
        }

    @mcp.tool()
    def manage_checkpoints(
        namespace: str,
        job_name: str,
    ) -> dict[str, Any]:
        """Get checkpoint information for a training job.

        Returns information about saved checkpoints including the latest
        checkpoint path and a list of all available checkpoints.

        Args:
            namespace: The namespace of the training job.
            job_name: The name of the training job.

        Returns:
            Checkpoint information including paths and steps.
        """
        client = TrainingClient(server.k8s)

        # Get checkpoint info from annotation
        resource = server.k8s.get(
            client._k8s.get_resource(
                __import__("rhoai_mcp_training.crds", fromlist=["TrainingCRDs"]).TrainingCRDs.TRAIN_JOB
            ),
            job_name,
            namespace=namespace,
        )

        annotations = getattr(resource.metadata, "annotations", {}) or {}
        checkpoint_annotation = annotations.get(CHECKPOINT_ANNOTATION, "")

        if not checkpoint_annotation:
            return {
                "job_name": job_name,
                "message": "No checkpoint information available. Checkpointing may not be configured.",
                "latest": None,
                "checkpoints": [],
            }

        try:
            checkpoint_data = json.loads(checkpoint_annotation)
        except json.JSONDecodeError:
            return {
                "job_name": job_name,
                "error": "Failed to parse checkpoint annotation",
                "latest": None,
                "checkpoints": [],
            }

        return {
            "job_name": job_name,
            "latest": checkpoint_data.get("latest"),
            "checkpoints": checkpoint_data.get("checkpoints", []),
        }


def _analyze_logs(logs: str) -> list[str]:
    """Analyze training logs for common issues."""
    issues: list[str] = []

    if not logs:
        return issues

    log_lower = logs.lower()

    # CUDA/GPU issues
    if "cuda out of memory" in log_lower or "oom" in log_lower:
        issues.append(
            "CUDA out of memory detected. Consider reducing batch size or using gradient checkpointing."
        )

    # NaN/Inf loss
    if "nan" in log_lower and "loss" in log_lower:
        issues.append(
            "NaN loss detected. This may indicate learning rate is too high or data issues."
        )

    if "inf" in log_lower and ("loss" in log_lower or "gradient" in log_lower):
        issues.append(
            "Infinite values detected in training. Check learning rate and data preprocessing."
        )

    # Gradient issues
    if "gradient overflow" in log_lower or "gradient underflow" in log_lower:
        issues.append(
            "Gradient overflow/underflow detected. Consider using gradient clipping."
        )

    # Connection issues
    if "connection refused" in log_lower or "connection reset" in log_lower:
        issues.append("Network connection issues detected. Check distributed training setup.")

    # Import errors
    if "modulenotfounderror" in log_lower or "importerror" in log_lower:
        issues.append(
            "Missing Python module. Verify runtime image has all required packages."
        )

    return issues
