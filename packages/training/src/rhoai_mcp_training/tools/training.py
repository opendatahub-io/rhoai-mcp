"""MCP Tools for training job creation and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp_training.client import TrainingClient
from rhoai_mcp_training.models import PeftMethod

if TYPE_CHECKING:
    from rhoai_mcp_core.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register training tools with the MCP server."""

    @mcp.tool()
    def train(
        namespace: str,
        model_id: str,
        dataset_id: str,
        runtime_name: str,
        job_name: str | None = None,
        method: str = "lora",
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_nodes: int = 1,
        gpus_per_node: int = 1,
        checkpoint_dir: str | None = None,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Create a training job for model fine-tuning.

        This is the primary interface for starting a training job. It supports
        various fine-tuning methods including LoRA, QLoRA, and full fine-tuning.

        The operation uses a two-step confirmation workflow:
        1. First call returns a preview of the job configuration
        2. Second call with confirmed=True actually creates the job

        Args:
            namespace: The namespace to create the job in.
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            dataset_id: Dataset identifier (e.g., "tatsu-lab/alpaca").
            runtime_name: Name of the ClusterTrainingRuntime to use.
            job_name: Optional job name (auto-generated if not provided).
            method: Fine-tuning method: "lora", "qlora", "dora", or "full".
            epochs: Number of training epochs (default: 3).
            batch_size: Per-device batch size (default: 32).
            learning_rate: Learning rate (default: 1e-4).
            num_nodes: Number of training nodes (default: 1).
            gpus_per_node: GPUs per node (default: 1).
            checkpoint_dir: Directory for checkpoints (PVC path).
            confirmed: Set to True to create the job.

        Returns:
            Job preview (if not confirmed) or creation confirmation.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        # Validate method
        try:
            peft_method = PeftMethod(method.lower())
        except ValueError:
            return {
                "error": f"Invalid method: {method}",
                "valid_methods": [m.value for m in PeftMethod],
            }

        # Generate job name if not provided
        if not job_name:
            import hashlib
            import time

            suffix = hashlib.md5(f"{model_id}-{time.time()}".encode()).hexdigest()[:8]
            job_name = f"train-{suffix}"

        # Build preview
        preview = {
            "job_name": job_name,
            "namespace": namespace,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "runtime_name": runtime_name,
            "method": peft_method.value,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "checkpoint_dir": checkpoint_dir,
        }

        if not confirmed:
            return {
                "preview": preview,
                "message": (
                    "Review the configuration above. To create the training job, "
                    "call train() again with confirmed=True."
                ),
            }

        # Create the training job
        client = TrainingClient(server.k8s)
        job = client.create_training_job(
            namespace=namespace,
            name=job_name,
            model_id=model_id,
            dataset_id=dataset_id,
            runtime_ref=runtime_name,
            method=peft_method,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )

        return {
            "success": True,
            "job_name": job.name,
            "namespace": job.namespace,
            "status": job.status.value,
            "message": (
                f"Training job '{job.name}' created. "
                "Use get_training_progress() to monitor progress."
            ),
        }

    @mcp.tool()
    def run_container_training_job(
        namespace: str,
        image: str,
        job_name: str | None = None,
        command: list[str] | None = None,
        args: list[str] | None = None,
        num_nodes: int = 1,
        gpus_per_node: int = 1,
        memory_gb: int = 16,
        env_vars: dict[str, str] | None = None,
        volume_mounts: list[dict[str, str]] | None = None,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Run a training job using a custom container image.

        Use this for pre-built training containers or custom training
        scripts that don't fit the standard fine-tuning workflow.

        Args:
            namespace: The namespace to create the job in.
            image: Container image with training code.
            job_name: Optional job name (auto-generated if not provided).
            command: Container command override.
            args: Container arguments.
            num_nodes: Number of training nodes (default: 1).
            gpus_per_node: GPUs per node (default: 1).
            memory_gb: Memory per node in GB (default: 16).
            env_vars: Environment variables to set.
            volume_mounts: Volume mounts (list of {name, mountPath, pvcName}).
            confirmed: Set to True to create the job.

        Returns:
            Job preview (if not confirmed) or creation confirmation.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        # Generate job name if not provided
        if not job_name:
            import hashlib
            import time

            suffix = hashlib.md5(f"{image}-{time.time()}".encode()).hexdigest()[:8]
            job_name = f"container-train-{suffix}"

        preview = {
            "job_name": job_name,
            "namespace": namespace,
            "image": image,
            "command": command,
            "args": args,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "memory_gb": memory_gb,
            "env_vars": env_vars,
            "volume_mounts": volume_mounts,
        }

        if not confirmed:
            return {
                "preview": preview,
                "message": (
                    "Review the configuration above. To create the container training job, "
                    "call run_container_training_job() again with confirmed=True."
                ),
            }

        # Build the job spec for container training
        from rhoai_mcp_training.crds import TrainingCRDs

        spec: dict[str, Any] = {
            "trainer": {
                "numNodes": num_nodes,
                "image": image,
                "resourcesPerNode": {
                    "requests": {
                        "memory": f"{memory_gb}Gi",
                        "nvidia.com/gpu": str(gpus_per_node),
                    },
                    "limits": {
                        "memory": f"{memory_gb}Gi",
                        "nvidia.com/gpu": str(gpus_per_node),
                    },
                },
            },
        }

        if command:
            spec["trainer"]["command"] = command
        if args:
            spec["trainer"]["args"] = args
        if env_vars:
            spec["trainer"]["env"] = [
                {"name": k, "value": v} for k, v in env_vars.items()
            ]

        body = {
            "apiVersion": TrainingCRDs.TRAIN_JOB.api_version,
            "kind": TrainingCRDs.TRAIN_JOB.kind,
            "metadata": {
                "name": job_name,
                "namespace": namespace,
            },
            "spec": spec,
        }

        resource = server.k8s.create(TrainingCRDs.TRAIN_JOB, body=body, namespace=namespace)

        return {
            "success": True,
            "job_name": resource.metadata.name,
            "namespace": namespace,
            "message": f"Container training job '{job_name}' created.",
        }

    @mcp.tool()
    def analyze_training_failure(
        namespace: str,
        job_name: str,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a training failure and suggest remediation.

        Examines job events, logs, and error messages to diagnose
        why a training job failed and provides actionable suggestions.

        Args:
            namespace: The namespace of the training job.
            job_name: The name of the training job.
            error_message: Optional specific error message to analyze.

        Returns:
            Diagnosis and suggested fixes.
        """
        client = TrainingClient(server.k8s)

        # Get job details
        job = client.get_training_job(namespace, job_name)

        # Get events
        events = client.get_job_events(namespace, job_name)

        # Get logs (try to get from failed container)
        logs = ""
        try:
            logs = client.get_training_logs(namespace, job_name, previous=True)
        except Exception:
            try:
                logs = client.get_training_logs(namespace, job_name, previous=False)
            except Exception:
                logs = "Unable to retrieve logs"

        # Analyze
        issues = []
        suggestions = []

        # Check events for issues
        for event in events:
            if event.get("type") == "Warning":
                reason = event.get("reason", "")
                message = event.get("message", "")

                if "OOMKilled" in reason or "OutOfMemory" in message:
                    issues.append("Out of memory - pod was killed")
                    suggestions.append("Reduce batch size or increase memory limits")
                    suggestions.append("Consider using gradient checkpointing")

                if "FailedScheduling" in reason:
                    if "gpu" in message.lower():
                        issues.append("Not enough GPU resources available")
                        suggestions.append("Wait for GPUs or reduce GPU requirements")
                    else:
                        issues.append("Pod scheduling failed")
                        suggestions.append("Check node resources and taints")

                if "ImagePullBackOff" in reason:
                    issues.append("Failed to pull container image")
                    suggestions.append("Verify image exists and pull secret is configured")

        # Analyze logs
        log_lower = logs.lower() if logs else ""

        if "cuda out of memory" in log_lower:
            issues.append("CUDA out of memory during training")
            suggestions.append("Reduce batch size")
            suggestions.append("Use gradient accumulation instead of larger batches")
            suggestions.append("Enable mixed precision training (fp16/bf16)")

        if "nan" in log_lower and "loss" in log_lower:
            issues.append("NaN loss detected")
            suggestions.append("Reduce learning rate")
            suggestions.append("Check dataset for invalid values")
            suggestions.append("Enable gradient clipping")

        if "connection refused" in log_lower or "nccl" in log_lower:
            issues.append("Distributed training communication error")
            suggestions.append("Check network configuration between nodes")
            suggestions.append("Verify NCCL environment variables")

        if error_message:
            issues.append(f"Reported error: {error_message}")

        if not issues:
            issues.append("No specific issues identified")
            suggestions.append("Review the full logs for more details")
            suggestions.append("Check pod events with kubectl describe")

        return {
            "job_name": job_name,
            "namespace": namespace,
            "job_status": job.status.value,
            "issues": issues,
            "suggestions": list(set(suggestions)),  # Deduplicate
            "event_count": len(events),
            "log_preview": logs[:500] if logs else None,
        }
