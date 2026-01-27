"""MCP Tools for training job discovery."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp_training.client import TrainingClient

if TYPE_CHECKING:
    from rhoai_mcp_core.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register training discovery tools with the MCP server."""

    @mcp.tool()
    def list_training_jobs(namespace: str) -> dict[str, Any]:
        """List all training jobs in a namespace.

        Returns information about all TrainJob resources in the specified
        namespace, including their status and progress.

        Args:
            namespace: The namespace to list training jobs from.

        Returns:
            List of training jobs with their status and metadata.
        """
        client = TrainingClient(server.k8s)
        jobs = client.list_training_jobs(namespace)

        job_list = []
        for job in jobs:
            job_info: dict[str, Any] = {
                "name": job.name,
                "status": job.status.value,
                "model_id": job.model_id,
                "dataset_id": job.dataset_id,
                "num_nodes": job.num_nodes,
                "created": job.creation_timestamp,
            }

            if job.progress:
                job_info["progress"] = {
                    "state": job.progress.state.value,
                    "current_epoch": job.progress.current_epoch,
                    "total_epochs": job.progress.total_epochs,
                    "progress_percent": round(job.progress.progress_percent, 1),
                }

            job_list.append(job_info)

        return {
            "namespace": namespace,
            "count": len(job_list),
            "jobs": job_list,
        }

    @mcp.tool()
    def get_training_job(namespace: str, name: str) -> dict[str, Any]:
        """Get detailed information about a specific training job.

        Returns comprehensive information about a TrainJob including its
        configuration, current status, and training progress.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.

        Returns:
            Detailed training job information.
        """
        client = TrainingClient(server.k8s)
        job = client.get_training_job(namespace, name)

        result: dict[str, Any] = {
            "name": job.name,
            "namespace": job.namespace,
            "status": job.status.value,
            "model_id": job.model_id,
            "dataset_id": job.dataset_id,
            "num_nodes": job.num_nodes,
            "gpus_per_node": job.gpus_per_node,
            "runtime_ref": job.runtime_ref,
            "checkpoint_dir": job.checkpoint_dir,
            "created": job.creation_timestamp,
        }

        if job.progress:
            result["progress"] = {
                "state": job.progress.state.value,
                "current_epoch": job.progress.current_epoch,
                "total_epochs": job.progress.total_epochs,
                "current_step": job.progress.current_step,
                "total_steps": job.progress.total_steps,
                "loss": job.progress.loss,
                "learning_rate": job.progress.learning_rate,
                "throughput": job.progress.throughput,
                "progress_percent": round(job.progress.progress_percent, 1),
                "progress_bar": job.progress.progress_bar(),
                "eta_seconds": job.progress.eta_seconds,
            }

        return result

    @mcp.tool()
    def get_cluster_resources() -> dict[str, Any]:
        """Get cluster-wide compute resources available for training.

        Returns information about CPU, memory, and GPU resources across
        all nodes in the cluster. Useful for planning training jobs and
        understanding cluster capacity.

        Returns:
            Cluster resource summary including CPU, memory, and GPU info.
        """
        client = TrainingClient(server.k8s)
        resources = client.get_cluster_resources()

        result: dict[str, Any] = {
            "cpu_total": resources.cpu_total,
            "cpu_allocatable": resources.cpu_allocatable,
            "memory_total_gb": round(resources.memory_total_gb, 1),
            "memory_allocatable_gb": round(resources.memory_allocatable_gb, 1),
            "node_count": resources.node_count,
            "has_gpus": resources.has_gpus,
        }

        if resources.gpu_info:
            result["gpu_info"] = {
                "type": resources.gpu_info.type,
                "total": resources.gpu_info.total,
                "available": resources.gpu_info.available,
                "nodes_with_gpu": resources.gpu_info.nodes_with_gpu,
            }

        # Include per-node details
        result["nodes"] = [
            {
                "name": node.name,
                "cpu": node.cpu_allocatable,
                "memory_gb": round(node.memory_allocatable_gb, 1),
                "gpus": node.gpu_count,
            }
            for node in resources.nodes
        ]

        return result

    @mcp.tool()
    def list_training_runtimes(namespace: str | None = None) -> dict[str, Any]:
        """List available training runtimes.

        Training runtimes define the container images, frameworks, and
        configurations used for training jobs. This includes both
        cluster-scoped and namespace-scoped runtimes.

        Args:
            namespace: Optional namespace to include namespace-scoped runtimes.

        Returns:
            List of available training runtimes.
        """
        client = TrainingClient(server.k8s)

        # Always get cluster-scoped runtimes
        runtimes = client.list_cluster_training_runtimes()

        # Optionally include namespace-scoped runtimes
        if namespace:
            ns_runtimes = client.list_training_runtimes(namespace)
            runtimes.extend(ns_runtimes)

        runtime_list = []
        for runtime in runtimes:
            runtime_list.append({
                "name": runtime.name,
                "namespace": runtime.namespace,
                "framework": runtime.framework,
                "has_model_initializer": runtime.has_model_initializer,
                "has_dataset_initializer": runtime.has_dataset_initializer,
                "scope": "cluster" if runtime.namespace is None else "namespace",
            })

        return {
            "count": len(runtime_list),
            "runtimes": runtime_list,
        }
