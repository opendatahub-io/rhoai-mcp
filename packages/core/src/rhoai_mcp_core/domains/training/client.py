"""Training client operations wrapping K8sClient."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rhoai_mcp_core.domains.training.crds import TrainingCRDs
from rhoai_mcp_core.domains.training.models import (
    ClusterResources,
    GPUInfo,
    NodeResources,
    PeftMethod,
    TrainingRuntime,
    TrainJob,
)

if TYPE_CHECKING:
    from rhoai_mcp_core.clients.base import K8sClient


class TrainingClient:
    """Client for Kubeflow Training Operator operations."""

    def __init__(self, k8s: K8sClient) -> None:
        """Initialize with a K8sClient instance."""
        self._k8s = k8s

    # -------------------------------------------------------------------------
    # TrainJob Operations
    # -------------------------------------------------------------------------

    def list_training_jobs(self, namespace: str) -> list[TrainJob]:
        """List all training jobs in a namespace.

        Args:
            namespace: The namespace to list jobs from.

        Returns:
            List of TrainJob models.
        """
        resources = self._k8s.list_resources(TrainingCRDs.TRAIN_JOB, namespace=namespace)
        return [TrainJob.from_resource(r) for r in resources]

    def get_training_job(self, namespace: str, name: str) -> TrainJob:
        """Get a specific training job.

        Args:
            namespace: The namespace of the job.
            name: The name of the job.

        Returns:
            TrainJob model.
        """
        resource = self._k8s.get(TrainingCRDs.TRAIN_JOB, name, namespace=namespace)
        return TrainJob.from_resource(resource)

    def create_training_job(
        self,
        namespace: str,
        name: str,
        model_id: str,
        dataset_id: str,
        runtime_ref: str,
        method: PeftMethod = PeftMethod.LORA,
        num_nodes: int = 1,
        gpus_per_node: int = 1,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        checkpoint_dir: str | None = None,
        tolerations: list[dict[str, Any]] | None = None,
        node_selector: dict[str, str] | None = None,
    ) -> TrainJob:
        """Create a new training job.

        Args:
            namespace: The namespace to create the job in.
            name: The name of the job.
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            dataset_id: Dataset identifier (e.g., "tatsu-lab/alpaca").
            runtime_ref: Name of the ClusterTrainingRuntime to use.
            method: Fine-tuning method (lora, qlora, full, dora).
            num_nodes: Number of training nodes.
            gpus_per_node: Number of GPUs per node.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            checkpoint_dir: Directory for checkpoints (PVC path).
            tolerations: Pod tolerations for scheduling.
            node_selector: Node selector labels.

        Returns:
            Created TrainJob model.
        """
        body = self._build_train_job_spec(
            namespace=namespace,
            name=name,
            model_id=model_id,
            dataset_id=dataset_id,
            runtime_ref=runtime_ref,
            method=method,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            tolerations=tolerations,
            node_selector=node_selector,
        )
        resource = self._k8s.create(TrainingCRDs.TRAIN_JOB, body=body, namespace=namespace)
        return TrainJob.from_resource(resource)

    def delete_training_job(self, namespace: str, name: str) -> None:
        """Delete a training job.

        Args:
            namespace: The namespace of the job.
            name: The name of the job.
        """
        self._k8s.delete(TrainingCRDs.TRAIN_JOB, name, namespace=namespace)

    def suspend_training_job(self, namespace: str, name: str) -> None:
        """Suspend a training job.

        This stops the training pods while preserving state.

        Args:
            namespace: The namespace of the job.
            name: The name of the job.
        """
        body = {"spec": {"suspend": True}}
        self._k8s.patch(TrainingCRDs.TRAIN_JOB, name, body=body, namespace=namespace)

    def resume_training_job(self, namespace: str, name: str) -> None:
        """Resume a suspended training job.

        Args:
            namespace: The namespace of the job.
            name: The name of the job.
        """
        body = {"spec": {"suspend": False}}
        self._k8s.patch(TrainingCRDs.TRAIN_JOB, name, body=body, namespace=namespace)

    # -------------------------------------------------------------------------
    # Training Runtime Operations
    # -------------------------------------------------------------------------

    def list_cluster_training_runtimes(self) -> list[TrainingRuntime]:
        """List all cluster training runtimes.

        Returns:
            List of TrainingRuntime models.
        """
        resources = self._k8s.list_resources(TrainingCRDs.CLUSTER_TRAINING_RUNTIME)
        return [TrainingRuntime.from_resource(r, is_cluster_scoped=True) for r in resources]

    def list_training_runtimes(self, namespace: str) -> list[TrainingRuntime]:
        """List namespace-scoped training runtimes.

        Args:
            namespace: The namespace to list runtimes from.

        Returns:
            List of TrainingRuntime models.
        """
        resources = self._k8s.list_resources(TrainingCRDs.TRAINING_RUNTIME, namespace=namespace)
        return [TrainingRuntime.from_resource(r, is_cluster_scoped=False) for r in resources]

    def get_cluster_training_runtime(self, name: str) -> TrainingRuntime:
        """Get a specific cluster training runtime.

        Args:
            name: The name of the runtime.

        Returns:
            TrainingRuntime model.
        """
        resource = self._k8s.get(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, name)
        return TrainingRuntime.from_resource(resource, is_cluster_scoped=True)

    def create_cluster_training_runtime(
        self,
        name: str,
        spec: dict[str, Any],
    ) -> TrainingRuntime:
        """Create a cluster training runtime.

        Args:
            name: The name of the runtime.
            spec: The runtime specification.

        Returns:
            Created TrainingRuntime model.
        """
        body = {
            "apiVersion": TrainingCRDs.CLUSTER_TRAINING_RUNTIME.api_version,
            "kind": TrainingCRDs.CLUSTER_TRAINING_RUNTIME.kind,
            "metadata": {"name": name},
            "spec": spec,
        }
        resource = self._k8s.create(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, body=body)
        return TrainingRuntime.from_resource(resource, is_cluster_scoped=True)

    # -------------------------------------------------------------------------
    # Pod and Log Operations
    # -------------------------------------------------------------------------

    def get_training_logs(
        self,
        namespace: str,
        job_name: str,
        container: str = "trainer",
        tail_lines: int = 100,
        previous: bool = False,
    ) -> str:
        """Get logs from a training job's pods.

        Args:
            namespace: The namespace of the job.
            job_name: The name of the job.
            container: Container name to get logs from.
            tail_lines: Number of lines to return.
            previous: Get logs from previous container instance.

        Returns:
            Log content as string.
        """
        # Find pods for this job
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"training.kubeflow.org/job-name={job_name}",
        )

        if not pods.items:
            return f"No pods found for job '{job_name}'"

        # Get logs from the first running or completed pod
        pod = pods.items[0]
        try:
            logs: str = self._k8s.core_v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
                container=container,
                tail_lines=tail_lines,
                previous=previous,
            )
            return logs
        except Exception as e:
            return f"Error getting logs: {e}"

    def get_job_events(self, namespace: str, job_name: str) -> list[dict[str, Any]]:
        """Get Kubernetes events for a training job.

        Args:
            namespace: The namespace of the job.
            job_name: The name of the job.

        Returns:
            List of event dictionaries.
        """
        events = self._k8s.core_v1.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={job_name}",
        )

        result = []
        for event in events.items:
            result.append(
                {
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "timestamp": str(event.last_timestamp) if event.last_timestamp else None,
                    "count": getattr(event, "count", 1),
                }
            )

        return result

    def list_training_job_pods(self, namespace: str, job_name: str) -> list[dict[str, Any]]:
        """List pods for a training job.

        Args:
            namespace: The namespace of the job.
            job_name: The name of the job.

        Returns:
            List of pod info dictionaries.
        """
        pods = self._k8s.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"training.kubeflow.org/job-name={job_name}",
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

    # -------------------------------------------------------------------------
    # Cluster Resource Discovery
    # -------------------------------------------------------------------------

    def get_cluster_resources(self) -> ClusterResources:
        """Get cluster-wide resource information.

        Returns:
            ClusterResources with CPU, memory, and GPU info.
        """
        nodes = self._k8s.core_v1.list_node()

        total_cpu = 0
        allocatable_cpu = 0
        total_memory_gb = 0.0
        allocatable_memory_gb = 0.0
        gpu_total = 0
        gpu_nodes = 0
        gpu_type = None
        node_resources = []

        for node in nodes.items:
            capacity = node.status.capacity or {}
            allocatable = node.status.allocatable or {}

            cpu = _parse_cpu(capacity.get("cpu", "0"))
            cpu_alloc = _parse_cpu(allocatable.get("cpu", "0"))
            memory = _parse_memory_gb(capacity.get("memory", "0"))
            memory_alloc = _parse_memory_gb(allocatable.get("memory", "0"))

            total_cpu += cpu
            allocatable_cpu += cpu_alloc
            total_memory_gb += memory
            allocatable_memory_gb += memory_alloc

            # Check for GPUs
            node_gpus = 0
            for key in ["nvidia.com/gpu", "amd.com/gpu"]:
                if key in capacity:
                    node_gpus = int(capacity[key])
                    if node_gpus > 0:
                        gpu_type = key
                        gpu_total += node_gpus
                        gpu_nodes += 1
                        break

            node_resources.append(
                NodeResources(
                    name=node.metadata.name,
                    cpu_total=cpu,
                    cpu_allocatable=cpu_alloc,
                    memory_total_gb=memory,
                    memory_allocatable_gb=memory_alloc,
                    gpu_count=node_gpus,
                    gpu_type=gpu_type if node_gpus > 0 else None,
                )
            )

        gpu_info = None
        if gpu_total > 0 and gpu_type:
            gpu_info = GPUInfo(
                type=gpu_type,
                total=gpu_total,
                available=gpu_total,  # Simplified; real impl would check allocations
                nodes_with_gpu=gpu_nodes,
            )

        return ClusterResources(
            cpu_total=total_cpu,
            cpu_allocatable=allocatable_cpu,
            memory_total_gb=total_memory_gb,
            memory_allocatable_gb=allocatable_memory_gb,
            gpu_info=gpu_info,
            node_count=len(nodes.items),
            nodes=node_resources,
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_train_job_spec(
        self,
        namespace: str,
        name: str,
        model_id: str,
        dataset_id: str,
        runtime_ref: str,
        method: PeftMethod = PeftMethod.LORA,
        num_nodes: int = 1,
        gpus_per_node: int = 1,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        checkpoint_dir: str | None = None,
        tolerations: list[dict[str, Any]] | None = None,
        node_selector: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build a TrainJob CR specification."""
        spec: dict[str, Any] = {
            "modelConfig": {
                "name": model_id,
            },
            "datasetConfig": {
                "name": dataset_id,
            },
            "trainer": {
                "numNodes": num_nodes,
                "resourcesPerNode": {
                    "requests": {
                        "nvidia.com/gpu": str(gpus_per_node),
                    },
                },
            },
            "runtimeRef": {
                "name": runtime_ref,
            },
        }

        # Add training parameters
        training_args: dict[str, Any] = {
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": learning_rate,
        }

        # Add PEFT method configuration
        if method != PeftMethod.FULL:
            training_args["peft_method"] = method.value

        spec["trainer"]["trainingArgs"] = training_args

        # Add checkpoint directory if specified
        if checkpoint_dir:
            spec["trainer"]["trainingArgs"]["output_dir"] = checkpoint_dir

        # Add tolerations if specified
        if tolerations:
            spec["trainer"]["tolerations"] = tolerations

        # Add node selector if specified
        if node_selector:
            spec["trainer"]["nodeSelector"] = node_selector

        return {
            "apiVersion": TrainingCRDs.TRAIN_JOB.api_version,
            "kind": TrainingCRDs.TRAIN_JOB.kind,
            "metadata": {
                "name": name,
                "namespace": namespace,
            },
            "spec": spec,
        }


def _parse_cpu(value: str) -> int:
    """Parse CPU value to cores."""
    if not value:
        return 0
    value = str(value)
    if value.endswith("m"):
        return int(value[:-1]) // 1000
    return int(value)


def _parse_memory_gb(value: str) -> float:
    """Parse memory value to GB."""
    if not value:
        return 0.0
    value = str(value)
    multipliers = {
        "Ki": 1 / (1024 * 1024),
        "Mi": 1 / 1024,
        "Gi": 1,
        "Ti": 1024,
        "K": 1 / (1000 * 1000),
        "M": 1 / 1000,
        "G": 1,
        "T": 1000,
    }
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * mult
    return float(value) / (1024 * 1024 * 1024)  # Assume bytes
