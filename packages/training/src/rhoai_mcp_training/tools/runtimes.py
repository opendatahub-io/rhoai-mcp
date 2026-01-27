"""MCP Tools for training runtime management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp_training.client import TrainingClient
from rhoai_mcp_training.crds import TrainingCRDs

if TYPE_CHECKING:
    from rhoai_mcp_core.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register training runtime tools with the MCP server."""

    @mcp.tool()
    def get_runtime_details(runtime_name: str) -> dict[str, Any]:
        """Get detailed information about a training runtime.

        Returns the complete configuration of a ClusterTrainingRuntime
        including initializers, container images, and framework settings.

        Args:
            runtime_name: Name of the ClusterTrainingRuntime.

        Returns:
            Runtime configuration details.
        """
        client = TrainingClient(server.k8s)
        runtime = client.get_cluster_training_runtime(runtime_name)

        # Get the full resource for additional details
        resource = server.k8s.get(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, runtime_name)
        spec = dict(resource.spec) if hasattr(resource.spec, "items") else resource.spec or {}

        # Parse template details
        template = spec.get("template", {})
        template_spec = template.get("spec", {})

        # Extract initializers
        initializers = []
        for init in template_spec.get("initializers", []):
            initializers.append({
                "type": init.get("type"),
                "image": init.get("image"),
                "config": init.get("config", {}),
            })

        # Extract trainer config
        trainer = template_spec.get("trainer", {})

        return {
            "name": runtime.name,
            "framework": runtime.framework,
            "has_model_initializer": runtime.has_model_initializer,
            "has_dataset_initializer": runtime.has_dataset_initializer,
            "trainer_image": trainer.get("image"),
            "initializers": initializers,
            "labels": dict(resource.metadata.labels) if resource.metadata.labels else {},
        }

    @mcp.tool()
    def create_runtime(
        name: str,
        trainer_image: str,
        framework: str = "transformers",
        model_initializer_image: str | None = None,
        dataset_initializer_image: str | None = None,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Create a new ClusterTrainingRuntime.

        Training runtimes define the container images and configuration
        used for training jobs. This creates a cluster-scoped runtime
        available to all namespaces.

        Args:
            name: Name for the new runtime.
            trainer_image: Container image for the trainer.
            framework: Training framework (default: "transformers").
            model_initializer_image: Optional image for model initialization.
            dataset_initializer_image: Optional image for dataset initialization.
            confirmed: Set to True to create the runtime.

        Returns:
            Runtime preview (if not confirmed) or creation confirmation.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        preview = {
            "name": name,
            "framework": framework,
            "trainer_image": trainer_image,
            "model_initializer_image": model_initializer_image,
            "dataset_initializer_image": dataset_initializer_image,
        }

        if not confirmed:
            return {
                "preview": preview,
                "message": (
                    "Review the configuration above. To create the runtime, "
                    "call create_runtime() again with confirmed=True."
                ),
            }

        # Build runtime spec
        spec = _build_runtime_spec(
            trainer_image=trainer_image,
            framework=framework,
            model_initializer_image=model_initializer_image,
            dataset_initializer_image=dataset_initializer_image,
        )

        body = {
            "apiVersion": TrainingCRDs.CLUSTER_TRAINING_RUNTIME.api_version,
            "kind": TrainingCRDs.CLUSTER_TRAINING_RUNTIME.kind,
            "metadata": {
                "name": name,
                "labels": {
                    "training.kubeflow.org/framework": framework,
                },
            },
            "spec": spec,
        }

        resource = server.k8s.create(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, body=body)

        return {
            "success": True,
            "name": resource.metadata.name,
            "message": f"ClusterTrainingRuntime '{name}' created.",
        }

    @mcp.tool()
    def setup_training_runtime(
        namespace: str,
        name: str = "mcp-transformers-runtime",
        framework: str = "transformers",
    ) -> dict[str, Any]:
        """Set up a training runtime for a namespace.

        Creates a pre-configured training runtime suitable for most
        fine-tuning tasks. Includes model and dataset initializers
        for HuggingFace-based training.

        Args:
            namespace: The namespace to configure.
            name: Name for the runtime (default: "mcp-transformers-runtime").
            framework: Training framework (default: "transformers").

        Returns:
            Runtime setup confirmation.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        # Check if runtime already exists
        try:
            server.k8s.get(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, name)
            return {
                "exists": True,
                "name": name,
                "message": f"ClusterTrainingRuntime '{name}' already exists.",
            }
        except Exception:
            pass  # Runtime doesn't exist, proceed to create

        # Build standard runtime spec
        spec = _build_runtime_spec(
            trainer_image="quay.io/modh/training:latest",
            framework=framework,
            model_initializer_image="quay.io/modh/model-initializer:latest",
            dataset_initializer_image="quay.io/modh/dataset-initializer:latest",
        )

        body = {
            "apiVersion": TrainingCRDs.CLUSTER_TRAINING_RUNTIME.api_version,
            "kind": TrainingCRDs.CLUSTER_TRAINING_RUNTIME.kind,
            "metadata": {
                "name": name,
                "labels": {
                    "training.kubeflow.org/framework": framework,
                    "app.kubernetes.io/managed-by": "rhoai-mcp",
                },
            },
            "spec": spec,
        }

        resource = server.k8s.create(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, body=body)

        return {
            "success": True,
            "name": resource.metadata.name,
            "message": (
                f"ClusterTrainingRuntime '{name}' has been set up. "
                "You can now use it to create training jobs."
            ),
        }

    @mcp.tool()
    def delete_runtime(name: str, confirm: bool = False) -> dict[str, Any]:
        """Delete a ClusterTrainingRuntime.

        Removes a training runtime. Existing training jobs using this
        runtime will not be affected, but new jobs cannot use it.

        Args:
            name: Name of the runtime to delete.
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
                    f"To delete runtime '{name}', set confirm=True. "
                    "Existing jobs using this runtime will not be affected."
                ),
            }

        server.k8s.delete(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, name)

        return {
            "success": True,
            "deleted": True,
            "name": name,
            "message": f"ClusterTrainingRuntime '{name}' has been deleted.",
        }


def _build_runtime_spec(
    trainer_image: str,
    framework: str = "transformers",
    model_initializer_image: str | None = None,
    dataset_initializer_image: str | None = None,
) -> dict[str, Any]:
    """Build a ClusterTrainingRuntime spec."""
    initializers = []

    if model_initializer_image:
        initializers.append({
            "type": "model",
            "image": model_initializer_image,
        })

    if dataset_initializer_image:
        initializers.append({
            "type": "dataset",
            "image": dataset_initializer_image,
        })

    spec: dict[str, Any] = {
        "template": {
            "spec": {
                "trainer": {
                    "image": trainer_image,
                },
            },
        },
    }

    if initializers:
        spec["template"]["spec"]["initializers"] = initializers

    return spec
