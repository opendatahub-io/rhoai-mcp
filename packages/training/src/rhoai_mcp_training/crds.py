"""CRD definitions for Kubeflow Training Operator resources."""

from rhoai_mcp_core.clients.base import CRDDefinition


class TrainingCRDs:
    """Kubeflow Training Operator CRD definitions."""

    TRAIN_JOB = CRDDefinition(
        group="trainer.kubeflow.org",
        version="v1",
        plural="trainjobs",
        kind="TrainJob",
    )

    CLUSTER_TRAINING_RUNTIME = CRDDefinition(
        group="trainer.kubeflow.org",
        version="v1",
        plural="clustertrainingruntimes",
        kind="ClusterTrainingRuntime",
    )

    TRAINING_RUNTIME = CRDDefinition(
        group="trainer.kubeflow.org",
        version="v1",
        plural="trainingruntimes",
        kind="TrainingRuntime",
    )

    @classmethod
    def all_crds(cls) -> list[CRDDefinition]:
        """Return all CRD definitions."""
        return [cls.TRAIN_JOB, cls.CLUSTER_TRAINING_RUNTIME, cls.TRAINING_RUNTIME]
