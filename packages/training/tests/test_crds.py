"""Tests for training CRD definitions."""

import pytest

from rhoai_mcp_training.crds import TrainingCRDs


class TestTrainingCRDs:
    """Test training CRD definitions."""

    def test_train_job_crd_properties(self) -> None:
        """Test TrainJob CRD has correct properties."""
        crd = TrainingCRDs.TRAIN_JOB
        assert crd.group == "trainer.kubeflow.org"
        assert crd.version == "v1"
        assert crd.plural == "trainjobs"
        assert crd.kind == "TrainJob"

    def test_train_job_api_version(self) -> None:
        """Test TrainJob CRD returns correct api_version."""
        crd = TrainingCRDs.TRAIN_JOB
        assert crd.api_version == "trainer.kubeflow.org/v1"

    def test_cluster_training_runtime_crd_properties(self) -> None:
        """Test ClusterTrainingRuntime CRD has correct properties."""
        crd = TrainingCRDs.CLUSTER_TRAINING_RUNTIME
        assert crd.group == "trainer.kubeflow.org"
        assert crd.version == "v1"
        assert crd.plural == "clustertrainingruntimes"
        assert crd.kind == "ClusterTrainingRuntime"

    def test_cluster_training_runtime_api_version(self) -> None:
        """Test ClusterTrainingRuntime CRD returns correct api_version."""
        crd = TrainingCRDs.CLUSTER_TRAINING_RUNTIME
        assert crd.api_version == "trainer.kubeflow.org/v1"

    def test_training_runtime_crd_properties(self) -> None:
        """Test TrainingRuntime CRD has correct properties."""
        crd = TrainingCRDs.TRAINING_RUNTIME
        assert crd.group == "trainer.kubeflow.org"
        assert crd.version == "v1"
        assert crd.plural == "trainingruntimes"
        assert crd.kind == "TrainingRuntime"

    def test_all_crds_returns_list(self) -> None:
        """Test all_crds returns all CRD definitions."""
        crds = TrainingCRDs.all_crds()
        assert len(crds) == 3
        assert TrainingCRDs.TRAIN_JOB in crds
        assert TrainingCRDs.CLUSTER_TRAINING_RUNTIME in crds
        assert TrainingCRDs.TRAINING_RUNTIME in crds
