"""Dataclasses representing mock cluster state for evaluations.

Provides a realistic RHOAI cluster with pre-populated data including
projects, workbenches, training jobs, inference services, and more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockMetadata:
    """Kubernetes-style object metadata."""

    name: str
    namespace: str | None = None
    uid: str = ""
    creation_timestamp: str = "2025-01-15T10:00:00Z"
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass
class MockResource:
    """Generic mock Kubernetes resource."""

    metadata: MockMetadata
    spec: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    kind: str = ""
    api_version: str = ""


@dataclass
class ClusterState:
    """Represents the full state of a mock RHOAI cluster.

    Resources are stored by CRD key (api_version/plural) -> list of resources.
    Core K8s resources (namespaces, secrets, PVCs) are stored separately.
    """

    # CRD resources keyed by "api_version/plural"
    resources: dict[str, list[MockResource]] = field(default_factory=dict)

    # Core K8s resources
    namespaces: list[MockResource] = field(default_factory=list)
    projects: list[MockResource] = field(default_factory=list)
    secrets: list[MockResource] = field(default_factory=list)
    pvcs: list[MockResource] = field(default_factory=list)


def _make_uid(kind: str, name: str) -> str:
    """Generate a deterministic mock UID."""
    return f"{kind.lower()}-{name}-uid-12345"


def create_default_cluster_state() -> ClusterState:
    """Create a realistic mock RHOAI cluster state.

    Includes:
    - 2 projects (ml-experiments, production-models)
    - 1 running workbench in ml-experiments
    - 1 completed training job (llama fine-tune)
    - 1 failed training job (for troubleshooting scenarios)
    - 1 ready inference service (granite-serving)
    - NVIDIA A100 accelerator profile
    - 1 cluster training runtime
    - PVCs and secrets for data connections
    """
    state = ClusterState()

    # --- Projects / Namespaces ---
    for ns_name, display_name in [
        ("ml-experiments", "ML Experiments"),
        ("production-models", "Production Models"),
    ]:
        ns = MockResource(
            metadata=MockMetadata(
                name=ns_name,
                labels={
                    "opendatahub.io/dashboard": "true",
                    "kubernetes.io/metadata.name": ns_name,
                },
                annotations={"openshift.io/display-name": display_name},
                uid=_make_uid("namespace", ns_name),
            ),
            kind="Namespace",
        )
        state.namespaces.append(ns)

        proj = MockResource(
            metadata=MockMetadata(
                name=ns_name,
                labels={
                    "opendatahub.io/dashboard": "true",
                    "kubernetes.io/metadata.name": ns_name,
                },
                annotations={"openshift.io/display-name": display_name},
                uid=_make_uid("project", ns_name),
            ),
            kind="Project",
            api_version="project.openshift.io/v1",
        )
        state.projects.append(proj)

    # --- DataScienceCluster ---
    dsc = MockResource(
        metadata=MockMetadata(
            name="default-dsc",
            uid=_make_uid("dsc", "default-dsc"),
        ),
        status={
            "phase": "Ready",
            "installedComponents": {
                "dashboard": True,
                "workbenches": True,
                "modelmeshserving": True,
                "kserve": True,
                "datasciencepipelines": True,
                "trainingoperator": True,
            },
        },
        kind="DataScienceCluster",
        api_version="datasciencecluster.opendatahub.io/v1",
    )
    state.resources.setdefault(
        "datasciencecluster.opendatahub.io/v1/datascienceclusters", []
    ).append(dsc)

    # --- AcceleratorProfile ---
    acc = MockResource(
        metadata=MockMetadata(
            name="nvidia-a100",
            annotations={
                "openshift.io/display-name": "NVIDIA A100 80GB",
                "openshift.io/description": "NVIDIA A100 80GB GPU accelerator",
            },
            uid=_make_uid("accelerator", "nvidia-a100"),
        ),
        spec={
            "enabled": True,
            "identifier": "nvidia.com/gpu",
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
        },
        kind="AcceleratorProfile",
        api_version="dashboard.opendatahub.io/v1",
    )
    state.resources.setdefault(
        "dashboard.opendatahub.io/v1/acceleratorprofiles", []
    ).append(acc)

    # --- Notebook (Workbench) ---
    notebook = MockResource(
        metadata=MockMetadata(
            name="my-workbench",
            namespace="ml-experiments",
            labels={
                "app": "my-workbench",
                "opendatahub.io/dashboard": "true",
                "opendatahub.io/odh-managed": "true",
            },
            annotations={
                "notebooks.opendatahub.io/inject-oauth": "true",
                "openshift.io/display-name": "My Workbench",
                "opendatahub.io/image-display-name": "Minimal Python",
            },
            uid=_make_uid("notebook", "my-workbench"),
        ),
        spec={
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "my-workbench",
                            "image": "quay.io/modh/odh-minimal-notebook:v2-2024a",
                            "resources": {
                                "limits": {"cpu": "2", "memory": "8Gi"},
                                "requests": {"cpu": "1", "memory": "4Gi"},
                            },
                        }
                    ],
                }
            }
        },
        status={
            "conditions": [
                {"type": "Ready", "status": "True"},
            ],
            "readyReplicas": 1,
        },
        kind="Notebook",
        api_version="kubeflow.org/v1",
    )
    state.resources.setdefault("kubeflow.org/v1/notebooks", []).append(notebook)

    # --- Training Jobs ---
    # Completed training job
    train_job_ok = MockResource(
        metadata=MockMetadata(
            name="llama-finetune-001",
            namespace="ml-experiments",
            labels={"trainer.kubeflow.org/runtime": "torchtune-llama"},
            annotations={},
            uid=_make_uid("trainjob", "llama-finetune-001"),
        ),
        spec={
            "modelConfig": {"name": "meta-llama/Llama-3.1-8B"},
            "datasetConfig": {"name": "alpaca-cleaned"},
            "trainer": {
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {"nvidia.com/gpu": "1"},
                },
            },
            "runtimeRef": {"name": "torchtune-llama"},
        },
        status={
            "conditions": [
                {"type": "Created", "status": "True"},
                {"type": "Completed", "status": "True"},
            ],
        },
        kind="TrainJob",
        api_version="trainer.kubeflow.org/v1",
    )

    # Failed training job (for troubleshooting)
    train_job_fail = MockResource(
        metadata=MockMetadata(
            name="failed-training-001",
            namespace="ml-experiments",
            labels={"trainer.kubeflow.org/runtime": "torchtune-llama"},
            annotations={},
            uid=_make_uid("trainjob", "failed-training-001"),
        ),
        spec={
            "modelConfig": {"name": "meta-llama/Llama-3.1-8B"},
            "datasetConfig": {"name": "bad-dataset"},
            "trainer": {
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {"nvidia.com/gpu": "2"},
                },
            },
            "runtimeRef": {"name": "torchtune-llama"},
        },
        status={
            "conditions": [
                {"type": "Created", "status": "True"},
                {"type": "Failed", "status": "True", "message": "OOMKilled: GPU out of memory"},
            ],
        },
        kind="TrainJob",
        api_version="trainer.kubeflow.org/v1",
    )

    state.resources.setdefault("trainer.kubeflow.org/v1/trainjobs", []).extend(
        [train_job_ok, train_job_fail]
    )

    # --- ClusterTrainingRuntime ---
    ctr = MockResource(
        metadata=MockMetadata(
            name="torchtune-llama",
            labels={"trainer.kubeflow.org/runtime-type": "torchtune"},
            annotations={
                "openshift.io/display-name": "TorchTune LLaMA Runtime",
            },
            uid=_make_uid("ctr", "torchtune-llama"),
        ),
        spec={
            "template": {
                "spec": {
                    "trainer": {
                        "image": "quay.io/modh/trainer-torchtune:latest",
                    },
                    "initializers": [
                        {
                            "type": "model",
                            "image": "quay.io/modh/model-initializer:latest",
                        },
                        {
                            "type": "dataset",
                            "image": "quay.io/modh/dataset-initializer:latest",
                        },
                    ],
                }
            }
        },
        kind="ClusterTrainingRuntime",
        api_version="trainer.kubeflow.org/v1",
    )
    state.resources.setdefault(
        "trainer.kubeflow.org/v1/clustertrainingruntimes", []
    ).append(ctr)

    # --- TrainingRuntime (namespace-scoped) ---
    tr = MockResource(
        metadata=MockMetadata(
            name="custom-training-runtime",
            namespace="ml-experiments",
            labels={"trainer.kubeflow.org/runtime-type": "custom"},
            uid=_make_uid("tr", "custom-training-runtime"),
        ),
        spec={
            "template": {
                "spec": {
                    "trainer": {
                        "image": "quay.io/modh/trainer-custom:latest",
                    },
                }
            }
        },
        kind="TrainingRuntime",
        api_version="trainer.kubeflow.org/v1",
    )
    state.resources.setdefault(
        "trainer.kubeflow.org/v1/trainingruntimes", []
    ).append(tr)

    # --- InferenceService ---
    isvc = MockResource(
        metadata=MockMetadata(
            name="granite-serving",
            namespace="production-models",
            labels={"serving.kserve.io/inferenceservice": "granite-serving"},
            annotations={
                "openshift.io/display-name": "Granite 3B Model",
            },
            uid=_make_uid("isvc", "granite-serving"),
        ),
        spec={
            "predictor": {
                "model": {
                    "modelFormat": {"name": "vLLM"},
                    "runtime": "vllm-runtime",
                    "storageUri": "s3://models/granite-3b",
                },
                "resources": {
                    "limits": {"nvidia.com/gpu": "1"},
                    "requests": {"nvidia.com/gpu": "1"},
                },
            }
        },
        status={
            "conditions": [
                {"type": "Ready", "status": "True"},
            ],
            "url": "https://granite-serving-production-models.apps.cluster.example.com",
        },
        kind="InferenceService",
        api_version="serving.kserve.io/v1beta1",
    )
    state.resources.setdefault(
        "serving.kserve.io/v1beta1/inferenceservices", []
    ).append(isvc)

    # --- ServingRuntime ---
    srt = MockResource(
        metadata=MockMetadata(
            name="vllm-runtime",
            namespace="production-models",
            labels={"opendatahub.io/dashboard": "true"},
            annotations={
                "openshift.io/display-name": "vLLM ServingRuntime",
            },
            uid=_make_uid("srt", "vllm-runtime"),
        ),
        spec={
            "supportedModelFormats": [
                {"name": "vLLM", "version": "1", "autoSelect": True},
            ],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": "quay.io/modh/vllm:latest",
                }
            ],
        },
        kind="ServingRuntime",
        api_version="serving.kserve.io/v1alpha1",
    )
    state.resources.setdefault(
        "serving.kserve.io/v1alpha1/servingruntimes", []
    ).append(srt)

    # --- Template (in redhat-ods-applications) ---
    tmpl = MockResource(
        metadata=MockMetadata(
            name="vllm-runtime-template",
            namespace="redhat-ods-applications",
            labels={"opendatahub.io/dashboard": "true"},
            uid=_make_uid("template", "vllm-runtime-template"),
        ),
        spec={},
        kind="Template",
        api_version="template.openshift.io/v1",
    )
    state.resources.setdefault(
        "template.openshift.io/v1/templates", []
    ).append(tmpl)

    # --- DSPA (DataSciencePipelinesApplication) ---
    dspa = MockResource(
        metadata=MockMetadata(
            name="dspa-default",
            namespace="ml-experiments",
            uid=_make_uid("dspa", "dspa-default"),
        ),
        spec={
            "apiServer": {"deploy": True},
            "database": {"mariaDB": {"deploy": True}},
            "objectStorage": {
                "externalStorage": {
                    "host": "s3.amazonaws.com",
                    "bucket": "pipeline-artifacts",
                }
            },
        },
        status={
            "conditions": [
                {"type": "Ready", "status": "True"},
            ],
        },
        kind="DataSciencePipelinesApplication",
        api_version="datasciencepipelinesapplications.opendatahub.io/v1alpha1",
    )
    state.resources.setdefault(
        "datasciencepipelinesapplications.opendatahub.io/v1alpha1/"
        "datasciencepipelinesapplications",
        [],
    ).append(dspa)

    # --- Secrets (data connections) ---
    s3_secret = MockResource(
        metadata=MockMetadata(
            name="aws-connection-models",
            namespace="ml-experiments",
            labels={
                "opendatahub.io/dashboard": "true",
                "opendatahub.io/managed": "true",
            },
            annotations={
                "opendatahub.io/connection-type": "s3",
                "openshift.io/display-name": "Model Storage",
            },
            uid=_make_uid("secret", "aws-connection-models"),
        ),
        kind="Secret",
    )
    state.secrets.append(s3_secret)

    # --- PVCs ---
    pvc = MockResource(
        metadata=MockMetadata(
            name="workbench-storage",
            namespace="ml-experiments",
            labels={"opendatahub.io/dashboard": "true"},
            annotations={"openshift.io/display-name": "Workbench Storage"},
            uid=_make_uid("pvc", "workbench-storage"),
        ),
        spec={
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": "20Gi"}},
        },
        status={"phase": "Bound"},
        kind="PersistentVolumeClaim",
    )
    state.pvcs.append(pvc)

    return state
