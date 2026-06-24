"""Pytest configuration and fixtures for RHOAI MCP tests."""

import pytest
from unittest.mock import MagicMock, patch

from rhoai_mcp.config import RHOAIConfig, AuthMode


@pytest.fixture
def mock_config() -> RHOAIConfig:
    """Create a test configuration."""
    return RHOAIConfig(
        auth_mode=AuthMode.KUBECONFIG,
        enable_dangerous_operations=True,
        read_only_mode=False,
    )


@pytest.fixture
def mock_k8s_client():
    """Create a mocked K8s client."""
    with patch("rhoai_mcp.clients.base.K8sClient") as mock_class:
        client = MagicMock()
        client.is_connected = True
        mock_class.return_value = client
        yield client


@pytest.fixture
def mock_core_v1_api():
    """Create a mocked CoreV1Api."""
    with patch("kubernetes.client.CoreV1Api") as mock_api:
        yield mock_api.return_value


@pytest.fixture
def sample_namespace():
    """Create a sample namespace object."""
    ns = MagicMock()
    ns.metadata.name = "test-project"
    ns.metadata.namespace = None
    ns.metadata.uid = "test-uid"
    ns.metadata.creation_timestamp = "2024-01-01T00:00:00Z"
    ns.metadata.labels = {
        "opendatahub.io/dashboard": "true",
        "modelmesh-enabled": "false",
    }
    ns.metadata.annotations = {
        "openshift.io/display-name": "Test Project",
        "openshift.io/description": "A test project",
    }
    ns.status.phase = "Active"
    return ns


@pytest.fixture
def sample_notebook():
    """Create a sample Notebook CR object."""
    nb = MagicMock()
    nb.metadata.name = "test-workbench"
    nb.metadata.namespace = "test-project"
    nb.metadata.uid = "notebook-uid"
    nb.metadata.creation_timestamp = "2024-01-01T00:00:00Z"
    nb.metadata.labels = {"notebook-name": "test-workbench"}
    nb.metadata.annotations = {
        "notebooks.opendatahub.io/inject-oauth": "true",
        "opendatahub.io/image-display-name": "Jupyter Data Science",
        "notebooks.opendatahub.io/last-size-selection": "Small",
    }
    nb.spec = {
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "test-workbench",
                        "image": "jupyter-datascience:2024.1",
                        "resources": {
                            "requests": {"cpu": "500m", "memory": "1Gi"},
                            "limits": {"cpu": "2", "memory": "4Gi"},
                        },
                    }
                ],
                "volumes": [
                    {
                        "name": "storage",
                        "persistentVolumeClaim": {"claimName": "test-workbench-pvc"},
                    }
                ],
            }
        }
    }
    nb.status = MagicMock()
    nb.status.conditions = []
    nb.status.readyReplicas = 1
    return nb


@pytest.fixture
def sample_inference_service():
    """Create a sample InferenceService CR object."""
    isvc = MagicMock()
    isvc.metadata.name = "test-model"
    isvc.metadata.namespace = "test-project"
    isvc.metadata.uid = "isvc-uid"
    isvc.metadata.creation_timestamp = "2024-01-01T00:00:00Z"
    isvc.metadata.labels = {}
    isvc.metadata.annotations = {
        "openshift.io/display-name": "Test Model",
    }
    isvc.spec = {
        "predictor": {
            "minReplicas": 1,
            "maxReplicas": 1,
            "model": {
                "modelFormat": {"name": "onnx"},
                "runtime": "ovms",
                "storageUri": "s3://bucket/model",
            },
        }
    }
    isvc.status = MagicMock()
    isvc.status.conditions = [
        MagicMock(type="Ready", status="True", reason=None, message=None)
    ]
    isvc.status.address = MagicMock(url="http://test-model.test-project.svc.cluster.local")
    return isvc


@pytest.fixture
def sample_secret():
    """Create a sample data connection Secret object."""
    import base64

    secret = MagicMock()
    secret.metadata.name = "test-connection"
    secret.metadata.namespace = "test-project"
    secret.metadata.uid = "secret-uid"
    secret.metadata.creation_timestamp = "2024-01-01T00:00:00Z"
    secret.metadata.labels = {"opendatahub.io/dashboard": "true"}
    secret.metadata.annotations = {
        "opendatahub.io/connection-type": "s3",
        "opendatahub.io/managed": "true",
        "openshift.io/display-name": "Test S3 Connection",
    }
    secret.data = {
        "AWS_ACCESS_KEY_ID": base64.b64encode(b"AKIAIOSFODNN7EXAMPLE").decode(),
        "AWS_SECRET_ACCESS_KEY": base64.b64encode(b"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY").decode(),
        "AWS_S3_ENDPOINT": base64.b64encode(b"https://s3.amazonaws.com").decode(),
        "AWS_S3_BUCKET": base64.b64encode(b"my-bucket").decode(),
        "AWS_DEFAULT_REGION": base64.b64encode(b"us-east-1").decode(),
    }
    return secret


@pytest.fixture
def sample_pvc():
    """Create a sample PVC object."""
    pvc = MagicMock()
    pvc.metadata.name = "test-storage"
    pvc.metadata.namespace = "test-project"
    pvc.metadata.uid = "pvc-uid"
    pvc.metadata.creation_timestamp = "2024-01-01T00:00:00Z"
    pvc.metadata.labels = {"opendatahub.io/dashboard": "true"}
    pvc.metadata.annotations = {
        "openshift.io/display-name": "Test Storage",
    }
    pvc.spec.access_modes = ["ReadWriteOnce"]
    pvc.spec.resources.requests = {"storage": "10Gi"}
    pvc.spec.storage_class_name = "gp3"
    pvc.spec.volume_name = None
    pvc.status.phase = "Bound"
    return pvc
