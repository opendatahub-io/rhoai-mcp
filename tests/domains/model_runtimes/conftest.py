"""Shared fixtures for model_runtimes tests."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_matrix_data() -> dict:
    """Sample CUDA compatibility matrix data."""
    return {
        "RHOAI serving runtime image": [
            {
                "image": "rhaiis/vllm-cuda-rhel9:3.0",
                "cuda_version": ["12.4"],
                "notes": "Test image",
            }
        ],
        "CUDA toolkit version": [
            {"cuda_version": ["12.4"], "min_driver_version": ["550.54.14"]}
        ],
        "GPU compute capability": [
            {"compute_capability": "8.0", "supported_cuda_versions": ["12.4"]}
        ],
    }


@pytest.fixture
def mock_k8s_client(sample_matrix_data: dict) -> MagicMock:
    """Mock K8s client with ConfigMap data."""
    mock = MagicMock()
    configmap = MagicMock()
    configmap.data = {"cuda_compat.json": json.dumps(sample_matrix_data)}
    mock.core_v1.read_namespaced_config_map.return_value = configmap
    return mock


@pytest.fixture
def mock_server(sample_matrix_data: dict) -> MagicMock:
    """Mock RHOAI server with K8s client."""
    server = MagicMock()
    configmap = MagicMock()
    configmap.data = {"cuda_compat.json": json.dumps(sample_matrix_data)}
    server.k8s.core_v1.read_namespaced_config_map.return_value = configmap
    return server


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register model_runtimes tools and return captured tool functions."""
    from rhoai_mcp.domains.model_runtimes.tools import register_tools

    mcp = MagicMock()
    registered_tools: dict[str, Any] = {}

    def capture_tool() -> Any:
        def decorator(func: Any) -> Any:
            registered_tools[func.__name__] = func
            return func

        return decorator

    mcp.tool = capture_tool
    register_tools(mcp, mock_server)
    return registered_tools
