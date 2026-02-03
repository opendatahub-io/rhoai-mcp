"""Fixtures for Model Registry tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rhoai_mcp.config import RHOAIConfig


@pytest.fixture
def mock_config() -> RHOAIConfig:
    """Create a mock RHOAIConfig for testing."""
    config = MagicMock(spec=RHOAIConfig)
    config.model_registry_url = "http://model-registry.test.svc:8080"
    config.model_registry_timeout = 30
    config.model_registry_enabled = True
    config.default_list_limit = None
    config.max_list_limit = 100
    return config


@pytest.fixture
def mock_http_response() -> MagicMock:
    """Create a mock HTTP response."""

    def _create_response(
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
    ) -> MagicMock:
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.raise_for_status = MagicMock()
        if status_code >= 400:
            from httpx import HTTPStatusError, Request, Response

            response.raise_for_status.side_effect = HTTPStatusError(
                message=f"HTTP {status_code}",
                request=MagicMock(spec=Request),
                response=MagicMock(spec=Response),
            )
        return response

    return _create_response


@pytest.fixture
def sample_registered_model() -> dict[str, Any]:
    """Sample registered model API response."""
    return {
        "id": "model-123",
        "name": "llama-2-7b",
        "description": "Llama 2 7B fine-tuned model",
        "owner": "data-science-team",
        "state": "LIVE",
        "customProperties": {
            "framework": "pytorch",
            "task": "text-generation",
        },
    }


@pytest.fixture
def sample_model_version() -> dict[str, Any]:
    """Sample model version API response."""
    return {
        "id": "version-456",
        "name": "v1.0.0",
        "registeredModelId": "model-123",
        "state": "LIVE",
        "description": "Production release",
        "author": "ml-engineer",
        "customProperties": {
            "accuracy": "0.95",
            "training_date": "2024-01-15",
        },
    }


@pytest.fixture
def sample_model_artifact() -> dict[str, Any]:
    """Sample model artifact API response."""
    return {
        "id": "artifact-789",
        "name": "model-weights",
        "uri": "s3://models/llama-2-7b/v1.0.0/weights.safetensors",
        "description": "SafeTensors model weights",
        "modelFormatName": "safetensors",
        "modelFormatVersion": "0.4.0",
        "storageKey": "models",
        "storagePath": "llama-2-7b/v1.0.0/weights.safetensors",
        "customProperties": {},
    }


@pytest.fixture
def mock_async_client() -> AsyncMock:
    """Create a mock async HTTP client."""
    client = AsyncMock()
    client.aclose = AsyncMock()
    return client
