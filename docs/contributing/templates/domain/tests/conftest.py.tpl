"""Test fixtures for {{DOMAIN_NAME}} domain."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.{{DOMAIN_NAME}}.client import {{DOMAIN_CLASS}}Client


@pytest.fixture
def mock_k8s() -> MagicMock:
    """Create a mock K8sClient."""
    return MagicMock()


@pytest.fixture
def client(mock_k8s: MagicMock) -> {{DOMAIN_CLASS}}Client:
    """Create a {{DOMAIN_CLASS}}Client with mock K8s."""
    return {{DOMAIN_CLASS}}Client(mock_k8s)
