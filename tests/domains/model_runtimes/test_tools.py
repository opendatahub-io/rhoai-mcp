"""Tests for Model Runtimes MCP tools."""

import json
from unittest.mock import MagicMock

import pytest

from tests.domains.model_runtimes.conftest import _register_tools


class TestModelRuntimesTools:
    """Test Model Runtimes MCP tools."""

    @pytest.mark.asyncio
    async def test_get_cuda_version_for_runtime_success(self, mock_server: MagicMock) -> None:
        """Test successful CUDA version lookup for runtime."""
        tools = _register_tools(mock_server)

        result = await tools["get_cuda_version_for_runtime"]("rhaiis/vllm-cuda-rhel9:3.0")

        assert result["image"] == "rhaiis/vllm-cuda-rhel9:3.0"
        assert result["cuda_versions"] == ["12.4"]

    @pytest.mark.asyncio
    async def test_get_cuda_version_for_runtime_not_found(self, mock_server: MagicMock) -> None:
        """Test CUDA version lookup for nonexistent runtime."""
        tools = _register_tools(mock_server)

        with pytest.raises(ValueError, match="Runtime image not found"):
            await tools["get_cuda_version_for_runtime"]("nonexistent/image:1.0")

    @pytest.mark.asyncio
    async def test_get_cuda_version_for_runtime_full_registry(
        self, mock_server: MagicMock
    ) -> None:
        """Test CUDA version lookup with full registry reference."""
        tools = _register_tools(mock_server)

        result = await tools["get_cuda_version_for_runtime"](
            "registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.0"
        )

        assert result["cuda_versions"] == ["12.4"]

    @pytest.mark.asyncio
    async def test_get_min_driver_for_cuda_version_success(
        self, mock_server: MagicMock
    ) -> None:
        """Test successful minimum driver lookup for CUDA version."""
        tools = _register_tools(mock_server)

        result = await tools["get_min_driver_for_cuda_version"]("12.4")

        assert result["cuda_version"] == "12.4"
        assert result["min_driver_versions"] == ["550.54.14"]

    @pytest.mark.asyncio
    async def test_get_min_driver_for_cuda_version_not_found(
        self, mock_server: MagicMock
    ) -> None:
        """Test minimum driver lookup for nonexistent CUDA version."""
        tools = _register_tools(mock_server)

        with pytest.raises(ValueError, match="CUDA version not found"):
            await tools["get_min_driver_for_cuda_version"]("99.9")

    @pytest.mark.asyncio
    async def test_get_supported_cuda_for_gpu_success(self, mock_server: MagicMock) -> None:
        """Test successful CUDA versions lookup for GPU compute capability."""
        tools = _register_tools(mock_server)

        result = await tools["get_supported_cuda_for_gpu"]("8.0")

        assert result["compute_capability"] == "8.0"
        assert result["supported_cuda_versions"] == ["12.4"]

    @pytest.mark.asyncio
    async def test_get_supported_cuda_for_gpu_not_found(self, mock_server: MagicMock) -> None:
        """Test CUDA versions lookup for nonexistent compute capability."""
        tools = _register_tools(mock_server)

        with pytest.raises(ValueError, match="GPU compute capability not found"):
            await tools["get_supported_cuda_for_gpu"]("99.9")

    def test_register_tools_creates_three_tools(self, mock_server: MagicMock) -> None:
        """Test that register_tools creates exactly 3 tools."""
        tools = _register_tools(mock_server)

        # Should register 3 tools
        assert len(tools) == 3
        assert "get_cuda_version_for_runtime" in tools
        assert "get_min_driver_for_cuda_version" in tools
        assert "get_supported_cuda_for_gpu" in tools



class TestToolErrorHandling:
    """Test error handling in tools."""

    @pytest.mark.asyncio
    async def test_configmap_not_found_raises_error(self) -> None:
        """Test that ConfigMap not found raises ValueError."""
        from kubernetes.client.exceptions import ApiException

        mock_server = MagicMock()
        mock_server.k8s.core_v1.read_namespaced_config_map.side_effect = ApiException(
            status=404
        )

        tools = _register_tools(mock_server)

        with pytest.raises(ValueError, match="not found"):
            await tools["get_cuda_version_for_runtime"]("rhaiis/vllm-cuda-rhel9:3.0")

    @pytest.mark.asyncio
    async def test_invalid_json_in_configmap_raises_error(self) -> None:
        """Test that invalid JSON in ConfigMap raises appropriate error."""
        mock_server = MagicMock()
        configmap = MagicMock()
        configmap.data = {"cuda_compat.json": "invalid json{{{"}
        mock_server.k8s.core_v1.read_namespaced_config_map.return_value = configmap

        tools = _register_tools(mock_server)

        with pytest.raises(json.JSONDecodeError):
            await tools["get_cuda_version_for_runtime"]("rhaiis/vllm-cuda-rhel9:3.0")


class TestLazyClientInitialization:
    """Test that client is lazy-initialized only when tools are called."""

    def test_client_not_initialized_during_registration(self, mock_server: MagicMock) -> None:
        """Test that K8s client is not accessed during tool registration."""
        # Register tools
        _register_tools(mock_server)

        # K8s should not be accessed yet
        mock_server.k8s.core_v1.read_namespaced_config_map.assert_not_called()

    @pytest.mark.asyncio
    async def test_client_initialized_on_first_tool_call(self, mock_server: MagicMock) -> None:
        """Test that client is initialized only when first tool is called."""
        tools = _register_tools(mock_server)

        # K8s not accessed yet
        mock_server.k8s.core_v1.read_namespaced_config_map.assert_not_called()

        # Call tool
        await tools["get_cuda_version_for_runtime"]("rhaiis/vllm-cuda-rhel9:3.0")

        # Now K8s should be accessed
        mock_server.k8s.core_v1.read_namespaced_config_map.assert_called_once()
