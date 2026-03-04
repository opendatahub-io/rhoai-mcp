"""Tests for configuration module."""

from pathlib import Path

import pytest

from rhoai_mcp.config import (
    AuthMode,
    LogLevel,
    RHOAIConfig,
    TransportMode,
)


class TestRHOAIConfig:
    """Tests for RHOAIConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RHOAIConfig()

        assert config.auth_mode == AuthMode.AUTO
        assert config.transport == TransportMode.STDIO
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.enable_dangerous_operations is False
        assert config.read_only_mode is False
        assert config.log_level == LogLevel.INFO

    def test_auth_mode_token_validation(self):
        """Test token auth mode validation."""
        config = RHOAIConfig(
            auth_mode=AuthMode.TOKEN,
            api_server="https://api.cluster.example.com:6443",
            api_token="sha256~token",
        )

        # Should not raise
        warnings = config.validate_auth_config()
        assert len(warnings) == 0

    def test_auth_mode_token_missing_server(self):
        """Test token auth mode without server raises error."""
        config = RHOAIConfig(
            auth_mode=AuthMode.TOKEN,
            api_token="sha256~token",
        )

        with pytest.raises(ValueError, match="api_server is required"):
            config.validate_auth_config()

    def test_auth_mode_token_missing_token(self):
        """Test token auth mode without token raises error."""
        config = RHOAIConfig(
            auth_mode=AuthMode.TOKEN,
            api_server="https://api.cluster.example.com:6443",
        )

        with pytest.raises(ValueError, match="api_token is required"):
            config.validate_auth_config()

    def test_is_operation_allowed_read_only(self):
        """Test read-only mode blocks write operations."""
        config = RHOAIConfig(read_only_mode=True)

        allowed, reason = config.is_operation_allowed("create")
        assert allowed is False
        assert "Read-only" in reason

        allowed, reason = config.is_operation_allowed("delete")
        assert allowed is False

        # Read should still be allowed
        allowed, reason = config.is_operation_allowed("get")
        assert allowed is True

    def test_is_operation_allowed_dangerous_disabled(self):
        """Test dangerous operations disabled by default."""
        config = RHOAIConfig(enable_dangerous_operations=False)

        allowed, reason = config.is_operation_allowed("delete")
        assert allowed is False
        assert "Dangerous operations are disabled" in reason

        # Non-dangerous operations should be allowed
        allowed, reason = config.is_operation_allowed("create")
        assert allowed is True

    def test_is_operation_allowed_dangerous_enabled(self):
        """Test dangerous operations when enabled."""
        config = RHOAIConfig(enable_dangerous_operations=True)

        allowed, reason = config.is_operation_allowed("delete")
        assert allowed is True
        assert reason is None

    def test_effective_kubeconfig_path_default(self):
        """Test default kubeconfig path."""
        config = RHOAIConfig()
        expected = Path.home() / ".kube" / "config"
        assert config.effective_kubeconfig_path == expected

    def test_effective_kubeconfig_path_explicit(self):
        """Test explicit kubeconfig path."""
        config = RHOAIConfig(kubeconfig_path="/custom/kubeconfig")
        assert config.effective_kubeconfig_path == Path("/custom/kubeconfig")

    def test_effective_kubeconfig_path_env(self, monkeypatch):
        """Test kubeconfig path from environment."""
        monkeypatch.setenv("KUBECONFIG", "/env/kubeconfig")
        config = RHOAIConfig()
        assert config.effective_kubeconfig_path == Path("/env/kubeconfig")

    def test_env_prefix(self, monkeypatch):
        """Test environment variable prefix."""
        monkeypatch.setenv("RHOAI_MCP_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("RHOAI_MCP_PORT", "9000")

        config = RHOAIConfig()
        assert config.log_level == LogLevel.DEBUG
        assert config.port == 9000

    def test_enabled_plugins_default_none(self):
        """Test enabled_plugins defaults to None (all plugins)."""
        config = RHOAIConfig()
        assert config.enabled_plugins is None

    def test_enabled_plugins_from_comma_separated_string(self):
        """Test parsing comma-separated string into list."""
        config = RHOAIConfig(enabled_plugins="projects,inference,training")
        assert config.enabled_plugins == ["projects", "inference", "training"]

    def test_enabled_plugins_strips_whitespace(self):
        """Test whitespace is stripped from plugin names."""
        config = RHOAIConfig(enabled_plugins=" projects , inference , training ")
        assert config.enabled_plugins == ["projects", "inference", "training"]

    def test_enabled_plugins_empty_string_returns_empty_list(self):
        """Test empty string returns empty list."""
        config = RHOAIConfig(enabled_plugins="")
        assert config.enabled_plugins == []

    def test_enabled_plugins_skips_empty_entries(self):
        """Test empty entries from extra commas are skipped."""
        config = RHOAIConfig(enabled_plugins="projects,,inference,")
        assert config.enabled_plugins == ["projects", "inference"]

    def test_enabled_plugins_from_list(self):
        """Test list input passes through unchanged."""
        config = RHOAIConfig(enabled_plugins=["projects", "inference"])
        assert config.enabled_plugins == ["projects", "inference"]

    def test_enabled_plugins_from_env(self, monkeypatch):
        """Test enabled_plugins from environment variable."""
        monkeypatch.setenv("RHOAI_MCP_ENABLED_PLUGINS", "projects,notebooks")
        config = RHOAIConfig()
        assert config.enabled_plugins == ["projects", "notebooks"]
