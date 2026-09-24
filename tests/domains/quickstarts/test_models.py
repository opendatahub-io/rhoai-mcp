"""Tests for quickstart models and parsing."""

import pytest

from rhoai_mcp.domains.quickstarts.models import (
    QuickstartManifest,
    QuickstartParameter,
    QuickstartRegistry,
    to_env_key,
)
from rhoai_mcp.utils.errors import RHOAIError
from tests.domains.quickstarts.conftest import MANIFEST_YAML, REGISTRY_YAML


class TestToEnvKey:
    """Tests for the to_env_key helper."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ollama.gpu.enabled", "OLLAMA_GPU_ENABLED"),
            ("keycloak.realm.testUser.password", "KEYCLOAK_REALM_TESTUSER_PASSWORD"),
            ("already_snake", "ALREADY_SNAKE"),
            ("--leading.trailing--", "LEADING_TRAILING"),
            ("mixed 123-abc", "MIXED_123_ABC"),
        ],
    )
    def test_to_env_key(self, name: str, expected: str) -> None:
        assert to_env_key(name) == expected


class TestRegistry:
    """Tests for QuickstartRegistry parsing."""

    def test_from_yaml_parses_entries(self) -> None:
        registry = QuickstartRegistry.from_yaml(REGISTRY_YAML)

        assert len(registry.quickstarts) == 1
        entry = registry.quickstarts[0]
        assert entry.name == "peoplemesh"
        assert entry.display_name == "Peoplemesh"
        assert entry.latest_manifest_version == "1.0.0"
        assert entry.available_manifest_versions[0].version == "1.0.0"
        assert entry.available_manifest_versions[0].status == "stable"
        assert entry.tags == ["llm", "rag"]
        assert entry.manifest_repo == "quay.io/rh-ai-quickstart/peoplemesh-manifest"

    def test_get_returns_entry(self) -> None:
        registry = QuickstartRegistry.from_yaml(REGISTRY_YAML)
        assert registry.get("peoplemesh") is not None
        assert registry.get("missing") is None

    def test_manifest_ref_defaults_to_latest(self) -> None:
        entry = QuickstartRegistry.from_yaml(REGISTRY_YAML).quickstarts[0]
        assert entry.manifest_ref() == "quay.io/rh-ai-quickstart/peoplemesh-manifest:1.0.0"

    def test_manifest_ref_with_explicit_version(self) -> None:
        entry = QuickstartRegistry.from_yaml(REGISTRY_YAML).quickstarts[0]
        assert entry.manifest_ref("2.0.0") == "quay.io/rh-ai-quickstart/peoplemesh-manifest:2.0.0"

    def test_from_yaml_rejects_non_mapping(self) -> None:
        with pytest.raises(RHOAIError):
            QuickstartRegistry.from_yaml(b"- just\n- a\n- list\n")


class TestManifest:
    """Tests for QuickstartManifest parsing."""

    def test_from_yaml_parses_deployment_and_params(self) -> None:
        manifest = QuickstartManifest.from_yaml(MANIFEST_YAML)

        assert manifest.name == "peoplemesh"
        assert manifest.version == "1.0.0"
        assert manifest.display_name == "Peoplemesh"
        assert manifest.deployment.installer.image.endswith("peoplemesh-installer:1.0.0")
        assert manifest.deployment.installer.command == ["/installer/entrypoint.sh"]
        assert manifest.deployment.default_namespace == "peoplemesh-quickstart"
        assert len(manifest.parameters.secrets) == 1
        assert len(manifest.parameters.configuration) == 1
        # Passthrough sections preserved.
        assert manifest.prerequisites == {"openshift": {"minimumVersion": "4.12"}}
        assert manifest.llm_context == {"whenToRecommend": "Recommend for talent discovery."}
        # The status block is preserved verbatim (previously dropped by extra="ignore").
        assert manifest.status == {"pollingInterval": "10s", "timeout": "15m"}

    def test_supports_action_case_insensitive(self) -> None:
        manifest = QuickstartManifest.from_yaml(MANIFEST_YAML)
        assert manifest.supports_action("install")
        assert manifest.supports_action("INSTALL")
        assert not manifest.supports_action("UPGRADE")

    def test_supports_mode_case_insensitive(self) -> None:
        manifest = QuickstartManifest.from_yaml(MANIFEST_YAML)
        assert manifest.supports_mode("demo")
        assert not manifest.supports_mode("production")

    def test_parameters_lookup_and_secret_detection(self) -> None:
        manifest = QuickstartManifest.from_yaml(MANIFEST_YAML)
        params = manifest.parameters

        assert params.get("ollama.gpu.enabled") is not None
        assert params.get("nope") is None
        assert params.is_secret("keycloak.realm.testUser.password")
        assert not params.is_secret("ollama.gpu.enabled")

    def test_from_yaml_missing_deployment_raises(self) -> None:
        bad = b"apiVersion: quickstart.redhat.com/v1\nkind: Quickstart\nmetadata:\n  name: x\n"
        with pytest.raises(RHOAIError):
            QuickstartManifest.from_yaml(bad)


class TestParameter:
    """Tests for QuickstartParameter env-var resolution."""

    def test_resolved_env_var_derived(self) -> None:
        param = QuickstartParameter(name="ollama.gpu.enabled")
        assert param.resolved_env_var() == "PARAM_OLLAMA_GPU_ENABLED"

    def test_resolved_env_var_explicit(self) -> None:
        param = QuickstartParameter.model_validate({"name": "x", "envVar": "MY_VAR"})
        assert param.resolved_env_var() == "MY_VAR"

    def test_to_dict_includes_guidance(self) -> None:
        param = QuickstartParameter.model_validate(
            {"name": "x", "llmGuidance": "ask the user", "required": True}
        )
        d = param.to_dict(is_secret=True)
        assert d["llm_guidance"] == "ask the user"
        assert d["secret"] is True
        assert d["required"] is True
        assert d["env_var"] == "PARAM_X"
