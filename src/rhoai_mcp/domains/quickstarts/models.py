"""Pydantic models for the quickstarts domain.

Two OCI artifacts back this domain:

* the **registry** — a lightweight catalog index listing every quickstart
  (one :class:`QuickstartSummary` per entry); and
* a per-quickstart **manifest** — the rich descriptor with prerequisites,
  parameters and deployment configuration (:class:`QuickstartManifest`).

Only the fields the tools operate on programmatically are modelled. Rich,
descriptive sections (``prerequisites``, ``classification``, ``llmContext`` …)
are preserved verbatim as passthrough dicts so an agent can reason over them
without this module having to track every schema addition upstream.
"""

from __future__ import annotations

import re
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rhoai_mcp.utils.errors import RHOAIError


def to_env_key(name: str) -> str:
    """Convert a parameter name to a canonical environment-variable key fragment.

    Uppercases, collapses each run of non-alphanumeric characters to a single
    underscore, and trims leading/trailing underscores — e.g.
    ``ollama.gpu.enabled`` becomes ``OLLAMA_GPU_ENABLED``. This mirrors the
    algorithm the quickstart installers expect.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


class QuickstartModel(BaseModel):
    """Base model: accept camelCase aliases, ignore unknown fields."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


def _parse_yaml(raw: bytes | str) -> dict[str, Any]:
    """Parse a YAML document into a mapping, raising on anything else."""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RHOAIError(f"invalid quickstart YAML: {exc}")
    if not isinstance(data, dict):
        raise RHOAIError("quickstart artifact is not a YAML mapping")
    return data


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RegistryVersion(QuickstartModel):
    """One available version of a quickstart in the registry index."""

    version: str
    status: str | None = None


class QuickstartSummary(QuickstartModel):
    """A single catalog entry from the registry index."""

    name: str
    display_name: str | None = Field(default=None, alias="displayName")
    short_description: str | None = Field(default=None, alias="shortDescription")
    latest_version: str | None = Field(default=None, alias="latestVersion")
    available_versions: list[RegistryVersion] = Field(
        default_factory=list, alias="availableVersions"
    )
    estimated_deployment_time: int | None = Field(default=None, alias="estimatedDeploymentTime")
    tags: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    manifest_repo: str | None = Field(default=None, alias="manifestRepo")
    installer_repo: str | None = Field(default=None, alias="installerRepo")

    def manifest_ref(self, version: str | None = None) -> str:
        """Build the OCI reference for this quickstart's manifest artifact.

        Args:
            version: Explicit version tag; defaults to ``latest_version``.

        Raises:
            RHOAIError: If no manifest repository or resolvable version exists.
        """
        if not self.manifest_repo:
            raise RHOAIError(f"quickstart '{self.name}' has no manifestRepo in the registry")
        resolved = version or self.latest_version
        if not resolved:
            raise RHOAIError(f"quickstart '{self.name}' has no version to resolve a manifest")
        return f"{self.manifest_repo}:{resolved}"

    def to_dict(self) -> dict[str, Any]:
        """Render a catalog-tile dict for tool responses."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "short_description": self.short_description,
            "latest_version": self.latest_version,
            "available_versions": [
                {"version": v.version, "status": v.status} for v in self.available_versions
            ],
            "estimated_deployment_time": self.estimated_deployment_time,
            "tags": self.tags,
            "industries": self.industries,
            "manifest_repo": self.manifest_repo,
            "installer_repo": self.installer_repo,
        }


class QuickstartRegistry(QuickstartModel):
    """The registry index artifact."""

    api_version: str | None = Field(default=None, alias="apiVersion")
    kind: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    quickstarts: list[QuickstartSummary] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, raw: bytes | str) -> QuickstartRegistry:
        """Parse the registry YAML payload into a model."""
        try:
            return cls.model_validate(_parse_yaml(raw))
        except ValidationError as exc:
            raise RHOAIError(f"invalid quickstart registry: {exc}")

    def get(self, name: str) -> QuickstartSummary | None:
        """Return the catalog entry with ``name``, or ``None``."""
        return next((q for q in self.quickstarts if q.name == name), None)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class QuickstartParameter(QuickstartModel):
    """A single configurable parameter (secret or configuration)."""

    name: str
    display_name: str | None = Field(default=None, alias="displayName")
    description: str | None = None
    type: str = "string"
    required: bool = False
    default: Any = None
    env_var: str | None = Field(default=None, alias="envVar")
    llm_guidance: str | None = Field(default=None, alias="llmGuidance")

    def resolved_env_var(self) -> str:
        """The environment variable name the installer expects for this parameter."""
        return self.env_var or f"PARAM_{to_env_key(self.name)}"

    def to_dict(self, *, is_secret: bool) -> dict[str, Any]:
        """Render this parameter for tool responses."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "env_var": self.resolved_env_var(),
            "secret": is_secret,
            "llm_guidance": self.llm_guidance,
        }


class QuickstartParameters(QuickstartModel):
    """The ``parameters`` block: secret and configuration parameters."""

    secrets: list[QuickstartParameter] = Field(default_factory=list)
    configuration: list[QuickstartParameter] = Field(default_factory=list)

    def all(self) -> list[QuickstartParameter]:
        """All parameters, secrets first."""
        return [*self.secrets, *self.configuration]

    def get(self, name: str) -> QuickstartParameter | None:
        """Return the parameter named ``name``, or ``None``."""
        return next((p for p in self.all() if p.name == name), None)

    def is_secret(self, name: str) -> bool:
        """Whether ``name`` is a secret parameter."""
        return any(p.name == name for p in self.secrets)


class QuickstartInstaller(QuickstartModel):
    """The installer container configuration."""

    image: str
    command: list[str] | None = None
    required_env: list[str] = Field(default_factory=list, alias="requiredEnv")


class QuickstartDeployment(QuickstartModel):
    """The ``deployment`` block driving action execution."""

    supported_actions: list[str] = Field(default_factory=list, alias="supportedActions")
    supported_modes: list[str] = Field(default_factory=list, alias="supportedModes")
    installer: QuickstartInstaller
    default_namespace: str | None = Field(default=None, alias="defaultNamespace")


class QuickstartManifest(QuickstartModel):
    """A quickstart's full manifest artifact."""

    api_version: str | None = Field(default=None, alias="apiVersion")
    kind: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    versioning: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    prerequisites: dict[str, Any] | None = None
    deployment: QuickstartDeployment
    parameters: QuickstartParameters = Field(default_factory=QuickstartParameters)
    status: dict[str, Any] | None = None
    access: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    documentation: dict[str, Any] | None = None
    llm_context: dict[str, Any] | None = Field(default=None, alias="llmContext")

    @classmethod
    def from_yaml(cls, raw: bytes | str) -> QuickstartManifest:
        """Parse a manifest YAML payload into a model."""
        try:
            return cls.model_validate(_parse_yaml(raw))
        except ValidationError as exc:
            raise RHOAIError(f"invalid quickstart manifest: {exc}")

    @property
    def name(self) -> str | None:
        """The quickstart identifier."""
        value = self.metadata.get("name")
        return value if isinstance(value, str) else None

    @property
    def version(self) -> str | None:
        """The quickstart release version."""
        value = self.metadata.get("version")
        return value if isinstance(value, str) else None

    @property
    def display_name(self) -> str | None:
        """Human-readable display name."""
        value = self.metadata.get("displayName")
        return value if isinstance(value, str) else None

    def supports_action(self, action: str) -> bool:
        """Whether ``action`` (case-insensitive) is supported."""
        upper = action.upper()
        return any(a.upper() == upper for a in self.deployment.supported_actions)

    def supports_mode(self, mode: str) -> bool:
        """Whether ``mode`` (case-insensitive) is supported."""
        upper = mode.upper()
        return any(m.upper() == upper for m in self.deployment.supported_modes)
