"""Pydantic models for Quickstarts domain."""

from enum import Enum

from pydantic import BaseModel, Field


class DeploymentMethod(str, Enum):
    """Detected deployment method for a quickstart."""

    HELM = "helm"
    KUSTOMIZE = "kustomize"
    MANIFESTS = "manifests"
    UNKNOWN = "unknown"


class Quickstart(BaseModel):
    """Metadata for a Red Hat AI Quickstart."""

    name: str = Field(description="Unique identifier for the quickstart")
    display_name: str = Field(description="Human-readable name")
    description: str = Field(description="Brief description of the quickstart")
    repo_url: str = Field(description="GitHub repository URL")
    tags: list[str] = Field(default_factory=list, description="Tags/categories")
    git_ref: str = Field(default="main", description="Git ref (tag, branch, or commit SHA) to pin")


class QuickstartReadme(BaseModel):
    """README content from a quickstart repository."""

    quickstart_name: str = Field(description="Name of the quickstart")
    content: str = Field(description="Raw README.md content")
    repo_url: str = Field(description="Source repository URL")


class DeploymentResult(BaseModel):
    """Result of a quickstart deployment operation."""

    quickstart_name: str = Field(description="Name of the quickstart")
    namespace: str = Field(description="Target namespace")
    method: DeploymentMethod = Field(description="Deployment method used")
    command: str = Field(description="Command executed or to be executed")
    dry_run: bool = Field(description="Whether this was a dry run")
    success: bool = Field(default=False, description="Whether deployment succeeded")
    stdout: str = Field(default="", description="Command stdout")
    stderr: str = Field(default="", description="Command stderr")
    error: str | None = Field(default=None, description="Error message if failed")
