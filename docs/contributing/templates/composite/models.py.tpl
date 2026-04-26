"""Pydantic models for {{DOMAIN_DESCRIPTION}} composite tools."""

from pydantic import BaseModel, Field


class {{RESOURCE_CLASS}}Summary(BaseModel):
    """Token-efficient summary of {{RESOURCE_NAME}} resources."""

    total: int = Field(0, description="Total resource count")
    status_summary: str = Field("", description="Compact status (e.g., '3/5 ready')")

    # TODO: Add summary fields specific to your composite.
