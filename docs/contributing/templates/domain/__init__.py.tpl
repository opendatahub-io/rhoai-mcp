"""{{DOMAIN_DESCRIPTION}} domain."""

from rhoai_mcp.domains.{{DOMAIN_NAME}}.client import {{DOMAIN_CLASS}}Client
from rhoai_mcp.domains.{{DOMAIN_NAME}}.models import (
    {{RESOURCE_CLASS}},
    {{RESOURCE_CLASS}}Status,
)

__all__ = [
    "{{DOMAIN_CLASS}}Client",
    "{{RESOURCE_CLASS}}",
    "{{RESOURCE_CLASS}}Status",
]
