"""Example domain — a working blueprint for contributors."""

from rhoai_mcp.domains._example.client import ExampleClient
from rhoai_mcp.domains._example.models import ExampleItem, ExampleStatus

__all__ = ["ExampleClient", "ExampleItem", "ExampleStatus"]
