# Adding a Domain Plugin

## 1. Overview

A **domain plugin** wraps a single Kubernetes resource type (CRD or core API) and exposes it through MCP tools. Each domain gets its own directory under `src/rhoai_mcp/domains/` and follows a consistent file layout.

Use a domain plugin when you are adding CRUD operations for a specific Kubernetes resource type. If your tool needs to aggregate data from multiple domains, create a [composite plugin](adding-a-composite.md) instead.

Before starting, read through the `_example` domain at [`src/rhoai_mcp/domains/_example/`](../../src/rhoai_mcp/domains/_example/). It is a fully working domain backed by ConfigMaps (no CRDs required) that demonstrates every pattern described in this guide.

## 2. Prerequisites

- Development environment set up per [CONTRIBUTING.md](../../CONTRIBUTING.md)
- Python 3.10+, `uv` package manager
- Familiarity with:
  - [pluggy](https://pluggy.readthedocs.io/) hook specifications and implementations
  - Kubernetes Python client patterns (core API and dynamic client)
  - Pydantic v2 models

Read the `_example` domain source before starting. It is the single best reference.

## 3. Quick Start

### Option A: Copy templates

Copy the template files from `docs/contributing/templates/domain/` into your new domain directory and replace the placeholders:

```bash
DOMAIN=my_feature
mkdir -p src/rhoai_mcp/domains/$DOMAIN
for f in docs/contributing/templates/domain/*.tpl; do
  cp "$f" "src/rhoai_mcp/domains/$DOMAIN/$(basename "${f%.tpl}")"
done
```

Then do the same for tests:

```bash
mkdir -p tests/domains/$DOMAIN
for f in docs/contributing/templates/domain/tests/*.tpl; do
  cp "$f" "tests/domains/$DOMAIN/$(basename "${f%.tpl}")"
done
```

### Option B: Claude Code command

Run `/scaffold-plugin` in Claude Code for interactive generation.

### Placeholder reference

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{{DOMAIN_NAME}}` | snake_case module name | `my_feature` |
| `{{DOMAIN_CLASS}}` | PascalCase class prefix | `MyFeature` |
| `{{DOMAIN_DESCRIPTION}}` | Human-readable description | `My Feature management` |
| `{{RESOURCE_NAME}}` | Singular resource name for tool naming | `widget` |
| `{{RESOURCE_CLASS}}` | PascalCase model class name | `Widget` |
| `{{CRD_GROUP}}` | CRD API group (CRD-based domains only) | `example.io` |
| `{{CRD_VERSION}}` | CRD API version (CRD-based domains only) | `v1` |
| `{{CRD_PLURAL}}` | CRD plural name (CRD-based domains only) | `widgets` |
| `{{CRD_KIND}}` | CRD kind (CRD-based domains only) | `Widget` |
| `{{CRD_KIND_UPPER}}` | Uppercased CRD kind (CRD-based domains only) | `WIDGET` |

## 4. Step-by-step walkthrough

### Directory structure

Every domain module follows this layout:

```
src/rhoai_mcp/domains/<name>/
├── __init__.py          # Public API exports
├── models.py            # Pydantic models
├── client.py            # Kubernetes client
├── tools.py             # MCP tool implementations
├── crds.py              # CRD definitions (optional, CRD-based domains only)
└── resources.py         # MCP resource endpoints (optional)
```

Tests mirror the source:

```
tests/domains/<name>/
├── __init__.py
├── test_models.py
├── test_client.py
└── test_tools.py
```

### `__init__.py`

Export the public API for your domain. This keeps imports clean for other modules.

```python
"""My Feature domain — Widget management."""

from rhoai_mcp.domains.my_feature.client import MyFeatureClient
from rhoai_mcp.domains.my_feature.models import Widget, WidgetStatus

__all__ = ["MyFeatureClient", "Widget", "WidgetStatus"]
```

See: [`_example/__init__.py`](../../src/rhoai_mcp/domains/_example/__init__.py)

### `models.py`

Define Pydantic models that represent your Kubernetes resource. Key patterns:

- **Status enum**: String enum inheriting from `str, Enum` for JSON serialization.
- **`from_k8s()` factory method**: Classmethod that converts a raw Kubernetes object to your model.
- **`ResourceMetadata`**: Import from `rhoai_mcp.models.common` for consistent metadata handling.

```python
"""Pydantic models for Widget resources."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import ResourceMetadata


class WidgetStatus(str, Enum):
    """Status values for widgets."""

    ACTIVE = "Active"
    INACTIVE = "Inactive"


class Widget(BaseModel):
    """A Widget resource."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Human-readable name")
    status: WidgetStatus = Field(WidgetStatus.ACTIVE, description="Widget status")

    @classmethod
    def from_k8s(cls, obj: Any) -> "Widget":
        """Create from a Kubernetes object."""
        meta = obj.metadata
        annotations = meta.annotations or {}

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                meta,
                kind="Widget",
                api_version="example.io/v1",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            status=WidgetStatus.ACTIVE,
        )
```

See: [`_example/models.py`](../../src/rhoai_mcp/domains/_example/models.py), [`storage/models.py`](../../src/rhoai_mcp/domains/storage/models.py)

### `client.py`

The client encapsulates all Kubernetes API calls for your resource type. Key patterns:

- **Constructor injection**: Accept `K8sClient` in `__init__` using a `TYPE_CHECKING` guard for the import.
- **List method**: Return `list[dict[str, Any]]` with a `_source` dict in each item.
- **Get method**: Return your Pydantic model.
- **Create/delete methods**: Optional, for write operations.

```python
"""Client for Widget operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.my_feature.models import Widget

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class MyFeatureClient:
    """Client for Widget operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_widgets(self, namespace: str) -> list[dict[str, Any]]:
        """List all widgets in a namespace."""
        # For core API resources:
        raw_list = self._k8s.core_v1.list_namespaced_config_map(
            namespace=namespace, label_selector="app=widget"
        )
        results = []
        for obj in raw_list.items:
            item = Widget.from_k8s(obj)
            results.append({
                "name": item.metadata.name,
                "display_name": item.display_name,
                "status": item.status.value,
                "_source": item.metadata.to_source_dict(),
            })
        return results

    def get_widget(self, name: str, namespace: str) -> Widget:
        """Get a single widget by name."""
        obj = self._k8s.core_v1.read_namespaced_config_map(
            name=name, namespace=namespace
        )
        return Widget.from_k8s(obj)
```

For CRD-based domains, use `self._k8s.get_resource(crd_definition)` or `self._k8s.list_resources(crd_definition, namespace)` from the dynamic client instead.

The `_source` dict is included in list results so AI agents can trace responses back to exact Kubernetes objects. It contains `kind`, `api_version`, `name`, `namespace`, and `uid`.

See: [`_example/client.py`](../../src/rhoai_mcp/domains/_example/client.py), [`storage/client.py`](../../src/rhoai_mcp/domains/storage/client.py)

### `tools.py`

This is where MCP tools are registered. Key patterns:

- **`register_tools(mcp, server)` function**: The single entry point called by the plugin class.
- **`@mcp.tool()` decorator**: Registers each function as an MCP tool. The docstring is shown to AI agents, so write it carefully.
- **Pagination**: Use `paginate()` and `PaginatedResponse.build()` from `rhoai_mcp.utils.response`.
- **Verbosity**: Use `Verbosity.from_str()` to support `minimal`/`standard`/`full` detail levels.
- **Operation guards**: Call `server.config.is_operation_allowed("create")` before write operations.

```python
"""MCP Tools for Widget operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.my_feature.client import MyFeatureClient
from rhoai_mcp.utils.response import PaginatedResponse, paginate

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register widget tools with the MCP server."""

    @mcp.tool()
    def list_widgets(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List widgets in a Data Science Project.

        Args:
            namespace: The project (namespace) name.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).

        Returns:
            Paginated list of widgets.
        """
        client = MyFeatureClient(server.k8s)
        all_items = client.list_widgets(namespace)

        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        paginated, total = paginate(all_items, offset, effective_limit)
        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_widget(
        name: str,
        namespace: str,
    ) -> dict[str, Any]:
        """Get a single widget by name.

        Args:
            name: The widget name.
            namespace: The project (namespace) name.

        Returns:
            Widget details.
        """
        client = MyFeatureClient(server.k8s)
        item = client.get_widget(name, namespace)

        return {
            "name": item.metadata.name,
            "namespace": item.metadata.namespace,
            "display_name": item.display_name,
            "status": item.status.value,
            "_source": item.metadata.to_source_dict(),
        }
```

For write operations (create, delete), always guard with `server.config.is_operation_allowed()`:

```python
    @mcp.tool()
    def create_widget(name: str, namespace: str) -> dict[str, Any]:
        """Create a widget."""
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}
        # ... create logic ...

    @mcp.tool()
    def delete_widget(name: str, namespace: str, confirm: bool = False) -> dict[str, Any]:
        """Delete a widget."""
        allowed, reason = server.config.is_operation_allowed("delete")
        if not allowed:
            return {"error": reason}
        if not confirm:
            return {"error": "Deletion not confirmed", "message": "Set confirm=True."}
        # ... delete logic ...
```

See: [`_example/tools.py`](../../src/rhoai_mcp/domains/_example/tools.py), [`storage/tools.py`](../../src/rhoai_mcp/domains/storage/tools.py)

### `crds.py` (optional)

Only needed for CRD-based domains. Define the CRD metadata so the server can verify CRD availability during startup.

```python
"""CRD definitions for Widget resources."""

from rhoai_mcp.clients.base import CRDDefinition


class WidgetCRDs:
    """Widget CRD definitions."""

    WIDGET = CRDDefinition(
        group="example.io",
        version="v1",
        plural="widgets",
        kind="Widget",
    )

    @classmethod
    def all_crds(cls) -> list[CRDDefinition]:
        """Return all CRD definitions."""
        return [cls.WIDGET]
```

See: [`training/crds.py`](../../src/rhoai_mcp/domains/training/crds.py)

### `resources.py` (optional)

Use MCP resources for data endpoints that expose static or semi-static data (e.g., configuration, status pages). Register them with `@mcp.resource()`.

### Plugin class in `domains/registry.py`

Add your plugin class to [`src/rhoai_mcp/domains/registry.py`](../../src/rhoai_mcp/domains/registry.py). This connects your domain to the server via pluggy hooks.

```python
class MyFeaturePlugin(BasePlugin):
    """Plugin for Widget management."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="my_feature",
                version="0.1.0",
                description="Widget management",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=["Widget"],  # empty list for core API domains
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.my_feature.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        from rhoai_mcp.domains.my_feature.crds import WidgetCRDs

        return WidgetCRDs.all_crds()
```

Then add the instance to `get_core_plugins()`:

```python
def get_core_plugins() -> list[BasePlugin]:
    plugins = [
        # ... existing plugins ...
        MyFeaturePlugin(),
    ]
    return plugins
```

Key points:

- **Lazy imports**: Import `register_tools` and CRD classes inside the hook methods, not at module level. This keeps startup fast and avoids circular imports.
- **`requires_crds`**: List the CRD kinds your domain needs. For core API domains (PVCs, ConfigMaps, Secrets), use an empty list.
- **Optional hooks**: Only implement the hooks you need. `BasePlugin` provides no-op defaults for all hooks.

### Health checks

The `BasePlugin` base class provides a default `rhoai_health_check` implementation that verifies all CRDs listed in `requires_crds` are accessible in the cluster. This works automatically if you implement `rhoai_get_crd_definitions`.

For core API domains (no CRDs), override the health check to return a simple success:

```python
    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:
        return True, "My Feature uses core Kubernetes API"
```

### Tests

Mirror the source structure under `tests/domains/<name>/`. The standard test approach:

1. **Mock the K8s client** with `unittest.mock.MagicMock`.
2. **Test tools** by calling them through `FastMCP._tool_manager.call_tool()`.
3. **Test models** by constructing mock K8s objects and calling `from_k8s()`.
4. **Test the client** by mocking the underlying K8s API calls.

```python
"""Tests for widget tools."""

from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.my_feature.tools import register_tools


class TestListWidgets:
    async def test_returns_paginated_response(self) -> None:
        mcp = FastMCP("test")
        server = MagicMock()
        server.config.max_list_limit = 100
        server.config.default_list_limit = None
        server.k8s = MagicMock()

        # Mock K8s response
        obj = MagicMock()
        obj.metadata.name = "widget-1"
        obj.metadata.namespace = "ns"
        obj.metadata.uid = "uid-1"
        obj.metadata.creation_timestamp = None
        obj.metadata.labels = {}
        obj.metadata.annotations = {}
        server.k8s.core_v1.list_namespaced_config_map.return_value.items = [obj]

        register_tools(mcp, server)

        result = await mcp._tool_manager.call_tool(
            "list_widgets", {"namespace": "ns"}
        )
        assert result["total"] == 1
        assert len(result["items"]) == 1
```

See: [`tests/domains/_example/test_tools.py`](../../tests/domains/_example/test_tools.py)

## 5. Conventions

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Directory/module | snake_case | `my_feature/` |
| Class | PascalCase | `MyFeatureClient`, `Widget` |
| List tool | `list_<resource>s` | `list_widgets` |
| Get tool | `get_<resource>` | `get_widget` |
| Create tool | `create_<resource>` | `create_widget` |
| Delete tool | `delete_<resource>` | `delete_widget` |

### Imports

- Use `TYPE_CHECKING` guard for `K8sClient` and `RHOAIServer` imports to avoid circular dependencies.
- Use lazy imports inside plugin hook methods (import inside the function body).

### Type hints

- Use `X | None` instead of `Optional[X]`.
- Use `dict[str, Any]` for tool return types consistently.
- All public functions must have type hints (enforced by mypy).

### Error handling

- Tools return `{"error": reason}` dicts on failure. Do not raise exceptions from tool functions.
- The client layer may raise exceptions (e.g., `NotFoundError`), which the tool layer catches and converts to error dicts.

### `_source` dict

Include `_source` in list and detail responses so AI agents can trace results back to Kubernetes objects:

```python
"_source": item.metadata.to_source_dict()
# Produces: {"kind": "Widget", "api_version": "example.io/v1",
#            "name": "w1", "namespace": "ns", "uid": "..."}
```

## 6. Checklist

- [ ] All files created: `__init__.py`, `models.py`, `client.py`, `tools.py`, optional `crds.py`/`resources.py`
- [ ] Plugin class added to `domains/registry.py` with `@hookimpl` decorators
- [ ] All tests pass: `uv run pytest tests/domains/<name>/ -q`
- [ ] Lint clean: `uv run ruff check src/rhoai_mcp/domains/<name>/`
- [ ] Type check clean: `uv run mypy src/rhoai_mcp/domains/<name>/`
- [ ] Tools documented with docstrings (these are shown to AI agents)
- [ ] PR references this guide
