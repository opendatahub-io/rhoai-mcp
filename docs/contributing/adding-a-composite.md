# Adding a Composite Plugin

## 1. Overview

A **composite plugin** orchestrates multiple domain clients to provide cross-cutting or token-efficient tools. Composites live under `src/rhoai_mcp/composites/<name>/` and aggregate data that would otherwise require multiple tool calls from the AI agent.

Existing composites:

| Composite | Purpose |
|-----------|---------|
| [`cluster/`](../../src/rhoai_mcp/composites/cluster/) | Cluster and project summaries, resource status, diagnostics |
| [`training/`](../../src/rhoai_mcp/composites/training/) | Training workflow orchestration (planning, storage, unified ops) |
| [`meta/`](../../src/rhoai_mcp/composites/meta/) | Tool discovery and workflow guidance for agents |

## 2. When to use

Use the decision table below to determine whether you need a domain or composite plugin:

| Scenario | Plugin type |
|----------|-------------|
| Wrapping a single K8s resource type with CRUD operations | **Domain** |
| Tool needs data from 2+ domain clients | **Composite** |
| Aggregating multiple resources into a token-efficient summary | **Composite** |
| Cross-domain workflow (e.g., validate storage + credentials + runtime) | **Composite** |
| Exposing a single resource's events, logs, or status | **Domain** |
| Providing a "one-call" exploration or diagnostic tool | **Composite** |

**Rule of thumb**: If your tool imports from more than one `domains/*/client.py`, it belongs in a composite.

## 3. Quick Start

Copy the template files from `docs/contributing/templates/composite/` and replace placeholders:

```bash
COMPOSITE=my_composite
mkdir -p src/rhoai_mcp/composites/$COMPOSITE
for f in docs/contributing/templates/composite/*.tpl; do
  cp "$f" "src/rhoai_mcp/composites/$COMPOSITE/$(basename "${f%.tpl}")"
done
```

Tests:

```bash
mkdir -p tests/composites/$COMPOSITE
for f in docs/contributing/templates/composite/tests/*.tpl; do
  cp "$f" "tests/composites/$COMPOSITE/$(basename "${f%.tpl}")"
done
```

Or use the Claude Code `/scaffold-plugin` command for interactive generation.

See the [domain guide](adding-a-domain.md#placeholder-reference) for the full placeholder reference.

## 4. Walkthrough

### Directory structure

Composites are simpler than domains:

```
src/rhoai_mcp/composites/<name>/
├── __init__.py          # Public API exports
├── models.py            # Summary/response models (optional)
└── tools.py             # MCP tool implementations
```

No `client.py` or `crds.py` since composites use existing domain clients.

### `models.py`

Define summary models that provide token-efficient representations. These are typically smaller than domain models, focusing on counts and status strings rather than full resource details.

```python
"""Models for my composite tools."""

from pydantic import BaseModel, Field


class FeatureSummary(BaseModel):
    """Compact summary optimized for AI agent context windows."""

    namespace: str = Field(..., description="Project namespace")
    widget_count: int = Field(0, description="Total widgets")
    active_widgets: str = Field("0/0", description="Active/total widgets")
    related_storage: int = Field(0, description="Associated PVCs")
```

See: [`cluster/models.py`](../../src/rhoai_mcp/composites/cluster/models.py)

### `tools.py`

The `register_tools()` function follows the same pattern as domain tools, but imports from multiple domain clients. Use lazy imports inside the tool functions to avoid loading unused domains.

```python
"""MCP Tools for my composite operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register my composite tools with the MCP server."""

    @mcp.tool()
    def feature_summary(namespace: str) -> dict[str, Any]:
        """Get a compact feature summary for a project.

        Aggregates widget and storage data into a single
        token-efficient response.

        Args:
            namespace: The project (namespace) name.

        Returns:
            Summary with widget counts and storage info.
        """
        from rhoai_mcp.domains.my_feature.client import MyFeatureClient
        from rhoai_mcp.domains.storage.client import StorageClient

        k8s = server.k8s
        widget_client = MyFeatureClient(k8s)
        storage_client = StorageClient(k8s)

        widgets = widget_client.list_widgets(namespace)
        storage = storage_client.list_storage(namespace)

        active = sum(1 for w in widgets if w.get("status") == "Active")

        return {
            "namespace": namespace,
            "widget_count": len(widgets),
            "active_widgets": f"{active}/{len(widgets)}",
            "related_storage": len(storage),
        }
```

See: [`cluster/tools.py`](../../src/rhoai_mcp/composites/cluster/tools.py)

### Plugin class in `composites/registry.py`

Add your plugin class to [`src/rhoai_mcp/composites/registry.py`](../../src/rhoai_mcp/composites/registry.py). The pattern is identical to domain plugins:

```python
class MyCompositesPlugin(BasePlugin):
    """Plugin for my composite tools."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="my-composites",
                version="1.0.0",
                description="My composite summary tools",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.composites.my_composite.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:
        return True, "My composites use existing domain clients"
```

Then add the instance to `get_composite_plugins()`:

```python
def get_composite_plugins() -> list[BasePlugin]:
    return [
        # ... existing plugins ...
        MyCompositesPlugin(),
    ]
```

### Health checks

Composite health checks typically return `True` directly, since they rely on domain clients that have their own health checks. Only override the default if your composite needs external services (see `NeuralNavCompositesPlugin` for an example).

## 5. Checklist

- [ ] All files created: `__init__.py`, `tools.py`, optional `models.py`
- [ ] Plugin class added to `composites/registry.py` with `@hookimpl` decorators
- [ ] Plugin instance added to `get_composite_plugins()`
- [ ] All tests pass: `uv run pytest tests/composites/<name>/ -q`
- [ ] Lint clean: `uv run ruff check src/rhoai_mcp/composites/<name>/`
- [ ] Type check clean: `uv run mypy src/rhoai_mcp/composites/<name>/`
- [ ] Tools documented with docstrings (these are shown to AI agents)
- [ ] PR references this guide
