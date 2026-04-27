# Scaffold Plugin

Scaffold a new domain or composite plugin for rhoai-mcp.

## Instructions

You are scaffolding a new plugin. Follow these steps exactly.

### Step 1: Gather Information

Ask the user these questions one at a time (wait for each answer before asking the next):

1. **Plugin type:** Domain (wraps a single K8s resource) or Composite (orchestrates multiple domains)?
2. **Plugin name:** snake_case module name (e.g., `my_feature`)
3. **Resource name:** Singular noun for the K8s resource (e.g., `widget`). Used in tool names like `list_widgets`, `get_widget`.
4. **Custom CRDs?** Does this domain use custom Kubernetes CRDs? If yes, ask for:
   - CRD group (e.g., `my.domain.io`)
   - CRD version (e.g., `v1`)
   - CRD plural (e.g., `widgets`)
   - CRD kind (e.g., `Widget`)
5. **Optional features:** Does it need MCP resources (data endpoints) or prompts (workflow guides)?
6. **Maintainer:** Email for the plugin maintainer (default: `rhoai-mcp@redhat.com`)

Derive these variables from the answers:

- `DOMAIN_NAME` = plugin name (snake_case)
- `DOMAIN_CLASS` = PascalCase version of plugin name (e.g., `my_feature` -> `MyFeature`)
- `DOMAIN_DESCRIPTION` = humanized version (e.g., "My Feature management")
- `RESOURCE_NAME` = resource name (snake_case singular)
- `RESOURCE_CLASS` = PascalCase version of resource name (e.g., `widget` -> `Widget`)
- `CRD_GROUP`, `CRD_VERSION`, `CRD_PLURAL`, `CRD_KIND` = CRD details (if applicable)
- `CRD_KIND_UPPER` = uppercased CRD kind (e.g., `Widget` -> `WIDGET`)
- `MAINTAINER` = maintainer email

### Step 2: Generate Files

Replace ALL `{PLACEHOLDER}` patterns in the code blocks below with the derived values.

---

#### Domain Plugin Files

If the user chose **Domain**, create the following files under `src/rhoai_mcp/domains/{DOMAIN_NAME}/`:

##### `__init__.py`

```python
"""{DOMAIN_DESCRIPTION} domain."""

from rhoai_mcp.domains.{DOMAIN_NAME}.client import {DOMAIN_CLASS}Client
from rhoai_mcp.domains.{DOMAIN_NAME}.models import (
    {RESOURCE_CLASS},
    {RESOURCE_CLASS}Status,
)

__all__ = [
    "{DOMAIN_CLASS}Client",
    "{RESOURCE_CLASS}",
    "{RESOURCE_CLASS}Status",
]
```

##### `models.py`

```python
"""Pydantic models for {DOMAIN_DESCRIPTION}."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import ResourceMetadata


class {RESOURCE_CLASS}Status(str, Enum):
    """Status values for {RESOURCE_NAME} resources."""

    READY = "Ready"
    PENDING = "Pending"
    ERROR = "Error"
    UNKNOWN = "Unknown"


class {RESOURCE_CLASS}(BaseModel):
    """{RESOURCE_CLASS} resource representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Human-readable display name")
    status: {RESOURCE_CLASS}Status = Field(
        {RESOURCE_CLASS}Status.UNKNOWN, description="Resource status"
    )

    # TODO: Add domain-specific fields here.

    @classmethod
    def from_k8s(cls, obj: Any) -> "{RESOURCE_CLASS}":
        """Create from a Kubernetes resource object.

        Args:
            obj: Raw Kubernetes API object (dict-like or ResourceField).

        Returns:
            Parsed {RESOURCE_CLASS} instance.
        """
        meta = obj.metadata
        annotations = meta.annotations or {}

        # TODO: Map Kubernetes status to {RESOURCE_CLASS}Status.
        status = {RESOURCE_CLASS}Status.UNKNOWN

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                meta,
                # For CRD-based domains, use kind="{CRD_KIND}" and
                # api_version="{CRD_GROUP}/{CRD_VERSION}".
                # For core API resources, use the appropriate kind/apiVersion.
                kind="{CRD_KIND}",
                api_version="{CRD_GROUP}/{CRD_VERSION}",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            status=status,
            # TODO: Populate domain-specific fields from obj.
        )
```

**Note:** If the domain does NOT use CRDs, replace the `kind=` and `api_version=` lines with the appropriate core API kind and apiVersion (e.g., `kind="ConfigMap"`, `api_version="v1"`).

##### `client.py`

```python
"""Client for {DOMAIN_DESCRIPTION} operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.{DOMAIN_NAME}.models import {RESOURCE_CLASS}

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class {DOMAIN_CLASS}Client:
    """Client for {DOMAIN_NAME} domain operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_{RESOURCE_NAME}s(self, namespace: str) -> list[dict[str, Any]]:
        """List all {RESOURCE_NAME} resources in a namespace.

        Args:
            namespace: Kubernetes namespace.

        Returns:
            List of formatted resource dicts.
        """
        # --- Option A: CRD-based resources ---
        # from rhoai_mcp.domains.{DOMAIN_NAME}.crds import {DOMAIN_CLASS}CRDs
        # items = self._k8s.list_resources(
        #     {DOMAIN_CLASS}CRDs.{CRD_KIND_UPPER},
        #     namespace=namespace,
        # )

        # --- Option B: Core API resources ---
        # items = self._k8s.core_v1.list_namespaced_config_map(
        #     namespace=namespace, label_selector="your-label=value"
        # ).items

        # TODO: Replace with actual K8s API call.
        items: list[Any] = []

        results = []
        for obj in items:
            resource = {RESOURCE_CLASS}.from_k8s(obj)
            results.append(
                {
                    "name": resource.metadata.name,
                    "display_name": resource.display_name,
                    "status": resource.status.value,
                    "_source": resource.metadata.to_source_dict(),
                }
            )
        return results

    def get_{RESOURCE_NAME}(self, name: str, namespace: str) -> {RESOURCE_CLASS}:
        """Get a single {RESOURCE_NAME} by name.

        Args:
            name: Resource name.
            namespace: Kubernetes namespace.

        Returns:
            Parsed {RESOURCE_CLASS} instance.
        """
        # TODO: Replace with actual K8s API call.
        # CRD: obj = self._k8s.get_resource(CRDs.X, name, namespace)
        # Core: obj = self._k8s.core_v1.read_namespaced_config_map(name=name, namespace=namespace)
        raise NotImplementedError("Replace with K8s API call")
```

##### `tools.py`

```python
"""MCP Tools for {DOMAIN_DESCRIPTION} operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.{DOMAIN_NAME}.client import {DOMAIN_CLASS}Client
from rhoai_mcp.utils.response import PaginatedResponse, paginate

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register {DOMAIN_NAME} tools with the MCP server."""

    @mcp.tool()
    def list_{RESOURCE_NAME}s(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List {RESOURCE_NAME} resources in a namespace.

        Args:
            namespace: The project (namespace) name.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            Paginated list of {RESOURCE_NAME} resources.
        """
        client = {DOMAIN_CLASS}Client(server.k8s)
        all_items = client.list_{RESOURCE_NAME}s(namespace)

        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        paginated, total = paginate(all_items, offset, effective_limit)

        # TODO: Apply verbosity filtering if needed.
        # from rhoai_mcp.utils.response import Verbosity
        # v = Verbosity.from_str(verbosity)
        # items = [ResponseBuilder.your_list_item(item, v) for item in paginated]

        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_{RESOURCE_NAME}(
        name: str,
        namespace: str,
    ) -> dict[str, Any]:
        """Get a single {RESOURCE_NAME} by name.

        Args:
            name: Resource name.
            namespace: The project (namespace) name.

        Returns:
            {RESOURCE_CLASS} details.
        """
        client = {DOMAIN_CLASS}Client(server.k8s)
        resource = client.get_{RESOURCE_NAME}(name, namespace)

        return {
            "name": resource.metadata.name,
            "namespace": resource.metadata.namespace,
            "display_name": resource.display_name,
            "status": resource.status.value,
            "_source": resource.metadata.to_source_dict(),
        }

    # --- Optional: write operations ---
    # Uncomment and implement as needed. Always check is_operation_allowed().
    #
    # @mcp.tool()
    # def create_{RESOURCE_NAME}(name: str, namespace: str, ...) -> dict[str, Any]:
    #     """Create a {RESOURCE_NAME}."""
    #     allowed, reason = server.config.is_operation_allowed("create")
    #     if not allowed:
    #         return {"error": reason}
    #     ...
    #
    # @mcp.tool()
    # def delete_{RESOURCE_NAME}(name: str, namespace: str, confirm: bool = False) -> dict[str, Any]:
    #     """Delete a {RESOURCE_NAME}."""
    #     allowed, reason = server.config.is_operation_allowed("delete")
    #     if not allowed:
    #         return {"error": reason}
    #     if not confirm:
    #         return {"error": "Deletion not confirmed", "message": "Set confirm=True to delete."}
    #     ...
```

##### `crds.py` (only if using custom CRDs)

```python
"""CRD definitions for {DOMAIN_DESCRIPTION}.

Only needed if your domain uses custom Kubernetes CRDs.
Delete this file if your domain uses core API resources only.
"""

from rhoai_mcp.clients.base import CRDDefinition


class {DOMAIN_CLASS}CRDs:
    """CRD definitions for the {DOMAIN_NAME} domain."""

    {CRD_KIND_UPPER} = CRDDefinition(
        group="{CRD_GROUP}",
        version="{CRD_VERSION}",
        plural="{CRD_PLURAL}",
        kind="{CRD_KIND}",
    )

    @classmethod
    def all_crds(cls) -> list[CRDDefinition]:
        """Return all CRD definitions for this domain."""
        return [cls.{CRD_KIND_UPPER}]
```

##### `resources.py` (only if user wants MCP resources)

```python
"""MCP Resources for {DOMAIN_DESCRIPTION}.

Optional -- only needed if your domain exposes MCP resources
(data endpoints that agents can read). Delete this file if not needed.
"""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.{DOMAIN_NAME}.client import {DOMAIN_CLASS}Client

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_resources(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register {DOMAIN_NAME} MCP resources."""

    @mcp.resource("rhoai://{DOMAIN_NAME}/{namespace}/status")
    def {DOMAIN_NAME}_status(namespace: str) -> dict[str, Any]:
        """Get {DOMAIN_NAME} status in a namespace."""
        client = {DOMAIN_CLASS}Client(server.k8s)
        items = client.list_{RESOURCE_NAME}s(namespace)
        return {
            "namespace": namespace,
            "count": len(items),
            "items": items,
        }
```

##### Domain Test Files

Create these files under `tests/domains/{DOMAIN_NAME}/`:

###### `__init__.py`

```python
```

(Empty file)

###### `conftest.py`

```python
"""Test fixtures for {DOMAIN_NAME} domain."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.{DOMAIN_NAME}.client import {DOMAIN_CLASS}Client


@pytest.fixture
def mock_k8s() -> MagicMock:
    """Create a mock K8sClient."""
    return MagicMock()


@pytest.fixture
def client(mock_k8s: MagicMock) -> {DOMAIN_CLASS}Client:
    """Create a {DOMAIN_CLASS}Client with mock K8s."""
    return {DOMAIN_CLASS}Client(mock_k8s)
```

###### `test_models.py`

```python
"""Tests for {DOMAIN_NAME} domain models."""

from unittest.mock import MagicMock

from rhoai_mcp.domains.{DOMAIN_NAME}.models import (
    {RESOURCE_CLASS},
    {RESOURCE_CLASS}Status,
)


class Test{RESOURCE_CLASS}Status:
    """Test {RESOURCE_CLASS}Status enum."""

    def test_ready_value(self) -> None:
        assert {RESOURCE_CLASS}Status.READY.value == "Ready"

    def test_unknown_value(self) -> None:
        assert {RESOURCE_CLASS}Status.UNKNOWN.value == "Unknown"


class Test{RESOURCE_CLASS}:
    """Test {RESOURCE_CLASS} model."""

    def test_from_k8s(self) -> None:
        """Test conversion from Kubernetes object."""
        obj = MagicMock()
        obj.metadata.name = "my-resource"
        obj.metadata.namespace = "test-ns"
        obj.metadata.uid = "uid-123"
        obj.metadata.creation_timestamp = None
        obj.metadata.labels = {}
        obj.metadata.annotations = {"openshift.io/display-name": "My Resource"}

        item = {RESOURCE_CLASS}.from_k8s(obj)

        assert item.metadata.name == "my-resource"
        assert item.display_name == "My Resource"
```

###### `test_client.py`

```python
"""Tests for {DOMAIN_NAME} domain client."""

from unittest.mock import MagicMock

from rhoai_mcp.domains.{DOMAIN_NAME}.client import {DOMAIN_CLASS}Client


class TestList{RESOURCE_CLASS}s:
    """Test {DOMAIN_CLASS}Client.list_{RESOURCE_NAME}s."""

    def test_returns_formatted_list(self) -> None:
        """Returns list of dicts with expected fields."""
        # TODO: Create a mock K8s object matching your resource type.
        obj = MagicMock()
        obj.metadata.name = "item-1"
        obj.metadata.namespace = "test-ns"
        obj.metadata.uid = "uid-1"
        obj.metadata.creation_timestamp = None
        obj.metadata.labels = {}
        obj.metadata.annotations = {}

        k8s = MagicMock()
        # TODO: Mock the correct K8s API call.
        # For CRDs: k8s.list_resources.return_value = [obj]
        # For core API: k8s.core_v1.list_namespaced_X.return_value.items = [obj]

        client = {DOMAIN_CLASS}Client(k8s)
        result = client.list_{RESOURCE_NAME}s("test-ns")

        assert len(result) == 1
        assert result[0]["name"] == "item-1"

    def test_empty_namespace(self) -> None:
        """Returns empty list when no resources exist."""
        k8s = MagicMock()
        # TODO: Mock empty response.

        client = {DOMAIN_CLASS}Client(k8s)
        result = client.list_{RESOURCE_NAME}s("test-ns")

        assert result == []


class TestGet{RESOURCE_CLASS}:
    """Test {DOMAIN_CLASS}Client.get_{RESOURCE_NAME}."""

    def test_returns_model(self) -> None:
        """Returns parsed {RESOURCE_CLASS} model."""
        obj = MagicMock()
        obj.metadata.name = "my-item"
        obj.metadata.namespace = "test-ns"
        obj.metadata.uid = "uid-1"
        obj.metadata.creation_timestamp = None
        obj.metadata.labels = {}
        obj.metadata.annotations = {}

        k8s = MagicMock()
        # TODO: Mock the correct K8s API call.

        client = {DOMAIN_CLASS}Client(k8s)
        item = client.get_{RESOURCE_NAME}("my-item", "test-ns")

        assert item.metadata.name == "my-item"
```

###### `test_tools.py`

```python
"""Tests for {DOMAIN_NAME} domain tools."""

from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.{DOMAIN_NAME}.tools import register_tools


class TestRegisterTools:
    """Test that register_tools registers the expected tools."""

    def test_registers_expected_tools(self) -> None:
        """register_tools should add list and get tools."""
        mcp = FastMCP("test")
        server = MagicMock()
        register_tools(mcp, server)

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "list_{RESOURCE_NAME}s" in tool_names
        assert "get_{RESOURCE_NAME}" in tool_names
```

---

#### Composite Plugin Files

If the user chose **Composite**, create the following files under `src/rhoai_mcp/composites/{DOMAIN_NAME}/`:

##### `__init__.py`

```python
"""{DOMAIN_DESCRIPTION} composite tools."""
```

##### `models.py`

```python
"""Pydantic models for {DOMAIN_DESCRIPTION} composite tools."""

from pydantic import BaseModel, Field


class {RESOURCE_CLASS}Summary(BaseModel):
    """Token-efficient summary of {RESOURCE_NAME} resources."""

    total: int = Field(0, description="Total resource count")
    status_summary: str = Field("", description="Compact status (e.g., '3/5 ready')")

    # TODO: Add summary fields specific to your composite.
```

##### `tools.py`

```python
"""MCP Tools for {DOMAIN_DESCRIPTION} composite operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register {DOMAIN_NAME} composite tools with the MCP server."""

    @mcp.tool()
    def {DOMAIN_NAME}_summary(
        namespace: str,
    ) -> dict[str, Any]:
        """Get a compact {DOMAIN_NAME} summary for a namespace.

        Aggregates data from multiple domain clients into a
        token-efficient response for AI agents.

        Args:
            namespace: The project (namespace) name.

        Returns:
            Compact summary with counts and status.
        """
        # Import domain clients as needed:
        # from rhoai_mcp.domains.X.client import XClient
        # from rhoai_mcp.domains.Y.client import YClient

        k8s = server.k8s

        # TODO: Aggregate data from domain clients.
        # x_client = XClient(k8s)
        # y_client = YClient(k8s)
        # items = x_client.list_items(namespace)

        return {
            "namespace": namespace,
            "total": 0,
            "status_summary": "0/0 ready",
            # TODO: Add aggregated fields.
        }
```

##### Composite Test Files

Create these files under `tests/composites/{DOMAIN_NAME}/`:

###### `__init__.py`

```python
```

(Empty file)

###### `test_tools.py`

```python
"""Tests for {DOMAIN_NAME} composite tools."""

from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.composites.{DOMAIN_NAME}.tools import register_tools


class TestRegisterTools:
    """Test that register_tools registers the expected tools."""

    def test_registers_expected_tools(self) -> None:
        """register_tools should add the summary tool."""
        mcp = FastMCP("test")
        server = MagicMock()
        register_tools(mcp, server)

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "{DOMAIN_NAME}_summary" in tool_names
```

---

### Step 3: Registry Entry

After generating all files, add the plugin to the appropriate registry.

#### For Domain Plugins

Add to `src/rhoai_mcp/domains/registry.py`:

1. Add a new plugin class (following the pattern of existing plugins like `StoragePlugin`).
2. Add an instance to the `get_core_plugins()` list.

The class should look like this:

```python
class {DOMAIN_CLASS}Plugin(BasePlugin):
    """Plugin for {DOMAIN_DESCRIPTION}."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="{DOMAIN_NAME}",
                version="0.1.0",
                description="{DOMAIN_DESCRIPTION}",
                maintainer="{MAINTAINER}",
                requires_crds=[],  # Add CRD kind names if applicable, e.g., ["{CRD_KIND}"]
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.{DOMAIN_NAME}.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "{DOMAIN_DESCRIPTION} uses core Kubernetes API"
```

If the domain uses CRDs, also add:

```python
    @hookimpl
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        from rhoai_mcp.domains.{DOMAIN_NAME}.crds import {DOMAIN_CLASS}CRDs

        return {DOMAIN_CLASS}CRDs.all_crds()
```

And update `requires_crds=["{CRD_KIND}"]` in the metadata.

If the domain has MCP resources, also add:

```python
    @hookimpl
    def rhoai_register_resources(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.{DOMAIN_NAME}.resources import register_resources

        register_resources(mcp, server)
```

Update the health check message accordingly:
- CRD-based: `"{DOMAIN_DESCRIPTION} requires {CRD_KIND} CRD"`
- Core API: `"{DOMAIN_DESCRIPTION} uses core Kubernetes API"`

Then add `{DOMAIN_CLASS}Plugin()` to the list in `get_core_plugins()`.

#### For Composite Plugins

Add to `src/rhoai_mcp/composites/registry.py`:

```python
class {DOMAIN_CLASS}CompositesPlugin(BasePlugin):
    """Plugin for {DOMAIN_DESCRIPTION} composite tools."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="{DOMAIN_NAME}-composites",
                version="1.0.0",
                description="{DOMAIN_DESCRIPTION}",
                maintainer="{MAINTAINER}",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.composites.{DOMAIN_NAME}.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "{DOMAIN_DESCRIPTION} composites use core domain clients"
```

Then add `{DOMAIN_CLASS}CompositesPlugin()` to the list in `get_composite_plugins()`.

### Step 4: Verify

Run lint and type checking on the generated files:

**For Domain plugins:**

```bash
uv run ruff check src/rhoai_mcp/domains/{DOMAIN_NAME}/
uv run ruff format --check src/rhoai_mcp/domains/{DOMAIN_NAME}/
uv run mypy src/rhoai_mcp/domains/{DOMAIN_NAME}/
```

**For Composite plugins:**

```bash
uv run ruff check src/rhoai_mcp/composites/{DOMAIN_NAME}/
uv run ruff format --check src/rhoai_mcp/composites/{DOMAIN_NAME}/
uv run mypy src/rhoai_mcp/composites/{DOMAIN_NAME}/
```

Fix any issues found. Common fixes:
- Formatting issues (run `uv run ruff format` on the files)

### Step 5: Summary

After everything is generated and verified, print:

1. **Files created** -- list every file with its full path
2. **Registry changes** -- which registry file was modified and what was added
3. **Next steps** for the developer:
   - Implement K8s client methods in `client.py` (replace TODO placeholders)
   - Add domain-specific fields to `models.py`
   - Update `from_k8s()` to map real Kubernetes status values
   - Write comprehensive tests (the stubs are starting points)
   - If using CRDs, uncomment the CRD-based code paths in `client.py`
   - Run the full test suite: `make test`
4. **Documentation references:**
   - Domain guide: `docs/contributing/adding-a-domain.md`
   - Composite guide: `docs/contributing/adding-a-composite.md`
   - Example domain: `src/rhoai_mcp/domains/_example/`
   - Templates: `docs/contributing/templates/`
