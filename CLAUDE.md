# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RHOAI MCP Server is an MCP (Model Context Protocol) server that enables AI agents to interact with Red Hat OpenShift AI (RHOAI) environments. It provides programmatic access to RHOAI features (projects, workbenches, model serving, pipelines, data connections, storage) through a hybrid architecture with core domain modules and external plugins.

## Build and Development Commands

```bash
# Setup development environment
uv sync --all-packages          # Install all workspace packages
make dev                         # Alias for setup

# Run the server locally
uv run rhoai-mcp                 # Default (stdio transport)
uv run rhoai-mcp --transport sse # HTTP transport

# Testing
make test                        # All tests
make test-unit                   # Unit tests only (packages/*/tests)
make test-integration            # Integration tests (tests/integration)
make test-package PKG=core       # Single package tests

# Code quality
make lint                        # ruff check
make format                      # ruff format + fix
make typecheck                   # mypy
make check                       # lint + typecheck

# Container operations
make build                       # Build container image
make run-http                    # Run with SSE transport
make run-stdio                   # Run with stdio transport
make run-dev                     # Debug logging + dangerous ops enabled
```

## Architecture

### UV Workspace Structure

This is a monorepo managed by `uv` with two packages in `packages/`:

- **core**: Contains all core functionality including:
  - Plugin system, K8s client, configuration, server entry point (`rhoai-mcp` command)
  - Domain modules (in `rhoai_mcp_core.domains.*`):
    - `projects`: Data Science Project (namespace) management
    - `notebooks`: Kubeflow Notebook/Workbench management
    - `inference`: KServe InferenceService model serving
    - `pipelines`: Data Science Pipelines (DSPA)
    - `connections`: S3 data connections
    - `storage`: PersistentVolumeClaim management
- **training**: External plugin for Kubeflow Training Operator integration (discovered via entry points)

### Hybrid Plugin Architecture

The server uses a hybrid architecture:

1. **Core Domain Modules** (in `rhoai_mcp_core.domains/`): Registered directly with the server, no entry point discovery needed. Each domain module has:
   - `client.py`: K8s resource client
   - `models.py`: Pydantic models
   - `tools.py`: MCP tool implementations
   - `crds.py`: CRD definitions (if applicable)

2. **External Plugins** (like `training`): Discovered via Python entry points in the `rhoai_mcp.plugins` group. Each plugin implements `RHOAIMCPPlugin` protocol (defined in `packages/core/src/rhoai_mcp_core/plugin.py`):

```python
# Entry point declaration in pyproject.toml:
[project.entry-points."rhoai_mcp.plugins"]
training = "rhoai_mcp_training.plugin:create_plugin"
```

Plugin interface:
- `metadata`: Returns `PluginMetadata` (name, version, required CRDs)
- `register_tools()`: Registers MCP tools with FastMCP
- `register_resources()`: Registers MCP resources
- `get_crd_definitions()`: Returns CRD definitions for the plugin
- `health_check()`: Verifies required CRDs are available (graceful degradation)

### Domain Module Structure

Each domain module in `packages/core/src/rhoai_mcp_core/domains/` follows this layout:
```
domains/<name>/
├── __init__.py
├── client.py            # K8s resource client
├── models.py            # Pydantic models
├── tools.py             # MCP tool implementations
├── crds.py              # CRD definitions (if applicable)
└── resources.py         # MCP resources (only projects has this)
```

The domain registry (`domains/registry.py`) defines all core domains and provides them to the server for direct registration.

### Configuration

Environment variables use `RHOAI_MCP_` prefix. Key settings:
- `AUTH_MODE`: auto | kubeconfig
- `TRANSPORT`: stdio | sse | streamable-http
- `KUBECONFIG_PATH`, `KUBECONFIG_CONTEXT`: For kubeconfig auth
- `ENABLE_DANGEROUS_OPERATIONS`: Enable delete operations
- `READ_ONLY_MODE`: Disable all writes

### Key Dependencies

- `mcp>=1.0.0`: Model Context Protocol (FastMCP)
- `kubernetes>=28.1.0`: K8s Python client
- `pydantic>=2.0.0`: Data validation and settings

## Development Principles

### Test-Driven Development

Follow TDD for all code changes:

1. **Write tests first**: Before implementing any feature or fix, write failing tests that define the expected behavior
2. **Red-Green-Refactor**: Run tests to see them fail (red), write minimal code to pass (green), then refactor while keeping tests green
3. **Test coverage**: All new code must have corresponding tests; run `make test` before committing
4. **Test types**: Unit tests go in `packages/*/tests/`, integration tests in `tests/integration/`

### Simplicity and Maintainability

Favor simple, maintainable solutions at all times:

- **KISS**: Choose the simplest solution that works; avoid premature optimization or over-abstraction
- **Single responsibility**: Each function, class, and module should do one thing well
- **Explicit over implicit**: Code should be self-documenting; avoid magic or clever tricks
- **Minimal dependencies**: Only add dependencies when truly necessary
- **Delete dead code**: Remove unused code rather than commenting it out
- **Small functions**: Keep functions short and focused; if a function needs extensive comments, it should be split

### Idiomatic Python

Write Pythonic code that follows community conventions:

- Use list/dict/set comprehensions where they improve readability
- Prefer `pathlib.Path` over `os.path` for file operations
- Use context managers (`with` statements) for resource management
- Leverage dataclasses and Pydantic models for structured data
- Use type hints consistently (required by mypy)
- Follow PEP 8 naming: `snake_case` for functions/variables, `PascalCase` for classes
- Use `typing` module for complex types; prefer `|` union syntax (Python 3.10+)
- Prefer raising specific exceptions over generic ones
- Use f-strings for string formatting

## Code Style

- Python 3.10+, line length 100
- Ruff for linting/formatting (isort included)
- Mypy with `disallow_untyped_defs=true`
- Pytest with `asyncio_mode = "auto"`
