"""Local mock MCP launch configuration, independent of optional agent SDKs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class LocalMCPConfig(TypedDict):
    type: str
    command: str
    args: list[str]
    env: dict[str, str]


def build_mcp_servers() -> dict[str, LocalMCPConfig]:
    """Use the embedded planner and bundled data with mock cluster state."""
    environment = {
        "RHOAI_MCP_PLANNER_MODE": "local",
        "RHOAI_MCP_PLANNER_MODEL_CATALOG_URL": "",
        "PLANNER_DETECT_CLUSTER_GPUS": "false",
    }
    # The MCP Python SDK inherits only a small system environment. Forward this
    # override explicitly so schema discovery and Claude see the same plugin set.
    plugins = os.environ.get("RHOAI_MCP_ENABLED_PLUGINS")
    if plugins is not None:
        environment["RHOAI_MCP_ENABLED_PLUGINS"] = plugins
    return {
        "rhoai-mcp": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "rhoai_mcp", "--transport", "stdio", "--mock-cluster"],
            "env": environment,
        }
    }
