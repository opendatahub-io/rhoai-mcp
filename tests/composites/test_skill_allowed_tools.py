"""Every skill ``allowed-tools`` entry must resolve to a real default tool.

This is the cheap, no-LLM guard from plan §3.2. Comparing the skills' declared
allowlists against the server's actual default tool list is exactly what
surfaced the ``get_cluster_resources`` packaging bug; this test would have caught
that drift the day it landed, instead of as a Phase 3 failure in a paid eval run.

The harness does not enforce ``allowed-tools`` and the field is optional per the
agentskills spec, so at runtime deviations are *reported*, not blocked. A skill
naming a tool the default server does not expose is still a packaging bug.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.plugin_manager import PluginManager
from rhoai_mcp.server import RHOAIServer

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

#: Claude spells MCP tools ``mcp__<server>__<tool>``. The server segment is
#: non-greedy so a server name containing an underscore (``mcp__rhoai_mcp__x``)
#: still parses; an earlier ``[^_]+`` form silently failed to match those and
#: skipped them as "not an MCP reference", making the whole guard vacuous.
MCP_TOOL_PREFIX = "mcp__"
MCP_TOOL_RE = re.compile(rf"^{MCP_TOOL_PREFIX}(?P<server>.+?)__(?P<tool>.+)$")


def default_plugins() -> list[str]:
    """The shipped ``enabled_plugins`` default, read from the config itself.

    Reading the configured default keeps this check aligned with the shipped server.

    Uses the field default rather than an instance so an exported
    ``RHOAI_MCP_ENABLED_PLUGINS`` cannot change what this test checks.
    """
    from rhoai_mcp.config import RHOAIConfig

    default = RHOAIConfig.model_fields["enabled_plugins"].default
    assert default is not None, "enabled_plugins default must be an explicit list"
    return list(default)


def skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/SKILL.md"))


def parse_allowed_tools(skill_md: Path) -> list[str]:
    """Read the ``allowed-tools`` frontmatter field as a list of tool names."""
    text = skill_md.read_text(encoding="utf-8")
    match = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return []
    for line in match.group(1).splitlines():
        if line.startswith("allowed-tools:"):
            value = line.split(":", 1)[1].strip()
            # Space-separated, with optional comma separators.
            return [t.strip().rstrip(",") for t in value.split() if t.strip()]
    return []


@pytest.fixture(scope="module")
def default_tool_names() -> set[str]:
    """Tool names a server exposes under the default plugin configuration."""
    manager = PluginManager()
    manager.load_core_plugins(enabled_plugins=default_plugins())

    names: set[str] = set()

    def capture_tool(*_args: object, **kwargs: object) -> object:
        explicit = kwargs.get("name")

        def decorator(func: object) -> object:
            names.add(str(explicit) if explicit else func.__name__)
            return func

        return decorator

    mcp = MagicMock()
    mcp.tool = capture_tool

    server = RHOAIServer()
    server._plugin_manager = manager
    manager.register_all_tools(mcp, server)
    return names


def test_skills_are_discovered() -> None:
    assert skill_files(), f"no SKILL.md found under {SKILLS_DIR}"


@pytest.mark.parametrize("skill_md", skill_files(), ids=lambda p: p.parent.name)
def test_skill_declares_allowed_tools(skill_md: Path) -> None:
    assert parse_allowed_tools(skill_md), f"{skill_md} declares no allowed-tools"


@pytest.mark.parametrize("skill_md", skill_files(), ids=lambda p: p.parent.name)
def test_every_allowed_tool_resolves_under_the_default_plugin_set(
    skill_md: Path, default_tool_names: set[str]
) -> None:
    declared = parse_allowed_tools(skill_md)

    unresolved = []
    unparsable = []
    for entry in declared:
        match = MCP_TOOL_RE.match(entry)
        if not match:
            if entry.startswith(MCP_TOOL_PREFIX):
                # An mcp__ entry we cannot parse must fail loudly: silently
                # skipping it is how a guard stops guarding.
                unparsable.append(entry)
            # Otherwise not an MCP tool reference (e.g. a built-in).
            continue
        tool = match.group("tool")
        if tool not in default_tool_names:
            unresolved.append(entry)

    assert not unparsable, (
        f"{skill_md.parent.name} declares mcp__ entries that do not parse as "
        f"mcp__<server>__<tool>: {unparsable}"
    )

    assert not unresolved, (
        f"{skill_md.parent.name} declares tools absent from the default server: "
        f"{unresolved}. Either expose the tool from a default plugin, enable the "
        f"owning plugin by default, or amend the skill."
    )
