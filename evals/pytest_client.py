"""Small backend-neutral facade for live pytest scenarios; no optional SDK imports."""

from __future__ import annotations

from typing import Any, cast

from evals.mcp_config import REPO_ROOT
from evals.result_types import AgentResult


class EvalClient:
    def __init__(self, client: Any, backend: str) -> None:
        self.client = client
        self.backend = backend
        self.skill: str | None = None

    async def query(self, task: str, *, skill: str | None = None) -> AgentResult:
        self.skill = skill
        if self.backend == "claude-code":
            return cast(AgentResult, await self.client.query(task, skill=skill))
        # LCS uses its server-configured instructions; it has no local skill loader.
        return cast(AgentResult, await self.client.query(task))

    async def follow_up(self, message: str, conversation_id: str) -> AgentResult:
        return cast(AgentResult, await self.client.follow_up(message, conversation_id))

    @property
    def agent_model(self) -> str:
        if self.backend == "claude-code":
            return str(self.client.initialization.get("model", ""))
        return ""

    @property
    def phase(self) -> str | None:
        if self.backend != "claude-code" or not self.skill:
            return None
        if (REPO_ROOT / ".claude/skills" / self.skill / "SKILL.md").is_file():
            return "skills"
        return "commands"
