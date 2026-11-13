"""DeepEval MCP metrics with one task per phase-gated conversation.

DeepEval 4.3's default task extraction splits at each customer reply, drops
two-message exchanges, and omits text accompanying a tool call. That loses
confirmed constraints, consent changes, and final handoffs. Score the entire
ordered dialogue using explicit item rubrics. Retain stock min/mean aggregation
and metric integration. Private hooks have compatibility tests.
"""

from __future__ import annotations

import json
from typing import Any

from deepeval.metrics import (
    BaseConversationalMetric,
    MCPTaskCompletionMetric,
    MultiTurnMCPUseMetric,
)
from deepeval.metrics.mcp.schema import ArgsScore, Task, TaskScore, ToolScore
from deepeval.metrics.mcp.utils import (
    available_mcp_servers_block,
    task_steps_taken_text,
    turn_mcp_interaction_text,
)
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_case import ConversationalTestCase, Turn

from evals.metrics.scorecards import Scorecard, aggregate, prompt, reason

CONVERSATION_METRIC_VERSION = "conversation-v3-items"


class _ConversationTask(BaseConversationalMetric):
    def __init__(self, *, context: str = "", **kwargs: Any) -> None:
        self.context = context
        self._conversation_turns: list[Turn] | None = None
        self.scorecards: list[dict[str, Any]] = []
        if kwargs.get("eval_mode") not in (None, "llm"):
            raise ValueError("Item scorecards require llm eval mode")
        kwargs["eval_mode"] = "llm"
        super().__init__(**kwargs)

    def measure(self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any) -> float:
        self._conversation_turns = test_case.turns
        self.scorecards = []
        # Concrete MCP implementations follow this mixin in both classes' MRO.
        return float(super().measure(test_case, *args, **kwargs))  # type: ignore[safe-super]

    async def a_measure(
        self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any
    ) -> float:
        self._conversation_turns = test_case.turns
        self.scorecards = []
        return float(await super().a_measure(test_case, *args, **kwargs))  # type: ignore[safe-super]

    def _get_tasks(self, unit_interactions: list[list[Turn]]) -> list[Task]:
        # DeepEval's unit splitter also drops an unanswered trailing user turn.
        # Preserve it on timeout: a final refusal must not disappear from scoring.
        turns = self._conversation_turns
        if turns is None:
            turns = [turn for unit in unit_interactions for turn in unit]
        if not turns:
            return []
        opening = next((turn.content for turn in turns if turn.role == "user"), "")
        task = (
            "Evaluate this entire conversation as ONE evolving task. Later user replies "
            "refine earlier requirements; they are not independent tasks. Apply each "
            "confirmation or refusal only from its position in the dialogue onward. "
            "A request to hold off changes the desired outcome to no deployment. "
            "An unresolved prerequisite does not itself change the requested outcome: "
            "waiting can be appropriate tool use while task completion is still partial.\n\n"
            f"Opening request:\n{opening}\n\n"
            f"Workflow specification and scenario intent:\n{self.context}"
        )
        steps: list[str] = []
        for index, turn in enumerate(turns):
            if turn.content:
                steps.append(f"[{index}] {turn.role.upper()}:\n{turn.content}")
            if turn._mcp_interaction:
                steps.append(
                    f"[{index}] INTERNAL MCP INTERACTION:\n{turn_mcp_interaction_text(turn)}"
                )
        return [Task(task=task, steps_taken=steps)]

    def _get_prompt(self, method: Any, *, template_class: str | None = None, **kwargs: Any) -> str:
        rubrics = {
            "get_tool_correctness_score": "tools",
            "get_args_correctness_score": "arguments",
            "get_task_completion_score": "completion",
        }
        if method in rubrics:
            return prompt(
                rubrics[method],
                task=kwargs["task"].task,
                steps=kwargs["steps_taken"],
                tools="\n".join(
                    str(kwargs.get(k, ""))
                    for k in ("available_tools", "available_resources", "available_prompts")
                ),
            )
        # DeepEval resolves some templates from the concrete class name, even
        # on the ordinary LLM path. Both metrics borrow the stock MCP templates.
        return str(
            super()._get_prompt(
                method, template_class=template_class or "MCPTaskCompletionMetric", **kwargs
            )
        )

    def _card_prompt(self, rubric: str, task: Task, case: ConversationalTestCase | None) -> str:
        if case is not None and case.multimodal:
            raise ValueError("Item scorecards currently support text conversations only")
        tools = (
            "\n".join(str(s) for s in available_mcp_servers_block(case.mcp_servers or []))
            if case
            else ""
        )
        method = {
            "tools": "get_tool_correctness_score",
            "arguments": "get_args_correctness_score",
            "completion": "get_task_completion_score",
        }[rubric]
        return self._get_prompt(
            method, task=task, steps_taken=task_steps_taken_text(task), available_tools=tools
        )

    def _card(self, rubric: str, task: Task, case: ConversationalTestCase | None = None) -> Any:
        result = generate_with_schema_and_extract(
            metric=self,
            prompt=self._card_prompt(rubric, task, case),
            schema_cls=Scorecard,
            extract_schema=lambda s: aggregate(rubric, s),
            extract_json=lambda data: aggregate(rubric, data),
        )
        return self._retain_card(result)

    async def _a_card(
        self, rubric: str, task: Task, case: ConversationalTestCase | None = None
    ) -> Any:
        result = await a_generate_with_schema_and_extract(
            metric=self,
            prompt=self._card_prompt(rubric, task, case),
            schema_cls=Scorecard,
            extract_schema=lambda s: aggregate(rubric, s),
            extract_json=lambda data: aggregate(rubric, data),
        )
        return self._retain_card(result)

    def _retain_card(self, card: dict[str, Any]) -> Any:
        self.scorecards.append(card)
        cls = {"tools": ToolScore, "arguments": ArgsScore, "completion": TaskScore}[card["rubric"]]
        return cls(score=card["score"], reason=reason(card))

    def _generate_reason(self, *_args: Any, **_kwargs: Any) -> str | None:
        if not self.include_reason:
            return None
        # DeepEval exports/caches the reason, but not arbitrary metric metadata.
        # Keep item evidence there too; no extra LLM summarization is needed.
        issues = [
            f"{c['rubric']}.{item['id']}: {item['rating']}/2 — {item['evidence']}"
            for c in self.scorecards
            for item in c["items"]
            if item["rating"] is not None and item["rating"] < 2
        ]
        summary = "; ".join(issues) or "All applicable rubric items fully met."
        return json.dumps(
            {
                "metric_version": CONVERSATION_METRIC_VERSION,
                "summary": summary,
                "scorecards": self.scorecards,
            },
            ensure_ascii=False,
        )


class ConversationMCPUseMetric(_ConversationTask, MultiTurnMCPUseMetric):
    """Minimum of weighted tool-choice and argument scorecards."""

    @property
    def __name__(self) -> str:
        return "Conversation MCP Use"

    def _get_tool_accuracy_score(self, task: Task, test_case: ConversationalTestCase) -> Any:
        return self._card("tools", task, test_case)

    async def _a_get_tool_accuracy_score(
        self, task: Task, test_case: ConversationalTestCase
    ) -> Any:
        return await self._a_card("tools", task, test_case)

    def _get_args_score(self, task: Task, test_case: ConversationalTestCase) -> Any:
        return self._card("arguments", task, test_case)

    async def _a_get_args_score(self, task: Task, test_case: ConversationalTestCase) -> Any:
        return await self._a_card("arguments", task, test_case)


class ConversationTaskCompletionMetric(_ConversationTask, MCPTaskCompletionMetric):
    """Weighted outcome/deliverable scorecard for the final customer request."""

    @property
    def __name__(self) -> str:
        return "Conversation MCP Task Completion"

    def _get_task_score(self, task: Task, *, multimodal: bool) -> Any:
        if multimodal:
            raise ValueError("Item scorecards currently support text conversations only")
        return self._card("completion", task)

    async def _a_get_task_score(self, task: Task, *, multimodal: bool) -> Any:
        if multimodal:
            raise ValueError("Item scorecards currently support text conversations only")
        return await self._a_card("completion", task)
