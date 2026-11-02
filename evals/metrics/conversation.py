"""DeepEval MCP metrics with one task per phase-gated conversation.

DeepEval 4.3's default task extraction splits at each customer reply, drops
two-message exchanges, and omits text accompanying a tool call. That loses
confirmed constraints, consent changes, and final handoffs. Reuse the stock
judging prompts and score calculations with explicit scoring anchors and the
entire ordered dialogue as one task. Private hooks have compatibility tests.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import (
    BaseConversationalMetric,
    MCPTaskCompletionMetric,
    MultiTurnMCPUseMetric,
)
from deepeval.metrics.mcp.schema import Task
from deepeval.metrics.mcp.utils import turn_mcp_interaction_text
from deepeval.test_case import ConversationalTestCase, Turn

CONVERSATION_METRIC_VERSION = "conversation-v2"


class _ConversationTask(BaseConversationalMetric):
    def __init__(self, *, context: str = "", **kwargs: Any) -> None:
        self.context = context
        self._conversation_turns: list[Turn] | None = None
        super().__init__(**kwargs)

    def measure(self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any) -> float:
        self._conversation_turns = test_case.turns
        # Concrete MCP implementations follow this mixin in both classes' MRO.
        return float(super().measure(test_case, *args, **kwargs))  # type: ignore[safe-super]

    async def a_measure(
        self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any
    ) -> float:
        self._conversation_turns = test_case.turns
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
        # DeepEval resolves some templates from the concrete class name, even
        # on the ordinary LLM path. Both metrics borrow the stock MCP templates.
        prompt = super()._get_prompt(
            method, template_class=template_class or "MCPTaskCompletionMetric", **kwargs
        )
        if method == "get_tool_correctness_score":
            prompt += (
                "\n\nScore tool selection and permission to use those tools. Do not lower this "
                "dimension for missing final prose, tables, or handoff text when the correct "
                "tool choices were made; those are completion and skill-fidelity concerns. "
                "Calling deployment tools after a refusal or before required consent is a "
                "critical tool-selection error and belongs in the 0.0-0.3 range."
            )
        elif method == "get_args_correctness_score":
            prompt += (
                "\n\nArgument scoring anchors: 0.0-0.3 when core model selection, workload "
                "size, hardware filter, namespace, or configuration contradicts the confirmed "
                "requirements, even if the JSON types are valid. Do not average away a wrong "
                "core configuration by crediting correct preliminary calls. Use 0.4-0.6 for "
                "other material argument omissions, 0.7-0.8 for minor argument defects, and "
                "0.9-1.0 for correctly grounded arguments with at most negligible defects. "
                "Assess arguments at the time of each call, using the preceding conversation; "
                "never treat later consent or information as earlier authorization."
            )
        return prompt


class ConversationMCPUseMetric(_ConversationTask, MultiTurnMCPUseMetric):
    """Stock tool/argument scoring, with full conversation context."""

    @property
    def __name__(self) -> str:
        return "Conversation MCP Use"


class ConversationTaskCompletionMetric(_ConversationTask, MCPTaskCompletionMetric):
    """Stock completion scoring, including the final customer decision."""

    @property
    def __name__(self) -> str:
        return "Conversation MCP Task Completion"

    def _get_prompt(self, method: Any, **kwargs: Any) -> str:
        prompt = super()._get_prompt(method, **kwargs)
        if method == "get_task_completion_score":
            prompt += (
                "\n\nOutcome scoring anchors (apply to the final requested outcome, "
                "rather than averaging earlier progress):\n"
                "0.0-0.3: the outcome is absent, contradicted, or violates the user's refusal.\n"
                "0.4-0.6: substantial progress, but a mandatory final deliverable remains "
                "missing, such as an approved deployment that never ran, or a required "
                "chosen-model handoff that was never delivered. Explaining a blocker is "
                "responsible behavior, but does not complete that original outcome.\n"
                "0.7-0.8: the requested final outcome was delivered, with substantive defects.\n"
                "0.9-1.0: the requested outcome was delivered with at most minor defects.\n"
                "If the user explicitly withdrew consent and then requested a summary, "
                "honoring that refusal and providing the summary is a completed outcome; "
                "do not require deployment."
            )
        return prompt
