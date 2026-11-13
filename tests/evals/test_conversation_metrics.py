"""Keep the actual conversation intact when DeepEval builds MCP tasks."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, MCPServer, MCPToolCall, Turn
from mcp.types import CallToolResult, TextContent


class UnusedJudge(DeepEvalBaseLLM):
    def get_model_name(self) -> str:
        return "offline-test"

    def load_model(self) -> None:
        return None

    def generate(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("This contract test must not call a judge")

    async def a_generate(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("This contract test must not call a judge")


@pytest.mark.parametrize(
    "metric_name", ["ConversationMCPUseMetric", "ConversationTaskCompletionMetric"]
)
def test_whole_conversation_keeps_constraints_tools_and_final_reply(metric_name: str) -> None:
    from deepeval.metrics.utils import get_unit_interactions
    from evals.metrics import conversation

    call = MCPToolCall(
        name="plan_deployment",
        args={"namespace": "production"},
        result=CallToolResult(
            content=[TextContent(type="text", text="storage missing")],
            structuredContent={"result": {"ready": False, "issues": ["storage missing"]}},
        ),
    )
    case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Plan a deployment for 50 users in production."),
            Turn(role="assistant", content="I will check the plan.", mcp_tools_called=[call]),
            Turn(role="assistant", content="Storage is missing. Shall we proceed?"),
            Turn(role="user", content="Hold off; do not deploy anything."),
            Turn(role="assistant", content="Nothing deployed. Here is the summary."),
        ],
    )
    metric = getattr(conversation, metric_name)(model=UnusedJudge(), context="Consent is required.")

    tasks = metric._get_tasks(get_unit_interactions(case.turns))

    assert len(tasks) == 1
    assert "50 users" in tasks[0].task
    assert "Consent is required." in tasks[0].task
    rendered = "\n".join(tasks[0].steps_taken)
    for text in [
        "I will check the plan.",
        "plan_deployment",
        "production",
        "storage missing",
        "Shall we proceed?",
        "Hold off; do not deploy anything.",
        "Nothing deployed.",
    ]:
        assert text in rendered
    assert rendered.index("plan_deployment") < rendered.index("Hold off")
    assert rendered.index("Hold off") < rendered.index("Nothing deployed")


@pytest.mark.parametrize(
    "metric_name", ["ConversationMCPUseMetric", "ConversationTaskCompletionMetric"]
)
def test_two_message_exchange_is_not_silently_dropped(metric_name: str) -> None:
    from evals.metrics import conversation

    metric = getattr(conversation, metric_name)(model=UnusedJudge(), context="")
    tasks = metric._get_tasks(
        [[Turn(role="user", content="Stop."), Turn(role="assistant", content="Stopped.")]]
    )
    assert len(tasks) == 1
    assert "Stopped." in tasks[0].steps_taken[-1]


def test_subclass_uses_deepeval_completion_templates() -> None:
    from deepeval.metrics.mcp.schema import Task
    from evals.metrics.conversation import ConversationTaskCompletionMetric

    metric = ConversationTaskCompletionMetric(model=UnusedJudge())
    prompt = metric._get_prompt(
        "get_task_completion_score",
        task=Task(task="Stop", steps_taken=["Stopped"]),
        steps_taken="Stopped",
    )
    assert "Stop" in prompt
    assert "Stopped" in prompt
    assert metric._get_prompt("_experimental_system_one_score")


def test_skill_factories_use_injected_judge_and_conversation_context() -> None:
    from evals.config import EvalConfig
    from evals.metrics.config import create_multi_turn_mcp_use_metric, create_task_completion_metric

    judge = UnusedJudge()
    config = EvalConfig(_env_file=None, mcp_use_threshold=0.6, task_completion_threshold=0.8)
    with patch(
        "evals.metrics.config.create_judge_llm", side_effect=AssertionError("unexpected provider")
    ):
        use = create_multi_turn_mcp_use_metric(config, model=judge, context="fixed workflow")
        completion = create_task_completion_metric(config, model=judge, context="fixed workflow")
    assert use.model is judge
    assert completion.model is judge
    assert use.context == completion.context == "fixed workflow"
    assert use.threshold == 0.6
    assert completion.threshold == 0.8


@pytest.mark.parametrize(
    "metric_name", ["ConversationMCPUseMetric", "ConversationTaskCompletionMetric"]
)
@pytest.mark.parametrize("async_mode", [False, True])
async def test_partial_history_keeps_final_unanswered_user_reply(
    metric_name: str, async_mode: bool
) -> None:
    from evals.metrics import conversation

    class FixedJudge(UnusedJudge):
        def generate(self, _prompt: str, schema: Any = None, **_kwargs: Any) -> Any:
            from evals.metrics.scorecards import Scorecard

            from tests.evals.test_scorecards import card

            if schema is Scorecard:
                rubric = _prompt.split("Evaluate the ", 1)[1].split(" dimension", 1)[0]
                return schema.model_validate(card(rubric))
            return schema(score=0.9, reason="offline contract test")

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            return self.generate(prompt, schema, **kwargs)

    case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Deploy the model."),
            Turn(role="assistant", content="May I proceed?"),
            Turn(role="user", content="Hold off; do not deploy."),
        ],
        mcp_servers=[MCPServer(server_name="offline")],
    )
    metric = getattr(conversation, metric_name)(
        model=FixedJudge(), context="", include_reason=False, async_mode=False
    )
    if async_mode:
        await metric.a_measure(case, _show_indicator=False)
    else:
        metric.measure(case, _show_indicator=False)
    assert len(metric.tasks) == 1
    assert "Hold off; do not deploy." in "\n".join(metric.tasks[0].steps_taken)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    ("metric_name", "expected_score", "expected_calls"),
    [("ConversationMCPUseMetric", 23 / 24, 2), ("ConversationTaskCompletionMetric", 33 / 34, 1)],
)
async def test_item_scores_drive_real_metric_and_retain_audit_evidence(
    metric_name: str, expected_score: float, expected_calls: int, async_mode: bool
) -> None:
    import json

    from evals.metrics import conversation

    from tests.evals.test_scorecards import card

    class ItemJudge(UnusedJudge):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, schema: Any = None, **_kwargs: Any) -> Any:
            self.calls += 1
            rubric = prompt.split("Evaluate the ", 1)[1].split(" dimension", 1)[0]
            return schema.model_validate(card(rubric, efficiency=1, constraints=1, clarity=1))

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            return self.generate(prompt, schema, **kwargs)

    judge = ItemJudge()
    metric = getattr(conversation, metric_name)(model=judge, async_mode=False)
    case = ConversationalTestCase(
        turns=[Turn(role="user", content="Stop"), Turn(role="assistant", content="Stopped")],
        mcp_servers=[MCPServer(server_name="offline")],
    )
    for _ in range(2):
        if async_mode:
            await metric.a_measure(case, _show_indicator=False)
        else:
            metric.measure(case, _show_indicator=False)
        assert metric.score == pytest.approx(expected_score)
        assert len(metric.scorecards) == expected_calls
        evidence = json.loads(metric.reason)
        assert evidence["metric_version"] == conversation.CONVERSATION_METRIC_VERSION
        assert evidence["scorecards"][0]["items"][0]["evidence"]
    assert judge.calls == expected_calls * 2  # no extra LLM summary requests
