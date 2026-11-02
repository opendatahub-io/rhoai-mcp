"""Adversarial agreement checks over frozen, synthetic conversations; no LLMs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")

from evals.agent.transcript import AttemptArtifact, from_pydantic_ai  # noqa: E402
from evals.skills.checks import check_navigator, find_user_turn_index  # noqa: E402
from evals.skills.goldens import (  # noqa: E402
    APPROVE_REPLY,
    CHOOSE_REPLY,
    CONFIRM_REPLY,
    DEPLOY_APPROVED,
    DEPLOY_CHOOSE_REPLY,
    DEPLOY_EXPECTATION,
    DEPLOY_STORAGE_URI,
    GPU_REPLY,
    NAVIGATOR_EXPECTATION,
    NAVIGATOR_HAPPY_PATH,
)
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart  # noqa: E402

from tests.evals.test_skill_checks import (  # noqa: E402
    DEPLOY_PLAN,
    DEPLOY_REVIEW,
    GOOD_TABLE,
    HANDOFF,
    assistant,
    call,
    load,
    result,
    returns,
    user,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
WORKLOAD = {
    "use_case": "chatbot_conversational",
    "user_count": 50,
    "prompt_tokens": 512,
    "output_tokens": 256,
    "expected_qps": 0.13,
    "ttft_target_ms": 300,
    "itl_target_ms": 30,
    "e2e_target_ms": 8620,
}
RECOMMENDATION = {
    "specification": {
        "use_case": "chatbot_conversational",
        "user_count": 50,
        "traffic_profile": {
            k: WORKLOAD[k] for k in ("prompt_tokens", "output_tokens", "expected_qps")
        },
    },
    "recommendations": {
        "top_balanced": {"model_id": MODEL_ID, "model": "llama-3.1-8b", "gpu": "1x A100-80"},
    },
}


def workflow(*, deploy: bool = False) -> list[Any]:
    messages = [
        user(f"A support chatbot for 50 agents. Selected model: {MODEL_ID}."),
        *load("navigator-deploy" if deploy else "navigator", "skill"),
        assistant(
            call("get_use_case_defaults", "defaults", use_case="chatbot_conversational"),
            call("get_expected_rps", "traffic", use_case="chatbot_conversational", user_count=50),
        ),
        returns(
            result(
                "get_use_case_defaults",
                "defaults",
                {
                    "use_case": "chatbot_conversational",
                    "workload": {"prompt_tokens": 512, "output_tokens": 256},
                    "slo_targets": {
                        f"{k}_ms": {"default": v}
                        for k, v in (("ttft", 300), ("itl", 30), ("e2e", 8620))
                    },
                },
            ),
            result(
                "get_expected_rps",
                "traffic",
                {
                    "use_case": "chatbot_conversational",
                    "user_count": 50,
                    "expected_rps": 0.06666666666666667,
                },
            ),
        ),
        assistant(TextPart(content="Here are the defaults for 50 users. Confirm?")),
        user(CONFIRM_REPLY),
        assistant(TextPart(content="Which GPUs does your cluster have?")),
        user(GPU_REPLY),
        assistant(
            call(
                "recommend_model",
                "recommend",
                use_case="chatbot_conversational",
                user_count=50,
                preferred_gpu_types=["A100-80"],
            )
        ),
        returns(result("recommend_model", "recommend", deepcopy(RECOMMENDATION))),
        assistant(
            TextPart(
                content=GOOD_TABLE.replace(
                    "| GPU | A100-80 | L4 | H100 | H100 |",
                    "| GPU | 1x A100-80 | 1x L4 | 2x H100 | 2x H100 |",
                )
            )
        ),
        user(DEPLOY_CHOOSE_REPLY if deploy else CHOOSE_REPLY),
    ]
    if not deploy:
        return messages + [assistant(TextPart(content=HANDOFF))]
    params = {
        **WORKLOAD,
        "category": "balanced",
        "namespace": "production-models",
        "preferred_gpu_types": ["A100-80"],
    }
    execution = {
        "model_id": MODEL_ID,
        "namespace": "production-models",
        "runtime": "vllm-runtime",
        "storage_uri": DEPLOY_STORAGE_URI,
        "gpu_count": 1,
        "gpu_type": "A100-80",
        "tensor_parallel": 1,
        "replicas": 1,
    }
    return messages + [
        assistant(call("plan_deployment", "plan", **params)),
        returns(result("plan_deployment", "plan", {**DEPLOY_PLAN, "tensor_parallel": 1})),
        assistant(TextPart(content="Please supply storage before approving.")),
        user(f"The weights are at {DEPLOY_STORAGE_URI}. Show me the resolved plan."),
        assistant(
            TextPart(
                content=DEPLOY_REVIEW.replace(
                    "oci://registry.example.com/models/llama:v1", DEPLOY_STORAGE_URI
                )
            )
        ),
        user(APPROVE_REPLY),
        assistant(call("execute_deployment", "execute", **execution)),
        returns(result("execute_deployment", "execute", {"deployed": True, "status": "Pending"})),
    ]


def tool_part(messages: list[Any], name: str, *, returned: bool = False) -> Any:
    kind = ToolReturnPart if returned else ToolCallPart
    return next(
        part
        for message in messages
        for part in message.parts
        if isinstance(part, kind) and part.tool_name == name
    )


def grade(messages: list[Any], *, deploy: bool = False) -> Any:
    return (DEPLOY_APPROVED if deploy else NAVIGATOR_HAPPY_PATH).grade(from_pydantic_ai(messages))


def failures(report: Any) -> set[str]:
    return {c.name for c in report.failures}


@pytest.mark.parametrize("deploy", [False, True])
def test_complete_workflow_agrees_after_artifact_roundtrip(deploy: bool, tmp_path: Any) -> None:
    artifact = AttemptArtifact.from_messages(workflow(deploy=deploy), attempt_id="a", run_id="r")
    restored = AttemptArtifact.load(artifact.save(tmp_path))
    report = (DEPLOY_APPROVED if deploy else NAVIGATOR_HAPPY_PATH).grade(restored.transcript())

    assert report.passed, [(c.name, c.detail) for c in report.failures]
    assert all(c.status == "pass" for c in report.results if "agrees_with" in c.name)


@pytest.mark.parametrize(
    "key,value",
    [
        ("use_case", "translation"),
        ("user_count", "5000"),
        ("prompt_tokens", "2048"),
        ("output_tokens", "1024"),
        ("expected_qps", "0.13"),
        ("expected_qps", "nan"),
        ("ttft_target_ms", "100"),
        ("itl_target_ms", "100"),
        ("e2e_target_ms", "100"),
        ("cluster_gpu_types", "H100"),
        ("cluster_gpu_types", "A100-80, H100"),
    ],
)
def test_handoff_rejects_wrong_workload_field(key: str, value: str) -> None:
    messages = workflow()
    text = messages[-1].parts[0].content
    lines = [
        f"{key}: {value}" if line.startswith(f"{key}:") else line for line in text.splitlines()
    ]
    messages[-1] = assistant(TextPart(content="\n".join(lines)))

    assert "navigator_handoff_agrees_with_workload" in failures(grade(messages))


@pytest.mark.parametrize(
    "old,new",
    [
        (f"model_id: {MODEL_ID}", "model_id: other/model"),
        ("chosen_category: balanced", "chosen_category: cost"),
    ],
)
def test_handoff_rejects_wrong_model_or_category(old: str, new: str) -> None:
    messages = workflow()
    messages[-1] = assistant(TextPart(content=HANDOFF.replace(old, new)))

    assert "navigator_handoff_agrees_with_selection" in failures(grade(messages))


def test_unrelated_later_slot_rerun_does_not_replace_customer_selection() -> None:
    messages = workflow()
    rerun = deepcopy(RECOMMENDATION)
    rerun["recommendations"]["top_balanced"] = {
        "model_id": "other/model",
        "model": "other model",
        "gpu": "2x H100",
    }
    messages[-2:-2] = [
        assistant(
            call(
                "recommend_model",
                "rerun",
                use_case="chatbot_conversational",
                user_count=50,
                min_quality=90,
            )
        ),
        returns(result("recommend_model", "rerun", rerun)),
    ]

    assert grade(messages).passed


def test_displayed_selection_cannot_be_replaced_by_another_model_in_same_slot() -> None:
    messages = workflow()
    tool_part(messages, "recommend_model", returned=True).content["recommendations"][
        "top_balanced"
    ]["model_id"] = "other/model"

    assert "navigator_handoff_agrees_with_selection" in failures(grade(messages))


@pytest.mark.parametrize("name", ["get_use_case_defaults", "get_expected_rps"])
def test_missing_preconfirmation_source_is_a_failure(name: str) -> None:
    messages = workflow()
    tool_part(messages, name, returned=True).content = {}

    assert "navigator_handoff_agrees_with_workload" in failures(grade(messages))


def test_corrupt_recommendation_and_handoff_cannot_redefine_scenario_user_count() -> None:
    messages = workflow()
    tool_part(messages, "recommend_model").args["user_count"] = 5000
    tool_part(messages, "recommend_model", returned=True).content["specification"]["user_count"] = (
        5000
    )
    messages[-1] = assistant(
        TextPart(content=HANDOFF.replace("user_count: 50", "user_count: 5000"))
    )

    assert {
        "navigator_handoff_agrees_with_workload",
        "navigator_handoff_agrees_with_selection",
    } <= failures(grade(messages))


@pytest.mark.parametrize("prompt", [CONFIRM_REPLY, CHOOSE_REPLY])
def test_missing_customer_gate_is_not_missing_coverage(prompt: str) -> None:
    messages = workflow()
    for message in messages:
        for part in message.parts:
            if getattr(part, "content", None) == prompt:
                part.content = "different reply"

    assert not grade(messages).passed


def test_handoff_before_choice_is_not_a_valid_handoff() -> None:
    messages = workflow()
    messages.insert(-2, messages.pop())

    assert "navigator_handoff_agrees_with_workload" in failures(grade(messages))


def test_unknown_gpu_inventory_may_have_an_empty_handoff_field() -> None:
    messages = workflow()
    tool_part(messages, "recommend_model").args["preferred_gpu_types"] = []
    messages[-1] = assistant(
        TextPart(content=HANDOFF.replace("cluster_gpu_types: A100-80", "cluster_gpu_types:"))
    )
    transcript = from_pydantic_ai(messages)
    report = check_navigator(
        transcript,
        confirmation_turn_index=find_user_turn_index(transcript, CONFIRM_REPLY),
        selection_turn_index=find_user_turn_index(transcript, CHOOSE_REPLY),
        expectation=replace(NAVIGATOR_EXPECTATION, cluster_gpu_types=()),
    )

    assert report.passed, [(c.name, c.detail) for c in report.failures]


@pytest.mark.parametrize(
    "key,value",
    [
        ("category", "cost"),
        ("namespace", "other-project"),
        ("user_count", 5000),
        ("use_case", "translation"),
        ("prompt_tokens", 1024),
        ("output_tokens", 1024),
        ("expected_qps", 5.0),
        ("ttft_target_ms", 100),
        ("itl_target_ms", 100),
        ("e2e_target_ms", 100),
        ("preferred_gpu_types", ["H100"]),
    ],
)
def test_plan_arguments_must_match_confirmed_selection(key: str, value: Any) -> None:
    messages = workflow(deploy=True)
    tool_part(messages, "plan_deployment").args[key] = value

    assert "deploy_plan_agrees_with_selection" in failures(grade(messages, deploy=True))


@pytest.mark.parametrize(
    "key,value", [("model_id", "other/model"), ("gpu_count", 2), ("gpu_type", "H100")]
)
def test_plan_cannot_silently_change_recommended_model_or_hardware(key: str, value: Any) -> None:
    messages = workflow(deploy=True)
    tool_part(messages, "plan_deployment", returned=True).content[key] = value
    tool_part(messages, "execute_deployment").args[key] = value

    assert "deploy_plan_agrees_with_selection" in failures(grade(messages, deploy=True))


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_id", "other/model"),
        ("namespace", "other-project"),
        ("runtime", "other-runtime"),
        ("storage_uri", "s3://other/weights"),
        ("gpu_count", 2),
        ("gpu_type", "H100"),
        ("tensor_parallel", 2),
        ("replicas", 2),
    ],
)
def test_every_execution_argument_must_match_reviewed_plan(key: str, value: Any) -> None:
    messages = workflow(deploy=True)
    tool_part(messages, "execute_deployment").args[key] = value

    assert "deploy_execution_agrees_with_reviewed_plan" in failures(grade(messages, deploy=True))


@pytest.mark.parametrize(
    "old,new",
    [
        (MODEL_ID, "other/model"),
        ("`production-models`", "`other-project`"),
        ("`vllm-runtime`", "`other-runtime`"),
        (DEPLOY_STORAGE_URI, "s3://other/weights"),
        ("1 × A100-80", "2 × H100"),
        ("| Replicas | 1 |", "| Replicas | 2 |"),
    ],
)
def test_actual_execution_cannot_disagree_with_displayed_plan(old: str, new: str) -> None:
    messages = workflow(deploy=True)
    review = messages[-4].parts[0].content
    assert old in review
    messages[-4] = assistant(TextPart(content=review.replace(old, new)))

    assert "deploy_execution_agrees_with_reviewed_plan" in failures(grade(messages, deploy=True))


def test_later_wrong_execution_is_not_hidden_by_an_earlier_correct_execution() -> None:
    messages = workflow(deploy=True)
    args = {**tool_part(messages, "execute_deployment").args, "gpu_type": "H100"}
    messages += [
        assistant(call("execute_deployment", "second-execution", **args)),
        returns(result("execute_deployment", "second-execution", {"deployed": True})),
    ]

    assert "deploy_execution_agrees_with_reviewed_plan" in failures(grade(messages, deploy=True))


def test_review_and_execution_cannot_collude_on_a_different_storage_uri() -> None:
    messages = workflow(deploy=True)
    messages[-4].parts[0].content = (
        messages[-4].parts[0].content.replace(DEPLOY_STORAGE_URI, "s3://wrong/model")
    )
    tool_part(messages, "execute_deployment").args["storage_uri"] = "s3://wrong/model"

    assert "deploy_execution_agrees_with_reviewed_plan" in failures(grade(messages, deploy=True))


def test_latest_review_cannot_be_hidden_by_an_earlier_valid_table() -> None:
    messages = workflow(deploy=True)
    messages.insert(
        -3,
        assistant(
            TextPart(
                content=messages[-4]
                .parts[0]
                .content.replace("| Replicas | 1 |", "| Replicas | 2 |")
            )
        ),
    )

    assert "deploy_execution_agrees_with_reviewed_plan" in failures(grade(messages, deploy=True))


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_malformed_recommendation_metadata_fails_without_crashing(value: Any) -> None:
    messages = workflow(deploy=True)
    tool_part(messages, "recommend_model", returned=True).content["specification"] = value

    assert "deploy_plan_agrees_with_selection" in failures(grade(messages, deploy=True))


def test_changing_expected_intent_changes_scenario_hash() -> None:
    changed = replace(DEPLOY_APPROVED, expectation=replace(DEPLOY_EXPECTATION, user_count=5000))

    assert changed.scenario_sha() != DEPLOY_APPROVED.scenario_sha()
    assert not changed.grade(from_pydantic_ai(workflow(deploy=True))).passed


@pytest.mark.parametrize("key", ["tensor_parallel", "replicas"])
def test_omitted_api_defaults_are_compared_as_effective_values(key: str) -> None:
    messages = workflow(deploy=True)
    del tool_part(messages, "execute_deployment").args[key]

    assert grade(messages, deploy=True).passed


@pytest.mark.parametrize("filter_value", [["H100"], [], None])
def test_recommendation_cannot_change_or_drop_known_hardware_filter(filter_value: Any) -> None:
    messages = workflow()
    tool_part(messages, "recommend_model").args["preferred_gpu_types"] = filter_value

    assert "navigator_handoff_agrees_with_selection" in failures(grade(messages))


def test_documented_unconstrained_fallback_is_allowed_after_empty_slot() -> None:
    messages = workflow()
    tool_part(messages, "recommend_model").args["preferred_gpu_types"] = []
    index = next(
        i
        for i, m in enumerate(messages)
        if any(isinstance(p, ToolCallPart) and p.tool_name == "recommend_model" for p in m.parts)
    )
    primary = deepcopy(RECOMMENDATION)
    primary["recommendations"]["top_balanced"] = None
    messages[index:index] = [
        assistant(
            call(
                "recommend_model",
                "primary",
                use_case="chatbot_conversational",
                user_count=50,
                preferred_gpu_types=["A100-80"],
            )
        ),
        returns(result("recommend_model", "primary", primary)),
    ]

    assert grade(messages).passed


def test_a_different_chosen_category_uses_its_own_slot() -> None:
    messages = workflow()
    rec = tool_part(messages, "recommend_model", returned=True).content
    rec["recommendations"]["top_cost"] = {
        "model_id": "mistral/Mistral-7B-Instruct",
        "model": "mistral-7b",
        "gpu": "1x L4",
    }
    choice = "Let's go with cost."
    messages[-2] = user(choice)
    messages[-1] = assistant(
        TextPart(
            content=HANDOFF.replace(MODEL_ID, "mistral/Mistral-7B-Instruct").replace(
                "chosen_category: balanced", "chosen_category: cost"
            )
        )
    )
    transcript = from_pydantic_ai(messages)
    report = check_navigator(
        transcript,
        confirmation_turn_index=find_user_turn_index(transcript, CONFIRM_REPLY),
        selection_turn_index=find_user_turn_index(transcript, choice),
        expectation=replace(NAVIGATOR_EXPECTATION, category="cost"),
    )

    assert report.passed, [(c.name, c.detail) for c in report.failures]


def test_same_model_from_wrong_category_cannot_satisfy_selection() -> None:
    messages = workflow()
    rec = tool_part(messages, "recommend_model", returned=True).content["recommendations"]
    rec["top_cost"] = rec.pop("top_balanced")

    assert "navigator_handoff_agrees_with_selection" in failures(grade(messages))


def test_explicit_slo_override_replaces_default_without_changing_source_tools() -> None:
    messages = workflow()
    messages[-1] = assistant(
        TextPart(content=HANDOFF.replace("ttft_target_ms: 300", "ttft_target_ms: 150"))
    )
    transcript = from_pydantic_ai(messages)
    report = check_navigator(
        transcript,
        confirmation_turn_index=find_user_turn_index(transcript, CONFIRM_REPLY),
        selection_turn_index=find_user_turn_index(transcript, CHOOSE_REPLY),
        expectation=replace(NAVIGATOR_EXPECTATION, slo_overrides={"ttft_target_ms": 150}),
    )

    assert report.passed, [(c.name, c.detail) for c in report.failures]


def test_later_recommendation_cannot_retroactively_authorize_handoff() -> None:
    messages = workflow()
    invented = deepcopy(RECOMMENDATION)
    invented["recommendations"]["top_balanced"]["model_id"] = "other/model"
    messages[-1] = assistant(TextPart(content=HANDOFF.replace(MODEL_ID, "other/model")))
    messages += [
        assistant(
            call(
                "recommend_model",
                "late",
                use_case="chatbot_conversational",
                user_count=50,
                preferred_gpu_types=["A100-80"],
            )
        ),
        returns(result("recommend_model", "late", invented)),
    ]

    assert "navigator_handoff_agrees_with_selection" in failures(grade(messages))


def test_latest_handoff_in_one_message_is_the_one_graded() -> None:
    messages = workflow()
    messages[-1] = assistant(
        TextPart(content=HANDOFF + HANDOFF.replace("user_count: 50", "user_count: 5000"))
    )

    assert "navigator_handoff_agrees_with_workload" in failures(grade(messages))


@pytest.mark.parametrize("value", ["nan", "inf", 0, True])
def test_invalid_numeric_source_cannot_be_copied_into_a_passing_handoff(value: Any) -> None:
    messages = workflow()
    tool_part(messages, "get_expected_rps", returned=True).content["expected_rps"] = value
    messages[-1] = assistant(
        TextPart(content=HANDOFF.replace("expected_qps: 0.067", f"expected_qps: {value}"))
    )

    assert "navigator_handoff_agrees_with_workload" in failures(grade(messages))


@pytest.mark.parametrize("deploy", [False, True])
def test_golden_cannot_silently_disable_agreement_by_omitting_intent(deploy: bool) -> None:
    golden = replace(DEPLOY_APPROVED if deploy else NAVIGATOR_HAPPY_PATH, expectation=None)
    report = golden.grade(from_pydantic_ai(workflow(deploy=deploy)))

    assert "scenario_selection_expectation_present" in failures(report)


def test_review_uri_prefix_is_not_a_match_for_another_uri() -> None:
    messages = workflow(deploy=True)
    messages[-4].parts[0].content = (
        messages[-4].parts[0].content.replace(DEPLOY_STORAGE_URI, DEPLOY_STORAGE_URI + ":other")
    )

    assert "deploy_execution_agrees_with_reviewed_plan" in failures(grade(messages, deploy=True))


def test_extra_gpu_in_review_cannot_be_ignored() -> None:
    messages = workflow(deploy=True)
    messages[-4].parts[0].content = (
        messages[-4].parts[0].content.replace("1 × A100-80", "1 × A100-80 + 1 × H100")
    )

    assert "deploy_execution_agrees_with_reviewed_plan" in failures(grade(messages, deploy=True))
