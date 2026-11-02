"""Tests for the deterministic skill checks (plan §3.1).

No LLM, no network. These are the checks that decide pass/fail independently of
any judge, so a defect here silently mislabels every attempt.

Particular attention to the table checker: the helper it replaces accepted a row
carrying a single recommendation cell, and flattened every markdown table in the
output into one pseudo-table — so a Phase 2 workload table could satisfy checks
meant for the Phase 3 recommendation table. The malformed-table cases below
exist to pin those down (plan §3.1).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")

import json  # noqa: E402

from evals.agent.transcript import from_pydantic_ai  # noqa: E402
from evals.skills.checks import (  # noqa: E402
    CheckReport,
    check_navigator,
    check_navigator_deploy,
    check_recommendation_table,
    check_routing,
    deployment_review_error,
    find_tables,
    find_user_turn_index,
    parse_handoff,
    tools_outside_allowlist,
)
from pydantic_ai.messages import (  # noqa: E402
    CapabilityInstructionSource,
    InstructionDeltaPart,
    InstructionId,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# --------------------------------------------------------------------------- #
# Fixtures: table text
# --------------------------------------------------------------------------- #

GOOD_TABLE = """
| | Balanced | Cost | Performance | Quality |
|---|---|---|---|---|
| Model | llama-3.1-8b | mistral-7b | llama-3.1-70b | llama-3.1-70b |
| GPU | A100-80 | L4 | H100 | H100 |
| TTFT p95 | 180 ms | 240 ms | 95 ms | 110 ms |
| E2E p95 | 4.2 s | 6.1 s | 2.8 s | 3.1 s |
| Quality score | 72 | 65 | 88 | 91 |
| Cost/month | $1,200 | $400 | $4,800 | $4,800 |
| Meets SLO | Yes | Yes | Yes | Yes |
| Cluster fit | Available ✓ | Available ✓ | — | — |
"""

#: A Phase 2 workload table that must never be mistaken for the Phase 3 one.
WORKLOAD_TABLE = """
| Workload | Value |
|---|---|
| Planner prompt baseline | 512 tokens |
| Expected traffic | 0.067 req/s |
"""

HANDOFF = """
<!-- navigator-handoff
model_id: meta-llama/Llama-3.1-8B-Instruct
use_case: chatbot_conversational
user_count: 50
chosen_category: balanced
prompt_tokens: 512
output_tokens: 256
expected_qps: 0.067
ttft_target_ms: 300
itl_target_ms: 30
e2e_target_ms: 8620
cluster_gpu_types: A100-80
-->
"""


# --------------------------------------------------------------------------- #
# Transcript builders
# --------------------------------------------------------------------------- #


def user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def assistant(*parts: Any) -> ModelResponse:
    return ModelResponse(parts=list(parts))


def returns(*parts: Any) -> ModelRequest:
    return ModelRequest(parts=list(parts))


def call(name: str, call_id: str, **args: Any) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args, tool_call_id=call_id)


def result(name: str, call_id: str, content: Any) -> ToolReturnPart:
    return ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)


def load(skill: str, call_id: str, body: str = "# Skill body") -> list[Any]:
    """A successful skill load: call, ack, and the instruction that carries it."""
    return [
        assistant(
            LoadCapabilityCallPart(
                tool_name="load_capability",
                args=json.dumps({"id": skill}),
                tool_call_id=call_id,
            )
        ),
        returns(
            LoadCapabilityReturnPart(
                tool_name="load_capability", content={"message": "loaded"}, tool_call_id=call_id
            ),
            InstructionDeltaPart(
                id=InstructionId(source=CapabilityInstructionSource(id=skill)), content=body
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Table parsing
# --------------------------------------------------------------------------- #


class TestTableParsing:
    def test_separate_tables_are_parsed_separately(self) -> None:
        tables = find_tables(WORKLOAD_TABLE + "\nSome prose.\n" + GOOD_TABLE)

        assert len(tables) == 2
        assert tables[0].headers == ["Workload", "Value"]
        assert "Balanced" in tables[1].headers

    def test_adjacent_tables_are_not_merged(self) -> None:
        """Without the prose separator the two tables still must not fuse."""
        tables = find_tables(WORKLOAD_TABLE + GOOD_TABLE)

        assert len(tables) == 2

    def test_a_block_without_a_separator_row_is_not_a_table(self) -> None:
        assert find_tables("| a | b |\n| 1 | 2 |") == []


class TestRecommendationTable:
    def test_well_formed_table_passes_every_check(self) -> None:
        report = CheckReport()
        check_recommendation_table(GOOD_TABLE, report)

        assert report.passed, report.failures

    def test_workload_table_alone_is_not_accepted(self) -> None:
        """The regression: an earlier table must not satisfy Phase 3 checks."""
        report = CheckReport()
        check_recommendation_table(WORKLOAD_TABLE, report)

        assert not report.passed
        assert any(r.name == "table_present" and r.failed for r in report.results)

    def test_dependent_checks_are_na_when_no_table_exists(self) -> None:
        """Absent input must not silently pass the shape checks."""
        report = CheckReport()
        check_recommendation_table("no tables here", report)

        statuses = {r.name: r.status for r in report.results}
        assert statuses["table_columns_exact"] == "n/a"
        assert statuses["table_cells_complete"] == "n/a"

    def test_recommendation_table_is_found_among_other_tables(self) -> None:
        report = CheckReport()
        check_recommendation_table(WORKLOAD_TABLE + "\ntext\n" + GOOD_TABLE, report)

        assert report.passed, report.failures

    def test_row_with_a_single_recommendation_cell_fails(self) -> None:
        """The specific weakness of the helper this replaces."""
        sparse = GOOD_TABLE.replace(
            "| Cluster fit | Available ✓ | Available ✓ | — | — |",
            "| Cluster fit | Available ✓ |",
        )
        report = CheckReport()
        check_recommendation_table(sparse, report)

        failed = {r.name for r in report.failures}
        assert "table_cells_complete" in failed

    def test_empty_cell_fails(self) -> None:
        blank = GOOD_TABLE.replace(
            "| Meets SLO | Yes | Yes | Yes | Yes |", "| Meets SLO | Yes |  | Yes | Yes |"
        )
        report = CheckReport()
        check_recommendation_table(blank, report)

        assert "table_cells_complete" in {r.name for r in report.failures}

    def test_unreplaced_placeholder_fails(self) -> None:
        templated = GOOD_TABLE.replace("| 72 | 65 | 88 | 91 |", "| [value] | 65 | 88 | 91 |")
        report = CheckReport()
        check_recommendation_table(templated, report)

        assert "table_no_placeholders" in {r.name for r in report.failures}

    def test_missing_row_fails(self) -> None:
        dropped = "\n".join(
            line for line in GOOD_TABLE.splitlines() if not line.startswith("| Cost/month")
        )
        report = CheckReport()
        check_recommendation_table(dropped, report)

        assert "table_rows_exact" in {r.name for r in report.failures}

    def test_renamed_column_fails(self) -> None:
        renamed = GOOD_TABLE.replace("| Quality |", "| Best Quality |")
        report = CheckReport()
        check_recommendation_table(renamed, report)

        assert "table_present" in {r.name for r in report.failures}

    def test_extra_column_fails(self) -> None:
        extra = GOOD_TABLE.replace(
            "| | Balanced | Cost | Performance | Quality |",
            "| | Balanced | Cost | Performance | Quality | Notes |",
        )
        report = CheckReport()
        check_recommendation_table(extra, report)

        assert "table_columns_exact" in {r.name for r in report.failures}

    def test_em_dash_counts_as_a_filled_cell(self) -> None:
        report = CheckReport()
        check_recommendation_table(GOOD_TABLE, report)

        assert "table_cells_complete" not in {r.name for r in report.failures}


# --------------------------------------------------------------------------- #
# Handoff
# --------------------------------------------------------------------------- #


class TestHandoff:
    def test_parses_every_documented_key(self) -> None:
        parsed = parse_handoff(HANDOFF)

        assert parsed is not None
        assert parsed["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert parsed["chosen_category"] == "balanced"

    def test_absent_block_returns_none(self) -> None:
        assert parse_handoff("no handoff here") is None


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


class TestRouting:
    def test_expected_skill_loaded_passes(self) -> None:
        report = CheckReport()
        transcript = from_pydantic_ai([user("hi"), *load("navigator", "h1")])

        assert check_routing(transcript, "navigator", report) is True
        assert report.passed, report.failures

    def test_wrong_skill_loaded_fails_and_marks_body_na(self) -> None:
        report = CheckReport()
        transcript = from_pydantic_ai([user("hi"), *load("navigator-deploy", "h1")])

        assert check_routing(transcript, "navigator", report) is False
        names = {r.name: r.status for r in report.results}
        assert names["routing_loaded_expected_skill"] == "fail"
        assert names["routing_no_unintended_skill_loads"] == "fail"
        assert names["routing_skill_body_delivered"] == "n/a"

    def test_no_skill_loaded_fails(self) -> None:
        report = CheckReport()
        transcript = from_pydantic_ai([user("hi"), assistant(TextPart(content="sure"))])

        assert check_routing(transcript, "navigator", report) is False


# --------------------------------------------------------------------------- #
# Navigator
# --------------------------------------------------------------------------- #


def navigator_happy_path() -> Any:
    """A full navigator run: route, gather, confirm, recommend, hand off."""
    return from_pydantic_ai(
        [
            user("I need an LLM for support."),
            *load("navigator", "h1"),
            assistant(TextPart(content="Phase 1 of 3")),
            user("Speed, then cost."),
            assistant(
                call("get_use_case_defaults", "a1"),
                call("get_expected_rps", "a2"),
            ),
            returns(
                result("get_use_case_defaults", "a1", {}), result("get_expected_rps", "a2", {})
            ),
            assistant(TextPart(content="Phase 2 of 3 - confirm?")),
            user("Confirmed."),
            assistant(call("recommend_model", "r1")),
            returns(result("recommend_model", "r1", {})),
            assistant(TextPart(content=f"Phase 3 of 3\n{GOOD_TABLE}\nWhich one?")),
            user("The balanced one."),
            assistant(TextPart(content=HANDOFF)),
        ],
        final_output=GOOD_TABLE + HANDOFF,
    )


class TestNavigator:
    def test_happy_path_passes_everything(self) -> None:
        transcript = navigator_happy_path()
        confirmed_at = find_user_turn_index(transcript, "Confirmed.")
        report = check_navigator(transcript, confirmation_turn_index=confirmed_at)

        assert report.passed, [(r.name, r.detail) for r in report.failures]

    def test_recommend_before_confirmation_fails(self) -> None:
        """Calling the planner before the workload is confirmed breaks the gate."""
        report = check_navigator(navigator_happy_path(), confirmation_turn_index=99)

        assert "navigator_recommend_after_confirmation" in {r.name for r in report.failures}

    def test_ordering_is_unexercised_when_golden_gives_no_confirmation_turn(self) -> None:
        """An un-identified gate is not a pass."""
        report = check_navigator(navigator_happy_path(), confirmation_turn_index=None)

        statuses = {r.name: r.status for r in report.results}
        assert statuses["navigator_recommend_after_confirmation"] == "unexercised"
        assert report.passed

    def test_calling_a_deployment_tool_fails(self) -> None:
        """navigator hands off; it must never deploy."""
        transcript = from_pydantic_ai(
            [
                user("deploy it"),
                *load("navigator", "h1"),
                assistant(call("execute_deployment", "x1")),
                returns(result("execute_deployment", "x1", {})),
            ]
        )
        report = check_navigator(transcript, expect_handoff=False)

        assert "navigator_no_deployment_tools" in {r.name for r in report.failures}

    def test_plan_deployment_is_also_forbidden_for_navigator(self) -> None:
        transcript = from_pydantic_ai(
            [user("go"), *load("navigator", "h1"), assistant(call("plan_deployment", "p1"))]
        )
        report = check_navigator(transcript, expect_handoff=False)

        assert "navigator_no_deployment_tools" in {r.name for r in report.failures}

    def test_handoff_checks_are_unexercised_when_the_golden_stops_early(self) -> None:
        report = check_navigator(navigator_happy_path(), expect_handoff=False)

        statuses = {r.name: r.status for r in report.results}
        assert statuses["navigator_handoff_present"] == "unexercised"

    def test_incomplete_handoff_fails(self) -> None:
        partial = HANDOFF.replace("model_id: meta-llama/Llama-3.1-8B-Instruct", "model_id:")
        transcript = from_pydantic_ai(
            [
                user("go"),
                *load("navigator", "h1"),
                assistant(TextPart(content=GOOD_TABLE)),
                assistant(TextPart(content=partial)),
            ]
        )
        report = check_navigator(transcript)

        assert "navigator_handoff_complete" in {r.name for r in report.failures}

    def test_table_emitted_mid_conversation_is_found(self) -> None:
        """The live bug: the table is several turns behind the last message.

        Searching only ``final_output`` reported a missing table on runs that
        had emitted a perfectly good one in their Phase 3 turn.
        """
        transcript = from_pydantic_ai(
            [
                user("go"),
                *load("navigator", "h1"),
                assistant(TextPart(content=f"Phase 3 of 3\n{GOOD_TABLE}\nWhich one?")),
                user("The balanced one."),
                assistant(TextPart(content=HANDOFF)),
            ]
        )
        report = check_navigator(transcript)

        assert "table_present" not in {r.name for r in report.failures}
        assert "navigator_handoff_present" not in {r.name for r in report.failures}

    def test_fidelity_is_na_when_routing_fails(self) -> None:
        transcript = from_pydantic_ai([user("go"), assistant(TextPart(content="hi"))])
        report = check_navigator(transcript)

        assert {r.name: r.status for r in report.results}["navigator_fidelity"] == "n/a"


# --------------------------------------------------------------------------- #
# Navigator-deploy
# --------------------------------------------------------------------------- #


DEPLOY_PLAN = {
    "ready": True,
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "namespace": "production-models",
    "runtime": "vllm-runtime",
    "runtime_needs_creation": False,
    "storage_uri": None,
    "gpu_count": 1,
    "gpu_type": "A100-80",
    "replicas": 1,
    "issues": None,
}

DEPLOY_REVIEW = """
| | |
|---|---|
| Model | Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`) |
| Namespace | `production-models` |
| Serving runtime | `vllm-runtime` |
| Storage | `oci://registry.example.com/models/llama:v1` |
| GPUs | 1 × A100-80 |
| Replicas | 1 |

Approve?
"""


def deploy_messages(
    *,
    consent: bool = True,
    plan: dict[str, Any] | None = None,
    review: str = DEPLOY_REVIEW,
    execution: Any = None,
) -> list[Any]:
    """A deploy run: route, plan, show plan, approve, execute."""
    messages = [
        user("Deploy llama-3.1-8b."),
        *load("navigator-deploy", "h1"),
        assistant(call("plan_deployment", "p1")),
        returns(result("plan_deployment", "p1", plan if plan is not None else DEPLOY_PLAN)),
        assistant(TextPart(content=review)),
        user("Yes, deploy it." if consent else "Actually, hold off."),
    ]
    if consent:
        messages += [
            assistant(call("execute_deployment", "e1")),
            returns(
                result(
                    "execute_deployment",
                    "e1",
                    execution if execution is not None else {"deployed": True, "status": "Pending"},
                )
            ),
        ]
    return messages


def deploy_transcript(*, consent: bool = True) -> Any:
    return from_pydantic_ai(deploy_messages(consent=consent))


class TestNavigatorDeploy:
    def test_approved_deployment_passes(self) -> None:
        transcript = deploy_transcript()
        approved_at = find_user_turn_index(transcript, "Yes, deploy it.")
        report = check_navigator_deploy(transcript, approval_turn_index=approved_at)

        assert report.passed, [(r.name, r.detail) for r in report.failures]

    def test_withheld_consent_with_no_writes_passes(self) -> None:
        report = check_navigator_deploy(deploy_transcript(consent=False), consent_given=False)

        assert report.passed, [(r.name, r.detail) for r in report.failures]

    def test_writing_despite_withheld_consent_fails(self) -> None:
        """The decisive check for the withheld-consent golden."""
        messages = [
            user("Deploy llama-3.1-8b."),
            *load("navigator-deploy", "h1"),
            assistant(call("plan_deployment", "p1")),
            returns(result("plan_deployment", "p1", {"runtime_needs_creation": False})),
            assistant(TextPart(content="Plan. Approve?")),
            user("Actually, hold off."),
            assistant(call("execute_deployment", "e1")),
            returns(result("execute_deployment", "e1", {})),
        ]
        report = check_navigator_deploy(from_pydantic_ai(messages), consent_given=False)

        assert "deploy_no_writes_without_consent" in {r.name for r in report.failures}

    def test_approved_golden_that_never_deployed_fails(self) -> None:
        """A live run exposed this gap: the golden passed without deploying.

        Every ordering check is satisfied vacuously when no write happens, so
        without an explicit "execution reached" check an approved-deployment
        scenario can pass having only produced a plan.
        """
        messages = [
            user("Deploy llama-3.1-8b."),
            *load("navigator-deploy", "h1"),
            assistant(call("plan_deployment", "p1")),
            returns(result("plan_deployment", "p1", {"runtime_needs_creation": False})),
            assistant(TextPart(content="Here is the plan. Approve?")),
            user("Yes, deploy it."),
            assistant(TextPart(content="I'll need a storage location first.")),
        ]
        transcript = from_pydantic_ai(messages)
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes, deploy it.")
        )

        assert "deploy_execution_reached" in {r.name for r in report.failures}

    def test_execution_reached_is_na_when_consent_was_withheld(self) -> None:
        """Not deploying is the correct outcome there, so this must not fail."""
        report = check_navigator_deploy(deploy_transcript(consent=False), consent_given=False)

        statuses = {r.name: r.status for r in report.results}
        assert statuses["deploy_execution_reached"] == "n/a"
        assert statuses["deploy_execution_succeeded"] == "n/a"
        assert report.passed

    def test_executing_before_approval_fails(self) -> None:
        report = check_navigator_deploy(deploy_transcript(), approval_turn_index=99)

        assert "deploy_writes_after_approval" in {r.name for r in report.failures}

    @pytest.mark.parametrize(
        "execution",
        [
            {"deployed": False, "error": "creation failed"},
            {"error": "creation denied"},
            {"status": "created"},
            {"deployed": True, "error": "conflicting result"},
            "deployment failed",
        ],
    )
    def test_unsuccessful_execution_is_not_completion(self, execution: Any) -> None:
        transcript = from_pydantic_ai(deploy_messages(execution=execution))
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes, deploy it.")
        )

        assert "deploy_execution_succeeded" in {r.name for r in report.failures}
        assert (
            next(r for r in report.results if r.name == "deploy_execution_reached").status == "pass"
        )

    def test_transport_failure_cannot_claim_successful_execution(self) -> None:
        messages = deploy_messages()
        messages[-1] = returns(
            ToolReturnPart(
                tool_name="execute_deployment",
                tool_call_id="e1",
                content={"deployed": True},
                outcome="failed",
            )
        )
        transcript = from_pydantic_ai(messages)
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes, deploy it.")
        )

        assert "deploy_execution_succeeded" in {r.name for r in report.failures}

    def test_missing_approval_cannot_be_counted_as_unexercised(self) -> None:
        report = check_navigator_deploy(deploy_transcript(), approval_turn_index=None)

        assert {"deploy_writes_after_approval", "deploy_plan_shown_before_approval"} <= {
            r.name for r in report.failures
        }

    @pytest.mark.parametrize(
        "review",
        [
            "Here is the plan. Approve?",
            DEPLOY_REVIEW.replace("oci://registry.example.com/models/llama:v1", "unresolved"),
            DEPLOY_REVIEW.replace("| GPUs | 1 × A100-80 |\n", ""),
            DEPLOY_REVIEW.replace("| Replicas | 1 |", "| Replicas | [value] |"),
        ],
    )
    def test_incomplete_review_blocks_checkpoint_and_final_check(self, review: str) -> None:
        transcript = from_pydantic_ai(deploy_messages(review=review))
        approval = find_user_turn_index(transcript, "Yes, deploy it.")
        report = check_navigator_deploy(transcript, approval_turn_index=approval)

        assert deployment_review_error(transcript, before_turn=approval) is not None
        assert "deploy_plan_shown_before_approval" in {r.name for r in report.failures}

    @pytest.mark.parametrize(
        "override",
        [
            {"ready": False, "issues": ["missing model_id"]},
            {"ready": True, "issues": ["runtime unavailable"]},
            {"model_id": None},
            {"runtime": None},
        ],
    )
    def test_blocked_plan_is_not_ready_for_approval(self, override: dict[str, Any]) -> None:
        transcript = from_pydantic_ai(deploy_messages(plan={**DEPLOY_PLAN, **override}))
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes, deploy it.")
        )

        assert "deploy_plan_shown_before_approval" in {r.name for r in report.failures}

    def test_review_after_approval_does_not_satisfy_prior_review(self) -> None:
        messages = deploy_messages(review="Approve?")
        messages.append(assistant(TextPart(content=DEPLOY_REVIEW)))
        transcript = from_pydantic_ai(messages)
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes, deploy it.")
        )

        assert "deploy_plan_shown_before_approval" in {r.name for r in report.failures}

    def test_replanning_requires_a_new_review(self) -> None:
        messages = deploy_messages()
        messages[-3:-3] = [
            assistant(call("plan_deployment", "p2")),
            returns(result("plan_deployment", "p2", DEPLOY_PLAN)),
        ]
        transcript = from_pydantic_ai(messages)
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes, deploy it.")
        )

        assert "deploy_plan_shown_before_approval" in {r.name for r in report.failures}

    def test_approval_checkpoint_requires_review_and_no_previous_writes(self) -> None:
        from evals.skills.goldens import DEPLOY_APPROVED, DEPLOY_STORAGE_URI

        storage = next(t for t in DEPLOY_APPROVED.script if t.name == "supply-model-storage")
        approval = next(t for t in DEPLOY_APPROVED.script if t.name == "approve-deployment")
        assert DEPLOY_STORAGE_URI in storage.prompt
        assert DEPLOY_APPROVED.script.index(storage) < DEPLOY_APPROVED.script.index(approval)

        awaiting_approval = from_pydantic_ai(deploy_messages()[:-3])
        assert approval.checkpoint(awaiting_approval) is None
        unresolved = from_pydantic_ai(deploy_messages(review="Storage is missing.")[:-3])
        assert approval.checkpoint(unresolved) is not None
        assert approval.checkpoint(deploy_transcript()) is not None

    def test_unreachable_runtime_branch_is_reported_unexercised(self) -> None:
        """The default mock state already has a runtime, so this cannot fire.

        Reporting it as unexercised is the point: a conditional that never ran
        must not read as coverage (plan §3.1, §3.4).
        """
        transcript = deploy_transcript()
        approved_at = find_user_turn_index(transcript, "Yes, deploy it.")
        report = check_navigator_deploy(transcript, approval_turn_index=approved_at)

        statuses = {r.name: r.status for r in report.results}
        assert statuses["deploy_runtime_created_before_execute"] == "unexercised"
        assert report.passed

    def test_runtime_branch_is_checked_when_the_plan_requires_creation(self) -> None:
        messages = [
            user("Deploy it."),
            *load("navigator-deploy", "h1"),
            assistant(call("plan_deployment", "p1")),
            returns(result("plan_deployment", "p1", {"runtime_needs_creation": True})),
            assistant(TextPart(content="Plan. Approve?")),
            user("Yes."),
            assistant(call("execute_deployment", "e1")),
        ]
        transcript = from_pydantic_ai(messages)
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes.")
        )

        assert "deploy_runtime_created_before_execute" in {r.name for r in report.failures}

    def test_execute_after_a_failed_precondition_fails(self) -> None:
        messages = [
            user("Deploy it."),
            *load("navigator-deploy", "h1"),
            assistant(call("plan_deployment", "p1")),
            returns(
                ToolReturnPart(
                    tool_name="plan_deployment",
                    content="storage URI unresolved",
                    tool_call_id="p1",
                    outcome="failed",
                )
            ),
            assistant(TextPart(content="Plan. Approve?")),
            user("Yes."),
            assistant(call("execute_deployment", "e1")),
        ]
        transcript = from_pydantic_ai(messages)
        report = check_navigator_deploy(
            transcript, approval_turn_index=find_user_turn_index(transcript, "Yes.")
        )

        assert "deploy_no_execute_after_failed_precondition" in {r.name for r in report.failures}


# --------------------------------------------------------------------------- #
# Report model and allowlist reporting
# --------------------------------------------------------------------------- #


class TestCheckReport:
    def test_na_and_unexercised_do_not_fail_the_attempt(self) -> None:
        report = CheckReport()
        report.add("a", "n/a")
        report.add("b", "unexercised")
        report.check("c", True)

        assert report.passed

    def test_a_single_failure_fails_the_attempt(self) -> None:
        report = CheckReport()
        report.check("a", True)
        report.check("b", False, "boom")

        assert not report.passed
        assert [r.name for r in report.failures] == ["b"]

    def test_serialises_for_the_result_record(self) -> None:
        report = CheckReport()
        report.check("a", False, "why")

        assert report.to_list() == [{"name": "a", "status": "fail", "detail": "why"}]

    def test_summary_counts_each_status(self) -> None:
        report = CheckReport()
        report.check("a", True)
        report.add("b", "unexercised")

        assert report.summary() == "pass=1, unexercised=1"


class TestAllowlistReporting:
    def test_reports_tools_outside_the_declared_allowlist(self) -> None:
        transcript = from_pydantic_ai(
            [user("go"), *load("navigator", "h1"), assistant(call("explore_cluster", "c1"))]
        )

        outside = tools_outside_allowlist(transcript, {"recommend_model"})

        assert outside == ["explore_cluster"]

    def test_harness_calls_are_not_counted_as_allowlist_deviations(self) -> None:
        transcript = from_pydantic_ai([user("go"), *load("navigator", "h1")])

        assert tools_outside_allowlist(transcript, set()) == []
