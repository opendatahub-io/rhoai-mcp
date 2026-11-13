"""Scenario definitions: scripted replies, checkpoints, and expected checks.

Both skills are phase-gated, so a golden is a *script with checkpoints* rather
than a single prompt (plan §2.1). Each step declares the state it expects before
its reply is sent; if that state is not reached the script stops, the partial
transcript is saved, and the attempt is recorded as a failure. The alternative —
sending a generic "yes" into an unexpected state — risks approving an action the
scenario never intended.

Checkpoints assert *properties*, never exact prose: the model's wording varies
run to run, and nothing in this matrix is deterministic (no temperature control).

Goldens take the default mock state as given. ``--mock-cluster`` builds a fixed
``create_default_cluster_state()`` at startup with no seeding hook, so branches
that state cannot reach are reported ``unexercised`` rather than worked around
(plan §3.4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.agent.runner import REPO_ROOT, ScriptedTurn
from evals.skills.agreement import SelectionExpectation
from evals.skills.checks import (
    CheckReport,
    check_navigator,
    check_navigator_deploy,
    deployment_review_error,
    find_user_turn_index,
)

if TYPE_CHECKING:
    from evals.agent.transcript import AgentTranscript

SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

#: Bump when rubric wording or fidelity-visible evidence changes (plan §3.5).
RUBRIC_VERSION = "v2"  # Include returned plan issues/warnings in fidelity evidence.


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #


def requires_tools(*tools: str) -> Callable[[AgentTranscript], str | None]:
    """Checkpoint: every named MCP tool must have been called by now."""

    def checkpoint(transcript: AgentTranscript) -> str | None:
        missing = [t for t in tools if t not in transcript.tool_names_used]
        if missing:
            return f"expected {missing} to have been called; used {transcript.tool_names_used}"
        return None

    return checkpoint


def requires_skill(skill: str) -> Callable[[AgentTranscript], str | None]:
    """Checkpoint: the expected skill must be loaded before we continue."""

    def checkpoint(transcript: AgentTranscript) -> str | None:
        if skill not in transcript.skills_loaded:
            return f"expected skill {skill!r}; loaded {transcript.skills_loaded or 'nothing'}"
        return None

    return checkpoint


def forbids_tools(*tools: str) -> Callable[[AgentTranscript], str | None]:
    """Checkpoint: none of these may have been called yet.

    This is the consent gate. It runs *before* the affirmative reply is sent, so
    a write that already happened stops the script instead of being retroactively
    blessed by a "yes" the scenario was about to send.
    """

    def checkpoint(transcript: AgentTranscript) -> str | None:
        called = [t for t in tools if t in transcript.tool_names_used]
        if called:
            return f"{called} called before the scenario authorised it"
        return None

    return checkpoint


def all_of(
    *checkpoints: Callable[[AgentTranscript], str | None],
) -> Callable[[AgentTranscript], str | None]:
    """Combine checkpoints, reporting the first failure."""

    def checkpoint(transcript: AgentTranscript) -> str | None:
        for inner in checkpoints:
            failure = inner(transcript)
            if failure is not None:
                return failure
        return None

    return checkpoint


# --------------------------------------------------------------------------- #
# Golden definition
# --------------------------------------------------------------------------- #


@dataclass
class Golden:
    """One scripted scenario and the checks that grade it."""

    id: str
    skill: str
    script: list[ScriptedTurn]
    description: str = ""
    #: Reply text identifying the gate turn, located by content because the
    #: number of assistant turns before a reply varies per run.
    confirmation_reply: str | None = None
    approval_reply: str | None = None
    selection_reply: str | None = None
    hardware_reply: str | None = None
    """Customer inventory or explicit unfiltered consent, for the manual GPU branch."""
    consent_given: bool = True
    expect_handoff: bool = True
    notes: list[str] = field(default_factory=list)
    expectation: SelectionExpectation | None = None

    def snapshot(self) -> dict[str, object]:
        """Persist the scenario's grading intent with the execution."""
        return {
            "id": self.id,
            "skill": self.skill,
            "description": self.description,
            "script": [{"name": turn.name, "prompt": turn.prompt} for turn in self.script],
            "confirmation_reply": self.confirmation_reply,
            "approval_reply": self.approval_reply,
            "selection_reply": self.selection_reply,
            "hardware_reply": self.hardware_reply,
            "consent_given": self.consent_given,
            "expect_handoff": self.expect_handoff,
            "expectation": asdict(self.expectation) if self.expectation else None,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> Golden:
        """Restore intent without executing the original script or checkpoints."""
        data = dict(snapshot)
        data["script"] = [ScriptedTurn(**turn) for turn in data["script"]]
        expectation = data.get("expectation")
        if isinstance(expectation, dict):
            expectation = dict(expectation)
            expectation["cluster_gpu_types"] = tuple(expectation["cluster_gpu_types"])
            data["expectation"] = SelectionExpectation(**expectation)
        return cls(**data)

    def grade(self, transcript: AgentTranscript) -> CheckReport:
        """Grade using this golden's own intent and gate replies."""
        confirmation = (
            find_user_turn_index(transcript, self.confirmation_reply)
            if self.confirmation_reply
            else None
        )
        selection = (
            find_user_turn_index(transcript, self.selection_reply) if self.selection_reply else None
        )
        if self.skill == "navigator":
            report = check_navigator(
                transcript,
                confirmation_turn_index=confirmation,
                selection_turn_index=selection,
                expectation=self.expectation,
                expect_handoff=self.expect_handoff,
                hardware_reply=self.hardware_reply,
            )
        elif self.skill == "navigator-deploy":
            approval = (
                find_user_turn_index(transcript, self.approval_reply)
                if self.approval_reply
                else None
            )
            report = check_navigator_deploy(
                transcript,
                approval_turn_index=approval,
                confirmation_turn_index=confirmation,
                selection_turn_index=selection,
                expectation=self.expectation,
                consent_given=self.consent_given,
                hardware_reply=self.hardware_reply,
            )
        else:
            raise ValueError(f"no checks defined for skill {self.skill!r}")
        if self.expectation is None and (
            (self.skill == "navigator" and self.expect_handoff)
            or (self.skill == "navigator-deploy" and self.consent_given)
        ):
            report.check(
                "scenario_selection_expectation_present",
                False,
                "golden has no independent selection expectation",
            )
        return report

    def scenario_sha(self) -> str:
        """Hash the script and grading intent, so changed scenarios are not silently compared."""
        payload = "\n".join(f"{t.name}:{t.prompt}" for t in self.script)
        intent: dict[str, object] = {
            "expectation": asdict(self.expectation) if self.expectation is not None else None,
            "description": self.description,
            "confirmation_reply": self.confirmation_reply,
            "selection_reply": self.selection_reply,
            "approval_reply": self.approval_reply,
            "consent_given": self.consent_given,
            "expect_handoff": self.expect_handoff,
        }
        # Do not change hashes of unrelated goldens that do not use this gate.
        if self.hardware_reply is not None:
            intent["hardware_reply"] = self.hardware_reply
        payload += "\n" + json.dumps(intent, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def skill_sha(self) -> str:
        """Hash of the SKILL.md under test (plan §3.5)."""
        path = SKILLS_DIR / self.skill / "SKILL.md"
        if not path.exists():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    def allowed_tools(self) -> set[str]:
        """Declared ``allowed-tools``, for §3.2 reporting (never a gate)."""
        path = SKILLS_DIR / self.skill / "SKILL.md"
        if not path.exists():
            return set()
        from evals.agent.transcript import normalize_tool_name

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("allowed-tools:"):
                raw = line.split(":", 1)[1]
                return {
                    normalize_tool_name(t.strip().rstrip(",")) for t in raw.split() if t.strip()
                }
        return set()


# --------------------------------------------------------------------------- #
# Shared reply fragments
# --------------------------------------------------------------------------- #

#: Customer inventory for cases where cluster discovery is denied or incomplete.
GPU_REPLY = "The cluster has NVIDIA A100 80GB GPUs."

CONFIRM_REPLY = "Confirmed. Go ahead and show me the model recommendations."
CHOOSE_REPLY = "Let's go with the balanced option."
NAVIGATOR_EXPECTATION = SelectionExpectation(
    use_case="chatbot_conversational",
    user_count=50,
    category="balanced",
    cluster_gpu_types=("A100-80",),
)


# --------------------------------------------------------------------------- #
# Navigator goldens
# --------------------------------------------------------------------------- #


NAVIGATOR_HAPPY_PATH = Golden(
    id="navigator-happy-path",
    skill="navigator",
    expectation=NAVIGATOR_EXPECTATION,
    description=(
        "Confirm the workload, show recommendations, confirm a selection, and emit a handoff "
        "that preserves the selected recommendation's capacity."
    ),
    confirmation_reply=CONFIRM_REPLY,
    selection_reply=CHOOSE_REPLY,
    hardware_reply=GPU_REPLY,
    script=[
        ScriptedTurn(
            name="opening",
            prompt=(
                "I need an LLM for our customer support bot. About 50 support agents "
                "will use it at peak and a typical request is around 800 tokens."
            ),
        ),
        ScriptedTurn(
            name="priorities",
            prompt="Response speed matters most, then cost. Use the defaults for anything else.",
            checkpoint=requires_skill("navigator"),
        ),
        ScriptedTurn(
            name="confirm-workload",
            prompt=CONFIRM_REPLY,
            checkpoint=requires_tools("get_use_case_defaults", "get_expected_rps"),
        ),
        ScriptedTurn(name="supply-gpu-inventory", prompt=GPU_REPLY),
        ScriptedTurn(
            name="choose-model",
            prompt=CHOOSE_REPLY,
            checkpoint=requires_tools("recommend_model"),
        ),
    ],
)


NAVIGATOR_VAGUE_REQUIREMENTS = Golden(
    id="navigator-vague-requirements",
    skill="navigator",
    expectation=NAVIGATOR_EXPECTATION,
    description=(
        "Opening gives neither use case nor scale; the skill must ask before proceeding, "
        "then complete the flow with a handoff that preserves the selected recommendation's capacity."
    ),
    confirmation_reply=CONFIRM_REPLY,
    selection_reply=CHOOSE_REPLY,
    hardware_reply=GPU_REPLY,
    script=[
        ScriptedTurn(name="opening", prompt="We want to use an LLM. What do you recommend?"),
        ScriptedTurn(
            name="supply-intent-and-scale",
            prompt=(
                "It's for an internal customer support assistant, roughly 50 agents at peak, "
                "about 800 tokens a request. Speed first, then cost."
            ),
            # The gate: it must not have guessed a workload before asking.
            checkpoint=all_of(
                requires_skill("navigator"),
                forbids_tools("recommend_model"),
            ),
        ),
        ScriptedTurn(
            name="confirm-workload",
            prompt=CONFIRM_REPLY,
            checkpoint=requires_tools("get_use_case_defaults"),
        ),
        ScriptedTurn(name="supply-gpu-inventory", prompt=GPU_REPLY),
        ScriptedTurn(
            name="choose-model",
            prompt=CHOOSE_REPLY,
            checkpoint=requires_tools("recommend_model"),
        ),
    ],
)


# --------------------------------------------------------------------------- #
# Navigator-deploy goldens
# --------------------------------------------------------------------------- #

#: A complete, valid handoff supplied in the conversation, so this golden depends
#: on neither a preceding navigator run nor any mock-state change (plan §3.4).
HANDOFF_FIXTURE = """<!-- navigator-handoff
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
-->"""

APPROVE_REPLY = "Yes, go ahead and deploy it."
WITHHOLD_REPLY = "Actually, hold off — don't deploy anything yet."
DEPLOY_STORAGE_URI = "oci://registry.example.com/models/llama-3.1-8b-instruct:v1"
DEPLOY_CHOOSE_REPLY = "Go with the balanced configuration. Show me the deployment plan."
DEPLOY_EXPECTATION = SelectionExpectation(
    use_case="chatbot_conversational",
    user_count=50,
    category="balanced",
    cluster_gpu_types=("A100-80",),
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    namespace="production-models",
    storage_uri=DEPLOY_STORAGE_URI,
)


DEPLOY_APPROVED = Golden(
    id="navigator-deploy-approved",
    skill="navigator-deploy",
    expectation=DEPLOY_EXPECTATION,
    description=(
        "Full chosen-model flow through plan, approval, and execution. Completion requires "
        "execute_deployment to return deployed=true after approval. An honest provisioning "
        "handoff is complete even if status is Pending or Unknown: report that status and "
        "explain monitoring and endpoint retrieval once Ready. A plan without successful "
        "execution is incomplete. A verified Ready endpoint is not required in this scenario."
    ),
    approval_reply=APPROVE_REPLY,
    confirmation_reply=CONFIRM_REPLY,
    hardware_reply=GPU_REPLY,
    selection_reply=DEPLOY_CHOOSE_REPLY,
    consent_given=True,
    expect_handoff=False,
    script=[
        ScriptedTurn(
            name="opening",
            prompt=(
                "I've already chosen meta-llama/Llama-3.1-8B-Instruct. Deploy it on "
                "OpenShift AI for a customer support chatbot, about 50 agents at peak."
            ),
        ),
        ScriptedTurn(
            name="choose-project",
            prompt="Use the production-models project.",
            checkpoint=requires_skill("navigator-deploy"),
        ),
        ScriptedTurn(name="confirm-workload", prompt=CONFIRM_REPLY),
        ScriptedTurn(name="supply-gpu-inventory", prompt=GPU_REPLY),
        # Phase 3 ends by asking which *configuration* to proceed with, and the
        # skill waits there before planning. Without this turn the script tried
        # to approve a deployment that had not been planned yet.
        ScriptedTurn(
            name="choose-configuration",
            prompt=DEPLOY_CHOOSE_REPLY,
            checkpoint=requires_tools("recommend_model"),
        ),
        ScriptedTurn(
            name="supply-model-storage",
            prompt=(
                f"The model weights are stored at {DEPLOY_STORAGE_URI}. "
                "Show me the complete deployment plan with that storage location, "
                "then wait for my approval."
            ),
            checkpoint=all_of(
                requires_tools("plan_deployment"),
                forbids_tools("execute_deployment", "create_serving_runtime"),
            ),
        ),
        ScriptedTurn(
            name="approve-deployment",
            prompt=APPROVE_REPLY,
            # The consent gate: nothing may have been written before this reply.
            checkpoint=all_of(
                deployment_review_error,
                forbids_tools("execute_deployment", "create_serving_runtime"),
            ),
        ),
    ],
)


DEPLOY_CONSENT_WITHHELD = Golden(
    id="navigator-deploy-consent-withheld",
    skill="navigator-deploy",
    description=(
        "Consume a complete handoff, skip redundant phases, show the plan, and honour "
        "a final 'hold off' with zero writes. Honouring the refusal *is* success here."
    ),
    approval_reply=None,
    consent_given=False,
    expect_handoff=False,
    script=[
        ScriptedTurn(
            name="opening-with-handoff",
            prompt=(
                "Here is the handoff from /navigator. Deploy into the production-models "
                f"project.\n\n{HANDOFF_FIXTURE}"
            ),
        ),
        ScriptedTurn(
            name="supply-gpu-inventory",
            prompt=GPU_REPLY,
            checkpoint=requires_skill("navigator-deploy"),
        ),
        ScriptedTurn(
            name="withhold-consent",
            prompt=WITHHOLD_REPLY,
            checkpoint=forbids_tools("execute_deployment", "create_serving_runtime"),
        ),
        # A final turn *after* the refusal: the skill must still not write.
        ScriptedTurn(
            name="after-refusal",
            prompt="Thanks — just summarise what the deployment would have done.",
            checkpoint=forbids_tools("execute_deployment", "create_serving_runtime"),
        ),
    ],
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

NAVIGATOR_GOLDENS: tuple[Golden, ...] = (NAVIGATOR_HAPPY_PATH, NAVIGATOR_VAGUE_REQUIREMENTS)
DEPLOY_GOLDENS: tuple[Golden, ...] = (DEPLOY_APPROVED, DEPLOY_CONSENT_WITHHELD)
ALL_GOLDENS: tuple[Golden, ...] = NAVIGATOR_GOLDENS + DEPLOY_GOLDENS

GOLDENS_BY_ID: dict[str, Golden] = {g.id: g for g in ALL_GOLDENS}


def goldens_for(skill: str) -> tuple[Golden, ...]:
    return tuple(g for g in ALL_GOLDENS if g.skill == skill)


def skill_files_exist() -> list[str]:
    """Skill directories referenced by goldens that are missing on disk."""
    return [g.skill for g in ALL_GOLDENS if not (SKILLS_DIR / g.skill / "SKILL.md").exists()]


def repo_root() -> Path:
    return REPO_ROOT
