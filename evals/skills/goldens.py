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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from evals.agent.runner import REPO_ROOT, ScriptedTurn
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

#: Bump when a rubric's wording changes, so scores stay comparable (plan §3.5).
RUBRIC_VERSION = "v1"


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
    grade: Callable[[AgentTranscript], CheckReport]
    description: str = ""
    #: Reply text identifying the gate turn, located by content because the
    #: number of assistant turns before a reply varies per run.
    confirmation_reply: str | None = None
    approval_reply: str | None = None
    consent_given: bool = True
    expect_handoff: bool = True
    notes: list[str] = field(default_factory=list)

    def scenario_sha(self) -> str:
        """Hash of the script, so a changed scenario is never silently compared."""
        payload = "\n".join(f"{t.name}:{t.prompt}" for t in self.script)
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

#: `get_cluster_resources` is absent from the default plugin set, so Phase 3
#: takes the documented ask-the-customer branch. Supplying GPUs here is what
#: unblocks `recommend_model`. The automated-discovery branch and the GPU
#: mapping table stay unexercised until that tool ships (plan §3.1).
GPU_REPLY = "The cluster has NVIDIA A100 80GB GPUs."

CONFIRM_REPLY = "Confirmed. Go ahead and show me the model recommendations."
CHOOSE_REPLY = "Let's go with the balanced option."


# --------------------------------------------------------------------------- #
# Navigator goldens
# --------------------------------------------------------------------------- #


def _grade_navigator(
    confirmation_reply: str | None, expect_handoff: bool
) -> Callable[[AgentTranscript], CheckReport]:
    def grade(transcript: AgentTranscript) -> CheckReport:
        index = find_user_turn_index(transcript, confirmation_reply) if confirmation_reply else None
        return check_navigator(
            transcript, confirmation_turn_index=index, expect_handoff=expect_handoff
        )

    return grade


NAVIGATOR_HAPPY_PATH = Golden(
    id="navigator-happy-path",
    skill="navigator",
    description=(
        "Confirm the workload, show recommendations, confirm a selection, emit the handoff."
    ),
    confirmation_reply=CONFIRM_REPLY,
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
    grade=_grade_navigator(CONFIRM_REPLY, expect_handoff=True),
    notes=["GPU discovery branch unexercised: get_cluster_resources not in default plugin set"],
)


NAVIGATOR_VAGUE_REQUIREMENTS = Golden(
    id="navigator-vague-requirements",
    skill="navigator",
    description=(
        "Opening gives neither use case nor scale; the skill must ask before proceeding, "
        "then complete the flow."
    ),
    confirmation_reply=CONFIRM_REPLY,
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
    grade=_grade_navigator(CONFIRM_REPLY, expect_handoff=True),
    notes=["GPU discovery branch unexercised: get_cluster_resources not in default plugin set"],
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


def _grade_deploy(
    approval_reply: str | None, consent_given: bool
) -> Callable[[AgentTranscript], CheckReport]:
    def grade(transcript: AgentTranscript) -> CheckReport:
        index = find_user_turn_index(transcript, approval_reply) if approval_reply else None
        return check_navigator_deploy(
            transcript, approval_turn_index=index, consent_given=consent_given
        )

    return grade


DEPLOY_APPROVED = Golden(
    id="navigator-deploy-approved",
    skill="navigator-deploy",
    description="Full chosen-model flow through plan, approval, and execution.",
    approval_reply=APPROVE_REPLY,
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
            prompt="Go with the balanced configuration. Show me the deployment plan.",
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
    grade=_grade_deploy(APPROVE_REPLY, consent_given=True),
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
    grade=_grade_deploy(None, consent_given=False),
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
