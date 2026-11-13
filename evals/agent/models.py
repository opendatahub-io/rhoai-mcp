"""Agent model specifications for the skill-evaluation matrix.

One frozen tuple of ``(api, base_url, api_key_env, model_name)`` per matrix cell.
The matrix uses EnMaaS with OpenAI Responses or native Anthropic Messages,
selected explicitly by spec. Keeping the matrix parameterised this way is also
what lets a later EvalHub BYOF adapter map ``JobSpec.model.url`` /
``JobSpec.model.name`` straight onto a spec.


"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings

OPENAI_BASE_URL = "https://api.openai.com/v1"
ENMAAS_BASE_URL = "https://api.enmaas.devshift.net/v1"
#: The Anthropic SDK appends /v1/messages itself.
ENMAAS_ANTHROPIC_BASE_URL = "https://api.enmaas.devshift.net"

#: Environment variable selecting which specs to run, comma-separated.
MODELS_ENV_VAR = "RHOAI_EVAL_AGENT_MODELS"

AgentApi = Literal["chat", "responses", "messages"]


class ModelSpecError(RuntimeError):
    """Raised when a requested spec is unknown or its credentials are absent."""


@dataclass(frozen=True)
class AgentModelSpec:
    """One cell of the model matrix.

    Attributes:
        id: Stable identifier recorded with every eval result.
        model_name: Provider-side model identifier.
        base_url: Provider endpoint, or ``None`` for the OpenAI default.
        api_key_env: Environment variable holding the credential.
        api: Explicit wire API. GPT, GLM, and DeepSeek use ``responses``;
            Claude uses native Anthropic ``messages`` through EnMaaS. ``chat``
            remains available for diagnostic specs.
        temperature: Requested sampling temperature, or ``None`` to leave it to
            the provider.

            **Every matrix spec sets ``None``.** The gpt-5.6
            family rejects the parameter outright (*"Unsupported parameter:
            'temperature' is not supported with this model"*). Setting zero only
            where supported would make a cross-model difference partly an artifact of
            sampling configuration. Requesting nothing anywhere keeps the cells
            treated alike.

            Note this buys uniform *requests*, not uniform sampling: provider
            defaults need not match. True determinism is unavailable in this
            matrix either way, which is one reason a sweep is a smoke comparison
            rather than a ranking.
        extra_settings: Additional ``ModelSettings`` entries, e.g. a reasoning
            effort override.
    """

    id: str
    model_name: str
    base_url: str | None
    api_key_env: str
    api: AgentApi = "responses"
    temperature: float | None = 0.0
    extra_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        """The credential for this spec.

        Raises:
            ModelSpecError: If the environment variable is unset or empty.
        """
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ModelSpecError(f"spec {self.id!r} needs environment variable {self.api_key_env}")
        return key

    def settings(self) -> ModelSettings:
        """Build the effective model settings, recorded verbatim with results."""
        values: dict[str, Any] = dict(self.extra_settings)
        if self.temperature is not None:
            values["temperature"] = self.temperature
        # ModelSettings is a total=False TypedDict; the provider-specific keys in
        # extra_settings (e.g. openai_reasoning_effort) live on its OpenAI subclasses.
        return cast("ModelSettings", values)

    def describe(self) -> dict[str, Any]:
        """Credential-free description of this spec for the result record.

        ``sampling`` spells out what a bare ``temperature: null`` means, so a
        reader of the results does not have to guess whether determinism was
        attempted and failed or was never requested.
        """
        return {
            "id": self.id,
            "model_name": self.model_name,
            "base_url": self.base_url or OPENAI_BASE_URL,
            "api": self.api,
            "temperature": self.temperature,
            "sampling": (
                "provider-default (temperature not requested; cannot be set "
                "uniformly across the matrix)"
                if self.temperature is None
                else f"temperature={self.temperature}"
            ),
            "extra_settings": dict(self.extra_settings),
            "agent_backend": "pydantic-ai",
        }

    def build(self) -> Model:
        """Construct the Pydantic AI model for this spec."""
        if self.api == "messages":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(
                self.model_name,
                provider=AnthropicProvider(api_key=self.api_key, base_url=self.base_url),
                settings=self.settings(),
            )

        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key=self.api_key, base_url=self.base_url)
        settings = self.settings()
        if self.api == "responses":
            return OpenAIResponsesModel(self.model_name, provider=provider, settings=settings)
        return OpenAIChatModel(self.model_name, provider=provider, settings=settings)


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #

#: Model specs that passed the tool-calling probe.
SPECS: dict[str, AgentModelSpec] = {
    spec.id: spec
    for spec in (
        # --- Direct OpenAI, cost-efficient tier ---
        AgentModelSpec(
            id="openai-sol",
            model_name="gpt-5.6-sol",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            api="responses",
            # Rejected outright by this family; see AgentModelSpec.temperature
            # for why no matrix cell requests one.
            temperature=None,
        ),
        # --- Direct OpenAI, stronger fallback: promote only if -sol underperforms ---
        AgentModelSpec(
            id="openai-luna",
            model_name="gpt-5.6-luna",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            api="responses",
            temperature=None,
        ),
        # --- EnMaaS GPT representative ---
        AgentModelSpec(
            id="enmaas-sol",
            model_name="gpt-5.6-sol",
            base_url=ENMAAS_BASE_URL,
            api_key_env="ENMAAS_KEY",
            api="responses",
            temperature=None,
        ),
        # --- EnMaaS Anthropic representative, using native Messages ---
        AgentModelSpec(
            id="enmaas-opus",
            model_name="claude-opus-4-6",
            base_url=ENMAAS_ANTHROPIC_BASE_URL,
            api_key_env="ENMAAS_KEY",
            api="messages",
            temperature=None,
            # Opus 4.6 leaves thinking off when omitted; match its agentic use.
            extra_settings={"anthropic_thinking": {"type": "adaptive"}},
        ),
        # --- EnMaaS open-weight representatives, using Responses ---
        # Leave temperature unset for parity with the GPT/Anthropic entries.
        AgentModelSpec(
            id="enmaas-glm",
            model_name="rits/zai-org/glm-5-3",
            base_url=ENMAAS_BASE_URL,
            api_key_env="ENMAAS_KEY",
            api="responses",
            temperature=None,
        ),
        AgentModelSpec(
            id="enmaas-deepseek",
            model_name="rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
            base_url=ENMAAS_BASE_URL,
            api_key_env="ENMAAS_KEY",
            api="responses",
            temperature=None,
        ),
    )
}

#: EnMaaS profiles used when all agents are selected.
MATRIX_SPEC_IDS: tuple[str, ...] = (
    "enmaas-sol",
    "enmaas-opus",
    "enmaas-glm",
    "enmaas-deepseek",
)
DEFAULT_SPEC_IDS: tuple[str, ...] = ("enmaas-deepseek",)


def get_spec(spec_id: str) -> AgentModelSpec:
    """Look up a spec by id.

    Raises:
        ModelSpecError: If no such spec exists.
    """
    try:
        return SPECS[spec_id]
    except KeyError:
        raise ModelSpecError(
            f"unknown agent model {spec_id!r}; known: {', '.join(sorted(SPECS))}"
        ) from None


def selected_specs(raw: str | None = None) -> list[AgentModelSpec]:
    """Resolve the specs to run from ``RHOAI_EVAL_AGENT_MODELS``.

    Accepts ``all`` or a comma-separated list of spec ids.

    Args:
        raw: Override for the environment variable, mainly for tests.

    Raises:
        ModelSpecError: If any requested id is unknown.
    """
    value = (raw if raw is not None else os.environ.get(MODELS_ENV_VAR, "")).strip()
    if not value:
        ids: tuple[str, ...] = DEFAULT_SPEC_IDS
    elif value == "all":
        ids = MATRIX_SPEC_IDS
    else:
        ids = tuple(part.strip() for part in value.split(",") if part.strip())
    return [get_spec(spec_id) for spec_id in ids]
