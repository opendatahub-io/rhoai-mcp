"""Phase 0 probe: multi-turn OpenAI-style tool-calling fidelity across candidate models.

Decides which models enter the skill-eval matrix. See
``docs/temp/pydantic-ai-skill-evals-plan.md`` §4 (Phase 0) and §6 open question 1.

The probe deliberately mirrors the shape the navigator skills demand of a model:

1. **Turn 1** - call a tool with a *non-trivial* argument shape (nested object,
   enum, array of enums) built from a natural-language request.
2. **Turn 2** - consume the tool result and issue a *second, different* tool call
   whose arguments are grounded in values returned by the first tool. This is the
   step cheap models typically fail: they re-call the first tool, hallucinate the
   GPU product, or emit both calls in parallel before seeing any result.
3. **Turn 3** - stop calling tools and answer in prose referencing the result.

A model that cannot do this produces ``model_failure`` rows that say nothing about
the skills, so it is better excluded knowingly than discovered mid-sweep.

Run::

    uv run --group skill-eval python -m evals.agent.probe_tool_calling
    uv run --group skill-eval python -m evals.agent.probe_tool_calling --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

OPENAI_BASE_URL = "https://api.openai.com/v1"
ENMAAS_BASE_URL = "https://api.enmaas.devshift.net/v1"


# --------------------------------------------------------------------------- #
# Candidate roster
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidate:
    """One endpoint/model/API triple to probe.

    ``api`` matters: the gpt-5.6 reasoning family rejects function tools on
    ``/v1/chat/completions`` unless ``reasoning_effort='none'``, and directs
    callers to ``/v1/responses`` instead. Both routes are probed so the matrix
    can be built on evidence rather than on a guess about which one works.
    """

    id: str
    model_name: str
    base_url: str | None
    api_key_env: str
    api: str = "chat"  # "chat" | "responses"
    extra: dict[str, Any] = field(default_factory=dict)


CANDIDATES: tuple[Candidate, ...] = (
    # --- Direct OpenAI, cost-efficient tier (the user's preferred default) ---
    # Responses API: the route OpenAI directs reasoning models to for tool use.
    Candidate("openai-sol-resp", "gpt-5.6-sol", OPENAI_BASE_URL, "OPENAI_API_KEY", "responses"),
    # Chat Completions with reasoning disabled: the only way this family tool-calls
    # on /v1/chat/completions. Cheaper and faster, but gives up reasoning.
    Candidate(
        "openai-sol-chat",
        "gpt-5.6-sol",
        OPENAI_BASE_URL,
        "OPENAI_API_KEY",
        "chat",
        {"reasoning_effort": "none"},
    ),
    # --- Direct OpenAI, stronger fallback ("only if required") ---
    Candidate("openai-luna-resp", "gpt-5.6-luna", OPENAI_BASE_URL, "OPENAI_API_KEY", "responses"),
    # --- EnMaaS proxying the same OpenAI model ---
    # Isolates gateway problems from model problems: if this fails while the direct
    # equivalent passes, the gateway is at fault, not the model.
    Candidate("enmaas-sol-resp", "gpt-5.6-sol", ENMAAS_BASE_URL, "ENMAAS_KEY", "responses"),
    Candidate(
        "enmaas-sol-chat",
        "gpt-5.6-sol",
        ENMAAS_BASE_URL,
        "ENMAAS_KEY",
        "chat",
        {"reasoning_effort": "none"},
    ),
    # --- EnMaaS third-party open-weight models from the plan's §2.3 matrix ---
    Candidate("enmaas-glm", "rits/zai-org/glm-5-3", ENMAAS_BASE_URL, "ENMAAS_KEY"),
    Candidate(
        "enmaas-deepseek",
        "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
        ENMAAS_BASE_URL,
        "ENMAAS_KEY",
    ),
)


# --------------------------------------------------------------------------- #
# Probe fixture: tools with a non-trivial argument shape
# --------------------------------------------------------------------------- #

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_use_case_defaults",
            "description": (
                "Look up the default workload profile for an LLM serving use case, "
                "including the GPU inventory available on the cluster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "use_case": {
                        "type": "string",
                        "enum": ["chat", "code_assist", "summarization", "rag"],
                        "description": "The customer's primary use case.",
                    },
                    "concurrency": {
                        "type": "object",
                        "description": "Expected concurrent load.",
                        "properties": {
                            "peak_users": {"type": "integer"},
                            "avg_tokens_per_request": {"type": "integer"},
                        },
                        "required": ["peak_users", "avg_tokens_per_request"],
                    },
                    "priorities": {
                        "type": "array",
                        "description": "Ranked optimisation priorities.",
                        "items": {
                            "type": "string",
                            "enum": ["latency", "throughput", "cost", "accuracy"],
                        },
                    },
                },
                "required": ["use_case", "concurrency", "priorities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_model",
            "description": (
                "Recommend LLM models for a confirmed workload profile on a specific "
                "GPU product. Call only after get_use_case_defaults has returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workload": {
                        "type": "object",
                        "properties": {
                            "use_case": {"type": "string"},
                            "expected_rps": {"type": "number"},
                        },
                        "required": ["use_case", "expected_rps"],
                    },
                    "gpu": {
                        "type": "object",
                        "properties": {
                            "product": {
                                "type": "string",
                                "description": "GPU product key, e.g. A100-80.",
                            },
                            "count": {"type": "integer"},
                        },
                        "required": ["product", "count"],
                    },
                    "top_k": {"type": "integer"},
                },
                "required": ["workload", "gpu"],
            },
        },
    },
]

# The Responses API takes a flattened tool shape (no nested "function" key).
# strict=False keeps the probe about tool-calling fidelity rather than about
# strict-schema conformance, which has its own separate failure modes.
RESPONSES_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "parameters": t["function"]["parameters"],
        "strict": False,
    }
    for t in TOOLS
]

SYSTEM_PROMPT = (
    "You are an assistant that helps select LLM models for OpenShift AI. "
    "Use the provided tools. Call get_use_case_defaults first, then use the GPU "
    "inventory it returns to call recommend_model. Do not guess GPU hardware."
)

USER_PROMPT = (
    "I need an LLM for our customer support chat bot. About 50 support agents will "
    "use it at peak, and a typical request is around 800 tokens. Low latency matters "
    "most, then cost. What should we run?"
)

# Returned to the model as the result of get_use_case_defaults. The GPU product is
# deliberately a value the model cannot guess from the prompt, so a correct
# recommend_model call proves the model actually read the tool result.
DEFAULTS_RESULT = {
    "use_case": "chat",
    "expected_rps": 4.2,
    "gpu_info": {
        "product": "NVIDIA-A100-SXM4-80GB",
        "products": ["NVIDIA-A100-SXM4-80GB"],
        "total": 8,
        "available": 8,
        "nodes_with_gpu": 2,
    },
    "note": "Map the GPU product to the short key 'A100-80' when calling recommend_model.",
}

RECOMMEND_RESULT = {
    "recommendations": [
        {"model": "meta-llama/Llama-3.1-8B-Instruct", "fits": True, "est_latency_ms": 180},
        {"model": "mistralai/Mistral-7B-Instruct-v0.3", "fits": True, "est_latency_ms": 165},
    ]
}

EXPECTED_MODEL_TOKEN = "Llama-3.1-8B"


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    """Outcome of probing one candidate."""

    candidate: Candidate
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    temperature_zero_supported: bool | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    turns: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A candidate qualifies only if every check passed and no API error occurred."""
        return self.error is None and bool(self.checks) and all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate.id,
            "model_name": self.candidate.model_name,
            "base_url": self.candidate.base_url,
            "api": self.candidate.api,
            "extra": self.candidate.extra,
            "passed": self.passed,
            "checks": self.checks,
            "notes": self.notes,
            "temperature_zero_supported": self.temperature_zero_supported,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 2),
            "turns": self.turns,
        }


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


@dataclass
class NormCall:
    """A tool call normalised across the Chat Completions and Responses APIs."""

    id: str
    name: str
    arguments: str

    def parsed(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None


class _ChatDriver:
    """Drives a conversation over ``/v1/chat/completions``."""

    def __init__(self, client: Any, candidate: Candidate, result: ProbeResult) -> None:
        self._client = client
        self._candidate = candidate
        self._result = result
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    async def send(self) -> tuple[list[NormCall], str]:
        kwargs: dict[str, Any] = {
            "model": self._candidate.model_name,
            "messages": self._messages,
            "tools": TOOLS,
            **self._candidate.extra,
        }
        resp = await _with_temperature_zero(
            self._client.chat.completions.create, kwargs, self._result
        )
        msg = resp.choices[0].message
        self._messages.append(msg.model_dump(exclude_none=True))
        calls = [
            NormCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "")
            for c in (msg.tool_calls or [])
        ]
        return calls, msg.content or ""

    def add_tool_result(self, call: NormCall, payload: str) -> None:
        self._messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})


class _ResponsesDriver:
    """Drives a conversation over ``/v1/responses``.

    Reasoning items are echoed back verbatim with ``response.output``, which is
    what lets a reasoning model carry state across tool calls.
    """

    def __init__(self, client: Any, candidate: Candidate, result: ProbeResult) -> None:
        self._client = client
        self._candidate = candidate
        self._result = result
        self._input: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    async def send(self) -> tuple[list[NormCall], str]:
        kwargs: dict[str, Any] = {
            "model": self._candidate.model_name,
            "input": self._input,
            "tools": RESPONSES_TOOLS,
            **self._candidate.extra,
        }
        resp = await _with_temperature_zero(self._client.responses.create, kwargs, self._result)
        output = list(resp.output or [])
        self._input.extend(item.model_dump(exclude_none=True) for item in output)

        calls = [
            NormCall(id=item.call_id, name=item.name, arguments=item.arguments or "")
            for item in output
            if getattr(item, "type", None) == "function_call"
        ]
        return calls, resp.output_text or ""

    def add_tool_result(self, call: NormCall, payload: str) -> None:
        self._input.append({"type": "function_call_output", "call_id": call.id, "output": payload})


async def _with_temperature_zero(create: Any, kwargs: dict[str, Any], result: ProbeResult) -> Any:
    """Call the endpoint, degrading from ``temperature=0`` only if it is rejected.

    Whether the fallback happened is recorded rather than silently absorbed
    (plan §2.3: do not silently fall back to different settings).
    """
    if result.temperature_zero_supported is not False:
        try:
            resp = await create(**kwargs, temperature=0)
            result.temperature_zero_supported = True
            return resp
        except Exception as exc:  # noqa: BLE001 - probing for capability
            if "temperature" not in str(exc).lower():
                raise
            result.temperature_zero_supported = False
            result.notes.append("temperature=0 rejected; using provider default")
    return await create(**kwargs)


async def probe(candidate: Candidate, timeout: float = 300.0) -> ProbeResult:
    """Run the three-turn tool-calling probe against one candidate."""
    from openai import AsyncOpenAI

    result = ProbeResult(candidate=candidate)
    api_key = os.environ.get(candidate.api_key_env)
    if not api_key:
        result.error = f"missing env var {candidate.api_key_env}"
        return result

    client = AsyncOpenAI(
        api_key=api_key, base_url=candidate.base_url, timeout=timeout, max_retries=1
    )
    driver: Any = (
        _ResponsesDriver(client, candidate, result)
        if candidate.api == "responses"
        else _ChatDriver(client, candidate, result)
    )
    start = time.monotonic()

    try:
        # ---- Turn 1: build a non-trivial argument shape from natural language ----
        calls, content = await driver.send()
        result.turns.append(
            {
                "turn": 1,
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in calls],
                "content": content[:400],
            }
        )

        result.checks["t1_called_exactly_one_tool"] = len(calls) == 1
        first = calls[0] if calls else None
        result.checks["t1_correct_tool"] = bool(first and first.name == "get_use_case_defaults")

        args = first.parsed() if first else None
        result.checks["t1_args_valid_json"] = args is not None
        conc = (args or {}).get("concurrency")
        result.checks["t1_nested_object_well_formed"] = (
            isinstance(conc, dict)
            and isinstance(conc.get("peak_users"), int)
            and isinstance(conc.get("avg_tokens_per_request"), int)
        )
        result.checks["t1_enum_respected"] = (args or {}).get("use_case") in {
            "chat",
            "code_assist",
            "summarization",
            "rag",
        }
        prios = (args or {}).get("priorities")
        result.checks["t1_enum_array_well_formed"] = (
            isinstance(prios, list)
            and len(prios) > 0
            and all(p in {"latency", "throughput", "cost", "accuracy"} for p in prios)
        )

        if not first:
            result.notes.append("no tool call on turn 1; aborting probe")
            return result

        # ---- Turn 2: second, different call grounded in the first tool's result ----
        driver.add_tool_result(first, json.dumps(DEFAULTS_RESULT))
        calls, content = await driver.send()
        result.turns.append(
            {
                "turn": 2,
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in calls],
                "content": content[:400],
            }
        )

        second = calls[0] if calls else None
        result.checks["t2_called_second_tool"] = bool(second and second.name == "recommend_model")
        args2 = second.parsed() if second else None
        gpu = (args2 or {}).get("gpu")
        # The GPU product appears nowhere in the user prompt, so a correct value
        # here proves the model consumed the tool result instead of guessing.
        result.checks["t2_grounded_in_tool_result"] = isinstance(gpu, dict) and (
            "A100" in str(gpu.get("product", ""))
        )
        workload = (args2 or {}).get("workload")
        result.checks["t2_carried_numeric_result"] = (
            isinstance(workload, dict)
            and isinstance(workload.get("expected_rps"), int | float)
            and abs(float(workload.get("expected_rps", 0)) - 4.2) < 0.01
        )

        if not second:
            result.notes.append("no second tool call; aborting probe")
            return result

        # ---- Turn 3: stop calling tools and answer in prose ----
        driver.add_tool_result(second, json.dumps(RECOMMEND_RESULT))
        calls, content = await driver.send()
        result.turns.append(
            {
                "turn": 3,
                "tool_calls": [{"name": c.name} for c in calls],
                "content": content[:400],
            }
        )

        result.checks["t3_stopped_calling_tools"] = len(calls) == 0
        result.checks["t3_answered_from_result"] = EXPECTED_MODEL_TOKEN.lower() in content.lower()

    except Exception as exc:  # noqa: BLE001 - the probe's job is to classify failures
        result.error = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        result.elapsed_s = time.monotonic() - start
        await client.close()

    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

CHECK_ORDER = [
    "t1_called_exactly_one_tool",
    "t1_correct_tool",
    "t1_args_valid_json",
    "t1_nested_object_well_formed",
    "t1_enum_respected",
    "t1_enum_array_well_formed",
    "t2_called_second_tool",
    "t2_grounded_in_tool_result",
    "t2_carried_numeric_result",
    "t3_stopped_calling_tools",
    "t3_answered_from_result",
]


def render(results: list[ProbeResult]) -> str:
    """Render a compact pass/fail matrix of candidates against checks."""
    width = max(len(c) for c in CHECK_ORDER) + 2
    lines = [f"{'check':<{width}}" + "".join(f"{r.candidate.id:>18}" for r in results)]
    lines.append("-" * len(lines[0]))
    for check in CHECK_ORDER:
        cells = []
        for r in results:
            if check not in r.checks:
                cells.append(f"{'-':>18}")
            else:
                cells.append(f"{'PASS' if r.checks[check] else 'FAIL':>18}")
        lines.append(f"{check:<{width}}" + "".join(cells))
    lines.append("-" * len(lines[0]))
    lines.append(
        f"{'VERDICT':<{width}}"
        + "".join(f"{('QUALIFIED' if r.passed else 'EXCLUDED'):>18}" for r in results)
    )
    lines.append(f"{'elapsed_s':<{width}}" + "".join(f"{r.elapsed_s:>18.1f}" for r in results))
    lines.append(
        f"{'temp=0':<{width}}"
        + "".join(f"{str(r.temperature_zero_supported):>18}" for r in results)
    )
    lines.append("")
    for r in results:
        if r.error:
            lines.append(f"[{r.candidate.id}] ERROR {r.error}")
        for note in r.notes:
            lines.append(f"[{r.candidate.id}] note: {note}")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", help="write full results to this path")
    parser.add_argument("--only", nargs="*", help="probe only these candidate ids", default=None)
    args = parser.parse_args()

    candidates = CANDIDATES
    if args.only:
        candidates = tuple(c for c in CANDIDATES if c.id in set(args.only))

    results = await asyncio.gather(*(probe(c) for c in candidates))
    print(render(list(results)))

    if args.json_path:
        from pathlib import Path

        Path(args.json_path).write_text(
            json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_path}")


if __name__ == "__main__":
    asyncio.run(main())
