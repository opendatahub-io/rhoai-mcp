"""Exercise real runner instrumentation without a collector or provider calls."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

from evals.agent import runner  # noqa: E402
from evals.agent.tracing import setup_tracing  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from tests.evals.test_runner import SPEC, MemorySession  # noqa: E402


async def test_runner_exports_attempt_sessions_turns_tools_and_errors(monkeypatch: Any) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("RHOAI_EVAL_OTLP_ENDPOINT", "http://localhost:5000/v1/traces")
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        lambda **_kwargs: exporter,
    )
    tracing = setup_tracing()

    def respond(messages: Any, _info: Any) -> ModelResponse:
        part = messages[-1].parts[-1]
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content="done")])
        if part.content == "fail":
            raise RuntimeError("model failed")
        return ModelResponse(parts=[ToolCallPart(tool_name="read_state", args={})])

    def build(*_args: Any, **_kwargs: Any) -> Any:
        session = MemorySession()
        return Agent(FunctionModel(respond), toolsets=[session]), session

    monkeypatch.setattr(runner, "build_agent", build)
    first, second = await asyncio.gather(
        runner.run_conversation(
            SPEC,
            ["one", "two"],
            run_id="sweep",
            attempt_id="a1",
            tracing=tracing,
            skill="navigator",
            golden_id="golden",
        ),
        runner.run_conversation(
            SPEC,
            ["fail"],
            run_id="sweep",
            attempt_id="a2",
            tracing=tracing,
            skill="navigator",
            golden_id="golden",
        ),
    )
    tracing.provider.force_flush()
    spans = exporter.get_finished_spans()
    roots = [span for span in spans if span.name.startswith("eval.turn")]
    assert len(roots) == 3
    assert len({span.context.trace_id for span in roots}) == 3
    assert {span.attributes["session.id"] for span in spans} == {"a1", "a2"}
    assert all(
        span.attributes["session.id"]
        == next(
            root.attributes["session.id"]
            for root in roots
            if root.context.trace_id == span.context.trace_id
        )
        for span in spans
    )
    assert any("read_state" in span.name for span in spans)
    root = next(span for span in roots if span.attributes["session.id"] == "a1")
    assert json.loads(root.attributes["mlflow.spanInputs"])["user"] in {"one", "two"}
    assert json.loads(root.attributes["mlflow.spanOutputs"])["assistant"] == "done"
    assert root.attributes["rhoai.skill"] == "navigator"
    assert root.attributes["rhoai.golden_id"] == "golden"
    assert root.attributes["rhoai.run_id"] == "sweep"
    assert all(span.attributes["mlflow.traceTag.rhoai.golden_id"] == "golden" for span in roots)
    assert all(span.attributes["mlflow.traceTag.rhoai.agent_model"] == SPEC.id for span in roots)
    assert all(
        span.attributes["mlflow.traceTag.rhoai.model_name"] == SPEC.model_name for span in roots
    )
    failed = next(span for span in roots if span.attributes["session.id"] == "a2")
    assert failed.status.status_code is StatusCode.ERROR
    assert first.trace_session_id == "a1" and second.trace_session_id == "a2"

    monkeypatch.setattr(runner, "build_agent", Mock(side_effect=RuntimeError("setup failed")))
    failed_setup = await runner.run_conversation(
        SPEC, ["start"], run_id="sweep", attempt_id="a3", tracing=tracing
    )
    tracing.provider.force_flush()
    setup_span = next(span for span in exporter.get_finished_spans() if span.name == "eval.setup")
    assert setup_span.status.status_code is StatusCode.ERROR
    assert setup_span.attributes["session.id"] == failed_setup.trace_session_id == "a3"
    session = MemorySession()
    monkeypatch.setattr(session, "list_tools", AsyncMock(side_effect=RuntimeError("schema failed")))
    monkeypatch.setattr(
        runner,
        "build_agent",
        Mock(return_value=(Agent(FunctionModel(respond), toolsets=[session]), session)),
    )
    failed_schema = await runner.run_conversation(
        SPEC, ["start"], run_id="sweep", attempt_id="a4", tracing=tracing
    )
    tracing.provider.force_flush()
    schema_span = next(
        span for span in exporter.get_finished_spans() if span.attributes.get("session.id") == "a4"
    )
    assert schema_span.status.status_code is StatusCode.ERROR
    assert failed_schema.error_stage == "tool_schema_fetch"
    tracing.shutdown()
