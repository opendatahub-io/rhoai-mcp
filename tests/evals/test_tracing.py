"""Tests for the OTel tracing helpers (plan §3.6, §4).

These use an in-memory exporter: no collector, no network, no MLflow.

The bug they exist to prevent is silent. OpenTelemetry ignores
``set_attribute`` on a span that has already ended — no error, no warning — so
recording a turn's reply one line outside the ``with`` block produced traces
whose Outputs were simply empty in the UI, with everything else working.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk", reason="requires the trace dependency group")

from evals.agent.tracing import (  # noqa: E402
    TracingSession,
    _SessionCapture,
    instrumentation,
    set_turn_output,
    turn_span,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture
def session() -> Any:
    """A tracing session wired to an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    capture = _SessionCapture()
    provider.add_span_processor(capture)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TracingSession(
        provider=provider, enabled=True, endpoint="memory", capture=capture
    ), exporter


class TestTurnSpan:
    def test_turn_span_records_the_user_prompt_as_inputs(self, session: Any) -> None:
        sess, exporter = session
        with turn_span(
            sess,
            run_id="r1",
            attempt_id="a1",
            golden_id="navigator-happy-path",
            model="enmaas-glm",
            turn_index=0,
            turn_name="opening",
            prompt="I need an LLM.",
        ):
            pass

        (span,) = exporter.get_finished_spans()
        assert "I need an LLM." in span.attributes["mlflow.spanInputs"]

    def test_turn_span_records_the_reply_as_outputs(self, session: Any) -> None:
        """The regression: Outputs were empty because the span had already ended."""
        sess, exporter = session
        with turn_span(
            sess,
            run_id="r1",
            attempt_id="a1",
            golden_id="g",
            model="m",
            turn_index=0,
            turn_name="opening",
            prompt="hi",
        ) as span:
            set_turn_output(span, "Phase 1 of 3 — what matters most?")

        (finished,) = exporter.get_finished_spans()
        assert "mlflow.spanOutputs" in finished.attributes
        assert "Phase 1 of 3" in finished.attributes["mlflow.spanOutputs"]

    def test_setting_output_after_the_span_ends_is_silently_lost(self, session: Any) -> None:
        """Documents *why* the call must stay inside the context manager."""
        sess, exporter = session
        with turn_span(
            sess,
            run_id="r1",
            attempt_id="a1",
            golden_id="g",
            model="m",
            turn_index=0,
            turn_name="opening",
            prompt="hi",
        ) as span:
            pass
        set_turn_output(span, "too late")

        (finished,) = exporter.get_finished_spans()
        assert "mlflow.spanOutputs" not in finished.attributes

    def test_correlation_attributes_are_stamped(self, session: Any) -> None:
        sess, exporter = session
        with turn_span(
            sess,
            run_id="run-1",
            attempt_id="att-1",
            golden_id="navigator-happy-path",
            model="enmaas-glm",
            turn_index=3,
            turn_name="supply-gpu-inventory",
            prompt="A100s",
        ):
            pass

        (span,) = exporter.get_finished_spans()
        assert span.attributes["rhoai.run_id"] == "run-1"
        assert span.attributes["rhoai.attempt_id"] == "att-1"
        assert span.attributes["rhoai.golden_id"] == "navigator-happy-path"
        assert span.attributes["rhoai.turn_index"] == 3
        assert span.name == "eval.turn[3] supply-gpu-inventory"

    def test_each_turn_is_its_own_root_span(self, session: Any) -> None:
        """MLflow's model is one trace per turn, so turns must not nest."""
        sess, exporter = session
        for i, name in enumerate(["opening", "priorities"]):
            with turn_span(
                sess,
                run_id="r",
                attempt_id="a",
                golden_id="g",
                model="m",
                turn_index=i,
                turn_name=name,
                prompt=f"p{i}",
            ):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        assert all(s.parent is None for s in spans)
        assert len({s.context.trace_id for s in spans}) == 2


class TestDisabledSession:
    """Tracing is optional: a disabled session must be a complete no-op."""

    def test_turn_span_yields_none_and_emits_nothing(self) -> None:
        disabled = TracingSession()
        with turn_span(
            disabled,
            run_id="r",
            attempt_id="a",
            golden_id="g",
            model="m",
            turn_index=0,
            turn_name="t",
            prompt="p",
        ) as span:
            assert span is None
            set_turn_output(span, "ignored")

    def test_instrumentation_is_empty(self) -> None:
        assert instrumentation(TracingSession()) == []

    def test_session_id_is_empty(self) -> None:
        assert TracingSession().mlflow_session_id == ""


class TestSessionCapture:
    """MLflow adopts gen_ai.conversation.id as the session; we observe it."""

    def test_captures_the_conversation_id(self, session: Any) -> None:
        sess, _ = session
        tracer = sess.provider.get_tracer("t")
        with tracer.start_as_current_span("parent"), tracer.start_as_current_span("child") as child:
            child.set_attribute("gen_ai.conversation.id", "conv-abc")

        assert sess.mlflow_session_id == "conv-abc"

    def test_reset_clears_between_attempts(self, session: Any) -> None:
        sess, _ = session
        tracer = sess.provider.get_tracer("t")
        with tracer.start_as_current_span("s") as s:
            s.set_attribute("gen_ai.conversation.id", "conv-1")
        assert sess.mlflow_session_id == "conv-1"

        sess.reset()

        assert sess.mlflow_session_id == ""

    def test_keeps_the_first_id_seen_within_one_attempt(self, session: Any) -> None:
        sess, _ = session
        tracer = sess.provider.get_tracer("t")
        for value in ("conv-1", "conv-2"):
            with tracer.start_as_current_span("s") as s:
                s.set_attribute("gen_ai.conversation.id", value)

        assert sess.mlflow_session_id == "conv-1"
