"""Optional OTLP tracing for live skill evaluations."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)
_attempt_id: ContextVar[str] = ContextVar("eval_attempt_id", default="")


def setup_tracing() -> TracingSession:
    """Use a private provider; failure to configure tracing must not stop evals."""
    endpoint = os.environ.get("RHOAI_EVAL_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return TracingSession()
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        class SessionProcessor(SpanProcessor):
            def on_start(self, span: Any, parent_context: Any = None) -> None:  # noqa: ARG002
                if attempt_id := _attempt_id.get():
                    span.set_attribute("session.id", attempt_id)

        provider = TracerProvider(resource=Resource.create({"service.name": "rhoai-skill-evals"}))
        provider.add_span_processor(SessionProcessor())
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=5))
        )
        return TracingSession(provider)
    except Exception as exc:  # noqa: BLE001 - tracing is optional
        logger.warning("Could not configure tracing (install the trace group): %s", exc)
        return TracingSession()


@dataclass
class TracingSession:
    provider: Any = None

    def instrumentation(self) -> list[Any]:
        if self.provider is None:
            return []
        from pydantic_ai import InstrumentationSettings
        from pydantic_ai.capabilities import Instrumentation

        return [
            Instrumentation(
                settings=InstrumentationSettings(
                    tracer_provider=self.provider, include_content=True, version=5
                )
            )
        ]

    def shutdown(self) -> None:
        if self.provider is not None:
            try:
                if not self.provider.force_flush(timeout_millis=6000):
                    logger.warning("Tracing flush timed out")
                self.provider.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tracing shutdown failed: %s", exc)

    @contextmanager
    def span(self, *, attempt_id: str, name: str, attributes: dict[str, Any]) -> Iterator[Any]:
        if self.provider is None:
            yield None
            return
        from opentelemetry.context import Context

        token = _attempt_id.set(attempt_id)
        try:
            with self.provider.get_tracer("rhoai-skill-evals").start_as_current_span(
                name, context=Context()
            ) as span:
                # Set explicitly as well, so injected providers used by callers work.
                span.set_attribute("session.id", attempt_id)
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                yield span
        finally:
            _attempt_id.reset(token)


def content(value: str, role: str) -> str:
    return json.dumps({role: value}, ensure_ascii=False)


def mark_error(span: Any, error: str) -> None:
    if span is not None:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, error))
