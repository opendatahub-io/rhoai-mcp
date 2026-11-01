"""Optional OTel tracing for agent runs, exported to a local MLflow server.

Opt-in and non-blocking by design (plan §3.6, §4): **an unavailable collector
must never prevent transcript capture or a normal evaluation run.** Everything
here degrades to a no-op if tracing is off, the SDK is missing, or the endpoint
is unreachable. The durable record remains the attempt artifact; a trace is a
convenience for looking at timing and shape, not a source of truth.

Uses Pydantic AI's **native** instrumentation with our own ``TracerProvider``
and exporter, deliberately not ``DeepEvalInstrumentationSettings``: that one
claims a global tracer provider and a Confident AI exporter, which would couple
scoring to the Pydantic AI tracing integration we chose to avoid (plan §1.3).

MLflow 3.x serves a real OTLP/HTTP ingestion endpoint at ``/v1/traces``
(``mlflow/server/otel_api.py``), so this is exporter configuration rather than a
bespoke integration. Spans are routed to an experiment with the
``x-mlflow-experiment-id`` header.

Enable with::

    mlflow server --host 127.0.0.1 --port 5000 \\
        --backend-store-uri sqlite:///mlflow.db

    export RHOAI_EVAL_OTLP_ENDPOINT=http://127.0.0.1:5000/v1/traces
    export RHOAI_EVAL_MLFLOW_EXPERIMENT_ID=1
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: OTLP/HTTP traces endpoint. Unset disables tracing entirely.
OTLP_ENDPOINT_ENV = "RHOAI_EVAL_OTLP_ENDPOINT"
#: MLflow experiment to route spans into.
MLFLOW_EXPERIMENT_ENV = "RHOAI_EVAL_MLFLOW_EXPERIMENT_ID"

#: Instrumentation format. Pinned rather than defaulted so a library upgrade
#: cannot silently change span shape mid-sweep (plan §3.6). Version 6 is opt-in
#: and reassigns tool-result message roles, which would change span shape.
INSTRUMENTATION_VERSION: Literal[2, 3, 4, 5, 6] = 5

SERVICE_NAME = "rhoai-skill-evals"


def _span_processor_base() -> Any:
    """``SpanProcessor`` when the SDK is present, else ``object``.

    Subclassing the real base matters: the SDK calls hook methods such as
    ``_on_ending`` that a duck-typed class does not provide, and the resulting
    ``AttributeError`` is raised from ``span.end()`` inside the agent run.
    """
    try:
        from opentelemetry.sdk.trace import SpanProcessor
    except ImportError:
        return object
    return SpanProcessor


class _SessionCapture(_span_processor_base()):  # type: ignore[misc]
    """Records the conversation id MLflow adopts as the trace session.

    MLflow derives ``mlflow.trace.session`` from the first span it sees carrying
    ``session.id`` or ``gen_ai.conversation.id``
    (``store/tracking/sqlalchemy_store.py``). Spans are exported on *end*, and a
    turn's root span ends last, so a child's ``gen_ai.conversation.id`` always
    arrives first and wins — our ``session.id`` on the root never does. Rather
    than fight the ordering, observe the value MLflow will actually use so the
    eval record can point at the real session.
    """

    def __init__(self) -> None:
        self.conversation_id: str = ""

    def reset(self) -> None:
        """Clear between attempts; each attempt is its own conversation."""
        self.conversation_id = ""

    def on_start(self, span: Any, parent_context: Any = None) -> None:  # noqa: ARG002
        return None

    def on_end(self, span: Any) -> None:
        if self.conversation_id:
            return
        value = (span.attributes or {}).get("gen_ai.conversation.id")
        if value:
            self.conversation_id = str(value)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True


@dataclass
class TracingSession:
    """A configured tracer provider, or a disabled no-op."""

    provider: Any = None
    enabled: bool = False
    endpoint: str = ""
    capture: Any = None

    def reset(self) -> None:
        if self.capture is not None:
            self.capture.reset()

    @property
    def mlflow_session_id(self) -> str:
        """The session MLflow will group this attempt's traces under."""
        return self.capture.conversation_id if self.capture is not None else ""

    def shutdown(self) -> None:
        """Flush and stop the exporter.

        Required: ``BatchSpanProcessor`` buffers, so without this a short run
        exits before its spans are sent and the trace never appears.
        """
        if self.provider is None:
            return
        try:
            self.provider.force_flush()
            self.provider.shutdown()
        except Exception as exc:  # noqa: BLE001 - never fail a run on teardown
            logger.warning(f"tracing shutdown failed: {exc}")


def tracing_enabled() -> bool:
    return bool(os.environ.get(OTLP_ENDPOINT_ENV, "").strip())


def setup_tracing(endpoint: str | None = None, experiment_id: str | None = None) -> TracingSession:
    """Build a tracer provider exporting to ``endpoint``.

    Returns a disabled session — never raises — when tracing is off or the SDK
    is unavailable, so callers can use it unconditionally.
    """
    url = endpoint or os.environ.get(OTLP_ENDPOINT_ENV, "").strip()
    if not url:
        return TracingSession()

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning(
            f"tracing requested but the OTel SDK is missing ({exc}); "
            "install the 'trace' dependency group. Continuing untraced."
        )
        return TracingSession()

    experiment = experiment_id or os.environ.get(MLFLOW_EXPERIMENT_ENV, "0")
    headers = {"x-mlflow-experiment-id": str(experiment)}

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    # Registered before the exporter so the session id is observed even if the
    # export itself fails.
    capture = _SessionCapture()
    provider.add_span_processor(capture)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url, headers=headers)))
    logger.info(f"tracing to {url} (mlflow experiment {experiment})")
    return TracingSession(provider=provider, enabled=True, endpoint=url, capture=capture)


def instrumentation(session: TracingSession) -> list[Any]:
    """Pydantic AI capabilities enabling instrumentation, or ``[]`` if disabled.

    ``include_content=True`` is what puts prompts, completions and tool
    arguments into the spans; without it a trace shows shape and timing only.
    """
    if not session.enabled:
        return []
    try:
        from pydantic_ai import InstrumentationSettings
        from pydantic_ai.capabilities import Instrumentation
    except ImportError:  # pragma: no cover - pydantic_ai is a hard dep here
        return []

    return [
        Instrumentation(
            settings=InstrumentationSettings(
                tracer_provider=session.provider,
                include_content=True,
                version=INSTRUMENTATION_VERSION,
            )
        )
    ]


@contextmanager
def turn_span(
    session: TracingSession,
    *,
    run_id: str,
    attempt_id: str,
    golden_id: str,
    model: str,
    turn_index: int,
    turn_name: str,
    prompt: str,
) -> Iterator[Any]:
    """Wrap **one scripted turn** as the root of its own trace.

    This follows MLflow's conversation model: *"Each trace in the session
    represents one turn: the user's input and the system's response"*, with
    turns linked by a shared session id
    (``mlflow/assistant/skills/analyze-mlflow-chat-session/SKILL.md``).

    An earlier version wrapped the whole conversation in a single parent span.
    That produced one trace whose root carried no inputs or outputs — exactly
    the "wrapper span" case MLflow's own guidance calls out — so the trace list
    showed nothing and each turn's new user message was buried at the bottom of
    a cumulative message list.

    ``session.id`` is set to the attempt id so grouping is explicit and
    meaningful: MLflow scans spans for ``session.id`` or
    ``gen_ai.conversation.id`` and takes the first it finds
    (``store/tracking/sqlalchemy_store.py``). Pydantic AI already emits a stable
    ``gen_ai.conversation.id``, so grouping would work regardless; using the
    attempt id instead makes the session key the same one the eval record and
    the saved artifact use.

    ``mlflow.spanInputs``/``mlflow.spanOutputs`` give the trace list a readable
    request and response, which the agent-run span does not provide on its own
    (Pydantic AI puts messages on model-request spans only).

    Yields:
        The span, so the caller can attach the turn's output once it exists.
        A disabled session yields ``None``.
    """
    if not session.enabled:
        yield None
        return

    tracer = session.provider.get_tracer(SERVICE_NAME)
    name = f"eval.turn[{turn_index}] {turn_name or golden_id}"
    with tracer.start_as_current_span(name) as span:
        # Kept as a fallback grouping key: it loses to Pydantic AI's
        # gen_ai.conversation.id on export ordering today, but would become the
        # session if that attribute ever stopped being emitted.
        span.set_attribute("session.id", attempt_id)
        span.set_attribute("rhoai.run_id", run_id)
        span.set_attribute("rhoai.attempt_id", attempt_id)
        span.set_attribute("rhoai.golden_id", golden_id)
        span.set_attribute("rhoai.agent_model", model)
        span.set_attribute("rhoai.turn_index", turn_index)
        span.set_attribute("mlflow.spanInputs", _json({"user": prompt}))
        yield span


def set_turn_output(span: Any, output: str) -> None:
    """Attach a turn's assistant reply, for the trace-list response preview."""
    if span is None:
        return
    span.set_attribute("mlflow.spanOutputs", _json({"assistant": output}))


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
