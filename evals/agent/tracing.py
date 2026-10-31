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
from typing import Any

logger = logging.getLogger(__name__)

#: OTLP/HTTP traces endpoint. Unset disables tracing entirely.
OTLP_ENDPOINT_ENV = "RHOAI_EVAL_OTLP_ENDPOINT"
#: MLflow experiment to route spans into.
MLFLOW_EXPERIMENT_ENV = "RHOAI_EVAL_MLFLOW_EXPERIMENT_ID"

#: Instrumentation format. Pinned rather than defaulted so a library upgrade
#: cannot silently change span shape mid-sweep (plan §3.6).
INSTRUMENTATION_VERSION = 5

SERVICE_NAME = "rhoai-skill-evals"


@dataclass
class TracingSession:
    """A configured tracer provider, or a disabled no-op."""

    provider: Any = None
    enabled: bool = False
    endpoint: str = ""

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
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url, headers=headers)))
    logger.info(f"tracing to {url} (mlflow experiment {experiment})")
    return TracingSession(provider=provider, enabled=True, endpoint=url)


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
def conversation_span(
    session: TracingSession,
    *,
    run_id: str,
    attempt_id: str,
    golden_id: str,
    model: str,
) -> Iterator[str]:
    """Wrap a whole multi-turn conversation in one parent span.

    Each ``agent.run()`` would otherwise start its own trace, leaving a
    five-turn conversation scattered across five unrelated traces. The parent
    span also carries the correlation ids, so a trace can be matched back to its
    saved record and artifact.

    Yields:
        The hex trace id, or ``""`` when tracing is disabled.
    """
    if not session.enabled:
        yield ""
        return

    tracer = session.provider.get_tracer(SERVICE_NAME)
    with tracer.start_as_current_span(f"eval.attempt {golden_id}") as span:
        span.set_attribute("rhoai.run_id", run_id)
        span.set_attribute("rhoai.attempt_id", attempt_id)
        span.set_attribute("rhoai.golden_id", golden_id)
        span.set_attribute("rhoai.agent_model", model)
        trace_id = f"{span.get_span_context().trace_id:032x}"
        yield trace_id
