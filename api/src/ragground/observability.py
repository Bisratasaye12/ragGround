"""OpenTelemetry tracing. Off by default; enable with RAGGROUND_OTEL_ENABLED=true and view traces
in Jaeger (http://localhost:16686). Spans: HTTP request (auto) → rag.retrieve → retrieval.* →
rag.generate, with token counts and model names as attributes."""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ragground.config import Settings


def setup_tracing(app: FastAPI, settings: Settings) -> None:
    if not settings.otel_enabled:
        return
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": settings.otel_service_name, "deployment.environment": settings.env}
        )
    )
    exporter = OTLPSpanExporter(endpoint=f"{settings.otel_endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,health/ready",
        # One span per streamed SSE chunk is pure noise; keep the request span and ours.
        exclude_spans=["send", "receive"],
    )
