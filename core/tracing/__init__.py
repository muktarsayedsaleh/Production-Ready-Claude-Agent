import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "production-ready-claude-agent"


def configure_tracing() -> TracerProvider | None:
    """Export spans over OTLP gRPC (e.g. to Jaeger) when OTEL_EXPORTER_OTLP_ENDPOINT is set.

    Without an endpoint, nothing is configured and OpenTelemetry's default no-op tracer is
    used, so instrumented code costs next to nothing. Call `shutdown()` on the returned
    provider before exiting to flush pending spans.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set, tracing disabled")
        return None

    service_name = os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("Exporting traces for %r to %s", service_name, endpoint)
    return provider


__all__ = ["DEFAULT_SERVICE_NAME", "configure_tracing"]
