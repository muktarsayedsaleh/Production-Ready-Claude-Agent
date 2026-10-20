import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode, Tracer

from core.llm import LLMResponse, ToolCall, TransientLLMError
from core.runtime import AgentRuntime, CostCapExceeded, RetryPolicy, Tool
from core.tracing import configure_tracing
from tests.fakes import FakeLLMClient

MODEL = "claude-opus-5-5"


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def response(*calls: ToolCall, text: str = "", cost: float = 0.01) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=list(calls),
        input_tokens=100,
        output_tokens=10,
        cost_usd=cost,
        model=MODEL,
    )


def echo_tool() -> Tool:
    async def handler(args: dict[str, Any]) -> str:
        return f"echo:{args['value']}"

    return Tool("echo", "Echo a value", {"type": "object"}, handler)


def children(spans: Sequence[ReadableSpan], parent: ReadableSpan) -> list[ReadableSpan]:
    return [s for s in spans if s.parent is not None and s.parent.span_id == parent.context.span_id]


def only_root(spans: Sequence[ReadableSpan]) -> ReadableSpan:
    roots = [s for s in spans if s.parent is None]
    assert len(roots) == 1
    return roots[0]


def test_task_span_tree(tracer: Tracer, exporter: InMemorySpanExporter) -> None:
    llm = FakeLLMClient(
        [
            response(ToolCall("call_1", "echo", {"value": "a"}), ToolCall("call_2", "missing", {})),
            response(text="done"),
        ]
    )

    asyncio.run(AgentRuntime(llm, [echo_tool()], tracer=tracer).run("task", "job-1"))

    spans = exporter.get_finished_spans()
    root = only_root(spans)
    assert root.name == "agent.task"
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}
    assert dict(root.attributes or {}) == {
        "agent.idempotency_key": "job-1",
        "agent.steps": 2,
        "agent.cost_usd": pytest.approx(0.02),
        "gen_ai.usage.input_tokens": 200,
        "gen_ai.usage.output_tokens": 20,
    }

    # Tool spans hang off the task, not the LLM call that requested them.
    kids = children(spans, root)
    assert sorted(s.name for s in kids) == ["llm.call", "llm.call", "tool.call", "tool.call"]
    assert all(children(spans, kid) == [] for kid in kids)

    for llm_span in (s for s in kids if s.name == "llm.call"):
        assert dict(llm_span.attributes or {}) == {
            "gen_ai.response.model": MODEL,
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 10,
            "agent.cost_usd": 0.01,
        }

    tools = {(s.attributes or {})["gen_ai.tool.name"]: s for s in kids if s.name == "tool.call"}
    assert (tools["echo"].attributes or {})["agent.tool.success"] is True
    assert tools["echo"].status.status_code == StatusCode.UNSET
    assert (tools["missing"].attributes or {})["agent.tool.success"] is False
    assert tools["missing"].status.status_code == StatusCode.ERROR
    for tool_span in tools.values():
        duration = (tool_span.attributes or {})["agent.tool.duration_ms"]
        assert isinstance(duration, float) and duration >= 0


def test_failed_llm_attempts_are_error_spans(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    llm = FakeLLMClient([TransientLLMError("overloaded"), response(text="ok")])

    async def no_sleep(delay: float) -> None:
        pass

    runtime = AgentRuntime(llm, retry=RetryPolicy(max_attempts=2), sleep=no_sleep, tracer=tracer)
    asyncio.run(runtime.run("task"))

    spans = exporter.get_finished_spans()
    attempts = sorted(children(spans, only_root(spans)), key=lambda s: s.start_time or 0)
    assert [s.status.status_code for s in attempts] == [StatusCode.ERROR, StatusCode.UNSET]
    assert attempts[0].events[0].name == "exception"


def test_aborted_task_marks_root_span_as_error(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    llm = FakeLLMClient([response(ToolCall("call_1", "echo", {"value": 1}), cost=5.0)])

    with pytest.raises(CostCapExceeded):
        asyncio.run(AgentRuntime(llm, [echo_tool()], tracer=tracer).run("task"))

    root = only_root(exporter.get_finished_spans())
    assert root.status.status_code == StatusCode.ERROR
    assert "exceeded its cost cap" in (root.status.description or "")


def test_configure_tracing_is_noop_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert configure_tracing() is None


def test_configure_tracing_uses_env_service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "agent-under-test")

    provider = configure_tracing()

    assert provider is not None
    assert provider.resource.attributes["service.name"] == "agent-under-test"
    provider.shutdown()
