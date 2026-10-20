import asyncio
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis

from core.llm import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, TransientLLMError
from core.queue import Job, JobStatus
from core.runtime import (
    AgentRuntime,
    CostCapExceeded,
    RedisResultStore,
    RetryPolicy,
    StepLimitExceeded,
    TaskResult,
    Tool,
)
from core.runtime.handler import make_job_handler


class FakeLLMClient(LLMClient):
    """Replays scripted responses (or raises scripted errors) and records every call."""

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self.script = list(script)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec] | None] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class InMemoryResultStore:
    def __init__(self) -> None:
        self.results: dict[str, TaskResult] = {}

    async def get(self, key: str) -> TaskResult | None:
        return self.results.get(key)

    async def save(self, key: str, result: TaskResult) -> None:
        self.results[key] = result


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def answer(text: str, cost: float = 0.01) -> LLMResponse:
    return LLMResponse(text=text, input_tokens=100, output_tokens=10, cost_usd=cost)


def tool_use(*calls: ToolCall, cost: float = 0.01) -> LLMResponse:
    return LLMResponse(
        text="", tool_calls=list(calls), input_tokens=100, output_tokens=10, cost_usd=cost
    )


def echo_tool() -> Tool:
    async def handler(args: dict[str, Any]) -> str:
        return f"echo:{args['value']}"

    return Tool("echo", "Echo a value", {"type": "object"}, handler)


NO_JITTER = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False)


def test_returns_final_answer_without_tools() -> None:
    llm = FakeLLMClient([answer("42")])

    result = asyncio.run(AgentRuntime(llm).run("What is 6 x 7?"))

    assert result == TaskResult(
        answer="42", steps=1, cost_usd=0.01, input_tokens=100, output_tokens=10
    )
    assert llm.calls[0] == [Message(role="user", text="What is 6 x 7?")]
    assert llm.tools_seen == [None]


def test_executes_tool_calls_and_feeds_results_back() -> None:
    llm = FakeLLMClient(
        [
            tool_use(
                ToolCall("call_1", "echo", {"value": "a"}),
                ToolCall("call_2", "echo", {"value": "b"}),
            ),
            answer("done"),
        ]
    )

    result = asyncio.run(AgentRuntime(llm, [echo_tool()]).run("echo twice"))

    assert result.answer == "done"
    assert result.steps == 2
    assert result.cost_usd == pytest.approx(0.02)
    assert result.input_tokens == 200
    assert llm.tools_seen[0] == [echo_tool().spec]

    tool_results = llm.calls[1][-1].tool_results
    assert [(r.tool_call_id, r.content, r.is_error) for r in tool_results] == [
        ("call_1", "echo:a", False),
        ("call_2", "echo:b", False),
    ]


def test_unknown_tool_and_tool_failure_are_reported_to_the_model() -> None:
    async def broken(args: dict[str, Any]) -> str:
        raise RuntimeError("disk full")

    llm = FakeLLMClient(
        [
            tool_use(ToolCall("call_1", "missing", {}), ToolCall("call_2", "broken", {})),
            answer("recovered"),
        ]
    )
    runtime = AgentRuntime(llm, [Tool("broken", "Always fails", {"type": "object"}, broken)])

    result = asyncio.run(runtime.run("try tools"))

    assert result.answer == "recovered"
    first, second = llm.calls[1][-1].tool_results
    assert (first.content, first.is_error) == ("Unknown tool: missing", True)
    assert (second.content, second.is_error) == ("RuntimeError: disk full", True)


def test_step_limit_aborts_the_task() -> None:
    llm = FakeLLMClient([tool_use(ToolCall(f"call_{i}", "echo", {"value": i})) for i in range(3)])
    runtime = AgentRuntime(llm, [echo_tool()], max_steps=3)

    with pytest.raises(StepLimitExceeded, match="did not finish within 3 steps"):
        asyncio.run(runtime.run("loop forever"))
    assert len(llm.calls) == 3


def test_cost_cap_aborts_with_clear_error() -> None:
    llm = FakeLLMClient(
        [tool_use(ToolCall(f"call_{i}", "echo", {"value": i}), cost=0.4) for i in range(5)]
    )
    runtime = AgentRuntime(llm, [echo_tool()], cost_cap_usd=1.0)

    with pytest.raises(CostCapExceeded) as excinfo:
        asyncio.run(runtime.run("expensive task"))

    assert str(excinfo.value) == (
        "Task exceeded its cost cap: spent $1.2000 against a cap of $1.0000 after 3 LLM call(s)"
    )
    assert excinfo.value.cost_usd == pytest.approx(1.2)
    assert len(llm.calls) == 3


def test_transient_errors_are_retried_with_exponential_backoff() -> None:
    llm = FakeLLMClient(
        [TransientLLMError("rate limited"), TransientLLMError("overloaded"), answer("ok")]
    )
    sleep = RecordingSleep()

    result = asyncio.run(AgentRuntime(llm, retry=NO_JITTER, sleep=sleep).run("hi"))

    assert result.answer == "ok"
    assert sleep.delays == [1.0, 2.0]


def test_retries_give_up_after_max_attempts() -> None:
    llm = FakeLLMClient([TransientLLMError(f"fail {i}") for i in range(3)])
    sleep = RecordingSleep()

    with pytest.raises(TransientLLMError, match="fail 2"):
        asyncio.run(AgentRuntime(llm, retry=NO_JITTER, sleep=sleep).run("hi"))
    assert len(llm.calls) == 3
    assert sleep.delays == [1.0, 2.0]


def test_non_transient_errors_are_not_retried() -> None:
    llm = FakeLLMClient([ValueError("bad request")])
    sleep = RecordingSleep()

    with pytest.raises(ValueError):
        asyncio.run(AgentRuntime(llm, retry=NO_JITTER, sleep=sleep).run("hi"))
    assert sleep.delays == []


def test_backoff_delay_is_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=False)

    assert [policy.delay(attempt) for attempt in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_same_idempotency_key_returns_stored_result_without_calling_llm() -> None:
    llm = FakeLLMClient([answer("first"), answer("second")])
    runtime = AgentRuntime(llm, results=InMemoryResultStore())

    async def scenario() -> tuple[TaskResult, TaskResult, TaskResult]:
        return (
            await runtime.run("task", idempotency_key="job-1"),
            await runtime.run("task", idempotency_key="job-1"),
            await runtime.run("task", idempotency_key="job-2"),
        )

    first, repeat, other = asyncio.run(scenario())

    assert repeat == first
    assert other.answer == "second"
    assert len(llm.calls) == 2


def test_failed_task_is_not_stored_so_it_can_be_retried() -> None:
    store = InMemoryResultStore()
    llm = FakeLLMClient([tool_use(ToolCall("call_1", "echo", {"value": 1}), cost=5.0)])

    with pytest.raises(CostCapExceeded):
        asyncio.run(
            AgentRuntime(llm, [echo_tool()], results=store).run("t", idempotency_key="job-1")
        )
    assert store.results == {}


def test_redis_result_store_round_trip() -> None:
    async def scenario() -> TaskResult | None:
        store = RedisResultStore(FakeAsyncRedis())
        await store.save("job-1", TaskResult("42", 1, 0.01, 100, 10))
        assert await store.get("missing") is None
        return await store.get("job-1")

    assert asyncio.run(scenario()) == TaskResult("42", 1, 0.01, 100, 10)


def test_job_handler_uses_job_id_as_default_idempotency_key() -> None:
    llm = FakeLLMClient([answer("a"), answer("b")])
    store = InMemoryResultStore()
    handle = make_job_handler(AgentRuntime(llm, results=store))

    async def scenario() -> None:
        await handle(Job(id="job-1", payload={"task": "t"}, status=JobStatus.RUNNING))
        await handle(Job(id="job-1", payload={"task": "t"}, status=JobStatus.RUNNING))
        await handle(
            Job(
                id="job-2",
                payload={"task": "t", "idempotency_key": "custom"},
                status=JobStatus.RUNNING,
            )
        )

    asyncio.run(scenario())

    assert len(llm.calls) == 2
    assert set(store.results) == {"job-1", "custom"}
