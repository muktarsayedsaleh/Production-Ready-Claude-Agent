import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer

from core.llm import LLMClient, LLMResponse, Message, ToolCall, ToolResult, ToolSpec
from core.runtime.errors import CostCapExceeded, StepLimitExceeded
from core.runtime.results import ResultStore, TaskResult
from core.runtime.retry import RetryPolicy, Sleep, with_retries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


class AgentRuntime:
    """Runs a task as an LLM/tool loop until a final answer, the step limit, or the cost cap.

    Each task is traced as an `agent.task` span with an `llm.call` child per model call
    (one per attempt, so retried failures show up) and a `tool.call` child per tool call.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[Tool] = (),
        *,
        results: ResultStore | None = None,
        max_steps: int = 20,
        cost_cap_usd: float = 1.0,
        max_tokens: int = 16000,
        retry: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        tracer: Tracer | None = None,
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._results = results
        self._max_steps = max_steps
        self._cost_cap_usd = cost_cap_usd
        self._max_tokens = max_tokens
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._tracer = tracer or trace.get_tracer(__name__)

    async def run(self, task: str, idempotency_key: str | None = None) -> TaskResult:
        with self._tracer.start_as_current_span("agent.task") as span:
            if idempotency_key:
                span.set_attribute("agent.idempotency_key", idempotency_key)
            result = await self._run(task, idempotency_key, span)
            span.set_attributes(
                {
                    "agent.steps": result.steps,
                    "agent.cost_usd": result.cost_usd,
                    "gen_ai.usage.input_tokens": result.input_tokens,
                    "gen_ai.usage.output_tokens": result.output_tokens,
                }
            )
            return result

    async def _run(self, task: str, idempotency_key: str | None, span: trace.Span) -> TaskResult:
        if idempotency_key and self._results:
            cached = await self._results.get(idempotency_key)
            span.set_attribute("agent.cached", cached is not None)
            if cached is not None:
                logger.info("Returning stored result for idempotency key %s", idempotency_key)
                return cached

        messages = [Message(role="user", text=task)]
        specs = [tool.spec for tool in self._tools.values()] or None
        cost_usd = 0.0
        input_tokens = output_tokens = 0

        for step in range(1, self._max_steps + 1):
            response = await with_retries(
                lambda: self._complete(messages, specs),
                self._retry,
                self._sleep,
            )
            cost_usd += response.cost_usd
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            if cost_usd > self._cost_cap_usd:
                raise CostCapExceeded(cost_usd, self._cost_cap_usd, step)

            messages.append(
                Message(
                    role="assistant",
                    text=response.text,
                    tool_calls=response.tool_calls,
                    provider_state=response.provider_state,
                )
            )
            if not response.tool_calls:
                result = TaskResult(
                    answer=response.text,
                    steps=step,
                    cost_usd=cost_usd,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                if idempotency_key and self._results:
                    await self._results.save(idempotency_key, result)
                return result

            tool_results = await asyncio.gather(
                *(self._execute(call) for call in response.tool_calls)
            )
            messages.append(Message(role="user", tool_results=list(tool_results)))

        raise StepLimitExceeded(self._max_steps, cost_usd)

    async def _complete(self, messages: list[Message], specs: list[ToolSpec] | None) -> LLMResponse:
        with self._tracer.start_as_current_span("llm.call") as span:
            response = await self._llm.complete(messages, specs, self._max_tokens)
            span.set_attributes(
                {
                    "gen_ai.response.model": response.model,
                    "gen_ai.usage.input_tokens": response.input_tokens,
                    "gen_ai.usage.output_tokens": response.output_tokens,
                    "agent.cost_usd": response.cost_usd,
                }
            )
            return response

    async def _execute(self, call: ToolCall) -> ToolResult:
        with self._tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("gen_ai.tool.name", call.name)
            started = time.perf_counter()
            result = await self._run_tool(call)
            span.set_attributes(
                {
                    "agent.tool.duration_ms": (time.perf_counter() - started) * 1000,
                    "agent.tool.success": not result.is_error,
                }
            )
            if result.is_error:
                span.set_status(Status(StatusCode.ERROR, result.content))
            return result

    async def _run_tool(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.id, f"Unknown tool: {call.name}", is_error=True)
        try:
            return ToolResult(call.id, await tool.handler(call.input))
        except Exception as exc:
            # Report tool failures to the model so it can recover, instead of failing the task.
            logger.exception("Tool %s failed", call.name)
            return ToolResult(call.id, f"{type(exc).__name__}: {exc}", is_error=True)
