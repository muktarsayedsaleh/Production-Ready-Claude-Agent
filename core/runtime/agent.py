import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.llm import LLMClient, Message, ToolCall, ToolResult, ToolSpec
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
    """Runs a task as an LLM/tool loop until a final answer, the step limit, or the cost cap."""

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
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._results = results
        self._max_steps = max_steps
        self._cost_cap_usd = cost_cap_usd
        self._max_tokens = max_tokens
        self._retry = retry or RetryPolicy()
        self._sleep = sleep

    async def run(self, task: str, idempotency_key: str | None = None) -> TaskResult:
        if idempotency_key and self._results:
            cached = await self._results.get(idempotency_key)
            if cached is not None:
                logger.info("Returning stored result for idempotency key %s", idempotency_key)
                return cached

        messages = [Message(role="user", text=task)]
        specs = [tool.spec for tool in self._tools.values()] or None
        cost_usd = 0.0
        input_tokens = output_tokens = 0

        for step in range(1, self._max_steps + 1):
            response = await with_retries(
                lambda: self._llm.complete(messages, specs, self._max_tokens),
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

    async def _execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.id, f"Unknown tool: {call.name}", is_error=True)
        try:
            return ToolResult(call.id, await tool.handler(call.input))
        except Exception as exc:
            # Report tool failures to the model so it can recover, instead of failing the task.
            logger.exception("Tool %s failed", call.name)
            return ToolResult(call.id, f"{type(exc).__name__}: {exc}", is_error=True)
