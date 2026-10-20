import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer

from core.llm import LLMClient, LLMResponse, Message, ToolCall, ToolResult, ToolSpec
from core.permissions import Decision, Policy, PolicyCheck
from core.runtime.checkpoint import CheckpointStore, TaskState
from core.runtime.errors import ApprovalRequired, CostCapExceeded, StepLimitExceeded
from core.runtime.results import ResultStore, TaskResult, ToolUse
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

    Every tool call is checked against the policy first. Denied calls return a structured
    error to the model; calls that need approval pause the task (ApprovalRequired) after
    checkpointing its state, and `resume()` continues it once a reviewer has decided.

    Each task is traced as an `agent.task` span with an `llm.call` child per model call
    (one per attempt, so retried failures show up) and a `tool.call` child per tool call.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[Tool] = (),
        *,
        policy: Policy | None = None,
        results: ResultStore | None = None,
        checkpoints: CheckpointStore | None = None,
        max_steps: int = 20,
        cost_cap_usd: float = 1.0,
        max_tokens: int = 16000,
        retry: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        tracer: Tracer | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._policy = policy or Policy.allow_all()
        if self._policy.requires_approval and checkpoints is None:
            raise ValueError("A policy with approval rules needs a CheckpointStore to pause tasks")
        self._results = results
        self._checkpoints = checkpoints
        self._max_steps = max_steps
        self._cost_cap_usd = cost_cap_usd
        self._max_tokens = max_tokens
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._tracer = tracer or trace.get_tracer(__name__)
        self._system_prompt = system_prompt

    async def run(self, task: str, idempotency_key: str | None = None) -> TaskResult:
        """Run a task. If it is paused under this key, continue it instead (once decided)."""
        return await self._traced(task, idempotency_key)

    async def resume(self, idempotency_key: str, decisions: Mapping[str, bool]) -> TaskResult:
        """Record reviewer decisions (tool call id -> approved) and continue the paused task."""
        await self.approve(idempotency_key, decisions)
        return await self._traced(None, idempotency_key)

    async def approve(self, idempotency_key: str, decisions: Mapping[str, bool]) -> None:
        """Record reviewer decisions without continuing; a later run() picks them up."""
        state = await self._paused_state(idempotency_key)
        awaiting = {call.id for call in self._awaiting(state)}
        unknown = set(decisions) - awaiting
        if unknown:
            raise ValueError(f"Not awaiting approval: {', '.join(sorted(unknown))}")
        missing = awaiting - set(decisions)
        if missing:
            raise ValueError(f"Missing decisions for: {', '.join(sorted(missing))}")
        state.decisions.update(decisions)
        assert self._checkpoints is not None
        await self._checkpoints.save(idempotency_key, state)

    async def pending_approvals(self, idempotency_key: str) -> list[ToolCall]:
        return self._awaiting(await self._paused_state(idempotency_key))

    async def _paused_state(self, idempotency_key: str) -> TaskState:
        state = await self._checkpoints.get(idempotency_key) if self._checkpoints else None
        if state is None:
            raise ValueError(f"No paused task for idempotency key {idempotency_key!r}")
        return state

    async def _traced(self, task: str | None, key: str | None) -> TaskResult:
        with self._tracer.start_as_current_span(
            "agent.task", record_exception=False, set_status_on_exception=False
        ) as span:
            if key:
                span.set_attribute("agent.idempotency_key", key)
            try:
                result = await self._run(task, key, span)
            except ApprovalRequired as exc:
                span.set_attribute("agent.awaiting_approval", [call.name for call in exc.pending])
                raise
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                # A failed task starts over on its next run rather than resuming mid-way.
                if key and self._checkpoints:
                    await self._checkpoints.delete(key)
                raise
            span.set_attributes(
                {
                    "agent.steps": result.steps,
                    "agent.cost_usd": result.cost_usd,
                    "gen_ai.usage.input_tokens": result.input_tokens,
                    "gen_ai.usage.output_tokens": result.output_tokens,
                }
            )
            return result

    async def _run(self, task: str | None, key: str | None, span: trace.Span) -> TaskResult:
        if key and self._results:
            cached = await self._results.get(key)
            span.set_attribute("agent.cached", cached is not None)
            if cached is not None:
                logger.info("Returning stored result for idempotency key %s", key)
                return cached

        state = await self._checkpoints.get(key) if key and self._checkpoints else None
        if state is not None:
            span.set_attribute("agent.resumed", True)
        elif task is None:
            raise ValueError(f"No paused task for idempotency key {key!r}")
        else:
            state = TaskState(messages=[Message(role="user", text=task)])

        specs = [tool.spec for tool in self._tools.values()] or None
        while True:
            if state.pending:
                await self._execute_pending(state, key)
            if state.step >= self._max_steps:
                raise StepLimitExceeded(self._max_steps, state.cost_usd)

            response = await with_retries(
                lambda: self._complete(state.messages, specs),
                self._retry,
                self._sleep,
            )
            state.step += 1
            state.cost_usd += response.cost_usd
            state.input_tokens += response.input_tokens
            state.output_tokens += response.output_tokens
            if state.cost_usd > self._cost_cap_usd:
                raise CostCapExceeded(state.cost_usd, self._cost_cap_usd, state.step)

            state.messages.append(
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
                    steps=state.step,
                    cost_usd=state.cost_usd,
                    input_tokens=state.input_tokens,
                    output_tokens=state.output_tokens,
                    tool_calls=state.tool_log,
                )
                if key and self._results:
                    await self._results.save(key, result)
                if key and self._checkpoints:
                    await self._checkpoints.delete(key)
                return result
            state.pending = list(response.tool_calls)

    def _plan(self, state: TaskState) -> list[tuple[ToolCall, PolicyCheck]]:
        """Check each pending call in order, counting earlier calls in the batch toward limits."""
        counts = dict(state.tool_calls_made)
        plan = []
        for call in state.pending:
            check = self._policy.check(call, counts)
            if check.decision is not Decision.DENY:
                counts[call.name] = counts.get(call.name, 0) + 1
            plan.append((call, check))
        return plan

    def _awaiting(self, state: TaskState) -> list[ToolCall]:
        return [
            call
            for call, check in self._plan(state)
            if check.decision is Decision.REQUIRE_APPROVAL and call.id not in state.decisions
        ]

    async def _execute_pending(self, state: TaskState, key: str | None) -> None:
        awaiting = self._awaiting(state)
        if awaiting:
            if not key or self._checkpoints is None:
                raise ValueError("Tool calls that need approval require an idempotency key")
            # Nothing in the batch runs until every approval is decided.
            await self._checkpoints.save(key, state)
            raise ApprovalRequired(key, awaiting)

        plan = self._plan(state)
        results = await asyncio.gather(
            *(self._execute(call, check, state.decisions.get(call.id)) for call, check in plan)
        )
        for (call, check), result in zip(plan, results, strict=True):
            state.tool_log.append(ToolUse(call.name, call.input, result.is_error))
            if _runs(check, state.decisions.get(call.id)):
                state.tool_calls_made[call.name] = state.tool_calls_made.get(call.name, 0) + 1
        state.messages.append(Message(role="user", tool_results=list(results)))
        state.pending = []
        state.decisions = {}

    async def _complete(self, messages: list[Message], specs: list[ToolSpec] | None) -> LLMResponse:
        with self._tracer.start_as_current_span("llm.call") as span:
            response = await self._llm.complete(
                messages, specs, self._max_tokens, system=self._system_prompt
            )
            span.set_attributes(
                {
                    "gen_ai.response.model": response.model,
                    "gen_ai.usage.input_tokens": response.input_tokens,
                    "gen_ai.usage.output_tokens": response.output_tokens,
                    "agent.cost_usd": response.cost_usd,
                }
            )
            return response

    async def _execute(
        self, call: ToolCall, check: PolicyCheck, approved: bool | None
    ) -> ToolResult:
        with self._tracer.start_as_current_span("tool.call") as span:
            span.set_attributes(
                {"gen_ai.tool.name": call.name, "agent.tool.decision": check.decision.value}
            )
            started = time.perf_counter()
            if check.decision is Decision.DENY:
                result = _refusal(call, "permission_denied", check.reason)
            elif not _runs(check, approved):
                result = _refusal(call, "approval_rejected", "a reviewer rejected this call")
            else:
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


def _runs(check: PolicyCheck, approved: bool | None) -> bool:
    if check.decision is Decision.ALLOW:
        return True
    return check.decision is Decision.REQUIRE_APPROVAL and approved is True


def _refusal(call: ToolCall, error: str, reason: str) -> ToolResult:
    content = json.dumps({"error": error, "tool": call.name, "reason": reason})
    return ToolResult(call.id, content, is_error=True)
