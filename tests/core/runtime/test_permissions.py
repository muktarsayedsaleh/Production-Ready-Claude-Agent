import asyncio
import json
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis

from core.llm import LLMResponse, ToolCall
from core.permissions import Policy
from core.queue import JobQueue, JobStatus
from core.queue.worker import Worker
from core.runtime import (
    AgentRuntime,
    ApprovalRequired,
    RedisCheckpointStore,
    Tool,
)
from core.runtime.handler import make_job_handler, resume_job
from tests.fakes import FakeLLMClient, InMemoryCheckpointStore, InMemoryJobStore

POLICY = Policy.from_dict(
    {
        "tools": {
            "search": {"decision": "allow", "max_calls_per_task": 2},
            "shell": {"decision": "deny"},
            "send_email": {"decision": "require_approval"},
        }
    }
)


class RecordingTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def tool(self, name: str) -> Tool:
        async def handler(args: dict[str, Any]) -> str:
            self.calls.append((name, args))
            return f"{name} ok"

        return Tool(name, f"The {name} tool", {"type": "object"}, handler)

    def all(self) -> list[Tool]:
        return [self.tool(name) for name in ("search", "shell", "send_email")]


def uses(*calls: ToolCall) -> LLMResponse:
    # provider_state stands in for raw provider blocks that must survive a checkpoint.
    return LLMResponse(
        text="",
        tool_calls=list(calls),
        cost_usd=0.01,
        provider_state=[{"type": "thinking", "thinking": "", "signature": "sig"}],
    )


def answer(text: str) -> LLMResponse:
    return LLMResponse(text=text, cost_usd=0.01)


def last_tool_results(llm: FakeLLMClient) -> list[tuple[str, bool]]:
    return [(r.content, r.is_error) for r in llm.calls[-1][-1].tool_results]


def approval_runtime(
    llm: FakeLLMClient, tools: RecordingTools, checkpoints: Any = None
) -> AgentRuntime:
    return AgentRuntime(
        llm, tools.all(), policy=POLICY, checkpoints=checkpoints or InMemoryCheckpointStore()
    )


def test_allowed_tool_runs() -> None:
    tools = RecordingTools()
    llm = FakeLLMClient([uses(ToolCall("c1", "search", {"q": "x"})), answer("done")])

    result = asyncio.run(approval_runtime(llm, tools).run("task"))

    assert result.answer == "done"
    assert tools.calls == [("search", {"q": "x"})]
    assert last_tool_results(llm) == [("search ok", False)]


def test_denied_tool_returns_structured_error_without_running() -> None:
    tools = RecordingTools()
    llm = FakeLLMClient([uses(ToolCall("c1", "shell", {"cmd": "rm -rf /"})), answer("ok")])

    asyncio.run(approval_runtime(llm, tools).run("task"))

    assert tools.calls == []
    [(content, is_error)] = last_tool_results(llm)
    assert is_error
    assert json.loads(content) == {
        "error": "permission_denied",
        "tool": "shell",
        "reason": "tool 'shell' is denied by policy",
    }


def test_call_limit_is_enforced_across_steps_and_within_a_batch() -> None:
    tools = RecordingTools()
    llm = FakeLLMClient(
        [
            uses(ToolCall("c1", "search", {"q": 1})),
            uses(ToolCall("c2", "search", {"q": 2}), ToolCall("c3", "search", {"q": 3})),
            answer("done"),
        ]
    )

    asyncio.run(approval_runtime(llm, tools).run("task"))

    assert tools.calls == [("search", {"q": 1}), ("search", {"q": 2})]
    allowed, denied = last_tool_results(llm)
    assert allowed == ("search ok", False)
    assert (
        json.loads(denied[0])["reason"] == "tool 'search' reached its limit of 2 call(s) per task"
    )


def test_approval_pauses_then_resume_runs_the_approved_call() -> None:
    tools = RecordingTools()
    checkpoints = InMemoryCheckpointStore()
    llm = FakeLLMClient(
        [
            uses(
                ToolCall("c1", "search", {"q": "contacts"}),
                ToolCall("c2", "send_email", {"to": "a@example.com"}),
            ),
            answer("sent"),
        ]
    )
    runtime = approval_runtime(llm, tools, checkpoints)

    with pytest.raises(ApprovalRequired) as excinfo:
        asyncio.run(runtime.run("email the team", idempotency_key="task-1"))

    # Nothing in the batch ran, and the pending call is reported for review.
    assert [c.name for c in excinfo.value.pending] == ["send_email"]
    assert tools.calls == []
    assert "task-1" in checkpoints.states
    pending = asyncio.run(runtime.pending_approvals("task-1"))
    assert [c.id for c in pending] == ["c2"]

    result = asyncio.run(runtime.resume("task-1", {"c2": True}))

    assert result.answer == "sent"
    assert result.steps == 2
    assert tools.calls == [("search", {"q": "contacts"}), ("send_email", {"to": "a@example.com"})]
    # The model was not re-asked before the pause was resolved, and saw both results after.
    assert len(llm.calls) == 2
    assert last_tool_results(llm) == [("search ok", False), ("send_email ok", False)]
    assert llm.calls[1][1].provider_state == [
        {"type": "thinking", "thinking": "", "signature": "sig"}
    ]
    assert checkpoints.states == {}


def test_rejected_approval_returns_structured_error() -> None:
    tools = RecordingTools()
    llm = FakeLLMClient([uses(ToolCall("c1", "send_email", {"to": "x"})), answer("not sent")])
    runtime = approval_runtime(llm, tools)

    with pytest.raises(ApprovalRequired):
        asyncio.run(runtime.run("task", idempotency_key="task-1"))
    result = asyncio.run(runtime.resume("task-1", {"c1": False}))

    assert result.answer == "not sent"
    assert tools.calls == []
    [(content, is_error)] = last_tool_results(llm)
    assert is_error
    assert json.loads(content)["error"] == "approval_rejected"


def test_resume_validates_decisions() -> None:
    llm = FakeLLMClient([uses(ToolCall("c1", "send_email", {}), ToolCall("c2", "send_email", {}))])
    runtime = approval_runtime(llm, RecordingTools())

    with pytest.raises(ApprovalRequired):
        asyncio.run(runtime.run("task", idempotency_key="task-1"))

    with pytest.raises(ValueError, match="Missing decisions for: c2"):
        asyncio.run(runtime.resume("task-1", {"c1": True}))
    with pytest.raises(ValueError, match="Not awaiting approval: c9"):
        asyncio.run(runtime.resume("task-1", {"c1": True, "c2": True, "c9": True}))
    with pytest.raises(ValueError, match="No paused task"):
        asyncio.run(runtime.resume("other", {}))


def test_approval_policy_requires_a_checkpoint_store() -> None:
    with pytest.raises(ValueError, match="needs a CheckpointStore"):
        AgentRuntime(FakeLLMClient([]), policy=POLICY)


def test_job_waits_for_approval_and_resumes_through_the_queue() -> None:
    async def scenario() -> None:
        tools = RecordingTools()
        llm = FakeLLMClient([uses(ToolCall("c1", "send_email", {"to": "x"})), answer("sent")])
        redis = FakeAsyncRedis()
        runtime = approval_runtime(llm, tools, RedisCheckpointStore(redis))
        store = InMemoryJobStore()
        queue = JobQueue(redis, store)
        worker = Worker(queue, store, make_job_handler(runtime))

        job_id = await queue.enqueue({"task": "email someone"})
        await worker.run_once(timeout=0.1)

        assert store.jobs[job_id].status is JobStatus.WAITING_APPROVAL
        assert tools.calls == []
        with pytest.raises(ValueError, match="not waiting for approval"):
            await resume_job("missing", {}, runtime=runtime, queue=queue, store=store)

        await resume_job(job_id, {"c1": True}, runtime=runtime, queue=queue, store=store)
        assert store.jobs[job_id].status is JobStatus.QUEUED
        await worker.run_once(timeout=0.1)

        job = store.jobs[job_id]
        assert (job.status, job.attempts) == (JobStatus.SUCCEEDED, 2)
        assert tools.calls == [("send_email", {"to": "x"})]

    asyncio.run(scenario())
