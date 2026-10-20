from dataclasses import replace
from typing import Any

from core.llm import LLMClient, LLMResponse, Message, ToolSpec
from core.queue import Job, JobStatus
from core.runtime import TaskResult, TaskState


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


class InMemoryJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    async def create(self, job_id: str, payload: dict[str, Any]) -> None:
        self.jobs[job_id] = Job(id=job_id, payload=payload, status=JobStatus.QUEUED)

    async def mark_running(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        job = replace(job, status=JobStatus.RUNNING, attempts=job.attempts + 1)
        self.jobs[job_id] = job
        return job

    async def mark_succeeded(self, job_id: str) -> None:
        self.jobs[job_id] = replace(self.jobs[job_id], status=JobStatus.SUCCEEDED, last_error=None)

    async def mark_failed(self, job_id: str, error: str) -> None:
        self.jobs[job_id] = replace(self.jobs[job_id], status=JobStatus.FAILED, last_error=error)

    async def mark_waiting_approval(self, job_id: str) -> None:
        self.jobs[job_id] = replace(self.jobs[job_id], status=JobStatus.WAITING_APPROVAL)

    async def mark_queued(self, job_id: str) -> None:
        self.jobs[job_id] = replace(self.jobs[job_id], status=JobStatus.QUEUED, last_error=None)

    async def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


class InMemoryResultStore:
    def __init__(self) -> None:
        self.results: dict[str, TaskResult] = {}

    async def get(self, key: str) -> TaskResult | None:
        return self.results.get(key)

    async def save(self, key: str, result: TaskResult) -> None:
        self.results[key] = result


class InMemoryCheckpointStore:
    """Round-trips through JSON, like the Redis store, so serialization bugs surface."""

    def __init__(self) -> None:
        self.states: dict[str, str] = {}

    async def get(self, key: str) -> TaskState | None:
        raw = self.states.get(key)
        return None if raw is None else TaskState.from_json(raw)

    async def save(self, key: str, state: TaskState) -> None:
        self.states[key] = state.to_json()

    async def delete(self, key: str) -> None:
        self.states.pop(key, None)
