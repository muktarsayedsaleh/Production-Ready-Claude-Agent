import asyncio
from dataclasses import replace
from typing import Any

from fakeredis import FakeAsyncRedis

from core.queue import Job, JobQueue, JobStatus
from core.queue.worker import Worker


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

    async def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


def make_queue() -> tuple[JobQueue, InMemoryJobStore]:
    store = InMemoryJobStore()
    return JobQueue(FakeAsyncRedis(), store), store


def test_enqueue_records_job_and_pushes_id() -> None:
    async def scenario() -> None:
        queue, store = make_queue()

        job_id = await queue.enqueue({"task": "summarize"})

        assert store.jobs[job_id] == Job(
            id=job_id, payload={"task": "summarize"}, status=JobStatus.QUEUED
        )
        assert await queue.pop(timeout=0.1) == job_id

    asyncio.run(scenario())


def test_jobs_are_processed_in_fifo_order() -> None:
    async def scenario() -> None:
        queue, store = make_queue()
        seen: list[dict[str, Any]] = []

        async def handler(payload: dict[str, Any]) -> None:
            seen.append(payload)

        first = await queue.enqueue({"n": 1})
        second = await queue.enqueue({"n": 2})
        worker = Worker(queue, store, handler)
        await worker.run_once(timeout=0.1)
        await worker.run_once(timeout=0.1)

        assert seen == [{"n": 1}, {"n": 2}]
        for job_id in (first, second):
            job = store.jobs[job_id]
            assert job.status == JobStatus.SUCCEEDED
            assert job.attempts == 1

    asyncio.run(scenario())


def test_failed_job_records_error() -> None:
    async def scenario() -> None:
        queue, store = make_queue()

        async def handler(payload: dict[str, Any]) -> None:
            raise RuntimeError("model unavailable")

        job_id = await queue.enqueue({"task": "boom"})
        await Worker(queue, store, handler).run_once(timeout=0.1)

        job = store.jobs[job_id]
        assert job.status == JobStatus.FAILED
        assert job.last_error == "model unavailable"
        assert job.attempts == 1

    asyncio.run(scenario())


def test_run_once_returns_false_when_queue_empty() -> None:
    async def scenario() -> None:
        queue, store = make_queue()

        async def handler(payload: dict[str, Any]) -> None:
            raise AssertionError("should not be called")

        assert await Worker(queue, store, handler).run_once(timeout=0.1) is False

    asyncio.run(scenario())
