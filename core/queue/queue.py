import uuid
from typing import Any

from redis.asyncio import Redis

from core.queue.store import JobStore

DEFAULT_QUEUE = "jobs"


class JobQueue:
    """FIFO queue of job ids in Redis; the job record itself lives in the JobStore."""

    def __init__(self, redis: Redis, store: JobStore, name: str = DEFAULT_QUEUE) -> None:
        self._redis = redis
        self._store = store
        self._key = f"queue:{name}"

    async def enqueue(self, job: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        # Record first so a worker never pops an id that has no row.
        await self._store.create(job_id, job)
        await self._redis.lpush(self._key, job_id)
        return job_id

    async def requeue(self, job_id: str) -> None:
        """Put an existing job back on the queue, e.g. after its approval was decided."""
        await self._store.mark_queued(job_id)
        await self._redis.lpush(self._key, job_id)

    async def pop(self, timeout: float = 5) -> str | None:
        """Block up to `timeout` seconds for the next job id."""
        result = await self._redis.brpop([self._key], timeout=timeout)
        if result is None:
            return None
        _, job_id = result
        return job_id.decode() if isinstance(job_id, bytes) else str(job_id)
