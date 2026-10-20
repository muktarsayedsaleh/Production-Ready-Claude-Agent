from typing import Any, Protocol

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from core.queue.models import Job, JobStatus


class JobStore(Protocol):
    async def create(self, job_id: str, payload: dict[str, Any]) -> None: ...

    async def mark_running(self, job_id: str) -> Job | None:
        """Mark the job running, increment attempts, and return it (None if unknown)."""
        ...

    async def mark_succeeded(self, job_id: str) -> None: ...

    async def mark_failed(self, job_id: str, error: str) -> None: ...

    async def mark_waiting_approval(self, job_id: str) -> None: ...

    async def mark_queued(self, job_id: str) -> None: ...

    async def get(self, job_id: str) -> Job | None: ...


class PostgresJobStore:
    def __init__(self, conn: AsyncConnection[Any]) -> None:
        self._conn = conn

    async def create(self, job_id: str, payload: dict[str, Any]) -> None:
        await self._conn.execute(
            "INSERT INTO jobs (id, payload, status) VALUES (%s, %s, %s)",
            (job_id, Jsonb(payload), JobStatus.QUEUED.value),
        )

    async def mark_running(self, job_id: str) -> Job | None:
        cur = await self._conn.execute(
            """
            UPDATE jobs SET status = %s, attempts = attempts + 1, updated_at = now()
            WHERE id = %s
            RETURNING id, payload, status, attempts, last_error
            """,
            (JobStatus.RUNNING.value, job_id),
        )
        return _to_job(await cur.fetchone())

    async def mark_succeeded(self, job_id: str) -> None:
        await self._set_status(job_id, JobStatus.SUCCEEDED, None)

    async def mark_failed(self, job_id: str, error: str) -> None:
        await self._set_status(job_id, JobStatus.FAILED, error)

    async def mark_waiting_approval(self, job_id: str) -> None:
        await self._set_status(job_id, JobStatus.WAITING_APPROVAL, None)

    async def mark_queued(self, job_id: str) -> None:
        await self._set_status(job_id, JobStatus.QUEUED, None)

    async def get(self, job_id: str) -> Job | None:
        cur = await self._conn.execute(
            "SELECT id, payload, status, attempts, last_error FROM jobs WHERE id = %s",
            (job_id,),
        )
        return _to_job(await cur.fetchone())

    async def _set_status(self, job_id: str, status: JobStatus, error: str | None) -> None:
        await self._conn.execute(
            "UPDATE jobs SET status = %s, last_error = %s, updated_at = now() WHERE id = %s",
            (status.value, error, job_id),
        )


def _to_job(row: tuple[Any, ...] | None) -> Job | None:
    if row is None:
        return None
    job_id, payload, status, attempts, last_error = row
    return Job(
        id=str(job_id),
        payload=payload,
        status=JobStatus(status),
        attempts=attempts,
        last_error=last_error,
    )
