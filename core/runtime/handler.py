from collections.abc import Mapping

from core.queue.models import Job, JobPaused, JobStatus
from core.queue.queue import JobQueue
from core.queue.store import JobStore
from core.queue.worker import Handler
from core.runtime.agent import AgentRuntime
from core.runtime.errors import ApprovalRequired


def idempotency_key(job: Job) -> str:
    """The payload's `idempotency_key`, or the job id."""
    key = job.payload.get("idempotency_key")
    return str(key) if key else job.id


def make_job_handler(runtime: AgentRuntime) -> Handler:
    """Queue handler that runs `payload["task"]`, keyed by the job's idempotency key.

    A redelivered job returns its stored result instead of calling the LLM again. A task
    that needs approval parks the job as waiting_approval; see `resume_job`.
    """

    async def handle(job: Job) -> None:
        try:
            await runtime.run(job.payload["task"], idempotency_key=idempotency_key(job))
        except ApprovalRequired as exc:
            raise JobPaused(str(exc)) from exc

    return handle


async def resume_job(
    job_id: str,
    decisions: Mapping[str, bool],
    *,
    runtime: AgentRuntime,
    queue: JobQueue,
    store: JobStore,
) -> None:
    """Record reviewer decisions for a waiting job and put it back on the queue.

    The next worker to pick it up continues the task from its checkpoint.
    """
    job = await store.get(job_id)
    if job is None or job.status is not JobStatus.WAITING_APPROVAL:
        raise ValueError(f"Job {job_id} is not waiting for approval")
    await runtime.approve(idempotency_key(job), decisions)
    await queue.requeue(job_id)
