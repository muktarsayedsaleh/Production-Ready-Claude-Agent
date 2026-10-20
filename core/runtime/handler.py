from core.queue.models import Job
from core.queue.worker import Handler
from core.runtime.agent import AgentRuntime


def make_job_handler(runtime: AgentRuntime) -> Handler:
    """Queue handler that runs `payload["task"]`, keyed by the job's idempotency key.

    The key defaults to the job id, so a redelivered job returns its stored result
    instead of calling the LLM again.
    """

    async def handle(job: Job) -> None:
        key = job.payload.get("idempotency_key") or job.id
        await runtime.run(job.payload["task"], idempotency_key=key)

    return handle
