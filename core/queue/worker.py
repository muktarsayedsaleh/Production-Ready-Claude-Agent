import argparse
import asyncio
import importlib
import logging
import os
from collections.abc import Awaitable, Callable

import psycopg
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from core.queue.connections import POP_TIMEOUT_SECONDS, create_redis
from core.queue.models import Job, JobPaused
from core.queue.queue import DEFAULT_QUEUE, JobQueue
from core.queue.store import JobStore, PostgresJobStore
from core.tracing import configure_tracing

logger = logging.getLogger(__name__)

Handler = Callable[[Job], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]

MAX_RECONNECT_DELAY_SECONDS = 30.0


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        store: JobStore,
        handler: Handler,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._queue = queue
        self._store = store
        self._handler = handler
        self._sleep = sleep

    async def run_once(self, timeout: float = POP_TIMEOUT_SECONDS) -> bool:
        """Process at most one job. Returns False if the queue was empty."""
        job_id = await self._queue.pop(timeout=timeout)
        if job_id is None:
            return False

        job = await self._store.mark_running(job_id)
        if job is None:
            logger.warning("Popped unknown job %s, skipping", job_id)
            return True

        try:
            await self._handler(job)
        except JobPaused as exc:
            logger.info("Job %s paused: %s", job_id, exc)
            await self._store.mark_waiting_approval(job_id)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            await self._store.mark_failed(job_id, str(exc) or type(exc).__name__)
        else:
            logger.info("Job %s succeeded", job_id)
            await self._store.mark_succeeded(job_id)
        return True

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Process jobs until `stop` is set. Redis outages are waited out, never fatal."""
        stop = stop or asyncio.Event()
        failures = 0
        while not stop.is_set():
            try:
                await self.run_once()
            except (RedisTimeoutError, RedisConnectionError) as exc:
                failures += 1
                delay = reconnect_delay(failures)
                logger.warning(
                    "Redis unavailable (%s: %s), reconnecting in %.0fs (attempt %d)",
                    type(exc).__name__,
                    exc,
                    delay,
                    failures,
                )
                await self._sleep(delay)
                await self._reconnect()
            else:
                if failures:
                    logger.info("Redis connection restored after %d attempt(s)", failures)
                failures = 0

    async def _reconnect(self) -> None:
        try:
            await self._queue.reconnect()
        except (RedisTimeoutError, RedisConnectionError):
            # Still down; the next run_once fails fast and backs off further.
            logger.debug("Reconnect failed", exc_info=True)


def reconnect_delay(failures: int) -> float:
    """1s, 2s, 4s, ... capped at MAX_RECONNECT_DELAY_SECONDS."""
    return min(MAX_RECONNECT_DELAY_SECONDS, 2.0 ** (failures - 1))


async def log_payload(job: Job) -> None:
    logger.info("Processing job %s: %s", job.id, job.payload)


def _load_handler(path: str) -> Handler:
    module_name, _, attr = path.partition(":")
    handler: Handler = getattr(importlib.import_module(module_name), attr)
    return handler


async def _main(queue_name: str, handler: Handler) -> None:
    redis = create_redis()
    database_url = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5432/agent")
    tracer_provider = configure_tracing()
    try:
        async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
            store = PostgresJobStore(conn)
            logger.info("Worker listening on queue %r", queue_name)
            await Worker(JobQueue(redis, store, queue_name), store, handler).run()
    finally:
        if tracer_provider is not None:
            tracer_provider.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a job queue worker.")
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument(
        "--handler",
        default="core.queue.worker:log_payload",
        help="Async job handler as module:function",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(_main(args.queue, _load_handler(args.handler)))
    except KeyboardInterrupt:
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
