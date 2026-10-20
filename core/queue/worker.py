import argparse
import asyncio
import importlib
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import psycopg
from redis.asyncio import Redis

from core.queue.queue import DEFAULT_QUEUE, JobQueue
from core.queue.store import JobStore, PostgresJobStore

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[None]]


class Worker:
    def __init__(self, queue: JobQueue, store: JobStore, handler: Handler) -> None:
        self._queue = queue
        self._store = store
        self._handler = handler

    async def run_once(self, timeout: float = 5) -> bool:
        """Process at most one job. Returns False if the queue was empty."""
        job_id = await self._queue.pop(timeout=timeout)
        if job_id is None:
            return False

        job = await self._store.mark_running(job_id)
        if job is None:
            logger.warning("Popped unknown job %s, skipping", job_id)
            return True

        try:
            await self._handler(job.payload)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            await self._store.mark_failed(job_id, str(exc) or type(exc).__name__)
        else:
            logger.info("Job %s succeeded", job_id)
            await self._store.mark_succeeded(job_id)
        return True

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        while not stop.is_set():
            await self.run_once()


async def log_payload(payload: dict[str, Any]) -> None:
    logger.info("Processing payload: %s", payload)


def _load_handler(path: str) -> Handler:
    module_name, _, attr = path.partition(":")
    handler: Handler = getattr(importlib.import_module(module_name), attr)
    return handler


async def _main(queue_name: str, handler: Handler) -> None:
    redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    database_url = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5432/agent")
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        store = PostgresJobStore(conn)
        await store.ensure_schema()
        logger.info("Worker listening on queue %r", queue_name)
        await Worker(JobQueue(redis, store, queue_name), store, handler).run()


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
