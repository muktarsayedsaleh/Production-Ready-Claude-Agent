import asyncio

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from core.queue import Job, JobStatus
from core.queue.connections import POP_TIMEOUT_SECONDS, create_redis
from core.queue.worker import Worker, reconnect_delay
from tests.fakes import InMemoryJobStore


class FlakyQueue:
    """Stands in for JobQueue: each pop() returns or raises the next scripted outcome."""

    def __init__(self, outcomes: list[str | None | Exception], stop: asyncio.Event) -> None:
        self.outcomes = list(outcomes)
        self.stop = stop
        self.reconnects = 0

    async def pop(self, timeout: float = POP_TIMEOUT_SECONDS) -> str | None:
        outcome = self.outcomes.pop(0)
        if not self.outcomes:
            self.stop.set()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def reconnect(self) -> None:
        self.reconnects += 1


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def noop_handler(job: Job) -> None:
    pass


def run_worker(
    outcomes: list[str | None | Exception], store: InMemoryJobStore | None = None
) -> tuple[FlakyQueue, RecordingSleep]:
    async def scenario() -> tuple[FlakyQueue, RecordingSleep]:
        stop = asyncio.Event()
        queue = FlakyQueue(outcomes, stop)
        sleep = RecordingSleep()
        worker = Worker(queue, store or InMemoryJobStore(), noop_handler, sleep=sleep)  # type: ignore[arg-type]
        await asyncio.wait_for(worker.run(stop), timeout=5)
        return queue, sleep

    return asyncio.run(scenario())


def test_pop_timeout_does_not_stop_the_loop() -> None:
    queue, sleep = run_worker(
        [
            RedisTimeoutError("Timeout reading from redis:6379"),
            RedisTimeoutError("Timeout reading from redis:6379"),
            RedisConnectionError("Connection refused"),
            None,
        ]
    )

    assert queue.outcomes == []  # every pop happened: the loop survived three Redis errors
    assert sleep.delays == [1.0, 2.0, 4.0]
    assert queue.reconnects == 3


def test_backoff_resets_after_a_successful_pop() -> None:
    _, sleep = run_worker(
        [RedisTimeoutError("t"), RedisTimeoutError("t"), None, RedisTimeoutError("t"), None]
    )

    assert sleep.delays == [1.0, 2.0, 1.0]


def test_jobs_are_processed_once_redis_recovers() -> None:
    async def setup() -> InMemoryJobStore:
        store = InMemoryJobStore()
        await store.create("job-1", {"task": "t"})
        return store

    store = asyncio.run(setup())
    run_worker([RedisConnectionError("down"), "job-1"], store)

    assert store.jobs["job-1"].status is JobStatus.SUCCEEDED


def test_non_redis_errors_still_propagate() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        run_worker([RuntimeError("boom"), None])


def test_reconnect_delay_doubles_and_caps_at_30s() -> None:
    assert [reconnect_delay(n) for n in range(1, 8)] == [1, 2, 4, 8, 16, 30, 30]


def test_redis_socket_timeout_outlasts_the_blocking_pop() -> None:
    kwargs = create_redis("redis://localhost:6379/0").connection_pool.connection_kwargs

    assert kwargs["socket_timeout"] > POP_TIMEOUT_SECONDS
    assert kwargs["socket_connect_timeout"] == 5
