"""Integration tests against a real Postgres, with the schema created by the Alembic migrations.

Set TEST_DATABASE_URL to a server where the user may create databases (the docker-compose
`agent` user can). Each test runs in a fresh, temporary database that is dropped afterwards.
"""

import asyncio
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from fakeredis import FakeAsyncRedis
from sqlalchemy.engine import make_url

from core.queue import Job, JobQueue, JobStatus, PostgresJobStore
from core.queue.worker import Worker

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"

pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


@pytest.fixture
def database_url(monkeypatch: pytest.MonkeyPatch, alembic_config: Config) -> Iterator[str]:
    assert TEST_DATABASE_URL
    name = f"test_{uuid.uuid4().hex[:12]}"
    url = make_url(TEST_DATABASE_URL).set(database=name).render_as_string(hide_password=False)

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(alembic_config, "head")
    try:
        yield url
    finally:
        with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


def jobs_columns(url: str) -> dict[str, str]:
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'jobs'"
        ).fetchall()
    return dict(rows)


def test_migration_creates_jobs_table(database_url: str) -> None:
    assert jobs_columns(database_url) == {
        "id": "uuid",
        "payload": "jsonb",
        "status": "text",
        "attempts": "integer",
        "last_error": "text",
        "created_at": "timestamp with time zone",
        "updated_at": "timestamp with time zone",
    }


def test_downgrade_drops_jobs_table(database_url: str, alembic_config: Config) -> None:
    command.downgrade(alembic_config, "base")

    assert jobs_columns(database_url) == {}


def test_queue_and_worker_record_job_outcomes(database_url: str) -> None:
    async def ok(job: Job) -> None:
        pass

    async def boom(job: Job) -> None:
        raise ValueError("bad payload")

    async def scenario() -> tuple[Job | None, Job | None]:
        async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
            store = PostgresJobStore(conn)
            queue = JobQueue(FakeAsyncRedis(), store)

            succeeded = await queue.enqueue({"n": 1})
            assert await Worker(queue, store, ok).run_once(timeout=0.1)
            failed = await queue.enqueue({"n": 2})
            assert await Worker(queue, store, boom).run_once(timeout=0.1)

            return await store.get(succeeded), await store.get(failed)

    succeeded, failed = asyncio.run(scenario())

    assert succeeded is not None and failed is not None
    assert (succeeded.payload, succeeded.status, succeeded.attempts, succeeded.last_error) == (
        {"n": 1},
        JobStatus.SUCCEEDED,
        1,
        None,
    )
    assert (failed.status, failed.attempts, failed.last_error) == (
        JobStatus.FAILED,
        1,
        "bad payload",
    )
