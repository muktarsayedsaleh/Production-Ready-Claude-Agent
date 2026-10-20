"""Exit 0 if Redis and Postgres are reachable, 1 otherwise. Used as the worker healthcheck.

python -m core.queue.healthcheck
"""

import asyncio
import os
import sys

import psycopg

from core.queue.connections import create_redis

DEFAULT_DATABASE_URL = "postgresql://agent:agent@localhost:5432/agent"
TIMEOUT_SECONDS = 3


async def check() -> list[str]:
    problems = []

    redis = create_redis()
    try:
        await asyncio.wait_for(redis.ping(), TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - any failure means unhealthy
        problems.append(f"redis: {type(exc).__name__}: {exc}")
    finally:
        await redis.aclose()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        async with await psycopg.AsyncConnection.connect(
            database_url, connect_timeout=TIMEOUT_SECONDS
        ) as conn:
            await conn.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - any failure means unhealthy
        problems.append(f"postgres: {type(exc).__name__}: {exc}")

    return problems


def main() -> None:
    problems = asyncio.run(check())
    for problem in problems:
        print(problem, file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
