"""Ask the research agent a question.

python -m agents.research "How does the retry backoff work?"          # enqueue for a worker
python -m agents.research --sync "How does the retry backoff work?"   # run in-process
"""

import argparse
import asyncio
import json
import logging
import os
import sys

import psycopg

from agents.research.agent import QUEUE_NAME, ResearchAgent, load_policy
from core.llm import ClaudeClient
from core.queue import JobQueue, PostgresJobStore
from core.queue.connections import create_redis
from core.runtime import ApprovalRequired, InMemoryCheckpointStore
from core.tracing import configure_tracing


async def run_sync(question: str, cost_cap_usd: float, as_json: bool) -> int:
    tracer_provider = configure_tracing()
    agent = ResearchAgent(
        ClaudeClient(),
        policy=load_policy(),
        checkpoints=InMemoryCheckpointStore(),
        cost_cap_usd=cost_cap_usd,
    )
    try:
        report = await agent.research(question, idempotency_key="sync")
    except ApprovalRequired as exc:
        print(f"{exc}\nApprovals need the queue; run without --sync.", file=sys.stderr)
        return 1
    finally:
        if tracer_provider is not None:
            tracer_provider.shutdown()
    print(json.dumps(report.to_dict(), indent=2) if as_json else report.format_text())
    return 0


async def enqueue(question: str) -> int:
    redis = create_redis()
    database_url = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5432/agent")
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        job_id = await JobQueue(redis, PostgresJobStore(conn), QUEUE_NAME).enqueue(
            {"question": question}
        )
    await redis.aclose()
    print(f"Enqueued research job {job_id} on queue {QUEUE_NAME!r}.")
    print(
        "A worker picks it up and logs the report:\n"
        f"  python -m core.queue.worker --queue {QUEUE_NAME} "
        "--handler agents.research.agent:handle_job"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the research agent a question.")
    parser.add_argument("question")
    parser.add_argument("--sync", action="store_true", help="run in-process instead of queueing")
    parser.add_argument("--json", action="store_true", help="with --sync, print the report as JSON")
    parser.add_argument("--cost-cap", type=float, default=1.0, help="with --sync, USD per task")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.sync:
        code = asyncio.run(run_sync(args.question, args.cost_cap, args.json))
    else:
        code = asyncio.run(enqueue(args.question))
    sys.exit(code)


if __name__ == "__main__":
    main()
