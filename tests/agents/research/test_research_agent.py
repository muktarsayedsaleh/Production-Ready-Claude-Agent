"""End-to-end tests of the research agent: real tools, real policy, scripted (fake) model."""

import asyncio
import json
from pathlib import Path

import pytest
from fakeredis import FakeAsyncRedis

from agents.research.agent import (
    PROMPT_PATH,
    QUEUE_NAME,
    ResearchAgent,
    make_research_handler,
    parse_final_answer,
)
from core.llm import LLMResponse, ToolCall
from core.permissions import Policy
from core.queue import JobQueue, JobStatus
from core.queue.worker import Worker
from core.runtime import RedisResultStore
from tests.fakes import FakeLLMClient, InMemoryCheckpointStore, InMemoryJobStore

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY = Policy.from_yaml(REPO_ROOT / "policies" / "default.yaml")

FINAL = json.dumps(
    {
        "answer": "Backoff doubles the delay after each failure. Four retries wait 15 seconds.",
        "sources": ["retries-and-backoff.md", "retries-and-backoff.md"],
    }
)


def scripted_llm() -> FakeLLMClient:
    """A plan: search the docs, try a page that isn't allowed, compute, then answer."""
    return FakeLLMClient(
        [
            LLMResponse(
                text="I'll check the docs and the web.",
                tool_calls=[
                    ToolCall("c1", "file_search", {"query": "exponential backoff"}),
                    ToolCall("c2", "fetch_url", {"url": "https://example.com/backoff"}),
                ],
                cost_usd=0.02,
            ),
            LLMResponse(
                text="",
                tool_calls=[ToolCall("c3", "calculator", {"expression": "1 + 2 + 4 + 8"})],
                cost_usd=0.01,
            ),
            LLMResponse(text=FINAL, cost_usd=0.01),
        ]
    )


def test_research_agent_end_to_end() -> None:
    llm = scripted_llm()
    agent = ResearchAgent(llm, policy=POLICY, checkpoints=InMemoryCheckpointStore())

    report = asyncio.run(agent.research("How long do four retries wait in total?"))

    assert report.answer.startswith("Backoff doubles the delay")
    assert report.sources == ["retries-and-backoff.md"]
    assert report.steps == [
        "file_search(query='exponential backoff')",
        "fetch_url(url='https://example.com/backoff') -> error",
        "calculator(expression='1 + 2 + 4 + 8')",
    ]
    assert report.llm_calls == 3
    assert report.cost_usd == pytest.approx(0.04)

    # The model got the system prompt and all three tools.
    assert llm.systems_seen == [PROMPT_PATH.read_text()] * 3
    assert sorted(t.name for t in llm.tools_seen[0] or []) == [
        "calculator",
        "fetch_url",
        "file_search",
    ]

    # Real tool output went back to the model: doc hits, a policy denial, the sum.
    search, fetch = llm.calls[1][-1].tool_results
    assert json.loads(search.content)[0]["file"] == "retries-and-backoff.md"
    assert json.loads(fetch.content)["error"] == "permission_denied"
    [calc] = llm.calls[2][-1].tool_results
    assert calc.content == "15"

    text = report.format_text()
    assert "  - retries-and-backoff.md" in text
    assert "LLM calls: 3    Total cost: $0.0400" in text


def test_research_job_runs_through_the_queue() -> None:
    async def scenario() -> None:
        redis = FakeAsyncRedis()
        results = RedisResultStore(redis)
        agent = ResearchAgent(
            scripted_llm(), policy=POLICY, checkpoints=InMemoryCheckpointStore(), results=results
        )
        store = InMemoryJobStore()
        queue = JobQueue(redis, store, QUEUE_NAME)

        job_id = await queue.enqueue({"question": "How long do four retries wait?"})
        await Worker(queue, store, make_research_handler(agent)).run_once(timeout=0.1)

        assert store.jobs[job_id].status is JobStatus.SUCCEEDED
        stored = await results.get(job_id)
        assert stored is not None
        assert [use.name for use in stored.tool_calls] == [
            "file_search",
            "fetch_url",
            "calculator",
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "A", "sources": ["x.md"]}', ("A", ["x.md"])),
        ('```json\n{"answer": "A", "sources": []}\n```', ("A", [])),
        ('{"answer": "A", "sources": "x.md"}', ("A", [])),
        ('{"answer": "A", "sources": ["x.md", 3]}', ("A", ["x.md"])),
        ("Plain text answer.", ("Plain text answer.", [])),
        ('{"result": "no answer key"}', ('{"result": "no answer key"}', [])),
    ],
)
def test_parse_final_answer(text: str, expected: tuple[str, list[str]]) -> None:
    assert parse_final_answer(text) == expected
