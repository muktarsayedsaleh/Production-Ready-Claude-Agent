"""The research agent: answers a question with file_search, calculator and fetch_url.

Run it with `python -m agents.research "your question"` (queued) or add `--sync` (in-process).
Workers run queued tasks with the handler in this module:

    python -m core.queue.worker --queue research --handler agents.research.agent:handle_job
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.research.tools import build_registry
from core.llm import ClaudeClient, LLMClient
from core.permissions import Policy
from core.queue import Job, JobPaused
from core.queue.connections import create_redis
from core.queue.worker import Handler
from core.runtime import (
    AgentRuntime,
    ApprovalRequired,
    CheckpointStore,
    RedisCheckpointStore,
    RedisResultStore,
    ResultStore,
    TaskResult,
    ToolUse,
)
from core.runtime.handler import idempotency_key

logger = logging.getLogger(__name__)

QUEUE_NAME = "research"
PROMPT_PATH = Path(__file__).with_name("prompt.md")
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "policies" / "default.yaml"


@dataclass(frozen=True)
class ResearchReport:
    question: str
    answer: str
    sources: list[str]
    steps: list[str]
    llm_calls: int
    cost_usd: float

    @classmethod
    def from_result(cls, question: str, result: TaskResult) -> "ResearchReport":
        answer, sources = parse_final_answer(result.answer)
        return cls(
            question=question,
            answer=answer,
            sources=sources,
            steps=[describe_tool_use(use) for use in result.tool_calls],
            llm_calls=result.steps,
            cost_usd=result.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        lines = [f"Question: {self.question}", "", self.answer, "", "Sources:"]
        lines += [f"  - {source}" for source in self.sources] or ["  (none)"]
        lines += ["", "Steps:"]
        lines += [f"  {i}. {step}" for i, step in enumerate(self.steps, 1)] or ["  (no tools used)"]
        lines += ["", f"LLM calls: {self.llm_calls}    Total cost: ${self.cost_usd:.4f}"]
        return "\n".join(lines)


def parse_final_answer(text: str) -> tuple[str, list[str]]:
    """Read the {"answer", "sources"} JSON the prompt asks for; fall back to the raw text."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return text.strip(), []
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return text.strip(), []
    sources = data.get("sources")
    if not isinstance(sources, list):
        sources = []
    return data["answer"].strip(), list(dict.fromkeys(s for s in sources if isinstance(s, str)))


def describe_tool_use(use: ToolUse) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in use.input.items())
    return f"{use.name}({args})" + (" -> error" if use.is_error else "")


class ResearchAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        policy: Policy,
        checkpoints: CheckpointStore,
        results: ResultStore | None = None,
        docs_dir: str | Path | None = None,
        cost_cap_usd: float = 1.0,
        max_steps: int = 15,
    ) -> None:
        registry = build_registry(policy) if docs_dir is None else build_registry(policy, docs_dir)
        self.runtime = AgentRuntime(
            llm,
            registry.tools(),
            policy=policy,
            results=results,
            checkpoints=checkpoints,
            cost_cap_usd=cost_cap_usd,
            max_steps=max_steps,
            system_prompt=PROMPT_PATH.read_text(),
        )

    async def research(self, question: str, idempotency_key: str | None = None) -> ResearchReport:
        result = await self.runtime.run(question, idempotency_key=idempotency_key)
        return ResearchReport.from_result(question, result)


def load_policy() -> Policy:
    return Policy.from_yaml(os.environ.get("AGENT_POLICY_FILE", DEFAULT_POLICY_PATH))


def make_research_handler(agent: ResearchAgent) -> Handler:
    """Queue handler: research `payload["question"]` and log the report."""

    async def handle(job: Job) -> None:
        question = job.payload["question"]
        try:
            report = await agent.research(question, idempotency_key=idempotency_key(job))
        except ApprovalRequired as exc:
            raise JobPaused(str(exc)) from exc
        logger.info("Research job %s finished:\n%s", job.id, report.format_text())

    return handle


_handler: Handler | None = None


async def handle_job(job: Job) -> None:
    """Worker entry point, configured from the environment on first use."""
    global _handler
    if _handler is None:
        redis = create_redis()
        agent = ResearchAgent(
            ClaudeClient(),
            policy=load_policy(),
            results=RedisResultStore(redis),
            checkpoints=RedisCheckpointStore(redis),
        )
        _handler = make_research_handler(agent)
    await _handler(job)
