"""TODO: your agent. A skeleton to copy; see README.md in this folder.

This mirrors agents/research/agent.py with the research-specific parts removed. Copy this
folder to a package name with underscores (e.g. agents/support_bot), then work through the
TODOs. Until then it is not importable, because 'your-agent' is not a valid module name.
"""

from dataclasses import dataclass
from pathlib import Path

from core.llm import LLMClient
from core.permissions import Policy
from core.queue import Job, JobPaused
from core.queue.worker import Handler
from core.runtime import (
    AgentRuntime,
    ApprovalRequired,
    CheckpointStore,
    ResultStore,
    TaskResult,
    ToolRegistry,
)
from core.runtime.handler import idempotency_key

# TODO: pick a queue name for your agent's jobs.
QUEUE_NAME = "your_agent"

# TODO: write the system prompt in prompt.md next to this file.
PROMPT_PATH = Path(__file__).with_name("prompt.md")


def build_registry(policy: Policy) -> ToolRegistry:
    """TODO: register your tools. Each needs name, description, input_schema and run()."""
    # Every tool also needs a rule in the policy file, or the runtime denies it.
    return ToolRegistry([])


@dataclass(frozen=True)
class YourReport:
    """TODO: the result your callers need."""

    answer: str
    cost_usd: float

    @classmethod
    def from_result(cls, result: TaskResult) -> "YourReport":
        # TODO: parse result.answer into your format (see parse_final_answer in the research
        # agent); result.tool_calls lists every tool call the task made.
        return cls(answer=result.answer, cost_usd=result.cost_usd)


class YourAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        policy: Policy,
        checkpoints: CheckpointStore,
        results: ResultStore | None = None,
        cost_cap_usd: float = 1.0,  # TODO: choose a cost cap per task
        max_steps: int = 15,  # TODO: and a step limit
    ) -> None:
        self.runtime = AgentRuntime(
            llm,
            build_registry(policy).tools(),
            policy=policy,
            results=results,
            checkpoints=checkpoints,
            cost_cap_usd=cost_cap_usd,
            max_steps=max_steps,
            system_prompt=PROMPT_PATH.read_text(),
        )

    async def run(self, task: str, idempotency_key: str | None = None) -> YourReport:
        return YourReport.from_result(await self.runtime.run(task, idempotency_key))


def make_handler(agent: YourAgent) -> Handler:
    """Queue handler. TODO: read your payload fields and decide what to do with the report."""

    async def handle(job: Job) -> None:
        try:
            await agent.run(job.payload["task"], idempotency_key=idempotency_key(job))
        except ApprovalRequired as exc:
            raise JobPaused(str(exc)) from exc

    return handle


# TODO: add handle_job() that builds YourAgent from the environment (ClaudeClient, Redis
# stores, policy), as in agents/research/agent.py, and a __main__.py for the CLI.
