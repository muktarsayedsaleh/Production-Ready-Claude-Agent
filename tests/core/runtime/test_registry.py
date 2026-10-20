import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agents.research.tools import build_registry
from core.llm import LLMResponse, ToolCall
from core.permissions import Policy
from core.runtime import AgentRuntime
from core.runtime.registry import ToolRegistry
from tests.fakes import FakeLLMClient, InMemoryCheckpointStore

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = Policy.from_yaml(REPO_ROOT / "policies" / "default.yaml")


class EchoTool:
    name = "echo"
    description = "Echo the input"
    input_schema: ClassVar[dict[str, Any]] = {"type": "object"}

    async def run(self, args: dict[str, object]) -> str:
        return json.dumps(args)


def test_register_get_and_convert_to_runtime_tools() -> None:
    registry = ToolRegistry([EchoTool()])

    assert "echo" in registry
    assert list(registry) == ["echo"]
    assert registry.get("echo").name == "echo"
    [tool] = registry.tools()
    assert (tool.name, tool.description, tool.input_schema) == (
        "echo",
        "Echo the input",
        {"type": "object"},
    )
    assert asyncio.run(tool.handler({"a": 1})) == '{"a": 1}'


def test_duplicate_and_unknown_tools() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())
    with pytest.raises(KeyError, match="No tool named 'missing'"):
        registry.get("missing")


def test_research_registry_takes_fetch_allowlist_from_policy() -> None:
    registry = build_registry(DEFAULT_POLICY)

    assert sorted(registry) == ["calculator", "fetch_url", "file_search"]
    # Off-allowlist URLs are denied by the policy before the tool is reached...
    call = ToolCall("c1", "fetch_url", {"url": "https://example.com/"})
    assert DEFAULT_POLICY.check(call, {}).decision.value == "deny"
    # ...and refused by the tool itself if it is ever called directly.
    with pytest.raises(PermissionError):
        asyncio.run(registry.get("fetch_url").run({"url": "https://example.com/"}))


def test_runtime_runs_registered_research_tools() -> None:
    llm = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall("c1", "calculator", {"expression": "(2 + 3) * 4"}),
                    ToolCall("c2", "file_search", {"query": "cost cap", "max_results": 1}),
                ],
            ),
            LLMResponse(text="done"),
        ]
    )
    runtime = AgentRuntime(
        llm,
        build_registry(DEFAULT_POLICY).tools(),
        policy=DEFAULT_POLICY,
        checkpoints=InMemoryCheckpointStore(),
    )

    asyncio.run(runtime.run("research"))

    calc, search = llm.calls[1][-1].tool_results
    assert (calc.content, calc.is_error) == ("20", False)
    assert json.loads(search.content)[0]["file"] == "cost-caps.md"
