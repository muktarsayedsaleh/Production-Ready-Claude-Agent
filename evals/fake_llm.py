import json
from collections.abc import Mapping, Sequence
from typing import Any

from core.llm import LLMClient, LLMResponse, Message, ToolCall, ToolSpec


class ScriptedLLMClient(LLMClient):
    """Replays a case's `fake_llm` turns so evals run in CI without an API key.

    Each turn is either tool calls or the final answer:

        - tool_calls:
            - {name: file_search, input: {query: backoff}}
          cost_usd: 0.002          # optional
        - answer: {answer: "...", sources: [...]}   # or a plain string

    The tool calls still go through the real tools and policy; only the model is scripted.
    """

    def __init__(self, turns: Sequence[Mapping[str, Any]]) -> None:
        self._responses = [_response(index, turn) for index, turn in enumerate(turns)]

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
        system: str | None = None,
    ) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("The fake_llm script ran out of turns before a final answer")
        return self._responses.pop(0)


def _response(index: int, turn: Mapping[str, Any]) -> LLMResponse:
    cost = float(turn.get("cost_usd", 0.0))
    if "answer" in turn:
        answer = turn["answer"]
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return LLMResponse(text=text, cost_usd=cost, model="fake")
    if "tool_calls" in turn:
        calls = [
            ToolCall(id=f"call_{index}_{i}", name=call["name"], input=dict(call.get("input", {})))
            for i, call in enumerate(turn["tool_calls"])
        ]
        return LLMResponse(text="", tool_calls=calls, cost_usd=cost, model="fake")
    raise ValueError(f"fake_llm turn {index} needs 'tool_calls' or 'answer'")
