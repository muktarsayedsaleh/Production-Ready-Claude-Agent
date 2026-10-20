import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.llm import ClaudeClient, ToolCall


class FakeMessages:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.response


class FakeAnthropic:
    def __init__(self, response: SimpleNamespace) -> None:
        self.messages = FakeMessages(response)


def make_response(content: list[SimpleNamespace], input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_complete_parses_text_tool_calls_and_cost() -> None:
    fake = FakeAnthropic(
        make_response(
            [
                SimpleNamespace(type="text", text="Checking the weather."),
                SimpleNamespace(
                    type="tool_use", id="toolu_1", name="get_weather", input={"city": "Paris"}
                ),
            ],
            input_tokens=1000,
            output_tokens=500,
        )
    )
    client = ClaudeClient(model="claude-opus-5-5", client=fake)  # type: ignore[arg-type]
    tools = [{"name": "get_weather", "description": "Get weather", "input_schema": {}}]

    response = asyncio.run(
        client.complete([{"role": "user", "content": "Weather in Paris?"}], tools, max_tokens=256)
    )

    assert response.text == "Checking the weather."
    assert response.tool_calls == [
        ToolCall(id="toolu_1", name="get_weather", input={"city": "Paris"})
    ]
    assert response.input_tokens == 1000
    assert response.output_tokens == 500
    # 1000 * $4/M + 500 * $20/M
    assert response.cost_usd == pytest.approx(0.014)

    call = fake.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 256
    assert call["tools"] == tools


def test_complete_omits_tools_when_none() -> None:
    fake = FakeAnthropic(make_response([SimpleNamespace(type="text", text="hi")], 10, 2))
    client = ClaudeClient(client=fake)  # type: ignore[arg-type]

    response = asyncio.run(client.complete([{"role": "user", "content": "hello"}]))

    assert response.tool_calls == []
    assert "tools" not in fake.messages.calls[0]


def test_unknown_model_fails_fast() -> None:
    fake = FakeAnthropic(make_response([], 0, 0))
    with pytest.raises(ValueError, match="No pricing configured"):
        ClaudeClient(model="claude-unknown", client=fake)  # type: ignore[arg-type]
