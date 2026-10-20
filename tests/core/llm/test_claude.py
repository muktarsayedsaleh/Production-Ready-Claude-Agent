import asyncio
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import TextBlock, ToolUseBlock

from core.llm import ClaudeClient, Message, ToolCall, ToolResult, ToolSpec, TransientLLMError


class FakeMessages:
    def __init__(self, response: SimpleNamespace | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeAnthropic:
    def __init__(self, response: SimpleNamespace | Exception) -> None:
        self.messages = FakeMessages(response)


def make_response(content: list[SimpleNamespace], input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        model="claude-opus-5-5",
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def status_error(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        f"HTTP {status}", response=httpx2.Response(status, request=request), body=None
    )


WEATHER_TOOL = ToolSpec(
    name="get_weather", description="Get weather", input_schema={"type": "object"}
)


def test_complete_parses_text_tool_calls_and_cost() -> None:
    content = [
        TextBlock(type="text", text="Checking the weather."),
        ToolUseBlock(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Paris"}),
    ]
    fake = FakeAnthropic(make_response(content, input_tokens=1000, output_tokens=500))
    client = ClaudeClient(model="claude-opus-5-5", client=fake)  # type: ignore[arg-type]

    response = asyncio.run(
        client.complete(
            [Message(role="user", text="Weather in Paris?")], [WEATHER_TOOL], max_tokens=256
        )
    )

    assert response.text == "Checking the weather."
    assert response.tool_calls == [
        ToolCall(id="toolu_1", name="get_weather", input={"city": "Paris"})
    ]
    assert response.input_tokens == 1000
    assert response.output_tokens == 500
    # 1000 * $4/M + 500 * $20/M
    assert response.cost_usd == pytest.approx(0.014)
    assert response.model == "claude-opus-5-5"
    assert response.provider_state == [
        {"type": "text", "text": "Checking the weather."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]

    call = fake.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 256
    assert call["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}
    ]
    assert call["tools"] == [
        {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}
    ]


def test_complete_omits_tools_when_none() -> None:
    fake = FakeAnthropic(make_response([TextBlock(type="text", text="hi")], 10, 2))
    client = ClaudeClient(client=fake)  # type: ignore[arg-type]

    response = asyncio.run(client.complete([Message(role="user", text="hello")]))

    assert response.tool_calls == []
    assert "tools" not in fake.messages.calls[0]


def test_complete_converts_history_and_echoes_provider_state() -> None:
    fake = FakeAnthropic(make_response([TextBlock(type="text", text="done")], 1, 1))
    client = ClaudeClient(client=fake)  # type: ignore[arg-type]
    raw_assistant = [{"type": "thinking", "thinking": "", "signature": "sig"}]

    asyncio.run(
        client.complete(
            [
                Message(role="user", text="go"),
                Message(role="assistant", provider_state=raw_assistant),
                Message(
                    role="user",
                    tool_results=[ToolResult("toolu_1", "boom", is_error=True)],
                ),
            ]
        )
    )

    sent = fake.messages.calls[0]["messages"]
    assert sent[1] == {"role": "assistant", "content": raw_assistant}
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "boom", "is_error": True}
        ],
    }


@pytest.mark.parametrize("status", [429, 500, 529])
def test_retryable_status_raises_transient_error(status: int) -> None:
    client = ClaudeClient(client=FakeAnthropic(status_error(status)))  # type: ignore[arg-type]

    with pytest.raises(TransientLLMError):
        asyncio.run(client.complete([Message(role="user", text="hi")]))


def test_connection_error_raises_transient_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = ClaudeClient(
        client=FakeAnthropic(anthropic.APIConnectionError(request=request))  # type: ignore[arg-type]
    )

    with pytest.raises(TransientLLMError):
        asyncio.run(client.complete([Message(role="user", text="hi")]))


def test_client_error_is_not_transient() -> None:
    client = ClaudeClient(client=FakeAnthropic(status_error(400)))  # type: ignore[arg-type]

    with pytest.raises(anthropic.APIStatusError):
        asyncio.run(client.complete([Message(role="user", text="hi")]))


def test_unknown_model_fails_fast() -> None:
    fake = FakeAnthropic(make_response([], 0, 0))
    with pytest.raises(ValueError, match="No pricing configured"):
        ClaudeClient(model="claude-unknown", client=fake)  # type: ignore[arg-type]
