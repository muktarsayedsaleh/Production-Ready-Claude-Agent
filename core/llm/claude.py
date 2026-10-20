from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam

from core.llm.base import LLMClient, LLMResponse, ToolCall
from core.llm.pricing import pricing_for

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, client: AsyncAnthropic | None = None) -> None:
        # Resolve pricing up front so an unpriced model fails at startup, not mid-run.
        self._pricing = pricing_for(model)
        self._model = model
        self._client = client or AsyncAnthropic()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = cast(list[ToolParam], tools)

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=cast(list[MessageParam], messages),
            **kwargs,
        )

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = [
            ToolCall(id=block.id, name=block.name, input=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._pricing.cost(input_tokens, output_tokens),
        )
