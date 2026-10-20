from typing import Any, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam

from core.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    TransientLLMError,
)
from core.llm.pricing import pricing_for

DEFAULT_MODEL = "claude-opus-5-5"

_TRANSIENT_STATUS_CODES = {408, 409, 429}


class ClaudeClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, client: AsyncAnthropic | None = None) -> None:
        # Resolve pricing up front so an unpriced model fails at startup, not mid-run.
        self._pricing = pricing_for(model)
        self._model = model
        # SDK retries are off: the runtime owns retries, so they aren't multiplied.
        self._client = client or AsyncAnthropic(max_retries=0)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [_to_tool_param(tool) for tool in tools]

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[_to_message_param(message) for message in messages],
                **kwargs,
            )
        except anthropic.APIConnectionError as exc:
            raise TransientLLMError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in _TRANSIENT_STATUS_CODES or exc.status_code >= 500:
                raise TransientLLMError(str(exc)) from exc
            raise

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
            model=response.model,
            # Plain dicts, so the state can be checkpointed as JSON and sent back as-is.
            provider_state=[block.to_dict() for block in response.content],
        )


def _to_tool_param(tool: ToolSpec) -> ToolParam:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": cast(Any, tool.input_schema),
    }


def _to_message_param(message: Message) -> MessageParam:
    # Echo our own assistant content back verbatim so thinking blocks are preserved.
    if message.provider_state is not None:
        return cast(MessageParam, {"role": message.role, "content": message.provider_state})

    content: list[dict[str, Any]] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.input})
    for result in message.tool_results:
        content.append(
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
        )
    return cast(MessageParam, {"role": message.role, "content": content})
