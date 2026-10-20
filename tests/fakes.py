from core.llm import LLMClient, LLMResponse, Message, ToolSpec


class FakeLLMClient(LLMClient):
    """Replays scripted responses (or raises scripted errors) and records every call."""

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self.script = list(script)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec] | None] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
