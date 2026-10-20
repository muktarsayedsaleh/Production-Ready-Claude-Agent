from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, described by a JSON Schema for its input."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Opaque provider content for an assistant turn (e.g. thinking blocks) that must be
    # sent back verbatim. Only the client that produced it should interpret it.
    provider_state: Any = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider_state: Any = None


class TransientLLMError(Exception):
    """A failure worth retrying: rate limits, overload, timeouts, connection errors."""


class LLMClient(ABC):
    """Provider-agnostic interface for a single model call."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse: ...
