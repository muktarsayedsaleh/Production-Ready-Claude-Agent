import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from redis.asyncio import Redis

from core.llm import Message, ToolCall, ToolResult


@dataclass
class TaskState:
    """Everything needed to continue a task: the conversation, spend, and tool calls in flight."""

    messages: list[Message]
    step: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls_made: dict[str, int] = field(default_factory=dict)
    # Tool calls from the latest LLM response that have not been executed yet.
    pending: list[ToolCall] = field(default_factory=list)
    # Reviewer decisions for pending calls that need approval, by tool call id.
    decisions: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "TaskState":
        data = json.loads(raw)
        return cls(
            messages=[_message(m) for m in data["messages"]],
            step=data["step"],
            cost_usd=data["cost_usd"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            tool_calls_made=data["tool_calls_made"],
            pending=[ToolCall(**c) for c in data["pending"]],
            decisions=data["decisions"],
        )


def _message(data: dict[str, Any]) -> Message:
    return Message(
        role=data["role"],
        text=data["text"],
        tool_calls=[ToolCall(**c) for c in data["tool_calls"]],
        tool_results=[ToolResult(**r) for r in data["tool_results"]],
        provider_state=data["provider_state"],
    )


class CheckpointStore(Protocol):
    """Paused task state keyed by idempotency key."""

    async def get(self, key: str) -> TaskState | None: ...

    async def save(self, key: str, state: TaskState) -> None: ...

    async def delete(self, key: str) -> None: ...


class RedisCheckpointStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 30 * 24 * 3600) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    async def get(self, key: str) -> TaskState | None:
        raw = await self._redis.get(self._key(key))
        return None if raw is None else TaskState.from_json(raw)

    async def save(self, key: str, state: TaskState) -> None:
        await self._redis.set(self._key(key), state.to_json(), ex=self._ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(self._key(key))

    @staticmethod
    def _key(key: str) -> str:
        return f"agent:checkpoint:{key}"
