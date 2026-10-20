import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from redis.asyncio import Redis


@dataclass(frozen=True)
class ToolUse:
    """One tool call the task made, including ones the policy denied."""

    name: str
    input: dict[str, Any]
    is_error: bool = False


@dataclass(frozen=True)
class TaskResult:
    answer: str
    steps: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolUse] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        tool_calls = [ToolUse(**use) for use in data.get("tool_calls", [])]
        return cls(**{**data, "tool_calls": tool_calls})


class ResultStore(Protocol):
    """Completed results keyed by idempotency key, so a re-run doesn't pay twice."""

    async def get(self, key: str) -> TaskResult | None: ...

    async def save(self, key: str, result: TaskResult) -> None: ...


class RedisResultStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 7 * 24 * 3600) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    async def get(self, key: str) -> TaskResult | None:
        raw = await self._redis.get(self._key(key))
        return None if raw is None else TaskResult.from_dict(json.loads(raw))

    async def save(self, key: str, result: TaskResult) -> None:
        await self._redis.set(self._key(key), json.dumps(asdict(result)), ex=self._ttl)

    @staticmethod
    def _key(key: str) -> str:
        return f"agent:result:{key}"
